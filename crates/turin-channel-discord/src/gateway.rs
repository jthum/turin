use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use std::time::Duration;
use tokio::time::sleep;
use tokio_tungstenite::tungstenite::protocol::Message as WsMessage;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};
use turin_channel_core::InboundEvent;

use crate::{DiscordChannelDriver, api::DiscordMessage};

type DiscordWsStream = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

pub(crate) struct GatewayConnection {
    pub(crate) stream: DiscordWsStream,
    heartbeat_interval: Duration,
    next_heartbeat_at: tokio::time::Instant,
    seq: Option<u64>,
}

enum GatewayProcessResult {
    Event(Box<InboundEvent>),
    Continue,
    Reconnect,
}

#[derive(Debug, Clone, Deserialize)]
struct GatewayPayload {
    op: u8,
    #[serde(default)]
    d: serde_json::Value,
    #[serde(default)]
    s: Option<u64>,
    #[serde(default)]
    t: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct GatewayHello {
    heartbeat_interval: u64,
}

#[derive(Debug, Clone, Deserialize)]
struct GatewayReady {
    session_id: String,
    resume_gateway_url: String,
}

impl DiscordChannelDriver {
    pub(crate) async fn next_gateway_event(&mut self) -> Result<Option<InboundEvent>> {
        loop {
            if let Some(event) = self.backlog.pop_front() {
                return Ok(Some(event));
            }
            if *self.shutdown_rx.borrow() {
                return Ok(None);
            }

            if let Err(_err) = self.ensure_gateway_connected().await {
                let backoff = self.next_reconnect_delay();
                if self.sleep_or_shutdown(backoff).await {
                    return Ok(None);
                }
                continue;
            }
            let mut connection = match self.gateway.take() {
                Some(connection) => connection,
                None => continue,
            };

            let mut reconnect = false;
            let mut emitted_event = None;

            let heartbeat_at = connection.next_heartbeat_at;
            let heartbeat_sleep = tokio::time::sleep_until(heartbeat_at);
            tokio::pin!(heartbeat_sleep);

            tokio::select! {
                _ = &mut heartbeat_sleep => {
                    self.send_gateway_heartbeat(&mut connection).await?;
                }
                changed = self.shutdown_rx.changed() => {
                    if changed.is_ok() && *self.shutdown_rx.borrow() {
                        return Ok(None);
                    }
                }
                maybe_msg = connection.stream.next() => {
                    let Some(msg_result) = maybe_msg else {
                        self.gateway = None;
                        continue;
                    };
                    let msg = msg_result.context("Discord gateway stream read failed")?;
                    match self.process_gateway_message(&mut connection, msg).await? {
                        GatewayProcessResult::Event(event) => emitted_event = Some(*event),
                        GatewayProcessResult::Reconnect => reconnect = true,
                        GatewayProcessResult::Continue => {}
                    }
                }
            }

            if reconnect {
                self.gateway = None;
                let backoff = self.next_reconnect_delay();
                if self.sleep_or_shutdown(backoff).await {
                    return Ok(None);
                }
                continue;
            }

            self.gateway = Some(connection);
            if let Some(event) = emitted_event {
                return Ok(Some(event));
            }
        }
    }

    async fn ensure_gateway_connected(&mut self) -> Result<()> {
        if self.gateway.is_some() {
            return Ok(());
        }
        self.gateway = Some(self.connect_gateway().await?);
        self.reconnect_attempts = 0;
        Ok(())
    }

    async fn connect_gateway(&mut self) -> Result<GatewayConnection> {
        let gateway_url = self
            .resume_gateway_url
            .as_deref()
            .unwrap_or(&self.config.gateway_url);
        let (mut stream, _) = connect_async(gateway_url).await.with_context(|| {
            format!(
                "[discord_gateway_connect_failed] Failed to connect to Discord gateway '{}'",
                gateway_url
            )
        })?;

        let hello_payload = loop {
            let Some(msg) = stream.next().await else {
                anyhow::bail!(
                    "[discord_gateway_closed_before_hello] Discord gateway closed before HELLO"
                );
            };
            if let Some(payload) = decode_gateway_payload(msg?)? {
                break payload;
            }
        };

        if hello_payload.op != 10 {
            anyhow::bail!(
                "[discord_gateway_unexpected_hello] Discord gateway expected HELLO (op=10), got op={} instead",
                hello_payload.op
            );
        }

        let hello: GatewayHello = serde_json::from_value(hello_payload.d).context(
            "[discord_gateway_decode_hello_failed] Failed to decode Discord HELLO payload",
        )?;
        let heartbeat_interval = Duration::from_millis(hello.heartbeat_interval.max(100));

        let payload = if let Some(resume) = self.resume_payload() {
            resume
        } else {
            self.identify_payload()
        };
        stream
            .send(WsMessage::Text(payload.to_string()))
            .await
            .context(
                "[discord_gateway_auth_payload_failed] Failed to send Discord gateway auth payload",
            )?;

        Ok(GatewayConnection {
            stream,
            heartbeat_interval,
            next_heartbeat_at: tokio::time::Instant::now() + heartbeat_interval,
            seq: self.last_gateway_seq.or(hello_payload.s),
        })
    }

    async fn send_gateway_heartbeat(&self, connection: &mut GatewayConnection) -> Result<()> {
        let heartbeat = serde_json::json!({
            "op": 1,
            "d": connection.seq,
        });
        connection
            .stream
            .send(WsMessage::Text(heartbeat.to_string()))
            .await
            .context("[discord_gateway_heartbeat_send_failed] Failed to send Discord heartbeat")?;
        connection.next_heartbeat_at = tokio::time::Instant::now() + connection.heartbeat_interval;
        Ok(())
    }

    async fn process_gateway_message(
        &mut self,
        connection: &mut GatewayConnection,
        message: WsMessage,
    ) -> Result<GatewayProcessResult> {
        match message {
            WsMessage::Ping(payload) => {
                connection
                    .stream
                    .send(WsMessage::Pong(payload))
                    .await
                    .context(
                        "[discord_gateway_pong_send_failed] Failed to respond to Discord ping",
                    )?;
                return Ok(GatewayProcessResult::Continue);
            }
            WsMessage::Close(frame) => {
                let close_code = frame.as_ref().map(|f| f.code.into()).unwrap_or(0u16);
                if is_fatal_gateway_close_code(close_code) {
                    anyhow::bail!(
                        "[discord_gateway_close_fatal_{}] Discord gateway closed with fatal close code {}",
                        close_code,
                        close_code
                    );
                }
                return Ok(GatewayProcessResult::Reconnect);
            }
            WsMessage::Pong(_) | WsMessage::Frame(_) => {
                return Ok(GatewayProcessResult::Continue);
            }
            _ => {}
        }

        let Some(payload) = decode_gateway_payload(message)? else {
            return Ok(GatewayProcessResult::Continue);
        };

        if let Some(seq) = payload.s {
            connection.seq = Some(seq);
            self.last_gateway_seq = Some(seq);
        }

        match payload.op {
            0 => self.process_gateway_dispatch(payload.t.as_deref(), payload.d),
            1 => {
                self.send_gateway_heartbeat(connection).await?;
                Ok(GatewayProcessResult::Continue)
            }
            7 => Ok(GatewayProcessResult::Reconnect),
            9 => {
                let can_resume = payload.d.as_bool().unwrap_or(false);
                if !can_resume {
                    self.clear_gateway_resume_state();
                }
                Ok(GatewayProcessResult::Reconnect)
            }
            10 => {
                let hello: GatewayHello = serde_json::from_value(payload.d)
                    .context("[discord_gateway_decode_reconnect_hello_failed] Failed to decode Discord HELLO payload during reconnect")?;
                connection.heartbeat_interval =
                    Duration::from_millis(hello.heartbeat_interval.max(100));
                connection.next_heartbeat_at =
                    tokio::time::Instant::now() + connection.heartbeat_interval;
                Ok(GatewayProcessResult::Continue)
            }
            11 => Ok(GatewayProcessResult::Continue),
            _ => Ok(GatewayProcessResult::Continue),
        }
    }

    fn process_gateway_dispatch(
        &mut self,
        event_name: Option<&str>,
        data: serde_json::Value,
    ) -> Result<GatewayProcessResult> {
        if event_name == Some("READY") {
            let ready: GatewayReady = serde_json::from_value(data).context(
                "[discord_gateway_decode_ready_failed] Failed to decode Discord READY payload",
            )?;
            self.gateway_session_id = Some(ready.session_id);
            self.resume_gateway_url = Some(ready.resume_gateway_url);
            return Ok(GatewayProcessResult::Continue);
        }

        if event_name == Some("RESUMED") {
            return Ok(GatewayProcessResult::Continue);
        }

        if event_name == Some("MESSAGE_CREATE") {
            let message: DiscordMessage =
                serde_json::from_value(data).context("[discord_gateway_decode_message_create_failed] Failed to decode Discord MESSAGE_CREATE")?;
            if message.channel_id != self.config.channel_id {
                return Ok(GatewayProcessResult::Continue);
            }
            if let Some(event) = self.normalize_message(message) {
                return Ok(GatewayProcessResult::Event(Box::new(event)));
            }
        }

        Ok(GatewayProcessResult::Continue)
    }

    fn identify_payload(&self) -> serde_json::Value {
        serde_json::json!({
            "op": 2,
            "d": {
                "token": self.config.token,
                "intents": self.config.gateway_intents,
                "properties": {
                    "os": "linux",
                    "browser": "turin",
                    "device": "turin"
                }
            }
        })
    }

    fn resume_payload(&self) -> Option<serde_json::Value> {
        let session_id = self.gateway_session_id.as_deref()?;
        let seq = self.last_gateway_seq?;
        Some(serde_json::json!({
            "op": 6,
            "d": {
                "token": self.config.token,
                "session_id": session_id,
                "seq": seq
            }
        }))
    }

    fn clear_gateway_resume_state(&mut self) {
        self.gateway_session_id = None;
        self.resume_gateway_url = None;
        self.last_gateway_seq = None;
    }

    fn next_reconnect_delay(&mut self) -> Duration {
        self.reconnect_attempts = self.reconnect_attempts.saturating_add(1);
        let exponent = self.reconnect_attempts.min(6);
        let base_ms = 250u64.saturating_mul(2u64.saturating_pow(exponent));
        Duration::from_millis(base_ms.min(8_000))
    }

    async fn sleep_or_shutdown(&mut self, duration: Duration) -> bool {
        tokio::select! {
            changed = self.shutdown_rx.changed() => {
                changed.is_ok() && *self.shutdown_rx.borrow()
            }
            _ = sleep(duration) => false
        }
    }
}

fn decode_gateway_payload(message: WsMessage) -> Result<Option<GatewayPayload>> {
    match message {
        WsMessage::Text(text) => {
            let payload = serde_json::from_str::<GatewayPayload>(&text)
                .context("[discord_gateway_invalid_text_payload] Invalid Discord text payload")?;
            Ok(Some(payload))
        }
        WsMessage::Binary(binary) => {
            let payload = serde_json::from_slice::<GatewayPayload>(&binary).context(
                "[discord_gateway_invalid_binary_payload] Invalid Discord binary payload",
            )?;
            Ok(Some(payload))
        }
        WsMessage::Ping(_) | WsMessage::Pong(_) | WsMessage::Close(_) | WsMessage::Frame(_) => {
            Ok(None)
        }
    }
}

fn is_fatal_gateway_close_code(code: u16) -> bool {
    matches!(code, 4004 | 4010 | 4011 | 4012 | 4013 | 4014)
}
