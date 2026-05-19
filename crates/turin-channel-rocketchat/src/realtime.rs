use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use tokio::time::sleep;
use tokio_tungstenite::tungstenite::protocol::Message as WsMessage;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};
use tracing::warn;
use turin_channel_core::InboundEvent;
use turin_channel_runner::ChannelStreamMode;

use crate::{
    DEFAULT_REALTIME_RECONNECT_DELAY_MS, ROCKETCHAT_REALTIME_CONNECT_TIMEOUT_SECONDS,
    ROCKETCHAT_REALTIME_HANDSHAKE_TIMEOUT_SECONDS, ROCKETCHAT_REALTIME_KEEPALIVE_SECONDS,
    ROCKETCHAT_REALTIME_STALE_SECONDS, ROCKETCHAT_TYPING_STATUS_INTERVAL_SECONDS,
    RocketChatChannelDriver, RocketChatTransportMode, build_http_client, subscription_request_id,
    subscription_room_id,
};

pub(super) type RocketChatWsStream = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

impl RocketChatChannelDriver {
    fn note_realtime_activity(&mut self) {
        self.last_realtime_activity_at = Some(Instant::now());
    }

    fn should_send_realtime_keepalive(&self) -> bool {
        let Some(last_activity) = self.last_realtime_activity_at else {
            return false;
        };
        if self.ws_stream.is_none() {
            return false;
        }
        if last_activity.elapsed() < Duration::from_secs(ROCKETCHAT_REALTIME_KEEPALIVE_SECONDS) {
            return false;
        }
        self.last_realtime_keepalive_at
            .is_none_or(|last_keepalive| {
                last_keepalive.elapsed()
                    >= Duration::from_secs(ROCKETCHAT_REALTIME_KEEPALIVE_SECONDS)
            })
    }

    fn realtime_connection_stale(&self) -> bool {
        self.ws_stream.is_some()
            && self.last_realtime_activity_at.is_some_and(|last_activity| {
                last_activity.elapsed() >= Duration::from_secs(ROCKETCHAT_REALTIME_STALE_SECONDS)
            })
    }

    async fn send_realtime_keepalive(&mut self) -> Result<()> {
        let Some(stream) = self.ws_stream.as_mut() else {
            return Ok(());
        };
        send_ws_json(stream, serde_json::json!({ "msg": "ping" }))
            .await
            .context(
                "[rocketchat_realtime_keepalive_failed] Failed to send Rocket.Chat realtime keepalive ping",
            )?;
        self.last_realtime_keepalive_at = Some(Instant::now());
        Ok(())
    }

    pub(super) fn reset_transport_state(&mut self) -> Result<()> {
        self.ws_stream = None;
        self.realtime_subscribed_room_ids.clear();
        self.last_typing_at.clear();
        self.last_realtime_activity_at = None;
        self.last_realtime_keepalive_at = None;
        self.client = build_http_client().context(
            "[rocketchat_http_client_rebuild_failed] Failed to rebuild Rocket.Chat HTTP client",
        )?;
        Ok(())
    }

    pub(super) async fn send_typing_status(&mut self, event: &InboundEvent) -> Result<()> {
        if self.config.stream_mode != ChannelStreamMode::Typing
            || self.config.transport_mode != RocketChatTransportMode::Realtime
        {
            return Ok(());
        }

        let Some(room_id) = event.conversation.room_id.as_deref() else {
            return Ok(());
        };

        let key = crate::progress_key(&event.conversation)?;
        let now = Instant::now();
        if self.last_typing_at.get(&key).is_some_and(|previous| {
            now.duration_since(*previous)
                < Duration::from_secs(ROCKETCHAT_TYPING_STATUS_INTERVAL_SECONDS)
        }) {
            return Ok(());
        }

        if self.bot_username.is_none()
            && let Err(err) = self.load_bot_identity().await
        {
            warn!(
                channel_id = %self.channel_id,
                error = ?err,
                "Rocket.Chat bot username lookup failed during typing update"
            );
            return Ok(());
        }
        let Some(username) = self.bot_username.clone() else {
            return Ok(());
        };

        self.ensure_realtime_connected().await?;
        self.send_room_notification(
            vec![
                serde_json::json!(format!("{room_id}/typing")),
                serde_json::json!(username.clone()),
                serde_json::json!(true),
            ],
            "[rocketchat_typing_failed] Failed to send Rocket.Chat typing notification",
        )
        .await?;
        if let Err(err) = self
            .send_room_notification(
                vec![
                    serde_json::json!(format!("{room_id}/user-activity")),
                    serde_json::json!(username.clone()),
                    serde_json::json!(["user-typing"]),
                    serde_json::json!({}),
                ],
                "[rocketchat_user_activity_failed] Failed to send Rocket.Chat user activity notification",
            )
            .await
        {
            warn!(
                channel_id = %self.channel_id,
                room_id = room_id,
                error = ?err,
                "Rocket.Chat user activity notification failed"
            );
        }
        self.last_typing_at.insert(key, now);
        Ok(())
    }

    async fn send_room_notification(
        &mut self,
        params: Vec<serde_json::Value>,
        error_context: &str,
    ) -> Result<()> {
        let request_id = self.next_request_id();
        let stream = self.ws_stream.as_mut().ok_or_else(|| {
            anyhow!("[rocketchat_realtime_missing_stream] Rocket.Chat websocket is not connected")
        })?;
        send_ws_json(
            stream,
            serde_json::json!({
                "msg": "method",
                "method": "stream-notify-room",
                "id": request_id,
                "params": params
            }),
        )
        .await
        .with_context(|| error_context.to_string())?;
        Ok(())
    }

    async fn ensure_realtime_connected(&mut self) -> Result<()> {
        if self.ws_stream.is_some() {
            return Ok(());
        }

        let websocket_url = self.config.websocket_url.clone();
        let (mut stream, _) = tokio::time::timeout(
            Duration::from_secs(ROCKETCHAT_REALTIME_CONNECT_TIMEOUT_SECONDS),
            connect_async(&websocket_url),
        )
        .await
        .with_context(|| {
            format!(
                "[rocketchat_realtime_connect_timeout] Timed out connecting to Rocket.Chat websocket '{}'",
                websocket_url
            )
        })?
        .with_context(|| {
            format!(
                "[rocketchat_realtime_connect_failed] Failed to connect to Rocket.Chat websocket '{}'",
                websocket_url
            )
        })?;

        send_ws_json(
            &mut stream,
            serde_json::json!({
                "msg": "connect",
                "version": "1",
                "support": ["1"]
            }),
        )
        .await
        .context("[rocketchat_realtime_connect_send_failed] Failed to send DDP connect message")?;

        tokio::time::timeout(
            Duration::from_secs(ROCKETCHAT_REALTIME_HANDSHAKE_TIMEOUT_SECONDS),
            self.await_connected(&mut stream),
        )
        .await
        .context(
            "[rocketchat_realtime_connect_timeout] Timed out waiting for Rocket.Chat DDP connect acknowledgement",
        )??;
        tokio::time::timeout(
            Duration::from_secs(ROCKETCHAT_REALTIME_HANDSHAKE_TIMEOUT_SECONDS),
            self.login_realtime(&mut stream),
        )
        .await
        .context(
            "[rocketchat_realtime_login_timeout] Timed out waiting for Rocket.Chat DDP login response",
        )??;

        self.ws_stream = Some(stream);
        self.note_realtime_activity();
        self.last_realtime_keepalive_at = None;
        self.realtime_subscribed_room_ids.clear();
        self.sync_realtime_subscriptions().await?;

        if self.rooms.values().any(|state| state.cursor_ts.is_some()) {
            self.poll_known_rooms().await?;
        }

        Ok(())
    }

    async fn await_connected(&mut self, stream: &mut RocketChatWsStream) -> Result<()> {
        loop {
            let frame = read_ddp_frame(stream).await?;
            match frame.msg.as_deref() {
                Some("connected") => return Ok(()),
                Some("failed") => {
                    anyhow::bail!(
                        "[rocketchat_realtime_connect_rejected] Rocket.Chat rejected the DDP connect negotiation"
                    );
                }
                _ => {}
            }
        }
    }

    async fn login_realtime(&mut self, stream: &mut RocketChatWsStream) -> Result<()> {
        let request_id = self.next_request_id();
        send_ws_json(
            stream,
            serde_json::json!({
                "msg": "method",
                "method": "login",
                "id": request_id,
                "params": [{
                    "resume": self.config.token
                }]
            }),
        )
        .await
        .context(
            "[rocketchat_realtime_login_send_failed] Failed to send Rocket.Chat DDP login request",
        )?;

        loop {
            let frame = read_ddp_frame(stream).await?;
            if frame.id.as_deref() != Some(request_id.as_str()) {
                continue;
            }

            if frame.msg.as_deref() == Some("result") {
                if let Some(error) = login_result_error(&frame) {
                    let maybe_error = error
                        .get("error")
                        .or_else(|| error.get("reason"))
                        .cloned()
                        .unwrap_or(error);
                    anyhow::bail!(
                        "[rocketchat_realtime_login_failed] Rocket.Chat DDP login failed: {}",
                        maybe_error
                    );
                }
                return Ok(());
            }

            if frame.msg.as_deref() == Some("error") {
                anyhow::bail!(
                    "[rocketchat_realtime_login_failed] Rocket.Chat DDP login returned an error"
                );
            }
        }
    }

    async fn sync_realtime_subscriptions(&mut self) -> Result<()> {
        let Some(stream) = self.ws_stream.as_mut() else {
            return Ok(());
        };

        let mut room_ids: Vec<String> = self.rooms.keys().cloned().collect();
        room_ids.sort();
        for room_id in room_ids {
            if self.realtime_subscribed_room_ids.contains(&room_id) {
                continue;
            }
            let request_id = subscription_request_id(&room_id);
            send_ws_json(
                stream,
                serde_json::json!({
                    "msg": "sub",
                    "id": request_id,
                    "name": "stream-room-messages",
                    "params": [room_id.clone(), false]
                }),
            )
            .await
            .with_context(|| {
                format!(
                    "[rocketchat_realtime_subscribe_send_failed] Failed to subscribe to Rocket.Chat room '{}'",
                    room_id
                )
            })?;
            self.realtime_subscribed_room_ids.insert(room_id);
        }
        Ok(())
    }

    pub(super) async fn next_realtime_event(&mut self) -> Result<Option<InboundEvent>> {
        loop {
            if let Err(err) = self.ensure_realtime_connected().await {
                warn!(
                    channel_id = %self.channel_id,
                    room_count = self.rooms.len(),
                    error = ?err,
                    "Rocket.Chat realtime connection failed"
                );
                if let Err(reset_err) = self.reset_transport_state() {
                    warn!(
                        channel_id = %self.channel_id,
                        error = ?reset_err,
                        "Rocket.Chat transport reset failed"
                    );
                }
                tokio::select! {
                    changed = self.shutdown_rx.changed() => {
                        if changed.is_ok() && *self.shutdown_rx.borrow() {
                            return Ok(None);
                        }
                    }
                    _ = sleep(Duration::from_millis(DEFAULT_REALTIME_RECONNECT_DELAY_MS)) => {}
                }
                continue;
            }

            if let Some(event) = self.backlog.pop_front() {
                return Ok(Some(event));
            }

            if self.realtime_connection_stale() {
                warn!(
                    channel_id = %self.channel_id,
                    stale_after_seconds = ROCKETCHAT_REALTIME_STALE_SECONDS,
                    "Rocket.Chat realtime connection went idle; resetting transport"
                );
                if let Err(reset_err) = self.reset_transport_state() {
                    warn!(
                        channel_id = %self.channel_id,
                        error = ?reset_err,
                        "Rocket.Chat transport reset failed"
                    );
                }
                continue;
            }

            if self.should_send_realtime_keepalive()
                && let Err(err) = self.send_realtime_keepalive().await
            {
                warn!(
                    channel_id = %self.channel_id,
                    error = ?err,
                    "Rocket.Chat realtime keepalive failed; resetting transport"
                );
                if let Err(reset_err) = self.reset_transport_state() {
                    warn!(
                        channel_id = %self.channel_id,
                        error = ?reset_err,
                        "Rocket.Chat transport reset failed"
                    );
                }
                continue;
            }

            if self
                .last_room_refresh
                .is_none_or(|last| last.elapsed() >= self.config.poll_interval)
            {
                if let Err(err) = self.refresh_rooms(false).await {
                    warn!(
                        channel_id = %self.channel_id,
                        error = ?err,
                        "Rocket.Chat room refresh failed"
                    );
                    if let Err(reset_err) = self.reset_transport_state() {
                        warn!(
                            channel_id = %self.channel_id,
                            error = ?reset_err,
                            "Rocket.Chat transport reset failed"
                        );
                    }
                    continue;
                } else if let Err(err) = self.sync_realtime_subscriptions().await {
                    warn!(
                        channel_id = %self.channel_id,
                        error = ?err,
                        "Rocket.Chat subscription sync failed"
                    );
                    if let Err(reset_err) = self.reset_transport_state() {
                        warn!(
                            channel_id = %self.channel_id,
                            error = ?reset_err,
                            "Rocket.Chat transport reset failed"
                        );
                    }
                    continue;
                }
                if let Some(event) = self.backlog.pop_front() {
                    return Ok(Some(event));
                }
            }

            let refresh_delay = self
                .last_room_refresh
                .map(|last| self.config.poll_interval.saturating_sub(last.elapsed()))
                .unwrap_or(Duration::from_secs(0));

            let result = {
                let stream = self
                    .ws_stream
                    .as_mut()
                    .expect("realtime stream established before reading events");
                tokio::select! {
                    changed = self.shutdown_rx.changed() => {
                        if changed.is_ok() && *self.shutdown_rx.borrow() {
                            return Ok(None);
                        }
                        Ok(None)
                    }
                    _ = sleep(refresh_delay) => Ok(None),
                    frame = read_ddp_frame(stream) => frame.map(Some),
                }
            };

            match result {
                Ok(None) => continue,
                Ok(Some(frame)) => {
                    self.note_realtime_activity();
                    self.last_realtime_keepalive_at = None;
                    if let Some(event) = self.process_realtime_frame(frame)? {
                        return Ok(Some(event));
                    }
                }
                Err(err) => {
                    warn!(
                        channel_id = %self.channel_id,
                        error = ?err,
                        "Rocket.Chat realtime stream failed; reconnecting"
                    );
                    if let Err(reset_err) = self.reset_transport_state() {
                        warn!(
                            channel_id = %self.channel_id,
                            error = ?reset_err,
                            "Rocket.Chat transport reset failed"
                        );
                    }
                }
            }
        }
    }

    fn process_realtime_frame(
        &mut self,
        frame: RocketChatDdpFrame,
    ) -> Result<Option<InboundEvent>> {
        if frame.msg.as_deref() == Some("nosub") {
            if let Some(room_id) = frame.id.as_deref().and_then(subscription_room_id) {
                self.realtime_subscribed_room_ids.remove(room_id);
                warn!(
                    channel_id = %self.channel_id,
                    room_id = room_id,
                    "Rocket.Chat room subscription was rejected"
                );
            }
            return Ok(None);
        }

        if frame.msg.as_deref() != Some("changed")
            || frame.collection.as_deref() != Some("stream-room-messages")
        {
            return Ok(None);
        }

        let fields = match frame.fields {
            Some(fields) => fields,
            None => return Ok(None),
        };
        let Some(room_id) = fields.event_name.as_deref() else {
            return Ok(None);
        };
        let Some(room) = self.rooms.get(room_id).map(|state| state.room.clone()) else {
            return Ok(None);
        };

        let Some(raw_message) = fields.args.into_iter().next() else {
            return Ok(None);
        };
        let message: crate::RocketChatMessage = serde_json::from_value(raw_message).context(
            "[rocketchat_realtime_decode_message_failed] Failed to decode Rocket.Chat room message from realtime event",
        )?;
        self.update_room_cursor(room_id, message.ts.clone());
        if self.seen_message_ids.contains(&message.id) {
            return Ok(None);
        }
        self.remember_message_id(message.id.clone());
        self.message_to_event(&room, message)
    }
}

async fn send_ws_json(stream: &mut RocketChatWsStream, payload: serde_json::Value) -> Result<()> {
    stream
        .send(WsMessage::Text(payload.to_string()))
        .await
        .context("Failed to send Rocket.Chat websocket frame")
}

async fn read_ddp_frame(stream: &mut RocketChatWsStream) -> Result<RocketChatDdpFrame> {
    loop {
        let message = stream
            .next()
            .await
            .ok_or_else(|| anyhow!("[rocketchat_realtime_closed] Rocket.Chat websocket closed"))?
            .context(
                "[rocketchat_realtime_receive_failed] Failed to read Rocket.Chat websocket frame",
            )?;

        match message {
            WsMessage::Text(text) => {
                let frame: RocketChatDdpFrame = serde_json::from_str(&text).context(
                    "[rocketchat_realtime_decode_failed] Failed to decode Rocket.Chat DDP frame",
                )?;
                if frame.msg.as_deref() == Some("ping") {
                    send_ws_json(stream, serde_json::json!({ "msg": "pong" }))
                        .await
                        .context("[rocketchat_realtime_pong_failed] Failed to respond to Rocket.Chat DDP ping")?;
                    continue;
                }
                return Ok(frame);
            }
            WsMessage::Binary(bytes) => {
                let frame: RocketChatDdpFrame = serde_json::from_slice(&bytes)
                    .context("[rocketchat_realtime_decode_failed] Failed to decode Rocket.Chat binary DDP frame")?;
                if frame.msg.as_deref() == Some("ping") {
                    send_ws_json(stream, serde_json::json!({ "msg": "pong" }))
                        .await
                        .context("[rocketchat_realtime_pong_failed] Failed to respond to Rocket.Chat DDP ping")?;
                    continue;
                }
                return Ok(frame);
            }
            WsMessage::Ping(payload) => {
                stream
                    .send(WsMessage::Pong(payload))
                    .await
                    .context("[rocketchat_realtime_pong_failed] Failed to respond to Rocket.Chat websocket ping")?;
            }
            WsMessage::Pong(_) => {}
            WsMessage::Close(_) => {
                anyhow::bail!("[rocketchat_realtime_closed] Rocket.Chat websocket closed");
            }
            WsMessage::Frame(_) => {}
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct RocketChatDdpFrame {
    #[serde(default)]
    pub(crate) msg: Option<String>,
    #[serde(default)]
    pub(crate) id: Option<String>,
    #[serde(default)]
    collection: Option<String>,
    #[serde(default)]
    fields: Option<RocketChatDdpChangedFields>,
    #[serde(default)]
    pub(crate) error: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct RocketChatDdpChangedFields {
    #[serde(rename = "eventName", default)]
    event_name: Option<String>,
    #[serde(default)]
    args: Vec<serde_json::Value>,
}

pub(crate) fn login_result_error(frame: &RocketChatDdpFrame) -> Option<serde_json::Value> {
    if frame.msg.as_deref() != Some("result") {
        return None;
    }
    frame.error.clone()
}
