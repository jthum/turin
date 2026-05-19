use anyhow::{Context, Result};
use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use std::collections::{HashSet, VecDeque};
use std::time::Duration;
use tokio::sync::watch;
use tokio::time::sleep;
use tokio_tungstenite::tungstenite::protocol::Message as WsMessage;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};
use turin_channel_core::{
    ChannelAttachment, ChannelCapabilities, ChannelConversationKey, ChannelKind, ChannelMessageRef,
    ChannelSessionScope, ChannelUser, InboundEvent, OutboundMessage, bound_inbound_text,
};
#[cfg(test)]
use turin_channel_core::{DEFAULT_MAX_INBOUND_TEXT_CHARS, MessageBlock};
use turin_channel_runner::ChannelDriver;

mod manifest;
mod render;
mod settings;
pub use manifest::{adapter_manifest, poll_auth_flow, start_auth_flow};
#[cfg(test)]
use render::DISCORD_CONTENT_MAX_LEN;
use render::{
    DiscordSendMessage, LocalAttachmentRef, discord_payload_from_message, render_outbound_messages,
};
pub use settings::{DiscordChannelDriverConfig, validate_settings};
#[cfg(test)]
pub(crate) use settings::{parse_settings, parse_transport_mode};

const DEFAULT_BASE_URL: &str = "https://discord.com/api/v10";
const DEFAULT_GATEWAY_URL: &str = "wss://gateway.discord.gg/?v=10&encoding=json";
const DEFAULT_GATEWAY_INTENTS: u64 = (1 << 9) | (1 << 12) | (1 << 15);
const SEEN_MESSAGE_IDS_LIMIT: usize = 1_024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscordTransportMode {
    Gateway,
    Polling,
}

type DiscordWsStream = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

struct GatewayConnection {
    stream: DiscordWsStream,
    heartbeat_interval: Duration,
    next_heartbeat_at: tokio::time::Instant,
    seq: Option<u64>,
}

enum GatewayProcessResult {
    Event(Box<InboundEvent>),
    Continue,
    Reconnect,
}

pub struct DiscordChannelDriver {
    channel_runtime_id: String,
    config: DiscordChannelDriverConfig,
    client: reqwest::Client,
    shutdown_rx: watch::Receiver<bool>,
    backlog: VecDeque<InboundEvent>,
    last_seen_message_id: Option<String>,
    initialized: bool,
    gateway: Option<GatewayConnection>,
    last_gateway_seq: Option<u64>,
    gateway_session_id: Option<String>,
    resume_gateway_url: Option<String>,
    seen_message_ids: VecDeque<String>,
    seen_message_set: HashSet<String>,
    reconnect_attempts: u32,
}

impl DiscordChannelDriver {
    pub async fn from_settings(
        channel_runtime_id: impl Into<String>,
        settings: &serde_json::Value,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Result<Self> {
        let config = DiscordChannelDriverConfig::from_settings(settings)?;
        let client = reqwest::Client::builder()
            .user_agent("turin-channel-discord/0.22.0")
            .build()
            .context(
                "[discord_http_client_init_failed] Failed to build Discord adapter HTTP client",
            )?;

        Ok(Self {
            channel_runtime_id: channel_runtime_id.into(),
            config,
            client,
            shutdown_rx,
            backlog: VecDeque::new(),
            last_seen_message_id: None,
            initialized: false,
            gateway: None,
            last_gateway_seq: None,
            gateway_session_id: None,
            resume_gateway_url: None,
            seen_message_ids: VecDeque::new(),
            seen_message_set: HashSet::new(),
            reconnect_attempts: 0,
        })
    }

    async fn next_poll_event(&mut self) -> Result<Option<InboundEvent>> {
        loop {
            if let Some(event) = self.backlog.pop_front() {
                return Ok(Some(event));
            }
            if *self.shutdown_rx.borrow() {
                return Ok(None);
            }

            self.poll_once().await?;
            if let Some(event) = self.backlog.pop_front() {
                return Ok(Some(event));
            }

            tokio::select! {
                changed = self.shutdown_rx.changed() => {
                    if changed.is_ok() && *self.shutdown_rx.borrow() {
                        return Ok(None);
                    }
                }
                _ = sleep(self.config.poll_interval) => {}
            }
        }
    }

    async fn next_gateway_event(&mut self) -> Result<Option<InboundEvent>> {
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

    async fn poll_once(&mut self) -> Result<()> {
        if !self.initialized && self.config.start_from_latest {
            let latest = self.fetch_latest_message_id().await?;
            self.last_seen_message_id = latest;
            self.initialized = true;
            return Ok(());
        }
        self.initialized = true;

        let mut messages = self
            .fetch_messages(
                self.last_seen_message_id.as_deref(),
                self.config.max_messages_per_poll,
            )
            .await?;
        if messages.is_empty() {
            return Ok(());
        }

        messages.sort_by_key(|message| parse_snowflake(&message.id).unwrap_or_default());
        let mut newest_id = self.last_seen_message_id.clone();
        for message in messages {
            if newest_id
                .as_ref()
                .is_none_or(|current| is_newer_snowflake(&message.id, current))
            {
                newest_id = Some(message.id.clone());
            }
            if let Some(event) = self.normalize_message(message) {
                self.backlog.push_back(event);
            }
        }
        self.last_seen_message_id = newest_id;
        Ok(())
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

    fn normalize_message(&mut self, message: DiscordMessage) -> Option<InboundEvent> {
        if !self.track_seen_message(&message.id) {
            return None;
        }
        if self.config.ignore_bot_messages && message.author.bot.unwrap_or(false) {
            return None;
        }
        if message.content.trim().is_empty() && message.attachments.is_empty() {
            return None;
        }

        let room_id = self
            .config
            .room_id
            .clone()
            .or(message.guild_id.clone())
            .or(Some(self.config.channel_id.clone()));

        let attachments = message
            .attachments
            .into_iter()
            .map(|attachment| ChannelAttachment {
                name: attachment.filename,
                content_type: attachment.content_type,
                url: Some(attachment.url),
                local_path: None,
            })
            .collect();

        let conversation = ChannelConversationKey {
            channel: ChannelKind::new("discord"),
            workspace_id: self.config.workspace_id.clone(),
            room_id,
            thread_id: message.channel_id.clone(),
            user_id: match self.config.session_scope {
                ChannelSessionScope::User => Some(message.author.id.clone()),
                ChannelSessionScope::Thread | ChannelSessionScope::Room => None,
            },
        };

        let mut metadata = serde_json::Map::new();
        metadata.insert(
            "discord_message_id".to_string(),
            serde_json::Value::String(message.id.clone()),
        );
        if let Some(guild_id) = message.guild_id {
            metadata.insert(
                "discord_guild_id".to_string(),
                serde_json::Value::String(guild_id),
            );
        }
        metadata.insert(
            "channel_runtime_id".to_string(),
            serde_json::Value::String(self.channel_runtime_id.clone()),
        );
        let text = bound_inbound_text(
            message.content,
            &mut metadata,
            self.config.max_inbound_text_chars,
        );

        Some(InboundEvent {
            message: ChannelMessageRef {
                conversation: conversation.clone(),
                message_id: message.id.clone(),
            },
            conversation,
            user: ChannelUser {
                id: message.author.id,
                display_name: message.author.global_name,
                username: Some(message.author.username),
            },
            session_scope: self.config.session_scope,
            text,
            attachments,
            metadata,
        })
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

    fn track_seen_message(&mut self, message_id: &str) -> bool {
        if self.seen_message_set.contains(message_id) {
            return false;
        }
        self.seen_message_set.insert(message_id.to_string());
        self.seen_message_ids.push_back(message_id.to_string());
        while self.seen_message_ids.len() > SEEN_MESSAGE_IDS_LIMIT {
            if let Some(old) = self.seen_message_ids.pop_front() {
                self.seen_message_set.remove(&old);
            }
        }
        true
    }

    async fn fetch_latest_message_id(&self) -> Result<Option<String>> {
        let mut messages = self.fetch_messages(None, 1).await?;
        Ok(messages.pop().map(|msg| msg.id))
    }

    async fn fetch_messages(&self, after: Option<&str>, limit: u16) -> Result<Vec<DiscordMessage>> {
        let url = format!(
            "{}/channels/{}/messages",
            self.config.base_url, self.config.channel_id
        );
        let mut params = vec![("limit".to_string(), limit.to_string())];
        if let Some(after) = after {
            params.push(("after".to_string(), after.to_string()));
        }

        let response = self
            .request_with_retry(|| {
                self.client
                    .get(&url)
                    .header("Authorization", format!("Bot {}", self.config.token))
                    .query(&params)
                    .build()
                    .context("[discord_http_build_messages_request_failed] Failed to build Discord messages request")
            })
            .await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!(
                "[discord_http_messages_failed] Discord messages request failed with {}: {}",
                status.as_u16(),
                body
            );
        }

        response.json::<Vec<DiscordMessage>>().await.context(
            "[discord_http_decode_messages_failed] Failed to decode Discord messages response",
        )
    }

    async fn post_message(&self, channel_id: &str, message: DiscordSendMessage) -> Result<()> {
        let url = format!("{}/channels/{}/messages", self.config.base_url, channel_id);
        let payload = discord_payload_from_message(&message);

        let response = if message.files.is_empty() {
            self.request_with_retry(|| {
                self.client
                    .post(&url)
                    .header("Authorization", format!("Bot {}", self.config.token))
                    .json(&payload)
                    .build()
                    .context("[discord_http_build_send_request_failed] Failed to build Discord send request")
            })
            .await?
        } else {
            let prepared = prepare_local_files(&message.files).await?;
            self.request_with_retry(|| {
                let payload_json = serde_json::to_string(&payload)
                    .context("Failed to encode Discord multipart payload")?;
                let mut form = reqwest::multipart::Form::new().text("payload_json", payload_json);
                for (index, file) in prepared.iter().enumerate() {
                    let mut part = reqwest::multipart::Part::bytes(file.bytes.clone())
                        .file_name(file.name.clone());
                    if let Some(content_type) = &file.content_type {
                        part = part.mime_str(content_type).with_context(|| {
                            format!("Invalid content type '{}' for '{}'", content_type, file.name)
                        })?;
                    }
                    form = form.part(format!("files[{index}]"), part);
                }

                self.client
                    .post(&url)
                    .header("Authorization", format!("Bot {}", self.config.token))
                    .multipart(form)
                    .build()
                    .context("[discord_http_build_multipart_send_request_failed] Failed to build Discord multipart send request")
            })
            .await?
        };

        let status = response.status();
        if status == reqwest::StatusCode::OK || status == reqwest::StatusCode::CREATED {
            return Ok(());
        }

        let body = response.text().await.unwrap_or_default();
        anyhow::bail!(
            "[discord_send_failed] Discord send request failed with {}: {}",
            status.as_u16(),
            body
        );
    }

    async fn request_with_retry<F>(&self, request_builder: F) -> Result<reqwest::Response>
    where
        F: Fn() -> Result<reqwest::Request>,
    {
        let mut attempts = 0;
        loop {
            attempts += 1;
            let request = request_builder()?;
            let response = match self.client.execute(request).await {
                Ok(response) => response,
                Err(error) => {
                    if attempts < 5 {
                        sleep(retry_backoff(attempts)).await;
                        continue;
                    }
                    return Err(error)
                        .context("[discord_http_request_failed] Discord request failed");
                }
            };

            if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS && attempts < 6 {
                let delay = parse_rate_limit_delay(response)
                    .await
                    .unwrap_or_else(|| retry_backoff(attempts));
                sleep(delay).await;
                continue;
            }

            if response.status().is_server_error() && attempts < 5 {
                sleep(retry_backoff(attempts)).await;
                continue;
            }
            return Ok(response);
        }
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

#[async_trait]
impl ChannelDriver for DiscordChannelDriver {
    fn kind(&self) -> ChannelKind {
        ChannelKind::new("discord")
    }

    fn user_matches_selector(&self, selector: &str, user: &ChannelUser) -> bool {
        let selector = selector.trim();
        if selector.is_empty() {
            return false;
        }
        let selector = selector.strip_prefix('@').unwrap_or(selector);
        user.id == selector
            || user
                .username
                .as_ref()
                .is_some_and(|username| username.eq_ignore_ascii_case(selector))
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities {
            rich_formatting: true,
            threads: true,
            attachments: true,
            ephemeral_messages: false,
        }
    }

    async fn next_event(&mut self) -> Result<Option<InboundEvent>> {
        match self.config.transport_mode {
            DiscordTransportMode::Gateway => self.next_gateway_event().await,
            DiscordTransportMode::Polling => self.next_poll_event().await,
        }
    }

    async fn send(
        &mut self,
        conversation: &ChannelConversationKey,
        message: OutboundMessage,
    ) -> Result<()> {
        let channel_id = if conversation.thread_id.trim().is_empty() {
            self.config.channel_id.clone()
        } else {
            conversation.thread_id.clone()
        };
        let outbound_messages = render_outbound_messages(message);
        for outbound in outbound_messages {
            self.post_message(&channel_id, outbound).await?;
        }
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<()> {
        if let Some(mut connection) = self.gateway.take() {
            let _ = connection.stream.close(None).await;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
struct DiscordMessage {
    id: String,
    channel_id: String,
    #[serde(default)]
    guild_id: Option<String>,
    #[serde(default)]
    content: String,
    author: DiscordAuthor,
    #[serde(default)]
    attachments: Vec<DiscordAttachment>,
}

#[derive(Debug, Clone, Deserialize)]
struct DiscordAuthor {
    id: String,
    username: String,
    #[serde(default)]
    global_name: Option<String>,
    #[serde(default)]
    bot: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct DiscordAttachment {
    filename: String,
    #[serde(default)]
    content_type: Option<String>,
    url: String,
}

#[derive(Debug, Clone, Deserialize)]
struct DiscordRateLimit {
    retry_after: f64,
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

#[derive(Debug, Clone)]
struct PreparedLocalFile {
    name: String,
    content_type: Option<String>,
    bytes: Vec<u8>,
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

fn parse_snowflake(value: &str) -> Option<u64> {
    value.parse::<u64>().ok()
}

fn is_newer_snowflake(candidate: &str, current: &str) -> bool {
    match (parse_snowflake(candidate), parse_snowflake(current)) {
        (Some(candidate), Some(current)) => candidate > current,
        _ => candidate > current,
    }
}

async fn prepare_local_files(files: &[LocalAttachmentRef]) -> Result<Vec<PreparedLocalFile>> {
    let mut prepared = Vec::new();
    for file in files {
        let bytes = tokio::fs::read(&file.path).await.with_context(|| {
            format!("Failed to read local attachment '{}'", file.path.display())
        })?;
        prepared.push(PreparedLocalFile {
            name: file.name.clone(),
            content_type: file.content_type.clone(),
            bytes,
        });
    }
    Ok(prepared)
}

async fn parse_rate_limit_delay(response: reqwest::Response) -> Option<Duration> {
    let header_delay = response
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|raw| raw.parse::<f64>().ok())
        .map(Duration::from_secs_f64);
    if header_delay.is_some() {
        return header_delay;
    }

    let reset_after = response
        .headers()
        .get("x-ratelimit-reset-after")
        .and_then(|value| value.to_str().ok())
        .and_then(|raw| raw.parse::<f64>().ok())
        .map(Duration::from_secs_f64);
    if reset_after.is_some() {
        return reset_after;
    }

    let body_delay = response
        .text()
        .await
        .ok()
        .and_then(|raw| serde_json::from_str::<DiscordRateLimit>(&raw).ok())
        .map(|rate| Duration::from_secs_f64(rate.retry_after.max(0.1)));
    if body_delay.is_some() {
        return body_delay;
    }

    None
}

fn retry_backoff(attempt: u32) -> Duration {
    let exponent = attempt.min(5);
    let millis = 200u64.saturating_mul(2u64.saturating_pow(exponent));
    Duration::from_millis(millis.min(6_000))
}

fn is_fatal_gateway_close_code(code: u16) -> bool {
    matches!(code, 4004 | 4010 | 4011 | 4012 | 4013 | 4014)
}

#[cfg(test)]
mod tests;
