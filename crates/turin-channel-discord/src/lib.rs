use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::watch;
use tokio::time::sleep;
use tokio_tungstenite::tungstenite::protocol::Message as WsMessage;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};
use turin_channel_core::{
    ChannelAdapterManifest, ChannelAttachment, ChannelCapabilities, ChannelConfigField,
    ChannelConfigFieldOption, ChannelConfigTarget, ChannelConfigTargetKind, ChannelConversationKey,
    ChannelEnumSetting, ChannelIdentitySelectors, ChannelInstallManifest, ChannelKind,
    ChannelMessageRef, ChannelRuntimeCapabilities, ChannelRuntimeManifest, ChannelSessionScope,
    ChannelSecretRequirement, ChannelSetupManifest, ChannelUser, ChannelValidationCheck,
    InboundEvent, MessageBlock, OutboundMessage,
};
use turin_channel_runner::ChannelDriver;

const DEFAULT_BASE_URL: &str = "https://discord.com/api/v10";
const DEFAULT_GATEWAY_URL: &str = "wss://gateway.discord.gg/?v=10&encoding=json";
const DEFAULT_GATEWAY_INTENTS: u64 = (1 << 9) | (1 << 12) | (1 << 15);
const DISCORD_CONTENT_MAX_LEN: usize = 2_000;
const DISCORD_EMBEDS_MAX: usize = 10;
const DISCORD_FILES_MAX: usize = 10;
const SEEN_MESSAGE_IDS_LIMIT: usize = 1_024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscordTransportMode {
    Gateway,
    Polling,
}

#[derive(Debug, Clone)]
pub struct DiscordChannelDriverConfig {
    pub base_url: String,
    pub gateway_url: String,
    pub transport_mode: DiscordTransportMode,
    pub gateway_intents: u64,
    pub workspace_id: String,
    pub room_id: Option<String>,
    pub channel_id: String,
    pub token: String,
    pub poll_interval: Duration,
    pub max_messages_per_poll: u16,
    pub start_from_latest: bool,
    pub ignore_bot_messages: bool,
    pub session_scope: ChannelSessionScope,
}

pub fn validate_settings(settings: &serde_json::Value) -> Result<()> {
    parse_settings(settings).map(|_| ())
}

pub fn adapter_manifest() -> ChannelAdapterManifest {
    ChannelAdapterManifest {
        protocol_version: 1,
        kind: "discord".to_string(),
        display_name: "Discord".to_string(),
        runtime: ChannelRuntimeManifest {
            session_scopes: vec!["user".to_string(), "thread".to_string()],
            enum_settings: vec![ChannelEnumSetting {
                key: "session_scope".to_string(),
                options: vec!["user".to_string(), "thread".to_string()],
            }],
            capabilities: ChannelRuntimeCapabilities {
                dm: true,
                groups: true,
                threads: true,
                attachments: true,
                streaming: false,
            },
            identity_selectors: ChannelIdentitySelectors {
                matching_rules: vec!["id".to_string(), "username".to_string()],
                examples: vec!["123456789012345678".to_string(), "jthum".to_string()],
            },
        },
        setup: Some(ChannelSetupManifest {
            required_secrets: vec![ChannelSecretRequirement {
                name: "discord_bot_token".to_string(),
                env_var: "DISCORD_BOT_TOKEN".to_string(),
                display_name: Some("Discord bot token".to_string()),
                help: Some("Get this from the Discord developer portal for your application.".to_string()),
                optional: false,
                hints: vec!["Usually a long bot token string issued by Discord.".to_string()],
            }],
            instructions: Some("Create a Discord application, add a bot, enable the intents you need, and invite it to the target server.".to_string()),
            setup_url: Some("https://discord.com/developers/applications".to_string()),
            validation_checks: vec![ChannelValidationCheck {
                kind: "http_get".to_string(),
                url_template: Some("https://discord.com/api/v10/users/@me".to_string()),
                message: Some("Verify that the supplied Discord bot token can authenticate.".to_string()),
            }],
            config_fields: vec![
                ChannelConfigField {
                    key: "channel_id".to_string(),
                    label: Some("Channel ID".to_string()),
                    field_type: "text".to_string(),
                    prompt: Some("Discord channel ID to connect Turin to".to_string()),
                    help: Some("Enable developer mode in Discord to copy the channel ID.".to_string()),
                    required: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "channel_id".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "workspace_id".to_string(),
                    label: Some("Workspace ID".to_string()),
                    field_type: "text".to_string(),
                    default: Some(serde_json::json!("discord")),
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "workspace_id".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "session_scope".to_string(),
                    label: Some("Session Scope".to_string()),
                    field_type: "select".to_string(),
                    default: Some(serde_json::json!("thread")),
                    options: vec![
                        ChannelConfigFieldOption {
                            value: "user".to_string(),
                            label: Some("Per user".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "thread".to_string(),
                            label: Some("Per thread".to_string()),
                        },
                    ],
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "session_scope".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
            ],
        }),
        install: Some(ChannelInstallManifest {
            binary_name: Some("turin-channel-discord".to_string()),
        }),
    }
}

impl DiscordChannelDriverConfig {
    pub fn from_settings(settings: &serde_json::Value) -> Result<Self> {
        let settings = parse_settings(settings)?;
        let token_env = settings.token_env.as_str();
        let token = std::env::var(token_env).map_err(|_| {
            anyhow!(
                "[discord_auth_missing_token] Discord bot token env var '{}' is not set for channel adapter",
                token_env
            )
        })?;

        Ok(Self {
            base_url: settings.base_url,
            gateway_url: settings.gateway_url,
            transport_mode: settings.transport_mode,
            gateway_intents: settings.gateway_intents,
            workspace_id: settings.workspace_id,
            room_id: settings.room_id,
            channel_id: settings.channel_id,
            token,
            poll_interval: Duration::from_millis(settings.poll_interval_ms),
            max_messages_per_poll: settings.max_messages_per_poll,
            start_from_latest: settings.start_from_latest,
            ignore_bot_messages: settings.ignore_bot_messages,
            session_scope: settings.session_scope,
        })
    }
}

#[derive(Debug, Clone)]
struct DiscordChannelSettings {
    token_env: String,
    base_url: String,
    gateway_url: String,
    transport_mode: DiscordTransportMode,
    gateway_intents: u64,
    workspace_id: String,
    room_id: Option<String>,
    channel_id: String,
    poll_interval_ms: u64,
    max_messages_per_poll: u16,
    start_from_latest: bool,
    ignore_bot_messages: bool,
    session_scope: ChannelSessionScope,
}

fn parse_settings(settings: &serde_json::Value) -> Result<DiscordChannelSettings> {
    let settings = settings
        .as_object()
        .ok_or_else(|| anyhow!("Discord channel settings must be a JSON object"))?;

    let token_env = read_required_non_empty_string(
        settings,
        "token_env",
        "[discord_config_missing_token_env] Discord channel setting 'token_env' is required",
        "[discord_config_invalid_token_env] Discord channel setting 'token_env' must not be empty",
    )?
    .to_string();
    let channel_id = read_required_non_empty_string(
        settings,
        "channel_id",
        "[discord_config_missing_channel_id] Discord channel setting 'channel_id' is required",
        "[discord_config_invalid_channel_id] Discord channel setting 'channel_id' must not be empty",
    )?
    .to_string();

    let poll_interval_ms = match settings.get("poll_interval_ms") {
        None => 1_000,
        Some(value) => {
            let interval = value.as_u64().ok_or_else(|| {
                anyhow!(
                    "[discord_config_invalid_poll_interval] Discord channel setting 'poll_interval_ms' must be a positive integer"
                )
            })?;
            if interval < 100 {
                anyhow::bail!(
                    "[discord_config_invalid_poll_interval] Discord channel setting 'poll_interval_ms' must be >= 100"
                );
            }
            interval
        }
    };

    let max_messages_per_poll = match settings.get("max_messages_per_poll") {
        None => 25,
        Some(value) => {
            let max = value.as_u64().ok_or_else(|| {
                anyhow!(
                    "[discord_config_invalid_max_messages] Discord channel setting 'max_messages_per_poll' must be a positive integer"
                )
            })?;
            if !(1..=100).contains(&max) {
                anyhow::bail!(
                    "[discord_config_invalid_max_messages] Discord channel setting 'max_messages_per_poll' must be in 1..=100"
                );
            }
            max as u16
        }
    };

    let gateway_intents = match settings.get("gateway_intents") {
        None => DEFAULT_GATEWAY_INTENTS,
        Some(value) => {
            let intents = value.as_u64().ok_or_else(|| {
                anyhow!(
                    "[discord_config_invalid_gateway_intents] Discord channel setting 'gateway_intents' must be a positive integer"
                )
            })?;
            if intents == 0 {
                anyhow::bail!(
                    "[discord_config_invalid_gateway_intents] Discord channel setting 'gateway_intents' must be > 0"
                );
            }
            intents
        }
    };

    Ok(DiscordChannelSettings {
        token_env,
        base_url: read_optional_non_empty_string(settings, "base_url", DEFAULT_BASE_URL)?
            .trim_end_matches('/')
            .to_string(),
        gateway_url: read_optional_non_empty_string(settings, "gateway_url", DEFAULT_GATEWAY_URL)?
            .to_string(),
        transport_mode: parse_transport_mode(
            settings
                .get("transport")
                .map(|value| {
                    value.as_str().ok_or_else(|| {
                        anyhow!(
                            "[discord_config_invalid_transport] Discord channel setting 'transport' must be a string"
                        )
                    })
                })
                .transpose()?,
        )?,
        gateway_intents,
        workspace_id: read_optional_non_empty_string(settings, "workspace_id", "discord")?
            .to_string(),
        room_id: read_optional_string(settings, "room_id")?,
        channel_id,
        poll_interval_ms,
        max_messages_per_poll,
        start_from_latest: read_optional_bool(settings, "start_from_latest", true)?,
        ignore_bot_messages: read_optional_bool(settings, "ignore_bot_messages", true)?,
        session_scope: read_discord_session_scope(settings.get("session_scope"))?,
    })
}

fn read_required_non_empty_string<'a>(
    settings: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
    missing_message: &str,
    empty_message: &str,
) -> Result<&'a str> {
    let value = settings
        .get(key)
        .and_then(|value| value.as_str())
        .ok_or_else(|| anyhow!(missing_message.to_string()))?;
    if value.trim().is_empty() {
        anyhow::bail!(empty_message.to_string());
    }
    Ok(value)
}

fn read_optional_non_empty_string<'a>(
    settings: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
    default: &'a str,
) -> Result<&'a str> {
    match settings.get(key) {
        None => Ok(default),
        Some(value) => {
            let text = value
                .as_str()
                .ok_or_else(|| anyhow!("Discord channel setting '{}' must be a string", key))?;
            if text.trim().is_empty() {
                anyhow::bail!("Discord channel setting '{}' must not be empty", key);
            }
            Ok(text)
        }
    }
}

fn read_optional_string(
    settings: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<Option<String>> {
    match settings.get(key) {
        None => Ok(None),
        Some(value) => {
            let text = value
                .as_str()
                .ok_or_else(|| anyhow!("Discord channel setting '{}' must be a string", key))?;
            if text.trim().is_empty() {
                anyhow::bail!("Discord channel setting '{}' must not be empty", key);
            }
            Ok(Some(text.to_string()))
        }
    }
}

fn read_optional_bool(
    settings: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    default: bool,
) -> Result<bool> {
    match settings.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| anyhow!("Discord channel setting '{}' must be a boolean", key)),
    }
}

fn read_discord_session_scope(value: Option<&serde_json::Value>) -> Result<ChannelSessionScope> {
    let Some(value) = value else {
        return Ok(ChannelSessionScope::User);
    };
    let scope = value.as_str().ok_or_else(|| {
        anyhow!(
            "[discord_config_invalid_session_scope] Discord channel setting 'session_scope' must be a string"
        )
    })?;
    match scope.trim().to_ascii_lowercase().as_str() {
        "user" => Ok(ChannelSessionScope::User),
        "thread" => Ok(ChannelSessionScope::Thread),
        _ => anyhow::bail!(
            "[discord_config_invalid_session_scope] Discord channel setting 'session_scope' must be one of: user, thread"
        ),
    }
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
            text: message.content,
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
struct LocalAttachmentRef {
    name: String,
    path: PathBuf,
    content_type: Option<String>,
}

#[derive(Debug, Clone)]
struct PreparedLocalFile {
    name: String,
    content_type: Option<String>,
    bytes: Vec<u8>,
}

#[derive(Debug, Clone)]
struct DiscordSendMessage {
    content: Option<String>,
    embeds: Vec<serde_json::Value>,
    components: Vec<serde_json::Value>,
    files: Vec<LocalAttachmentRef>,
}

fn parse_transport_mode(raw: Option<&str>) -> Result<DiscordTransportMode> {
    match raw.unwrap_or("gateway") {
        "gateway" => Ok(DiscordTransportMode::Gateway),
        "polling" => Ok(DiscordTransportMode::Polling),
        other => anyhow::bail!(
            "[discord_config_invalid_transport] Invalid Discord transport '{}'; expected 'gateway' or 'polling'",
            other
        ),
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

fn parse_snowflake(value: &str) -> Option<u64> {
    value.parse::<u64>().ok()
}

fn is_newer_snowflake(candidate: &str, current: &str) -> bool {
    match (parse_snowflake(candidate), parse_snowflake(current)) {
        (Some(candidate), Some(current)) => candidate > current,
        _ => candidate > current,
    }
}

fn render_outbound_messages(message: OutboundMessage) -> Vec<DiscordSendMessage> {
    let mut text_chunks = split_for_discord_content(render_text_blocks(&message.blocks));
    let mut embeds = message.embeds;
    if embeds.is_empty() {
        embeds = extract_embeds_from_metadata(&message.metadata);
    }
    let mut components = extract_components_from_metadata(&message.metadata);
    if components.is_empty() {
        components = message.components;
    }

    let mut local_files = Vec::new();
    let mut remote_attachment_urls = Vec::new();
    for attachment in message.attachments {
        if let Some(local_path) = attachment.local_path {
            local_files.push(LocalAttachmentRef {
                name: attachment.name,
                path: PathBuf::from(local_path),
                content_type: attachment.content_type,
            });
            continue;
        }
        if let Some(url) = attachment.url {
            remote_attachment_urls.push(url);
        }
    }
    if !remote_attachment_urls.is_empty() {
        let urls = remote_attachment_urls.join("\n");
        if !urls.trim().is_empty() {
            text_chunks.extend(split_for_discord_content(urls));
        }
    }

    let mut embed_queue: VecDeque<serde_json::Value> = embeds.into_iter().collect();
    let mut file_queue: VecDeque<LocalAttachmentRef> = local_files.into_iter().collect();
    let mut text_queue: VecDeque<String> = text_chunks.into_iter().collect();
    let mut output = Vec::new();
    let mut first = true;

    while !text_queue.is_empty() || !embed_queue.is_empty() || !file_queue.is_empty() || first {
        let content = text_queue.pop_front();
        let mut embeds_for_message = Vec::new();
        while embeds_for_message.len() < DISCORD_EMBEDS_MAX {
            let Some(embed) = embed_queue.pop_front() else {
                break;
            };
            embeds_for_message.push(embed);
        }

        let mut files_for_message = Vec::new();
        while files_for_message.len() < DISCORD_FILES_MAX {
            let Some(file) = file_queue.pop_front() else {
                break;
            };
            files_for_message.push(file);
        }

        let components_for_message = if first {
            components.clone()
        } else {
            Vec::new()
        };

        if content.is_none()
            && embeds_for_message.is_empty()
            && files_for_message.is_empty()
            && components_for_message.is_empty()
        {
            break;
        }

        output.push(DiscordSendMessage {
            content,
            embeds: embeds_for_message,
            components: components_for_message,
            files: files_for_message,
        });
        first = false;
    }

    if output.is_empty() {
        output.push(DiscordSendMessage {
            content: Some("(no output)".to_string()),
            embeds: Vec::new(),
            components: Vec::new(),
            files: Vec::new(),
        });
    }

    output
}

fn render_text_blocks(blocks: &[MessageBlock]) -> String {
    let mut chunks = Vec::new();
    for block in blocks {
        match block {
            MessageBlock::Text { text } => {
                if !text.trim().is_empty() {
                    chunks.push(text.clone());
                }
            }
            MessageBlock::CodeBlock { language, code } => {
                let prefix = language.clone().unwrap_or_default();
                chunks.push(format!("```{}\n{}\n```", prefix, code));
            }
        }
    }
    chunks.join("\n\n")
}

fn split_for_discord_content(content: String) -> Vec<String> {
    let mut out = Vec::new();
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return out;
    }

    let mut current = String::new();
    for line in trimmed.lines() {
        if line.chars().count() > DISCORD_CONTENT_MAX_LEN {
            if !current.is_empty() {
                out.push(current.clone());
                current.clear();
            }
            let mut segment = String::new();
            for ch in line.chars() {
                segment.push(ch);
                if segment.chars().count() >= DISCORD_CONTENT_MAX_LEN {
                    out.push(segment.clone());
                    segment.clear();
                }
            }
            if !segment.is_empty() {
                out.push(segment);
            }
            continue;
        }

        let tentative = if current.is_empty() {
            line.to_string()
        } else {
            format!("{current}\n{line}")
        };
        if tentative.chars().count() > DISCORD_CONTENT_MAX_LEN {
            if !current.is_empty() {
                out.push(current.clone());
            }
            current = line.to_string();
        } else {
            current = tentative;
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn extract_embeds_from_metadata(
    metadata: &serde_json::Map<String, serde_json::Value>,
) -> Vec<serde_json::Value> {
    metadata
        .get("discord_embeds")
        .or_else(|| metadata.get("embeds"))
        .and_then(|value| value.as_array())
        .map(|entries| {
            entries
                .iter()
                .filter(|entry| entry.is_object())
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

fn extract_components_from_metadata(
    metadata: &serde_json::Map<String, serde_json::Value>,
) -> Vec<serde_json::Value> {
    metadata
        .get("discord_components")
        .or_else(|| metadata.get("components"))
        .and_then(|value| value.as_array())
        .map(|entries| {
            entries
                .iter()
                .filter(|entry| entry.is_object())
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

fn discord_payload_from_message(message: &DiscordSendMessage) -> serde_json::Value {
    let mut payload = serde_json::Map::new();
    if let Some(content) = &message.content {
        payload.insert(
            "content".to_string(),
            serde_json::Value::String(content.clone()),
        );
    }
    if !message.embeds.is_empty() {
        payload.insert(
            "embeds".to_string(),
            serde_json::Value::Array(message.embeds.clone()),
        );
    }
    if !message.components.is_empty() {
        payload.insert(
            "components".to_string(),
            serde_json::Value::Array(message.components.clone()),
        );
    }
    serde_json::Value::Object(payload)
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
mod tests {
    use super::*;

    #[test]
    fn parse_transport_mode_defaults_to_gateway() {
        assert_eq!(
            parse_transport_mode(None).expect("default transport should parse"),
            DiscordTransportMode::Gateway
        );
    }

    #[test]
    fn parse_transport_mode_accepts_polling() {
        assert_eq!(
            parse_transport_mode(Some("polling")).expect("polling transport should parse"),
            DiscordTransportMode::Polling
        );
    }

    #[test]
    fn parse_transport_mode_rejects_invalid_value() {
        let error = parse_transport_mode(Some("unknown")).expect_err("transport should fail");
        assert!(error.to_string().contains("Invalid Discord transport"));
    }

    #[test]
    fn validate_settings_rejects_small_poll_interval() {
        let error = validate_settings(&serde_json::json!({
            "token_env": "DISCORD_TOKEN",
            "channel_id": "123",
            "poll_interval_ms": 10
        }))
        .expect_err("too-small poll interval should fail");
        assert!(error.to_string().contains("poll_interval_ms"));
    }

    #[test]
    fn validate_settings_rejects_zero_gateway_intents() {
        let error = validate_settings(&serde_json::json!({
            "token_env": "DISCORD_TOKEN",
            "channel_id": "123",
            "gateway_intents": 0
        }))
        .expect_err("zero gateway intents should fail");
        assert!(error.to_string().contains("gateway_intents"));
    }

    #[test]
    fn validate_settings_rejects_unsupported_room_session_scope() {
        let error = validate_settings(&serde_json::json!({
            "token_env": "DISCORD_TOKEN",
            "channel_id": "123",
            "session_scope": "room"
        }))
        .expect_err("room scope rejected for discord");
        assert!(error.to_string().contains("session_scope"));
    }

    #[test]
    fn render_outbound_preserves_code_blocks() {
        let batch = render_outbound_messages(OutboundMessage {
            blocks: vec![
                MessageBlock::Text {
                    text: "summary".to_string(),
                },
                MessageBlock::CodeBlock {
                    language: Some("rust".to_string()),
                    code: "fn main() {}".to_string(),
                },
            ],
            ..OutboundMessage::default()
        });
        assert_eq!(batch.len(), 1);
        let output = batch[0]
            .content
            .as_ref()
            .expect("first message should contain content");
        assert!(output.contains("summary"));
        assert!(output.contains("```rust"));
    }

    #[test]
    fn render_outbound_includes_embeds_and_components() {
        let batch = render_outbound_messages(OutboundMessage {
            blocks: vec![MessageBlock::Text {
                text: "summary".to_string(),
            }],
            embeds: vec![serde_json::json!({ "title": "Build Summary" })],
            components: vec![serde_json::json!({ "type": 1, "components": [] })],
            ..OutboundMessage::default()
        });
        assert_eq!(batch.len(), 1);
        assert_eq!(batch[0].embeds.len(), 1);
        assert_eq!(batch[0].components.len(), 1);
    }

    #[test]
    fn render_outbound_splits_long_content() {
        let long_text = "a".repeat(DISCORD_CONTENT_MAX_LEN + 200);
        let batch = render_outbound_messages(OutboundMessage {
            blocks: vec![MessageBlock::Text { text: long_text }],
            ..OutboundMessage::default()
        });
        assert!(
            batch.len() >= 2,
            "long payload should be split into multiple outbound messages"
        );
        assert!(batch.iter().all(|entry| {
            entry
                .content
                .as_ref()
                .map(|text| text.chars().count())
                .unwrap_or(0)
                <= DISCORD_CONTENT_MAX_LEN
        }));
    }

    #[test]
    fn normalize_ignores_bot_messages() {
        let config = DiscordChannelDriverConfig {
            base_url: DEFAULT_BASE_URL.to_string(),
            gateway_url: DEFAULT_GATEWAY_URL.to_string(),
            transport_mode: DiscordTransportMode::Gateway,
            gateway_intents: DEFAULT_GATEWAY_INTENTS,
            workspace_id: "discord".to_string(),
            room_id: None,
            channel_id: "123".to_string(),
            token: "token".to_string(),
            poll_interval: Duration::from_millis(250),
            max_messages_per_poll: 10,
            start_from_latest: true,
            ignore_bot_messages: true,
            session_scope: ChannelSessionScope::User,
        };
        let (_tx, rx) = watch::channel(false);
        let mut driver = DiscordChannelDriver {
            channel_runtime_id: "discord-runtime".to_string(),
            config,
            client: reqwest::Client::new(),
            shutdown_rx: rx,
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
        };
        let message = DiscordMessage {
            id: "1".to_string(),
            channel_id: "123".to_string(),
            guild_id: Some("guild".to_string()),
            content: "hello".to_string(),
            author: DiscordAuthor {
                id: "bot".to_string(),
                username: "bot".to_string(),
                global_name: None,
                bot: Some(true),
            },
            attachments: Vec::new(),
        };
        assert!(driver.normalize_message(message).is_none());
    }

    #[test]
    fn normalize_dedupes_message_ids() {
        let config = DiscordChannelDriverConfig {
            base_url: DEFAULT_BASE_URL.to_string(),
            gateway_url: DEFAULT_GATEWAY_URL.to_string(),
            transport_mode: DiscordTransportMode::Gateway,
            gateway_intents: DEFAULT_GATEWAY_INTENTS,
            workspace_id: "discord".to_string(),
            room_id: None,
            channel_id: "123".to_string(),
            token: "token".to_string(),
            poll_interval: Duration::from_millis(250),
            max_messages_per_poll: 10,
            start_from_latest: true,
            ignore_bot_messages: false,
            session_scope: ChannelSessionScope::User,
        };
        let (_tx, rx) = watch::channel(false);
        let mut driver = DiscordChannelDriver {
            channel_runtime_id: "discord-runtime".to_string(),
            config,
            client: reqwest::Client::new(),
            shutdown_rx: rx,
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
        };
        let message = DiscordMessage {
            id: "dup".to_string(),
            channel_id: "123".to_string(),
            guild_id: Some("guild".to_string()),
            content: "hello".to_string(),
            author: DiscordAuthor {
                id: "user".to_string(),
                username: "user".to_string(),
                global_name: None,
                bot: Some(false),
            },
            attachments: Vec::new(),
        };

        assert!(driver.normalize_message(message.clone()).is_some());
        assert!(driver.normalize_message(message).is_none());
    }

    #[test]
    fn normalize_thread_scope_shares_channel_across_users() {
        let config = DiscordChannelDriverConfig {
            base_url: DEFAULT_BASE_URL.to_string(),
            gateway_url: DEFAULT_GATEWAY_URL.to_string(),
            transport_mode: DiscordTransportMode::Gateway,
            gateway_intents: DEFAULT_GATEWAY_INTENTS,
            workspace_id: "discord".to_string(),
            room_id: None,
            channel_id: "123".to_string(),
            token: "token".to_string(),
            poll_interval: Duration::from_millis(250),
            max_messages_per_poll: 10,
            start_from_latest: true,
            ignore_bot_messages: false,
            session_scope: ChannelSessionScope::Thread,
        };
        let (_tx, rx) = watch::channel(false);
        let mut driver = DiscordChannelDriver {
            channel_runtime_id: "discord-runtime".to_string(),
            config,
            client: reqwest::Client::new(),
            shutdown_rx: rx,
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
        };
        let message = DiscordMessage {
            id: "1".to_string(),
            channel_id: "123".to_string(),
            guild_id: Some("guild".to_string()),
            content: "hello".to_string(),
            author: DiscordAuthor {
                id: "user".to_string(),
                username: "user".to_string(),
                global_name: None,
                bot: Some(false),
            },
            attachments: Vec::new(),
        };

        let event = driver.normalize_message(message).expect("normalized event");
        assert_eq!(event.session_scope, ChannelSessionScope::Thread);
        assert_eq!(event.conversation.thread_id, "123");
        assert_eq!(event.conversation.user_id, None);
    }

    #[test]
    fn adapter_manifest_exposes_discord_enum_settings() {
        let manifest = adapter_manifest();
        assert_eq!(manifest.kind, "discord");
        assert_eq!(
            manifest
                .enum_setting("session_scope")
                .expect("session scope setting")
                .options,
            vec!["user", "thread"]
        );
    }
}
