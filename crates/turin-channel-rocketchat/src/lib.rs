use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use std::collections::{HashSet, VecDeque};
use std::time::Duration;
use tokio::sync::watch;
use tokio::time::sleep;
use tracing::warn;
use turin_channel_core::{
    ChannelAdapterManifest, ChannelAttachment, ChannelAuthFlowPollRequest,
    ChannelAuthFlowPollResponse, ChannelAuthFlowStartRequest, ChannelAuthFlowStartResponse,
    ChannelCapabilities, ChannelConfigField, ChannelConfigFieldOption, ChannelConfigTarget,
    ChannelConfigTargetKind, ChannelConversationKey, ChannelEnumSetting, ChannelIdentitySelectors,
    ChannelInstallManifest, ChannelKind, ChannelMessageRef, ChannelRuntimeCapabilities,
    ChannelRuntimeManifest, ChannelSecretRequirement, ChannelSessionScope, ChannelSetupManifest,
    ChannelUser, InboundEvent, MessageBlock, OutboundMessage,
};
use turin_channel_runner::ChannelDriver;

const DEFAULT_BASE_URL: &str = "http://localhost:3000";
const DEFAULT_POLL_INTERVAL_MS: u64 = 1_000;
const DEFAULT_MAX_MESSAGES_PER_POLL: u16 = 50;
const MAX_MESSAGES_PER_POLL: u16 = 100;
const ROCKETCHAT_MESSAGE_MAX_LEN: usize = 4_000;
const SEEN_MESSAGE_IDS_LIMIT: usize = 1_024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RocketChatRespondMode {
    All,
    Mentions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RocketChatRoomType {
    Channel,
    PrivateGroup,
    DirectMessage,
}

#[derive(Debug, Clone)]
pub struct RocketChatChannelDriverConfig {
    pub base_url: String,
    pub workspace_id: String,
    pub room_id: Option<String>,
    pub room_name: Option<String>,
    pub user_id: String,
    pub token: String,
    pub poll_interval: Duration,
    pub max_messages_per_poll: u16,
    pub start_from_latest: bool,
    pub ignore_bot_messages: bool,
    pub respond_mode: RocketChatRespondMode,
    pub session_scope: ChannelSessionScope,
}

#[derive(Debug, Clone)]
struct RocketChatChannelSettings {
    token_env: String,
    base_url: String,
    workspace_id: String,
    room_id: Option<String>,
    room_name: Option<String>,
    user_id: String,
    poll_interval_ms: u64,
    max_messages_per_poll: u16,
    start_from_latest: bool,
    ignore_bot_messages: bool,
    respond_mode: RocketChatRespondMode,
    session_scope: ChannelSessionScope,
}

#[derive(Debug, Clone)]
struct RocketChatResolvedRoom {
    id: String,
    room_type: RocketChatRoomType,
    latest_message_id: Option<String>,
    latest_message_ts: Option<String>,
}

pub fn validate_settings(settings: &serde_json::Value) -> Result<()> {
    parse_settings(settings).map(|_| ())
}

pub fn start_auth_flow(
    _request: &ChannelAuthFlowStartRequest,
) -> Result<ChannelAuthFlowStartResponse> {
    anyhow::bail!("Rocket.Chat does not expose manifest auth flows")
}

pub fn poll_auth_flow(
    _request: &ChannelAuthFlowPollRequest,
) -> Result<ChannelAuthFlowPollResponse> {
    anyhow::bail!("Rocket.Chat does not expose manifest auth flows")
}

pub fn adapter_manifest() -> ChannelAdapterManifest {
    ChannelAdapterManifest {
        protocol_version: turin_channel_core::CHANNEL_ADAPTER_PROTOCOL_VERSION,
        kind: "rocketchat".to_string(),
        display_name: "Rocket.Chat".to_string(),
        runtime: ChannelRuntimeManifest {
            session_scopes: vec!["user".to_string(), "thread".to_string(), "room".to_string()],
            enum_settings: vec![
                ChannelEnumSetting {
                    key: "respond_mode".to_string(),
                    options: vec!["all".to_string(), "mentions".to_string()],
                },
                ChannelEnumSetting {
                    key: "session_scope".to_string(),
                    options: vec!["user".to_string(), "thread".to_string(), "room".to_string()],
                },
            ],
            capabilities: ChannelRuntimeCapabilities {
                dm: true,
                groups: true,
                threads: true,
                attachments: false,
                streaming: false,
            },
            identity_selectors: ChannelIdentitySelectors {
                matching_rules: vec!["id".to_string(), "username".to_string()],
                examples: vec![
                    "rbAXPnMktTFbNpwtJ".to_string(),
                    "rocket.cat".to_string(),
                ],
            },
        },
        setup: Some(ChannelSetupManifest {
            required_secrets: vec![ChannelSecretRequirement {
                name: "rocketchat_auth_token".to_string(),
                env_var: "ROCKETCHAT_AUTH_TOKEN".to_string(),
                display_name: Some("Rocket.Chat auth token".to_string()),
                help: Some(
                    "Create a personal access token or a bot auth token in your Rocket.Chat workspace."
                        .to_string(),
                ),
                optional: false,
                hints: vec!["Looks like RScctEHSmLGZGywfIhWyRpyofhKOiMoUIpimhvheU3f".to_string()],
                target: Some(ChannelConfigTarget {
                    kind: ChannelConfigTargetKind::ChannelSetting,
                    name: "token_env".to_string(),
                }),
                validate: None,
            }],
            instructions: Some(
                "Create or choose a Rocket.Chat bot/user, copy its auth token and user ID, then point Turin at the target room."
                    .to_string(),
            ),
            setup_url: Some("https://developer.rocket.chat/apidocs".to_string()),
            validation_checks: vec![],
            config_fields: vec![
                ChannelConfigField {
                    key: "base_url".to_string(),
                    label: Some("Server URL".to_string()),
                    field_type: "text".to_string(),
                    prompt: Some("Rocket.Chat workspace base URL".to_string()),
                    help: Some("Example: https://chat.example.com".to_string()),
                    default: Some(serde_json::json!(DEFAULT_BASE_URL)),
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "base_url".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "user_id".to_string(),
                    label: Some("User ID".to_string()),
                    field_type: "text".to_string(),
                    prompt: Some("Rocket.Chat user ID for the bot/user token".to_string()),
                    help: Some(
                        "Rocket.Chat requires both X-Auth-Token and X-User-Id headers for API requests."
                            .to_string(),
                    ),
                    required: true,
                    hint: Some("Looks like rbAXPnMktTFbNpwtJ".to_string()),
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "user_id".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "room_id".to_string(),
                    label: Some("Room ID".to_string()),
                    field_type: "text".to_string(),
                    prompt: Some("Rocket.Chat room ID to connect Turin to".to_string()),
                    help: Some(
                        "Use the room ID for the public channel, private group, or DM that Turin should monitor."
                            .to_string(),
                    ),
                    required: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "room_id".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "workspace_id".to_string(),
                    label: Some("Workspace ID".to_string()),
                    field_type: "text".to_string(),
                    default: Some(serde_json::json!("rocketchat")),
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "workspace_id".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "respond_mode".to_string(),
                    label: Some("Respond Mode".to_string()),
                    field_type: "select".to_string(),
                    prompt: Some("When should Turin respond in shared rooms?".to_string()),
                    help: Some("Direct messages always go through; mentions mode is safer for channels and groups.".to_string()),
                    default: Some(serde_json::json!("mentions")),
                    options: vec![
                        ChannelConfigFieldOption {
                            value: "all".to_string(),
                            label: Some("Every message".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "mentions".to_string(),
                            label: Some("Mentions only".to_string()),
                        },
                    ],
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "respond_mode".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "session_scope".to_string(),
                    label: Some("Session Scope".to_string()),
                    field_type: "select".to_string(),
                    prompt: Some("How should Rocket.Chat conversation memory be scoped?".to_string()),
                    help: Some("Thread scope keeps each thread isolated; room shares one memory for the whole room.".to_string()),
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
                        ChannelConfigFieldOption {
                            value: "room".to_string(),
                            label: Some("Per room".to_string()),
                        },
                    ],
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "session_scope".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "allowed_users".to_string(),
                    label: Some("Allowed Users".to_string()),
                    field_type: "string_list".to_string(),
                    prompt: Some("Optional usernames or IDs allowed to interact".to_string()),
                    help: Some("Leave empty to allow any user in the configured room.".to_string()),
                    advanced: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "allowed_users".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "banned_users".to_string(),
                    label: Some("Banned Users".to_string()),
                    field_type: "string_list".to_string(),
                    prompt: Some("Optional usernames or IDs that should always be denied".to_string()),
                    advanced: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "banned_users".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "poll_interval_ms".to_string(),
                    label: Some("Poll Interval (ms)".to_string()),
                    field_type: "number".to_string(),
                    default: Some(serde_json::json!(DEFAULT_POLL_INTERVAL_MS)),
                    advanced: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "poll_interval_ms".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "start_from_latest".to_string(),
                    label: Some("Start From Latest".to_string()),
                    field_type: "boolean".to_string(),
                    help: Some("Skip older room history and only process new messages from now on.".to_string()),
                    default: Some(serde_json::json!(true)),
                    advanced: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "start_from_latest".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "ignore_bot_messages".to_string(),
                    label: Some("Ignore Bot Messages".to_string()),
                    field_type: "boolean".to_string(),
                    default: Some(serde_json::json!(true)),
                    advanced: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "ignore_bot_messages".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
            ],
            auth_flows: vec![],
        }),
        install: Some(ChannelInstallManifest {
            binary_name: Some("turin-channel-rocketchat".to_string()),
        }),
    }
}

impl RocketChatChannelDriverConfig {
    pub fn from_settings(settings: &serde_json::Value) -> Result<Self> {
        let settings = parse_settings(settings)?;
        let token = std::env::var(&settings.token_env).map_err(|_| {
            anyhow!(
                "[rocketchat_auth_missing_token] Rocket.Chat auth token env var '{}' is not set for channel adapter",
                settings.token_env
            )
        })?;

        Ok(Self {
            base_url: settings.base_url,
            workspace_id: settings.workspace_id,
            room_id: settings.room_id,
            room_name: settings.room_name,
            user_id: settings.user_id,
            token,
            poll_interval: Duration::from_millis(settings.poll_interval_ms),
            max_messages_per_poll: settings.max_messages_per_poll,
            start_from_latest: settings.start_from_latest,
            ignore_bot_messages: settings.ignore_bot_messages,
            respond_mode: settings.respond_mode,
            session_scope: settings.session_scope,
        })
    }
}

pub struct RocketChatChannelDriver {
    channel_id: String,
    client: Client,
    config: RocketChatChannelDriverConfig,
    shutdown_rx: watch::Receiver<bool>,
    room: RocketChatResolvedRoom,
    backlog: VecDeque<InboundEvent>,
    seen_message_ids: HashSet<String>,
    seen_message_order: VecDeque<String>,
    cursor_ts: Option<String>,
}

impl RocketChatChannelDriver {
    pub async fn from_settings(
        channel_id: impl Into<String>,
        settings: &serde_json::Value,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Result<Self> {
        let config = RocketChatChannelDriverConfig::from_settings(settings)?;
        let client = Client::builder().build()?;
        let room = fetch_room_info(&client, &config).await?;

        let mut driver = Self {
            channel_id: channel_id.into(),
            client,
            config,
            shutdown_rx,
            room,
            backlog: VecDeque::new(),
            seen_message_ids: HashSet::new(),
            seen_message_order: VecDeque::new(),
            cursor_ts: None,
        };

        if driver.config.start_from_latest {
            if let Some(message_id) = driver.room.latest_message_id.clone() {
                driver.remember_message_id(message_id);
            }
            driver.cursor_ts = driver.room.latest_message_ts.clone();
        }

        Ok(driver)
    }

    async fn poll_messages(&mut self) -> Result<()> {
        let messages = fetch_room_messages(
            &self.client,
            &self.config,
            &self.room,
            self.cursor_ts.as_deref(),
        )
        .await?;

        for message in messages {
            self.cursor_ts = Some(message.ts.clone());
            if self.seen_message_ids.contains(&message.id) {
                continue;
            }
            self.remember_message_id(message.id.clone());

            let Some(event) = self.message_to_event(message)? else {
                continue;
            };
            self.backlog.push_back(event);
        }

        Ok(())
    }

    fn message_to_event(&self, message: RocketChatMessage) -> Result<Option<InboundEvent>> {
        if message.kind.is_some() {
            return Ok(None);
        }

        let user = message.user.as_ref().ok_or_else(|| {
            anyhow!(
                "[rocketchat_message_missing_user] Rocket.Chat message '{}' is missing user metadata",
                message.id
            )
        })?;

        if self.config.ignore_bot_messages && user.id == self.config.user_id {
            return Ok(None);
        }

        if !self.should_accept_message(&message, user) {
            return Ok(None);
        }

        let mut text = message.text.clone().unwrap_or_default();
        let attachments = collect_attachments(&self.config.base_url, &message);
        if text.trim().is_empty() && attachments.is_empty() {
            return Ok(None);
        }
        if text.trim().is_empty() && !attachments.is_empty() {
            text = "[Attachment]".to_string();
        }

        let user = ChannelUser {
            id: user.id.clone(),
            display_name: user.name.clone(),
            username: user.username.clone(),
        };
        let conversation = ChannelConversationKey {
            channel: ChannelKind::new("rocketchat"),
            workspace_id: self.config.workspace_id.clone(),
            room_id: Some(self.room.id.clone()),
            thread_id: self.thread_id_for_message(&message),
            user_id: if matches!(self.config.session_scope, ChannelSessionScope::User) {
                Some(user.id.clone())
            } else {
                None
            },
        };

        let mut metadata = serde_json::Map::new();
        metadata.insert(
            "rocketchat_message_id".to_string(),
            serde_json::json!(message.id),
        );
        metadata.insert(
            "rocketchat_room_id".to_string(),
            serde_json::json!(self.room.id),
        );
        if let Some(tmid) = message.thread_root_id {
            metadata.insert("rocketchat_thread_id".to_string(), serde_json::json!(tmid));
        }

        Ok(Some(InboundEvent {
            message: ChannelMessageRef {
                conversation: conversation.clone(),
                message_id: metadata["rocketchat_message_id"]
                    .as_str()
                    .expect("message id inserted")
                    .to_string(),
            },
            conversation,
            user,
            session_scope: self.config.session_scope,
            text,
            attachments,
            metadata,
        }))
    }

    fn should_accept_message(
        &self,
        message: &RocketChatMessage,
        user: &RocketChatMessageUser,
    ) -> bool {
        if matches!(self.room.room_type, RocketChatRoomType::DirectMessage) {
            return user.id != self.config.user_id || !self.config.ignore_bot_messages;
        }

        match self.config.respond_mode {
            RocketChatRespondMode::All => true,
            RocketChatRespondMode::Mentions => message
                .mentions
                .iter()
                .any(|mention| mention.id.as_deref() == Some(self.config.user_id.as_str())),
        }
    }

    fn thread_id_for_message(&self, message: &RocketChatMessage) -> String {
        match self.config.session_scope {
            ChannelSessionScope::Room => self.room.id.clone(),
            ChannelSessionScope::Thread => message
                .thread_root_id
                .clone()
                .unwrap_or_else(|| message.id.clone()),
            ChannelSessionScope::User => message
                .thread_root_id
                .clone()
                .unwrap_or_else(|| self.room.id.clone()),
        }
    }

    fn remember_message_id(&mut self, message_id: String) {
        if self.seen_message_ids.insert(message_id.clone()) {
            self.seen_message_order.push_back(message_id);
            while self.seen_message_order.len() > SEEN_MESSAGE_IDS_LIMIT {
                if let Some(oldest) = self.seen_message_order.pop_front() {
                    self.seen_message_ids.remove(&oldest);
                }
            }
        }
    }
}

#[async_trait]
impl ChannelDriver for RocketChatChannelDriver {
    fn kind(&self) -> ChannelKind {
        ChannelKind::new("rocketchat")
    }

    fn user_matches_selector(&self, selector: &str, user: &ChannelUser) -> bool {
        let selector = selector.trim().trim_start_matches('@');
        if selector.is_empty() {
            return false;
        }
        user.id == selector
            || user
                .username
                .as_ref()
                .is_some_and(|username| username.eq_ignore_ascii_case(selector))
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities {
            rich_formatting: false,
            threads: true,
            attachments: false,
            ephemeral_messages: false,
        }
    }

    async fn next_event(&mut self) -> Result<Option<InboundEvent>> {
        loop {
            if *self.shutdown_rx.borrow() {
                return Ok(None);
            }

            if let Some(event) = self.backlog.pop_front() {
                return Ok(Some(event));
            }

            if let Err(err) = self.poll_messages().await {
                warn!(
                    channel_id = %self.channel_id,
                    room_id = %self.room.id,
                    error = %err,
                    "Rocket.Chat polling failed"
                );
            }

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

    async fn send(
        &mut self,
        conversation: &ChannelConversationKey,
        message: OutboundMessage,
    ) -> Result<()> {
        let room_id = conversation
            .room_id
            .as_deref()
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                anyhow!("[rocketchat_send_missing_room] outbound conversation is missing room_id")
            })?;

        let thread_id = if conversation.thread_id != room_id {
            Some(conversation.thread_id.as_str())
        } else {
            None
        };

        for chunk in split_for_rocketchat_content(render_text_blocks(&message.blocks)) {
            let payload = serde_json::json!({
                "roomId": room_id,
                "text": chunk,
                "parseUrls": false,
                "tmid": thread_id,
            });
            let response = self
                .client
                .post(api_url(&self.config.base_url, "chat.postMessage"))
                .header("X-Auth-Token", &self.config.token)
                .header("X-User-Id", &self.config.user_id)
                .json(&payload)
                .send()
                .await
                .context("Failed to send Rocket.Chat message")?;
            if !response.status().is_success() {
                let status = response.status();
                let body = response.text().await.unwrap_or_default();
                anyhow::bail!(
                    "[rocketchat_send_failed] Rocket.Chat chat.postMessage failed with status {}: {}",
                    status,
                    body
                );
            }
        }

        Ok(())
    }

    async fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
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

fn split_for_rocketchat_content(content: String) -> Vec<String> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return vec![" ".to_string()];
    }

    let mut out = Vec::new();
    let mut current = String::new();
    for line in trimmed.lines() {
        if line.chars().count() > ROCKETCHAT_MESSAGE_MAX_LEN {
            if !current.is_empty() {
                out.push(current.clone());
                current.clear();
            }
            let mut segment = String::new();
            for ch in line.chars() {
                segment.push(ch);
                if segment.chars().count() >= ROCKETCHAT_MESSAGE_MAX_LEN {
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
        if tentative.chars().count() > ROCKETCHAT_MESSAGE_MAX_LEN {
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

fn collect_attachments(base_url: &str, message: &RocketChatMessage) -> Vec<ChannelAttachment> {
    let mut attachments = Vec::new();
    if let Some(file) = &message.file {
        attachments.push(ChannelAttachment {
            name: file.name.clone(),
            content_type: file.content_type.clone(),
            url: file.url.as_ref().map(|url| absolute_url(base_url, url)),
            local_path: None,
        });
    }
    for attachment in &message.attachments {
        if let Some(url) = attachment
            .title_link
            .as_ref()
            .or(attachment.image_url.as_ref())
            .or(attachment.audio_url.as_ref())
            .or(attachment.video_url.as_ref())
        {
            attachments.push(ChannelAttachment {
                name: attachment
                    .title
                    .clone()
                    .or_else(|| attachment.text.clone())
                    .unwrap_or_else(|| "attachment".to_string()),
                content_type: None,
                url: Some(absolute_url(base_url, url)),
                local_path: None,
            });
        }
    }
    attachments
}

fn absolute_url(base_url: &str, raw: &str) -> String {
    if raw.starts_with("http://") || raw.starts_with("https://") {
        raw.to_string()
    } else {
        format!(
            "{}/{}",
            base_url.trim_end_matches('/'),
            raw.trim_start_matches('/')
        )
    }
}

fn api_url(base_url: &str, path: &str) -> String {
    format!("{}/api/v1/{}", base_url.trim_end_matches('/'), path)
}

async fn fetch_room_info(
    client: &Client,
    config: &RocketChatChannelDriverConfig,
) -> Result<RocketChatResolvedRoom> {
    let mut request = client
        .get(api_url(&config.base_url, "rooms.info"))
        .header("X-Auth-Token", &config.token)
        .header("X-User-Id", &config.user_id);

    if let Some(room_id) = &config.room_id {
        request = request.query(&[("roomId", room_id)]);
    } else if let Some(room_name) = &config.room_name {
        request = request.query(&[("roomName", room_name)]);
    } else {
        anyhow::bail!(
            "[rocketchat_config_missing_room] Rocket.Chat channel requires 'room_id' or 'room_name'"
        );
    }

    let response = request
        .send()
        .await
        .context("Failed to query Rocket.Chat room info")?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!(
            "[rocketchat_room_info_failed] Rocket.Chat rooms.info failed with status {}: {}",
            status,
            body
        );
    }
    let parsed: RocketChatRoomInfoResponse =
        serde_json::from_str(&body).context("Failed to decode Rocket.Chat room info response")?;

    Ok(RocketChatResolvedRoom {
        id: parsed.room.id,
        room_type: RocketChatRoomType::parse(&parsed.room.kind)?,
        latest_message_id: parsed
            .room
            .last_message
            .as_ref()
            .map(|message| message.id.clone()),
        latest_message_ts: parsed.room.last_message_at.or_else(|| {
            parsed
                .room
                .last_message
                .as_ref()
                .map(|message| message.ts.clone())
        }),
    })
}

async fn fetch_room_messages(
    client: &Client,
    config: &RocketChatChannelDriverConfig,
    room: &RocketChatResolvedRoom,
    cursor_ts: Option<&str>,
) -> Result<Vec<RocketChatMessage>> {
    let endpoint = match room.room_type {
        RocketChatRoomType::Channel => "channels.history",
        RocketChatRoomType::PrivateGroup => "groups.history",
        RocketChatRoomType::DirectMessage => "dm.history",
    };

    let mut request = client
        .get(api_url(&config.base_url, endpoint))
        .header("X-Auth-Token", &config.token)
        .header("X-User-Id", &config.user_id)
        .query(&[("roomId", room.id.as_str())]);

    if let Some(cursor_ts) = cursor_ts {
        request = request.query(&[("oldest", cursor_ts), ("inclusive", "true")]);
        request = request.query(&[("count", config.max_messages_per_poll.to_string())]);
        request = request.query(&[("sort", "{\"ts\":1,\"_id\":1}")]);
        request = request.query(&[("showThreadMessages", "true")]);
    } else {
        request = request.query(&[("count", config.max_messages_per_poll.to_string())]);
        request = request.query(&[("sort", "{\"ts\":-1,\"_id\":-1}")]);
        request = request.query(&[("showThreadMessages", "true")]);
    }

    let response = request
        .send()
        .await
        .context("Failed to query Rocket.Chat room history")?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!(
            "[rocketchat_history_failed] Rocket.Chat history request failed with status {}: {}",
            status,
            body
        );
    }

    let mut parsed: RocketChatHistoryResponse =
        serde_json::from_str(&body).context("Failed to decode Rocket.Chat history response")?;
    parsed
        .messages
        .sort_by(|left, right| left.ts.cmp(&right.ts).then_with(|| left.id.cmp(&right.id)));
    Ok(parsed.messages)
}

fn parse_settings(settings: &serde_json::Value) -> Result<RocketChatChannelSettings> {
    let settings = settings
        .as_object()
        .ok_or_else(|| anyhow!("Rocket.Chat channel settings must be a JSON object"))?;

    let token_env = read_required_non_empty_string(
        settings,
        "token_env",
        "[rocketchat_config_missing_token_env] Rocket.Chat channel setting 'token_env' is required",
        "[rocketchat_config_invalid_token_env] Rocket.Chat channel setting 'token_env' must not be empty",
    )?
    .to_string();
    let user_id = read_required_non_empty_string(
        settings,
        "user_id",
        "[rocketchat_config_missing_user_id] Rocket.Chat channel setting 'user_id' is required",
        "[rocketchat_config_invalid_user_id] Rocket.Chat channel setting 'user_id' must not be empty",
    )?
    .to_string();
    let room_id = read_optional_non_empty_string(
        settings,
        "room_id",
        "[rocketchat_config_invalid_room_id] Rocket.Chat channel setting 'room_id' must not be empty",
    )?
    .map(ToString::to_string);
    let room_name = read_optional_non_empty_string(
        settings,
        "room_name",
        "[rocketchat_config_invalid_room_name] Rocket.Chat channel setting 'room_name' must not be empty",
    )?
    .map(ToString::to_string);

    if room_id.is_none() && room_name.is_none() {
        anyhow::bail!(
            "[rocketchat_config_missing_room] Rocket.Chat channel requires 'room_id' or 'room_name'"
        );
    }

    let base_url = read_optional_non_empty_string(
        settings,
        "base_url",
        "[rocketchat_config_invalid_base_url] Rocket.Chat channel setting 'base_url' must not be empty",
    )?
    .unwrap_or(DEFAULT_BASE_URL)
    .trim_end_matches('/')
    .to_string();

    let poll_interval_ms = read_u64_with_min(
        settings.get("poll_interval_ms"),
        DEFAULT_POLL_INTERVAL_MS,
        100,
        "[rocketchat_config_invalid_poll_interval] Rocket.Chat channel setting 'poll_interval_ms' must be a positive integer >= 100",
    )?;

    let max_messages_per_poll = read_u64_with_min(
        settings.get("max_messages_per_poll"),
        DEFAULT_MAX_MESSAGES_PER_POLL as u64,
        1,
        "[rocketchat_config_invalid_max_messages] Rocket.Chat channel setting 'max_messages_per_poll' must be in 1..=100",
    )?;
    if max_messages_per_poll > MAX_MESSAGES_PER_POLL as u64 {
        anyhow::bail!(
            "[rocketchat_config_invalid_max_messages] Rocket.Chat channel setting 'max_messages_per_poll' must be in 1..=100"
        );
    }

    Ok(RocketChatChannelSettings {
        token_env,
        base_url,
        workspace_id: read_optional_non_empty_string(
            settings,
            "workspace_id",
            "[rocketchat_config_invalid_workspace_id] Rocket.Chat channel setting 'workspace_id' must not be empty",
        )?
        .unwrap_or("rocketchat")
        .to_string(),
        room_id,
        room_name,
        user_id,
        poll_interval_ms,
        max_messages_per_poll: max_messages_per_poll as u16,
        start_from_latest: read_bool(settings.get("start_from_latest"), true, "start_from_latest")?,
        ignore_bot_messages: read_bool(
            settings.get("ignore_bot_messages"),
            true,
            "ignore_bot_messages",
        )?,
        respond_mode: read_respond_mode(settings.get("respond_mode"))?,
        session_scope: read_session_scope(settings.get("session_scope"))?,
    })
}

fn read_required_non_empty_string<'a>(
    settings: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
    missing_message: &str,
    invalid_message: &str,
) -> Result<&'a str> {
    let value = settings
        .get(key)
        .ok_or_else(|| anyhow!("{missing_message}"))?
        .as_str()
        .ok_or_else(|| anyhow!("{invalid_message}"))?;
    if value.trim().is_empty() {
        anyhow::bail!("{invalid_message}");
    }
    Ok(value)
}

fn read_optional_non_empty_string<'a>(
    settings: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
    invalid_message: &str,
) -> Result<Option<&'a str>> {
    match settings.get(key) {
        None => Ok(None),
        Some(value) => {
            let value = value.as_str().ok_or_else(|| anyhow!("{invalid_message}"))?;
            if value.trim().is_empty() {
                anyhow::bail!("{invalid_message}");
            }
            Ok(Some(value))
        }
    }
}

fn read_u64_with_min(
    value: Option<&serde_json::Value>,
    default: u64,
    min: u64,
    invalid_message: &str,
) -> Result<u64> {
    match value {
        None => Ok(default),
        Some(value) => {
            let parsed = value.as_u64().ok_or_else(|| anyhow!("{invalid_message}"))?;
            if parsed < min {
                anyhow::bail!("{invalid_message}");
            }
            Ok(parsed)
        }
    }
}

fn read_bool(value: Option<&serde_json::Value>, default: bool, key: &str) -> Result<bool> {
    match value {
        None => Ok(default),
        Some(value) => value.as_bool().ok_or_else(|| {
            anyhow!(
                "[rocketchat_config_invalid_bool] Rocket.Chat channel setting '{}' must be true or false",
                key
            )
        }),
    }
}

fn read_respond_mode(value: Option<&serde_json::Value>) -> Result<RocketChatRespondMode> {
    let raw = match value {
        None => return Ok(RocketChatRespondMode::Mentions),
        Some(value) => value.as_str().ok_or_else(|| {
            anyhow!(
                "[rocketchat_config_invalid_respond_mode] Rocket.Chat channel setting 'respond_mode' must be a string"
            )
        })?,
    };
    match raw {
        "all" => Ok(RocketChatRespondMode::All),
        "mentions" => Ok(RocketChatRespondMode::Mentions),
        _ => anyhow::bail!(
            "[rocketchat_config_invalid_respond_mode] Rocket.Chat channel setting 'respond_mode' must be one of: all, mentions"
        ),
    }
}

fn read_session_scope(value: Option<&serde_json::Value>) -> Result<ChannelSessionScope> {
    let raw = match value {
        None => return Ok(ChannelSessionScope::Thread),
        Some(value) => value.as_str().ok_or_else(|| {
            anyhow!(
                "[rocketchat_config_invalid_session_scope] Rocket.Chat channel setting 'session_scope' must be a string"
            )
        })?,
    };
    match raw {
        "user" => Ok(ChannelSessionScope::User),
        "thread" => Ok(ChannelSessionScope::Thread),
        "room" => Ok(ChannelSessionScope::Room),
        _ => anyhow::bail!(
            "[rocketchat_config_invalid_session_scope] Rocket.Chat channel setting 'session_scope' must be one of: user, thread, room"
        ),
    }
}

impl RocketChatRoomType {
    fn parse(raw: &str) -> Result<Self> {
        match raw {
            "c" => Ok(Self::Channel),
            "p" => Ok(Self::PrivateGroup),
            "d" => Ok(Self::DirectMessage),
            other => anyhow::bail!(
                "[rocketchat_room_type_unsupported] Rocket.Chat room type '{}' is not supported yet",
                other
            ),
        }
    }
}

#[derive(Debug, Deserialize)]
struct RocketChatRoomInfoResponse {
    room: RocketChatRoomInfo,
}

#[derive(Debug, Deserialize)]
struct RocketChatRoomInfo {
    #[serde(rename = "_id")]
    id: String,
    #[serde(rename = "t")]
    kind: String,
    #[serde(rename = "name")]
    _name: Option<String>,
    #[serde(rename = "fname")]
    _friendly_name: Option<String>,
    #[serde(rename = "lm")]
    last_message_at: Option<String>,
    #[serde(rename = "lastMessage")]
    last_message: Option<RocketChatMessage>,
}

#[derive(Debug, Deserialize)]
struct RocketChatHistoryResponse {
    #[serde(default)]
    messages: Vec<RocketChatMessage>,
}

#[derive(Debug, Deserialize)]
struct RocketChatMessage {
    #[serde(rename = "_id")]
    id: String,
    #[serde(rename = "msg")]
    text: Option<String>,
    ts: String,
    #[serde(rename = "u")]
    user: Option<RocketChatMessageUser>,
    #[serde(rename = "t")]
    kind: Option<String>,
    #[serde(rename = "tmid")]
    thread_root_id: Option<String>,
    #[serde(default)]
    mentions: Vec<RocketChatMention>,
    #[serde(default)]
    attachments: Vec<RocketChatApiAttachment>,
    #[serde(default)]
    file: Option<RocketChatFileInfo>,
}

#[derive(Debug, Deserialize)]
struct RocketChatMessageUser {
    #[serde(rename = "_id")]
    id: String,
    username: Option<String>,
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RocketChatMention {
    #[serde(rename = "_id")]
    id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RocketChatApiAttachment {
    text: Option<String>,
    title: Option<String>,
    #[serde(rename = "title_link")]
    title_link: Option<String>,
    #[serde(rename = "image_url")]
    image_url: Option<String>,
    #[serde(rename = "audio_url")]
    audio_url: Option<String>,
    #[serde(rename = "video_url")]
    video_url: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RocketChatFileInfo {
    name: String,
    #[serde(rename = "type")]
    content_type: Option<String>,
    url: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adapter_manifest_is_valid() {
        let manifest = adapter_manifest();
        assert_eq!(manifest.kind, "rocketchat");
        manifest.validate().expect("valid manifest");
    }

    #[test]
    fn parse_settings_accepts_room_id_and_defaults() {
        let settings = serde_json::json!({
            "token_env": "ROCKETCHAT_AUTH_TOKEN",
            "user_id": "rbAXPnMktTFbNpwtJ",
            "room_id": "GENERAL123"
        });
        let parsed = parse_settings(&settings).expect("settings parse");
        assert_eq!(parsed.base_url, DEFAULT_BASE_URL);
        assert_eq!(parsed.workspace_id, "rocketchat");
        assert_eq!(parsed.max_messages_per_poll, DEFAULT_MAX_MESSAGES_PER_POLL);
        assert_eq!(parsed.respond_mode, RocketChatRespondMode::Mentions);
        assert_eq!(parsed.session_scope, ChannelSessionScope::Thread);
    }

    #[test]
    fn parse_settings_requires_room_reference() {
        let settings = serde_json::json!({
            "token_env": "ROCKETCHAT_AUTH_TOKEN",
            "user_id": "rbAXPnMktTFbNpwtJ"
        });
        let error = parse_settings(&settings).expect_err("missing room should fail");
        assert!(error.to_string().contains("room_id"));
    }

    #[test]
    fn render_outbound_preserves_code_blocks() {
        let rendered = render_text_blocks(&[
            MessageBlock::Text {
                text: "hello".to_string(),
            },
            MessageBlock::CodeBlock {
                language: Some("rust".to_string()),
                code: "fn main() {}".to_string(),
            },
        ]);
        assert!(rendered.contains("hello"));
        assert!(rendered.contains("```rust"));
    }

    #[test]
    fn user_scope_uses_room_id_for_top_level_messages() {
        let config = RocketChatChannelDriverConfig {
            base_url: DEFAULT_BASE_URL.to_string(),
            workspace_id: "rocketchat".to_string(),
            room_id: Some("room1".to_string()),
            room_name: None,
            user_id: "bot".to_string(),
            token: "token".to_string(),
            poll_interval: Duration::from_millis(DEFAULT_POLL_INTERVAL_MS),
            max_messages_per_poll: DEFAULT_MAX_MESSAGES_PER_POLL,
            start_from_latest: true,
            ignore_bot_messages: true,
            respond_mode: RocketChatRespondMode::Mentions,
            session_scope: ChannelSessionScope::User,
        };
        let driver = RocketChatChannelDriver {
            channel_id: "rocketchat".to_string(),
            client: Client::new(),
            config,
            shutdown_rx: watch::channel(false).1,
            room: RocketChatResolvedRoom {
                id: "room1".to_string(),
                room_type: RocketChatRoomType::Channel,
                latest_message_id: None,
                latest_message_ts: None,
            },
            backlog: VecDeque::new(),
            seen_message_ids: HashSet::new(),
            seen_message_order: VecDeque::new(),
            cursor_ts: None,
        };
        let message = RocketChatMessage {
            id: "m1".to_string(),
            text: Some("hi".to_string()),
            ts: "2026-03-29T00:00:00.000Z".to_string(),
            user: Some(RocketChatMessageUser {
                id: "user1".to_string(),
                username: Some("alice".to_string()),
                name: Some("Alice".to_string()),
            }),
            kind: None,
            thread_root_id: None,
            mentions: vec![],
            attachments: vec![],
            file: None,
        };
        assert_eq!(driver.thread_id_for_message(&message), "room1");
    }
}
