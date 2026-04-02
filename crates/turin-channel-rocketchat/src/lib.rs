use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use futures_util::{SinkExt, StreamExt};
use reqwest::Client;
use serde::{Deserialize, Deserializer};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;
use tokio::sync::watch;
use tokio::time::sleep;
use tokio_tungstenite::tungstenite::protocol::Message as WsMessage;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};
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
use turin_channel_runner::{ChannelDriver, ChannelProgressUpdate, ChannelStreamMode};

const DEFAULT_BASE_URL: &str = "http://localhost:3000";
const DEFAULT_TRANSPORT_MODE: &str = "realtime";
const DEFAULT_STREAM_MODE: &str = "typing";
const DEFAULT_POLL_INTERVAL_MS: u64 = 1_000;
const DEFAULT_MAX_MESSAGES_PER_POLL: u16 = 50;
const MAX_MESSAGES_PER_POLL: u16 = 100;
const ROCKETCHAT_MESSAGE_MAX_LEN: usize = 4_000;
const SEEN_MESSAGE_IDS_LIMIT: usize = 1_024;
const RECENT_SENT_MESSAGE_IDS_LIMIT: usize = 256;
const DEFAULT_REALTIME_RECONNECT_DELAY_MS: u64 = 2_000;
const ROCKETCHAT_TYPING_STATUS_INTERVAL_SECS: u64 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RocketChatRespondMode {
    All,
    Mentions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RocketChatTransportMode {
    Realtime,
    Polling,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RocketChatReplyMode {
    Thread,
    Channel,
    ThreadAndChannel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RocketChatRoomType {
    Channel,
    PrivateGroup,
    DirectMessage,
}

type RocketChatWsStream = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

#[derive(Debug, Clone)]
pub struct RocketChatChannelDriverConfig {
    pub base_url: String,
    pub websocket_url: String,
    pub transport_mode: RocketChatTransportMode,
    pub workspace_id: String,
    pub accept_all_rooms: bool,
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
    pub session_scope_dm: Option<ChannelSessionScope>,
    pub session_scope_group: Option<ChannelSessionScope>,
    pub session_scope_channel: Option<ChannelSessionScope>,
    pub reply_mode: RocketChatReplyMode,
    pub stream_mode: ChannelStreamMode,
    pub persist_thinking: bool,
}

#[derive(Debug, Clone)]
struct RocketChatChannelSettings {
    token_env: String,
    base_url: String,
    websocket_url: String,
    transport_mode: RocketChatTransportMode,
    workspace_id: String,
    accept_all_rooms: bool,
    room_id: Option<String>,
    room_name: Option<String>,
    user_id: String,
    poll_interval_ms: u64,
    max_messages_per_poll: u16,
    start_from_latest: bool,
    ignore_bot_messages: bool,
    respond_mode: RocketChatRespondMode,
    session_scope: ChannelSessionScope,
    session_scope_dm: Option<ChannelSessionScope>,
    session_scope_group: Option<ChannelSessionScope>,
    session_scope_channel: Option<ChannelSessionScope>,
    reply_mode: RocketChatReplyMode,
    stream_mode: ChannelStreamMode,
    persist_thinking: bool,
}

#[derive(Debug, Clone)]
struct RocketChatResolvedRoom {
    id: String,
    room_type: RocketChatRoomType,
    name: Option<String>,
    friendly_name: Option<String>,
    usernames: Vec<String>,
    latest_message: Option<RocketChatMessage>,
    latest_message_id: Option<String>,
    latest_message_ts: Option<String>,
}

#[derive(Debug, Clone)]
struct RocketChatRoomState {
    room: RocketChatResolvedRoom,
    cursor_ts: Option<String>,
}

pub fn validate_settings(
    settings: &serde_json::Value,
    allow_unconfigured_rooms: bool,
) -> Result<()> {
    parse_settings(settings, allow_unconfigured_rooms).map(|_| ())
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
                    key: "transport_mode".to_string(),
                    options: vec!["realtime".to_string(), "polling".to_string()],
                },
                ChannelEnumSetting {
                    key: "respond_mode".to_string(),
                    options: vec!["all".to_string(), "mentions".to_string()],
                },
                ChannelEnumSetting {
                    key: "session_scope".to_string(),
                    options: vec!["user".to_string(), "thread".to_string(), "room".to_string()],
                },
                ChannelEnumSetting {
                    key: "session_scope_dm".to_string(),
                    options: vec!["user".to_string(), "thread".to_string(), "room".to_string()],
                },
                ChannelEnumSetting {
                    key: "session_scope_group".to_string(),
                    options: vec!["user".to_string(), "thread".to_string(), "room".to_string()],
                },
                ChannelEnumSetting {
                    key: "session_scope_channel".to_string(),
                    options: vec!["user".to_string(), "thread".to_string(), "room".to_string()],
                },
                ChannelEnumSetting {
                    key: "reply_mode".to_string(),
                    options: vec![
                        "thread".to_string(),
                        "channel".to_string(),
                        "thread_and_channel".to_string(),
                    ],
                },
                ChannelEnumSetting {
                    key: "stream_mode".to_string(),
                    options: vec!["off".to_string(), "typing".to_string()],
                },
            ],
            capabilities: ChannelRuntimeCapabilities {
                dm: true,
                groups: true,
                threads: true,
                attachments: false,
                streaming: true,
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
                "Create or choose a Rocket.Chat bot/user, copy its auth token and user ID, then choose whether Turin should pair new rooms dynamically or stay pinned to a specific room."
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
                    key: "workspace_id".to_string(),
                    label: Some("Workspace ID".to_string()),
                    field_type: "text".to_string(),
                    help: Some("Defaults to 'rocketchat' and is usually fine to leave alone.".to_string()),
                    default: Some(serde_json::json!("rocketchat")),
                    advanced: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "workspace_id".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "pairing_mode".to_string(),
                    label: Some("Pairing Mode".to_string()),
                    field_type: "select".to_string(),
                    prompt: Some("How should new Rocket.Chat rooms and DMs be admitted?".to_string()),
                    help: Some("Auto approves newly seen rooms from trusted senders; pending records them for manual approval.".to_string()),
                    default: Some(serde_json::json!("auto")),
                    options: vec![
                        ChannelConfigFieldOption {
                            value: "auto".to_string(),
                            label: Some("Auto approve new rooms".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "pending".to_string(),
                            label: Some("Require manual approval".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "off".to_string(),
                            label: Some("Disable pairing".to_string()),
                        },
                    ],
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "pairing_mode".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "pairing_users".to_string(),
                    label: Some("Pairing Users".to_string()),
                    field_type: "string_list".to_string(),
                    prompt: Some("Optional usernames or IDs allowed to pair new Rocket.Chat rooms".to_string()),
                    help: Some("Leave empty to allow any sender to trigger room pairing.".to_string()),
                    advanced: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "pairing_users".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "transport_mode".to_string(),
                    label: Some("Transport Mode".to_string()),
                    field_type: "select".to_string(),
                    prompt: Some("How should Turin receive Rocket.Chat messages?".to_string()),
                    help: Some("Realtime uses Rocket.Chat's websocket/DDP path; polling remains available as a fallback.".to_string()),
                    default: Some(serde_json::json!(DEFAULT_TRANSPORT_MODE)),
                    advanced: true,
                    options: vec![
                        ChannelConfigFieldOption {
                            value: "realtime".to_string(),
                            label: Some("Realtime websocket".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "polling".to_string(),
                            label: Some("REST polling".to_string()),
                        },
                    ],
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "transport_mode".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "websocket_url".to_string(),
                    label: Some("WebSocket URL".to_string()),
                    field_type: "text".to_string(),
                    prompt: Some("Optional Rocket.Chat websocket URL override".to_string()),
                    help: Some("Leave empty to derive it automatically from the server URL as ws(s)://.../websocket.".to_string()),
                    advanced: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "websocket_url".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "room_id".to_string(),
                    label: Some("Room ID".to_string()),
                    field_type: "text".to_string(),
                    prompt: Some("Optional Rocket.Chat room ID filter".to_string()),
                    help: Some(
                        "Leave empty to let Turin discover rooms dynamically through pairing and approval. Set it to pin this channel to one specific room or DM."
                            .to_string(),
                    ),
                    advanced: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "room_id".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "room_name".to_string(),
                    label: Some("Room Name".to_string()),
                    field_type: "text".to_string(),
                    prompt: Some("Optional Rocket.Chat room name filter".to_string()),
                    help: Some(
                        "Alternative to room_id. Leave empty unless you want to pin Turin to a specific named room."
                            .to_string(),
                    ),
                    advanced: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "room_name".to_string(),
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
                    key: "reply_mode".to_string(),
                    label: Some("Reply Mode".to_string()),
                    field_type: "select".to_string(),
                    prompt: Some("How should Turin post replies back into Rocket.Chat?".to_string()),
                    help: Some("Thread keeps replies nested, thread and channel also shows them in the room, channel replies inline with an attachment-style quote of the triggering message.".to_string()),
                    default: Some(serde_json::json!("thread")),
                    options: vec![
                        ChannelConfigFieldOption {
                            value: "thread".to_string(),
                            label: Some("Thread only".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "thread_and_channel".to_string(),
                            label: Some("Thread and channel".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "channel".to_string(),
                            label: Some("Channel only".to_string()),
                        },
                    ],
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "reply_mode".to_string(),
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
                    key: "session_scope_dm".to_string(),
                    label: Some("DM Session Scope".to_string()),
                    field_type: "select".to_string(),
                    prompt: Some("Optional session scope override for direct messages".to_string()),
                    help: Some("Leave empty to reuse the main session scope.".to_string()),
                    advanced: true,
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
                        name: "session_scope_dm".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "session_scope_group".to_string(),
                    label: Some("Group Session Scope".to_string()),
                    field_type: "select".to_string(),
                    prompt: Some("Optional session scope override for private groups".to_string()),
                    help: Some("Leave empty to reuse the main session scope.".to_string()),
                    advanced: true,
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
                        name: "session_scope_group".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "session_scope_channel".to_string(),
                    label: Some("Channel Session Scope".to_string()),
                    field_type: "select".to_string(),
                    prompt: Some("Optional session scope override for public channels".to_string()),
                    help: Some("Leave empty to reuse the main session scope.".to_string()),
                    advanced: true,
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
                        name: "session_scope_channel".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "stream_mode".to_string(),
                    label: Some("Progress Mode".to_string()),
                    field_type: "select".to_string(),
                    prompt: Some("How should Turin signal that it is working on a reply?".to_string()),
                    help: Some("Typing sends Rocket.Chat room activity notifications while the turn is active.".to_string()),
                    default: Some(serde_json::json!(DEFAULT_STREAM_MODE)),
                    options: vec![
                        ChannelConfigFieldOption {
                            value: "off".to_string(),
                            label: Some("No progress".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "typing".to_string(),
                            label: Some("Typing indicator".to_string()),
                        },
                    ],
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "stream_mode".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "persist_thinking".to_string(),
                    label: Some("Include Final Thinking".to_string()),
                    field_type: "boolean".to_string(),
                    help: Some("When enabled, Turin prepends the model's final thinking to the posted reply.".to_string()),
                    default: Some(serde_json::json!(false)),
                    advanced: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "persist_thinking".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "allowed_users".to_string(),
                    label: Some("Allowed Users".to_string()),
                    field_type: "string_list".to_string(),
                    prompt: Some("Optional usernames or IDs allowed to interact".to_string()),
                    help: Some("Leave empty to allow any user in approved rooms.".to_string()),
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
    pub fn from_settings(
        settings: &serde_json::Value,
        allow_unconfigured_rooms: bool,
    ) -> Result<Self> {
        let settings = parse_settings(settings, allow_unconfigured_rooms)?;
        let token = std::env::var(&settings.token_env).map_err(|_| {
            anyhow!(
                "[rocketchat_auth_missing_token] Rocket.Chat auth token env var '{}' is not set for channel adapter",
                settings.token_env
            )
        })?;

        Ok(Self {
            base_url: settings.base_url,
            websocket_url: settings.websocket_url,
            transport_mode: settings.transport_mode,
            workspace_id: settings.workspace_id,
            accept_all_rooms: settings.accept_all_rooms,
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
            session_scope_dm: settings.session_scope_dm,
            session_scope_group: settings.session_scope_group,
            session_scope_channel: settings.session_scope_channel,
            reply_mode: settings.reply_mode,
            stream_mode: settings.stream_mode,
            persist_thinking: settings.persist_thinking,
        })
    }
}

pub struct RocketChatChannelDriver {
    channel_id: String,
    client: Client,
    config: RocketChatChannelDriverConfig,
    shutdown_rx: watch::Receiver<bool>,
    bot_username: Option<String>,
    bot_display_name: Option<String>,
    rooms: HashMap<String, RocketChatRoomState>,
    ws_stream: Option<RocketChatWsStream>,
    realtime_subscribed_room_ids: HashSet<String>,
    active_thread_keys: HashSet<String>,
    backlog: VecDeque<InboundEvent>,
    seen_message_ids: HashSet<String>,
    seen_message_order: VecDeque<String>,
    recent_sent_message_ids: HashSet<String>,
    recent_sent_message_order: VecDeque<String>,
    rooms_updated_since: Option<String>,
    last_room_refresh: Option<Instant>,
    last_typing_at: HashMap<String, Instant>,
    next_realtime_request_id: u64,
}

impl RocketChatChannelDriver {
    pub async fn from_settings(
        channel_id: impl Into<String>,
        settings: &serde_json::Value,
        allow_unconfigured_rooms: bool,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Result<Self> {
        let config =
            RocketChatChannelDriverConfig::from_settings(settings, allow_unconfigured_rooms)?;
        let client = Client::builder().build()?;

        let mut driver = Self {
            channel_id: channel_id.into(),
            client,
            config,
            shutdown_rx,
            bot_username: None,
            bot_display_name: None,
            rooms: HashMap::new(),
            ws_stream: None,
            realtime_subscribed_room_ids: HashSet::new(),
            active_thread_keys: HashSet::new(),
            backlog: VecDeque::new(),
            seen_message_ids: HashSet::new(),
            seen_message_order: VecDeque::new(),
            recent_sent_message_ids: HashSet::new(),
            recent_sent_message_order: VecDeque::new(),
            rooms_updated_since: None,
            last_room_refresh: None,
            last_typing_at: HashMap::new(),
            next_realtime_request_id: 1,
        };

        if let Err(err) = driver.load_bot_identity().await {
            warn!(
                channel_id = %driver.channel_id,
                error = ?err,
                "Rocket.Chat bot username lookup failed; typing indicators disabled until it succeeds"
            );
        }
        driver.refresh_rooms(true).await?;
        if !driver.config.start_from_latest
            && matches!(
                driver.config.transport_mode,
                RocketChatTransportMode::Realtime
            )
        {
            driver.poll_messages().await?;
        }

        Ok(driver)
    }

    async fn refresh_rooms(&mut self, initial: bool) -> Result<()> {
        let update = fetch_rooms(
            &self.client,
            &self.config,
            self.rooms_updated_since.as_deref(),
        )
        .await?;

        for room_id in update.remove_room_ids {
            self.rooms.remove(&room_id);
            self.realtime_subscribed_room_ids.remove(&room_id);
        }

        for room in update.rooms {
            if !self.room_matches_filters(&room) {
                continue;
            }
            self.upsert_room(room, initial).await?;
        }

        if let Some(updated_since) = update.next_updated_since {
            let should_replace = self
                .rooms_updated_since
                .as_deref()
                .is_none_or(|current| current < updated_since.as_str());
            if should_replace {
                self.rooms_updated_since = Some(updated_since);
            }
        }
        self.last_room_refresh = Some(Instant::now());

        if initial && !self.config.accept_all_rooms && self.rooms.is_empty() {
            anyhow::bail!(
                "[rocketchat_room_not_found] Rocket.Chat could not find a room matching the configured room filter"
            );
        }

        Ok(())
    }

    fn room_matches_filters(&self, room: &RocketChatResolvedRoom) -> bool {
        if let Some(expected_room_id) = self.config.room_id.as_deref()
            && room.id != expected_room_id
        {
            return false;
        }
        if let Some(expected_room_name) = self.config.room_name.as_deref() {
            let matches = room
                .name
                .as_deref()
                .is_some_and(|value| value.eq_ignore_ascii_case(expected_room_name))
                || room
                    .friendly_name
                    .as_deref()
                    .is_some_and(|value| value.eq_ignore_ascii_case(expected_room_name));
            if !matches {
                return false;
            }
        }
        true
    }

    async fn upsert_room(&mut self, room: RocketChatResolvedRoom, initial: bool) -> Result<()> {
        let room_id = room.id.clone();
        if let Some(existing) = self.rooms.get_mut(&room_id) {
            existing.room = room;
            return Ok(());
        }

        let mut state = RocketChatRoomState {
            room,
            cursor_ts: None,
        };
        if initial && self.config.start_from_latest {
            if let Some(message_id) = state.room.latest_message_id.clone() {
                self.remember_message_id(message_id);
            }
            state.cursor_ts = state.room.latest_message_ts.clone();
        }
        self.rooms.insert(room_id.clone(), state);

        if !initial {
            if self.config.start_from_latest {
                self.seed_new_room_from_latest(&room_id)?;
            } else {
                self.poll_room_messages(&room_id).await?;
            }
        }

        Ok(())
    }

    fn seed_new_room_from_latest(&mut self, room_id: &str) -> Result<()> {
        let Some(state) = self.rooms.get(room_id).cloned() else {
            return Ok(());
        };

        if let Some(cursor_ts) = state.room.latest_message_ts.clone() {
            self.update_room_cursor(room_id, cursor_ts);
        }
        if let Some(message_id) = state.room.latest_message_id.clone() {
            self.remember_message_id(message_id);
        }
        if let Some(message) = state.room.latest_message.clone() {
            if self.seen_message_ids.contains(&message.id) {
                return Ok(());
            }
            self.remember_message_id(message.id.clone());
            self.update_room_cursor(room_id, message.ts.clone());
            if let Some(event) = self.message_to_event(&state.room, message)? {
                self.backlog.push_back(event);
            }
        }

        Ok(())
    }

    fn update_room_cursor(&mut self, room_id: &str, cursor_ts: String) {
        if let Some(state) = self.rooms.get_mut(room_id) {
            state.cursor_ts = Some(cursor_ts);
        }
    }

    async fn poll_messages(&mut self) -> Result<()> {
        self.refresh_rooms(false).await?;
        self.poll_known_rooms().await
    }

    async fn poll_known_rooms(&mut self) -> Result<()> {
        let mut room_ids: Vec<String> = self.rooms.keys().cloned().collect();
        room_ids.sort();
        for room_id in room_ids {
            self.poll_room_messages(&room_id).await?;
        }
        Ok(())
    }

    async fn poll_room_messages(&mut self, room_id: &str) -> Result<()> {
        let Some(state) = self.rooms.get(room_id).cloned() else {
            return Ok(());
        };
        let messages = fetch_room_messages(
            &self.client,
            &self.config,
            &state.room,
            state.cursor_ts.as_deref(),
        )
        .await?;

        for message in messages {
            self.update_room_cursor(room_id, message.ts.clone());
            if self.seen_message_ids.contains(&message.id) {
                continue;
            }
            self.remember_message_id(message.id.clone());

            let Some(event) = self.message_to_event(&state.room, message)? else {
                continue;
            };
            self.backlog.push_back(event);
        }

        Ok(())
    }

    fn message_to_event(
        &self,
        room: &RocketChatResolvedRoom,
        message: RocketChatMessage,
    ) -> Result<Option<InboundEvent>> {
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

        if !self.should_accept_message(room, &message, user) {
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
        let session_scope = self.effective_session_scope(room);
        let conversation = ChannelConversationKey {
            channel: ChannelKind::new("rocketchat"),
            workspace_id: self.config.workspace_id.clone(),
            room_id: Some(room.id.clone()),
            thread_id: self.thread_id_for_message(room, &message, session_scope),
            user_id: if matches!(session_scope, ChannelSessionScope::User) {
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
            "rocketchat_message_ts".to_string(),
            serde_json::json!(message.ts),
        );
        metadata.insert("rocketchat_room_id".to_string(), serde_json::json!(room.id));
        if let Some(message_link) = build_rocketchat_message_link(
            &self.config.base_url,
            room,
            self.bot_username.as_deref(),
            &message.id,
        ) {
            metadata.insert(
                "rocketchat_message_link".to_string(),
                serde_json::json!(message_link),
            );
        }
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
            session_scope,
            text,
            attachments,
            metadata,
        }))
    }

    fn effective_session_scope(&self, room: &RocketChatResolvedRoom) -> ChannelSessionScope {
        let configured_scope = match room.room_type {
            RocketChatRoomType::DirectMessage => self
                .config
                .session_scope_dm
                .unwrap_or(self.config.session_scope),
            RocketChatRoomType::Channel => self
                .config
                .session_scope_channel
                .unwrap_or(self.config.session_scope),
            RocketChatRoomType::PrivateGroup => self
                .config
                .session_scope_group
                .unwrap_or(self.config.session_scope),
        };

        if matches!(self.config.reply_mode, RocketChatReplyMode::Channel)
            && matches!(configured_scope, ChannelSessionScope::Thread)
        {
            ChannelSessionScope::Room
        } else {
            configured_scope
        }
    }

    fn should_accept_message(
        &self,
        room: &RocketChatResolvedRoom,
        message: &RocketChatMessage,
        user: &RocketChatMessageUser,
    ) -> bool {
        if matches!(room.room_type, RocketChatRoomType::DirectMessage) {
            return user.id != self.config.user_id || !self.config.ignore_bot_messages;
        }

        match self.config.respond_mode {
            RocketChatRespondMode::All => true,
            RocketChatRespondMode::Mentions => {
                message
                    .mentions
                    .iter()
                    .any(|mention| mention.id.as_deref() == Some(self.config.user_id.as_str()))
                    || self.message_quotes_bot_reply(message)
                    || message.thread_root_id.as_deref().is_some_and(|thread_id| {
                        self.active_thread_keys
                            .contains(&active_thread_key(&room.id, thread_id))
                    })
            }
        }
    }

    fn message_quotes_bot_reply(&self, message: &RocketChatMessage) -> bool {
        message.attachments.iter().any(|attachment| {
            attachment.message_link.as_deref().is_some_and(|link| {
                self.recent_sent_message_ids
                    .iter()
                    .any(|message_id| link.contains(message_id))
            }) || attachment
                .author_name
                .as_deref()
                .is_some_and(|author_name| self.is_bot_identity_label(author_name))
        })
    }

    fn thread_id_for_message(
        &self,
        room: &RocketChatResolvedRoom,
        message: &RocketChatMessage,
        session_scope: ChannelSessionScope,
    ) -> String {
        match session_scope {
            ChannelSessionScope::Room => room.id.clone(),
            ChannelSessionScope::Thread => message
                .thread_root_id
                .clone()
                .unwrap_or_else(|| message.id.clone()),
            ChannelSessionScope::User => message
                .thread_root_id
                .clone()
                .unwrap_or_else(|| room.id.clone()),
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

    fn next_request_id(&mut self) -> String {
        let id = format!("turin-{}", self.next_realtime_request_id);
        self.next_realtime_request_id += 1;
        id
    }

    fn is_bot_identity_label(&self, raw: &str) -> bool {
        let normalized = normalize_identity_label(raw);
        if normalized.is_empty() {
            return false;
        }
        self.bot_username
            .as_deref()
            .is_some_and(|value| normalize_identity_label(value) == normalized)
            || self
                .bot_display_name
                .as_deref()
                .is_some_and(|value| normalize_identity_label(value) == normalized)
    }

    async fn load_bot_identity(&mut self) -> Result<()> {
        let identity = fetch_bot_identity(&self.client, &self.config).await?;
        self.bot_username = Some(identity.username);
        self.bot_display_name = identity.display_name;
        Ok(())
    }

    fn reset_transport_state(&mut self) -> Result<()> {
        self.ws_stream = None;
        self.realtime_subscribed_room_ids.clear();
        self.last_typing_at.clear();
        self.client = Client::builder().build().context(
            "[rocketchat_http_client_rebuild_failed] Failed to rebuild Rocket.Chat HTTP client",
        )?;
        Ok(())
    }

    fn remember_sent_message_id(&mut self, message_id: String) {
        if self.recent_sent_message_ids.insert(message_id.clone()) {
            self.recent_sent_message_order.push_back(message_id);
            while self.recent_sent_message_order.len() > RECENT_SENT_MESSAGE_IDS_LIMIT {
                if let Some(oldest) = self.recent_sent_message_order.pop_front() {
                    self.recent_sent_message_ids.remove(&oldest);
                }
            }
        }
    }

    async fn send_typing_status(&mut self, event: &InboundEvent) -> Result<()> {
        if self.config.stream_mode != ChannelStreamMode::Typing
            || self.config.transport_mode != RocketChatTransportMode::Realtime
        {
            return Ok(());
        }

        let Some(room_id) = event.conversation.room_id.as_deref() else {
            return Ok(());
        };

        let key = progress_key(&event.conversation)?;
        let now = Instant::now();
        if self.last_typing_at.get(&key).is_some_and(|previous| {
            now.duration_since(*previous)
                < Duration::from_secs(ROCKETCHAT_TYPING_STATUS_INTERVAL_SECS)
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
        let (mut stream, _) = connect_async(&websocket_url)
            .await
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

        self.await_connected(&mut stream).await?;
        self.login_realtime(&mut stream).await?;

        self.ws_stream = Some(stream);
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

    async fn next_realtime_event(&mut self) -> Result<Option<InboundEvent>> {
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
        let message: RocketChatMessage = serde_json::from_value(raw_message).context(
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

            if matches!(
                self.config.transport_mode,
                RocketChatTransportMode::Realtime
            ) {
                return self.next_realtime_event().await;
            }

            if let Err(err) = self.poll_messages().await {
                warn!(
                    channel_id = %self.channel_id,
                    room_count = self.rooms.len(),
                    error = ?err,
                    "Rocket.Chat polling failed"
                );
                if let Err(reset_err) = self.reset_transport_state() {
                    warn!(
                        channel_id = %self.channel_id,
                        error = ?reset_err,
                        "Rocket.Chat transport reset failed"
                    );
                }
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

        let reply_target =
            resolve_reply_target(room_id, conversation, &message, self.config.reply_mode);
        let chunks = split_for_rocketchat_content(render_rocketchat_message(
            &message,
            self.config.persist_thinking,
        ));

        for (index, chunk) in chunks.into_iter().enumerate() {
            let rendered_chunk =
                if index == 0 && matches!(self.config.reply_mode, RocketChatReplyMode::Channel) {
                    prepend_channel_reply_quote(&chunk, &message)
                } else {
                    chunk
                };
            let payload =
                build_rocketchat_send_payload(room_id, &rendered_chunk, reply_target, &[]);
            if let Some(thread_id) = reply_target.thread_id
                && index == 0
            {
                self.active_thread_keys
                    .insert(active_thread_key(room_id, thread_id));
            }
            let response = self
                .client
                .post(api_url(&self.config.base_url, "chat.sendMessage"))
                .header("X-Auth-Token", &self.config.token)
                .header("X-User-Id", &self.config.user_id)
                .json(&payload)
                .send()
                .await
                .context("Failed to send Rocket.Chat message")?;
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            if !status.is_success() {
                anyhow::bail!(
                    "[rocketchat_send_failed] Rocket.Chat chat.sendMessage failed with status {}: {}",
                    status,
                    body
                );
            }
            if let Ok(parsed) = serde_json::from_str::<RocketChatSendMessageResponse>(&body)
                && let Some(sent_message) = parsed.message
            {
                self.remember_sent_message_id(sent_message.id);
            }
        }

        Ok(())
    }

    fn enrich_outbound_for_event(
        &self,
        event: &InboundEvent,
        mut outbound: OutboundMessage,
    ) -> OutboundMessage {
        if !outbound
            .metadata
            .contains_key("rocketchat_reply_to_message_id")
            && let Some(message_id) = event.metadata.get("rocketchat_message_id")
        {
            outbound.metadata.insert(
                "rocketchat_reply_to_message_id".to_string(),
                message_id.clone(),
            );
        }
        if !outbound.metadata.contains_key("rocketchat_thread_id")
            && let Some(thread_id) = event.metadata.get("rocketchat_thread_id")
        {
            outbound
                .metadata
                .insert("rocketchat_thread_id".to_string(), thread_id.clone());
        }
        if !outbound.metadata.contains_key("rocketchat_reply_to_label") {
            outbound.metadata.insert(
                "rocketchat_reply_to_label".to_string(),
                serde_json::json!(event.user.prompt_label()),
            );
        }
        if !outbound
            .metadata
            .contains_key("rocketchat_reply_to_excerpt")
            && !event.text.trim().is_empty()
        {
            outbound.metadata.insert(
                "rocketchat_reply_to_excerpt".to_string(),
                serde_json::json!(reply_excerpt(&event.text)),
            );
        }
        if !outbound
            .metadata
            .contains_key("rocketchat_reply_to_message_ts")
            && let Some(message_ts) = event.metadata.get("rocketchat_message_ts")
        {
            outbound.metadata.insert(
                "rocketchat_reply_to_message_ts".to_string(),
                message_ts.clone(),
            );
        }
        if !outbound
            .metadata
            .contains_key("rocketchat_reply_to_message_link")
            && let Some(message_link) = event.metadata.get("rocketchat_message_link")
        {
            outbound.metadata.insert(
                "rocketchat_reply_to_message_link".to_string(),
                message_link.clone(),
            );
        }
        outbound
    }

    fn stream_mode(&self) -> ChannelStreamMode {
        self.config.stream_mode
    }

    fn persist_thinking(&self) -> bool {
        self.config.persist_thinking
    }

    async fn send_progress(
        &mut self,
        event: &InboundEvent,
        update: ChannelProgressUpdate,
    ) -> Result<()> {
        match update {
            ChannelProgressUpdate::Typing => self.send_typing_status(event).await,
            ChannelProgressUpdate::StreamingPreview { .. } => Ok(()),
        }
    }

    async fn shutdown(&mut self) -> Result<()> {
        self.ws_stream = None;
        self.realtime_subscribed_room_ids.clear();
        self.last_typing_at.clear();
        Ok(())
    }
}

fn render_text_blocks(blocks: &[MessageBlock]) -> String {
    let mut chunks = Vec::new();
    for block in blocks {
        match block {
            MessageBlock::Text { text } => {
                if !text.trim().is_empty() {
                    chunks.push(wrap_markdown_tables(text));
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

#[derive(Debug, Clone, Copy)]
struct RocketChatReplyTarget<'a> {
    thread_id: Option<&'a str>,
    show_in_channel: bool,
}

fn build_rocketchat_send_payload(
    room_id: &str,
    text: &str,
    reply_target: RocketChatReplyTarget<'_>,
    attachments: &[serde_json::Value],
) -> serde_json::Value {
    let mut message = serde_json::Map::new();
    message.insert("rid".to_string(), serde_json::json!(room_id));
    message.insert("msg".to_string(), serde_json::json!(text));
    message.insert("parseUrls".to_string(), serde_json::json!(false));
    if !attachments.is_empty() {
        message.insert(
            "attachments".to_string(),
            serde_json::Value::Array(attachments.to_vec()),
        );
    }
    if let Some(thread_id) = reply_target.thread_id {
        message.insert("tmid".to_string(), serde_json::json!(thread_id));
        if reply_target.show_in_channel {
            message.insert("tshow".to_string(), serde_json::json!(true));
        }
    }

    serde_json::json!({ "message": message })
}

fn render_rocketchat_message(message: &OutboundMessage, persist_thinking: bool) -> String {
    let mut rendered = render_text_blocks(&message.blocks);
    if persist_thinking
        && let Some(thinking) = message
            .metadata
            .get("channel_final_thinking")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
    {
        rendered = prepend_final_thinking_text(&rendered, thinking);
    }

    rendered
}

fn prepend_channel_reply_quote(text: &str, message: &OutboundMessage) -> String {
    let reply_label = message
        .metadata
        .get("rocketchat_reply_to_label")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let reply_link = message
        .metadata
        .get("rocketchat_reply_to_message_link")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let reply_excerpt = message
        .metadata
        .get("rocketchat_reply_to_excerpt")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let mut quote_lines = Vec::new();
    if let Some(reply_label) = reply_label {
        let first_line = if let Some(reply_link) = reply_link {
            format!("> [{}]({})", reply_label, reply_link)
        } else {
            format!("> {}", reply_label)
        };
        quote_lines.push(first_line);
    }
    if let Some(reply_excerpt) = reply_excerpt {
        for line in reply_excerpt.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                quote_lines.push(format!("> {}", trimmed));
            }
        }
    }

    if quote_lines.is_empty() {
        return text.to_string();
    }

    if text.is_empty() {
        quote_lines.join("\n")
    } else {
        format!("{}\n\n{}", quote_lines.join("\n"), text)
    }
}

fn resolve_reply_target<'a>(
    room_id: &'a str,
    conversation: &'a ChannelConversationKey,
    message: &'a OutboundMessage,
    reply_mode: RocketChatReplyMode,
) -> RocketChatReplyTarget<'a> {
    let metadata_thread_id = metadata_str(&message.metadata, "rocketchat_thread_id");
    let reply_to_message_id = metadata_str(&message.metadata, "rocketchat_reply_to_message_id");
    let conversation_thread_id =
        (conversation.thread_id != room_id).then_some(conversation.thread_id.as_str());
    let thread_id = match reply_mode {
        RocketChatReplyMode::Channel => None,
        RocketChatReplyMode::Thread | RocketChatReplyMode::ThreadAndChannel => metadata_thread_id
            .or(reply_to_message_id)
            .or(conversation_thread_id),
    };

    RocketChatReplyTarget {
        thread_id,
        show_in_channel: matches!(reply_mode, RocketChatReplyMode::ThreadAndChannel),
    }
}

fn metadata_str<'a>(
    metadata: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<&'a str> {
    metadata.get(key).and_then(|value| value.as_str())
}

fn prepend_final_thinking_text(rendered: &str, thinking: &str) -> String {
    let trimmed = rendered.trim();
    if trimmed.is_empty() {
        format!("Thinking:\n{}", thinking)
    } else {
        format!("Thinking:\n{}\n\nReply:\n{}", thinking, trimmed)
    }
}

fn reply_excerpt(text: &str) -> String {
    let lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(3)
        .map(|line| {
            let excerpt = line.chars().take(120).collect::<String>();
            if line.chars().count() > excerpt.chars().count() {
                format!("{excerpt}...")
            } else {
                excerpt
            }
        })
        .collect::<Vec<_>>();
    if lines.is_empty() {
        String::new()
    } else {
        lines.join("\n")
    }
}

fn wrap_markdown_tables(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return text.to_string();
    }

    let mut out = Vec::new();
    let mut index = 0usize;
    let mut in_fence = false;
    while index < lines.len() {
        let line = lines[index];
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            out.push(line.to_string());
            index += 1;
            continue;
        }

        if !in_fence && is_markdown_table_row(line) {
            let start = index;
            let mut end = index;
            let mut has_separator = false;
            while end < lines.len() && is_markdown_table_row(lines[end]) {
                has_separator |= is_markdown_table_separator(lines[end]);
                end += 1;
            }
            if has_separator && end.saturating_sub(start) >= 2 {
                out.push("```".to_string());
                out.extend(lines[start..end].iter().map(|value| (*value).to_string()));
                out.push("```".to_string());
                index = end;
                continue;
            }
        }

        out.push(line.to_string());
        index += 1;
    }

    out.join("\n")
}

fn is_markdown_table_row(line: &str) -> bool {
    let trimmed = line.trim();
    !trimmed.is_empty() && trimmed.contains('|') && !trimmed.starts_with("```")
}

fn is_markdown_table_separator(line: &str) -> bool {
    let trimmed = line.trim();
    !trimmed.is_empty()
        && trimmed.contains('-')
        && trimmed
            .chars()
            .all(|ch| matches!(ch, '|' | ':' | '-' | ' ' | '\t'))
}

fn progress_key(conversation: &ChannelConversationKey) -> Result<String> {
    serde_json::to_string(conversation)
        .with_context(|| "[rocketchat_progress_key_invalid] Failed to serialize conversation key")
}

fn active_thread_key(room_id: &str, thread_id: &str) -> String {
    format!("{room_id}:{thread_id}")
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

fn build_rocketchat_message_link(
    base_url: &str,
    room: &RocketChatResolvedRoom,
    bot_username: Option<&str>,
    message_id: &str,
) -> Option<String> {
    let path = match room.room_type {
        RocketChatRoomType::Channel => room
            .name
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .map(|name| format!("channel/{}", name)),
        RocketChatRoomType::PrivateGroup => room
            .name
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .map(|name| format!("group/{}", name)),
        RocketChatRoomType::DirectMessage => {
            let bot_username = bot_username?;
            room.usernames
                .iter()
                .find(|username| {
                    let username = username.trim();
                    !username.is_empty() && username != bot_username
                })
                .map(|username| format!("direct/{}", username))
        }
    }?;

    Some(format!(
        "{}/{}?msg={}",
        base_url.trim_end_matches('/'),
        path,
        message_id
    ))
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

async fn fetch_bot_identity(
    client: &Client,
    config: &RocketChatChannelDriverConfig,
) -> Result<RocketChatBotIdentity> {
    let response = client
        .get(api_url(&config.base_url, "users.info"))
        .header("X-Auth-Token", &config.token)
        .header("X-User-Id", &config.user_id)
        .query(&[("userId", config.user_id.as_str())])
        .send()
        .await
        .context("Failed to query Rocket.Chat user info")?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!(
            "[rocketchat_user_info_failed] Rocket.Chat users.info failed with status {}: {}",
            status,
            body
        );
    }

    let parsed: RocketChatUserInfoResponse =
        serde_json::from_str(&body).context("Failed to decode Rocket.Chat users.info response")?;
    let username = parsed
        .user
        .username
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            anyhow!(
                "[rocketchat_user_info_missing_username] Rocket.Chat users.info did not return a username for '{}'",
                config.user_id
            )
        })?;
    let display_name = parsed
        .user
        .name
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    Ok(RocketChatBotIdentity {
        username,
        display_name,
    })
}

async fn fetch_rooms(
    client: &Client,
    config: &RocketChatChannelDriverConfig,
    updated_since: Option<&str>,
) -> Result<RocketChatRoomsUpdate> {
    let mut request = client
        .get(api_url(&config.base_url, "rooms.get"))
        .header("X-Auth-Token", &config.token)
        .header("X-User-Id", &config.user_id);

    if let Some(updated_since) = updated_since {
        request = request.query(&[("updatedSince", updated_since)]);
    }

    let response = request
        .send()
        .await
        .context("Failed to query Rocket.Chat rooms")?;
    let status = response.status();
    let body = response.text().await.unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!(
            "[rocketchat_rooms_get_failed] Rocket.Chat rooms.get failed with status {}: {}",
            status,
            body
        );
    }

    let parsed: RocketChatRoomsResponse =
        serde_json::from_str(&body).context("Failed to decode Rocket.Chat rooms.get response")?;
    let next_updated_since = parsed
        .update
        .iter()
        .filter_map(|room| room.updated_at.clone())
        .max();
    let rooms = parsed
        .update
        .into_iter()
        .map(RocketChatResolvedRoom::try_from)
        .collect::<Result<Vec<_>>>()?;

    Ok(RocketChatRoomsUpdate {
        rooms,
        remove_room_ids: parsed.remove,
        next_updated_since,
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

fn parse_settings(
    settings: &serde_json::Value,
    allow_unconfigured_rooms: bool,
) -> Result<RocketChatChannelSettings> {
    let settings = settings
        .as_object()
        .ok_or_else(|| anyhow!("Rocket.Chat channel settings must be a JSON object"))?;
    reject_deprecated_session_scope_keys(settings)?;

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

    let accept_all_rooms = room_id.is_none() && room_name.is_none() && allow_unconfigured_rooms;

    if room_id.is_none() && room_name.is_none() && !allow_unconfigured_rooms {
        anyhow::bail!(
            "[rocketchat_config_missing_room] Rocket.Chat channel requires 'room_id' or 'room_name' unless pairing is enabled"
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
    let websocket_url = read_optional_non_empty_string(
        settings,
        "websocket_url",
        "[rocketchat_config_invalid_websocket_url] Rocket.Chat channel setting 'websocket_url' must not be empty",
    )?
    .map(ToString::to_string)
    .unwrap_or_else(|| default_websocket_url(&base_url));

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
        websocket_url,
        transport_mode: read_transport_mode(settings.get("transport_mode"))?,
        workspace_id: read_optional_non_empty_string(
            settings,
            "workspace_id",
            "[rocketchat_config_invalid_workspace_id] Rocket.Chat channel setting 'workspace_id' must not be empty",
        )?
        .unwrap_or("rocketchat")
        .to_string(),
        accept_all_rooms,
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
        session_scope_dm: read_optional_session_scope(
            settings.get("session_scope_dm"),
            "session_scope_dm",
        )?,
        session_scope_group: read_optional_session_scope(
            settings.get("session_scope_group"),
            "session_scope_group",
        )?,
        session_scope_channel: read_optional_session_scope(
            settings.get("session_scope_channel"),
            "session_scope_channel",
        )?,
        reply_mode: read_reply_mode(settings.get("reply_mode"))?,
        stream_mode: read_stream_mode(settings.get("stream_mode"))?,
        persist_thinking: read_bool(
            settings.get("persist_thinking"),
            false,
            "persist_thinking",
        )?,
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

fn read_transport_mode(value: Option<&serde_json::Value>) -> Result<RocketChatTransportMode> {
    let raw = match value {
        None => return Ok(RocketChatTransportMode::Realtime),
        Some(value) => value.as_str().ok_or_else(|| {
            anyhow!(
                "[rocketchat_config_invalid_transport_mode] Rocket.Chat channel setting 'transport_mode' must be a string"
            )
        })?,
    };
    match raw {
        "realtime" => Ok(RocketChatTransportMode::Realtime),
        "polling" => Ok(RocketChatTransportMode::Polling),
        _ => anyhow::bail!(
            "[rocketchat_config_invalid_transport_mode] Rocket.Chat channel setting 'transport_mode' must be one of: realtime, polling"
        ),
    }
}

fn read_reply_mode(value: Option<&serde_json::Value>) -> Result<RocketChatReplyMode> {
    let raw = match value {
        None => return Ok(RocketChatReplyMode::Thread),
        Some(value) => value.as_str().ok_or_else(|| {
            anyhow!(
                "[rocketchat_config_invalid_reply_mode] Rocket.Chat channel setting 'reply_mode' must be a string"
            )
        })?,
    };
    match raw {
        "thread" => Ok(RocketChatReplyMode::Thread),
        "channel" => Ok(RocketChatReplyMode::Channel),
        "thread_and_channel" => Ok(RocketChatReplyMode::ThreadAndChannel),
        _ => anyhow::bail!(
            "[rocketchat_config_invalid_reply_mode] Rocket.Chat channel setting 'reply_mode' must be one of: thread, channel, thread_and_channel"
        ),
    }
}

fn read_stream_mode(value: Option<&serde_json::Value>) -> Result<ChannelStreamMode> {
    let raw = match value {
        None => return Ok(ChannelStreamMode::Typing),
        Some(value) => value.as_str().ok_or_else(|| {
            anyhow!(
                "[rocketchat_config_invalid_stream_mode] Rocket.Chat channel setting 'stream_mode' must be a string"
            )
        })?,
    };
    match raw {
        "off" => Ok(ChannelStreamMode::Off),
        "typing" => Ok(ChannelStreamMode::Typing),
        _ => anyhow::bail!(
            "[rocketchat_config_invalid_stream_mode] Rocket.Chat channel setting 'stream_mode' must be one of: off, typing"
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

fn read_optional_session_scope(
    value: Option<&serde_json::Value>,
    key: &str,
) -> Result<Option<ChannelSessionScope>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let scope = value.as_str().ok_or_else(|| {
        anyhow!(
            "[rocketchat_config_invalid_session_scope] Rocket.Chat channel setting '{}' must be a string",
            key
        )
    })?;
    match scope {
        "user" => Ok(Some(ChannelSessionScope::User)),
        "thread" => Ok(Some(ChannelSessionScope::Thread)),
        "room" => Ok(Some(ChannelSessionScope::Room)),
        _ => anyhow::bail!(
            "[rocketchat_config_invalid_session_scope] Rocket.Chat channel setting '{}' must be one of: user, thread, room",
            key
        ),
    }
}

fn reject_deprecated_session_scope_keys(
    settings: &serde_json::Map<String, serde_json::Value>,
) -> Result<()> {
    for (legacy, replacement) in [
        ("dm_session_scope", "session_scope_dm"),
        ("group_session_scope", "session_scope_group"),
        ("channel_session_scope", "session_scope_channel"),
    ] {
        if settings.contains_key(legacy) {
            anyhow::bail!(
                "[rocketchat_config_deprecated_session_scope_key] Rocket.Chat channel setting '{}' is no longer supported; use '{}' instead",
                legacy,
                replacement
            );
        }
    }
    Ok(())
}

fn default_websocket_url(base_url: &str) -> String {
    if let Some(rest) = base_url.strip_prefix("https://") {
        return format!("wss://{}/websocket", rest.trim_end_matches('/'));
    }
    if let Some(rest) = base_url.strip_prefix("http://") {
        return format!("ws://{}/websocket", rest.trim_end_matches('/'));
    }
    if base_url.starts_with("wss://") || base_url.starts_with("ws://") {
        return format!("{}/websocket", base_url.trim_end_matches('/'));
    }
    format!("ws://{}/websocket", base_url.trim_end_matches('/'))
}

fn subscription_request_id(room_id: &str) -> String {
    format!("room:{room_id}")
}

fn subscription_room_id(request_id: &str) -> Option<&str> {
    request_id.strip_prefix("room:")
}

fn login_result_error(frame: &RocketChatDdpFrame) -> Option<serde_json::Value> {
    if frame.msg.as_deref() != Some("result") {
        return None;
    }
    frame.error.clone()
}

fn deserialize_rocketchat_timestamp<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    normalize_rocketchat_timestamp_value(value).map_err(serde::de::Error::custom)
}

fn deserialize_optional_rocketchat_timestamp<'de, D>(
    deserializer: D,
) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<serde_json::Value>::deserialize(deserializer)?;
    value
        .map(normalize_rocketchat_timestamp_value)
        .transpose()
        .map_err(serde::de::Error::custom)
}

fn normalize_rocketchat_timestamp_value(value: serde_json::Value) -> Result<String> {
    match value {
        serde_json::Value::String(raw) => normalize_rocketchat_timestamp_string(&raw),
        serde_json::Value::Object(map) => {
            if let Some(inner) = map.get("$date") {
                return normalize_rocketchat_timestamp_value(inner.clone());
            }
            anyhow::bail!(
                "[rocketchat_timestamp_invalid] Rocket.Chat timestamp object must contain '$date'"
            );
        }
        serde_json::Value::Number(number) => normalize_rocketchat_timestamp_number(&number),
        other => anyhow::bail!(
            "[rocketchat_timestamp_invalid] Rocket.Chat timestamp must be a string, number, or {{$date: ...}}, got {}",
            other
        ),
    }
}

fn normalize_rocketchat_timestamp_string(raw: &str) -> Result<String> {
    match OffsetDateTime::parse(raw, &Rfc3339) {
        Ok(parsed) => parsed
            .format(&Rfc3339)
            .map_err(anyhow::Error::from)
            .context("[rocketchat_timestamp_format_failed] Failed to format Rocket.Chat timestamp"),
        Err(_) => Ok(raw.to_string()),
    }
}

fn normalize_rocketchat_timestamp_number(number: &serde_json::Number) -> Result<String> {
    let timestamp = if let Some(value) = number.as_i64() {
        value
    } else if let Some(value) = number.as_u64() {
        i64::try_from(value).context(
            "[rocketchat_timestamp_out_of_range] Rocket.Chat timestamp number does not fit in i64",
        )?
    } else {
        anyhow::bail!(
            "[rocketchat_timestamp_invalid] Rocket.Chat floating-point timestamps are not supported"
        );
    };

    let nanos = if timestamp.abs() >= 10_000_000_000 {
        i128::from(timestamp) * 1_000_000
    } else {
        i128::from(timestamp) * 1_000_000_000
    };
    let parsed = OffsetDateTime::from_unix_timestamp_nanos(nanos).context(
        "[rocketchat_timestamp_out_of_range] Rocket.Chat numeric timestamp is out of range",
    )?;
    parsed
        .format(&Rfc3339)
        .map_err(anyhow::Error::from)
        .context("[rocketchat_timestamp_format_failed] Failed to format Rocket.Chat timestamp")
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

impl TryFrom<RocketChatRoomInfo> for RocketChatResolvedRoom {
    type Error = anyhow::Error;

    fn try_from(value: RocketChatRoomInfo) -> Result<Self> {
        Ok(Self {
            id: value.id,
            room_type: RocketChatRoomType::parse(&value.kind)?,
            name: value.name,
            friendly_name: value.friendly_name,
            usernames: value.usernames,
            latest_message_id: value
                .last_message
                .as_ref()
                .map(|message| message.id.clone()),
            latest_message_ts: value.last_message_at.or_else(|| {
                value
                    .last_message
                    .as_ref()
                    .map(|message| message.ts.clone())
            }),
            latest_message: value.last_message,
        })
    }
}

#[derive(Debug)]
struct RocketChatRoomsUpdate {
    rooms: Vec<RocketChatResolvedRoom>,
    remove_room_ids: Vec<String>,
    next_updated_since: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RocketChatRoomsResponse {
    #[serde(default)]
    update: Vec<RocketChatRoomInfo>,
    #[serde(default)]
    remove: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct RocketChatRoomInfo {
    #[serde(rename = "_id")]
    id: String,
    #[serde(rename = "t")]
    kind: String,
    #[serde(rename = "name")]
    name: Option<String>,
    #[serde(rename = "fname")]
    friendly_name: Option<String>,
    #[serde(default)]
    usernames: Vec<String>,
    #[serde(
        rename = "_updatedAt",
        default,
        deserialize_with = "deserialize_optional_rocketchat_timestamp"
    )]
    updated_at: Option<String>,
    #[serde(
        rename = "lm",
        default,
        deserialize_with = "deserialize_optional_rocketchat_timestamp"
    )]
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
struct RocketChatUserInfoResponse {
    user: RocketChatApiUser,
}

#[derive(Debug, Deserialize)]
struct RocketChatApiUser {
    username: Option<String>,
    name: Option<String>,
}

#[derive(Debug)]
struct RocketChatBotIdentity {
    username: String,
    display_name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RocketChatSendMessageResponse {
    message: Option<RocketChatSentMessage>,
}

#[derive(Debug, Deserialize)]
struct RocketChatSentMessage {
    #[serde(rename = "_id")]
    id: String,
}

#[derive(Debug, Deserialize)]
struct RocketChatDdpFrame {
    #[serde(default)]
    msg: Option<String>,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    collection: Option<String>,
    #[serde(default)]
    fields: Option<RocketChatDdpChangedFields>,
    #[serde(default)]
    error: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct RocketChatDdpChangedFields {
    #[serde(rename = "eventName", default)]
    event_name: Option<String>,
    #[serde(default)]
    args: Vec<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
struct RocketChatMessage {
    #[serde(rename = "_id")]
    id: String,
    #[serde(rename = "msg")]
    text: Option<String>,
    #[serde(deserialize_with = "deserialize_rocketchat_timestamp")]
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

#[derive(Debug, Clone, Deserialize)]
struct RocketChatMessageUser {
    #[serde(rename = "_id")]
    id: String,
    username: Option<String>,
    name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct RocketChatMention {
    #[serde(rename = "_id")]
    id: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct RocketChatApiAttachment {
    text: Option<String>,
    title: Option<String>,
    #[serde(rename = "title_link")]
    title_link: Option<String>,
    #[serde(rename = "message_link")]
    message_link: Option<String>,
    #[serde(rename = "author_name")]
    author_name: Option<String>,
    #[serde(rename = "image_url")]
    image_url: Option<String>,
    #[serde(rename = "audio_url")]
    audio_url: Option<String>,
    #[serde(rename = "video_url")]
    video_url: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct RocketChatFileInfo {
    name: String,
    #[serde(rename = "type")]
    content_type: Option<String>,
    url: Option<String>,
}

fn normalize_identity_label(raw: &str) -> String {
    raw.trim().trim_start_matches('@').to_ascii_lowercase()
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
        let parsed = parse_settings(&settings, false).expect("settings parse");
        assert_eq!(parsed.base_url, DEFAULT_BASE_URL);
        assert_eq!(parsed.websocket_url, "ws://localhost:3000/websocket");
        assert_eq!(parsed.transport_mode, RocketChatTransportMode::Realtime);
        assert_eq!(parsed.workspace_id, "rocketchat");
        assert!(!parsed.accept_all_rooms);
        assert_eq!(parsed.max_messages_per_poll, DEFAULT_MAX_MESSAGES_PER_POLL);
        assert_eq!(parsed.respond_mode, RocketChatRespondMode::Mentions);
        assert_eq!(parsed.session_scope, ChannelSessionScope::Thread);
        assert_eq!(parsed.session_scope_dm, None);
        assert_eq!(parsed.session_scope_group, None);
        assert_eq!(parsed.session_scope_channel, None);
        assert_eq!(parsed.reply_mode, RocketChatReplyMode::Thread);
        assert_eq!(parsed.stream_mode, ChannelStreamMode::Typing);
        assert!(!parsed.persist_thinking);
    }

    #[test]
    fn parse_settings_requires_room_reference() {
        let settings = serde_json::json!({
            "token_env": "ROCKETCHAT_AUTH_TOKEN",
            "user_id": "rbAXPnMktTFbNpwtJ"
        });
        let error = parse_settings(&settings, false).expect_err("missing room should fail");
        assert!(error.to_string().contains("room_id"));
    }

    #[test]
    fn parse_settings_accepts_dynamic_room_discovery_when_pairing_enabled() {
        let settings = serde_json::json!({
            "token_env": "ROCKETCHAT_AUTH_TOKEN",
            "user_id": "rbAXPnMktTFbNpwtJ"
        });
        let parsed = parse_settings(&settings, true).expect("settings parse");
        assert!(parsed.accept_all_rooms);
        assert!(parsed.room_id.is_none());
        assert!(parsed.room_name.is_none());
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
            websocket_url: default_websocket_url(DEFAULT_BASE_URL),
            transport_mode: RocketChatTransportMode::Realtime,
            workspace_id: "rocketchat".to_string(),
            accept_all_rooms: false,
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
            session_scope_dm: None,
            session_scope_group: None,
            session_scope_channel: None,
            reply_mode: RocketChatReplyMode::Thread,
            stream_mode: ChannelStreamMode::Typing,
            persist_thinking: false,
        };
        let driver = RocketChatChannelDriver {
            channel_id: "rocketchat".to_string(),
            client: Client::new(),
            config,
            shutdown_rx: watch::channel(false).1,
            bot_username: None,
            bot_display_name: None,
            rooms: HashMap::from([(
                "room1".to_string(),
                RocketChatRoomState {
                    room: RocketChatResolvedRoom {
                        id: "room1".to_string(),
                        room_type: RocketChatRoomType::Channel,
                        name: Some("general".to_string()),
                        friendly_name: Some("General".to_string()),
                        usernames: vec![],
                        latest_message: None,
                        latest_message_id: None,
                        latest_message_ts: None,
                    },
                    cursor_ts: None,
                },
            )]),
            ws_stream: None,
            realtime_subscribed_room_ids: HashSet::new(),
            active_thread_keys: HashSet::new(),
            backlog: VecDeque::new(),
            seen_message_ids: HashSet::new(),
            seen_message_order: VecDeque::new(),
            recent_sent_message_ids: HashSet::new(),
            recent_sent_message_order: VecDeque::new(),
            rooms_updated_since: None,
            last_room_refresh: None,
            last_typing_at: HashMap::new(),
            next_realtime_request_id: 1,
        };
        let room = driver.rooms.get("room1").expect("room state").room.clone();
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
        assert_eq!(
            driver.thread_id_for_message(&room, &message, ChannelSessionScope::User),
            "room1"
        );
    }

    #[test]
    fn reset_transport_state_clears_realtime_subscriptions() {
        let config = RocketChatChannelDriverConfig {
            base_url: DEFAULT_BASE_URL.to_string(),
            websocket_url: default_websocket_url(DEFAULT_BASE_URL),
            transport_mode: RocketChatTransportMode::Realtime,
            workspace_id: "rocketchat".to_string(),
            accept_all_rooms: false,
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
            session_scope_dm: None,
            session_scope_group: None,
            session_scope_channel: None,
            reply_mode: RocketChatReplyMode::Thread,
            stream_mode: ChannelStreamMode::Typing,
            persist_thinking: false,
        };
        let mut driver = RocketChatChannelDriver {
            channel_id: "rocketchat".to_string(),
            client: Client::new(),
            config,
            shutdown_rx: watch::channel(false).1,
            bot_username: None,
            bot_display_name: None,
            rooms: HashMap::new(),
            ws_stream: None,
            realtime_subscribed_room_ids: HashSet::from(["room1".to_string()]),
            active_thread_keys: HashSet::new(),
            backlog: VecDeque::new(),
            seen_message_ids: HashSet::new(),
            seen_message_order: VecDeque::new(),
            recent_sent_message_ids: HashSet::new(),
            recent_sent_message_order: VecDeque::new(),
            rooms_updated_since: Some("2026-03-29T17:12:01Z".to_string()),
            last_room_refresh: None,
            last_typing_at: HashMap::new(),
            next_realtime_request_id: 1,
        };

        driver.reset_transport_state().expect("transport reset");

        assert!(driver.ws_stream.is_none());
        assert!(driver.realtime_subscribed_room_ids.is_empty());
    }

    #[test]
    fn mentions_mode_accepts_followups_in_active_turin_threads() {
        let config = RocketChatChannelDriverConfig {
            base_url: DEFAULT_BASE_URL.to_string(),
            websocket_url: default_websocket_url(DEFAULT_BASE_URL),
            transport_mode: RocketChatTransportMode::Realtime,
            workspace_id: "rocketchat".to_string(),
            accept_all_rooms: false,
            room_id: Some("room1".to_string()),
            room_name: None,
            user_id: "bot".to_string(),
            token: "token".to_string(),
            poll_interval: Duration::from_millis(DEFAULT_POLL_INTERVAL_MS),
            max_messages_per_poll: DEFAULT_MAX_MESSAGES_PER_POLL,
            start_from_latest: true,
            ignore_bot_messages: true,
            respond_mode: RocketChatRespondMode::Mentions,
            session_scope: ChannelSessionScope::Thread,
            session_scope_dm: None,
            session_scope_group: None,
            session_scope_channel: None,
            reply_mode: RocketChatReplyMode::Thread,
            stream_mode: ChannelStreamMode::Typing,
            persist_thinking: false,
        };
        let driver = RocketChatChannelDriver {
            channel_id: "rocketchat".to_string(),
            client: Client::new(),
            config,
            shutdown_rx: watch::channel(false).1,
            bot_username: None,
            bot_display_name: None,
            rooms: HashMap::new(),
            ws_stream: None,
            realtime_subscribed_room_ids: HashSet::new(),
            active_thread_keys: HashSet::from([active_thread_key("room1", "root-message")]),
            backlog: VecDeque::new(),
            seen_message_ids: HashSet::new(),
            seen_message_order: VecDeque::new(),
            recent_sent_message_ids: HashSet::new(),
            recent_sent_message_order: VecDeque::new(),
            rooms_updated_since: None,
            last_room_refresh: None,
            last_typing_at: HashMap::new(),
            next_realtime_request_id: 1,
        };
        let room = RocketChatResolvedRoom {
            id: "room1".to_string(),
            room_type: RocketChatRoomType::Channel,
            name: Some("general".to_string()),
            friendly_name: Some("General".to_string()),
            usernames: vec![],
            latest_message: None,
            latest_message_id: None,
            latest_message_ts: None,
        };
        let message = RocketChatMessage {
            id: "m2".to_string(),
            text: Some("follow up".to_string()),
            ts: "2026-03-29T00:00:00.000Z".to_string(),
            user: Some(RocketChatMessageUser {
                id: "user1".to_string(),
                username: Some("alice".to_string()),
                name: Some("Alice".to_string()),
            }),
            kind: None,
            thread_root_id: Some("root-message".to_string()),
            mentions: vec![],
            attachments: vec![],
            file: None,
        };

        assert!(driver.should_accept_message(
            &room,
            &message,
            message.user.as_ref().expect("user"),
        ));
    }

    #[test]
    fn direct_messages_can_override_session_scope() {
        let config = RocketChatChannelDriverConfig {
            base_url: DEFAULT_BASE_URL.to_string(),
            websocket_url: default_websocket_url(DEFAULT_BASE_URL),
            transport_mode: RocketChatTransportMode::Realtime,
            workspace_id: "rocketchat".to_string(),
            accept_all_rooms: false,
            room_id: Some("dm-room".to_string()),
            room_name: None,
            user_id: "bot".to_string(),
            token: "token".to_string(),
            poll_interval: Duration::from_millis(DEFAULT_POLL_INTERVAL_MS),
            max_messages_per_poll: DEFAULT_MAX_MESSAGES_PER_POLL,
            start_from_latest: true,
            ignore_bot_messages: true,
            respond_mode: RocketChatRespondMode::Mentions,
            session_scope: ChannelSessionScope::Thread,
            session_scope_dm: Some(ChannelSessionScope::Room),
            session_scope_group: None,
            session_scope_channel: None,
            reply_mode: RocketChatReplyMode::Thread,
            stream_mode: ChannelStreamMode::Typing,
            persist_thinking: false,
        };
        let driver = RocketChatChannelDriver {
            channel_id: "rocketchat".to_string(),
            client: Client::new(),
            config,
            shutdown_rx: watch::channel(false).1,
            bot_username: None,
            bot_display_name: None,
            rooms: HashMap::new(),
            ws_stream: None,
            realtime_subscribed_room_ids: HashSet::new(),
            active_thread_keys: HashSet::new(),
            backlog: VecDeque::new(),
            seen_message_ids: HashSet::new(),
            seen_message_order: VecDeque::new(),
            recent_sent_message_ids: HashSet::new(),
            recent_sent_message_order: VecDeque::new(),
            rooms_updated_since: None,
            last_room_refresh: None,
            last_typing_at: HashMap::new(),
            next_realtime_request_id: 1,
        };
        let room = RocketChatResolvedRoom {
            id: "dm-room".to_string(),
            room_type: RocketChatRoomType::DirectMessage,
            name: None,
            friendly_name: None,
            usernames: vec!["bot".to_string(), "alice".to_string()],
            latest_message: None,
            latest_message_id: None,
            latest_message_ts: None,
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

        assert_eq!(
            driver.effective_session_scope(&room),
            ChannelSessionScope::Room
        );
        assert_eq!(
            driver.thread_id_for_message(&room, &message, driver.effective_session_scope(&room)),
            "dm-room"
        );
    }

    #[test]
    fn channel_reply_mode_downgrades_thread_scope_to_room_scope() {
        let config = RocketChatChannelDriverConfig {
            base_url: DEFAULT_BASE_URL.to_string(),
            websocket_url: default_websocket_url(DEFAULT_BASE_URL),
            transport_mode: RocketChatTransportMode::Realtime,
            workspace_id: "rocketchat".to_string(),
            accept_all_rooms: false,
            room_id: Some("room1".to_string()),
            room_name: None,
            user_id: "bot".to_string(),
            token: "token".to_string(),
            poll_interval: Duration::from_millis(DEFAULT_POLL_INTERVAL_MS),
            max_messages_per_poll: DEFAULT_MAX_MESSAGES_PER_POLL,
            start_from_latest: true,
            ignore_bot_messages: true,
            respond_mode: RocketChatRespondMode::Mentions,
            session_scope: ChannelSessionScope::Thread,
            session_scope_dm: None,
            session_scope_group: None,
            session_scope_channel: None,
            reply_mode: RocketChatReplyMode::Channel,
            stream_mode: ChannelStreamMode::Typing,
            persist_thinking: false,
        };
        let driver = RocketChatChannelDriver {
            channel_id: "rocketchat".to_string(),
            client: Client::new(),
            config,
            shutdown_rx: watch::channel(false).1,
            bot_username: None,
            bot_display_name: None,
            rooms: HashMap::new(),
            ws_stream: None,
            realtime_subscribed_room_ids: HashSet::new(),
            active_thread_keys: HashSet::new(),
            backlog: VecDeque::new(),
            seen_message_ids: HashSet::new(),
            seen_message_order: VecDeque::new(),
            recent_sent_message_ids: HashSet::new(),
            recent_sent_message_order: VecDeque::new(),
            rooms_updated_since: None,
            last_room_refresh: None,
            last_typing_at: HashMap::new(),
            next_realtime_request_id: 1,
        };
        let room = RocketChatResolvedRoom {
            id: "room1".to_string(),
            room_type: RocketChatRoomType::PrivateGroup,
            name: Some("turin".to_string()),
            friendly_name: Some("Turin".to_string()),
            usernames: vec![],
            latest_message: None,
            latest_message_id: None,
            latest_message_ts: None,
        };
        let message = RocketChatMessage {
            id: "m1".to_string(),
            text: Some("@nux hello".to_string()),
            ts: "2026-03-31T00:00:00.000Z".to_string(),
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

        assert_eq!(
            driver.effective_session_scope(&room),
            ChannelSessionScope::Room
        );
        assert_eq!(
            driver.thread_id_for_message(&room, &message, driver.effective_session_scope(&room)),
            "room1"
        );
    }

    #[test]
    fn validate_settings_rejects_deprecated_session_scope_aliases() {
        let error = validate_settings(
            &serde_json::json!({
                "token_env": "ROCKETCHAT_AUTH_TOKEN",
                "user_id": "rbAXPnMktTFbNpwtJ",
                "room_id": "GENERAL123",
                "dm_session_scope": "room"
            }),
            false,
        )
        .expect_err("deprecated alias should fail");

        assert!(error.to_string().contains("session_scope_dm"));
    }

    #[test]
    fn resolve_reply_target_starts_thread_from_triggering_message() {
        let conversation = ChannelConversationKey {
            channel: ChannelKind::new("rocketchat"),
            workspace_id: "rocketchat".to_string(),
            room_id: Some("room1".to_string()),
            thread_id: "room1".to_string(),
            user_id: None,
        };
        let mut outbound = OutboundMessage::text("reply");
        outbound.metadata.insert(
            "rocketchat_reply_to_message_id".to_string(),
            serde_json::json!("message-42"),
        );

        let reply_target = resolve_reply_target(
            "room1",
            &conversation,
            &outbound,
            RocketChatReplyMode::Thread,
        );
        assert_eq!(reply_target.thread_id, Some("message-42"));
        assert!(!reply_target.show_in_channel);
    }

    #[test]
    fn build_rocketchat_send_payload_uses_send_message_shape() {
        let payload = build_rocketchat_send_payload(
            "room1",
            "hello",
            RocketChatReplyTarget {
                thread_id: Some("message-42"),
                show_in_channel: true,
            },
            &[],
        );

        assert_eq!(payload["message"]["rid"], "room1");
        assert_eq!(payload["message"]["msg"], "hello");
        assert_eq!(payload["message"]["parseUrls"], false);
        assert_eq!(payload["message"]["tmid"], "message-42");
        assert_eq!(payload["message"]["tshow"], true);
        assert!(payload["message"].get("attachments").is_none());
        assert!(payload.get("roomId").is_none());
        assert!(payload.get("channel").is_none());
    }

    #[test]
    fn render_rocketchat_message_wraps_markdown_tables_and_thinking() {
        let mut outbound =
            OutboundMessage::text("| Name | Score |\n| --- | --- |\n| Alice | 10 |\n| Bob | 9 |");
        outbound.metadata.insert(
            "channel_final_thinking".to_string(),
            serde_json::json!("brief reasoning"),
        );

        let rendered = render_rocketchat_message(&outbound, true);
        assert!(rendered.contains("Thinking:"));
        assert!(rendered.contains("```"));
        assert!(rendered.contains("| Alice | 10 |"));
    }

    #[test]
    fn channel_reply_quote_renders_reply_context() {
        let mut outbound = OutboundMessage::text("reply");
        outbound.metadata.insert(
            "rocketchat_reply_to_label".to_string(),
            serde_json::json!("Jayadeep Thum (@jayadeep)"),
        );
        outbound.metadata.insert(
            "rocketchat_reply_to_message_link".to_string(),
            serde_json::json!("https://chat.example.com/group/turin?msg=m1"),
        );
        outbound.metadata.insert(
            "rocketchat_reply_to_excerpt".to_string(),
            serde_json::json!("Line one\nLine two\nLine three"),
        );

        let quoted = prepend_channel_reply_quote("reply", &outbound);
        assert_eq!(
            quoted,
            "> [Jayadeep Thum (@jayadeep)](https://chat.example.com/group/turin?msg=m1)\n> Line one\n> Line two\n> Line three\n\nreply"
        );
    }

    #[test]
    fn mentions_mode_accepts_quoted_messages_from_recent_bot_replies() {
        let config = RocketChatChannelDriverConfig {
            base_url: DEFAULT_BASE_URL.to_string(),
            websocket_url: default_websocket_url(DEFAULT_BASE_URL),
            transport_mode: RocketChatTransportMode::Realtime,
            workspace_id: "rocketchat".to_string(),
            accept_all_rooms: false,
            room_id: Some("room1".to_string()),
            room_name: None,
            user_id: "bot".to_string(),
            token: "token".to_string(),
            poll_interval: Duration::from_millis(DEFAULT_POLL_INTERVAL_MS),
            max_messages_per_poll: DEFAULT_MAX_MESSAGES_PER_POLL,
            start_from_latest: true,
            ignore_bot_messages: true,
            respond_mode: RocketChatRespondMode::Mentions,
            session_scope: ChannelSessionScope::Thread,
            session_scope_dm: None,
            session_scope_group: None,
            session_scope_channel: None,
            reply_mode: RocketChatReplyMode::Channel,
            stream_mode: ChannelStreamMode::Typing,
            persist_thinking: false,
        };
        let driver = RocketChatChannelDriver {
            channel_id: "rocketchat".to_string(),
            client: Client::new(),
            config,
            shutdown_rx: watch::channel(false).1,
            bot_username: Some("turinbot".to_string()),
            bot_display_name: Some("Turin".to_string()),
            rooms: HashMap::new(),
            ws_stream: None,
            realtime_subscribed_room_ids: HashSet::new(),
            active_thread_keys: HashSet::new(),
            backlog: VecDeque::new(),
            seen_message_ids: HashSet::new(),
            seen_message_order: VecDeque::new(),
            recent_sent_message_ids: HashSet::from(["bot-message-1".to_string()]),
            recent_sent_message_order: VecDeque::from(["bot-message-1".to_string()]),
            rooms_updated_since: None,
            last_room_refresh: None,
            last_typing_at: HashMap::new(),
            next_realtime_request_id: 1,
        };
        let room = RocketChatResolvedRoom {
            id: "room1".to_string(),
            room_type: RocketChatRoomType::Channel,
            name: Some("general".to_string()),
            friendly_name: Some("General".to_string()),
            usernames: vec![],
            latest_message: None,
            latest_message_id: None,
            latest_message_ts: None,
        };
        let message = RocketChatMessage {
            id: "m2".to_string(),
            text: Some("follow up".to_string()),
            ts: "2026-03-30T00:00:00.000Z".to_string(),
            user: Some(RocketChatMessageUser {
                id: "user1".to_string(),
                username: Some("alice".to_string()),
                name: Some("Alice".to_string()),
            }),
            kind: None,
            thread_root_id: None,
            mentions: vec![],
            attachments: vec![RocketChatApiAttachment {
                text: Some("Earlier reply".to_string()),
                title: None,
                title_link: None,
                message_link: Some(
                    "https://chat.example.com/channel/general?msg=bot-message-1".to_string(),
                ),
                author_name: Some("Turin".to_string()),
                image_url: None,
                audio_url: None,
                video_url: None,
            }],
            file: None,
        };

        assert!(driver.should_accept_message(
            &room,
            &message,
            message.user.as_ref().expect("user"),
        ));
    }

    #[test]
    fn build_rocketchat_message_link_matches_room_paths() {
        let channel_room = RocketChatResolvedRoom {
            id: "room1".to_string(),
            room_type: RocketChatRoomType::Channel,
            name: Some("general".to_string()),
            friendly_name: Some("General".to_string()),
            usernames: vec![],
            latest_message: None,
            latest_message_id: None,
            latest_message_ts: None,
        };
        assert_eq!(
            build_rocketchat_message_link(
                "https://chat.example.com",
                &channel_room,
                Some("nux"),
                "abc123"
            )
            .as_deref(),
            Some("https://chat.example.com/channel/general?msg=abc123")
        );

        let group_room = RocketChatResolvedRoom {
            id: "room2".to_string(),
            room_type: RocketChatRoomType::PrivateGroup,
            name: Some("turin".to_string()),
            friendly_name: Some("Turin".to_string()),
            usernames: vec![],
            latest_message: None,
            latest_message_id: None,
            latest_message_ts: None,
        };
        assert_eq!(
            build_rocketchat_message_link(
                "https://chat.example.com",
                &group_room,
                Some("nux"),
                "def456"
            )
            .as_deref(),
            Some("https://chat.example.com/group/turin?msg=def456")
        );

        let dm_room = RocketChatResolvedRoom {
            id: "room3".to_string(),
            room_type: RocketChatRoomType::DirectMessage,
            name: None,
            friendly_name: None,
            usernames: vec!["jayadeep".to_string(), "nux".to_string()],
            latest_message: None,
            latest_message_id: None,
            latest_message_ts: None,
        };
        assert_eq!(
            build_rocketchat_message_link(
                "https://chat.example.com",
                &dm_room,
                Some("nux"),
                "ghi789"
            )
            .as_deref(),
            Some("https://chat.example.com/direct/jayadeep?msg=ghi789")
        );
    }

    #[test]
    fn default_websocket_url_tracks_base_url_scheme() {
        assert_eq!(
            default_websocket_url("https://chat.example.com"),
            "wss://chat.example.com/websocket"
        );
        assert_eq!(
            default_websocket_url("http://chat.example.com"),
            "ws://chat.example.com/websocket"
        );
    }

    #[test]
    fn ddp_frame_deserializes_success_result_payloads() {
        let frame: RocketChatDdpFrame = serde_json::from_value(serde_json::json!({
            "msg": "result",
            "id": "turin-1",
            "result": {
                "id": "user-id",
                "token": "resume-token",
                "tokenExpires": { "$date": null },
                "type": "resume"
            }
        }))
        .expect("frame");

        assert_eq!(frame.msg.as_deref(), Some("result"));
        assert_eq!(frame.id.as_deref(), Some("turin-1"));
        assert!(login_result_error(&frame).is_none());
    }

    #[test]
    fn rocketchat_message_accepts_ejson_timestamp() {
        let message: RocketChatMessage = serde_json::from_value(serde_json::json!({
            "_id": "message-id",
            "msg": "hello",
            "ts": { "$date": "2026-03-29T17:12:01.123Z" },
            "u": {
                "_id": "user-id",
                "username": "alice",
                "name": "Alice"
            },
            "mentions": [],
            "attachments": []
        }))
        .expect("message");

        assert_eq!(message.ts, "2026-03-29T17:12:01.123Z");
    }

    #[test]
    fn rocketchat_room_info_accepts_ejson_timestamps() {
        let room: RocketChatRoomInfo = serde_json::from_value(serde_json::json!({
            "_id": "room-id",
            "t": "c",
            "_updatedAt": { "$date": "2026-03-29T17:12:01.123Z" },
            "lm": { "$date": "2026-03-29T17:10:00.000Z" }
        }))
        .expect("room");

        assert_eq!(room.updated_at.as_deref(), Some("2026-03-29T17:12:01.123Z"));
        assert_eq!(
            room.last_message_at.as_deref(),
            Some("2026-03-29T17:10:00Z")
        );
    }
}
