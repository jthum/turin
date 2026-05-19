use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Deserializer};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;
use tokio::sync::watch;
use tokio::time::sleep;
use tracing::warn;
#[cfg(test)]
use turin_channel_core::MessageBlock;
use turin_channel_core::{
    ChannelAdapterManifest, ChannelAttachment, ChannelAuthFlowPollRequest,
    ChannelAuthFlowPollResponse, ChannelAuthFlowStartRequest, ChannelAuthFlowStartResponse,
    ChannelCapabilities, ChannelConfigField, ChannelConfigFieldOption, ChannelConfigTarget,
    ChannelConfigTargetKind, ChannelConversationKey, ChannelEnumSetting, ChannelIdentitySelectors,
    ChannelInstallManifest, ChannelKind, ChannelMessageRef, ChannelRuntimeCapabilities,
    ChannelRuntimeManifest, ChannelSecretRequirement, ChannelSessionScope, ChannelSetupManifest,
    ChannelUser, DEFAULT_MAX_INBOUND_TEXT_CHARS, InboundEvent, OutboundMessage, bound_inbound_text,
};
use turin_channel_runner::{ChannelDriver, ChannelProgressUpdate, ChannelStreamMode};

mod realtime;
mod render;
mod settings;
use realtime::RocketChatWsStream;
#[cfg(test)]
use realtime::{RocketChatDdpFrame, login_result_error};
#[cfg(test)]
use render::RocketChatReplyTarget;
#[cfg(test)]
use render::render_text_blocks_for_test as render_text_blocks;
use render::{
    build_rocketchat_send_payload, prepend_channel_reply_quote, render_rocketchat_message,
    reply_excerpt, resolve_reply_target, split_for_rocketchat_content,
};
pub use settings::{RocketChatChannelDriverConfig, validate_settings};
#[cfg(test)]
pub(crate) use settings::{default_websocket_url, parse_settings};

const DEFAULT_BASE_URL: &str = "http://localhost:3000";
const DEFAULT_TRANSPORT_MODE: &str = "realtime";
const DEFAULT_STREAM_MODE: &str = "typing";
const DEFAULT_POLL_INTERVAL_MS: u64 = 1_000;
const DEFAULT_MAX_MESSAGES_PER_POLL: u16 = 50;
const MAX_MESSAGES_PER_POLL: u16 = 100;
const SEEN_MESSAGE_IDS_LIMIT: usize = 1_024;
const RECENT_SENT_MESSAGE_IDS_LIMIT: usize = 256;
const DEFAULT_REALTIME_RECONNECT_DELAY_MS: u64 = 2_000;
const ROCKETCHAT_TYPING_STATUS_INTERVAL_SECONDS: u64 = 4;
const ROCKETCHAT_HTTP_TIMEOUT_SECONDS: u64 = 30;
const ROCKETCHAT_HTTP_CONNECT_TIMEOUT_SECONDS: u64 = 10;
const ROCKETCHAT_REALTIME_CONNECT_TIMEOUT_SECONDS: u64 = 15;
const ROCKETCHAT_REALTIME_HANDSHAKE_TIMEOUT_SECONDS: u64 = 15;
const ROCKETCHAT_REALTIME_KEEPALIVE_SECONDS: u64 = 15;
const ROCKETCHAT_REALTIME_STALE_SECONDS: u64 = 45;

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
                ChannelConfigField {
                    key: "max_inbound_text_chars".to_string(),
                    label: Some("Max Inbound Text Chars".to_string()),
                    field_type: "number".to_string(),
                    help: Some(
                        "Safety cap for inbound Rocket.Chat text retained before Turin truncates it."
                            .to_string(),
                    ),
                    default: Some(serde_json::json!(DEFAULT_MAX_INBOUND_TEXT_CHARS)),
                    advanced: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "max_inbound_text_chars".to_string(),
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
    last_realtime_activity_at: Option<Instant>,
    last_realtime_keepalive_at: Option<Instant>,
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
        let client = build_http_client()?;

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
            last_realtime_activity_at: None,
            last_realtime_keepalive_at: None,
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
        text = bound_inbound_text(text, &mut metadata, self.config.max_inbound_text_chars);

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

fn progress_key(conversation: &ChannelConversationKey) -> Result<String> {
    serde_json::to_string(conversation)
        .with_context(|| "[rocketchat_progress_key_invalid] Failed to serialize conversation key")
}

fn active_thread_key(room_id: &str, thread_id: &str) -> String {
    format!("{room_id}:{thread_id}")
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

fn build_http_client() -> Result<Client> {
    Client::builder()
        .connect_timeout(Duration::from_secs(ROCKETCHAT_HTTP_CONNECT_TIMEOUT_SECONDS))
        .timeout(Duration::from_secs(ROCKETCHAT_HTTP_TIMEOUT_SECONDS))
        .build()
        .context("[rocketchat_http_client_build_failed] Failed to build Rocket.Chat HTTP client")
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

fn subscription_request_id(room_id: &str) -> String {
    format!("room:{room_id}")
}

fn subscription_room_id(request_id: &str) -> Option<&str> {
    request_id.strip_prefix("room:")
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
mod tests;
