use anyhow::Result;
use turin_channel_core::{
    ChannelAdapterManifest, ChannelAuthFlowPollRequest, ChannelAuthFlowPollResponse,
    ChannelAuthFlowStartRequest, ChannelAuthFlowStartResponse, ChannelConfigField,
    ChannelConfigFieldOption, ChannelIdentitySelectors, ChannelInstallManifest,
    ChannelRuntimeCapabilities, ChannelRuntimeManifest, ChannelSecretRequirement,
    ChannelSetupManifest, channel_enum_setting, channel_setting_target_opt,
    max_inbound_text_chars_field,
};

use crate::{
    DEFAULT_BASE_URL, DEFAULT_POLL_INTERVAL_MS, DEFAULT_STREAM_MODE, DEFAULT_TRANSPORT_MODE,
};

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
                channel_enum_setting("transport_mode", ["realtime", "polling"]),
                channel_enum_setting("respond_mode", ["all", "mentions"]),
                channel_enum_setting("session_scope", ["user", "thread", "room"]),
                channel_enum_setting("session_scope_dm", ["user", "thread", "room"]),
                channel_enum_setting("session_scope_group", ["user", "thread", "room"]),
                channel_enum_setting("session_scope_channel", ["user", "thread", "room"]),
                channel_enum_setting("reply_mode", ["thread", "channel", "thread_and_channel"]),
                channel_enum_setting("stream_mode", ["off", "typing"]),
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
                target: channel_setting_target_opt("token_env"),
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
                    target: channel_setting_target_opt("base_url"),
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
                    target: channel_setting_target_opt("user_id"),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "workspace_id".to_string(),
                    label: Some("Workspace ID".to_string()),
                    field_type: "text".to_string(),
                    help: Some("Defaults to 'rocketchat' and is usually fine to leave alone.".to_string()),
                    default: Some(serde_json::json!("rocketchat")),
                    advanced: true,
                    target: channel_setting_target_opt("workspace_id"),
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
                    target: channel_setting_target_opt("pairing_mode"),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "pairing_users".to_string(),
                    label: Some("Pairing Users".to_string()),
                    field_type: "string_list".to_string(),
                    prompt: Some("Optional usernames or IDs allowed to pair new Rocket.Chat rooms".to_string()),
                    help: Some("Leave empty to allow any sender to trigger room pairing.".to_string()),
                    advanced: true,
                    target: channel_setting_target_opt("pairing_users"),
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
                    target: channel_setting_target_opt("transport_mode"),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "websocket_url".to_string(),
                    label: Some("WebSocket URL".to_string()),
                    field_type: "text".to_string(),
                    prompt: Some("Optional Rocket.Chat websocket URL override".to_string()),
                    help: Some("Leave empty to derive it automatically from the server URL as ws(s)://.../websocket.".to_string()),
                    advanced: true,
                    target: channel_setting_target_opt("websocket_url"),
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
                    target: channel_setting_target_opt("room_id"),
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
                    target: channel_setting_target_opt("room_name"),
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
                    target: channel_setting_target_opt("respond_mode"),
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
                    target: channel_setting_target_opt("reply_mode"),
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
                    target: channel_setting_target_opt("session_scope"),
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
                    target: channel_setting_target_opt("session_scope_dm"),
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
                    target: channel_setting_target_opt("session_scope_group"),
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
                    target: channel_setting_target_opt("session_scope_channel"),
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
                    target: channel_setting_target_opt("stream_mode"),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "persist_thinking".to_string(),
                    label: Some("Include Final Thinking".to_string()),
                    field_type: "boolean".to_string(),
                    help: Some("When enabled, Turin prepends the model's final thinking to the posted reply.".to_string()),
                    default: Some(serde_json::json!(false)),
                    advanced: true,
                    target: channel_setting_target_opt("persist_thinking"),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "allowed_users".to_string(),
                    label: Some("Allowed Users".to_string()),
                    field_type: "string_list".to_string(),
                    prompt: Some("Optional usernames or IDs allowed to interact".to_string()),
                    help: Some("Leave empty to allow any user in approved rooms.".to_string()),
                    advanced: true,
                    target: channel_setting_target_opt("allowed_users"),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "banned_users".to_string(),
                    label: Some("Banned Users".to_string()),
                    field_type: "string_list".to_string(),
                    prompt: Some("Optional usernames or IDs that should always be denied".to_string()),
                    advanced: true,
                    target: channel_setting_target_opt("banned_users"),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "poll_interval_ms".to_string(),
                    label: Some("Poll Interval (ms)".to_string()),
                    field_type: "number".to_string(),
                    default: Some(serde_json::json!(DEFAULT_POLL_INTERVAL_MS)),
                    advanced: true,
                    target: channel_setting_target_opt("poll_interval_ms"),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "start_from_latest".to_string(),
                    label: Some("Start From Latest".to_string()),
                    field_type: "boolean".to_string(),
                    help: Some("Skip older room history and only process new messages from now on.".to_string()),
                    default: Some(serde_json::json!(true)),
                    advanced: true,
                    target: channel_setting_target_opt("start_from_latest"),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "ignore_bot_messages".to_string(),
                    label: Some("Ignore Bot Messages".to_string()),
                    field_type: "boolean".to_string(),
                    default: Some(serde_json::json!(true)),
                    advanced: true,
                    target: channel_setting_target_opt("ignore_bot_messages"),
                    ..ChannelConfigField::default()
                },
                max_inbound_text_chars_field(
                    "Safety cap for inbound Rocket.Chat text retained before Turin truncates it.",
                ),
            ],
            auth_flows: vec![],
        }),
        install: Some(ChannelInstallManifest {
            binary_name: Some("turin-channel-rocketchat".to_string()),
        }),
    }
}
