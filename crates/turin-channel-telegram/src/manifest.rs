use anyhow::Result;
use turin_channel_core::{
    ChannelAdapterManifest, ChannelAuthFlowPollRequest, ChannelAuthFlowPollResponse,
    ChannelAuthFlowStartRequest, ChannelAuthFlowStartResponse, ChannelConfigField,
    ChannelConfigFieldOption, ChannelIdentitySelectors, ChannelInstallManifest,
    ChannelRuntimeCapabilities, ChannelRuntimeManifest, ChannelSecretRequirement,
    ChannelSetupManifest, ChannelValidationCheck, channel_enum_setting, channel_setting_target_opt,
    max_inbound_text_chars_field,
};

pub fn start_auth_flow(
    _request: &ChannelAuthFlowStartRequest,
) -> Result<ChannelAuthFlowStartResponse> {
    anyhow::bail!("Telegram does not expose manifest auth flows")
}

pub fn poll_auth_flow(
    _request: &ChannelAuthFlowPollRequest,
) -> Result<ChannelAuthFlowPollResponse> {
    anyhow::bail!("Telegram does not expose manifest auth flows")
}

pub fn adapter_manifest() -> ChannelAdapterManifest {
    ChannelAdapterManifest {
        protocol_version: turin_channel_core::CHANNEL_ADAPTER_PROTOCOL_VERSION,
        kind: "telegram".to_string(),
        display_name: "Telegram".to_string(),
        runtime: ChannelRuntimeManifest {
            session_scopes: vec!["user".to_string(), "thread".to_string(), "room".to_string()],
            enum_settings: vec![
                channel_enum_setting(
                    "respond_mode",
                    ["all", "mentions", "replies", "mentions_or_replies"],
                ),
                channel_enum_setting("session_scope", ["user", "thread", "room"]),
                channel_enum_setting("session_scope_dm", ["user", "thread", "room"]),
                channel_enum_setting("session_scope_group", ["user", "thread", "room"]),
                channel_enum_setting("session_scope_channel", ["user", "thread", "room"]),
            ],
            capabilities: ChannelRuntimeCapabilities {
                dm: true,
                groups: true,
                threads: true,
                attachments: true,
                streaming: true,
            },
            identity_selectors: ChannelIdentitySelectors {
                matching_rules: vec!["id".to_string(), "username".to_string()],
                examples: vec!["498502840".to_string(), "jthum".to_string()],
            },
        },
        setup: Some(ChannelSetupManifest {
            required_secrets: vec![ChannelSecretRequirement {
                name: "telegram_bot_token".to_string(),
                env_var: "TELEGRAM_BOT_TOKEN".to_string(),
                display_name: Some("Telegram bot token".to_string()),
                help: Some("Get this from @BotFather on Telegram.".to_string()),
                optional: false,
                hints: vec!["Looks like 123456789:AABBccDDeeFFgg...".to_string()],
                target: channel_setting_target_opt("token_env"),
                validate: Some(ChannelValidationCheck {
                    kind: "http_get".to_string(),
                    url_template: Some(
                        "https://api.telegram.org/bot{telegram_bot_token}/getMe".to_string(),
                    ),
                    message: Some(
                        "Verify that the supplied Telegram bot token is valid.".to_string(),
                    ),
                }),
            }],
            instructions: Some("Create a bot with BotFather, copy the token, and choose the channel settings you want Turin to apply.".to_string()),
            setup_url: Some("https://t.me/BotFather".to_string()),
            validation_checks: vec![],
            config_fields: vec![
                ChannelConfigField {
                    key: "workspace_id".to_string(),
                    label: Some("Workspace ID".to_string()),
                    field_type: "text".to_string(),
                    prompt: Some("Workspace identifier used when routing Telegram conversations into Turin".to_string()),
                    help: Some("Defaults to 'telegram' and is usually fine to leave alone.".to_string()),
                    default: Some(serde_json::json!("telegram")),
                    target: channel_setting_target_opt("workspace_id"),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "pairing_mode".to_string(),
                    label: Some("Pairing Mode".to_string()),
                    field_type: "select".to_string(),
                    prompt: Some("How should new Telegram chats be admitted?".to_string()),
                    help: Some("Auto is the easiest onboarding mode; pending requires explicit approval later.".to_string()),
                    default: Some(serde_json::json!("auto")),
                    options: vec![
                        ChannelConfigFieldOption {
                            value: "auto".to_string(),
                            label: Some("Auto approve new chats".to_string()),
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
                    key: "respond_mode".to_string(),
                    label: Some("Respond Mode".to_string()),
                    field_type: "select".to_string(),
                    prompt: Some("When should the bot respond in shared chats?".to_string()),
                    help: Some("Mentions or replies is a safe default for groups.".to_string()),
                    default: Some(serde_json::json!("mentions_or_replies")),
                    options: vec![
                        ChannelConfigFieldOption {
                            value: "all".to_string(),
                            label: Some("Every message".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "mentions".to_string(),
                            label: Some("Mentions only".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "replies".to_string(),
                            label: Some("Replies only".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "mentions_or_replies".to_string(),
                            label: Some("Mentions or replies".to_string()),
                        },
                    ],
                    target: channel_setting_target_opt("respond_mode"),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "session_scope".to_string(),
                    label: Some("Session Scope".to_string()),
                    field_type: "select".to_string(),
                    prompt: Some("How should Telegram conversation memory be scoped?".to_string()),
                    help: Some("Room shares memory across the room; user keeps memory isolated per sender.".to_string()),
                    default: Some(serde_json::json!("user")),
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
                    prompt: Some("Optional session scope override for private Telegram chats".to_string()),
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
                    prompt: Some("Optional session scope override for Telegram groups and supergroups".to_string()),
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
                    prompt: Some("Optional session scope override for Telegram channels".to_string()),
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
                    key: "pairing_users".to_string(),
                    label: Some("Pairing Users".to_string()),
                    field_type: "string_list".to_string(),
                    prompt: Some("Optional usernames or IDs allowed to pair new rooms".to_string()),
                    help: Some("Leave empty to allow any sender to trigger pairing.".to_string()),
                    advanced: true,
                    target: channel_setting_target_opt("pairing_users"),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "allowed_users".to_string(),
                    label: Some("Allowed Users".to_string()),
                    field_type: "string_list".to_string(),
                    prompt: Some("Optional usernames or IDs allowed to interact after approval".to_string()),
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
                max_inbound_text_chars_field(
                    "Safety cap for inbound Telegram text retained before Turin truncates it.",
                ),
            ],
            auth_flows: vec![],
        }),
        install: Some(ChannelInstallManifest {
            binary_name: Some("turin-channel-telegram".to_string()),
        }),
    }
}
