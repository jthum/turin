use anyhow::Result;
use turin_channel_core::{
    ChannelAdapterManifest, ChannelAuthFlowPollRequest, ChannelAuthFlowPollResponse,
    ChannelAuthFlowStartRequest, ChannelAuthFlowStartResponse, ChannelConfigField,
    ChannelIdentitySelectors, ChannelInstallManifest, ChannelRuntimeCapabilities,
    ChannelRuntimeManifest, ChannelSecretRequirement, ChannelSetupManifest, channel_enum_setting,
    channel_setting_target_opt, config_field_options, max_inbound_text_chars_field,
};

pub fn start_auth_flow(
    _request: &ChannelAuthFlowStartRequest,
) -> Result<ChannelAuthFlowStartResponse> {
    anyhow::bail!("Discord does not expose manifest auth flows")
}

pub fn poll_auth_flow(
    _request: &ChannelAuthFlowPollRequest,
) -> Result<ChannelAuthFlowPollResponse> {
    anyhow::bail!("Discord does not expose manifest auth flows")
}

pub fn adapter_manifest() -> ChannelAdapterManifest {
    ChannelAdapterManifest {
        protocol_version: turin_channel_core::CHANNEL_ADAPTER_PROTOCOL_VERSION,
        kind: "discord".to_string(),
        display_name: "Discord".to_string(),
        runtime: ChannelRuntimeManifest {
            session_scopes: vec!["user".to_string(), "thread".to_string()],
            enum_settings: vec![channel_enum_setting("session_scope", ["user", "thread"])],
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
                help: Some(
                    "Get this from the Discord developer portal for your application.".to_string(),
                ),
                optional: false,
                hints: vec!["Usually a long bot token string issued by Discord.".to_string()],
                target: channel_setting_target_opt("token_env"),
                validate: None,
            }],
            instructions: Some("Create a Discord application, add a bot, enable the intents you need, and invite it to the target server.".to_string()),
            setup_url: Some("https://discord.com/developers/applications".to_string()),
            validation_checks: vec![],
            config_fields: vec![
                ChannelConfigField {
                    key: "channel_id".to_string(),
                    label: Some("Channel ID".to_string()),
                    field_type: "text".to_string(),
                    prompt: Some("Discord channel ID to connect Turin to".to_string()),
                    help: Some("Enable developer mode in Discord to copy the channel ID.".to_string()),
                    required: true,
                    target: channel_setting_target_opt("channel_id"),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "workspace_id".to_string(),
                    label: Some("Workspace ID".to_string()),
                    field_type: "text".to_string(),
                    default: Some(serde_json::json!("discord")),
                    target: channel_setting_target_opt("workspace_id"),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "session_scope".to_string(),
                    label: Some("Session Scope".to_string()),
                    field_type: "select".to_string(),
                    default: Some(serde_json::json!("thread")),
                    options: config_field_options([("user", "Per user"), ("thread", "Per thread")]),
                    target: channel_setting_target_opt("session_scope"),
                    ..ChannelConfigField::default()
                },
                max_inbound_text_chars_field(
                    "Safety cap for inbound text retained from Discord before Turin truncates it.",
                ),
            ],
            auth_flows: vec![],
        }),
        install: Some(ChannelInstallManifest {
            binary_name: Some("turin-channel-discord".to_string()),
        }),
    }
}
