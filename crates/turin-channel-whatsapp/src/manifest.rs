use serde_json::json;
use turin_channel_core::{
    ChannelAdapterManifest, ChannelAuthFlow, ChannelAuthFlowKind, ChannelConfigField,
    ChannelConfigFieldOption, ChannelConfigTarget, ChannelConfigTargetKind,
    ChannelFieldVisibilityRule, ChannelIdentitySelectors, ChannelInstallManifest,
    ChannelRuntimeCapabilities, ChannelRuntimeManifest, ChannelSetupManifest,
    DEFAULT_MAX_INBOUND_TEXT_CHARS,
};

use crate::{DEFAULT_AUTH_FLOW_ID, DEFAULT_PERSONAL_TRIGGER_PREFIX, DEFAULT_WORKSPACE_ID};

pub fn adapter_manifest() -> ChannelAdapterManifest {
    ChannelAdapterManifest {
        protocol_version: turin_channel_core::CHANNEL_ADAPTER_PROTOCOL_VERSION,
        kind: "whatsapp".to_string(),
        display_name: "WhatsApp".to_string(),
        runtime: ChannelRuntimeManifest {
            session_scopes: vec!["user".to_string(), "room".to_string()],
            enum_settings: vec![turin_channel_core::ChannelEnumSetting {
                key: "session_scope".to_string(),
                options: vec!["user".to_string(), "room".to_string()],
            }],
            capabilities: ChannelRuntimeCapabilities {
                dm: true,
                groups: true,
                threads: false,
                attachments: true,
                streaming: false,
            },
            identity_selectors: ChannelIdentitySelectors {
                matching_rules: vec!["jid".to_string(), "phone".to_string()],
                examples: vec![
                    "15551234567".to_string(),
                    "15551234567@s.whatsapp.net".to_string(),
                ],
            },
        },
        setup: Some(ChannelSetupManifest {
            required_secrets: vec![],
            instructions: Some(
                "Link Turin to WhatsApp Web by scanning a QR code. For headless servers, provide a phone number to receive a pairing code instead.".to_string(),
            ),
            setup_url: None,
            validation_checks: vec![],
            config_fields: vec![
                ChannelConfigField {
                    key: "account_mode".to_string(),
                    label: Some("Account Mode".to_string()),
                    field_type: "select".to_string(),
                    prompt: Some(
                        "Will Turin use a personal WhatsApp account or a dedicated agent number?"
                            .to_string(),
                    ),
                    help: Some(
                        "Personal mode defaults to a trigger prefix so normal chats stay quiet. Dedicated mode is better when the linked account belongs only to the agent.".to_string(),
                    ),
                    default: Some(json!("personal")),
                    options: vec![
                        ChannelConfigFieldOption {
                            value: "personal".to_string(),
                            label: Some("Personal account".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "dedicated".to_string(),
                            label: Some("Dedicated agent number".to_string()),
                        },
                    ],
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "account_mode".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "workspace_id".to_string(),
                    label: Some("Workspace ID".to_string()),
                    field_type: "text".to_string(),
                    prompt: Some(
                        "Workspace identifier used when routing WhatsApp conversations into Turin"
                            .to_string(),
                    ),
                    help: Some(
                        "Defaults to 'whatsapp' and is usually fine to leave alone.".to_string(),
                    ),
                    default: Some(json!(DEFAULT_WORKSPACE_ID)),
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
                    prompt: Some("How should new WhatsApp chats be admitted?".to_string()),
                    help: Some(
                        "Auto is the easiest onboarding mode; pending requires explicit approval later."
                            .to_string(),
                    ),
                    default: Some(json!("auto")),
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
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "pairing_mode".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "session_scope".to_string(),
                    label: Some("Session Scope".to_string()),
                    field_type: "select".to_string(),
                    prompt: Some("How should WhatsApp conversation memory be scoped?".to_string()),
                    help: Some(
                        "Room shares memory across the whole chat; user keeps memory isolated per sender."
                            .to_string(),
                    ),
                    default: Some(json!("user")),
                    options: vec![
                        ChannelConfigFieldOption {
                            value: "user".to_string(),
                            label: Some("Per user".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "room".to_string(),
                            label: Some("Per chat".to_string()),
                        },
                    ],
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "session_scope".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "trigger_prefix".to_string(),
                    label: Some("Trigger Prefix".to_string()),
                    field_type: "text".to_string(),
                    prompt: Some(
                        "Prefix required before Turin should answer in this WhatsApp account"
                            .to_string(),
                    ),
                    help: Some(
                        "Personal mode defaults to '/turin'. Dedicated accounts can usually leave this empty."
                            .to_string(),
                    ),
                    default: Some(json!(DEFAULT_PERSONAL_TRIGGER_PREFIX)),
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "trigger_prefix".to_string(),
                    }),
                    visible_if: Some(ChannelFieldVisibilityRule {
                        key: "account_mode".to_string(),
                        equals: json!("personal"),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "max_inbound_text_chars".to_string(),
                    label: Some("Max Inbound Text Chars".to_string()),
                    field_type: "number".to_string(),
                    help: Some(
                        "Safety cap for inbound WhatsApp text retained before Turin truncates it."
                            .to_string(),
                    ),
                    default: Some(json!(DEFAULT_MAX_INBOUND_TEXT_CHARS)),
                    advanced: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "max_inbound_text_chars".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "pairing_users".to_string(),
                    label: Some("Pairing Users".to_string()),
                    field_type: "string_list".to_string(),
                    prompt: Some(
                        "Optional phone numbers or JIDs allowed to pair new chats".to_string(),
                    ),
                    help: Some("Leave empty to allow any sender to trigger pairing.".to_string()),
                    advanced: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "pairing_users".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "allowed_users".to_string(),
                    label: Some("Allowed Users".to_string()),
                    field_type: "string_list".to_string(),
                    prompt: Some(
                        "Optional phone numbers or JIDs allowed to interact after approval"
                            .to_string(),
                    ),
                    help: Some("Leave empty to allow any user in approved chats.".to_string()),
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
                    prompt: Some(
                        "Optional phone numbers or JIDs that should always be denied".to_string(),
                    ),
                    advanced: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "banned_users".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "allowed_chats".to_string(),
                    label: Some("Allowed Chats".to_string()),
                    field_type: "string_list".to_string(),
                    prompt: Some(
                        "Optional WhatsApp chats where Turin is allowed to listen".to_string(),
                    ),
                    help: Some(
                        "Use full JIDs like '15551234567@s.whatsapp.net' or group IDs. Leave empty to allow any chat not explicitly banned.".to_string(),
                    ),
                    advanced: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "allowed_chats".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "banned_chats".to_string(),
                    label: Some("Banned Chats".to_string()),
                    field_type: "string_list".to_string(),
                    prompt: Some(
                        "Optional WhatsApp chats where Turin must stay silent".to_string(),
                    ),
                    help: Some(
                        "Banned chats override allowed chats. Use this to keep personal or administrative chats out of the agent.".to_string(),
                    ),
                    advanced: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "banned_chats".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "session_store_path".to_string(),
                    label: Some("Session Store Path".to_string()),
                    field_type: "text".to_string(),
                    prompt: Some(
                        "Optional SQLite path for the WhatsApp linked-device session".to_string(),
                    ),
                    help: Some(
                        "Leave empty to let the pairing flow generate a session store automatically."
                            .to_string(),
                    ),
                    advanced: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "session_store_path".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "pair_code_phone_number".to_string(),
                    label: Some("Pair Code Phone Number".to_string()),
                    field_type: "text".to_string(),
                    prompt: Some(
                        "Optional international phone number for headless pairing-code auth"
                            .to_string(),
                    ),
                    help: Some(
                        "Leave empty to use QR-only pairing. This value is cleared after pairing completes."
                            .to_string(),
                    ),
                    advanced: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "pair_code_phone_number".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "pair_code_custom_code".to_string(),
                    label: Some("Custom Pair Code".to_string()),
                    field_type: "text".to_string(),
                    prompt: Some(
                        "Optional custom 8-character pairing code for headless linking".to_string(),
                    ),
                    help: Some(
                        "Uses the Crockford Base32 alphabet and is cleared after pairing completes."
                            .to_string(),
                    ),
                    advanced: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "pair_code_custom_code".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
            ],
            auth_flows: vec![ChannelAuthFlow {
                id: DEFAULT_AUTH_FLOW_ID.to_string(),
                kind: ChannelAuthFlowKind::QrPairing,
                label: Some("Link WhatsApp account".to_string()),
                prompt: Some(
                    "Start WhatsApp linking now (QR by default, pair code when phone number is set)"
                        .to_string(),
                ),
                help: Some(
                    "The sidecar opens a temporary WhatsApp session, shows a QR code, and optionally generates a pairing code for headless servers.".to_string(),
                ),
                hint: Some(
                    "You can rerun this later if the linked-device session is revoked.".to_string(),
                ),
                advanced: false,
                visible_if: None,
            }],
        }),
        install: Some(ChannelInstallManifest {
            binary_name: Some("turin-channel-whatsapp".to_string()),
        }),
    }
}
