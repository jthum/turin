use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use pulldown_cmark::{CodeBlockKind, Event, Options, Parser, Tag, TagEnd};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
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
    ChannelUser, ChannelValidationCheck, DEFAULT_MAX_INBOUND_TEXT_CHARS, InboundEvent,
    MessageBlock, OutboundMessage, bound_inbound_text,
};
use turin_channel_runner::{ChannelDriver, ChannelProgressUpdate, ChannelStreamMode};

const DEFAULT_BASE_URL: &str = "https://api.telegram.org";
const TELEGRAM_MESSAGE_MAX_LEN: usize = 4_096;
const MAX_STARTUP_SKIP_BATCHES: usize = 32;
const MAX_API_REQUEST_ATTEMPTS: u32 = 5;

#[derive(Debug, Clone)]
pub struct TelegramChannelDriverConfig {
    pub base_url: String,
    pub workspace_id: String,
    pub chat_ids: Vec<String>,
    pub accept_all_chats: bool,
    pub token: String,
    pub poll_timeout_seconds: u64,
    pub poll_interval: Duration,
    pub max_updates_per_poll: u8,
    pub max_inbound_text_chars: usize,
    pub start_from_latest: bool,
    pub ignore_bot_messages: bool,
    pub respond_mode: TelegramRespondMode,
    pub session_scope: ChannelSessionScope,
    pub session_scope_dm: Option<ChannelSessionScope>,
    pub session_scope_group: Option<ChannelSessionScope>,
    pub session_scope_channel: Option<ChannelSessionScope>,
    pub stream_mode: ChannelStreamMode,
    pub stream_thinking: bool,
    pub persist_thinking: bool,
}

pub fn validate_settings(
    settings: &serde_json::Value,
    allow_unconfigured_chats: bool,
) -> Result<()> {
    parse_settings(settings, allow_unconfigured_chats).map(|_| ())
}

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
                ChannelEnumSetting {
                    key: "respond_mode".to_string(),
                    options: vec![
                        "all".to_string(),
                        "mentions".to_string(),
                        "replies".to_string(),
                        "mentions_or_replies".to_string(),
                    ],
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
                target: Some(ChannelConfigTarget {
                    kind: ChannelConfigTargetKind::ChannelSetting,
                    name: "token_env".to_string(),
                }),
                validate: Some(ChannelValidationCheck {
                    kind: "http_get".to_string(),
                    url_template: Some(
                        "https://api.telegram.org/bot{telegram_bot_token}/getMe".to_string(),
                    ),
                    message: Some("Verify that the supplied Telegram bot token is valid.".to_string()),
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
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "pairing_mode".to_string(),
                    }),
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
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "session_scope_channel".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "pairing_users".to_string(),
                    label: Some("Pairing Users".to_string()),
                    field_type: "string_list".to_string(),
                    prompt: Some("Optional usernames or IDs allowed to pair new rooms".to_string()),
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
                    prompt: Some("Optional usernames or IDs allowed to interact after approval".to_string()),
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
                    key: "max_inbound_text_chars".to_string(),
                    label: Some("Max Inbound Text Chars".to_string()),
                    field_type: "number".to_string(),
                    help: Some(
                        "Safety cap for inbound Telegram text retained before Turin truncates it."
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
            binary_name: Some("turin-channel-telegram".to_string()),
        }),
    }
}

impl TelegramChannelDriverConfig {
    pub fn from_settings(
        settings: &serde_json::Value,
        allow_unconfigured_chats: bool,
    ) -> Result<Self> {
        let settings = parse_settings(settings, allow_unconfigured_chats)?;
        let token_env = settings.token_env.as_str();
        let token = std::env::var(token_env).map_err(|_| {
            anyhow!(
                "[telegram_auth_missing_token] Telegram bot token env var '{}' is not set for channel adapter",
                token_env
            )
        })?;

        Ok(Self {
            base_url: settings.base_url,
            workspace_id: settings.workspace_id,
            chat_ids: settings.chat_ids,
            accept_all_chats: settings.accept_all_chats,
            token,
            poll_timeout_seconds: settings.poll_timeout_seconds,
            poll_interval: Duration::from_millis(settings.poll_interval_ms),
            max_updates_per_poll: settings.max_updates_per_poll,
            max_inbound_text_chars: settings.max_inbound_text_chars,
            start_from_latest: settings.start_from_latest,
            ignore_bot_messages: settings.ignore_bot_messages,
            respond_mode: settings.respond_mode,
            session_scope: settings.session_scope,
            session_scope_dm: settings.session_scope_dm,
            session_scope_group: settings.session_scope_group,
            session_scope_channel: settings.session_scope_channel,
            stream_mode: settings.stream_mode,
            stream_thinking: settings.stream_thinking,
            persist_thinking: settings.persist_thinking,
        })
    }

    fn primary_chat_id(&self) -> &str {
        self.chat_ids
            .first()
            .map(String::as_str)
            .unwrap_or_default()
    }

    fn allows_chat_id(&self, chat_id: &str) -> bool {
        self.chat_ids.iter().any(|allowed| allowed == chat_id)
    }
}

#[derive(Debug, Clone)]
struct TelegramChannelSettings {
    token_env: String,
    base_url: String,
    workspace_id: String,
    chat_ids: Vec<String>,
    poll_timeout_seconds: u64,
    poll_interval_ms: u64,
    max_updates_per_poll: u8,
    max_inbound_text_chars: usize,
    start_from_latest: bool,
    ignore_bot_messages: bool,
    respond_mode: TelegramRespondMode,
    session_scope: ChannelSessionScope,
    session_scope_dm: Option<ChannelSessionScope>,
    session_scope_group: Option<ChannelSessionScope>,
    session_scope_channel: Option<ChannelSessionScope>,
    stream_mode: ChannelStreamMode,
    stream_thinking: bool,
    persist_thinking: bool,
    accept_all_chats: bool,
}

fn parse_settings(
    settings: &serde_json::Value,
    allow_unconfigured_chats: bool,
) -> Result<TelegramChannelSettings> {
    let settings = settings
        .as_object()
        .ok_or_else(|| anyhow!("Telegram channel settings must be a JSON object"))?;
    reject_deprecated_session_scope_keys(settings)?;

    let token_env = read_required_string(
        settings,
        "token_env",
        "[telegram_config_missing_token_env] Telegram channel setting 'token_env' is required",
        "[telegram_config_invalid_token_env] Telegram channel setting 'token_env' must not be empty",
    )?
    .to_string();

    let chat_ids = match read_chat_ids(settings) {
        Ok(ids) => ids,
        Err(_) if allow_unconfigured_chats => Vec::new(),
        Err(err) => {
            return Err(anyhow!(
                "[telegram_config_missing_chat_id] Telegram channel setting 'chat_id' or 'chat_ids' is required: {}",
                err
            ));
        }
    };

    let poll_timeout_seconds = match settings.get("poll_timeout_seconds") {
        None => 30,
        Some(value) => {
            let timeout = value.as_u64().ok_or_else(|| {
                anyhow!(
                    "[telegram_config_invalid_poll_timeout] Telegram channel setting 'poll_timeout_seconds' must be a non-negative integer"
                )
            })?;
            if timeout > 50 {
                anyhow::bail!(
                    "[telegram_config_invalid_poll_timeout] Telegram channel setting 'poll_timeout_seconds' must be <= 50"
                );
            }
            timeout
        }
    };

    let poll_interval_ms = match settings.get("poll_interval_ms") {
        None => 250,
        Some(value) => {
            let interval = value.as_u64().ok_or_else(|| {
                anyhow!(
                    "[telegram_config_invalid_poll_interval] Telegram channel setting 'poll_interval_ms' must be a positive integer"
                )
            })?;
            if interval < 25 {
                anyhow::bail!(
                    "[telegram_config_invalid_poll_interval] Telegram channel setting 'poll_interval_ms' must be >= 25"
                );
            }
            interval
        }
    };

    let max_updates_per_poll = match settings.get("max_updates_per_poll") {
        None => 25,
        Some(value) => {
            let max = value.as_u64().ok_or_else(|| {
                anyhow!(
                    "[telegram_config_invalid_max_updates] Telegram channel setting 'max_updates_per_poll' must be a positive integer"
                )
            })?;
            if !(1..=100).contains(&max) {
                anyhow::bail!(
                    "[telegram_config_invalid_max_updates] Telegram channel setting 'max_updates_per_poll' must be in 1..=100"
                );
            }
            max as u8
        }
    };

    let max_inbound_text_chars = match settings.get("max_inbound_text_chars") {
        None => DEFAULT_MAX_INBOUND_TEXT_CHARS,
        Some(value) => {
            let max = value.as_u64().ok_or_else(|| {
                anyhow!(
                    "[telegram_config_invalid_max_inbound_text_chars] Telegram channel setting 'max_inbound_text_chars' must be a positive integer"
                )
            })?;
            let max = usize::try_from(max).map_err(|_| {
                anyhow!(
                    "[telegram_config_invalid_max_inbound_text_chars] Telegram channel setting 'max_inbound_text_chars' is too large"
                )
            })?;
            if max == 0 {
                anyhow::bail!(
                    "[telegram_config_invalid_max_inbound_text_chars] Telegram channel setting 'max_inbound_text_chars' must be > 0"
                );
            }
            max
        }
    };

    Ok(TelegramChannelSettings {
        token_env,
        base_url: settings
            .get("base_url")
            .map(|value| {
                value.as_str().ok_or_else(|| {
                    anyhow!(
                        "[telegram_config_invalid_base_url] Telegram channel setting 'base_url' must be a string"
                    )
                })
            })
            .transpose()?
            .unwrap_or(DEFAULT_BASE_URL)
            .trim_end_matches('/')
            .to_string(),
        workspace_id: settings
            .get("workspace_id")
            .map(|value| {
                let text = value.as_str().ok_or_else(|| {
                    anyhow!(
                        "[telegram_config_invalid_workspace_id] Telegram channel setting 'workspace_id' must be a string"
                    )
                })?;
                if text.trim().is_empty() {
                    anyhow::bail!(
                        "[telegram_config_invalid_workspace_id] Telegram channel setting 'workspace_id' must not be empty"
                    );
                }
                Ok::<String, anyhow::Error>(text.to_string())
            })
            .transpose()?
            .unwrap_or_else(|| "telegram".to_string()),
        chat_ids,
        poll_timeout_seconds,
        poll_interval_ms,
        max_updates_per_poll,
        max_inbound_text_chars,
        start_from_latest: settings
            .get("start_from_latest")
            .map(|value| {
                value.as_bool().ok_or_else(|| {
                    anyhow!(
                        "[telegram_config_invalid_start_from_latest] Telegram channel setting 'start_from_latest' must be a boolean"
                    )
                })
            })
            .transpose()?
            .unwrap_or(true),
        ignore_bot_messages: settings
            .get("ignore_bot_messages")
            .map(|value| {
                value.as_bool().ok_or_else(|| {
                    anyhow!(
                        "[telegram_config_invalid_ignore_bot_messages] Telegram channel setting 'ignore_bot_messages' must be a boolean"
                    )
                })
            })
            .transpose()?
            .unwrap_or(true),
        respond_mode: read_respond_mode(settings.get("respond_mode"))?,
        session_scope: read_telegram_session_scope(settings.get("session_scope"))?,
        session_scope_dm: read_optional_telegram_session_scope(
            settings.get("session_scope_dm"),
            "session_scope_dm",
        )?,
        session_scope_group: read_optional_telegram_session_scope(
            settings.get("session_scope_group"),
            "session_scope_group",
        )?,
        session_scope_channel: read_optional_telegram_session_scope(
            settings.get("session_scope_channel"),
            "session_scope_channel",
        )?,
        stream_mode: read_stream_mode(settings.get("stream_mode"))?,
        stream_thinking: settings
            .get("stream_thinking")
            .map(|value| {
                value.as_bool().ok_or_else(|| {
                    anyhow!(
                        "[telegram_config_invalid_stream_thinking] Telegram channel setting 'stream_thinking' must be a boolean"
                    )
                })
            })
            .transpose()?
            .unwrap_or(false),
        persist_thinking: settings
            .get("persist_thinking")
            .map(|value| {
                value.as_bool().ok_or_else(|| {
                    anyhow!(
                        "[telegram_config_invalid_persist_thinking] Telegram channel setting 'persist_thinking' must be a boolean"
                    )
                })
            })
            .transpose()?
            .unwrap_or(false),
        accept_all_chats: allow_unconfigured_chats,
    })
}

pub struct TelegramChannelDriver {
    channel_runtime_id: String,
    config: TelegramChannelDriverConfig,
    media_dir: PathBuf,
    client: reqwest::Client,
    shutdown_rx: watch::Receiver<bool>,
    backlog: VecDeque<InboundEvent>,
    next_update_offset: Option<i64>,
    initialized: bool,
    consecutive_poll_failures: u32,
    progress_states: HashMap<String, TelegramProgressState>,
    last_chat_action_at: HashMap<String, Instant>,
    next_draft_id: i64,
    bot_identity: Option<TelegramBotIdentity>,
}

impl TelegramChannelDriver {
    pub async fn from_settings(
        channel_runtime_id: impl Into<String>,
        settings: &serde_json::Value,
        shutdown_rx: watch::Receiver<bool>,
        allow_unconfigured_chats: bool,
    ) -> Result<Self> {
        Self::from_settings_with_media_dir(
            channel_runtime_id,
            settings,
            None,
            shutdown_rx,
            allow_unconfigured_chats,
        )
        .await
    }

    pub async fn from_settings_with_media_dir(
        channel_runtime_id: impl Into<String>,
        settings: &serde_json::Value,
        media_dir: Option<PathBuf>,
        shutdown_rx: watch::Receiver<bool>,
        allow_unconfigured_chats: bool,
    ) -> Result<Self> {
        let config =
            TelegramChannelDriverConfig::from_settings(settings, allow_unconfigured_chats)?;
        Self::from_config_with_media_dir(channel_runtime_id, config, media_dir, shutdown_rx)
    }

    pub fn from_config(
        channel_runtime_id: impl Into<String>,
        config: TelegramChannelDriverConfig,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Result<Self> {
        Self::from_config_with_media_dir(channel_runtime_id, config, None, shutdown_rx)
    }

    pub fn from_config_with_media_dir(
        channel_runtime_id: impl Into<String>,
        config: TelegramChannelDriverConfig,
        media_dir: Option<PathBuf>,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Result<Self> {
        let channel_runtime_id = channel_runtime_id.into();
        let timeout = Duration::from_secs(config.poll_timeout_seconds.saturating_add(10).max(10));
        let client = reqwest::Client::builder()
            .user_agent("turin-channel-telegram/0.24.0")
            .timeout(timeout)
            .build()
            .context(
                "[telegram_http_client_init_failed] Failed to build Telegram adapter HTTP client",
            )?;
        let media_dir =
            media_dir.unwrap_or_else(|| default_media_dir_for_runtime(&channel_runtime_id));

        Ok(Self {
            channel_runtime_id,
            config,
            media_dir,
            client,
            shutdown_rx,
            backlog: VecDeque::new(),
            next_update_offset: None,
            initialized: false,
            consecutive_poll_failures: 0,
            progress_states: HashMap::new(),
            last_chat_action_at: HashMap::new(),
            next_draft_id: 1,
            bot_identity: None,
        })
    }

    async fn skip_pending_updates(&mut self) -> std::result::Result<(), TelegramApiError> {
        for _ in 0..MAX_STARTUP_SKIP_BATCHES {
            let updates = self.fetch_updates(self.next_update_offset, 100, 0).await?;
            if updates.is_empty() {
                break;
            }
            self.advance_offset(&updates);
            if updates.len() < 100 {
                break;
            }
        }
        Ok(())
    }

    async fn poll_once(&mut self) -> std::result::Result<bool, TelegramApiError> {
        let updates = self
            .fetch_updates(
                self.next_update_offset,
                self.config.max_updates_per_poll,
                self.config.poll_timeout_seconds,
            )
            .await?;
        if updates.is_empty() {
            return Ok(false);
        }

        self.advance_offset(&updates);
        for update in updates {
            let update_id = update.update_id;
            let Some(message) = update.message.or(update.channel_post) else {
                continue;
            };
            if let Some(mut event) = self.normalize_message(update_id, message.clone()) {
                match self.collect_inbound_attachments(&message).await {
                    Ok(attachments) => {
                        event.attachments = attachments;
                    }
                    Err(error) => {
                        warn!(
                            channel_runtime_id = %self.channel_runtime_id,
                            update_id,
                            message_id = message.message_id,
                            error = %error,
                            "Telegram attachment collection failed; continuing without attachments"
                        );
                    }
                }
                if event.text.trim().is_empty() && event.attachments.is_empty() {
                    continue;
                }
                self.backlog.push_back(event);
            }
        }

        Ok(!self.backlog.is_empty())
    }

    async fn fetch_updates(
        &self,
        offset: Option<i64>,
        limit: u8,
        timeout_seconds: u64,
    ) -> std::result::Result<Vec<TelegramUpdate>, TelegramApiError> {
        let payload = serde_json::json!({
            "offset": offset,
            "limit": limit,
            "timeout": timeout_seconds,
            "allowed_updates": ["message", "channel_post"]
        });
        self.request_with_retry("getUpdates", &payload).await
    }

    async fn collect_inbound_attachments(
        &self,
        message: &TelegramMessage,
    ) -> Result<Vec<ChannelAttachment>> {
        let refs = message.attachment_refs();
        if refs.is_empty() {
            return Ok(Vec::new());
        }

        tokio::fs::create_dir_all(&self.media_dir)
            .await
            .with_context(|| {
                format!(
                    "Failed to create Telegram media directory '{}'",
                    self.media_dir.display()
                )
            })?;

        let mut attachments = Vec::with_capacity(refs.len());
        for attachment in refs {
            attachments.push(self.download_inbound_attachment(&attachment).await?);
        }
        Ok(attachments)
    }

    async fn download_inbound_attachment(
        &self,
        attachment: &TelegramAttachmentRef,
    ) -> Result<ChannelAttachment> {
        let file: TelegramFile = self
            .request_with_retry(
                "getFile",
                &serde_json::json!({ "file_id": attachment.file_id }),
            )
            .await
            .map_err(TelegramApiError::into_anyhow)?;
        let file_path = file.file_path.context(format!(
            "Telegram getFile response missing file_path for '{}'",
            attachment.file_id
        ))?;
        let download_url = self.telegram_file_url(&file_path);
        let response = self
            .client
            .get(&download_url)
            .send()
            .await
            .with_context(|| format!("Telegram file download failed for '{}'", attachment.name))?
            .error_for_status()
            .with_context(|| {
                format!(
                    "Telegram file download returned error status for '{}'",
                    attachment.name
                )
            })?;
        let fetched_content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(|value| value.split(';').next().unwrap_or(value).trim().to_string());
        let bytes = response
            .bytes()
            .await
            .with_context(|| format!("Failed to read Telegram file '{}'", attachment.name))?;
        let target_path = self.media_dir.join(unique_media_name(
            &attachment.name,
            Some(file_path.as_str()),
        ));
        tokio::fs::write(&target_path, bytes)
            .await
            .with_context(|| {
                format!(
                    "Failed to write Telegram media attachment '{}'",
                    target_path.display()
                )
            })?;
        Ok(ChannelAttachment {
            name: attachment.name.clone(),
            content_type: attachment
                .content_type
                .clone()
                .or(fetched_content_type)
                .or_else(|| match attachment.kind {
                    TelegramAttachmentKind::Image => Some("image/jpeg".to_string()),
                    TelegramAttachmentKind::File => None,
                }),
            url: None,
            local_path: Some(target_path.display().to_string()),
        })
    }

    fn telegram_file_url(&self, file_path: &str) -> String {
        format!(
            "{}/file/bot{}/{}",
            self.config.base_url,
            self.config.token,
            file_path.trim_start_matches('/')
        )
    }

    async fn send_batches(
        &self,
        conversation: &ChannelConversationKey,
        message: &OutboundMessage,
    ) -> Result<()> {
        let chat_id = conversation_chat_id(self.config.primary_chat_id(), conversation);
        let message_thread_id = resolve_message_thread_id(conversation)?;
        let reply_to_message_id = metadata_i64(&message.metadata, "telegram_reply_to_message_id")?;
        let payloads = telegram_batches_from_message(&chat_id, message_thread_id, message)?;
        let reply_for_attachments = if payloads.is_empty() {
            reply_to_message_id
        } else {
            None
        };
        for payload in payloads {
            let _: TelegramSentMessage = self
                .request_with_retry("sendMessage", &payload)
                .await
                .map_err(TelegramApiError::into_anyhow)?;
        }
        self.send_attachment_messages(
            &chat_id,
            message_thread_id,
            &message.attachments,
            reply_for_attachments,
        )
        .await?;
        Ok(())
    }

    async fn send_attachment_messages(
        &self,
        chat_id: &str,
        message_thread_id: Option<i64>,
        attachments: &[ChannelAttachment],
        mut reply_to_message_id: Option<i64>,
    ) -> Result<()> {
        for attachment in attachments {
            self.send_attachment_message(
                chat_id,
                message_thread_id,
                attachment,
                reply_to_message_id.take(),
            )
            .await?;
        }
        Ok(())
    }

    async fn send_attachment_message(
        &self,
        chat_id: &str,
        message_thread_id: Option<i64>,
        attachment: &ChannelAttachment,
        reply_to_message_id: Option<i64>,
    ) -> Result<()> {
        let method = if attachment
            .content_type
            .as_deref()
            .is_some_and(|content_type| content_type.starts_with("image/"))
        {
            "sendPhoto"
        } else {
            "sendDocument"
        };
        let field_name = if method == "sendPhoto" {
            "photo"
        } else {
            "document"
        };

        if let Some(local_path) = attachment.local_path.as_deref() {
            let attachment_name = attachment.name.clone();
            let content_type = attachment.content_type.clone();
            let local_path = PathBuf::from(local_path);
            let chat_id = chat_id.to_string();
            let _: TelegramSentMessage = self
                .multipart_request_with_retry(method, || {
                    let bytes = std::fs::read(&local_path).with_context(|| {
                        format!(
                            "Failed to read Telegram attachment '{}'",
                            local_path.display()
                        )
                    })?;
                    let mut form = reqwest::multipart::Form::new().text("chat_id", chat_id.clone());
                    if let Some(message_thread_id) = message_thread_id {
                        form = form.text("message_thread_id", message_thread_id.to_string());
                    }
                    if let Some(reply_to_message_id) = reply_to_message_id {
                        form = form.text("reply_to_message_id", reply_to_message_id.to_string());
                    }

                    let mut part =
                        reqwest::multipart::Part::bytes(bytes).file_name(attachment_name.clone());
                    if let Some(content_type) = &content_type {
                        part = part.mime_str(content_type).with_context(|| {
                            format!(
                                "Invalid Telegram attachment content type '{}'",
                                content_type
                            )
                        })?;
                    }
                    Ok(form.part(field_name.to_string(), part))
                })
                .await
                .map_err(TelegramApiError::into_anyhow)?;
            return Ok(());
        }

        let remote = attachment.url.as_deref().ok_or_else(|| {
            anyhow!(
                "[telegram_send_missing_attachment_source] attachment '{}' is missing local_path and url",
                attachment.name
            )
        })?;
        let mut payload = serde_json::Map::new();
        payload.insert("chat_id".to_string(), serde_json::json!(chat_id));
        payload.insert(field_name.to_string(), serde_json::json!(remote));
        if let Some(message_thread_id) = message_thread_id {
            payload.insert(
                "message_thread_id".to_string(),
                serde_json::json!(message_thread_id),
            );
        }
        if let Some(reply_to_message_id) = reply_to_message_id {
            payload.insert(
                "reply_to_message_id".to_string(),
                serde_json::json!(reply_to_message_id),
            );
        }
        let _: TelegramSentMessage = self
            .request_with_retry(method, &serde_json::Value::Object(payload))
            .await
            .map_err(TelegramApiError::into_anyhow)?;
        Ok(())
    }

    async fn send_chat_action(&mut self, event: &InboundEvent) -> Result<()> {
        let key = progress_key(&event.conversation)?;
        let now = Instant::now();
        if self
            .last_chat_action_at
            .get(&key)
            .is_some_and(|previous| now.duration_since(*previous) < Duration::from_secs(4))
        {
            return Ok(());
        }

        let chat_id = conversation_chat_id(self.config.primary_chat_id(), &event.conversation);
        let message_thread_id = resolve_message_thread_id(&event.conversation)?;
        let mut payload = serde_json::Map::new();
        payload.insert("chat_id".to_string(), serde_json::json!(chat_id));
        payload.insert("action".to_string(), serde_json::json!("typing"));
        if let Some(message_thread_id) = message_thread_id {
            payload.insert(
                "message_thread_id".to_string(),
                serde_json::json!(message_thread_id),
            );
        }

        let _: bool = self
            .request_with_retry("sendChatAction", &serde_json::Value::Object(payload))
            .await
            .map_err(TelegramApiError::into_anyhow)?;
        self.last_chat_action_at.insert(key, now);
        Ok(())
    }

    async fn send_stream_preview(
        &mut self,
        event: &InboundEvent,
        text: &str,
        thinking: Option<&str>,
    ) -> Result<()> {
        let preview = render_stream_preview(text, thinking);
        if preview.is_empty() {
            return Ok(());
        }

        let key = progress_key(&event.conversation)?;
        let chat_id = conversation_chat_id(self.config.primary_chat_id(), &event.conversation);
        let message_thread_id = resolve_message_thread_id(&event.conversation)?;
        let reply_to_message_id = event
            .metadata
            .get("telegram_message_id")
            .and_then(|value| value.as_i64())
            .or_else(|| {
                event
                    .metadata
                    .get("telegram_message_id")
                    .and_then(|value| value.as_str())
                    .and_then(|value| value.parse::<i64>().ok())
            });

        let existing_state = self.progress_states.get(&key).cloned();
        let next_state = match existing_state {
            Some(TelegramProgressState {
                sink: TelegramProgressSink::Draft { draft_id },
            }) => {
                self.send_message_draft(&chat_id, message_thread_id, draft_id, &preview)
                    .await?;
                Some(TelegramProgressState {
                    sink: TelegramProgressSink::Draft { draft_id },
                })
            }
            Some(TelegramProgressState {
                sink: TelegramProgressSink::Placeholder { message_id },
            }) => {
                self.edit_stream_placeholder(&chat_id, message_id, &preview)
                    .await?;
                Some(TelegramProgressState {
                    sink: TelegramProgressSink::Placeholder { message_id },
                })
            }
            None => {
                self.start_progress_sink(&chat_id, message_thread_id, reply_to_message_id, &preview)
                    .await?
            }
        };

        if let Some(state) = next_state {
            self.progress_states.insert(key, state);
        } else {
            self.progress_states.remove(&key);
        }
        Ok(())
    }

    async fn start_progress_sink(
        &mut self,
        chat_id: &str,
        message_thread_id: Option<i64>,
        reply_to_message_id: Option<i64>,
        preview: &str,
    ) -> Result<Option<TelegramProgressState>> {
        if self.config.stream_mode == ChannelStreamMode::Draft && chat_id_is_private(chat_id) {
            let draft_id = self.allocate_draft_id();
            match self
                .send_message_draft(chat_id, message_thread_id, draft_id, preview)
                .await
            {
                Ok(()) => {
                    return Ok(Some(TelegramProgressState {
                        sink: TelegramProgressSink::Draft { draft_id },
                    }));
                }
                Err(err) => {
                    warn!(error = %err, "Telegram draft streaming failed; falling back to placeholder edits");
                }
            }
        }

        let payload = telegram_payload(
            chat_id,
            message_thread_id,
            preview.to_string(),
            None,
            reply_to_message_id,
            true,
            false,
        );
        let sent: TelegramSentMessage = self
            .request_with_retry("sendMessage", &payload)
            .await
            .map_err(TelegramApiError::into_anyhow)?;
        Ok(Some(TelegramProgressState {
            sink: TelegramProgressSink::Placeholder {
                message_id: sent.message_id,
            },
        }))
    }

    async fn send_message_draft(
        &self,
        chat_id: &str,
        message_thread_id: Option<i64>,
        draft_id: i64,
        preview: &str,
    ) -> Result<()> {
        let mut payload = serde_json::Map::new();
        payload.insert("chat_id".to_string(), serde_json::json!(chat_id));
        payload.insert("draft_id".to_string(), serde_json::json!(draft_id));
        payload.insert("text".to_string(), serde_json::json!(preview));
        if let Some(message_thread_id) = message_thread_id {
            payload.insert(
                "message_thread_id".to_string(),
                serde_json::json!(message_thread_id),
            );
        }
        let _: bool = self
            .request_with_retry("sendMessageDraft", &serde_json::Value::Object(payload))
            .await
            .map_err(TelegramApiError::into_anyhow)?;
        Ok(())
    }

    async fn edit_stream_placeholder(
        &self,
        chat_id: &str,
        message_id: i64,
        preview: &str,
    ) -> Result<()> {
        let payload = telegram_edit_payload(chat_id, message_id, preview.to_string(), None, true);
        let _: TelegramSentMessage = self
            .request_with_retry("editMessageText", &payload)
            .await
            .map_err(TelegramApiError::into_anyhow)?;
        Ok(())
    }

    async fn send_final_message(
        &mut self,
        conversation: &ChannelConversationKey,
        message: &OutboundMessage,
    ) -> Result<()> {
        let key = progress_key(conversation)?;
        let progress_state = self.progress_states.remove(&key);
        let attachment_placeholder_id = match progress_state.as_ref() {
            Some(TelegramProgressState {
                sink: TelegramProgressSink::Placeholder { message_id },
            }) => Some(*message_id),
            _ => None,
        };
        let chat_id = conversation_chat_id(self.config.primary_chat_id(), conversation);
        let message_thread_id = resolve_message_thread_id(conversation)?;
        let payloads = telegram_batches_from_message(&chat_id, message_thread_id, message)?;
        let reply_to_message_id = metadata_i64(&message.metadata, "telegram_reply_to_message_id")?;

        if let Some(TelegramProgressState {
            sink: TelegramProgressSink::Placeholder { message_id },
        }) = progress_state
            && let Some((first, rest)) = payloads.split_first()
        {
            let payload = telegram_edit_payload(
                &chat_id,
                message_id,
                first["text"].as_str().unwrap_or_default().to_string(),
                first["parse_mode"].as_str(),
                first["disable_web_page_preview"].as_bool().unwrap_or(true),
            );
            match self
                .request_with_retry::<TelegramSentMessage>("editMessageText", &payload)
                .await
            {
                Ok(_) => {
                    for payload in rest {
                        let _: TelegramSentMessage = self
                            .request_with_retry("sendMessage", payload)
                            .await
                            .map_err(TelegramApiError::into_anyhow)?;
                    }
                    self.send_attachment_messages(
                        &chat_id,
                        message_thread_id,
                        &message.attachments,
                        None,
                    )
                    .await?;
                    return Ok(());
                }
                Err(error) if error.is_message_not_modified() => {
                    for payload in rest {
                        let _: TelegramSentMessage = self
                            .request_with_retry("sendMessage", payload)
                            .await
                            .map_err(TelegramApiError::into_anyhow)?;
                    }
                    self.send_attachment_messages(
                        &chat_id,
                        message_thread_id,
                        &message.attachments,
                        None,
                    )
                    .await?;
                    return Ok(());
                }
                Err(error) => {
                    warn!(
                        error_code = %error.code,
                        error = %error.message,
                        "Telegram placeholder finalization failed; sending final message normally"
                    );
                }
            }
        }

        if payloads.is_empty() && !message.attachments.is_empty() {
            if let Some(message_id) = attachment_placeholder_id {
                let summary = attachment_preview_text(&message.attachments);
                let payload = telegram_edit_payload(&chat_id, message_id, summary, None, true);
                let _ = self
                    .request_with_retry::<TelegramSentMessage>("editMessageText", &payload)
                    .await;
            }
            self.send_attachment_messages(
                &chat_id,
                message_thread_id,
                &message.attachments,
                reply_to_message_id,
            )
            .await?;
            return Ok(());
        }

        self.send_batches(conversation, message).await
    }

    fn allocate_draft_id(&mut self) -> i64 {
        let draft_id = self.next_draft_id.max(1);
        self.next_draft_id = self.next_draft_id.saturating_add(1).max(1);
        draft_id
    }

    fn advance_offset(&mut self, updates: &[TelegramUpdate]) {
        if let Some(next) = updates.iter().map(|update| update.update_id).max() {
            self.next_update_offset = Some(next.saturating_add(1));
        }
    }

    #[cfg(test)]
    fn normalize_update(&self, update: TelegramUpdate) -> Option<InboundEvent> {
        let update_id = update.update_id;
        let message = update.message.or(update.channel_post)?;
        self.normalize_message(update_id, message)
    }

    fn normalize_message(&self, update_id: i64, message: TelegramMessage) -> Option<InboundEvent> {
        let chat_id = message.chat.id.to_string();
        if !self.config.accept_all_chats && !self.config.allows_chat_id(&chat_id) {
            return None;
        }

        if self.config.ignore_bot_messages
            && message.from.as_ref().and_then(|user| user.is_bot) == Some(true)
        {
            return None;
        }

        if !self.should_accept_message(&message) {
            return None;
        }

        let text = message
            .body_text()
            .map(|value| value.trim().to_string())
            .unwrap_or_default();

        let user = message.channel_user()?;
        let scoped_thread_id = message
            .message_thread_id
            .map(|value| value.to_string())
            .unwrap_or_else(|| chat_id.clone());

        let mut metadata = serde_json::Map::new();
        metadata.insert(
            "telegram_update_id".to_string(),
            serde_json::json!(update_id),
        );
        metadata.insert(
            "telegram_message_id".to_string(),
            serde_json::json!(message.message_id),
        );
        metadata.insert(
            "telegram_chat_id".to_string(),
            serde_json::json!(message.chat.id),
        );
        if let Some(message_thread_id) = message.message_thread_id {
            metadata.insert(
                "telegram_message_thread_id".to_string(),
                serde_json::json!(message_thread_id),
            );
        }
        metadata.insert(
            "telegram_chat_type".to_string(),
            serde_json::json!(message.chat.chat_type),
        );
        let text = bound_inbound_text(text, &mut metadata, self.config.max_inbound_text_chars);

        let session_scope = effective_telegram_session_scope(&self.config, &message.chat);
        let conversation = ChannelConversationKey {
            channel: ChannelKind::new("telegram"),
            workspace_id: self.config.workspace_id.clone(),
            room_id: Some(chat_id.clone()),
            thread_id: match session_scope {
                ChannelSessionScope::User | ChannelSessionScope::Thread => scoped_thread_id,
                ChannelSessionScope::Room => chat_id,
            },
            user_id: match session_scope {
                ChannelSessionScope::User => Some(user.id.clone()),
                ChannelSessionScope::Thread | ChannelSessionScope::Room => None,
            },
        };

        Some(InboundEvent {
            message: ChannelMessageRef {
                conversation: conversation.clone(),
                message_id: message.message_id.to_string(),
            },
            conversation,
            user,
            session_scope,
            text,
            attachments: Vec::new(),
            metadata,
        })
    }

    fn should_accept_message(&self, message: &TelegramMessage) -> bool {
        if message.chat.is_private() {
            return true;
        }

        match self.config.respond_mode {
            TelegramRespondMode::All => true,
            TelegramRespondMode::Mentions => {
                self.message_mentions_bot(message) || self.message_targets_bot_command(message)
            }
            TelegramRespondMode::Replies => self.message_replies_to_bot(message),
            TelegramRespondMode::MentionsOrReplies => {
                self.message_mentions_bot(message)
                    || self.message_targets_bot_command(message)
                    || self.message_replies_to_bot(message)
            }
        }
    }

    fn message_mentions_bot(&self, message: &TelegramMessage) -> bool {
        let Some(identity) = self.bot_identity.as_ref() else {
            return false;
        };
        let Some(username) = identity.username.as_deref() else {
            return false;
        };
        let Some(body) = message.body_text() else {
            return false;
        };
        let mention = format!("@{}", username);

        for entity in message.body_entities() {
            match entity.kind.as_str() {
                "mention" => {
                    let Some(slice) = utf16_slice(body, entity.offset, entity.length) else {
                        continue;
                    };
                    if slice.eq_ignore_ascii_case(&mention) {
                        return true;
                    }
                }
                "text_mention" if entity.user.as_ref().map(|user| user.id) == Some(identity.id) => {
                    return true;
                }
                _ => {}
            }
        }

        false
    }

    fn message_targets_bot_command(&self, message: &TelegramMessage) -> bool {
        let Some(identity) = self.bot_identity.as_ref() else {
            return false;
        };
        let Some(username) = identity.username.as_deref() else {
            return false;
        };
        let Some(body) = message.body_text() else {
            return false;
        };

        for entity in message.body_entities() {
            if entity.kind != "bot_command" {
                continue;
            }
            let Some(slice) = utf16_slice(body, entity.offset, entity.length) else {
                continue;
            };
            let Some((_, target)) = slice.split_once('@') else {
                continue;
            };
            if target.eq_ignore_ascii_case(username) {
                return true;
            }
        }

        false
    }

    fn message_replies_to_bot(&self, message: &TelegramMessage) -> bool {
        let Some(replied) = message.reply_to_message.as_deref() else {
            return false;
        };
        let Some(identity) = self.bot_identity.as_ref() else {
            return false;
        };

        if replied.from.as_ref().map(|user| user.id) == Some(identity.id) {
            return true;
        }

        replied
            .from
            .as_ref()
            .and_then(|user| user.username.as_deref())
            .zip(identity.username.as_deref())
            .is_some_and(|(reply_username, bot_username)| {
                reply_username.eq_ignore_ascii_case(bot_username)
            })
    }

    async fn api_request_once<T: DeserializeOwned>(
        &self,
        method: &str,
        payload: &serde_json::Value,
    ) -> std::result::Result<T, TelegramApiError> {
        let url = format!(
            "{}/bot{}/{}",
            self.config.base_url, self.config.token, method
        );
        let response = self
            .client
            .post(&url)
            .json(payload)
            .send()
            .await
            .map_err(|error| TelegramApiError {
                code: "telegram_http_request_failed".to_string(),
                message: format!("Telegram {} request failed: {}", method, error),
                retriable: true,
                retry_after: None,
            })?;

        self.decode_api_response(method, response).await
    }

    async fn api_multipart_request_once<T: DeserializeOwned>(
        &self,
        method: &str,
        form: reqwest::multipart::Form,
    ) -> std::result::Result<T, TelegramApiError> {
        let url = format!(
            "{}/bot{}/{}",
            self.config.base_url, self.config.token, method
        );
        let response = self
            .client
            .post(&url)
            .multipart(form)
            .send()
            .await
            .map_err(|error| TelegramApiError {
                code: "telegram_http_request_failed".to_string(),
                message: format!("Telegram {} multipart request failed: {}", method, error),
                retriable: true,
                retry_after: None,
            })?;

        self.decode_api_response(method, response).await
    }

    async fn decode_api_response<T: DeserializeOwned>(
        &self,
        method: &str,
        response: reqwest::Response,
    ) -> std::result::Result<T, TelegramApiError> {
        let status = response.status();
        let body = response
            .text()
            .await
            .with_context(|| {
                format!(
                    "[telegram_http_decode_failed] Failed to read Telegram {} response body",
                    method
                )
            })
            .map_err(|error| TelegramApiError {
                code: "telegram_http_decode_failed".to_string(),
                message: error.to_string(),
                retriable: true,
                retry_after: None,
            })?;

        let envelope: TelegramApiEnvelope<T> = serde_json::from_str(&body)
            .with_context(|| {
                format!(
                    "[telegram_http_decode_failed] Failed to decode Telegram {} response: {}",
                    method, body
                )
            })
            .map_err(|error| TelegramApiError {
                code: "telegram_http_decode_failed".to_string(),
                message: error.to_string(),
                retriable: false,
                retry_after: None,
            })?;

        if !status.is_success() || !envelope.ok {
            let description = envelope.description.clone().unwrap_or_else(|| body.clone());
            let error_code = envelope.error_code.unwrap_or(status.as_u16() as i64);
            let code = classify_api_error(method, status.as_u16(), &description);
            let retriable = is_retriable_api_error(&code, status.as_u16())
                && !is_not_modified_description(&description);
            return Err(TelegramApiError {
                retriable,
                retry_after: envelope
                    .parameters
                    .as_ref()
                    .and_then(|parameters| parameters.retry_after)
                    .map(Duration::from_secs),
                code,
                message: format!(
                    "Telegram {} request failed with {}: {}",
                    method, error_code, description
                ),
            });
        }

        envelope
            .result
            .context(format!("Telegram {} response missing result", method))
            .map_err(|error| TelegramApiError {
                code: "telegram_missing_result".to_string(),
                message: error.to_string(),
                retriable: false,
                retry_after: None,
            })
    }

    async fn request_with_retry<T: DeserializeOwned>(
        &self,
        method: &str,
        payload: &serde_json::Value,
    ) -> std::result::Result<T, TelegramApiError> {
        let mut attempts: u32 = 0;
        loop {
            attempts = attempts.saturating_add(1);
            match self.api_request_once(method, payload).await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    if !error.retriable || attempts >= MAX_API_REQUEST_ATTEMPTS {
                        return Err(error);
                    }

                    let delay = error.retry_after.unwrap_or_else(|| retry_backoff(attempts));
                    warn!(
                        channel_runtime_id = %self.channel_runtime_id,
                        method,
                        attempt = attempts,
                        delay_ms = delay.as_millis() as u64,
                        error_code = %error.code,
                        error = %error.message,
                        "Retrying Telegram request after transient failure"
                    );
                    sleep(delay).await;
                }
            }
        }
    }

    async fn multipart_request_with_retry<T, F>(
        &self,
        method: &str,
        form_builder: F,
    ) -> std::result::Result<T, TelegramApiError>
    where
        T: DeserializeOwned,
        F: Fn() -> Result<reqwest::multipart::Form>,
    {
        let mut attempts: u32 = 0;
        loop {
            attempts = attempts.saturating_add(1);
            let form = form_builder().map_err(|error| TelegramApiError {
                code: "telegram_multipart_build_failed".to_string(),
                message: error.to_string(),
                retriable: false,
                retry_after: None,
            })?;

            match self.api_multipart_request_once(method, form).await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    if !error.retriable || attempts >= MAX_API_REQUEST_ATTEMPTS {
                        return Err(error);
                    }

                    let delay = error.retry_after.unwrap_or_else(|| retry_backoff(attempts));
                    warn!(
                        channel_runtime_id = %self.channel_runtime_id,
                        method,
                        attempt = attempts,
                        delay_ms = delay.as_millis() as u64,
                        error_code = %error.code,
                        error = %error.message,
                        "Retrying Telegram multipart request after transient failure"
                    );
                    sleep(delay).await;
                }
            }
        }
    }

    async fn sleep_or_shutdown(&self, duration: Duration) -> bool {
        let mut shutdown_rx = self.shutdown_rx.clone();
        tokio::select! {
            changed = shutdown_rx.changed() => changed.is_ok() && *shutdown_rx.borrow(),
            _ = sleep(duration) => false,
        }
    }

    async fn handle_transient_poll_error(&mut self, phase: &str, error: TelegramApiError) -> bool {
        self.consecutive_poll_failures = self.consecutive_poll_failures.saturating_add(1);
        let delay = error
            .retry_after
            .unwrap_or_else(|| retry_backoff(self.consecutive_poll_failures));
        warn!(
            channel_runtime_id = %self.channel_runtime_id,
            phase,
            error_code = %error.code,
            error = %error.message,
            delay_ms = delay.as_millis() as u64,
            "Telegram polling hit a transient failure; backing off"
        );
        self.sleep_or_shutdown(delay).await
    }

    async fn ensure_bot_identity(&mut self) -> std::result::Result<(), TelegramApiError> {
        if self.bot_identity.is_some() || !self.config.respond_mode.requires_bot_identity() {
            return Ok(());
        }

        let bot: TelegramUser = self
            .request_with_retry("getMe", &serde_json::json!({}))
            .await?;
        self.bot_identity = Some(TelegramBotIdentity {
            id: bot.id,
            username: bot.username.map(|username| username.to_ascii_lowercase()),
        });
        Ok(())
    }
}

#[async_trait]
impl ChannelDriver for TelegramChannelDriver {
    fn kind(&self) -> ChannelKind {
        ChannelKind::new("telegram")
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
        loop {
            if let Some(event) = self.backlog.pop_front() {
                return Ok(Some(event));
            }
            if *self.shutdown_rx.borrow() {
                return Ok(None);
            }

            if !self.initialized {
                match self.ensure_bot_identity().await {
                    Ok(()) => {
                        self.consecutive_poll_failures = 0;
                    }
                    Err(error) if error.retriable => {
                        if self.handle_transient_poll_error("getMe", error).await {
                            return Ok(None);
                        }
                        continue;
                    }
                    Err(error) => return Err(error.into_anyhow()),
                }
                if self.config.start_from_latest {
                    match self.skip_pending_updates().await {
                        Ok(()) => {
                            self.consecutive_poll_failures = 0;
                        }
                        Err(error) if error.retriable => {
                            if self
                                .handle_transient_poll_error("startup skip", error)
                                .await
                            {
                                return Ok(None);
                            }
                            continue;
                        }
                        Err(error) => return Err(error.into_anyhow()),
                    }
                }
                self.initialized = true;
                continue;
            }

            let mut shutdown_rx = self.shutdown_rx.clone();
            let got_backlog = tokio::select! {
                changed = shutdown_rx.changed() => {
                    if changed.is_ok() && *shutdown_rx.borrow() {
                        return Ok(None);
                    }
                    Ok(false)
                }
                result = self.poll_once() => result,
            };

            let got_backlog = match got_backlog {
                Ok(got_backlog) => {
                    self.consecutive_poll_failures = 0;
                    got_backlog
                }
                Err(error) if error.retriable => {
                    if self.handle_transient_poll_error("poll", error).await {
                        return Ok(None);
                    }
                    continue;
                }
                Err(error) => return Err(error.into_anyhow()),
            };

            if let Some(event) = self.backlog.pop_front() {
                return Ok(Some(event));
            }

            if !got_backlog && self.sleep_or_shutdown(self.config.poll_interval).await {
                return Ok(None);
            }
        }
    }

    async fn send(
        &mut self,
        conversation: &ChannelConversationKey,
        message: OutboundMessage,
    ) -> Result<()> {
        self.send_final_message(conversation, &message).await
    }

    fn enrich_outbound_for_event(
        &self,
        event: &InboundEvent,
        mut outbound: OutboundMessage,
    ) -> OutboundMessage {
        if !outbound
            .metadata
            .contains_key("telegram_reply_to_message_id")
            && let Some(message_id) = event.metadata.get("telegram_message_id")
        {
            outbound.metadata.insert(
                "telegram_reply_to_message_id".to_string(),
                message_id.clone(),
            );
        }
        outbound
    }

    fn stream_mode(&self) -> ChannelStreamMode {
        self.config.stream_mode
    }

    fn stream_thinking(&self) -> bool {
        self.config.stream_mode.streams_text() && self.config.stream_thinking
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
            ChannelProgressUpdate::Typing => self.send_chat_action(event).await,
            ChannelProgressUpdate::StreamingPreview { text, thinking } => {
                self.send_stream_preview(event, &text, thinking.as_deref())
                    .await
            }
        }
    }

    async fn shutdown(&mut self) -> Result<()> {
        let _ = &self.channel_runtime_id;
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
struct TelegramApiEnvelope<T> {
    ok: bool,
    result: Option<T>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    error_code: Option<i64>,
    #[serde(default)]
    parameters: Option<TelegramApiParameters>,
}

#[derive(Debug, Clone, Deserialize)]
struct TelegramApiParameters {
    #[serde(default)]
    retry_after: Option<u64>,
}

#[derive(Debug, Clone)]
struct TelegramApiError {
    code: String,
    message: String,
    retriable: bool,
    retry_after: Option<Duration>,
}

impl TelegramApiError {
    fn into_anyhow(self) -> anyhow::Error {
        anyhow!("[{}] {}", self.code, self.message)
    }

    fn is_message_not_modified(&self) -> bool {
        self.code == "telegram_edit_message_failed" && is_not_modified_description(&self.message)
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TelegramUpdate {
    update_id: i64,
    #[serde(default)]
    message: Option<TelegramMessage>,
    #[serde(default)]
    channel_post: Option<TelegramMessage>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TelegramMessage {
    message_id: i64,
    chat: TelegramChat,
    #[serde(default)]
    from: Option<TelegramUser>,
    #[serde(default)]
    sender_chat: Option<TelegramChat>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    caption: Option<String>,
    #[serde(default)]
    entities: Vec<TelegramMessageEntity>,
    #[serde(default)]
    caption_entities: Vec<TelegramMessageEntity>,
    #[serde(default)]
    photo: Vec<TelegramPhotoSize>,
    #[serde(default)]
    document: Option<TelegramDocument>,
    #[serde(default)]
    video: Option<TelegramVideo>,
    #[serde(default)]
    audio: Option<TelegramAudio>,
    #[serde(default)]
    voice: Option<TelegramVoice>,
    #[serde(default)]
    message_thread_id: Option<i64>,
    #[serde(default)]
    reply_to_message: Option<Box<TelegramMessage>>,
}

impl TelegramMessage {
    fn channel_user(&self) -> Option<ChannelUser> {
        if let Some(user) = &self.from {
            let display_name = match (&user.first_name, &user.last_name) {
                (Some(first), Some(last)) if !last.trim().is_empty() => {
                    Some(format!("{} {}", first, last))
                }
                (Some(first), _) => Some(first.clone()),
                _ => user.username.clone(),
            };
            return Some(ChannelUser {
                id: user.id.to_string(),
                display_name,
                username: user.username.clone(),
            });
        }

        self.sender_chat.as_ref().map(|chat| ChannelUser {
            id: chat.id.to_string(),
            display_name: chat
                .title
                .clone()
                .or_else(|| chat.first_name.clone())
                .or_else(|| chat.username.clone()),
            username: chat.username.clone(),
        })
    }

    fn body_text(&self) -> Option<&String> {
        self.text.as_ref().or(self.caption.as_ref())
    }

    fn body_entities(&self) -> &[TelegramMessageEntity] {
        if self.text.is_some() {
            &self.entities
        } else {
            &self.caption_entities
        }
    }

    fn attachment_refs(&self) -> Vec<TelegramAttachmentRef> {
        let mut attachments = Vec::new();
        if let Some(photo) = self.photo.iter().max_by_key(|photo| {
            (
                u64::from(photo.width) * u64::from(photo.height),
                photo.file_size.unwrap_or_default(),
            )
        }) {
            attachments.push(TelegramAttachmentRef {
                file_id: photo.file_id.clone(),
                name: photo
                    .file_unique_id
                    .as_deref()
                    .map(|id| format!("{id}.jpg"))
                    .unwrap_or_else(|| format!("photo-{}.jpg", self.message_id)),
                content_type: Some("image/jpeg".to_string()),
                kind: TelegramAttachmentKind::Image,
            });
        }
        if let Some(document) = &self.document {
            attachments.push(TelegramAttachmentRef {
                file_id: document.file_id.clone(),
                name: document
                    .file_name
                    .clone()
                    .unwrap_or_else(|| format!("document-{}", self.message_id)),
                content_type: document.mime_type.clone(),
                kind: attachment_kind_from_content_type(document.mime_type.as_deref()),
            });
        }
        if let Some(video) = &self.video {
            attachments.push(TelegramAttachmentRef {
                file_id: video.file_id.clone(),
                name: video
                    .file_name
                    .clone()
                    .unwrap_or_else(|| format!("video-{}.mp4", self.message_id)),
                content_type: video
                    .mime_type
                    .clone()
                    .or_else(|| Some("video/mp4".to_string())),
                kind: TelegramAttachmentKind::File,
            });
        }
        if let Some(audio) = &self.audio {
            attachments.push(TelegramAttachmentRef {
                file_id: audio.file_id.clone(),
                name: audio.file_name.clone().unwrap_or_else(|| {
                    infer_audio_name(audio).unwrap_or_else(|| format!("audio-{}", self.message_id))
                }),
                content_type: audio.mime_type.clone(),
                kind: TelegramAttachmentKind::File,
            });
        }
        if let Some(voice) = &self.voice {
            attachments.push(TelegramAttachmentRef {
                file_id: voice.file_id.clone(),
                name: format!("voice-{}.ogg", self.message_id),
                content_type: voice
                    .mime_type
                    .clone()
                    .or_else(|| Some("audio/ogg".to_string())),
                kind: TelegramAttachmentKind::File,
            });
        }
        attachments
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TelegramUser {
    id: i64,
    #[serde(default)]
    is_bot: Option<bool>,
    #[serde(default)]
    first_name: Option<String>,
    #[serde(default)]
    last_name: Option<String>,
    #[serde(default)]
    username: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TelegramPhotoSize {
    file_id: String,
    #[serde(default)]
    file_unique_id: Option<String>,
    width: u32,
    height: u32,
    #[serde(default)]
    file_size: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TelegramDocument {
    file_id: String,
    #[serde(default)]
    file_name: Option<String>,
    #[serde(default)]
    mime_type: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TelegramVideo {
    file_id: String,
    #[serde(default)]
    file_name: Option<String>,
    #[serde(default)]
    mime_type: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TelegramAudio {
    file_id: String,
    #[serde(default)]
    file_name: Option<String>,
    #[serde(default)]
    mime_type: Option<String>,
    #[serde(default)]
    performer: Option<String>,
    #[serde(default)]
    title: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TelegramVoice {
    file_id: String,
    #[serde(default)]
    mime_type: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TelegramFile {
    #[serde(default)]
    file_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TelegramAttachmentKind {
    Image,
    File,
}

#[derive(Debug, Clone)]
struct TelegramAttachmentRef {
    file_id: String,
    name: String,
    content_type: Option<String>,
    kind: TelegramAttachmentKind,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TelegramMessageEntity {
    #[serde(rename = "type")]
    kind: String,
    offset: usize,
    length: usize,
    #[serde(default)]
    user: Option<TelegramUser>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TelegramChat {
    id: i64,
    #[serde(rename = "type")]
    chat_type: String,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    username: Option<String>,
    #[serde(default)]
    first_name: Option<String>,
}

impl TelegramChat {
    fn is_private(&self) -> bool {
        self.chat_type == "private"
    }
}

#[derive(Debug, Clone, Deserialize)]
struct TelegramSentMessage {
    message_id: i64,
}

#[derive(Debug, Clone)]
struct TelegramBotIdentity {
    id: i64,
    username: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelegramRespondMode {
    All,
    Mentions,
    Replies,
    MentionsOrReplies,
}

impl TelegramRespondMode {
    fn requires_bot_identity(self) -> bool {
        !matches!(self, Self::All)
    }
}

#[derive(Debug, Clone)]
struct TelegramProgressState {
    sink: TelegramProgressSink,
}

#[derive(Debug, Clone)]
enum TelegramProgressSink {
    Draft { draft_id: i64 },
    Placeholder { message_id: i64 },
}

fn read_required_string<'a>(
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

fn read_chat_id(value: Option<&serde_json::Value>) -> Result<String> {
    let Some(value) = value else {
        anyhow::bail!("missing value");
    };

    if let Some(id) = value.as_i64() {
        if id == 0 {
            anyhow::bail!("chat_id must not be zero");
        }
        return Ok(id.to_string());
    }
    if let Some(id) = value.as_u64() {
        if id == 0 {
            anyhow::bail!("chat_id must not be zero");
        }
        return Ok(id.to_string());
    }

    let text = value
        .as_str()
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| anyhow!("chat_id must be a non-empty integer or integer string"))?;

    let is_valid = text
        .strip_prefix('-')
        .unwrap_or(text)
        .chars()
        .all(|ch| ch.is_ascii_digit());
    if !is_valid || text == "-" || text == "0" || text == "-0" {
        anyhow::bail!("chat_id must be a non-zero integer or integer string");
    }

    Ok(text.to_string())
}

fn read_chat_ids(settings: &serde_json::Map<String, serde_json::Value>) -> Result<Vec<String>> {
    if let Some(value) = settings.get("chat_ids") {
        return read_chat_id_list(value);
    }

    Ok(vec![read_chat_id(settings.get("chat_id"))?])
}

fn read_chat_id_list(value: &serde_json::Value) -> Result<Vec<String>> {
    let mut ids = Vec::new();
    match value {
        serde_json::Value::Array(values) => {
            for item in values {
                ids.push(read_chat_id(Some(item))?);
            }
        }
        serde_json::Value::String(text) => {
            for item in text.split(',') {
                ids.push(read_chat_id(Some(&serde_json::Value::String(
                    item.trim().to_string(),
                )))?);
            }
        }
        _ => ids.push(read_chat_id(Some(value))?),
    }

    let mut seen = HashSet::new();
    ids.retain(|id| seen.insert(id.clone()));
    if ids.is_empty() {
        anyhow::bail!("chat_ids must include at least one numeric chat id");
    }
    Ok(ids)
}

fn read_respond_mode(value: Option<&serde_json::Value>) -> Result<TelegramRespondMode> {
    let Some(value) = value else {
        return Ok(TelegramRespondMode::All);
    };
    let mode = value.as_str().ok_or_else(|| {
        anyhow!(
            "[telegram_config_invalid_respond_mode] Telegram channel setting 'respond_mode' must be a string"
        )
    })?;
    match mode.trim().to_ascii_lowercase().as_str() {
        "all" => Ok(TelegramRespondMode::All),
        "mentions" => Ok(TelegramRespondMode::Mentions),
        "replies" => Ok(TelegramRespondMode::Replies),
        "mentions_or_replies" => Ok(TelegramRespondMode::MentionsOrReplies),
        _ => anyhow::bail!(
            "[telegram_config_invalid_respond_mode] Telegram channel setting 'respond_mode' must be one of: all, mentions, replies, mentions_or_replies"
        ),
    }
}

fn read_telegram_session_scope(value: Option<&serde_json::Value>) -> Result<ChannelSessionScope> {
    let Some(value) = value else {
        return Ok(ChannelSessionScope::User);
    };
    let scope = value.as_str().ok_or_else(|| {
        anyhow!(
            "[telegram_config_invalid_session_scope] Telegram channel setting 'session_scope' must be a string"
        )
    })?;
    match scope.trim().to_ascii_lowercase().as_str() {
        "user" => Ok(ChannelSessionScope::User),
        "thread" => Ok(ChannelSessionScope::Thread),
        "room" => Ok(ChannelSessionScope::Room),
        _ => anyhow::bail!(
            "[telegram_config_invalid_session_scope] Telegram channel setting 'session_scope' must be one of: user, thread, room"
        ),
    }
}

fn read_optional_telegram_session_scope(
    value: Option<&serde_json::Value>,
    key: &str,
) -> Result<Option<ChannelSessionScope>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let scope = value.as_str().ok_or_else(|| {
        anyhow!(
            "[telegram_config_invalid_session_scope] Telegram channel setting '{}' must be a string",
            key
        )
    })?;
    match scope.trim().to_ascii_lowercase().as_str() {
        "user" => Ok(Some(ChannelSessionScope::User)),
        "thread" => Ok(Some(ChannelSessionScope::Thread)),
        "room" => Ok(Some(ChannelSessionScope::Room)),
        _ => anyhow::bail!(
            "[telegram_config_invalid_session_scope] Telegram channel setting '{}' must be one of: user, thread, room",
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
                "[telegram_config_deprecated_session_scope_key] Telegram channel setting '{}' is no longer supported; use '{}' instead",
                legacy,
                replacement
            );
        }
    }
    Ok(())
}

fn effective_telegram_session_scope(
    config: &TelegramChannelDriverConfig,
    chat: &TelegramChat,
) -> ChannelSessionScope {
    match chat.chat_type.as_str() {
        "private" => config.session_scope_dm.unwrap_or(config.session_scope),
        "channel" => config.session_scope_channel.unwrap_or(config.session_scope),
        "group" | "supergroup" => config.session_scope_group.unwrap_or(config.session_scope),
        _ => config.session_scope,
    }
}

fn read_stream_mode(value: Option<&serde_json::Value>) -> Result<ChannelStreamMode> {
    let Some(value) = value else {
        return Ok(ChannelStreamMode::Off);
    };
    let mode = value.as_str().ok_or_else(|| {
        anyhow!(
            "[telegram_config_invalid_stream_mode] Telegram channel setting 'stream_mode' must be a string"
        )
    })?;
    match mode.trim().to_ascii_lowercase().as_str() {
        "off" => Ok(ChannelStreamMode::Off),
        "typing" => Ok(ChannelStreamMode::Typing),
        "draft" => Ok(ChannelStreamMode::Draft),
        "block" => Ok(ChannelStreamMode::Block),
        _ => anyhow::bail!(
            "[telegram_config_invalid_stream_mode] Telegram channel setting 'stream_mode' must be one of: off, typing, draft, block"
        ),
    }
}

fn progress_key(conversation: &ChannelConversationKey) -> Result<String> {
    serde_json::to_string(conversation)
        .with_context(|| "[telegram_progress_key_invalid] Failed to serialize conversation key")
}

fn conversation_chat_id(default_chat_id: &str, conversation: &ChannelConversationKey) -> String {
    conversation
        .room_id
        .as_ref()
        .filter(|value| !value.trim().is_empty())
        .cloned()
        .unwrap_or_else(|| default_chat_id.to_string())
}

fn utf16_slice(text: &str, offset: usize, length: usize) -> Option<&str> {
    let end = offset.saturating_add(length);
    let mut utf16_index = 0usize;
    let mut start_byte = None;
    let mut end_byte = None;

    for (byte_index, ch) in text.char_indices() {
        if utf16_index == offset {
            start_byte = Some(byte_index);
        }
        if utf16_index == end {
            end_byte = Some(byte_index);
            break;
        }

        utf16_index = utf16_index.saturating_add(ch.len_utf16());

        if utf16_index == offset {
            start_byte = Some(byte_index + ch.len_utf8());
        }
        if utf16_index == end {
            end_byte = Some(byte_index + ch.len_utf8());
            break;
        }
    }

    if offset == utf16_index && start_byte.is_none() {
        start_byte = Some(text.len());
    }
    if end == utf16_index && end_byte.is_none() {
        end_byte = Some(text.len());
    }

    Some(&text[start_byte?..end_byte?])
}

fn chat_id_is_private(chat_id: &str) -> bool {
    !chat_id.trim_start().starts_with('-')
}

fn stream_preview_text(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let mut out = String::new();
    for ch in trimmed.chars() {
        if out.chars().count() >= TELEGRAM_MESSAGE_MAX_LEN.saturating_sub(1) {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

fn render_stream_preview(text: &str, thinking: Option<&str>) -> String {
    let text = text.trim();
    let thinking = thinking.map(str::trim).unwrap_or_default();

    if text.is_empty() && thinking.is_empty() {
        return String::new();
    }

    let mut preview = String::new();
    if !thinking.is_empty() {
        preview.push_str("Thinking…\n");
        preview.push_str(thinking);
    }
    if !text.is_empty() {
        if !preview.is_empty() {
            preview.push_str("\n\nReply\n");
        }
        preview.push_str(text);
    }

    stream_preview_text(&preview)
}

fn resolve_message_thread_id(conversation: &ChannelConversationKey) -> Result<Option<i64>> {
    let Some(room_id) = conversation.room_id.as_deref() else {
        return Ok(None);
    };
    if conversation.thread_id == room_id {
        return Ok(None);
    }

    conversation
        .thread_id
        .parse::<i64>()
        .map(Some)
        .with_context(|| {
            format!(
                "[telegram_invalid_thread_id] Telegram conversation thread id '{}' is not a valid numeric message thread id",
                conversation.thread_id
            )
        })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TelegramRenderMode {
    PlainText,
    Html,
}

#[derive(Debug, Clone)]
struct TelegramRenderedMessage {
    chunks: Vec<String>,
    parse_mode: Option<&'static str>,
    reply_to_message_id: Option<i64>,
    disable_web_page_preview: bool,
    disable_notification: bool,
}

fn telegram_batches_from_message(
    chat_id: &str,
    message_thread_id: Option<i64>,
    message: &OutboundMessage,
) -> Result<Vec<serde_json::Value>> {
    let rendered = render_telegram_message(message)?;
    Ok(rendered
        .chunks
        .into_iter()
        .map(|text| {
            telegram_payload(
                chat_id,
                message_thread_id,
                text,
                rendered.parse_mode,
                rendered.reply_to_message_id,
                rendered.disable_web_page_preview,
                rendered.disable_notification,
            )
        })
        .collect())
}

fn render_telegram_message(message: &OutboundMessage) -> Result<TelegramRenderedMessage> {
    let render_mode = resolve_render_mode(message);
    let reply_to_message_id = metadata_i64(&message.metadata, "telegram_reply_to_message_id")?;
    let disable_web_page_preview = message
        .metadata
        .get("telegram_disable_web_page_preview")
        .and_then(|value| value.as_bool())
        .unwrap_or(true);
    let disable_notification = message
        .metadata
        .get("telegram_disable_notification")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let final_thinking = message
        .metadata
        .get("channel_final_thinking")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty());

    let mut chunks = match render_mode {
        TelegramRenderMode::PlainText => {
            let mut rendered = render_text_blocks(&message.blocks);
            if let Some(thinking) = final_thinking {
                rendered = prepend_final_thinking_text(&rendered, thinking);
            }
            split_for_telegram_message(rendered)
        }
        TelegramRenderMode::Html => render_html_chunks(message, final_thinking),
    };

    if chunks.is_empty() && message.attachments.is_empty() {
        chunks.push("(no output)".to_string());
    }

    Ok(TelegramRenderedMessage {
        chunks,
        parse_mode: match render_mode {
            TelegramRenderMode::PlainText => None,
            TelegramRenderMode::Html => Some("HTML"),
        },
        reply_to_message_id,
        disable_web_page_preview,
        disable_notification,
    })
}

fn resolve_render_mode(message: &OutboundMessage) -> TelegramRenderMode {
    if message
        .metadata
        .get("telegram_format")
        .and_then(|value| value.as_str())
        .is_some_and(|value| {
            value.eq_ignore_ascii_case("plain") || value.eq_ignore_ascii_case("text")
        })
    {
        return TelegramRenderMode::PlainText;
    }
    TelegramRenderMode::Html
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

fn render_html_chunks(message: &OutboundMessage, final_thinking: Option<&str>) -> Vec<String> {
    let mut segments = Vec::new();
    if let Some(thinking) = final_thinking {
        segments.push("<i>Thinking</i>".to_string());
        segments.extend(split_wrapped_segment(thinking, "<pre>", "</pre>"));
        segments.push("<i>Reply</i>".to_string());
    }
    for block in &message.blocks {
        segments.extend(render_html_segments_for_block(block));
    }

    pack_segments(segments)
}

fn prepend_final_thinking_text(rendered: &str, thinking: &str) -> String {
    let trimmed = rendered.trim();
    if trimmed.is_empty() {
        format!("Thinking:\n{}\n", thinking)
    } else {
        format!("Thinking:\n{}\n\nReply:\n{}", thinking, trimmed)
    }
}

fn render_html_segments_for_block(block: &MessageBlock) -> Vec<String> {
    match block {
        MessageBlock::Text { text } => render_markdown_segments(text),
        MessageBlock::CodeBlock { code, .. } => split_wrapped_segment(code, "<pre>", "</pre>"),
    }
}

fn attachment_preview_text(attachments: &[ChannelAttachment]) -> String {
    match attachments.len() {
        0 => "(no output)".to_string(),
        1 => format!("Sent attachment: {}", attachments[0].name),
        count => format!("Sent {count} attachments"),
    }
}

fn attachment_kind_from_content_type(content_type: Option<&str>) -> TelegramAttachmentKind {
    if content_type.is_some_and(|value| value.starts_with("image/")) {
        TelegramAttachmentKind::Image
    } else {
        TelegramAttachmentKind::File
    }
}

fn infer_audio_name(audio: &TelegramAudio) -> Option<String> {
    match (audio.performer.as_deref(), audio.title.as_deref()) {
        (Some(performer), Some(title))
            if !performer.trim().is_empty() && !title.trim().is_empty() =>
        {
            Some(format!("{performer} - {title}.mp3"))
        }
        (_, Some(title)) if !title.trim().is_empty() => Some(format!("{title}.mp3")),
        _ => None,
    }
}

fn default_media_dir_for_runtime(channel_runtime_id: &str) -> PathBuf {
    std::env::temp_dir()
        .join("turin")
        .join("channels")
        .join("telegram")
        .join(sanitize_runtime_component(channel_runtime_id))
        .join("media")
}

fn sanitize_runtime_component(raw: &str) -> String {
    let mut out = String::new();
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('-');
        }
    }
    let trimmed = out.trim_matches('-');
    if trimmed.is_empty() {
        "default".to_string()
    } else {
        trimmed.to_string()
    }
}

fn unique_media_name(name: &str, fallback_path: Option<&str>) -> String {
    let extension = media_extension(name, fallback_path)
        .map(|value| format!(".{value}"))
        .unwrap_or_default();
    format!("{}{}", uuid::Uuid::now_v7().simple(), extension)
}

fn media_extension(name: &str, fallback_path: Option<&str>) -> Option<String> {
    std::path::Path::new(name)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::trim)
        .filter(|ext| !ext.is_empty())
        .map(str::to_ascii_lowercase)
        .or_else(|| {
            fallback_path
                .and_then(|path| Path::new(path).extension().and_then(|ext| ext.to_str()))
                .map(str::trim)
                .filter(|ext| !ext.is_empty())
                .map(str::to_ascii_lowercase)
        })
}

#[derive(Debug, Clone, Copy)]
struct MarkdownListState {
    ordered: bool,
    next_index: u64,
}

#[derive(Debug, Default, Clone)]
struct MarkdownTableState {
    rows: Vec<Vec<String>>,
    current_row: Vec<String>,
    current_cell: String,
    header_rows: usize,
}

fn render_markdown_segments(markdown: &str) -> Vec<String> {
    let trimmed = markdown.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    let mut options = Options::empty();
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_TASKLISTS);

    let parser = Parser::new_ext(trimmed, options);
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut blockquote_depth = 0usize;
    let mut list_stack: Vec<MarkdownListState> = Vec::new();
    let mut code_block: Option<String> = None;
    let mut table_state: Option<MarkdownTableState> = None;

    for event in parser {
        match event {
            Event::Start(tag) => match tag {
                Tag::Paragraph => {}
                Tag::Heading { .. } => {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str("<b>");
                }
                Tag::BlockQuote(_) => {
                    blockquote_depth = blockquote_depth.saturating_add(1);
                }
                Tag::List(start) => {
                    list_stack.push(MarkdownListState {
                        ordered: start.is_some(),
                        next_index: start.unwrap_or(1),
                    });
                }
                Tag::Item => {
                    flush_rich_segment(&mut segments, &mut current);
                    current.push_str(&blockquote_prefix(blockquote_depth));
                    if let Some(state) = list_stack.last_mut() {
                        if state.ordered {
                            current.push_str(&format!("{}. ", state.next_index));
                            state.next_index = state.next_index.saturating_add(1);
                        } else {
                            current.push_str("• ");
                        }
                    }
                }
                Tag::Emphasis => {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str("<i>");
                }
                Tag::Strong => {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str("<b>");
                }
                Tag::Strikethrough => {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str("<s>");
                }
                Tag::Link { dest_url, .. } => {
                    if table_state.is_some() {
                        continue;
                    }
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str("<a href=\"");
                    current.push_str(&escape_html(dest_url.as_ref()));
                    current.push_str("\">");
                }
                Tag::Table(_) => {
                    flush_rich_segment(&mut segments, &mut current);
                    table_state = Some(MarkdownTableState::default());
                }
                Tag::TableHead => {}
                Tag::TableRow => {
                    if let Some(table) = table_state.as_mut() {
                        table.current_row.clear();
                    }
                }
                Tag::TableCell => {
                    if let Some(table) = table_state.as_mut() {
                        table.current_cell.clear();
                    }
                }
                Tag::CodeBlock(kind) => {
                    flush_rich_segment(&mut segments, &mut current);
                    let mut rendered = String::new();
                    if let CodeBlockKind::Fenced(language) = kind {
                        let language = language.trim();
                        if !language.is_empty() {
                            rendered.push_str(language);
                            rendered.push('\n');
                        }
                    }
                    code_block = Some(rendered);
                }
                _ => {}
            },
            Event::End(tag) => match tag {
                TagEnd::Paragraph if list_stack.is_empty() => {
                    flush_rich_segment(&mut segments, &mut current);
                }
                TagEnd::Heading(_) => {
                    current.push_str("</b>");
                    flush_rich_segment(&mut segments, &mut current);
                }
                TagEnd::BlockQuote(_) => {
                    flush_rich_segment(&mut segments, &mut current);
                    blockquote_depth = blockquote_depth.saturating_sub(1);
                }
                TagEnd::List(_) => {
                    flush_rich_segment(&mut segments, &mut current);
                    list_stack.pop();
                }
                TagEnd::Item => {
                    flush_rich_segment(&mut segments, &mut current);
                }
                TagEnd::Emphasis => current.push_str("</i>"),
                TagEnd::Strong => current.push_str("</b>"),
                TagEnd::Strikethrough => current.push_str("</s>"),
                TagEnd::Table => {
                    if let Some(table) = table_state.take() {
                        let rendered = render_markdown_table(&table);
                        if !rendered.trim().is_empty() {
                            segments.extend(split_wrapped_segment(&rendered, "<pre>", "</pre>"));
                        }
                    }
                }
                TagEnd::TableHead => {
                    if let Some(table) = table_state.as_mut() {
                        if !table.current_row.is_empty() {
                            table.rows.push(std::mem::take(&mut table.current_row));
                        }
                        table.header_rows = table.rows.len();
                    }
                }
                TagEnd::TableRow => {
                    if let Some(table) = table_state.as_mut()
                        && !table.current_row.is_empty()
                    {
                        table.rows.push(std::mem::take(&mut table.current_row));
                    }
                }
                TagEnd::TableCell => {
                    if let Some(table) = table_state.as_mut() {
                        table
                            .current_row
                            .push(normalize_table_cell(&table.current_cell));
                        table.current_cell.clear();
                    }
                }
                TagEnd::Link if table_state.is_none() => {
                    current.push_str("</a>");
                }
                TagEnd::CodeBlock => {
                    if let Some(rendered) = code_block.take() {
                        segments.extend(split_wrapped_segment(&rendered, "<pre>", "</pre>"));
                    }
                }
                _ => {}
            },
            Event::Text(text) => {
                if let Some(code) = code_block.as_mut() {
                    code.push_str(text.as_ref());
                } else if let Some(table) = table_state.as_mut() {
                    table.current_cell.push_str(text.as_ref());
                } else {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str(&escape_html(text.as_ref()));
                }
            }
            Event::Code(text) => {
                if let Some(table) = table_state.as_mut() {
                    table.current_cell.push_str(text.as_ref());
                } else {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str("<code>");
                    current.push_str(&escape_html(text.as_ref()));
                    current.push_str("</code>");
                }
            }
            Event::SoftBreak | Event::HardBreak => {
                if let Some(code) = code_block.as_mut() {
                    code.push('\n');
                } else if let Some(table) = table_state.as_mut() {
                    if !table.current_cell.ends_with(' ') && !table.current_cell.is_empty() {
                        table.current_cell.push(' ');
                    }
                } else {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push('\n');
                }
            }
            Event::Rule => {
                flush_rich_segment(&mut segments, &mut current);
                segments.push("────────".to_string());
            }
            Event::TaskListMarker(checked) => {
                if let Some(table) = table_state.as_mut() {
                    table
                        .current_cell
                        .push_str(if checked { "[x] " } else { "[ ] " });
                } else {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str(if checked { "[x] " } else { "[ ] " });
                }
            }
            Event::Html(html) | Event::InlineHtml(html) => {
                if let Some(code) = code_block.as_mut() {
                    code.push_str(html.as_ref());
                } else if let Some(table) = table_state.as_mut() {
                    table.current_cell.push_str(html.as_ref());
                } else {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str(&escape_html(html.as_ref()));
                }
            }
            Event::InlineMath(text) | Event::DisplayMath(text) => {
                if let Some(table) = table_state.as_mut() {
                    table.current_cell.push_str(text.as_ref());
                } else {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str(&escape_html(text.as_ref()));
                }
            }
            Event::FootnoteReference(reference) => {
                if let Some(table) = table_state.as_mut() {
                    table.current_cell.push('[');
                    table.current_cell.push_str(reference.as_ref());
                    table.current_cell.push(']');
                } else {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push('[');
                    current.push_str(&escape_html(reference.as_ref()));
                    current.push(']');
                }
            }
        }
    }

    flush_rich_segment(&mut segments, &mut current);
    pack_segments(segments)
}

fn ensure_prefix(current: &mut String, blockquote_depth: usize) {
    if current.is_empty() {
        current.push_str(&blockquote_prefix(blockquote_depth));
    }
}

fn blockquote_prefix(depth: usize) -> String {
    "&gt; ".repeat(depth)
}

fn flush_rich_segment(segments: &mut Vec<String>, current: &mut String) {
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        segments.extend(split_rich_segment(trimmed));
    }
    current.clear();
}

fn normalize_table_cell(cell: &str) -> String {
    cell.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn render_markdown_table(table: &MarkdownTableState) -> String {
    if table.rows.is_empty() {
        return String::new();
    }

    let column_count = table.rows.iter().map(Vec::len).max().unwrap_or(0);
    if column_count == 0 {
        return String::new();
    }

    let mut widths = vec![0usize; column_count];
    for row in &table.rows {
        for (index, cell) in row.iter().enumerate() {
            widths[index] = widths[index].max(cell.chars().count());
        }
    }

    let format_row = |row: &[String]| {
        let mut out = String::from("|");
        for (index, width) in widths.iter().enumerate() {
            let cell = row.get(index).map(String::as_str).unwrap_or("");
            out.push(' ');
            out.push_str(cell);
            let padding = width.saturating_sub(cell.chars().count());
            if padding > 0 {
                out.push_str(&" ".repeat(padding));
            }
            out.push(' ');
            out.push('|');
        }
        out
    };

    let separator = {
        let mut out = String::from("|");
        for width in &widths {
            out.push(' ');
            out.push_str(&"-".repeat((*width).max(3)));
            out.push(' ');
            out.push('|');
        }
        out
    };

    let mut lines = Vec::new();
    for (index, row) in table.rows.iter().enumerate() {
        lines.push(format_row(row));
        if table.header_rows > 0 && index + 1 == table.header_rows {
            lines.push(separator.clone());
        }
    }

    lines.join("\n")
}

fn split_rich_segment(content: &str) -> Vec<String> {
    if content.chars().count() <= TELEGRAM_MESSAGE_MAX_LEN {
        return vec![content.to_string()];
    }

    let mut out = Vec::new();
    let mut current = String::new();
    for line in content.lines() {
        let tentative = if current.is_empty() {
            line.to_string()
        } else {
            format!("{current}\n{line}")
        };
        if tentative.chars().count() > TELEGRAM_MESSAGE_MAX_LEN {
            if !current.is_empty() {
                out.push(current.clone());
                current.clear();
            }
            if line.chars().count() > TELEGRAM_MESSAGE_MAX_LEN {
                out.extend(split_plain_segment(line));
            } else {
                current = line.to_string();
            }
        } else {
            current = tentative;
        }
    }

    if !current.is_empty() {
        out.push(current);
    }

    out
}

fn split_plain_segment(content: &str) -> Vec<String> {
    split_content_to_limit(content, TELEGRAM_MESSAGE_MAX_LEN)
}

fn split_wrapped_segment(content: &str, prefix: &str, suffix: &str) -> Vec<String> {
    let limit = TELEGRAM_MESSAGE_MAX_LEN
        .saturating_sub(prefix.chars().count())
        .saturating_sub(suffix.chars().count())
        .max(1);
    split_content_to_limit(&escape_html(content), limit)
        .into_iter()
        .map(|chunk| format!("{prefix}{chunk}{suffix}"))
        .collect()
}

fn split_content_to_limit(content: &str, limit: usize) -> Vec<String> {
    let mut out = Vec::new();
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return out;
    }

    let mut current = String::new();
    for ch in trimmed.chars() {
        current.push(ch);
        if current.chars().count() >= limit {
            out.push(current.clone());
            current.clear();
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn pack_segments(segments: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();

    for segment in segments {
        let segment = segment.trim().to_string();
        if segment.is_empty() {
            continue;
        }

        let tentative = if current.is_empty() {
            segment.clone()
        } else {
            format!("{current}\n\n{segment}")
        };
        if tentative.chars().count() > TELEGRAM_MESSAGE_MAX_LEN {
            if !current.is_empty() {
                out.push(current.clone());
                current.clear();
            }
            current = segment;
        } else {
            current = tentative;
        }
    }

    if !current.is_empty() {
        out.push(current);
    }

    out
}

fn escape_html(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn split_for_telegram_message(content: String) -> Vec<String> {
    let mut out = Vec::new();
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return out;
    }

    let mut current = String::new();
    for line in trimmed.lines() {
        if line.chars().count() > TELEGRAM_MESSAGE_MAX_LEN {
            if !current.is_empty() {
                out.push(current.clone());
                current.clear();
            }

            let mut segment = String::new();
            for ch in line.chars() {
                segment.push(ch);
                if segment.chars().count() >= TELEGRAM_MESSAGE_MAX_LEN {
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
        if tentative.chars().count() > TELEGRAM_MESSAGE_MAX_LEN {
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

fn telegram_payload(
    chat_id: &str,
    message_thread_id: Option<i64>,
    text: String,
    parse_mode: Option<&'static str>,
    reply_to_message_id: Option<i64>,
    disable_web_page_preview: bool,
    disable_notification: bool,
) -> serde_json::Value {
    let mut payload = serde_json::Map::new();
    payload.insert("chat_id".to_string(), serde_json::json!(chat_id));
    payload.insert("text".to_string(), serde_json::json!(text));
    payload.insert(
        "disable_web_page_preview".to_string(),
        serde_json::json!(disable_web_page_preview),
    );
    payload.insert(
        "disable_notification".to_string(),
        serde_json::json!(disable_notification),
    );
    if let Some(message_thread_id) = message_thread_id {
        payload.insert(
            "message_thread_id".to_string(),
            serde_json::json!(message_thread_id),
        );
    }
    if let Some(parse_mode) = parse_mode {
        payload.insert("parse_mode".to_string(), serde_json::json!(parse_mode));
    }
    if let Some(reply_to_message_id) = reply_to_message_id {
        payload.insert(
            "reply_to_message_id".to_string(),
            serde_json::json!(reply_to_message_id),
        );
        payload.insert(
            "allow_sending_without_reply".to_string(),
            serde_json::json!(true),
        );
    }
    serde_json::Value::Object(payload)
}

fn telegram_edit_payload(
    chat_id: &str,
    message_id: i64,
    text: String,
    parse_mode: Option<&str>,
    disable_web_page_preview: bool,
) -> serde_json::Value {
    let mut payload = serde_json::Map::new();
    payload.insert("chat_id".to_string(), serde_json::json!(chat_id));
    payload.insert("message_id".to_string(), serde_json::json!(message_id));
    payload.insert("text".to_string(), serde_json::json!(text));
    payload.insert(
        "disable_web_page_preview".to_string(),
        serde_json::json!(disable_web_page_preview),
    );
    if let Some(parse_mode) = parse_mode {
        payload.insert("parse_mode".to_string(), serde_json::json!(parse_mode));
    }
    serde_json::Value::Object(payload)
}

fn metadata_i64(
    metadata: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<Option<i64>> {
    let Some(value) = metadata.get(key) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    if let Some(number) = value.as_i64() {
        return Ok(Some(number));
    }
    if let Some(number) = value.as_u64() {
        return i64::try_from(number).map(Some).map_err(|_| {
            anyhow!(
                "[telegram_invalid_metadata] Telegram metadata '{}' is too large for i64",
                key
            )
        });
    }
    if let Some(text) = value.as_str() {
        return text.parse::<i64>().map(Some).map_err(|_| {
            anyhow!(
                "[telegram_invalid_metadata] Telegram metadata '{}' must be an integer or integer string",
                key
            )
        });
    }
    anyhow::bail!(
        "[telegram_invalid_metadata] Telegram metadata '{}' must be an integer or integer string",
        key
    );
}

fn classify_api_error(method: &str, status_code: u16, description: &str) -> String {
    let lower = description.to_ascii_lowercase();
    if status_code == 401 || lower.contains("unauthorized") {
        return "telegram_auth_invalid_token".to_string();
    }
    if status_code == 429 || lower.contains("too many requests") {
        return "telegram_rate_limited".to_string();
    }

    match method {
        "getUpdates" => {
            if lower.contains("webhook") {
                "telegram_polling_webhook_active".to_string()
            } else if lower.contains("terminated by other getupdates request")
                || lower.contains("terminated by other long poll")
            {
                "telegram_polling_conflict".to_string()
            } else {
                "telegram_get_updates_failed".to_string()
            }
        }
        "sendMessage" => {
            if lower.contains("chat not found") {
                "telegram_send_chat_not_found".to_string()
            } else {
                "telegram_send_failed".to_string()
            }
        }
        "sendMessageDraft" => "telegram_send_draft_failed".to_string(),
        "editMessageText" => "telegram_edit_message_failed".to_string(),
        "sendChatAction" => "telegram_chat_action_failed".to_string(),
        _ => "telegram_api_failed".to_string(),
    }
}

fn is_retriable_api_error(code: &str, status_code: u16) -> bool {
    status_code == 429
        || status_code >= 500
        || matches!(
            code,
            "telegram_http_request_failed"
                | "telegram_http_decode_failed"
                | "telegram_rate_limited"
                | "telegram_send_failed"
                | "telegram_send_draft_failed"
                | "telegram_edit_message_failed"
                | "telegram_chat_action_failed"
                | "telegram_get_updates_failed"
        )
}

fn is_not_modified_description(description: &str) -> bool {
    description
        .to_ascii_lowercase()
        .contains("message is not modified")
}

fn retry_backoff(attempt: u32) -> Duration {
    let exponent = attempt.min(5);
    let millis = 250u64.saturating_mul(2u64.saturating_pow(exponent));
    Duration::from_millis(millis.min(8_000))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> TelegramChannelDriverConfig {
        TelegramChannelDriverConfig {
            base_url: DEFAULT_BASE_URL.to_string(),
            workspace_id: "telegram".to_string(),
            chat_ids: vec!["-10012345".to_string()],
            accept_all_chats: false,
            token: "token".to_string(),
            poll_timeout_seconds: 30,
            poll_interval: Duration::from_millis(250),
            max_updates_per_poll: 25,
            max_inbound_text_chars: DEFAULT_MAX_INBOUND_TEXT_CHARS,
            start_from_latest: false,
            ignore_bot_messages: true,
            respond_mode: TelegramRespondMode::All,
            session_scope: ChannelSessionScope::User,
            session_scope_dm: None,
            session_scope_group: None,
            session_scope_channel: None,
            stream_mode: ChannelStreamMode::Off,
            stream_thinking: false,
            persist_thinking: false,
        }
    }

    fn driver() -> TelegramChannelDriver {
        let (_tx, rx) = watch::channel(false);
        TelegramChannelDriver::from_config("telegram-runtime", config(), rx).unwrap()
    }

    fn sample_event_with_message_id(message_id: i64) -> InboundEvent {
        let key = ChannelConversationKey {
            channel: ChannelKind::new("telegram"),
            workspace_id: "telegram".into(),
            room_id: Some("-10012345".into()),
            thread_id: "-10012345".into(),
            user_id: Some("user-1".into()),
        };
        let mut metadata = serde_json::Map::new();
        metadata.insert(
            "telegram_message_id".to_string(),
            serde_json::json!(message_id),
        );
        InboundEvent {
            conversation: key.clone(),
            message: ChannelMessageRef {
                conversation: key,
                message_id: format!("m-{message_id}"),
            },
            user: ChannelUser {
                id: "user-1".into(),
                display_name: Some("User One".into()),
                username: Some("user1".into()),
            },
            session_scope: ChannelSessionScope::User,
            text: "hello".into(),
            attachments: vec![],
            metadata,
        }
    }

    #[test]
    fn normalize_uses_chat_id_as_default_thread() {
        let driver = driver();
        let update = TelegramUpdate {
            update_id: 1,
            message: Some(TelegramMessage {
                message_id: 99,
                chat: TelegramChat {
                    id: -10012345,
                    chat_type: "supergroup".to_string(),
                    title: Some("Ops".to_string()),
                    username: None,
                    first_name: None,
                },
                from: Some(TelegramUser {
                    id: 7,
                    is_bot: Some(false),
                    first_name: Some("Ava".to_string()),
                    last_name: Some("Stone".to_string()),
                    username: Some("ava".to_string()),
                }),
                sender_chat: None,
                text: Some("hello".to_string()),
                caption: None,
                entities: Vec::new(),
                caption_entities: Vec::new(),
                message_thread_id: None,
                reply_to_message: None,
                ..Default::default()
            }),
            channel_post: None,
        };

        let event = driver.normalize_update(update).expect("normalized event");
        assert_eq!(event.conversation.channel, ChannelKind::new("telegram"));
        assert_eq!(event.conversation.room_id.as_deref(), Some("-10012345"));
        assert_eq!(event.conversation.thread_id, "-10012345");
        assert_eq!(event.user.display_name.as_deref(), Some("Ava Stone"));
    }

    #[test]
    fn normalize_uses_topic_thread_id_when_present() {
        let driver = driver();
        let update = TelegramUpdate {
            update_id: 2,
            message: Some(TelegramMessage {
                message_id: 100,
                chat: TelegramChat {
                    id: -10012345,
                    chat_type: "supergroup".to_string(),
                    title: Some("Ops".to_string()),
                    username: None,
                    first_name: None,
                },
                from: Some(TelegramUser {
                    id: 8,
                    is_bot: Some(false),
                    first_name: Some("Mia".to_string()),
                    last_name: None,
                    username: Some("mia".to_string()),
                }),
                sender_chat: None,
                text: Some("topic ping".to_string()),
                caption: None,
                entities: Vec::new(),
                caption_entities: Vec::new(),
                message_thread_id: Some(444),
                reply_to_message: None,
                ..Default::default()
            }),
            channel_post: None,
        };

        let event = driver.normalize_update(update).expect("normalized event");
        assert_eq!(event.conversation.thread_id, "444");
        assert_eq!(event.metadata["telegram_message_thread_id"], 444);
    }

    #[test]
    fn normalize_thread_scope_shares_topic_across_users() {
        let mut config = config();
        config.session_scope = ChannelSessionScope::Thread;
        let (_tx, rx) = watch::channel(false);
        let driver = TelegramChannelDriver::from_config("telegram-runtime", config, rx).unwrap();
        let update = TelegramUpdate {
            update_id: 2,
            message: Some(TelegramMessage {
                message_id: 100,
                chat: TelegramChat {
                    id: -10012345,
                    chat_type: "supergroup".to_string(),
                    title: Some("Ops".to_string()),
                    username: None,
                    first_name: None,
                },
                from: Some(TelegramUser {
                    id: 8,
                    is_bot: Some(false),
                    first_name: Some("Mia".to_string()),
                    last_name: None,
                    username: Some("mia".to_string()),
                }),
                sender_chat: None,
                text: Some("topic ping".to_string()),
                caption: None,
                entities: Vec::new(),
                caption_entities: Vec::new(),
                message_thread_id: Some(444),
                reply_to_message: None,
                ..Default::default()
            }),
            channel_post: None,
        };

        let event = driver.normalize_update(update).expect("normalized event");
        assert_eq!(event.session_scope, ChannelSessionScope::Thread);
        assert_eq!(event.conversation.thread_id, "444");
        assert_eq!(event.conversation.user_id, None);
    }

    #[test]
    fn normalize_room_scope_collapses_topics_and_users() {
        let mut config = config();
        config.session_scope = ChannelSessionScope::Room;
        let (_tx, rx) = watch::channel(false);
        let driver = TelegramChannelDriver::from_config("telegram-runtime", config, rx).unwrap();
        let update = TelegramUpdate {
            update_id: 2,
            message: Some(TelegramMessage {
                message_id: 100,
                chat: TelegramChat {
                    id: -10012345,
                    chat_type: "supergroup".to_string(),
                    title: Some("Ops".to_string()),
                    username: None,
                    first_name: None,
                },
                from: Some(TelegramUser {
                    id: 8,
                    is_bot: Some(false),
                    first_name: Some("Mia".to_string()),
                    last_name: None,
                    username: Some("mia".to_string()),
                }),
                sender_chat: None,
                text: Some("topic ping".to_string()),
                caption: None,
                entities: Vec::new(),
                caption_entities: Vec::new(),
                message_thread_id: Some(444),
                reply_to_message: None,
                ..Default::default()
            }),
            channel_post: None,
        };

        let event = driver.normalize_update(update).expect("normalized event");
        assert_eq!(event.session_scope, ChannelSessionScope::Room);
        assert_eq!(event.conversation.thread_id, "-10012345");
        assert_eq!(event.conversation.user_id, None);
    }

    #[test]
    fn normalize_ignores_bot_messages() {
        let driver = driver();
        let update = TelegramUpdate {
            update_id: 3,
            message: Some(TelegramMessage {
                message_id: 101,
                chat: TelegramChat {
                    id: -10012345,
                    chat_type: "supergroup".to_string(),
                    title: Some("Ops".to_string()),
                    username: None,
                    first_name: None,
                },
                from: Some(TelegramUser {
                    id: 9,
                    is_bot: Some(true),
                    first_name: Some("Bot".to_string()),
                    last_name: None,
                    username: Some("bot".to_string()),
                }),
                sender_chat: None,
                text: Some("ignore me".to_string()),
                caption: None,
                entities: Vec::new(),
                caption_entities: Vec::new(),
                message_thread_id: None,
                reply_to_message: None,
                ..Default::default()
            }),
            channel_post: None,
        };

        assert!(driver.normalize_update(update).is_none());
    }

    #[test]
    fn normalize_accepts_updates_from_any_configured_chat_id() {
        let mut config = config();
        config.chat_ids.push("-10099999".to_string());
        let (_tx, rx) = watch::channel(false);
        let driver = TelegramChannelDriver::from_config("telegram-runtime", config, rx).unwrap();
        let update = TelegramUpdate {
            update_id: 4,
            message: Some(TelegramMessage {
                message_id: 102,
                chat: TelegramChat {
                    id: -10099999,
                    chat_type: "supergroup".to_string(),
                    title: Some("Second Ops".to_string()),
                    username: None,
                    first_name: None,
                },
                from: Some(TelegramUser {
                    id: 10,
                    is_bot: Some(false),
                    first_name: Some("Rei".to_string()),
                    last_name: None,
                    username: Some("rei".to_string()),
                }),
                sender_chat: None,
                text: Some("hello second room".to_string()),
                caption: None,
                entities: Vec::new(),
                caption_entities: Vec::new(),
                message_thread_id: None,
                reply_to_message: None,
                ..Default::default()
            }),
            channel_post: None,
        };

        let event = driver.normalize_update(update).expect("normalized event");
        assert_eq!(event.conversation.room_id.as_deref(), Some("-10099999"));
        assert_eq!(event.conversation.thread_id, "-10099999");
    }

    #[test]
    fn normalize_mentions_only_requires_explicit_bot_mention_in_groups() {
        let mut config = config();
        config.respond_mode = TelegramRespondMode::Mentions;
        let (_tx, rx) = watch::channel(false);
        let mut driver =
            TelegramChannelDriver::from_config("telegram-runtime", config, rx).unwrap();
        driver.bot_identity = Some(TelegramBotIdentity {
            id: 42,
            username: Some("turin_bot".to_string()),
        });

        let update_without_mention = TelegramUpdate {
            update_id: 5,
            message: Some(TelegramMessage {
                message_id: 103,
                chat: TelegramChat {
                    id: -10012345,
                    chat_type: "supergroup".to_string(),
                    title: Some("Ops".to_string()),
                    username: None,
                    first_name: None,
                },
                from: Some(TelegramUser {
                    id: 11,
                    is_bot: Some(false),
                    first_name: Some("Nora".to_string()),
                    last_name: None,
                    username: Some("nora".to_string()),
                }),
                sender_chat: None,
                text: Some("hello there".to_string()),
                caption: None,
                entities: Vec::new(),
                caption_entities: Vec::new(),
                message_thread_id: None,
                reply_to_message: None,
                ..Default::default()
            }),
            channel_post: None,
        };
        assert!(driver.normalize_update(update_without_mention).is_none());

        let update_with_mention = TelegramUpdate {
            update_id: 6,
            message: Some(TelegramMessage {
                message_id: 104,
                chat: TelegramChat {
                    id: -10012345,
                    chat_type: "supergroup".to_string(),
                    title: Some("Ops".to_string()),
                    username: None,
                    first_name: None,
                },
                from: Some(TelegramUser {
                    id: 11,
                    is_bot: Some(false),
                    first_name: Some("Nora".to_string()),
                    last_name: None,
                    username: Some("nora".to_string()),
                }),
                sender_chat: None,
                text: Some("@turin_bot hello there".to_string()),
                caption: None,
                entities: vec![TelegramMessageEntity {
                    kind: "mention".to_string(),
                    offset: 0,
                    length: 10,
                    user: None,
                }],
                caption_entities: Vec::new(),
                message_thread_id: None,
                reply_to_message: None,
                ..Default::default()
            }),
            channel_post: None,
        };
        assert!(driver.normalize_update(update_with_mention).is_some());
    }

    #[test]
    fn normalize_replies_mode_accepts_replies_to_the_bot() {
        let mut config = config();
        config.respond_mode = TelegramRespondMode::Replies;
        let (_tx, rx) = watch::channel(false);
        let mut driver =
            TelegramChannelDriver::from_config("telegram-runtime", config, rx).unwrap();
        driver.bot_identity = Some(TelegramBotIdentity {
            id: 42,
            username: Some("turin_bot".to_string()),
        });

        let update = TelegramUpdate {
            update_id: 7,
            message: Some(TelegramMessage {
                message_id: 105,
                chat: TelegramChat {
                    id: -10012345,
                    chat_type: "supergroup".to_string(),
                    title: Some("Ops".to_string()),
                    username: None,
                    first_name: None,
                },
                from: Some(TelegramUser {
                    id: 12,
                    is_bot: Some(false),
                    first_name: Some("Ira".to_string()),
                    last_name: None,
                    username: Some("ira".to_string()),
                }),
                sender_chat: None,
                text: Some("following up".to_string()),
                caption: None,
                entities: Vec::new(),
                caption_entities: Vec::new(),
                message_thread_id: None,
                reply_to_message: Some(Box::new(TelegramMessage {
                    message_id: 1000,
                    chat: TelegramChat {
                        id: -10012345,
                        chat_type: "supergroup".to_string(),
                        title: Some("Ops".to_string()),
                        username: None,
                        first_name: None,
                    },
                    from: Some(TelegramUser {
                        id: 42,
                        is_bot: Some(true),
                        first_name: Some("Turin".to_string()),
                        last_name: None,
                        username: Some("turin_bot".to_string()),
                    }),
                    sender_chat: None,
                    text: Some("prior answer".to_string()),
                    caption: None,
                    entities: Vec::new(),
                    caption_entities: Vec::new(),
                    message_thread_id: None,
                    reply_to_message: None,
                    ..Default::default()
                })),
                ..Default::default()
            }),
            channel_post: None,
        };

        assert!(driver.normalize_update(update).is_some());
    }

    #[test]
    fn normalize_mentions_or_replies_accepts_addressed_bot_commands_in_groups() {
        let mut config = config();
        config.respond_mode = TelegramRespondMode::MentionsOrReplies;
        let (_tx, rx) = watch::channel(false);
        let mut driver =
            TelegramChannelDriver::from_config("telegram-runtime", config, rx).unwrap();
        driver.bot_identity = Some(TelegramBotIdentity {
            id: 42,
            username: Some("turin_bot".to_string()),
        });

        let update = TelegramUpdate {
            update_id: 8,
            message: Some(TelegramMessage {
                message_id: 106,
                chat: TelegramChat {
                    id: -10012345,
                    chat_type: "supergroup".to_string(),
                    title: Some("Ops".to_string()),
                    username: None,
                    first_name: None,
                },
                from: Some(TelegramUser {
                    id: 13,
                    is_bot: Some(false),
                    first_name: Some("Rin".to_string()),
                    last_name: None,
                    username: Some("rin".to_string()),
                }),
                sender_chat: None,
                text: Some("/start@turin_bot".to_string()),
                caption: None,
                entities: vec![TelegramMessageEntity {
                    kind: "bot_command".to_string(),
                    offset: 0,
                    length: 16,
                    user: None,
                }],
                caption_entities: Vec::new(),
                message_thread_id: None,
                reply_to_message: None,
                ..Default::default()
            }),
            channel_post: None,
        };

        assert!(driver.normalize_update(update).is_some());
    }

    #[test]
    fn adapter_manifest_exposes_telegram_enum_settings() {
        let manifest = adapter_manifest();
        assert_eq!(manifest.kind, "telegram");
        manifest.validate().expect("valid manifest");
        assert_eq!(
            manifest
                .enum_setting("session_scope")
                .expect("session scope setting")
                .options,
            vec!["user", "thread", "room"]
        );
        assert_eq!(
            manifest
                .enum_setting("respond_mode")
                .expect("respond mode setting")
                .options,
            vec!["all", "mentions", "replies", "mentions_or_replies"]
        );
    }

    #[test]
    fn enrich_outbound_defaults_to_replying_to_source_message() {
        let driver = driver();
        let event = sample_event_with_message_id(42);
        let enriched = driver.enrich_outbound_for_event(&event, OutboundMessage::text("reply"));
        assert_eq!(enriched.metadata["telegram_reply_to_message_id"], 42);
    }

    #[test]
    fn enrich_outbound_keeps_explicit_reply_override() {
        let driver = driver();
        let event = sample_event_with_message_id(42);
        let mut outbound = OutboundMessage::text("reply");
        outbound.metadata.insert(
            "telegram_reply_to_message_id".to_string(),
            serde_json::json!(7),
        );
        let enriched = driver.enrich_outbound_for_event(&event, outbound);
        assert_eq!(enriched.metadata["telegram_reply_to_message_id"], 7);
    }

    #[test]
    fn enrich_outbound_allows_clearing_default_reply_target() {
        let driver = driver();
        let event = sample_event_with_message_id(42);
        let mut outbound = OutboundMessage::text("reply");
        outbound.metadata.insert(
            "telegram_reply_to_message_id".to_string(),
            serde_json::Value::Null,
        );
        let enriched = driver.enrich_outbound_for_event(&event, outbound);
        assert_eq!(
            enriched.metadata.get("telegram_reply_to_message_id"),
            Some(&serde_json::Value::Null)
        );
    }

    #[test]
    fn outbound_batches_split_long_messages_and_keep_thread() {
        let long_text = "x".repeat(TELEGRAM_MESSAGE_MAX_LEN + 200);
        let payloads = telegram_batches_from_message(
            "-10012345",
            Some(555),
            &OutboundMessage {
                blocks: vec![MessageBlock::Text { text: long_text }],
                ..OutboundMessage::default()
            },
        )
        .expect("render telegram payloads");

        assert!(payloads.len() >= 2);
        assert!(payloads.iter().all(|payload| {
            payload["text"]
                .as_str()
                .map(|text| text.chars().count() <= TELEGRAM_MESSAGE_MAX_LEN)
                .unwrap_or(false)
        }));
        assert!(
            payloads
                .iter()
                .all(|payload| payload["message_thread_id"] == 555)
        );
    }

    #[test]
    fn code_blocks_render_as_html_with_parse_mode() {
        let payloads = telegram_batches_from_message(
            "-10012345",
            None,
            &OutboundMessage {
                blocks: vec![MessageBlock::CodeBlock {
                    language: Some("rust".to_string()),
                    code: "fn main() { println!(\"hi\"); }".to_string(),
                }],
                ..OutboundMessage::default()
            },
        )
        .expect("render telegram payloads");

        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0]["parse_mode"], "HTML");
        assert!(
            payloads[0]["text"]
                .as_str()
                .is_some_and(|text| text.contains("<pre>") && text.contains("fn main()")),
            "payload should render Telegram HTML code block: {}",
            payloads[0]
        );
    }

    #[test]
    fn text_messages_render_markdown_as_html_by_default() {
        let payloads = telegram_batches_from_message(
            "-10012345",
            None,
            &OutboundMessage::text(
                "# Heading\n\n**bold** and `code`\n\n- first\n- second\n\n[site](https://example.com)",
            ),
        )
        .expect("render telegram payloads");

        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0]["parse_mode"], "HTML");
        let text = payloads[0]["text"]
            .as_str()
            .expect("telegram text should be a string");
        assert!(text.contains("<b>Heading</b>"), "payload text: {text}");
        assert!(text.contains("<b>bold</b>"), "payload text: {text}");
        assert!(text.contains("<code>code</code>"), "payload text: {text}");
        assert!(text.contains("• first"), "payload text: {text}");
        assert!(
            text.contains("<a href=\"https://example.com\">site</a>"),
            "payload text: {text}"
        );
    }

    #[test]
    fn markdown_tables_render_as_preformatted_blocks() {
        let payloads = telegram_batches_from_message(
            "-10012345",
            None,
            &OutboundMessage::text("| Name | Value |\n| --- | --- |\n| alpha | 1 |\n| beta | 22 |"),
        )
        .expect("render telegram payloads");

        assert_eq!(payloads.len(), 1);
        assert_eq!(payloads[0]["parse_mode"], "HTML");
        let text = payloads[0]["text"]
            .as_str()
            .expect("telegram text should be a string");
        assert!(text.contains("<pre>"), "payload text: {text}");
        assert!(text.contains("| Name  | Value |"), "payload text: {text}");
        assert!(text.contains("| alpha | 1     |"), "payload text: {text}");
        assert!(text.contains("| beta  | 22    |"), "payload text: {text}");
    }

    #[test]
    fn stream_preview_can_include_thinking_sections() {
        let preview = render_stream_preview("Partial answer", Some("Reasoning step"));
        assert!(preview.contains("Thinking…"));
        assert!(preview.contains("Reasoning step"));
        assert!(preview.contains("Reply"));
        assert!(preview.contains("Partial answer"));
    }

    #[test]
    fn stream_preview_returns_text_only_when_no_thinking_is_present() {
        let preview = render_stream_preview("Partial answer", None);
        assert_eq!(preview, "Partial answer");
    }

    #[test]
    fn final_message_can_include_persisted_thinking() {
        let mut message = OutboundMessage::text("Final answer");
        message.metadata.insert(
            "channel_final_thinking".to_string(),
            serde_json::json!("Step 1\nStep 2"),
        );

        let payloads = telegram_batches_from_message("-10012345", None, &message)
            .expect("render telegram payloads");
        let text = payloads[0]["text"].as_str().expect("telegram text payload");
        assert!(text.contains("Thinking"));
        assert!(text.contains("<pre>"));
        assert!(text.contains("Step 1"));
        assert!(text.contains("Reply"));
        assert!(text.contains("Final answer"));
    }

    #[test]
    fn telegram_api_error_recognizes_not_modified_edit_failures() {
        let error = TelegramApiError {
            code: "telegram_edit_message_failed".to_string(),
            message: "Telegram editMessageText request failed with 400: Bad Request: message is not modified: specified new message content and reply markup are exactly the same as a current content and reply markup of the message".to_string(),
            retriable: false,
            retry_after: None,
        };
        assert!(error.is_message_not_modified());
    }

    #[test]
    fn config_supports_chat_lists_and_telegram_stream_settings() {
        unsafe {
            std::env::set_var("TELEGRAM_BOT_TOKEN", "token");
        }
        let config = TelegramChannelDriverConfig::from_settings(
            &serde_json::json!({
                "token_env": "TELEGRAM_BOT_TOKEN",
                "chat_ids": [498502840, -10012345],
                "respond_mode": "mentions_or_replies",
                "stream_mode": "block",
                "stream_thinking": true,
                "persist_thinking": true
            }),
            false,
        )
        .expect("telegram config should parse");

        assert_eq!(config.chat_ids, vec!["498502840", "-10012345"]);
        assert_eq!(config.respond_mode, TelegramRespondMode::MentionsOrReplies);
        assert_eq!(config.session_scope_dm, None);
        assert_eq!(config.session_scope_group, None);
        assert_eq!(config.session_scope_channel, None);
        assert_eq!(
            config.max_inbound_text_chars,
            DEFAULT_MAX_INBOUND_TEXT_CHARS
        );
        assert!(config.stream_thinking);
        assert!(config.persist_thinking);
    }

    #[test]
    fn private_chats_can_override_session_scope() {
        let config = TelegramChannelDriverConfig {
            base_url: DEFAULT_BASE_URL.to_string(),
            workspace_id: "telegram".to_string(),
            chat_ids: vec!["498502840".to_string()],
            accept_all_chats: false,
            token: "token".to_string(),
            poll_timeout_seconds: 30,
            poll_interval: Duration::from_millis(250),
            max_updates_per_poll: 25,
            max_inbound_text_chars: DEFAULT_MAX_INBOUND_TEXT_CHARS,
            start_from_latest: true,
            ignore_bot_messages: true,
            respond_mode: TelegramRespondMode::MentionsOrReplies,
            session_scope: ChannelSessionScope::User,
            session_scope_dm: Some(ChannelSessionScope::Room),
            session_scope_group: None,
            session_scope_channel: None,
            stream_mode: ChannelStreamMode::Off,
            stream_thinking: false,
            persist_thinking: false,
        };
        let chat = TelegramChat {
            id: 498502840,
            title: None,
            username: Some("jthum".to_string()),
            first_name: Some("Jay".to_string()),
            chat_type: "private".to_string(),
        };

        assert_eq!(
            effective_telegram_session_scope(&config, &chat),
            ChannelSessionScope::Room
        );
    }

    #[test]
    fn validate_settings_rejects_deprecated_session_scope_aliases() {
        let error = validate_settings(
            &serde_json::json!({
                "token_env": "TELEGRAM_BOT_TOKEN",
                "chat_id": "498502840",
                "dm_session_scope": "room"
            }),
            false,
        )
        .expect_err("deprecated alias should fail");

        assert!(error.to_string().contains("session_scope_dm"));
    }

    #[test]
    fn validate_settings_does_not_require_live_token_env() {
        validate_settings(
            &serde_json::json!({
                "token_env": "TELEGRAM_TOKEN_NOT_SET_FOR_VALIDATION",
                "chat_ids": [498502840],
                "respond_mode": "mentions_or_replies"
            }),
            false,
        )
        .expect("settings validation should not require the token env var to exist");
    }

    #[test]
    fn validate_settings_allows_missing_chat_ids_when_unconfigured_chats_are_enabled() {
        validate_settings(
            &serde_json::json!({
                "token_env": "TELEGRAM_TOKEN_NOT_SET_FOR_VALIDATION",
                "respond_mode": "mentions_or_replies"
            }),
            true,
        )
        .expect("discovery mode should allow telegram channels without explicit chat ids");
    }

    #[test]
    fn validate_settings_rejects_invalid_session_scope() {
        let error = validate_settings(
            &serde_json::json!({
                "token_env": "TELEGRAM_TOKEN_NOT_SET_FOR_VALIDATION",
                "chat_ids": [498502840],
                "session_scope": "guild"
            }),
            false,
        )
        .expect_err("invalid session scope rejected");
        assert!(error.to_string().contains("session_scope"));
    }

    #[test]
    fn metadata_can_set_reply_target_and_disable_notification() {
        let mut message = OutboundMessage::text("hello");
        message.metadata.insert(
            "telegram_reply_to_message_id".to_string(),
            serde_json::json!(77),
        );
        message.metadata.insert(
            "telegram_disable_notification".to_string(),
            serde_json::json!(true),
        );

        let payloads = telegram_batches_from_message("-10012345", None, &message)
            .expect("render telegram payloads");
        assert_eq!(payloads[0]["reply_to_message_id"], 77);
        assert_eq!(payloads[0]["allow_sending_without_reply"], true);
        assert_eq!(payloads[0]["disable_notification"], true);
    }

    #[test]
    fn telegram_format_plain_disables_html_rendering_for_code_blocks() {
        let mut message = OutboundMessage {
            blocks: vec![MessageBlock::CodeBlock {
                language: Some("rust".to_string()),
                code: "fn main() {}".to_string(),
            }],
            ..OutboundMessage::default()
        };
        message
            .metadata
            .insert("telegram_format".to_string(), serde_json::json!("plain"));

        let payloads = telegram_batches_from_message("-10012345", None, &message)
            .expect("render telegram payloads");

        assert_eq!(payloads.len(), 1);
        assert!(payloads[0].get("parse_mode").is_none());
        assert_eq!(payloads[0]["text"], "```rust\nfn main() {}\n```");
    }
}
