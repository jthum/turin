use std::fs;
use std::io::Cursor;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use tokio::sync::{mpsc, watch};
use tracing::{info, warn};
use turin_channel_core::{
    ChannelAdapterManifest, ChannelAttachment, ChannelAuthFlow, ChannelAuthFlowDisplay,
    ChannelAuthFlowKind, ChannelAuthFlowPollRequest, ChannelAuthFlowPollResponse,
    ChannelAuthFlowResolvedValue, ChannelAuthFlowStartRequest, ChannelAuthFlowStartResponse,
    ChannelCapabilities, ChannelConfigField, ChannelConfigFieldOption, ChannelConfigTarget,
    ChannelConfigTargetKind, ChannelConversationKey, ChannelFieldVisibilityRule,
    ChannelIdentitySelectors, ChannelInstallManifest, ChannelKind, ChannelMessageRef,
    ChannelRuntimeCapabilities, ChannelRuntimeManifest, ChannelSessionScope, ChannelSetupManifest,
    ChannelUser, DEFAULT_MAX_INBOUND_TEXT_CHARS, InboundEvent, OutboundMessage, bound_inbound_text,
};
use turin_channel_runner::ChannelDriver;
use uuid::Uuid;
use whatsapp_rust::Jid;
use whatsapp_rust::TokioRuntime;
use whatsapp_rust::bot::{Bot, BotHandle};
use whatsapp_rust::download::{Downloadable, MediaType};
use whatsapp_rust::pair_code::PairCodeOptions;
use whatsapp_rust::proto_helpers::MessageExt;
use whatsapp_rust::store::SqliteStore;
use whatsapp_rust::types::events::Event;
use whatsapp_rust::types::message::MessageInfo;
use whatsapp_rust::waproto::whatsapp as wa;
use whatsapp_rust_tokio_transport::TokioWebSocketTransportFactory;
use whatsapp_rust_ureq_http_client::UreqHttpClient;

const DEFAULT_WORKSPACE_ID: &str = "whatsapp";
const DEFAULT_AUTH_FLOW_ID: &str = "pair";
const DEFAULT_AUTH_POLL_INTERVAL_SECS: u64 = 3;
const DEFAULT_AUTH_TIMEOUT_SECS: u64 = 300;
const DEFAULT_RUNTIME_STORE_BASENAME: &str = "whatsapp-session.db";
const DEFAULT_PERSONAL_TRIGGER_PREFIX: &str = "/turin";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WhatsAppAccountMode {
    Personal,
    Dedicated,
}

#[derive(Debug, Clone)]
pub struct WhatsAppChannelDriverConfig {
    pub workspace_id: String,
    account_mode: WhatsAppAccountMode,
    pub session_scope: ChannelSessionScope,
    pub session_store_path: PathBuf,
    media_dir: PathBuf,
    max_inbound_text_chars: usize,
    trigger_prefix: Option<String>,
    allowed_chats: Vec<String>,
    banned_chats: Vec<String>,
}

pub struct WhatsAppChannelDriver {
    config: WhatsAppChannelDriverConfig,
    shutdown_rx: watch::Receiver<bool>,
    client: Arc<whatsapp_rust::Client>,
    bot_handle: BotHandle,
    event_rx: mpsc::UnboundedReceiver<DriverEvent>,
}

enum DriverEvent {
    Message(Box<wa::Message>, Box<MessageInfo>),
    LoggedOut(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WhatsAppAuthSession {
    ticket: String,
    state_path: PathBuf,
    store_path: PathBuf,
    phone_number: Option<String>,
    custom_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum WhatsAppAuthPhase {
    Pending,
    Complete,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct WhatsAppAuthState {
    phase: WhatsAppAuthPhase,
    display: ChannelAuthFlowDisplay,
    message: Option<String>,
}

#[derive(Clone)]
struct AuthStateWriter {
    path: PathBuf,
    lock: Arc<Mutex<()>>,
}

impl AuthStateWriter {
    fn new(path: PathBuf) -> Self {
        Self {
            path,
            lock: Arc::new(Mutex::new(())),
        }
    }

    fn load(&self) -> Result<WhatsAppAuthState> {
        let bytes = fs::read(&self.path)
            .with_context(|| format!("Failed to read auth flow state '{}'", self.path.display()))?;
        serde_json::from_slice(&bytes).context("Failed to decode WhatsApp auth flow state")
    }

    fn write(&self, state: &WhatsAppAuthState) -> Result<()> {
        let _guard = self
            .lock
            .lock()
            .map_err(|_| anyhow!("WhatsApp auth flow state lock was poisoned"))?;
        let body = serde_json::to_vec_pretty(state).context("Failed to encode auth flow state")?;
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "Failed to create auth flow state directory '{}'",
                    parent.display()
                )
            })?;
        }
        fs::write(&self.path, body)
            .with_context(|| format!("Failed to write auth flow state '{}'", self.path.display()))
    }
}

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
                    help: Some("Defaults to 'whatsapp' and is usually fine to leave alone.".to_string()),
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
                    prompt: Some(
                        "How should WhatsApp conversation memory be scoped?".to_string(),
                    ),
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
                        "Personal mode defaults to '/turin'. Dedicated accounts can usually leave this empty.".to_string(),
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
                        "Optional phone numbers or JIDs that should always be denied"
                            .to_string(),
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
                        "Optional custom 8-character pairing code for headless linking"
                            .to_string(),
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

pub fn validate_settings(settings: &Value, _allow_unconfigured_chats: bool) -> Result<()> {
    parse_settings(settings, None).map(|_| ())
}

pub fn start_auth_flow(
    request: &ChannelAuthFlowStartRequest,
) -> Result<ChannelAuthFlowStartResponse> {
    if request.flow_id != DEFAULT_AUTH_FLOW_ID {
        bail!("Unsupported WhatsApp auth flow '{}'", request.flow_id);
    }

    let settings = settings_object(&request.current_settings)?;
    let store_path = resolve_auth_store_path(settings)?;
    let phone_number = optional_nonempty_string(settings, "pair_code_phone_number")?;
    let custom_code = optional_nonempty_string(settings, "pair_code_custom_code")?;
    validate_pair_code_fields(phone_number.as_deref(), custom_code.as_deref())?;

    let session = WhatsAppAuthSession {
        ticket: Uuid::new_v4().to_string(),
        state_path: std::env::temp_dir()
            .join("turin-whatsapp-auth")
            .join(Uuid::new_v4().to_string())
            .join("state.json"),
        store_path,
        phone_number,
        custom_code,
    };
    let writer = AuthStateWriter::new(session.state_path.clone());
    writer.write(&WhatsAppAuthState {
        phase: WhatsAppAuthPhase::Pending,
        display: ChannelAuthFlowDisplay {
            message: Some("Starting WhatsApp pairing...".to_string()),
            poll_interval_secs: Some(DEFAULT_AUTH_POLL_INTERVAL_SECS),
            ..ChannelAuthFlowDisplay::default()
        },
        message: None,
    })?;
    spawn_auth_flow_worker(&session)?;

    let display = writer
        .load()
        .map(|state| state.display)
        .unwrap_or(ChannelAuthFlowDisplay {
            message: Some("Starting WhatsApp pairing...".to_string()),
            poll_interval_secs: Some(DEFAULT_AUTH_POLL_INTERVAL_SECS),
            ..ChannelAuthFlowDisplay::default()
        });

    Ok(ChannelAuthFlowStartResponse {
        session: serde_json::to_value(session).context("Failed to encode auth flow session")?,
        display,
    })
}

pub fn poll_auth_flow(request: &ChannelAuthFlowPollRequest) -> Result<ChannelAuthFlowPollResponse> {
    if request.flow_id != DEFAULT_AUTH_FLOW_ID {
        bail!("Unsupported WhatsApp auth flow '{}'", request.flow_id);
    }

    let session: WhatsAppAuthSession = serde_json::from_value(request.session.clone())
        .context("Failed to decode WhatsApp auth flow session")?;
    let writer = AuthStateWriter::new(session.state_path.clone());
    let state = writer.load()?;

    Ok(match state.phase {
        WhatsAppAuthPhase::Pending => ChannelAuthFlowPollResponse::Pending {
            display: state.display,
        },
        WhatsAppAuthPhase::Complete => ChannelAuthFlowPollResponse::Complete {
            values: vec![
                ChannelAuthFlowResolvedValue {
                    target: ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "session_store_path".to_string(),
                    },
                    value: Value::String(session.store_path.display().to_string()),
                },
                ChannelAuthFlowResolvedValue {
                    target: ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "pair_code_phone_number".to_string(),
                    },
                    value: Value::Null,
                },
                ChannelAuthFlowResolvedValue {
                    target: ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "pair_code_custom_code".to_string(),
                    },
                    value: Value::Null,
                },
            ],
            message: state.message,
        },
        WhatsAppAuthPhase::Failed => ChannelAuthFlowPollResponse::Failed {
            message: state
                .message
                .unwrap_or_else(|| "WhatsApp pairing failed".to_string()),
        },
    })
}

pub async fn run_auth_flow_worker(session_json: &str) -> Result<()> {
    let session: WhatsAppAuthSession =
        serde_json::from_str(session_json).context("Failed to parse auth flow worker session")?;
    let writer = AuthStateWriter::new(session.state_path.clone());
    let (client, mut bot) = build_bot(
        &session.store_path,
        session.phone_number.clone(),
        session.custom_code.clone(),
        Some(writer.clone()),
        None,
    )
    .await
    .with_context(|| {
        format!(
            "Failed to initialize WhatsApp pairing worker using store '{}'",
            session.store_path.display()
        )
    })?;
    let mut bot_handle = bot.run().await.context("Failed to start WhatsApp bot")?;
    let deadline = Instant::now() + Duration::from_secs(DEFAULT_AUTH_TIMEOUT_SECS);

    loop {
        if client.is_logged_in() {
            writer.write(&WhatsAppAuthState {
                phase: WhatsAppAuthPhase::Complete,
                display: ChannelAuthFlowDisplay {
                    message: Some("WhatsApp pairing complete.".to_string()),
                    poll_interval_secs: Some(DEFAULT_AUTH_POLL_INTERVAL_SECS),
                    ..ChannelAuthFlowDisplay::default()
                },
                message: Some(format!(
                    "WhatsApp pairing complete. Session store: {}",
                    session.store_path.display()
                )),
            })?;
            break;
        }

        if Instant::now() >= deadline {
            writer.write(&WhatsAppAuthState {
                phase: WhatsAppAuthPhase::Failed,
                display: ChannelAuthFlowDisplay::default(),
                message: Some("WhatsApp pairing timed out before the session linked.".to_string()),
            })?;
            break;
        }

        tokio::select! {
            result = &mut bot_handle => {
                if let Err(err) = result {
                    writer.write(&WhatsAppAuthState {
                        phase: WhatsAppAuthPhase::Failed,
                        display: ChannelAuthFlowDisplay::default(),
                        message: Some(format!("WhatsApp pairing worker was cancelled: {err}")),
                    })?;
                } else if !client.is_logged_in() {
                    writer.write(&WhatsAppAuthState {
                        phase: WhatsAppAuthPhase::Failed,
                        display: ChannelAuthFlowDisplay::default(),
                        message: Some("WhatsApp pairing worker exited before pairing completed.".to_string()),
                    })?;
                }
                break;
            }
            _ = tokio::time::sleep(Duration::from_secs(1)) => {}
        }
    }

    client.disconnect().await;
    bot_handle.abort();
    Ok(())
}

impl WhatsAppChannelDriver {
    pub async fn from_settings(
        _channel_id: &str,
        settings: &Value,
        runtime_dir: &Path,
        shutdown_rx: watch::Receiver<bool>,
        _allow_unconfigured_chats: bool,
    ) -> Result<Self> {
        let config = parse_settings(settings, Some(runtime_dir))?;
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let (runtime_client, mut bot) = build_bot(
            &config.session_store_path,
            None,
            None,
            None,
            Some(event_tx.clone()),
        )
        .await
        .with_context(|| {
            format!(
                "Failed to initialize WhatsApp runtime session store '{}'",
                config.session_store_path.display()
            )
        })?;

        let bot_handle = bot
            .run()
            .await
            .context("Failed to start WhatsApp runtime bot")?;

        Ok(Self {
            config,
            shutdown_rx,
            client: runtime_client,
            bot_handle,
            event_rx,
        })
    }

    async fn message_to_event(
        &self,
        message: Box<wa::Message>,
        info: MessageInfo,
    ) -> Result<Option<InboundEvent>> {
        if info.source.is_from_me || info.source.chat.to_string() == "status@broadcast" {
            return Ok(None);
        }

        let chat_id = info.source.chat.to_string();
        if !chat_is_allowed(
            &chat_id,
            &self.config.allowed_chats,
            &self.config.banned_chats,
        ) {
            return Ok(None);
        }

        let base_message = message.get_base_message();
        let attachments: Vec<ChannelAttachment> = self
            .collect_inbound_attachments(base_message, &info.id)
            .await?;
        let raw_text = message.text_content().or_else(|| message.get_caption());
        let text = match raw_text {
            Some(text) => match inbound_text(
                text,
                self.config.account_mode,
                self.config.trigger_prefix.as_deref(),
            ) {
                Some(value) => value,
                None => return Ok(None),
            },
            None if attachments.is_empty() => return Ok(None),
            None if matches!(self.config.account_mode, WhatsAppAccountMode::Personal)
                && self.config.trigger_prefix.is_some() =>
            {
                return Ok(None);
            }
            None => String::new(),
        };
        let sender_id = info.source.sender.to_string();
        let thread_id = match self.config.session_scope {
            ChannelSessionScope::User if info.source.is_group => {
                format!("room:{chat_id}:user:{sender_id}")
            }
            ChannelSessionScope::User => format!("user:{sender_id}"),
            ChannelSessionScope::Thread => format!("room:{chat_id}:user:{sender_id}"),
            ChannelSessionScope::Room => format!("room:{chat_id}"),
        };

        let conversation = ChannelConversationKey {
            channel: ChannelKind::new("whatsapp"),
            workspace_id: self.config.workspace_id.clone(),
            room_id: Some(chat_id.clone()),
            thread_id,
            user_id: Some(sender_id.clone()),
        };

        let mut metadata = Map::new();
        metadata.insert("chat_jid".to_string(), Value::String(chat_id.clone()));
        metadata.insert("sender_jid".to_string(), Value::String(sender_id.clone()));
        metadata.insert("is_group".to_string(), Value::Bool(info.source.is_group));
        let text = bound_inbound_text(text, &mut metadata, self.config.max_inbound_text_chars);

        Ok(Some(InboundEvent {
            message: ChannelMessageRef {
                conversation: conversation.clone(),
                message_id: info.id,
            },
            conversation,
            user: ChannelUser {
                id: sender_id,
                display_name: None,
                username: None,
            },
            session_scope: self.config.session_scope,
            text,
            attachments,
            metadata,
        }))
    }

    async fn collect_inbound_attachments(
        &self,
        message: &wa::Message,
        message_id: &str,
    ) -> Result<Vec<ChannelAttachment>> {
        fs::create_dir_all(&self.config.media_dir).with_context(|| {
            format!(
                "Failed to create WhatsApp media directory '{}'",
                self.config.media_dir.display()
            )
        })?;

        let mut attachments = Vec::new();
        if let Some(image) = &message.image_message {
            attachments.push(
                self.download_whatsapp_attachment(
                    &**image,
                    message_id,
                    image.mimetype.clone(),
                    image_name(image, message_id),
                )
                .await?,
            );
        }
        if let Some(document) = &message.document_message {
            attachments.push(
                self.download_whatsapp_attachment(
                    &**document,
                    message_id,
                    document.mimetype.clone(),
                    document_name(document, message_id),
                )
                .await?,
            );
        }
        if let Some(video) = &message.video_message {
            attachments.push(
                self.download_whatsapp_attachment(
                    &**video,
                    message_id,
                    video.mimetype.clone(),
                    format!("video-{message_id}.mp4"),
                )
                .await?,
            );
        }
        if let Some(audio) = &message.audio_message {
            attachments.push(
                self.download_whatsapp_attachment(
                    &**audio,
                    message_id,
                    audio.mimetype.clone(),
                    format!("audio-{message_id}.ogg"),
                )
                .await?,
            );
        }
        Ok(attachments)
    }

    async fn download_whatsapp_attachment<D: Downloadable>(
        &self,
        media: &D,
        message_id: &str,
        content_type: Option<String>,
        suggested_name: String,
    ) -> Result<ChannelAttachment> {
        let mut data = Cursor::new(Vec::new());
        self.client
            .download_to_file(media, &mut data)
            .await
            .context("Failed to download WhatsApp media attachment")?;
        let target_path = self.config.media_dir.join(format!(
            "{}-{}",
            Uuid::new_v4(),
            sanitize_component(&suggested_name)
        ));
        fs::write(&target_path, data.into_inner()).with_context(|| {
            format!(
                "Failed to write WhatsApp media attachment '{}'",
                target_path.display()
            )
        })?;
        let final_name = if Path::new(&suggested_name).extension().is_some() {
            suggested_name
        } else {
            infer_media_name(message_id, content_type.as_deref(), &suggested_name)
        };
        Ok(ChannelAttachment {
            name: final_name,
            content_type,
            url: None,
            local_path: Some(target_path.display().to_string()),
        })
    }
}

impl WhatsAppChannelDriver {
    async fn send_attachment(
        &self,
        chat: Jid,
        attachment: &turin_channel_core::ChannelAttachment,
    ) -> Result<()> {
        let local_path = attachment.local_path.as_deref().ok_or_else(|| {
            anyhow!(
                "[whatsapp_send_missing_attachment_source] attachment '{}' is missing local_path",
                attachment.name
            )
        })?;
        let bytes = fs::read(local_path)
            .with_context(|| format!("Failed to read WhatsApp attachment '{}'", local_path))?;
        let media_type = whatsapp_media_type(attachment.content_type.as_deref());
        let upload = self
            .client
            .upload(bytes, media_type)
            .await
            .context("Failed to upload WhatsApp attachment")?;
        let mime_type = attachment
            .content_type
            .clone()
            .or_else(|| whatsapp_default_mime_type(media_type).map(str::to_string));
        let message = match media_type {
            MediaType::Image => wa::Message {
                image_message: Some(Box::new(wa::message::ImageMessage {
                    mimetype: mime_type,
                    url: Some(upload.url),
                    direct_path: Some(upload.direct_path),
                    media_key: Some(upload.media_key),
                    file_enc_sha256: Some(upload.file_enc_sha256),
                    file_sha256: Some(upload.file_sha256),
                    file_length: Some(upload.file_length),
                    ..Default::default()
                })),
                ..Default::default()
            },
            _ => wa::Message {
                document_message: Some(Box::new(wa::message::DocumentMessage {
                    mimetype: mime_type,
                    title: Some(attachment.name.clone()),
                    file_name: Some(attachment.name.clone()),
                    url: Some(upload.url),
                    direct_path: Some(upload.direct_path),
                    media_key: Some(upload.media_key),
                    file_enc_sha256: Some(upload.file_enc_sha256),
                    file_sha256: Some(upload.file_sha256),
                    file_length: Some(upload.file_length),
                    ..Default::default()
                })),
                ..Default::default()
            },
        };
        self.client
            .send_message(chat, message)
            .await
            .context("Failed to send WhatsApp attachment")?;
        Ok(())
    }
}

#[async_trait]
impl ChannelDriver for WhatsAppChannelDriver {
    fn kind(&self) -> ChannelKind {
        ChannelKind::new("whatsapp")
    }

    fn user_matches_selector(&self, selector: &str, user: &ChannelUser) -> bool {
        let selector = selector.trim().trim_start_matches('@');
        if selector.is_empty() {
            return false;
        }
        user.id.eq_ignore_ascii_case(selector)
            || user
                .id
                .split('@')
                .next()
                .is_some_and(|phone| phone.eq_ignore_ascii_case(selector))
            || user
                .username
                .as_ref()
                .is_some_and(|username| username.eq_ignore_ascii_case(selector))
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities {
            rich_formatting: false,
            threads: false,
            attachments: true,
            ephemeral_messages: false,
        }
    }

    async fn next_event(&mut self) -> Result<Option<InboundEvent>> {
        loop {
            tokio::select! {
                changed = self.shutdown_rx.changed() => {
                    if changed.is_err() || *self.shutdown_rx.borrow() {
                        return Ok(None);
                    }
                }
                maybe_event = self.event_rx.recv() => {
                    match maybe_event {
                        Some(DriverEvent::Message(message, info)) => {
                            if let Some(event) = self.message_to_event(message, *info).await? {
                                return Ok(Some(event));
                            }
                        }
                        Some(DriverEvent::LoggedOut(reason)) => {
                            bail!("WhatsApp linked session was logged out: {reason}");
                        }
                        None => return Ok(None),
                    }
                }
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
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                anyhow!("[whatsapp_send_missing_room] outbound conversation is missing room_id")
            })?;
        let chat: Jid = room_id
            .parse()
            .with_context(|| format!("Invalid WhatsApp chat JID '{room_id}'"))?;
        let rendered = render_whatsapp_message(&message);
        if !rendered.trim().is_empty() {
            self.client
                .send_message(
                    chat.clone(),
                    wa::Message {
                        conversation: Some(rendered),
                        ..Default::default()
                    },
                )
                .await
                .context("Failed to send WhatsApp message")?;
        }
        for attachment in &message.attachments {
            self.send_attachment(chat.clone(), attachment).await?;
        }
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<()> {
        self.client.disconnect().await;
        self.bot_handle.abort();
        Ok(())
    }
}

fn parse_settings(
    settings: &Value,
    runtime_dir: Option<&Path>,
) -> Result<WhatsAppChannelDriverConfig> {
    let map = settings_object(settings)?;
    let workspace_id = optional_nonempty_string(map, "workspace_id")?
        .unwrap_or_else(|| DEFAULT_WORKSPACE_ID.to_string());
    let account_mode = parse_account_mode(map.get("account_mode"))?;
    let session_scope = parse_session_scope(map.get("session_scope"))?;
    let trigger_prefix = optional_nonempty_string(map, "trigger_prefix")?;
    let trigger_prefix = match (account_mode, trigger_prefix) {
        (WhatsAppAccountMode::Personal, None) => Some(DEFAULT_PERSONAL_TRIGGER_PREFIX.to_string()),
        (_, value) => value,
    };
    let allowed_chats = parse_string_list(map.get("allowed_chats"), "allowed_chats")?;
    let banned_chats = parse_string_list(map.get("banned_chats"), "banned_chats")?;

    let pair_code_phone_number = optional_nonempty_string(map, "pair_code_phone_number")?;
    let pair_code_custom_code = optional_nonempty_string(map, "pair_code_custom_code")?;
    validate_pair_code_fields(
        pair_code_phone_number.as_deref(),
        pair_code_custom_code.as_deref(),
    )?;

    let session_store_path = match optional_nonempty_string(map, "session_store_path")? {
        Some(raw) => resolve_runtime_store_path(&raw, runtime_dir),
        None => runtime_dir
            .map(|dir| dir.join(DEFAULT_RUNTIME_STORE_BASENAME))
            .unwrap_or_else(|| {
                default_auth_store_path(
                    &std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
                    &workspace_id,
                )
            }),
    };

    let max_inbound_text_chars = match map.get("max_inbound_text_chars") {
        None => DEFAULT_MAX_INBOUND_TEXT_CHARS,
        Some(value) => {
            let max = value.as_u64().ok_or_else(|| {
                anyhow!(
                    "[whatsapp_config_invalid_max_inbound_text_chars] WhatsApp channel setting 'max_inbound_text_chars' must be a positive integer"
                )
            })?;
            let max = usize::try_from(max).map_err(|_| {
                anyhow!(
                    "[whatsapp_config_invalid_max_inbound_text_chars] WhatsApp channel setting 'max_inbound_text_chars' is too large"
                )
            })?;
            if max == 0 {
                bail!(
                    "[whatsapp_config_invalid_max_inbound_text_chars] WhatsApp channel setting 'max_inbound_text_chars' must be > 0"
                );
            }
            max
        }
    };

    Ok(WhatsAppChannelDriverConfig {
        workspace_id: workspace_id.clone(),
        account_mode,
        session_scope,
        session_store_path,
        media_dir: runtime_dir.map(|dir| dir.join("media")).unwrap_or_else(|| {
            std::env::temp_dir()
                .join("turin")
                .join("channels")
                .join("whatsapp")
                .join(sanitize_component(&workspace_id))
                .join("media")
        }),
        max_inbound_text_chars,
        trigger_prefix,
        allowed_chats,
        banned_chats,
    })
}

async fn build_bot(
    session_store_path: &Path,
    phone_number: Option<String>,
    custom_code: Option<String>,
    auth_state_writer: Option<AuthStateWriter>,
    driver_event_tx: Option<mpsc::UnboundedSender<DriverEvent>>,
) -> Result<(Arc<whatsapp_rust::Client>, Bot)> {
    if let Some(parent) = session_store_path.parent() {
        fs::create_dir_all(parent).with_context(|| {
            format!(
                "Failed to create WhatsApp session store directory '{}'",
                parent.display()
            )
        })?;
        tighten_path_permissions(parent, 0o700)
            .with_context(|| format!("Failed to harden permissions for '{}'", parent.display()))?;
    }

    let database_url = session_store_path.to_string_lossy().to_string();
    let backend = Arc::new(SqliteStore::new(&database_url).await.with_context(|| {
        format!(
            "Failed to open WhatsApp session store '{}'",
            session_store_path.display()
        )
    })?);
    tighten_path_permissions(session_store_path, 0o600).with_context(|| {
        format!(
            "Failed to harden permissions for WhatsApp session store '{}'",
            session_store_path.display()
        )
    })?;
    let transport_factory = TokioWebSocketTransportFactory::new();
    let http_client = UreqHttpClient::new();

    let mut builder = Bot::builder()
        .with_backend(backend)
        .with_transport_factory(transport_factory)
        .with_http_client(http_client)
        .with_runtime(TokioRuntime);

    if let Some(phone_number) = phone_number {
        builder = builder.with_pair_code(PairCodeOptions {
            phone_number,
            custom_code,
            ..Default::default()
        });
    }

    let writer_for_callback = auth_state_writer.clone();
    let driver_event_tx = driver_event_tx.clone();
    let store_path_for_display = session_store_path.display().to_string();
    let bot = builder
        .on_event(move |event, _client| {
            let writer_for_callback = writer_for_callback.clone();
            let driver_event_tx = driver_event_tx.clone();
            let store_path_for_display = store_path_for_display.clone();
            async move {
                match event {
                    Event::Message(message, info) => {
                        if let Some(driver_event_tx) = driver_event_tx.as_ref()
                            && !info.source.is_from_me
                        {
                            let _ = driver_event_tx.send(DriverEvent::Message(message, Box::new(info)));
                        }
                    }
                    Event::LoggedOut(reason) => {
                        if let Some(driver_event_tx) = driver_event_tx.as_ref() {
                            let _ = driver_event_tx.send(DriverEvent::LoggedOut(format!("{reason:?}")));
                        }
                        if let Some(writer) = writer_for_callback.as_ref() {
                            let _ = writer.write(&WhatsAppAuthState {
                                phase: WhatsAppAuthPhase::Failed,
                                display: ChannelAuthFlowDisplay::default(),
                                message: Some(format!(
                                    "WhatsApp session logged out during pairing: {reason:?}"
                                )),
                            });
                        }
                    }
                    Event::PairingQrCode { code, timeout } => {
                        if driver_event_tx.is_some() {
                            warn!(
                                expires_in_secs = timeout.as_secs(),
                                "WhatsApp runtime requested QR pairing; pair the session through setup before using the channel"
                            );
                        }
                        let Some(writer) = writer_for_callback.as_ref() else {
                            return;
                        };
                        let _ = writer.write(&WhatsAppAuthState {
                            phase: WhatsAppAuthPhase::Pending,
                            display: ChannelAuthFlowDisplay {
                                message: Some(
                                    "Scan this QR code in WhatsApp > Linked Devices.".to_string(),
                                ),
                                qr_text: Some(code),
                                expires_in_secs: Some(timeout.as_secs()),
                                poll_interval_secs: Some(DEFAULT_AUTH_POLL_INTERVAL_SECS),
                                ..ChannelAuthFlowDisplay::default()
                            },
                            message: None,
                        });
                    }
                    Event::PairingCode { code, timeout } => {
                        if driver_event_tx.is_some() {
                            warn!(
                                expires_in_secs = timeout.as_secs(),
                                "WhatsApp runtime requested a pairing code; pair the session through setup before using the channel"
                            );
                        }
                        let Some(writer) = writer_for_callback.as_ref() else {
                            return;
                        };
                        let _ = writer.write(&WhatsAppAuthState {
                            phase: WhatsAppAuthPhase::Pending,
                            display: ChannelAuthFlowDisplay {
                                message: Some(
                                    "Enter this pairing code in WhatsApp > Linked Devices > Link with phone number instead.".to_string(),
                                ),
                                pairing_code: Some(code),
                                expires_in_secs: Some(timeout.as_secs()),
                                poll_interval_secs: Some(DEFAULT_AUTH_POLL_INTERVAL_SECS),
                                ..ChannelAuthFlowDisplay::default()
                            },
                            message: None,
                        });
                    }
                    Event::PairSuccess(_) | Event::Connected(_) => {
                        if driver_event_tx.is_some() {
                            info!("WhatsApp channel connected");
                        }
                        let Some(writer) = writer_for_callback.as_ref() else {
                            return;
                        };
                        let _ = writer.write(&WhatsAppAuthState {
                            phase: WhatsAppAuthPhase::Complete,
                            display: ChannelAuthFlowDisplay {
                                message: Some("WhatsApp pairing complete.".to_string()),
                                poll_interval_secs: Some(DEFAULT_AUTH_POLL_INTERVAL_SECS),
                                ..ChannelAuthFlowDisplay::default()
                            },
                            message: Some(format!(
                                "WhatsApp pairing complete. Session store: {store_path_for_display}"
                            )),
                        });
                    }
                    Event::PairError(err) => {
                        let Some(writer) = writer_for_callback.as_ref() else {
                            return;
                        };
                        let _ = writer.write(&WhatsAppAuthState {
                            phase: WhatsAppAuthPhase::Failed,
                            display: ChannelAuthFlowDisplay::default(),
                            message: Some(format!("WhatsApp pairing failed: {err:?}")),
                        });
                    }
                    _ => {}
                }
            }
        })
        .build()
        .await
        .context("Failed to build WhatsApp bot")?;
    let client = bot.client();
    Ok((client, bot))
}

#[cfg(unix)]
fn tighten_path_permissions(path: &Path, mode: u32) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let mut permissions = fs::metadata(path)?.permissions();
    permissions.set_mode(mode);
    fs::set_permissions(path, permissions)?;
    Ok(())
}

#[cfg(not(unix))]
fn tighten_path_permissions(_path: &Path, _mode: u32) -> Result<()> {
    Ok(())
}

fn render_whatsapp_message(message: &OutboundMessage) -> String {
    let mut parts = Vec::new();
    for block in &message.blocks {
        match block {
            turin_channel_core::MessageBlock::Text { text } => {
                if !text.trim().is_empty() {
                    parts.push(text.trim().to_string());
                }
            }
            turin_channel_core::MessageBlock::CodeBlock { language, code } => {
                let mut fenced = String::from("```");
                if let Some(language) = language.as_deref()
                    && !language.trim().is_empty()
                {
                    fenced.push_str(language.trim());
                }
                fenced.push('\n');
                fenced.push_str(code.trim_end());
                fenced.push_str("\n```");
                parts.push(fenced);
            }
        }
    }
    parts.join("\n\n")
}

fn image_name(message: &wa::message::ImageMessage, message_id: &str) -> String {
    let extension = content_type_extension(message.mimetype.as_deref()).unwrap_or("jpg");
    format!("image-{message_id}.{extension}")
}

fn document_name(message: &wa::message::DocumentMessage, message_id: &str) -> String {
    message
        .file_name
        .clone()
        .or_else(|| message.title.clone())
        .unwrap_or_else(|| {
            let extension = content_type_extension(message.mimetype.as_deref()).unwrap_or("bin");
            format!("document-{message_id}.{extension}")
        })
}

fn infer_media_name(message_id: &str, content_type: Option<&str>, fallback_stem: &str) -> String {
    if let Some(extension) = content_type_extension(content_type) {
        format!("{fallback_stem}.{extension}")
    } else {
        format!("{fallback_stem}-{message_id}")
    }
}

fn content_type_extension(content_type: Option<&str>) -> Option<&'static str> {
    match content_type.unwrap_or_default() {
        "image/jpeg" => Some("jpg"),
        "image/png" => Some("png"),
        "image/webp" => Some("webp"),
        "application/pdf" => Some("pdf"),
        "video/mp4" => Some("mp4"),
        "audio/mpeg" => Some("mp3"),
        "audio/ogg" => Some("ogg"),
        _ => None,
    }
}

fn whatsapp_media_type(content_type: Option<&str>) -> MediaType {
    match content_type.unwrap_or_default() {
        value if value.starts_with("image/") => MediaType::Image,
        value if value.starts_with("audio/") => MediaType::Audio,
        value if value.starts_with("video/") => MediaType::Video,
        _ => MediaType::Document,
    }
}

fn whatsapp_default_mime_type(media_type: MediaType) -> Option<&'static str> {
    match media_type {
        MediaType::Image => Some("image/jpeg"),
        MediaType::Video => Some("video/mp4"),
        MediaType::Audio => Some("audio/ogg"),
        MediaType::Document => Some("application/octet-stream"),
        _ => None,
    }
}

fn inbound_text(
    raw: &str,
    account_mode: WhatsAppAccountMode,
    trigger_prefix: Option<&str>,
) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }

    let required_prefix = trigger_prefix.or(match account_mode {
        WhatsAppAccountMode::Personal => Some(DEFAULT_PERSONAL_TRIGGER_PREFIX),
        WhatsAppAccountMode::Dedicated => None,
    });

    let Some(prefix) = required_prefix else {
        return Some(trimmed.to_string());
    };

    let candidate = trimmed.strip_prefix(prefix)?.trim_start();
    if candidate.is_empty() {
        None
    } else {
        Some(candidate.to_string())
    }
}

fn chat_is_allowed(chat_jid: &str, allowed_chats: &[String], banned_chats: &[String]) -> bool {
    if selector_matches_chat_list(chat_jid, banned_chats) {
        return false;
    }
    allowed_chats.is_empty() || selector_matches_chat_list(chat_jid, allowed_chats)
}

fn selector_matches_chat_list(chat_jid: &str, selectors: &[String]) -> bool {
    selectors
        .iter()
        .any(|selector| chat_selector_matches(selector, chat_jid))
}

fn chat_selector_matches(selector: &str, chat_jid: &str) -> bool {
    let selector = selector.trim();
    if selector.is_empty() {
        return false;
    }

    selector.eq_ignore_ascii_case(chat_jid)
        || selector
            .strip_prefix('@')
            .is_some_and(|value| value.eq_ignore_ascii_case(chat_jid))
        || selector.eq_ignore_ascii_case(chat_jid.split('@').next().unwrap_or(chat_jid))
}

fn settings_object(settings: &Value) -> Result<&Map<String, Value>> {
    settings
        .as_object()
        .ok_or_else(|| anyhow!("Channel settings must be a JSON object"))
}

fn optional_nonempty_string(map: &Map<String, Value>, key: &str) -> Result<Option<String>> {
    match map.get(key) {
        Some(Value::String(value)) if !value.trim().is_empty() => {
            Ok(Some(value.trim().to_string()))
        }
        Some(Value::String(_)) | Some(Value::Null) | None => Ok(None),
        Some(_) => bail!("channel setting '{key}' must be a string"),
    }
}

fn parse_session_scope(value: Option<&Value>) -> Result<ChannelSessionScope> {
    let scope = match value {
        None | Some(Value::Null) => return Ok(ChannelSessionScope::User),
        Some(Value::String(value)) => value.as_str(),
        Some(_) => bail!("channel setting 'session_scope' must be a string"),
    };

    match scope {
        "user" => Ok(ChannelSessionScope::User),
        "room" => Ok(ChannelSessionScope::Room),
        other => {
            bail!("channel setting 'session_scope' must be one of: user, room (got '{other}')")
        }
    }
}

fn parse_account_mode(value: Option<&Value>) -> Result<WhatsAppAccountMode> {
    let mode = match value {
        None | Some(Value::Null) => "personal",
        Some(Value::String(value)) => value.as_str(),
        Some(_) => bail!("channel setting 'account_mode' must be a string"),
    };

    match mode {
        "personal" => Ok(WhatsAppAccountMode::Personal),
        "dedicated" => Ok(WhatsAppAccountMode::Dedicated),
        other => bail!(
            "channel setting 'account_mode' must be one of: personal, dedicated (got '{other}')"
        ),
    }
}

fn parse_string_list(value: Option<&Value>, key: &str) -> Result<Vec<String>> {
    match value {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(values)) => {
            let mut out = Vec::with_capacity(values.len());
            for value in values {
                let Some(value) = value.as_str() else {
                    bail!("channel setting '{key}' must contain only strings");
                };
                let trimmed = value.trim();
                if !trimmed.is_empty() {
                    out.push(trimmed.to_string());
                }
            }
            Ok(out)
        }
        Some(_) => bail!("channel setting '{key}' must be an array of strings"),
    }
}

fn validate_pair_code_fields(phone_number: Option<&str>, custom_code: Option<&str>) -> Result<()> {
    if custom_code.is_some() && phone_number.is_none() {
        bail!("channel setting 'pair_code_custom_code' requires 'pair_code_phone_number'");
    }

    if let Some(custom_code) = custom_code {
        if custom_code.len() != 8 {
            bail!("channel setting 'pair_code_custom_code' must be exactly 8 characters");
        }
        if !custom_code
            .chars()
            .all(|ch| matches!(ch, '1'..='9' | 'A'..='H' | 'J'..='N' | 'P'..='T' | 'V'..='Z' | 'a'..='h' | 'j'..='n' | 'p'..='t' | 'v'..='z'))
        {
            bail!("channel setting 'pair_code_custom_code' must use Crockford Base32 characters");
        }
    }

    Ok(())
}

fn resolve_runtime_store_path(raw: &str, runtime_dir: Option<&Path>) -> PathBuf {
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        path
    } else if let Some(runtime_dir) = runtime_dir {
        runtime_dir.join(path)
    } else {
        path
    }
}

fn resolve_auth_store_path(settings: &Map<String, Value>) -> Result<PathBuf> {
    let cwd =
        std::env::current_dir().context("Failed to resolve current directory for auth flow")?;
    if let Some(raw) = optional_nonempty_string(settings, "session_store_path")? {
        let configured = PathBuf::from(raw);
        if configured.is_absolute() {
            return Ok(configured);
        }
        return Ok(cwd.join(configured));
    }

    let workspace_id = optional_nonempty_string(settings, "workspace_id")?
        .unwrap_or_else(|| DEFAULT_WORKSPACE_ID.to_string());
    Ok(default_auth_store_path(&cwd, &workspace_id))
}

fn default_auth_store_path(cwd: &Path, workspace_id: &str) -> PathBuf {
    let workspace_component = sanitize_component(workspace_id);
    cwd.join(".turin")
        .join("data")
        .join("channels")
        .join(format!("whatsapp-{workspace_component}.db"))
}

fn sanitize_component(raw: &str) -> String {
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

fn spawn_auth_flow_worker(session: &WhatsAppAuthSession) -> Result<()> {
    let current_exe =
        std::env::current_exe().context("Failed to resolve WhatsApp runner executable")?;
    let session_json =
        serde_json::to_string(session).context("Failed to encode WhatsApp auth flow session")?;
    Command::new(current_exe)
        .arg("auth-flow-worker")
        .arg("--session-json")
        .arg(session_json)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .context("Failed to spawn WhatsApp auth flow worker")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn adapter_manifest_is_valid() {
        let manifest = adapter_manifest();
        assert_eq!(manifest.kind, "whatsapp");
        assert!(manifest.runtime.capabilities.attachments);
        manifest.validate().expect("valid manifest");
        let setup = manifest.setup.expect("setup manifest");
        assert_eq!(setup.auth_flows.len(), 1);
        assert_eq!(setup.auth_flows[0].id, DEFAULT_AUTH_FLOW_ID);
    }

    #[test]
    fn validate_pair_code_requires_phone_number() {
        let err = validate_pair_code_fields(None, Some("ABCD1234")).expect_err("invalid");
        assert!(err.to_string().contains("pair_code_phone_number"));
    }

    #[test]
    fn parse_settings_resolves_runtime_default_store() {
        let temp = tempfile::tempdir().expect("tempdir");
        let config = parse_settings(&json!({}), Some(temp.path())).expect("settings");
        assert_eq!(
            config.session_store_path,
            temp.path().join(DEFAULT_RUNTIME_STORE_BASENAME)
        );
        assert_eq!(config.account_mode, WhatsAppAccountMode::Personal);
        assert_eq!(
            config.trigger_prefix.as_deref(),
            Some(DEFAULT_PERSONAL_TRIGGER_PREFIX)
        );
        assert_eq!(
            config.max_inbound_text_chars,
            DEFAULT_MAX_INBOUND_TEXT_CHARS
        );
        assert_eq!(config.media_dir, temp.path().join("media"));
    }

    #[test]
    fn dedicated_mode_does_not_force_trigger_prefix() {
        let temp = tempfile::tempdir().expect("tempdir");
        let config = parse_settings(&json!({"account_mode": "dedicated"}), Some(temp.path()))
            .expect("settings");
        assert_eq!(config.account_mode, WhatsAppAccountMode::Dedicated);
        assert_eq!(config.trigger_prefix, None);
    }

    #[test]
    fn inbound_text_requires_prefix_for_personal_mode() {
        assert_eq!(
            inbound_text(
                "/turin status",
                WhatsAppAccountMode::Personal,
                Some(DEFAULT_PERSONAL_TRIGGER_PREFIX)
            ),
            Some("status".to_string())
        );
        assert_eq!(
            inbound_text("status", WhatsAppAccountMode::Personal, Some("/turin")),
            None
        );
        assert_eq!(
            inbound_text("status", WhatsAppAccountMode::Dedicated, None),
            Some("status".to_string())
        );
    }

    #[test]
    fn banned_chats_override_allowed_chats() {
        let allowed = vec![
            "15551234567@s.whatsapp.net".to_string(),
            "120363123456789@g.us".to_string(),
        ];
        let banned = vec!["15551234567".to_string()];
        assert!(!chat_is_allowed(
            "15551234567@s.whatsapp.net",
            &allowed,
            &banned
        ));
        assert!(chat_is_allowed("120363123456789@g.us", &allowed, &banned));
    }

    #[test]
    fn poll_complete_returns_store_path_and_clears_pair_code_fields() {
        let temp = tempfile::tempdir().expect("tempdir");
        let state_path = temp.path().join("state.json");
        let writer = AuthStateWriter::new(state_path.clone());
        writer
            .write(&WhatsAppAuthState {
                phase: WhatsAppAuthPhase::Complete,
                display: ChannelAuthFlowDisplay::default(),
                message: Some("done".to_string()),
            })
            .expect("state written");

        let response = poll_auth_flow(&ChannelAuthFlowPollRequest {
            flow_id: DEFAULT_AUTH_FLOW_ID.to_string(),
            session: serde_json::to_value(WhatsAppAuthSession {
                ticket: "t".to_string(),
                state_path,
                store_path: PathBuf::from("/tmp/whatsapp.db"),
                phone_number: Some("15551234567".to_string()),
                custom_code: Some("ABCD1234".to_string()),
            })
            .expect("session"),
            current_settings: json!({}),
        })
        .expect("poll response");

        match response {
            ChannelAuthFlowPollResponse::Complete { values, .. } => {
                let values_by_name: HashMap<_, _> = values
                    .into_iter()
                    .map(|value| (value.target.name, value.value))
                    .collect();
                assert_eq!(
                    values_by_name.get("session_store_path"),
                    Some(&Value::String("/tmp/whatsapp.db".to_string()))
                );
                assert_eq!(
                    values_by_name.get("pair_code_phone_number"),
                    Some(&Value::Null)
                );
                assert_eq!(
                    values_by_name.get("pair_code_custom_code"),
                    Some(&Value::Null)
                );
            }
            other => panic!("unexpected response: {other:?}"),
        }
    }
}
