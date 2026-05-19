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

mod render;
mod settings;
use render::render_whatsapp_message;
pub(crate) use settings::{
    WhatsAppAccountMode, optional_nonempty_string, parse_settings, resolve_auth_store_path,
    sanitize_component, settings_object, validate_pair_code_fields,
};
pub use settings::{WhatsAppChannelDriverConfig, validate_settings};

const DEFAULT_WORKSPACE_ID: &str = "whatsapp";
const DEFAULT_AUTH_FLOW_ID: &str = "pair";
const DEFAULT_AUTH_POLL_INTERVAL_SECONDS: u64 = 3;
const DEFAULT_AUTH_TIMEOUT_SECONDS: u64 = 300;
const DEFAULT_RUNTIME_STORE_BASENAME: &str = "whatsapp-session.db";
const DEFAULT_PERSONAL_TRIGGER_PREFIX: &str = "/turin";

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
            poll_interval_seconds: Some(DEFAULT_AUTH_POLL_INTERVAL_SECONDS),
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
            poll_interval_seconds: Some(DEFAULT_AUTH_POLL_INTERVAL_SECONDS),
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
    let deadline = Instant::now() + Duration::from_secs(DEFAULT_AUTH_TIMEOUT_SECONDS);

    loop {
        if client.is_logged_in() {
            writer.write(&WhatsAppAuthState {
                phase: WhatsAppAuthPhase::Complete,
                display: ChannelAuthFlowDisplay {
                    message: Some("WhatsApp pairing complete.".to_string()),
                    poll_interval_seconds: Some(DEFAULT_AUTH_POLL_INTERVAL_SECONDS),
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
                                expires_in_seconds = timeout.as_secs(),
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
                                expires_in_seconds: Some(timeout.as_secs()),
                                poll_interval_seconds: Some(DEFAULT_AUTH_POLL_INTERVAL_SECONDS),
                                ..ChannelAuthFlowDisplay::default()
                            },
                            message: None,
                        });
                    }
                    Event::PairingCode { code, timeout } => {
                        if driver_event_tx.is_some() {
                            warn!(
                                expires_in_seconds = timeout.as_secs(),
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
                                expires_in_seconds: Some(timeout.as_secs()),
                                poll_interval_seconds: Some(DEFAULT_AUTH_POLL_INTERVAL_SECONDS),
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
                                poll_interval_seconds: Some(DEFAULT_AUTH_POLL_INTERVAL_SECONDS),
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
mod tests;
