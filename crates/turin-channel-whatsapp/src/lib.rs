use std::fs;
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
    ChannelAdapterManifest, ChannelAuthFlow, ChannelAuthFlowDisplay, ChannelAuthFlowKind,
    ChannelAuthFlowPollRequest, ChannelAuthFlowPollResponse, ChannelAuthFlowResolvedValue,
    ChannelAuthFlowStartRequest, ChannelAuthFlowStartResponse, ChannelCapabilities,
    ChannelConfigField, ChannelConfigFieldOption, ChannelConfigTarget, ChannelConfigTargetKind,
    ChannelConversationKey, ChannelIdentitySelectors, ChannelInstallManifest, ChannelKind,
    ChannelMessageRef, ChannelRuntimeCapabilities, ChannelRuntimeManifest, ChannelSessionScope,
    ChannelSetupManifest, ChannelUser, InboundEvent, OutboundMessage,
};
use turin_channel_runner::ChannelDriver;
use uuid::Uuid;
use whatsapp_rust::Jid;
use whatsapp_rust::TokioRuntime;
use whatsapp_rust::bot::{Bot, BotHandle};
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

#[derive(Debug, Clone)]
pub struct WhatsAppChannelDriverConfig {
    pub workspace_id: String,
    pub session_scope: ChannelSessionScope,
    pub session_store_path: PathBuf,
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
                attachments: false,
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

    fn message_to_event(
        &self,
        message: Box<wa::Message>,
        info: MessageInfo,
    ) -> Option<InboundEvent> {
        if info.source.is_from_me || info.source.chat.to_string() == "status@broadcast" {
            return None;
        }

        let text = message.text_content()?.trim().to_string();
        if text.is_empty() {
            return None;
        }

        let chat_id = info.source.chat.to_string();
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

        Some(InboundEvent {
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
            attachments: vec![],
            metadata,
        })
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
            attachments: false,
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
                            if let Some(event) = self.message_to_event(message, *info) {
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
        if rendered.trim().is_empty() {
            return Ok(());
        }
        self.client
            .send_message(
                chat,
                wa::Message {
                    conversation: Some(rendered),
                    ..Default::default()
                },
            )
            .await
            .context("Failed to send WhatsApp message")?;
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
    let session_scope = parse_session_scope(map.get("session_scope"))?;

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

    Ok(WhatsAppChannelDriverConfig {
        workspace_id,
        session_scope,
        session_store_path,
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
    }

    let database_url = session_store_path.to_string_lossy().to_string();
    let backend = Arc::new(SqliteStore::new(&database_url).await.with_context(|| {
        format!(
            "Failed to open WhatsApp session store '{}'",
            session_store_path.display()
        )
    })?);
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
