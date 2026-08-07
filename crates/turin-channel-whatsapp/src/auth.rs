use std::fs;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use turin_channel_core::{
    ChannelAuthFlowDisplay, ChannelAuthFlowPollRequest, ChannelAuthFlowPollResponse,
    ChannelAuthFlowResolvedValue, ChannelAuthFlowStartRequest, ChannelAuthFlowStartResponse,
    ChannelConfigTarget, ChannelConfigTargetKind,
};
use uuid::Uuid;

use crate::{
    DEFAULT_AUTH_FLOW_ID, DEFAULT_AUTH_POLL_INTERVAL_SECONDS, DEFAULT_AUTH_TIMEOUT_SECONDS,
    bot::build_bot,
    settings::{
        optional_nonempty_string, resolve_auth_store_path, settings_object,
        validate_pair_code_fields,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WhatsAppAuthSession {
    pub(crate) ticket: String,
    pub(crate) state_path: PathBuf,
    pub(crate) store_path: PathBuf,
    pub(crate) phone_number: Option<String>,
    pub(crate) custom_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WhatsAppAuthPhase {
    Pending,
    Complete,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct WhatsAppAuthState {
    pub(crate) phase: WhatsAppAuthPhase,
    pub(crate) display: ChannelAuthFlowDisplay,
    pub(crate) message: Option<String>,
}

#[derive(Clone)]
pub(crate) struct AuthStateWriter {
    path: PathBuf,
    lock: Arc<Mutex<()>>,
}

impl AuthStateWriter {
    pub(crate) fn new(path: PathBuf) -> Self {
        Self {
            path,
            lock: Arc::new(Mutex::new(())),
        }
    }

    pub(crate) fn load(&self) -> Result<WhatsAppAuthState> {
        let bytes = fs::read(&self.path)
            .with_context(|| format!("Failed to read auth flow state '{}'", self.path.display()))?;
        serde_json::from_slice(&bytes).context("Failed to decode WhatsApp auth flow state")
    }

    pub(crate) fn write(&self, state: &WhatsAppAuthState) -> Result<()> {
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
    let (client, bot) = build_bot(
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
    let mut bot_handle = bot.spawn();
    let mut bot_finished = false;
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
            _ = &mut bot_handle => {
                bot_finished = true;
                if !client.is_logged_in() {
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

    if !bot_finished {
        bot_handle.shutdown().await;
    }
    Ok(())
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
