use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::mpsc;
use tracing::{info, warn};
use whatsapp_rust::TokioRuntime;
use whatsapp_rust::bot::Bot;
use whatsapp_rust::pair_code::PairCodeOptions;
use whatsapp_rust::store::SqliteStore;
use whatsapp_rust::types::events::Event;
use whatsapp_rust::types::message::MessageInfo;
use whatsapp_rust::waproto::whatsapp as wa;
use whatsapp_rust_tokio_transport::TokioWebSocketTransportFactory;
use whatsapp_rust_ureq_http_client::UreqHttpClient;

use crate::{
    DEFAULT_AUTH_POLL_INTERVAL_SECONDS,
    auth::{AuthStateWriter, WhatsAppAuthPhase, WhatsAppAuthState},
};

pub(crate) enum DriverEvent {
    Message(Box<wa::Message>, Box<MessageInfo>),
    LoggedOut(String),
}

pub(crate) async fn build_bot(
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
                            let _ =
                                driver_event_tx.send(DriverEvent::Message(message, Box::new(info)));
                        }
                    }
                    Event::LoggedOut(reason) => {
                        if let Some(driver_event_tx) = driver_event_tx.as_ref() {
                            let _ = driver_event_tx
                                .send(DriverEvent::LoggedOut(format!("{reason:?}")));
                        }
                        if let Some(writer) = writer_for_callback.as_ref() {
                            let _ = writer.write(&WhatsAppAuthState {
                                phase: WhatsAppAuthPhase::Failed,
                                display: turin_channel_core::ChannelAuthFlowDisplay::default(),
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
                            display: turin_channel_core::ChannelAuthFlowDisplay {
                                message: Some(
                                    "Scan this QR code in WhatsApp > Linked Devices.".to_string(),
                                ),
                                qr_text: Some(code),
                                expires_in_seconds: Some(timeout.as_secs()),
                                poll_interval_seconds: Some(DEFAULT_AUTH_POLL_INTERVAL_SECONDS),
                                ..turin_channel_core::ChannelAuthFlowDisplay::default()
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
                            display: turin_channel_core::ChannelAuthFlowDisplay {
                                message: Some(
                                    "Enter this pairing code in WhatsApp > Linked Devices > Link with phone number instead.".to_string(),
                                ),
                                pairing_code: Some(code),
                                expires_in_seconds: Some(timeout.as_secs()),
                                poll_interval_seconds: Some(DEFAULT_AUTH_POLL_INTERVAL_SECONDS),
                                ..turin_channel_core::ChannelAuthFlowDisplay::default()
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
                            display: turin_channel_core::ChannelAuthFlowDisplay {
                                message: Some("WhatsApp pairing complete.".to_string()),
                                poll_interval_seconds: Some(DEFAULT_AUTH_POLL_INTERVAL_SECONDS),
                                ..turin_channel_core::ChannelAuthFlowDisplay::default()
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
                            display: turin_channel_core::ChannelAuthFlowDisplay::default(),
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
