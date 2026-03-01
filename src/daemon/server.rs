use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{Mutex, watch};
use tracing::{error, info, warn};

use crate::daemon::protocol::{RequestEnvelope, ResponseEnvelope};
use crate::daemon::state::DaemonState;

pub async fn serve(config_path: &Path) -> Result<()> {
    let state = Arc::new(Mutex::new(DaemonState::load(config_path).await?));
    let socket_path = {
        let guard = state.lock().await;
        guard.socket_path().to_path_buf()
    };

    if let Some(parent) = socket_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("Failed to create socket directory '{}'", parent.display()))?;
    }

    cleanup_stale_socket(&socket_path).await?;
    let listener = UnixListener::bind(&socket_path)
        .with_context(|| format!("Failed to bind socket '{}'", socket_path.display()))?;

    info!(socket = %socket_path.display(), "Turin daemon started");

    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("Daemon shutdown requested via signal");
                break;
            }
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    info!("Daemon shutdown requested via control API");
                    break;
                }
            }
            accept_res = listener.accept() => {
                match accept_res {
                    Ok((stream, _)) => {
                        let state = Arc::clone(&state);
                        let shutdown_tx = shutdown_tx.clone();
                        tokio::spawn(async move {
                            if let Err(err) = handle_client(stream, state, shutdown_tx).await {
                                error!(error = %err, "Daemon client handler failed");
                            }
                        });
                    }
                    Err(err) => {
                        warn!(error = %err, "Failed to accept daemon socket connection");
                    }
                }
            }
        }
    }

    tokio::fs::remove_file(&socket_path).await.ok();
    Ok(())
}

async fn cleanup_stale_socket(socket_path: &Path) -> Result<()> {
    if !socket_path.exists() {
        return Ok(());
    }

    match UnixStream::connect(socket_path).await {
        Ok(_) => anyhow::bail!(
            "Daemon socket '{}' is already in use",
            socket_path.display()
        ),
        Err(_) => {
            tokio::fs::remove_file(socket_path).await.with_context(|| {
                format!("Failed to remove stale socket '{}'", socket_path.display())
            })?;
        }
    }

    Ok(())
}

async fn handle_client(
    stream: UnixStream,
    state: Arc<Mutex<DaemonState>>,
    shutdown_tx: watch::Sender<bool>,
) -> Result<()> {
    let (reader, mut writer) = stream.into_split();
    let mut lines = BufReader::new(reader).lines();

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }

        let request: RequestEnvelope = match serde_json::from_str(&line) {
            Ok(req) => req,
            Err(err) => {
                let response = ResponseEnvelope::err(
                    None,
                    "invalid_request",
                    format!("Failed to parse request: {}", err),
                    None,
                );
                writer
                    .write_all(serde_json::to_string(&response)?.as_bytes())
                    .await?;
                writer.write_all(b"\n").await?;
                continue;
            }
        };

        let response = dispatch(request, Arc::clone(&state), shutdown_tx.clone()).await;
        writer
            .write_all(serde_json::to_string(&response)?.as_bytes())
            .await?;
        writer.write_all(b"\n").await?;
    }

    Ok(())
}

async fn dispatch(
    request: RequestEnvelope,
    state: Arc<Mutex<DaemonState>>,
    shutdown_tx: watch::Sender<bool>,
) -> ResponseEnvelope {
    match request.op.as_str() {
        "daemon.ping" => ResponseEnvelope::ok(
            request.id,
            json!({
                "pong": true,
                "version": env!("CARGO_PKG_VERSION"),
            }),
        ),
        "daemon.status" => {
            let guard = state.lock().await;
            match serde_json::to_value(guard.status()) {
                Ok(value) => ResponseEnvelope::ok(request.id, value),
                Err(err) => ResponseEnvelope::err(
                    request.id,
                    "serialize_error",
                    format!("Failed to serialize daemon status: {}", err),
                    None,
                ),
            }
        }
        "runtime.rescan" => {
            let mut guard = state.lock().await;
            match guard.rescan().await {
                Ok(status) => match serde_json::to_value(status) {
                    Ok(value) => ResponseEnvelope::ok(request.id, value),
                    Err(err) => ResponseEnvelope::err(
                        request.id,
                        "serialize_error",
                        format!("Failed to serialize rescan result: {}", err),
                        None,
                    ),
                },
                Err(err) => {
                    ResponseEnvelope::err(request.id, "rescan_failed", err.to_string(), None)
                }
            }
        }
        "agent.list" => {
            let guard = state.lock().await;
            ResponseEnvelope::ok(
                request.id,
                json!({ "agents": guard.registry_snapshot().agents }),
            )
        }
        "harness.list" => {
            let guard = state.lock().await;
            ResponseEnvelope::ok(
                request.id,
                json!({
                    "harnesses": guard.status().harnesses
                }),
            )
        }
        "daemon.stop" => {
            let _ = shutdown_tx.send(true);
            ResponseEnvelope::ok(request.id, json!({ "stopping": true }))
        }
        _ => ResponseEnvelope::err(
            request.id,
            "unknown_operation",
            format!("Unknown daemon operation '{}'", request.op),
            None,
        ),
    }
}
