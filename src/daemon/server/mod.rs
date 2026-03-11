mod dispatch;
mod events;
#[cfg(test)]
mod tests;
mod watch;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{RwLock, broadcast, watch as watch_channel};
use tracing::{error, info, warn};

use crate::daemon::channels::ChannelRuntimeManager;
use crate::daemon::protocol::{DaemonRequest, ErrorCode, RequestEnvelope, ResponseEnvelope};
use crate::daemon::state::DaemonState;

#[derive(Clone)]
struct ClientContext {
    state: Arc<RwLock<DaemonState>>,
    watcher_slot: Arc<std::sync::Mutex<Option<notify::RecommendedWatcher>>>,
    daemon_watcher_tx: tokio::sync::mpsc::Sender<Vec<PathBuf>>,
    event_tx: broadcast::Sender<crate::daemon::protocol::EventEnvelope>,
    channel_runtimes: Arc<ChannelRuntimeManager>,
    shutdown_tx: watch_channel::Sender<bool>,
    shutdown_rx: watch_channel::Receiver<bool>,
}

pub async fn serve(config_path: &Path) -> Result<()> {
    let state = Arc::new(RwLock::new(DaemonState::load(config_path).await?));
    let socket_path = {
        let guard = state.read().await;
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

    let (shutdown_tx, mut shutdown_rx) = watch_channel::channel(false);
    let (event_tx, _) = broadcast::channel(512);
    let watcher_slot = Arc::new(std::sync::Mutex::new(None));
    let channel_runtimes = Arc::new(ChannelRuntimeManager::new(
        socket_path.clone(),
        event_tx.clone(),
    ));
    {
        let guard = state.read().await;
        let workspace_root = PathBuf::from(&guard.bootstrap_config.kernel.workspace_root);
        let channels = guard.registry_load.channels.clone();
        channel_runtimes.sync(workspace_root, channels).await?;
    }
    let daemon_watcher_tx = watch::start_daemon_watcher(
        Arc::clone(&state),
        Arc::clone(&watcher_slot),
        Arc::clone(&channel_runtimes),
        event_tx.clone(),
    )
    .await?;
    events::start_task_event_poller(Arc::clone(&state), event_tx.clone(), shutdown_rx.clone());
    let client_ctx = ClientContext {
        state: Arc::clone(&state),
        watcher_slot: Arc::clone(&watcher_slot),
        daemon_watcher_tx: daemon_watcher_tx.clone(),
        event_tx: event_tx.clone(),
        channel_runtimes: Arc::clone(&channel_runtimes),
        shutdown_tx: shutdown_tx.clone(),
        shutdown_rx: shutdown_rx.clone(),
    };

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
                        let client_ctx = client_ctx.clone();
                        tokio::spawn(async move {
                            if let Err(err) = handle_client(stream, client_ctx).await {
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

    {
        let mut slot = watcher_slot
            .lock()
            .expect("daemon watcher mutex poisoned during shutdown");
        *slot = None;
    }
    channel_runtimes.shutdown().await;
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

async fn handle_client(stream: UnixStream, ctx: ClientContext) -> Result<()> {
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
                    ErrorCode::InvalidRequest,
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

        if matches!(request.request, DaemonRequest::RuntimeEventsSubscribe(_)) {
            events::stream_events(
                request,
                Arc::clone(&ctx.state),
                Arc::clone(&ctx.channel_runtimes),
                ctx.event_tx.subscribe(),
                ctx.shutdown_rx.clone(),
                &mut writer,
            )
            .await?;
            break;
        }

        let response = dispatch::dispatch(
            request,
            Arc::clone(&ctx.state),
            Arc::clone(&ctx.watcher_slot),
            ctx.daemon_watcher_tx.clone(),
            ctx.event_tx.clone(),
            Arc::clone(&ctx.channel_runtimes),
            ctx.shutdown_tx.clone(),
        )
        .await;
        writer
            .write_all(serde_json::to_string(&response)?.as_bytes())
            .await?;
        writer.write_all(b"\n").await?;
    }

    Ok(())
}
