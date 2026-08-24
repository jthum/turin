mod dispatch;
mod events;
mod scheduler;
#[cfg(test)]
mod tests;
mod watch;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{RwLock, broadcast, watch as watch_channel};
use tracing::{debug, error, info, warn};
use turin_local_ipc::{
    BoxedLocalIpcStream, LocalIpcListener, cleanup_stale_endpoint, remove_endpoint,
    split as split_local_ipc,
};

use crate::daemon::protocol::{DaemonRequest, ErrorCode, RequestEnvelope, ResponseEnvelope};
use crate::daemon::state::DaemonState;
use crate::kernel::harness_runtime::HarnessAdapterFactory;

#[derive(Clone)]
struct ClientContext {
    state: Arc<RwLock<DaemonState>>,
    watcher_slot: Arc<std::sync::Mutex<Option<notify::RecommendedWatcher>>>,
    daemon_watcher_tx: tokio::sync::mpsc::Sender<Vec<PathBuf>>,
    event_tx: broadcast::Sender<crate::daemon::protocol::EventEnvelope>,
    shutdown_tx: watch_channel::Sender<bool>,
    shutdown_rx: watch_channel::Receiver<bool>,
}

pub async fn serve_with_harness_adapter(
    config_path: &Path,
    script_harness_adapter: Arc<dyn HarnessAdapterFactory>,
) -> Result<()> {
    let state = Arc::new(RwLock::new(
        DaemonState::load_with_harness_adapter(config_path, script_harness_adapter).await?,
    ));
    let endpoint = {
        let guard = state.read().await;
        guard.endpoint().to_path_buf()
    };

    #[cfg(unix)]
    if let Some(parent) = endpoint.parent() {
        tokio::fs::create_dir_all(parent).await.with_context(|| {
            format!("Failed to create endpoint directory '{}'", parent.display())
        })?;
    }

    cleanup_stale_endpoint(&endpoint).await.with_context(|| {
        format!(
            "Failed to prepare local IPC endpoint '{}'",
            endpoint.display()
        )
    })?;
    let mut listener = LocalIpcListener::bind(&endpoint)
        .with_context(|| format!("Failed to bind local IPC endpoint '{}'", endpoint.display()))?;

    info!(endpoint = %endpoint.display(), "Turin daemon started");

    let (shutdown_tx, mut shutdown_rx) = watch_channel::channel(false);
    let (event_tx, _) = broadcast::channel(512);
    #[cfg(feature = "perf-diagnostics")]
    crate::perf_diagnostics::install_event_sink(event_tx.clone());
    let watcher_slot = Arc::new(std::sync::Mutex::new(None));
    let authorization_requests = {
        let guard = state.read().await;
        guard.tool_authorization_broker().subscribe_requests()
    };
    let daemon_watcher_tx = watch::start_daemon_watcher(
        Arc::clone(&state),
        Arc::clone(&watcher_slot),
        event_tx.clone(),
    )
    .await?;
    events::start_task_event_poller(Arc::clone(&state), event_tx.clone(), shutdown_rx.clone());
    events::start_tool_authorization_events(
        authorization_requests,
        event_tx.clone(),
        shutdown_rx.clone(),
    );
    scheduler::start_internal_scheduler(Arc::clone(&state), shutdown_rx.clone());
    let client_ctx = ClientContext {
        state: Arc::clone(&state),
        watcher_slot: Arc::clone(&watcher_slot),
        daemon_watcher_tx: daemon_watcher_tx.clone(),
        event_tx: event_tx.clone(),
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
                    Ok(stream) => {
                        let client_ctx = client_ctx.clone();
                        tokio::spawn(async move {
                            if let Err(err) = handle_client(stream, client_ctx).await {
                                if is_expected_client_disconnect(&err) {
                                    debug!(error = %err, "Daemon client disconnected");
                                } else {
                                    error!(error = %err, "Daemon client handler failed");
                                }
                            }
                        });
                    }
                    Err(err) => {
                        warn!(error = %err, "Failed to accept daemon IPC connection");
                    }
                }
            }
        }
    }

    let _ = shutdown_tx.send(true);
    {
        let mut slot = watcher_slot
            .lock()
            .expect("daemon watcher mutex poisoned during shutdown");
        *slot = None;
    }
    state.write().await.shutdown().await;
    remove_endpoint(&endpoint).await.ok();
    Ok(())
}

async fn handle_client(stream: BoxedLocalIpcStream, ctx: ClientContext) -> Result<()> {
    let (reader, mut writer) = split_local_ipc(stream);
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

fn is_expected_client_disconnect(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        cause.downcast_ref::<std::io::Error>().is_some_and(|io| {
            matches!(
                io.kind(),
                std::io::ErrorKind::BrokenPipe
                    | std::io::ErrorKind::ConnectionReset
                    | std::io::ErrorKind::ConnectionAborted
                    | std::io::ErrorKind::UnexpectedEof
            )
        })
    })
}
