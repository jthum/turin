use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use notify::Event;
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{Mutex, watch};
use tracing::{error, info, warn};

use crate::daemon::protocol::{RequestEnvelope, ResponseEnvelope};
use crate::daemon::state::{
    CreateAgentInput, DaemonState, DaemonStatus, DaemonWatchPaths, UpdateAgentInput,
};

#[derive(Debug, serde::Deserialize)]
struct CreateAgentParams {
    id: String,
    provider: String,
    model: String,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    thinking: Option<crate::kernel::config::ThinkingConfig>,
    #[serde(default)]
    mode: Option<crate::kernel::config::AgentMode>,
    #[serde(default)]
    harness: Option<String>,
    #[serde(default)]
    idle_grace_secs: Option<u64>,
    #[serde(default = "default_enabled")]
    enabled: bool,
}

#[derive(Debug, serde::Deserialize)]
struct AgentIdParams {
    id: String,
}

#[derive(Debug, serde::Deserialize)]
struct UpdateAgentParams {
    id: String,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    thinking: Option<crate::kernel::config::ThinkingConfig>,
    #[serde(default)]
    mode: Option<crate::kernel::config::AgentMode>,
    #[serde(default)]
    idle_grace_secs: Option<u64>,
}

#[derive(Debug, serde::Deserialize)]
struct SubmitTaskParams {
    agent_id: String,
    prompt: String,
}

#[derive(Debug, serde::Deserialize)]
struct TaskIdParams {
    request_id: String,
}

fn default_enabled() -> bool {
    true
}

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
    let watcher_slot = Arc::new(std::sync::Mutex::new(None));
    let daemon_watcher_tx =
        start_daemon_watcher(Arc::clone(&state), Arc::clone(&watcher_slot)).await?;

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
                        let watcher_slot = Arc::clone(&watcher_slot);
                        let daemon_watcher_tx = daemon_watcher_tx.clone();
                        let shutdown_tx = shutdown_tx.clone();
                        tokio::spawn(async move {
                            if let Err(err) =
                                handle_client(
                                    stream,
                                    state,
                                    watcher_slot,
                                    daemon_watcher_tx,
                                    shutdown_tx,
                                )
                                .await
                            {
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
    watcher_slot: Arc<std::sync::Mutex<Option<notify::RecommendedWatcher>>>,
    daemon_watcher_tx: tokio::sync::mpsc::Sender<Vec<PathBuf>>,
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

        let response = dispatch(
            request,
            Arc::clone(&state),
            Arc::clone(&watcher_slot),
            daemon_watcher_tx.clone(),
            shutdown_tx.clone(),
        )
        .await;
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
    watcher_slot: Arc<std::sync::Mutex<Option<notify::RecommendedWatcher>>>,
    daemon_watcher_tx: tokio::sync::mpsc::Sender<Vec<PathBuf>>,
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
            match rescan_and_refresh_watcher(state, watcher_slot, daemon_watcher_tx).await {
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
        "agent.get" => {
            let params: AgentIdParams = match serde_json::from_value(request.params) {
                Ok(params) => params,
                Err(err) => {
                    return ResponseEnvelope::err(
                        request.id,
                        "invalid_params",
                        format!("Failed to parse agent.get params: {}", err),
                        None,
                    );
                }
            };
            let guard = state.lock().await;
            match guard.agent_detail(&params.id) {
                Ok(Some(agent)) => match serde_json::to_value(agent) {
                    Ok(value) => ResponseEnvelope::ok(request.id, value),
                    Err(err) => ResponseEnvelope::err(
                        request.id,
                        "serialize_error",
                        format!("Failed to serialize agent detail: {}", err),
                        None,
                    ),
                },
                Ok(None) => ResponseEnvelope::err(
                    request.id,
                    "agent_not_found",
                    format!("Agent '{}' not found", params.id),
                    None,
                ),
                Err(err) => {
                    ResponseEnvelope::err(request.id, "agent_get_failed", err.to_string(), None)
                }
            }
        }
        "agent.create" => {
            let params: CreateAgentParams = match serde_json::from_value(request.params) {
                Ok(params) => params,
                Err(err) => {
                    return ResponseEnvelope::err(
                        request.id,
                        "invalid_params",
                        format!("Failed to parse agent.create params: {}", err),
                        None,
                    );
                }
            };
            let mut guard = state.lock().await;
            match guard
                .create_agent(CreateAgentInput {
                    id: params.id,
                    provider: params.provider,
                    model: params.model,
                    system_prompt: params.system_prompt,
                    thinking: params.thinking,
                    mode: params.mode,
                    harness: params.harness,
                    idle_grace_secs: params.idle_grace_secs,
                    enabled: params.enabled,
                })
                .await
            {
                Ok(agent) => match serde_json::to_value(agent) {
                    Ok(value) => ResponseEnvelope::ok(request.id, value),
                    Err(err) => ResponseEnvelope::err(
                        request.id,
                        "serialize_error",
                        format!("Failed to serialize created agent: {}", err),
                        None,
                    ),
                },
                Err(err) => {
                    ResponseEnvelope::err(request.id, "agent_create_failed", err.to_string(), None)
                }
            }
        }
        "agent.enable" | "agent.disable" => {
            let params: AgentIdParams = match serde_json::from_value(request.params) {
                Ok(params) => params,
                Err(err) => {
                    return ResponseEnvelope::err(
                        request.id,
                        "invalid_params",
                        format!("Failed to parse agent toggle params: {}", err),
                        None,
                    );
                }
            };
            let enabled = request.op == "agent.enable";
            let mut guard = state.lock().await;
            match guard.set_agent_enabled(&params.id, enabled).await {
                Ok(agent) => match serde_json::to_value(agent) {
                    Ok(value) => ResponseEnvelope::ok(request.id, value),
                    Err(err) => ResponseEnvelope::err(
                        request.id,
                        "serialize_error",
                        format!("Failed to serialize agent toggle result: {}", err),
                        None,
                    ),
                },
                Err(err) => {
                    ResponseEnvelope::err(request.id, "agent_toggle_failed", err.to_string(), None)
                }
            }
        }
        "agent.update" => {
            let params: UpdateAgentParams = match serde_json::from_value(request.params) {
                Ok(params) => params,
                Err(err) => {
                    return ResponseEnvelope::err(
                        request.id,
                        "invalid_params",
                        format!("Failed to parse agent.update params: {}", err),
                        None,
                    );
                }
            };
            let mut guard = state.lock().await;
            match guard
                .update_agent(
                    &params.id,
                    UpdateAgentInput {
                        provider: params.provider,
                        model: params.model,
                        system_prompt: params.system_prompt,
                        thinking: params.thinking,
                        mode: params.mode,
                        idle_grace_secs: params.idle_grace_secs,
                    },
                )
                .await
            {
                Ok(agent) => match serde_json::to_value(agent) {
                    Ok(value) => ResponseEnvelope::ok(request.id, value),
                    Err(err) => ResponseEnvelope::err(
                        request.id,
                        "serialize_error",
                        format!("Failed to serialize updated agent: {}", err),
                        None,
                    ),
                },
                Err(err) => {
                    ResponseEnvelope::err(request.id, "agent_update_failed", err.to_string(), None)
                }
            }
        }
        "agent.delete" => {
            let params: AgentIdParams = match serde_json::from_value(request.params) {
                Ok(params) => params,
                Err(err) => {
                    return ResponseEnvelope::err(
                        request.id,
                        "invalid_params",
                        format!("Failed to parse agent.delete params: {}", err),
                        None,
                    );
                }
            };
            let mut guard = state.lock().await;
            match guard.delete_agent(&params.id).await {
                Ok(status) => match serde_json::to_value(status) {
                    Ok(value) => ResponseEnvelope::ok(request.id, value),
                    Err(err) => ResponseEnvelope::err(
                        request.id,
                        "serialize_error",
                        format!("Failed to serialize delete status: {}", err),
                        None,
                    ),
                },
                Err(err) => {
                    ResponseEnvelope::err(request.id, "agent_delete_failed", err.to_string(), None)
                }
            }
        }
        "task.submit" => {
            let params: SubmitTaskParams = match serde_json::from_value(request.params) {
                Ok(params) => params,
                Err(err) => {
                    return ResponseEnvelope::err(
                        request.id,
                        "invalid_params",
                        format!("Failed to parse task.submit params: {}", err),
                        None,
                    );
                }
            };
            let guard = state.lock().await;
            match guard.submit_task(&params.agent_id, params.prompt).await {
                Ok(task) => match serde_json::to_value(task) {
                    Ok(value) => ResponseEnvelope::ok(request.id, value),
                    Err(err) => ResponseEnvelope::err(
                        request.id,
                        "serialize_error",
                        format!("Failed to serialize submitted task: {}", err),
                        None,
                    ),
                },
                Err(err) => {
                    ResponseEnvelope::err(request.id, "task_submit_failed", err.to_string(), None)
                }
            }
        }
        "task.get" => {
            let params: TaskIdParams = match serde_json::from_value(request.params) {
                Ok(params) => params,
                Err(err) => {
                    return ResponseEnvelope::err(
                        request.id,
                        "invalid_params",
                        format!("Failed to parse task.get params: {}", err),
                        None,
                    );
                }
            };
            let guard = state.lock().await;
            match guard.get_task(&params.request_id).await {
                Some(task) => match serde_json::to_value(task) {
                    Ok(value) => ResponseEnvelope::ok(request.id, value),
                    Err(err) => ResponseEnvelope::err(
                        request.id,
                        "serialize_error",
                        format!("Failed to serialize task: {}", err),
                        None,
                    ),
                },
                None => ResponseEnvelope::err(
                    request.id,
                    "task_not_found",
                    format!("Task '{}' not found", params.request_id),
                    None,
                ),
            }
        }
        "task.list" => {
            let guard = state.lock().await;
            ResponseEnvelope::ok(request.id, json!({ "tasks": guard.list_tasks().await }))
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

async fn start_daemon_watcher(
    state: Arc<Mutex<DaemonState>>,
    watcher_slot: Arc<std::sync::Mutex<Option<notify::RecommendedWatcher>>>,
) -> Result<tokio::sync::mpsc::Sender<Vec<PathBuf>>> {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<PathBuf>>(32);
    let watcher_tx = tx.clone();
    let task_watcher_tx = watcher_tx.clone();
    let state_for_task = Arc::clone(&state);
    let watcher_slot_for_task = Arc::clone(&watcher_slot);

    tokio::spawn(async move {
        while let Some(mut changed_paths) = rx.recv().await {
            tokio::time::sleep(Duration::from_millis(200)).await;
            while let Ok(mut more_paths) = rx.try_recv() {
                changed_paths.append(&mut more_paths);
            }

            let watch_paths = {
                let guard = state_for_task.lock().await;
                guard.watch_paths()
            };

            if !should_rescan_daemon(&watch_paths, &changed_paths) {
                continue;
            }

            info!(
                ?changed_paths,
                "Daemon filesystem rescan triggered by file change"
            );

            if let Err(err) = rescan_and_refresh_watcher(
                Arc::clone(&state_for_task),
                Arc::clone(&watcher_slot_for_task),
                task_watcher_tx.clone(),
            )
            .await
            {
                error!(error = %err, "Daemon filesystem rescan failed");
            }
        }
    });

    let watch_paths = {
        let guard = state.lock().await;
        guard.watch_paths()
    };
    let watcher = build_daemon_watcher(&watch_paths, tx)?;
    let mut slot = watcher_slot
        .lock()
        .expect("daemon watcher mutex poisoned during startup");
    *slot = watcher;

    Ok(watcher_tx)
}

async fn rescan_and_refresh_watcher(
    state: Arc<Mutex<DaemonState>>,
    watcher_slot: Arc<std::sync::Mutex<Option<notify::RecommendedWatcher>>>,
    tx: tokio::sync::mpsc::Sender<Vec<PathBuf>>,
) -> Result<DaemonStatus> {
    let (status, watch_paths) = {
        let mut guard = state.lock().await;
        let status = guard.rescan().await?;
        let watch_paths = guard.watch_paths();
        (status, watch_paths)
    };

    let watcher = build_daemon_watcher(&watch_paths, tx)?;
    let mut slot = watcher_slot
        .lock()
        .expect("daemon watcher mutex poisoned during refresh");
    *slot = watcher;

    Ok(status)
}

fn build_daemon_watcher(
    watch_paths: &DaemonWatchPaths,
    tx: tokio::sync::mpsc::Sender<Vec<PathBuf>>,
) -> Result<Option<notify::RecommendedWatcher>> {
    use notify::{RecursiveMode, Watcher};

    let roots = collect_daemon_watch_roots(watch_paths);
    if roots.is_empty() {
        warn!("No daemon watch roots available, skipping daemon watcher");
        return Ok(None);
    }

    let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| match res {
        Ok(event) => {
            if event.kind.is_modify() || event.kind.is_create() || event.kind.is_remove() {
                let _ = tx.blocking_send(event.paths.clone());
            }
        }
        Err(err) => error!(error = %err, "Daemon watcher channel error"),
    })?;

    for root in roots {
        if !root.path.exists() && root.recursive {
            continue;
        }

        let mode = if root.recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };
        watcher.watch(&root.path, mode)?;
        info!(
            path = %root.path.display(),
            recursive = matches!(mode, RecursiveMode::Recursive),
            "Watching daemon path"
        );
    }

    Ok(Some(watcher))
}

fn should_rescan_daemon(watch_paths: &DaemonWatchPaths, changed_paths: &[PathBuf]) -> bool {
    changed_paths.iter().any(|path| {
        path == &watch_paths.config_path
            || is_agent_toml(path, &watch_paths.agents_dir)
            || is_direct_child(path, &watch_paths.agents_dir)
            || is_direct_child(path, &watch_paths.harnesses_dir)
            || is_agent_harness_dir(path, &watch_paths.agents_dir)
            || path == &watch_paths.agents_dir
            || path == &watch_paths.harnesses_dir
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DaemonWatchRoot {
    path: PathBuf,
    recursive: bool,
}

fn collect_daemon_watch_roots(watch_paths: &DaemonWatchPaths) -> Vec<DaemonWatchRoot> {
    let mut roots = Vec::new();
    push_watch_root(
        &mut roots,
        watch_paths
            .config_path
            .parent()
            .unwrap_or_else(|| Path::new(".")),
        false,
    );
    push_watch_root(
        &mut roots,
        watch_paths
            .agents_dir
            .parent()
            .unwrap_or_else(|| Path::new(".")),
        false,
    );
    push_watch_root(
        &mut roots,
        watch_paths
            .harnesses_dir
            .parent()
            .unwrap_or_else(|| Path::new(".")),
        false,
    );
    if watch_paths.agents_dir.exists() {
        push_watch_root(&mut roots, &watch_paths.agents_dir, true);
    }
    if watch_paths.harnesses_dir.exists() {
        push_watch_root(&mut roots, &watch_paths.harnesses_dir, true);
    }
    roots
}

fn push_watch_root(roots: &mut Vec<DaemonWatchRoot>, path: &Path, recursive: bool) {
    let root = DaemonWatchRoot {
        path: path.to_path_buf(),
        recursive,
    };
    if !roots.contains(&root) {
        roots.push(root);
    }
}

fn is_agent_toml(path: &Path, agents_dir: &Path) -> bool {
    path.file_name().and_then(|name| name.to_str()) == Some("agent.toml")
        && path.starts_with(agents_dir)
}

fn is_direct_child(path: &Path, parent: &Path) -> bool {
    path.parent() == Some(parent)
}

fn is_agent_harness_dir(path: &Path, agents_dir: &Path) -> bool {
    path.file_name().and_then(|name| name.to_str()) == Some("harness")
        && path
            .parent()
            .and_then(Path::parent)
            .is_some_and(|grandparent| grandparent == agents_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rescan_filter_ignores_harness_script_edits_but_tracks_registry_changes() {
        let watch_paths = DaemonWatchPaths {
            config_path: PathBuf::from("/tmp/turin/turin.toml"),
            agents_dir: PathBuf::from("/tmp/turin/agents"),
            harnesses_dir: PathBuf::from("/tmp/turin/harnesses"),
        };

        assert!(should_rescan_daemon(
            &watch_paths,
            &[PathBuf::from("/tmp/turin/turin.toml")]
        ));
        assert!(should_rescan_daemon(
            &watch_paths,
            &[PathBuf::from("/tmp/turin/agents/docs/agent.toml")]
        ));
        assert!(should_rescan_daemon(
            &watch_paths,
            &[PathBuf::from("/tmp/turin/agents/docs")]
        ));
        assert!(should_rescan_daemon(
            &watch_paths,
            &[PathBuf::from("/tmp/turin/agents/docs/harness")]
        ));
        assert!(should_rescan_daemon(
            &watch_paths,
            &[PathBuf::from("/tmp/turin/harnesses/reviewer")]
        ));

        assert!(!should_rescan_daemon(
            &watch_paths,
            &[PathBuf::from("/tmp/turin/agents/docs/harness/main.lua")]
        ));
        assert!(!should_rescan_daemon(
            &watch_paths,
            &[PathBuf::from("/tmp/turin/harnesses/reviewer/main.lua")]
        ));
    }
}
