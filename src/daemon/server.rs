use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use notify::Event;
use serde::Serialize;
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{Mutex, broadcast, watch};
use tracing::{error, info, warn};

use crate::daemon::protocol::{
    BindHarnessParams, CreateAgentParams, DaemonRequest, EntityIdParams, EventEnvelope,
    RequestEnvelope, ResponseEnvelope, SessionIdParams, SessionListParams, SubmitTaskParams,
    TaskIdParams, UpdateAgentParams, WaitTaskParams,
};
use crate::daemon::state::{
    CreateAgentInput, DaemonState, DaemonStatus, DaemonWatchPaths, UpdateAgentInput,
};

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
    let (event_tx, _) = broadcast::channel(512);
    let watcher_slot = Arc::new(std::sync::Mutex::new(None));
    let daemon_watcher_tx = start_daemon_watcher(
        Arc::clone(&state),
        Arc::clone(&watcher_slot),
        event_tx.clone(),
    )
    .await?;
    start_task_event_poller(Arc::clone(&state), event_tx.clone(), shutdown_rx.clone());

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
                        let event_tx = event_tx.clone();
                        let shutdown_tx = shutdown_tx.clone();
                        let shutdown_rx = shutdown_rx.clone();
                        tokio::spawn(async move {
                            if let Err(err) =
                                handle_client(
                                    stream,
                                    state,
                                    watcher_slot,
                                    daemon_watcher_tx,
                                    event_tx,
                                    shutdown_tx,
                                    shutdown_rx,
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
    event_tx: broadcast::Sender<EventEnvelope>,
    shutdown_tx: watch::Sender<bool>,
    shutdown_rx: watch::Receiver<bool>,
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

        if matches!(request.request, DaemonRequest::RuntimeEventsSubscribe(_)) {
            stream_events(
                request,
                Arc::clone(&state),
                event_tx.subscribe(),
                shutdown_rx,
                &mut writer,
            )
            .await?;
            break;
        }

        let response = dispatch(
            request,
            Arc::clone(&state),
            Arc::clone(&watcher_slot),
            daemon_watcher_tx.clone(),
            event_tx.clone(),
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
    event_tx: broadcast::Sender<EventEnvelope>,
    shutdown_tx: watch::Sender<bool>,
) -> ResponseEnvelope {
    match request.request {
        DaemonRequest::DaemonPing(_) => ResponseEnvelope::ok(
            request.id,
            json!({
                "pong": true,
                "version": env!("CARGO_PKG_VERSION"),
            }),
        ),
        DaemonRequest::DaemonStatus(_) => {
            let guard = state.lock().await;
            serialize_response(request.id, guard.status().await, "daemon status")
        }
        DaemonRequest::RuntimeRescan(_) => {
            match rescan_and_refresh_watcher(state, watcher_slot, daemon_watcher_tx, event_tx).await
            {
                Ok(status) => serialize_response(request.id, status, "rescan result"),
                Err(err) => {
                    ResponseEnvelope::err(request.id, "rescan_failed", err.to_string(), None)
                }
            }
        }
        DaemonRequest::RuntimeReload(_) => {
            match rescan_and_refresh_watcher(
                state,
                watcher_slot,
                daemon_watcher_tx,
                event_tx.clone(),
            )
            .await
            {
                Ok(status) => serialize_response_with_event(
                    request.id,
                    status,
                    "reload result",
                    &event_tx,
                    "runtime.reloaded",
                ),
                Err(err) => ResponseEnvelope::err(
                    request.id,
                    "runtime_reload_failed",
                    err.to_string(),
                    None,
                ),
            }
        }
        DaemonRequest::RuntimeErrors(_) => {
            let guard = state.lock().await;
            ResponseEnvelope::ok(request.id, json!({ "issues": guard.runtime_errors() }))
        }
        DaemonRequest::AgentList(_) => {
            let guard = state.lock().await;
            ResponseEnvelope::ok(
                request.id,
                json!({ "agents": guard.registry_snapshot().agents }),
            )
        }
        DaemonRequest::AgentGet(EntityIdParams { id }) => {
            let guard = state.lock().await;
            match guard.agent_detail(&id) {
                Ok(Some(agent)) => serialize_response(request.id, agent, "agent detail"),
                Ok(None) => ResponseEnvelope::err(
                    request.id,
                    "agent_not_found",
                    format!("Agent '{}' not found", id),
                    None,
                ),
                Err(err) => {
                    ResponseEnvelope::err(request.id, "agent_get_failed", err.to_string(), None)
                }
            }
        }
        DaemonRequest::AgentStatus(EntityIdParams { id }) => {
            let guard = state.lock().await;
            match guard.agent_runtime_status(&id).await {
                Ok(Some(status)) => serialize_response(request.id, status, "agent status"),
                Ok(None) => ResponseEnvelope::err(
                    request.id,
                    "agent_not_found",
                    format!("Agent '{}' not found", id),
                    None,
                ),
                Err(err) => {
                    ResponseEnvelope::err(request.id, "agent_status_failed", err.to_string(), None)
                }
            }
        }
        DaemonRequest::AgentIssues(EntityIdParams { id }) => {
            let guard = state.lock().await;
            match guard.agent_issues(&id) {
                Ok(Some(issues)) => {
                    ResponseEnvelope::ok(request.id, json!({ "agent_id": id, "issues": issues }))
                }
                Ok(None) => ResponseEnvelope::err(
                    request.id,
                    "agent_not_found",
                    format!("Agent '{}' not found", id),
                    None,
                ),
                Err(err) => {
                    ResponseEnvelope::err(request.id, "agent_issues_failed", err.to_string(), None)
                }
            }
        }
        DaemonRequest::AgentCreate(CreateAgentParams {
            id,
            provider,
            model,
            system_prompt,
            thinking,
            mode,
            harness,
            idle_grace_secs,
            enabled,
        }) => {
            let mut guard = state.lock().await;
            match guard
                .create_agent(CreateAgentInput {
                    id,
                    provider,
                    model,
                    system_prompt,
                    thinking,
                    mode,
                    harness,
                    idle_grace_secs,
                    enabled,
                })
                .await
            {
                Ok(agent) => serialize_response_with_event(
                    request.id,
                    agent,
                    "created agent",
                    &event_tx,
                    "agent.created",
                ),
                Err(err) => {
                    ResponseEnvelope::err(request.id, "agent_create_failed", err.to_string(), None)
                }
            }
        }
        DaemonRequest::AgentEnable(EntityIdParams { id }) => {
            let mut guard = state.lock().await;
            match guard.set_agent_enabled(&id, true).await {
                Ok(agent) => serialize_response_with_event(
                    request.id,
                    agent,
                    "agent toggle result",
                    &event_tx,
                    "agent.enabled",
                ),
                Err(err) => {
                    ResponseEnvelope::err(request.id, "agent_toggle_failed", err.to_string(), None)
                }
            }
        }
        DaemonRequest::AgentDisable(EntityIdParams { id }) => {
            let mut guard = state.lock().await;
            match guard.set_agent_enabled(&id, false).await {
                Ok(agent) => serialize_response_with_event(
                    request.id,
                    agent,
                    "agent toggle result",
                    &event_tx,
                    "agent.disabled",
                ),
                Err(err) => {
                    ResponseEnvelope::err(request.id, "agent_toggle_failed", err.to_string(), None)
                }
            }
        }
        DaemonRequest::AgentUpdate(UpdateAgentParams {
            id,
            provider,
            model,
            system_prompt,
            thinking,
            mode,
            idle_grace_secs,
        }) => {
            let mut guard = state.lock().await;
            match guard
                .update_agent(
                    &id,
                    UpdateAgentInput {
                        provider,
                        model,
                        system_prompt,
                        thinking,
                        mode,
                        idle_grace_secs,
                    },
                )
                .await
            {
                Ok(agent) => serialize_response_with_event(
                    request.id,
                    agent,
                    "updated agent",
                    &event_tx,
                    "agent.updated",
                ),
                Err(err) => {
                    ResponseEnvelope::err(request.id, "agent_update_failed", err.to_string(), None)
                }
            }
        }
        DaemonRequest::AgentReload(EntityIdParams { id }) => {
            let mut guard = state.lock().await;
            match guard.reload_agent(&id).await {
                Ok(agent) => serialize_response_with_event(
                    request.id,
                    agent,
                    "reloaded agent",
                    &event_tx,
                    "agent.reloaded",
                ),
                Err(err) => {
                    ResponseEnvelope::err(request.id, "agent_reload_failed", err.to_string(), None)
                }
            }
        }
        DaemonRequest::AgentBindHarness(BindHarnessParams { id, harness_id }) => {
            let mut guard = state.lock().await;
            match guard.bind_agent_shared_harness(&id, &harness_id).await {
                Ok(agent) => match serialize_value(&request.id, agent, "rebound agent") {
                    Ok(value) => {
                        emit_event(&event_tx, "agent.updated", value.clone());
                        emit_event(
                            &event_tx,
                            "agent.harness_bound",
                            json!({ "id": id, "harness_id": harness_id }),
                        );
                        ResponseEnvelope::ok(request.id, value)
                    }
                    Err(response) => *response,
                },
                Err(err) => ResponseEnvelope::err(
                    request.id,
                    "agent_bind_harness_failed",
                    err.to_string(),
                    None,
                ),
            }
        }
        DaemonRequest::AgentUseLocalHarness(EntityIdParams { id }) => {
            let mut guard = state.lock().await;
            match guard.use_local_agent_harness(&id).await {
                Ok(agent) => match serialize_value(&request.id, agent, "local-harness agent") {
                    Ok(value) => {
                        emit_event(&event_tx, "agent.updated", value.clone());
                        emit_event(
                            &event_tx,
                            "agent.local_harness_enabled",
                            json!({ "id": id }),
                        );
                        ResponseEnvelope::ok(request.id, value)
                    }
                    Err(response) => *response,
                },
                Err(err) => ResponseEnvelope::err(
                    request.id,
                    "agent_use_local_harness_failed",
                    err.to_string(),
                    None,
                ),
            }
        }
        DaemonRequest::AgentDelete(EntityIdParams { id }) => {
            let mut guard = state.lock().await;
            match guard.delete_agent(&id).await {
                Ok(status) => match serialize_value(&request.id, &status, "delete status") {
                    Ok(value) => {
                        emit_event(&event_tx, "agent.deleted", json!({ "id": id }));
                        emit_event(&event_tx, "runtime.rescanned", value.clone());
                        emit_registry_issue_events(&event_tx, &status);
                        ResponseEnvelope::ok(request.id, value)
                    }
                    Err(response) => *response,
                },
                Err(err) => {
                    ResponseEnvelope::err(request.id, "agent_delete_failed", err.to_string(), None)
                }
            }
        }
        DaemonRequest::TaskSubmit(SubmitTaskParams { agent_id, prompt }) => {
            let guard = state.lock().await;
            match guard.submit_task(&agent_id, prompt).await {
                Ok(task) => serialize_response_with_event(
                    request.id,
                    task,
                    "submitted task",
                    &event_tx,
                    "task.submitted",
                ),
                Err(err) => {
                    ResponseEnvelope::err(request.id, "task_submit_failed", err.to_string(), None)
                }
            }
        }
        DaemonRequest::TaskGet(TaskIdParams { request_id }) => {
            let guard = state.lock().await;
            match guard.get_task(&request_id).await {
                Some(task) => serialize_response(request.id, task, "task"),
                None => ResponseEnvelope::err(
                    request.id,
                    "task_not_found",
                    format!("Task '{}' not found", request_id),
                    None,
                ),
            }
        }
        DaemonRequest::TaskWait(WaitTaskParams {
            request_id,
            timeout_ms,
        }) => {
            let guard = state.lock().await;
            match guard.wait_for_task(&request_id, timeout_ms).await {
                Ok(task) => ResponseEnvelope::ok(request.id, json!(task)),
                Err(err) => {
                    ResponseEnvelope::err(request.id, "task_wait_failed", err.to_string(), None)
                }
            }
        }
        DaemonRequest::TaskCancel(TaskIdParams { request_id }) => {
            let guard = state.lock().await;
            match guard.cancel_task(&request_id).await {
                Ok(task) => {
                    let value = json!(task);
                    let event_name = if value.get("state").and_then(|state| state.as_str())
                        == Some("cancelling")
                    {
                        "task.cancel_requested"
                    } else {
                        "task.cancelled"
                    };
                    emit_event(&event_tx, event_name, value.clone());
                    ResponseEnvelope::ok(request.id, value)
                }
                Err(err) => {
                    ResponseEnvelope::err(request.id, "task_cancel_failed", err.to_string(), None)
                }
            }
        }
        DaemonRequest::TaskList(_) => {
            let guard = state.lock().await;
            ResponseEnvelope::ok(request.id, json!({ "tasks": guard.list_tasks().await }))
        }
        DaemonRequest::SessionList(SessionListParams { limit, offset }) => {
            let guard = state.lock().await;
            match guard.list_sessions(limit, offset).await {
                Ok(sessions) => ResponseEnvelope::ok(request.id, json!({ "sessions": sessions })),
                Err(err) => {
                    ResponseEnvelope::err(request.id, "session_list_failed", err.to_string(), None)
                }
            }
        }
        DaemonRequest::SessionGet(SessionIdParams { session_id }) => {
            let guard = state.lock().await;
            match guard.get_session(&session_id).await {
                Ok(Some(session)) => serialize_response(request.id, session, "session detail"),
                Ok(None) => ResponseEnvelope::err(
                    request.id,
                    "session_not_found",
                    format!("Session '{}' not found", session_id),
                    None,
                ),
                Err(err) => {
                    ResponseEnvelope::err(request.id, "session_get_failed", err.to_string(), None)
                }
            }
        }
        DaemonRequest::SessionCancel(SessionIdParams { session_id }) => {
            let guard = state.lock().await;
            match guard.cancel_session(&session_id).await {
                Ok(result) => {
                    emit_event(&event_tx, "session.cancel_requested", result.clone());
                    ResponseEnvelope::ok(request.id, result)
                }
                Err(err) => ResponseEnvelope::err(
                    request.id,
                    "session_cancel_failed",
                    err.to_string(),
                    None,
                ),
            }
        }
        DaemonRequest::SessionKill(SessionIdParams { session_id }) => {
            let guard = state.lock().await;
            match guard.kill_session(&session_id).await {
                Ok(result) => {
                    emit_event(&event_tx, "session.killed", result.clone());
                    ResponseEnvelope::ok(request.id, result)
                }
                Err(err) => {
                    ResponseEnvelope::err(request.id, "session_kill_failed", err.to_string(), None)
                }
            }
        }
        DaemonRequest::HarnessList(_) => {
            let guard = state.lock().await;
            ResponseEnvelope::ok(
                request.id,
                json!({
                    "harnesses": guard.status().await.harnesses
                }),
            )
        }
        DaemonRequest::HarnessCreate(EntityIdParams { id }) => {
            let mut guard = state.lock().await;
            match guard.create_shared_harness(&id).await {
                Ok(harness) => serialize_response_with_event(
                    request.id,
                    harness,
                    "created harness",
                    &event_tx,
                    "harness.created",
                ),
                Err(err) => ResponseEnvelope::err(
                    request.id,
                    "harness_create_failed",
                    err.to_string(),
                    None,
                ),
            }
        }
        DaemonRequest::HarnessGet(EntityIdParams { id }) => {
            let guard = state.lock().await;
            match guard.harness_detail(&id) {
                Some(harness) => serialize_response(request.id, harness, "harness detail"),
                None => ResponseEnvelope::err(
                    request.id,
                    "harness_not_found",
                    format!("Harness '{}' not found", id),
                    None,
                ),
            }
        }
        DaemonRequest::HarnessIssues(EntityIdParams { id }) => {
            let guard = state.lock().await;
            match guard.harness_issues(&id) {
                Ok(Some(issues)) => {
                    ResponseEnvelope::ok(request.id, json!({ "harness_id": id, "issues": issues }))
                }
                Ok(None) => ResponseEnvelope::err(
                    request.id,
                    "harness_not_found",
                    format!("Harness '{}' not found", id),
                    None,
                ),
                Err(err) => ResponseEnvelope::err(
                    request.id,
                    "harness_issues_failed",
                    err.to_string(),
                    None,
                ),
            }
        }
        DaemonRequest::HarnessReload(EntityIdParams { id }) => {
            let mut guard = state.lock().await;
            match guard.reload_harness(&id).await {
                Ok(harness) => serialize_response_with_event(
                    request.id,
                    harness,
                    "harness reload result",
                    &event_tx,
                    "harness.reloaded",
                ),
                Err(err) => ResponseEnvelope::err(
                    request.id,
                    "harness_reload_failed",
                    err.to_string(),
                    None,
                ),
            }
        }
        DaemonRequest::HarnessValidate(EntityIdParams { id }) => {
            let guard = state.lock().await;
            match guard.validate_harness(&id) {
                Ok(result) => {
                    emit_event(&event_tx, "harness.validated", result.clone());
                    ResponseEnvelope::ok(request.id, result)
                }
                Err(err) => ResponseEnvelope::err(
                    request.id,
                    "harness_validate_failed",
                    err.to_string(),
                    None,
                ),
            }
        }
        DaemonRequest::HarnessDelete(EntityIdParams { id }) => {
            let mut guard = state.lock().await;
            match guard.delete_shared_harness(&id).await {
                Ok(status) => {
                    match serialize_value(&request.id, &status, "harness delete result") {
                        Ok(value) => {
                            emit_event(&event_tx, "harness.deleted", json!({ "id": id }));
                            emit_event(&event_tx, "runtime.rescanned", value.clone());
                            emit_registry_issue_events(&event_tx, &status);
                            ResponseEnvelope::ok(request.id, value)
                        }
                        Err(response) => *response,
                    }
                }
                Err(err) => ResponseEnvelope::err(
                    request.id,
                    "harness_delete_failed",
                    err.to_string(),
                    None,
                ),
            }
        }
        DaemonRequest::RuntimeEventsSubscribe(_) => ResponseEnvelope::err(
            request.id,
            "invalid_operation_context",
            "runtime.events.subscribe must be handled by the event stream path",
            None,
        ),
        DaemonRequest::DaemonStop(_) => {
            emit_event(&event_tx, "daemon.stopping", json!({}));
            let _ = shutdown_tx.send(true);
            ResponseEnvelope::ok(request.id, json!({ "stopping": true }))
        }
    }
}

fn serialize_response<T: Serialize>(
    id: Option<String>,
    value: T,
    context: &str,
) -> ResponseEnvelope {
    match serialize_value(&id, value, context) {
        Ok(value) => ResponseEnvelope::ok(id, value),
        Err(response) => *response,
    }
}

fn serialize_response_with_event<T: Serialize>(
    id: Option<String>,
    value: T,
    context: &str,
    event_tx: &broadcast::Sender<EventEnvelope>,
    event_name: &str,
) -> ResponseEnvelope {
    match serialize_value(&id, value, context) {
        Ok(value) => {
            emit_event(event_tx, event_name, value.clone());
            ResponseEnvelope::ok(id, value)
        }
        Err(response) => *response,
    }
}

fn serialize_value<T: Serialize>(
    id: &Option<String>,
    value: T,
    context: &str,
) -> Result<serde_json::Value, Box<ResponseEnvelope>> {
    serde_json::to_value(value).map_err(|err| {
        Box::new(ResponseEnvelope::err(
            id.clone(),
            "serialize_error",
            format!("Failed to serialize {}: {}", context, err),
            None,
        ))
    })
}

fn emit_event(tx: &broadcast::Sender<EventEnvelope>, event: &str, data: serde_json::Value) {
    let _ = tx.send(EventEnvelope::new(event, data));
}

fn emit_registry_issue_events(tx: &broadcast::Sender<EventEnvelope>, status: &DaemonStatus) {
    for issue in &status.registry.issues {
        if let Ok(data) = serde_json::to_value(issue) {
            emit_event(tx, "runtime.issue", data);
        }
        if let Some((event_name, data)) = classify_registry_issue(status, issue) {
            emit_event(tx, event_name, data);
        }
    }
}

fn classify_registry_issue(
    status: &DaemonStatus,
    issue: &crate::daemon::registry::RegistryIssue,
) -> Option<(&'static str, serde_json::Value)> {
    let issue_path = Path::new(&issue.path);
    let agents_dir = Path::new(&status.registry.agents_dir);
    if let Ok(relative) = issue_path.strip_prefix(agents_dir)
        && let Some(agent_id) = relative.components().next()
    {
        return Some((
            "agent.load_failed",
            json!({
                "agent_id": agent_id.as_os_str().to_string_lossy(),
                "path": issue.path,
                "message": issue.message,
            }),
        ));
    }

    let harnesses_dir = Path::new(&status.registry.harnesses_dir);
    if let Ok(relative) = issue_path.strip_prefix(harnesses_dir)
        && let Some(harness_id) = relative.components().next()
    {
        return Some((
            "harness.load_failed",
            json!({
                "harness_id": harness_id.as_os_str().to_string_lossy(),
                "path": issue.path,
                "message": issue.message,
            }),
        ));
    }

    None
}

async fn stream_events(
    request: RequestEnvelope,
    state: Arc<Mutex<DaemonState>>,
    mut event_rx: broadcast::Receiver<EventEnvelope>,
    mut shutdown_rx: watch::Receiver<bool>,
    writer: &mut tokio::net::unix::OwnedWriteHalf,
) -> Result<()> {
    let ack = ResponseEnvelope::ok(request.id, json!({ "subscribed": true }));
    writer
        .write_all(serde_json::to_string(&ack)?.as_bytes())
        .await?;
    writer.write_all(b"\n").await?;

    let snapshot = {
        let guard = state.lock().await;
        serde_json::to_value(guard.status().await)?
    };
    let snapshot_event = EventEnvelope::new("runtime.snapshot", snapshot);
    writer
        .write_all(serde_json::to_string(&snapshot_event)?.as_bytes())
        .await?;
    writer.write_all(b"\n").await?;

    let status: DaemonStatus = {
        let guard = state.lock().await;
        guard.status().await
    };
    for issue in &status.registry.issues {
        if let Some((event_name, data)) = classify_registry_issue(&status, issue) {
            let event = EventEnvelope::new(event_name, data);
            writer
                .write_all(serde_json::to_string(&event)?.as_bytes())
                .await?;
            writer.write_all(b"\n").await?;
        }
    }

    loop {
        tokio::select! {
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() {
                    break;
                }
            }
            event = event_rx.recv() => {
                match event {
                    Ok(event) => {
                        writer
                            .write_all(serde_json::to_string(&event)?.as_bytes())
                            .await?;
                        writer.write_all(b"\n").await?;
                    }
                    Err(broadcast::error::RecvError::Lagged(skipped)) => {
                        let lagged = EventEnvelope::new("runtime.events_lagged", json!({ "skipped": skipped }));
                        writer
                            .write_all(serde_json::to_string(&lagged)?.as_bytes())
                            .await?;
                        writer.write_all(b"\n").await?;
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        }
    }

    Ok(())
}

fn start_task_event_poller(
    state: Arc<Mutex<DaemonState>>,
    event_tx: broadcast::Sender<EventEnvelope>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    tokio::spawn(async move {
        let mut seen: std::collections::HashMap<String, serde_json::Value> =
            std::collections::HashMap::new();

        loop {
            tokio::select! {
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() {
                        break;
                    }
                }
                _ = tokio::time::sleep(Duration::from_millis(250)) => {
                    let tasks = {
                        let guard = state.lock().await;
                        guard.list_tasks().await
                    };

                    for task in tasks {
                        let Ok(value) = serde_json::to_value(&task) else {
                            continue;
                        };
                        let changed = seen
                            .get(&task.request_id)
                            .map(|previous| previous != &value)
                            .unwrap_or(true);
                        if changed {
                            emit_event(&event_tx, "task.updated", value.clone());
                            seen.insert(task.request_id.clone(), value);
                        }
                    }
                }
            }
        }
    });
}

async fn start_daemon_watcher(
    state: Arc<Mutex<DaemonState>>,
    watcher_slot: Arc<std::sync::Mutex<Option<notify::RecommendedWatcher>>>,
    event_tx: broadcast::Sender<EventEnvelope>,
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
                event_tx.clone(),
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
    event_tx: broadcast::Sender<EventEnvelope>,
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

    emit_event(&event_tx, "runtime.rescanned", json!(status.clone()));
    emit_registry_issue_events(&event_tx, &status);
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
    use crate::daemon::registry::{RegistryIssue, RegistrySnapshot};
    use crate::daemon::state::DaemonStatus;

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

    #[test]
    fn classify_registry_issue_recognizes_agent_and_harness_paths() {
        let status = DaemonStatus {
            config_path: "turin.toml".to_string(),
            workspace_root: ".".to_string(),
            socket_path: ".turin/daemon.sock".to_string(),
            registry: RegistrySnapshot {
                agents_dir: "/tmp/work/agents".to_string(),
                harnesses_dir: "/tmp/work/harnesses".to_string(),
                agents: Vec::new(),
                shared_harnesses: Vec::new(),
                issues: Vec::new(),
            },
            harnesses: Vec::new(),
            agent_runtimes: Vec::new(),
        };

        let agent_issue = RegistryIssue {
            path: "/tmp/work/agents/docs-reviewer/agent.toml".to_string(),
            message: "bad toml".to_string(),
        };
        let harness_issue = RegistryIssue {
            path: "/tmp/work/harnesses/reviewer/main.lua".to_string(),
            message: "bad lua".to_string(),
        };

        let (agent_event, agent_data) =
            classify_registry_issue(&status, &agent_issue).expect("agent issue classified");
        assert_eq!(agent_event, "agent.load_failed");
        assert_eq!(agent_data["agent_id"], "docs-reviewer");

        let (harness_event, harness_data) =
            classify_registry_issue(&status, &harness_issue).expect("harness issue classified");
        assert_eq!(harness_event, "harness.load_failed");
        assert_eq!(harness_data["harness_id"], "reviewer");
    }
}
