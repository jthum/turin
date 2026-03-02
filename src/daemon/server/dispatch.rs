use std::path::Path;
use std::sync::Arc;

use serde::Serialize;
use serde_json::json;
use tokio::sync::{RwLock, broadcast, watch};

use crate::daemon::protocol::{
    BindHarnessParams, CreateAgentParams, DaemonRequest, EntityIdParams, EventEnvelope,
    RequestEnvelope, ResponseEnvelope, SessionIdParams, SessionListParams, SubmitTaskParams,
    TaskIdParams, UpdateAgentParams, WaitTaskParams,
};
use crate::daemon::state::{CreateAgentInput, DaemonState, DaemonStatus, UpdateAgentInput};

use super::watch::rescan_and_refresh_watcher;

pub(super) async fn dispatch(
    request: RequestEnvelope,
    state: Arc<RwLock<DaemonState>>,
    watcher_slot: Arc<std::sync::Mutex<Option<notify::RecommendedWatcher>>>,
    daemon_watcher_tx: tokio::sync::mpsc::Sender<Vec<std::path::PathBuf>>,
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
            let guard = state.read().await;
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
            let guard = state.read().await;
            ResponseEnvelope::ok(request.id, json!({ "issues": guard.runtime_errors() }))
        }
        DaemonRequest::AgentList(_) => {
            let guard = state.read().await;
            ResponseEnvelope::ok(
                request.id,
                json!({ "agents": guard.registry_snapshot().agents }),
            )
        }
        DaemonRequest::AgentGet(EntityIdParams { id }) => {
            let guard = state.read().await;
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
            let guard = state.read().await;
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
            let guard = state.read().await;
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
            let mut guard = state.write().await;
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
            let mut guard = state.write().await;
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
            let mut guard = state.write().await;
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
            let mut guard = state.write().await;
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
            let mut guard = state.write().await;
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
            let mut guard = state.write().await;
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
            let mut guard = state.write().await;
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
            let mut guard = state.write().await;
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
            let guard = state.read().await;
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
            let guard = state.read().await;
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
            let guard = state.read().await;
            match guard.wait_for_task(&request_id, timeout_ms).await {
                Ok(task) => ResponseEnvelope::ok(request.id, json!(task)),
                Err(err) => {
                    ResponseEnvelope::err(request.id, "task_wait_failed", err.to_string(), None)
                }
            }
        }
        DaemonRequest::TaskCancel(TaskIdParams { request_id }) => {
            let guard = state.read().await;
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
            let guard = state.read().await;
            ResponseEnvelope::ok(request.id, json!({ "tasks": guard.list_tasks().await }))
        }
        DaemonRequest::SessionList(SessionListParams { limit, offset }) => {
            let guard = state.read().await;
            match guard.list_sessions(limit, offset).await {
                Ok(sessions) => ResponseEnvelope::ok(request.id, json!({ "sessions": sessions })),
                Err(err) => {
                    ResponseEnvelope::err(request.id, "session_list_failed", err.to_string(), None)
                }
            }
        }
        DaemonRequest::SessionGet(SessionIdParams { session_id }) => {
            let guard = state.read().await;
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
            let guard = state.read().await;
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
            let guard = state.read().await;
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
            let guard = state.read().await;
            ResponseEnvelope::ok(
                request.id,
                json!({
                    "harnesses": guard.status().await.harnesses
                }),
            )
        }
        DaemonRequest::HarnessCreate(EntityIdParams { id }) => {
            let mut guard = state.write().await;
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
            let guard = state.read().await;
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
            let guard = state.read().await;
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
            let mut guard = state.write().await;
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
            let guard = state.read().await;
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
            let mut guard = state.write().await;
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

pub(super) fn emit_event(
    tx: &broadcast::Sender<EventEnvelope>,
    event: &str,
    data: serde_json::Value,
) {
    let _ = tx.send(EventEnvelope::new(event, data));
}

pub(super) fn emit_registry_issue_events(
    tx: &broadcast::Sender<EventEnvelope>,
    status: &DaemonStatus,
) {
    for issue in &status.registry.issues {
        if let Ok(data) = serde_json::to_value(issue) {
            emit_event(tx, "runtime.issue", data);
        }
        if let Some((event_name, data)) = classify_registry_issue(status, issue) {
            emit_event(tx, event_name, data);
        }
    }
}

pub(super) fn classify_registry_issue(
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
