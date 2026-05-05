use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Serialize;
use serde_json::json;
use tokio::sync::{RwLock, broadcast, watch};

use crate::daemon::channels::ChannelRuntimeManager;
use crate::daemon::protocol::{
    DaemonRequest, ErrorCode, EventEnvelope, RequestEnvelope, ResponseEnvelope,
};
use crate::daemon::state::{DaemonRuntimeSnapshot, DaemonState, DaemonStatus};

mod agent;
mod channel;
mod daemon;
mod harness;
mod runtime;
mod schedule;
mod session;
mod task;

pub(super) struct DispatchContext {
    pub(super) state: Arc<RwLock<DaemonState>>,
    pub(super) watcher_slot: Arc<std::sync::Mutex<Option<notify::RecommendedWatcher>>>,
    pub(super) daemon_watcher_tx: tokio::sync::mpsc::Sender<Vec<PathBuf>>,
    pub(super) event_tx: broadcast::Sender<EventEnvelope>,
    pub(super) channel_runtimes: Arc<ChannelRuntimeManager>,
    pub(super) shutdown_tx: watch::Sender<bool>,
}

pub(super) async fn dispatch(
    request: RequestEnvelope,
    state: Arc<RwLock<DaemonState>>,
    watcher_slot: Arc<std::sync::Mutex<Option<notify::RecommendedWatcher>>>,
    daemon_watcher_tx: tokio::sync::mpsc::Sender<Vec<PathBuf>>,
    event_tx: broadcast::Sender<EventEnvelope>,
    channel_runtimes: Arc<ChannelRuntimeManager>,
    shutdown_tx: watch::Sender<bool>,
) -> ResponseEnvelope {
    let RequestEnvelope { id, request } = request;
    let context = DispatchContext {
        state,
        watcher_slot,
        daemon_watcher_tx,
        event_tx,
        channel_runtimes,
        shutdown_tx,
    };

    match request {
        DaemonRequest::DaemonPing(params) => daemon::ping(id, params, &context).await,
        DaemonRequest::DaemonStatus(params) => daemon::status(id, params, &context).await,
        DaemonRequest::DaemonStop(params) => daemon::stop(id, params, &context).await,
        DaemonRequest::RuntimeRescan(params) => runtime::rescan(id, params, &context).await,
        DaemonRequest::RuntimeReload(params) => runtime::reload(id, params, &context).await,
        DaemonRequest::RuntimeErrors(params) => runtime::errors(id, params, &context).await,
        DaemonRequest::RuntimeEventsSubscribe(_) => ResponseEnvelope::err(
            id,
            ErrorCode::UnsupportedOperation,
            "runtime.events.subscribe must be handled by the event stream path",
            None,
        ),
        DaemonRequest::AgentList(params) => agent::list(id, params, &context).await,
        DaemonRequest::AgentGet(params) => agent::get(id, params, &context).await,
        DaemonRequest::AgentStatus(params) => agent::status(id, params, &context).await,
        DaemonRequest::AgentIssues(params) => agent::issues(id, params, &context).await,
        DaemonRequest::AgentCreate(params) => agent::create(id, params, &context).await,
        DaemonRequest::AgentEnable(params) => agent::enable(id, params, &context).await,
        DaemonRequest::AgentDisable(params) => agent::disable(id, params, &context).await,
        DaemonRequest::AgentUpdate(params) => agent::update(id, params, &context).await,
        DaemonRequest::AgentReload(params) => agent::reload(id, params, &context).await,
        DaemonRequest::AgentBindHarness(params) => agent::bind_harness(id, params, &context).await,
        DaemonRequest::AgentUseLocalHarness(params) => {
            agent::use_local_harness(id, params, &context).await
        }
        DaemonRequest::AgentDelete(params) => agent::delete(id, params, &context).await,
        DaemonRequest::TaskSubmit(params) => task::submit(id, params, &context).await,
        DaemonRequest::TaskSidestep(params) => task::sidestep(id, params, &context).await,
        DaemonRequest::TaskGet(params) => task::get(id, params, &context).await,
        DaemonRequest::TaskWait(params) => task::wait(id, params, &context).await,
        DaemonRequest::TaskPromote(params) => task::promote(id, params, &context).await,
        DaemonRequest::TaskCancel(params) => task::cancel(id, params, &context).await,
        DaemonRequest::TaskList(params) => task::list(id, params, &context).await,
        DaemonRequest::ScheduleCreate(params) => schedule::create(id, params, &context).await,
        DaemonRequest::ScheduleList(params) => schedule::list(id, params, &context).await,
        DaemonRequest::SessionList(params) => session::list(id, params, &context).await,
        DaemonRequest::SessionListLive(params) => session::list_live(id, params, &context).await,
        DaemonRequest::SessionSearch(params) => session::search(id, params, &context).await,
        DaemonRequest::SessionOpen(params) => session::open(id, params, &context).await,
        DaemonRequest::SessionResume(params) => session::resume(id, params, &context).await,
        DaemonRequest::SessionGet(params) => session::get(id, params, &context).await,
        DaemonRequest::SessionSetTitle(params) => session::set_title(id, params, &context).await,
        DaemonRequest::SessionBranchList(params) => {
            session::branch_list(id, params, &context).await
        }
        DaemonRequest::SessionBranchCreate(params) => {
            session::branch_create(id, params, &context).await
        }
        DaemonRequest::SessionBranchCheckout(params) => {
            session::branch_checkout(id, params, &context).await
        }
        DaemonRequest::SessionBranchSiblings(params) => {
            session::branch_siblings(id, params, &context).await
        }
        DaemonRequest::SessionCancel(params) => session::cancel(id, params, &context).await,
        DaemonRequest::SessionKill(params) => session::kill(id, params, &context).await,
        DaemonRequest::HarnessList(params) => harness::list(id, params, &context).await,
        DaemonRequest::HarnessCreate(params) => harness::create(id, params, &context).await,
        DaemonRequest::HarnessGet(params) => harness::get(id, params, &context).await,
        DaemonRequest::HarnessIssues(params) => harness::issues(id, params, &context).await,
        DaemonRequest::HarnessReload(params) => harness::reload(id, params, &context).await,
        DaemonRequest::HarnessValidate(params) => harness::validate(id, params, &context).await,
        DaemonRequest::HarnessDelete(params) => harness::delete(id, params, &context).await,
        DaemonRequest::ChannelList(params) => channel::list(id, params, &context).await,
        DaemonRequest::ChannelCreate(params) => channel::create(id, params, &context).await,
        DaemonRequest::ChannelGet(params) => channel::get(id, params, &context).await,
        DaemonRequest::ChannelStatus(params) => channel::status(id, params, &context).await,
        DaemonRequest::ChannelIssues(params) => channel::issues(id, params, &context).await,
        DaemonRequest::ChannelEnable(params) => channel::enable(id, params, &context).await,
        DaemonRequest::ChannelDisable(params) => channel::disable(id, params, &context).await,
        DaemonRequest::ChannelUpdate(params) => channel::update(id, params, &context).await,
        DaemonRequest::ChannelAccessGet(params) => channel::access_get(id, params, &context).await,
        DaemonRequest::ChannelAccessApprove(params) => {
            channel::access_approve(id, params, &context).await
        }
        DaemonRequest::ChannelAccessReject(params) => {
            channel::access_reject(id, params, &context).await
        }
        DaemonRequest::ChannelAccessRevoke(params) => {
            channel::access_revoke(id, params, &context).await
        }
        DaemonRequest::ChannelRunnerHello(params) => {
            channel::runner_hello(id, params, &context).await
        }
        DaemonRequest::ChannelRunnerHeartbeat(params) => {
            channel::runner_heartbeat(id, params, &context).await
        }
        DaemonRequest::ChannelDelete(params) => channel::delete(id, params, &context).await,
    }
}

pub(super) fn serialize_response<T: Serialize>(
    id: Option<String>,
    value: T,
    context: &str,
) -> ResponseEnvelope {
    match serialize_value(&id, value, context) {
        Ok(value) => ResponseEnvelope::ok(id, value),
        Err(response) => *response,
    }
}

pub(super) fn serialize_response_with_event<T: Serialize>(
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

pub(super) fn serialize_value<T: Serialize>(
    id: &Option<String>,
    value: T,
    context: &str,
) -> Result<serde_json::Value, Box<ResponseEnvelope>> {
    serde_json::to_value(value).map_err(|err| {
        Box::new(ResponseEnvelope::err(
            id.clone(),
            ErrorCode::InternalError,
            format!("Failed to serialize {}: {}", context, err),
            None,
        ))
    })
}

pub(super) fn not_found_error(
    id: Option<String>,
    code: ErrorCode,
    message: impl Into<String>,
) -> ResponseEnvelope {
    ResponseEnvelope::err(id, code, message, None)
}

pub(super) fn validation_error(
    id: Option<String>,
    err: impl std::fmt::Display,
) -> ResponseEnvelope {
    ResponseEnvelope::err(id, ErrorCode::ValidationFailed, err.to_string(), None)
}

pub(super) fn resource_busy_error(
    id: Option<String>,
    err: impl std::fmt::Display,
) -> ResponseEnvelope {
    ResponseEnvelope::err(id, ErrorCode::ResourceBusy, err.to_string(), None)
}

pub(super) fn internal_error(id: Option<String>, err: impl std::fmt::Display) -> ResponseEnvelope {
    ResponseEnvelope::err(id, ErrorCode::InternalError, err.to_string(), None)
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

pub(super) async fn build_runtime_snapshot(
    state: &Arc<RwLock<DaemonState>>,
    channel_runtimes: &ChannelRuntimeManager,
) -> DaemonRuntimeSnapshot {
    let status = {
        let guard = state.read().await;
        guard.status().await
    };
    let channel_runtimes = channel_runtimes.list().await;
    DaemonRuntimeSnapshot::from_parts(status, channel_runtimes)
}

pub(super) async fn sync_channel_runtimes(ctx: &DispatchContext) -> Result<(), anyhow::Error> {
    let (workspace_root, channels) = {
        let guard = ctx.state.read().await;
        (
            PathBuf::from(&guard.bootstrap_config.kernel.workspace_root),
            guard.registry_load.channels.clone(),
        )
    };
    ctx.channel_runtimes.sync(workspace_root, channels).await
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

    let channels_dir = Path::new(&status.registry.channels_dir);
    if let Ok(relative) = issue_path.strip_prefix(channels_dir)
        && let Some(channel_id) = relative.components().next()
    {
        return Some((
            "channel.load_failed",
            json!({
                "channel_id": channel_id.as_os_str().to_string_lossy(),
                "path": issue.path,
                "message": issue.message,
            }),
        ));
    }

    None
}
