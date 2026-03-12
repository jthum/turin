use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use turin_daemon_client::{DaemonClient, DaemonHealth, DaemonHealthState};
use turin_local_ipc::{connect as connect_local_ipc, split as split_local_ipc};

use turin::daemon::protocol::{
    DaemonRequest, ErrorCode, ErrorEnvelope, EventEnvelope, RequestEnvelope, ResponseEnvelope,
    RuntimeEventsSubscribeParams,
};
use turin::kernel::config::TurinConfig;

#[derive(Debug, Deserialize)]
struct DaemonStatusView {
    config_path: String,
    workspace_root: String,
    endpoint: String,
    registry: RegistrySnapshotView,
    harnesses: Vec<HarnessRuntimeView>,
    agent_runtimes: Vec<AgentRuntimeView>,
}

#[derive(Debug, Deserialize)]
struct RegistrySnapshotView {
    agents: Vec<AgentSummaryView>,
    shared_harnesses: Vec<SharedHarnessView>,
    channels: Vec<ChannelSummaryView>,
    issues: Vec<IssueView>,
}

#[derive(Debug, Deserialize)]
struct AgentSummaryView {
    id: String,
    enabled: bool,
    provider: String,
    model: String,
    harness_ref: String,
}

#[derive(Debug, Deserialize)]
struct SharedHarnessView {
    id: String,
}

#[derive(Debug, Deserialize)]
struct ChannelSummaryView {
    id: String,
    enabled: bool,
    kind: String,
    agent_id: String,
}

#[derive(Debug, Deserialize)]
struct HarnessRuntimeView {
    harness_id: String,
    bound_agents: Vec<String>,
    watched_roots: Vec<String>,
    loaded_scripts: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct AgentRuntimeView {
    agent_id: String,
    running: bool,
    active_tasks: usize,
    queued_tasks: usize,
    awaiting_results: usize,
    current_session_id: Option<String>,
    current_request_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct IssueView {
    path: String,
    message: String,
}

#[derive(Debug, Deserialize)]
struct IssueListView {
    issues: Vec<IssueView>,
}

#[derive(Debug, Deserialize)]
struct AgentDetailView {
    id: String,
    directory: String,
    enabled: bool,
    provider: String,
    model: String,
    system_prompt: Option<String>,
    mode: Option<String>,
    harness: Option<String>,
    idle_grace_secs: Option<u64>,
    has_local_harness: bool,
}

#[derive(Debug, Deserialize)]
struct HarnessDetailView {
    harness_id: String,
    directory: String,
    bound_agents: Vec<String>,
    watched_roots: Vec<String>,
    loaded_scripts: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ChannelDetailView {
    id: String,
    directory: String,
    enabled: bool,
    kind: String,
    agent_id: String,
    idle_ttl_secs: Option<u64>,
    settings: Value,
}

#[derive(Debug, Deserialize)]
struct ChannelRuntimeView {
    id: String,
    kind: String,
    agent_id: String,
    directory: String,
    state: String,
    last_error: Option<String>,
    last_error_code: Option<String>,
    start_count: u64,
    restart_count: u64,
    failure_count: u64,
    last_transition_unix_ms: u64,
    last_started_unix_ms: Option<u64>,
    last_stopped_unix_ms: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct TaskStatusView {
    request_id: String,
    agent_id: String,
    slot_id: String,
    trace_id: String,
    state: String,
    runtime_task_id: Option<String>,
    status: Option<String>,
    task_turn_count: Option<u32>,
    output: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TaskListView {
    tasks: Vec<TaskStatusView>,
}

#[derive(Debug, Deserialize)]
struct SessionSummaryView {
    internal_id: i64,
    session_id: String,
    agent_id: String,
    metadata: Option<Value>,
    created_at: String,
}

#[derive(Debug, Deserialize)]
struct SessionListView {
    sessions: Vec<SessionSummaryView>,
}

#[derive(Debug, Deserialize)]
struct LiveSessionView {
    agent_id: String,
    slot_id: String,
    session_id: String,
    running: bool,
    active_tasks: usize,
    queued_tasks: usize,
    current_request_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LiveSessionListView {
    sessions: Vec<LiveSessionView>,
}

#[derive(Debug, Deserialize)]
struct SessionEventDetailView {
    id: i64,
    event_type: String,
    payload: Value,
    created_at: String,
}

#[derive(Debug, Deserialize)]
struct SessionMessageDetailView {
    id: i64,
    turn_index: u32,
    role: String,
    content: Value,
    token_count: Option<u64>,
    created_at: String,
}

#[derive(Debug, Deserialize)]
struct SessionToolExecutionDetailView {
    id: i64,
    turn_index: u32,
    tool_call_id: String,
    tool_name: String,
    args: Value,
    output: Option<Value>,
    is_error: bool,
    duration_ms: Option<u64>,
    verdict: String,
    created_at: String,
}

#[derive(Debug, Deserialize)]
struct SessionDetailView {
    session: SessionSummaryView,
    events: Vec<SessionEventDetailView>,
    messages: Vec<SessionMessageDetailView>,
    tool_executions: Vec<SessionToolExecutionDetailView>,
}

#[derive(Debug, Serialize)]
struct DaemonHealthReport {
    state: String,
    ready: bool,
    endpoint: String,
    error: Option<String>,
    version: Option<String>,
    protocol_version: Option<u32>,
    transport: Option<String>,
    wire_format: Option<String>,
    issue_count: usize,
    agent_count: usize,
    harness_count: usize,
    channel_count: usize,
    running_agent_count: usize,
    active_task_count: usize,
    queued_task_count: usize,
    awaiting_result_count: usize,
    channel_runtime_count: usize,
    failed_channel_count: usize,
}

#[derive(Debug, Serialize)]
struct DaemonStartReport {
    started: bool,
    endpoint: String,
    log_path: String,
    health: DaemonHealthReport,
}

#[derive(Debug, Serialize)]
struct DaemonLogReport {
    path: String,
    exists: bool,
    lines: Vec<String>,
}

pub async fn run_start(
    config_path: &Path,
    background: bool,
    wait_timeout_ms: u64,
    json_output: bool,
    log_level: &str,
    log_file_override: Option<&Path>,
) -> Result<()> {
    if !background {
        return turin::daemon::server::serve(config_path).await;
    }

    let report = ensure_background_daemon(
        config_path,
        Duration::from_millis(wait_timeout_ms),
        Duration::from_millis(100),
        log_level,
        log_file_override,
    )
    .await?;
    print_start_report(report, json_output)
}

pub async fn run_health(config_path: &Path, json_output: bool) -> Result<()> {
    let report = daemon_health_report(config_path).await?;
    print_health_report(&report, json_output)
}

pub async fn run_wait(
    config_path: &Path,
    timeout_ms: u64,
    poll_interval_ms: u64,
    json_output: bool,
) -> Result<()> {
    let client = daemon_client_from_config(config_path)?;
    client
        .wait_until_ready(
            Duration::from_millis(timeout_ms),
            Duration::from_millis(poll_interval_ms),
        )
        .await
        .map_err(|err| wrap_daemon_client_error(config_path, err))?;
    let report = daemon_health_report(config_path).await?;
    print_health_report(&report, json_output)
}

pub async fn run_ensure(
    config_path: &Path,
    timeout_ms: u64,
    poll_interval_ms: u64,
    json_output: bool,
    log_level: &str,
    log_file_override: Option<&Path>,
) -> Result<()> {
    let report = ensure_background_daemon(
        config_path,
        Duration::from_millis(timeout_ms),
        Duration::from_millis(poll_interval_ms),
        log_level,
        log_file_override,
    )
    .await?;
    print_start_report(report, json_output)
}

pub async fn run_logs(
    config_path: &Path,
    lines: usize,
    path_only: bool,
    json_output: bool,
    log_file_override: Option<&Path>,
) -> Result<()> {
    let log_path = resolve_daemon_log_path(config_path, log_file_override)?;
    let exists = log_path.exists();
    let rendered_lines = if path_only || !exists {
        Vec::new()
    } else {
        tail_lines(&log_path, lines)?
    };
    let report = DaemonLogReport {
        path: log_path.display().to_string(),
        exists,
        lines: rendered_lines,
    };

    if json_output {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    if path_only {
        println!("{}", report.path);
        return Ok(());
    }

    println!("Log Path: {}", report.path);
    println!("Exists:   {}", yes_no(report.exists));
    if !report.exists {
        println!("No daemon log file exists yet.");
        return Ok(());
    }
    if report.lines.is_empty() {
        println!("No log lines available.");
        return Ok(());
    }
    println!();
    for line in report.lines {
        println!("{}", line);
    }
    Ok(())
}

pub async fn run_ping(config_path: &Path, json_output: bool) -> Result<()> {
    let response = send_request(config_path, "daemon.ping", json!({})).await?;
    print_response(response, json_output)
}

pub async fn run_status(config_path: &Path, json_output: bool) -> Result<()> {
    let response = send_request(config_path, "daemon.status", json!({})).await?;
    if json_output {
        return print_response(response, true);
    }

    let status: DaemonStatusView = decode_result(response)?;
    print_daemon_status(status);
    Ok(())
}

pub async fn run_rescan(config_path: &std::path::Path, json_output: bool) -> Result<()> {
    let response = send_request(config_path, "runtime.rescan", json!({})).await?;
    print_response(response, json_output)
}

pub async fn run_reload(config_path: &std::path::Path, json_output: bool) -> Result<()> {
    let response = send_request(config_path, "runtime.reload", json!({})).await?;
    print_response(response, json_output)
}

pub async fn run_agent_list(config_path: &std::path::Path, json_output: bool) -> Result<()> {
    let response = send_request(config_path, "daemon.status", json!({})).await?;
    if json_output {
        return print_response(response, true);
    }

    let status: DaemonStatusView = decode_result(response)?;
    print_agent_list(status);
    Ok(())
}

pub async fn run_agent_get(
    config_path: &std::path::Path,
    agent_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "agent.get", json!({ "id": agent_id })).await?;
    if json_output {
        return print_response(response, true);
    }

    let agent: AgentDetailView = decode_result(response)?;
    print_agent_detail(agent);
    Ok(())
}

pub async fn run_agent_status(
    config_path: &std::path::Path,
    agent_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "agent.status", json!({ "id": agent_id })).await?;
    if json_output {
        return print_response(response, true);
    }

    let status: AgentRuntimeView = decode_result(response)?;
    print_agent_runtime_status(status);
    Ok(())
}

pub async fn run_agent_issues(
    config_path: &std::path::Path,
    agent_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "agent.issues", json!({ "id": agent_id })).await?;
    if json_output {
        return print_response(response, true);
    }

    let issues: IssueListView = decode_result(response)?;
    print_issue_list(&format!("Agent '{}' issues", agent_id), &issues.issues);
    Ok(())
}

pub async fn run_agent_create(
    config_path: &std::path::Path,
    params: Value,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "agent.create", params).await?;
    print_response(response, json_output)
}

pub async fn run_agent_enable(
    config_path: &std::path::Path,
    agent_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "agent.enable", json!({ "id": agent_id })).await?;
    print_response(response, json_output)
}

pub async fn run_agent_disable(
    config_path: &std::path::Path,
    agent_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "agent.disable", json!({ "id": agent_id })).await?;
    print_response(response, json_output)
}

pub async fn run_agent_update(
    config_path: &std::path::Path,
    params: Value,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "agent.update", params).await?;
    print_response(response, json_output)
}

pub async fn run_agent_reload(
    config_path: &std::path::Path,
    agent_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "agent.reload", json!({ "id": agent_id })).await?;
    print_response(response, json_output)
}

pub async fn run_agent_bind_harness(
    config_path: &std::path::Path,
    agent_id: &str,
    harness_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(
        config_path,
        "agent.bind_harness",
        json!({ "id": agent_id, "harness_id": harness_id }),
    )
    .await?;
    print_response(response, json_output)
}

pub async fn run_agent_use_local_harness(
    config_path: &std::path::Path,
    agent_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(
        config_path,
        "agent.use_local_harness",
        json!({ "id": agent_id }),
    )
    .await?;
    print_response(response, json_output)
}

pub async fn run_agent_delete(
    config_path: &std::path::Path,
    agent_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "agent.delete", json!({ "id": agent_id })).await?;
    print_response(response, json_output)
}

pub async fn run_task_submit(
    config_path: &std::path::Path,
    agent_id: Option<&str>,
    session_id: Option<&str>,
    prompt: &str,
    wait: bool,
    timeout_ms: Option<u64>,
    json_output: bool,
) -> Result<()> {
    let response = send_request(
        config_path,
        "task.submit",
        json!({ "agent_id": agent_id, "session_id": session_id, "prompt": prompt }),
    )
    .await?;
    if !wait {
        if json_output {
            return print_response(response, true);
        }
        let task: TaskStatusView = decode_result(response)?;
        print_task_status("Submitted task", &task);
        return Ok(());
    }

    if !response.ok {
        return print_response(response, json_output);
    }

    let request_id = response
        .result
        .as_ref()
        .and_then(|result| result.get("request_id"))
        .and_then(|value| value.as_str())
        .ok_or_else(|| anyhow::anyhow!("Daemon task.submit response did not include request_id"))?;

    run_task_wait(config_path, request_id, timeout_ms, json_output).await
}

pub async fn run_session_open(
    config_path: &std::path::Path,
    agent_id: &str,
    slot_id: Option<&str>,
    json_output: bool,
) -> Result<()> {
    let response = send_request(
        config_path,
        "session.open",
        json!({ "agent_id": agent_id, "slot_id": slot_id }),
    )
    .await?;
    if json_output {
        return print_response(response, true);
    }

    let session: LiveSessionView = decode_result(response)?;
    print_live_session("Opened live session", &session);
    Ok(())
}

pub async fn run_session_resume(
    config_path: &std::path::Path,
    session_id: &str,
    slot_id: Option<&str>,
    json_output: bool,
) -> Result<()> {
    let response = send_request(
        config_path,
        "session.resume",
        json!({ "session_id": session_id, "slot_id": slot_id }),
    )
    .await?;
    if json_output {
        return print_response(response, true);
    }

    let session: LiveSessionView = decode_result(response)?;
    print_live_session("Resumed live session", &session);
    Ok(())
}

pub async fn run_session_list_live(config_path: &std::path::Path, json_output: bool) -> Result<()> {
    let response = send_request(config_path, "session.list_live", json!({})).await?;
    if json_output {
        return print_response(response, true);
    }

    let sessions: LiveSessionListView = decode_result(response)?;
    print_live_session_list(sessions);
    Ok(())
}

pub async fn run_task_get(
    config_path: &std::path::Path,
    request_id: &str,
    json_output: bool,
) -> Result<()> {
    let response =
        send_request(config_path, "task.get", json!({ "request_id": request_id })).await?;
    if json_output {
        return print_response(response, true);
    }

    let task: TaskStatusView = decode_result(response)?;
    print_task_status("Task", &task);
    Ok(())
}

pub async fn run_task_wait(
    config_path: &std::path::Path,
    request_id: &str,
    timeout_ms: Option<u64>,
    json_output: bool,
) -> Result<()> {
    let response = send_request(
        config_path,
        "task.wait",
        json!({ "request_id": request_id, "timeout_ms": timeout_ms }),
    )
    .await?;
    if json_output {
        return print_response(response, true);
    }

    let task: TaskStatusView = decode_result(response)?;
    print_task_status("Task", &task);
    Ok(())
}

pub async fn run_task_cancel(
    config_path: &std::path::Path,
    request_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(
        config_path,
        "task.cancel",
        json!({ "request_id": request_id }),
    )
    .await?;
    if json_output {
        return print_response(response, true);
    }

    let task: TaskStatusView = decode_result(response)?;
    print_task_status("Cancelled task", &task);
    Ok(())
}

pub async fn run_task_list(config_path: &std::path::Path, json_output: bool) -> Result<()> {
    let response = send_request(config_path, "task.list", json!({})).await?;
    if json_output {
        return print_response(response, true);
    }

    let tasks: TaskListView = decode_result(response)?;
    print_task_list(tasks);
    Ok(())
}

pub async fn run_session_list(
    config_path: &std::path::Path,
    limit: usize,
    offset: usize,
    json_output: bool,
) -> Result<()> {
    let response = send_request(
        config_path,
        "session.list",
        json!({ "limit": limit, "offset": offset }),
    )
    .await?;
    if json_output {
        return print_response(response, true);
    }

    let sessions: SessionListView = decode_result(response)?;
    print_session_list(sessions);
    Ok(())
}

pub async fn run_session_get(
    config_path: &std::path::Path,
    session_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(
        config_path,
        "session.get",
        json!({ "session_id": session_id }),
    )
    .await?;
    if json_output {
        return print_response(response, true);
    }

    let session: SessionDetailView = decode_result(response)?;
    print_session_detail(session);
    Ok(())
}

pub async fn run_session_cancel(
    config_path: &std::path::Path,
    session_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(
        config_path,
        "session.cancel",
        json!({ "session_id": session_id }),
    )
    .await?;
    print_response(response, json_output)
}

pub async fn run_session_kill(
    config_path: &std::path::Path,
    session_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(
        config_path,
        "session.kill",
        json!({ "session_id": session_id }),
    )
    .await?;
    print_response(response, json_output)
}

pub async fn run_harness_list(config_path: &std::path::Path, json_output: bool) -> Result<()> {
    let response = send_request(config_path, "daemon.status", json!({})).await?;
    if json_output {
        return print_response(response, true);
    }

    let status: DaemonStatusView = decode_result(response)?;
    print_harness_list(status);
    Ok(())
}

pub async fn run_harness_create(
    config_path: &std::path::Path,
    harness_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "harness.create", json!({ "id": harness_id })).await?;
    print_response(response, json_output)
}

pub async fn run_harness_get(
    config_path: &std::path::Path,
    harness_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "harness.get", json!({ "id": harness_id })).await?;
    if json_output {
        return print_response(response, true);
    }

    let harness: HarnessDetailView = decode_result(response)?;
    print_harness_detail(harness);
    Ok(())
}

pub async fn run_harness_issues(
    config_path: &std::path::Path,
    harness_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "harness.issues", json!({ "id": harness_id })).await?;
    if json_output {
        return print_response(response, true);
    }

    let issues: IssueListView = decode_result(response)?;
    print_issue_list(&format!("Harness '{}' issues", harness_id), &issues.issues);
    Ok(())
}

pub async fn run_harness_reload(
    config_path: &std::path::Path,
    harness_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "harness.reload", json!({ "id": harness_id })).await?;
    print_response(response, json_output)
}

pub async fn run_harness_validate(
    config_path: &std::path::Path,
    harness_id: &str,
    json_output: bool,
) -> Result<()> {
    let response =
        send_request(config_path, "harness.validate", json!({ "id": harness_id })).await?;
    print_response(response, json_output)
}

pub async fn run_harness_delete(
    config_path: &std::path::Path,
    harness_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "harness.delete", json!({ "id": harness_id })).await?;
    print_response(response, json_output)
}

pub async fn run_channel_list(config_path: &std::path::Path, json_output: bool) -> Result<()> {
    let response = send_request(config_path, "daemon.status", json!({})).await?;
    if json_output {
        let response = send_request(config_path, "channel.list", json!({})).await?;
        return print_response(response, true);
    }

    let status: DaemonStatusView = decode_result(response)?;
    print_channel_list(status);
    Ok(())
}

pub async fn run_channel_create(
    config_path: &std::path::Path,
    params: Value,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "channel.create", params).await?;
    print_response(response, json_output)
}

pub async fn run_channel_get(
    config_path: &std::path::Path,
    channel_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "channel.get", json!({ "id": channel_id })).await?;
    if json_output {
        return print_response(response, true);
    }

    let channel: ChannelDetailView = decode_result(response)?;
    print_channel_detail(channel);
    Ok(())
}

pub async fn run_channel_status(
    config_path: &std::path::Path,
    channel_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "channel.status", json!({ "id": channel_id })).await?;
    if json_output {
        return print_response(response, true);
    }

    let channel: ChannelRuntimeView = decode_result(response)?;
    print_channel_runtime(channel);
    Ok(())
}

pub async fn run_channel_issues(
    config_path: &std::path::Path,
    channel_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "channel.issues", json!({ "id": channel_id })).await?;
    if json_output {
        return print_response(response, true);
    }

    let issues: IssueListView = decode_result(response)?;
    print_issue_list(&format!("Channel '{}' issues", channel_id), &issues.issues);
    Ok(())
}

pub async fn run_channel_enable(
    config_path: &std::path::Path,
    channel_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "channel.enable", json!({ "id": channel_id })).await?;
    print_response(response, json_output)
}

pub async fn run_channel_disable(
    config_path: &std::path::Path,
    channel_id: &str,
    json_output: bool,
) -> Result<()> {
    let response =
        send_request(config_path, "channel.disable", json!({ "id": channel_id })).await?;
    print_response(response, json_output)
}

pub async fn run_channel_update(
    config_path: &std::path::Path,
    params: Value,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "channel.update", params).await?;
    print_response(response, json_output)
}

pub async fn run_channel_delete(
    config_path: &std::path::Path,
    channel_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "channel.delete", json!({ "id": channel_id })).await?;
    print_response(response, json_output)
}

pub async fn run_events(config_path: &std::path::Path, json_output: bool) -> Result<()> {
    let endpoint = resolve_endpoint_path(config_path)?;
    let stream = connect_local_ipc(&endpoint)
        .await
        .with_context(|| {
            format!(
                "Failed to connect to daemon endpoint '{}'",
                endpoint.display()
            )
        })
        .map_err(|err| wrap_daemon_client_error(config_path, err))?;

    let (reader, mut writer) = split_local_ipc(stream);
    let request = RequestEnvelope::new(
        Some(format!("req-{}", uuid::Uuid::new_v4())),
        DaemonRequest::RuntimeEventsSubscribe(RuntimeEventsSubscribeParams::default()),
    );

    writer
        .write_all(serde_json::to_string(&request)?.as_bytes())
        .await?;
    writer.write_all(b"\n").await?;

    let mut lines = BufReader::new(reader).lines();
    let Some(line) = lines.next_line().await? else {
        anyhow::bail!("Daemon closed event stream before acknowledging subscription");
    };
    let response: ResponseEnvelope =
        serde_json::from_str(&line).with_context(|| "Failed to parse daemon subscription ack")?;
    if !response.ok {
        let error = response.error.unwrap_or(ErrorEnvelope {
            code: ErrorCode::InternalError,
            message: "Unknown daemon error".to_string(),
            details: None,
        });
        anyhow::bail!("{}: {}", error.code, error.message);
    }

    while let Some(line) = lines.next_line().await? {
        let event: EventEnvelope =
            serde_json::from_str(&line).with_context(|| "Failed to parse daemon event")?;
        if json_output {
            println!("{}", serde_json::to_string(&event)?);
        } else if event.data.is_null()
            || event.data == serde_json::Value::Object(Default::default())
        {
            println!("{}", event.event);
        } else {
            println!(
                "{} {}",
                event.event,
                serde_json::to_string_pretty(&event.data)?
            );
        }
    }

    Ok(())
}

pub async fn run_stop(config_path: &std::path::Path, json_output: bool) -> Result<()> {
    let response = send_request(config_path, "daemon.stop", json!({})).await?;
    print_response(response, json_output)
}

pub async fn run_runtime_errors(config_path: &std::path::Path, json_output: bool) -> Result<()> {
    let response = send_request(config_path, "runtime.errors", json!({})).await?;
    if json_output {
        return print_response(response, true);
    }

    let issues: IssueListView = decode_result(response)?;
    print_issue_list("Runtime issues", &issues.issues);
    Ok(())
}

fn print_response(response: ResponseEnvelope, json_output: bool) -> Result<()> {
    if json_output {
        println!("{}", serde_json::to_string_pretty(&response)?);
        return Ok(());
    }

    if response.ok {
        if let Some(result) = response.result {
            println!("{}", serde_json::to_string_pretty(&result)?);
        } else {
            println!("ok");
        }
        Ok(())
    } else {
        let error = response.error.unwrap_or(ErrorEnvelope {
            code: ErrorCode::InternalError,
            message: "Unknown daemon error".to_string(),
            details: None,
        });
        anyhow::bail!("{}: {}", error.code, error.message);
    }
}

fn decode_result<T: serde::de::DeserializeOwned>(response: ResponseEnvelope) -> Result<T> {
    if response.ok {
        let value = response
            .result
            .ok_or_else(|| anyhow::anyhow!("Daemon response did not include a result payload"))?;
        Ok(serde_json::from_value(value)?)
    } else {
        let error = response.error.unwrap_or(ErrorEnvelope {
            code: ErrorCode::InternalError,
            message: "Unknown daemon error".to_string(),
            details: None,
        });
        anyhow::bail!("{}: {}", error.code, error.message);
    }
}

fn print_daemon_status(status: DaemonStatusView) {
    println!("Config:    {}", status.config_path);
    println!("Workspace: {}", status.workspace_root);
    println!("Endpoint:  {}", status.endpoint);
    println!(
        "Agents:    {} daemon-managed, {} shared harnesses, {} channels, {} issues",
        status.registry.agents.len(),
        status.registry.shared_harnesses.len(),
        status.registry.channels.len(),
        status.registry.issues.len()
    );

    if !status.agent_runtimes.is_empty() {
        println!("\nAgent Runtimes");
        let mut runtime_by_agent: HashMap<_, _> = status
            .agent_runtimes
            .into_iter()
            .map(|runtime| (runtime.agent_id.clone(), runtime))
            .collect();

        let mut rows = Vec::new();
        rows.push(vec![
            "AGENT".to_string(),
            "ENABLED".to_string(),
            "RUNNING".to_string(),
            "ACTIVE".to_string(),
            "QUEUED".to_string(),
            "AWAIT".to_string(),
            "SESSION".to_string(),
            "HARNESS".to_string(),
            "MODEL".to_string(),
        ]);

        for agent in &status.registry.agents {
            let runtime = runtime_by_agent
                .remove(&agent.id)
                .unwrap_or(AgentRuntimeView {
                    agent_id: agent.id.clone(),
                    running: false,
                    active_tasks: 0,
                    queued_tasks: 0,
                    awaiting_results: 0,
                    current_session_id: None,
                    current_request_id: None,
                });
            rows.push(vec![
                agent.id.clone(),
                yes_no(agent.enabled),
                yes_no(runtime.running),
                runtime.active_tasks.to_string(),
                runtime.queued_tasks.to_string(),
                runtime.awaiting_results.to_string(),
                runtime
                    .current_session_id
                    .clone()
                    .unwrap_or_else(|| "-".to_string()),
                agent.harness_ref.clone(),
                agent.model.clone(),
            ]);
        }

        for (agent_id, runtime) in runtime_by_agent {
            rows.push(vec![
                agent_id,
                "bootstrap".to_string(),
                yes_no(runtime.running),
                runtime.active_tasks.to_string(),
                runtime.queued_tasks.to_string(),
                runtime.awaiting_results.to_string(),
                runtime
                    .current_session_id
                    .unwrap_or_else(|| "-".to_string()),
                "default".to_string(),
                "-".to_string(),
            ]);
        }

        print_table(&rows);
    }

    if !status.harnesses.is_empty() {
        println!("\nHarness Runtimes");
        let mut rows = Vec::new();
        rows.push(vec![
            "HARNESS".to_string(),
            "KIND".to_string(),
            "BOUND".to_string(),
            "SCRIPTS".to_string(),
            "WATCHED".to_string(),
        ]);
        for harness in status.harnesses {
            let kind = if harness.harness_id == "default" {
                "default"
            } else if harness.harness_id.starts_with("agent::") {
                "local"
            } else {
                "shared"
            };
            rows.push(vec![
                harness.harness_id,
                kind.to_string(),
                harness.bound_agents.len().to_string(),
                harness.loaded_scripts.len().to_string(),
                harness.watched_roots.len().to_string(),
            ]);
        }
        print_table(&rows);
    }

    if !status.registry.issues.is_empty() {
        println!();
        print_issue_list("Runtime issues", &status.registry.issues);
    }
}

fn print_agent_list(status: DaemonStatusView) {
    let runtime_by_agent: HashMap<_, _> = status
        .agent_runtimes
        .into_iter()
        .map(|runtime| (runtime.agent_id.clone(), runtime))
        .collect();

    let mut rows = Vec::new();
    rows.push(vec![
        "AGENT".to_string(),
        "ENABLED".to_string(),
        "RUNNING".to_string(),
        "ACTIVE".to_string(),
        "QUEUED".to_string(),
        "AWAIT".to_string(),
        "SESSION".to_string(),
        "HARNESS".to_string(),
        "PROVIDER".to_string(),
        "MODEL".to_string(),
    ]);

    for agent in status.registry.agents {
        let runtime = runtime_by_agent.get(&agent.id);
        rows.push(vec![
            agent.id,
            yes_no(agent.enabled),
            yes_no(runtime.map(|r| r.running).unwrap_or(false)),
            runtime.map(|r| r.active_tasks).unwrap_or(0).to_string(),
            runtime.map(|r| r.queued_tasks).unwrap_or(0).to_string(),
            runtime.map(|r| r.awaiting_results).unwrap_or(0).to_string(),
            runtime
                .and_then(|r| r.current_session_id.clone())
                .unwrap_or_else(|| "-".to_string()),
            agent.harness_ref,
            agent.provider,
            agent.model,
        ]);
    }

    print_table(&rows);
}

fn print_agent_detail(agent: AgentDetailView) {
    println!("Agent");
    println!("  id:                {}", agent.id);
    println!("  enabled:           {}", yes_no(agent.enabled));
    println!("  provider:          {}", agent.provider);
    println!("  model:             {}", agent.model);
    println!(
        "  mode:              {}",
        agent.mode.unwrap_or_else(|| "-".to_string())
    );
    println!(
        "  harness:           {}",
        agent.harness.unwrap_or_else(|| "local".to_string())
    );
    println!("  local_harness:     {}", yes_no(agent.has_local_harness));
    println!("  directory:         {}", agent.directory);
    if let Some(idle_grace_secs) = agent.idle_grace_secs {
        println!("  idle_grace_secs:   {}", idle_grace_secs);
    }
    if let Some(system_prompt) = &agent.system_prompt {
        println!("  system_prompt:");
        print_indented(system_prompt);
    }
}

fn print_agent_runtime_status(status: AgentRuntimeView) {
    println!("Agent Runtime");
    println!("  agent:           {}", status.agent_id);
    println!("  running:         {}", yes_no(status.running));
    println!("  active_tasks:    {}", status.active_tasks);
    println!("  queued_tasks:    {}", status.queued_tasks);
    println!("  awaiting_results: {}", status.awaiting_results);
    println!(
        "  current_session: {}",
        status.current_session_id.unwrap_or_else(|| "-".to_string())
    );
    println!(
        "  current_request: {}",
        status.current_request_id.unwrap_or_else(|| "-".to_string())
    );
}

fn print_harness_list(status: DaemonStatusView) {
    let shared_ids: std::collections::HashSet<_> = status
        .registry
        .shared_harnesses
        .into_iter()
        .map(|shared| shared.id)
        .collect();

    let mut rows = Vec::new();
    rows.push(vec![
        "HARNESS".to_string(),
        "KIND".to_string(),
        "BOUND".to_string(),
        "SCRIPTS".to_string(),
        "WATCHED".to_string(),
    ]);

    for harness in status.harnesses {
        let kind = if harness.harness_id == "default" {
            "default"
        } else if shared_ids.contains(&harness.harness_id) {
            "shared"
        } else {
            "local"
        };
        rows.push(vec![
            harness.harness_id,
            kind.to_string(),
            harness.bound_agents.len().to_string(),
            harness.loaded_scripts.len().to_string(),
            harness.watched_roots.len().to_string(),
        ]);
    }

    print_table(&rows);
}

fn print_harness_detail(harness: HarnessDetailView) {
    println!("Harness");
    println!("  harness_id:   {}", harness.harness_id);
    println!("  directory:    {}", harness.directory);
    println!("  bound_agents: {}", harness.bound_agents.len());
    if !harness.bound_agents.is_empty() {
        println!("    {}", harness.bound_agents.join(", "));
    }
    println!("  watched_roots: {}", harness.watched_roots.len());
    for root in harness.watched_roots {
        println!("    {}", root);
    }
    println!("  loaded_scripts: {}", harness.loaded_scripts.len());
    for script in harness.loaded_scripts {
        println!("    {}", script);
    }
}

fn print_channel_list(status: DaemonStatusView) {
    let mut rows = Vec::new();
    rows.push(vec![
        "CHANNEL".to_string(),
        "ENABLED".to_string(),
        "KIND".to_string(),
        "AGENT".to_string(),
    ]);

    for channel in status.registry.channels {
        rows.push(vec![
            channel.id,
            yes_no(channel.enabled),
            channel.kind,
            channel.agent_id,
        ]);
    }

    print_table(&rows);
}

fn print_channel_detail(channel: ChannelDetailView) {
    println!("Channel");
    println!("  id:            {}", channel.id);
    println!("  kind:          {}", channel.kind);
    println!("  agent:         {}", channel.agent_id);
    println!("  enabled:       {}", yes_no(channel.enabled));
    println!("  directory:     {}", channel.directory);
    if let Some(idle_ttl_secs) = channel.idle_ttl_secs {
        println!("  idle_ttl_secs: {}", idle_ttl_secs);
    }
    if channel.settings.is_object()
        && !channel
            .settings
            .as_object()
            .is_some_and(|map| map.is_empty())
    {
        println!("  settings:");
        print_indented(&serde_json::to_string_pretty(&channel.settings).unwrap_or_default());
    }
}

fn print_channel_runtime(channel: ChannelRuntimeView) {
    println!("Channel Runtime:");
    println!("  id:            {}", channel.id);
    println!("  kind:          {}", channel.kind);
    println!("  agent_id:      {}", channel.agent_id);
    println!("  directory:     {}", channel.directory);
    println!("  state:         {}", channel.state);
    println!("  start_count:   {}", channel.start_count);
    println!("  restart_count: {}", channel.restart_count);
    println!("  failure_count: {}", channel.failure_count);
    println!("  transitioned:  {}", channel.last_transition_unix_ms);
    if let Some(last_started) = channel.last_started_unix_ms {
        println!("  last_started:  {}", last_started);
    }
    if let Some(last_stopped) = channel.last_stopped_unix_ms {
        println!("  last_stopped:  {}", last_stopped);
    }
    if let Some(code) = channel.last_error_code {
        println!("  error_code:    {}", code);
    }
    if let Some(error) = channel.last_error {
        println!("  last_error:    {}", error);
    }
}

fn print_task_status(title: &str, task: &TaskStatusView) {
    println!("{}", title);
    println!("  request_id:      {}", task.request_id);
    println!("  trace_id:        {}", task.trace_id);
    println!("  agent:           {}", task.agent_id);
    println!("  slot_id:         {}", task.slot_id);
    println!("  state:           {}", task.state);
    if let Some(runtime_task_id) = &task.runtime_task_id {
        println!("  runtime_task_id: {}", runtime_task_id);
    }
    if let Some(status) = &task.status {
        println!("  terminal_status: {}", status);
    }
    if let Some(turns) = task.task_turn_count {
        println!("  task_turns:      {}", turns);
    }
    if let Some(error) = &task.error {
        println!("  error:");
        print_indented(error);
    }
    if let Some(output) = &task.output {
        println!("  output:");
        print_indented(output);
    }
}

fn print_task_list(tasks: TaskListView) {
    let mut rows = Vec::new();
    rows.push(vec![
        "REQUEST".to_string(),
        "TRACE".to_string(),
        "AGENT".to_string(),
        "SLOT".to_string(),
        "STATE".to_string(),
        "STATUS".to_string(),
        "TURNS".to_string(),
        "OUT".to_string(),
        "ERR".to_string(),
    ]);

    for task in tasks.tasks {
        rows.push(vec![
            task.request_id,
            task.trace_id,
            task.agent_id,
            task.slot_id,
            task.state,
            task.status.unwrap_or_else(|| "-".to_string()),
            task.task_turn_count
                .map(|count| count.to_string())
                .unwrap_or_else(|| "-".to_string()),
            yes_no(task.output.is_some()),
            yes_no(task.error.is_some()),
        ]);
    }

    print_table(&rows);
}

fn print_live_session(title: &str, session: &LiveSessionView) {
    println!("{}", title);
    println!("  agent:           {}", session.agent_id);
    println!("  slot_id:         {}", session.slot_id);
    println!("  session_id:      {}", session.session_id);
    println!("  running:         {}", yes_no(session.running));
    println!("  active_tasks:    {}", session.active_tasks);
    println!("  queued_tasks:    {}", session.queued_tasks);
    println!(
        "  current_request: {}",
        session
            .current_request_id
            .clone()
            .unwrap_or_else(|| "-".to_string())
    );
}

fn print_live_session_list(sessions: LiveSessionListView) {
    let mut rows = Vec::new();
    rows.push(vec![
        "AGENT".to_string(),
        "SLOT".to_string(),
        "SESSION".to_string(),
        "RUNNING".to_string(),
        "ACTIVE".to_string(),
        "QUEUED".to_string(),
        "REQUEST".to_string(),
    ]);

    for session in sessions.sessions {
        rows.push(vec![
            session.agent_id,
            session.slot_id,
            session.session_id,
            yes_no(session.running),
            session.active_tasks.to_string(),
            session.queued_tasks.to_string(),
            session
                .current_request_id
                .unwrap_or_else(|| "-".to_string()),
        ]);
    }

    print_table(&rows);
}

fn print_session_list(sessions: SessionListView) {
    let mut rows = Vec::new();
    rows.push(vec![
        "SESSION".to_string(),
        "AGENT".to_string(),
        "CREATED".to_string(),
        "META".to_string(),
        "DB_ID".to_string(),
    ]);

    for session in sessions.sessions {
        rows.push(vec![
            session.session_id,
            session.agent_id,
            session.created_at,
            yes_no(session.metadata.is_some()),
            session.internal_id.to_string(),
        ]);
    }

    print_table(&rows);
}

fn print_session_detail(session: SessionDetailView) {
    println!("Session");
    println!("  session_id: {}", session.session.session_id);
    println!("  agent:      {}", session.session.agent_id);
    println!("  created_at: {}", session.session.created_at);
    println!("  db_id:      {}", session.session.internal_id);
    if let Some(metadata) = &session.session.metadata {
        println!("  metadata:   {}", json_snippet(metadata, 120));
    }

    println!();
    println!(
        "Counts: {} events, {} messages, {} tool executions",
        session.events.len(),
        session.messages.len(),
        session.tool_executions.len()
    );

    if !session.events.is_empty() {
        println!("\nEvents");
        let mut rows = Vec::new();
        rows.push(vec![
            "ID".to_string(),
            "TYPE".to_string(),
            "CREATED".to_string(),
            "PAYLOAD".to_string(),
        ]);
        for event in session.events {
            rows.push(vec![
                event.id.to_string(),
                event.event_type,
                event.created_at,
                json_snippet(&event.payload, 72),
            ]);
        }
        print_table(&rows);
    }

    if !session.messages.is_empty() {
        println!("\nMessages");
        let mut rows = Vec::new();
        rows.push(vec![
            "ID".to_string(),
            "TURN".to_string(),
            "ROLE".to_string(),
            "TOKENS".to_string(),
            "CREATED".to_string(),
            "CONTENT".to_string(),
        ]);
        for message in session.messages {
            rows.push(vec![
                message.id.to_string(),
                message.turn_index.to_string(),
                message.role,
                message
                    .token_count
                    .map(|count| count.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                message.created_at,
                json_snippet(&message.content, 72),
            ]);
        }
        print_table(&rows);
    }

    if !session.tool_executions.is_empty() {
        println!("\nTool Executions");
        let mut rows = Vec::new();
        rows.push(vec![
            "ID".to_string(),
            "TURN".to_string(),
            "TOOL".to_string(),
            "VERDICT".to_string(),
            "ERR".to_string(),
            "DURATION".to_string(),
            "ARGS".to_string(),
            "OUTPUT".to_string(),
            "CALL_ID".to_string(),
            "CREATED".to_string(),
        ]);
        for execution in session.tool_executions {
            rows.push(vec![
                execution.id.to_string(),
                execution.turn_index.to_string(),
                execution.tool_name,
                execution.verdict,
                yes_no(execution.is_error),
                execution
                    .duration_ms
                    .map(|ms| format!("{}ms", ms))
                    .unwrap_or_else(|| "-".to_string()),
                json_snippet(&execution.args, 48),
                execution
                    .output
                    .as_ref()
                    .map(|value| json_snippet(value, 48))
                    .unwrap_or_else(|| "-".to_string()),
                execution.tool_call_id,
                execution.created_at,
            ]);
        }
        print_table(&rows);
    }
}

fn print_issue_list(title: &str, issues: &[IssueView]) {
    println!("{}", title);
    if issues.is_empty() {
        println!("  none");
        return;
    }
    for issue in issues {
        println!("- {}", issue.path);
        println!("  {}", issue.message);
    }
}

fn yes_no(value: bool) -> String {
    if value { "yes" } else { "no" }.to_string()
}

fn json_snippet(value: &Value, max_chars: usize) -> String {
    let mut rendered = match value {
        Value::String(text) => text.clone(),
        other => serde_json::to_string(other).unwrap_or_else(|_| "<unserializable>".to_string()),
    };
    rendered = rendered.replace('\n', "\\n");
    let char_count = rendered.chars().count();
    if char_count > max_chars {
        let truncated: String = rendered.chars().take(max_chars.saturating_sub(1)).collect();
        format!("{}…", truncated)
    } else {
        rendered
    }
}

fn print_indented(text: &str) {
    for line in text.lines() {
        println!("    {}", line);
    }
}

fn print_table(rows: &[Vec<String>]) {
    if rows.is_empty() {
        return;
    }
    let cols = rows[0].len();
    let mut widths = vec![0usize; cols];
    for row in rows {
        for (idx, cell) in row.iter().enumerate() {
            widths[idx] = widths[idx].max(cell.len());
        }
    }

    for (row_idx, row) in rows.iter().enumerate() {
        let line = row
            .iter()
            .enumerate()
            .map(|(idx, cell)| format!("{:width$}", cell, width = widths[idx]))
            .collect::<Vec<_>>()
            .join("  ");
        println!("{}", line);
        if row_idx == 0 {
            let sep = widths
                .iter()
                .map(|width| "-".repeat(*width))
                .collect::<Vec<_>>()
                .join("  ");
            println!("{}", sep);
        }
    }
}

async fn send_request(config_path: &Path, op: &str, params: Value) -> Result<ResponseEnvelope> {
    let client = daemon_client_from_config(config_path)?;
    let request: RequestEnvelope = serde_json::from_value(json!({
        "id": format!("req-{}", uuid::Uuid::new_v4()),
        "op": op,
        "params": params,
    }))
    .with_context(|| format!("Failed to build daemon request '{}'", op))?;
    client
        .send(request)
        .await
        .map_err(|err| wrap_daemon_client_error(config_path, err))
}

fn resolve_endpoint_path(config_path: &Path) -> Result<PathBuf> {
    let config = TurinConfig::from_file(config_path)?;
    let config_base = config_path.parent().unwrap_or_else(|| Path::new("."));
    Ok(config.resolve_daemon_endpoint(config_base))
}

fn daemon_client_from_config(config_path: &Path) -> Result<DaemonClient> {
    let config = TurinConfig::from_file(config_path)?;
    let config_base = config_path.parent().unwrap_or_else(|| Path::new("."));
    Ok(DaemonClient::new(
        config.resolve_daemon_endpoint(config_base),
    ))
}

async fn daemon_health_report(config_path: &Path) -> Result<DaemonHealthReport> {
    let client = daemon_client_from_config(config_path)?;
    match client.health().await {
        Ok(health) => Ok(DaemonHealthReport::from_health(health)),
        Err(err) if is_daemon_offline_error(&err) => Ok(DaemonHealthReport::offline(
            client.endpoint().display().to_string(),
            err.to_string(),
        )),
        Err(err) => Err(wrap_daemon_client_error(config_path, err)),
    }
}

async fn ensure_background_daemon(
    config_path: &Path,
    timeout: Duration,
    poll_interval: Duration,
    log_level: &str,
    log_file_override: Option<&Path>,
) -> Result<DaemonStartReport> {
    let client = daemon_client_from_config(config_path)?;
    match client.health().await {
        Ok(health) => {
            let log_path = resolve_daemon_log_path(config_path, log_file_override)?;
            return Ok(DaemonStartReport {
                started: false,
                endpoint: health.endpoint.clone(),
                log_path: log_path.display().to_string(),
                health: DaemonHealthReport::from_health(health),
            });
        }
        Err(err) if is_daemon_offline_error(&err) => {}
        Err(err) => return Err(wrap_daemon_client_error(config_path, err)),
    }

    spawn_background_daemon(config_path, log_level, log_file_override)?;
    client
        .wait_until_ready(timeout, poll_interval)
        .await
        .map_err(|err| wrap_daemon_client_error(config_path, err))?;
    let health = client
        .health()
        .await
        .map_err(|err| wrap_daemon_client_error(config_path, err))?;
    let log_path = resolve_daemon_log_path(config_path, log_file_override)?;
    Ok(DaemonStartReport {
        started: true,
        endpoint: health.endpoint.clone(),
        log_path: log_path.display().to_string(),
        health: DaemonHealthReport::from_health(health),
    })
}

fn spawn_background_daemon(
    config_path: &Path,
    log_level: &str,
    log_file_override: Option<&Path>,
) -> Result<()> {
    let current_exe =
        std::env::current_exe().with_context(|| "Failed to resolve current Turin binary path")?;
    let config_path = absolute_path(config_path)?;
    let log_path = resolve_daemon_log_path(config_path.as_path(), log_file_override)?;
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "Failed to create daemon log directory '{}'",
                parent.display()
            )
        })?;
    }
    let log_file = File::options()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("Failed to open daemon log '{}'", log_path.display()))?;
    let stderr = log_file
        .try_clone()
        .with_context(|| format!("Failed to clone daemon log handle '{}'", log_path.display()))?;

    let mut child = Command::new(current_exe);
    child
        .arg("--log-level")
        .arg(log_level)
        .arg("daemon")
        .arg("start")
        .arg("--config")
        .arg(&config_path)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(stderr));

    child.spawn().with_context(|| {
        format!(
            "Failed to spawn background daemon with config '{}'",
            config_path.display()
        )
    })?;
    Ok(())
}

fn resolve_daemon_log_path(
    config_path: &Path,
    log_file_override: Option<&Path>,
) -> Result<PathBuf> {
    if let Some(path) = log_file_override {
        return Ok(absolute_path(path)?);
    }

    let config = TurinConfig::from_file(config_path)?;
    let config_base = config_path.parent().unwrap_or_else(|| Path::new("."));
    Ok(config
        .resolve_workspace_root(config_base)
        .join(".turin/daemon.log"))
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    Ok(std::env::current_dir()
        .with_context(|| "Failed to resolve current working directory")?
        .join(path))
}

fn tail_lines(path: &Path, count: usize) -> Result<Vec<String>> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read daemon log '{}'", path.display()))?;
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut lines: Vec<String> = contents.lines().map(str::to_string).collect();
    if lines.len() > count {
        lines = lines.split_off(lines.len() - count);
    }
    Ok(lines)
}

fn is_daemon_offline_error(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| cause.is::<std::io::Error>())
        || err.to_string().contains("Failed to connect to")
        || err
            .to_string()
            .contains("Daemon closed connection before response")
        || err
            .to_string()
            .contains("Daemon closed connection before subscription ack")
}

fn wrap_daemon_client_error(config_path: &Path, err: anyhow::Error) -> anyhow::Error {
    if is_daemon_offline_error(&err) {
        return err.context(format!(
            "Turin daemon is not running. Start it with `turin daemon ensure --config {}` or `turin daemon start --background --config {}`",
            config_path.display(),
            config_path.display()
        ));
    }
    err
}

impl DaemonHealthReport {
    fn from_health(health: DaemonHealth) -> Self {
        Self {
            state: match health.state {
                DaemonHealthState::Ready => "ready".to_string(),
                DaemonHealthState::Degraded => "degraded".to_string(),
            },
            ready: health.ready,
            endpoint: health.endpoint,
            error: None,
            version: Some(health.version),
            protocol_version: Some(health.protocol_version),
            transport: Some(health.transport),
            wire_format: Some(health.wire_format),
            issue_count: health.issue_count,
            agent_count: health.agent_count,
            harness_count: health.harness_count,
            channel_count: health.channel_count,
            running_agent_count: health.running_agent_count,
            active_task_count: health.active_task_count,
            queued_task_count: health.queued_task_count,
            awaiting_result_count: health.awaiting_result_count,
            channel_runtime_count: health.channel_runtime_count,
            failed_channel_count: health.failed_channel_count,
        }
    }

    fn offline(endpoint: String, error: String) -> Self {
        Self {
            state: "offline".to_string(),
            ready: false,
            endpoint,
            error: Some(error),
            version: None,
            protocol_version: None,
            transport: None,
            wire_format: None,
            issue_count: 0,
            agent_count: 0,
            harness_count: 0,
            channel_count: 0,
            running_agent_count: 0,
            active_task_count: 0,
            queued_task_count: 0,
            awaiting_result_count: 0,
            channel_runtime_count: 0,
            failed_channel_count: 0,
        }
    }
}

fn print_health_report(report: &DaemonHealthReport, json_output: bool) -> Result<()> {
    if json_output {
        println!("{}", serde_json::to_string_pretty(report)?);
        return Ok(());
    }

    println!("State:     {}", report.state);
    println!("Ready:     {}", yes_no(report.ready));
    println!("Endpoint:  {}", report.endpoint);
    if let Some(error) = &report.error {
        println!("Error:     {}", error);
        return Ok(());
    }
    if let Some(version) = &report.version {
        println!("Version:   {}", version);
    }
    if let Some(protocol_version) = report.protocol_version {
        println!("Protocol:  {}", protocol_version);
    }
    if let Some(transport) = &report.transport {
        println!("Transport: {}", transport);
    }
    println!(
        "Counts:    {} agents, {} shared harnesses, {} channels, {} issues",
        report.agent_count, report.harness_count, report.channel_count, report.issue_count
    );
    println!(
        "Load:      {} running agents, {} active tasks, {} queued tasks, {} failed channels",
        report.running_agent_count,
        report.active_task_count,
        report.queued_task_count,
        report.failed_channel_count
    );
    Ok(())
}

fn print_start_report(report: DaemonStartReport, json_output: bool) -> Result<()> {
    if json_output {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    if report.started {
        println!("Daemon started in the background.");
    } else {
        println!("Daemon already running.");
    }
    println!("Endpoint:  {}", report.endpoint);
    println!("Logs:      {}", report.log_path);
    println!();
    print_health_report(&report.health, false)
}
