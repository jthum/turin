use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixStream;

use turin::daemon::protocol::{ErrorEnvelope, EventEnvelope, RequestEnvelope, ResponseEnvelope};
use turin::kernel::config::TurinConfig;

#[derive(Debug, Deserialize)]
struct DaemonStatusView {
    config_path: String,
    workspace_root: String,
    socket_path: String,
    registry: RegistrySnapshotView,
    harnesses: Vec<HarnessRuntimeView>,
    agent_runtimes: Vec<AgentRuntimeView>,
}

#[derive(Debug, Deserialize)]
struct RegistrySnapshotView {
    agents: Vec<AgentSummaryView>,
    shared_harnesses: Vec<SharedHarnessView>,
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
struct TaskStatusView {
    request_id: String,
    agent_id: String,
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

pub async fn run_start(config_path: &std::path::Path) -> Result<()> {
    turin::daemon::server::serve(config_path).await
}

pub async fn run_ping(config_path: &std::path::Path, json_output: bool) -> Result<()> {
    let response = send_request(config_path, "daemon.ping", json!({})).await?;
    print_response(response, json_output)
}

pub async fn run_status(config_path: &std::path::Path, json_output: bool) -> Result<()> {
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
    print_response(response, json_output)
}

pub async fn run_agent_status(
    config_path: &std::path::Path,
    agent_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "agent.status", json!({ "id": agent_id })).await?;
    print_response(response, json_output)
}

pub async fn run_agent_issues(
    config_path: &std::path::Path,
    agent_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "agent.issues", json!({ "id": agent_id })).await?;
    print_response(response, json_output)
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
    agent_id: &str,
    prompt: &str,
    wait: bool,
    timeout_ms: Option<u64>,
    json_output: bool,
) -> Result<()> {
    let response = send_request(
        config_path,
        "task.submit",
        json!({ "agent_id": agent_id, "prompt": prompt }),
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
    print_response(response, json_output)
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

pub async fn run_events(config_path: &std::path::Path, json_output: bool) -> Result<()> {
    let socket_path = resolve_socket_path(config_path)?;
    let stream = UnixStream::connect(&socket_path).await.with_context(|| {
        format!(
            "Failed to connect to daemon socket '{}'",
            socket_path.display()
        )
    })?;

    let (reader, mut writer) = stream.into_split();
    let request = RequestEnvelope {
        id: Some(format!("req-{}", uuid::Uuid::new_v4())),
        op: "runtime.events.subscribe".to_string(),
        params: json!({}),
    };

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
            code: "unknown_error".to_string(),
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
            code: "unknown_error".to_string(),
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
            code: "unknown_error".to_string(),
            message: "Unknown daemon error".to_string(),
            details: None,
        });
        anyhow::bail!("{}: {}", error.code, error.message);
    }
}

fn print_daemon_status(status: DaemonStatusView) {
    println!("Config:    {}", status.config_path);
    println!("Workspace: {}", status.workspace_root);
    println!("Socket:    {}", status.socket_path);
    println!(
        "Agents:    {} daemon-managed, {} shared harnesses, {} issues",
        status.registry.agents.len(),
        status.registry.shared_harnesses.len(),
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
                });
            rows.push(vec![
                agent.id.clone(),
                yes_no(agent.enabled),
                yes_no(runtime.running),
                runtime.active_tasks.to_string(),
                runtime.queued_tasks.to_string(),
                runtime.awaiting_results.to_string(),
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
            agent.harness_ref,
            agent.provider,
            agent.model,
        ]);
    }

    print_table(&rows);
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

fn print_task_status(title: &str, task: &TaskStatusView) {
    println!("{}", title);
    println!("  request_id:      {}", task.request_id);
    println!("  agent:           {}", task.agent_id);
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
        "AGENT".to_string(),
        "STATE".to_string(),
        "STATUS".to_string(),
        "TURNS".to_string(),
        "OUT".to_string(),
        "ERR".to_string(),
    ]);

    for task in tasks.tasks {
        rows.push(vec![
            task.request_id,
            task.agent_id,
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

async fn send_request(
    config_path: &std::path::Path,
    op: &str,
    params: Value,
) -> Result<ResponseEnvelope> {
    let socket_path = resolve_socket_path(config_path)?;
    let stream = UnixStream::connect(&socket_path).await.with_context(|| {
        format!(
            "Failed to connect to daemon socket '{}'",
            socket_path.display()
        )
    })?;

    let (reader, mut writer) = stream.into_split();
    let request = RequestEnvelope {
        id: Some(format!("req-{}", uuid::Uuid::new_v4())),
        op: op.to_string(),
        params,
    };

    writer
        .write_all(serde_json::to_string(&request)?.as_bytes())
        .await?;
    writer.write_all(b"\n").await?;

    let mut lines = BufReader::new(reader).lines();
    let Some(line) = lines.next_line().await? else {
        anyhow::bail!("Daemon closed connection without sending a response");
    };

    let response: ResponseEnvelope =
        serde_json::from_str(&line).with_context(|| "Failed to parse daemon response")?;
    Ok(response)
}

fn resolve_socket_path(config_path: &std::path::Path) -> Result<std::path::PathBuf> {
    let config = TurinConfig::from_file(config_path)?;
    let config_base = config_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));
    Ok(config.resolve_daemon_socket_path(config_base))
}
