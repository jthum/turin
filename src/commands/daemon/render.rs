use anyhow::Result;
use serde_json::Value;
use std::collections::{HashMap, HashSet};

use turin::daemon::protocol::{ErrorCode, ErrorEnvelope, ResponseEnvelope};

use super::{
    AgentDetailView, AgentRuntimeView, ChannelDetailView, ChannelRuntimeView, DaemonHealthReport,
    DaemonStartReport, DaemonStatusView, HarnessDetailView, IssueView, LiveSessionListView,
    LiveSessionView, SessionDetailView, SessionListView, TaskListView, TaskStatusView,
};

pub(super) fn print_response(response: ResponseEnvelope, json_output: bool) -> Result<()> {
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

pub(super) fn decode_result<T: serde::de::DeserializeOwned>(
    response: ResponseEnvelope,
) -> Result<T> {
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

pub(super) fn print_daemon_status(status: DaemonStatusView) {
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

pub(super) fn print_agent_list(status: DaemonStatusView) {
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

pub(super) fn print_agent_detail(agent: AgentDetailView) {
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

pub(super) fn print_agent_runtime_status(status: AgentRuntimeView) {
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

pub(super) fn print_harness_list(status: DaemonStatusView) {
    let shared_ids: HashSet<_> = status
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

pub(super) fn print_harness_detail(harness: HarnessDetailView) {
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

pub(super) fn print_channel_list(status: DaemonStatusView) {
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

pub(super) fn print_channel_detail(channel: ChannelDetailView) {
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

pub(super) fn print_channel_runtime(channel: ChannelRuntimeView) {
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

pub(super) fn print_task_status(title: &str, task: &TaskStatusView) {
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

pub(super) fn print_task_list(tasks: TaskListView) {
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

pub(super) fn print_live_session(title: &str, session: &LiveSessionView) {
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

pub(super) fn print_live_session_list(sessions: LiveSessionListView) {
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

pub(super) fn print_session_list(sessions: SessionListView) {
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

pub(super) fn print_session_detail(session: SessionDetailView) {
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

pub(super) fn print_issue_list(title: &str, issues: &[IssueView]) {
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

pub(super) fn print_health_report(report: &DaemonHealthReport, json_output: bool) -> Result<()> {
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

pub(super) fn print_start_report(report: DaemonStartReport, json_output: bool) -> Result<()> {
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

pub(super) fn yes_no(value: bool) -> String {
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
