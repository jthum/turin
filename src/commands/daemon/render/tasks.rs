use super::common::{format_context_target, print_indented, print_table, yes_no};
use super::types::{LiveSessionListView, LiveSessionView, TaskListView, TaskStatusView};

pub(in crate::commands::daemon) fn print_task_status(title: &str, task: &TaskStatusView) {
    println!("{}", title);
    println!("  request_id:      {}", task.request_id);
    println!("  trace_id:        {}", task.trace_id);
    println!("  agent:           {}", task.agent_id);
    println!("  slot_id:         {}", task.slot_id);
    println!("  state:           {}", task.state);
    println!("  execution_id:    {}", task.execution.execution_id);
    println!(
        "  context_target:  {}",
        format_context_target(&task.execution.context_target)
    );
    println!("  write_policy:    {}", task.execution.write_policy);
    println!("  durability:      {}", task.execution.durability);
    println!("  visibility:      {}", task.execution.visibility);
    if let Some(runtime_task_id) = &task.runtime_task_id {
        println!("  runtime_task_id: {}", runtime_task_id);
    }
    if let Some(status) = &task.status {
        println!("  terminal_status: {}", status);
    }
    if let Some(turns) = task.task_turn_count {
        println!("  task_turns:      {}", turns);
    }
    if let Some(branch_outcome) = &task.branch_outcome {
        println!(
            "  branch_outcome:  {}",
            branch_outcome
                .get("kind")
                .and_then(|value| value.as_str())
                .unwrap_or("present")
        );
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

pub(in crate::commands::daemon) fn print_task_list(tasks: TaskListView) {
    let mut rows = Vec::new();
    rows.push(vec![
        "REQUEST".to_string(),
        "TRACE".to_string(),
        "AGENT".to_string(),
        "SLOT".to_string(),
        "STATE".to_string(),
        "STATUS".to_string(),
        "EXEC".to_string(),
        "TARGET".to_string(),
        "WRITE".to_string(),
        "TURNS".to_string(),
        "BRANCH".to_string(),
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
            task.execution.execution_id,
            format_context_target(&task.execution.context_target),
            task.execution.write_policy,
            task.task_turn_count
                .map(|count| count.to_string())
                .unwrap_or_else(|| "-".to_string()),
            task.branch_outcome
                .as_ref()
                .and_then(|value| value.get("kind"))
                .and_then(|value| value.as_str())
                .map(str::to_string)
                .unwrap_or_else(|| "-".to_string()),
            yes_no(task.output.is_some()),
            yes_no(task.error.is_some()),
        ]);
    }

    print_table(&rows);
}

pub(in crate::commands::daemon) fn print_live_session(title: &str, session: &LiveSessionView) {
    println!("{}", title);
    println!("  agent:           {}", session.agent_id);
    println!("  slot_id:         {}", session.slot_id);
    println!("  session_id:      {}", session.session_id);
    println!("  running:         {}", yes_no(session.running));
    println!("  active_tasks:    {}", session.active_tasks);
    println!("  queued_tasks:    {}", session.queued_tasks);
    println!("  execution_id:    {}", session.execution.execution_id);
    println!(
        "  context_target:  {}",
        format_context_target(&session.execution.context_target)
    );
    println!("  write_policy:    {}", session.execution.write_policy);
    println!("  durability:      {}", session.execution.durability);
    println!("  visibility:      {}", session.execution.visibility);
    println!("  conflict_policy: {}", session.conflict_policy);
    println!(
        "  current_request: {}",
        session
            .current_request_id
            .clone()
            .unwrap_or_else(|| "-".to_string())
    );
}

pub(in crate::commands::daemon) fn print_live_session_list(sessions: LiveSessionListView) {
    let mut rows = Vec::new();
    rows.push(vec![
        "AGENT".to_string(),
        "SLOT".to_string(),
        "SESSION".to_string(),
        "RUNNING".to_string(),
        "ACTIVE".to_string(),
        "QUEUED".to_string(),
        "EXEC".to_string(),
        "TARGET".to_string(),
        "WRITE".to_string(),
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
            session.execution.execution_id,
            format_context_target(&session.execution.context_target),
            session.execution.write_policy,
            session
                .current_request_id
                .unwrap_or_else(|| "-".to_string()),
        ]);
    }

    print_table(&rows);
}
