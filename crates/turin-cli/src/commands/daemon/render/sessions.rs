use super::common::{json_snippet, print_table, yes_no};
use super::types::{
    SessionBranchDetailView, SessionBranchListView, SessionDetailView, SessionListView,
};

pub(in crate::commands::daemon) fn print_session_list(sessions: SessionListView) {
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

pub(in crate::commands::daemon) fn print_session_detail(session: SessionDetailView) {
    println!("Session");
    println!("  session_id: {}", session.session.session_id);
    println!("  agent:      {}", session.session.agent_id);
    println!("  created_at: {}", session.session.created_at);
    println!("  db_id:      {}", session.session.internal_id);
    if let Some(metadata) = &session.session.metadata {
        println!("  metadata:   {}", json_snippet(metadata, 120));
    }

    if !session.branches.is_empty() {
        println!();
        println!("Branches");
        let mut rows = Vec::new();
        rows.push(vec![
            "BRANCH".to_string(),
            "NAME".to_string(),
            "HEAD".to_string(),
            "ACTIVE".to_string(),
            "CREATED".to_string(),
        ]);
        for branch in &session.branches {
            rows.push(vec![
                branch.branch_id.clone(),
                branch.name.clone(),
                branch
                    .head_turn_index
                    .map(|idx| idx.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                yes_no(branch.active),
                branch.created_at.clone(),
            ]);
        }
        print_table(&rows);
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

pub(in crate::commands::daemon) fn print_session_branch_list(
    session_id: &str,
    branches: SessionBranchListView,
) {
    println!("Session Branches");
    println!("  session_id: {}", session_id);
    if branches.branches.is_empty() {
        println!("  none");
        return;
    }

    let mut rows = Vec::new();
    rows.push(vec![
        "BRANCH".to_string(),
        "NAME".to_string(),
        "HEAD".to_string(),
        "ACTIVE".to_string(),
        "CREATED".to_string(),
    ]);
    for branch in branches.branches {
        rows.push(vec![
            branch.branch_id,
            branch.name,
            branch
                .head_turn_index
                .map(|idx| idx.to_string())
                .unwrap_or_else(|| "-".to_string()),
            yes_no(branch.active),
            branch.created_at,
        ]);
    }
    print_table(&rows);
}

pub(in crate::commands::daemon) fn print_session_branch(
    title: &str,
    branch: SessionBranchDetailView,
) {
    println!("{}", title);
    println!("  branch_id:    {}", branch.branch_id);
    println!("  name:         {}", branch.name);
    println!(
        "  head_turn:    {}",
        branch
            .head_turn_index
            .map(|idx| idx.to_string())
            .unwrap_or_else(|| "-".to_string())
    );
    println!("  active:       {}", yes_no(branch.active));
    println!("  created_at:   {}", branch.created_at);
}
