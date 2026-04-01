use super::*;

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

pub async fn run_session_list(
    config_path: &std::path::Path,
    limit: usize,
    offset: usize,
    store: Option<&str>,
    path: Option<&str>,
    json_output: bool,
) -> Result<()> {
    let response = send_request(
        config_path,
        "session.list",
        json!({
            "limit": limit,
            "offset": offset,
            "store": store,
            "path": path,
        }),
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

pub async fn run_session_branch_list(
    config_path: &std::path::Path,
    session_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(
        config_path,
        "session.branch_list",
        json!({ "session_id": session_id }),
    )
    .await?;
    if json_output {
        return print_response(response, true);
    }

    let branches: SessionBranchListView = decode_result(response)?;
    print_session_branch_list(session_id, branches);
    Ok(())
}

pub async fn run_session_branch_create(
    config_path: &std::path::Path,
    session_id: &str,
    name: &str,
    from_turn: Option<u32>,
    activate: bool,
    json_output: bool,
) -> Result<()> {
    let response = send_request(
        config_path,
        "session.branch_create",
        json!({
            "session_id": session_id,
            "name": name,
            "from_turn_index": from_turn,
            "activate": activate,
        }),
    )
    .await?;
    if json_output {
        return print_response(response, true);
    }

    let branch: SessionBranchDetailView = decode_result(response)?;
    print_session_branch("Created session branch", branch);
    Ok(())
}

pub async fn run_session_branch_checkout(
    config_path: &std::path::Path,
    session_id: &str,
    branch: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(
        config_path,
        "session.branch_checkout",
        json!({ "session_id": session_id, "branch": branch }),
    )
    .await?;
    if json_output {
        return print_response(response, true);
    }

    let branch: SessionBranchDetailView = decode_result(response)?;
    print_session_branch("Checked out session branch", branch);
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
