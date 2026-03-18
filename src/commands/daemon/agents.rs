use super::*;

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
