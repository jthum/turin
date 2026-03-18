use super::*;

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
