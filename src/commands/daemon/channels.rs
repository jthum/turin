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

pub async fn run_channel_access(
    config_path: &std::path::Path,
    channel_id: &str,
    json_output: bool,
) -> Result<()> {
    let response = send_request(
        config_path,
        "channel.access.get",
        json!({ "id": channel_id }),
    )
    .await?;
    if json_output {
        return print_response(response, true);
    }

    let access: ChannelAccessView = decode_result(response)?;
    print_channel_access(channel_id, &access);
    Ok(())
}

pub async fn run_channel_approve(
    config_path: &std::path::Path,
    params: Value,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "channel.access.approve", params).await?;
    print_response(response, json_output)
}

pub async fn run_channel_reject(
    config_path: &std::path::Path,
    params: Value,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "channel.access.reject", params).await?;
    print_response(response, json_output)
}

pub async fn run_channel_revoke(
    config_path: &std::path::Path,
    params: Value,
    json_output: bool,
) -> Result<()> {
    let response = send_request(config_path, "channel.access.revoke", params).await?;
    print_response(response, json_output)
}

fn print_channel_access(channel_id: &str, access: &ChannelAccessView) {
    println!("Channel '{}' access", channel_id);
    println!("  pending:  {}", access.pending_rooms.len());
    for room in &access.pending_rooms {
        println!(
            "    - {} workspace={} room={} thread={} sample_user={} first_seen={} last_seen={}",
            room.room.channel,
            room.room.workspace_id,
            room.room.room_id.as_deref().unwrap_or("-"),
            room.room.thread_id,
            room.sample_username
                .as_deref()
                .or(room.sample_user_id.as_deref())
                .unwrap_or("-"),
            room.first_seen_unix_seconds,
            room.last_seen_unix_seconds,
        );
    }
    println!("  approved: {}", access.approved_rooms.len());
    for room in &access.approved_rooms {
        println!(
            "    - {} workspace={} room={} thread={} approved_by={} approved_at={}",
            room.room.channel,
            room.room.workspace_id,
            room.room.room_id.as_deref().unwrap_or("-"),
            room.room.thread_id,
            room.approved_by_username
                .as_deref()
                .or(room.approved_by_user_id.as_deref())
                .unwrap_or("-"),
            room.approved_at_unix_seconds,
        );
    }
}
