use super::*;

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
