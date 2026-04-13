use super::*;

pub struct TaskSubmitCommand<'a> {
    pub config_path: &'a std::path::Path,
    pub agent_id: Option<&'a str>,
    pub session_id: Option<&'a str>,
    pub slot_id: Option<&'a str>,
    pub prompt: &'a str,
    pub wait: bool,
    pub timeout_ms: Option<u64>,
    pub json_output: bool,
}

pub async fn run_task_submit(command: TaskSubmitCommand<'_>) -> Result<()> {
    let response = send_request(
        command.config_path,
        "task.submit",
        json!({
            "agent_id": command.agent_id,
            "session_id": command.session_id,
            "slot_id": command.slot_id,
            "prompt": command.prompt
        }),
    )
    .await?;
    if !command.wait {
        if command.json_output {
            return print_response(response, true);
        }
        let task: TaskStatusView = decode_result(response)?;
        print_task_status("Submitted task", &task);
        return Ok(());
    }

    if !response.ok {
        return print_response(response, command.json_output);
    }

    let request_id = response
        .result
        .as_ref()
        .and_then(|result| result.get("request_id"))
        .and_then(|value| value.as_str())
        .ok_or_else(|| anyhow::anyhow!("Daemon task.submit response did not include request_id"))?;

    run_task_wait(
        command.config_path,
        request_id,
        command.timeout_ms,
        command.json_output,
    )
    .await
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
