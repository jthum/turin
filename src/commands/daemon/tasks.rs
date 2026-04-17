use super::*;

pub struct TaskSubmitCommand<'a> {
    pub config_path: &'a std::path::Path,
    pub agent_id: Option<&'a str>,
    pub session_id: Option<&'a str>,
    pub slot_id: Option<&'a str>,
    pub prompt: &'a str,
    pub conflict_policy: Option<&'a str>,
    pub wait: bool,
    pub timeout_ms: Option<u64>,
    pub json_output: bool,
}

pub struct TaskSidestepCommand<'a> {
    pub config_path: &'a std::path::Path,
    pub session_id: &'a str,
    pub slot_id: Option<&'a str>,
    pub branch_head_id: Option<i64>,
    pub turn_id: Option<i64>,
    pub prompt: &'a str,
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
            "prompt": command.prompt,
            "conflict_policy": command.conflict_policy,
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

pub async fn run_task_sidestep(command: TaskSidestepCommand<'_>) -> Result<()> {
    let context_target = match (command.branch_head_id, command.turn_id) {
        (Some(branch_head_id), None) => {
            Some(json!({ "kind": "branch_head", "branch_head_id": branch_head_id }))
        }
        (None, Some(turn_id)) => Some(json!({ "kind": "turn_id", "turn_id": turn_id })),
        (None, None) => None,
        (Some(_), Some(_)) => {
            anyhow::bail!("task sidestep accepts at most one of --branch-head-id or --turn-id")
        }
    };

    let response = send_request(
        command.config_path,
        "task.sidestep",
        json!({
            "session_id": command.session_id,
            "slot_id": command.slot_id,
            "prompt": command.prompt,
            "context_target": context_target,
            "timeout_ms": command.timeout_ms,
        }),
    )
    .await?;
    if command.json_output {
        return print_response(response, true);
    }

    let task: TaskStatusView = decode_result(response)?;
    print_task_status("Sidestep task", &task);
    Ok(())
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
