use crate::daemon::protocol::{
    NoParams, ResponseEnvelope, SubmitTaskParams, TaskIdParams, WaitTaskParams,
};

use super::{
    DispatchContext, emit_event, not_found_error, serialize_response,
    serialize_response_with_event, validation_error,
};
use crate::daemon::protocol::ErrorCode;

pub(super) async fn submit(
    id: Option<String>,
    params: SubmitTaskParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard
        .submit_task(
            params.agent_id.as_deref(),
            params.session_id.as_deref(),
            params.prompt,
            params.content,
            params.tools,
        )
        .await
    {
        Ok(task) => serialize_response_with_event(
            id,
            task,
            "submitted task",
            &ctx.event_tx,
            "task.submitted",
        ),
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn get(
    id: Option<String>,
    params: TaskIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard.get_task(&params.request_id).await {
        Some(task) => serialize_response(id, task, "task"),
        None => not_found_error(
            id,
            ErrorCode::TaskNotFound,
            format!("Task '{}' not found", params.request_id),
        ),
    }
}

pub(super) async fn wait(
    id: Option<String>,
    params: WaitTaskParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard
        .wait_for_task(&params.request_id, params.timeout_ms)
        .await
    {
        Ok(task) => ResponseEnvelope::ok(id, serde_json::json!(task)),
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn cancel(
    id: Option<String>,
    params: TaskIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard.cancel_task(&params.request_id).await {
        Ok(task) => {
            let value = serde_json::json!(task);
            let event_name =
                if value.get("state").and_then(|state| state.as_str()) == Some("cancelling") {
                    "task.cancel_requested"
                } else {
                    "task.cancelled"
                };
            emit_event(&ctx.event_tx, event_name, value.clone());
            ResponseEnvelope::ok(id, value)
        }
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn list(
    id: Option<String>,
    _params: NoParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    ResponseEnvelope::ok(id, serde_json::json!({ "tasks": guard.list_tasks().await }))
}
