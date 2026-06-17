use serde_json::json;

use crate::daemon::protocol::ErrorCode;
use crate::daemon::protocol::{EntityIdParams, HarnessActionRunParams, NoParams, ResponseEnvelope};

use super::{
    DispatchContext, build_runtime_snapshot, emit_event, emit_registry_issue_events,
    internal_error, not_found_error, serialize_response, serialize_response_with_event,
    serialize_value, sync_channel_runtimes, validation_error,
};

pub(super) async fn list(
    id: Option<String>,
    _params: NoParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    ResponseEnvelope::ok(id, json!({ "harnesses": guard.status().await.harnesses }))
}

pub(super) async fn create(
    id: Option<String>,
    params: EntityIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let mut guard = ctx.state.write().await;
    let response = match guard.create_shared_harness(&params.id).await {
        Ok(harness) => serialize_response_with_event(
            id,
            harness,
            "created harness",
            &ctx.event_tx,
            "harness.created",
        ),
        Err(err) => validation_error(id, err),
    };
    drop(guard);
    if response.ok
        && let Err(err) = sync_channel_runtimes(ctx).await
    {
        return internal_error(response.id, err);
    }
    response
}

pub(super) async fn get(
    id: Option<String>,
    params: EntityIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard.harness_detail(&params.id) {
        Some(harness) => serialize_response(id, harness, "harness detail"),
        None => not_found_error(
            id,
            ErrorCode::HarnessNotFound,
            format!("Harness '{}' not found", params.id),
        ),
    }
}

pub(super) async fn issues(
    id: Option<String>,
    params: EntityIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard.harness_issues(&params.id) {
        Ok(Some(issues)) => {
            ResponseEnvelope::ok(id, json!({ "harness_id": params.id, "issues": issues }))
        }
        Ok(None) => not_found_error(
            id,
            ErrorCode::HarnessNotFound,
            format!("Harness '{}' not found", params.id),
        ),
        Err(err) => internal_error(id, err),
    }
}

pub(super) async fn reload(
    id: Option<String>,
    params: EntityIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let mut guard = ctx.state.write().await;
    let response = match guard.reload_harness(&params.id).await {
        Ok(harness) => serialize_response_with_event(
            id,
            harness,
            "harness reload result",
            &ctx.event_tx,
            "harness.reloaded",
        ),
        Err(err) => validation_error(id, err),
    };
    drop(guard);
    if response.ok
        && let Err(err) = sync_channel_runtimes(ctx).await
    {
        return internal_error(response.id, err);
    }
    response
}

pub(super) async fn validate(
    id: Option<String>,
    params: EntityIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard.validate_harness(&params.id) {
        Ok(result) => {
            emit_event(&ctx.event_tx, "harness.validated", result.clone());
            ResponseEnvelope::ok(id, result)
        }
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn action_run(
    id: Option<String>,
    params: HarnessActionRunParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard.run_harness_action(params) {
        Ok(result) => serialize_response_with_event(
            id,
            result,
            "harness action result",
            &ctx.event_tx,
            "harness.action_ran",
        ),
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn delete(
    id: Option<String>,
    params: EntityIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let mut guard = ctx.state.write().await;
    let status = match guard.delete_shared_harness(&params.id).await {
        Ok(status) => status,
        Err(err) => return validation_error(id, err),
    };
    drop(guard);

    if let Err(err) = sync_channel_runtimes(ctx).await {
        return internal_error(id, err);
    }

    let runtime_snapshot = build_runtime_snapshot(&ctx.state, &ctx.channel_runtimes).await;
    match serialize_value(&id, runtime_snapshot, "harness delete result") {
        Ok(value) => {
            emit_event(&ctx.event_tx, "harness.deleted", json!({ "id": params.id }));
            emit_event(&ctx.event_tx, "runtime.rescanned", value.clone());
            emit_registry_issue_events(&ctx.event_tx, &status);
            ResponseEnvelope::ok(id, value)
        }
        Err(response) => *response,
    }
}
