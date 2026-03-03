use serde_json::json;

use crate::daemon::protocol::{
    BindHarnessParams, CreateAgentParams, EntityIdParams, ResponseEnvelope, UpdateAgentParams,
};
use crate::daemon::state::{CreateAgentInput, UpdateAgentInput};

use super::{
    DispatchContext, emit_event, emit_registry_issue_events, internal_error, not_found_error,
    serialize_response, serialize_response_with_event, serialize_value, validation_error,
};
use crate::daemon::protocol::ErrorCode;

pub(super) async fn list(
    id: Option<String>,
    _params: crate::daemon::protocol::NoParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    ResponseEnvelope::ok(id, json!({ "agents": guard.registry_snapshot().agents }))
}

pub(super) async fn get(
    id: Option<String>,
    params: EntityIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard.agent_detail(&params.id) {
        Ok(Some(agent)) => serialize_response(id, agent, "agent detail"),
        Ok(None) => not_found_error(
            id,
            ErrorCode::AgentNotFound,
            format!("Agent '{}' not found", params.id),
        ),
        Err(err) => internal_error(id, err),
    }
}

pub(super) async fn status(
    id: Option<String>,
    params: EntityIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard.agent_runtime_status(&params.id).await {
        Ok(Some(status)) => serialize_response(id, status, "agent status"),
        Ok(None) => not_found_error(
            id,
            ErrorCode::AgentNotFound,
            format!("Agent '{}' not found", params.id),
        ),
        Err(err) => internal_error(id, err),
    }
}

pub(super) async fn issues(
    id: Option<String>,
    params: EntityIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard.agent_issues(&params.id) {
        Ok(Some(issues)) => {
            ResponseEnvelope::ok(id, json!({ "agent_id": params.id, "issues": issues }))
        }
        Ok(None) => not_found_error(
            id,
            ErrorCode::AgentNotFound,
            format!("Agent '{}' not found", params.id),
        ),
        Err(err) => internal_error(id, err),
    }
}

pub(super) async fn create(
    id: Option<String>,
    params: CreateAgentParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let mut guard = ctx.state.write().await;
    match guard
        .create_agent(CreateAgentInput {
            id: params.id,
            provider: params.provider,
            model: params.model,
            system_prompt: params.system_prompt,
            thinking: params.thinking,
            mode: params.mode,
            harness: params.harness,
            idle_grace_secs: params.idle_grace_secs,
            enabled: params.enabled,
        })
        .await
    {
        Ok(agent) => serialize_response_with_event(
            id,
            agent,
            "created agent",
            &ctx.event_tx,
            "agent.created",
        ),
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn enable(
    id: Option<String>,
    params: EntityIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    set_enabled(id, params.id, true, ctx).await
}

pub(super) async fn disable(
    id: Option<String>,
    params: EntityIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    set_enabled(id, params.id, false, ctx).await
}

async fn set_enabled(
    id: Option<String>,
    agent_id: String,
    enabled: bool,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let mut guard = ctx.state.write().await;
    match guard.set_agent_enabled(&agent_id, enabled).await {
        Ok(agent) => serialize_response_with_event(
            id,
            agent,
            "agent toggle result",
            &ctx.event_tx,
            if enabled {
                "agent.enabled"
            } else {
                "agent.disabled"
            },
        ),
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn update(
    id: Option<String>,
    params: UpdateAgentParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let mut guard = ctx.state.write().await;
    match guard
        .update_agent(
            &params.id,
            UpdateAgentInput {
                provider: params.provider,
                model: params.model,
                system_prompt: params.system_prompt,
                thinking: params.thinking,
                mode: params.mode,
                idle_grace_secs: params.idle_grace_secs,
            },
        )
        .await
    {
        Ok(agent) => serialize_response_with_event(
            id,
            agent,
            "updated agent",
            &ctx.event_tx,
            "agent.updated",
        ),
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn reload(
    id: Option<String>,
    params: EntityIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let mut guard = ctx.state.write().await;
    match guard.reload_agent(&params.id).await {
        Ok(agent) => serialize_response_with_event(
            id,
            agent,
            "reloaded agent",
            &ctx.event_tx,
            "agent.reloaded",
        ),
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn bind_harness(
    id: Option<String>,
    params: BindHarnessParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let mut guard = ctx.state.write().await;
    match guard
        .bind_agent_shared_harness(&params.id, &params.harness_id)
        .await
    {
        Ok(agent) => match serialize_value(&id, agent, "rebound agent") {
            Ok(value) => {
                emit_event(&ctx.event_tx, "agent.updated", value.clone());
                emit_event(
                    &ctx.event_tx,
                    "agent.harness_bound",
                    json!({ "id": params.id, "harness_id": params.harness_id }),
                );
                ResponseEnvelope::ok(id, value)
            }
            Err(response) => *response,
        },
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn use_local_harness(
    id: Option<String>,
    params: EntityIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let mut guard = ctx.state.write().await;
    match guard.use_local_agent_harness(&params.id).await {
        Ok(agent) => match serialize_value(&id, agent, "local-harness agent") {
            Ok(value) => {
                emit_event(&ctx.event_tx, "agent.updated", value.clone());
                emit_event(
                    &ctx.event_tx,
                    "agent.local_harness_enabled",
                    json!({ "id": params.id }),
                );
                ResponseEnvelope::ok(id, value)
            }
            Err(response) => *response,
        },
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn delete(
    id: Option<String>,
    params: EntityIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let mut guard = ctx.state.write().await;
    match guard.delete_agent(&params.id).await {
        Ok(status) => match serialize_value(&id, &status, "delete status") {
            Ok(value) => {
                emit_event(&ctx.event_tx, "agent.deleted", json!({ "id": params.id }));
                emit_event(&ctx.event_tx, "runtime.rescanned", value.clone());
                emit_registry_issue_events(&ctx.event_tx, &status);
                ResponseEnvelope::ok(id, value)
            }
            Err(response) => *response,
        },
        Err(err) => validation_error(id, err),
    }
}
