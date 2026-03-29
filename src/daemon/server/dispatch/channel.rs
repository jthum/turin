use serde_json::json;

use crate::daemon::protocol::ErrorCode;
use crate::daemon::protocol::{
    ChannelAccessParams, ChannelAccessRoomParams, ChannelRunnerHelloParams, CreateChannelParams,
    EntityIdParams, NoParams, ResponseEnvelope, UpdateChannelParams,
};
use crate::daemon::state::{CreateChannelInput, UpdateChannelInput};

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
    ResponseEnvelope::ok(
        id,
        json!({ "channels": guard.status().await.registry.channels }),
    )
}

pub(super) async fn get(
    id: Option<String>,
    params: EntityIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard.channel_detail(&params.id) {
        Some(channel) => serialize_response(id, channel, "channel detail"),
        None => not_found_error(
            id,
            ErrorCode::ChannelNotFound,
            format!("Channel '{}' not found", params.id),
        ),
    }
}

pub(super) async fn status(
    id: Option<String>,
    params: EntityIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    match ctx.channel_runtimes.get(&params.id).await {
        Some(status) => serialize_response(id, status, "channel status"),
        None => not_found_error(
            id,
            ErrorCode::ChannelNotFound,
            format!("Channel '{}' not found", params.id),
        ),
    }
}

pub(super) async fn runner_hello(
    id: Option<String>,
    params: ChannelRunnerHelloParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    match ctx.channel_runtimes.record_external_hello(params).await {
        Ok(snapshot) => serialize_response(id, snapshot, "channel runner hello"),
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn issues(
    id: Option<String>,
    params: EntityIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard.channel_issues(&params.id) {
        Ok(Some(issues)) => {
            ResponseEnvelope::ok(id, json!({ "channel_id": params.id, "issues": issues }))
        }
        Ok(None) => not_found_error(
            id,
            ErrorCode::ChannelNotFound,
            format!("Channel '{}' not found", params.id),
        ),
        Err(err) => internal_error(id, err),
    }
}

pub(super) async fn create(
    id: Option<String>,
    params: CreateChannelParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let mut guard = ctx.state.write().await;
    let response = match guard
        .create_channel(CreateChannelInput {
            id: params.id,
            kind: params.kind,
            agent_id: params.agent_id,
            idle_ttl_secs: params.idle_ttl_secs,
            enabled: params.enabled,
            settings: params.settings.unwrap_or_else(|| json!({})),
        })
        .await
    {
        Ok(channel) => serialize_response_with_event(
            id,
            channel,
            "created channel",
            &ctx.event_tx,
            "channel.created",
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
    channel_id: String,
    enabled: bool,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let mut guard = ctx.state.write().await;
    let response = match guard.set_channel_enabled(&channel_id, enabled).await {
        Ok(channel) => serialize_response_with_event(
            id,
            channel,
            "channel toggle result",
            &ctx.event_tx,
            if enabled {
                "channel.enabled"
            } else {
                "channel.disabled"
            },
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

pub(super) async fn update(
    id: Option<String>,
    params: UpdateChannelParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let mut guard = ctx.state.write().await;
    let response = match guard
        .update_channel(
            &params.id,
            UpdateChannelInput {
                kind: params.kind,
                agent_id: params.agent_id,
                idle_ttl_secs: params.idle_ttl_secs,
                settings: params.settings,
            },
        )
        .await
    {
        Ok(channel) => serialize_response_with_event(
            id,
            channel,
            "updated channel",
            &ctx.event_tx,
            "channel.updated",
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

pub(super) async fn access_get(
    id: Option<String>,
    params: ChannelAccessParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard.channel_access_snapshot(&params.id).await {
        Ok(Some(snapshot)) => serialize_response(id, snapshot, "channel access state"),
        Ok(None) => not_found_error(
            id,
            ErrorCode::ChannelNotFound,
            format!("Channel '{}' not found", params.id),
        ),
        Err(err) => internal_error(id, err),
    }
}

pub(super) async fn access_approve(
    id: Option<String>,
    params: ChannelAccessRoomParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard
        .approve_channel_room(
            &params.id,
            params.workspace_id,
            params.room_id,
            params.thread_id,
        )
        .await
    {
        Ok(Some(snapshot)) => serialize_response(id, snapshot, "channel access state"),
        Ok(None) => not_found_error(
            id,
            ErrorCode::ChannelNotFound,
            format!("Channel '{}' not found", params.id),
        ),
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn access_reject(
    id: Option<String>,
    params: ChannelAccessRoomParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard
        .reject_channel_room(
            &params.id,
            params.workspace_id,
            params.room_id,
            params.thread_id,
        )
        .await
    {
        Ok(Some(snapshot)) => serialize_response(id, snapshot, "channel access state"),
        Ok(None) => not_found_error(
            id,
            ErrorCode::ChannelNotFound,
            format!("Channel '{}' not found", params.id),
        ),
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn access_revoke(
    id: Option<String>,
    params: ChannelAccessRoomParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard
        .revoke_channel_room(
            &params.id,
            params.workspace_id,
            params.room_id,
            params.thread_id,
        )
        .await
    {
        Ok(Some(snapshot)) => serialize_response(id, snapshot, "channel access state"),
        Ok(None) => not_found_error(
            id,
            ErrorCode::ChannelNotFound,
            format!("Channel '{}' not found", params.id),
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
    let status = match guard.delete_channel(&params.id).await {
        Ok(status) => status,
        Err(err) => return validation_error(id, err),
    };
    drop(guard);

    if let Err(err) = sync_channel_runtimes(ctx).await {
        return internal_error(id, err);
    }

    let runtime_snapshot = build_runtime_snapshot(&ctx.state, &ctx.channel_runtimes).await;
    match serialize_value(&id, runtime_snapshot, "delete status") {
        Ok(value) => {
            emit_event(&ctx.event_tx, "channel.deleted", json!({ "id": params.id }));
            emit_event(&ctx.event_tx, "runtime.rescanned", value.clone());
            emit_registry_issue_events(&ctx.event_tx, &status);
            ResponseEnvelope::ok(id, value)
        }
        Err(response) => *response,
    }
}
