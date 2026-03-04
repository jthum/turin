use serde_json::json;

use crate::daemon::protocol::ErrorCode;
use crate::daemon::protocol::{EntityIdParams, NoParams, ResponseEnvelope};

use super::{DispatchContext, internal_error, not_found_error, serialize_response};

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
            ErrorCode::InvalidParams,
            format!("Channel '{}' not found", params.id),
        ),
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
            ErrorCode::InvalidParams,
            format!("Channel '{}' not found", params.id),
        ),
        Err(err) => internal_error(id, err),
    }
}
