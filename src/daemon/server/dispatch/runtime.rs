use crate::daemon::protocol::{NoParams, ResponseEnvelope};

use super::{DispatchContext, emit_event, resource_busy_error, serialize_response};
use crate::daemon::server::watch::rescan_and_refresh_watcher;

pub(super) async fn rescan(
    id: Option<String>,
    _params: NoParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    match rescan_and_refresh_watcher(
        ctx.state.clone(),
        ctx.watcher_slot.clone(),
        ctx.daemon_watcher_tx.clone(),
        ctx.event_tx.clone(),
    )
    .await
    {
        Ok(status) => serialize_response(id, status, "rescan result"),
        Err(err) => resource_busy_error(id, err),
    }
}

pub(super) async fn reload(
    id: Option<String>,
    _params: NoParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    match rescan_and_refresh_watcher(
        ctx.state.clone(),
        ctx.watcher_slot.clone(),
        ctx.daemon_watcher_tx.clone(),
        ctx.event_tx.clone(),
    )
    .await
    {
        Ok(status) => {
            let value = serde_json::to_value(status).expect("runtime snapshot serializes");
            emit_event(&ctx.event_tx, "runtime.reloaded", value.clone());
            ResponseEnvelope::ok(id, value)
        }
        Err(err) => resource_busy_error(id, err),
    }
}

pub(super) async fn errors(
    id: Option<String>,
    _params: NoParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    ResponseEnvelope::ok(id, serde_json::json!({ "issues": guard.runtime_errors() }))
}
