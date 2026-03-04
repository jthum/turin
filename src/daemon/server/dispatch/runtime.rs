use crate::daemon::protocol::{NoParams, ResponseEnvelope};

use super::{
    DispatchContext, resource_busy_error, serialize_response, serialize_response_with_event,
};
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
        ctx.channel_runtimes.clone(),
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
        ctx.channel_runtimes.clone(),
        ctx.event_tx.clone(),
    )
    .await
    {
        Ok(status) => serialize_response_with_event(
            id,
            status,
            "reload result",
            &ctx.event_tx,
            "runtime.reloaded",
        ),
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
