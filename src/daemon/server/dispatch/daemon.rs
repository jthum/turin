use serde_json::json;

use crate::daemon::protocol::{NoParams, ResponseEnvelope};

use super::{DispatchContext, build_runtime_snapshot, emit_event};

pub(super) async fn ping(
    id: Option<String>,
    _params: NoParams,
    _ctx: &DispatchContext,
) -> ResponseEnvelope {
    ResponseEnvelope::ok(
        id,
        json!({
            "pong": true,
            "version": env!("CARGO_PKG_VERSION"),
        }),
    )
}

pub(super) async fn status(
    id: Option<String>,
    _params: NoParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let status = build_runtime_snapshot(&ctx.state, &ctx.channel_runtimes).await;
    ResponseEnvelope::ok(
        id,
        serde_json::to_value(status).expect("runtime snapshot serializes"),
    )
}

pub(super) async fn stop(
    id: Option<String>,
    _params: NoParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    emit_event(&ctx.event_tx, "daemon.stopping", json!({}));
    let _ = ctx.shutdown_tx.send(true);
    ResponseEnvelope::ok(id, json!({ "stopping": true }))
}
