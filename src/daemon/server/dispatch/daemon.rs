use serde_json::json;

use crate::daemon::protocol::{NoParams, ResponseEnvelope};

use super::{DispatchContext, emit_event};

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
    let guard = ctx.state.read().await;
    let status = guard.status().await;
    drop(guard);
    let channel_runtimes = ctx.channel_runtimes.list().await;
    ResponseEnvelope::ok(
        id,
        json!({
            "config_path": status.config_path,
            "workspace_root": status.workspace_root,
            "socket_path": status.socket_path,
            "registry": status.registry,
            "harnesses": status.harnesses,
            "agent_runtimes": status.agent_runtimes,
            "channel_runtimes": channel_runtimes,
        }),
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
