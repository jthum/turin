use serde_json::json;
use turin_local_ipc::current_transport_name;

use crate::daemon::protocol::{
    DAEMON_PROTOCOL_VERSION, DAEMON_WIRE_FORMAT_NDJSON, DaemonCapabilities, DaemonHandshake,
    NoParams, ResponseEnvelope,
};

use super::{DispatchContext, build_runtime_snapshot, emit_event};

pub(super) async fn ping(
    id: Option<String>,
    _params: NoParams,
    _ctx: &DispatchContext,
) -> ResponseEnvelope {
    let handshake = DaemonHandshake {
        pong: true,
        version: env!("CARGO_PKG_VERSION").to_string(),
        protocol_version: DAEMON_PROTOCOL_VERSION,
        transport: current_transport_name().to_string(),
        wire_format: DAEMON_WIRE_FORMAT_NDJSON.to_string(),
        capabilities: DaemonCapabilities {
            runtime_snapshot_v1: true,
            scoped_event_snapshots: true,
            lag_resnapshot: true,
            watcher_rescan_failed_events: true,
            channels: true,
        },
    };
    ResponseEnvelope::ok(
        id,
        serde_json::to_value(handshake).expect("daemon handshake serializes"),
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
