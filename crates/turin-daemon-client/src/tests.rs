use super::*;
use tempfile::tempdir;
use turin_daemon_protocol::{
    DAEMON_PROTOCOL_VERSION, DaemonCapabilities, DaemonRequest, NoParams, ResponseEnvelope,
};

#[test]
fn decode_ok_rejects_error_response() {
    let response = ResponseEnvelope::err(
        None,
        turin_daemon_protocol::ErrorCode::ValidationFailed,
        "bad",
        None,
    );
    let err = decode_ok::<serde_json::Value>(response).expect_err("error response rejected");
    assert!(err.to_string().contains("validation_failed"));
}

#[test]
fn encode_params_round_trips_json() {
    let value = encode_params(NoParams::default());
    assert!(value.is_object());
    let request = DaemonRequest::DaemonPing(NoParams::default());
    let envelope = RequestEnvelope::new(Some("x".into()), request);
    let encoded = serde_json::to_value(envelope).expect("serialize");
    assert_eq!(encoded["op"], "daemon.ping");
}

#[test]
fn compatible_handshake_is_accepted() {
    let handshake = DaemonHandshake {
        pong: true,
        version: env!("CARGO_PKG_VERSION").into(),
        protocol_version: DAEMON_PROTOCOL_VERSION,
        transport: current_transport_name().into(),
        wire_format: "ndjson".into(),
        capabilities: DaemonCapabilities {
            runtime_snapshot_v1: true,
            scoped_event_snapshots: true,
            lag_resnapshot: true,
            watcher_rescan_failed_events: true,
            channels: true,
        },
    };
    ensure_compatible_handshake(&handshake).expect("handshake accepted");
}

#[test]
fn incompatible_protocol_version_is_rejected() {
    let handshake = DaemonHandshake {
        pong: true,
        version: env!("CARGO_PKG_VERSION").into(),
        protocol_version: DAEMON_PROTOCOL_VERSION + 1,
        transport: current_transport_name().into(),
        wire_format: "ndjson".into(),
        capabilities: DaemonCapabilities {
            runtime_snapshot_v1: true,
            scoped_event_snapshots: true,
            lag_resnapshot: true,
            watcher_rescan_failed_events: true,
            channels: true,
        },
    };
    let err = ensure_compatible_handshake(&handshake).expect_err("version mismatch rejected");
    assert!(
        err.to_string()
            .contains("Unsupported daemon protocol version")
    );
}

#[test]
fn managed_subscribe_defaults_are_wrapper_friendly() {
    let options = ManagedSubscribeOptions::default();
    assert_eq!(options.initial_backoff, Duration::from_millis(100));
    assert_eq!(options.max_backoff, Duration::from_secs(1));
}

#[test]
fn io_errors_are_recoverable_for_managed_subscriptions() {
    let err = anyhow!(std::io::Error::new(
        std::io::ErrorKind::ConnectionRefused,
        "refused",
    ));
    assert!(is_recoverable_subscription_error(&err));
}

#[test]
fn protocol_mismatch_is_not_recoverable_for_managed_subscriptions() {
    let err = anyhow!("Unsupported daemon protocol version 99 (client expects 1)");
    assert!(!is_recoverable_subscription_error(&err));
}

#[tokio::test]
async fn from_config_uses_daemon_endpoint_key() {
    let tempdir = tempdir().expect("tempdir");
    let config_path = tempdir.path().join("turin.toml");
    std::fs::write(
        &config_path,
        r#"
[kernel]
workspace_root = "workspace"

[daemon]
endpoint = ".turin/gui.sock"
"#,
    )
    .expect("write config");

    let client = DaemonClient::from_config(&config_path)
        .await
        .expect("load client from config");
    assert_eq!(
        client.endpoint(),
        tempdir.path().join("workspace/.turin/gui.sock").as_path()
    );
}

#[test]
fn next_backoff_caps_at_maximum() {
    assert_eq!(
        next_backoff(Duration::from_millis(750), Duration::from_secs(1)),
        Duration::from_secs(1)
    );
}
