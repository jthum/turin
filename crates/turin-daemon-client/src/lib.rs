use anyhow::{Context, Result, anyhow};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::time::sleep;
use turin_daemon_protocol::{
    DAEMON_PROTOCOL_VERSION, DaemonHandshake, DaemonRequest, EventEnvelope, NoParams,
    RequestEnvelope, ResponseEnvelope, RuntimeEventsSubscribeParams,
};
use turin_local_ipc::{
    LocalIpcReadHalf, connect as connect_local_ipc, current_transport_name,
    resolve_endpoint as resolve_local_ipc_endpoint, split as split_local_ipc,
};

#[derive(Debug, Clone)]
pub struct DaemonClient {
    endpoint: PathBuf,
}

#[derive(Debug, Clone, Copy)]
pub struct ManagedSubscribeOptions {
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
}

impl Default for ManagedSubscribeOptions {
    fn default() -> Self {
        Self {
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(1),
        }
    }
}

impl DaemonClient {
    pub fn new(endpoint: impl Into<PathBuf>) -> Self {
        Self {
            endpoint: endpoint.into(),
        }
    }

    pub async fn from_config(config_path: impl AsRef<Path>) -> Result<Self> {
        let raw = tokio::fs::read_to_string(config_path.as_ref())
            .await
            .with_context(|| format!("Failed to read '{}'", config_path.as_ref().display()))?;
        let value: toml::Value = toml::from_str(&raw)
            .with_context(|| format!("Failed to parse '{}'", config_path.as_ref().display()))?;
        let workspace_root = value
            .get("kernel")
            .and_then(|k| k.get("workspace_root"))
            .and_then(|v| v.as_str())
            .unwrap_or(".");
        let endpoint = value
            .get("daemon")
            .and_then(|d| d.get("endpoint"))
            .and_then(|v| v.as_str())
            .unwrap_or(".turin/daemon.sock");
        Ok(Self::new(resolve_local_ipc_endpoint(
            config_path.as_ref().parent().unwrap_or(Path::new(".")),
            workspace_root,
            endpoint,
        )))
    }

    pub fn endpoint(&self) -> &Path {
        &self.endpoint
    }

    pub async fn send(&self, request: RequestEnvelope) -> Result<ResponseEnvelope> {
        let mut stream = connect_local_ipc(&self.endpoint)
            .await
            .with_context(|| format!("Failed to connect to '{}'", self.endpoint.display()))?;
        let body = serde_json::to_string(&request)?;
        stream.write_all(body.as_bytes()).await?;
        stream.write_all(b"\n").await?;

        let (reader, _) = split_local_ipc(stream);
        let mut lines = BufReader::new(reader).lines();
        let line = lines
            .next_line()
            .await?
            .ok_or_else(|| anyhow!("Daemon closed connection before response"))?;
        serde_json::from_str(&line).context("Failed to decode daemon response")
    }

    pub async fn request(
        &self,
        id: Option<String>,
        request: DaemonRequest,
    ) -> Result<ResponseEnvelope> {
        self.send(RequestEnvelope::new(id, request)).await
    }

    pub async fn request_ok<T: DeserializeOwned>(
        &self,
        id: Option<String>,
        request: DaemonRequest,
    ) -> Result<T> {
        let response = self.request(id, request).await?;
        decode_ok(response)
    }

    pub async fn handshake(&self) -> Result<DaemonHandshake> {
        let handshake: DaemonHandshake = self
            .request_ok(None, DaemonRequest::DaemonPing(NoParams::default()))
            .await?;
        ensure_compatible_handshake(&handshake)?;
        Ok(handshake)
    }

    pub async fn subscribe(
        &self,
        id: Option<String>,
        filter: RuntimeEventsSubscribeParams,
    ) -> Result<EventStream> {
        let mut stream = connect_local_ipc(&self.endpoint)
            .await
            .with_context(|| format!("Failed to connect to '{}'", self.endpoint.display()))?;
        let request = RequestEnvelope::new(id, DaemonRequest::RuntimeEventsSubscribe(filter));
        let body = serde_json::to_string(&request)?;
        stream.write_all(body.as_bytes()).await?;
        stream.write_all(b"\n").await?;

        let (reader, _) = split_local_ipc(stream);
        let mut lines = BufReader::new(reader).lines();
        let ack_line = lines
            .next_line()
            .await?
            .ok_or_else(|| anyhow!("Daemon closed connection before subscription ack"))?;
        let ack: ResponseEnvelope =
            serde_json::from_str(&ack_line).context("Failed to decode subscription ack")?;
        if !ack.ok {
            return Err(anyhow!(format_error(&ack)));
        }
        Ok(EventStream { lines })
    }

    pub async fn subscribe_managed(
        &self,
        filter: RuntimeEventsSubscribeParams,
    ) -> Result<ManagedEventStream> {
        self.subscribe_managed_with_options(filter, ManagedSubscribeOptions::default())
            .await
    }

    pub async fn subscribe_managed_with_options(
        &self,
        filter: RuntimeEventsSubscribeParams,
        options: ManagedSubscribeOptions,
    ) -> Result<ManagedEventStream> {
        let stream = self.subscribe(None, filter.clone()).await?;
        Ok(ManagedEventStream {
            client: self.clone(),
            filter,
            options,
            stream: Some(stream),
        })
    }
}

pub struct EventStream {
    lines: tokio::io::Lines<BufReader<LocalIpcReadHalf>>,
}

impl EventStream {
    pub async fn next(&mut self) -> Result<Option<EventEnvelope>> {
        match self.lines.next_line().await? {
            Some(line) => Ok(Some(
                serde_json::from_str(&line).context("Failed to decode daemon event")?,
            )),
            None => Ok(None),
        }
    }
}

pub struct ManagedEventStream {
    client: DaemonClient,
    filter: RuntimeEventsSubscribeParams,
    options: ManagedSubscribeOptions,
    stream: Option<EventStream>,
}

impl ManagedEventStream {
    pub async fn next_event(&mut self) -> Result<EventEnvelope> {
        loop {
            if self.stream.is_none() {
                self.stream = Some(self.reconnect().await?);
            }

            match self
                .stream
                .as_mut()
                .expect("managed stream set before polling")
                .next()
                .await
            {
                Ok(Some(event)) => return Ok(event),
                Ok(None) => {
                    self.stream = None;
                }
                Err(err) if is_recoverable_subscription_error(&err) => {
                    self.stream = None;
                }
                Err(err) => return Err(err),
            }
        }
    }

    async fn reconnect(&self) -> Result<EventStream> {
        let mut delay = self.options.initial_backoff;
        loop {
            match self.client.handshake().await {
                Ok(_) => {}
                Err(err) if is_recoverable_subscription_error(&err) => {
                    sleep(delay).await;
                    delay = next_backoff(delay, self.options.max_backoff);
                    continue;
                }
                Err(err) => return Err(err),
            }

            match self.client.subscribe(None, self.filter.clone()).await {
                Ok(stream) => return Ok(stream),
                Err(err) if is_recoverable_subscription_error(&err) => {
                    sleep(delay).await;
                    delay = next_backoff(delay, self.options.max_backoff);
                }
                Err(err) => return Err(err),
            }
        }
    }
}

pub fn decode_ok<T: DeserializeOwned>(response: ResponseEnvelope) -> Result<T> {
    if !response.ok {
        return Err(anyhow!(format_error(&response)));
    }
    let result = response
        .result
        .ok_or_else(|| anyhow!("Daemon response missing result payload"))?;
    serde_json::from_value(result).context("Failed to decode daemon result payload")
}

pub fn encode_params<T: Serialize>(value: T) -> Value {
    serde_json::to_value(value).expect("daemon params must serialize")
}

pub fn ensure_compatible_handshake(handshake: &DaemonHandshake) -> Result<()> {
    if handshake.protocol_version != DAEMON_PROTOCOL_VERSION {
        return Err(anyhow!(
            "Unsupported daemon protocol version {} (client expects {})",
            handshake.protocol_version,
            DAEMON_PROTOCOL_VERSION
        ));
    }
    if handshake.transport != current_transport_name() {
        return Err(anyhow!(
            "Unsupported daemon transport '{}' (client expects '{}')",
            handshake.transport,
            current_transport_name()
        ));
    }
    Ok(())
}

fn is_recoverable_subscription_error(err: &anyhow::Error) -> bool {
    if err.chain().any(|cause| cause.is::<std::io::Error>()) {
        return true;
    }

    let message = err.to_string();
    message.contains("Failed to connect to")
        || message.contains("Daemon closed connection before response")
        || message.contains("Daemon closed connection before subscription ack")
}

fn next_backoff(current: Duration, max: Duration) -> Duration {
    current.checked_mul(2).unwrap_or(max).min(max)
}

fn format_error(response: &ResponseEnvelope) -> String {
    match &response.error {
        Some(err) => match &err.details {
            Some(details) => format!("{}: {} ({})", err.code, err.message, details),
            None => format!("{}: {}", err.code, err.message),
        },
        None => "daemon request failed without error envelope".to_string(),
    }
}

#[cfg(test)]
mod tests {
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
            version: "0.23.0".into(),
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
            version: "0.23.0".into(),
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
}
