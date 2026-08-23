mod support;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use futures::StreamExt;
use reqwest::header::AUTHORIZATION;
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::task::JoinHandle;
use tokio::time::{Instant, sleep, timeout};
use tokio_tungstenite::{
    client_async,
    tungstenite::{client::IntoClientRequest, protocol::Message},
};
use turin::remote::{RemoteServeOptions, start as start_remote};
use turin_daemon_protocol::DAEMON_PROTOCOL_VERSION;

struct DaemonHarness {
    _tempdir: Arc<TempDir>,
    endpoint: PathBuf,
    config_path: PathBuf,
    join: JoinHandle<Result<()>>,
}

impl DaemonHarness {
    async fn start() -> Result<Self> {
        let tempdir = Arc::new(tempfile::tempdir()?);
        let workspace_root = tempdir.path().join("workspace");
        let config_path =
            support::write_mock_runtime_config(&workspace_root, "Remote integration", "PONG")?;
        let endpoint = support::workspace_daemon_socket(&workspace_root);
        let serve_config_path = config_path.clone();
        let join =
            tokio::spawn(async move { turin_harness_lua::serve_daemon(&serve_config_path).await });

        let deadline = Instant::now() + Duration::from_secs(10);
        let client = turin_daemon_client::DaemonClient::new(&endpoint);
        loop {
            if client.handshake().await.is_ok() {
                break;
            }
            if join.is_finished() {
                let result = join
                    .await
                    .context("daemon task join failed before endpoint bind")?;
                return Err(result
                    .err()
                    .unwrap_or_else(|| anyhow!("daemon exited before creating daemon endpoint")));
            }
            if Instant::now() >= deadline {
                return Err(anyhow!(
                    "Timed out waiting for daemon endpoint '{}'",
                    endpoint.display()
                ));
            }
            sleep(Duration::from_millis(25)).await;
        }

        Ok(Self {
            _tempdir: tempdir,
            endpoint,
            config_path,
            join,
        })
    }

    async fn stop(self) -> Result<()> {
        let client = turin_daemon_client::DaemonClient::new(&self.endpoint);
        let _: Value = client
            .request_ok(
                None,
                turin_daemon_protocol::DaemonRequest::DaemonStop(Default::default()),
            )
            .await?;
        let _ = timeout(Duration::from_secs(5), self.join)
            .await
            .context("timed out waiting for daemon to exit")??;
        Ok(())
    }
}

struct RemoteHarness {
    base_url: String,
    server: turin::remote::RunningRemoteServer,
}

impl RemoteHarness {
    async fn start(config_path: &Path) -> Result<Self> {
        Self::start_with_options(
            config_path,
            RemoteServeOptions {
                bind: Some("127.0.0.1:0".to_string()),
                auth_token: Some("test-token".to_string()),
                auth_token_env: None,
                event_keepalive_seconds: Some(1),
                allow_non_loopback: Some(false),
            },
        )
        .await
    }

    async fn start_with_options(config_path: &Path, options: RemoteServeOptions) -> Result<Self> {
        let server = start_remote(config_path, options).await?;
        Ok(Self {
            base_url: format!("http://{}", server.local_addr()),
            server,
        })
    }

    async fn stop(self) -> Result<()> {
        self.server.stop().await
    }
}

struct SseReader<S> {
    stream: S,
    buffer: String,
}

impl<S> SseReader<S>
where
    S: futures::Stream<Item = std::result::Result<bytes::Bytes, reqwest::Error>> + Unpin,
{
    async fn next_event(&mut self) -> Result<(String, Value)> {
        loop {
            if let Some(index) = self.buffer.find("\n\n") {
                let chunk = self.buffer[..index].to_string();
                self.buffer = self.buffer[index + 2..].to_string();
                if chunk.trim().is_empty() || chunk.starts_with(':') {
                    continue;
                }

                let mut event_name = None;
                let mut data = None;
                for line in chunk.lines() {
                    if let Some(value) = line.strip_prefix("event: ") {
                        event_name = Some(value.to_string());
                    } else if let Some(value) = line.strip_prefix("data: ") {
                        data = Some(serde_json::from_str(value)?);
                    }
                }

                return Ok((
                    event_name.context("missing SSE event name")?,
                    data.context("missing SSE data")?,
                ));
            }

            let next = self
                .stream
                .next()
                .await
                .context("SSE stream closed unexpectedly")??;
            self.buffer.push_str(std::str::from_utf8(&next)?);
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn remote_health_and_request_proxy_require_auth() -> Result<()> {
    let daemon = DaemonHarness::start().await?;
    let remote = RemoteHarness::start(&daemon.config_path).await?;
    let client = reqwest::Client::new();

    let public_health = client
        .get(format!("{}/healthz", remote.base_url))
        .send()
        .await?;
    assert_eq!(public_health.status(), reqwest::StatusCode::OK);

    let unauthorized = client
        .get(format!("{}/v1/health", remote.base_url))
        .send()
        .await?;
    assert_eq!(unauthorized.status(), reqwest::StatusCode::UNAUTHORIZED);

    let authorized = client
        .get(format!("{}/v1/health", remote.base_url))
        .header(AUTHORIZATION, "Bearer test-token")
        .send()
        .await?;
    assert_eq!(authorized.status(), reqwest::StatusCode::OK);
    let authorized: Value = authorized.json().await?;
    assert_eq!(authorized["remote"]["ready"], true);
    assert_eq!(
        authorized["daemon"]["protocol_version"],
        DAEMON_PROTOCOL_VERSION
    );

    let request = client
        .post(format!("{}/v1/daemon/request", remote.base_url))
        .header(AUTHORIZATION, "Bearer test-token")
        .json(&json!({
            "op": "daemon.ping",
            "params": {}
        }))
        .send()
        .await?;
    assert_eq!(request.status(), reqwest::StatusCode::OK);
    let request: Value = request.json().await?;
    assert_eq!(request["ok"], true);
    assert_eq!(
        request["result"]["protocol_version"],
        DAEMON_PROTOCOL_VERSION
    );

    remote.stop().await?;
    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn remote_sse_streams_runtime_events() -> Result<()> {
    let daemon = DaemonHarness::start().await?;
    let remote = RemoteHarness::start(&daemon.config_path).await?;
    let client = reqwest::Client::new();

    let response = client
        .get(format!("{}/v1/events", remote.base_url))
        .header(AUTHORIZATION, "Bearer test-token")
        .send()
        .await?;
    assert_eq!(response.status(), reqwest::StatusCode::OK);

    let mut reader = SseReader {
        stream: response.bytes_stream(),
        buffer: String::new(),
    };
    let (first_event, _) = timeout(Duration::from_secs(5), reader.next_event())
        .await
        .context("timed out waiting for initial SSE event")??;
    assert_eq!(first_event, "runtime.snapshot");

    let created = client
        .post(format!("{}/v1/daemon/request", remote.base_url))
        .header(AUTHORIZATION, "Bearer test-token")
        .json(&json!({
            "op": "agent.create",
            "params": {
                "id": "remote-reviewer",
                "provider": "mock",
                "model": "mock-model",
                "system_prompt": "Review from remote",
                "enabled": true
            }
        }))
        .send()
        .await?;
    assert_eq!(created.status(), reqwest::StatusCode::OK);
    let created: Value = created.json().await?;
    assert_eq!(created["ok"], true);

    let deadline = Instant::now() + Duration::from_secs(5);
    let mut saw_rescan = false;
    while Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let (event_name, data) = timeout(remaining, reader.next_event())
            .await
            .context("timed out waiting for runtime.rescanned SSE event")??;
        if event_name == "runtime.rescanned" {
            saw_rescan = data["registry"]["agents"]
                .as_array()
                .is_some_and(|agents| agents.iter().any(|agent| agent["id"] == "remote-reviewer"));
            if saw_rescan {
                break;
            }
        }
    }

    assert!(
        saw_rescan,
        "runtime.rescanned event should include new agent"
    );

    remote.stop().await?;
    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn remote_websocket_streams_runtime_events() -> Result<()> {
    let daemon = DaemonHarness::start().await?;
    let remote = RemoteHarness::start(&daemon.config_path).await?;
    let mut request =
        format!("ws://{}/v1/events/ws", remote.server.local_addr()).into_client_request()?;
    request
        .headers_mut()
        .insert(AUTHORIZATION, "Bearer test-token".parse()?);
    let stream = tokio::net::TcpStream::connect(remote.server.local_addr()).await?;
    let (mut websocket, _) = client_async(request, stream).await?;

    let first = next_websocket_text(&mut websocket).await?;
    let first: Value = serde_json::from_str(&first)?;
    assert_eq!(first["event"], "runtime.snapshot");
    assert_eq!(first["data"]["agent_runtimes"][0]["agent_id"], "default");

    let client = reqwest::Client::new();
    let created = client
        .post(format!("{}/v1/daemon/request", remote.base_url))
        .header(AUTHORIZATION, "Bearer test-token")
        .json(&json!({
            "op": "agent.create",
            "params": {
                "id": "remote-ws-reviewer",
                "provider": "mock",
                "model": "mock-model",
                "system_prompt": "Review from websocket",
                "enabled": true
            }
        }))
        .send()
        .await?;
    assert_eq!(created.status(), reqwest::StatusCode::OK);

    let deadline = Instant::now() + Duration::from_secs(5);
    let mut saw_rescan = false;
    while Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let next = timeout(remaining, websocket.next())
            .await
            .context("timed out waiting for runtime.rescanned websocket event")?
            .context("websocket stream closed unexpectedly")??;
        let Message::Text(next) = next else {
            continue;
        };
        let payload: Value = serde_json::from_str(&next)?;
        if payload["event"] == "runtime.rescanned" {
            saw_rescan = payload["data"]["registry"]["agents"]
                .as_array()
                .is_some_and(|agents| {
                    agents
                        .iter()
                        .any(|agent| agent["id"] == "remote-ws-reviewer")
                });
            if saw_rescan {
                break;
            }
        }
    }

    assert!(
        saw_rescan,
        "runtime.rescanned websocket event should include new agent"
    );

    websocket.close(None).await?;
    remote.stop().await?;
    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn remote_refuses_non_loopback_bind_without_opt_in() -> Result<()> {
    let daemon = DaemonHarness::start().await?;
    let err = start_remote(
        &daemon.config_path,
        RemoteServeOptions {
            bind: Some("0.0.0.0:0".to_string()),
            auth_token: Some("test-token".to_string()),
            auth_token_env: None,
            event_keepalive_seconds: Some(1),
            allow_non_loopback: Some(false),
        },
    )
    .await
    .expect_err("non-loopback bind should require explicit opt-in");
    assert!(err.to_string().contains("allow_non_loopback"));
    daemon.stop().await
}

async fn next_websocket_text<S>(websocket: &mut S) -> Result<String>
where
    S: futures::Stream<Item = std::result::Result<Message, tokio_tungstenite::tungstenite::Error>>
        + Unpin,
{
    let deadline = Instant::now() + Duration::from_secs(5);
    while Instant::now() < deadline {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let next = timeout(remaining, websocket.next())
            .await
            .context("timed out waiting for websocket text frame")?
            .context("websocket stream closed unexpectedly")??;
        match next {
            Message::Text(text) => return Ok(text.to_string()),
            Message::Ping(_) | Message::Pong(_) | Message::Binary(_) | Message::Frame(_) => {}
            Message::Close(_) => return Err(anyhow!("websocket closed before text frame")),
        }
    }
    Err(anyhow!("timed out waiting for websocket text frame"))
}
