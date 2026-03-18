use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use futures::StreamExt;
use reqwest::header::AUTHORIZATION;
use serde_json::{Value, json};
use tempfile::TempDir;
use tokio::task::JoinHandle;
use tokio::time::{Instant, sleep, timeout};
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
        let harness_dir = workspace_root.join(".turin/harnesses");
        let agents_dir = workspace_root.join("agents");
        let harnesses_dir = workspace_root.join("harnesses");

        std::fs::create_dir_all(&harness_dir)?;
        std::fs::create_dir_all(&agents_dir)?;
        std::fs::create_dir_all(&harnesses_dir)?;
        std::fs::write(
            harness_dir.join("main.lua"),
            "-- remote integration harness\n",
        )?;

        let config_path = tempdir.path().join("turin.toml");
        let config_toml = format!(
            r#"[agent]
id = "default"
model = "mock-model"
provider = "mock"
system_prompt = "Remote integration"

[kernel]
workspace_root = "{workspace_root}"
max_turns = 4
heartbeat_interval_secs = 30
initial_spawn_depth = 0

[persistence]
database_path = "{database_path}"

[harness]
directory = "{harness_directory}"
fs_root = "."

[providers.mock]
type = "mock"
base_url = "PONG"
"#,
            workspace_root = workspace_root.display(),
            database_path = workspace_root.join("test.db").display(),
            harness_directory = harness_dir.display(),
        );
        std::fs::write(&config_path, config_toml)?;

        let endpoint = workspace_root.join(".turin/daemon.sock");
        let serve_config_path = config_path.clone();
        let join =
            tokio::spawn(async move { turin::daemon::server::serve(&serve_config_path).await });

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
    async fn start(config_path: &PathBuf) -> Result<Self> {
        let server = start_remote(
            config_path,
            RemoteServeOptions {
                bind: Some("127.0.0.1:0".to_string()),
                auth_token: Some("test-token".to_string()),
                auth_token_env: None,
                event_keepalive_secs: Some(1),
            },
        )
        .await?;
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
