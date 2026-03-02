use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use serde_json::Value;
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::net::UnixStream;
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::task::JoinHandle;
use tokio::time::{Instant, sleep, timeout};
use turin::daemon::protocol::{DaemonRequest, EventEnvelope, RequestEnvelope, ResponseEnvelope};

struct DaemonHarness {
    _tempdir: TempDir,
    socket_path: PathBuf,
    join: JoinHandle<Result<()>>,
}

struct EventSubscription {
    _writer: OwnedWriteHalf,
    lines: Lines<BufReader<OwnedReadHalf>>,
}

impl DaemonHarness {
    async fn start() -> Result<Self> {
        let tempdir = tempfile::tempdir()?;
        let workspace_root = tempdir.path().join("workspace");
        let harness_dir = workspace_root.join(".turin/harnesses");
        let agents_dir = workspace_root.join("agents");
        let harnesses_dir = workspace_root.join("harnesses");

        std::fs::create_dir_all(&harness_dir)?;
        std::fs::create_dir_all(&agents_dir)?;
        std::fs::create_dir_all(&harnesses_dir)?;
        std::fs::write(
            harness_dir.join("main.lua"),
            "-- daemon integration harness\n",
        )?;

        let config_path = tempdir.path().join("turin.toml");
        let config_toml = format!(
            r#"[agent]
id = "default"
model = "mock-model"
provider = "mock"
system_prompt = "Harness example test"

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
        let socket_path = workspace_root.join(".turin/daemon.sock");

        let serve_config_path = config_path.clone();
        let join =
            tokio::spawn(async move { turin::daemon::server::serve(&serve_config_path).await });

        let deadline = Instant::now() + Duration::from_secs(5);
        while !socket_path.exists() {
            if join.is_finished() {
                let result = join
                    .await
                    .context("daemon task join failed before socket bind")?;
                return Err(result
                    .err()
                    .unwrap_or_else(|| anyhow!("daemon exited before creating socket")));
            }
            if Instant::now() >= deadline {
                join.abort();
                return Err(anyhow!(
                    "Timed out waiting for daemon socket '{}'",
                    socket_path.display()
                ));
            }
            sleep(Duration::from_millis(25)).await;
        }

        Ok(Self {
            _tempdir: tempdir,
            socket_path,
            join,
        })
    }

    async fn request(&self, request: DaemonRequest) -> Result<ResponseEnvelope> {
        let stream = UnixStream::connect(&self.socket_path)
            .await
            .with_context(|| {
                format!(
                    "Failed to connect to daemon socket '{}'",
                    self.socket_path.display()
                )
            })?;
        let (reader, mut writer) = stream.into_split();
        let request = RequestEnvelope::new(Some(format!("req-{}", uuid::Uuid::new_v4())), request);
        writer
            .write_all(serde_json::to_string(&request)?.as_bytes())
            .await?;
        writer.write_all(b"\n").await?;

        let mut lines = BufReader::new(reader).lines();
        let Some(line) = lines.next_line().await? else {
            return Err(anyhow!("Daemon closed connection without a response"));
        };
        Ok(serde_json::from_str(&line)?)
    }

    async fn subscribe(&self) -> Result<(ResponseEnvelope, EventEnvelope, EventSubscription)> {
        let stream = UnixStream::connect(&self.socket_path).await?;
        let (reader, mut writer) = stream.into_split();
        let request = RequestEnvelope::new(
            Some(format!("req-{}", uuid::Uuid::new_v4())),
            DaemonRequest::RuntimeEventsSubscribe(Default::default()),
        );
        writer
            .write_all(serde_json::to_string(&request)?.as_bytes())
            .await?;
        writer.write_all(b"\n").await?;

        let mut lines = BufReader::new(reader).lines();
        let ack: ResponseEnvelope =
            serde_json::from_str(&lines.next_line().await?.context("missing subscribe ack")?)?;
        let snapshot: EventEnvelope = serde_json::from_str(
            &lines
                .next_line()
                .await?
                .context("missing runtime.snapshot event")?,
        )?;

        Ok((
            ack,
            snapshot,
            EventSubscription {
                _writer: writer,
                lines,
            },
        ))
    }

    async fn stop(self) -> Result<()> {
        let response = self
            .request(DaemonRequest::DaemonStop(Default::default()))
            .await?;
        if !response.ok {
            return Err(anyhow!("daemon.stop failed: {:?}", response.error));
        }
        let _ = timeout(Duration::from_secs(5), self.join)
            .await
            .context("timed out waiting for daemon to exit")??;
        Ok(())
    }
}

impl EventSubscription {
    async fn next_event(&mut self) -> Result<EventEnvelope> {
        let line = self
            .lines
            .next_line()
            .await?
            .context("event stream closed unexpectedly")?;
        Ok(serde_json::from_str(&line)?)
    }

    async fn wait_for(&mut self, event_name: &str) -> Result<EventEnvelope> {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let event = timeout(remaining, self.next_event())
                .await
                .with_context(|| format!("timed out waiting for event '{}'", event_name))??;
            if event.event == event_name {
                return Ok(event);
            }
        }
    }
}

fn result_value(response: ResponseEnvelope) -> Value {
    assert!(response.ok, "daemon response failed: {:?}", response.error);
    response.result.expect("daemon response missing result")
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_agent_crud_round_trip_over_socket() -> Result<()> {
    let daemon = DaemonHarness::start().await?;

    let created = result_value(
        daemon
            .request(DaemonRequest::AgentCreate(
                turin::daemon::protocol::CreateAgentParams {
                    id: "docs-reviewer".to_string(),
                    provider: "mock".to_string(),
                    model: "mock-model".to_string(),
                    system_prompt: Some("Review docs".to_string()),
                    thinking: None,
                    mode: None,
                    harness: None,
                    idle_grace_secs: Some(45),
                    enabled: true,
                },
            ))
            .await?,
    );
    assert_eq!(created["id"], "docs-reviewer");
    assert_eq!(created["enabled"], true);

    let fetched = result_value(
        daemon
            .request(DaemonRequest::AgentGet(
                turin::daemon::protocol::EntityIdParams {
                    id: "docs-reviewer".to_string(),
                },
            ))
            .await?,
    );
    assert_eq!(fetched["provider"], "mock");

    let updated = result_value(
        daemon
            .request(DaemonRequest::AgentUpdate(
                turin::daemon::protocol::UpdateAgentParams {
                    id: "docs-reviewer".to_string(),
                    provider: None,
                    model: Some("mock-model-v2".to_string()),
                    system_prompt: Some("Review docs carefully".to_string()),
                    thinking: None,
                    mode: None,
                    idle_grace_secs: Some(60),
                },
            ))
            .await?,
    );
    assert_eq!(updated["model"], "mock-model-v2");

    let disabled = result_value(
        daemon
            .request(DaemonRequest::AgentDisable(
                turin::daemon::protocol::EntityIdParams {
                    id: "docs-reviewer".to_string(),
                },
            ))
            .await?,
    );
    assert_eq!(disabled["enabled"], false);

    let deleted = result_value(
        daemon
            .request(DaemonRequest::AgentDelete(
                turin::daemon::protocol::EntityIdParams {
                    id: "docs-reviewer".to_string(),
                },
            ))
            .await?,
    );
    let agents = deleted["registry"]["agents"]
        .as_array()
        .expect("registry agents array");
    assert!(!agents.iter().any(|agent| agent["id"] == "docs-reviewer"));

    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_task_wait_and_session_round_trip_over_socket() -> Result<()> {
    let daemon = DaemonHarness::start().await?;

    let submitted = result_value(
        daemon
            .request(DaemonRequest::TaskSubmit(
                turin::daemon::protocol::SubmitTaskParams {
                    agent_id: "default".to_string(),
                    prompt: "Say pong".to_string(),
                },
            ))
            .await?,
    );
    let request_id = submitted["request_id"]
        .as_str()
        .expect("request id")
        .to_string();

    let completed = result_value(
        daemon
            .request(DaemonRequest::TaskWait(
                turin::daemon::protocol::WaitTaskParams {
                    request_id,
                    timeout_ms: Some(5_000),
                },
            ))
            .await?,
    );
    assert_eq!(completed["state"], "completed");
    assert_eq!(completed["output"], "PONG");

    let sessions = result_value(
        daemon
            .request(DaemonRequest::SessionList(
                turin::daemon::protocol::SessionListParams {
                    limit: 10,
                    offset: 0,
                },
            ))
            .await?,
    );
    let session_id = sessions["sessions"][0]["session_id"]
        .as_str()
        .expect("session id")
        .to_string();

    let session = result_value(
        daemon
            .request(DaemonRequest::SessionGet(
                turin::daemon::protocol::SessionIdParams { session_id },
            ))
            .await?,
    );
    assert_eq!(session["session"]["agent_id"], "default");
    assert!(
        session["messages"]
            .as_array()
            .expect("messages array")
            .iter()
            .any(|message| message["role"] == "assistant")
    );

    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_event_subscription_receives_snapshot_and_mutation() -> Result<()> {
    let daemon = DaemonHarness::start().await?;
    let (ack, snapshot, mut subscription) = daemon.subscribe().await?;

    assert!(ack.ok, "subscription ack failed: {:?}", ack.error);
    assert_eq!(snapshot.event, "runtime.snapshot");

    let _ = daemon
        .request(DaemonRequest::AgentCreate(
            turin::daemon::protocol::CreateAgentParams {
                id: "writer".to_string(),
                provider: "mock".to_string(),
                model: "mock-model".to_string(),
                system_prompt: Some("Write".to_string()),
                thinking: None,
                mode: None,
                harness: None,
                idle_grace_secs: None,
                enabled: true,
            },
        ))
        .await?;

    let created = subscription.wait_for("agent.created").await?;
    assert_eq!(created.data["id"], "writer");

    daemon.stop().await
}
