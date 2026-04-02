use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use serde_json::Value;
use tempfile::TempDir;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::task::JoinHandle;
use tokio::time::{Instant, sleep, timeout};
use turin::daemon::protocol::{DaemonRequest, EventEnvelope, RequestEnvelope, ResponseEnvelope};
use turin::kernel::session_refs::parse_session_reference;
use turin::persistence::state::StateStore;
use turin_daemon_protocol::DAEMON_PROTOCOL_VERSION;
use turin_local_ipc::{
    LocalIpcReadHalf, LocalIpcWriteHalf, connect as connect_local_ipc, current_transport_name,
    split as split_local_ipc,
};

struct DaemonHarness {
    tempdir: std::sync::Arc<TempDir>,
    endpoint: PathBuf,
    join: JoinHandle<Result<()>>,
}

struct EventSubscription {
    _writer: LocalIpcWriteHalf,
    lines: Lines<BufReader<LocalIpcReadHalf>>,
}

impl DaemonHarness {
    async fn start() -> Result<Self> {
        let tempdir = std::sync::Arc::new(tempfile::tempdir()?);
        Self::start_in(tempdir).await
    }

    async fn start_in(tempdir: std::sync::Arc<TempDir>) -> Result<Self> {
        let workspace_root = tempdir.path().join("workspace");
        let harness_dir = workspace_root.join(".turin/harnesses");
        let agents_dir = workspace_root.join(".turin/runtime/agents");
        let channels_dir = workspace_root.join(".turin/runtime/channels");
        let harnesses_dir = workspace_root.join(".turin/harnesses");

        std::fs::create_dir_all(&harness_dir)?;
        std::fs::create_dir_all(&agents_dir)?;
        std::fs::create_dir_all(&channels_dir)?;
        std::fs::create_dir_all(&harnesses_dir)?;
        std::fs::write(
            harness_dir.join("main.lua"),
            "-- daemon integration harness\n",
        )?;

        let config_path = workspace_root.join(".turin/config.toml");
        std::fs::create_dir_all(config_path.parent().expect("config parent"))?;
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

[persistence.state]
path = "{database_path}"

[harness]
directory = "{harness_directory}"
fs_root = "."

[providers.mock]
type = "mock"
base_url = "PONG"

[remote]
bind = "127.0.0.1:0"
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

        let deadline = Instant::now() + Duration::from_secs(5);
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
                join.abort();
                return Err(anyhow!(
                    "Timed out waiting for daemon endpoint '{}'",
                    endpoint.display()
                ));
            }
            sleep(Duration::from_millis(25)).await;
        }

        Ok(Self {
            tempdir,
            endpoint,
            join,
        })
    }

    async fn restart(self) -> Result<Self> {
        let tempdir = std::sync::Arc::clone(&self.tempdir);
        self.stop().await?;
        Self::start_in(tempdir).await
    }

    async fn request(&self, request: DaemonRequest) -> Result<ResponseEnvelope> {
        let stream = connect_local_ipc(&self.endpoint).await.with_context(|| {
            format!(
                "Failed to connect to daemon endpoint '{}'",
                self.endpoint.display()
            )
        })?;
        let (reader, mut writer) = split_local_ipc(stream);
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

    async fn subscribe(
        &self,
        params: turin::daemon::protocol::RuntimeEventsSubscribeParams,
    ) -> Result<(ResponseEnvelope, EventEnvelope, EventSubscription)> {
        let stream = connect_local_ipc(&self.endpoint).await?;
        let (reader, mut writer) = split_local_ipc(stream);
        let request = RequestEnvelope::new(
            Some(format!("req-{}", uuid::Uuid::new_v4())),
            DaemonRequest::RuntimeEventsSubscribe(params),
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

    async fn expect_no_event(&mut self, timeout_ms: u64) -> Result<()> {
        match timeout(Duration::from_millis(timeout_ms), self.next_event()).await {
            Ok(Ok(event)) => Err(anyhow!("unexpected event: {}", event.event)),
            Ok(Err(err)) => Err(err),
            Err(_) => Ok(()),
        }
    }
}

fn result_value(response: ResponseEnvelope) -> Value {
    assert!(response.ok, "daemon response failed: {:?}", response.error);
    response.result.expect("daemon response missing result")
}

async fn wait_for_channel_state(
    daemon: &DaemonHarness,
    channel_id: &str,
    expected: &str,
    timeout_secs: u64,
) -> Result<Value> {
    let deadline = Instant::now() + Duration::from_secs(timeout_secs);
    let mut last_state = None;
    loop {
        let response = daemon
            .request(DaemonRequest::ChannelStatus(
                turin::daemon::protocol::EntityIdParams {
                    id: channel_id.to_string(),
                },
            ))
            .await?;
        if response.ok {
            let result = response.result.context("channel.status missing result")?;
            let state = result
                .get("state")
                .and_then(|value| value.as_str())
                .unwrap_or("unknown");
            last_state = Some(state.to_string());
            if state == expected {
                return Ok(result);
            }
        }

        if Instant::now() >= deadline {
            return Err(anyhow!(
                "Timed out waiting for channel '{}' to reach state '{}' (last state: {:?})",
                channel_id,
                expected,
                last_state
            ));
        }
        sleep(Duration::from_millis(25)).await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_agent_crud_round_trip_over_endpoint() -> Result<()> {
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
                    tools: Default::default(),
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
                    tools: None,
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
async fn daemon_task_wait_and_session_round_trip_over_endpoint() -> Result<()> {
    let daemon = DaemonHarness::start().await?;

    let live = result_value(
        daemon
            .request(DaemonRequest::SessionOpen(
                turin::daemon::protocol::OpenSessionParams {
                    agent_id: "default".to_string(),
                    slot_id: Some("chat-thread-1".to_string()),
                    channel_id: None,
                },
            ))
            .await?,
    );
    let live_session_id = live["session_id"]
        .as_str()
        .expect("live session id")
        .to_string();
    assert_eq!(live["slot_id"], "chat-thread-1");

    let live_sessions = result_value(
        daemon
            .request(DaemonRequest::SessionListLive(
                turin::daemon::protocol::NoParams::default(),
            ))
            .await?,
    );
    assert!(
        live_sessions["sessions"]
            .as_array()
            .expect("live sessions array")
            .iter()
            .any(|session| session["session_id"] == live_session_id)
    );

    let submitted = result_value(
        daemon
            .request(DaemonRequest::TaskSubmit(
                turin::daemon::protocol::SubmitTaskParams {
                    agent_id: None,
                    session_id: Some(live_session_id.clone()),
                    prompt: "Say pong".to_string(),
                    tools: Default::default(),
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
    assert_eq!(completed["slot_id"], "chat-thread-1");
    assert_eq!(completed["output"], "PONG");

    let sessions = result_value(
        daemon
            .request(DaemonRequest::SessionList(
                turin::daemon::protocol::SessionListParams {
                    limit: 10,
                    offset: 0,
                    store: None,
                    path: None,
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
async fn daemon_session_resume_round_trip_over_restart() -> Result<()> {
    let daemon = DaemonHarness::start().await?;

    let opened = result_value(
        daemon
            .request(DaemonRequest::SessionOpen(
                turin::daemon::protocol::OpenSessionParams {
                    agent_id: "default".to_string(),
                    slot_id: Some("restart-thread".to_string()),
                    channel_id: None,
                },
            ))
            .await?,
    );
    let session_id = opened["session_id"]
        .as_str()
        .context("session.open should return session_id")?
        .to_string();

    let submitted = result_value(
        daemon
            .request(DaemonRequest::TaskSubmit(
                turin::daemon::protocol::SubmitTaskParams {
                    agent_id: None,
                    session_id: Some(session_id.clone()),
                    prompt: "resume me".to_string(),
                    tools: Default::default(),
                },
            ))
            .await?,
    );
    let request_id = submitted["request_id"]
        .as_str()
        .context("task.submit should return request_id")?
        .to_string();
    let waited = result_value(
        daemon
            .request(DaemonRequest::TaskWait(
                turin::daemon::protocol::WaitTaskParams {
                    request_id,
                    timeout_ms: Some(5_000),
                },
            ))
            .await?,
    );
    assert_eq!(waited["status"], "success");

    wait_for_persisted_user_messages(&daemon, &session_id, 1).await?;

    let daemon = daemon.restart().await?;

    let resumed = result_value(
        daemon
            .request(DaemonRequest::SessionResume(
                turin::daemon::protocol::ResumeSessionParams {
                    session_id: session_id.clone(),
                    slot_id: Some("restart-thread".to_string()),
                },
            ))
            .await?,
    );
    assert_eq!(resumed["session_id"], session_id);
    assert_eq!(resumed["slot_id"], "restart-thread");

    let resubmitted = result_value(
        daemon
            .request(DaemonRequest::TaskSubmit(
                turin::daemon::protocol::SubmitTaskParams {
                    agent_id: None,
                    session_id: Some(session_id.clone()),
                    prompt: "resume me again".to_string(),
                    tools: Default::default(),
                },
            ))
            .await?,
    );
    let request_id = resubmitted["request_id"]
        .as_str()
        .context("resumed task.submit should return request_id")?
        .to_string();
    let waited = result_value(
        daemon
            .request(DaemonRequest::TaskWait(
                turin::daemon::protocol::WaitTaskParams {
                    request_id,
                    timeout_ms: Some(5_000),
                },
            ))
            .await?,
    );
    assert_eq!(waited["status"], "success");

    wait_for_persisted_user_messages(&daemon, &session_id, 2).await?;

    daemon.stop().await
}

async fn wait_for_persisted_user_messages(
    daemon: &DaemonHarness,
    session_id: &str,
    expected_user_count: usize,
) -> Result<()> {
    let persistence_deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let detail = result_value(
            daemon
                .request(DaemonRequest::SessionGet(
                    turin::daemon::protocol::SessionIdParams {
                        session_id: session_id.to_string(),
                    },
                ))
                .await?,
        );
        let messages = detail["messages"]
            .as_array()
            .context("session detail should include messages")?;
        let persisted_user_count = messages
            .iter()
            .filter(|message| message["role"] == "user")
            .count();
        if persisted_user_count >= expected_user_count {
            return Ok(());
        }
        if Instant::now() >= persistence_deadline {
            anyhow::bail!(
                "Timed out waiting for persisted session history to reach {} user messages",
                expected_user_count
            );
        }
        sleep(Duration::from_millis(25)).await;
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_event_subscription_receives_snapshot_and_mutation() -> Result<()> {
    let daemon = DaemonHarness::start().await?;
    let (ack, snapshot, mut subscription) = daemon.subscribe(Default::default()).await?;

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
                tools: Default::default(),
            },
        ))
        .await?;

    let created = subscription.wait_for("agent.created").await?;
    assert_eq!(created.data["id"], "writer");

    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_ping_exposes_typed_handshake() -> Result<()> {
    let daemon = DaemonHarness::start().await?;
    let client = turin_daemon_client::DaemonClient::new(&daemon.endpoint);

    let handshake = client.handshake().await?;
    assert!(handshake.pong);
    assert_eq!(handshake.protocol_version, DAEMON_PROTOCOL_VERSION);
    assert_eq!(handshake.transport, current_transport_name());
    assert_eq!(handshake.wire_format, "ndjson");
    assert!(handshake.capabilities.runtime_snapshot_v1);
    assert!(handshake.capabilities.scoped_event_snapshots);
    assert!(handshake.capabilities.lag_resnapshot);
    assert!(handshake.capabilities.watcher_rescan_failed_events);
    assert!(handshake.capabilities.channels);

    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_managed_subscription_reconnects_after_restart() -> Result<()> {
    let daemon = DaemonHarness::start().await?;
    let client = turin_daemon_client::DaemonClient::new(&daemon.endpoint);
    let mut managed = client.subscribe_managed(Default::default()).await?;

    let snapshot = timeout(Duration::from_secs(5), managed.next_event())
        .await
        .context("timed out waiting for initial managed runtime snapshot")??;
    assert_eq!(snapshot.event, "runtime.snapshot");

    let created = result_value(
        daemon
            .request(DaemonRequest::AgentCreate(
                turin::daemon::protocol::CreateAgentParams {
                    id: "before-restart".to_string(),
                    provider: "mock".to_string(),
                    model: "mock-model".to_string(),
                    system_prompt: Some("Before restart".to_string()),
                    thinking: None,
                    mode: None,
                    harness: None,
                    idle_grace_secs: None,
                    enabled: true,
                    tools: Default::default(),
                },
            ))
            .await?,
    );
    assert_eq!(created["id"], "before-restart");

    loop {
        let event = timeout(Duration::from_secs(5), managed.next_event())
            .await
            .context("timed out waiting for pre-restart agent.created")??;
        if event.event == "agent.created" {
            assert_eq!(event.data["id"], "before-restart");
            break;
        }
    }

    let daemon = daemon.restart().await?;

    let deadline = Instant::now() + Duration::from_secs(10);
    let snapshot = loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        let event = timeout(remaining, managed.next_event())
            .await
            .context("timed out waiting for managed resubscribe snapshot after restart")??;
        if event.event == "runtime.snapshot" {
            break event;
        }
    };
    assert!(
        snapshot.data["endpoint"]
            .as_str()
            .is_some_and(|endpoint| !endpoint.is_empty()),
        "managed runtime snapshot should include endpoint after reconnect"
    );

    let created = result_value(
        daemon
            .request(DaemonRequest::AgentCreate(
                turin::daemon::protocol::CreateAgentParams {
                    id: "after-restart".to_string(),
                    provider: "mock".to_string(),
                    model: "mock-model".to_string(),
                    system_prompt: Some("After restart".to_string()),
                    thinking: None,
                    mode: None,
                    harness: None,
                    idle_grace_secs: None,
                    enabled: true,
                    tools: Default::default(),
                },
            ))
            .await?,
    );
    assert_eq!(created["id"], "after-restart");

    loop {
        let event = timeout(Duration::from_secs(5), managed.next_event())
            .await
            .context("timed out waiting for post-restart agent.created")??;
        if event.event == "agent.created" {
            assert_eq!(event.data["id"], "after-restart");
            break;
        }
    }

    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_event_subscription_filters_by_agent_and_session() -> Result<()> {
    let daemon = DaemonHarness::start().await?;

    let (_ack, _snapshot, mut agent_subscription) = daemon
        .subscribe(turin::daemon::protocol::RuntimeEventsSubscribeParams {
            agent_id: Some("writer".to_string()),
            session_id: None,
        })
        .await?;

    let _ = daemon
        .request(DaemonRequest::AgentCreate(
            turin::daemon::protocol::CreateAgentParams {
                id: "other".to_string(),
                provider: "mock".to_string(),
                model: "mock-model".to_string(),
                system_prompt: Some("Other".to_string()),
                thinking: None,
                mode: None,
                harness: None,
                idle_grace_secs: None,
                enabled: true,
                tools: Default::default(),
            },
        ))
        .await?;
    agent_subscription.expect_no_event(250).await?;

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
                tools: Default::default(),
            },
        ))
        .await?;
    let created = agent_subscription.wait_for("agent.created").await?;
    assert_eq!(created.data["id"], "writer");

    let opened = result_value(
        daemon
            .request(DaemonRequest::SessionOpen(
                turin::daemon::protocol::OpenSessionParams {
                    agent_id: "default".to_string(),
                    slot_id: Some("filter-session".to_string()),
                    channel_id: None,
                },
            ))
            .await?,
    );
    let session_id = opened["session_id"]
        .as_str()
        .context("session.open should return session_id")?
        .to_string();

    let (_ack, _snapshot, mut session_subscription) = daemon
        .subscribe(turin::daemon::protocol::RuntimeEventsSubscribeParams {
            agent_id: None,
            session_id: Some(session_id.clone()),
        })
        .await?;

    let _ = daemon
        .request(DaemonRequest::SessionOpen(
            turin::daemon::protocol::OpenSessionParams {
                agent_id: "default".to_string(),
                slot_id: Some("other-session".to_string()),
                channel_id: None,
            },
        ))
        .await?;
    session_subscription.expect_no_event(250).await?;

    let _ = daemon
        .request(DaemonRequest::SessionResume(
            turin::daemon::protocol::ResumeSessionParams {
                session_id: session_id.clone(),
                slot_id: Some("filter-session".to_string()),
            },
        ))
        .await?;
    let resumed = session_subscription.wait_for("session.resumed").await?;
    assert_eq!(resumed.data["session_id"], session_id);

    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_session_subscription_receives_kernel_stream_events() -> Result<()> {
    let daemon = DaemonHarness::start().await?;

    let opened = result_value(
        daemon
            .request(DaemonRequest::SessionOpen(
                turin::daemon::protocol::OpenSessionParams {
                    agent_id: "default".to_string(),
                    slot_id: Some("stream-session".to_string()),
                    channel_id: None,
                },
            ))
            .await?,
    );
    let session_id = opened["session_id"]
        .as_str()
        .context("session.open should return session_id")?
        .to_string();

    let (_ack, _snapshot, mut subscription) = daemon
        .subscribe(turin::daemon::protocol::RuntimeEventsSubscribeParams {
            agent_id: None,
            session_id: Some(session_id.clone()),
        })
        .await?;

    let submitted = result_value(
        daemon
            .request(DaemonRequest::TaskSubmit(
                turin::daemon::protocol::SubmitTaskParams {
                    agent_id: None,
                    session_id: Some(session_id.clone()),
                    prompt: "Say pong".to_string(),
                    tools: Default::default(),
                },
            ))
            .await?,
    );
    let request_id = submitted["request_id"]
        .as_str()
        .context("task.submit should return request_id")?
        .to_string();

    let task_start = subscription.wait_for("task_start").await?;
    assert_eq!(task_start.data["session_id"], session_id);
    assert_eq!(task_start.data["agent_id"], "default");

    let message_delta = subscription.wait_for("message_delta").await?;
    assert_eq!(message_delta.data["session_id"], session_id);
    assert_eq!(message_delta.data["agent_id"], "default");
    assert_eq!(message_delta.data["content_delta"], "PONG");

    let _ = daemon
        .request(DaemonRequest::TaskWait(
            turin::daemon::protocol::WaitTaskParams {
                request_id,
                timeout_ms: Some(5_000),
            },
        ))
        .await?;

    let task_complete = subscription.wait_for("task_complete").await?;
    assert_eq!(task_complete.data["session_id"], session_id);

    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_event_subscription_scopes_initial_snapshot_and_includes_channel_runtimes()
-> Result<()> {
    let daemon = DaemonHarness::start().await?;

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
                tools: Default::default(),
            },
        ))
        .await?;

    let _ = daemon
        .request(DaemonRequest::ChannelCreate(
            turin::daemon::protocol::CreateChannelParams {
                id: "writer-fs".to_string(),
                kind: "fs".to_string(),
                agent_id: "writer".to_string(),
                idle_ttl_secs: Some(600),
                enabled: true,
                settings: Some(serde_json::json!({
                    "inbox_dir": "inbox",
                    "outbox_dir": "outbox",
                    "processed_dir": "processed",
                    "failed_dir": "failed",
                    "poll_interval_ms": 25,
                })),
            },
        ))
        .await?;

    let (_ack, snapshot, _subscription) = daemon
        .subscribe(turin::daemon::protocol::RuntimeEventsSubscribeParams {
            agent_id: Some("writer".to_string()),
            session_id: None,
        })
        .await?;

    assert_eq!(snapshot.event, "runtime.snapshot");
    let agents = snapshot.data["registry"]["agents"]
        .as_array()
        .context("snapshot registry.agents should be an array")?;
    assert_eq!(agents.len(), 1);
    assert_eq!(agents[0]["id"], "writer");
    let channel_runtimes = snapshot.data["channel_runtimes"]
        .as_array()
        .context("snapshot channel_runtimes should be an array")?;
    assert_eq!(channel_runtimes.len(), 1);
    assert_eq!(channel_runtimes[0]["agent_id"], "writer");

    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_event_subscription_receives_channel_runtime_events() -> Result<()> {
    let daemon = DaemonHarness::start().await?;
    let (_ack, _snapshot, mut subscription) = daemon.subscribe(Default::default()).await?;

    let _created = result_value(
        daemon
            .request(DaemonRequest::ChannelCreate(
                turin::daemon::protocol::CreateChannelParams {
                    id: "fs-events".to_string(),
                    kind: "fs".to_string(),
                    agent_id: "default".to_string(),
                    idle_ttl_secs: Some(600),
                    enabled: true,
                    settings: Some(serde_json::json!({
                        "inbox_dir": "inbox",
                        "outbox_dir": "outbox",
                        "processed_dir": "processed",
                        "failed_dir": "failed",
                        "poll_interval_ms": 25,
                    })),
                },
            ))
            .await?,
    );

    let deadline = Instant::now() + Duration::from_secs(5);
    let mut saw_update = false;
    while Instant::now() < deadline {
        let event = timeout(Duration::from_millis(750), subscription.next_event()).await;
        let Ok(Ok(event)) = event else {
            continue;
        };
        if event.event == "channel.runtime.updated" && event.data["id"] == "fs-events" {
            saw_update = true;
            break;
        }
    }
    assert!(saw_update, "expected channel.runtime.updated for fs-events");

    let _deleted = result_value(
        daemon
            .request(DaemonRequest::ChannelDelete(
                turin::daemon::protocol::EntityIdParams {
                    id: "fs-events".to_string(),
                },
            ))
            .await?,
    );

    let deadline = Instant::now() + Duration::from_secs(5);
    let mut saw_removed = false;
    while Instant::now() < deadline {
        let event = timeout(Duration::from_millis(750), subscription.next_event()).await;
        let Ok(Ok(event)) = event else {
            continue;
        };
        if event.event == "channel.runtime.removed" && event.data["id"] == "fs-events" {
            saw_removed = true;
            break;
        }
    }
    assert!(
        saw_removed,
        "expected channel.runtime.removed for fs-events"
    );

    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_channel_registry_round_trip_over_endpoint() -> Result<()> {
    let daemon = DaemonHarness::start().await?;
    let channels_dir = daemon
        .tempdir
        .path()
        .join("workspace/.turin/runtime/channels/discord");
    std::fs::create_dir_all(&channels_dir)?;
    std::fs::write(
        channels_dir.join("config.toml"),
        r#"
kind = "discord"
agent_id = "default"
enabled = true
idle_ttl_secs = 600
token_env = "DISCORD_TOKEN"
channel_id = "1234567890"
"#,
    )?;

    let _ = daemon
        .request(DaemonRequest::RuntimeRescan(
            turin::daemon::protocol::NoParams::default(),
        ))
        .await?;

    let listed = result_value(
        daemon
            .request(DaemonRequest::ChannelList(
                turin::daemon::protocol::NoParams::default(),
            ))
            .await?,
    );
    let channels = listed["channels"]
        .as_array()
        .context("channel list should be an array")?;
    assert!(channels.iter().any(|channel| channel["id"] == "discord"));

    let detail = result_value(
        daemon
            .request(DaemonRequest::ChannelGet(
                turin::daemon::protocol::EntityIdParams {
                    id: "discord".to_string(),
                },
            ))
            .await?,
    );
    assert_eq!(detail["kind"], "discord");
    assert_eq!(detail["agent_id"], "default");
    assert_eq!(detail["settings"]["token_env"], "DISCORD_TOKEN");

    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_channel_management_round_trip_over_endpoint() -> Result<()> {
    let daemon = DaemonHarness::start().await?;

    let created = result_value(
        daemon
            .request(DaemonRequest::ChannelCreate(
                turin::daemon::protocol::CreateChannelParams {
                    id: "discord".to_string(),
                    kind: "discord".to_string(),
                    agent_id: "default".to_string(),
                    idle_ttl_secs: Some(600),
                    enabled: true,
                    settings: Some(serde_json::json!({
                        "token_env": "DISCORD_TOKEN",
                        "channel_id": "1234567890",
                        "allow_dm": true,
                    })),
                },
            ))
            .await?,
    );
    assert_eq!(created["id"], "discord");
    assert_eq!(created["settings"]["token_env"], "DISCORD_TOKEN");

    let updated = result_value(
        daemon
            .request(DaemonRequest::ChannelUpdate(
                turin::daemon::protocol::UpdateChannelParams {
                    id: "discord".to_string(),
                    kind: None,
                    agent_id: None,
                    idle_ttl_secs: Some(900),
                    settings: Some(serde_json::json!({
                        "token_env": "NEW_DISCORD_TOKEN",
                        "guild_id": "12345",
                    })),
                },
            ))
            .await?,
    );
    assert_eq!(updated["idle_ttl_secs"], 900);
    assert_eq!(updated["settings"]["token_env"], "NEW_DISCORD_TOKEN");
    assert_eq!(updated["settings"]["guild_id"], "12345");
    assert_eq!(updated["settings"]["channel_id"], "1234567890");

    let disabled = result_value(
        daemon
            .request(DaemonRequest::ChannelDisable(
                turin::daemon::protocol::EntityIdParams {
                    id: "discord".to_string(),
                },
            ))
            .await?,
    );
    assert_eq!(disabled["enabled"], false);

    let enabled = result_value(
        daemon
            .request(DaemonRequest::ChannelEnable(
                turin::daemon::protocol::EntityIdParams {
                    id: "discord".to_string(),
                },
            ))
            .await?,
    );
    assert_eq!(enabled["enabled"], true);

    let delete_result = result_value(
        daemon
            .request(DaemonRequest::ChannelDelete(
                turin::daemon::protocol::EntityIdParams {
                    id: "discord".to_string(),
                },
            ))
            .await?,
    );
    let channels = delete_result["registry"]["channels"]
        .as_array()
        .context("registry channels array")?;
    assert!(channels.iter().all(|channel| channel["id"] != "discord"));
    assert!(
        !daemon
            .tempdir
            .path()
            .join("workspace/.turin/runtime/channels/discord")
            .exists()
    );

    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_channel_create_rejects_invalid_known_kind_settings() -> Result<()> {
    let daemon = DaemonHarness::start().await?;

    let response = daemon
        .request(DaemonRequest::ChannelCreate(
            turin::daemon::protocol::CreateChannelParams {
                id: "discord-bad".to_string(),
                kind: "discord".to_string(),
                agent_id: "default".to_string(),
                idle_ttl_secs: Some(600),
                enabled: true,
                settings: Some(serde_json::json!({
                    "token_env": "DISCORD_TOKEN_ONLY"
                })),
            },
        ))
        .await?;

    assert!(!response.ok, "invalid settings should be rejected");
    let error_message = response
        .error
        .as_ref()
        .map(|error| error.message.clone())
        .unwrap_or_default();
    assert!(
        error_message.contains("channel_id"),
        "unexpected validation error: {}",
        error_message
    );

    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_fs_channel_runtime_processes_inbox_and_reports_runtime_status() -> Result<()> {
    let daemon = DaemonHarness::start().await?;

    let created = result_value(
        daemon
            .request(DaemonRequest::ChannelCreate(
                turin::daemon::protocol::CreateChannelParams {
                    id: "fs-local".to_string(),
                    kind: "fs".to_string(),
                    agent_id: "default".to_string(),
                    idle_ttl_secs: Some(600),
                    enabled: true,
                    settings: Some(serde_json::json!({
                        "inbox_dir": "inbox",
                        "outbox_dir": "outbox",
                        "processed_dir": "processed",
                        "failed_dir": "failed",
                        "poll_interval_ms": 25,
                    })),
                },
            ))
            .await?,
    );
    assert_eq!(created["id"], "fs-local");

    let runtime = wait_for_channel_state(&daemon, "fs-local", "running", 10).await?;
    assert_eq!(runtime["kind"], "fs");
    assert_eq!(runtime["agent_id"], "default");
    assert!(
        runtime["start_count"].as_u64().unwrap_or_default() >= 1,
        "fs runtime should record at least one start"
    );

    let daemon_status = result_value(
        daemon
            .request(DaemonRequest::DaemonStatus(
                turin::daemon::protocol::NoParams::default(),
            ))
            .await?,
    );
    assert!(
        daemon_status["endpoint"]
            .as_str()
            .is_some_and(|endpoint| !endpoint.is_empty()),
        "daemon.status should expose a non-empty endpoint"
    );
    let runtime_list = daemon_status["channel_runtimes"]
        .as_array()
        .context("daemon.status channel_runtimes should be an array")?;
    assert!(
        runtime_list
            .iter()
            .any(|entry| entry["id"] == "fs-local" && entry["state"] == "running")
    );

    let channel_dir = daemon
        .tempdir
        .path()
        .join("workspace/.turin/runtime/channels/fs-local");
    tokio::fs::create_dir_all(channel_dir.join("inbox")).await?;
    let inbound = serde_json::json!({
        "conversation": {
            "channel": "fs",
            "workspace_id": "workspace",
            "room_id": "room",
            "thread_id": "thread-1",
            "user_id": "user-1"
        },
        "message_id": "m-1",
        "user": {
            "id": "user-1",
            "display_name": "User One",
            "username": "user1"
        },
        "text": "Say pong"
    });
    tokio::fs::write(
        channel_dir.join("inbox/in-1.json"),
        serde_json::to_string_pretty(&inbound)?,
    )
    .await?;

    let deadline = Instant::now() + Duration::from_secs(10);
    let mut outbound_body = None;
    while Instant::now() < deadline {
        let outbox_dir = channel_dir.join("outbox");
        if outbox_dir.exists() {
            let mut entries = tokio::fs::read_dir(&outbox_dir).await?;
            if let Some(entry) = entries.next_entry().await? {
                outbound_body = Some(tokio::fs::read_to_string(entry.path()).await?);
                break;
            }
        }
        sleep(Duration::from_millis(25)).await;
    }

    let outbound = outbound_body.context("fs-local channel did not produce outbound response")?;
    assert!(outbound.contains("PONG"), "outbound payload: {}", outbound);
    assert!(channel_dir.join("processed/in-1.json").exists());

    let disabled = result_value(
        daemon
            .request(DaemonRequest::ChannelDisable(
                turin::daemon::protocol::EntityIdParams {
                    id: "fs-local".to_string(),
                },
            ))
            .await?,
    );
    assert_eq!(disabled["enabled"], false);

    let status_after_disable = daemon
        .request(DaemonRequest::ChannelStatus(
            turin::daemon::protocol::EntityIdParams {
                id: "fs-local".to_string(),
            },
        ))
        .await?;
    assert!(
        !status_after_disable.ok,
        "disabled channel should not be running"
    );
    assert!(matches!(
        status_after_disable.error.as_ref().map(|error| &error.code),
        Some(turin::daemon::protocol::ErrorCode::ChannelNotFound)
    ));

    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_session_open_uses_channel_owned_state_path() -> Result<()> {
    let tempdir = std::sync::Arc::new(tempfile::tempdir()?);
    let workspace_root = tempdir.path().join("workspace");
    let channel_dir = workspace_root.join(".turin/runtime/channels/fs-isolated");
    std::fs::create_dir_all(&channel_dir)?;
    std::fs::write(
        channel_dir.join("config.toml"),
        r#"kind = "fs"
agent_id = "default"

[persistence.state]
path = ".turin/runtime/channels/fs-isolated/state.db"

inbox_dir = "inbox"
outbox_dir = "outbox"
processed_dir = "processed"
failed_dir = "failed"
poll_interval_ms = 25
"#,
    )?;

    let daemon = DaemonHarness::start_in(tempdir.clone()).await?;

    let opened = result_value(
        daemon
            .request(DaemonRequest::SessionOpen(
                turin::daemon::protocol::OpenSessionParams {
                    agent_id: "default".to_string(),
                    slot_id: Some("isolated-slot".to_string()),
                    channel_id: Some("fs-isolated".to_string()),
                },
            ))
            .await?,
    );
    let session_id = opened["session_id"]
        .as_str()
        .context("session.open should return session_id")?
        .to_string();
    assert!(
        session_id.contains("@.turin/runtime/channels/fs-isolated/state.db"),
        "session id should be qualified with the channel-owned store path: {session_id}"
    );

    let session_ref = parse_session_reference(&session_id)?;
    let public_id = uuid::Uuid::parse_str(&session_ref.public_id)?;

    let default_store = StateStore::open(&workspace_root.join("test.db").to_string_lossy()).await?;
    assert!(
        default_store
            .get_session_row_by_public_id(public_id)
            .await?
            .is_none(),
        "channel-owned session should not be persisted in the default state db"
    );

    let channel_store = StateStore::open(
        &workspace_root
            .join(".turin/runtime/channels/fs-isolated/state.db")
            .to_string_lossy(),
    )
    .await?;
    assert!(
        channel_store
            .get_session_row_by_public_id(public_id)
            .await?
            .is_some(),
        "channel-owned session should be persisted in the channel-local state db"
    );

    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_discord_channel_reports_failed_runtime_when_token_is_missing() -> Result<()> {
    let daemon = DaemonHarness::start().await?;

    let created = result_value(
        daemon
            .request(DaemonRequest::ChannelCreate(
                turin::daemon::protocol::CreateChannelParams {
                    id: "discord-local".to_string(),
                    kind: "discord".to_string(),
                    agent_id: "default".to_string(),
                    idle_ttl_secs: Some(600),
                    enabled: true,
                    settings: Some(serde_json::json!({
                        "token_env": "DISCORD_TOKEN_MISSING_FOR_TEST",
                        "channel_id": "1234567890",
                        "poll_interval_ms": 250,
                    })),
                },
            ))
            .await?,
    );
    assert_eq!(created["id"], "discord-local");

    let failed = wait_for_channel_state(&daemon, "discord-local", "failed", 10).await?;
    let error = failed["last_error"]
        .as_str()
        .context("discord-local failed state should include last_error")?;
    let error_code = failed["last_error_code"]
        .as_str()
        .context("discord-local failed state should include last_error_code")?;
    assert!(
        error_code.contains("discord_auth_missing_token")
            || error_code.contains("auth_missing_token"),
        "unexpected discord runtime error code: {}",
        error_code
    );
    assert!(
        error.contains("DISCORD_TOKEN_MISSING_FOR_TEST"),
        "unexpected discord runtime error: {}",
        error
    );

    let daemon_status = result_value(
        daemon
            .request(DaemonRequest::DaemonStatus(
                turin::daemon::protocol::NoParams::default(),
            ))
            .await?,
    );
    assert!(
        daemon_status["endpoint"]
            .as_str()
            .is_some_and(|endpoint| !endpoint.is_empty()),
        "daemon.status should expose a non-empty endpoint"
    );
    let runtime_list = daemon_status["channel_runtimes"]
        .as_array()
        .context("daemon.status channel_runtimes should be an array")?;
    assert!(
        runtime_list
            .iter()
            .any(|entry| entry["id"] == "discord-local" && entry["state"] == "failed")
    );

    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_telegram_channel_management_round_trip_over_endpoint() -> Result<()> {
    let daemon = DaemonHarness::start().await?;

    let created = result_value(
        daemon
            .request(DaemonRequest::ChannelCreate(
                turin::daemon::protocol::CreateChannelParams {
                    id: "telegram-local".to_string(),
                    kind: "telegram".to_string(),
                    agent_id: "default".to_string(),
                    idle_ttl_secs: Some(600),
                    enabled: false,
                    settings: Some(serde_json::json!({
                        "token_env": "TELEGRAM_BOT_TOKEN",
                        "chat_id": -100123456,
                        "poll_timeout_secs": 10,
                    })),
                },
            ))
            .await?,
    );
    assert_eq!(created["id"], "telegram-local");
    assert_eq!(created["kind"], "telegram");
    assert_eq!(created["settings"]["chat_id"], -100123456);

    let updated = result_value(
        daemon
            .request(DaemonRequest::ChannelUpdate(
                turin::daemon::protocol::UpdateChannelParams {
                    id: "telegram-local".to_string(),
                    kind: None,
                    agent_id: None,
                    idle_ttl_secs: Some(900),
                    settings: Some(serde_json::json!({
                        "workspace_id": "ops",
                        "poll_interval_ms": 250,
                    })),
                },
            ))
            .await?,
    );
    assert_eq!(updated["idle_ttl_secs"], 900);
    assert_eq!(updated["settings"]["workspace_id"], "ops");
    assert_eq!(updated["settings"]["chat_id"], -100123456);

    let enabled = result_value(
        daemon
            .request(DaemonRequest::ChannelEnable(
                turin::daemon::protocol::EntityIdParams {
                    id: "telegram-local".to_string(),
                },
            ))
            .await?,
    );
    assert_eq!(enabled["enabled"], true);

    let disabled = result_value(
        daemon
            .request(DaemonRequest::ChannelDisable(
                turin::daemon::protocol::EntityIdParams {
                    id: "telegram-local".to_string(),
                },
            ))
            .await?,
    );
    assert_eq!(disabled["enabled"], false);

    let delete_result = result_value(
        daemon
            .request(DaemonRequest::ChannelDelete(
                turin::daemon::protocol::EntityIdParams {
                    id: "telegram-local".to_string(),
                },
            ))
            .await?,
    );
    let channels = delete_result["registry"]["channels"]
        .as_array()
        .context("registry channels array")?;
    assert!(
        channels
            .iter()
            .all(|channel| channel["id"] != "telegram-local")
    );

    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_channel_create_rejects_invalid_telegram_settings() -> Result<()> {
    let daemon = DaemonHarness::start().await?;

    let response = daemon
        .request(DaemonRequest::ChannelCreate(
            turin::daemon::protocol::CreateChannelParams {
                id: "telegram-bad".to_string(),
                kind: "telegram".to_string(),
                agent_id: "default".to_string(),
                idle_ttl_secs: Some(600),
                enabled: true,
                settings: Some(serde_json::json!({
                    "token_env": "TELEGRAM_BOT_TOKEN",
                    "chat_id": "@ops"
                })),
            },
        ))
        .await?;

    assert!(!response.ok, "invalid settings should be rejected");
    let error_message = response
        .error
        .as_ref()
        .map(|error| error.message.clone())
        .unwrap_or_default();
    assert!(
        error_message.contains("chat_id"),
        "unexpected validation error: {}",
        error_message
    );

    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_channel_create_accepts_multi_chat_telegram_settings() -> Result<()> {
    let daemon = DaemonHarness::start().await?;

    let created = result_value(
        daemon
            .request(DaemonRequest::ChannelCreate(
                turin::daemon::protocol::CreateChannelParams {
                    id: "telegram-multi".to_string(),
                    kind: "telegram".to_string(),
                    agent_id: "default".to_string(),
                    idle_ttl_secs: Some(600),
                    enabled: false,
                    settings: Some(serde_json::json!({
                        "token_env": "TELEGRAM_BOT_TOKEN",
                        "chat_ids": [-100123456, -100654321],
                        "respond_mode": "mentions_or_replies",
                    })),
                },
            ))
            .await?,
    );
    assert_eq!(created["id"], "telegram-multi");
    assert_eq!(
        created["settings"]["chat_ids"],
        serde_json::json!([-100123456, -100654321])
    );
    assert_eq!(created["settings"]["respond_mode"], "mentions_or_replies");

    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_channel_access_commands_manage_pairing_state() -> Result<()> {
    let daemon = DaemonHarness::start().await?;

    let created = result_value(
        daemon
            .request(DaemonRequest::ChannelCreate(
                turin::daemon::protocol::CreateChannelParams {
                    id: "telegram-pairing".to_string(),
                    kind: "telegram".to_string(),
                    agent_id: "default".to_string(),
                    idle_ttl_secs: Some(600),
                    enabled: false,
                    settings: Some(serde_json::json!({
                        "token_env": "TELEGRAM_BOT_TOKEN",
                        "pairing_mode": "pending",
                        "respond_mode": "mentions_or_replies",
                    })),
                },
            ))
            .await?,
    );
    assert_eq!(created["id"], "telegram-pairing");

    let empty = result_value(
        daemon
            .request(DaemonRequest::ChannelAccessGet(
                turin::daemon::protocol::ChannelAccessParams {
                    id: "telegram-pairing".to_string(),
                },
            ))
            .await?,
    );
    assert_eq!(empty["approved_rooms"], serde_json::json!([]));
    assert_eq!(empty["pending_rooms"], serde_json::json!([]));

    let approved = result_value(
        daemon
            .request(DaemonRequest::ChannelAccessApprove(
                turin::daemon::protocol::ChannelAccessRoomParams {
                    id: "telegram-pairing".to_string(),
                    workspace_id: "telegram".to_string(),
                    room_id: Some("-100123456".to_string()),
                    thread_id: "-100123456".to_string(),
                },
            ))
            .await?,
    );
    assert_eq!(
        approved["approved_rooms"].as_array().map(|v| v.len()),
        Some(1)
    );

    let revoked = result_value(
        daemon
            .request(DaemonRequest::ChannelAccessRevoke(
                turin::daemon::protocol::ChannelAccessRoomParams {
                    id: "telegram-pairing".to_string(),
                    workspace_id: "telegram".to_string(),
                    room_id: Some("-100123456".to_string()),
                    thread_id: "-100123456".to_string(),
                },
            ))
            .await?,
    );
    assert_eq!(revoked["approved_rooms"], serde_json::json!([]));

    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn daemon_telegram_channel_reports_failed_runtime_when_token_is_missing() -> Result<()> {
    let daemon = DaemonHarness::start().await?;

    let created = result_value(
        daemon
            .request(DaemonRequest::ChannelCreate(
                turin::daemon::protocol::CreateChannelParams {
                    id: "telegram-runtime".to_string(),
                    kind: "telegram".to_string(),
                    agent_id: "default".to_string(),
                    idle_ttl_secs: Some(600),
                    enabled: true,
                    settings: Some(serde_json::json!({
                        "token_env": "TELEGRAM_TOKEN_MISSING_FOR_TEST",
                        "chat_id": -100123456,
                        "poll_timeout_secs": 0,
                        "poll_interval_ms": 25,
                    })),
                },
            ))
            .await?,
    );
    assert_eq!(created["id"], "telegram-runtime");

    let failed = wait_for_channel_state(&daemon, "telegram-runtime", "failed", 10).await?;
    let error = failed["last_error"]
        .as_str()
        .context("telegram-runtime failed state should include last_error")?;
    let error_code = failed["last_error_code"]
        .as_str()
        .context("telegram-runtime failed state should include last_error_code")?;
    assert!(
        error_code.contains("telegram_auth_missing_token")
            || error_code.contains("auth_missing_token"),
        "unexpected telegram runtime error code: {}",
        error_code
    );
    assert!(
        error.contains("TELEGRAM_TOKEN_MISSING_FOR_TEST"),
        "unexpected telegram runtime error: {}",
        error
    );

    let daemon_status = result_value(
        daemon
            .request(DaemonRequest::DaemonStatus(
                turin::daemon::protocol::NoParams::default(),
            ))
            .await?,
    );
    let runtime_list = daemon_status["channel_runtimes"]
        .as_array()
        .context("daemon.status channel_runtimes should be an array")?;
    assert!(
        runtime_list
            .iter()
            .any(|entry| entry["id"] == "telegram-runtime" && entry["state"] == "failed")
    );

    daemon.stop().await
}
