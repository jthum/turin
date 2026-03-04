use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use tempfile::TempDir;
use tokio::task::JoinHandle;
use tokio::time::{Instant, sleep, timeout};
use turin_channel_core::{
    ChannelConversationKey, ChannelKind, ChannelMessageRef, ChannelUser, InboundEvent,
    OutboundMessage,
};
use turin_channel_runner::{ChannelDriver, ChannelRunner, RunnerConfig};

struct DaemonHarness {
    _tempdir: Arc<TempDir>,
    workspace_root: PathBuf,
    socket_path: PathBuf,
    join: JoinHandle<Result<()>>,
}

struct MockDriver {
    events: VecDeque<InboundEvent>,
    sent: Arc<Mutex<Vec<OutboundMessage>>>,
    shutdown_called: Arc<Mutex<bool>>,
}

impl DaemonHarness {
    async fn start() -> Result<Self> {
        let tempdir = Arc::new(tempfile::tempdir()?);
        let workspace_root = tempdir.path().join("workspace");
        let harness_dir = workspace_root.join(".turin/harnesses");

        std::fs::create_dir_all(&harness_dir)?;
        std::fs::write(
            harness_dir.join("main.lua"),
            "-- channel runner integration harness\n",
        )?;

        let config_path = tempdir.path().join("turin.toml");
        let config_toml = format!(
            r#"[agent]
id = "default"
model = "mock-model"
provider = "mock"
system_prompt = "Channel runner test"

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
            workspace_root,
            socket_path,
            join,
        })
    }

    fn runner(&self) -> ChannelRunner {
        ChannelRunner::new(
            turin_daemon_client::DaemonClient::new(&self.socket_path),
            RunnerConfig {
                state_path: self.workspace_root.join(".turin/channel-bindings.json"),
                idle_ttl: Some(Duration::from_secs(600)),
            },
        )
    }

    async fn stop(self) -> Result<()> {
        let client = turin_daemon_client::DaemonClient::new(&self.socket_path);
        let _: serde_json::Value = client
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

impl MockDriver {
    fn new(events: Vec<InboundEvent>) -> Self {
        Self {
            events: events.into(),
            sent: Arc::new(Mutex::new(Vec::new())),
            shutdown_called: Arc::new(Mutex::new(false)),
        }
    }
}

#[async_trait]
impl ChannelDriver for MockDriver {
    fn kind(&self) -> ChannelKind {
        ChannelKind::Other("mock".into())
    }

    async fn next_event(&mut self) -> Result<Option<InboundEvent>> {
        Ok(self.events.pop_front())
    }

    async fn send(
        &mut self,
        _conversation: &ChannelConversationKey,
        message: OutboundMessage,
    ) -> Result<()> {
        self.sent.lock().expect("sent lock poisoned").push(message);
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<()> {
        *self.shutdown_called.lock().expect("shutdown lock poisoned") = true;
        Ok(())
    }
}

fn sample_event() -> InboundEvent {
    let conversation = ChannelConversationKey {
        channel: ChannelKind::Discord,
        workspace_id: "guild".into(),
        room_id: Some("room".into()),
        thread_id: "thread-1".into(),
        user_id: Some("user-1".into()),
    };
    InboundEvent {
        message: ChannelMessageRef {
            conversation: conversation.clone(),
            message_id: "m-1".into(),
        },
        conversation,
        user: ChannelUser {
            id: "user-1".into(),
            display_name: Some("User One".into()),
            username: Some("user1".into()),
        },
        text: "Say pong".into(),
        attachments: vec![],
        metadata: Default::default(),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn channel_runner_drives_daemon_and_emits_outbound_messages() -> Result<()> {
    let daemon = DaemonHarness::start().await?;
    let runner = daemon.runner();
    let mut driver = MockDriver::new(vec![sample_event()]);
    let sent = Arc::clone(&driver.sent);
    let shutdown_called = Arc::clone(&driver.shutdown_called);

    runner
        .run_driver("default", &mut driver, Some(5_000))
        .await?;

    let rendered = {
        let messages = sent.lock().expect("sent lock poisoned");
        assert_eq!(messages.len(), 1);
        serde_json::to_string(&messages[0])?
    };
    assert!(rendered.contains("PONG"));
    assert!(*shutdown_called.lock().expect("shutdown lock poisoned"));

    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn channel_runner_reset_requests_start_fresh_session() -> Result<()> {
    let daemon = DaemonHarness::start().await?;
    let runner = daemon.runner();
    let event = sample_event();

    let initial = runner
        .ensure_session("default", &event.conversation, false)
        .await?;
    let reused = runner
        .ensure_session("default", &event.conversation, false)
        .await?;
    let reset = runner
        .ensure_session("default", &event.conversation, true)
        .await?;

    assert_eq!(initial.session_id, reused.session_id);
    assert_ne!(initial.session_id, reset.session_id);

    daemon.stop().await
}
