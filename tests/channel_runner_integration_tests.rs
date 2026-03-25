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
use turin_channel_runner::{
    ChannelDriver, ChannelProgressUpdate, ChannelRunner, ChannelStreamMode, RunnerConfig,
};

struct DaemonHarness {
    _tempdir: Arc<TempDir>,
    workspace_root: PathBuf,
    endpoint: PathBuf,
    join: JoinHandle<Result<()>>,
}

struct MockDriver {
    events: VecDeque<InboundEvent>,
    sent: Arc<Mutex<Vec<OutboundMessage>>>,
    progress: Arc<Mutex<Vec<String>>>,
    shutdown_called: Arc<Mutex<bool>>,
    stream_mode: ChannelStreamMode,
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
            _tempdir: tempdir,
            workspace_root,
            endpoint,
            join,
        })
    }

    fn runner(&self) -> ChannelRunner {
        ChannelRunner::new(
            turin_daemon_client::DaemonClient::new(&self.endpoint),
            RunnerConfig {
                state_path: self.workspace_root.join(".turin/channel-bindings.json"),
                idle_ttl: Some(Duration::from_secs(600)),
            },
        )
    }

    async fn stop(self) -> Result<()> {
        let client = turin_daemon_client::DaemonClient::new(&self.endpoint);
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
            progress: Arc::new(Mutex::new(Vec::new())),
            shutdown_called: Arc::new(Mutex::new(false)),
            stream_mode: ChannelStreamMode::Off,
        }
    }

    fn with_stream_mode(events: Vec<InboundEvent>, stream_mode: ChannelStreamMode) -> Self {
        Self {
            stream_mode,
            ..Self::new(events)
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

    fn stream_mode(&self) -> ChannelStreamMode {
        self.stream_mode
    }

    async fn send_progress(
        &mut self,
        _event: &InboundEvent,
        update: ChannelProgressUpdate,
    ) -> Result<()> {
        let value = match update {
            ChannelProgressUpdate::Typing => "typing".to_string(),
            ChannelProgressUpdate::StreamingPreview { text, thinking } => match thinking {
                Some(thinking) => format!("preview:{text}|thinking:{thinking}"),
                None => format!("text:{text}"),
            },
        };
        self.progress
            .lock()
            .expect("progress lock poisoned")
            .push(value);
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

#[tokio::test(flavor = "multi_thread")]
async fn channel_runner_emits_progress_updates_for_opted_in_driver() -> Result<()> {
    let daemon = DaemonHarness::start().await?;
    let runner = daemon.runner();
    let mut driver = MockDriver::with_stream_mode(vec![sample_event()], ChannelStreamMode::Draft);
    let progress = Arc::clone(&driver.progress);

    runner
        .run_driver("default", &mut driver, Some(5_000))
        .await?;

    let progress = progress.lock().expect("progress lock poisoned");
    assert!(
        progress.iter().any(|entry| entry == "typing"),
        "progress log should contain typing updates: {progress:?}"
    );
    assert!(
        progress.iter().any(|entry| entry.starts_with("text:PONG")),
        "progress log should contain streamed assistant text: {progress:?}"
    );

    daemon.stop().await
}
