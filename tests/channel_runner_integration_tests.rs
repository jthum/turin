mod support;

use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use tempfile::TempDir;
use tokio::task::JoinHandle;
use tokio::time::{Instant, sleep, timeout};
use turin_channel_core::{
    ChannelConversationKey, ChannelKind, ChannelMessageRef, ChannelSessionScope, ChannelUser,
    InboundEvent, MessageBlock, OutboundMessage,
};
use turin_channel_runner::{
    ChannelAccessPolicy, ChannelDriver, ChannelProgressUpdate, ChannelRunner, ChannelStreamMode,
    RunnerConfig,
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

struct RecordingDriver {
    events: VecDeque<InboundEvent>,
    sent: Arc<Mutex<Vec<(String, String, Instant)>>>,
    shutdown_called: Arc<Mutex<bool>>,
}

impl DaemonHarness {
    async fn start() -> Result<Self> {
        Self::start_with_mock_response("PONG").await
    }

    async fn start_with_mock_response(mock_response: &str) -> Result<Self> {
        let tempdir = Arc::new(tempfile::tempdir()?);
        let workspace_root = tempdir.path().join("workspace");
        let config_path = support::write_mock_runtime_config(
            &workspace_root,
            "Channel runner test",
            mock_response,
        )?;
        std::fs::create_dir_all(support::channel_runtime_dir(&workspace_root, "mock"))?;
        let endpoint = support::workspace_daemon_socket(&workspace_root);

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
                channel_id: "mock".to_string(),
                state_path: support::channel_runtime_dir(&self.workspace_root, "mock")
                    .join("bindings.json"),
                access_state_path: support::channel_runtime_dir(&self.workspace_root, "mock")
                    .join("access.json"),
                idle_ttl: Some(Duration::from_secs(600)),
                access_policy: Default::default(),
                tools: Default::default(),
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

impl RecordingDriver {
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
        ChannelKind::new("mock")
    }

    fn user_matches_selector(&self, selector: &str, user: &ChannelUser) -> bool {
        let selector = selector.trim();
        if selector.is_empty() {
            return false;
        }
        user.id == selector
            || user
                .username
                .as_ref()
                .is_some_and(|username| username.eq_ignore_ascii_case(selector))
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

#[async_trait]
impl ChannelDriver for RecordingDriver {
    fn kind(&self) -> ChannelKind {
        ChannelKind::new("telegram")
    }

    fn user_matches_selector(&self, selector: &str, user: &ChannelUser) -> bool {
        let selector = selector.trim();
        if selector.is_empty() {
            return false;
        }
        user.id == selector
            || user
                .username
                .as_ref()
                .is_some_and(|username| username.eq_ignore_ascii_case(selector))
    }

    async fn next_event(&mut self) -> Result<Option<InboundEvent>> {
        Ok(self.events.pop_front())
    }

    async fn send(
        &mut self,
        _conversation: &ChannelConversationKey,
        message: OutboundMessage,
    ) -> Result<()> {
        let reply_to = message
            .metadata
            .get("telegram_reply_to_message_id")
            .and_then(|value| value.as_str())
            .unwrap_or_default()
            .to_string();
        let text = message
            .blocks
            .iter()
            .filter_map(|block| match block {
                MessageBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        self.sent
            .lock()
            .expect("sent lock poisoned")
            .push((reply_to, text, Instant::now()));
        Ok(())
    }

    fn enrich_outbound_for_event(
        &self,
        event: &InboundEvent,
        mut outbound: OutboundMessage,
    ) -> OutboundMessage {
        if let Some(message_id) = event.metadata.get("telegram_message_id") {
            outbound.metadata.insert(
                "telegram_reply_to_message_id".to_string(),
                message_id.clone(),
            );
        }
        outbound
    }

    async fn shutdown(&mut self) -> Result<()> {
        *self.shutdown_called.lock().expect("shutdown lock poisoned") = true;
        Ok(())
    }
}

fn sample_event() -> InboundEvent {
    let conversation = ChannelConversationKey {
        channel: ChannelKind::new("discord"),
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
        session_scope: ChannelSessionScope::User,
        text: "Say pong".into(),
        attachments: vec![],
        metadata: Default::default(),
    }
}

fn sample_telegram_event(thread_id: &str, user_id: &str, message_id: &str) -> InboundEvent {
    let conversation = ChannelConversationKey {
        channel: ChannelKind::new("telegram"),
        workspace_id: "telegram".into(),
        room_id: Some("group-1".into()),
        thread_id: thread_id.into(),
        user_id: Some(user_id.into()),
    };
    let mut metadata = serde_json::Map::new();
    metadata.insert(
        "telegram_message_id".into(),
        serde_json::Value::String(message_id.into()),
    );
    InboundEvent {
        message: ChannelMessageRef {
            conversation: conversation.clone(),
            message_id: message_id.into(),
        },
        conversation,
        user: ChannelUser {
            id: user_id.into(),
            display_name: Some(format!("User {user_id}")),
            username: Some(format!("user_{user_id}")),
        },
        session_scope: ChannelSessionScope::User,
        text: "Say pong".into(),
        attachments: vec![],
        metadata,
    }
}

fn sample_scoped_event(
    thread_id: &str,
    user_id: &str,
    message_id: &str,
    session_scope: ChannelSessionScope,
) -> InboundEvent {
    let conversation = ChannelConversationKey {
        channel: ChannelKind::new("telegram"),
        workspace_id: "telegram".into(),
        room_id: Some("group-1".into()),
        thread_id: thread_id.into(),
        user_id: matches!(session_scope, ChannelSessionScope::User).then(|| user_id.into()),
    };
    InboundEvent {
        message: ChannelMessageRef {
            conversation: conversation.clone(),
            message_id: message_id.into(),
        },
        conversation,
        user: ChannelUser {
            id: user_id.into(),
            display_name: Some(format!("User {user_id}")),
            username: Some(format!("user_{user_id}")),
        },
        session_scope,
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
async fn channel_runner_session_scope_controls_conversation_sharing() -> Result<()> {
    let daemon = DaemonHarness::start().await?;
    let runner = daemon.runner();

    let user_a = sample_scoped_event("thread-a", "user-a", "msg-a", ChannelSessionScope::User);
    let user_b = sample_scoped_event("thread-a", "user-b", "msg-b", ChannelSessionScope::User);
    let user_a_binding = runner
        .ensure_session("default", &user_a.conversation, false)
        .await?;
    let user_b_binding = runner
        .ensure_session("default", &user_b.conversation, false)
        .await?;
    assert_ne!(
        user_a_binding.session_id, user_b_binding.session_id,
        "user-scoped channel conversations should isolate users in the same thread"
    );

    let shared_a = sample_scoped_event("thread-b", "user-a", "msg-c", ChannelSessionScope::Thread);
    let shared_b = sample_scoped_event("thread-b", "user-b", "msg-d", ChannelSessionScope::Thread);
    let shared_a_binding = runner
        .ensure_session("default", &shared_a.conversation, false)
        .await?;
    let shared_b_binding = runner
        .ensure_session("default", &shared_b.conversation, false)
        .await?;
    assert_eq!(
        shared_a_binding.session_id, shared_b_binding.session_id,
        "thread-scoped channel conversations should reuse one session across users"
    );

    daemon.stop().await
}

#[tokio::test]
async fn channel_runner_ignores_disallowed_user_without_opening_session() -> Result<()> {
    let tempdir = tempfile::tempdir()?;
    let runner = ChannelRunner::new(
        turin_daemon_client::DaemonClient::new(tempdir.path().join("missing.sock")),
        RunnerConfig {
            channel_id: "mock".to_string(),
            state_path: tempdir.path().join("bindings.json"),
            access_state_path: tempdir.path().join("access.json"),
            idle_ttl: Some(Duration::from_secs(600)),
            access_policy: ChannelAccessPolicy {
                allowed_users: HashSet::from(["user-a".to_string()]),
                ..Default::default()
            },
            tools: Default::default(),
        },
    );
    let mut driver = MockDriver::new(vec![sample_scoped_event(
        "thread-a",
        "user-b",
        "msg-b",
        ChannelSessionScope::User,
    )]);
    let sent = Arc::clone(&driver.sent);
    let shutdown_called = Arc::clone(&driver.shutdown_called);

    runner.run_driver("default", &mut driver, Some(100)).await?;

    assert!(
        sent.lock().expect("sent lock poisoned").is_empty(),
        "disallowed users should not receive task output or error output"
    );
    assert!(*shutdown_called.lock().expect("shutdown lock poisoned"));
    assert!(
        !tempdir.path().join("bindings.json").exists(),
        "ignored channel events should not create session bindings"
    );

    Ok(())
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

    {
        let progress = progress.lock().expect("progress lock poisoned");
        assert!(
            progress.iter().any(|entry| entry == "typing"),
            "progress log should contain typing updates: {progress:?}"
        );
        assert!(
            progress.iter().any(|entry| entry.starts_with("text:PONG")),
            "progress log should contain streamed assistant text: {progress:?}"
        );
    }

    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn channel_runner_processes_different_conversations_in_parallel() -> Result<()> {
    let daemon = DaemonHarness::start_with_mock_response("delay_ms=600;PONG").await?;
    let runner = daemon.runner();
    let event_a = sample_telegram_event("thread-a", "user-a", "msg-a");
    let event_b = sample_telegram_event("thread-b", "user-b", "msg-b");
    let mut driver = RecordingDriver::new(vec![event_a, event_b]);
    let sent = Arc::clone(&driver.sent);
    let started_at = Instant::now();

    runner
        .run_driver("default", &mut driver, Some(5_000))
        .await?;

    let elapsed = started_at.elapsed();
    let send_gap = {
        let sent = sent.lock().expect("sent lock poisoned");
        assert_eq!(sent.len(), 2);
        assert!(
            sent.iter().all(|(_, text, _)| text.contains("PONG")),
            "parallel conversations should both succeed, sent={sent:?}"
        );
        sent[0]
            .2
            .checked_duration_since(sent[1].2)
            .unwrap_or_else(|| sent[1].2.duration_since(sent[0].2))
    };
    {
        let sent = sent.lock().expect("sent lock poisoned");
        assert_eq!(sent.len(), 2);
    }
    assert!(
        send_gap < Duration::from_millis(450),
        "different conversations should overlap, elapsed={elapsed:?}, send_gap={send_gap:?}"
    );

    daemon.stop().await
}

#[tokio::test(flavor = "multi_thread")]
async fn channel_runner_serializes_same_conversation_events() -> Result<()> {
    let daemon = DaemonHarness::start_with_mock_response("delay_ms=600;PONG").await?;
    let runner = daemon.runner();
    let event_a = sample_telegram_event("thread-a", "user-a", "msg-a");
    let event_b = sample_telegram_event("thread-a", "user-a", "msg-b");
    let mut driver = RecordingDriver::new(vec![event_a, event_b]);
    let sent = Arc::clone(&driver.sent);
    let started_at = Instant::now();

    runner
        .run_driver("default", &mut driver, Some(5_000))
        .await?;

    let elapsed = started_at.elapsed();
    let send_gap = {
        let sent = sent.lock().expect("sent lock poisoned");
        assert_eq!(sent.len(), 2);
        assert_eq!(sent[0].0, "msg-a");
        assert_eq!(sent[1].0, "msg-b");
        assert!(
            sent.iter().all(|(_, text, _)| text.contains("PONG")),
            "serialized conversation should still succeed, sent={sent:?}"
        );
        sent[1].2.duration_since(sent[0].2)
    };
    assert!(
        send_gap >= Duration::from_millis(500),
        "same conversation should stay serialized, elapsed={elapsed:?}, send_gap={send_gap:?}"
    );

    daemon.stop().await
}
