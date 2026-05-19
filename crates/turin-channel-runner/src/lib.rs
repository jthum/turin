use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{Mutex, mpsc};
use tokio::time::{Instant, MissedTickBehavior};
use turin_channel_core::{
    ChannelAdapterManifest, ChannelCapabilities, ChannelConversationKey, ChannelKind, ChannelUser,
    ConversationBinding, InboundEvent, OutboundMessage, RoutingDecision, decide_routing,
};
use turin_daemon_client::DaemonClient;
use turin_daemon_protocol::{
    ChannelRunnerHeartbeatParams, ChannelRunnerHelloParams, OpenSessionParams, ResumeSessionParams,
    RuntimeEventsSubscribeParams, SubmitTaskParams, WaitTaskParams,
};
use turin_types::ToolsConfig;

mod access;
mod bindings;
mod stream;
mod task_payloads;

pub use access::{
    ApprovedRoomView, ChannelAccessPolicy, ChannelAccessSnapshot, ChannelRoomRef,
    FileAccessStateStore, PairingMode, PendingRoomView,
};
pub use bindings::FileBindingStore;
pub use stream::{ChannelProgressUpdate, ChannelStreamMode};
pub use task_payloads::TaskSnapshot;

#[cfg(test)]
pub(crate) use access::AccessStateFile;
pub(crate) use access::{
    ApprovedRoom, ChannelRoomKey, PendingRoom, serialize_room_key, unix_seconds,
};
pub(crate) use bindings::serialize_binding_key;
pub(crate) use stream::{
    WorkerStreamConfig, attach_final_thinking, preview_char_count, preview_thinking,
    should_flush_preview, should_subscribe_to_session_events,
};
pub(crate) use task_payloads::{
    task_input_content_from_event, task_prompt_for_submission, task_to_outbound,
};

const RUNNER_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15);

#[derive(Debug, Clone)]
pub struct RunnerPresence {
    pub manifest: ChannelAdapterManifest,
    pub runner_binary: Option<String>,
    pub runner_version: Option<String>,
    pub pid: Option<u32>,
}

pub async fn announce_runner_presence(
    daemon: &DaemonClient,
    channel_id: &str,
    presence: RunnerPresence,
) -> Result<()> {
    daemon
        .channel_runner_hello(ChannelRunnerHelloParams {
            channel_id: channel_id.to_string(),
            manifest: presence.manifest,
            runner_binary: presence.runner_binary,
            runner_version: presence.runner_version,
            pid: presence.pid,
        })
        .await
}

pub fn spawn_runner_heartbeat(
    daemon: DaemonClient,
    channel_id: String,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(RUNNER_HEARTBEAT_INTERVAL);
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                changed = shutdown_rx.changed() => {
                    if changed.is_err() || *shutdown_rx.borrow() {
                        break;
                    }
                }
                _ = interval.tick() => {
                    let _ = daemon.channel_runner_heartbeat(ChannelRunnerHeartbeatParams {
                        channel_id: channel_id.clone(),
                    }).await;
                }
            }
        }
    })
}

#[derive(Debug, Clone)]
pub struct RunnerConfig {
    pub channel_id: String,
    pub state_path: PathBuf,
    pub access_state_path: PathBuf,
    pub idle_ttl: Option<Duration>,
    pub access_policy: ChannelAccessPolicy,
    pub tools: ToolsConfig,
}

pub fn task_timeout_ms_from_settings(settings: &Value) -> Result<Option<u64>> {
    let map = settings
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Channel settings must be a JSON object"))?;
    read_task_timeout_ms(map.get("task_timeout_ms"))
}

pub fn tools_config_from_settings(settings: &Value) -> Result<ToolsConfig> {
    let map = settings
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Channel settings must be a JSON object"))?;
    let Some(tools) = map.get("tools") else {
        return Ok(ToolsConfig::default());
    };
    serde_json::from_value(tools.clone()).context("failed to parse 'tools' settings")
}

enum EventAccessDecision {
    Allow,
    Pending { notify: bool },
    Ignore,
}

#[async_trait]
pub trait ChannelDriver {
    fn kind(&self) -> ChannelKind;

    fn user_matches_selector(&self, selector: &str, user: &ChannelUser) -> bool;

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities::default()
    }

    async fn next_event(&mut self) -> Result<Option<InboundEvent>>;

    async fn send(
        &mut self,
        conversation: &ChannelConversationKey,
        message: OutboundMessage,
    ) -> Result<()>;

    fn enrich_outbound_for_event(
        &self,
        _event: &InboundEvent,
        outbound: OutboundMessage,
    ) -> OutboundMessage {
        outbound
    }

    fn stream_mode(&self) -> ChannelStreamMode {
        ChannelStreamMode::Off
    }

    fn stream_thinking(&self) -> bool {
        false
    }

    fn persist_thinking(&self) -> bool {
        false
    }

    async fn send_progress(
        &mut self,
        _event: &InboundEvent,
        _update: ChannelProgressUpdate,
    ) -> Result<()> {
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<()>;
}

#[derive(Clone)]
pub struct ChannelRunner {
    daemon: DaemonClient,
    channel_id: String,
    bindings: FileBindingStore,
    bindings_lock: Arc<Mutex<()>>,
    access_state: FileAccessStateStore,
    idle_ttl: Option<Duration>,
    access_policy: ChannelAccessPolicy,
    tools: ToolsConfig,
}

#[derive(Debug, Clone)]
struct QueuedInboundEvent {
    conversation_id: String,
    event: InboundEvent,
    reset_requested: bool,
    stream: WorkerStreamConfig,
}

#[derive(Debug)]
enum WorkerAction {
    Progress {
        event: InboundEvent,
        update: ChannelProgressUpdate,
    },
    Completed {
        conversation_id: String,
        event: InboundEvent,
        outbound: OutboundMessage,
    },
}

struct DriverDispatchState {
    action_tx: mpsc::UnboundedSender<WorkerAction>,
    active_conversations: HashSet<String>,
    queued_events: HashMap<String, VecDeque<QueuedInboundEvent>>,
}

impl DriverDispatchState {
    fn new(action_tx: mpsc::UnboundedSender<WorkerAction>) -> Self {
        Self {
            action_tx,
            active_conversations: HashSet::new(),
            queued_events: HashMap::new(),
        }
    }
}

struct WorkerTaskContext<'a> {
    event: &'a InboundEvent,
    binding: &'a ConversationBinding,
    submitted: &'a TaskSnapshot,
}

impl ChannelRunner {
    pub fn new(daemon: DaemonClient, config: RunnerConfig) -> Self {
        Self {
            daemon,
            channel_id: config.channel_id,
            bindings: FileBindingStore::new(config.state_path),
            bindings_lock: Arc::new(Mutex::new(())),
            access_state: FileAccessStateStore::new(config.access_state_path),
            idle_ttl: config.idle_ttl,
            access_policy: config.access_policy,
            tools: config.tools,
        }
    }

    pub async fn ensure_session(
        &self,
        agent_id: &str,
        key: &ChannelConversationKey,
        reset_requested: bool,
    ) -> Result<ConversationBinding> {
        let binding_key = serialize_binding_key(key)?;
        let (current, decision) = {
            let _guard = self.bindings_lock.lock().await;
            let bindings = self.bindings.load().await?;
            let current = bindings.get(&binding_key).cloned();
            let decision = decide_routing(
                key,
                current.as_ref(),
                SystemTime::now(),
                self.idle_ttl,
                reset_requested,
            );
            (current, decision)
        };

        let session: serde_json::Value = match decision {
            RoutingDecision::Reuse {
                slot_id,
                session_id,
            } => {
                self.daemon
                    .request_ok(
                        None,
                        turin_daemon_protocol::DaemonRequest::SessionResume(ResumeSessionParams {
                            session_id,
                            slot_id: Some(slot_id),
                        }),
                    )
                    .await?
            }
            RoutingDecision::StartFresh { slot_id } => {
                if let Some(binding) = current {
                    let _ = self
                        .daemon
                        .request_ok::<serde_json::Value>(
                            None,
                            turin_daemon_protocol::DaemonRequest::SessionKill(
                                turin_daemon_protocol::LiveSessionTargetParams {
                                    session_id: binding.session_id.clone(),
                                    slot_id: Some(binding.slot_id.clone()),
                                },
                            ),
                        )
                        .await;
                }
                self.daemon
                    .request_ok(
                        None,
                        turin_daemon_protocol::DaemonRequest::SessionOpen(OpenSessionParams {
                            agent_id: agent_id.to_string(),
                            slot_id: Some(slot_id),
                            channel_id: Some(self.channel_id.clone()),
                        }),
                    )
                    .await?
            }
        };

        let session_id = session
            .get("session_id")
            .and_then(|v| v.as_str())
            .context("daemon session response missing session_id")?;
        let mut binding = ConversationBinding::new(agent_id, session_id, key, SystemTime::now());
        binding.touch(SystemTime::now());
        {
            let _guard = self.bindings_lock.lock().await;
            let mut bindings = self.bindings.load().await?;
            bindings.insert(binding_key, binding.clone());
            self.bindings.save(&bindings).await?;
        }
        Ok(binding)
    }

    pub async fn submit(
        &self,
        agent_id: &str,
        event: &InboundEvent,
        reset_requested: bool,
    ) -> Result<TaskSnapshot> {
        let binding = self
            .ensure_session(agent_id, &event.conversation, reset_requested)
            .await?;
        self.submit_with_binding(&binding, event).await
    }

    pub async fn submit_with_binding(
        &self,
        binding: &ConversationBinding,
        event: &InboundEvent,
    ) -> Result<TaskSnapshot> {
        let content = task_input_content_from_event(event);
        self.daemon
            .request_ok(
                None,
                turin_daemon_protocol::DaemonRequest::TaskSubmit(SubmitTaskParams {
                    agent_id: None,
                    session_id: Some(binding.session_id.clone()),
                    slot_id: Some(binding.slot_id.clone()),
                    prompt: task_prompt_for_submission(event),
                    content: (!content.is_empty()).then_some(content),
                    tools: (!self.tools.is_empty()).then_some(self.tools.clone()),
                    conflict_policy: None,
                }),
            )
            .await
    }

    pub async fn submit_and_wait(
        &self,
        agent_id: &str,
        event: &InboundEvent,
        reset_requested: bool,
        timeout_ms: Option<u64>,
    ) -> Result<TaskSnapshot> {
        let submitted = self.submit(agent_id, event, reset_requested).await?;
        self.daemon
            .request_ok(
                None,
                turin_daemon_protocol::DaemonRequest::TaskWait(WaitTaskParams {
                    request_id: submitted.request_id,
                    timeout_ms,
                }),
            )
            .await
    }

    pub async fn handle_event(
        &self,
        agent_id: &str,
        event: &InboundEvent,
        reset_requested: bool,
        timeout_ms: Option<u64>,
    ) -> Result<OutboundMessage> {
        let task = self
            .submit_and_wait(agent_id, event, reset_requested, timeout_ms)
            .await?;
        Ok(task_to_outbound(&task))
    }

    async fn authorize_event<D: ChannelDriver + Send>(
        &self,
        driver: &D,
        event: &InboundEvent,
    ) -> Result<EventAccessDecision> {
        if matches!(self.access_policy.pairing_mode, PairingMode::Off) {
            return Ok(
                if self
                    .access_policy
                    .allows_interaction(&event.user, |selector, user| {
                        driver.user_matches_selector(selector, user)
                    })
                {
                    EventAccessDecision::Allow
                } else {
                    EventAccessDecision::Ignore
                },
            );
        }

        let room = ChannelRoomKey::from(&event.conversation);
        let room_key = serialize_room_key(&room)?;
        let mut state = self.access_state.load().await?;
        if state.approved_rooms.contains_key(&room_key) {
            return Ok(
                if self
                    .access_policy
                    .allows_interaction(&event.user, |selector, user| {
                        driver.user_matches_selector(selector, user)
                    })
                {
                    EventAccessDecision::Allow
                } else {
                    EventAccessDecision::Ignore
                },
            );
        }

        if self.access_policy.is_banned(&event.user, |selector, user| {
            driver.user_matches_selector(selector, user)
        }) {
            return Ok(EventAccessDecision::Ignore);
        }

        if !self
            .access_policy
            .allows_pairing(&event.user, |selector, user| {
                driver.user_matches_selector(selector, user)
            })
        {
            return Ok(EventAccessDecision::Ignore);
        }

        match self.access_policy.pairing_mode {
            PairingMode::Off => Ok(EventAccessDecision::Allow),
            PairingMode::Auto => {
                let now = SystemTime::now();
                state.approved_rooms.insert(
                    room_key.clone(),
                    ApprovedRoom {
                        room,
                        approved_at_unix_seconds: unix_seconds(now),
                        approved_by_user_id: Some(event.user.id.clone()),
                        approved_by_username: event.user.username.clone(),
                    },
                );
                state.pending_rooms.remove(&room_key);
                self.access_state.save(&state).await?;
                Ok(EventAccessDecision::Allow)
            }
            PairingMode::Pending => {
                let now_seconds = unix_seconds(SystemTime::now());
                let notify = match state.pending_rooms.get_mut(&room_key) {
                    Some(existing) => {
                        existing.last_seen_unix_seconds = now_seconds;
                        false
                    }
                    None => {
                        state.pending_rooms.insert(
                            room_key,
                            PendingRoom {
                                room,
                                first_seen_unix_seconds: now_seconds,
                                last_seen_unix_seconds: now_seconds,
                                sample_user_id: Some(event.user.id.clone()),
                                sample_username: event.user.username.clone(),
                            },
                        );
                        true
                    }
                };
                self.access_state.save(&state).await?;
                Ok(EventAccessDecision::Pending { notify })
            }
        }
    }

    pub async fn run_driver<D: ChannelDriver + Send>(
        &self,
        agent_id: &str,
        driver: &mut D,
        timeout_ms: Option<u64>,
    ) -> Result<()> {
        let (action_tx, mut action_rx) = mpsc::unbounded_channel::<WorkerAction>();
        let mut dispatch_state = DriverDispatchState::new(action_tx);
        let mut driver_closed = false;

        let run_result = async {
            loop {
                while let Ok(action) = action_rx.try_recv() {
                    self.handle_worker_action(
                        driver,
                        timeout_ms,
                        agent_id,
                        action,
                        &mut dispatch_state,
                    )
                    .await?;
                }

                if driver_closed && dispatch_state.active_conversations.is_empty() {
                    break;
                }

                if driver_closed {
                    match action_rx.recv().await {
                        Some(action) => {
                            self.handle_worker_action(
                                driver,
                                timeout_ms,
                                agent_id,
                                action,
                                &mut dispatch_state,
                            )
                            .await?;
                        }
                        None => break,
                    }
                    continue;
                }

                if dispatch_state.active_conversations.is_empty() {
                    match driver.next_event().await? {
                        Some(event) => {
                            self.handle_inbound_event(
                                agent_id,
                                driver,
                                event,
                                timeout_ms,
                                &mut dispatch_state,
                            )
                            .await?;
                        }
                        None => driver_closed = true,
                    }
                    continue;
                }

                enum DriverLoopOutcome {
                    Event(Result<Option<InboundEvent>>),
                    Action(Option<WorkerAction>),
                }

                let outcome = {
                    let next_event = driver.next_event();
                    tokio::pin!(next_event);
                    tokio::select! {
                        event_result = &mut next_event => DriverLoopOutcome::Event(event_result),
                        maybe_action = action_rx.recv() => DriverLoopOutcome::Action(maybe_action),
                    }
                };

                match outcome {
                    DriverLoopOutcome::Event(event_result) => match event_result? {
                        Some(event) => {
                            self.handle_inbound_event(
                                agent_id,
                                driver,
                                event,
                                timeout_ms,
                                &mut dispatch_state,
                            )
                            .await?;
                        }
                        None => driver_closed = true,
                    },
                    DriverLoopOutcome::Action(maybe_action) => match maybe_action {
                        Some(action) => {
                            self.handle_worker_action(
                                driver,
                                timeout_ms,
                                agent_id,
                                action,
                                &mut dispatch_state,
                            )
                            .await?;
                        }
                        None => break,
                    },
                }
            }
            Result::<()>::Ok(())
        }
        .await;

        let shutdown_result = driver.shutdown().await;
        run_result?;
        shutdown_result
    }

    async fn handle_inbound_event<D: ChannelDriver + Send>(
        &self,
        agent_id: &str,
        driver: &mut D,
        event: InboundEvent,
        timeout_ms: Option<u64>,
        dispatch_state: &mut DriverDispatchState,
    ) -> Result<()> {
        match self.authorize_event(driver, &event).await? {
            EventAccessDecision::Allow => {}
            EventAccessDecision::Ignore => return Ok(()),
            EventAccessDecision::Pending { notify } => {
                if notify {
                    driver
                        .send(&event.conversation, pending_approval_message())
                        .await?;
                }
                return Ok(());
            }
        }

        let reset_requested = event
            .metadata
            .get("reset_session")
            .and_then(|value| value.as_bool())
            .unwrap_or(false);
        let queued = QueuedInboundEvent {
            conversation_id: serialize_binding_key(&event.conversation)?,
            event,
            reset_requested,
            stream: WorkerStreamConfig {
                mode: driver.stream_mode(),
                stream_thinking: driver.stream_thinking(),
                persist_thinking: driver.persist_thinking(),
            },
        };

        if dispatch_state
            .active_conversations
            .contains(&queued.conversation_id)
        {
            dispatch_state
                .queued_events
                .entry(queued.conversation_id.clone())
                .or_default()
                .push_back(queued);
            return Ok(());
        }

        self.spawn_worker(
            agent_id,
            queued,
            timeout_ms,
            &dispatch_state.action_tx,
            &mut dispatch_state.active_conversations,
        );
        Ok(())
    }

    async fn handle_worker_action<D: ChannelDriver + Send>(
        &self,
        driver: &mut D,
        timeout_ms: Option<u64>,
        agent_id: &str,
        action: WorkerAction,
        dispatch_state: &mut DriverDispatchState,
    ) -> Result<()> {
        match action {
            WorkerAction::Progress { event, update } => {
                let _ = driver.send_progress(&event, update).await;
            }
            WorkerAction::Completed {
                conversation_id,
                event,
                outbound,
            } => {
                let outbound = driver.enrich_outbound_for_event(&event, outbound);
                driver.send(&event.conversation, outbound).await?;
                dispatch_state.active_conversations.remove(&conversation_id);
                if let Some(queue) = dispatch_state.queued_events.get_mut(&conversation_id) {
                    if let Some(next) = queue.pop_front() {
                        self.spawn_worker(
                            agent_id,
                            next,
                            timeout_ms,
                            &dispatch_state.action_tx,
                            &mut dispatch_state.active_conversations,
                        );
                    }
                    if queue.is_empty() {
                        dispatch_state.queued_events.remove(&conversation_id);
                    }
                }
            }
        }
        Ok(())
    }

    fn spawn_worker(
        &self,
        agent_id: &str,
        queued: QueuedInboundEvent,
        timeout_ms: Option<u64>,
        action_tx: &mpsc::UnboundedSender<WorkerAction>,
        active_conversations: &mut HashSet<String>,
    ) {
        active_conversations.insert(queued.conversation_id.clone());
        let runner = self.clone();
        let agent_id = agent_id.to_string();
        let action_tx = action_tx.clone();
        tokio::spawn(async move {
            let event = queued.event.clone();
            let outbound = match runner
                .handle_event_with_progress(
                    &agent_id,
                    &queued.event,
                    queued.reset_requested,
                    timeout_ms,
                    queued.stream,
                    &action_tx,
                )
                .await
            {
                Ok(message) => message,
                Err(err) => OutboundMessage::text(format!("Turin error: {}", err)),
            };
            let _ = action_tx.send(WorkerAction::Completed {
                conversation_id: queued.conversation_id,
                event,
                outbound,
            });
        });
    }

    async fn handle_event_with_progress(
        &self,
        agent_id: &str,
        event: &InboundEvent,
        reset_requested: bool,
        timeout_ms: Option<u64>,
        stream: WorkerStreamConfig,
        action_tx: &mpsc::UnboundedSender<WorkerAction>,
    ) -> Result<OutboundMessage> {
        let binding = self
            .ensure_session(agent_id, &event.conversation, reset_requested)
            .await?;
        let session_events = if should_subscribe_to_session_events(&stream) {
            self.daemon
                .subscribe_managed(RuntimeEventsSubscribeParams {
                    agent_id: None,
                    session_id: Some(binding.session_id.clone()),
                    slot_id: Some(binding.slot_id.clone()),
                })
                .await
                .ok()
        } else {
            None
        };
        let submitted = self.submit_with_binding(&binding, event).await?;
        let task_ctx = WorkerTaskContext {
            event,
            binding: &binding,
            submitted: &submitted,
        };
        let (task, final_thinking) = self
            .wait_for_task_with_progress(&task_ctx, session_events, timeout_ms, &stream, action_tx)
            .await?;
        let outbound = task_to_outbound(&task);
        Ok(attach_final_thinking(outbound, final_thinking))
    }

    async fn wait_for_task_with_progress(
        &self,
        task_ctx: &WorkerTaskContext<'_>,
        mut session_events: Option<turin_daemon_client::ManagedEventStream>,
        timeout_ms: Option<u64>,
        stream: &WorkerStreamConfig,
        action_tx: &mpsc::UnboundedSender<WorkerAction>,
    ) -> Result<(TaskSnapshot, Option<String>)> {
        let capture_thinking = stream.stream_thinking || stream.persist_thinking;
        if stream.mode == ChannelStreamMode::Off {
            let task = self
                .daemon
                .request_ok(
                    None,
                    turin_daemon_protocol::DaemonRequest::TaskWait(WaitTaskParams {
                        request_id: task_ctx.submitted.request_id.clone(),
                        timeout_ms,
                    }),
                )
                .await?;
            return Ok((task, None));
        }

        if stream.mode.sends_typing() {
            self.emit_worker_progress(action_tx, task_ctx.event, ChannelProgressUpdate::Typing);
        }

        let wait_task = self.daemon.request_ok(
            None,
            turin_daemon_protocol::DaemonRequest::TaskWait(WaitTaskParams {
                request_id: task_ctx.submitted.request_id.clone(),
                timeout_ms,
            }),
        );
        tokio::pin!(wait_task);

        let mut typing_tick = tokio::time::interval_at(
            Instant::now() + Duration::from_secs(4),
            Duration::from_secs(4),
        );
        typing_tick.set_missed_tick_behavior(MissedTickBehavior::Delay);

        let mut task_started = false;
        let mut text_preview = String::new();
        let mut thinking_preview = String::new();
        let mut last_flushed_chars = 0usize;
        let mut last_flush_at = Instant::now();

        loop {
            tokio::select! {
                result = &mut wait_task => {
                    if stream.mode.streams_text()
                        && preview_char_count(&text_preview, stream.stream_thinking.then_some(thinking_preview.as_str())) > last_flushed_chars
                    {
                        self.emit_worker_progress(
                            action_tx,
                            task_ctx.event,
                            ChannelProgressUpdate::StreamingPreview {
                                text: text_preview.clone(),
                                thinking: preview_thinking(stream.stream_thinking, &thinking_preview),
                            },
                        );
                    }
                    let task = result?;
                    let final_thinking = preview_thinking(capture_thinking, &thinking_preview);
                    return Ok((task, final_thinking));
                }
                _ = typing_tick.tick(), if stream.mode.sends_typing() => {
                    self.emit_worker_progress(action_tx, task_ctx.event, ChannelProgressUpdate::Typing);
                }
                event_result = next_managed_event(session_events.as_mut()), if session_events.is_some() => {
                    let Ok(kernel_event) = event_result else {
                        session_events = None;
                        continue;
                    };
                    if kernel_event.data.get("session_id").and_then(|value| value.as_str()) != Some(task_ctx.binding.session_id.as_str()) {
                        continue;
                    }

                    match kernel_event.event.as_str() {
                        "task_start" if kernel_event.data.get("trace_id").and_then(|value| value.as_str()) == Some(task_ctx.submitted.trace_id.as_str()) => {
                            task_started = true;
                        }
                        "message_delta" if task_started => {
                            if let Some(delta) = kernel_event.data.get("content_delta").and_then(|value| value.as_str()) {
                                text_preview.push_str(delta);
                            }
                            if should_flush_preview(
                                stream.mode,
                                &text_preview,
                                stream.stream_thinking.then_some(thinking_preview.as_str()),
                                last_flushed_chars,
                                last_flush_at,
                            ) {
                                self.emit_worker_progress(
                                    action_tx,
                                    task_ctx.event,
                                    ChannelProgressUpdate::StreamingPreview {
                                        text: text_preview.clone(),
                                        thinking: preview_thinking(stream.stream_thinking, &thinking_preview),
                                    },
                                );
                                last_flushed_chars = preview_char_count(
                                    &text_preview,
                                    stream.stream_thinking.then_some(thinking_preview.as_str()),
                                );
                                last_flush_at = Instant::now();
                            }
                        }
                        "thinking_delta" if task_started && capture_thinking => {
                            if let Some(delta) = kernel_event.data.get("thinking").and_then(|value| value.as_str()) {
                                thinking_preview.push_str(delta);
                            }
                            if should_flush_preview(
                                stream.mode,
                                &text_preview,
                                stream.stream_thinking.then_some(thinking_preview.as_str()),
                                last_flushed_chars,
                                last_flush_at,
                            ) {
                                self.emit_worker_progress(
                                    action_tx,
                                    task_ctx.event,
                                    ChannelProgressUpdate::StreamingPreview {
                                        text: text_preview.clone(),
                                        thinking: preview_thinking(stream.stream_thinking, &thinking_preview),
                                    },
                                );
                                last_flushed_chars = preview_char_count(
                                    &text_preview,
                                    stream.stream_thinking.then_some(thinking_preview.as_str()),
                                );
                                last_flush_at = Instant::now();
                            }
                        }
                        "message_end"
                            if task_started
                                && preview_char_count(
                                    &text_preview,
                                    stream.stream_thinking.then_some(thinking_preview.as_str()),
                                ) > last_flushed_chars =>
                        {
                            self.emit_worker_progress(
                                action_tx,
                                task_ctx.event,
                                ChannelProgressUpdate::StreamingPreview {
                                    text: text_preview.clone(),
                                    thinking: preview_thinking(
                                        stream.stream_thinking,
                                        &thinking_preview,
                                    ),
                                },
                            );
                            last_flushed_chars = preview_char_count(
                                &text_preview,
                                stream.stream_thinking.then_some(thinking_preview.as_str()),
                            );
                            last_flush_at = Instant::now();
                        }
                        "task_complete" if kernel_event.data.get("trace_id").and_then(|value| value.as_str()) == Some(task_ctx.submitted.trace_id.as_str()) => {
                            task_started = false;
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    fn emit_worker_progress(
        &self,
        action_tx: &mpsc::UnboundedSender<WorkerAction>,
        event: &InboundEvent,
        update: ChannelProgressUpdate,
    ) {
        let _ = action_tx.send(WorkerAction::Progress {
            event: event.clone(),
            update,
        });
    }

    pub async fn clear_binding(&self, key: &ChannelConversationKey) -> Result<()> {
        let _guard = self.bindings_lock.lock().await;
        let mut bindings = self.bindings.load().await?;
        bindings.remove(&serialize_binding_key(key)?);
        self.bindings.save(&bindings).await
    }

    pub async fn prune_expired(&self) -> Result<()> {
        let Some(ttl) = self.idle_ttl else {
            return Ok(());
        };
        let _guard = self.bindings_lock.lock().await;
        let mut bindings = self.bindings.load().await?;
        let now = SystemTime::now();
        bindings.retain(|_, binding| !binding.is_expired(now, ttl));
        self.bindings.save(&bindings).await
    }
}

fn read_task_timeout_ms(value: Option<&Value>) -> Result<Option<u64>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let timeout_ms = value.as_u64().ok_or_else(|| {
        anyhow::anyhow!("channel setting 'task_timeout_ms' must be a non-negative integer")
    })?;
    if timeout_ms == 0 {
        Ok(None)
    } else {
        Ok(Some(timeout_ms))
    }
}

fn pending_approval_message() -> OutboundMessage {
    OutboundMessage::text(
        "This conversation is pending approval. Turin will not respond here until the operator approves this room.",
    )
}

async fn next_managed_event(
    stream: Option<&mut turin_daemon_client::ManagedEventStream>,
) -> Result<turin_daemon_protocol::EventEnvelope> {
    let stream = stream.context("managed event stream missing")?;
    stream.next_event().await
}

#[cfg(test)]
mod tests;
