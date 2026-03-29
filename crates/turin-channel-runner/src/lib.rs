use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::time::{Duration, SystemTime};
use tokio::sync::mpsc;
use tokio::time::{Instant, MissedTickBehavior};
use turin_channel_core::{
    ChannelCapabilities, ChannelConversationKey, ChannelKind, ChannelUser, ConversationBinding,
    InboundEvent, OutboundMessage, RoutingDecision, decide_routing,
};
use turin_daemon_client::DaemonClient;
use turin_daemon_protocol::{
    OpenSessionParams, ResumeSessionParams, RuntimeEventsSubscribeParams, SubmitTaskParams,
    WaitTaskParams,
};
use turin_types::ToolsConfig;

#[derive(Debug, Clone)]
pub struct RunnerConfig {
    pub state_path: PathBuf,
    pub access_state_path: PathBuf,
    pub idle_ttl: Option<Duration>,
    pub access_policy: ChannelAccessPolicy,
    pub tools: ToolsConfig,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct BindingFile {
    bindings: HashMap<String, ConversationBinding>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairingMode {
    Off,
    Pending,
    Auto,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelAccessPolicy {
    pub pairing_mode: PairingMode,
    pub pairing_users: HashSet<String>,
    pub allowed_users: HashSet<String>,
    pub banned_users: HashSet<String>,
}

impl Default for ChannelAccessPolicy {
    fn default() -> Self {
        Self {
            pairing_mode: PairingMode::Off,
            pairing_users: HashSet::new(),
            allowed_users: HashSet::new(),
            banned_users: HashSet::new(),
        }
    }
}

impl ChannelAccessPolicy {
    pub fn from_settings(settings: &Value) -> Result<Self> {
        let map = settings
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("Channel settings must be a JSON object"))?;
        let pairing_mode = parse_pairing_mode(map.get("pairing_mode"))?;
        Ok(Self {
            pairing_mode,
            pairing_users: parse_string_set(map.get("pairing_users"), "pairing_users")?,
            allowed_users: parse_string_set(map.get("allowed_users"), "allowed_users")?,
            banned_users: parse_string_set(map.get("banned_users"), "banned_users")?,
        })
    }

    pub fn validate_settings(settings: &Value) -> Result<()> {
        Self::from_settings(settings).map(|_| ())
    }

    pub fn requires_unconfigured_inbound(&self) -> bool {
        !matches!(self.pairing_mode, PairingMode::Off)
    }

    fn matches_any<D: ChannelDriver>(
        &self,
        driver: &D,
        selectors: &HashSet<String>,
        user: &ChannelUser,
    ) -> bool {
        selectors
            .iter()
            .any(|selector| driver.user_matches_selector(selector, user))
    }

    fn is_banned<D: ChannelDriver>(&self, driver: &D, user: &ChannelUser) -> bool {
        !self.banned_users.is_empty() && self.matches_any(driver, &self.banned_users, user)
    }

    fn allows_pairing<D: ChannelDriver>(&self, driver: &D, user: &ChannelUser) -> bool {
        self.pairing_users.is_empty() || self.matches_any(driver, &self.pairing_users, user)
    }

    fn allows_interaction<D: ChannelDriver>(&self, driver: &D, user: &ChannelUser) -> bool {
        if self.is_banned(driver, user) {
            return false;
        }
        self.allowed_users.is_empty() || self.matches_any(driver, &self.allowed_users, user)
    }
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelRoomRef {
    pub channel: ChannelKind,
    pub workspace_id: String,
    pub room_id: Option<String>,
    pub thread_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovedRoomView {
    pub room: ChannelRoomRef,
    pub approved_at_unix_secs: u64,
    pub approved_by_user_id: Option<String>,
    pub approved_by_username: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingRoomView {
    pub room: ChannelRoomRef,
    pub first_seen_unix_secs: u64,
    pub last_seen_unix_secs: u64,
    pub sample_user_id: Option<String>,
    pub sample_username: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelAccessSnapshot {
    pub approved_rooms: Vec<ApprovedRoomView>,
    pub pending_rooms: Vec<PendingRoomView>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct AccessStateFile {
    #[serde(default)]
    approved_rooms: HashMap<String, ApprovedRoom>,
    #[serde(default)]
    pending_rooms: HashMap<String, PendingRoom>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ApprovedRoom {
    room: ChannelRoomKey,
    approved_at_unix_secs: u64,
    approved_by_user_id: Option<String>,
    approved_by_username: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PendingRoom {
    room: ChannelRoomKey,
    first_seen_unix_secs: u64,
    last_seen_unix_secs: u64,
    sample_user_id: Option<String>,
    sample_username: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
struct ChannelRoomKey {
    channel: ChannelKind,
    workspace_id: String,
    room_id: Option<String>,
    thread_id: String,
}

impl From<&ChannelConversationKey> for ChannelRoomKey {
    fn from(value: &ChannelConversationKey) -> Self {
        Self {
            channel: value.channel.clone(),
            workspace_id: value.workspace_id.clone(),
            room_id: value.room_id.clone(),
            thread_id: value.thread_id.clone(),
        }
    }
}

impl From<&ChannelRoomKey> for ChannelRoomRef {
    fn from(value: &ChannelRoomKey) -> Self {
        Self {
            channel: value.channel.clone(),
            workspace_id: value.workspace_id.clone(),
            room_id: value.room_id.clone(),
            thread_id: value.thread_id.clone(),
        }
    }
}

impl From<&ChannelRoomRef> for ChannelRoomKey {
    fn from(value: &ChannelRoomRef) -> Self {
        Self {
            channel: value.channel.clone(),
            workspace_id: value.workspace_id.clone(),
            room_id: value.room_id.clone(),
            thread_id: value.thread_id.clone(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TaskSnapshot {
    pub request_id: String,
    pub agent_id: String,
    pub slot_id: String,
    pub trace_id: String,
    pub state: String,
    pub runtime_task_id: Option<String>,
    pub status: Option<String>,
    pub task_turn_count: Option<u32>,
    pub output: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelStreamMode {
    Off,
    Typing,
    Draft,
    Block,
}

impl ChannelStreamMode {
    pub fn sends_typing(self) -> bool {
        matches!(self, Self::Typing | Self::Draft | Self::Block)
    }

    pub fn streams_text(self) -> bool {
        matches!(self, Self::Draft | Self::Block)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChannelProgressUpdate {
    Typing,
    StreamingPreview {
        text: String,
        thinking: Option<String>,
    },
}

#[derive(Clone)]
pub struct FileBindingStore {
    path: PathBuf,
}

impl FileBindingStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub async fn load(&self) -> Result<HashMap<String, ConversationBinding>> {
        if !self.path.exists() {
            return Ok(HashMap::new());
        }
        let raw = tokio::fs::read_to_string(&self.path)
            .await
            .with_context(|| format!("Failed to read '{}'", self.path.display()))?;
        let file: BindingFile = serde_json::from_str(&raw)
            .with_context(|| format!("Failed to parse '{}'", self.path.display()))?;
        Ok(file.bindings)
    }

    pub async fn save(&self, bindings: &HashMap<String, ConversationBinding>) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let tmp = self.path.with_extension("json.tmp");
        let body = serde_json::to_string_pretty(&BindingFile {
            bindings: bindings.clone(),
        })?;
        tokio::fs::write(&tmp, body).await?;
        tokio::fs::rename(&tmp, &self.path).await?;
        Ok(())
    }
}

#[derive(Clone)]
pub struct FileAccessStateStore {
    path: PathBuf,
}

impl FileAccessStateStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    async fn load(&self) -> Result<AccessStateFile> {
        if !self.path.exists() {
            return Ok(AccessStateFile::default());
        }
        let raw = tokio::fs::read_to_string(&self.path)
            .await
            .with_context(|| format!("Failed to read '{}'", self.path.display()))?;
        serde_json::from_str(&raw)
            .with_context(|| format!("Failed to parse '{}'", self.path.display()))
    }

    async fn save(&self, state: &AccessStateFile) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let tmp = self.path.with_extension("json.tmp");
        let body = serde_json::to_string_pretty(state)?;
        tokio::fs::write(&tmp, body).await?;
        tokio::fs::rename(&tmp, &self.path).await?;
        Ok(())
    }

    pub async fn snapshot(&self) -> Result<ChannelAccessSnapshot> {
        let state = self.load().await?;
        Ok(channel_access_snapshot(&state))
    }

    pub async fn approve(
        &self,
        room: &ChannelRoomRef,
        approved_by_user_id: Option<String>,
        approved_by_username: Option<String>,
    ) -> Result<ChannelAccessSnapshot> {
        let mut state = self.load().await?;
        let room_key = ChannelRoomKey::from(room);
        let serialized_room = serialize_room_key(&room_key)?;
        state.pending_rooms.remove(&serialized_room);
        state.approved_rooms.insert(
            serialized_room,
            ApprovedRoom {
                room: room_key,
                approved_at_unix_secs: unix_secs(SystemTime::now()),
                approved_by_user_id,
                approved_by_username,
            },
        );
        self.save(&state).await?;
        Ok(channel_access_snapshot(&state))
    }

    pub async fn reject_pending(&self, room: &ChannelRoomRef) -> Result<ChannelAccessSnapshot> {
        let mut state = self.load().await?;
        state
            .pending_rooms
            .remove(&serialize_room_key(&ChannelRoomKey::from(room))?);
        self.save(&state).await?;
        Ok(channel_access_snapshot(&state))
    }

    pub async fn revoke(&self, room: &ChannelRoomRef) -> Result<ChannelAccessSnapshot> {
        let mut state = self.load().await?;
        state
            .approved_rooms
            .remove(&serialize_room_key(&ChannelRoomKey::from(room))?);
        self.save(&state).await?;
        Ok(channel_access_snapshot(&state))
    }
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
    bindings: FileBindingStore,
    access_state: FileAccessStateStore,
    idle_ttl: Option<Duration>,
    access_policy: ChannelAccessPolicy,
    tools: ToolsConfig,
}

#[derive(Debug, Clone)]
struct WorkerStreamConfig {
    mode: ChannelStreamMode,
    stream_thinking: bool,
    persist_thinking: bool,
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
            bindings: FileBindingStore::new(config.state_path),
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
        let mut bindings = self.bindings.load().await?;
        let binding_key = serialize_binding_key(key)?;
        let current = bindings.get(&binding_key);
        let decision = decide_routing(
            key,
            current,
            SystemTime::now(),
            self.idle_ttl,
            reset_requested,
        );

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
                                turin_daemon_protocol::SessionIdParams {
                                    session_id: binding.session_id.clone(),
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
        bindings.insert(binding_key, binding.clone());
        self.bindings.save(&bindings).await?;
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
        self.daemon
            .request_ok(
                None,
                turin_daemon_protocol::DaemonRequest::TaskSubmit(SubmitTaskParams {
                    agent_id: None,
                    session_id: Some(binding.session_id.clone()),
                    prompt: event.prompt_text(),
                    tools: (!self.tools.is_empty()).then_some(self.tools.clone()),
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
                if self.access_policy.allows_interaction(driver, &event.user) {
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
                if self.access_policy.allows_interaction(driver, &event.user) {
                    EventAccessDecision::Allow
                } else {
                    EventAccessDecision::Ignore
                },
            );
        }

        if self.access_policy.is_banned(driver, &event.user) {
            return Ok(EventAccessDecision::Ignore);
        }

        if !self.access_policy.allows_pairing(driver, &event.user) {
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
                        approved_at_unix_secs: unix_secs(now),
                        approved_by_user_id: Some(event.user.id.clone()),
                        approved_by_username: event.user.username.clone(),
                    },
                );
                state.pending_rooms.remove(&room_key);
                self.access_state.save(&state).await?;
                Ok(EventAccessDecision::Allow)
            }
            PairingMode::Pending => {
                let now_secs = unix_secs(SystemTime::now());
                let notify = match state.pending_rooms.get_mut(&room_key) {
                    Some(existing) => {
                        existing.last_seen_unix_secs = now_secs;
                        false
                    }
                    None => {
                        state.pending_rooms.insert(
                            room_key,
                            PendingRoom {
                                room,
                                first_seen_unix_secs: now_secs,
                                last_seen_unix_secs: now_secs,
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
        let session_events = if stream.mode.streams_text() || stream.stream_thinking {
            self.daemon
                .subscribe_managed(RuntimeEventsSubscribeParams {
                    agent_id: None,
                    session_id: Some(binding.session_id.clone()),
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
                        "message_end" if task_started => {
                            if preview_char_count(
                                &text_preview,
                                stream.stream_thinking.then_some(thinking_preview.as_str()),
                            ) > last_flushed_chars {
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
        let mut bindings = self.bindings.load().await?;
        bindings.remove(&serialize_binding_key(key)?);
        self.bindings.save(&bindings).await
    }

    pub async fn prune_expired(&self) -> Result<()> {
        let Some(ttl) = self.idle_ttl else {
            return Ok(());
        };
        let mut bindings = self.bindings.load().await?;
        let now = SystemTime::now();
        bindings.retain(|_, binding| !binding.is_expired(now, ttl));
        self.bindings.save(&bindings).await
    }
}

fn serialize_binding_key(key: &ChannelConversationKey) -> Result<String> {
    Ok(serde_json::to_string(key)?)
}

fn serialize_room_key(key: &ChannelRoomKey) -> Result<String> {
    Ok(serde_json::to_string(key)?)
}

fn channel_access_snapshot(state: &AccessStateFile) -> ChannelAccessSnapshot {
    let mut approved_rooms: Vec<_> = state
        .approved_rooms
        .values()
        .map(|room| ApprovedRoomView {
            room: ChannelRoomRef::from(&room.room),
            approved_at_unix_secs: room.approved_at_unix_secs,
            approved_by_user_id: room.approved_by_user_id.clone(),
            approved_by_username: room.approved_by_username.clone(),
        })
        .collect();
    approved_rooms.sort_by(|left, right| {
        left.room
            .workspace_id
            .cmp(&right.room.workspace_id)
            .then_with(|| left.room.room_id.cmp(&right.room.room_id))
            .then_with(|| left.room.thread_id.cmp(&right.room.thread_id))
    });

    let mut pending_rooms: Vec<_> = state
        .pending_rooms
        .values()
        .map(|room| PendingRoomView {
            room: ChannelRoomRef::from(&room.room),
            first_seen_unix_secs: room.first_seen_unix_secs,
            last_seen_unix_secs: room.last_seen_unix_secs,
            sample_user_id: room.sample_user_id.clone(),
            sample_username: room.sample_username.clone(),
        })
        .collect();
    pending_rooms.sort_by(|left, right| {
        left.room
            .workspace_id
            .cmp(&right.room.workspace_id)
            .then_with(|| left.room.room_id.cmp(&right.room.room_id))
            .then_with(|| left.room.thread_id.cmp(&right.room.thread_id))
    });

    ChannelAccessSnapshot {
        approved_rooms,
        pending_rooms,
    }
}

fn parse_pairing_mode(value: Option<&Value>) -> Result<PairingMode> {
    let Some(value) = value else {
        return Ok(PairingMode::Off);
    };
    let mode = value
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("channel setting 'pairing_mode' must be a string"))?;
    match mode.trim().to_ascii_lowercase().as_str() {
        "off" => Ok(PairingMode::Off),
        "pending" => Ok(PairingMode::Pending),
        "auto" => Ok(PairingMode::Auto),
        _ => anyhow::bail!("channel setting 'pairing_mode' must be one of: off, pending, auto"),
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

fn parse_string_set(value: Option<&Value>, key: &str) -> Result<HashSet<String>> {
    let mut out = HashSet::new();
    let Some(value) = value else {
        return Ok(out);
    };

    match value {
        Value::Array(values) => {
            for item in values {
                let text = item.as_str().ok_or_else(|| {
                    anyhow::anyhow!("channel setting '{}' must be an array of strings", key)
                })?;
                let normalized = normalize_string_item(text).ok_or_else(|| {
                    anyhow::anyhow!(
                        "channel setting '{}' must not contain empty string values",
                        key
                    )
                })?;
                out.insert(normalized);
            }
        }
        Value::String(text) => {
            for item in text.split(',') {
                let normalized = normalize_string_item(item).ok_or_else(|| {
                    anyhow::anyhow!(
                        "channel setting '{}' must not contain empty string values",
                        key
                    )
                })?;
                out.insert(normalized);
            }
        }
        _ => {
            anyhow::bail!(
                "channel setting '{}' must be a string or array of strings",
                key
            );
        }
    }

    Ok(out)
}

fn normalize_string_item(text: &str) -> Option<String> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn pending_approval_message() -> OutboundMessage {
    OutboundMessage::text(
        "This conversation is pending approval. Turin will not respond here until the operator approves this room.",
    )
}

fn unix_secs(time: SystemTime) -> u64 {
    time.duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn task_to_outbound(task: &TaskSnapshot) -> OutboundMessage {
    if let Some(output) = task.output.as_ref() {
        if let Some(structured) = try_parse_structured_outbound(output) {
            structured
        } else {
            OutboundMessage::text(output.clone())
        }
    } else if let Some(error) = task.error.as_ref() {
        OutboundMessage::text(format!("Turin error: {}", error))
    } else {
        OutboundMessage::text(format!("Task {} finished without output", task.request_id))
    }
}

async fn next_managed_event(
    stream: Option<&mut turin_daemon_client::ManagedEventStream>,
) -> Result<turin_daemon_protocol::EventEnvelope> {
    let stream = stream.context("managed event stream missing")?;
    stream.next_event().await
}

fn should_flush_preview(
    stream_mode: ChannelStreamMode,
    text_preview: &str,
    thinking_preview: Option<&str>,
    last_flushed_chars: usize,
    last_flush_at: Instant,
) -> bool {
    let current_chars = preview_char_count(text_preview, thinking_preview);
    if current_chars <= last_flushed_chars {
        return false;
    }

    let new_chars = current_chars.saturating_sub(last_flushed_chars);
    match stream_mode {
        ChannelStreamMode::Draft => {
            new_chars >= 32 || last_flush_at.elapsed() >= Duration::from_millis(800)
        }
        ChannelStreamMode::Block => {
            new_chars >= 160
                || (new_chars >= 64
                    && last_flush_at.elapsed() >= Duration::from_millis(1500)
                    && (text_preview.ends_with('\n') || text_preview.ends_with(". ")))
        }
        _ => false,
    }
}

fn preview_char_count(text_preview: &str, thinking_preview: Option<&str>) -> usize {
    text_preview.chars().count()
        + thinking_preview
            .map(|thinking| thinking.chars().count())
            .unwrap_or_default()
}

fn preview_thinking(include_thinking: bool, thinking_preview: &str) -> Option<String> {
    if !include_thinking || thinking_preview.trim().is_empty() {
        return None;
    }
    Some(thinking_preview.to_string())
}

fn attach_final_thinking(
    mut outbound: OutboundMessage,
    thinking: Option<String>,
) -> OutboundMessage {
    let Some(thinking) = thinking.map(|value| value.trim().to_string()) else {
        return outbound;
    };
    if thinking.is_empty() {
        return outbound;
    }
    outbound.metadata.insert(
        "channel_final_thinking".to_string(),
        Value::String(thinking),
    );
    outbound
}

#[derive(Debug, Clone, Deserialize)]
struct StructuredOutbound {
    #[serde(default)]
    _turin_channel_outbound: bool,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    blocks: Vec<turin_channel_core::MessageBlock>,
    #[serde(default)]
    attachments: Vec<turin_channel_core::ChannelAttachment>,
    #[serde(default)]
    embeds: Vec<Value>,
    #[serde(default)]
    components: Vec<Value>,
    #[serde(default)]
    metadata: Map<String, Value>,
}

fn try_parse_structured_outbound(raw: &str) -> Option<OutboundMessage> {
    let trimmed = raw.trim();
    if !trimmed.starts_with('{') {
        return None;
    }

    let parsed: StructuredOutbound = serde_json::from_str(trimmed).ok()?;
    if !parsed._turin_channel_outbound {
        return None;
    }

    let mut blocks = parsed.blocks;
    if blocks.is_empty()
        && let Some(content) = parsed.content
        && !content.trim().is_empty()
    {
        blocks.push(turin_channel_core::MessageBlock::Text { text: content });
    }

    Some(OutboundMessage {
        blocks,
        attachments: parsed.attachments,
        embeds: parsed.embeds,
        components: parsed.components,
        metadata: parsed.metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use turin_channel_core::{
        ChannelKind, ChannelMessageRef, ChannelSessionScope, ChannelUser, MessageBlock,
    };

    struct TestDriver;

    #[async_trait::async_trait]
    impl ChannelDriver for TestDriver {
        fn kind(&self) -> ChannelKind {
            ChannelKind::new("test")
        }

        fn user_matches_selector(&self, selector: &str, user: &ChannelUser) -> bool {
            let selector = selector.trim();
            if selector.is_empty() {
                return false;
            }
            let selector = selector.strip_prefix('@').unwrap_or(selector);
            user.id == selector
                || user
                    .username
                    .as_ref()
                    .is_some_and(|username| username.eq_ignore_ascii_case(selector))
        }

        async fn next_event(&mut self) -> Result<Option<InboundEvent>> {
            Ok(None)
        }

        async fn send(
            &mut self,
            _conversation: &ChannelConversationKey,
            _message: OutboundMessage,
        ) -> Result<()> {
            Ok(())
        }

        async fn shutdown(&mut self) -> Result<()> {
            Ok(())
        }
    }

    fn sample_key() -> ChannelConversationKey {
        ChannelConversationKey {
            channel: ChannelKind::new("discord"),
            workspace_id: "guild".into(),
            room_id: Some("room".into()),
            thread_id: "thread".into(),
            user_id: Some("user".into()),
        }
    }

    #[tokio::test]
    async fn file_binding_store_round_trips() {
        let dir = tempdir().unwrap();
        let store = FileBindingStore::new(dir.path().join("bindings.json"));
        let key = serialize_binding_key(&sample_key()).unwrap();
        let mut map = HashMap::new();
        map.insert(
            key,
            ConversationBinding::new("writer", "session-1", &sample_key(), SystemTime::UNIX_EPOCH),
        );
        store.save(&map).await.unwrap();
        let loaded = store.load().await.unwrap();
        assert_eq!(loaded.len(), 1);
    }

    #[tokio::test]
    async fn file_access_state_store_round_trips() {
        let dir = tempdir().unwrap();
        let store = FileAccessStateStore::new(dir.path().join("access.json"));
        let room = ChannelRoomKey::from(&sample_key());
        let key = serialize_room_key(&room).unwrap();
        let mut state = AccessStateFile::default();
        state.approved_rooms.insert(
            key,
            ApprovedRoom {
                room,
                approved_at_unix_secs: 1,
                approved_by_user_id: Some("user".into()),
                approved_by_username: Some("owner".into()),
            },
        );
        store.save(&state).await.unwrap();
        let loaded = store.load().await.unwrap();
        assert_eq!(loaded.approved_rooms.len(), 1);
    }

    #[tokio::test]
    async fn file_access_state_store_manages_public_snapshot() {
        let dir = tempdir().unwrap();
        let store = FileAccessStateStore::new(dir.path().join("access.json"));
        let room = ChannelRoomRef {
            channel: ChannelKind::new("telegram"),
            workspace_id: "telegram".into(),
            room_id: Some("-100123".into()),
            thread_id: "-100123".into(),
        };

        let snapshot = store
            .approve(&room, Some("owner".into()), Some("jay".into()))
            .await
            .unwrap();
        assert_eq!(snapshot.approved_rooms.len(), 1);
        assert!(snapshot.pending_rooms.is_empty());

        let snapshot = store.reject_pending(&room).await.unwrap();
        assert_eq!(snapshot.approved_rooms.len(), 1);
        assert!(snapshot.pending_rooms.is_empty());

        let snapshot = store.revoke(&room).await.unwrap();
        assert!(snapshot.approved_rooms.is_empty());
    }

    #[test]
    fn access_policy_parses_pairing_allowed_and_banned_users() {
        let policy = ChannelAccessPolicy::from_settings(&serde_json::json!({
            "pairing_mode": "auto",
            "pairing_users": ["123", "@owner"],
            "allowed_users": "friend1,friend2",
            "banned_users": ["intruder"]
        }))
        .expect("policy should parse");
        assert_eq!(policy.pairing_mode, PairingMode::Auto);
        assert!(policy.pairing_users.contains("123"));
        assert!(policy.pairing_users.contains("@owner"));
        assert!(policy.allowed_users.contains("friend1"));
        assert!(policy.allowed_users.contains("friend2"));
        assert!(policy.banned_users.contains("intruder"));
    }

    #[test]
    fn task_timeout_ms_defaults_to_none_and_accepts_zero_as_unbounded() {
        assert_eq!(
            task_timeout_ms_from_settings(&serde_json::json!({})).unwrap(),
            None
        );
        assert_eq!(
            task_timeout_ms_from_settings(&serde_json::json!({ "task_timeout_ms": 0 })).unwrap(),
            None
        );
        assert_eq!(
            task_timeout_ms_from_settings(&serde_json::json!({ "task_timeout_ms": 45000 }))
                .unwrap(),
            Some(45_000)
        );
    }

    #[test]
    fn tools_settings_parse_string_lists() {
        let tools = tools_config_from_settings(&serde_json::json!({
            "tools": {
                "allow": ["group:web", "read_file"],
                "exclude": "web_search"
            }
        }))
        .unwrap();
        assert_eq!(
            tools.selection.allow,
            Some(vec!["group:web".to_string(), "read_file".to_string()])
        );
        assert_eq!(tools.selection.exclude, vec!["web_search".to_string()]);
    }

    fn test_runner(dir: &tempfile::TempDir, policy: ChannelAccessPolicy) -> ChannelRunner {
        ChannelRunner::new(
            turin_daemon_client::DaemonClient::new(dir.path().join("dummy.sock")),
            RunnerConfig {
                state_path: dir.path().join("bindings.json"),
                access_state_path: dir.path().join("access.json"),
                idle_ttl: Some(Duration::from_secs(600)),
                access_policy: policy,
                tools: Default::default(),
            },
        )
    }

    #[tokio::test]
    async fn authorize_event_records_pending_rooms_once() {
        let dir = tempdir().unwrap();
        let runner = test_runner(
            &dir,
            ChannelAccessPolicy {
                pairing_mode: PairingMode::Pending,
                ..Default::default()
            },
        );
        let event = InboundEvent {
            conversation: sample_key(),
            message: ChannelMessageRef {
                conversation: sample_key(),
                message_id: "m1".into(),
            },
            user: ChannelUser {
                id: "u1".into(),
                display_name: Some("User".into()),
                username: Some("user".into()),
            },
            session_scope: ChannelSessionScope::User,
            text: "hello".into(),
            attachments: vec![],
            metadata: Default::default(),
        };
        let driver = TestDriver;

        assert!(matches!(
            runner.authorize_event(&driver, &event).await.unwrap(),
            EventAccessDecision::Pending { notify: true }
        ));
        assert!(matches!(
            runner.authorize_event(&driver, &event).await.unwrap(),
            EventAccessDecision::Pending { notify: false }
        ));
    }

    #[tokio::test]
    async fn authorize_event_auto_approves_pairing_users() {
        let dir = tempdir().unwrap();
        let runner = test_runner(
            &dir,
            ChannelAccessPolicy {
                pairing_mode: PairingMode::Auto,
                pairing_users: HashSet::from(["u1".to_string()]),
                ..Default::default()
            },
        );
        let event = InboundEvent {
            conversation: sample_key(),
            message: ChannelMessageRef {
                conversation: sample_key(),
                message_id: "m1".into(),
            },
            user: ChannelUser {
                id: "u1".into(),
                display_name: Some("User".into()),
                username: Some("user".into()),
            },
            session_scope: ChannelSessionScope::User,
            text: "hello".into(),
            attachments: vec![],
            metadata: Default::default(),
        };
        let driver = TestDriver;

        assert!(matches!(
            runner.authorize_event(&driver, &event).await.unwrap(),
            EventAccessDecision::Allow
        ));
        assert!(matches!(
            runner.authorize_event(&driver, &event).await.unwrap(),
            EventAccessDecision::Allow
        ));
    }

    #[tokio::test]
    async fn authorize_event_ignores_senders_not_allowed_to_pair() {
        let dir = tempdir().unwrap();
        let runner = test_runner(
            &dir,
            ChannelAccessPolicy {
                pairing_mode: PairingMode::Auto,
                pairing_users: HashSet::from(["owner".to_string()]),
                ..Default::default()
            },
        );
        let event = InboundEvent {
            conversation: sample_key(),
            message: ChannelMessageRef {
                conversation: sample_key(),
                message_id: "m1".into(),
            },
            user: ChannelUser {
                id: "intruder".into(),
                display_name: Some("Intruder".into()),
                username: Some("intruder".into()),
            },
            session_scope: ChannelSessionScope::User,
            text: "hello".into(),
            attachments: vec![],
            metadata: Default::default(),
        };
        let driver = TestDriver;

        assert!(matches!(
            runner.authorize_event(&driver, &event).await.unwrap(),
            EventAccessDecision::Ignore
        ));
    }

    #[tokio::test]
    async fn authorize_event_allows_open_interaction_after_pairing() {
        let dir = tempdir().unwrap();
        let runner = test_runner(
            &dir,
            ChannelAccessPolicy {
                pairing_mode: PairingMode::Auto,
                pairing_users: HashSet::from(["owner".to_string()]),
                ..Default::default()
            },
        );
        let driver = TestDriver;

        let owner_event = InboundEvent {
            conversation: sample_key(),
            message: ChannelMessageRef {
                conversation: sample_key(),
                message_id: "m1".into(),
            },
            user: ChannelUser {
                id: "owner".into(),
                display_name: Some("Owner".into()),
                username: Some("jay".into()),
            },
            session_scope: ChannelSessionScope::User,
            text: "pair room".into(),
            attachments: vec![],
            metadata: Default::default(),
        };

        let friend_event = InboundEvent {
            conversation: sample_key(),
            message: ChannelMessageRef {
                conversation: sample_key(),
                message_id: "m2".into(),
            },
            user: ChannelUser {
                id: "friend".into(),
                display_name: Some("Friend".into()),
                username: Some("friend".into()),
            },
            session_scope: ChannelSessionScope::User,
            text: "hello".into(),
            attachments: vec![],
            metadata: Default::default(),
        };

        assert!(matches!(
            runner.authorize_event(&driver, &owner_event).await.unwrap(),
            EventAccessDecision::Allow
        ));
        assert!(matches!(
            runner
                .authorize_event(&driver, &friend_event)
                .await
                .unwrap(),
            EventAccessDecision::Allow
        ));
    }

    #[tokio::test]
    async fn authorize_event_applies_allowed_users_after_pairing() {
        let dir = tempdir().unwrap();
        let runner = test_runner(
            &dir,
            ChannelAccessPolicy {
                pairing_mode: PairingMode::Auto,
                pairing_users: HashSet::from(["owner".to_string()]),
                allowed_users: HashSet::from(["friend".to_string()]),
                ..Default::default()
            },
        );
        let driver = TestDriver;

        let owner_event = InboundEvent {
            conversation: sample_key(),
            message: ChannelMessageRef {
                conversation: sample_key(),
                message_id: "m1".into(),
            },
            user: ChannelUser {
                id: "owner".into(),
                display_name: Some("Owner".into()),
                username: Some("jay".into()),
            },
            session_scope: ChannelSessionScope::User,
            text: "pair room".into(),
            attachments: vec![],
            metadata: Default::default(),
        };

        let intruder_event = InboundEvent {
            conversation: sample_key(),
            message: ChannelMessageRef {
                conversation: sample_key(),
                message_id: "m2".into(),
            },
            user: ChannelUser {
                id: "intruder".into(),
                display_name: Some("Intruder".into()),
                username: Some("intruder".into()),
            },
            session_scope: ChannelSessionScope::User,
            text: "hello".into(),
            attachments: vec![],
            metadata: Default::default(),
        };

        let friend_event = InboundEvent {
            conversation: sample_key(),
            message: ChannelMessageRef {
                conversation: sample_key(),
                message_id: "m3".into(),
            },
            user: ChannelUser {
                id: "friend".into(),
                display_name: Some("Friend".into()),
                username: Some("friend".into()),
            },
            session_scope: ChannelSessionScope::User,
            text: "hello".into(),
            attachments: vec![],
            metadata: Default::default(),
        };

        assert!(matches!(
            runner.authorize_event(&driver, &owner_event).await.unwrap(),
            EventAccessDecision::Allow
        ));
        assert!(matches!(
            runner
                .authorize_event(&driver, &intruder_event)
                .await
                .unwrap(),
            EventAccessDecision::Ignore
        ));
        assert!(matches!(
            runner
                .authorize_event(&driver, &friend_event)
                .await
                .unwrap(),
            EventAccessDecision::Allow
        ));
    }

    #[tokio::test]
    async fn authorize_event_banned_users_override_approval() {
        let dir = tempdir().unwrap();
        let runner = test_runner(
            &dir,
            ChannelAccessPolicy {
                pairing_mode: PairingMode::Auto,
                pairing_users: HashSet::from(["owner".to_string()]),
                banned_users: HashSet::from(["friend".to_string()]),
                ..Default::default()
            },
        );
        let driver = TestDriver;

        let owner_event = InboundEvent {
            conversation: sample_key(),
            message: ChannelMessageRef {
                conversation: sample_key(),
                message_id: "m1".into(),
            },
            user: ChannelUser {
                id: "owner".into(),
                display_name: Some("Owner".into()),
                username: Some("jay".into()),
            },
            session_scope: ChannelSessionScope::User,
            text: "pair room".into(),
            attachments: vec![],
            metadata: Default::default(),
        };

        let friend_event = InboundEvent {
            conversation: sample_key(),
            message: ChannelMessageRef {
                conversation: sample_key(),
                message_id: "m2".into(),
            },
            user: ChannelUser {
                id: "friend".into(),
                display_name: Some("Friend".into()),
                username: Some("friend".into()),
            },
            session_scope: ChannelSessionScope::User,
            text: "hello".into(),
            attachments: vec![],
            metadata: Default::default(),
        };

        assert!(matches!(
            runner.authorize_event(&driver, &owner_event).await.unwrap(),
            EventAccessDecision::Allow
        ));
        assert!(matches!(
            runner
                .authorize_event(&driver, &friend_event)
                .await
                .unwrap(),
            EventAccessDecision::Ignore
        ));
    }

    #[test]
    fn inbound_event_shape_is_runner_compatible() {
        let key = sample_key();
        let event = InboundEvent {
            conversation: key.clone(),
            message: ChannelMessageRef {
                conversation: key,
                message_id: "m1".into(),
            },
            user: ChannelUser {
                id: "u1".into(),
                display_name: Some("User".into()),
                username: Some("user".into()),
            },
            session_scope: ChannelSessionScope::User,
            text: "hello".into(),
            attachments: vec![],
            metadata: Default::default(),
        };
        assert_eq!(event.text, "hello");
    }

    #[test]
    fn task_to_outbound_prefers_output() {
        let outbound = task_to_outbound(&TaskSnapshot {
            request_id: "req-1".into(),
            agent_id: "writer".into(),
            slot_id: "slot-1".into(),
            trace_id: "trace-1".into(),
            state: "completed".into(),
            runtime_task_id: None,
            status: Some("completed".into()),
            task_turn_count: Some(1),
            output: Some("hello".into()),
            error: Some("bad".into()),
        });
        assert_eq!(
            outbound.blocks,
            vec![MessageBlock::Text {
                text: "hello".into(),
            }]
        );
    }

    #[test]
    fn task_to_outbound_parses_structured_payload() {
        let outbound = task_to_outbound(&TaskSnapshot {
            request_id: "req-1".into(),
            agent_id: "writer".into(),
            slot_id: "slot-1".into(),
            trace_id: "trace-1".into(),
            state: "completed".into(),
            runtime_task_id: None,
            status: Some("completed".into()),
            task_turn_count: Some(1),
            output: Some(
                serde_json::json!({
                    "_turin_channel_outbound": true,
                    "content": "overview",
                    "embeds": [{ "title": "Build result" }],
                    "components": [{ "type": 1, "components": [] }],
                    "metadata": { "priority": "high" }
                })
                .to_string(),
            ),
            error: None,
        });
        assert_eq!(outbound.blocks.len(), 1);
        assert_eq!(outbound.embeds.len(), 1);
        assert_eq!(outbound.components.len(), 1);
        assert_eq!(outbound.metadata["priority"], "high");
    }
}
