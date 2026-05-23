mod allocator;
mod cancellation;
mod operations;
mod peer_runtime;
mod runtime_registry;
mod tasks;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex, OnceLock, RwLock as StdRwLock};

use crate::harness::scheduler::HarnessSchedulerAccess;
use crate::inference::embeddings::EmbeddingProvider;
use crate::inference::provider::ProviderClient;
use crate::kernel::config::InferenceOverrideConfig;
use crate::kernel::config::TurinConfig;
use crate::kernel::event::{KernelEvent, TaskBranchOutcome, TaskTerminalStatus};
use crate::kernel::execution_host::SessionPersistenceCoordinator;
use crate::kernel::governance::GovernanceManager;
use crate::kernel::harness_manager::HarnessManager;
use crate::kernel::policy::RuntimePolicyManager;
pub use crate::kernel::session::ExecutionStatusSnapshot;
use crate::kernel::session::{ExecutionConflictPolicy, QueuedTask, SessionState};
pub use crate::kernel::task_promotion::{PromotedTaskBranch, TaskPromotionCandidate};
use crate::persistence::manager::StoreManager;
use crate::tools::registry::ToolRegistry;
use tokio::sync::{Notify, RwLock, oneshot};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use turin_types::TaskInputContent;

pub(crate) type SessionEventRecord = (Option<i64>, KernelEvent);
pub(crate) type SessionEventSender = tokio::sync::broadcast::Sender<SessionEventRecord>;
pub(crate) type SessionEventReceiver = tokio::sync::broadcast::Receiver<SessionEventRecord>;

#[derive(Debug, Clone, serde::Serialize)]
pub struct PeerAgentTaskResult {
    pub request_id: String,
    pub agent_id: String,
    pub slot_id: String,
    pub trace_id: String,
    pub runtime_task_id: String,
    pub execution: ExecutionStatusSnapshot,
    pub status: TaskTerminalStatus,
    pub task_turn_count: u32,
    pub branch_outcome: Option<TaskBranchOutcome>,
    pub promotion_candidate: Option<TaskPromotionCandidate>,
    pub promoted_branch: Option<PromotedTaskBranch>,
    pub output: Option<String>,
    pub assistant_content: Option<Vec<TaskInputContent>>,
    #[serde(skip_serializing)]
    pub promotion_input_content: Option<Vec<TaskInputContent>>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AgentStatusSnapshot {
    pub agent_id: String,
    pub running: bool,
    pub active_tasks: usize,
    pub queued_tasks: usize,
    pub awaiting_results: usize,
    pub current_session_id: Option<String>,
    pub current_request_id: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TaskStatusSnapshot {
    pub request_id: String,
    pub agent_id: String,
    pub slot_id: String,
    pub trace_id: String,
    pub state: String,
    pub runtime_task_id: Option<String>,
    pub execution: ExecutionStatusSnapshot,
    pub status: Option<TaskTerminalStatus>,
    pub task_turn_count: Option<u32>,
    pub branch_outcome: Option<TaskBranchOutcome>,
    pub promotion_candidate: Option<TaskPromotionCandidate>,
    pub promoted_branch: Option<PromotedTaskBranch>,
    pub output: Option<String>,
    pub assistant_content: Option<Vec<TaskInputContent>>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TaskStatusFingerprint {
    pub(crate) request_id: String,
    state: &'static str,
    runtime_task_id: Option<String>,
    status: Option<TaskTerminalStatus>,
    task_turn_count: Option<u32>,
    branch_outcome: Option<TaskBranchOutcomeFingerprint>,
    promotion_candidate: Option<(String, i64)>,
    promoted_branch: Option<PromotedTaskBranchFingerprint>,
    output_bytes: usize,
    assistant_content_items: usize,
    assistant_content_bytes: usize,
    error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TaskBranchOutcomeFingerprint {
    ForkSibling {
        branch_id: i64,
        branch_public_id: String,
        source_turn_id: Option<i64>,
        persisted_active_head_unchanged: bool,
    },
    SidestepSibling {
        branch_id: i64,
        branch_public_id: String,
        source_turn_id: Option<i64>,
        persisted_active_head_unchanged: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PromotedTaskBranchFingerprint {
    branch_id: String,
    name: String,
    head_turn_index: Option<u32>,
    source_turn_id: Option<i64>,
    origin_kind: String,
    origin_task_id: Option<String>,
    origin_execution_id: Option<String>,
    active: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LiveSessionSnapshot {
    pub agent_id: String,
    pub slot_id: String,
    pub session_id: String,
    pub running: bool,
    pub active_tasks: usize,
    pub queued_tasks: usize,
    pub current_request_id: Option<String>,
    pub execution: ExecutionStatusSnapshot,
    pub conflict_policy: ExecutionConflictPolicy,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub history: Option<LiveSessionHistorySnapshot>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct LiveSessionHistorySnapshot {
    pub len: usize,
    pub message_offset: usize,
}

impl LiveSessionHistorySnapshot {
    fn from_session(session: &SessionState) -> Self {
        Self {
            len: session.history.len(),
            message_offset: session.history_message_offset,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RuntimeSlotKey {
    agent_id: String,
    slot_id: String,
}

impl RuntimeSlotKey {
    const DEFAULT_SLOT_ID: &str = "default";

    fn default_for(agent_id: &str) -> Self {
        Self {
            agent_id: agent_id.to_string(),
            slot_id: Self::DEFAULT_SLOT_ID.to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingTaskState {
    Queued,
    Running,
    Cancelling,
}

#[derive(Debug, Clone)]
struct PendingTaskRecord {
    runtime_key: RuntimeSlotKey,
    trace_id: String,
    state: PendingTaskState,
    runtime_task_id: Option<String>,
    execution: ExecutionStatusSnapshot,
}

pub(crate) struct RuntimeControl {
    state: StdRwLock<RuntimeControlState>,
    session_reset_request: Mutex<Option<SessionResetRequest>>,
}

#[derive(Clone)]
pub(crate) struct RuntimeControlSnapshot {
    session_id: Option<String>,
    session_events: Option<SessionEventSender>,
    session_context: SessionContextOverrides,
    execution: Option<ExecutionStatusSnapshot>,
    conflict_policy: ExecutionConflictPolicy,
    history: Option<LiveSessionHistorySnapshot>,
    request_id: Option<String>,
    runtime_task_id: Option<String>,
    cancel_token: Option<CancellationToken>,
    generation: u64,
}

#[derive(Clone)]
struct RuntimeControlState {
    session_id: Option<String>,
    session_events: Option<SessionEventSender>,
    session_context: SessionContextOverrides,
    execution: Option<ExecutionStatusSnapshot>,
    conflict_policy: ExecutionConflictPolicy,
    history: Option<LiveSessionHistorySnapshot>,
    request_id: Option<String>,
    runtime_task_id: Option<String>,
    cancel_token: Option<CancellationToken>,
    generation: u64,
}

impl Default for RuntimeControlState {
    fn default() -> Self {
        Self {
            session_id: None,
            session_events: None,
            session_context: SessionContextOverrides::default(),
            execution: None,
            conflict_policy: ExecutionConflictPolicy::Reject,
            history: None,
            request_id: None,
            runtime_task_id: None,
            cancel_token: None,
            generation: 0,
        }
    }
}

impl Default for RuntimeControl {
    fn default() -> Self {
        Self {
            state: StdRwLock::new(RuntimeControlState::default()),
            session_reset_request: Mutex::new(None),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct SessionContextOverrides {
    pub(crate) channel_id: Option<String>,
    pub(crate) inference: InferenceOverrideConfig,
}

#[derive(Debug, Clone)]
pub(crate) enum SessionResetRequest {
    Fresh(SessionContextOverrides),
    Resume {
        session_id: String,
        context: SessionContextOverrides,
    },
}

impl RuntimeControl {
    fn set_current_session(
        &self,
        session_id: Option<String>,
        event_tx: Option<SessionEventSender>,
        context: SessionContextOverrides,
        execution: Option<ExecutionStatusSnapshot>,
        conflict_policy: ExecutionConflictPolicy,
        history: Option<LiveSessionHistorySnapshot>,
    ) {
        let mut state = self.write_state();
        state.session_id = session_id;
        state.session_events = event_tx;
        state.session_context = context;
        state.execution = execution;
        state.conflict_policy = conflict_policy;
        state.history = history;
        state.generation = state.generation.wrapping_add(1);
    }

    #[cfg(test)]
    fn set_current_session_id(&self, session_id: Option<String>) {
        self.set_current_session(
            session_id,
            None,
            SessionContextOverrides::default(),
            None,
            ExecutionConflictPolicy::Reject,
            None,
        );
    }

    fn current_session_id(&self) -> Option<String> {
        self.snapshot().session_id
    }

    fn session_generation(&self) -> u64 {
        self.snapshot().generation
    }

    fn current_session_context(&self) -> SessionContextOverrides {
        self.snapshot().session_context
    }

    fn current_execution(&self) -> Option<ExecutionStatusSnapshot> {
        self.snapshot().execution
    }

    fn set_current_conflict_policy(&self, conflict_policy: ExecutionConflictPolicy) {
        self.write_state().conflict_policy = conflict_policy;
    }

    fn current_conflict_policy(&self) -> ExecutionConflictPolicy {
        self.snapshot().conflict_policy
    }

    fn set_current_execution_snapshot(&self, execution: ExecutionStatusSnapshot) {
        self.write_state().execution = Some(execution);
    }

    fn set_current_history_snapshot(&self, history: LiveSessionHistorySnapshot) {
        self.write_state().history = Some(history);
    }

    #[cfg(test)]
    fn set_current_execution_conflict_policy(&self, conflict_policy: ExecutionConflictPolicy) {
        self.set_current_conflict_policy(conflict_policy);
    }

    fn subscribe_current_session_events(&self) -> Option<SessionEventReceiver> {
        self.snapshot()
            .session_events
            .as_ref()
            .map(SessionEventSender::subscribe)
    }

    fn activate_task(
        &self,
        request_id: Option<String>,
        runtime_task_id: String,
        cancel_token: CancellationToken,
    ) {
        let mut state = self.write_state();
        state.request_id = request_id;
        state.runtime_task_id = Some(runtime_task_id);
        state.cancel_token = Some(cancel_token);
    }

    fn clear_active_task(&self) {
        let mut state = self.write_state();
        state.request_id = None;
        state.runtime_task_id = None;
        state.cancel_token = None;
    }

    fn current_request_id(&self) -> Option<String> {
        self.snapshot().request_id
    }

    fn current_runtime_task_id(&self) -> Option<String> {
        self.snapshot().runtime_task_id
    }

    fn request_task_cancel(&self) -> bool {
        let token = self.snapshot().cancel_token;
        if let Some(token) = token {
            token.cancel();
            true
        } else {
            false
        }
    }

    fn request_session_cancel(&self) -> bool {
        *self
            .session_reset_request
            .lock()
            .expect("runtime control session reset lock poisoned") =
            Some(SessionResetRequest::Fresh(self.current_session_context()));
        self.request_task_cancel()
    }

    fn request_session_resume(&self, session_id: String, context: SessionContextOverrides) {
        *self
            .session_reset_request
            .lock()
            .expect("runtime control session reset lock poisoned") =
            Some(SessionResetRequest::Resume {
                session_id,
                context,
            });
    }

    fn take_session_reset_request(&self) -> Option<SessionResetRequest> {
        self.session_reset_request
            .lock()
            .expect("runtime control session reset lock poisoned")
            .take()
    }

    pub(crate) fn snapshot(&self) -> RuntimeControlSnapshot {
        let state = self.read_state();
        RuntimeControlSnapshot {
            session_id: state.session_id.clone(),
            session_events: state.session_events.clone(),
            session_context: state.session_context.clone(),
            execution: state.execution.clone(),
            conflict_policy: state.conflict_policy,
            history: state.history.clone(),
            request_id: state.request_id.clone(),
            runtime_task_id: state.runtime_task_id.clone(),
            cancel_token: state.cancel_token.clone(),
            generation: state.generation,
        }
    }

    fn read_state(&self) -> std::sync::RwLockReadGuard<'_, RuntimeControlState> {
        self.state
            .read()
            .expect("runtime control state lock poisoned")
    }

    fn write_state(&self) -> std::sync::RwLockWriteGuard<'_, RuntimeControlState> {
        self.state
            .write()
            .expect("runtime control state lock poisoned")
    }
}

struct PeerAgentTaskEnvelope {
    task: QueuedTask,
    request_id: Option<String>,
    result_tx: Option<oneshot::Sender<PeerAgentTaskResult>>,
    delegated_capabilities: Option<BTreeMap<String, bool>>,
}

/// A handle to a running peer agent.
pub struct AgentRuntimeHandle {
    /// Explicit queued envelopes awaiting execution.
    queue: Arc<Mutex<VecDeque<PeerAgentTaskEnvelope>>>,
    /// Notification used to wake the background runtime when new work arrives.
    notify: Arc<Notify>,
    /// Shared execution/session control state for the runtime.
    control: Arc<RuntimeControl>,
    /// The background task running the agent's event loop.
    task: Option<JoinHandle<()>>,
    /// Approximate number of tasks currently queued in the runtime channel.
    queued_tasks: Arc<AtomicUsize>,
    /// Number of tasks currently executing inside the runtime loop.
    active_tasks: Arc<AtomicUsize>,
}

impl AgentRuntimeHandle {
    fn is_running(&self) -> bool {
        self.task
            .as_ref()
            .map(|jh| !jh.is_finished())
            .unwrap_or(false)
    }
}

#[derive(Clone)]
pub(crate) struct SharedPeerRuntimeContext {
    pub(crate) json: bool,
    pub(crate) tool_registry: ToolRegistry,
    pub(crate) policy_manager: Arc<RuntimePolicyManager>,
    pub(crate) governance_manager: Arc<GovernanceManager>,
    pub(crate) harness_manager: Arc<HarnessManager>,
    pub(crate) persistence_locks: Arc<SessionPersistenceCoordinator>,
}

#[derive(Clone, Default)]
pub(crate) struct SharedInferenceState {
    pub(crate) clients: HashMap<String, ProviderClient>,
    pub(crate) embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
}

#[derive(Default)]
struct CompletedTaskCache {
    order: VecDeque<String>,
    results: HashMap<String, PeerAgentTaskResult>,
}

impl CompletedTaskCache {
    const MAX_ENTRIES: usize = 1024;

    fn insert(&mut self, result: PeerAgentTaskResult) {
        let request_id = result.request_id.clone();
        if !self.results.contains_key(&request_id) {
            self.order.push_back(request_id.clone());
        }
        self.results.insert(request_id, result);
        while self.order.len() > Self::MAX_ENTRIES {
            if let Some(evicted) = self.order.pop_front() {
                self.results.remove(&evicted);
            }
        }
    }

    fn mark_promoted(&mut self, request_id: &str, branch: PromotedTaskBranch) {
        if let Some(result) = self.results.get_mut(request_id) {
            result.promoted_branch = Some(branch);
        }
    }
}

/// Orchestrates peer agents, spinning them up on demand and routing tasks to their independent runtimes.
pub struct AgentManager {
    /// The full configuration, used to look up agent profiles and instantiate kernels.
    config: Arc<TurinConfig>,
    /// Reference to the shared StoreManager for database operations.
    store_manager: Arc<StoreManager>,
    /// The list of active, running agent handles, keyed by runtime slot.
    runtimes: RwLock<HashMap<RuntimeSlotKey, Arc<AgentRuntimeHandle>>>,
    /// Task result receivers keyed by runtime request id (consumed by `await_result`).
    pending_results: RwLock<HashMap<String, oneshot::Receiver<PeerAgentTaskResult>>>,
    /// Mapping of request id -> current non-terminal task state for status accounting.
    pending_task_states: RwLock<HashMap<String, PendingTaskRecord>>,
    /// Bounded cache of completed task results for daemon/control-plane inspection.
    completed_results: RwLock<CompletedTaskCache>,
    /// Shared runtime pieces used to fork peer kernels without cloning the whole kernel topology.
    shared_runtime: OnceLock<SharedPeerRuntimeContext>,
    /// Live inference state copied from the root kernel after provider initialization.
    shared_inference: Mutex<SharedInferenceState>,
    /// Optional daemon-owned scheduler access propagated to peer runtimes.
    shared_scheduler: Mutex<Option<Arc<HarnessSchedulerAccess>>>,
}

impl AgentManager {
    /// Create a new AgentManager.
    pub fn new(config: Arc<TurinConfig>, store_manager: Arc<StoreManager>) -> Self {
        Self {
            config,
            store_manager,
            runtimes: RwLock::new(HashMap::new()),
            pending_results: RwLock::new(HashMap::new()),
            pending_task_states: RwLock::new(HashMap::new()),
            completed_results: RwLock::new(CompletedTaskCache::default()),
            shared_runtime: OnceLock::new(),
            shared_inference: Mutex::new(SharedInferenceState::default()),
            shared_scheduler: Mutex::new(None),
        }
    }

    pub(crate) fn bind_shared_runtime(&self, runtime: SharedPeerRuntimeContext) {
        let _ = self.shared_runtime.set(runtime);
    }

    pub(crate) fn shared_runtime(&self) -> Option<&SharedPeerRuntimeContext> {
        self.shared_runtime.get()
    }

    pub(crate) fn bind_inference_state(
        &self,
        clients: HashMap<String, ProviderClient>,
        embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    ) {
        *self
            .shared_inference
            .lock()
            .expect("agent manager shared inference mutex poisoned") = SharedInferenceState {
            clients,
            embedding_provider,
        };
    }

    pub(crate) fn bind_scheduler_access(&self, scheduler: Option<Arc<HarnessSchedulerAccess>>) {
        *self
            .shared_scheduler
            .lock()
            .expect("agent manager shared scheduler mutex poisoned") = scheduler;
    }

    pub(crate) fn shared_scheduler(&self) -> Option<Arc<HarnessSchedulerAccess>> {
        self.shared_scheduler
            .lock()
            .expect("agent manager shared scheduler mutex poisoned")
            .clone()
    }
}
