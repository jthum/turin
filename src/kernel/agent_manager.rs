mod cancellation;
mod operations;
mod peer_runtime;
mod runtime_registry;
#[cfg(test)]
mod tests;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize};
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
use crate::kernel::session::{ExecutionConflictPolicy, QueuedTask};
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
    current_session_id: StdRwLock<Option<String>>,
    current_session_events: StdRwLock<Option<SessionEventSender>>,
    current_session_context: StdRwLock<SessionContextOverrides>,
    current_execution: StdRwLock<Option<ExecutionStatusSnapshot>>,
    current_conflict_policy: StdRwLock<ExecutionConflictPolicy>,
    current_request_id: StdRwLock<Option<String>>,
    current_runtime_task_id: StdRwLock<Option<String>>,
    current_cancel_token: Mutex<Option<CancellationToken>>,
    session_reset_request: Mutex<Option<SessionResetRequest>>,
    session_generation: AtomicU64,
}

impl Default for RuntimeControl {
    fn default() -> Self {
        Self {
            current_session_id: StdRwLock::new(None),
            current_session_events: StdRwLock::new(None),
            current_session_context: StdRwLock::new(SessionContextOverrides::default()),
            current_execution: StdRwLock::new(None),
            current_conflict_policy: StdRwLock::new(ExecutionConflictPolicy::Reject),
            current_request_id: StdRwLock::new(None),
            current_runtime_task_id: StdRwLock::new(None),
            current_cancel_token: Mutex::new(None),
            session_reset_request: Mutex::new(None),
            session_generation: AtomicU64::new(0),
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
    ) {
        *self
            .current_session_id
            .write()
            .expect("runtime control session lock poisoned") = session_id;
        *self
            .current_session_events
            .write()
            .expect("runtime control session events lock poisoned") = event_tx;
        *self
            .current_session_context
            .write()
            .expect("runtime control session context lock poisoned") = context;
        *self
            .current_execution
            .write()
            .expect("runtime control execution snapshot lock poisoned") = execution;
        *self
            .current_conflict_policy
            .write()
            .expect("runtime control conflict policy lock poisoned") = conflict_policy;
        self.session_generation
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    #[cfg(test)]
    fn set_current_session_id(&self, session_id: Option<String>) {
        self.set_current_session(
            session_id,
            None,
            SessionContextOverrides::default(),
            None,
            ExecutionConflictPolicy::Reject,
        );
    }

    fn current_session_id(&self) -> Option<String> {
        self.current_session_id
            .read()
            .expect("runtime control session lock poisoned")
            .clone()
    }

    fn session_generation(&self) -> u64 {
        self.session_generation
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    fn current_session_context(&self) -> SessionContextOverrides {
        self.current_session_context
            .read()
            .expect("runtime control session context lock poisoned")
            .clone()
    }

    fn current_execution(&self) -> Option<ExecutionStatusSnapshot> {
        self.current_execution
            .read()
            .expect("runtime control execution snapshot lock poisoned")
            .clone()
    }

    fn set_current_conflict_policy(&self, conflict_policy: ExecutionConflictPolicy) {
        *self
            .current_conflict_policy
            .write()
            .expect("runtime control conflict policy lock poisoned") = conflict_policy;
    }

    fn current_conflict_policy(&self) -> ExecutionConflictPolicy {
        *self
            .current_conflict_policy
            .read()
            .expect("runtime control conflict policy lock poisoned")
    }

    fn set_current_execution_snapshot(&self, execution: ExecutionStatusSnapshot) {
        *self
            .current_execution
            .write()
            .expect("runtime control execution snapshot lock poisoned") = Some(execution);
    }

    #[cfg(test)]
    fn set_current_execution_conflict_policy(&self, conflict_policy: ExecutionConflictPolicy) {
        self.set_current_conflict_policy(conflict_policy);
    }

    fn subscribe_current_session_events(&self) -> Option<SessionEventReceiver> {
        self.current_session_events
            .read()
            .expect("runtime control session events lock poisoned")
            .as_ref()
            .map(SessionEventSender::subscribe)
    }

    fn activate_task(
        &self,
        request_id: Option<String>,
        runtime_task_id: String,
        cancel_token: CancellationToken,
    ) {
        *self
            .current_request_id
            .write()
            .expect("runtime control request lock poisoned") = request_id;
        *self
            .current_runtime_task_id
            .write()
            .expect("runtime control task id lock poisoned") = Some(runtime_task_id);
        *self
            .current_cancel_token
            .lock()
            .expect("runtime control cancel lock poisoned") = Some(cancel_token);
    }

    fn clear_active_task(&self) {
        *self
            .current_request_id
            .write()
            .expect("runtime control request lock poisoned") = None;
        *self
            .current_runtime_task_id
            .write()
            .expect("runtime control task id lock poisoned") = None;
        *self
            .current_cancel_token
            .lock()
            .expect("runtime control cancel lock poisoned") = None;
    }

    fn current_request_id(&self) -> Option<String> {
        self.current_request_id
            .read()
            .expect("runtime control request lock poisoned")
            .clone()
    }

    fn current_runtime_task_id(&self) -> Option<String> {
        self.current_runtime_task_id
            .read()
            .expect("runtime control task id lock poisoned")
            .clone()
    }

    fn request_task_cancel(&self) -> bool {
        let token = self
            .current_cancel_token
            .lock()
            .expect("runtime control cancel lock poisoned")
            .clone();
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
