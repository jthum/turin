mod allocator;
mod caches;
mod cancellation;
mod operations;
mod peer_runtime;
mod records;
mod runtime_control;
mod runtime_registry;
mod task_status;
mod tasks;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, OnceLock, RwLock as StdRwLock};

use crate::harness::scheduler::HarnessSchedulerAccess;
use crate::inference::embeddings::EmbeddingProvider;
use crate::inference::provider::ProviderClient;
use crate::kernel::config::TurinConfig;
use crate::kernel::event::{KernelEvent, TaskBranchOutcome, TaskTerminalStatus};
use crate::kernel::execution_host::SessionPersistenceCoordinator;
use crate::kernel::governance::GovernanceManager;
use crate::kernel::harness_manager::HarnessManager;
use crate::kernel::policy::RuntimePolicyManager;
pub use crate::kernel::session::ExecutionStatusSnapshot;
use crate::kernel::session::{ExecutionConflictPolicy, SessionState};
pub use crate::kernel::task_promotion::{PromotedTaskBranch, TaskPromotionCandidate};
use crate::persistence::manager::StoreManager;
use crate::tools::registry::ToolRegistry;
use tokio::sync::{Mutex as AsyncMutex, RwLock, oneshot};
use turin_types::TaskInputContent;

use caches::{CompletedTaskCache, DelegationBudgetCache};
pub use records::AgentRuntimeHandle;
use records::{
    DelegationAdmission, LinkedSessionTarget, PeerAgentTaskEnvelope, PeerTaskSubmission,
    PendingTaskRecord, PendingTaskState, RuntimeSlotKey, TaskSessionTarget, task_prompt_preview,
};
pub(crate) use runtime_control::{
    RuntimeControl, RuntimeControlSnapshot, SessionContextOverrides, SessionResetRequest,
};

pub(crate) type SessionEventRecord = (Option<i64>, KernelEvent);
pub(crate) type SessionEventSender = tokio::sync::broadcast::Sender<SessionEventRecord>;
pub(crate) type SessionEventReceiver = tokio::sync::broadcast::Receiver<SessionEventRecord>;

#[derive(Debug, Clone, serde::Serialize)]
pub struct PeerAgentTaskResult {
    pub request_id: String,
    pub agent_id: String,
    pub slot_id: String,
    pub session_id: Option<String>,
    pub trace_id: String,
    pub title: Option<String>,
    pub prompt_preview: String,
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
pub struct InferenceContextStatusSnapshot {
    pub id: String,
    pub provider: String,
    pub model: String,
    pub is_default: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AgentStatusSnapshot {
    pub agent_id: String,
    pub provider: String,
    pub model: String,
    pub harness_id: String,
    pub inference_contexts: Vec<InferenceContextStatusSnapshot>,
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
    pub session_id: Option<String>,
    pub trace_id: String,
    pub title: Option<String>,
    pub prompt_preview: String,
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
    session_id: Option<String>,
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
    pub has_prior_history: bool,
}

impl LiveSessionHistorySnapshot {
    fn from_session(session: &SessionState) -> Self {
        Self {
            len: session.history.len(),
            has_prior_history: session.history.has_prior_history(),
        }
    }
}

/// Selects whether linked delegation reuses a logical child context or creates a new one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkedSessionMode {
    Thread(String),
    Fresh,
}

impl LinkedSessionMode {
    fn into_thread_key(self) -> String {
        match self {
            Self::Thread(key) => key,
            Self::Fresh => format!("fresh-{}", uuid::Uuid::now_v7().simple()),
        }
    }
}

#[derive(Clone)]
pub(crate) struct SharedPeerRuntimeContext {
    pub(crate) json: bool,
    pub(crate) tool_registry: ToolRegistry,
    pub(crate) policy_manager: Arc<RuntimePolicyManager>,
    pub(crate) governance_manager: Arc<GovernanceManager>,
    pub(crate) harness_manager: Arc<StdRwLock<Arc<HarnessManager>>>,
    pub(crate) persistence_locks: Arc<SessionPersistenceCoordinator>,
}

#[derive(Clone, Default)]
pub(crate) struct SharedInferenceState {
    pub(crate) clients: HashMap<String, ProviderClient>,
    pub(crate) embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
}

/// Orchestrates peer agents, spinning them up on demand and routing tasks to their independent runtimes.
pub struct AgentManager {
    /// The full configuration, used to look up agent profiles and instantiate kernels.
    config: StdRwLock<Arc<TurinConfig>>,
    /// Reference to the shared StoreManager for database operations.
    store_manager: Arc<StoreManager>,
    /// The list of active, running agent handles, keyed by runtime slot.
    runtimes: RwLock<HashMap<RuntimeSlotKey, Arc<AgentRuntimeHandle>>>,
    /// Per-agent gates serialize rare catalog replacement against task admission.
    catalog_gates: Mutex<HashMap<String, Arc<tokio::sync::RwLock<()>>>>,
    /// Keeps config and harness snapshots coherent while publishing a generation.
    catalog_snapshot_gate: StdRwLock<()>,
    /// Task result receivers keyed by runtime request id (consumed by `await_result`).
    pending_results: RwLock<HashMap<String, oneshot::Receiver<PeerAgentTaskResult>>>,
    /// Mapping of request id -> current non-terminal task state for status accounting.
    pending_task_states: RwLock<HashMap<String, PendingTaskRecord>>,
    /// Bounded cache of completed task results for daemon/control-plane inspection.
    completed_results: RwLock<CompletedTaskCache>,
    /// Serializes the rare completed-task promotion path so one task cannot fork twice.
    task_promotion: AsyncMutex<()>,
    delegation_budgets: Mutex<DelegationBudgetCache>,
    /// Shared runtime pieces used to fork peer kernels without cloning the whole kernel topology.
    shared_runtime: OnceLock<SharedPeerRuntimeContext>,
    /// Live inference state copied from the root kernel after provider initialization.
    shared_inference: Mutex<SharedInferenceState>,
    /// Optional daemon-owned scheduler access propagated to peer runtimes.
    shared_scheduler: Mutex<Option<Arc<HarnessSchedulerAccess>>>,
    /// Prevents new runtimes from being created after shutdown begins.
    shutting_down: AtomicBool,
}

impl AgentManager {
    /// Create a new AgentManager.
    pub fn new(config: Arc<TurinConfig>, store_manager: Arc<StoreManager>) -> Self {
        Self {
            config: StdRwLock::new(config),
            store_manager,
            runtimes: RwLock::new(HashMap::new()),
            catalog_gates: Mutex::new(HashMap::new()),
            catalog_snapshot_gate: StdRwLock::new(()),
            pending_results: RwLock::new(HashMap::new()),
            pending_task_states: RwLock::new(HashMap::new()),
            completed_results: RwLock::new(CompletedTaskCache::default()),
            task_promotion: AsyncMutex::new(()),
            delegation_budgets: Mutex::new(DelegationBudgetCache::default()),
            shared_runtime: OnceLock::new(),
            shared_inference: Mutex::new(SharedInferenceState::default()),
            shared_scheduler: Mutex::new(None),
            shutting_down: AtomicBool::new(false),
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

    pub(crate) fn config_snapshot(&self) -> Arc<TurinConfig> {
        self.config
            .read()
            .expect("agent manager config lock poisoned")
            .clone()
    }

    pub(crate) fn install_runtime_catalog(
        &self,
        config: Arc<TurinConfig>,
        harness_manager: Arc<HarnessManager>,
    ) {
        let _snapshot_guard = self
            .catalog_snapshot_gate
            .write()
            .expect("agent manager catalog snapshot lock poisoned");
        *self
            .config
            .write()
            .expect("agent manager config lock poisoned") = config;
        if let Some(shared) = self.shared_runtime() {
            *shared
                .harness_manager
                .write()
                .expect("agent manager harness catalog lock poisoned") = harness_manager;
        }
    }

    pub(crate) fn runtime_catalog_snapshot(&self) -> (Arc<TurinConfig>, Arc<HarnessManager>) {
        let _snapshot_guard = self
            .catalog_snapshot_gate
            .read()
            .expect("agent manager catalog snapshot lock poisoned");
        let config = self.config_snapshot();
        let harness_manager = self
            .shared_runtime()
            .expect("AgentManager shared runtime not bound")
            .harness_manager
            .read()
            .expect("agent manager harness catalog lock poisoned")
            .clone();
        (config, harness_manager)
    }

    fn catalog_gate(&self, agent_id: &str) -> Arc<tokio::sync::RwLock<()>> {
        let mut gates = self
            .catalog_gates
            .lock()
            .expect("agent manager catalog gates lock poisoned");
        Arc::clone(
            gates
                .entry(agent_id.to_string())
                .or_insert_with(|| Arc::new(tokio::sync::RwLock::new(()))),
        )
    }
}
