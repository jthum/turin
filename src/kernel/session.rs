use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, broadcast, mpsc};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::inference::provider::InferenceMessage;
use crate::kernel::config::InferenceOverrideConfig;
use crate::kernel::event::KernelEvent;
use crate::kernel::harness_runtime::HarnessInstance;
use crate::kernel::identity::RuntimeIdentity;
use crate::persistence::manager::StoreSelector;
use turin_types::{TaskInputContent, ToolsConfig};

pub type SessionHarnessEngine = Arc<std::sync::Mutex<HarnessInstance>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionVisibility {
    Visible,
    Hidden,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionDurability {
    Durable,
    Ephemeral,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionWritePolicy {
    AdvanceBranchHead,
    Detached,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecutionContextTarget {
    BranchHead { branch_head_id: Option<i64> },
    TurnId { turn_id: i64 },
    SelectedPath { turn_ids: Vec<i64> },
    ExternalReference { reference: String },
    SummarySource { source_turn_id: i64 },
}

impl ExecutionContextTarget {
    pub fn branch_head_id(&self) -> Option<i64> {
        match self {
            Self::BranchHead { branch_head_id } => *branch_head_id,
            _ => None,
        }
    }

    pub fn turn_id(&self) -> Option<i64> {
        match self {
            Self::TurnId { turn_id }
            | Self::SummarySource {
                source_turn_id: turn_id,
            } => Some(*turn_id),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ExecutionContext {
    pub execution_id: String,
    pub context_target: ExecutionContextTarget,
    pub visibility: ExecutionVisibility,
    pub durability: ExecutionDurability,
    pub write_policy: ExecutionWritePolicy,
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionContext {
    pub fn new() -> Self {
        Self {
            execution_id: new_execution_id(),
            context_target: ExecutionContextTarget::BranchHead {
                branch_head_id: None,
            },
            visibility: ExecutionVisibility::Visible,
            durability: ExecutionDurability::Durable,
            write_policy: ExecutionWritePolicy::AdvanceBranchHead,
        }
    }
}

/// One queued unit of work to be executed by the kernel.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QueuedTask {
    pub task_id: String,
    pub plan_id: Option<String>,
    pub title: Option<String>,
    pub prompt: String,
    #[serde(default)]
    pub content: Option<Vec<TaskInputContent>>,
    #[serde(default)]
    pub tools: Option<ToolsConfig>,
    #[serde(default = "new_trace_id")]
    pub trace_id: String,
}

#[derive(Debug, Clone)]
pub struct PersistedKernelEvent {
    pub internal_id: Option<i64>,
    pub branch_head_id: Option<i64>,
    pub turn_index: Option<u32>,
    pub event: KernelEvent,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ContextCompactionCheckpoint {
    pub summary: String,
    pub covered_message_count: usize,
    pub generated_at_turn_index: u32,
    pub provider_name: String,
    pub model: String,
}

pub enum PersistedKernelRecord {
    Event(Box<PersistedKernelEvent>),
    Barrier(tokio::sync::oneshot::Sender<()>),
}

impl QueuedTask {
    pub fn ad_hoc(prompt: impl Into<String>) -> Self {
        Self {
            task_id: String::new(), // Assigned by SessionState
            plan_id: None,
            title: None,
            prompt: prompt.into(),
            content: None,
            tools: None,
            trace_id: new_trace_id(),
        }
    }

    pub fn with_plan(
        prompt: impl Into<String>,
        plan_id: impl Into<String>,
        title: Option<String>,
    ) -> Self {
        Self {
            task_id: String::new(), // Assigned by SessionState
            plan_id: Some(plan_id.into()),
            title,
            prompt: prompt.into(),
            content: None,
            tools: None,
            trace_id: new_trace_id(),
        }
    }

    pub fn with_inherited_trace(mut self, trace_id: Option<&str>) -> Self {
        if let Some(trace_id) = trace_id
            && !trace_id.is_empty()
        {
            self.trace_id = trace_id.to_string();
        }
        self
    }
}

/// Lightweight in-memory progress tracker for a plan.
#[derive(Debug, Clone, serde::Serialize)]
pub struct PlanProgress {
    pub plan_id: String,
    pub title: String,
    pub total_tasks: usize,
    pub completed_tasks: usize,
}

impl PlanProgress {
    pub fn pending_tasks(&self) -> usize {
        self.total_tasks.saturating_sub(self.completed_tasks)
    }

    pub fn is_complete(&self) -> bool {
        self.completed_tasks >= self.total_tasks
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    Inactive,
    Active,
}

#[derive(Debug, Clone)]
pub struct ToolRateLimitState {
    window_started_at: Instant,
    call_count_in_window: usize,
}

impl ToolRateLimitState {
    fn new() -> Self {
        Self {
            window_started_at: Instant::now(),
            call_count_in_window: 0,
        }
    }

    fn reserve(&mut self, requested: usize, max_calls: usize, window: Duration) -> usize {
        if self.window_started_at.elapsed() >= window {
            self.window_started_at = Instant::now();
            self.call_count_in_window = 0;
        }

        let granted = max_calls
            .saturating_sub(self.call_count_in_window)
            .min(requested);
        self.call_count_in_window += granted;
        granted
    }
}

/// Holds the state of an active agent session.
pub struct SessionState {
    pub identity: RuntimeIdentity,
    pub internal_id: Option<i64>,
    pub runtime_slot_id: Option<String>,
    pub store_selector: StoreSelector,
    pub default_store_selector: Option<StoreSelector>,
    pub inference: InferenceOverrideConfig,
    pub context_checkpoint: Option<ContextCompactionCheckpoint>,
    pub history: Vec<InferenceMessage>,
    pub execution: ExecutionContext,
    pub harness_engine: Option<SessionHarnessEngine>,
    pub harness_generation: u64,
    pub queue: Arc<Mutex<VecDeque<QueuedTask>>>,
    pub plans: HashMap<String, PlanProgress>,
    pub turn_index: u32,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    // Event channel for this session
    pub event_tx: broadcast::Sender<(Option<i64>, KernelEvent)>,
    /// Reliable durability lane (separate from observer fanout).
    pub durability_tx: Option<mpsc::UnboundedSender<PersistedKernelRecord>>,
    /// Serializes branch-scoped persistence so turn creation stays consistent.
    pub persistence_lock: Arc<Mutex<()>>,
    pub event_task: Option<Arc<Mutex<Option<JoinHandle<()>>>>>,
    /// Token to cooperatively cancel the currently running task/turn.
    pub cancel_token: CancellationToken,
    // Internal counters for task scheduling
    pub next_task_id: u32,
    pub next_plan_id: u32,
    pub status: SessionStatus,
    pub mode: crate::kernel::config::AgentMode,
    pub stop_requested: bool,
    pub restored_from_persistence: bool,
    pub tool_rate_limit: ToolRateLimitState,
}

impl Default for SessionState {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionState {
    pub fn new() -> Self {
        let session_id = uuid::Uuid::now_v7().simple().to_string();
        let (tx, _rx) = broadcast::channel(1024);
        Self {
            identity: RuntimeIdentity::new(session_id, "default"),
            internal_id: None,
            runtime_slot_id: None,
            store_selector: StoreSelector::Alias("state".to_string()),
            default_store_selector: None,
            inference: InferenceOverrideConfig::default(),
            context_checkpoint: None,
            history: Vec::new(),
            execution: ExecutionContext::new(),
            harness_engine: None,
            harness_generation: 0,
            queue: Arc::new(Mutex::new(VecDeque::new())),
            plans: HashMap::new(),
            turn_index: 0,
            total_input_tokens: 0,
            total_output_tokens: 0,
            event_tx: tx,
            durability_tx: None,
            persistence_lock: Arc::new(Mutex::new(())),
            event_task: Some(Arc::new(Mutex::new(None))),
            cancel_token: CancellationToken::new(),
            next_task_id: 1,
            next_plan_id: 1,
            status: SessionStatus::Inactive,
            mode: crate::kernel::config::AgentMode::Auto,
            stop_requested: false,
            restored_from_persistence: false,
            tool_rate_limit: ToolRateLimitState::new(),
        }
    }

    pub fn reserve_tool_calls(
        &mut self,
        requested: usize,
        max_calls: usize,
        window: Duration,
    ) -> usize {
        self.tool_rate_limit.reserve(requested, max_calls, window)
    }

    pub fn execution_id(&self) -> &str {
        &self.execution.execution_id
    }

    pub fn context_target(&self) -> &ExecutionContextTarget {
        &self.execution.context_target
    }

    pub fn set_context_target(&mut self, context_target: ExecutionContextTarget) {
        self.execution.context_target = context_target;
    }

    pub fn selected_branch_head_id(&self) -> Option<i64> {
        self.execution.context_target.branch_head_id()
    }

    pub fn set_selected_branch_head_id(&mut self, branch_head_id: Option<i64>) {
        self.set_context_target(ExecutionContextTarget::BranchHead { branch_head_id });
    }

    pub fn selected_turn_id(&self) -> Option<i64> {
        self.execution.context_target.turn_id()
    }

    pub fn set_selected_turn_id(&mut self, turn_id: i64) {
        self.set_context_target(ExecutionContextTarget::TurnId { turn_id });
    }
}

fn new_trace_id() -> String {
    format!("tr_{}", uuid::Uuid::now_v7().simple())
}

fn new_execution_id() -> String {
    format!("ex_{}", uuid::Uuid::now_v7().simple())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_rate_limit_caps_reserved_calls_within_window() {
        let mut session = SessionState::new();
        assert_eq!(session.reserve_tool_calls(3, 4, Duration::from_secs(60)), 3);
        assert_eq!(session.reserve_tool_calls(3, 4, Duration::from_secs(60)), 1);
        assert_eq!(session.reserve_tool_calls(1, 4, Duration::from_secs(60)), 0);
    }

    #[test]
    fn tool_rate_limit_resets_after_window_elapses() {
        let mut session = SessionState::new();
        assert_eq!(
            session.reserve_tool_calls(2, 2, Duration::from_millis(1)),
            2
        );
        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(
            session.reserve_tool_calls(2, 2, Duration::from_millis(1)),
            2
        );
    }

    #[test]
    fn session_defaults_to_visible_durable_execution_context() {
        let session = SessionState::new();
        assert!(session.execution_id().starts_with("ex_"));
        assert_eq!(session.selected_branch_head_id(), None);
        assert_eq!(
            session.context_target(),
            &ExecutionContextTarget::BranchHead {
                branch_head_id: None
            }
        );
        assert_eq!(session.execution.visibility, ExecutionVisibility::Visible);
        assert_eq!(session.execution.durability, ExecutionDurability::Durable);
        assert_eq!(
            session.execution.write_policy,
            ExecutionWritePolicy::AdvanceBranchHead
        );
    }
}
