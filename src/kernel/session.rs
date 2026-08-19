use std::collections::{HashMap, VecDeque};
use std::ops::{Deref, Index};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::{Mutex, RwLock, broadcast, mpsc};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::inference::provider::InferenceMessage;
use crate::kernel::config::InferenceOverrideConfig;
use crate::kernel::event::KernelEvent;
use crate::kernel::event::TaskBranchOutcome;
use crate::kernel::harness_runtime::HarnessInstance;
use crate::kernel::identity::RuntimeIdentity;
use crate::persistence::manager::StoreSelector;
use crate::persistence::state::TurnWriteTarget;

mod completed_tasks;
mod queued_tasks;

pub use completed_tasks::{
    CompletedLocalTaskResults, CompletedLocalTaskResultsHandle, LocalTaskResult,
};
pub use queued_tasks::QueuedTask;

pub type SessionHarnessEngine = Arc<std::sync::Mutex<HarnessInstance>>;

/// Transient per-task execution state that is reset when a task completes.
#[derive(Debug, Default)]
pub struct ActiveTaskState {
    pub conflict_policy: Option<ExecutionConflictPolicy>,
    pub branch_outcome: Option<TaskBranchOutcome>,
    pub conflict_detached: bool,
    pub turn_target: Option<TurnWriteTarget>,
    budget: Option<ActiveTaskBudget>,
    execution_restore: Option<TaskExecutionRestoreState>,
}

#[derive(Debug, Clone)]
struct ActiveTaskBudget {
    started_at_unix_ms: u64,
    started_at: Instant,
    input_tokens_at_start: u64,
    output_tokens_at_start: u64,
}

#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct TaskBudgetSnapshot {
    pub task_started_at_unix_ms: Option<u64>,
    pub task_elapsed_ms: u64,
    pub task_input_tokens: u64,
    pub task_output_tokens: u64,
    pub task_total_tokens: u64,
    pub task_turn_count: u32,
}

impl ActiveTaskBudget {
    fn start(input_tokens_at_start: u64, output_tokens_at_start: u64) -> Self {
        Self {
            started_at_unix_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            started_at: Instant::now(),
            input_tokens_at_start,
            output_tokens_at_start,
        }
    }
}

#[derive(Debug, Clone)]
struct TaskExecutionRestoreState {
    execution: ExecutionContext,
    selected_branch_head_cursor: Option<BranchHeadCursor>,
}

/// Tracks the persisted turn currently at the tip of the selected branch head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BranchHeadCursor {
    pub turn_id: i64,
    pub turn_index: u32,
}

/// Controls whether an execution is visible to normal session consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionVisibility {
    /// Show this execution as part of the normal visible session flow.
    Visible,
    /// Keep this execution hidden from the default visible session flow.
    Hidden,
}

/// Controls whether an execution is intended to persist beyond runtime memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionDurability {
    /// Persist durable state for this execution when its write policy allows it.
    Durable,
    /// Treat this execution as ephemeral runtime state.
    Ephemeral,
}

/// Controls how an execution writes new turns into persisted session state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionWritePolicy {
    /// Advance the selected branch head by allocating and writing a new turn.
    AdvanceBranchHead,
    /// Avoid creating new durable turns for this execution.
    Detached,
}

/// Determines how the kernel resolves stale branch-head writes during execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionConflictPolicy {
    /// Terminate the task with `Conflict` status.
    Reject,
    /// Continue the task without durable writes after a stale branch-head conflict.
    Detached,
    /// Create a sibling branch from the expected source turn and continue durably there.
    ForkSibling,
}

/// Controls how a sidestep execution branches away from the current path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SidestepMode {
    /// Explore a point-in-time snapshot without creating durable turns.
    Ephemeral,
    /// Create a durable sibling branch before running the sidestep.
    ForkSibling,
}

/// Selects which persisted context path is materialized for an execution.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecutionContextTarget {
    /// Follow a session branch head, optionally overriding the persisted active head.
    BranchHead { branch_head_id: Option<i64> },
    /// Materialize history up to a specific turn.
    TurnId { turn_id: i64 },
    /// Materialize an explicit ordered set of turns as the execution context.
    SelectedPath { turn_ids: Vec<i64> },
    /// Materialize context from another persisted session reference.
    ExternalReference { reference: String },
    /// Materialize a summary source turn without treating it as a writable branch target.
    SummarySource { source_turn_id: i64 },
}

impl ExecutionContextTarget {
    pub fn default_write_policy(&self) -> ExecutionWritePolicy {
        match self {
            Self::BranchHead { .. } => ExecutionWritePolicy::AdvanceBranchHead,
            Self::TurnId { .. }
            | Self::SelectedPath { .. }
            | Self::ExternalReference { .. }
            | Self::SummarySource { .. } => ExecutionWritePolicy::Detached,
        }
    }

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

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExecutionContext {
    /// Stable identifier for this live execution instance.
    pub execution_id: String,
    /// Which persisted context path this execution is currently using.
    pub context_target: ExecutionContextTarget,
    /// Whether the execution is part of the default visible session flow.
    pub visibility: ExecutionVisibility,
    /// Whether the execution should persist durable state.
    pub durability: ExecutionDurability,
    /// How the execution writes turns when it is allowed to persist.
    pub write_policy: ExecutionWritePolicy,
    /// How the execution responds when its intended branch write becomes stale.
    pub conflict_policy: ExecutionConflictPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ExecutionStatusSnapshot {
    pub execution_id: String,
    pub context_target: ExecutionContextTarget,
    pub visibility: ExecutionVisibility,
    pub durability: ExecutionDurability,
    pub write_policy: ExecutionWritePolicy,
}

impl ExecutionStatusSnapshot {
    pub fn from_execution(
        execution: &ExecutionContext,
        effective_write_policy: ExecutionWritePolicy,
    ) -> Self {
        Self {
            execution_id: execution.execution_id.clone(),
            context_target: execution.context_target.clone(),
            visibility: execution.visibility,
            durability: execution.durability,
            write_policy: effective_write_policy,
        }
    }
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
            conflict_policy: ExecutionConflictPolicy::Reject,
        }
    }
}

/// Optional per-task execution overrides layered on top of the live session execution.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TaskExecutionOverrides {
    #[serde(default)]
    pub context_target: Option<ExecutionContextTarget>,
    #[serde(default)]
    pub visibility: Option<ExecutionVisibility>,
    #[serde(default)]
    pub durability: Option<ExecutionDurability>,
    #[serde(default)]
    pub write_policy: Option<ExecutionWritePolicy>,
}

impl TaskExecutionOverrides {
    pub fn is_empty(&self) -> bool {
        self.context_target.is_none()
            && self.visibility.is_none()
            && self.durability.is_none()
            && self.write_policy.is_none()
    }

    pub fn apply_to_execution(&self, execution: &mut ExecutionContext) -> Result<(), String> {
        if let Some(context_target) = &self.context_target {
            execution.context_target = context_target.clone();
            execution.write_policy = context_target.default_write_policy();
        }
        if let Some(visibility) = self.visibility {
            execution.visibility = visibility;
        }
        if let Some(durability) = self.durability {
            execution.durability = durability;
        }
        if let Some(write_policy) = self.write_policy {
            execution.write_policy = write_policy;
        }

        if execution.write_policy == ExecutionWritePolicy::AdvanceBranchHead
            && !matches!(
                execution.context_target,
                ExecutionContextTarget::BranchHead { .. }
            )
        {
            return Err(
                "advance_branch_head write policy requires a branch_head execution target"
                    .to_string(),
            );
        }

        Ok(())
    }
}

/// Precomputed execution policy for a queued sidestep task.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedSidestepExecution {
    pub execution: TaskExecutionOverrides,
    pub conflict_policy: ExecutionConflictPolicy,
    pub branch_outcome: Option<TaskBranchOutcome>,
}

impl ExecutionConflictPolicy {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Reject => "reject",
            Self::Detached => "detached",
            Self::ForkSibling => "fork_sibling",
        }
    }
}

impl std::str::FromStr for ExecutionConflictPolicy {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "reject" => Ok(Self::Reject),
            "detached" => Ok(Self::Detached),
            "fork_sibling" => Ok(Self::ForkSibling),
            other => Err(format!(
                "invalid conflict policy '{other}'; expected reject|detached|fork_sibling"
            )),
        }
    }
}

impl SidestepMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ephemeral => "ephemeral",
            Self::ForkSibling => "fork_sibling",
        }
    }
}

impl std::str::FromStr for SidestepMode {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "ephemeral" => Ok(Self::Ephemeral),
            "fork_sibling" => Ok(Self::ForkSibling),
            other => Err(format!(
                "invalid sidestep mode '{other}'; expected ephemeral|fork_sibling"
            )),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PersistedKernelEvent {
    pub internal_id: Option<i64>,
    pub turn_target: Option<TurnWriteTarget>,
    pub event: KernelEvent,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct ContextCompactionCheckpoint {
    pub summary: String,
    pub covered_through_turn_id: i64,
    pub covered_through_turn_index: u32,
    pub generated_at_turn_index: u32,
    pub provider_name: String,
    pub model: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistoryOrigin {
    pub turn_id: i64,
    pub turn_index: u32,
}

#[derive(Debug, Clone, Default)]
pub struct ResidentHistory {
    messages: Vec<InferenceMessage>,
    origins: Vec<Option<HistoryOrigin>>,
    has_prior_history: bool,
}

impl ResidentHistory {
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    pub fn messages(&self) -> &[InferenceMessage] {
        &self.messages
    }

    pub fn messages_mut(&mut self) -> &mut [InferenceMessage] {
        &mut self.messages
    }

    pub fn iter(&self) -> std::slice::Iter<'_, InferenceMessage> {
        self.messages.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, InferenceMessage> {
        self.messages.iter_mut()
    }

    pub fn to_messages(&self) -> Vec<InferenceMessage> {
        self.messages.clone()
    }

    pub fn into_messages(self) -> Vec<InferenceMessage> {
        self.messages
    }

    pub fn push(&mut self, message: InferenceMessage) {
        self.push_with_origin(message, None);
    }

    pub fn push_with_origin(&mut self, message: InferenceMessage, origin: Option<HistoryOrigin>) {
        self.messages.push(message);
        self.origins.push(origin);
    }

    pub fn replace(
        &mut self,
        entries: Vec<(InferenceMessage, Option<HistoryOrigin>)>,
        has_prior_history: bool,
    ) {
        let (messages, origins) = entries.into_iter().unzip();
        self.messages = messages;
        self.origins = origins;
        self.has_prior_history = has_prior_history;
    }

    pub fn replace_untracked(&mut self, messages: Vec<InferenceMessage>, has_prior_history: bool) {
        self.origins = vec![None; messages.len()];
        self.messages = messages;
        self.has_prior_history = has_prior_history;
    }

    pub fn drain_prefix(&mut self, count: usize) {
        self.messages.drain(0..count);
        self.origins.drain(0..count);
        self.messages.shrink_to_fit();
        self.origins.shrink_to_fit();
        self.has_prior_history = true;
    }

    pub fn clear(&mut self) {
        self.messages.clear();
        self.origins.clear();
        self.has_prior_history = false;
    }

    pub fn has_prior_history(&self) -> bool {
        self.has_prior_history
    }

    pub fn origin(&self, index: usize) -> Option<HistoryOrigin> {
        self.origins.get(index).copied().flatten()
    }

    pub fn untracked_suffix(&self) -> &[InferenceMessage] {
        let start = self
            .origins
            .iter()
            .rposition(Option::is_some)
            .map_or(0, |index| index + 1);
        &self.messages[start..]
    }

    pub fn suffix_after_turn(&self, turn_id: i64, turn_index: u32) -> &[InferenceMessage] {
        if let Some(index) = self.index_after_turn(turn_id) {
            return &self.messages[index..];
        }
        let starts_after_checkpoint = self
            .origins
            .iter()
            .flatten()
            .all(|origin| origin.turn_index > turn_index);
        if starts_after_checkpoint {
            &self.messages
        } else {
            &[]
        }
    }

    pub fn into_suffix_after_turn(
        mut self,
        turn_id: i64,
        turn_index: u32,
    ) -> Vec<InferenceMessage> {
        if let Some(index) = self.index_after_turn(turn_id) {
            self.messages.drain(..index);
            return self.messages;
        }
        if self
            .origins
            .iter()
            .flatten()
            .all(|origin| origin.turn_index > turn_index)
        {
            self.messages
        } else {
            Vec::new()
        }
    }

    pub fn index_after_turn(&self, turn_id: i64) -> Option<usize> {
        self.origins
            .iter()
            .rposition(|origin| origin.is_some_and(|origin| origin.turn_id == turn_id))
            .map(|index| index + 1)
    }
}

impl Index<usize> for ResidentHistory {
    type Output = InferenceMessage;

    fn index(&self, index: usize) -> &Self::Output {
        &self.messages[index]
    }
}

impl Deref for ResidentHistory {
    type Target = [InferenceMessage];

    fn deref(&self) -> &Self::Target {
        &self.messages
    }
}

impl<'a> IntoIterator for &'a ResidentHistory {
    type Item = &'a InferenceMessage;
    type IntoIter = std::slice::Iter<'a, InferenceMessage>;

    fn into_iter(self) -> Self::IntoIter {
        self.messages.iter()
    }
}

impl<'a> IntoIterator for &'a mut ResidentHistory {
    type Item = &'a mut InferenceMessage;
    type IntoIter = std::slice::IterMut<'a, InferenceMessage>;

    fn into_iter(self) -> Self::IntoIter {
        self.messages.iter_mut()
    }
}

pub enum PersistedKernelRecord {
    Event(Box<PersistedKernelEvent>),
    Barrier(tokio::sync::oneshot::Sender<()>),
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
    pub history: ResidentHistory,
    pub execution: ExecutionContext,
    pub active_task: ActiveTaskState,
    pub selected_branch_head_cursor: Option<BranchHeadCursor>,
    pub harness_engine: Option<SessionHarnessEngine>,
    pub harness_generation: u64,
    pub queue: Arc<Mutex<VecDeque<QueuedTask>>>,
    pub completed_task_results: CompletedLocalTaskResultsHandle,
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
            history: ResidentHistory::default(),
            execution: ExecutionContext::new(),
            active_task: ActiveTaskState::default(),
            selected_branch_head_cursor: None,
            harness_engine: None,
            harness_generation: 0,
            queue: Arc::new(Mutex::new(VecDeque::new())),
            completed_task_results: Arc::new(RwLock::new(CompletedLocalTaskResults::default())),
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

    pub fn begin_active_task_budget(&mut self) {
        self.active_task.budget = Some(ActiveTaskBudget::start(
            self.total_input_tokens,
            self.total_output_tokens,
        ));
    }

    pub fn active_task_budget_snapshot(&self, task_turn_count: u32) -> TaskBudgetSnapshot {
        let Some(budget) = self.active_task.budget.as_ref() else {
            return TaskBudgetSnapshot {
                task_turn_count,
                ..TaskBudgetSnapshot::default()
            };
        };
        let task_input_tokens = self
            .total_input_tokens
            .saturating_sub(budget.input_tokens_at_start);
        let task_output_tokens = self
            .total_output_tokens
            .saturating_sub(budget.output_tokens_at_start);
        TaskBudgetSnapshot {
            task_started_at_unix_ms: Some(budget.started_at_unix_ms),
            task_elapsed_ms: budget.started_at.elapsed().as_millis() as u64,
            task_input_tokens,
            task_output_tokens,
            task_total_tokens: task_input_tokens + task_output_tokens,
            task_turn_count,
        }
    }

    pub fn history_is_pruned(&self) -> bool {
        self.history.has_prior_history()
    }

    pub fn replace_full_history(&mut self, history: Vec<InferenceMessage>) {
        self.history.replace_untracked(history, false);
    }

    pub fn execution_id(&self) -> &str {
        &self.execution.execution_id
    }

    pub fn context_target(&self) -> &ExecutionContextTarget {
        &self.execution.context_target
    }

    pub fn set_context_target(&mut self, context_target: ExecutionContextTarget) {
        self.execution.write_policy = context_target.default_write_policy();
        self.execution.context_target = context_target;
        if self.execution.write_policy != ExecutionWritePolicy::AdvanceBranchHead {
            self.selected_branch_head_cursor = None;
        }
        self.active_task.conflict_detached = false;
        self.active_task.turn_target = None;
    }

    pub fn replace_context_target_preserving_policy(
        &mut self,
        context_target: ExecutionContextTarget,
    ) {
        self.execution.context_target = context_target;
        if self.execution.write_policy != ExecutionWritePolicy::AdvanceBranchHead {
            self.selected_branch_head_cursor = None;
        }
        self.active_task.conflict_detached = false;
        self.active_task.turn_target = None;
    }

    pub fn selected_branch_head_id(&self) -> Option<i64> {
        self.execution.context_target.branch_head_id()
    }

    pub fn set_selected_branch_head_id(&mut self, branch_head_id: Option<i64>) {
        self.set_context_target(ExecutionContextTarget::BranchHead { branch_head_id });
    }

    pub fn selected_branch_head_turn_id(&self) -> Option<i64> {
        self.selected_branch_head_cursor
            .map(|cursor| cursor.turn_id)
    }

    pub fn set_selected_branch_head_turn_id(&mut self, turn_id: Option<i64>) {
        self.set_selected_branch_head_cursor(turn_id, self.selected_branch_head_turn_index());
    }

    pub fn selected_branch_head_turn_index(&self) -> Option<u32> {
        self.selected_branch_head_cursor
            .map(|cursor| cursor.turn_index)
    }

    pub fn set_selected_branch_head_turn_index(&mut self, turn_index: Option<u32>) {
        self.set_selected_branch_head_cursor(self.selected_branch_head_turn_id(), turn_index);
    }

    pub fn set_selected_branch_head_cursor(
        &mut self,
        turn_id: Option<i64>,
        turn_index: Option<u32>,
    ) {
        self.selected_branch_head_cursor = match (turn_id, turn_index) {
            (Some(turn_id), Some(turn_index)) => Some(BranchHeadCursor {
                turn_id,
                turn_index,
            }),
            _ => None,
        };
    }

    pub fn selected_turn_id(&self) -> Option<i64> {
        self.execution.context_target.turn_id()
    }

    pub fn set_selected_turn_id(&mut self, turn_id: i64) {
        self.set_context_target(ExecutionContextTarget::TurnId { turn_id });
    }

    pub fn next_turn_write_target_request(&self) -> Option<TurnWriteTarget> {
        match self.effective_write_policy() {
            ExecutionWritePolicy::AdvanceBranchHead => {
                let next_turn_index = self
                    .selected_branch_head_cursor
                    .map_or(0, |cursor| cursor.turn_index + 1);
                Some(TurnWriteTarget::branch_head_with_expectation(
                    self.selected_branch_head_id(),
                    self.selected_branch_head_cursor
                        .map(|cursor| cursor.turn_id),
                    next_turn_index,
                ))
            }
            ExecutionWritePolicy::Detached => None,
        }
    }

    pub fn active_turn_write_target(&self) -> Option<TurnWriteTarget> {
        self.active_task.turn_target
    }

    pub fn active_history_origin(&self) -> Option<HistoryOrigin> {
        match self.active_turn_write_target()? {
            TurnWriteTarget::ExistingTurn {
                turn_id,
                turn_index,
            } => Some(HistoryOrigin {
                turn_id,
                turn_index,
            }),
            TurnWriteTarget::BranchAdvance { .. } => None,
        }
    }

    pub fn set_active_turn_write_target(&mut self, target: Option<TurnWriteTarget>) {
        self.active_task.turn_target = target;
    }

    pub fn effective_write_policy(&self) -> ExecutionWritePolicy {
        if self.active_task.conflict_detached {
            ExecutionWritePolicy::Detached
        } else {
            self.execution.write_policy
        }
    }

    pub fn effective_conflict_policy(&self) -> ExecutionConflictPolicy {
        self.active_task
            .conflict_policy
            .unwrap_or(self.execution.conflict_policy)
    }

    pub fn set_active_task_conflict_policy(
        &mut self,
        conflict_policy: Option<ExecutionConflictPolicy>,
    ) {
        self.active_task.conflict_policy = conflict_policy;
    }

    pub fn current_task_branch_outcome(&self) -> Option<&TaskBranchOutcome> {
        self.active_task.branch_outcome.as_ref()
    }

    pub fn set_current_task_branch_outcome(&mut self, outcome: Option<TaskBranchOutcome>) {
        self.active_task.branch_outcome = outcome;
    }

    pub fn begin_conflict_detached_task(&mut self) {
        self.active_task.conflict_detached = true;
        self.active_task.turn_target = None;
    }

    pub fn begin_task_execution_override(
        &mut self,
        overrides: Option<&TaskExecutionOverrides>,
    ) -> Result<bool, String> {
        let Some(overrides) = overrides.filter(|overrides| !overrides.is_empty()) else {
            return Ok(false);
        };

        let previous_execution = self.execution.clone();
        let previous_target = previous_execution.context_target.clone();
        let mut task_execution = previous_execution.clone();
        task_execution.execution_id = new_execution_id();
        overrides.apply_to_execution(&mut task_execution)?;

        self.active_task.execution_restore = Some(TaskExecutionRestoreState {
            execution: previous_execution,
            selected_branch_head_cursor: self.selected_branch_head_cursor,
        });
        self.execution = task_execution;
        self.active_task.conflict_detached = false;
        self.active_task.turn_target = None;

        let target_changed = self.execution.context_target != previous_target;
        if target_changed {
            self.selected_branch_head_cursor = None;
        }

        Ok(target_changed)
    }

    pub fn finish_task_execution_scope(&mut self) -> bool {
        let restore = self.active_task.execution_restore.take();
        let should_refresh = restore.as_ref().is_some_and(|restore| {
            self.execution.context_target != restore.execution.context_target
        });
        if let Some(restore) = restore {
            self.execution = restore.execution;
            self.selected_branch_head_cursor = restore.selected_branch_head_cursor;
        }
        self.active_task = ActiveTaskState::default();
        should_refresh
    }
}

impl ExecutionStatusSnapshot {
    pub(crate) fn from_session(session: &SessionState) -> Self {
        Self::from_execution(&session.execution, session.effective_write_policy())
    }
}

pub(super) fn new_execution_id() -> String {
    format!("ex_{}", uuid::Uuid::now_v7().simple())
}

#[cfg(test)]
#[path = "tests/session.rs"]
mod tests;
