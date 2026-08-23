use std::collections::{HashMap, VecDeque};
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
mod execution;
mod queued_tasks;
mod resident_history;

pub use completed_tasks::{
    CompletedLocalTaskResults, CompletedLocalTaskResultsHandle, LocalTaskResult,
};
pub use execution::{
    BranchHeadCursor, ExecutionConflictPolicy, ExecutionContext, ExecutionContextTarget,
    ExecutionDurability, ExecutionStatusSnapshot, ExecutionVisibility, ExecutionWritePolicy,
    PreparedSidestepExecution, SidestepMode, TaskExecutionOverrides,
};
pub use queued_tasks::QueuedTask;
pub use resident_history::{HistoryOrigin, ResidentHistory};

pub(crate) type SessionHarnessEngine = Arc<std::sync::Mutex<Box<dyn HarnessInstance>>>;

/// Transient per-task execution state that is reset when a task completes.
#[derive(Debug, Default)]
pub struct ActiveTaskState {
    pub conflict_policy: Option<ExecutionConflictPolicy>,
    pub branch_outcome: Option<TaskBranchOutcome>,
    pub conflict_detached: bool,
    pub turn_target: Option<TurnWriteTarget>,
    budget: Option<ActiveTaskBudget>,
    execution_restore: Option<TaskExecutionRestoreState>,
    delegation_budget: Option<Arc<crate::kernel::delegation_budget::DelegationBudget>>,
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

pub enum PersistedKernelRecord {
    Event(Box<PersistedKernelEvent>),
    Barrier(tokio::sync::oneshot::Sender<Result<(), String>>),
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
    pub(crate) harness_engine: Option<SessionHarnessEngine>,
    pub(crate) harness_generation: u64,
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

    pub(crate) fn record_delegation_tokens(&self, tokens: u64) {
        if let Some(budget) = self.active_task.delegation_budget.as_ref() {
            budget.record_tokens(tokens);
        }
    }

    pub(crate) fn reserve_delegation_tool_calls(&self, requested: usize) -> usize {
        self.active_task
            .delegation_budget
            .as_ref()
            .map_or(requested, |budget| budget.reserve_tool_calls(requested))
    }

    pub(crate) fn set_active_delegation_budget(
        &mut self,
        budget: Option<Arc<crate::kernel::delegation_budget::DelegationBudget>>,
    ) {
        self.active_task.delegation_budget = budget;
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
        task_execution.execution_id = execution::new_execution_id();
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

#[cfg(test)]
#[path = "tests/session.rs"]
mod tests;
