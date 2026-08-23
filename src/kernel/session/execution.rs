use crate::kernel::event::TaskBranchOutcome;

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

pub(super) fn new_execution_id() -> String {
    format!("ex_{}", uuid::Uuid::now_v7().simple())
}
