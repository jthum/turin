use serde_json::Value;
#[cfg(feature = "lua")]
use serde_json::json;
use std::collections::{BTreeSet, HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio_util::sync::CancellationToken;

use super::event::{KernelEvent, TaskBranchOutcome, TaskTerminalStatus};
use super::governance::GovernanceSnapshot;
use super::identity::RuntimeIdentity;
use super::session::ExecutionStatusSnapshot;
use super::session::{
    CompletedLocalTaskResultsHandle, ExecutionConflictPolicy, ExecutionContextTarget,
    ExecutionDurability, ExecutionVisibility, ExecutionWritePolicy, PersistedKernelRecord,
    QueuedTask,
};
use crate::persistence::manager::StoreSelector;

mod request_options;
pub use request_options::RequestOptionsOverride;
pub(crate) use request_options::build_merged_request_options;

#[derive(Clone, Debug, Default)]
pub struct ToolExposure {
    selected: Option<BTreeSet<String>>,
    excluded: BTreeSet<String>,
}

impl ToolExposure {
    pub fn exposes(&self, name: &str) -> bool {
        self.selected
            .as_ref()
            .is_none_or(|selected| selected.contains(name))
            && !self.excluded.contains(name)
    }

    pub fn only(&mut self, names: BTreeSet<String>) {
        self.selected = Some(names);
        self.excluded.clear();
    }

    pub fn include(&mut self, names: BTreeSet<String>) {
        for name in names {
            self.excluded.remove(&name);
            if let Some(selected) = self.selected.as_mut() {
                selected.insert(name);
            }
        }
    }

    pub fn exclude(&mut self, names: BTreeSet<String>) {
        if let Some(selected) = self.selected.as_mut() {
            for name in names {
                selected.remove(&name);
            }
        } else {
            self.excluded.extend(names);
        }
    }

    pub fn expose_all(&mut self) {
        self.selected = None;
        self.excluded.clear();
    }
}

pub(crate) type SessionQueue = Arc<Mutex<VecDeque<QueuedTask>>>;

#[derive(Clone)]
#[cfg_attr(not(feature = "lua"), allow(dead_code))]
pub(crate) struct HarnessEventContext {
    pub(crate) json: bool,
    pub(crate) internal_id: Option<i64>,
    pub(crate) turn_id: Option<i64>,
    pub(crate) event_tx: tokio::sync::broadcast::Sender<(Option<i64>, KernelEvent)>,
    pub(crate) durability_tx: Option<tokio::sync::mpsc::UnboundedSender<PersistedKernelRecord>>,
}

#[derive(Clone)]
#[cfg_attr(not(feature = "lua"), allow(dead_code))]
pub(crate) struct HarnessExecutionMetadata {
    pub(crate) execution_id: String,
    pub(crate) context_target: ExecutionContextTarget,
    pub(crate) visibility: ExecutionVisibility,
    pub(crate) durability: ExecutionDurability,
    pub(crate) write_policy: ExecutionWritePolicy,
    pub(crate) conflict_policy: ExecutionConflictPolicy,
}

#[derive(Clone)]
#[cfg_attr(not(feature = "lua"), allow(dead_code))]
pub(crate) struct HarnessExecutionBinding {
    pub(crate) agent_id: String,
    pub(crate) session_id: String,
    pub(crate) store_selector: StoreSelector,
    pub(crate) default_store_selector: Option<StoreSelector>,
    pub(crate) execution: HarnessExecutionMetadata,
    pub(crate) runtime_slot_id: Option<String>,
    pub(crate) trace_id: String,
    pub(crate) completed_task_results: CompletedLocalTaskResultsHandle,
    pub(crate) event_context: HarnessEventContext,
    pub(crate) cancel_token: CancellationToken,
}
use crate::inference::provider::{InferenceMessage, ProviderClient};
use crate::kernel::config::{InferenceOverrideConfig, TurinConfig};

/// Mutable provider request and turn metadata presented to `on_turn_prepare`.
///
/// Ownership moves into this value before the hook and back into the provider request
/// afterward. Native harnesses mutate it directly; scripting adapters may temporarily
/// wrap it without making JSON the canonical representation.
pub struct HarnessTurnRequest {
    pub inference: Option<String>,
    pub model: String,
    pub provider: String,
    pub system_prompt: String,
    pub messages: Vec<InferenceMessage>,
    pub turn_index: u32,
    pub task_turn_index: u32,
    pub is_first_turn_in_task: bool,
    pub task_id: String,
    pub plan_id: Option<String>,
    pub token_count: u32,
    pub token_limit: u32,
    pub thinking_budget: u32,
    pub request_options: RequestOptionsOverride,
    pub agent_id: String,
    pub session_inference: InferenceOverrideConfig,
    pub session_id: String,
    pub session_title: Option<String>,
    pub available_tools: BTreeSet<String>,
    pub tool_exposure: ToolExposure,
}

/// A durable runtime signal delivered to a harness subscription.
///
/// The contract borrows delivery data and deliberately omits persistence bookkeeping
/// such as database row IDs and retry counters.
#[derive(Clone, Copy, Debug)]
pub struct HarnessSignal<'a> {
    pub signal_id: Option<uuid::Uuid>,
    pub topic: &'a str,
    pub source_agent_id: &'a str,
    pub target_agent_id: &'a str,
    pub source_session_id: Option<&'a str>,
    pub target_session_id: Option<&'a str>,
    pub payload: &'a str,
    pub created_at: &'a str,
}

/// A named harness action invoked by a client, schedule, or runtime facility.
pub struct HarnessActionRequest<'a> {
    pub agent_id: &'a str,
    pub name: &'a str,
    pub params: Value,
}

#[cfg_attr(not(feature = "lua"), allow(dead_code))]
pub(crate) struct HarnessTurnServices<'a> {
    pub(crate) clients: &'a HashMap<String, ProviderClient>,
    pub(crate) config: &'a Arc<TurinConfig>,
}

/// Typed lifecycle and policy input presented to a session harness.
///
/// Scripting adapters own conversion from this contract into language values. Native
/// harnesses can inspect the borrowed domain values directly.
pub enum HarnessHook<'a> {
    SessionStart {
        identity: &'a RuntimeIdentity,
        session_id: &'a str,
        governance: &'a GovernanceSnapshot,
    },
    SessionEnd {
        identity: &'a RuntimeIdentity,
        session_id: &'a str,
        turn_count: u32,
        total_input_tokens: u64,
        total_output_tokens: u64,
    },
    TaskStart {
        identity: &'a RuntimeIdentity,
        session_id: &'a str,
        task_id: &'a str,
        trace_id: &'a str,
        plan_id: Option<&'a str>,
        title: Option<&'a str>,
        prompt: &'a str,
        queue_depth: usize,
    },
    TaskComplete {
        identity: &'a RuntimeIdentity,
        session_id: &'a str,
        task_id: &'a str,
        trace_id: &'a str,
        plan_id: Option<&'a str>,
        status: TaskTerminalStatus,
        task_turn_count: u32,
        task_started_at_unix_ms: Option<u64>,
        task_elapsed_ms: u64,
        task_input_tokens: u64,
        task_output_tokens: u64,
        task_total_tokens: u64,
        turn_count: u32,
        execution: &'a ExecutionStatusSnapshot,
        branch_outcome: Option<&'a TaskBranchOutcome>,
        error: Option<&'a str>,
    },
    PlanComplete {
        identity: &'a RuntimeIdentity,
        session_id: &'a str,
        plan_id: &'a str,
        title: &'a str,
        total_tasks: usize,
        completed_tasks: usize,
    },
    AllTasksComplete {
        identity: &'a RuntimeIdentity,
        session_id: &'a str,
        turn_count: u32,
    },
    InferenceError {
        identity: &'a RuntimeIdentity,
        session_id: &'a str,
        task_id: &'a str,
        trace_id: &'a str,
        plan_id: Option<&'a str>,
        turn_count: u32,
        error: &'a str,
    },
    TurnStart {
        identity: &'a RuntimeIdentity,
        session_id: &'a str,
        task_id: &'a str,
        trace_id: &'a str,
        plan_id: Option<&'a str>,
        turn_index: u32,
        task_turn_index: u32,
    },
    TurnEnd {
        identity: &'a RuntimeIdentity,
        session_id: &'a str,
        task_id: &'a str,
        trace_id: &'a str,
        plan_id: Option<&'a str>,
        turn_index: u32,
        task_turn_index: u32,
        has_tool_calls: bool,
    },
    ToolCall {
        name: &'a str,
        id: &'a str,
        args: &'a Value,
    },
    ToolResult {
        id: &'a str,
        name: &'a str,
        args: &'a Value,
        output: &'a str,
        is_error: bool,
    },
    TokenUsage {
        input_tokens: u64,
        output_tokens: u64,
        task_started_at_unix_ms: Option<u64>,
        task_elapsed_ms: u64,
        task_input_tokens: u64,
        task_output_tokens: u64,
        task_turn_count: u32,
    },
    PlanSubmit {
        title: &'a str,
        tasks: &'a [String],
        clear_existing: bool,
    },
    KernelEvent(&'a KernelEvent),
}

impl HarnessHook<'_> {
    #[cfg(feature = "lua")]
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::SessionStart { .. } => "on_session_start",
            Self::SessionEnd { .. } => "on_session_end",
            Self::TaskStart { .. } => "on_task_start",
            Self::TaskComplete { .. } => "on_task_complete",
            Self::PlanComplete { .. } => "on_plan_complete",
            Self::AllTasksComplete { .. } => "on_all_tasks_complete",
            Self::InferenceError { .. } => "on_inference_error",
            Self::TurnStart { .. } => "on_turn_start",
            Self::TurnEnd { .. } => "on_turn_end",
            Self::ToolCall { .. } => "on_tool_call",
            Self::ToolResult { .. } => "on_tool_result",
            Self::TokenUsage { .. } => "on_token_usage",
            Self::PlanSubmit { .. } => "on_plan_submit",
            Self::KernelEvent(_) => "on_kernel_event",
        }
    }

    #[cfg(feature = "lua")]
    pub(crate) fn lua_payload(&self) -> Value {
        match self {
            Self::SessionStart {
                identity,
                session_id,
                governance,
            } => {
                json!({ "identity": identity, "session_id": session_id, "governance": governance })
            }
            Self::SessionEnd {
                identity,
                session_id,
                turn_count,
                total_input_tokens,
                total_output_tokens,
            } => {
                json!({ "identity": identity, "session_id": session_id, "turn_count": turn_count, "total_input_tokens": total_input_tokens, "total_output_tokens": total_output_tokens })
            }
            Self::TaskStart {
                identity,
                session_id,
                task_id,
                trace_id,
                plan_id,
                title,
                prompt,
                queue_depth,
            } => {
                json!({ "identity": identity, "session_id": session_id, "task_id": task_id, "trace_id": trace_id, "plan_id": plan_id, "title": title, "prompt": prompt, "queue_depth": queue_depth })
            }
            Self::TaskComplete {
                identity,
                session_id,
                task_id,
                trace_id,
                plan_id,
                status,
                task_turn_count,
                task_started_at_unix_ms,
                task_elapsed_ms,
                task_input_tokens,
                task_output_tokens,
                task_total_tokens,
                turn_count,
                execution,
                branch_outcome,
                error,
            } => {
                json!({ "identity": identity, "session_id": session_id, "task_id": task_id, "trace_id": trace_id, "plan_id": plan_id, "status": status, "task_turn_count": task_turn_count, "task_started_at_unix_ms": task_started_at_unix_ms, "task_elapsed_ms": task_elapsed_ms, "task_input_tokens": task_input_tokens, "task_output_tokens": task_output_tokens, "task_total_tokens": task_total_tokens, "turn_count": turn_count, "execution": execution, "branch_outcome": branch_outcome, "error": error })
            }
            Self::PlanComplete {
                identity,
                session_id,
                plan_id,
                title,
                total_tasks,
                completed_tasks,
            } => {
                json!({ "identity": identity, "session_id": session_id, "plan_id": plan_id, "title": title, "total_tasks": total_tasks, "completed_tasks": completed_tasks })
            }
            Self::AllTasksComplete {
                identity,
                session_id,
                turn_count,
            } => {
                json!({ "identity": identity, "session_id": session_id, "turn_count": turn_count })
            }
            Self::InferenceError {
                identity,
                session_id,
                task_id,
                trace_id,
                plan_id,
                turn_count,
                error,
            } => {
                json!({ "identity": identity, "session_id": session_id, "task_id": task_id, "trace_id": trace_id, "plan_id": plan_id, "turn_count": turn_count, "error": error })
            }
            Self::TurnStart {
                identity,
                session_id,
                task_id,
                trace_id,
                plan_id,
                turn_index,
                task_turn_index,
            } => {
                json!({ "identity": identity, "session_id": session_id, "task_id": task_id, "trace_id": trace_id, "plan_id": plan_id, "turn_index": turn_index, "task_turn_index": task_turn_index })
            }
            Self::TurnEnd {
                identity,
                session_id,
                task_id,
                trace_id,
                plan_id,
                turn_index,
                task_turn_index,
                has_tool_calls,
            } => {
                json!({ "identity": identity, "session_id": session_id, "task_id": task_id, "trace_id": trace_id, "plan_id": plan_id, "turn_index": turn_index, "task_turn_index": task_turn_index, "has_tool_calls": has_tool_calls })
            }
            Self::ToolCall { name, id, args } => json!({ "name": name, "id": id, "args": args }),
            Self::ToolResult {
                id,
                name,
                args,
                output,
                is_error,
            } => {
                json!({ "id": id, "name": name, "args": args, "output": output, "is_error": is_error })
            }
            Self::TokenUsage {
                input_tokens,
                output_tokens,
                task_started_at_unix_ms,
                task_elapsed_ms,
                task_input_tokens,
                task_output_tokens,
                task_turn_count,
            } => {
                json!({ "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens, "task_started_at_unix_ms": task_started_at_unix_ms, "task_elapsed_ms": task_elapsed_ms, "task_input_tokens": task_input_tokens, "task_output_tokens": task_output_tokens, "task_total_tokens": task_input_tokens + task_output_tokens, "task_turn_count": task_turn_count })
            }
            Self::PlanSubmit {
                title,
                tasks,
                clear_existing,
            } => json!({ "title": title, "tasks": tasks, "clear_existing": clear_existing }),
            Self::KernelEvent(event) => serde_json::to_value(event).unwrap_or_default(),
        }
    }
}
