use serde_json::{Value, json};

use super::event::{KernelEvent, TaskBranchOutcome, TaskTerminalStatus};
use super::governance::GovernanceSnapshot;
use super::identity::RuntimeIdentity;
use super::session::ExecutionStatusSnapshot;

/// Typed lifecycle and policy input presented to a session harness.
///
/// Scripting adapters own conversion from this contract into language values. Native
/// harnesses can inspect the borrowed domain values directly.
pub(crate) enum HarnessHook<'a> {
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
