use std::collections::{BTreeMap, HashMap, HashSet};

use crate::daemon::state::{
    SessionCompactionDetail, SessionEfficiencyDetail, SessionExecutionContextDetail,
    SessionExecutionDetail, SessionPlanExecutionDetail, SessionRequestEfficiencyDetail,
    SessionTaskExecutionDetail, SessionTaskTurnDetail, SessionTurnEfficiencyDetail,
};
use crate::kernel::event::{
    InferenceRequestMetrics, KernelEvent, LifecycleEvent, TaskTerminalStatus,
};
use crate::kernel::session::ExecutionStatusSnapshot;

use super::SESSION_EXECUTION_EVENT_LIMIT;

pub(super) fn session_execution_from_events(
    events: Vec<crate::persistence::schema::EventRow>,
    truncated: bool,
) -> SessionExecutionDetail {
    let mut tasks = Vec::<SessionTaskExecutionDetail>::new();
    let mut task_indexes = HashMap::<String, usize>::new();
    let mut plan_completions = BTreeMap::<String, SessionPlanExecutionDetail>::new();

    for event in events {
        let Ok(KernelEvent::Lifecycle(lifecycle)) = serde_json::from_str(&event.payload) else {
            continue;
        };
        match lifecycle {
            LifecycleEvent::TaskStart {
                identity,
                task_id,
                trace_id,
                plan_id,
                title,
                prompt,
                queue_depth,
                execution,
            } => {
                let index = tasks.len();
                task_indexes.insert(task_id.clone(), index);
                tasks.push(SessionTaskExecutionDetail {
                    task_id,
                    trace_id,
                    plan_id,
                    run_id: identity.run_id().map(str::to_string),
                    agent_id: identity.agent_id().to_string(),
                    title,
                    prompt,
                    status: "running".to_string(),
                    queue_depth,
                    task_turn_count: 0,
                    execution: execution_context_detail(execution),
                    turns: Vec::new(),
                    branch_outcome: None,
                    error: None,
                    started_at: event.created_at,
                    completed_at: None,
                });
            }
            LifecycleEvent::TaskComplete {
                identity,
                task_id,
                trace_id,
                plan_id,
                status,
                task_turn_count,
                execution,
                branch_outcome,
                error,
            } => {
                let index = if let Some(index) = task_indexes.get(&task_id).copied() {
                    index
                } else {
                    let index = tasks.len();
                    task_indexes.insert(task_id.clone(), index);
                    tasks.push(SessionTaskExecutionDetail {
                        task_id: task_id.clone(),
                        trace_id: trace_id.clone(),
                        plan_id: plan_id.clone(),
                        run_id: identity.run_id().map(str::to_string),
                        agent_id: identity.agent_id().to_string(),
                        title: None,
                        prompt: String::new(),
                        status: task_terminal_status(status).to_string(),
                        queue_depth: 0,
                        task_turn_count,
                        execution: execution_context_detail(execution.clone()),
                        turns: Vec::new(),
                        branch_outcome: None,
                        error: None,
                        started_at: event.created_at.clone(),
                        completed_at: None,
                    });
                    index
                };
                let task = &mut tasks[index];
                task.trace_id = trace_id;
                task.plan_id = plan_id;
                task.status = task_terminal_status(status).to_string();
                task.task_turn_count = task_turn_count;
                task.execution = execution_context_detail(execution);
                task.branch_outcome =
                    branch_outcome.and_then(|outcome| serde_json::to_value(outcome).ok());
                task.error = error;
                task.completed_at = Some(event.created_at);
            }
            LifecycleEvent::TurnStart {
                turn_index,
                task_id,
                task_turn_index,
                ..
            } => {
                let Some(index) = task_indexes.get(&task_id).copied() else {
                    continue;
                };
                tasks[index].turns.push(SessionTaskTurnDetail {
                    turn_index,
                    task_turn_index,
                    has_tool_calls: None,
                    started_at: event.created_at,
                    completed_at: None,
                });
            }
            LifecycleEvent::TurnEnd {
                turn_index,
                task_id,
                task_turn_index,
                has_tool_calls,
                ..
            } => {
                let Some(index) = task_indexes.get(&task_id).copied() else {
                    continue;
                };
                let task = &mut tasks[index];
                if let Some(turn) = task
                    .turns
                    .iter_mut()
                    .rev()
                    .find(|turn| turn.turn_index == turn_index)
                {
                    turn.has_tool_calls = Some(has_tool_calls);
                    turn.completed_at = Some(event.created_at);
                } else {
                    task.turns.push(SessionTaskTurnDetail {
                        turn_index,
                        task_turn_index,
                        has_tool_calls: Some(has_tool_calls),
                        started_at: event.created_at.clone(),
                        completed_at: Some(event.created_at),
                    });
                }
            }
            LifecycleEvent::PlanComplete {
                plan_id,
                title,
                total_tasks,
                completed_tasks,
                ..
            } => {
                plan_completions.insert(
                    plan_id.clone(),
                    SessionPlanExecutionDetail {
                        plan_id,
                        title: Some(title),
                        status: "complete".to_string(),
                        total_tasks,
                        completed_tasks,
                        started_at: event.created_at.clone(),
                        completed_at: Some(event.created_at),
                    },
                );
            }
            _ => {}
        }
    }

    for task in &tasks {
        let Some(plan_id) = task.plan_id.as_ref() else {
            continue;
        };
        let plan =
            plan_completions
                .entry(plan_id.clone())
                .or_insert_with(|| SessionPlanExecutionDetail {
                    plan_id: plan_id.clone(),
                    title: None,
                    status: "running".to_string(),
                    total_tasks: 0,
                    completed_tasks: 0,
                    started_at: task.started_at.clone(),
                    completed_at: None,
                });
        if plan.completed_at.is_none() {
            plan.total_tasks = plan.total_tasks.saturating_add(1);
            if task.status != "running" {
                plan.completed_tasks = plan.completed_tasks.saturating_add(1);
            }
            if task.started_at < plan.started_at {
                plan.started_at.clone_from(&task.started_at);
            }
        }
    }

    tasks.sort_by(|left, right| {
        let left_date = left.completed_at.as_ref().unwrap_or(&left.started_at);
        let right_date = right.completed_at.as_ref().unwrap_or(&right.started_at);
        right_date.cmp(left_date)
    });
    let mut plans = plan_completions.into_values().collect::<Vec<_>>();
    plans.sort_by(|left, right| right.started_at.cmp(&left.started_at));

    SessionExecutionDetail {
        tasks,
        plans,
        event_limit: SESSION_EXECUTION_EVENT_LIMIT,
        truncated,
    }
}

fn execution_context_detail(execution: ExecutionStatusSnapshot) -> SessionExecutionContextDetail {
    SessionExecutionContextDetail {
        execution_id: execution.execution_id,
        context_target: serde_json::to_value(execution.context_target)
            .unwrap_or(serde_json::Value::Null),
        visibility: serialized_enum_name(&execution.visibility),
        durability: serialized_enum_name(&execution.durability),
        write_policy: serialized_enum_name(&execution.write_policy),
    }
}

fn serialized_enum_name(value: &impl serde::Serialize) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|value| value.as_str().map(str::to_string))
        .unwrap_or_else(|| "unknown".to_string())
}

fn task_terminal_status(status: TaskTerminalStatus) -> &'static str {
    match status {
        TaskTerminalStatus::Success => "success",
        TaskTerminalStatus::Rejected => "rejected",
        TaskTerminalStatus::Conflict => "conflict",
        TaskTerminalStatus::MaxTurns => "max_turns",
        TaskTerminalStatus::Error => "error",
        TaskTerminalStatus::Cancelled => "cancelled",
        TaskTerminalStatus::TimedOut => "timed_out",
        TaskTerminalStatus::Killed => "killed",
    }
}

pub(super) fn session_efficiency_from_events(
    events: Vec<crate::persistence::schema::EventRow>,
    visible_turn_indexes: Option<&HashSet<u32>>,
) -> SessionEfficiencyDetail {
    let mut turns = BTreeMap::<u32, SessionTurnEfficiencyDetail>::new();
    let mut latest_compaction = None;
    let mut total_input_tokens = 0_u64;
    let mut total_output_tokens = 0_u64;
    let mut total_cache_read_input_tokens = 0_u64;
    let mut total_cache_creation_input_tokens = 0_u64;
    let mut total_request_count = 0_usize;
    let mut provider_cache_metrics_available = false;

    for event in events {
        let Ok(payload) = serde_json::from_str::<serde_json::Value>(&event.payload) else {
            continue;
        };
        match event.event_type.as_str() {
            "inference_request" => {
                total_request_count = total_request_count.saturating_add(1);
                let (Some(turn_index), Some(metrics)) = (
                    event.turn_index,
                    payload.get("metrics").cloned().and_then(|value| {
                        serde_json::from_value::<InferenceRequestMetrics>(value).ok()
                    }),
                ) else {
                    continue;
                };
                if visible_turn_indexes.is_some_and(|visible| !visible.contains(&turn_index)) {
                    continue;
                }
                let turn = turns
                    .entry(turn_index)
                    .or_insert_with(|| empty_turn_efficiency(turn_index, event.created_at.clone()));
                turn.requests.push(SessionRequestEfficiencyDetail {
                    metrics: Some(metrics),
                    input_tokens: None,
                    output_tokens: None,
                    cache_read_input_tokens: None,
                    cache_creation_input_tokens: None,
                    created_at: event.created_at.clone(),
                });
                turn.created_at = event.created_at;
            }
            "message_end" => {
                let Some(turn_index) = event.turn_index else {
                    continue;
                };
                let input_tokens = payload
                    .get("input_tokens")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                let output_tokens = payload
                    .get("output_tokens")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0);
                let cache_read_input_tokens = payload
                    .get("cache_read_input_tokens")
                    .and_then(serde_json::Value::as_u64);
                let cache_creation_input_tokens = payload
                    .get("cache_creation_input_tokens")
                    .and_then(serde_json::Value::as_u64);
                total_input_tokens = total_input_tokens.saturating_add(input_tokens);
                total_output_tokens = total_output_tokens.saturating_add(output_tokens);
                if let Some(tokens) = cache_read_input_tokens {
                    provider_cache_metrics_available = true;
                    total_cache_read_input_tokens =
                        total_cache_read_input_tokens.saturating_add(tokens);
                }
                if let Some(tokens) = cache_creation_input_tokens {
                    provider_cache_metrics_available = true;
                    total_cache_creation_input_tokens =
                        total_cache_creation_input_tokens.saturating_add(tokens);
                }
                if visible_turn_indexes.is_some_and(|visible| !visible.contains(&turn_index)) {
                    continue;
                }
                let turn = turns
                    .entry(turn_index)
                    .or_insert_with(|| empty_turn_efficiency(turn_index, event.created_at.clone()));
                turn.input_tokens = turn.input_tokens.saturating_add(input_tokens);
                turn.output_tokens = turn.output_tokens.saturating_add(output_tokens);
                if let Some(request) = turn
                    .requests
                    .iter_mut()
                    .rev()
                    .find(|request| request.input_tokens.is_none())
                {
                    request.input_tokens = Some(input_tokens);
                    request.output_tokens = Some(output_tokens);
                    request.cache_read_input_tokens = cache_read_input_tokens;
                    request.cache_creation_input_tokens = cache_creation_input_tokens;
                } else {
                    turn.requests.push(SessionRequestEfficiencyDetail {
                        metrics: None,
                        input_tokens: Some(input_tokens),
                        output_tokens: Some(output_tokens),
                        cache_read_input_tokens,
                        cache_creation_input_tokens,
                        created_at: event.created_at.clone(),
                    });
                }
                turn.created_at = event.created_at;
            }
            "context_compaction" => {
                let Some(checkpoint) = payload.get("checkpoint") else {
                    continue;
                };
                latest_compaction = Some(SessionCompactionDetail {
                    covered_through_turn_id: checkpoint
                        .get("covered_through_turn_id")
                        .and_then(serde_json::Value::as_i64)
                        .unwrap_or(0),
                    covered_through_turn_index: checkpoint
                        .get("covered_through_turn_index")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0) as u32,
                    generated_at_turn_index: checkpoint
                        .get("generated_at_turn_index")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0) as u32,
                    provider: checkpoint
                        .get("provider_name")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("unknown")
                        .to_string(),
                    model: checkpoint
                        .get("model")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or("unknown")
                        .to_string(),
                    created_at: event.created_at,
                });
            }
            _ => {}
        }
    }

    let turns = turns.into_values().collect::<Vec<_>>();
    SessionEfficiencyDetail {
        total_input_tokens,
        total_output_tokens,
        total_cache_read_input_tokens,
        total_cache_creation_input_tokens,
        total_request_count,
        turns,
        latest_compaction,
        provider_cache_metrics_available,
    }
}

fn empty_turn_efficiency(turn_index: u32, created_at: String) -> SessionTurnEfficiencyDetail {
    SessionTurnEfficiencyDetail {
        turn_index,
        requests: Vec::new(),
        input_tokens: 0,
        output_tokens: 0,
        created_at,
    }
}
