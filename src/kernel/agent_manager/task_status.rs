use std::sync::Arc;

use anyhow::Result;

use crate::kernel::session::{ExecutionContext, ExecutionStatusSnapshot, QueuedTask};

use super::{
    AgentRuntimeHandle, PeerAgentTaskResult, PendingTaskRecord, PendingTaskState,
    PromotedTaskBranch, PromotedTaskBranchFingerprint, TaskBranchOutcomeFingerprint,
    TaskStatusFingerprint, TaskStatusSnapshot,
};

pub(super) fn intended_task_execution_snapshot(
    handle: &Arc<AgentRuntimeHandle>,
    task: &QueuedTask,
) -> Result<ExecutionStatusSnapshot> {
    let mut execution = handle
        .control
        .current_execution()
        .map(|snapshot| ExecutionContext {
            execution_id: snapshot.execution_id,
            context_target: snapshot.context_target,
            visibility: snapshot.visibility,
            durability: snapshot.durability,
            write_policy: snapshot.write_policy,
            conflict_policy: handle.control.current_conflict_policy(),
        })
        .unwrap_or_default();

    if let Some(overrides) = task.execution.as_ref() {
        overrides
            .apply_to_execution(&mut execution)
            .map_err(anyhow::Error::msg)?;
    }
    if let Some(conflict_policy) = task.conflict_policy {
        execution.conflict_policy = conflict_policy;
    }

    Ok(ExecutionStatusSnapshot::from_execution(
        &execution,
        execution.write_policy,
    ))
}

pub(super) fn pending_task_snapshot(
    request_id: &str,
    pending: &PendingTaskRecord,
) -> TaskStatusSnapshot {
    TaskStatusSnapshot {
        request_id: request_id.to_string(),
        agent_id: pending.runtime_key.agent_id.clone(),
        slot_id: pending.runtime_key.slot_id.clone(),
        session_id: pending.session_target.session_id.clone(),
        trace_id: pending.trace_id.clone(),
        title: pending.title.clone(),
        prompt_preview: pending.prompt_preview.clone(),
        state: match pending.state {
            PendingTaskState::Queued => "queued".to_string(),
            PendingTaskState::Running => "running".to_string(),
            PendingTaskState::Cancelling => "cancelling".to_string(),
        },
        runtime_task_id: pending.runtime_task_id.clone(),
        execution: pending.execution.clone(),
        status: None,
        task_turn_count: None,
        branch_outcome: None,
        promotion_candidate: None,
        promoted_branch: None,
        output: None,
        assistant_content: None,
        error: None,
    }
}

pub(super) fn completed_task_snapshot(result: &PeerAgentTaskResult) -> TaskStatusSnapshot {
    TaskStatusSnapshot {
        request_id: result.request_id.clone(),
        agent_id: result.agent_id.clone(),
        slot_id: result.slot_id.clone(),
        session_id: result.session_id.clone(),
        trace_id: result.trace_id.clone(),
        title: result.title.clone(),
        prompt_preview: result.prompt_preview.clone(),
        state: "completed".to_string(),
        runtime_task_id: Some(result.runtime_task_id.clone()),
        execution: result.execution.clone(),
        status: Some(result.status),
        task_turn_count: Some(result.task_turn_count),
        branch_outcome: result.branch_outcome.clone(),
        promotion_candidate: result.promotion_candidate.clone(),
        promoted_branch: result.promoted_branch.clone(),
        output: result.output.clone(),
        assistant_content: result.assistant_content.clone(),
        error: result.error.clone(),
    }
}

pub(super) fn pending_task_fingerprint(
    request_id: &str,
    pending: &PendingTaskRecord,
) -> TaskStatusFingerprint {
    TaskStatusFingerprint {
        request_id: request_id.to_string(),
        state: match pending.state {
            PendingTaskState::Queued => "queued",
            PendingTaskState::Running => "running",
            PendingTaskState::Cancelling => "cancelling",
        },
        runtime_task_id: pending.runtime_task_id.clone(),
        session_id: pending.session_target.session_id.clone(),
        status: None,
        task_turn_count: None,
        branch_outcome: None,
        promotion_candidate: None,
        promoted_branch: None,
        output_bytes: 0,
        assistant_content_items: 0,
        assistant_content_bytes: 0,
        error: None,
    }
}

pub(super) fn completed_task_fingerprint(result: &PeerAgentTaskResult) -> TaskStatusFingerprint {
    let assistant_content = result.assistant_content.as_deref().unwrap_or_default();
    TaskStatusFingerprint {
        request_id: result.request_id.clone(),
        state: "completed",
        runtime_task_id: Some(result.runtime_task_id.clone()),
        session_id: result.session_id.clone(),
        status: Some(result.status),
        task_turn_count: Some(result.task_turn_count),
        branch_outcome: result
            .branch_outcome
            .as_ref()
            .map(task_branch_outcome_fingerprint),
        promotion_candidate: result
            .promotion_candidate
            .as_ref()
            .map(|candidate| (candidate.session_id.clone(), candidate.source_turn_id)),
        promoted_branch: result
            .promoted_branch
            .as_ref()
            .map(promoted_task_branch_fingerprint),
        output_bytes: result.output.as_deref().map(str::len).unwrap_or(0),
        assistant_content_items: assistant_content.len(),
        assistant_content_bytes: assistant_content.iter().map(task_input_content_size).sum(),
        error: result.error.clone(),
    }
}

fn task_branch_outcome_fingerprint(
    outcome: &crate::kernel::event::TaskBranchOutcome,
) -> TaskBranchOutcomeFingerprint {
    match outcome {
        crate::kernel::event::TaskBranchOutcome::ForkSibling {
            branch_id,
            branch_public_id,
            source_turn_id,
            persisted_active_head_unchanged,
            ..
        } => TaskBranchOutcomeFingerprint::ForkSibling {
            branch_id: *branch_id,
            branch_public_id: branch_public_id.clone(),
            source_turn_id: *source_turn_id,
            persisted_active_head_unchanged: *persisted_active_head_unchanged,
        },
        crate::kernel::event::TaskBranchOutcome::SidestepSibling {
            branch_id,
            branch_public_id,
            source_turn_id,
            persisted_active_head_unchanged,
            ..
        } => TaskBranchOutcomeFingerprint::SidestepSibling {
            branch_id: *branch_id,
            branch_public_id: branch_public_id.clone(),
            source_turn_id: *source_turn_id,
            persisted_active_head_unchanged: *persisted_active_head_unchanged,
        },
    }
}

fn promoted_task_branch_fingerprint(branch: &PromotedTaskBranch) -> PromotedTaskBranchFingerprint {
    PromotedTaskBranchFingerprint {
        branch_id: branch.branch_id.clone(),
        name: branch.name.clone(),
        head_turn_index: branch.head_turn_index,
        source_turn_id: branch.source_turn_id,
        origin_kind: branch.origin_kind.clone(),
        origin_task_id: branch.origin_task_id.clone(),
        origin_execution_id: branch.origin_execution_id.clone(),
        active: branch.active,
    }
}

fn task_input_content_size(content: &turin_types::TaskInputContent) -> usize {
    match content {
        turin_types::TaskInputContent::Text { text } => text.len(),
        turin_types::TaskInputContent::Image {
            name,
            content_type,
            url,
            local_path,
            detail,
        } => {
            option_string_len(name)
                + option_string_len(content_type)
                + option_string_len(url)
                + option_string_len(local_path)
                + option_string_len(detail)
        }
        turin_types::TaskInputContent::File {
            name,
            content_type,
            url,
            local_path,
        } => {
            option_string_len(name)
                + option_string_len(content_type)
                + option_string_len(url)
                + option_string_len(local_path)
        }
    }
}

fn option_string_len(value: &Option<String>) -> usize {
    value.as_deref().map(str::len).unwrap_or(0)
}
