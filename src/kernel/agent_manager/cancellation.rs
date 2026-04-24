use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::Result;

use crate::kernel::event::TaskTerminalStatus;
use crate::kernel::session::{ExecutionContext, ExecutionStatusSnapshot, ExecutionWritePolicy};

use super::{
    AgentManager, AgentRuntimeHandle, PeerAgentTaskResult, PendingTaskState, RuntimeSlotKey,
    TaskStatusSnapshot,
};

fn default_execution_snapshot() -> ExecutionStatusSnapshot {
    ExecutionStatusSnapshot::from_execution(
        &ExecutionContext::new(),
        ExecutionWritePolicy::AdvanceBranchHead,
    )
}

impl AgentManager {
    pub async fn cancel_task(&self, request_id: &str) -> Result<TaskStatusSnapshot> {
        if let Some(result) = self.completed_result(request_id).await {
            anyhow::bail!(
                "Task '{}' is already terminal ({:?})",
                request_id,
                result.status
            );
        }

        let pending = self
            .pending_task_states
            .read()
            .await
            .get(request_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Task '{}' not found", request_id))?;

        if pending.state == PendingTaskState::Running
            || pending.state == PendingTaskState::Cancelling
        {
            let handle = {
                let runtimes = self.runtimes.read().await;
                runtimes.get(&pending.runtime_key).cloned().ok_or_else(|| {
                    anyhow::anyhow!(
                        "Agent runtime '{}' [{}] is not available",
                        pending.runtime_key.agent_id,
                        pending.runtime_key.slot_id
                    )
                })?
            };

            let current_request_id = handle.control.current_request_id();
            if current_request_id.as_deref() != Some(request_id) {
                anyhow::bail!("Task '{}' is no longer the active running task", request_id);
            }
            if !handle.control.request_task_cancel() {
                anyhow::bail!(
                    "Task '{}' is running without a cancellable execution token",
                    request_id
                );
            }
            if let Some(pending) = self.pending_task_states.write().await.get_mut(request_id) {
                pending.state = PendingTaskState::Cancelling;
            }
            let snapshot = self.get_task(request_id).await.ok_or_else(|| {
                anyhow::anyhow!("Task '{}' disappeared after cancellation", request_id)
            })?;
            return Ok(snapshot);
        }

        let handle = {
            let runtimes = self.runtimes.read().await;
            runtimes.get(&pending.runtime_key).cloned().ok_or_else(|| {
                anyhow::anyhow!(
                    "Agent runtime '{}' [{}] is not available",
                    pending.runtime_key.agent_id,
                    pending.runtime_key.slot_id
                )
            })?
        };

        let removed = {
            let mut queue = handle
                .queue
                .lock()
                .expect("agent runtime queue mutex poisoned");
            let Some(index) = queue
                .iter()
                .position(|envelope| envelope.request_id.as_deref() == Some(request_id))
            else {
                anyhow::bail!("Task '{}' is no longer queued", request_id);
            };
            queue.remove(index)
        };

        let Some(envelope) = removed else {
            anyhow::bail!("Task '{}' is no longer queued", request_id);
        };

        handle.queued_tasks.fetch_sub(1, Ordering::Relaxed);

        let completed = PeerAgentTaskResult {
            request_id: request_id.to_string(),
            agent_id: pending.runtime_key.agent_id.clone(),
            slot_id: pending.runtime_key.slot_id.clone(),
            trace_id: pending.trace_id.clone(),
            runtime_task_id: String::new(),
            execution: pending.execution,
            status: TaskTerminalStatus::Cancelled,
            task_turn_count: 0,
            branch_outcome: None,
            promotion_candidate: None,
            promoted_branch: None,
            output: None,
            assistant_content: None,
            promotion_input_content: None,
            error: Some("Task cancelled before execution".to_string()),
        };

        if let Some(tx_result) = envelope.result_tx {
            let _ = tx_result.send(completed.clone());
        }

        self.record_completed_result(completed.clone()).await;

        Ok(TaskStatusSnapshot {
            request_id: completed.request_id,
            agent_id: completed.agent_id,
            slot_id: completed.slot_id,
            trace_id: completed.trace_id,
            state: "completed".to_string(),
            runtime_task_id: Some(completed.runtime_task_id),
            execution: completed.execution,
            status: Some(completed.status),
            task_turn_count: Some(completed.task_turn_count),
            branch_outcome: completed.branch_outcome,
            promotion_candidate: completed.promotion_candidate,
            promoted_branch: completed.promoted_branch,
            output: completed.output,
            assistant_content: completed.assistant_content,
            error: completed.error,
        })
    }

    pub async fn cancel_session(
        &self,
        session_id: &str,
        slot_id: Option<&str>,
    ) -> Result<(String, String, String)> {
        let (runtime_key, handle) = self.runtime_by_session_target(session_id, slot_id).await?;

        self.cancel_queued_requests_for_runtime(&runtime_key, "Session cancelled before execution")
            .await;

        if handle.control.current_request_id().is_none() {
            handle.control.request_session_cancel();
            handle.notify.notify_one();
        } else if !handle.control.request_session_cancel() {
            anyhow::bail!(
                "Session '{}' has no cancellable active execution",
                session_id
            );
        }

        Ok((
            runtime_key.agent_id,
            runtime_key.slot_id,
            session_id.to_string(),
        ))
    }

    pub async fn kill_session(
        &self,
        session_id: &str,
        slot_id: Option<&str>,
    ) -> Result<(String, String, String)> {
        let (runtime_key, handle) = self.runtime_by_session_target(session_id, slot_id).await?;

        self.kill_runtime_requests(&runtime_key, &handle, "Session killed")
            .await;

        if let Some(task) = &handle.task {
            task.abort();
        }

        self.runtimes.write().await.remove(&runtime_key);

        Ok((
            runtime_key.agent_id,
            runtime_key.slot_id,
            session_id.to_string(),
        ))
    }

    async fn cancel_queued_requests_for_runtime(&self, runtime_key: &RuntimeSlotKey, reason: &str) {
        let Some(handle) = self.runtimes.read().await.get(runtime_key).cloned() else {
            return;
        };

        let drained: Vec<_> = {
            let mut queue = handle
                .queue
                .lock()
                .expect("agent runtime queue mutex poisoned");
            queue.drain(..).collect()
        };
        if drained.is_empty() {
            return;
        }
        handle.queued_tasks.store(0, Ordering::Relaxed);

        for envelope in drained {
            let request_id = envelope
                .request_id
                .unwrap_or_else(|| uuid::Uuid::now_v7().simple().to_string());
            let completed = PeerAgentTaskResult {
                request_id: request_id.clone(),
                agent_id: runtime_key.agent_id.clone(),
                slot_id: runtime_key.slot_id.clone(),
                trace_id: envelope.task.trace_id.clone(),
                runtime_task_id: String::new(),
                execution: handle
                    .control
                    .current_execution()
                    .unwrap_or_else(default_execution_snapshot),
                status: TaskTerminalStatus::Cancelled,
                task_turn_count: 0,
                branch_outcome: None,
                promotion_candidate: None,
                promoted_branch: None,
                output: None,
                assistant_content: None,
                promotion_input_content: None,
                error: Some(reason.to_string()),
            };
            if let Some(tx_result) = envelope.result_tx {
                let _ = tx_result.send(completed.clone());
            }
            self.record_completed_result(completed).await;
        }
    }

    async fn kill_runtime_requests(
        &self,
        runtime_key: &RuntimeSlotKey,
        handle: &Arc<AgentRuntimeHandle>,
        reason: &str,
    ) {
        let drained: Vec<_> = {
            let mut queue = handle
                .queue
                .lock()
                .expect("agent runtime queue mutex poisoned");
            queue.drain(..).collect()
        };
        handle.queued_tasks.store(0, Ordering::Relaxed);

        for envelope in drained {
            let request_id = envelope
                .request_id
                .unwrap_or_else(|| uuid::Uuid::now_v7().simple().to_string());
            let completed = PeerAgentTaskResult {
                request_id: request_id.clone(),
                agent_id: runtime_key.agent_id.clone(),
                slot_id: runtime_key.slot_id.clone(),
                trace_id: envelope.task.trace_id.clone(),
                runtime_task_id: String::new(),
                execution: handle
                    .control
                    .current_execution()
                    .unwrap_or_else(default_execution_snapshot),
                status: TaskTerminalStatus::Killed,
                task_turn_count: 0,
                branch_outcome: None,
                promotion_candidate: None,
                promoted_branch: None,
                output: None,
                assistant_content: None,
                promotion_input_content: None,
                error: Some(reason.to_string()),
            };
            if let Some(tx_result) = envelope.result_tx {
                let _ = tx_result.send(completed.clone());
            }
            self.record_completed_result(completed).await;
        }

        if let Some(request_id) = handle.control.current_request_id() {
            let trace_id = self
                .pending_task_states
                .read()
                .await
                .get(&request_id)
                .map(|pending| pending.trace_id.clone())
                .unwrap_or_default();
            let completed = PeerAgentTaskResult {
                request_id: request_id.clone(),
                agent_id: runtime_key.agent_id.clone(),
                slot_id: runtime_key.slot_id.clone(),
                trace_id,
                runtime_task_id: handle.control.current_runtime_task_id().unwrap_or_default(),
                execution: handle
                    .control
                    .current_execution()
                    .unwrap_or_else(default_execution_snapshot),
                status: TaskTerminalStatus::Killed,
                task_turn_count: 0,
                branch_outcome: None,
                promotion_candidate: None,
                promoted_branch: None,
                output: None,
                assistant_content: None,
                promotion_input_content: None,
                error: Some(reason.to_string()),
            };
            self.record_completed_result(completed).await;
        }
    }
}
