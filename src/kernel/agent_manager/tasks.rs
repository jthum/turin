use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::{Result, anyhow};
use tokio::sync::oneshot;

use crate::kernel::session::{ExecutionContext, ExecutionStatusSnapshot, QueuedTask};
use crate::kernel::task_promotion::promote_task_result;

use super::{
    AgentManager, AgentRuntimeHandle, PeerAgentTaskEnvelope, PeerAgentTaskResult,
    PendingTaskRecord, PendingTaskState, PromotedTaskBranch, RuntimeSlotKey, TaskStatusSnapshot,
};

fn intended_task_execution_snapshot(
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

impl AgentManager {
    /// Submit a task to a peer agent and return a request ID for later `await_result`.
    pub async fn submit(
        self: &Arc<Self>,
        agent_id: &str,
        task: QueuedTask,
        delegated_capabilities: Option<BTreeMap<String, bool>>,
    ) -> Result<String> {
        self.submit_to_runtime(
            RuntimeSlotKey::default_for(agent_id),
            task,
            delegated_capabilities,
        )
        .await
    }

    pub async fn submit_to_session(
        self: &Arc<Self>,
        session_id: &str,
        slot_id: Option<&str>,
        task: QueuedTask,
        delegated_capabilities: Option<BTreeMap<String, bool>>,
    ) -> Result<String> {
        let (runtime_key, _) = self.runtime_by_session_target(session_id, slot_id).await?;
        self.submit_to_runtime(runtime_key, task, delegated_capabilities)
            .await
    }

    async fn submit_to_runtime(
        self: &Arc<Self>,
        runtime_key: RuntimeSlotKey,
        task: QueuedTask,
        delegated_capabilities: Option<BTreeMap<String, bool>>,
    ) -> Result<String> {
        let trace_id = task.trace_id.clone();
        let request_id = uuid::Uuid::now_v7().simple().to_string();
        let (tx_result, rx_result) = oneshot::channel();
        let handle = self.ensure_runtime_slot(runtime_key.clone()).await?;
        {
            let mut pending = self.pending_results.write().await;
            pending.insert(request_id.clone(), rx_result);
        }
        {
            let mut pending = self.pending_task_states.write().await;
            pending.insert(
                request_id.clone(),
                PendingTaskRecord {
                    runtime_key: runtime_key.clone(),
                    trace_id,
                    state: PendingTaskState::Queued,
                    runtime_task_id: None,
                    execution: intended_task_execution_snapshot(&handle, &task)?,
                },
            );
        }

        if let Err(e) = self
            .enqueue_runtime_task(
                &handle,
                PeerAgentTaskEnvelope {
                    task,
                    request_id: Some(request_id.clone()),
                    result_tx: Some(tx_result),
                    delegated_capabilities,
                },
            )
            .await
        {
            self.remove_pending_request(&request_id).await;
            return Err(e);
        }

        Ok(request_id)
    }

    /// Await a previously submitted peer task result.
    pub async fn await_result(
        &self,
        request_id: &str,
        timeout_ms: Option<u64>,
    ) -> Result<PeerAgentTaskResult> {
        if let Some(result) = self.completed_result(request_id).await {
            return Ok(result);
        }

        let mut rx = {
            let mut pending = self.pending_results.write().await;
            pending.remove(request_id).ok_or_else(|| {
                anyhow::anyhow!("Unknown or already-awaited peer task '{}'", request_id)
            })?
        };

        let result = if let Some(ms) = timeout_ms {
            match tokio::time::timeout(std::time::Duration::from_millis(ms), &mut rx).await {
                Ok(Ok(res)) => Ok(res),
                Ok(Err(_)) => Err(anyhow::anyhow!(
                    "Peer task '{}' result channel closed",
                    request_id
                )),
                Err(_) => {
                    self.pending_results
                        .write()
                        .await
                        .insert(request_id.to_string(), rx);
                    Err(anyhow::anyhow!(
                        "Timed out waiting for peer task '{}'",
                        request_id
                    ))
                }
            }
        } else {
            match rx.await {
                Ok(res) => Ok(res),
                Err(_) => Err(anyhow::anyhow!(
                    "Peer task '{}' result channel closed",
                    request_id
                )),
            }
        };

        let result = result?;
        self.record_completed_result(result.clone()).await;
        Ok(result)
    }

    pub async fn list_tasks(&self) -> Vec<TaskStatusSnapshot> {
        let pending = self.pending_task_states.read().await;
        let mut snapshots: Vec<_> = pending
            .iter()
            .map(|(request_id, pending)| TaskStatusSnapshot {
                request_id: request_id.clone(),
                agent_id: pending.runtime_key.agent_id.clone(),
                slot_id: pending.runtime_key.slot_id.clone(),
                trace_id: pending.trace_id.clone(),
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
            })
            .collect();
        drop(pending);

        let completed = self.completed_results.read().await;
        snapshots.extend(
            completed
                .results
                .values()
                .cloned()
                .map(|result| TaskStatusSnapshot {
                    request_id: result.request_id,
                    agent_id: result.agent_id,
                    slot_id: result.slot_id,
                    trace_id: result.trace_id,
                    state: "completed".to_string(),
                    runtime_task_id: Some(result.runtime_task_id),
                    execution: result.execution,
                    status: Some(result.status),
                    task_turn_count: Some(result.task_turn_count),
                    branch_outcome: result.branch_outcome,
                    promotion_candidate: result.promotion_candidate,
                    promoted_branch: result.promoted_branch,
                    output: result.output,
                    assistant_content: result.assistant_content,
                    error: result.error,
                }),
        );
        snapshots.sort_by(|a, b| a.request_id.cmp(&b.request_id));
        snapshots
    }

    pub async fn get_task(&self, request_id: &str) -> Option<TaskStatusSnapshot> {
        self.list_tasks()
            .await
            .into_iter()
            .find(|task| task.request_id == request_id)
    }

    async fn enqueue_runtime_task(
        &self,
        handle: &Arc<AgentRuntimeHandle>,
        envelope: PeerAgentTaskEnvelope,
    ) -> Result<()> {
        {
            let mut queue = handle
                .queue
                .lock()
                .expect("agent runtime queue mutex poisoned");
            queue.push_back(envelope);
        }
        handle.queued_tasks.fetch_add(1, Ordering::Relaxed);
        handle.notify.notify_one();
        Ok(())
    }

    async fn remove_pending_request(&self, request_id: &str) {
        self.pending_results.write().await.remove(request_id);
        self.pending_task_states.write().await.remove(request_id);
    }

    pub(crate) async fn record_completed_result(&self, result: PeerAgentTaskResult) {
        self.pending_results
            .write()
            .await
            .remove(&result.request_id);
        self.pending_task_states
            .write()
            .await
            .remove(&result.request_id);
        let mut completed = self.completed_results.write().await;
        completed.insert(result);
    }

    pub(super) async fn completed_result(&self, request_id: &str) -> Option<PeerAgentTaskResult> {
        self.completed_results
            .read()
            .await
            .results
            .get(request_id)
            .cloned()
    }

    pub async fn promote_completed_task(
        &self,
        request_id: &str,
        branch_name: Option<&str>,
    ) -> Result<PromotedTaskBranch> {
        let result = self
            .completed_result(request_id)
            .await
            .ok_or_else(|| anyhow!("Task '{}' not found", request_id))?;
        if let Some(branch) = result.promoted_branch {
            return Ok(branch);
        }
        let promotion = result
            .promotion_candidate
            .clone()
            .ok_or_else(|| anyhow!("Task '{}' is not promotable", request_id))?;
        let assistant_content = result
            .assistant_content
            .as_ref()
            .filter(|content| !content.is_empty())
            .ok_or_else(|| anyhow!("Task '{}' has no promotable assistant output", request_id))?;
        let input_content = result
            .promotion_input_content
            .as_ref()
            .filter(|content| !content.is_empty())
            .ok_or_else(|| anyhow!("Task '{}' is missing promotable task input", request_id))?;
        let branch = promote_task_result(
            &self.store_manager,
            &promotion,
            input_content,
            assistant_content,
            Some(request_id),
            branch_name,
        )
        .await?;
        self.completed_results
            .write()
            .await
            .mark_promoted(request_id, branch.clone());
        Ok(branch)
    }

    pub(crate) async fn mark_task_running(&self, request_id: &str, runtime_task_id: String) {
        if let Some(pending) = self.pending_task_states.write().await.get_mut(request_id) {
            pending.state = PendingTaskState::Running;
            pending.runtime_task_id = Some(runtime_task_id);
        }
    }
}
