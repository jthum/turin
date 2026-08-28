use anyhow::{Result, anyhow};

use crate::kernel::event::TaskTerminalStatus;
use crate::kernel::task_promotion::{TaskPromotionSelection, promote_task_result};

use super::task_status::{
    completed_task_fingerprint, completed_task_snapshot, pending_task_fingerprint,
    pending_task_snapshot,
};
use super::{
    AgentManager, PeerAgentTaskResult, PendingTaskState, PromotedTaskBranch, TaskStatusFingerprint,
    TaskStatusSnapshot,
};

impl AgentManager {
    /// Await a previously submitted peer task result.
    pub(super) async fn await_result_inner(
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
                Ok(Err(_)) => {
                    self.record_lost_task_result(request_id, "Peer task result channel closed")
                        .await;
                    Err(anyhow::anyhow!(
                        "Peer task '{}' result channel closed",
                        request_id
                    ))
                }
                Err(_) => match rx.try_recv() {
                    Ok(result) => Ok(result),
                    Err(tokio::sync::oneshot::error::TryRecvError::Closed) => {
                        self.record_lost_task_result(request_id, "Peer task result channel closed")
                            .await;
                        Err(anyhow::anyhow!(
                            "Peer task '{}' result channel closed",
                            request_id
                        ))
                    }
                    Err(tokio::sync::oneshot::error::TryRecvError::Empty) => {
                        let completed = {
                            let mut pending_results = self.pending_results.write().await;
                            let completed = self
                                .completed_results
                                .read()
                                .await
                                .results
                                .get(request_id)
                                .cloned();
                            if completed.is_none() {
                                pending_results.insert(request_id.to_string(), rx);
                            }
                            completed
                        };
                        completed.ok_or_else(|| {
                            anyhow::anyhow!("Timed out waiting for peer task '{}'", request_id)
                        })
                    }
                },
            }
        } else {
            match rx.await {
                Ok(res) => Ok(res),
                Err(_) => {
                    self.record_lost_task_result(request_id, "Peer task result channel closed")
                        .await;
                    Err(anyhow::anyhow!(
                        "Peer task '{}' result channel closed",
                        request_id
                    ))
                }
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
            .map(|(request_id, pending)| pending_task_snapshot(request_id, pending))
            .collect();
        drop(pending);

        let completed = self.completed_results.read().await;
        snapshots.extend(completed.results.values().map(completed_task_snapshot));
        snapshots.sort_by(|a, b| a.request_id.cmp(&b.request_id));
        snapshots
    }

    pub(crate) async fn list_task_fingerprints(&self) -> Vec<TaskStatusFingerprint> {
        let pending = self.pending_task_states.read().await;
        let mut fingerprints: Vec<_> = pending
            .iter()
            .map(|(request_id, pending)| pending_task_fingerprint(request_id, pending))
            .collect();
        drop(pending);

        let completed = self.completed_results.read().await;
        fingerprints.extend(completed.results.values().map(completed_task_fingerprint));
        fingerprints.sort_by(|a, b| a.request_id.cmp(&b.request_id));
        fingerprints
    }

    pub async fn get_task(&self, request_id: &str) -> Option<TaskStatusSnapshot> {
        {
            let pending = self.pending_task_states.read().await;
            if let Some(task) = pending.get(request_id) {
                return Some(pending_task_snapshot(request_id, task));
            }
        }

        let completed = self.completed_results.read().await;
        completed
            .results
            .get(request_id)
            .map(completed_task_snapshot)
    }

    async fn record_lost_task_result(&self, request_id: &str, reason: &str) {
        let Some(pending) = self
            .pending_task_states
            .read()
            .await
            .get(request_id)
            .cloned()
        else {
            return;
        };
        self.record_completed_result(pending.into_terminal_result(
            request_id.to_string(),
            TaskTerminalStatus::Error,
            reason,
        ))
        .await;
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

    pub(super) async fn promote_completed_task_inner(
        &self,
        request_id: &str,
        branch_name: Option<&str>,
        source_turn_id: Option<i64>,
    ) -> Result<PromotedTaskBranch> {
        let _promotion_guard = self.task_promotion.lock().await;
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
        let assistant_content = result.assistant_content.as_deref().unwrap_or_default();
        let input_content = result
            .promotion_input_content
            .as_deref()
            .unwrap_or_default();
        let branch = promote_task_result(
            &self.store_manager,
            &promotion,
            input_content,
            assistant_content,
            Some(request_id),
            branch_name,
            source_turn_id
                .map(TaskPromotionSelection::LinkedTurn)
                .unwrap_or(TaskPromotionSelection::Result),
        )
        .await?;
        self.completed_results
            .write()
            .await
            .mark_promoted(request_id, branch.clone());
        Ok(branch)
    }

    pub(crate) async fn mark_task_running(
        &self,
        request_id: &str,
        runtime_task_id: String,
        session_id: Option<String>,
    ) -> bool {
        if let Some(pending) = self.pending_task_states.write().await.get_mut(request_id) {
            let cancellation_requested = pending.state == PendingTaskState::Cancelling;
            if !cancellation_requested {
                pending.state = PendingTaskState::Running;
            }
            pending.runtime_task_id = Some(runtime_task_id);
            pending.session_target.session_id = session_id;
            cancellation_requested
        } else {
            false
        }
    }
}
