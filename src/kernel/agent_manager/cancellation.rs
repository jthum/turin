use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::Result;
use tracing::warn;

use crate::kernel::event::TaskTerminalStatus;
use crate::kernel::session::{ExecutionContext, ExecutionStatusSnapshot, ExecutionWritePolicy};
use crate::kernel::session_refs::{format_session_reference, parse_session_reference};

use super::task_status::completed_task_snapshot;
use super::{
    AgentManager, AgentRuntimeHandle, PeerAgentTaskResult, PendingTaskRecord, PendingTaskState,
    RuntimeSlotKey, TaskStatusSnapshot,
};

fn default_execution_snapshot() -> ExecutionStatusSnapshot {
    ExecutionStatusSnapshot::from_execution(
        &ExecutionContext::new(),
        ExecutionWritePolicy::AdvanceBranchHead,
    )
}

impl AgentManager {
    pub(super) const SHUTDOWN_GRACE: Duration = Duration::from_secs(6);

    /// Stop accepting work, cancel queued and active tasks, and bound runtime cleanup.
    pub(crate) async fn shutdown(&self) {
        self.shutdown_with_grace(Self::SHUTDOWN_GRACE).await;
    }

    pub(super) async fn shutdown_with_grace(&self, grace: Duration) {
        self.shutting_down.store(true, Ordering::Release);

        let runtimes: Vec<_> = self
            .runtimes
            .read()
            .await
            .iter()
            .map(|(key, handle)| (key.clone(), Arc::clone(handle)))
            .collect();

        for (runtime_key, handle) in &runtimes {
            self.cancel_queued_requests_for_runtime(
                runtime_key,
                "Runtime shutting down before execution",
            )
            .await;
            handle.control.request_task_cancel();
            handle.shutdown_token.cancel();
            handle.notify.notify_one();
        }

        let deadline = tokio::time::Instant::now() + grace;
        while runtimes.iter().any(|(_, handle)| handle.is_running()) {
            let now = tokio::time::Instant::now();
            if now >= deadline {
                break;
            }
            tokio::time::sleep((deadline - now).min(Duration::from_millis(10))).await;
        }

        // Close the narrow race where a submission passed its shutdown check just
        // before the manager gate closed but reached the queue after the first drain.
        for (runtime_key, _) in &runtimes {
            self.cancel_queued_requests_for_runtime(
                runtime_key,
                "Runtime shutting down before execution",
            )
            .await;
        }

        let stalled: Vec<_> = runtimes
            .iter()
            .filter(|(_, handle)| handle.is_running())
            .collect();
        for (runtime_key, handle) in &stalled {
            warn!(
                agent_id = %runtime_key.agent_id,
                slot_id = %runtime_key.slot_id,
                "Peer runtime exceeded shutdown grace period; aborting"
            );
            if let Some(task) = &handle.task {
                task.abort();
            }
        }
        tokio::task::yield_now().await;

        for (runtime_key, handle) in stalled {
            if handle.control.current_request_id().is_some() {
                self.kill_runtime_requests(
                    runtime_key,
                    handle,
                    "Runtime killed after shutdown grace period",
                )
                .await;
            }
        }

        self.runtimes.write().await.clear();
        tokio::task::yield_now().await;
    }

    pub async fn cancel_task(&self, request_id: &str) -> Result<TaskStatusSnapshot> {
        self.cancel_task_with_reason(request_id, "Task cancelled before execution")
            .await
    }

    async fn cancel_task_with_reason(
        &self,
        request_id: &str,
        queued_reason: &str,
    ) -> Result<TaskStatusSnapshot> {
        loop {
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

            if pending.state == PendingTaskState::Queued {
                match self
                    .complete_queued_task(request_id, TaskTerminalStatus::Cancelled, queued_reason)
                    .await
                {
                    Ok(snapshot) => return Ok(snapshot),
                    Err(error) => {
                        let state = self
                            .pending_task_states
                            .read()
                            .await
                            .get(request_id)
                            .map(|pending| pending.state);
                        if matches!(
                            state,
                            Some(PendingTaskState::Running | PendingTaskState::Cancelling)
                        ) {
                            continue;
                        }
                        return Err(error);
                    }
                }
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

            if let Some(pending) = self.pending_task_states.write().await.get_mut(request_id) {
                pending.state = PendingTaskState::Cancelling;
            }
            if handle.control.current_request_id().as_deref() == Some(request_id) {
                handle.control.request_task_cancel();
            }
            return self.get_task(request_id).await.ok_or_else(|| {
                anyhow::anyhow!("Task '{}' disappeared after cancellation", request_id)
            });
        }
    }

    pub async fn cancel_session(
        &self,
        session_id: &str,
        slot_id: Option<&str>,
    ) -> Result<(String, String, String)> {
        let (runtime_key, handle, matching_pending) = self
            .logical_session_runtime_target(session_id, slot_id)
            .await?;
        for (request_id, _) in matching_pending {
            self.cancel_task_with_reason(&request_id, "Session cancelled before execution")
                .await?;
        }

        if !runtime_key.is_linked() {
            let had_active_request = handle.control.current_request_id().is_some();
            if !handle.control.request_session_cancel() && had_active_request {
                anyhow::bail!(
                    "Session '{}' has no cancellable active execution",
                    session_id
                );
            }
            handle.notify.notify_one();
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
        let (runtime_key, handle, matching_pending) = self
            .logical_session_runtime_target(session_id, slot_id)
            .await?;
        let current_is_target = handle.control.current_session_id().is_some_and(|current| {
            crate::kernel::session_refs::session_references_match(&current, session_id)
        });

        if !current_is_target {
            for (request_id, pending) in matching_pending {
                if pending.state != PendingTaskState::Queued {
                    anyhow::bail!(
                        "Session '{}' changed runtime state while force-kill was requested",
                        session_id
                    );
                }
                self.complete_queued_task(
                    &request_id,
                    TaskTerminalStatus::Killed,
                    "Session killed before execution",
                )
                .await?;
            }
            return Ok((
                runtime_key.agent_id,
                runtime_key.slot_id,
                session_id.to_string(),
            ));
        }

        if runtime_key.is_linked() {
            let has_unrelated_queued_work = handle
                .queue
                .lock()
                .expect("agent runtime queue mutex poisoned")
                .iter()
                .any(|envelope| !envelope.session_target.matches_session(session_id));
            if has_unrelated_queued_work {
                anyhow::bail!(
                    "Cannot force-kill session '{}' while linked runtime slot '{}' has unrelated queued work; cancel the session or wait for the lane to drain",
                    session_id,
                    runtime_key.slot_id
                );
            }
        }

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

    pub async fn cancel_session_family(&self, session_id: &str) -> Result<(String, String, usize)> {
        let config = self.config_snapshot();
        let session_ref = parse_session_reference(session_id)?;
        let store_selector = session_ref
            .store_selector
            .unwrap_or(config.persistence.top_level_state_selector()?);
        let store = self.store_manager.open(&store_selector).await?;
        let public_id = uuid::Uuid::parse_str(&session_ref.public_id)?;
        let parent = store
            .get_session_row_by_public_id(public_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Session '{}' not found", session_id))?;
        let mut family = store.list_linked_session_descendants(parent.id).await?;
        family.push(parent.clone());
        let family_session_ids = family
            .iter()
            .map(|row| {
                let public_id = uuid::Uuid::from_slice(&row.public_id)?.simple().to_string();
                Ok(format_session_reference(&public_id, &store_selector))
            })
            .collect::<Result<Vec<_>>>()?;
        let family_persisted_ids = family
            .iter()
            .map(|row| (store_selector.clone(), row.id))
            .collect::<std::collections::HashSet<_>>();
        let request_ids = self
            .pending_task_states
            .read()
            .await
            .iter()
            .filter(|(_, pending)| {
                pending
                    .session_target
                    .belongs_to_family(&family_session_ids, &family_persisted_ids)
            })
            .map(|(request_id, _)| request_id.clone())
            .collect::<Vec<_>>();

        self.cancel_pending_requests(&request_ids).await?;

        let runtimes = self.runtimes.read().await;
        for (runtime_key, handle) in runtimes.iter() {
            let current_is_family = handle.control.current_session_id().is_some_and(|current| {
                family_session_ids.iter().any(|family_session_id| {
                    crate::kernel::session_refs::session_references_match(
                        &current,
                        family_session_id,
                    )
                })
            });
            if current_is_family && !runtime_key.is_linked() {
                handle.control.request_session_cancel();
                handle.notify.notify_one();
            }
        }

        Ok((parent.agent_id, session_id.to_string(), request_ids.len()))
    }

    pub(super) async fn cancel_pending_requests(&self, request_ids: &[String]) -> Result<()> {
        for request_id in request_ids {
            if let Err(error) = self.cancel_task(request_id).await {
                // The cancellation sweep works from a snapshot. Completion may win the
                // race after that snapshot; terminal or already-removed work is settled.
                let is_settled = self.completed_result(request_id).await.is_some()
                    || !self
                        .pending_task_states
                        .read()
                        .await
                        .contains_key(request_id);
                if !is_settled {
                    return Err(error);
                }
            }
        }
        Ok(())
    }

    async fn logical_session_runtime_target(
        &self,
        session_id: &str,
        slot_id: Option<&str>,
    ) -> Result<(
        RuntimeSlotKey,
        Arc<AgentRuntimeHandle>,
        Vec<(String, PendingTaskRecord)>,
    )> {
        let pending_matches: Vec<_> = self
            .pending_task_states
            .read()
            .await
            .iter()
            .filter(|(_, pending)| pending.session_target.matches_session(session_id))
            .map(|(request_id, pending)| (request_id.clone(), pending.clone()))
            .collect();
        let mut matches = self.find_runtimes_by_session(session_id).await;
        let runtimes = self.runtimes.read().await;
        for (_, pending) in &pending_matches {
            if !matches
                .iter()
                .any(|(runtime_key, _)| runtime_key == &pending.runtime_key)
                && let Some(handle) = runtimes.get(&pending.runtime_key)
            {
                matches.push((pending.runtime_key.clone(), Arc::clone(handle)));
            }
        }
        drop(runtimes);

        if let Some(slot_id) = slot_id {
            matches.retain(|(runtime_key, _)| runtime_key.slot_id == slot_id);
        }
        if matches.is_empty() {
            anyhow::bail!(
                "Session '{}' is not an active or queued managed runtime session",
                session_id
            );
        }
        if slot_id.is_none() && matches.len() > 1 {
            anyhow::bail!(
                "Session '{}' is active or queued in multiple runtime slots; specify slot_id",
                session_id
            );
        }
        let (runtime_key, handle) = matches
            .into_iter()
            .next()
            .expect("non-empty matching runtime set");
        let pending_matches = pending_matches
            .into_iter()
            .filter(|(_, pending)| pending.runtime_key == runtime_key)
            .collect();
        Ok((runtime_key, handle, pending_matches))
    }

    async fn complete_queued_task(
        &self,
        request_id: &str,
        status: TaskTerminalStatus,
        reason: &str,
    ) -> Result<TaskStatusSnapshot> {
        let pending_snapshot = self
            .pending_task_states
            .read()
            .await
            .get(request_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Task '{}' not found", request_id))?;
        let handle = self
            .runtimes
            .read()
            .await
            .get(&pending_snapshot.runtime_key)
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Agent runtime '{}' [{}] is not available",
                    pending_snapshot.runtime_key.agent_id,
                    pending_snapshot.runtime_key.slot_id
                )
            })?;
        let (pending, envelope) = {
            let pending = self.pending_task_states.write().await;
            let current = pending
                .get(request_id)
                .filter(|pending| pending.state == PendingTaskState::Queued)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("Task '{}' is no longer queued", request_id))?;
            let mut queue = handle
                .queue
                .lock()
                .expect("agent runtime queue mutex poisoned");
            let index = queue
                .iter()
                .position(|envelope| envelope.request_id.as_deref() == Some(request_id))
                .ok_or_else(|| anyhow::anyhow!("Task '{}' is no longer queued", request_id))?;
            let envelope = queue
                .remove(index)
                .expect("queued task index should remain valid while queue is locked");
            handle.queued_tasks.store(queue.len(), Ordering::Relaxed);
            (current, envelope)
        };

        let completed = pending.into_terminal_result(request_id.to_string(), status, reason);
        if let Some(tx_result) = envelope.result_tx {
            let _ = tx_result.send(completed.clone());
        }
        self.record_completed_result(completed.clone()).await;

        Ok(completed_task_snapshot(&completed))
    }

    pub(super) async fn cancel_queued_requests_for_runtime(
        &self,
        runtime_key: &RuntimeSlotKey,
        reason: &str,
    ) {
        let Some(handle) = self.runtimes.read().await.get(runtime_key).cloned() else {
            return;
        };

        let drained: Vec<_> = {
            let mut queue = handle
                .queue
                .lock()
                .expect("agent runtime queue mutex poisoned");
            let drained = queue.drain(..).collect();
            handle.queued_tasks.store(queue.len(), Ordering::Relaxed);
            drained
        };
        if drained.is_empty() {
            return;
        }
        for envelope in drained {
            let (tx_result, completed) = envelope.into_terminal_result(
                runtime_key,
                handle
                    .control
                    .current_execution()
                    .unwrap_or_else(default_execution_snapshot),
                TaskTerminalStatus::Cancelled,
                reason,
            );
            if let Some(tx_result) = tx_result {
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
            let drained = queue.drain(..).collect();
            handle.queued_tasks.store(queue.len(), Ordering::Relaxed);
            drained
        };

        for envelope in drained {
            let (tx_result, completed) = envelope.into_terminal_result(
                runtime_key,
                handle
                    .control
                    .current_execution()
                    .unwrap_or_else(default_execution_snapshot),
                TaskTerminalStatus::Killed,
                reason,
            );
            if let Some(tx_result) = tx_result {
                let _ = tx_result.send(completed.clone());
            }
            self.record_completed_result(completed).await;
        }

        if let Some(request_id) = handle.control.current_request_id() {
            let description = self
                .pending_task_states
                .read()
                .await
                .get(&request_id)
                .map(|pending| {
                    (
                        pending.trace_id.clone(),
                        pending.title.clone(),
                        pending.prompt_preview.clone(),
                    )
                })
                .unwrap_or_default();
            let completed = PeerAgentTaskResult {
                request_id: request_id.clone(),
                agent_id: runtime_key.agent_id.clone(),
                slot_id: runtime_key.slot_id.clone(),
                session_id: handle.control.current_session_id(),
                trace_id: description.0,
                title: description.1,
                prompt_preview: description.2,
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
