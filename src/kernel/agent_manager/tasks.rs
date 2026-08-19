use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::{Context, Result, anyhow};
use tokio::sync::oneshot;

use crate::kernel::session::{ExecutionContext, ExecutionStatusSnapshot, QueuedTask};
use crate::kernel::session_refs::{format_session_reference, parse_session_reference};
use crate::kernel::task_promotion::promote_task_result;
use crate::persistence::schema::LinkedSessionCreate;

use super::{
    AgentManager, AgentRuntimeHandle, PeerAgentTaskEnvelope, PeerAgentTaskResult,
    PendingTaskRecord, PendingTaskState, PromotedTaskBranch, PromotedTaskBranchFingerprint,
    RuntimeSlotKey, SessionContextOverrides, TaskBranchOutcomeFingerprint, TaskStatusFingerprint,
    TaskStatusSnapshot, task_prompt_preview,
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

fn pending_task_snapshot(request_id: &str, pending: &PendingTaskRecord) -> TaskStatusSnapshot {
    TaskStatusSnapshot {
        request_id: request_id.to_string(),
        agent_id: pending.runtime_key.agent_id.clone(),
        slot_id: pending.runtime_key.slot_id.clone(),
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

fn completed_task_snapshot(result: &PeerAgentTaskResult) -> TaskStatusSnapshot {
    TaskStatusSnapshot {
        request_id: result.request_id.clone(),
        agent_id: result.agent_id.clone(),
        slot_id: result.slot_id.clone(),
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

fn pending_task_fingerprint(
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

fn completed_task_fingerprint(result: &PeerAgentTaskResult) -> TaskStatusFingerprint {
    let assistant_content = result.assistant_content.as_deref().unwrap_or_default();
    TaskStatusFingerprint {
        request_id: result.request_id.clone(),
        state: "completed",
        runtime_task_id: Some(result.runtime_task_id.clone()),
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

    /// Submit into an agent-owned child session scoped to the originating session.
    pub async fn submit_linked(
        self: &Arc<Self>,
        origin_session_id: &str,
        origin_turn_id: Option<i64>,
        agent_id: &str,
        thread_key: &str,
        task: QueuedTask,
        delegated_capabilities: Option<BTreeMap<String, bool>>,
    ) -> Result<String> {
        let thread_key = thread_key.trim();
        anyhow::ensure!(!thread_key.is_empty(), "Peer thread key must not be empty");
        anyhow::ensure!(
            thread_key.chars().count() <= 120,
            "Peer thread key must be at most 120 characters"
        );

        let session_ref = parse_session_reference(origin_session_id)?;
        let state_selector = session_ref
            .store_selector
            .unwrap_or(self.config.persistence.top_level_state_selector()?);
        let store = self.store_manager.open(&state_selector).await?;
        let parent_public_id = uuid::Uuid::parse_str(&session_ref.public_id)?;
        let parent = store
            .get_session_row_by_public_id(parent_public_id)
            .await?
            .ok_or_else(|| anyhow!("Origin session '{}' not found", origin_session_id))?;
        let parent_session_reference =
            format_session_reference(&session_ref.public_id, &state_selector);
        let runtime_key =
            RuntimeSlotKey::linked_for(agent_id, &parent_session_reference, thread_key);

        let handle = if let Some(linked) = store
            .find_linked_session(parent.id, agent_id, thread_key)
            .await?
        {
            let public_id = uuid::Uuid::from_slice(&linked.public_id)
                .context("Linked session has an invalid public id")?
                .simple()
                .to_string();
            self.ensure_runtime_slot_resumed(
                runtime_key.clone(),
                format_session_reference(&public_id, &state_selector),
                SessionContextOverrides::default(),
            )
            .await?
        } else {
            self.ensure_runtime_slot_linked(
                runtime_key.clone(),
                state_selector,
                None,
                SessionContextOverrides::default(),
                LinkedSessionCreate {
                    parent_session_id: parent.id,
                    origin_turn_id,
                    relation_kind: "delegated".to_string(),
                    thread_key: thread_key.to_string(),
                    visibility: "contextual".to_string(),
                },
            )
            .await?
        };

        self.submit_to_handle(runtime_key, handle, task, delegated_capabilities)
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
        let handle = self.ensure_runtime_slot(runtime_key.clone()).await?;
        self.submit_to_handle(runtime_key, handle, task, delegated_capabilities)
            .await
    }

    async fn submit_to_handle(
        &self,
        runtime_key: RuntimeSlotKey,
        handle: Arc<AgentRuntimeHandle>,
        task: QueuedTask,
        delegated_capabilities: Option<BTreeMap<String, bool>>,
    ) -> Result<String> {
        let trace_id = task.trace_id.clone();
        let title = task.title.clone();
        let prompt_preview = task_prompt_preview(&task.prompt);
        let request_id = uuid::Uuid::now_v7().simple().to_string();
        let (tx_result, rx_result) = oneshot::channel();
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
                    title,
                    prompt_preview,
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

    async fn enqueue_runtime_task(
        &self,
        handle: &Arc<AgentRuntimeHandle>,
        envelope: PeerAgentTaskEnvelope,
    ) -> Result<()> {
        if self.shutting_down.load(Ordering::Acquire) {
            anyhow::bail!("Agent manager is shutting down");
        }
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
