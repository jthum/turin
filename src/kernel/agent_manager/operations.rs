use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::{Context, Result, anyhow};
use tokio::sync::oneshot;

use crate::kernel::config::InferenceOverrideConfig;
use crate::kernel::event::KernelEvent;
use crate::kernel::session::{
    ExecutionContextTarget, ExecutionDurability, ExecutionVisibility, ExecutionWritePolicy,
    QueuedTask,
};
use crate::kernel::session_refs::parse_session_reference;
use crate::kernel::task_promotion::promote_task_result;
use crate::persistence::manager::StoreSelector;

use super::{
    AgentManager, AgentRuntimeHandle, AgentStatusSnapshot, ExecutionStatusSnapshot,
    LiveSessionSnapshot, PeerAgentTaskEnvelope, PeerAgentTaskResult, PendingTaskRecord,
    PendingTaskState, PromotedTaskBranch, RuntimeSlotKey, TaskStatusSnapshot,
};

fn live_execution_snapshot(handle: &Arc<AgentRuntimeHandle>) -> ExecutionStatusSnapshot {
    handle
        .control
        .current_execution()
        .unwrap_or(ExecutionStatusSnapshot {
            execution_id: String::new(),
            context_target: ExecutionContextTarget::BranchHead {
                branch_head_id: None,
            },
            visibility: ExecutionVisibility::Visible,
            durability: ExecutionDurability::Durable,
            write_policy: ExecutionWritePolicy::AdvanceBranchHead,
        })
}

impl AgentManager {
    /// Dispatch a task to an agent by ID. If the agent isn't running, it will be started automatically.
    pub async fn send(
        self: &Arc<Self>,
        agent_id: &str,
        task: QueuedTask,
        delegated_capabilities: Option<BTreeMap<String, bool>>,
    ) -> Result<()> {
        let handle = self.ensure_runtime(agent_id).await?;
        self.enqueue_runtime_task(
            &handle,
            PeerAgentTaskEnvelope {
                task,
                request_id: None,
                result_tx: None,
                delegated_capabilities,
            },
        )
        .await?;
        Ok(())
    }

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

    pub async fn open_session(
        self: &Arc<Self>,
        agent_id: &str,
        slot_id: Option<&str>,
        initial_state_selector: Option<StoreSelector>,
        initial_default_store_selector: Option<StoreSelector>,
        channel_id: Option<String>,
        initial_inference: InferenceOverrideConfig,
    ) -> Result<LiveSessionSnapshot> {
        let runtime_key = RuntimeSlotKey {
            agent_id: agent_id.to_string(),
            slot_id: slot_id
                .map(str::to_string)
                .unwrap_or_else(|| format!("sl_{}", uuid::Uuid::now_v7().simple())),
        };
        let handle = self
            .ensure_runtime_slot_in_store(
                runtime_key.clone(),
                initial_state_selector,
                initial_default_store_selector,
                super::SessionContextOverrides {
                    channel_id,
                    inference: initial_inference,
                },
            )
            .await?;
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
        let session_id = loop {
            if let Some(session_id) = handle.control.current_session_id() {
                break session_id;
            }
            if tokio::time::Instant::now() >= deadline {
                anyhow::bail!(
                    "Agent runtime '{}' [{}] did not expose a live session",
                    runtime_key.agent_id,
                    runtime_key.slot_id
                );
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        };
        Ok(LiveSessionSnapshot {
            agent_id: runtime_key.agent_id,
            slot_id: runtime_key.slot_id,
            session_id,
            running: handle.is_running(),
            active_tasks: handle.active_tasks.load(Ordering::Relaxed),
            queued_tasks: handle.queued_tasks.load(Ordering::Relaxed),
            current_request_id: handle.control.current_request_id(),
            execution: live_execution_snapshot(&handle),
            conflict_policy: handle.control.current_conflict_policy(),
        })
    }

    pub async fn resume_session(
        self: &Arc<Self>,
        session_id: &str,
        slot_id: Option<&str>,
        channel_id: Option<String>,
        initial_inference: InferenceOverrideConfig,
    ) -> Result<LiveSessionSnapshot> {
        let live_matches = self.find_runtimes_by_session(session_id).await;
        if let Some(requested_slot_id) = slot_id {
            if let Some((runtime_key, handle)) = live_matches
                .iter()
                .find(|(runtime_key, _)| runtime_key.slot_id == requested_slot_id)
                .cloned()
            {
                return Ok(LiveSessionSnapshot {
                    agent_id: runtime_key.agent_id,
                    slot_id: runtime_key.slot_id,
                    session_id: session_id.to_string(),
                    running: handle.is_running(),
                    active_tasks: handle.active_tasks.load(Ordering::Relaxed),
                    queued_tasks: handle.queued_tasks.load(Ordering::Relaxed),
                    current_request_id: handle.control.current_request_id(),
                    execution: live_execution_snapshot(&handle),
                    conflict_policy: handle.control.current_conflict_policy(),
                });
            }
        } else {
            match live_matches.as_slice() {
                [] => {}
                [(runtime_key, handle)] => {
                    return Ok(LiveSessionSnapshot {
                        agent_id: runtime_key.agent_id.clone(),
                        slot_id: runtime_key.slot_id.clone(),
                        session_id: session_id.to_string(),
                        running: handle.is_running(),
                        active_tasks: handle.active_tasks.load(Ordering::Relaxed),
                        queued_tasks: handle.queued_tasks.load(Ordering::Relaxed),
                        current_request_id: handle.control.current_request_id(),
                        execution: live_execution_snapshot(handle),
                        conflict_policy: handle.control.current_conflict_policy(),
                    });
                }
                _ => {
                    anyhow::bail!(
                        "Session '{}' is active in multiple runtime slots; specify slot_id",
                        session_id
                    );
                }
            }
        }

        let session_ref = parse_session_reference(session_id)?;
        let public_id = uuid::Uuid::parse_str(&session_ref.public_id)
            .with_context(|| format!("Invalid session id '{}'", session_ref.public_id))?;
        let store_selector = session_ref
            .store_selector
            .unwrap_or_else(|| StoreSelector::Alias("state".to_string()));
        let store = self.store_manager.open(&store_selector).await?;
        let row = store
            .get_session_row_by_public_id(public_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Session '{}' not found", session_id))?;
        let agent_id = row.agent_id.clone();

        let runtime_key = RuntimeSlotKey {
            agent_id: agent_id.clone(),
            slot_id: slot_id
                .map(str::to_string)
                .unwrap_or_else(|| format!("sl_{}", uuid::Uuid::now_v7().simple())),
        };

        if agent_id != self.config.agent.id {
            self.config
                .agents
                .get(&agent_id)
                .ok_or_else(|| anyhow::anyhow!("Unknown agent profile '{}'", agent_id))?;
        }

        let existing = {
            let runtimes = self.runtimes.read().await;
            runtimes.get(&runtime_key).cloned()
        };

        let handle = if let Some(handle) = existing {
            if handle.active_tasks.load(Ordering::Relaxed) > 0
                || handle.queued_tasks.load(Ordering::Relaxed) > 0
            {
                anyhow::bail!(
                    "Runtime slot '{}' for agent '{}' is busy",
                    runtime_key.slot_id,
                    runtime_key.agent_id
                );
            }
            handle.control.request_session_resume(
                session_id.to_string(),
                super::SessionContextOverrides {
                    channel_id,
                    inference: initial_inference,
                },
            );
            handle.notify.notify_one();
            handle
        } else {
            self.ensure_runtime_slot_resumed(
                runtime_key.clone(),
                session_id.to_string(),
                super::SessionContextOverrides {
                    channel_id,
                    inference: initial_inference,
                },
            )
            .await?
        };

        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            if handle.control.current_session_id().as_deref() == Some(session_id) {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                anyhow::bail!(
                    "Agent runtime '{}' [{}] did not resume session '{}'",
                    runtime_key.agent_id,
                    runtime_key.slot_id,
                    session_id
                );
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }

        Ok(LiveSessionSnapshot {
            agent_id: runtime_key.agent_id,
            slot_id: runtime_key.slot_id,
            session_id: session_id.to_string(),
            running: handle.is_running(),
            active_tasks: handle.active_tasks.load(Ordering::Relaxed),
            queued_tasks: handle.queued_tasks.load(Ordering::Relaxed),
            current_request_id: handle.control.current_request_id(),
            execution: live_execution_snapshot(&handle),
            conflict_policy: handle.control.current_conflict_policy(),
        })
    }

    pub async fn reload_session(
        self: &Arc<Self>,
        session_id: &str,
        slot_id: Option<&str>,
    ) -> Result<LiveSessionSnapshot> {
        let (runtime_key, handle) = self.runtime_by_session_target(session_id, slot_id).await?;

        if handle.active_tasks.load(Ordering::Relaxed) > 0
            || handle.queued_tasks.load(Ordering::Relaxed) > 0
        {
            anyhow::bail!(
                "Runtime slot '{}' for agent '{}' is busy",
                runtime_key.slot_id,
                runtime_key.agent_id
            );
        }

        let wanted = parse_session_reference(session_id)
            .map(|session_ref| session_ref.public_id)
            .unwrap_or_else(|_| session_id.to_string());
        let generation = handle.control.session_generation();
        let context = handle.control.current_session_context();
        handle
            .control
            .request_session_resume(session_id.to_string(), context);
        handle.notify.notify_one();

        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(2);
        loop {
            let current_matches = handle
                .control
                .current_session_id()
                .as_deref()
                .map(|current| {
                    parse_session_reference(current)
                        .map(|session_ref| session_ref.public_id == wanted)
                        .unwrap_or(current == wanted)
                })
                .unwrap_or(false);
            if current_matches && handle.control.session_generation() > generation {
                break;
            }
            if tokio::time::Instant::now() >= deadline {
                anyhow::bail!(
                    "Agent runtime '{}' [{}] did not reload session '{}'",
                    runtime_key.agent_id,
                    runtime_key.slot_id,
                    session_id
                );
            }
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }

        Ok(LiveSessionSnapshot {
            agent_id: runtime_key.agent_id,
            slot_id: runtime_key.slot_id,
            session_id: handle
                .control
                .current_session_id()
                .unwrap_or_else(|| session_id.to_string()),
            running: handle.is_running(),
            active_tasks: handle.active_tasks.load(Ordering::Relaxed),
            queued_tasks: handle.queued_tasks.load(Ordering::Relaxed),
            current_request_id: handle.control.current_request_id(),
            execution: live_execution_snapshot(&handle),
            conflict_policy: handle.control.current_conflict_policy(),
        })
    }

    pub async fn reload_session_if_live(
        self: &Arc<Self>,
        session_id: &str,
        slot_id: Option<&str>,
    ) -> Result<bool> {
        let live_matches = self.find_runtimes_by_session(session_id).await;
        if let Some(slot_id) = slot_id {
            if !live_matches
                .iter()
                .any(|(runtime_key, _)| runtime_key.slot_id == slot_id)
            {
                return Ok(false);
            }
            self.reload_session(session_id, Some(slot_id)).await?;
            return Ok(true);
        }

        if live_matches.is_empty() {
            return Ok(false);
        }
        self.reload_session(session_id, None).await?;
        Ok(true)
    }

    pub async fn subscribe_session_events(
        &self,
        session_id: &str,
        slot_id: Option<&str>,
    ) -> Option<(
        String,
        String,
        tokio::sync::broadcast::Receiver<(Option<i64>, KernelEvent)>,
    )> {
        match self.runtime_by_session_target(session_id, slot_id).await {
            Ok((runtime_key, handle)) => handle
                .control
                .subscribe_current_session_events()
                .map(|receiver| (runtime_key.agent_id, runtime_key.slot_id, receiver)),
            Err(_) => None,
        }
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
                },
            );
        }

        let handle = match self.ensure_runtime_slot(runtime_key).await {
            Ok(handle) => handle,
            Err(e) => {
                self.remove_pending_request(&request_id).await;
                return Err(e);
            }
        };

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

    /// List configured agents with runtime status.
    pub async fn list_statuses(&self) -> Vec<AgentStatusSnapshot> {
        let runtimes = self.runtimes.read().await;
        let pending = self.pending_task_states.read().await;
        let mut awaiting_by_agent: HashMap<&str, usize> = HashMap::new();
        for pending in pending.values() {
            *awaiting_by_agent
                .entry(pending.runtime_key.agent_id.as_str())
                .or_default() += 1;
        }

        let mut ids = vec![self.config.agent.id.clone()];
        ids.extend(self.config.agents.keys().cloned());
        ids.sort();
        ids.dedup();

        ids.into_iter()
            .map(|agent_id| {
                let matching: Vec<_> = runtimes
                    .iter()
                    .filter(|(key, _)| key.agent_id == agent_id)
                    .collect();
                let running = matching.iter().any(|(_, h)| h.is_running());
                let awaiting_results = *awaiting_by_agent.get(agent_id.as_str()).unwrap_or(&0);
                let queued_tasks = matching
                    .iter()
                    .map(|(_, h)| h.queued_tasks.load(Ordering::Relaxed))
                    .sum();
                let active_tasks = matching
                    .iter()
                    .map(|(_, h)| h.active_tasks.load(Ordering::Relaxed))
                    .sum();
                let default_handle = runtimes.get(&RuntimeSlotKey::default_for(&agent_id));
                let single_handle = if matching.len() == 1 {
                    matching.first().map(|(_, h)| *h)
                } else {
                    None
                };
                let display_handle = default_handle.or(single_handle);
                AgentStatusSnapshot {
                    agent_id,
                    running,
                    active_tasks,
                    queued_tasks,
                    awaiting_results,
                    current_session_id: display_handle.and_then(|h| h.control.current_session_id()),
                    current_request_id: display_handle.and_then(|h| h.control.current_request_id()),
                }
            })
            .collect()
    }

    pub async fn list_live_sessions(&self, agent_id: Option<&str>) -> Vec<LiveSessionSnapshot> {
        let runtimes = self.runtimes.read().await;
        let mut sessions: Vec<_> = runtimes
            .iter()
            .filter_map(|(runtime_key, handle)| {
                if agent_id.is_some_and(|wanted| runtime_key.agent_id != wanted) {
                    return None;
                }
                let session_id = handle.control.current_session_id()?;
                Some(LiveSessionSnapshot {
                    agent_id: runtime_key.agent_id.clone(),
                    slot_id: runtime_key.slot_id.clone(),
                    session_id,
                    running: handle.is_running(),
                    active_tasks: handle.active_tasks.load(Ordering::Relaxed),
                    queued_tasks: handle.queued_tasks.load(Ordering::Relaxed),
                    current_request_id: handle.control.current_request_id(),
                    execution: live_execution_snapshot(handle),
                    conflict_policy: handle.control.current_conflict_policy(),
                })
            })
            .collect();
        sessions.sort_by(|a, b| {
            a.agent_id
                .cmp(&b.agent_id)
                .then_with(|| a.slot_id.cmp(&b.slot_id))
        });
        sessions
    }

    /// Get status for a single agent.
    pub async fn get_status(&self, agent_id: &str) -> Option<AgentStatusSnapshot> {
        self.list_statuses()
            .await
            .into_iter()
            .find(|s| s.agent_id == agent_id)
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

    pub(super) async fn find_runtimes_by_session(
        &self,
        session_id: &str,
    ) -> Vec<(RuntimeSlotKey, Arc<AgentRuntimeHandle>)> {
        let wanted = parse_session_reference(session_id)
            .map(|session_ref| session_ref.public_id)
            .ok();
        let runtimes = self.runtimes.read().await;
        let mut matches: Vec<_> = runtimes
            .iter()
            .filter_map(|(runtime_key, handle)| {
                let current = handle.control.current_session_id()?;
                let current_public_id = parse_session_reference(&current)
                    .map(|session_ref| session_ref.public_id)
                    .ok();
                if current == session_id || (wanted.is_some() && wanted == current_public_id) {
                    Some((runtime_key.clone(), Arc::clone(handle)))
                } else {
                    None
                }
            })
            .collect();
        matches.sort_by(|(left, _), (right, _)| {
            left.agent_id
                .cmp(&right.agent_id)
                .then_with(|| left.slot_id.cmp(&right.slot_id))
        });
        matches
    }

    pub(super) async fn runtime_by_session_target(
        &self,
        session_id: &str,
        slot_id: Option<&str>,
    ) -> Result<(RuntimeSlotKey, Arc<AgentRuntimeHandle>)> {
        let matches = self.find_runtimes_by_session(session_id).await;
        if let Some(slot_id) = slot_id {
            return matches
                .into_iter()
                .find(|(runtime_key, _)| runtime_key.slot_id == slot_id)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Session '{}' is not active in runtime slot '{}'",
                        session_id,
                        slot_id
                    )
                });
        }
        match matches.len() {
            0 => anyhow::bail!(
                "Session '{}' is not an active managed runtime session",
                session_id
            ),
            1 => Ok(matches
                .into_iter()
                .next()
                .expect("single runtime match should exist")),
            _ => anyhow::bail!(
                "Session '{}' is active in multiple runtime slots; specify slot_id",
                session_id
            ),
        }
    }
}
