use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use anyhow::{Context, Result, anyhow};
use tokio::sync::oneshot;

use crate::kernel::delegation_budget::DelegationBudgetLimits;
use crate::kernel::policy::{PolicyScope, RuntimePolicy};
use crate::kernel::session::QueuedTask;
use crate::kernel::session_refs::{format_session_reference, parse_session_reference};
use crate::kernel::task_promotion::{
    TaskPromotionCandidate, TaskPromotionSelection, promote_task_result,
};
use crate::persistence::schema::LinkedSessionCreate;
use crate::persistence::schema::SessionRow;
use crate::persistence::state::StateStore;

use super::task_status::{
    completed_task_fingerprint, completed_task_snapshot, intended_task_execution_snapshot,
    pending_task_fingerprint, pending_task_snapshot,
};
use super::{
    AgentManager, AgentRuntimeHandle, DelegationAdmission, LinkedSessionMode, LinkedSessionTarget,
    PeerAgentTaskEnvelope, PeerAgentTaskResult, PeerTaskSubmission, PendingTaskRecord,
    PendingTaskState, PromotedTaskBranch, RuntimeSlotKey, SessionContextOverrides,
    TaskSessionTarget, TaskStatusFingerprint, TaskStatusSnapshot, task_prompt_preview,
};

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
        mode: LinkedSessionMode,
        mut task: QueuedTask,
        delegated_capabilities: Option<BTreeMap<String, bool>>,
    ) -> Result<String> {
        let thread_key = mode.into_thread_key();
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
        let policy = match self.shared_runtime() {
            Some(shared) => {
                shared
                    .policy_manager
                    .snapshot(&PolicyScope {
                        agent_id: Some(parent.agent_id.clone()),
                        session_id: Some(parent_session_reference.clone()),
                        ..PolicyScope::default()
                    })
                    .await
            }
            None => RuntimePolicy::default().to_map(),
        };
        let max_depth = policy_usize(&policy, "spawn.max_depth", 3);
        let max_fan_out = policy_usize(&policy, "spawn.max_fan_out", 64);
        let max_concurrent_children = policy_usize(&policy, "spawn.max_concurrent_children", 16);
        let budget_limits = DelegationBudgetLimits {
            max_total_tokens: policy_optional_u64(&policy, "spawn.root_max_total_tokens"),
            max_duration_ms: policy_optional_u64(&policy, "spawn.root_max_duration_ms"),
            max_tool_calls: policy_optional_u64(&policy, "spawn.root_max_tool_calls"),
        };
        let family_stats = store
            .linked_session_family_stats(parent.id)
            .await?
            .ok_or_else(|| anyhow!("Origin session '{}' disappeared", origin_session_id))?;
        anyhow::ensure!(
            family_stats.depth < max_depth,
            "Policy denial: spawn.max_depth={} reached at session depth {}",
            max_depth,
            family_stats.depth
        );
        let lane_capacity = self.config.linked_runtime_lanes_for_agent(agent_id)?;
        let linked = store
            .find_linked_session(parent.id, agent_id, thread_key)
            .await?;
        if let Some(linked) = linked.as_ref()
            && linked.visibility == "archived"
        {
            store.restore_linked_session(linked.id).await?;
        }
        let promotion_origin_turn_id = linked
            .as_ref()
            .and_then(|session| session.origin_turn_id)
            .or(origin_turn_id);
        let linked_session_id = if let Some(linked) = linked.as_ref() {
            let public_id = uuid::Uuid::from_slice(&linked.public_id)
                .context("Linked session has an invalid public id")?
                .simple()
                .to_string();
            Some(format_session_reference(&public_id, &state_selector))
        } else {
            None
        };
        if !budget_limits.is_unbounded() {
            let budget = self
                .delegation_budgets
                .lock()
                .expect("delegation budget mutex poisoned")
                .get_or_create(&task.trace_id, budget_limits);
            budget.check_admission()?;
            task.delegation_budget = Some(budget);
        }
        let runtime_key = if let Some(linked_session_id) = linked_session_id.as_deref() {
            self.find_runtimes_by_session(linked_session_id)
                .await
                .into_iter()
                .find_map(|(runtime_key, _)| {
                    (runtime_key.agent_id == agent_id && runtime_key.is_linked())
                        .then_some(runtime_key)
                })
        } else {
            None
        };
        let runtime_key = match runtime_key {
            Some(runtime_key) => runtime_key,
            None => {
                let excluded_slots = self
                    .occupied_ancestor_linked_slots(&store, &parent, agent_id, lane_capacity)
                    .await?;
                RuntimeSlotKey::linked_for_excluding(
                    agent_id,
                    &parent_session_reference,
                    thread_key,
                    &excluded_slots,
                    lane_capacity,
                )
                .ok_or_else(|| {
                    anyhow!(
                        "Same-agent delegation requires another linked runtime lane, but all {} lanes are occupied by awaiting ancestors",
                        lane_capacity
                    )
                })?
            }
        };
        let link = LinkedSessionCreate {
            parent_session_id: parent.id,
            origin_turn_id,
            relation_kind: "delegated".to_string(),
            thread_key: thread_key.to_string(),
            visibility: "contextual".to_string(),
        };
        let session_context = SessionContextOverrides::default();
        let handle = self
            .ensure_runtime_slot_for_linked(
                runtime_key.clone(),
                linked_session_id.as_deref(),
                state_selector.clone(),
                None,
                session_context.clone(),
                link.clone(),
            )
            .await?;

        let promotion_candidate =
            promotion_origin_turn_id.map(|source_turn_id| TaskPromotionCandidate {
                session_id: parent_session_reference,
                source_turn_id,
                source_session_id: linked_session_id.clone(),
            });
        let session_target = TaskSessionTarget {
            session_id: linked_session_id.clone(),
            store_selector: Some(state_selector.clone()),
            linked_parent_session_id: Some(parent.id),
            thread_key: Some(thread_key.to_string()),
            reserves_new_child: linked_session_id.is_none(),
        };

        self.submit_to_handle(
            runtime_key,
            handle,
            task,
            PeerTaskSubmission {
                delegated_capabilities,
                promotion_candidate,
                linked_session: Some(LinkedSessionTarget {
                    state_selector,
                    default_store_selector: None,
                    context: session_context,
                    link,
                }),
                session_target,
                delegation_admission: Some(DelegationAdmission {
                    parent_session_id: parent.id,
                    persisted_direct_children: family_stats.direct_child_count,
                    max_fan_out,
                    max_concurrent_children,
                }),
            },
        )
        .await
    }

    pub(super) async fn occupied_ancestor_linked_slots(
        &self,
        store: &StateStore,
        parent: &SessionRow,
        target_agent_id: &str,
        lane_capacity: usize,
    ) -> Result<HashSet<String>> {
        let live_ancestor_slots = {
            let runtimes = self.runtimes.read().await;
            runtimes
                .iter()
                .filter_map(|(runtime_key, handle)| {
                    if runtime_key.agent_id != target_agent_id
                        || !runtime_key.is_linked()
                        || (handle.active_tasks.load(Ordering::Relaxed) == 0
                            && handle.queued_tasks.load(Ordering::Relaxed) == 0)
                    {
                        return None;
                    }
                    let session_id = handle.control.current_session_id()?;
                    let public_id = parse_session_reference(&session_id).ok()?.public_id;
                    Some((public_id, runtime_key.slot_id.clone()))
                })
                .collect::<HashMap<_, _>>()
        };
        if live_ancestor_slots.is_empty() {
            return Ok(HashSet::new());
        }

        let mut occupied = HashSet::new();
        let mut current = Some(parent.clone());
        let mut visited = HashSet::new();
        while let Some(session) = current {
            anyhow::ensure!(
                visited.insert(session.id),
                "Linked session ancestry contains a cycle at session '{}'",
                session.id
            );
            let public_id = uuid::Uuid::from_slice(&session.public_id)?
                .simple()
                .to_string();
            if let Some(slot_id) = live_ancestor_slots.get(&public_id) {
                occupied.insert(slot_id.clone());
                if occupied.len() == lane_capacity {
                    break;
                }
            }
            current = match session.parent_session_id {
                Some(parent_id) => store.get_session_row(parent_id).await?,
                None => None,
            };
        }
        Ok(occupied)
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
        let session_id = handle.control.current_session_id();
        self.submit_to_handle(
            runtime_key,
            handle,
            task,
            PeerTaskSubmission {
                delegated_capabilities,
                promotion_candidate: None,
                linked_session: None,
                session_target: TaskSessionTarget {
                    session_id,
                    ..TaskSessionTarget::default()
                },
                delegation_admission: None,
            },
        )
        .await
    }

    pub(super) async fn submit_to_handle(
        &self,
        runtime_key: RuntimeSlotKey,
        handle: Arc<AgentRuntimeHandle>,
        task: QueuedTask,
        submission: PeerTaskSubmission,
    ) -> Result<String> {
        let trace_id = task.trace_id.clone();
        let title = task.title.clone();
        let prompt_preview = task_prompt_preview(&task.prompt);
        let request_id = uuid::Uuid::now_v7().simple().to_string();
        let (tx_result, rx_result) = oneshot::channel();
        {
            let mut pending = self.pending_task_states.write().await;
            if let Some(admission) = submission.delegation_admission {
                let same_parent = |record: &&PendingTaskRecord| {
                    record.session_target.linked_parent_session_id
                        == Some(admission.parent_session_id)
                        && record.session_target.store_selector
                            == submission.session_target.store_selector
                };
                let outstanding_children = pending.values().filter(same_parent).count();
                anyhow::ensure!(
                    outstanding_children < admission.max_concurrent_children,
                    "Policy denial: spawn.max_concurrent_children={} reached",
                    admission.max_concurrent_children
                );
                if submission.session_target.reserves_new_child {
                    let reserved_threads = pending
                        .values()
                        .filter(same_parent)
                        .filter(|record| record.session_target.reserves_new_child)
                        .filter_map(|record| record.session_target.thread_key.as_deref())
                        .collect::<HashSet<_>>();
                    let reserves_distinct_child = submission
                        .session_target
                        .thread_key
                        .as_deref()
                        .is_some_and(|thread| !reserved_threads.contains(thread));
                    let projected_children = admission
                        .persisted_direct_children
                        .saturating_add(reserved_threads.len())
                        .saturating_add(usize::from(reserves_distinct_child));
                    anyhow::ensure!(
                        projected_children <= admission.max_fan_out,
                        "Policy denial: spawn.max_fan_out={} reached",
                        admission.max_fan_out
                    );
                }
            }
            pending.insert(
                request_id.clone(),
                PendingTaskRecord {
                    runtime_key: runtime_key.clone(),
                    session_target: submission.session_target.clone(),
                    trace_id,
                    title,
                    prompt_preview,
                    state: PendingTaskState::Queued,
                    runtime_task_id: None,
                    execution: intended_task_execution_snapshot(&handle, &task)?,
                },
            );
        }
        self.pending_results
            .write()
            .await
            .insert(request_id.clone(), rx_result);

        if let Err(e) = self
            .enqueue_runtime_task(
                &handle,
                PeerAgentTaskEnvelope {
                    task,
                    request_id: Some(request_id.clone()),
                    result_tx: Some(tx_result),
                    delegated_capabilities: submission.delegated_capabilities,
                    promotion_candidate: submission.promotion_candidate,
                    linked_session: submission.linked_session,
                    session_target: submission.session_target,
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
                Ok(Err(_)) => {
                    self.record_lost_task_result(request_id, "Peer task result channel closed")
                        .await;
                    Err(anyhow::anyhow!(
                        "Peer task '{}' result channel closed",
                        request_id
                    ))
                }
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

    pub(crate) async fn session_family_work_count(
        &self,
        session_ids: &[String],
        persisted_ids: &HashSet<(crate::persistence::manager::StoreSelector, i64)>,
    ) -> usize {
        let matching_tasks: Vec<_> = self
            .pending_task_states
            .read()
            .await
            .values()
            .filter(|pending| {
                pending
                    .session_target
                    .belongs_to_family(session_ids, persisted_ids)
            })
            .map(|pending| (pending.runtime_key.clone(), pending.session_target.clone()))
            .collect();
        let live_without_task = self
            .runtimes
            .read()
            .await
            .iter()
            .filter(|(runtime_key, handle)| {
                let Some(current) = handle.control.current_session_id() else {
                    return false;
                };
                let current_is_family = session_ids.iter().any(|session_id| {
                    crate::kernel::session_refs::session_references_match(&current, session_id)
                });
                current_is_family
                    && !matching_tasks.iter().any(|(pending_key, target)| {
                        pending_key == *runtime_key && target.matches_session(&current)
                    })
            })
            .count();
        matching_tasks.len() + live_without_task
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
            if handle.shutdown_token.is_cancelled() {
                anyhow::bail!("Agent runtime stopped before task submission");
            }
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
        self.record_completed_result(PeerAgentTaskResult {
            request_id: request_id.to_string(),
            agent_id: pending.runtime_key.agent_id,
            slot_id: pending.runtime_key.slot_id,
            session_id: pending.session_target.session_id,
            trace_id: pending.trace_id,
            title: pending.title,
            prompt_preview: pending.prompt_preview,
            runtime_task_id: pending.runtime_task_id.unwrap_or_default(),
            execution: pending.execution,
            status: crate::kernel::event::TaskTerminalStatus::Error,
            task_turn_count: 0,
            branch_outcome: None,
            promotion_candidate: None,
            promoted_branch: None,
            output: None,
            assistant_content: None,
            promotion_input_content: None,
            error: Some(reason.to_string()),
        })
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

    pub async fn promote_completed_task(
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
    ) {
        if let Some(pending) = self.pending_task_states.write().await.get_mut(request_id) {
            pending.state = PendingTaskState::Running;
            pending.runtime_task_id = Some(runtime_task_id);
            pending.session_target.session_id = session_id;
        }
    }
}

fn policy_usize(
    policy: &std::collections::HashMap<String, serde_json::Value>,
    key: &str,
    default: usize,
) -> usize {
    policy
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(default)
}

fn policy_optional_u64(
    policy: &std::collections::HashMap<String, serde_json::Value>,
    key: &str,
) -> Option<u64> {
    policy.get(key).and_then(serde_json::Value::as_u64)
}
