use anyhow::Result;
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

use crate::harness::verdict::Verdict;
use crate::kernel::TaskExecutionResult;
use crate::kernel::event::{KernelEvent, LifecycleEvent, TaskTerminalStatus};
use crate::kernel::execution_host::{ExecutionHost, TaskRunAttempt};
use crate::kernel::harness_contract::HarnessHook;
use crate::kernel::session::QueuedTask;
use turin_types::TaskInputContent;

use super::{
    AgentManager, ExecutionStatusSnapshot, LiveSessionHistorySnapshot, PeerAgentTaskEnvelope,
    PeerAgentTaskResult, RuntimeControl, TaskPromotionCandidate, task_prompt_preview,
};

pub(super) struct PeerRuntime {
    pub(super) manager: Arc<AgentManager>,
    pub(super) control: Arc<RuntimeControl>,
    pub(super) host: ExecutionHost,
    pub(super) session: crate::kernel::session::SessionState,
    pub(super) agent_id: String,
    pub(super) slot_id: String,
}

#[derive(Debug)]
pub(super) struct PeerRunOutcome {
    pub(super) runtime_task_id: String,
    pub(super) execution: ExecutionStatusSnapshot,
    pub(super) status: TaskTerminalStatus,
    pub(super) task_turn_count: u32,
    pub(super) branch_outcome: Option<crate::kernel::event::TaskBranchOutcome>,
    pub(super) promotion_candidate: Option<TaskPromotionCandidate>,
    pub(super) output: Option<String>,
    pub(super) assistant_content: Option<Vec<turin_types::TaskInputContent>>,
    pub(super) promotion_input_content: Option<Vec<TaskInputContent>>,
}

impl PeerRuntime {
    pub(super) async fn handle_envelope(&mut self, mut envelope: PeerAgentTaskEnvelope) {
        let request_id = envelope.request_id.clone();
        let intended_session_id = envelope.session_target.session_id.clone();
        let trace_id = envelope.task.trace_id.clone();
        let title = envelope.task.title.clone();
        let prompt_preview = task_prompt_preview(&envelope.task.prompt);
        let activation = if let Some(target) = envelope.linked_session.take() {
            self.activate_linked_session(target).await
        } else {
            Ok(())
        };
        let result = match activation {
            Ok(()) => {
                if let Some(candidate) = envelope.promotion_candidate.as_mut()
                    && candidate.source_session_id.is_none()
                {
                    candidate.source_session_id = self.control.current_session_id();
                }
                let runtime_task_id = self.allocate_runtime_task_id(&mut envelope.task);
                self.prepare_task_execution(
                    request_id.clone(),
                    runtime_task_id.clone(),
                    envelope.task.delegation_budget.as_deref(),
                );
                if let Some(request_id) = request_id.as_deref() {
                    let cancellation_requested = self
                        .manager
                        .mark_task_running(
                            request_id,
                            runtime_task_id.clone(),
                            self.control.current_session_id(),
                        )
                        .await;
                    if cancellation_requested {
                        self.control.request_task_cancel();
                    }
                }
                self.run_queued_task(
                    envelope.task,
                    envelope.delegated_capabilities,
                    envelope.promotion_candidate,
                )
                .await
            }
            Err(err) => Err(err),
        };

        if let Some(tx_result) = envelope.result_tx {
            let request_id = envelope
                .request_id
                .unwrap_or_else(|| uuid::Uuid::now_v7().simple().to_string());
            let completed = match result {
                Ok(ok) => PeerAgentTaskResult {
                    request_id,
                    agent_id: self.agent_id.clone(),
                    slot_id: self.slot_id.clone(),
                    session_id: self.control.current_session_id(),
                    trace_id,
                    title,
                    prompt_preview,
                    runtime_task_id: ok.runtime_task_id,
                    execution: ok.execution,
                    status: ok.status,
                    task_turn_count: ok.task_turn_count,
                    branch_outcome: ok.branch_outcome,
                    promotion_candidate: ok.promotion_candidate,
                    promoted_branch: None,
                    output: ok.output,
                    assistant_content: ok.assistant_content,
                    promotion_input_content: ok.promotion_input_content,
                    error: None,
                },
                Err(e) => PeerAgentTaskResult {
                    request_id,
                    agent_id: self.agent_id.clone(),
                    slot_id: self.slot_id.clone(),
                    session_id: intended_session_id,
                    trace_id,
                    title,
                    prompt_preview,
                    runtime_task_id: String::new(),
                    execution: ExecutionStatusSnapshot::from_session(&self.session),
                    status: TaskTerminalStatus::Error,
                    task_turn_count: 0,
                    branch_outcome: None,
                    promotion_candidate: None,
                    promoted_branch: None,
                    output: None,
                    assistant_content: None,
                    promotion_input_content: None,
                    error: Some(e.to_string()),
                },
            };
            let _ = tx_result.send(completed.clone());
            self.manager.record_completed_result(completed).await;
        } else if let Err(e) = result {
            error!(agent_id = %self.agent_id, error = %e, "Peer agent task failed");
        }
        self.sync_control_execution_state();
        self.control.clear_active_task();
        if let Err(err) = self.reset_session_if_requested().await {
            error!(agent_id = %self.agent_id, error = %err, "Peer runtime failed to reset session");
        }
    }

    pub(super) fn allocate_runtime_task_id(&mut self, task: &mut QueuedTask) -> String {
        if task.task_id.is_empty() {
            task.task_id = format!("t_{}", self.session.next_task_id);
            self.session.next_task_id += 1;
        }
        task.task_id.clone()
    }

    async fn run_queued_task(
        &mut self,
        mut task: QueuedTask,
        delegated_capabilities: Option<BTreeMap<String, bool>>,
        linked_promotion_candidate: Option<TaskPromotionCandidate>,
    ) -> Result<PeerRunOutcome> {
        self.allocate_runtime_task_id(&mut task);

        self.set_capability_ceiling(delegated_capabilities.clone());
        self.session
            .set_active_task_conflict_policy(task.conflict_policy);
        self.session
            .set_current_task_branch_outcome(task.branch_outcome.clone());
        self.session.begin_active_task_budget();
        if let Err(error) = self
            .host
            .begin_task_execution_scope(&mut self.session, &task)
            .await
        {
            let error_message = error.to_string();
            let completion = self
                .host
                .complete_task(
                    &mut self.session,
                    &task,
                    TaskTerminalStatus::Error,
                    0,
                    None,
                    Some(error_message),
                )
                .await;
            let cleanup = self
                .host
                .finish_task_execution_scope(&mut self.session)
                .await;
            self.sync_control_execution_state();
            self.clear_capability_ceiling();
            completion?;
            cleanup?;
            return Ok(self.empty_outcome(task.task_id, TaskTerminalStatus::Error));
        }
        self.sync_control_execution_state();
        let outcome = async {
            self.host.persist_event(
                &self.session,
                &KernelEvent::Lifecycle(LifecycleEvent::TaskStart {
                    identity: self.session.identity.clone(),
                    task_id: task.task_id.clone(),
                    trace_id: task.trace_id.clone(),
                    plan_id: task.plan_id.clone(),
                    title: task.title.clone(),
                    prompt: task.prompt.clone(),
                    queue_depth: 0,
                    execution: ExecutionStatusSnapshot::from_session(&self.session),
                }),
            ).await;

            let task_start_verdict = {
                if let Some(harness) = self.host.session_harness_engine(&self.session) {
                    let engine = harness.lock().expect("session harness mutex poisoned");
                    let session_id = self.host.session_reference(&self.session);
                    match engine.evaluate_hook(HarnessHook::TaskStart {
                        identity: &self.session.identity,
                        session_id: &session_id,
                        task_id: &task.task_id,
                        trace_id: &task.trace_id,
                        plan_id: task.plan_id.as_deref(),
                        title: task.title.as_deref(),
                        prompt: &task.prompt,
                        queue_depth: 0,
                    }) {
                        Ok(v) => v,
                        Err(e) => {
                            warn!(error = %e, "Harness on_task_start error");
                            Verdict::Allow
                        }
                    }
                } else {
                    Verdict::Allow
                }
            };

            match task_start_verdict {
                Verdict::Reject(reason) => {
                    warn!(
                        task_id = %task.task_id,
                        trace_id = %task.trace_id,
                        reason = %reason,
                        "Peer task rejected by on_task_start"
                    );
                    self.host
                        .complete_task(
                            &mut self.session,
                            &task,
                            TaskTerminalStatus::Rejected,
                            0,
                            None,
                            None,
                        )
                        .await?;
                    return Ok(self.empty_outcome(task.task_id, TaskTerminalStatus::Rejected));
                }
                Verdict::Modify(val) => {
                    if let Some(obj) = val.as_object() {
                        if let Some(prompt) = obj.get("prompt").and_then(|v| v.as_str()) {
                            task.prompt = prompt.to_string();
                        }
                        if let Some(title) = obj.get("title").and_then(|v| v.as_str()) {
                            task.title = Some(title.to_string());
                        }
                    }
                }
                Verdict::Escalate(reason) => {
                    warn!(
                        task_id = %task.task_id,
                        trace_id = %task.trace_id,
                        reason = %reason,
                        "Peer task escalated at on_task_start; treating as rejected"
                    );
                    self.host
                        .complete_task(
                            &mut self.session,
                            &task,
                            TaskTerminalStatus::Rejected,
                            0,
                            None,
                            None,
                        )
                        .await?;
                    return Ok(self.empty_outcome(task.task_id, TaskTerminalStatus::Rejected));
                }
                Verdict::Allow => {}
            }

            info!(task_id = %task.task_id, trace_id = %task.trace_id, prompt = %task.prompt, "Running peer task");

            let run_result: TaskExecutionResult = match self
                .host
                .run_task_with_conflict_handling(&mut self.session, &task)
                .await?
            {
                TaskRunAttempt::Completed(result) => {
                    self.host
                        .complete_task(
                            &mut self.session,
                            &task,
                            result.status,
                            result.task_turn_count,
                            result.branch_outcome.clone(),
                            None,
                        )
                        .await?;
                    self.host
                        .apply_pending_branch_checkout(&mut self.session)
                        .await?;
                    result
                }
                TaskRunAttempt::Terminal {
                    status,
                    error_message,
                } => {
                    self.host
                        .complete_task(
                            &mut self.session,
                            &task,
                            status,
                            0,
                            None,
                            Some(error_message),
                        )
                        .await?;
                    self.host
                        .apply_pending_branch_checkout(&mut self.session)
                        .await?;
                    return Ok(self.empty_outcome(task.task_id, status));
                }
                TaskRunAttempt::Error {
                    error,
                    error_message,
                    recovered,
                } => {
                    self.host
                        .complete_task(
                            &mut self.session,
                            &task,
                            TaskTerminalStatus::Error,
                            0,
                            None,
                            Some(error_message),
                        )
                        .await?;
                    self.host
                        .apply_pending_branch_checkout(&mut self.session)
                        .await?;
                    if recovered {
                        return Ok(self.empty_outcome(task.task_id, TaskTerminalStatus::Error));
                    }
                    return Err(error);
                }
            };

            let output = self.host.last_assistant_text(&self.session);
            let assistant_content = self.host.last_assistant_content(&self.session);
            let promotion_candidate = (run_result.status == TaskTerminalStatus::Success)
                .then_some(linked_promotion_candidate)
                .flatten()
                .or_else(|| {
                    self.host
                        .promotable_detached_candidate(&self.session, run_result.status)
                });

            Ok(PeerRunOutcome {
                runtime_task_id: task.task_id.clone(),
                execution: ExecutionStatusSnapshot::from_session(&self.session),
                status: run_result.status,
                task_turn_count: run_result.task_turn_count,
                branch_outcome: run_result.branch_outcome,
                promotion_candidate,
                output,
                assistant_content,
                promotion_input_content: Some(ExecutionHost::task_input_content(&task)),
            })
        }
        .await;
        let finish_scope = self
            .host
            .finish_task_execution_scope(&mut self.session)
            .await;
        self.clear_capability_ceiling();
        self.sync_control_execution_state();
        finish_scope?;
        outcome
    }

    fn empty_outcome(&self, runtime_task_id: String, status: TaskTerminalStatus) -> PeerRunOutcome {
        PeerRunOutcome {
            runtime_task_id,
            execution: ExecutionStatusSnapshot::from_session(&self.session),
            status,
            task_turn_count: 0,
            branch_outcome: None,
            promotion_candidate: None,
            output: None,
            assistant_content: None,
            promotion_input_content: None,
        }
    }

    fn set_capability_ceiling(&self, caps: Option<BTreeMap<String, bool>>) {
        if let Some(harness) = self.host.session_harness_engine(&self.session) {
            let engine = harness.lock().expect("session harness mutex poisoned");
            engine.set_active_capability_delegation(caps);
        }
    }

    fn clear_capability_ceiling(&self) {
        if let Some(harness) = self.host.session_harness_engine(&self.session) {
            let engine = harness.lock().expect("session harness mutex poisoned");
            engine.set_active_capability_delegation(None);
        }
    }

    fn prepare_task_execution(
        &mut self,
        request_id: Option<String>,
        runtime_task_id: String,
        delegation_budget: Option<&crate::kernel::delegation_budget::DelegationBudget>,
    ) {
        self.session.cancel_token = delegation_budget.map_or_else(
            CancellationToken::new,
            crate::kernel::delegation_budget::DelegationBudget::child_cancellation_token,
        );
        self.control.activate_task(
            request_id,
            runtime_task_id,
            self.session.cancel_token.clone(),
        );
    }

    pub(super) fn sync_control_execution_state(&self) {
        self.control
            .set_current_execution_snapshot(ExecutionStatusSnapshot::from_session(&self.session));
        self.control
            .set_current_conflict_policy(self.session.effective_conflict_policy());
        self.control
            .set_current_history_snapshot(LiveSessionHistorySnapshot::from_session(&self.session));
    }
}
