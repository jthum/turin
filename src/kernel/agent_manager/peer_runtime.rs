use anyhow::Result;
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

use crate::harness::verdict::Verdict;
use crate::kernel::TaskExecutionResult;
use crate::kernel::event::{KernelEvent, LifecycleEvent, TaskTerminalStatus};
use crate::kernel::execution_host::{ExecutionHost, TaskRunAttempt};
use crate::kernel::session::QueuedTask;
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::SignalRow;
use turin_types::TaskInputContent;

use super::{
    AgentManager, ExecutionStatusSnapshot, LiveSessionHistorySnapshot, PeerAgentTaskEnvelope,
    PeerAgentTaskResult, RuntimeControl, SessionContextOverrides, TaskPromotionCandidate,
    task_prompt_preview,
};

pub(super) struct PeerRuntime {
    manager: Arc<AgentManager>,
    control: Arc<RuntimeControl>,
    host: ExecutionHost,
    session: crate::kernel::session::SessionState,
    agent_id: String,
    slot_id: String,
}

#[derive(Debug, Clone, Default)]
pub(super) struct SessionBootstrap {
    pub(super) initial_session_id: Option<String>,
    pub(super) initial_state_selector: Option<StoreSelector>,
    pub(super) initial_default_store_selector: Option<StoreSelector>,
    pub(super) context: SessionContextOverrides,
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
    pub(super) async fn start(
        manager: Arc<AgentManager>,
        agent_id: &str,
        slot_id: &str,
        control: Arc<RuntimeControl>,
        bootstrap: SessionBootstrap,
    ) -> Result<Self> {
        let mut host = fork_peer_kernel(&manager);
        if host.clients.is_empty() {
            host.init_clients()?;
        }

        let mut session = if let Some(session_id) = bootstrap.initial_session_id.as_deref() {
            host.resume_session_for_agent_with_context(
                agent_id,
                session_id,
                bootstrap.context.channel_id.clone(),
                bootstrap.context.inference.clone(),
            )
            .await?
        } else {
            host.create_session_for_agent_with_context(
                agent_id,
                bootstrap.initial_state_selector,
                bootstrap.initial_default_store_selector,
                bootstrap.context.channel_id.clone(),
                bootstrap.context.inference.clone(),
            )
            .await
        };
        session.runtime_slot_id = Some(slot_id.to_string());
        host.start_session(&mut session).await?;
        control.set_current_session(
            Some(host.session_reference(&session)),
            Some(session.event_tx.clone()),
            session_context_from_session(&session),
            Some(ExecutionStatusSnapshot::from_session(&session)),
            session.execution.conflict_policy,
            Some(LiveSessionHistorySnapshot::from_session(&session)),
        );

        Ok(Self {
            manager,
            control,
            host,
            session,
            agent_id: agent_id.to_string(),
            slot_id: slot_id.to_string(),
        })
    }

    pub(super) async fn handle_envelope(&mut self, mut envelope: PeerAgentTaskEnvelope) {
        let runtime_task_id = self.allocate_runtime_task_id(&mut envelope.task);
        let request_id = envelope.request_id.clone();
        let trace_id = envelope.task.trace_id.clone();
        let title = envelope.task.title.clone();
        let prompt_preview = task_prompt_preview(&envelope.task.prompt);
        self.prepare_task_execution(request_id.clone(), runtime_task_id);
        let result = self
            .run_queued_task(envelope.task, envelope.delegated_capabilities)
            .await;

        if let Some(tx_result) = envelope.result_tx {
            let request_id = envelope
                .request_id
                .unwrap_or_else(|| uuid::Uuid::now_v7().simple().to_string());
            let completed = match result {
                Ok(ok) => PeerAgentTaskResult {
                    request_id,
                    agent_id: self.agent_id.clone(),
                    slot_id: self.slot_id.clone(),
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

    pub(super) async fn shutdown(mut self) {
        if let Err(e) = self.host.end_session(&mut self.session).await {
            warn!(agent_id = %self.agent_id, error = %e, "Peer agent session end error");
        }
        self.control.clear_active_task();
        self.control.set_current_session(
            None,
            None,
            SessionContextOverrides::default(),
            None,
            crate::kernel::session::ExecutionConflictPolicy::Reject,
            None,
        );
        self.host.shutdown_mcp_clients().await;
        super::allocator::trim_after_peer_idle_if_enabled();
        info!(agent_id = %self.agent_id, "Peer runtime shut down");
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
                .finish_task_execution_scope(&mut self.session)
                .await?;
            self.sync_control_execution_state();
            self.clear_capability_ceiling();
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
            );

            let task_start_verdict = {
                if let Some(harness) = self.host.session_harness_engine(&self.session) {
                    let engine = harness.lock().expect("session harness mutex poisoned");
                    match engine.evaluate(
                        "on_task_start",
                        serde_json::json!({
                            "identity": self.session.identity.clone(),
                            "session_id": self.host.session_reference(&self.session),
                            "task_id": task.task_id.clone(),
                            "trace_id": task.trace_id.clone(),
                            "plan_id": task.plan_id.clone(),
                            "title": task.title.clone(),
                            "prompt": task.prompt.clone(),
                            "queue_depth": 0,
                        }),
                    ) {
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
            let promotion_candidate = self
                .host
                .promotable_detached_candidate(&self.session, run_result.status);

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

    fn prepare_task_execution(&mut self, request_id: Option<String>, runtime_task_id: String) {
        self.session.cancel_token = CancellationToken::new();
        self.control.activate_task(
            request_id,
            runtime_task_id,
            self.session.cancel_token.clone(),
        );
    }

    pub(super) async fn reset_session_if_requested(&mut self) -> Result<bool> {
        let Some(request) = self.control.take_session_reset_request() else {
            return Ok(false);
        };

        match request {
            super::SessionResetRequest::Fresh(context) => self.reset_session(context).await?,
            super::SessionResetRequest::Resume {
                session_id,
                context,
            } => self.restore_session(&session_id, context).await?,
        }
        Ok(true)
    }

    pub(super) async fn process_pending_signals(&mut self) -> Result<usize> {
        let Some(runtime_scheduler) = self.host.scheduler.as_ref() else {
            return Ok(0);
        };
        let store = runtime_scheduler.runtime_store();
        let signals = store.list_signals_for_agent(&self.agent_id, 64).await?;
        if signals.is_empty() {
            return Ok(0);
        }

        let mut processed = 0usize;
        for signal in signals {
            store.record_signal_attempt(signal.id).await?;
            match self.dispatch_signal(&signal).await {
                Ok(_) => {
                    store.delete_signal(signal.id).await?;
                    processed += 1;
                }
                Err(err) => {
                    let error_message = err.to_string();
                    store.set_signal_error(signal.id, &error_message).await?;
                    return Err(err);
                }
            }
        }

        Ok(processed)
    }

    async fn reset_session(&mut self, context: SessionContextOverrides) -> Result<()> {
        self.host.end_session(&mut self.session).await?;
        let context = effective_session_context(&self.session, context);
        let mut session = self
            .host
            .create_session_for_agent_with_context(
                &self.agent_id,
                Some(self.session.store_selector.clone()),
                self.session.default_store_selector.clone(),
                context.channel_id.clone(),
                context.inference.clone(),
            )
            .await;
        session.runtime_slot_id = Some(self.slot_id.clone());
        self.host.start_session(&mut session).await?;
        self.control.set_current_session(
            Some(self.host.session_reference(&session)),
            Some(session.event_tx.clone()),
            session_context_from_session(&session),
            Some(ExecutionStatusSnapshot::from_session(&session)),
            session.execution.conflict_policy,
            Some(LiveSessionHistorySnapshot::from_session(&session)),
        );
        self.session = session;
        Ok(())
    }

    async fn restore_session(
        &mut self,
        session_id: &str,
        context: SessionContextOverrides,
    ) -> Result<()> {
        self.host.end_session(&mut self.session).await?;
        let context = effective_session_context(&self.session, context);
        let mut session = self
            .host
            .resume_session_for_agent_with_context(
                &self.agent_id,
                session_id,
                context.channel_id.clone(),
                context.inference.clone(),
            )
            .await?;
        session.runtime_slot_id = Some(self.slot_id.clone());
        self.host.start_session(&mut session).await?;
        self.control.set_current_session(
            Some(self.host.session_reference(&session)),
            Some(session.event_tx.clone()),
            session_context_from_session(&session),
            Some(ExecutionStatusSnapshot::from_session(&session)),
            session.execution.conflict_policy,
            Some(LiveSessionHistorySnapshot::from_session(&session)),
        );
        self.session = session;
        Ok(())
    }

    fn sync_control_execution_state(&self) {
        self.control
            .set_current_execution_snapshot(ExecutionStatusSnapshot::from_session(&self.session));
        self.control
            .set_current_conflict_policy(self.session.effective_conflict_policy());
        self.control
            .set_current_history_snapshot(LiveSessionHistorySnapshot::from_session(&self.session));
    }

    async fn dispatch_signal(&mut self, signal: &SignalRow) -> Result<usize> {
        self.host.ensure_session_harness_engine(&mut self.session)?;
        let trace_task = QueuedTask::ad_hoc(format!("signal:{}", signal.topic));
        self.host
            .bind_harness_execution_context(&self.session, &trace_task);
        let result = {
            let harness = self
                .host
                .session_harness_engine(&self.session)
                .expect("session harness engine should be present after ensure");
            let engine = harness.lock().expect("session harness mutex poisoned");
            engine.dispatch_runtime_signal(signal)
        };
        self.host.unbind_harness_execution_context(&self.session);
        result
    }
}

fn session_context_from_session(
    session: &crate::kernel::session::SessionState,
) -> SessionContextOverrides {
    SessionContextOverrides {
        channel_id: session.identity.channel_id().map(ToOwned::to_owned),
        inference: session.inference.clone(),
    }
}

fn effective_session_context(
    session: &crate::kernel::session::SessionState,
    requested: SessionContextOverrides,
) -> SessionContextOverrides {
    SessionContextOverrides {
        channel_id: requested
            .channel_id
            .or_else(|| session.identity.channel_id().map(ToOwned::to_owned)),
        inference: if requested.inference.is_empty() {
            session.inference.clone()
        } else {
            requested.inference
        },
    }
}

pub(super) fn fork_peer_kernel(manager: &Arc<AgentManager>) -> ExecutionHost {
    let shared = manager
        .shared_runtime()
        .expect("AgentManager shared runtime not bound");
    let inference = manager
        .shared_inference
        .lock()
        .expect("agent manager shared inference mutex poisoned")
        .clone();

    ExecutionHost {
        config: Arc::clone(&manager.config),
        json: shared.json,
        tool_registry: shared.tool_registry.clone(),
        store_manager: Arc::clone(&manager.store_manager),
        agent_manager: Arc::clone(manager),
        policy_manager: Arc::clone(&shared.policy_manager),
        governance_manager: Arc::clone(&shared.governance_manager),
        harness_manager: Arc::clone(&shared.harness_manager),
        scheduler: manager.shared_scheduler(),
        persistence_locks: Arc::clone(&shared.persistence_locks),
        clients: inference.clients,
        embedding_provider: inference.embedding_provider,
        mcp_clients: Vec::new(),
    }
}
