use anyhow::Result;
use std::collections::BTreeMap;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::{error, info, warn};

use crate::harness::verdict::Verdict;
use crate::inference::provider::{InferenceContent, InferenceRole};
use crate::kernel::TaskExecutionResult;
use crate::kernel::event::{KernelEvent, LifecycleEvent, TaskTerminalStatus};
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::session::QueuedTask;
use crate::persistence::manager::StoreSelector;

use super::{AgentManager, PeerAgentTaskEnvelope, PeerAgentTaskResult, RuntimeControl};

pub(super) struct PeerRuntime {
    manager: Arc<AgentManager>,
    control: Arc<RuntimeControl>,
    host: ExecutionHost,
    session: crate::kernel::session::SessionState,
    agent_id: String,
    slot_id: String,
}

#[derive(Debug)]
pub(super) struct PeerRunOutcome {
    pub(super) runtime_task_id: String,
    pub(super) status: TaskTerminalStatus,
    pub(super) task_turn_count: u32,
    pub(super) output: Option<String>,
}

impl PeerRuntime {
    pub(super) async fn start(
        manager: Arc<AgentManager>,
        agent_id: &str,
        slot_id: &str,
        control: Arc<RuntimeControl>,
        initial_session_id: Option<&str>,
        initial_state_selector: Option<StoreSelector>,
        initial_default_store_selector: Option<StoreSelector>,
    ) -> Result<Self> {
        let mut host = fork_peer_kernel(&manager);
        if host.clients.is_empty() {
            host.init_clients()?;
        }

        let mut session = if let Some(session_id) = initial_session_id {
            host.resume_session_for_agent(agent_id, session_id).await?
        } else {
            host.create_session_for_agent_in_store(
                agent_id,
                initial_state_selector,
                initial_default_store_selector,
            )
            .await
        };
        host.start_session(&mut session).await?;
        control.set_current_session(
            Some(host.session_reference(&session)),
            Some(session.event_tx.clone()),
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
                    runtime_task_id: ok.runtime_task_id,
                    status: ok.status,
                    task_turn_count: ok.task_turn_count,
                    output: ok.output,
                    error: None,
                },
                Err(e) => PeerAgentTaskResult {
                    request_id,
                    agent_id: self.agent_id.clone(),
                    slot_id: self.slot_id.clone(),
                    trace_id,
                    runtime_task_id: String::new(),
                    status: TaskTerminalStatus::Error,
                    task_turn_count: 0,
                    output: None,
                    error: Some(e.to_string()),
                },
            };
            let _ = tx_result.send(completed.clone());
            self.manager.record_completed_result(completed).await;
        } else if let Err(e) = result {
            error!(agent_id = %self.agent_id, error = %e, "Peer agent task failed");
        }
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
        self.control.set_current_session(None, None);
        self.host.shutdown_mcp_clients().await;
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
                }),
            );

            let task_start_verdict = {
                let runtime = self.host.runtime_for_session(&self.session);
                let harness = runtime.lock_engine();
                if let Some(ref engine) = *harness {
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
                        )
                        .await?;
                    return Ok(PeerRunOutcome {
                        runtime_task_id: task.task_id,
                        status: TaskTerminalStatus::Rejected,
                        task_turn_count: 0,
                        output: None,
                    });
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
                        )
                        .await?;
                    return Ok(PeerRunOutcome {
                        runtime_task_id: task.task_id,
                        status: TaskTerminalStatus::Rejected,
                        task_turn_count: 0,
                        output: None,
                    });
                }
                Verdict::Allow => {}
            }

            info!(task_id = %task.task_id, trace_id = %task.trace_id, prompt = %task.prompt, "Running peer task");

            let run_result: TaskExecutionResult =
                match self.host.run_task(&mut self.session, &task).await {
                    Ok(result) => {
                        self.host
                            .complete_task(
                                &mut self.session,
                                &task,
                                result.status,
                                result.task_turn_count,
                                None,
                            )
                            .await?;
                        result
                    }
                    Err(e) => {
                        error!(task_id = %task.task_id, trace_id = %task.trace_id, error = %e, "Peer task failed with runtime error");
                        let error_message = e.to_string();
                        let recovered = self
                            .host
                            .handle_inference_error(&mut self.session, &task, &error_message)
                            .await?;
                        self.host
                            .complete_task(
                                &mut self.session,
                                &task,
                                TaskTerminalStatus::Error,
                                0,
                                Some(error_message),
                            )
                            .await?;
                        if recovered {
                            return Ok(PeerRunOutcome {
                                runtime_task_id: task.task_id,
                                status: TaskTerminalStatus::Error,
                                task_turn_count: 0,
                                output: None,
                            });
                        }
                        return Err(e);
                    }
                };

            let output = self.last_assistant_text();

            Ok(PeerRunOutcome {
                runtime_task_id: task.task_id,
                status: run_result.status,
                task_turn_count: run_result.task_turn_count,
                output,
            })
        }
        .await;
        self.clear_capability_ceiling();
        outcome
    }

    fn last_assistant_text(&self) -> Option<String> {
        self.session.history.iter().rev().find_map(|msg| {
            if msg.role != InferenceRole::Assistant {
                return None;
            }
            msg.content.iter().find_map(|c| match c {
                InferenceContent::Text { text } => Some(text.clone()),
                _ => None,
            })
        })
    }

    fn set_capability_ceiling(&self, caps: Option<BTreeMap<String, bool>>) {
        let runtime = self.host.runtime_for_session(&self.session);
        let harness = runtime.lock_engine();
        if let Some(ref engine) = *harness {
            engine.set_active_capability_delegation(caps);
        }
    }

    fn clear_capability_ceiling(&self) {
        let runtime = self.host.runtime_for_session(&self.session);
        let harness = runtime.lock_engine();
        if let Some(ref engine) = *harness {
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
            super::SessionResetRequest::Fresh => self.reset_session().await?,
            super::SessionResetRequest::Resume(session_id) => {
                self.restore_session(&session_id).await?
            }
        }
        Ok(true)
    }

    async fn reset_session(&mut self) -> Result<()> {
        self.host.end_session(&mut self.session).await?;
        let mut session = self.host.create_session_for_agent(&self.agent_id).await;
        self.host.start_session(&mut session).await?;
        self.control.set_current_session(
            Some(self.host.session_reference(&session)),
            Some(session.event_tx.clone()),
        );
        self.session = session;
        Ok(())
    }

    async fn restore_session(&mut self, session_id: &str) -> Result<()> {
        self.host.end_session(&mut self.session).await?;
        let mut session = self
            .host
            .resume_session_for_agent(&self.agent_id, session_id)
            .await?;
        self.host.start_session(&mut session).await?;
        self.control.set_current_session(
            Some(self.host.session_reference(&session)),
            Some(session.event_tx.clone()),
        );
        self.session = session;
        Ok(())
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
        clients: inference.clients,
        embedding_provider: inference.embedding_provider,
        mcp_clients: Vec::new(),
    }
}
