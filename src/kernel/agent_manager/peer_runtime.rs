use anyhow::Result;
use std::collections::BTreeMap;
use tracing::{error, info, warn};

use crate::harness::verdict::Verdict;
use crate::inference::provider::{InferenceContent, InferenceRole};
use crate::kernel::TaskExecutionResult;
use crate::kernel::event::{KernelEvent, LifecycleEvent, TaskTerminalStatus};
use crate::kernel::session::QueuedTask;

use super::{AgentManager, PeerAgentTaskEnvelope, PeerAgentTaskResult};

pub(super) struct PeerRuntime {
    kernel: crate::kernel::Kernel,
    session: crate::kernel::session::SessionState,
    agent_id: String,
}

#[derive(Debug)]
pub(super) struct PeerRunOutcome {
    pub(super) runtime_task_id: String,
    pub(super) status: TaskTerminalStatus,
    pub(super) task_turn_count: u32,
    pub(super) output: Option<String>,
}

impl PeerRuntime {
    pub(super) async fn start(manager: &AgentManager, agent_id: &str) -> Result<Self> {
        let mut kernel = manager.build_shared_peer_kernel()?;
        if kernel.clients.is_empty() {
            kernel.init_clients()?;
        }

        let mut session = kernel.create_session_for_agent(agent_id).await;
        kernel.start_session(&mut session).await?;

        Ok(Self {
            kernel,
            session,
            agent_id: agent_id.to_string(),
        })
    }

    pub(super) async fn handle_envelope(&mut self, envelope: PeerAgentTaskEnvelope) {
        let result = self
            .run_queued_task(envelope.task, envelope.delegated_capabilities)
            .await;

        if let Some(tx_result) = envelope.result_tx {
            let request_id = envelope
                .request_id
                .unwrap_or_else(|| uuid::Uuid::now_v7().simple().to_string());
            let _ = tx_result.send(match result {
                Ok(ok) => PeerAgentTaskResult {
                    request_id,
                    agent_id: self.agent_id.clone(),
                    runtime_task_id: ok.runtime_task_id,
                    status: ok.status,
                    task_turn_count: ok.task_turn_count,
                    output: ok.output,
                    error: None,
                },
                Err(e) => PeerAgentTaskResult {
                    request_id,
                    agent_id: self.agent_id.clone(),
                    runtime_task_id: String::new(),
                    status: TaskTerminalStatus::Error,
                    task_turn_count: 0,
                    output: None,
                    error: Some(e.to_string()),
                },
            });
        } else if let Err(e) = result {
            error!(agent_id = %self.agent_id, error = %e, "Peer agent task failed");
        }
    }

    pub(super) async fn shutdown(mut self) {
        if let Err(e) = self.kernel.end_session(&mut self.session).await {
            warn!(agent_id = %self.agent_id, error = %e, "Peer agent session end error");
        }
        self.kernel.shutdown_mcp_clients().await;
        info!(agent_id = %self.agent_id, "Peer runtime shut down");
    }

    async fn run_queued_task(
        &mut self,
        mut task: QueuedTask,
        delegated_capabilities: Option<BTreeMap<String, bool>>,
    ) -> Result<PeerRunOutcome> {
        if task.task_id.is_empty() {
            task.task_id = format!("t_{}", self.session.next_task_id);
            self.session.next_task_id += 1;
        }

        self.set_capability_ceiling(delegated_capabilities.clone());
        let outcome = async {
            self.kernel.persist_event(
                &self.session,
                &KernelEvent::Lifecycle(LifecycleEvent::TaskStart {
                    identity: self.session.identity.clone(),
                    task_id: task.task_id.clone(),
                    plan_id: task.plan_id.clone(),
                    title: task.title.clone(),
                    prompt: task.prompt.clone(),
                    queue_depth: 0,
                }),
            );

            let task_start_verdict = {
                let runtime = self.kernel.runtime_for_session(&self.session);
                let harness = runtime.lock_engine();
                if let Some(ref engine) = *harness {
                    match engine.evaluate(
                        "on_task_start",
                        serde_json::json!({
                            "identity": self.session.identity.clone(),
                            "session_id": self.session.identity.session_id(),
                            "task_id": task.task_id.clone(),
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
                        reason = %reason,
                        "Peer task rejected by on_task_start"
                    );
                    self.kernel
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
                        reason = %reason,
                        "Peer task escalated at on_task_start; treating as rejected"
                    );
                    self.kernel
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

            info!(task_id = %task.task_id, prompt = %task.prompt, "Running peer task");

            let run_result: TaskExecutionResult =
                match self.kernel.run_task(&mut self.session, &task).await {
                    Ok(result) => {
                        self.kernel
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
                        error!(task_id = %task.task_id, error = %e, "Peer task failed with runtime error");
                        let error_message = e.to_string();
                        let recovered = self
                            .kernel
                            .handle_inference_error(&mut self.session, &task, &error_message)
                            .await?;
                        self.kernel
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
        let runtime = self.kernel.runtime_for_session(&self.session);
        let harness = runtime.lock_engine();
        if let Some(ref engine) = *harness {
            engine.set_active_capability_delegation(caps);
        }
    }

    fn clear_capability_ceiling(&self) {
        let runtime = self.kernel.runtime_for_session(&self.session);
        let harness = runtime.lock_engine();
        if let Some(ref engine) = *harness {
            engine.set_active_capability_delegation(None);
        }
    }
}
