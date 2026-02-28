use std::collections::BTreeMap;

use anyhow::Result;
use tracing::{error, info, warn};

use crate::harness::verdict::Verdict;
use crate::inference::provider::{InferenceContent, InferenceRole};
use crate::kernel::event::{KernelEvent, LifecycleEvent, TaskTerminalStatus};
use crate::kernel::session::{QueuedTask, SessionState};
use crate::kernel::{Kernel, TaskExecutionResult};

#[derive(Debug)]
pub(super) struct PeerRunOutcome {
    pub(super) runtime_task_id: String,
    pub(super) status: TaskTerminalStatus,
    pub(super) task_turn_count: u32,
    pub(super) output: Option<String>,
}

pub(super) async fn run_peer_task(
    kernel: &mut Kernel,
    session: &mut SessionState,
    mut task: QueuedTask,
    delegated_capabilities: Option<BTreeMap<String, bool>>,
) -> Result<PeerRunOutcome> {
    if task.task_id.is_empty() {
        task.task_id = format!("t_{}", session.next_task_id);
        session.next_task_id += 1;
    }

    set_peer_task_capability_ceiling(kernel, delegated_capabilities.clone());
    let outcome = async {
        kernel.persist_event(
            session,
            &KernelEvent::Lifecycle(LifecycleEvent::TaskStart {
                identity: session.identity.clone(),
                task_id: task.task_id.clone(),
                plan_id: task.plan_id.clone(),
                title: task.title.clone(),
                prompt: task.prompt.clone(),
                queue_depth: 0,
            }),
        );

        let task_start_verdict = {
            let harness = kernel.lock_harness();
            if let Some(ref engine) = *harness {
                match engine.evaluate(
                    "on_task_start",
                    serde_json::json!({
                        "identity": session.identity.clone(),
                        "session_id": session.identity.session_id(),
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
                kernel
                    .complete_task(session, &task, TaskTerminalStatus::Rejected, 0, None)
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
                kernel
                    .complete_task(session, &task, TaskTerminalStatus::Rejected, 0, None)
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

        let run_result: TaskExecutionResult = match kernel.run_task(session, &task).await {
            Ok(result) => {
                kernel
                    .complete_task(session, &task, result.status, result.task_turn_count, None)
                    .await?;
                result
            }
            Err(e) => {
                error!(task_id = %task.task_id, error = %e, "Peer task failed with runtime error");
                let error_message = e.to_string();
                let recovered = kernel
                    .handle_inference_error(session, &task, &error_message)
                    .await?;
                kernel
                    .complete_task(
                        session,
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

        let output = last_assistant_text(session);

        Ok(PeerRunOutcome {
            runtime_task_id: task.task_id,
            status: run_result.status,
            task_turn_count: run_result.task_turn_count,
            output,
        })
    }
    .await;
    clear_peer_task_capability_ceiling(kernel);
    outcome
}

fn last_assistant_text(session: &SessionState) -> Option<String> {
    session.history.iter().rev().find_map(|msg| {
        if msg.role != InferenceRole::Assistant {
            return None;
        }
        msg.content.iter().find_map(|c| match c {
            InferenceContent::Text { text } => Some(text.clone()),
            _ => None,
        })
    })
}

fn set_peer_task_capability_ceiling(kernel: &Kernel, caps: Option<BTreeMap<String, bool>>) {
    let harness = kernel.lock_harness();
    if let Some(ref engine) = *harness {
        engine.set_active_capability_delegation(caps);
    }
}

fn clear_peer_task_capability_ceiling(kernel: &Kernel) {
    let harness = kernel.lock_harness();
    if let Some(ref engine) = *harness {
        engine.set_active_capability_delegation(None);
    }
}
