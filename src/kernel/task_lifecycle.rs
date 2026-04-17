use anyhow::Result;
use tokio::sync::oneshot;
use tracing::{info, warn};

use crate::harness::verdict::Verdict;
use crate::kernel::event::{KernelEvent, LifecycleEvent, TaskBranchOutcome, TaskTerminalStatus};
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::session::{PersistedKernelRecord, QueuedTask, SessionState};

impl ExecutionHost {
    pub(crate) async fn complete_task(
        &mut self,
        session: &mut SessionState,
        task: &QueuedTask,
        status: TaskTerminalStatus,
        task_turn_count: u32,
        branch_outcome: Option<TaskBranchOutcome>,
        error_message: Option<String>,
    ) -> Result<()> {
        self.persist_event(
            session,
            &KernelEvent::Lifecycle(LifecycleEvent::TaskComplete {
                identity: session.identity.clone(),
                task_id: task.task_id.clone(),
                trace_id: task.trace_id.clone(),
                plan_id: task.plan_id.clone(),
                status,
                task_turn_count,
                branch_outcome: branch_outcome.clone(),
                error: error_message.clone(),
            }),
        );

        let verdict_result = {
            self.ensure_session_harness_engine(session)?;
            if let Some(harness) = self.session_harness_engine(session) {
                let engine = harness.lock().expect("session harness mutex poisoned");
                Some(engine.evaluate(
                    "on_task_complete",
                    serde_json::json!({
                        "identity": session.identity.clone(),
                        "session_id": self.session_reference(session),
                        "task_id": task.task_id.clone(),
                        "trace_id": task.trace_id.clone(),
                        "plan_id": task.plan_id.clone(),
                        "status": status,
                        "task_turn_count": task_turn_count,
                        "turn_count": session.turn_index,
                        "branch_outcome": branch_outcome,
                        "error": error_message,
                    }),
                ))
            } else {
                None
            }
        };

        if let Some(result) = verdict_result {
            match result {
                Ok(Verdict::Modify(new_tasks_val)) => {
                    let new_tasks =
                        Self::parse_task_list(&new_tasks_val, None, None, Some(&task.trace_id));
                    if !new_tasks.is_empty() {
                        let mut q = session.queue.lock().await;
                        for queued in new_tasks {
                            q.push_back(queued);
                        }
                        info!("on_task_complete queued additional tasks via MODIFY");
                    }
                }
                Ok(Verdict::Reject(reason)) => {
                    warn!(task_id = %task.task_id, trace_id = %task.trace_id, reason = %reason, "on_task_complete rejected");
                }
                Ok(Verdict::Escalate(reason)) => {
                    warn!(task_id = %task.task_id, trace_id = %task.trace_id, reason = %reason, "on_task_complete escalated");
                }
                Ok(Verdict::Allow) => {}
                Err(e) => {
                    warn!(error = %e, "Harness on_task_complete error");
                }
            }
        }

        if let Some(plan_id) = &task.plan_id {
            let completed_plan = if let Some(progress) = session.plans.get_mut(plan_id) {
                progress.completed_tasks += 1;
                if progress.is_complete() {
                    Some(progress.clone())
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(plan) = completed_plan {
                self.persist_event(
                    session,
                    &KernelEvent::Lifecycle(LifecycleEvent::PlanComplete {
                        identity: session.identity.clone(),
                        plan_id: plan.plan_id.clone(),
                        title: plan.title.clone(),
                        total_tasks: plan.total_tasks,
                        completed_tasks: plan.completed_tasks,
                    }),
                );

                {
                    if let Some(harness) = self.session_harness_engine(session)
                        && let Ok(engine) = harness.lock()
                        && let Err(e) = engine.evaluate(
                            "on_plan_complete",
                            serde_json::json!({
                                "identity": session.identity.clone(),
                                "session_id": self.session_reference(session),
                                "plan_id": plan.plan_id.clone(),
                                "title": plan.title.clone(),
                                "total_tasks": plan.total_tasks,
                                "completed_tasks": plan.completed_tasks,
                            }),
                        )
                    {
                        warn!(error = %e, "Harness on_plan_complete failed");
                    }
                }

                session.plans.remove(plan_id);
            }
        }

        self.wait_for_session_durability(session).await;
        Ok(())
    }

    async fn wait_for_session_durability(&self, session: &SessionState) {
        let Some(durability_tx) = &session.durability_tx else {
            return;
        };
        let (tx, rx) = oneshot::channel();
        if durability_tx
            .send(PersistedKernelRecord::Barrier(tx))
            .is_err()
        {
            warn!("Event durability barrier send failed — persistence task unavailable");
            return;
        }
        if tokio::time::timeout(std::time::Duration::from_secs(5), rx)
            .await
            .is_err()
        {
            warn!("Timed out waiting for event durability barrier");
        }
    }

    pub(super) async fn handle_inference_error(
        &mut self,
        session: &mut SessionState,
        task: &QueuedTask,
        error: &str,
    ) -> Result<bool> {
        let verdict_result = {
            self.ensure_session_harness_engine(session)?;
            if let Some(harness) = self.session_harness_engine(session) {
                self.bind_harness_execution_context(session, task);
                let engine = harness.lock().expect("session harness mutex poisoned");
                let result = engine.evaluate(
                    "on_inference_error",
                    serde_json::json!({
                        "identity": session.identity.clone(),
                        "session_id": self.session_reference(session),
                        "task_id": task.task_id.clone(),
                        "trace_id": task.trace_id.clone(),
                        "plan_id": task.plan_id.clone(),
                        "turn_count": session.turn_index,
                        "error": error,
                    }),
                );
                drop(engine);
                self.unbind_harness_execution_context(session);
                Some(result)
            } else {
                None
            }
        };

        if let Some(result) = verdict_result {
            match result {
                Ok(Verdict::Modify(new_tasks_val)) => {
                    let new_tasks = Self::parse_task_list(
                        &new_tasks_val,
                        task.plan_id.as_deref(),
                        task.title.as_deref(),
                        Some(&task.trace_id),
                    );
                    if !new_tasks.is_empty() {
                        let mut q = session.queue.lock().await;
                        for queued in new_tasks {
                            q.push_back(queued);
                        }
                        info!(
                            task_id = %task.task_id,
                            trace_id = %task.trace_id,
                            "on_inference_error queued additional tasks via MODIFY"
                        );
                        return Ok(true);
                    }
                }
                Ok(Verdict::Reject(reason)) => {
                    warn!(task_id = %task.task_id, trace_id = %task.trace_id, reason = %reason, "on_inference_error rejected");
                }
                Ok(Verdict::Escalate(reason)) => {
                    warn!(task_id = %task.task_id, trace_id = %task.trace_id, reason = %reason, "on_inference_error escalated");
                }
                Ok(Verdict::Allow) => {}
                Err(e) => {
                    warn!(error = %e, "Harness on_inference_error error");
                }
            }
        }

        Ok(false)
    }
}
