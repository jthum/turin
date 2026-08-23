use anyhow::Result;
use tokio::sync::oneshot;
use tracing::{info, warn};

use crate::harness::verdict::Verdict;
use crate::kernel::event::{KernelEvent, LifecycleEvent, TaskBranchOutcome, TaskTerminalStatus};
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::harness_contract::HarnessHook;
use crate::kernel::session::{
    ExecutionStatusSnapshot, PersistedKernelRecord, QueuedTask, SessionState,
};

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
                execution: ExecutionStatusSnapshot::from_session(session),
                branch_outcome: branch_outcome.clone(),
                error: error_message.clone(),
            }),
        )
        .await;
        let task_budget = session.active_task_budget_snapshot(task_turn_count);

        let verdict_result = {
            self.ensure_session_harness_engine(session)?;
            if let Some(harness) = self.session_harness_engine(session) {
                let engine = harness.lock().expect("session harness mutex poisoned");
                let session_id = self.session_reference(session);
                let execution = ExecutionStatusSnapshot::from_session(session);
                Some(engine.evaluate_hook(HarnessHook::TaskComplete {
                    identity: &session.identity,
                    session_id: &session_id,
                    task_id: &task.task_id,
                    trace_id: &task.trace_id,
                    plan_id: task.plan_id.as_deref(),
                    status,
                    task_turn_count,
                    task_started_at_unix_ms: task_budget.task_started_at_unix_ms,
                    task_elapsed_ms: task_budget.task_elapsed_ms,
                    task_input_tokens: task_budget.task_input_tokens,
                    task_output_tokens: task_budget.task_output_tokens,
                    task_total_tokens: task_budget.task_total_tokens,
                    turn_count: session.turn_index,
                    execution: &execution,
                    branch_outcome: branch_outcome.as_ref(),
                    error: error_message.as_deref(),
                }))
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
                )
                .await;

                {
                    if let Some(harness) = self.session_harness_engine(session)
                        && let Ok(engine) = harness.lock()
                    {
                        let session_id = self.session_reference(session);
                        if let Err(e) = engine.evaluate_hook(HarnessHook::PlanComplete {
                            identity: &session.identity,
                            session_id: &session_id,
                            plan_id: &plan.plan_id,
                            title: &plan.title,
                            total_tasks: plan.total_tasks,
                            completed_tasks: plan.completed_tasks,
                        }) {
                            warn!(error = %e, "Harness on_plan_complete failed");
                        }
                    }
                }

                session.plans.remove(plan_id);
            }
        }

        self.record_local_completed_task(
            session,
            task,
            status,
            task_turn_count,
            branch_outcome,
            error_message,
        )
        .await;

        self.wait_for_session_durability(session).await?;
        Ok(())
    }

    pub(super) async fn wait_for_session_durability(&self, session: &SessionState) -> Result<()> {
        let Some(durability_tx) = &session.durability_tx else {
            return Ok(());
        };
        let (tx, rx) = oneshot::channel();
        durability_tx
            .send(PersistedKernelRecord::Barrier(tx))
            .await
            .map_err(|_| anyhow::anyhow!("Event durability lane is unavailable"))?;
        match tokio::time::timeout(std::time::Duration::from_secs(5), rx).await {
            Ok(Ok(Ok(()))) => Ok(()),
            Ok(Ok(Err(error))) => Err(anyhow::anyhow!("Event durability write failed: {error}")),
            Ok(Err(_)) => Err(anyhow::anyhow!(
                "Event durability lane closed before acknowledging the barrier"
            )),
            Err(_) => Err(anyhow::anyhow!(
                "Timed out waiting for event durability barrier"
            )),
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
                let session_id = self.session_reference(session);
                let result = engine.evaluate_hook(HarnessHook::InferenceError {
                    identity: &session.identity,
                    session_id: &session_id,
                    task_id: &task.task_id,
                    trace_id: &task.trace_id,
                    plan_id: task.plan_id.as_deref(),
                    turn_count: session.turn_index,
                    error,
                });
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
