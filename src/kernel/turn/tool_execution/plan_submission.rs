use tracing::error;

use crate::harness::verdict::Verdict;
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::session::{PlanProgress, QueuedTask, SessionState};

impl ExecutionHost {
    pub(super) async fn handle_plan_submission(
        &mut self,
        session: &mut SessionState,
        title: &str,
        tasks: Vec<String>,
        clear_existing: bool,
    ) -> (String, bool) {
        let mut plan_title = title.to_string();
        let mut plan_tasks = tasks;
        let mut should_clear_existing = clear_existing;

        let verdict_result = {
            let runtime = self.runtime_for_session(session);
            let harness = runtime.lock_engine();
            (*harness).as_ref().map(|engine| {
                engine.evaluate(
                    "on_plan_submit",
                    serde_json::json!({
                        "title": plan_title.clone(),
                        "tasks": plan_tasks.clone(),
                        "clear_existing": should_clear_existing,
                    }),
                )
            })
        };

        match verdict_result {
            Some(Ok(Verdict::Allow)) | None => {}
            Some(Ok(Verdict::Reject(reason))) => {
                return (format!("Plan rejected by harness: {}", reason), true);
            }
            Some(Ok(Verdict::Escalate(reason))) => {
                if !self.prompt_for_approval(&reason) {
                    return (format!("Plan escalation denied by user: {}", reason), true);
                }
            }
            Some(Ok(Verdict::Modify(new_val))) => {
                if let Some(obj) = new_val.as_object() {
                    if let Some(new_title) = obj.get("title").and_then(|v| v.as_str()) {
                        plan_title = new_title.to_string();
                    }
                    if let Some(new_clear) = obj.get("clear_existing").and_then(|v| v.as_bool()) {
                        should_clear_existing = new_clear;
                    }
                    if let Some(new_tasks_val) = obj.get("tasks") {
                        plan_tasks = ExecutionHost::parse_task_list(new_tasks_val, None, None)
                            .into_iter()
                            .map(|t| t.prompt)
                            .collect();
                    }
                } else if new_val.is_array() {
                    plan_tasks = ExecutionHost::parse_task_list(&new_val, None, None)
                        .into_iter()
                        .map(|t| t.prompt)
                        .collect();
                }
            }
            Some(Err(e)) => {
                error!(error = %e, "Failed to evaluate on_plan_submit");
            }
        }

        if plan_tasks.is_empty() {
            return (
                "Plan submission rejected: no tasks were provided".to_string(),
                true,
            );
        }

        let cancelled_count = if should_clear_existing {
            match self.cancel_queued_tasks(session).await {
                Ok(cancelled) => cancelled,
                Err(e) => {
                    return (
                        format!("Plan submission failed while clearing queue: {}", e),
                        true,
                    );
                }
            }
        } else {
            0
        };

        let plan_id = format!("p_{}", session.next_plan_id);
        session.next_plan_id += 1;
        session.plans.insert(
            plan_id.clone(),
            PlanProgress {
                plan_id: plan_id.clone(),
                title: plan_title.clone(),
                total_tasks: plan_tasks.len(),
                completed_tasks: 0,
            },
        );

        let scheduled_tasks = plan_tasks
            .into_iter()
            .map(|prompt| {
                let mut qt =
                    QueuedTask::with_plan(prompt, plan_id.clone(), Some(plan_title.clone()));
                qt.task_id = format!("t_{}", session.next_task_id);
                session.next_task_id += 1;
                qt
            })
            .collect::<Vec<_>>();

        let queued_count = scheduled_tasks.len();
        {
            let mut q = session.queue.lock().await;
            for task in scheduled_tasks {
                q.push_back(task);
            }
        }

        if cancelled_count > 0 {
            (
                format!(
                    "Plan '{}' submitted (plan_id: {}) with {} tasks. Cancelled {} queued tasks.",
                    plan_title, plan_id, queued_count, cancelled_count
                ),
                false,
            )
        } else {
            (
                format!(
                    "Plan '{}' submitted (plan_id: {}) with {} tasks.",
                    plan_title, plan_id, queued_count
                ),
                false,
            )
        }
    }
}
