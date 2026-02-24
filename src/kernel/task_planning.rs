use anyhow::Result;

use crate::kernel::Kernel;
use crate::kernel::event::TaskTerminalStatus;
use crate::kernel::session::{PlanProgress, QueuedTask, SessionState};

impl Kernel {
    /// Add a prompt to the end of the queue as an implicit single-task plan.
    pub async fn queue_prompt(&self, session: &mut SessionState, prompt: String) {
        let plan_id = format!("p_{}", session.next_plan_id);
        session.next_plan_id += 1;
        session.plans.insert(
            plan_id.clone(),
            PlanProgress {
                plan_id: plan_id.clone(),
                title: "queued_prompt".to_string(),
                total_tasks: 1,
                completed_tasks: 0,
            },
        );
        let mut q = session.queue.lock().await;
        let mut task = QueuedTask::with_plan(prompt, plan_id, Some("queued_prompt".to_string()));
        task.task_id = format!("t_{}", session.next_task_id);
        session.next_task_id += 1;
        q.push_back(task);
    }

    pub(crate) fn parse_task_list(
        tasks_val: &serde_json::Value,
        default_plan_id: Option<&str>,
        default_title: Option<&str>,
    ) -> Vec<QueuedTask> {
        let Some(items) = tasks_val.as_array() else {
            return Vec::new();
        };

        items
            .iter()
            .filter_map(|item| {
                if let Some(prompt) = item.as_str() {
                    if let Some(plan_id) = default_plan_id {
                        return Some(QueuedTask::with_plan(
                            prompt.to_string(),
                            plan_id.to_string(),
                            default_title.map(ToString::to_string),
                        ));
                    }
                    return Some(QueuedTask::ad_hoc(prompt.to_string()));
                }

                let obj = item.as_object()?;
                let prompt = obj.get("prompt").and_then(|v| v.as_str())?;
                let plan_id = obj
                    .get("plan_id")
                    .and_then(|v| v.as_str())
                    .map(ToString::to_string)
                    .or_else(|| default_plan_id.map(ToString::to_string));
                let title = obj
                    .get("title")
                    .and_then(|v| v.as_str())
                    .map(ToString::to_string)
                    .or_else(|| default_title.map(ToString::to_string));
                match plan_id {
                    Some(plan_id) => {
                        Some(QueuedTask::with_plan(prompt.to_string(), plan_id, title))
                    }
                    None => Some(QueuedTask::ad_hoc(prompt.to_string())),
                }
            })
            .collect()
    }

    pub(crate) async fn cancel_queued_tasks(
        &mut self,
        session: &mut SessionState,
    ) -> Result<usize> {
        let drained_tasks: Vec<QueuedTask> = {
            let mut q = session.queue.lock().await;
            q.drain(..).collect()
        };

        let cancelled = drained_tasks.len();
        for queued in drained_tasks {
            self.complete_task(session, &queued, TaskTerminalStatus::Cancelled, 0, None)
                .await?;
        }
        Ok(cancelled)
    }
}
