use anyhow::{Result, anyhow};

use crate::kernel::event::TaskTerminalStatus;
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::policy::PolicyScope;
use crate::kernel::session::{PlanProgress, QueuedTask, SessionState, TaskExecutionOverrides};

impl ExecutionHost {
    pub(crate) fn policy_scope_for_session(&self, session: &SessionState) -> PolicyScope {
        PolicyScope {
            agent_id: Some(session.identity.agent_id().to_string()),
            session_id: Some(session.identity.session_id().to_string()),
            run_id: session.identity.run_id().map(str::to_string),
            ..PolicyScope::default()
        }
    }

    pub(crate) async fn enqueue_session_task(
        &self,
        session: &mut SessionState,
        mut task: QueuedTask,
    ) -> Result<()> {
        let policy = self
            .policy_manager
            .typed_snapshot(&self.policy_scope_for_session(session))
            .await;
        let mut q = session.queue.lock().await;
        if q.len() >= policy.queue_max_depth {
            return Err(anyhow!(
                "Policy denial: queue.max_depth={} reached",
                policy.queue_max_depth
            ));
        }
        if task.task_id.is_empty() {
            task.task_id = format!("t_{}", session.next_task_id);
            session.next_task_id += 1;
        }
        q.push_back(task);
        Ok(())
    }

    /// Add a prompt to the end of the queue as an implicit single-task plan.
    pub async fn queue_prompt(&self, session: &mut SessionState, prompt: String) -> Result<()> {
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
        let task = QueuedTask::with_plan(prompt, plan_id, Some("queued_prompt".to_string()));
        self.enqueue_session_task(session, task).await
    }

    pub(crate) fn parse_task_list(
        tasks_val: &serde_json::Value,
        default_plan_id: Option<&str>,
        default_title: Option<&str>,
        default_trace_id: Option<&str>,
    ) -> Vec<QueuedTask> {
        let Some(items) = tasks_val.as_array() else {
            return Vec::new();
        };

        items
            .iter()
            .filter_map(|item| {
                if let Some(prompt) = item.as_str() {
                    if let Some(plan_id) = default_plan_id {
                        return Some(
                            QueuedTask::with_plan(
                                prompt.to_string(),
                                plan_id.to_string(),
                                default_title.map(ToString::to_string),
                            )
                            .with_inherited_trace(default_trace_id),
                        );
                    }
                    return Some(
                        QueuedTask::ad_hoc(prompt.to_string())
                            .with_inherited_trace(default_trace_id),
                    );
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
                let execution = obj
                    .get("execution")
                    .and_then(|value| {
                        serde_json::from_value::<TaskExecutionOverrides>(value.clone()).ok()
                    })
                    .filter(|execution| !execution.is_empty());
                match plan_id {
                    Some(plan_id) => Some(
                        QueuedTask::with_plan(prompt.to_string(), plan_id, title)
                            .with_inherited_trace(default_trace_id)
                            .with_execution(execution),
                    ),
                    None => Some(
                        QueuedTask::ad_hoc(prompt.to_string())
                            .with_inherited_trace(default_trace_id)
                            .with_execution(execution),
                    ),
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
            self.complete_task(
                session,
                &queued,
                TaskTerminalStatus::Cancelled,
                0,
                None,
                None,
            )
            .await?;
        }
        Ok(cancelled)
    }
}
