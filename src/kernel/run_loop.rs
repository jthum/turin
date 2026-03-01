use anyhow::Result;
use tracing::{debug, error, info, instrument, warn};

use crate::harness::verdict::Verdict;
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::event::TaskTerminalStatus;
use crate::kernel::event::{KernelEvent, LifecycleEvent};
use crate::kernel::session::{PlanProgress, QueuedTask, SessionState};

impl ExecutionHost {
    /// Run the agent loop with the given prompt.
    #[instrument(skip(self, session), fields(session_id = %session.identity.session_id()))]
    pub async fn run(&mut self, session: &mut SessionState, prompt: Option<String>) -> Result<()> {
        // Ensure session is started
        self.start_session(session).await?;

        if let Some(p) = prompt {
            self.enqueue_initial_run_prompt(session, p).await;
        }

        while let Some((mut task, queue_depth_after_pop)) = self.dequeue_next_task(session).await {
            if !self
                .prepare_task_start(session, &mut task, queue_depth_after_pop)
                .await?
            {
                continue;
            }

            info!(task_id = %task.task_id, prompt = %task.prompt, "Running task");

            let task_result = match self.run_task(session, &task).await {
                Ok(result) => result,
                Err(e) => {
                    error!(task_id = %task.task_id, error = %e, "Task failed with runtime error");
                    let error_message = e.to_string();
                    let recovered = self
                        .handle_inference_error(session, &task, &error_message)
                        .await?;
                    self.complete_task(
                        session,
                        &task,
                        TaskTerminalStatus::Error,
                        0,
                        Some(error_message),
                    )
                    .await?;
                    if recovered {
                        continue;
                    }
                    return Err(e);
                }
            };

            self.complete_task(
                session,
                &task,
                task_result.status,
                task_result.task_turn_count,
                None,
            )
            .await?;

            if session.stop_requested {
                info!(
                    session_id = %session.identity.session_id(),
                    "Stopping run loop due to session stop request"
                );
                break;
            }
        }

        Ok(())
    }

    pub(super) async fn enqueue_initial_run_prompt(
        &self,
        session: &mut SessionState,
        prompt: String,
    ) {
        let plan_id = format!("p_{}", session.next_plan_id);
        session.next_plan_id += 1;

        session.plans.insert(
            plan_id.clone(),
            PlanProgress {
                plan_id: plan_id.clone(),
                title: "user_request".to_string(),
                total_tasks: 1,
                completed_tasks: 0,
            },
        );

        let mut q = session.queue.lock().await;
        let mut task = QueuedTask::with_plan(prompt, plan_id, Some("user_request".to_string()));
        task.task_id = format!("t_{}", session.next_task_id);
        session.next_task_id += 1;
        q.push_back(task);
    }

    pub(super) async fn dequeue_next_task(
        &mut self,
        session: &mut SessionState,
    ) -> Option<(QueuedTask, usize)> {
        loop {
            let next = {
                let mut q = session.queue.lock().await;
                if q.is_empty() {
                    None
                } else {
                    let task = q.pop_front().expect("queue checked non-empty");
                    let depth = q.len();
                    Some((task, depth))
                }
            };

            if let Some(task) = next {
                return Some(task);
            }

            if self.handle_all_tasks_complete(session).await {
                continue;
            }

            return None;
        }
    }

    async fn handle_all_tasks_complete(&mut self, session: &mut SessionState) -> bool {
        debug!("Queue empty, firing on_all_tasks_complete");
        self.persist_event(
            session,
            &KernelEvent::Lifecycle(LifecycleEvent::AllTasksComplete {
                identity: session.identity.clone(),
            }),
        );

        let verdict = {
            let runtime = self.runtime_for_session(session);
            let harness = runtime.lock_engine();
            if let Some(ref engine) = *harness {
                match engine.evaluate(
                    "on_all_tasks_complete",
                    serde_json::json!({
                        "identity": session.identity.clone(),
                        "session_id": session.identity.session_id(),
                        "turn_count": session.turn_index,
                    }),
                ) {
                    Ok(v) => Some(v),
                    Err(e) => {
                        warn!(error = %e, "Harness on_all_tasks_complete failed");
                        None
                    }
                }
            } else {
                None
            }
        };

        if let Some(Verdict::Modify(new_tasks_val)) = verdict {
            let new_tasks = ExecutionHost::parse_task_list(&new_tasks_val, None, None);
            if !new_tasks.is_empty() {
                let mut q = session.queue.lock().await;
                for task in new_tasks {
                    q.push_back(task);
                }
                return true;
            }
        }

        false
    }

    pub(super) async fn prepare_task_start(
        &mut self,
        session: &mut SessionState,
        task: &mut QueuedTask,
        queue_depth_after_pop: usize,
    ) -> Result<bool> {
        self.persist_event(
            session,
            &KernelEvent::Lifecycle(LifecycleEvent::TaskStart {
                identity: session.identity.clone(),
                task_id: task.task_id.clone(),
                plan_id: task.plan_id.clone(),
                title: task.title.clone(),
                prompt: task.prompt.clone(),
                queue_depth: queue_depth_after_pop,
            }),
        );

        match self.evaluate_task_start_verdict(session, task, queue_depth_after_pop) {
            Verdict::Reject(reason) => {
                warn!(
                    task_id = %task.task_id,
                    reason = %reason,
                    "Task rejected by on_task_start"
                );
                self.complete_task(session, task, TaskTerminalStatus::Rejected, 0, None)
                    .await?;
                Ok(false)
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
                Ok(true)
            }
            Verdict::Escalate(reason) => {
                warn!(
                    task_id = %task.task_id,
                    reason = %reason,
                    "Task escalated at on_task_start; treating as rejected"
                );
                self.complete_task(session, task, TaskTerminalStatus::Rejected, 0, None)
                    .await?;
                Ok(false)
            }
            Verdict::Allow => Ok(true),
        }
    }

    fn evaluate_task_start_verdict(
        &self,
        session: &SessionState,
        task: &QueuedTask,
        queue_depth_after_pop: usize,
    ) -> Verdict {
        let runtime = self.runtime_for_session(session);
        let harness = runtime.lock_engine();
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
                    "queue_depth": queue_depth_after_pop,
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
    }
}
