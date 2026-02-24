use tracing::{debug, warn};

use crate::harness::verdict::Verdict;
use crate::kernel::event::{KernelEvent, LifecycleEvent};
use crate::kernel::session::{PlanProgress, QueuedTask, SessionState};
use crate::kernel::Kernel;

impl Kernel {
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
            let harness = self.lock_harness();
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
            let new_tasks = Self::parse_task_list(&new_tasks_val, None, None);
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
}
