use anyhow::{Context, Result};
use tracing::{debug, info, instrument, warn};

use crate::harness::verdict::Verdict;
use crate::kernel::event::TaskTerminalStatus;
use crate::kernel::event::{KernelEvent, LifecycleEvent};
use crate::kernel::execution_host::{ExecutionHost, TaskRunAttempt};
use crate::kernel::harness_contract::HarnessHook;
use crate::kernel::session::{PlanProgress, QueuedTask, SessionState};
use crate::kernel::session_refs::describe_store_selector;

impl ExecutionHost {
    /// Run the agent loop with the given prompt.
    #[instrument(
        skip(self, session),
        fields(
            session_id = %self.session_reference(session),
            store = %describe_store_selector(&session.store_selector)
        )
    )]
    pub(crate) async fn run(
        &mut self,
        session: &mut SessionState,
        prompt: Option<String>,
    ) -> Result<()> {
        // Ensure session is started
        self.start_session(session).await?;

        if let Some(p) = prompt {
            self.enqueue_initial_run_prompt(session, p).await?;
        }

        while let Some((mut task, queue_depth_after_pop)) = self.dequeue_next_task(session).await {
            session.set_active_task_conflict_policy(task.conflict_policy);
            session.set_current_task_branch_outcome(task.branch_outcome.clone());
            session.begin_active_task_budget();
            if let Err(error) = self.begin_task_execution_scope(session, &task).await {
                let error_message = error.to_string();
                let completion = self
                    .complete_task(
                        session,
                        &task,
                        TaskTerminalStatus::Error,
                        0,
                        None,
                        Some(error_message),
                    )
                    .await;
                self.finish_task_scope_after(session, completion).await?;
                if session.stop_requested {
                    break;
                }
                continue;
            }
            if session.cancel_token.is_cancelled() {
                let completion = self
                    .complete_task(session, &task, TaskTerminalStatus::Cancelled, 0, None, None)
                    .await;
                self.finish_task_scope_after(session, completion).await?;
                if session.stop_requested {
                    break;
                }
                continue;
            }
            let should_start = self
                .prepare_task_start(session, &mut task, queue_depth_after_pop)
                .await;
            let should_start = match should_start {
                Ok(should_start) => should_start,
                Err(error) => {
                    return self.finish_task_scope_after(session, Err(error)).await;
                }
            };
            if !should_start {
                self.finish_task_execution_scope(session).await?;
                continue;
            }

            info!(
                task_id = %task.task_id,
                trace_id = %task.trace_id,
                store = %describe_store_selector(&session.store_selector),
                prompt_bytes = task.prompt.len(),
                "Running task"
            );

            let task_attempt = self.run_task_with_conflict_handling(session, &task).await;
            let task_attempt = match task_attempt {
                Ok(task_attempt) => task_attempt,
                Err(error) => {
                    return self.finish_task_scope_after(session, Err(error)).await;
                }
            };
            let task_result = match task_attempt {
                TaskRunAttempt::Completed(result) => result,
                TaskRunAttempt::Terminal {
                    status,
                    error_message,
                } => {
                    let finalization = async {
                        self.complete_task(session, &task, status, 0, None, Some(error_message))
                            .await?;
                        self.apply_pending_branch_checkout(session).await
                    }
                    .await;
                    self.finish_task_scope_after(session, finalization).await?;
                    if session.stop_requested {
                        break;
                    }
                    continue;
                }
                TaskRunAttempt::Error {
                    error,
                    error_message,
                    recovered,
                } => {
                    let finalization = async {
                        self.complete_task(
                            session,
                            &task,
                            TaskTerminalStatus::Error,
                            0,
                            None,
                            Some(error_message),
                        )
                        .await?;
                        self.apply_pending_branch_checkout(session).await
                    }
                    .await;
                    self.finish_task_scope_after(session, finalization).await?;
                    if recovered {
                        continue;
                    }
                    return Err(error);
                }
            };

            let finalization = async {
                self.complete_task(
                    session,
                    &task,
                    task_result.status,
                    task_result.task_turn_count,
                    task_result.branch_outcome,
                    None,
                )
                .await?;
                self.apply_pending_branch_checkout(session).await
            }
            .await;
            self.finish_task_scope_after(session, finalization).await?;

            if session.stop_requested || task_result.status == TaskTerminalStatus::Cancelled {
                info!(
                    session_id = %self.session_reference(session),
                    "Stopping run loop due to session stop request"
                );
                break;
            }
        }

        Ok(())
    }

    async fn finish_task_scope_after<T>(
        &mut self,
        session: &mut SessionState,
        result: Result<T>,
    ) -> Result<T> {
        let cleanup = self.finish_task_execution_scope(session).await;
        match (result, cleanup) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(cleanup_error)) => Err(cleanup_error),
            (Err(error), Err(cleanup_error)) => Err(error.context(format!(
                "Task execution scope cleanup also failed: {cleanup_error}"
            ))),
        }
    }

    pub(crate) async fn apply_pending_branch_checkout(
        &mut self,
        session: &mut SessionState,
    ) -> Result<()> {
        let branch_name = {
            let Some(harness) = self.session_harness_engine(session) else {
                return Ok(());
            };
            let engine = harness.lock().expect("session harness mutex poisoned");
            engine.take_pending_session_branch_checkout()
        };
        let Some(branch_name) = branch_name else {
            return Ok(());
        };

        let Some(internal_id) = session.internal_id else {
            warn!(
                session_id = %self.session_reference(session),
                store = %describe_store_selector(&session.store_selector),
                branch = %branch_name,
                "Skipping deferred branch checkout for session without persistence id"
            );
            return Ok(());
        };

        let store = self
            .store_manager
            .open(&session.store_selector)
            .await
            .context("Deferred branch checkout requires a configured persistent state store")?;
        match store
            .checkout_branch_head_by_name(internal_id, &branch_name)
            .await
        {
            Ok(Some(branch)) => {
                session.set_selected_branch_head_id(Some(branch.id));
                info!(
                    session_id = %self.session_reference(session),
                    store = %describe_store_selector(&session.store_selector),
                    branch = %branch_name,
                    "Applied deferred branch checkout"
                );
                self.refresh_session_from_persistence(session).await?;
            }
            Ok(None) => {
                warn!(
                    session_id = %self.session_reference(session),
                    store = %describe_store_selector(&session.store_selector),
                    branch = %branch_name,
                    "Deferred branch checkout target no longer exists"
                );
            }
            Err(error) => {
                warn!(
                    session_id = %self.session_reference(session),
                    store = %describe_store_selector(&session.store_selector),
                    branch = %branch_name,
                    error = %error,
                    "Deferred branch checkout failed"
                );
            }
        }

        Ok(())
    }

    pub(super) async fn enqueue_initial_run_prompt(
        &self,
        session: &mut SessionState,
        prompt: String,
    ) -> Result<()> {
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

        let task = QueuedTask::with_plan(prompt, plan_id, Some("user_request".to_string()));
        self.enqueue_session_task(session, task).await
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
        )
        .await;

        let verdict = {
            if let Err(error) = self.ensure_session_harness_engine(session) {
                warn!(error = %error, "Failed to refresh session harness before on_all_tasks_complete");
                None
            } else if let Some(harness) = self.session_harness_engine(session) {
                let engine = harness.lock().expect("session harness mutex poisoned");
                let session_id = self.session_reference(session);
                match engine.evaluate_hook(HarnessHook::AllTasksComplete {
                    identity: &session.identity,
                    session_id: &session_id,
                    turn_count: session.turn_index,
                }) {
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
            let new_tasks = ExecutionHost::parse_task_list(&new_tasks_val, None, None, None);
            if !new_tasks.is_empty() {
                for task in new_tasks {
                    if let Err(error) = self.enqueue_session_task(session, task).await {
                        warn!(%error, "on_all_tasks_complete enqueue denied");
                        break;
                    }
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
        self.ensure_session_harness_engine(session)?;
        self.persist_event(
            session,
            &KernelEvent::Lifecycle(LifecycleEvent::TaskStart {
                identity: session.identity.clone(),
                task_id: task.task_id.clone(),
                trace_id: task.trace_id.clone(),
                plan_id: task.plan_id.clone(),
                title: task.title.clone(),
                prompt: task.prompt.clone(),
                queue_depth: queue_depth_after_pop,
                execution: crate::kernel::session::ExecutionStatusSnapshot::from_session(session),
            }),
        )
        .await;

        match self.evaluate_task_start_verdict(session, task, queue_depth_after_pop) {
            Verdict::Reject(reason) => {
                warn!(
                    task_id = %task.task_id,
                    trace_id = %task.trace_id,
                    reason = %reason,
                    "Task rejected by on_task_start"
                );
                self.complete_task(session, task, TaskTerminalStatus::Rejected, 0, None, None)
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
                    trace_id = %task.trace_id,
                    reason = %reason,
                    "Task escalated at on_task_start; treating as rejected"
                );
                self.complete_task(session, task, TaskTerminalStatus::Rejected, 0, None, None)
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
        if let Some(harness) = self.session_harness_engine(session) {
            let engine = harness.lock().expect("session harness mutex poisoned");
            let session_id = self.session_reference(session);
            match engine.evaluate_hook(HarnessHook::TaskStart {
                identity: &session.identity,
                session_id: &session_id,
                task_id: &task.task_id,
                trace_id: &task.trace_id,
                plan_id: task.plan_id.as_deref(),
                title: task.title.as_deref(),
                prompt: &task.prompt,
                queue_depth: queue_depth_after_pop,
            }) {
                Ok(v) => v,
                Err(e) => self.harness_eval_error_verdict("on_task_start", e)
            }
        } else {
            Verdict::Allow
        }
    }
}
