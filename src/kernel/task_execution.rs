use anyhow::Result;
use std::sync::Arc;
use tracing::{error, instrument, warn};

use crate::inference::provider::{InferenceContent, InferenceMessage, InferenceRole};
use crate::kernel::TaskExecutionResult;
use crate::kernel::config::AgentMode;
use crate::kernel::event::TaskTerminalStatus;
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::harness_hooks::TokenUsageHookAction;
use crate::kernel::session::{QueuedTask, SessionState};
use crate::kernel::turn;
use crate::tools::ToolContext;

impl ExecutionHost {
    /// Execute a single task (one specific prompt) within the persistent session.
    #[instrument(skip(self, session, task), fields(task_id = %task.task_id, trace_id = %task.trace_id))]
    pub(super) async fn run_task(
        &mut self,
        session: &mut SessionState,
        task: &QueuedTask,
    ) -> Result<TaskExecutionResult> {
        let session_id = session.identity.session_id().to_string();
        let prompt = task.prompt.as_str();

        if session.cancel_token.is_cancelled() {
            return Ok(TaskExecutionResult {
                status: TaskTerminalStatus::Cancelled,
                task_turn_count: 0,
            });
        }

        self.append_task_user_message(session, prompt);

        let effective_tools = crate::tools::policy::resolve_effective_tools_config(
            &self.config,
            session.identity.agent_id(),
            task.tools.as_ref(),
        )?;
        let allowed_native_tools = Arc::new(
            effective_tools
                .selection
                .allow
                .clone()
                .unwrap_or_default()
                .into_iter()
                .collect(),
        );

        let tool_ctx = ToolContext {
            workspace_root: std::path::PathBuf::from(&self.config.kernel.workspace_root),
            session_id: session_id.clone(),
            agent_id: session.identity.agent_id().to_string(),
            store_manager: Some(self.store_manager.clone()),
            embedding_provider: self.embedding_provider.clone(),
            config: Some(self.config.clone()),
            allowed_native_tools: Arc::clone(&allowed_native_tools),
            tools: Arc::new(effective_tools),
        };

        self.persist_task_user_message(session, prompt).await;
        self.set_task_active_session(session, task);

        let task_status_result = self.run_task_turn_loop(session, task, &tool_ctx).await;

        self.clear_task_active_session(session);

        let (status, task_turn_count) = task_status_result?;
        Ok(TaskExecutionResult {
            status,
            task_turn_count,
        })
    }

    fn append_task_user_message(&self, session: &mut SessionState, prompt: &str) {
        session.history.push(InferenceMessage {
            role: InferenceRole::User,
            content: vec![InferenceContent::Text {
                text: prompt.to_string(),
            }],
            tool_call_id: None,
        });
    }

    async fn persist_task_user_message(&self, session: &SessionState, prompt: &str) {
        if let Ok(store) = self.store_manager.open(&session.store_selector).await {
            if let Some(iid) = session.internal_id {
                let _ = store
                    .insert_message(
                        iid,
                        session.turn_index,
                        "user",
                        &serde_json::json!([{"type": "text", "text": prompt}]),
                        None,
                    )
                    .await;
            } else {
                warn!("Session missing internal_id, skipping message persistence");
            }
        }
    }

    fn set_task_active_session(&self, session: &SessionState, task: &QueuedTask) {
        let runtime = self.runtime_for_session(session);
        let harness = runtime.lock_engine();
        if let Some(ref engine) = *harness {
            engine.set_active_session(
                Some(session.identity.session_id()),
                Some(session.store_selector.clone()),
                Some(session.mode.clone()),
            );
            engine.set_active_trace_id(Some(&task.trace_id));
            engine.set_active_event_context(Some(crate::harness::globals::HarnessEventContext {
                json: self.json,
                internal_id: session.internal_id,
                event_tx: session.event_tx.clone(),
                durability_tx: session.durability_tx.clone(),
            }));
        }
    }

    fn clear_task_active_session(&self, session: &SessionState) {
        let runtime = self.runtime_for_session(session);
        let harness = runtime.lock_engine();
        if let Some(ref engine) = *harness {
            engine.set_active_session(None, None, None);
            engine.set_active_trace_id(None);
            engine.set_active_event_context(None);
        }
    }

    async fn run_task_turn_loop(
        &mut self,
        session: &mut SessionState,
        task: &QueuedTask,
        tool_ctx: &ToolContext,
    ) -> Result<(TaskTerminalStatus, u32)> {
        let mut task_turn_count = 0;
        let max_task_turns = self.config.kernel.max_turns;

        let task_status_result: Result<TaskTerminalStatus> = loop {
            if session.cancel_token.is_cancelled() {
                break Ok(TaskTerminalStatus::Cancelled);
            }
            if task_turn_count >= max_task_turns {
                error!(
                    max_turns = max_task_turns,
                    "Max turns reached for this task"
                );
                break Ok(TaskTerminalStatus::MaxTurns);
            }

            let turn_ctx = turn::TurnContext {
                task_id: task.task_id.clone(),
                trace_id: task.trace_id.clone(),
                plan_id: task.plan_id.clone(),
                task_turn_index: task_turn_count,
                allowed_native_tools: Arc::clone(&tool_ctx.allowed_native_tools),
            };
            let completed_turn = match self.execute_turn(session, tool_ctx, &turn_ctx).await {
                Ok(outcome) => outcome,
                Err(err) => break Err(err),
            };

            let token_usage_action = self.evaluate_token_usage(session).await;
            session.turn_index += 1;
            task_turn_count += 1;
            self.refresh_task_session_mode(session);

            match token_usage_action {
                TokenUsageHookAction::Continue => {}
                TokenUsageHookAction::RejectTask { reason } => {
                    warn!(
                        task_id = %task.task_id,
                        trace_id = %task.trace_id,
                        reason = %reason,
                        "Task rejected by token usage policy"
                    );
                    break Ok(TaskTerminalStatus::Rejected);
                }
                TokenUsageHookAction::RejectSession { reason } => {
                    warn!(
                        task_id = %task.task_id,
                        trace_id = %task.trace_id,
                        reason = %reason,
                        "Session stop requested by token usage policy"
                    );
                    session.stop_requested = true;
                    session.queue.lock().await.clear();
                    break Ok(TaskTerminalStatus::Rejected);
                }
            }

            if session.mode == AgentMode::Stateless {
                match completed_turn {
                    turn::TurnOutcome::Continue | turn::TurnOutcome::Complete => {
                        break Ok(TaskTerminalStatus::Success);
                    }
                    turn::TurnOutcome::Rejected => {
                        break Ok(TaskTerminalStatus::Rejected);
                    }
                    turn::TurnOutcome::Cancelled => {
                        break Ok(TaskTerminalStatus::Cancelled);
                    }
                }
            }

            match completed_turn {
                turn::TurnOutcome::Continue => {}
                turn::TurnOutcome::Complete => {
                    break Ok(TaskTerminalStatus::Success);
                }
                turn::TurnOutcome::Rejected => {
                    break Ok(TaskTerminalStatus::Rejected);
                }
                turn::TurnOutcome::Cancelled => {
                    break Ok(TaskTerminalStatus::Cancelled);
                }
            }
        };

        let task_status = task_status_result?;
        Ok((task_status, task_turn_count))
    }

    fn refresh_task_session_mode(&self, session: &mut SessionState) {
        let runtime = self.runtime_for_session(session);
        let harness = runtime.lock_engine();
        if let Some(ref engine) = *harness
            && let Some(m) = engine.get_active_session_mode()
        {
            session.mode = m;
        }
    }
}
