use anyhow::{Context, Result, anyhow};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{error, instrument, warn};

use crate::inference::content::{encode_content_json, materialize_task_input_content};
use crate::inference::provider::{InferenceContent, InferenceMessage, InferenceRole};
use crate::kernel::TaskExecutionResult;
use crate::kernel::config::AgentMode;
use crate::kernel::event::{TaskBranchOutcome, TaskTerminalStatus};
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::harness_hooks::TokenUsageHookAction;
use crate::kernel::session::{ExecutionConflictPolicy, QueuedTask, SessionState};
use crate::kernel::turn;
use crate::tools::ToolContext;
use turin_types::layout::{DEFAULT_LAYOUT_ROOT, resolve_relative_to};

impl ExecutionHost {
    /// Execute a single task (one specific prompt) within the persistent session.
    #[instrument(skip(self, session, task), fields(task_id = %task.task_id, trace_id = %task.trace_id))]
    pub(super) async fn run_task(
        &mut self,
        session: &mut SessionState,
        task: &QueuedTask,
    ) -> Result<TaskExecutionResult> {
        self.ensure_session_harness_engine(session)?;
        let session_id = self.session_reference(session);
        let prompt = task.prompt.as_str();
        let user_content = if let Some(content) = task.content.as_ref() {
            let media_dir = self.managed_media_dir();
            materialize_task_input_content(content, &media_dir).await?
        } else {
            vec![InferenceContent::Text {
                text: prompt.to_string(),
            }]
        };

        if session.cancel_token.is_cancelled() {
            return Ok(TaskExecutionResult {
                status: TaskTerminalStatus::Cancelled,
                task_turn_count: 0,
                branch_outcome: None,
            });
        }

        self.begin_turn_persistence(session).await?;
        self.append_task_user_message(session, &user_content);

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

        self.persist_task_user_message(session, &user_content).await;
        self.set_task_active_session(session, task);

        let task_status_result = self.run_task_turn_loop(session, task, &tool_ctx).await;

        self.clear_task_active_session(session);

        let (status, task_turn_count) = task_status_result?;
        Ok(TaskExecutionResult {
            status,
            task_turn_count,
            branch_outcome: session.current_task_branch_outcome.clone(),
        })
    }

    fn managed_media_dir(&self) -> PathBuf {
        let data_dir = PathBuf::from(&self.config.layout.data_dir);
        if data_dir.is_absolute() {
            return data_dir.join("media");
        }

        let workspace_root = PathBuf::from(&self.config.kernel.workspace_root);
        let layout_root = self
            .config
            .layout
            .root
            .as_deref()
            .map(std::path::Path::new)
            .map(|root| resolve_relative_to(&workspace_root, root))
            .unwrap_or_else(|| workspace_root.join(DEFAULT_LAYOUT_ROOT));
        layout_root.join(data_dir).join("media")
    }

    fn append_task_user_message(&self, session: &mut SessionState, content: &[InferenceContent]) {
        session.history.push(InferenceMessage {
            role: InferenceRole::User,
            content: content.to_vec(),
            tool_call_id: None,
        });
    }

    async fn persist_task_user_message(
        &self,
        session: &SessionState,
        content: &[InferenceContent],
    ) {
        if let Ok(store) = self.store_manager.open(&session.store_selector).await {
            if let (Some(iid), Some(target)) =
                (session.internal_id, session.active_turn_write_target())
            {
                let _guard = session.persistence_lock.lock().await;
                let _ = store
                    .insert_message(iid, target, "user", &encode_content_json(content), None)
                    .await;
            } else if session.internal_id.is_none() {
                warn!("Session missing internal_id, skipping message persistence");
            }
        }
    }

    fn set_task_active_session(&self, session: &SessionState, task: &QueuedTask) {
        if let Some(harness) = self.session_harness_engine(session) {
            let engine = harness.lock().expect("session harness mutex poisoned");
            let session_id = self.session_reference(session);
            engine.set_active_session(
                Some(&session_id),
                Some(session.store_selector.clone()),
                session.default_store_selector.clone(),
                Some(session.mode.clone()),
            );
            engine.set_active_execution_metadata(
                Some(session.execution_id()),
                Some(session.context_target().clone()),
                Some(session.execution.visibility),
                Some(session.execution.durability),
                Some(session.effective_write_policy()),
                Some(session.effective_conflict_policy()),
            );
            engine.set_active_runtime_slot_id(session.runtime_slot_id.as_deref());
            engine.set_active_trace_id(Some(&task.trace_id));
            engine.set_active_event_context(Some(crate::harness::globals::HarnessEventContext {
                json: self.json,
                internal_id: session.internal_id,
                branch_head_id: session.selected_branch_head_id(),
                execution_id: session.execution_id().to_string(),
                event_tx: session.event_tx.clone(),
                durability_tx: session.durability_tx.clone(),
            }));
        }
    }

    fn clear_task_active_session(&self, session: &SessionState) {
        if let Some(harness) = self.session_harness_engine(session) {
            let engine = harness.lock().expect("session harness mutex poisoned");
            engine.set_active_session(None, None, None, None);
            engine.set_active_execution_metadata(None, None, None, None, None, None);
            engine.set_active_runtime_slot_id(None);
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

            self.begin_turn_persistence(session).await?;

            let turn_ctx = turn::TurnContext {
                task_id: task.task_id.clone(),
                trace_id: task.trace_id.clone(),
                plan_id: task.plan_id.clone(),
                task_turn_index: task_turn_count,
                allowed_native_tools: Arc::clone(&tool_ctx.allowed_native_tools),
            };
            let completed_turn = match self.execute_turn(session, tool_ctx, &turn_ctx).await {
                Ok(outcome) => {
                    self.clear_turn_persistence(session);
                    outcome
                }
                Err(err) => {
                    self.clear_turn_persistence(session);
                    break Err(err);
                }
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
        if let Some(harness) = self.session_harness_engine(session)
            && let Ok(engine) = harness.lock()
            && let Some(m) = engine.get_active_session_mode()
        {
            session.mode = m;
        }
    }

    async fn begin_turn_persistence(&self, session: &mut SessionState) -> Result<()> {
        if session.active_turn_write_target().is_some() {
            return Ok(());
        }
        let Some(request) = session.next_turn_write_target_request() else {
            return Ok(());
        };
        let Some(internal_id) = session.internal_id else {
            warn!("Session missing internal_id, skipping turn target preparation");
            return Ok(());
        };
        let Ok(store) = self.store_manager.open(&session.store_selector).await else {
            return Ok(());
        };

        let prepare_result = {
            let _guard = session.persistence_lock.lock().await;
            store.prepare_turn_write_target(internal_id, request).await
        };
        let resolved = match prepare_result {
            Ok(Some(resolved)) => resolved,
            Ok(None) => {
                return Err(anyhow!(
                    "No active branch head available for session {}",
                    internal_id
                ));
            }
            Err(error)
                if crate::persistence::state::is_turn_write_conflict(&error)
                    && session.effective_conflict_policy() == ExecutionConflictPolicy::Detached =>
            {
                warn!(
                    execution_id = %session.execution_id(),
                    "Turn write target became stale; continuing with detached task execution"
                );
                session.begin_conflict_detached_task();
                return Ok(());
            }
            Err(error)
                if crate::persistence::state::is_turn_write_conflict(&error)
                    && session.effective_conflict_policy()
                        == ExecutionConflictPolicy::ForkSibling =>
            {
                let resolved = self
                    .prepare_fork_sibling_turn_target(session, request)
                    .await?;
                warn!(
                    execution_id = %session.execution_id(),
                    branch_head_id = session.selected_branch_head_id(),
                    "Turn write target became stale; continuing on a forked sibling branch"
                );
                resolved
            }
            Err(error) => return Err(error),
        };
        if let crate::persistence::state::TurnWriteTarget::ExistingTurn {
            turn_id,
            turn_index,
        } = resolved
        {
            session.set_selected_branch_head_turn_id(Some(turn_id));
            session.set_selected_branch_head_turn_index(Some(turn_index));
        }
        session.set_active_turn_write_target(Some(resolved));
        Ok(())
    }

    async fn prepare_fork_sibling_turn_target(
        &self,
        session: &mut SessionState,
        request: crate::persistence::state::TurnWriteTarget,
    ) -> Result<crate::persistence::state::TurnWriteTarget> {
        let crate::persistence::state::TurnWriteTarget::BranchAdvance {
            expected_head_turn_id,
            turn_index,
            ..
        } = request
        else {
            anyhow::bail!("ForkSibling conflict policy requires a branch-advance target");
        };

        let internal_id = session
            .internal_id
            .ok_or_else(|| anyhow!("Session missing internal persistence id"))?;
        let source_turn_index = turn_index.checked_sub(1);
        let fork_branch_name = format!("fork-{}-{}", session.execution_id(), turn_index);
        let store = self
            .store_manager
            .open(&session.store_selector)
            .await
            .context("ForkSibling conflict policy requires a configured persistent state store")?;

        let (branch, resolved) = {
            let _guard = session.persistence_lock.lock().await;
            let branch = store
                .create_branch_head_from_turn_index(
                    internal_id,
                    &fork_branch_name,
                    source_turn_index,
                    false,
                )
                .await?;
            let fork_request = match expected_head_turn_id {
                Some(turn_id) => {
                    crate::persistence::state::TurnWriteTarget::branch_head_with_expectation(
                        Some(branch.id),
                        Some(turn_id),
                        turn_index,
                    )
                }
                None => crate::persistence::state::TurnWriteTarget::branch_head(
                    Some(branch.id),
                    turn_index,
                ),
            };
            let resolved = store
                .prepare_turn_write_target(internal_id, fork_request)
                .await?
                .ok_or_else(|| anyhow!("Forked sibling branch did not yield a writable turn"))?;
            (branch, resolved)
        };

        let branch_public_id = uuid::Uuid::from_slice(&branch.public_id)
            .map(|value| value.to_string())
            .context("Forked sibling branch public id was invalid")?;
        session.set_selected_branch_head_id(Some(branch.id));
        session.set_selected_branch_head_turn_id(branch.head_turn_id);
        session.set_selected_branch_head_turn_index(source_turn_index);
        session.set_current_task_branch_outcome(Some(TaskBranchOutcome::ForkSibling {
            branch_id: branch.id,
            branch_public_id,
            branch_name: branch.name.clone(),
            source_turn_id: branch.created_from_turn_id,
            persisted_active_head_unchanged: !branch.is_active,
        }));

        Ok(resolved)
    }

    fn clear_turn_persistence(&self, session: &mut SessionState) {
        session.set_active_turn_write_target(None);
    }
}
