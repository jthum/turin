use anyhow::{Context, Result, anyhow};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{error, instrument, warn};

use crate::inference::content::{encode_content_json, materialize_task_input_content};
use crate::inference::provider::{InferenceContent, InferenceMessage, InferenceRole};
use crate::kernel::TaskExecutionResult;
use crate::kernel::event::{TaskBranchOutcome, TaskTerminalStatus};
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::harness_hooks::TokenUsageHookAction;
use crate::kernel::session::{ExecutionConflictPolicy, QueuedTask, SessionState};
use crate::kernel::turn;
use crate::persistence::schema::BranchProvenance;
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

        self.begin_turn_persistence(session, Some(task)).await?;
        self.persist_turn_message(session, "user", &encode_content_json(&user_content))
            .await?;
        self.append_task_user_message(session, &user_content);

        let effective_tools = crate::tools::policy::resolve_effective_tools_config_for_registry(
            &self.config,
            session.identity.agent_id(),
            task.tools.as_ref(),
            &self.tool_registry.names(),
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

        self.set_task_active_session(session, task);

        let task_status_result = self.run_task_turn_loop(session, task, &tool_ctx).await;

        self.clear_task_active_session(session);

        let (status, task_turn_count) = task_status_result?;
        Ok(TaskExecutionResult {
            status,
            task_turn_count,
            branch_outcome: session.current_task_branch_outcome().cloned(),
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
        let origin = session.active_history_origin();
        session.history.push_with_origin(
            InferenceMessage {
                role: InferenceRole::User,
                content: content.to_vec(),
                tool_call_id: None,
            },
            origin,
        );
    }

    pub(super) async fn persist_turn_message(
        &self,
        session: &SessionState,
        role: &str,
        content: &serde_json::Value,
    ) -> Result<()> {
        let (Some(internal_id), Some(target)) =
            (session.internal_id, session.active_turn_write_target())
        else {
            return Ok(());
        };
        let store = self
            .store_manager
            .open(&session.store_selector)
            .await
            .context("Failed to open state store for turn message persistence")?;
        let _guard = session.persistence_lock.lock().await;
        let token_count = crate::kernel::estimate_persisted_message_input_tokens(role, content)
            .map(|tokens| tokens as u64);
        store
            .insert_message(internal_id, target, role, content, token_count)
            .await
            .with_context(|| format!("Failed to persist {role} turn message"))
    }

    fn set_task_active_session(&self, session: &SessionState, task: &QueuedTask) {
        self.bind_harness_execution_context(session, task);
    }

    fn clear_task_active_session(&self, session: &SessionState) {
        self.unbind_harness_execution_context(session);
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

            self.begin_turn_persistence(session, Some(task)).await?;

            let turn_ctx = turn::TurnContext {
                task_id: task.task_id.clone(),
                trace_id: task.trace_id.clone(),
                plan_id: task.plan_id.clone(),
                task_turn_index: task_turn_count,
                inference_context: task.inference_context.clone(),
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

            let token_usage_action = self
                .evaluate_token_usage(session, task_turn_count + 1)
                .await;
            session.turn_index += 1;
            task_turn_count += 1;
            self.prune_session_hot_history(session);

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

    async fn begin_turn_persistence(
        &self,
        session: &mut SessionState,
        task: Option<&QueuedTask>,
    ) -> Result<()> {
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
        let store = self
            .store_manager
            .open(&session.store_selector)
            .await
            .with_context(|| {
                format!("Failed to open persisted session store (internal_id={internal_id})")
            })?;

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
                    .prepare_fork_sibling_turn_target(session, task, request)
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
            session.set_selected_branch_head_cursor(Some(turn_id), Some(turn_index));
        }
        session.set_active_turn_write_target(Some(resolved));
        Ok(())
    }

    async fn prepare_fork_sibling_turn_target(
        &self,
        session: &mut SessionState,
        task: Option<&QueuedTask>,
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
        let provenance = BranchProvenance::conflict_fork(
            task.and_then(|task| (!task.task_id.is_empty()).then(|| task.task_id.clone())),
            Some(session.execution_id().to_string()),
        );
        let store = self
            .store_manager
            .open(&session.store_selector)
            .await
            .context("ForkSibling conflict policy requires a configured persistent state store")?;

        let (branch, resolved) = {
            let _guard = session.persistence_lock.lock().await;
            let branch = store
                .create_branch_head_from_turn_index_with_provenance(
                    internal_id,
                    &fork_branch_name,
                    source_turn_index,
                    false,
                    provenance,
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
        session.set_selected_branch_head_cursor(branch.head_turn_id, source_turn_index);
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
