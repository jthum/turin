use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use tokio::sync::Mutex as AsyncMutex;
use tracing::{debug, info, warn};

mod materialization;
mod sidestep;

use crate::kernel::config::ContextPersistenceConfig;
use crate::kernel::event::{AuditEvent, KernelEvent, LifecycleEvent};
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::identity::RuntimeIdentity;
use crate::kernel::session::{
    ExecutionContextTarget, ExecutionWritePolicy, PersistedKernelRecord, SessionState,
    SessionStatus,
};
use crate::kernel::session_lifecycle::materialization::{
    MaterializedExecutionTarget, TokenContextBounds, materialize_execution_target,
    materialize_token_bounded_messages, rebuild_history,
};
pub(crate) use crate::kernel::session_lifecycle::sidestep::prepare_persisted_session_sidestep;
use crate::kernel::session_metadata::{
    create_session_metadata, session_channel_id_from_metadata,
    session_default_store_selector_from_metadata,
};
use crate::kernel::session_refs::{
    describe_store_selector, format_session_reference, parse_session_reference,
};
use crate::perf_diagnostics::{perf_session_scope, perf_stage, perf_stage_finish};
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::SessionRow;
use crate::persistence::state::StateStore;

#[cfg_attr(not(feature = "perf-diagnostics"), allow(unused_variables))]
async fn materialize_session_target(
    host: &ExecutionHost,
    store: &StateStore,
    store_selector: &StoreSelector,
    row: &SessionRow,
    context_target: &ExecutionContextTarget,
    session_id: &str,
    operation: &'static str,
) -> Result<MaterializedExecutionTarget> {
    perf_session_scope!(session_id, async {
        perf_stage!(
            materialize_stage,
            operation,
            Some(session_id),
            serde_json::json!({
                "internal_session_id": row.id,
                "context_target": format!("{context_target:?}"),
            })
        );
        let materialized =
            materialize_execution_target(host, store, store_selector, row, context_target).await;
        match materialized {
            Ok(materialized) => {
                perf_stage_finish!(
                    materialize_stage,
                    "ok",
                    serde_json::json!({
                        "message_rows": materialized.messages.len(),
                        "has_context_checkpoint": materialized.context_checkpoint.is_some(),
                    })
                );
                Ok(materialized)
            }
            Err(error) => {
                perf_stage_finish!(materialize_stage, "error", serde_json::json!({}));
                Err(error)
            }
        }
    })
    .await
}

impl ExecutionHost {
    pub(crate) async fn load_token_bounded_history(
        &self,
        session: &SessionState,
        token_budget: u64,
        minimum_messages: usize,
        max_turns: usize,
    ) -> Result<crate::kernel::session::ResidentHistory> {
        let (store, row) = self
            .load_current_session_row(
                session,
                "Token-bounded context retrieval requires a configured persistent state store",
            )
            .await?;
        let context_target = resolved_execution_context_target(session.context_target(), &row);
        let selected = materialize_token_bounded_messages(
            self,
            &store,
            &session.store_selector,
            &row,
            &context_target,
            TokenContextBounds {
                token_budget,
                minimum_messages,
                max_turns,
            },
        )
        .await?;
        let (history, _) = rebuild_history(&selected.messages)?;
        let mut resident = crate::kernel::session::ResidentHistory::default();
        resident.replace(history, selected.has_prior_history);
        Ok(resident)
    }

    /// Create a new session.
    pub async fn create_session(&self) -> SessionState {
        self.create_session_for_agent(&self.config.agent.id).await
    }

    /// Create a new session bound to a specific configured agent profile.
    pub async fn create_session_for_agent(&self, agent_id: &str) -> SessionState {
        self.create_session_for_agent_in_store(agent_id, None, None)
            .await
    }

    pub async fn create_session_for_agent_in_store(
        &self,
        agent_id: &str,
        state_selector: Option<StoreSelector>,
        default_store_selector: Option<StoreSelector>,
    ) -> SessionState {
        self.create_session_for_agent_with_context(
            agent_id,
            state_selector,
            default_store_selector,
            None,
            crate::kernel::config::InferenceOverrideConfig::default(),
        )
        .await
    }

    pub async fn create_session_for_agent_with_context(
        &self,
        agent_id: &str,
        state_selector: Option<StoreSelector>,
        default_store_selector: Option<StoreSelector>,
        channel_id: Option<String>,
        inference: crate::kernel::config::InferenceOverrideConfig,
    ) -> SessionState {
        let mut session = SessionState::new();
        session.identity.set_agent_id(agent_id.to_string());
        session.identity.set_channel_id(channel_id);
        session.store_selector =
            state_selector.unwrap_or_else(|| self.resolve_agent_state_selector(agent_id));
        session.default_store_selector =
            default_store_selector.or_else(|| self.resolve_agent_default_store_selector(agent_id));
        session.inference = inference;
        self.attach_session_persistence(&mut session, true).await;
        session
    }

    pub(crate) fn session_reference(&self, session: &SessionState) -> String {
        format_session_reference(session.identity.session_id(), &session.store_selector)
    }

    /// Resume an existing persisted session into a live runtime.
    pub async fn resume_session_for_agent(
        &self,
        agent_id: &str,
        session_id: &str,
    ) -> Result<SessionState> {
        self.resume_session_for_agent_with_context(
            agent_id,
            session_id,
            None,
            crate::kernel::config::InferenceOverrideConfig::default(),
        )
        .await
    }

    pub async fn resume_session_for_agent_with_context(
        &self,
        agent_id: &str,
        session_id: &str,
        channel_id: Option<String>,
        inference: crate::kernel::config::InferenceOverrideConfig,
    ) -> Result<SessionState> {
        let session_ref = parse_session_reference(session_id)?;
        let public_id = uuid::Uuid::parse_str(&session_ref.public_id)
            .with_context(|| format!("Invalid session id '{}'", session_ref.public_id))?;
        let store_selector = session_ref
            .store_selector
            .unwrap_or_else(|| self.resolve_agent_state_selector(agent_id));
        let store = self
            .store_manager
            .open(&store_selector)
            .await
            .context("Session resume requires a configured persistent state store")?;
        let row = store
            .get_session_row_by_public_id(public_id)
            .await?
            .ok_or_else(|| anyhow!("Session '{}' not found", session_ref.public_id))?;
        if row.agent_id != agent_id {
            anyhow::bail!(
                "Session '{}' belongs to agent '{}' not '{}'",
                session_ref.public_id,
                row.agent_id,
                agent_id
            );
        }

        let context_target = ExecutionContextTarget::BranchHead {
            branch_head_id: row.active_branch_head_id,
        };
        let materialized = materialize_session_target(
            self,
            &store,
            &store_selector,
            &row,
            &context_target,
            session_id,
            "session.resume.materialize",
        )
        .await?;
        let (history, turn_index) = rebuild_history(&materialized.messages)?;
        let counters = store.get_session_counters(row.id).await?;

        let mut session = SessionState::new();
        session.identity = RuntimeIdentity::new(session_ref.public_id, agent_id);
        session.identity.set_channel_id(
            channel_id.or_else(|| session_channel_id_from_metadata(row.metadata.as_deref())),
        );
        session.internal_id = Some(row.id);
        session.store_selector = store_selector;
        session.default_store_selector =
            session_default_store_selector_from_metadata(row.metadata.as_deref());
        session.inference = inference;
        session.context_checkpoint = materialized.context_checkpoint;
        session
            .history
            .replace(history, materialized.has_prior_history);
        session.replace_context_target_preserving_policy(context_target);
        session.set_selected_branch_head_cursor(
            materialized.branch_head_turn_id,
            materialized.branch_head_turn_index,
        );
        session.turn_index = materialized
            .branch_head_turn_index
            .map_or(turn_index, |head| turn_index.max(head.saturating_add(1)));
        session.total_input_tokens = counters.total_input_tokens;
        session.total_output_tokens = counters.total_output_tokens;
        session.next_task_id = counters.next_task_id;
        session.next_plan_id = counters.next_plan_id;
        session.restored_from_persistence = true;
        self.prune_session_hot_history(&mut session);
        self.attach_session_persistence(&mut session, false).await;
        Ok(session)
    }

    pub async fn refresh_session_from_persistence(&self, session: &mut SessionState) -> Result<()> {
        let (store, row) = self
            .load_current_session_row(
                session,
                "Session refresh requires a configured persistent state store",
            )
            .await?;

        let context_target = resolved_execution_context_target(session.context_target(), &row);
        let session_reference = self.session_reference(session);
        let materialized = materialize_session_target(
            self,
            &store,
            &session.store_selector,
            &row,
            &context_target,
            &session_reference,
            "session.refresh.materialize",
        )
        .await?;
        let (history, turn_index) = rebuild_history(&materialized.messages)?;
        let counters = store.get_session_counters(row.id).await?;

        session.default_store_selector =
            session_default_store_selector_from_metadata(row.metadata.as_deref());
        session
            .identity
            .set_channel_id(session_channel_id_from_metadata(row.metadata.as_deref()));
        session.context_checkpoint = materialized.context_checkpoint;
        session
            .history
            .replace(history, materialized.has_prior_history);
        session.set_context_target(context_target);
        session.set_selected_branch_head_cursor(
            materialized.branch_head_turn_id,
            materialized.branch_head_turn_index,
        );
        session.turn_index = materialized
            .branch_head_turn_index
            .map_or(turn_index, |head| turn_index.max(head.saturating_add(1)));
        session.total_input_tokens = counters.total_input_tokens;
        session.total_output_tokens = counters.total_output_tokens;
        session.next_task_id = counters.next_task_id;
        session.next_plan_id = counters.next_plan_id;
        session.restored_from_persistence = true;
        self.prune_session_hot_history(session);
        Ok(())
    }

    pub(crate) fn prune_session_hot_history(&self, session: &mut SessionState) {
        if session.internal_id.is_none()
            || session.effective_write_policy() != ExecutionWritePolicy::AdvanceBranchHead
        {
            return;
        }
        let hot_history = &self.config.inference.hot_history;
        let report = crate::kernel::hot_history::apply(session, hot_history);
        if report.applied() {
            let dropped_messages = report
                .prune
                .map(|report| report.dropped_messages)
                .unwrap_or(0);
            let retained_messages = report
                .prune
                .map(|report| report.retained_messages)
                .unwrap_or(session.history.len());
            let has_prior_history = report
                .prune
                .map(|report| report.has_prior_history)
                .unwrap_or_else(|| session.history.has_prior_history());
            let trimmed_tool_results = report
                .payload_trim
                .map(|report| report.trimmed_tool_results)
                .unwrap_or(0);
            let dropped_payload_bytes = report
                .payload_trim
                .map(|report| report.dropped_bytes)
                .unwrap_or(0);
            debug!(
                dropped_messages,
                retained_messages,
                has_prior_history,
                trimmed_tool_results,
                dropped_payload_bytes,
                "Applied hot-history memory policy"
            );
        }
    }

    pub async fn select_session_branch_by_name_local(
        &self,
        session: &mut SessionState,
        branch_name: &str,
    ) -> Result<bool> {
        let internal_id = require_persisted_session_id(session)?;
        let store = self
            .store_manager
            .open(&session.store_selector)
            .await
            .context("Local branch selection requires a configured persistent state store")?;
        let Some(branch) = store
            .get_branch_head_by_name(internal_id, branch_name)
            .await?
        else {
            return Ok(false);
        };

        ensure_local_session_target_idle(session, "branch").await?;
        session.set_selected_branch_head_id(Some(branch.id));
        self.refresh_session_from_persistence(session).await?;
        Ok(true)
    }

    pub async fn select_session_turn_local(
        &self,
        session: &mut SessionState,
        turn_id: i64,
    ) -> Result<bool> {
        let internal_id = require_persisted_session_id(session)?;
        let store = self
            .store_manager
            .open(&session.store_selector)
            .await
            .context("Local turn selection requires a configured persistent state store")?;
        let Some(turn) = store.get_turn_row(turn_id).await? else {
            return Ok(false);
        };
        if turn.session_id != internal_id {
            return Ok(false);
        }

        ensure_local_session_target_idle(session, "target").await?;
        session.set_selected_turn_id(turn_id);
        self.refresh_session_from_persistence(session).await?;
        Ok(true)
    }

    pub async fn select_session_external_reference_local(
        &self,
        session: &mut SessionState,
        reference: &str,
    ) -> Result<bool> {
        let session_ref = parse_session_reference(reference)?;
        let store_selector = session_ref
            .store_selector
            .clone()
            .unwrap_or_else(|| session.store_selector.clone());
        let public_id = uuid::Uuid::parse_str(&session_ref.public_id)
            .with_context(|| format!("Invalid session id '{}'", session_ref.public_id))?;
        let store =
            self.store_manager.open(&store_selector).await.context(
                "External reference selection requires a configured persistent state store",
            )?;
        let Some(_) = store.get_session_row_by_public_id(public_id).await? else {
            return Ok(false);
        };

        ensure_local_session_target_idle(session, "target").await?;
        session.set_context_target(ExecutionContextTarget::ExternalReference {
            reference: format_session_reference(&session_ref.public_id, &store_selector),
        });
        self.refresh_session_from_persistence(session).await?;
        Ok(true)
    }

    async fn attach_session_persistence(&self, session: &mut SessionState, create_row: bool) {
        if let Ok(store) = self.store_manager.open(&session.store_selector).await {
            if create_row
                && let Ok(public_id) = uuid::Uuid::parse_str(session.identity.session_id())
            {
                let metadata = create_session_metadata(
                    session.default_store_selector.as_ref(),
                    session.identity.channel_id(),
                );
                match store
                    .create_session(public_id, session.identity.agent_id(), metadata.as_deref())
                    .await
                {
                    Ok(id) => {
                        session.internal_id = Some(id);
                        match store.get_active_branch_head(id).await {
                            Ok(Some(branch)) => {
                                session.set_selected_branch_head_id(Some(branch.id));
                                let branch_head_turn_id = branch.head_turn_id;
                                let head_turn_index = match branch.head_turn_id {
                                    Some(turn_id) => match store.get_turn_row(turn_id).await {
                                        Ok(Some(turn)) => Some(turn.branch_depth),
                                        Ok(None) => None,
                                        Err(e) => {
                                            warn!(
                                                error = %e,
                                                turn_id,
                                                "Failed to load initial branch head turn depth"
                                            );
                                            None
                                        }
                                    },
                                    None => None,
                                };
                                session.set_selected_branch_head_cursor(
                                    branch_head_turn_id,
                                    head_turn_index,
                                );
                            }
                            Ok(None) => {
                                warn!("Created session is missing an active branch head");
                            }
                            Err(e) => {
                                warn!(error = %e, "Failed to load initial branch head for session");
                            }
                        }
                        if let Err(e) = self.bind_session_persistence_lock(session).await {
                            warn!(error = %e, "Failed to bind shared persistence lock for session");
                        }
                    }
                    Err(e) => warn!(error = %e, "Failed to create session row in DB"),
                }
            }

            if create_row
                && session.internal_id.is_none()
                && let Err(e) = self.bind_session_persistence_lock(session).await
            {
                warn!(error = %e, "Failed to bind shared persistence lock for session");
            }

            if !create_row && let Err(e) = self.bind_session_persistence_lock(session).await {
                warn!(error = %e, "Failed to bind shared persistence lock for resumed session");
            }

            let (durability_tx, mut durability_rx) =
                tokio::sync::mpsc::unbounded_channel::<PersistedKernelRecord>();
            session.durability_tx = Some(durability_tx);
            let store_clone = store.clone();
            let persistence_lock = Arc::clone(&session.persistence_lock);
            let handle = tokio::spawn(async move {
                let mut event_writer = None;
                let mut pending_error = None;
                while let Some(record) = durability_rx.recv().await {
                    match record {
                        PersistedKernelRecord::Event(record) => {
                            let event = record.event;
                            let event_type = event.event_type().to_string();
                            let payload = serde_json::to_value(&event).unwrap_or_default();
                            if let Some(iid) = record.internal_id {
                                let _guard = persistence_lock.lock().await;
                                if event_writer.is_none() {
                                    match store_clone.event_writer().await {
                                        Ok(writer) => event_writer = Some(writer),
                                        Err(e) => {
                                            warn!(error = %e, "Background persistence error");
                                            pending_error.get_or_insert_with(|| e.to_string());
                                            continue;
                                        }
                                    }
                                }
                                let writer = event_writer
                                    .as_ref()
                                    .expect("event writer initialized before insert");
                                if let Err(e) = writer
                                    .insert_event(iid, record.turn_target, &event_type, &payload)
                                    .await
                                {
                                    warn!(error = %e, "Background persistence error");
                                    pending_error.get_or_insert_with(|| e.to_string());
                                    event_writer = None;
                                }
                            } else {
                                warn!("Dropping event: no internal_id for session");
                            }
                        }
                        PersistedKernelRecord::Barrier(tx) => {
                            let _guard = persistence_lock.lock().await;
                            let result = pending_error.take().map_or(Ok(()), Err);
                            let _ = tx.send(result);
                        }
                    }
                }
            });
            session.event_task = Some(Arc::new(AsyncMutex::new(Some(handle))));
        }
    }

    pub(crate) fn resolve_agent_state_selector(&self, agent_id: &str) -> StoreSelector {
        self.config
            .persistence
            .resolve_context_state_selector(self.agent_persistence_context(agent_id))
            .unwrap_or_else(|err| {
                warn!(
                    agent_id = %agent_id,
                    error = %err,
                    "Falling back to default state selector for agent"
                );
                StoreSelector::Alias("state".to_string())
            })
    }

    pub(crate) fn resolve_agent_default_store_selector(
        &self,
        agent_id: &str,
    ) -> Option<StoreSelector> {
        let context = self.agent_persistence_context(agent_id)?;
        if context.store.is_none() && context.state.is_none() {
            return None;
        }
        self.config
            .persistence
            .resolve_context_store_selector(Some(context))
            .map(Some)
            .unwrap_or_else(|err| {
                warn!(
                    agent_id = %agent_id,
                    error = %err,
                    "Falling back to default store selector for agent"
                );
                None
            })
    }

    fn agent_persistence_context(&self, agent_id: &str) -> Option<&ContextPersistenceConfig> {
        if agent_id == self.config.agent.id {
            Some(&self.config.agent.persistence)
        } else {
            self.config
                .agents
                .get(agent_id)
                .map(|agent| &agent.persistence)
        }
    }

    /// Start a new session.
    pub async fn start_session(&self, session: &mut SessionState) -> Result<()> {
        if session.status == SessionStatus::Active {
            return Ok(());
        }

        let session_id = self.session_reference(session);
        info!(
            session_id = %session_id,
            store = %describe_store_selector(&session.store_selector),
            resumed = session.restored_from_persistence,
            "Starting new session"
        );

        self.ensure_session_harness_engine(session)?;

        self.persist_event(
            session,
            &KernelEvent::Lifecycle(if session.restored_from_persistence {
                LifecycleEvent::SessionResume {
                    identity: session.identity.clone(),
                }
            } else {
                LifecycleEvent::SessionStart {
                    identity: session.identity.clone(),
                }
            }),
        );
        let governance_snapshot = self
            .governance_manager
            .snapshot_for_agent(Some(session.identity.agent_id()));
        self.persist_event(
            session,
            &KernelEvent::Audit(AuditEvent::GovernanceSnapshot {
                snapshot: governance_snapshot.clone(),
            }),
        );

        if let Some(harness) = self.session_harness_engine(session) {
            let engine = harness.lock().expect("session harness mutex poisoned");
            if let Err(e) = engine.evaluate(
                "on_session_start",
                serde_json::json!({
                    "identity": session.identity.clone(),
                    "session_id": session_id,
                    "governance": governance_snapshot,
                }),
            ) {
                warn!(error = %e, "Harness on_session_start failed");
            }
        }

        session.stop_requested = false;
        session.status = SessionStatus::Active;
        Ok(())
    }

    /// End the session and emit SessionEnd event.
    pub async fn end_session(&self, session: &mut SessionState) -> Result<()> {
        if session.status == SessionStatus::Inactive {
            return Ok(());
        }

        info!(
            session_id = %self.session_reference(session),
            store = %describe_store_selector(&session.store_selector),
            turn_count = session.turn_index,
            "Ending session"
        );

        self.persist_event(
            session,
            &KernelEvent::Lifecycle(LifecycleEvent::SessionEnd {
                identity: session.identity.clone(),
                turn_count: session.turn_index,
                total_input_tokens: session.total_input_tokens,
                total_output_tokens: session.total_output_tokens,
            }),
        );

        if let Some(harness) = self.session_harness_engine(session) {
            let engine = harness.lock().expect("session harness mutex poisoned");
            if let Err(e) = engine.evaluate(
                "on_session_end",
                serde_json::json!({
                    "identity": session.identity.clone(),
                    "session_id": self.session_reference(session),
                    "turn_count": session.turn_index,
                    "total_input_tokens": session.total_input_tokens,
                    "total_output_tokens": session.total_output_tokens,
                }),
            ) {
                warn!(error = %e, "Harness on_session_end failed");
            }
            engine.set_active_queue(None);
        }

        let mut durability_error = self.wait_for_session_durability(session).await.err();

        // Close durability lane and await background persistence flush.
        session.durability_tx.take();
        if let Some(task_slot) = &session.event_task
            && let Some(handle) = task_slot.lock().await.take()
            && let Err(e) = handle.await
        {
            warn!(error = %e, "Background persistence task join error");
            durability_error
                .get_or_insert_with(|| anyhow!("Background persistence task failed: {e}"));
        }
        session.cancel_token.cancel();

        session.status = SessionStatus::Inactive;
        self.clear_session_harness_engine(session);
        durability_error.map_or(Ok(()), Err)
    }
}

fn require_persisted_session_id(session: &SessionState) -> Result<i64> {
    session
        .internal_id
        .ok_or_else(|| anyhow!("Session has no internal persistence id"))
}

impl ExecutionHost {
    async fn load_current_session_row(
        &self,
        session: &SessionState,
        context: &'static str,
    ) -> Result<(Arc<StateStore>, SessionRow)> {
        let internal_id = require_persisted_session_id(session)?;
        let store = self
            .store_manager
            .open(&session.store_selector)
            .await
            .context(context)?;
        let row = store.get_session_row(internal_id).await?.ok_or_else(|| {
            anyhow!(
                "Persisted session '{}' not found",
                session.identity.session_id()
            )
        })?;
        Ok((store, row))
    }
}

async fn ensure_local_session_target_idle(session: &SessionState, target_kind: &str) -> Result<()> {
    if !session.queue.lock().await.is_empty() {
        anyhow::bail!("Cannot switch local session {target_kind} while tasks are queued");
    }
    Ok(())
}

fn resolved_execution_context_target(
    current: &ExecutionContextTarget,
    row: &SessionRow,
) -> ExecutionContextTarget {
    match current {
        ExecutionContextTarget::BranchHead {
            branch_head_id: None,
        } => ExecutionContextTarget::BranchHead {
            branch_head_id: row.active_branch_head_id,
        },
        _ => current.clone(),
    }
}
