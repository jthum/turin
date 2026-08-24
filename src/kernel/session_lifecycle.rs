use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use tracing::{debug, info, warn};

mod materialization;
mod persistence;
mod sidestep;

use crate::kernel::config::ContextPersistenceConfig;
use crate::kernel::event::{AuditEvent, KernelEvent, LifecycleEvent};
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::harness_contract::HarnessHook;
use crate::kernel::identity::RuntimeIdentity;
use crate::kernel::session::{
    ExecutionContextTarget, ExecutionWritePolicy, SessionState, SessionStatus,
};

const SESSION_PERSISTENCE_SHUTDOWN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
use crate::kernel::session_lifecycle::materialization::{
    MaterializedExecutionTarget, TokenContextBounds, materialize_execution_target,
    materialize_token_bounded_messages, rebuild_history,
};
pub use crate::kernel::session_lifecycle::sidestep::prepare_persisted_session_sidestep;
use crate::kernel::session_metadata::session_default_store_selector_from_metadata;
use crate::kernel::session_refs::{
    describe_store_selector, format_session_reference, parse_session_reference,
};
use crate::perf_diagnostics::{perf_session_scope, perf_stage, perf_stage_finish};
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::{LinkedSessionCreate, SessionRow};
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
        origin_id: Option<String>,
        inference: crate::kernel::config::InferenceOverrideConfig,
    ) -> SessionState {
        self.create_session_for_agent_with_context_and_link(
            agent_id,
            state_selector,
            default_store_selector,
            origin_id,
            inference,
            None,
        )
        .await
    }

    pub(crate) async fn create_linked_session_for_agent_with_context(
        &self,
        agent_id: &str,
        state_selector: StoreSelector,
        default_store_selector: Option<StoreSelector>,
        origin_id: Option<String>,
        inference: crate::kernel::config::InferenceOverrideConfig,
        link: LinkedSessionCreate,
    ) -> Result<SessionState> {
        let session = self
            .create_session_for_agent_with_context_and_link(
                agent_id,
                Some(state_selector),
                default_store_selector,
                origin_id,
                inference,
                Some(link),
            )
            .await;
        anyhow::ensure!(
            session.internal_id.is_some(),
            "Failed to create linked session in its parent state store"
        );
        Ok(session)
    }

    async fn create_session_for_agent_with_context_and_link(
        &self,
        agent_id: &str,
        state_selector: Option<StoreSelector>,
        default_store_selector: Option<StoreSelector>,
        origin_id: Option<String>,
        inference: crate::kernel::config::InferenceOverrideConfig,
        link: Option<LinkedSessionCreate>,
    ) -> SessionState {
        let mut session = SessionState::new();
        session.identity.set_agent_id(agent_id.to_string());
        session.identity.set_origin_id(origin_id);
        session.store_selector =
            state_selector.unwrap_or_else(|| self.resolve_agent_state_selector(agent_id));
        session.default_store_selector =
            default_store_selector.or_else(|| self.resolve_agent_default_store_selector(agent_id));
        session.inference = inference;
        self.attach_session_persistence(&mut session, true, link.as_ref())
            .await;
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
        origin_id: Option<String>,
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
        session.identity.set_origin_id(origin_id.or(row.origin_id));
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
        self.attach_session_persistence(&mut session, false, None)
            .await;
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
        self.refresh_session_from_materialized_target(session, &store, &row, context_target)
            .await
    }

    async fn refresh_session_from_target(
        &self,
        session: &mut SessionState,
        context_target: ExecutionContextTarget,
    ) -> Result<()> {
        let (store, row) = self
            .load_current_session_row(
                session,
                "Session refresh requires a configured persistent state store",
            )
            .await?;
        self.refresh_session_from_materialized_target(session, &store, &row, context_target)
            .await
    }

    async fn refresh_session_from_materialized_target(
        &self,
        session: &mut SessionState,
        store: &StateStore,
        row: &SessionRow,
        context_target: ExecutionContextTarget,
    ) -> Result<()> {
        let session_reference = self.session_reference(session);
        let materialized = materialize_session_target(
            self,
            store,
            &session.store_selector,
            row,
            &context_target,
            &session_reference,
            "session.refresh.materialize",
        )
        .await?;
        let (history, turn_index) = rebuild_history(&materialized.messages)?;
        let counters = store.get_session_counters(row.id).await?;

        session.default_store_selector =
            session_default_store_selector_from_metadata(row.metadata.as_deref());
        session.identity.set_origin_id(row.origin_id.clone());
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
        self.refresh_session_from_target(
            session,
            ExecutionContextTarget::BranchHead {
                branch_head_id: Some(branch.id),
            },
        )
        .await?;
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
        self.refresh_session_from_target(session, ExecutionContextTarget::TurnId { turn_id })
            .await?;
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
        self.refresh_session_from_target(
            session,
            ExecutionContextTarget::ExternalReference {
                reference: format_session_reference(&session_ref.public_id, &store_selector),
            },
        )
        .await?;
        Ok(true)
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
        if session.status == SessionStatus::Ended {
            anyhow::bail!(
                "Ended sessions cannot be restarted; resume the persisted session instead"
            );
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
        )
        .await;
        let governance_snapshot = self
            .governance_manager
            .snapshot_for_agent(Some(session.identity.agent_id()));
        self.persist_event(
            session,
            &KernelEvent::Audit(AuditEvent::GovernanceSnapshot {
                snapshot: governance_snapshot.clone(),
            }),
        )
        .await;

        if let Some(harness) = self.session_harness_engine(session) {
            let engine = harness.lock().expect("session harness mutex poisoned");
            if let Err(e) = engine.evaluate_hook(HarnessHook::SessionStart {
                identity: &session.identity,
                session_id: &session_id,
                governance: &governance_snapshot,
            }) {
                warn!(error = %e, "Harness on_session_start failed");
            }
        }

        session.stop_requested = false;
        session.status = SessionStatus::Active;
        Ok(())
    }

    /// End the session and emit SessionEnd event.
    pub async fn end_session(&self, session: &mut SessionState) -> Result<()> {
        if session.status == SessionStatus::Ended {
            return Ok(());
        }

        info!(
            session_id = %self.session_reference(session),
            store = %describe_store_selector(&session.store_selector),
            turn_count = session.turn_index,
            "Ending session"
        );

        if session.status == SessionStatus::Active {
            self.persist_event(
                session,
                &KernelEvent::Lifecycle(LifecycleEvent::SessionEnd {
                    identity: session.identity.clone(),
                    turn_count: session.turn_index,
                    total_input_tokens: session.total_input_tokens,
                    total_output_tokens: session.total_output_tokens,
                }),
            )
            .await;

            if let Some(harness) = self.session_harness_engine(session) {
                let engine = harness.lock().expect("session harness mutex poisoned");
                let session_id = self.session_reference(session);
                if let Err(e) = engine.evaluate_hook(HarnessHook::SessionEnd {
                    identity: &session.identity,
                    session_id: &session_id,
                    turn_count: session.turn_index,
                    total_input_tokens: session.total_input_tokens,
                    total_output_tokens: session.total_output_tokens,
                }) {
                    warn!(error = %e, "Harness on_session_end failed");
                }
                engine.set_active_queue(None);
            }
        }

        let mut durability_error = self.wait_for_session_durability(session).await.err();

        // Close durability lane and await background persistence flush.
        session.durability_tx.take();
        if let Some(task_slot) = &session.event_task
            && let Some(handle) = task_slot.lock().await.take()
            && let Err(e) =
                finish_session_persistence_task(handle, SESSION_PERSISTENCE_SHUTDOWN_TIMEOUT).await
        {
            warn!(error = %e, "Background persistence task shutdown error");
            durability_error.get_or_insert(e);
        }
        session.cancel_token.cancel();
        self.policy_manager
            .clear_transient_scopes(session.identity.session_id(), session.identity.run_id())
            .await;

        session.status = SessionStatus::Ended;
        self.clear_session_harness_engine(session);
        durability_error.map_or(Ok(()), Err)
    }
}

async fn finish_session_persistence_task(
    mut handle: tokio::task::JoinHandle<()>,
    timeout: std::time::Duration,
) -> Result<()> {
    match tokio::time::timeout(timeout, &mut handle).await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(error)) => Err(anyhow!("Background persistence task failed: {error}")),
        Err(_) => {
            handle.abort();
            let _ = handle.await;
            Err(anyhow!(
                "Timed out waiting for background persistence task shutdown"
            ))
        }
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

#[cfg(test)]
mod tests {
    use super::finish_session_persistence_task;

    #[tokio::test]
    async fn stalled_persistence_task_shutdown_is_bounded() {
        let handle = tokio::spawn(std::future::pending());
        let error = finish_session_persistence_task(handle, std::time::Duration::from_millis(10))
            .await
            .expect_err("stalled task should time out");

        assert!(
            error
                .to_string()
                .contains("Timed out waiting for background persistence task shutdown")
        );
    }
}
