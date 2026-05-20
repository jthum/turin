use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use tokio::sync::Mutex as AsyncMutex;
use tracing::{debug, info, warn};

use crate::inference::content::decode_content_json;
use crate::kernel::config::StoreTargetConfig;
use crate::kernel::event::{AuditEvent, KernelEvent, LifecycleEvent};
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::session::{
    ContextCompactionCheckpoint, ExecutionConflictPolicy, ExecutionContextTarget,
    ExecutionDurability, ExecutionVisibility, ExecutionWritePolicy, PersistedKernelRecord,
    PreparedSidestepExecution, SessionState, SessionStatus, SidestepMode, TaskExecutionOverrides,
};
use crate::kernel::session_refs::{
    describe_store_selector, format_session_reference, parse_session_reference,
};
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::{
    BranchHeadRow, BranchProvenance, EventRow, MessageRow, SessionRow,
};
use crate::persistence::state::{SessionReadTarget, StateStore};
use crate::{
    inference::provider::{InferenceMessage, InferenceRole},
    kernel::identity::RuntimeIdentity,
};

impl ExecutionHost {
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
        let materialized =
            materialize_execution_target(self, &store, &store_selector, &row, &context_target)
                .await?;
        let events = store.get_all_events(row.id).await?;
        let (history, turn_index) = rebuild_history(&materialized.messages)?;
        let (next_task_id, next_plan_id, total_input_tokens, total_output_tokens) =
            rebuild_session_counters(&events);

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
        session.context_checkpoint = rebuild_context_checkpoint(&materialized.active_events);
        session.replace_full_history(history);
        session.replace_context_target_preserving_policy(context_target);
        session.set_selected_branch_head_cursor(
            materialized.branch_head_turn_id,
            materialized.branch_head_turn_index,
        );
        session.turn_index = turn_index;
        session.total_input_tokens = total_input_tokens;
        session.total_output_tokens = total_output_tokens;
        session.next_task_id = next_task_id;
        session.next_plan_id = next_plan_id;
        session.restored_from_persistence = true;
        self.prune_session_hot_history(&mut session);
        self.attach_session_persistence(&mut session, false).await;
        Ok(session)
    }

    pub async fn refresh_session_from_persistence(&self, session: &mut SessionState) -> Result<()> {
        let internal_id = session
            .internal_id
            .ok_or_else(|| anyhow!("Session has no internal persistence id"))?;
        let store = self
            .store_manager
            .open(&session.store_selector)
            .await
            .context("Session refresh requires a configured persistent state store")?;
        let row = store.get_session_row(internal_id).await?.ok_or_else(|| {
            anyhow!(
                "Persisted session '{}' not found",
                session.identity.session_id()
            )
        })?;

        let context_target = resolved_execution_context_target(session.context_target(), &row);
        let materialized = materialize_execution_target(
            self,
            &store,
            &session.store_selector,
            &row,
            &context_target,
        )
        .await?;
        let events = store.get_all_events(row.id).await?;
        let (history, turn_index) = rebuild_history(&materialized.messages)?;
        let (next_task_id, next_plan_id, total_input_tokens, total_output_tokens) =
            rebuild_session_counters(&events);

        session.default_store_selector =
            session_default_store_selector_from_metadata(row.metadata.as_deref());
        session
            .identity
            .set_channel_id(session_channel_id_from_metadata(row.metadata.as_deref()));
        session.context_checkpoint = rebuild_context_checkpoint(&materialized.active_events);
        session.replace_full_history(history);
        session.set_context_target(context_target);
        session.set_selected_branch_head_cursor(
            materialized.branch_head_turn_id,
            materialized.branch_head_turn_index,
        );
        session.turn_index = turn_index;
        session.total_input_tokens = total_input_tokens;
        session.total_output_tokens = total_output_tokens;
        session.next_task_id = next_task_id;
        session.next_plan_id = next_plan_id;
        session.restored_from_persistence = true;
        self.prune_session_hot_history(session);
        Ok(())
    }

    pub(crate) async fn ensure_full_history_materialized(
        &self,
        session: &mut SessionState,
    ) -> Result<()> {
        if !session.history_is_pruned() {
            return Ok(());
        }
        let internal_id = session
            .internal_id
            .ok_or_else(|| anyhow!("Session has no internal persistence id"))?;
        let store = self
            .store_manager
            .open(&session.store_selector)
            .await
            .context(
                "Session history materialization requires a configured persistent state store",
            )?;
        let row = store.get_session_row(internal_id).await?.ok_or_else(|| {
            anyhow!(
                "Persisted session '{}' not found",
                session.identity.session_id()
            )
        })?;
        let context_target = resolved_execution_context_target(session.context_target(), &row);
        let materialized = materialize_execution_target(
            self,
            &store,
            &session.store_selector,
            &row,
            &context_target,
        )
        .await?;
        let (history, _) = rebuild_history(&materialized.messages)?;
        session.replace_full_history(history);
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
            let retained_offset = report
                .prune
                .map(|report| report.retained_offset)
                .unwrap_or(session.history_message_offset);
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
                retained_offset,
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
        let internal_id = session
            .internal_id
            .ok_or_else(|| anyhow!("Session has no internal persistence id"))?;
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
        let internal_id = session
            .internal_id
            .ok_or_else(|| anyhow!("Session has no internal persistence id"))?;
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
                let metadata = session_create_metadata(session);
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
                while let Some(record) = durability_rx.recv().await {
                    match record {
                        PersistedKernelRecord::Event(record) => {
                            let event = record.event;
                            let event_type = event.event_type().to_string();
                            let payload = serde_json::to_value(&event).unwrap_or_default();
                            if let Some(iid) = record.internal_id {
                                let _guard = persistence_lock.lock().await;
                                if let Err(e) = store_clone
                                    .insert_event(iid, record.turn_target, &event_type, &payload)
                                    .await
                                {
                                    warn!(error = %e, "Background persistence error");
                                }
                            } else {
                                warn!("Dropping event: no internal_id for session");
                            }
                        }
                        PersistedKernelRecord::Barrier(tx) => {
                            let _guard = persistence_lock.lock().await;
                            let _ = tx.send(());
                        }
                    }
                }
            });
            session.event_task = Some(Arc::new(AsyncMutex::new(Some(handle))));
        }
    }

    pub(crate) fn resolve_agent_state_selector(&self, agent_id: &str) -> StoreSelector {
        let context = if agent_id == self.config.agent.id {
            Some(&self.config.agent.persistence)
        } else {
            self.config
                .agents
                .get(agent_id)
                .map(|agent| &agent.persistence)
        };
        self.config
            .persistence
            .resolve_context_state_selector(context)
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
        let context = if agent_id == self.config.agent.id {
            Some(&self.config.agent.persistence)
        } else {
            self.config
                .agents
                .get(agent_id)
                .map(|agent| &agent.persistence)
        };
        let context = context?;
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

        // Close durability lane and await background persistence flush.
        session.durability_tx.take();
        if let Some(task_slot) = &session.event_task
            && let Some(handle) = task_slot.lock().await.take()
            && let Err(e) = handle.await
        {
            warn!(error = %e, "Background persistence task join error");
        }
        session.cancel_token.cancel();

        session.status = SessionStatus::Inactive;
        self.clear_session_harness_engine(session);
        Ok(())
    }
}

fn session_create_metadata(session: &SessionState) -> Option<String> {
    let mut turin_meta = serde_json::Map::new();
    if let Some(default_store) = session
        .default_store_selector
        .as_ref()
        .and_then(store_target_from_selector)
    {
        turin_meta.insert(
            "default_store".to_string(),
            serde_json::json!(default_store),
        );
    }
    if let Some(channel_id) = session.identity.channel_id() {
        turin_meta.insert("channel_id".to_string(), serde_json::json!(channel_id));
    }
    if turin_meta.is_empty() {
        return None;
    }
    Some(
        serde_json::json!({
            "_turin": turin_meta,
        })
        .to_string(),
    )
}

async fn ensure_local_session_target_idle(session: &SessionState, target_kind: &str) -> Result<()> {
    if !session.queue.lock().await.is_empty() {
        anyhow::bail!("Cannot switch local session {target_kind} while tasks are queued");
    }
    Ok(())
}

fn session_default_store_selector_from_metadata(metadata: Option<&str>) -> Option<StoreSelector> {
    let parsed = metadata
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
        .and_then(|value| value.get("_turin").cloned())
        .and_then(|value| value.get("default_store").cloned())
        .and_then(|value| serde_json::from_value::<StoreTargetConfig>(value).ok());

    parsed.and_then(store_selector_from_target)
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

struct MaterializedExecutionTarget {
    messages: Vec<MessageRow>,
    active_events: Vec<EventRow>,
    branch_head_turn_id: Option<i64>,
    branch_head_turn_index: Option<u32>,
}

async fn materialize_execution_target(
    host: &ExecutionHost,
    store: &StateStore,
    current_store_selector: &StoreSelector,
    row: &SessionRow,
    target: &ExecutionContextTarget,
) -> Result<MaterializedExecutionTarget> {
    match target {
        ExecutionContextTarget::BranchHead { branch_head_id } => {
            let branch_head_id = branch_head_id.or(row.active_branch_head_id);
            let target = SessionReadTarget::branch_head(branch_head_id);
            let branch_head_turn_id = match branch_head_id {
                Some(branch_head_id) => store
                    .get_branch_head(row.id, branch_head_id)
                    .await?
                    .and_then(|branch| branch.head_turn_id),
                None => store
                    .get_active_branch_head(row.id)
                    .await?
                    .and_then(|branch| branch.head_turn_id),
            };
            Ok(MaterializedExecutionTarget {
                messages: store.get_messages(row.id, &target).await?,
                active_events: store.get_events(row.id, &target).await?,
                branch_head_turn_id,
                branch_head_turn_index: match branch_head_turn_id {
                    Some(turn_id) => store
                        .get_turn_row(turn_id)
                        .await?
                        .map(|turn| turn.branch_depth),
                    None => None,
                },
            })
        }
        ExecutionContextTarget::TurnId { turn_id } => {
            let target = SessionReadTarget::TurnId(*turn_id);
            Ok(MaterializedExecutionTarget {
                messages: store.get_messages(row.id, &target).await?,
                active_events: store.get_events(row.id, &target).await?,
                branch_head_turn_id: None,
                branch_head_turn_index: None,
            })
        }
        ExecutionContextTarget::SelectedPath { turn_ids } => {
            let target = SessionReadTarget::SelectedPath(turn_ids.clone());
            Ok(MaterializedExecutionTarget {
                messages: store.get_messages(row.id, &target).await?,
                active_events: store.get_events(row.id, &target).await?,
                branch_head_turn_id: None,
                branch_head_turn_index: None,
            })
        }
        ExecutionContextTarget::SummarySource { source_turn_id } => {
            let target = SessionReadTarget::TurnId(*source_turn_id);
            Ok(MaterializedExecutionTarget {
                messages: store.get_messages(row.id, &target).await?,
                active_events: store.get_events(row.id, &target).await?,
                branch_head_turn_id: None,
                branch_head_turn_index: None,
            })
        }
        ExecutionContextTarget::ExternalReference { reference } => {
            let session_ref = parse_session_reference(reference)?;
            let target_selector = session_ref
                .store_selector
                .clone()
                .unwrap_or_else(|| current_store_selector.clone());
            let target_store = host
                .store_manager
                .open(&target_selector)
                .await
                .with_context(|| {
                    format!(
                        "Execution context target '{}' requires a configured persistent state store",
                        reference
                    )
                })?;
            let public_id = uuid::Uuid::parse_str(&session_ref.public_id)
                .with_context(|| format!("Invalid session id '{}'", session_ref.public_id))?;
            let referenced_row = target_store
                .get_session_row_by_public_id(public_id)
                .await?
                .ok_or_else(|| anyhow!("Execution context target '{}' was not found", reference))?;
            let target = SessionReadTarget::branch_head(referenced_row.active_branch_head_id);
            Ok(MaterializedExecutionTarget {
                messages: target_store
                    .get_messages(referenced_row.id, &target)
                    .await?,
                active_events: target_store.get_events(referenced_row.id, &target).await?,
                branch_head_turn_id: None,
                branch_head_turn_index: None,
            })
        }
    }
}

fn session_channel_id_from_metadata(metadata: Option<&str>) -> Option<String> {
    metadata
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
        .and_then(|value| value.get("_turin").cloned())
        .and_then(|value| value.get("channel_id").cloned())
        .and_then(|value| value.as_str().map(ToString::to_string))
}

fn store_target_from_selector(selector: &StoreSelector) -> Option<StoreTargetConfig> {
    match selector {
        StoreSelector::Alias(alias) => Some(StoreTargetConfig::from_alias(alias.clone())),
        StoreSelector::Path(path) => Some(StoreTargetConfig::from_path(path.clone())),
        StoreSelector::Handle(_) => None,
    }
}

fn store_selector_from_target(target: StoreTargetConfig) -> Option<StoreSelector> {
    if let Some(path) = target.path {
        Some(StoreSelector::Path(path))
    } else {
        target.alias.map(StoreSelector::Alias)
    }
}

pub(crate) async fn prepare_persisted_session_sidestep(
    store_manager: &Arc<crate::persistence::manager::StoreManager>,
    session_id: &str,
    default_target: &ExecutionContextTarget,
    mode: SidestepMode,
    requested_target: Option<ExecutionContextTarget>,
) -> Result<PreparedSidestepExecution> {
    let (store_selector, row) = resolve_persisted_session_row(store_manager, session_id).await?;
    let store = store_manager.open(&store_selector).await?;
    let resolved_target = requested_target.unwrap_or_else(|| default_target.clone());
    let resolved_target = normalize_sidestep_target(
        store_manager,
        &store_selector,
        &store,
        &row,
        resolved_target,
    )
    .await?;

    match mode {
        SidestepMode::Ephemeral => Ok(PreparedSidestepExecution {
            execution: TaskExecutionOverrides {
                context_target: Some(
                    snapshot_sidestep_target(&store, &row, resolved_target).await?,
                ),
                visibility: Some(ExecutionVisibility::Hidden),
                durability: Some(ExecutionDurability::Ephemeral),
                write_policy: Some(ExecutionWritePolicy::Detached),
            },
            conflict_policy: ExecutionConflictPolicy::Detached,
            branch_outcome: None,
        }),
        SidestepMode::ForkSibling => {
            let source = resolve_sidestep_branch_source(&store, &row, resolved_target).await?;
            let branch_name = format!("sidestep-{}", uuid::Uuid::now_v7().simple());
            let branch = store
                .create_branch_head_from_turn_index_with_provenance(
                    row.id,
                    &branch_name,
                    source.turn_index,
                    false,
                    BranchProvenance::sidestep(),
                )
                .await?;
            let branch_public_id = uuid::Uuid::from_slice(&branch.public_id)
                .map(|value| value.to_string())
                .map_err(anyhow::Error::from)?;

            Ok(PreparedSidestepExecution {
                execution: TaskExecutionOverrides {
                    context_target: Some(ExecutionContextTarget::BranchHead {
                        branch_head_id: Some(branch.id),
                    }),
                    visibility: Some(ExecutionVisibility::Hidden),
                    durability: Some(ExecutionDurability::Durable),
                    write_policy: Some(ExecutionWritePolicy::AdvanceBranchHead),
                },
                conflict_policy: ExecutionConflictPolicy::Reject,
                branch_outcome: Some(crate::kernel::event::TaskBranchOutcome::SidestepSibling {
                    branch_id: branch.id,
                    branch_public_id,
                    branch_name: branch.name,
                    source_turn_id: branch.created_from_turn_id,
                    persisted_active_head_unchanged: !branch.is_active,
                }),
            })
        }
    }
}

async fn normalize_sidestep_target(
    store_manager: &Arc<crate::persistence::manager::StoreManager>,
    default_store_selector: &StoreSelector,
    store: &StateStore,
    row: &SessionRow,
    target: ExecutionContextTarget,
) -> Result<ExecutionContextTarget> {
    match target {
        ExecutionContextTarget::BranchHead { branch_head_id } => match branch_head_id {
            Some(branch_head_id) => {
                let branch = store
                    .get_branch_head(row.id, branch_head_id)
                    .await?
                    .ok_or_else(|| anyhow!("Branch head '{}' not found", branch_head_id))?;
                Ok(ExecutionContextTarget::BranchHead {
                    branch_head_id: Some(branch.id),
                })
            }
            None => Ok(ExecutionContextTarget::BranchHead {
                branch_head_id: None,
            }),
        },
        ExecutionContextTarget::TurnId { turn_id } => {
            validate_session_turn_target(store, row.id, turn_id, "sidestep target").await?;
            Ok(ExecutionContextTarget::TurnId { turn_id })
        }
        ExecutionContextTarget::SelectedPath { turn_ids } => {
            store
                .turn_rows_for_selected_path(row.id, &turn_ids)
                .await
                .context("Invalid selected sidestep path")?;
            Ok(ExecutionContextTarget::SelectedPath { turn_ids })
        }
        ExecutionContextTarget::SummarySource { source_turn_id } => {
            validate_session_turn_target(store, row.id, source_turn_id, "sidestep summary source")
                .await?;
            Ok(ExecutionContextTarget::SummarySource { source_turn_id })
        }
        ExecutionContextTarget::ExternalReference { reference } => {
            let session_ref = parse_session_reference(&reference)?;
            let resolved_selector = session_ref
                .store_selector
                .clone()
                .unwrap_or_else(|| default_store_selector.clone());
            let external_store = store_manager.open(&resolved_selector).await?;
            let public_id = uuid::Uuid::parse_str(&session_ref.public_id)
                .with_context(|| format!("Invalid session id '{}'", session_ref.public_id))?;
            let Some(_) = external_store
                .get_session_row_by_public_id(public_id)
                .await?
            else {
                anyhow::bail!("External sidestep reference '{}' not found", reference);
            };
            Ok(ExecutionContextTarget::ExternalReference {
                reference: format_session_reference(&session_ref.public_id, &resolved_selector),
            })
        }
    }
}

async fn snapshot_sidestep_target(
    store: &StateStore,
    row: &SessionRow,
    target: ExecutionContextTarget,
) -> Result<ExecutionContextTarget> {
    match target {
        ExecutionContextTarget::BranchHead { branch_head_id } => {
            let branch = match branch_head_id {
                Some(branch_head_id) => store.get_branch_head(row.id, branch_head_id).await?,
                None => store.get_active_branch_head(row.id).await?,
            };
            Ok(snapshot_target_from_branch_head(branch, branch_head_id))
        }
        other => Ok(other),
    }
}

async fn resolve_persisted_session_row(
    store_manager: &Arc<crate::persistence::manager::StoreManager>,
    session_id: &str,
) -> Result<(StoreSelector, SessionRow)> {
    let session_ref = parse_session_reference(session_id)?;
    let public_id = uuid::Uuid::parse_str(&session_ref.public_id)
        .with_context(|| format!("Invalid session id '{}'", session_ref.public_id))?;
    let store_selector = session_ref
        .store_selector
        .unwrap_or_else(|| StoreSelector::Alias("state".to_string()));
    let store = store_manager.open(&store_selector).await?;
    let row = store
        .get_session_row_by_public_id(public_id)
        .await?
        .ok_or_else(|| anyhow!("Session '{}' not found", session_id))?;
    Ok((store_selector, row))
}

struct SidestepBranchSource {
    turn_index: Option<u32>,
}

async fn resolve_sidestep_branch_source(
    store: &StateStore,
    row: &SessionRow,
    target: ExecutionContextTarget,
) -> Result<SidestepBranchSource> {
    match target {
        ExecutionContextTarget::BranchHead { branch_head_id } => {
            let branch = match branch_head_id {
                Some(branch_head_id) => store.get_branch_head(row.id, branch_head_id).await?,
                None => store.get_active_branch_head(row.id).await?,
            };
            sidestep_branch_source_from_branch(branch)
        }
        ExecutionContextTarget::TurnId { turn_id } => {
            sidestep_branch_source_from_turn(store, row.id, turn_id).await
        }
        ExecutionContextTarget::SelectedPath { turn_ids } => {
            let Some(turn_id) = turn_ids.last().copied() else {
                anyhow::bail!("Selected sidestep path must include at least one turn");
            };
            sidestep_branch_source_from_turn(store, row.id, turn_id).await
        }
        ExecutionContextTarget::SummarySource { source_turn_id } => {
            sidestep_branch_source_from_turn(store, row.id, source_turn_id).await
        }
        ExecutionContextTarget::ExternalReference { .. } => {
            anyhow::bail!("fork_sibling sidesteps do not support external_reference targets")
        }
    }
}

fn snapshot_target_from_branch_head(
    branch: Option<BranchHeadRow>,
    explicit_branch_head_id: Option<i64>,
) -> ExecutionContextTarget {
    match branch.and_then(|branch| branch.head_turn_id) {
        Some(turn_id) => ExecutionContextTarget::TurnId { turn_id },
        None => ExecutionContextTarget::BranchHead {
            branch_head_id: explicit_branch_head_id,
        },
    }
}

fn sidestep_branch_source_from_branch(
    branch: Option<BranchHeadRow>,
) -> Result<SidestepBranchSource> {
    let Some(branch) = branch else {
        anyhow::bail!("No branch head available for sidestep source");
    };
    Ok(SidestepBranchSource {
        turn_index: branch.head_turn_depth,
    })
}

async fn validate_session_turn_target(
    store: &StateStore,
    session_internal_id: i64,
    turn_id: i64,
    label: &str,
) -> Result<()> {
    let Some(turn) = store.get_turn_row(turn_id).await? else {
        anyhow::bail!("{} '{}' not found", label, turn_id);
    };
    if turn.session_id != session_internal_id {
        anyhow::bail!(
            "{} '{}' does not belong to the target session",
            label,
            turn_id
        );
    }
    Ok(())
}

async fn sidestep_branch_source_from_turn(
    store: &StateStore,
    session_internal_id: i64,
    turn_id: i64,
) -> Result<SidestepBranchSource> {
    validate_session_turn_target(store, session_internal_id, turn_id, "sidestep source turn")
        .await?;
    let turn = store
        .get_turn_row(turn_id)
        .await?
        .expect("validated sidestep source turn should exist");
    Ok(SidestepBranchSource {
        turn_index: Some(turn.branch_depth),
    })
}

fn rebuild_history(messages: &[MessageRow]) -> Result<(Vec<InferenceMessage>, u32)> {
    let mut history = Vec::new();
    let mut max_turn_index = None;

    for message in messages {
        max_turn_index =
            Some(max_turn_index.map_or(message.turn_index, |max: u32| max.max(message.turn_index)));
        let content_json: serde_json::Value = serde_json::from_str(&message.content)
            .with_context(|| format!("Failed to parse persisted message {}", message.id))?;
        let content = decode_content_json(content_json)
            .with_context(|| format!("Failed to rebuild persisted message {}", message.id))?;
        history.push(InferenceMessage {
            role: decode_role(&message.role)?,
            content,
            tool_call_id: None,
        });
    }

    Ok((history, max_turn_index.map_or(0, |idx| idx + 1)))
}

fn decode_role(role: &str) -> Result<InferenceRole> {
    match role {
        "user" => Ok(InferenceRole::User),
        "assistant" => Ok(InferenceRole::Assistant),
        "tool_result" => Ok(InferenceRole::Tool),
        other => anyhow::bail!("Unsupported persisted role '{}'", other),
    }
}

fn rebuild_session_counters(events: &[EventRow]) -> (u32, u32, u64, u64) {
    let mut next_task_id = 1;
    let mut next_plan_id = 1;
    let mut total_input_tokens = 0;
    let mut total_output_tokens = 0;

    for event in events {
        let Ok(payload) = serde_json::from_str::<serde_json::Value>(&event.payload) else {
            continue;
        };
        if let Some(task_id) = payload.get("task_id").and_then(|value| value.as_str()) {
            next_task_id = next_task_id.max(next_numeric_suffix(task_id, "t_"));
        }
        if let Some(plan_id) = payload.get("plan_id").and_then(|value| value.as_str()) {
            next_plan_id = next_plan_id.max(next_numeric_suffix(plan_id, "p_"));
        }
        match event.event_type.as_str() {
            "message_end" => {
                total_input_tokens += payload
                    .get("input_tokens")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(0);
                total_output_tokens += payload
                    .get("output_tokens")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(0);
            }
            "session_end" => {
                total_input_tokens = payload
                    .get("total_input_tokens")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(total_input_tokens);
                total_output_tokens = payload
                    .get("total_output_tokens")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(total_output_tokens);
            }
            _ => {}
        }
    }

    (
        next_task_id,
        next_plan_id,
        total_input_tokens,
        total_output_tokens,
    )
}

fn rebuild_context_checkpoint(events: &[EventRow]) -> Option<ContextCompactionCheckpoint> {
    let mut checkpoint = None;

    for event in events {
        if event.event_type != "context_compaction" {
            continue;
        }

        let Ok(KernelEvent::Audit(AuditEvent::ContextCompaction {
            checkpoint: persisted,
        })) = serde_json::from_str::<KernelEvent>(&event.payload)
        else {
            continue;
        };

        checkpoint = Some(persisted);
    }

    checkpoint
}

fn next_numeric_suffix(value: &str, prefix: &str) -> u32 {
    value
        .strip_prefix(prefix)
        .and_then(|suffix| suffix.parse::<u32>().ok())
        .map(|value| value + 1)
        .unwrap_or(1)
}
