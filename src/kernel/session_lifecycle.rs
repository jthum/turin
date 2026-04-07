use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use tokio::sync::Mutex as AsyncMutex;
use tracing::{info, warn};

use crate::kernel::config::StoreTargetConfig;
use crate::kernel::event::{AuditEvent, KernelEvent, LifecycleEvent};
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::session::{PersistedKernelRecord, SessionState, SessionStatus};
use crate::kernel::session_refs::{
    describe_store_selector, format_session_reference, parse_session_reference,
};
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::{EventRow, MessageRow};
use crate::{
    inference::provider::{InferenceContent, InferenceMessage, InferenceRole},
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

        let messages = store.get_messages(row.id).await?;
        let events = store.get_all_events(row.id).await?;
        let (history, turn_index) = rebuild_history(&messages)?;
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
        session.history = history;
        session.turn_index = turn_index;
        session.total_input_tokens = total_input_tokens;
        session.total_output_tokens = total_output_tokens;
        session.next_task_id = next_task_id;
        session.next_plan_id = next_plan_id;
        session.restored_from_persistence = true;
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

        let messages = store.get_messages(row.id).await?;
        let events = store.get_all_events(row.id).await?;
        let (history, turn_index) = rebuild_history(&messages)?;
        let (next_task_id, next_plan_id, total_input_tokens, total_output_tokens) =
            rebuild_session_counters(&events);

        session.default_store_selector =
            session_default_store_selector_from_metadata(row.metadata.as_deref());
        session
            .identity
            .set_channel_id(session_channel_id_from_metadata(row.metadata.as_deref()));
        session.history = history;
        session.turn_index = turn_index;
        session.total_input_tokens = total_input_tokens;
        session.total_output_tokens = total_output_tokens;
        session.next_task_id = next_task_id;
        session.next_plan_id = next_plan_id;
        session.restored_from_persistence = true;
        Ok(())
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
                    Ok(id) => session.internal_id = Some(id),
                    Err(e) => warn!(error = %e, "Failed to create session row in DB"),
                }
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
                                    .insert_event_with_turn_index(
                                        iid,
                                        record.turn_index,
                                        &event_type,
                                        &payload,
                                    )
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

        {
            let runtime = self.runtime_for_session(session);
            let harness = runtime.lock_engine();
            if let Some(ref engine) = *harness {
                engine.set_active_queue(Some(session.queue.clone()));
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

        {
            let runtime = self.runtime_for_session(session);
            let harness = runtime.lock_engine();
            if let Some(ref engine) = *harness {
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

fn session_default_store_selector_from_metadata(metadata: Option<&str>) -> Option<StoreSelector> {
    let parsed = metadata
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
        .and_then(|value| value.get("_turin").cloned())
        .and_then(|value| value.get("default_store").cloned())
        .and_then(|value| serde_json::from_value::<StoreTargetConfig>(value).ok());

    parsed.and_then(store_selector_from_target)
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

fn rebuild_history(messages: &[MessageRow]) -> Result<(Vec<InferenceMessage>, u32)> {
    let mut history = Vec::new();
    let mut max_turn_index = None;

    for message in messages {
        max_turn_index =
            Some(max_turn_index.map_or(message.turn_index, |max: u32| max.max(message.turn_index)));
        let content_json: serde_json::Value = serde_json::from_str(&message.content)
            .with_context(|| format!("Failed to parse persisted message {}", message.id))?;
        let content = decode_persisted_content(&message.role, content_json)
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

fn decode_persisted_content(role: &str, value: serde_json::Value) -> Result<Vec<InferenceContent>> {
    let parts = value
        .as_array()
        .ok_or_else(|| anyhow!("Persisted message content for '{}' is not an array", role))?;
    let mut content = Vec::with_capacity(parts.len());
    for part in parts {
        let part_type = part
            .get("type")
            .and_then(|value| value.as_str())
            .ok_or_else(|| anyhow!("Persisted message content missing type"))?;
        let item = match part_type {
            "text" => InferenceContent::Text {
                text: part
                    .get("text")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default()
                    .to_string(),
            },
            "tool_use" => InferenceContent::ToolUse {
                id: part
                    .get("id")
                    .and_then(|value| value.as_str())
                    .ok_or_else(|| anyhow!("tool_use content missing id"))?
                    .to_string(),
                name: part
                    .get("name")
                    .and_then(|value| value.as_str())
                    .ok_or_else(|| anyhow!("tool_use content missing name"))?
                    .to_string(),
                input: part
                    .get("input")
                    .cloned()
                    .unwrap_or_else(|| serde_json::json!({})),
            },
            "tool_result" => InferenceContent::ToolResult {
                tool_use_id: part
                    .get("tool_use_id")
                    .and_then(|value| value.as_str())
                    .ok_or_else(|| anyhow!("tool_result content missing tool_use_id"))?
                    .to_string(),
                content: part
                    .get("content")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default()
                    .to_string(),
                is_error: part
                    .get("is_error")
                    .and_then(|value| value.as_bool())
                    .unwrap_or(false),
            },
            "thinking" => InferenceContent::Thinking {
                content: part
                    .get("content")
                    .and_then(|value| value.as_str())
                    .unwrap_or_default()
                    .to_string(),
                signature: part
                    .get("signature")
                    .and_then(|value| value.as_str())
                    .map(str::to_string),
            },
            other => anyhow::bail!("Unsupported persisted content type '{}'", other),
        };
        content.push(item);
    }
    Ok(content)
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

fn next_numeric_suffix(value: &str, prefix: &str) -> u32 {
    value
        .strip_prefix(prefix)
        .and_then(|suffix| suffix.parse::<u32>().ok())
        .map(|value| value + 1)
        .unwrap_or(1)
}
