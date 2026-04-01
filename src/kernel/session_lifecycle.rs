use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use tokio::sync::Mutex as AsyncMutex;
use tracing::{info, warn};

use crate::kernel::event::{AuditEvent, KernelEvent, LifecycleEvent};
use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::session::{SessionState, SessionStatus};
use crate::kernel::session_refs::{format_session_reference, parse_session_reference};
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
        self.create_session_for_agent_in_store(agent_id, None).await
    }

    pub async fn create_session_for_agent_in_store(
        &self,
        agent_id: &str,
        store_selector: Option<StoreSelector>,
    ) -> SessionState {
        let mut session = SessionState::new();
        session.identity.set_agent_id(agent_id.to_string());
        session.store_selector =
            store_selector.unwrap_or_else(|| self.resolve_agent_state_selector(agent_id));
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
        let events = store.get_events(row.id).await?;
        let (history, turn_index) = rebuild_history(&messages)?;
        let (next_task_id, next_plan_id, total_input_tokens, total_output_tokens) =
            rebuild_session_counters(&events);

        let mut session = SessionState::new();
        session.identity = RuntimeIdentity::new(session_ref.public_id, agent_id);
        session.internal_id = Some(row.id);
        session.store_selector = store_selector;
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

    async fn attach_session_persistence(&self, session: &mut SessionState, create_row: bool) {
        if let Ok(store) = self.store_manager.open(&session.store_selector).await {
            if create_row
                && let Ok(public_id) = uuid::Uuid::parse_str(session.identity.session_id())
            {
                match store
                    .create_session(public_id, session.identity.agent_id(), None)
                    .await
                {
                    Ok(id) => session.internal_id = Some(id),
                    Err(e) => warn!(error = %e, "Failed to create session row in DB"),
                }
            }

            let (durability_tx, mut durability_rx) =
                tokio::sync::mpsc::unbounded_channel::<(Option<i64>, KernelEvent)>();
            session.durability_tx = Some(durability_tx);
            let store_clone = store.clone();
            let handle = tokio::spawn(async move {
                while let Some((session_id, event)) = durability_rx.recv().await {
                    let event_type = event.event_type().to_string();
                    let payload = serde_json::to_value(&event).unwrap_or_default();
                    if let Some(iid) = session_id {
                        if let Err(e) = store_clone.insert_event(iid, &event_type, &payload).await {
                            warn!(error = %e, "Background persistence error");
                        }
                    } else {
                        warn!("Dropping event: no internal_id for session");
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

    /// Start a new session.
    pub async fn start_session(&self, session: &mut SessionState) -> Result<()> {
        if session.status == SessionStatus::Active {
            return Ok(());
        }

        let session_id = self.session_reference(session);
        info!(session_id = %session_id, "Starting new session");

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
