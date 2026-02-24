use std::sync::Arc;

use anyhow::Result;
use tokio::sync::Mutex as AsyncMutex;
use tracing::{info, warn};

use crate::kernel::Kernel;
use crate::kernel::event::{AuditEvent, KernelEvent, LifecycleEvent};
use crate::kernel::session::{SessionState, SessionStatus};

impl Kernel {
    /// Create a new session.
    pub async fn create_session(&self) -> SessionState {
        let mut session = SessionState::new();
        session.identity.set_agent_id(self.config.agent.id.clone());

        // Spawn background persistence if state is available.
        if let Ok(store) = self.store_manager.get_default().await {
            // Create the session row eagerly so we have an internal_id for later persistence.
            if let Ok(public_id) = uuid::Uuid::parse_str(session.identity.session_id()) {
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

        session
    }

    /// Start a new session.
    pub async fn start_session(&self, session: &mut SessionState) -> Result<()> {
        if session.status == SessionStatus::Active {
            return Ok(());
        }

        let session_id = session.identity.session_id().to_string();
        info!(session_id = %session_id, "Starting new session");

        self.persist_event(
            session,
            &KernelEvent::Lifecycle(LifecycleEvent::SessionStart {
                identity: session.identity.clone(),
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
            let harness = self.lock_harness();
            if let Some(ref engine) = *harness
                && let Err(e) = engine.evaluate(
                    "on_session_start",
                    serde_json::json!({
                        "identity": session.identity.clone(),
                        "session_id": session_id,
                        "governance": governance_snapshot,
                    }),
                )
            {
                warn!(error = %e, "Harness on_session_start failed");
            }
        }

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
            let harness = self.lock_harness();
            if let Some(ref engine) = *harness
                && let Err(e) = engine.evaluate(
                    "on_session_end",
                    serde_json::json!({
                        "identity": session.identity.clone(),
                        "session_id": session.identity.session_id(),
                        "turn_count": session.turn_index,
                        "total_input_tokens": session.total_input_tokens,
                        "total_output_tokens": session.total_output_tokens,
                    }),
                )
            {
                warn!(error = %e, "Harness on_session_end failed");
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

        // Clear active queue.
        {
            let mut aq = self.active_queue.lock().await;
            *aq = None;
        }

        session.status = SessionStatus::Inactive;
        Ok(())
    }
}
