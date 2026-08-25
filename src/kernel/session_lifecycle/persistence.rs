use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::Mutex as AsyncMutex;
use tracing::warn;

use crate::kernel::execution_host::ExecutionHost;
use crate::kernel::session::{PersistedKernelRecord, SessionState};
use crate::kernel::session_metadata::create_session_metadata;
use crate::persistence::schema::LinkedSessionCreate;

impl ExecutionHost {
    pub(super) async fn attach_session_persistence(
        &self,
        session: &mut SessionState,
        create_row: bool,
        link: Option<&LinkedSessionCreate>,
    ) -> Result<()> {
        let store = self
            .store_manager
            .open(&session.store_selector)
            .await
            .with_context(|| {
                format!(
                    "Failed to open session store for {}",
                    session.identity.session_id()
                )
            })?;

        if create_row && let Ok(public_id) = uuid::Uuid::parse_str(session.identity.session_id()) {
            let metadata = create_session_metadata(session.default_store_selector.as_ref());
            let created = match link {
                Some(link) => {
                    store
                        .create_linked_session(
                            public_id,
                            session.identity.agent_id(),
                            metadata.as_deref(),
                            link,
                        )
                        .await
                }
                None => {
                    store
                        .create_session_with_origin(
                            public_id,
                            session.identity.agent_id(),
                            session.identity.origin_id(),
                            metadata.as_deref(),
                        )
                        .await
                }
            };
            match created {
                Ok(id) => {
                    session.internal_id = Some(id);
                    initialize_branch_cursor(&store, session, id).await;
                    if let Err(error) = self.bind_session_persistence_lock(session).await {
                        warn!(%error, "Failed to bind shared persistence lock for session");
                    }
                }
                Err(error) => {
                    return Err(error).context("Failed to create session row in DB");
                }
            }
        }

        if create_row
            && session.internal_id.is_none()
            && let Err(error) = self.bind_session_persistence_lock(session).await
        {
            warn!(%error, "Failed to bind shared persistence lock for session");
        }
        if !create_row && let Err(error) = self.bind_session_persistence_lock(session).await {
            warn!(%error, "Failed to bind shared persistence lock for resumed session");
        }

        let (durability_tx, mut durability_rx) =
            tokio::sync::mpsc::channel::<PersistedKernelRecord>(256);
        session.durability_tx = Some(durability_tx);
        let persistence_lock = Arc::clone(&session.persistence_lock);
        let handle = tokio::spawn(async move {
            let mut event_writer = None;
            let mut pending_error = None;
            while let Some(record) = durability_rx.recv().await {
                match record {
                    PersistedKernelRecord::Event(record) => {
                        let event = record.event;
                        let event_type = event.event_type().to_string();
                        let payload = match serde_json::to_value(&event) {
                            Ok(payload) => payload,
                            Err(error) => {
                                warn!(%error, "Failed to serialize event for persistence");
                                pending_error.get_or_insert_with(|| error.to_string());
                                continue;
                            }
                        };
                        let Some(internal_id) = record.internal_id else {
                            warn!("Dropping event: no internal_id for session");
                            continue;
                        };
                        let _guard = persistence_lock.lock().await;
                        if event_writer.is_none() {
                            match store.event_writer().await {
                                Ok(writer) => event_writer = Some(writer),
                                Err(error) => {
                                    warn!(%error, "Background persistence error");
                                    pending_error.get_or_insert_with(|| error.to_string());
                                    continue;
                                }
                            }
                        }
                        let Some(writer) = event_writer.as_ref() else {
                            continue;
                        };
                        if let Err(error) = writer
                            .insert_event(internal_id, record.turn_target, &event_type, &payload)
                            .await
                        {
                            warn!(%error, "Background persistence error");
                            pending_error.get_or_insert_with(|| error.to_string());
                            event_writer = None;
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
        Ok(())
    }
}

async fn initialize_branch_cursor(
    store: &crate::persistence::state::StateStore,
    session: &mut SessionState,
    internal_id: i64,
) {
    match store.get_active_branch_head(internal_id).await {
        Ok(Some(branch)) => {
            session.set_selected_branch_head_id(Some(branch.id));
            let head_turn_index = match branch.head_turn_id {
                Some(turn_id) => match store.get_turn_row(turn_id).await {
                    Ok(Some(turn)) => Some(turn.branch_depth),
                    Ok(None) => None,
                    Err(error) => {
                        warn!(%error, turn_id, "Failed to load initial branch head turn depth");
                        None
                    }
                },
                None => None,
            };
            session.set_selected_branch_head_cursor(branch.head_turn_id, head_turn_index);
        }
        Ok(None) => warn!("Created session is missing an active branch head"),
        Err(error) => warn!(%error, "Failed to load initial branch head for session"),
    }
}
