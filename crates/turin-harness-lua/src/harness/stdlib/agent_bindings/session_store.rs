use std::sync::Arc;

use crate::harness::globals::ActiveHarnessExecutionContext;
use crate::kernel::session_refs::{SessionReference, parse_session_reference};
use crate::persistence::manager::{StoreManager, StoreSelector};
use crate::persistence::schema::SessionRow;
use crate::persistence::state::StateStore;

pub(super) struct SessionStoreLookup {
    pub(super) row: Option<SessionRow>,
    session_ref: SessionReference,
    selector: StoreSelector,
    store: Arc<StateStore>,
}

impl SessionStoreLookup {
    pub(super) fn require_row(self) -> Result<ResolvedSessionStore, String> {
        let row = self.row.ok_or_else(|| "Session not found".to_string())?;
        Ok(ResolvedSessionStore {
            session_ref: self.session_ref,
            selector: self.selector,
            store: self.store,
            row,
        })
    }
}

pub(super) struct ResolvedSessionStore {
    pub(super) session_ref: SessionReference,
    pub(super) selector: StoreSelector,
    pub(super) store: Arc<StateStore>,
    pub(super) row: SessionRow,
}

pub(super) fn current_session_store_selector(
    execution_ctx: &ActiveHarnessExecutionContext,
) -> Result<StoreSelector, String> {
    execution_ctx
        .lock()
        .map_err(|_| "execution context mutex poisoned".to_string())
        .map(|lock| {
            lock.session_store_selector
                .clone()
                .unwrap_or_else(|| StoreSelector::Alias("state".to_string()))
        })
}

pub(super) fn current_completed_task_results(
    execution_ctx: &ActiveHarnessExecutionContext,
) -> Result<crate::kernel::session::CompletedLocalTaskResultsHandle, String> {
    execution_ctx
        .lock()
        .map_err(|_| "execution context mutex poisoned".to_string())?
        .completed_task_results
        .clone()
        .ok_or_else(|| "No active session completed-task cache".to_string())
}

fn resolve_session_reference(
    execution_ctx: &ActiveHarnessExecutionContext,
    requested: Option<String>,
) -> Result<SessionReference, String> {
    let implicit_selector = current_session_store_selector(execution_ctx)?;
    let raw = match requested {
        Some(session_id) => session_id,
        None => execution_ctx
            .lock()
            .map_err(|_| "execution context mutex poisoned".to_string())?
            .session_id
            .clone()
            .ok_or_else(|| "No active session context".to_string())?,
    };
    let mut session_ref = parse_session_reference(&raw).map_err(|e| e.to_string())?;
    if session_ref.store_selector.is_none() {
        session_ref.store_selector = Some(implicit_selector);
    }
    Ok(session_ref)
}

pub(super) fn current_session_matches(
    execution_ctx: &ActiveHarnessExecutionContext,
    target: &SessionReference,
    target_slot_id: Option<&str>,
) -> Result<bool, String> {
    let (current, current_slot_id) = {
        let lock = execution_ctx
            .lock()
            .map_err(|_| "execution context mutex poisoned".to_string())?;
        (lock.session_id.clone(), lock.runtime_slot_id.clone())
    };
    let Some(current) = current else {
        return Ok(false);
    };
    let current_ref = resolve_session_reference(execution_ctx, Some(current))?;
    Ok(current_ref.public_id == target.public_id
        && current_ref.store_selector == target.store_selector
        && match target_slot_id {
            Some(slot_id) => current_slot_id.as_deref() == Some(slot_id),
            None => true,
        })
}

pub(super) async fn lookup_session_store(
    store_manager: &Arc<StoreManager>,
    execution_ctx: &ActiveHarnessExecutionContext,
    requested: Option<String>,
) -> Result<SessionStoreLookup, String> {
    let session_ref = resolve_session_reference(execution_ctx, requested)?;
    let selector = session_ref
        .store_selector
        .clone()
        .ok_or_else(|| "Session reference store could not be resolved".to_string())?;
    let store = store_manager
        .open(&selector)
        .await
        .map_err(|err| err.to_string())?;
    let uuid = uuid::Uuid::parse_str(&session_ref.public_id).map_err(|err| err.to_string())?;
    let row = store
        .get_session_row_by_public_id(uuid)
        .await
        .map_err(|err| err.to_string())?;
    Ok(SessionStoreLookup {
        session_ref,
        selector,
        store,
        row,
    })
}

pub(super) async fn require_session_store(
    store_manager: &Arc<StoreManager>,
    execution_ctx: &ActiveHarnessExecutionContext,
    requested: Option<String>,
) -> Result<ResolvedSessionStore, String> {
    lookup_session_store(store_manager, execution_ctx, requested)
        .await?
        .require_row()
}
