use crate::daemon::protocol::{
    ErrorCode, ResponseEnvelope, WorklistItemsParams, WorklistList, WorklistListParams,
    WorklistTargetParams,
};

use super::{DispatchContext, not_found_error, serialize_response, validation_error};

pub(super) async fn list(
    id: Option<String>,
    params: WorklistListParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard
        .list_worklists(
            params.persistence.as_ref(),
            params.name.as_deref(),
            params.scope.as_deref(),
        )
        .await
    {
        Ok(worklists) => serialize_response(id, WorklistList { worklists }, "worklist list"),
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn get(
    id: Option<String>,
    params: WorklistTargetParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard
        .worklist_detail(&params.id, params.persistence.as_ref())
        .await
    {
        Ok(Some(worklist)) => serialize_response(id, worklist, "worklist detail"),
        Ok(None) => not_found_error(
            id,
            ErrorCode::WorklistNotFound,
            format!("Worklist '{}' not found", params.id),
        ),
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn items(
    id: Option<String>,
    params: WorklistItemsParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard
        .worklist_items(
            &params.id,
            params.persistence.as_ref(),
            params.status.as_deref(),
            params.parent_id.as_deref(),
            params.claimed_only,
            params.limit,
        )
        .await
    {
        Ok(Some(items)) => serialize_response(id, items, "worklist items"),
        Ok(None) => not_found_error(
            id,
            ErrorCode::WorklistNotFound,
            format!("Worklist '{}' not found", params.id),
        ),
        Err(err) => validation_error(id, err),
    }
}
