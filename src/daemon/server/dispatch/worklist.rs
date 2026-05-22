use crate::daemon::protocol::{
    ErrorCode, ResponseEnvelope, WorkItemTargetParams, WorklistItemsParams, WorklistList,
    WorklistListParams, WorklistTargetParams,
};
use crate::daemon::state::WorklistItemsQuery;

use super::{DispatchContext, optional_response, serialize_response, validation_error};

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
    let result = guard
        .worklist_detail(&params.id, params.persistence.as_ref())
        .await;
    optional_response(
        id,
        result,
        "worklist detail",
        ErrorCode::WorklistNotFound,
        || format!("Worklist '{}' not found", params.id),
    )
}

pub(super) async fn items(
    id: Option<String>,
    params: WorklistItemsParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    let result = guard
        .worklist_items(WorklistItemsQuery {
            public_id: &params.id,
            persistence: params.persistence.as_ref(),
            status: params.status.as_deref(),
            parent_public_id: params.parent_id.as_deref(),
            where_filter: params.r#where.as_ref(),
            claimed_only: params.claimed_only,
            paused_only: params.paused_only,
            due_only: params.due_only,
            limit: params.limit,
        })
        .await;
    optional_response(
        id,
        result,
        "worklist items",
        ErrorCode::WorklistNotFound,
        || format!("Worklist '{}' not found", params.id),
    )
}

pub(super) async fn item_get(
    id: Option<String>,
    params: WorkItemTargetParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    let result = guard
        .work_item_detail(&params.id, params.persistence.as_ref())
        .await;
    optional_response(
        id,
        result,
        "work item detail",
        ErrorCode::WorkItemNotFound,
        || format!("Work item '{}' not found", params.id),
    )
}
