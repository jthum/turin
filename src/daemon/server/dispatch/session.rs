use crate::daemon::protocol::{
    LiveSessionTargetParams, NoParams, OpenSessionParams, ResponseEnvelope, ResumeSessionParams,
    SessionBranchCheckoutParams, SessionBranchCreateParams, SessionBranchSiblingsParams,
    SessionGetParams, SessionIdParams, SessionListParams, SessionSearchParams, SessionTitleParams,
};
use crate::daemon::state::session_store_selector_from_filters;

use super::{
    DispatchContext, emit_event, not_found_error, optional_response, optional_response_with_event,
    serialize_response_with_event, validation_error,
};
use crate::daemon::protocol::ErrorCode;

pub(super) async fn list(
    id: Option<String>,
    params: SessionListParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    let store_selector = match session_store_selector_from_filters(
        params.store.as_deref(),
        params.path.as_deref(),
    ) {
        Ok(selector) => selector,
        Err(err) => return validation_error(id, err),
    };
    match guard
        .list_sessions(params.limit, params.offset, store_selector)
        .await
    {
        Ok(sessions) => ResponseEnvelope::ok(id, serde_json::json!({ "sessions": sessions })),
        Err(err) => super::internal_error(id, err),
    }
}

pub(super) async fn list_live(
    id: Option<String>,
    _params: NoParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    ResponseEnvelope::ok(
        id,
        serde_json::json!({ "sessions": guard.list_live_sessions().await }),
    )
}

pub(super) async fn search(
    id: Option<String>,
    params: SessionSearchParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    let store_selector = match session_store_selector_from_filters(
        params.store.as_deref(),
        params.path.as_deref(),
    ) {
        Ok(selector) => selector,
        Err(err) => return validation_error(id, err),
    };
    match guard
        .search_sessions(
            params.query.as_str(),
            params
                .scope
                .unwrap_or(turin_daemon_protocol::SessionSearchScope::All),
            params.limit,
            params.offset,
            store_selector,
        )
        .await
    {
        Ok(hits) => ResponseEnvelope::ok(id, serde_json::json!({ "hits": hits })),
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn open(
    id: Option<String>,
    params: OpenSessionParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard
        .open_session(
            &params.agent_id,
            params.slot_id.as_deref(),
            params.channel_id.as_deref(),
        )
        .await
    {
        Ok(session) => serialize_response_with_event(
            id,
            session,
            "opened session",
            &ctx.event_tx,
            "session.opened",
        ),
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn resume(
    id: Option<String>,
    params: ResumeSessionParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard
        .resume_session(&params.session_id, params.slot_id.as_deref())
        .await
    {
        Ok(session) => serialize_response_with_event(
            id,
            session,
            "resumed session",
            &ctx.event_tx,
            "session.resumed",
        ),
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn get(
    id: Option<String>,
    params: SessionGetParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    use crate::perf_diagnostics::perf_session_scope;

    let guard = ctx.state.read().await;
    let result = perf_session_scope!(
        &params.session_id,
        guard.get_session_projection(
            &params.session_id,
            params.message_limit,
            params.message_offset,
            params.include_events.unwrap_or(true),
            params.include_efficiency.unwrap_or(false),
        )
    )
    .await;
    optional_response(
        id,
        result,
        "session detail",
        ErrorCode::SessionNotFound,
        || format!("Session '{}' not found", params.session_id),
    )
}

pub(super) async fn graph_get(
    id: Option<String>,
    params: SessionIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    optional_response(
        id,
        guard.get_session_graph(&params.session_id).await,
        "session graph",
        ErrorCode::SessionNotFound,
        || format!("Session '{}' not found", params.session_id),
    )
}

pub(super) async fn set_title(
    id: Option<String>,
    params: SessionTitleParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    let result = guard
        .set_session_title(&params.session_id, params.title.as_deref())
        .await;
    optional_response_with_event(
        id,
        result,
        "updated session title",
        &ctx.event_tx,
        "session.title_updated",
        ErrorCode::SessionNotFound,
        || format!("Session '{}' not found", params.session_id),
    )
}

pub(super) async fn branch_list(
    id: Option<String>,
    params: SessionIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard.list_session_branches(&params.session_id).await {
        Ok(Some(branches)) => ResponseEnvelope::ok(id, serde_json::json!({ "branches": branches })),
        Ok(None) => not_found_error(
            id,
            ErrorCode::SessionNotFound,
            format!("Session '{}' not found", params.session_id),
        ),
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn branch_create(
    id: Option<String>,
    params: SessionBranchCreateParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    let result = match (params.from_turn_index, params.from_turn_id) {
        (Some(_), Some(_)) => {
            return validation_error(
                id,
                anyhow::anyhow!("Only one of from_turn_index or from_turn_id may be supplied"),
            );
        }
        (_, Some(turn_id)) => {
            guard
                .create_session_branch_from_turn_id(
                    &params.session_id,
                    &params.name,
                    params.slot_id.as_deref(),
                    turn_id,
                    params.activate,
                )
                .await
        }
        (from_turn_index, None) => {
            guard
                .create_session_branch(
                    &params.session_id,
                    &params.name,
                    params.slot_id.as_deref(),
                    from_turn_index,
                    params.activate,
                )
                .await
        }
    };
    optional_response_with_event(
        id,
        result,
        "created session branch",
        &ctx.event_tx,
        "session.branch_created",
        ErrorCode::SessionNotFound,
        || format!("Session '{}' not found", params.session_id),
    )
}

pub(super) async fn branch_checkout(
    id: Option<String>,
    params: SessionBranchCheckoutParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    let result = guard
        .checkout_session_branch(
            &params.session_id,
            &params.branch,
            params.slot_id.as_deref(),
        )
        .await;
    optional_response_with_event(
        id,
        result,
        "checked out session branch",
        &ctx.event_tx,
        "session.branch_checked_out",
        ErrorCode::SessionNotFound,
        || {
            format!(
                "Session '{}' or branch '{}' not found",
                params.session_id, params.branch
            )
        },
    )
}

pub(super) async fn branch_siblings(
    id: Option<String>,
    params: SessionBranchSiblingsParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard
        .list_session_branch_siblings(&params.session_id, params.source_turn_id)
        .await
    {
        Ok(Some(branches)) => ResponseEnvelope::ok(id, serde_json::json!({ "branches": branches })),
        Ok(None) => not_found_error(
            id,
            ErrorCode::SessionNotFound,
            format!("Session '{}' not found", params.session_id),
        ),
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn cancel(
    id: Option<String>,
    params: LiveSessionTargetParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard
        .cancel_session(&params.session_id, params.slot_id.as_deref())
        .await
    {
        Ok(result) => {
            emit_event(&ctx.event_tx, "session.cancel_requested", result.clone());
            ResponseEnvelope::ok(id, result)
        }
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn kill(
    id: Option<String>,
    params: LiveSessionTargetParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard
        .kill_session(&params.session_id, params.slot_id.as_deref())
        .await
    {
        Ok(result) => {
            emit_event(&ctx.event_tx, "session.killed", result.clone());
            ResponseEnvelope::ok(id, result)
        }
        Err(err) => validation_error(id, err),
    }
}
