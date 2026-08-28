use crate::daemon::protocol::{
    LiveSessionTargetParams, NoParams, OpenSessionParams, ResponseEnvelope, ResumeSessionParams,
    SessionBranchCheckoutParams, SessionBranchCreateParams, SessionBranchSiblingsParams,
    SessionGetParams, SessionIdParams, SessionListParams, SessionSearchParams, SessionTitleParams,
};
use crate::daemon::state::{
    DEFAULT_SESSION_EVENT_LIMIT, SessionEventProjection, SessionProjectionRequest,
    session_store_selector_from_filters,
};

use super::{
    DispatchContext, emit_event, not_found_error, optional_response, optional_response_with_event,
    resource_busy_error, serialize_response_with_event, validation_error,
};
use crate::daemon::protocol::ErrorCode;
use crate::daemon::state::SessionDeleteBusy;

pub(super) async fn list(
    id: Option<String>,
    params: SessionListParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    if let Some(parent_session_id) = params.parent_session_id.as_deref() {
        if params.store.is_some() || params.path.is_some() || params.origin_id.is_some() {
            return validation_error(
                id,
                anyhow::anyhow!(
                    "'store', 'path', and 'origin_id' cannot be combined with 'parent_session_id'"
                ),
            );
        }
        return match guard
            .list_linked_sessions(parent_session_id, params.limit, params.offset)
            .await
        {
            Ok(Some(sessions)) => {
                ResponseEnvelope::ok(id, serde_json::json!({ "sessions": sessions }))
            }
            Ok(None) => not_found_error(
                id,
                ErrorCode::SessionNotFound,
                format!("Session '{parent_session_id}' not found"),
            ),
            Err(err) => super::internal_error(id, err),
        };
    }
    let store_selector = match session_store_selector_from_filters(
        params.store.as_deref(),
        params.path.as_deref(),
    ) {
        Ok(selector) => selector,
        Err(err) => return validation_error(id, err),
    };
    match guard
        .list_sessions(
            params.limit,
            params.offset,
            store_selector,
            params.origin_id.as_deref(),
        )
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
            params.origin_id.as_deref(),
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

    let projection = match session_projection_request(&params) {
        Ok(projection) => projection,
        Err(error) => return validation_error(id, error),
    };
    let guard = ctx.state.read().await;
    let result = perf_session_scope!(
        &params.session_id,
        guard.get_session_projection(&params.session_id, projection)
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

fn session_projection_request(
    params: &SessionGetParams,
) -> anyhow::Result<SessionProjectionRequest> {
    let has_event_options = params.event_limit.is_some()
        || params.event_offset.is_some()
        || params.event_types.is_some();
    let events = match params.include_events {
        Some(false) => {
            anyhow::ensure!(
                !has_event_options,
                "event paging and filters require events to be included"
            );
            SessionEventProjection::None
        }
        Some(true) => match params.event_limit {
            Some(limit) => SessionEventProjection::Window {
                limit,
                offset: params.event_offset,
                event_types: params.event_types.clone(),
            },
            None => {
                anyhow::ensure!(
                    params.event_offset.is_none(),
                    "event_offset requires event_limit when all events are explicitly requested"
                );
                SessionEventProjection::All {
                    event_types: params.event_types.clone(),
                }
            }
        },
        None => SessionEventProjection::Window {
            limit: params.event_limit.unwrap_or(DEFAULT_SESSION_EVENT_LIMIT),
            offset: params.event_offset,
            event_types: params.event_types.clone(),
        },
    };
    Ok(SessionProjectionRequest {
        target_turn_id: params.target_turn_id,
        message_limit: params.message_limit,
        message_offset: params.message_offset,
        events,
        include_efficiency: params.include_efficiency.unwrap_or(false),
    })
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

pub(super) async fn family_get(
    id: Option<String>,
    params: SessionIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    optional_response(
        id,
        guard.get_session_family(&params.session_id).await,
        "session family",
        ErrorCode::SessionNotFound,
        || format!("Session '{}' not found", params.session_id),
    )
}

pub(super) async fn archive(
    id: Option<String>,
    params: SessionIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard.archive_linked_session(&params.session_id).await {
        Ok(Some(archived)) => ResponseEnvelope::ok(
            id,
            serde_json::json!({ "session_id": params.session_id, "archived": archived }),
        ),
        Ok(None) => not_found_error(
            id,
            ErrorCode::SessionNotFound,
            format!("Session '{}' not found", params.session_id),
        ),
        Err(error) => validation_error(id, error),
    }
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

pub(super) async fn delete(
    id: Option<String>,
    params: SessionIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.write().await;
    match guard.delete_session(&params.session_id).await {
        Ok(true) => {
            let result = serde_json::json!({ "deleted": params.session_id });
            emit_event(&ctx.event_tx, "session.deleted", result.clone());
            ResponseEnvelope::ok(id, result)
        }
        Ok(false) => not_found_error(
            id,
            ErrorCode::SessionNotFound,
            format!("Session '{}' not found", params.session_id),
        ),
        Err(err) if err.downcast_ref::<SessionDeleteBusy>().is_some() => {
            resource_busy_error(id, err)
        }
        Err(err) => validation_error(id, err),
    }
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
    let result = if params.recursive {
        guard.cancel_session_family(&params.session_id).await
    } else {
        guard
            .cancel_session(&params.session_id, params.slot_id.as_deref())
            .await
    };
    match result {
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
    if params.recursive {
        return validation_error(
            id,
            anyhow::anyhow!(
                "Recursive force-kill is unsafe for pooled runtime lanes; use recursive cancellation"
            ),
        );
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> SessionGetParams {
        SessionGetParams {
            session_id: "session".to_string(),
            target_turn_id: None,
            message_limit: None,
            message_offset: None,
            include_events: None,
            event_limit: None,
            event_offset: None,
            event_types: None,
            include_efficiency: None,
        }
    }

    #[test]
    fn omitted_event_options_use_bounded_default() {
        let request = session_projection_request(&params()).unwrap();
        assert!(matches!(
            request.events,
            SessionEventProjection::Window {
                limit: DEFAULT_SESSION_EVENT_LIMIT,
                offset: None,
                event_types: None,
            }
        ));
    }

    #[test]
    fn explicit_event_inclusion_without_limit_requests_all() {
        let mut params = params();
        params.include_events = Some(true);
        let request = session_projection_request(&params).unwrap();
        assert!(matches!(
            request.events,
            SessionEventProjection::All { event_types: None }
        ));
    }

    #[test]
    fn excluded_events_reject_paging_options() {
        let mut params = params();
        params.include_events = Some(false);
        params.event_limit = Some(10);
        assert!(session_projection_request(&params).is_err());
    }
}
