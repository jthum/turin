use crate::daemon::protocol::{
    NoParams, OpenSessionParams, ResponseEnvelope, ResumeSessionParams, SessionIdParams,
    SessionListParams, SessionSearchParams, SessionTitleParams,
};

use super::{
    DispatchContext, emit_event, not_found_error, serialize_response,
    serialize_response_with_event, validation_error,
};
use crate::daemon::protocol::ErrorCode;

pub(super) async fn list(
    id: Option<String>,
    params: SessionListParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard.list_sessions(params.limit, params.offset).await {
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
    match guard
        .search_sessions(
            params.query.as_str(),
            params
                .scope
                .unwrap_or(turin_daemon_protocol::SessionSearchScope::All),
            params.limit,
            params.offset,
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
        .open_session(&params.agent_id, params.slot_id.as_deref())
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
    params: SessionIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard.get_session(&params.session_id).await {
        Ok(Some(session)) => serialize_response(id, session, "session detail"),
        Ok(None) => not_found_error(
            id,
            ErrorCode::SessionNotFound,
            format!("Session '{}' not found", params.session_id),
        ),
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn set_title(
    id: Option<String>,
    params: SessionTitleParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard
        .set_session_title(&params.session_id, params.title.as_deref())
        .await
    {
        Ok(Some(session)) => serialize_response_with_event(
            id,
            session,
            "updated session title",
            &ctx.event_tx,
            "session.title_updated",
        ),
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
    params: SessionIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard.cancel_session(&params.session_id).await {
        Ok(result) => {
            emit_event(&ctx.event_tx, "session.cancel_requested", result.clone());
            ResponseEnvelope::ok(id, result)
        }
        Err(err) => validation_error(id, err),
    }
}

pub(super) async fn kill(
    id: Option<String>,
    params: SessionIdParams,
    ctx: &DispatchContext,
) -> ResponseEnvelope {
    let guard = ctx.state.read().await;
    match guard.kill_session(&params.session_id).await {
        Ok(result) => {
            emit_event(&ctx.event_tx, "session.killed", result.clone());
            ResponseEnvelope::ok(id, result)
        }
        Err(err) => validation_error(id, err),
    }
}
