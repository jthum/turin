use turin_daemon_protocol::{
    ErrorCode, NoParams, ResponseEnvelope, ToolAuthorizationListResult,
    ToolAuthorizationResolveParams,
};

use super::{DispatchContext, serialize_response, serialize_response_with_event};

pub(super) async fn list(
    id: Option<String>,
    _params: NoParams,
    context: &DispatchContext,
) -> ResponseEnvelope {
    let requests = context.state.read().await.list_tool_authorizations().await;
    serialize_response(
        id,
        ToolAuthorizationListResult { requests },
        "tool authorization list",
    )
}

pub(super) async fn resolve(
    id: Option<String>,
    params: ToolAuthorizationResolveParams,
    context: &DispatchContext,
) -> ResponseEnvelope {
    let request_id = params.request_id.clone();
    let result = context
        .state
        .read()
        .await
        .resolve_tool_authorization(params)
        .await;
    match result {
        Some(result) => serialize_response_with_event(
            id,
            result,
            "tool authorization resolution",
            &context.event_tx,
            "tool_authorization.resolved",
        ),
        None => super::not_found_error(
            id,
            ErrorCode::ToolAuthorizationNotFound,
            format!("Tool authorization request '{}' is not pending", request_id),
        ),
    }
}
