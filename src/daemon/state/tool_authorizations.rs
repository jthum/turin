use turin_daemon_protocol::{
    ToolAuthorizationRequestDetail, ToolAuthorizationResolution, ToolAuthorizationResolveParams,
    ToolAuthorizationResolveResult,
};

use crate::kernel::tool_authorization::ToolAuthorizationDecision;
use crate::kernel::tool_authorization::ToolAuthorizationRequest;

use super::DaemonState;

impl From<ToolAuthorizationRequest> for ToolAuthorizationRequestDetail {
    fn from(request: ToolAuthorizationRequest) -> Self {
        Self {
            id: request.id,
            agent_id: request.identity.agent_id().to_string(),
            session_id: request.identity.session_id().to_string(),
            slot_id: request.runtime_slot_id,
            tool_call_id: request.tool_call_id,
            tool_name: request.tool_name,
            arguments: request.arguments,
            reason: request.reason,
            requested_at_unix_ms: request.requested_at_unix_ms,
        }
    }
}

impl DaemonState {
    pub async fn list_tool_authorizations(&self) -> Vec<ToolAuthorizationRequestDetail> {
        self.tool_authorization_broker
            .list_pending()
            .await
            .into_iter()
            .map(Into::into)
            .collect()
    }

    pub async fn resolve_tool_authorization(
        &self,
        params: ToolAuthorizationResolveParams,
    ) -> Option<ToolAuthorizationResolveResult> {
        let decision = match params.decision {
            ToolAuthorizationResolution::Approve => ToolAuthorizationDecision::Approve,
            ToolAuthorizationResolution::Deny => ToolAuthorizationDecision::deny(params.reason),
        };
        self.tool_authorization_broker
            .resolve(&params.request_id, decision)
            .await
            .then_some(ToolAuthorizationResolveResult {
                request_id: params.request_id,
                decision: params.decision,
            })
    }
}
