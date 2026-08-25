use anyhow::Result;
use turin_daemon_protocol::{
    DaemonRequest, NoParams, ToolAuthorizationListResult, ToolAuthorizationRequestDetail,
    ToolAuthorizationResolution, ToolAuthorizationResolveParams, ToolAuthorizationResolveResult,
};

use crate::client::Client;

impl Client {
    pub async fn list_tool_authorizations(&self) -> Result<Vec<ToolAuthorizationRequestDetail>> {
        let result: ToolAuthorizationListResult = self
            .request_ok(
                None,
                DaemonRequest::ToolAuthorizationList(NoParams::default()),
            )
            .await?;
        Ok(result.requests)
    }

    pub async fn approve_tool_authorization(
        &self,
        request_id: impl Into<String>,
    ) -> Result<ToolAuthorizationResolveResult> {
        self.resolve_tool_authorization(request_id, ToolAuthorizationResolution::Approve, None)
            .await
    }

    pub async fn deny_tool_authorization(
        &self,
        request_id: impl Into<String>,
        reason: Option<String>,
    ) -> Result<ToolAuthorizationResolveResult> {
        self.resolve_tool_authorization(request_id, ToolAuthorizationResolution::Deny, reason)
            .await
    }

    async fn resolve_tool_authorization(
        &self,
        request_id: impl Into<String>,
        decision: ToolAuthorizationResolution,
        reason: Option<String>,
    ) -> Result<ToolAuthorizationResolveResult> {
        self.request_ok(
            None,
            DaemonRequest::ToolAuthorizationResolve(ToolAuthorizationResolveParams {
                request_id: request_id.into(),
                decision,
                reason,
            }),
        )
        .await
    }
}
