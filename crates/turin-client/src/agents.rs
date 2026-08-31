use anyhow::Result;
use turin_daemon_protocol::{DaemonRequest, EntityIdParams};

use crate::client::Client;
use crate::models::{AgentDetail, AgentRuntime, Issue, IssueList};

impl Client {
    pub async fn get_agent(&self, agent_id: impl Into<String>) -> Result<AgentDetail> {
        self.request_ok(
            None,
            DaemonRequest::AgentGet(EntityIdParams {
                id: agent_id.into(),
            }),
        )
        .await
    }

    pub async fn get_agent_status(&self, agent_id: impl Into<String>) -> Result<AgentRuntime> {
        self.request_ok(
            None,
            DaemonRequest::AgentStatus(EntityIdParams {
                id: agent_id.into(),
            }),
        )
        .await
    }

    pub async fn list_agent_issues(&self, agent_id: impl Into<String>) -> Result<Vec<Issue>> {
        let response: IssueList = self
            .request_ok(
                None,
                DaemonRequest::AgentIssues(EntityIdParams {
                    id: agent_id.into(),
                }),
            )
            .await?;
        Ok(response.issues)
    }

    pub async fn set_agent_enabled(
        &self,
        agent_id: impl Into<String>,
        enabled: bool,
    ) -> Result<AgentDetail> {
        let params = EntityIdParams {
            id: agent_id.into(),
        };
        self.request_ok(
            None,
            if enabled {
                DaemonRequest::AgentEnable(params)
            } else {
                DaemonRequest::AgentDisable(params)
            },
        )
        .await
    }

    pub async fn reload_agent(&self, agent_id: impl Into<String>) -> Result<AgentDetail> {
        self.request_ok(
            None,
            DaemonRequest::AgentReload(EntityIdParams {
                id: agent_id.into(),
            }),
        )
        .await
    }
}
