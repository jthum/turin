use anyhow::Result;
use serde_json::Value;
use turin_daemon_protocol::{
    ChannelAccessParams, ChannelAccessRoomParams, DaemonRequest, EntityIdParams,
    UpdateChannelParams,
};

use crate::client::ControlClient;
use crate::models::{AgentDetail, ChannelAccessState, ChannelDetail, ChannelRuntime};

impl ControlClient {
    pub async fn get_agent(&self, agent_id: &str) -> Result<AgentDetail> {
        self.request_ok(
            None,
            DaemonRequest::AgentGet(EntityIdParams {
                id: agent_id.to_string(),
            }),
        )
        .await
    }

    pub async fn get_channel(&self, channel_id: &str) -> Result<ChannelDetail> {
        self.request_ok(
            None,
            DaemonRequest::ChannelGet(EntityIdParams {
                id: channel_id.to_string(),
            }),
        )
        .await
    }

    pub async fn update_channel_settings(
        &self,
        channel_id: &str,
        settings: Value,
    ) -> Result<ChannelDetail> {
        self.request_ok(
            None,
            DaemonRequest::ChannelUpdate(UpdateChannelParams {
                id: channel_id.to_string(),
                kind: None,
                agent_id: None,
                idle_timeout_seconds: None,
                settings: Some(settings),
            }),
        )
        .await
    }

    pub async fn channel_status(&self, channel_id: &str) -> Result<ChannelRuntime> {
        self.request_ok(
            None,
            DaemonRequest::ChannelStatus(EntityIdParams {
                id: channel_id.to_string(),
            }),
        )
        .await
    }

    pub async fn channel_access(&self, channel_id: &str) -> Result<ChannelAccessState> {
        self.request_ok(
            None,
            DaemonRequest::ChannelAccessGet(ChannelAccessParams {
                id: channel_id.to_string(),
            }),
        )
        .await
    }

    pub async fn approve_channel_room(
        &self,
        channel_id: &str,
        workspace_id: &str,
        room_id: Option<&str>,
        thread_id: &str,
    ) -> Result<ChannelAccessState> {
        self.request_ok(
            None,
            DaemonRequest::ChannelAccessApprove(ChannelAccessRoomParams {
                id: channel_id.to_string(),
                workspace_id: workspace_id.to_string(),
                room_id: room_id.map(str::to_string),
                thread_id: thread_id.to_string(),
            }),
        )
        .await
    }

    pub async fn reject_channel_room(
        &self,
        channel_id: &str,
        workspace_id: &str,
        room_id: Option<&str>,
        thread_id: &str,
    ) -> Result<ChannelAccessState> {
        self.request_ok(
            None,
            DaemonRequest::ChannelAccessReject(ChannelAccessRoomParams {
                id: channel_id.to_string(),
                workspace_id: workspace_id.to_string(),
                room_id: room_id.map(str::to_string),
                thread_id: thread_id.to_string(),
            }),
        )
        .await
    }

    pub async fn revoke_channel_room(
        &self,
        channel_id: &str,
        workspace_id: &str,
        room_id: Option<&str>,
        thread_id: &str,
    ) -> Result<ChannelAccessState> {
        self.request_ok(
            None,
            DaemonRequest::ChannelAccessRevoke(ChannelAccessRoomParams {
                id: channel_id.to_string(),
                workspace_id: workspace_id.to_string(),
                room_id: room_id.map(str::to_string),
                thread_id: thread_id.to_string(),
            }),
        )
        .await
    }
}
