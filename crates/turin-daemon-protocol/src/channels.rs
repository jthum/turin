use serde::{Deserialize, Serialize};
use serde_json::Value;
use turin_channel_core::ChannelAdapterManifest;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CreateChannelParams {
    pub id: String,
    pub kind: String,
    pub agent_id: String,
    #[serde(default)]
    pub idle_timeout_seconds: Option<u64>,
    #[serde(default = "crate::default_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub settings: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UpdateChannelParams {
    pub id: String,
    #[serde(default)]
    pub kind: Option<String>,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub idle_timeout_seconds: Option<u64>,
    #[serde(default)]
    pub settings: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChannelAccessParams {
    pub id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChannelAccessRoomParams {
    pub id: String,
    pub workspace_id: String,
    #[serde(default)]
    pub room_id: Option<String>,
    pub thread_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChannelRunnerHelloParams {
    pub channel_id: String,
    pub manifest: ChannelAdapterManifest,
    #[serde(default)]
    pub runner_binary: Option<String>,
    #[serde(default)]
    pub runner_version: Option<String>,
    #[serde(default)]
    pub pid: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChannelRunnerHeartbeatParams {
    pub channel_id: String,
}
