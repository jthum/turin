use serde::{Deserialize, Serialize};

use crate::manifest::{ChannelConfigTarget, ChannelFieldVisibilityRule};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelAuthFlowKind {
    OauthDeviceCode,
    QrPairing,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelAuthFlow {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: ChannelAuthFlowKind,
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub help: Option<String>,
    #[serde(default)]
    pub hint: Option<String>,
    #[serde(default)]
    pub advanced: bool,
    #[serde(default)]
    pub visible_if: Option<ChannelFieldVisibilityRule>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelAuthFlowResolvedValue {
    pub target: ChannelConfigTarget,
    pub value: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ChannelAuthFlowDisplay {
    #[serde(default)]
    pub message: Option<String>,
    #[serde(default)]
    pub verification_uri: Option<String>,
    #[serde(default)]
    pub verification_uri_complete: Option<String>,
    #[serde(default)]
    pub user_code: Option<String>,
    #[serde(default)]
    pub qr_text: Option<String>,
    #[serde(default)]
    pub pairing_code: Option<String>,
    #[serde(default)]
    pub expires_in_seconds: Option<u64>,
    #[serde(default)]
    pub poll_interval_seconds: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelAuthFlowStartRequest {
    pub flow_id: String,
    #[serde(default)]
    pub current_settings: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelAuthFlowStartResponse {
    pub session: serde_json::Value,
    pub display: ChannelAuthFlowDisplay,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelAuthFlowPollRequest {
    pub flow_id: String,
    pub session: serde_json::Value,
    #[serde(default)]
    pub current_settings: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ChannelAuthFlowPollResponse {
    Pending {
        display: ChannelAuthFlowDisplay,
    },
    Complete {
        #[serde(default)]
        values: Vec<ChannelAuthFlowResolvedValue>,
        #[serde(default)]
        message: Option<String>,
    },
    Failed {
        message: String,
    },
}
