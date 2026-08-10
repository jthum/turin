use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenSessionParams {
    pub agent_id: String,
    #[serde(default)]
    pub slot_id: Option<String>,
    #[serde(default)]
    pub channel_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResumeSessionParams {
    pub session_id: String,
    #[serde(default)]
    pub slot_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SessionIdParams {
    pub session_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SessionGetParams {
    pub session_id: String,
    #[serde(default)]
    pub message_limit: Option<usize>,
    #[serde(default)]
    pub message_offset: Option<usize>,
    #[serde(default)]
    pub include_events: Option<bool>,
    #[serde(default)]
    pub include_efficiency: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LiveSessionTargetParams {
    pub session_id: String,
    #[serde(default)]
    pub slot_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SessionListParams {
    #[serde(default = "crate::default_session_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
    #[serde(default)]
    pub store: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SessionSearchScope {
    All,
    Sessions,
    Messages,
    ToolExecutions,
    Events,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SessionSearchHitKind {
    Session,
    Message,
    ToolExecution,
    Event,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SessionSearchParams {
    pub query: String,
    #[serde(default)]
    pub scope: Option<SessionSearchScope>,
    #[serde(default = "crate::default_search_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
    #[serde(default)]
    pub store: Option<String>,
    #[serde(default)]
    pub path: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SessionTitleParams {
    pub session_id: String,
    #[serde(default)]
    pub title: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SessionBranchCreateParams {
    pub session_id: String,
    pub name: String,
    #[serde(default)]
    pub slot_id: Option<String>,
    #[serde(default)]
    pub from_turn_index: Option<u32>,
    #[serde(default)]
    pub activate: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SessionBranchCheckoutParams {
    pub session_id: String,
    pub branch: String,
    #[serde(default)]
    pub slot_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SessionBranchSiblingsParams {
    pub session_id: String,
    pub source_turn_id: i64,
}
