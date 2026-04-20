use serde::Serialize;
use turin_channel_core::ChannelAdapterManifest;
use turin_daemon_protocol::SessionSearchHitKind;

#[derive(Debug, Clone, Serialize)]
pub struct AgentDetail {
    pub id: String,
    pub directory: String,
    pub enabled: bool,
    pub provider: String,
    pub model: String,
    pub system_prompt: Option<String>,
    pub mode: Option<String>,
    pub harness: Option<String>,
    pub idle_grace_secs: Option<u64>,
    pub has_local_harness: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct HarnessDetail {
    pub harness_id: String,
    pub directory: String,
    pub bound_agents: Vec<String>,
    pub watched_roots: Vec<String>,
    pub loaded_scripts: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChannelDetail {
    pub id: String,
    pub directory: String,
    pub enabled: bool,
    pub kind: String,
    pub agent_id: String,
    pub idle_ttl_secs: Option<u64>,
    pub settings: serde_json::Value,
    pub adapter: Option<ChannelAdapterManifest>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionSummary {
    pub internal_id: i64,
    pub session_id: String,
    pub agent_id: String,
    pub metadata: Option<serde_json::Value>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionEventDetail {
    pub id: i64,
    pub event_type: String,
    pub payload: serde_json::Value,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionMessageDetail {
    pub id: i64,
    pub turn_index: u32,
    pub role: String,
    pub content: serde_json::Value,
    pub token_count: Option<u64>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionToolExecutionDetail {
    pub id: i64,
    pub turn_index: u32,
    pub tool_call_id: String,
    pub tool_name: String,
    pub args: serde_json::Value,
    pub output: Option<serde_json::Value>,
    pub is_error: bool,
    pub duration_ms: Option<u64>,
    pub verdict: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionDetail {
    pub session: SessionSummary,
    pub branches: Vec<SessionBranchDetail>,
    pub events: Vec<SessionEventDetail>,
    pub messages: Vec<SessionMessageDetail>,
    pub tool_executions: Vec<SessionToolExecutionDetail>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionBranchDetail {
    pub branch_id: String,
    pub name: String,
    pub head_turn_index: Option<u32>,
    pub source_turn_id: Option<i64>,
    pub origin_kind: String,
    pub origin_task_id: Option<String>,
    pub origin_execution_id: Option<String>,
    pub origin_metadata: Option<serde_json::Value>,
    pub active: bool,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionSearchHit {
    pub kind: SessionSearchHitKind,
    pub score: i64,
    pub session_id: String,
    pub agent_id: String,
    pub title: Option<String>,
    pub created_at: String,
    pub turn_index: Option<u32>,
    pub role: Option<String>,
    pub tool_name: Option<String>,
    pub event_type: Option<String>,
    pub summary: String,
    pub snippet: String,
}
