use serde::Serialize;
use turin_daemon_protocol::{SessionSearchHitKind, UiIntentMessage};

use crate::kernel::event::InferenceRequestMetrics;

#[derive(Debug, Clone, Serialize)]
pub struct AgentDetail {
    pub id: String,
    pub directory: String,
    pub enabled: bool,
    pub provider: String,
    pub model: String,
    pub system_prompt: Option<String>,
    pub harness: Option<String>,
    pub idle_timeout_seconds: Option<u64>,
    pub has_local_harness: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct HarnessDetail {
    pub harness_id: String,
    pub directory: String,
    pub bound_agents: Vec<String>,
    pub watched_roots: Vec<String>,
    pub loaded_scripts: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ui_intents: Vec<UiIntentMessage>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionSummary {
    pub internal_id: i64,
    pub session_id: String,
    pub agent_id: String,
    pub origin_id: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub parent_internal_id: Option<i64>,
    pub root_internal_id: Option<i64>,
    pub origin_turn_id: Option<i64>,
    pub relation_kind: Option<String>,
    pub thread_key: Option<String>,
    pub visibility: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionFamilyMember {
    pub session: SessionSummary,
    pub depth: usize,
    pub direct_children: usize,
    pub live_slots: Vec<String>,
    pub active_tasks: usize,
    pub queued_tasks: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionFamilyDetail {
    pub requested_session_id: String,
    pub root_session_id: String,
    pub requested_depth: usize,
    pub direct_children: usize,
    pub descendants: usize,
    pub family_size: usize,
    pub members: Vec<SessionFamilyMember>,
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
    pub turn_id: i64,
    pub turn_index: u32,
    pub role: String,
    pub content: serde_json::Value,
    pub token_count: Option<u64>,
    pub estimated_token_count: Option<u32>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub efficiency: Option<SessionEfficiencyDetail>,
    pub execution: SessionExecutionDetail,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message_window: Option<SessionMessageWindow>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_window: Option<SessionEventWindow>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionExecutionDetail {
    pub tasks: Vec<SessionTaskExecutionDetail>,
    pub plans: Vec<SessionPlanExecutionDetail>,
    pub event_limit: usize,
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionTaskExecutionDetail {
    pub task_id: String,
    pub trace_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    pub agent_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub prompt: String,
    pub status: String,
    pub queue_depth: usize,
    pub task_turn_count: u32,
    pub execution: SessionExecutionContextDetail,
    pub turns: Vec<SessionTaskTurnDetail>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch_outcome: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub started_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionExecutionContextDetail {
    pub execution_id: String,
    pub context_target: serde_json::Value,
    pub visibility: String,
    pub durability: String,
    pub write_policy: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionTaskTurnDetail {
    pub turn_index: u32,
    pub task_turn_index: u32,
    pub has_tool_calls: Option<bool>,
    pub started_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionPlanExecutionDetail {
    pub plan_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub status: String,
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub started_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionEfficiencyDetail {
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_cache_read_input_tokens: u64,
    pub total_cache_creation_input_tokens: u64,
    pub total_request_count: usize,
    pub turns: Vec<SessionTurnEfficiencyDetail>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latest_compaction: Option<SessionCompactionDetail>,
    pub provider_cache_metrics_available: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionTurnEfficiencyDetail {
    pub turn_index: u32,
    pub requests: Vec<SessionRequestEfficiencyDetail>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionRequestEfficiencyDetail {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metrics: Option<InferenceRequestMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionCompactionDetail {
    pub covered_through_turn_id: i64,
    pub covered_through_turn_index: u32,
    pub generated_at_turn_index: u32,
    pub provider: String,
    pub model: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionMessageWindow {
    pub offset: usize,
    pub total: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionEventWindow {
    pub offset: usize,
    pub total: usize,
    pub has_more: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionBranchDetail {
    pub branch_id: String,
    pub name: String,
    pub head_turn_id: Option<i64>,
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
pub struct SessionGraphDetail {
    pub session: SessionSummary,
    pub turns: Vec<SessionGraphTurnDetail>,
    pub branches: Vec<SessionBranchDetail>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionGraphTurnDetail {
    pub turn_id: i64,
    pub turn_public_id: String,
    pub parent_turn_id: Option<i64>,
    pub turn_index: u32,
    pub message_count: usize,
    pub tool_execution_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub preview: Option<String>,
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
