use serde::{Deserialize, Serialize};
use serde_json::Value;
use turin_channel_core::ChannelAdapterManifest;
use turin_daemon_protocol::{SessionSearchHitKind, UiIntentMessage};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonStatus {
    pub config_path: String,
    pub workspace_root: String,
    pub endpoint: String,
    pub registry: RegistrySnapshot,
    #[serde(default)]
    pub harnesses: Vec<HarnessRuntime>,
    #[serde(default)]
    pub agent_runtimes: Vec<AgentRuntime>,
    #[serde(default)]
    pub live_sessions: Vec<LiveSession>,
    #[serde(default)]
    pub channel_runtimes: Vec<ChannelRuntime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistrySnapshot {
    #[serde(default)]
    pub agents: Vec<AgentSummary>,
    #[serde(default)]
    pub shared_harnesses: Vec<SharedHarnessSummary>,
    #[serde(default)]
    pub channels: Vec<ChannelSummary>,
    #[serde(default)]
    pub issues: Vec<Issue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSummary {
    pub id: String,
    pub enabled: bool,
    pub provider: String,
    pub model: String,
    pub harness_ref: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedHarnessSummary {
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelSummary {
    pub id: String,
    pub enabled: bool,
    pub kind: String,
    pub agent_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessRuntime {
    pub harness_id: String,
    #[serde(default)]
    pub bound_agents: Vec<String>,
    #[serde(default)]
    pub watched_roots: Vec<String>,
    #[serde(default)]
    pub loaded_scripts: Vec<String>,
    #[serde(default)]
    pub ui_intents: Vec<UiIntentMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct HarnessRuntimeList {
    #[serde(default)]
    pub(crate) harnesses: Vec<HarnessRuntime>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessDetail {
    pub harness_id: String,
    pub directory: String,
    #[serde(default)]
    pub bound_agents: Vec<String>,
    #[serde(default)]
    pub watched_roots: Vec<String>,
    #[serde(default)]
    pub loaded_scripts: Vec<String>,
    #[serde(default)]
    pub ui_intents: Vec<UiIntentMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceContextStatus {
    pub id: String,
    pub provider: String,
    pub model: String,
    pub is_default: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRuntime {
    pub agent_id: String,
    #[serde(default)]
    pub provider: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub harness_id: String,
    #[serde(default)]
    pub inference_contexts: Vec<InferenceContextStatus>,
    pub running: bool,
    pub active_tasks: usize,
    pub queued_tasks: usize,
    pub awaiting_results: usize,
    pub current_session_id: Option<String>,
    pub current_request_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Issue {
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IssueList {
    #[serde(default)]
    pub issues: Vec<Issue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelDetail {
    pub id: String,
    pub directory: String,
    pub enabled: bool,
    pub kind: String,
    pub agent_id: String,
    pub idle_timeout_seconds: Option<u64>,
    pub settings: Value,
    #[serde(default)]
    pub adapter: Option<ChannelAdapterManifest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelRunnerHandshake {
    pub display_name: String,
    pub protocol_version: u32,
    pub runner_binary: Option<String>,
    pub runner_version: Option<String>,
    pub pid: Option<u32>,
    pub last_handshake_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelRuntime {
    pub id: String,
    pub kind: String,
    pub agent_id: String,
    pub directory: String,
    pub state: String,
    pub last_error: Option<String>,
    pub last_error_code: Option<String>,
    pub start_count: u64,
    pub restart_count: u64,
    pub failure_count: u64,
    pub last_transition_unix_ms: u64,
    pub last_started_unix_ms: Option<u64>,
    pub last_stopped_unix_ms: Option<u64>,
    #[serde(default)]
    pub handshake: Option<ChannelRunnerHandshake>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelAccessRoom {
    pub channel: String,
    pub workspace_id: String,
    pub room_id: Option<String>,
    pub thread_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovedChannelRoom {
    pub room: ChannelAccessRoom,
    pub approved_at_unix_seconds: u64,
    pub approved_by_user_id: Option<String>,
    pub approved_by_username: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingChannelRoom {
    pub room: ChannelAccessRoom,
    pub first_seen_unix_seconds: u64,
    pub last_seen_unix_seconds: u64,
    pub sample_user_id: Option<String>,
    pub sample_username: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChannelAccessState {
    #[serde(default)]
    pub approved_rooms: Vec<ApprovedChannelRoom>,
    #[serde(default)]
    pub pending_rooms: Vec<PendingChannelRoom>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskStatus {
    pub request_id: String,
    pub agent_id: String,
    pub slot_id: String,
    pub trace_id: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub prompt_preview: String,
    pub state: String,
    pub runtime_task_id: Option<String>,
    pub execution: LiveExecution,
    pub status: Option<String>,
    pub task_turn_count: Option<u32>,
    pub branch_outcome: Option<Value>,
    pub promotion_candidate: Option<TaskPromotionCandidate>,
    pub promoted_branch: Option<SessionBranchDetail>,
    pub output: Option<String>,
    #[serde(default)]
    pub assistant_content: Option<Vec<turin_types::TaskInputContent>>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskPromotionCandidate {
    pub session_id: String,
    pub source_turn_id: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TaskList {
    pub(crate) tasks: Vec<TaskStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub internal_id: i64,
    pub session_id: String,
    pub agent_id: String,
    pub metadata: Option<Value>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SessionList {
    pub(crate) sessions: Vec<SessionSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveExecution {
    pub execution_id: String,
    pub context_target: Value,
    pub visibility: String,
    pub durability: String,
    pub write_policy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveSession {
    pub agent_id: String,
    pub slot_id: String,
    pub session_id: String,
    pub running: bool,
    pub active_tasks: usize,
    pub queued_tasks: usize,
    pub current_request_id: Option<String>,
    pub execution: LiveExecution,
    pub conflict_policy: String,
    #[serde(default)]
    pub history: Option<LiveSessionHistory>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveSessionHistory {
    pub len: usize,
    pub message_offset: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct LiveSessionList {
    pub(crate) sessions: Vec<LiveSession>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEventDetail {
    pub id: i64,
    pub event_type: String,
    pub payload: Value,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessageDetail {
    pub id: i64,
    pub turn_id: i64,
    pub turn_index: u32,
    pub role: String,
    pub content: Value,
    pub token_count: Option<u64>,
    #[serde(default)]
    pub estimated_token_count: Option<u32>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionToolExecutionDetail {
    pub id: i64,
    pub turn_index: u32,
    pub tool_call_id: String,
    pub tool_name: String,
    pub args: Value,
    pub output: Option<Value>,
    pub is_error: bool,
    pub duration_ms: Option<u64>,
    pub verdict: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionBranchDetail {
    pub branch_id: String,
    pub name: String,
    #[serde(default)]
    pub head_turn_id: Option<i64>,
    pub head_turn_index: Option<u32>,
    pub source_turn_id: Option<i64>,
    #[serde(default)]
    pub origin_kind: String,
    #[serde(default)]
    pub origin_task_id: Option<String>,
    #[serde(default)]
    pub origin_execution_id: Option<String>,
    #[serde(default)]
    pub origin_metadata: Option<Value>,
    pub active: bool,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionGraphDetail {
    pub session: SessionSummary,
    #[serde(default)]
    pub turns: Vec<SessionGraphTurnDetail>,
    #[serde(default)]
    pub branches: Vec<SessionBranchDetail>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionGraphTurnDetail {
    pub turn_id: i64,
    pub turn_public_id: String,
    pub parent_turn_id: Option<i64>,
    pub turn_index: u32,
    pub message_count: usize,
    pub tool_execution_count: usize,
    #[serde(default)]
    pub preview: Option<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDetail {
    pub session: SessionSummary,
    #[serde(default)]
    pub branches: Vec<SessionBranchDetail>,
    #[serde(default)]
    pub events: Vec<SessionEventDetail>,
    #[serde(default)]
    pub messages: Vec<SessionMessageDetail>,
    #[serde(default)]
    pub tool_executions: Vec<SessionToolExecutionDetail>,
    #[serde(default)]
    pub efficiency: Option<SessionEfficiencyDetail>,
    #[serde(default)]
    pub execution: SessionExecutionDetail,
    #[serde(default)]
    pub message_window: Option<SessionMessageWindow>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionExecutionDetail {
    #[serde(default)]
    pub tasks: Vec<SessionTaskExecutionDetail>,
    #[serde(default)]
    pub plans: Vec<SessionPlanExecutionDetail>,
    #[serde(default)]
    pub event_limit: usize,
    #[serde(default)]
    pub truncated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTaskExecutionDetail {
    pub task_id: String,
    pub trace_id: String,
    #[serde(default)]
    pub plan_id: Option<String>,
    #[serde(default)]
    pub run_id: Option<String>,
    pub agent_id: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub prompt: String,
    pub status: String,
    #[serde(default)]
    pub queue_depth: usize,
    #[serde(default)]
    pub task_turn_count: u32,
    pub execution: SessionExecutionContextDetail,
    #[serde(default)]
    pub turns: Vec<SessionTaskTurnDetail>,
    #[serde(default)]
    pub branch_outcome: Option<Value>,
    #[serde(default)]
    pub error: Option<String>,
    pub started_at: String,
    #[serde(default)]
    pub completed_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionExecutionContextDetail {
    pub execution_id: String,
    pub context_target: Value,
    pub visibility: String,
    pub durability: String,
    pub write_policy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTaskTurnDetail {
    pub turn_index: u32,
    pub task_turn_index: u32,
    #[serde(default)]
    pub has_tool_calls: Option<bool>,
    pub started_at: String,
    #[serde(default)]
    pub completed_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionPlanExecutionDetail {
    pub plan_id: String,
    #[serde(default)]
    pub title: Option<String>,
    pub status: String,
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub started_at: String,
    #[serde(default)]
    pub completed_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEfficiencyDetail {
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    #[serde(default)]
    pub total_cache_read_input_tokens: u64,
    #[serde(default)]
    pub total_cache_creation_input_tokens: u64,
    #[serde(default)]
    pub total_request_count: usize,
    #[serde(default)]
    pub turns: Vec<SessionTurnEfficiencyDetail>,
    #[serde(default)]
    pub latest_compaction: Option<SessionCompactionDetail>,
    #[serde(default)]
    pub provider_cache_metrics_available: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTurnEfficiencyDetail {
    pub turn_index: u32,
    #[serde(default)]
    pub requests: Vec<SessionRequestEfficiencyDetail>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRequestEfficiencyDetail {
    #[serde(default)]
    pub metrics: Option<InferenceRequestMetrics>,
    #[serde(default)]
    pub input_tokens: Option<u64>,
    #[serde(default)]
    pub output_tokens: Option<u64>,
    #[serde(default)]
    pub cache_read_input_tokens: Option<u64>,
    #[serde(default)]
    pub cache_creation_input_tokens: Option<u64>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequestMetrics {
    pub provider: String,
    pub model: String,
    pub requested_context: String,
    pub resolved_context: String,
    pub compaction_mode: String,
    pub estimated_input_tokens_before_compaction: u32,
    pub estimated_input_tokens: u32,
    pub system_prompt_tokens: u32,
    pub message_tokens: u32,
    pub tool_definition_tokens: u32,
    pub reusable_prefix_tokens: u32,
    pub context_window_tokens: u32,
    pub context_window_configured: bool,
    pub input_budget_tokens: u32,
    pub max_output_tokens: Option<u32>,
    pub thinking_budget_tokens: Option<u32>,
    pub available_message_count: usize,
    pub sent_message_count: usize,
    pub history_message_offset: usize,
    pub checkpoint_covered_message_count: usize,
    pub truncated_tool_results: usize,
    pub dropped_messages: usize,
    pub estimated_payload_bytes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionCompactionDetail {
    pub covered_message_count: usize,
    pub generated_at_turn_index: u32,
    pub provider: String,
    pub model: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessageWindow {
    pub offset: usize,
    pub total: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SessionBranchList {
    pub(crate) branches: Vec<SessionBranchDetail>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionActionResult {
    pub agent_id: String,
    #[serde(default)]
    pub slot_id: Option<String>,
    pub session_id: String,
    pub action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SessionSearchResultList {
    pub(crate) hits: Vec<SessionSearchHit>,
}
