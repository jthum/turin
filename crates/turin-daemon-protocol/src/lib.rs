use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;
use turin_channel_core::ChannelAdapterManifest;
use turin_types::{TaskInputContent, ThinkingConfig, ToolsConfig};

pub const DAEMON_PROTOCOL_VERSION: u32 = 1;
pub const DAEMON_TRANSPORT_UNIX: &str = "unix";
pub const DAEMON_TRANSPORT_NAMED_PIPE: &str = "named_pipe";
pub const DAEMON_WIRE_FORMAT_NDJSON: &str = "ndjson";

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct NoParams {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct DaemonCapabilities {
    pub runtime_snapshot_v1: bool,
    pub scoped_event_snapshots: bool,
    pub lag_resnapshot: bool,
    pub watcher_rescan_failed_events: bool,
    pub channels: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct DaemonHandshake {
    pub pong: bool,
    pub version: String,
    pub protocol_version: u32,
    pub transport: String,
    pub wire_format: String,
    pub capabilities: DaemonCapabilities,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct RuntimeEventsSubscribeParams {
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub slot_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CreateAgentParams {
    pub id: String,
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
    #[serde(default)]
    pub harness: Option<String>,
    #[serde(default)]
    pub idle_timeout_seconds: Option<u64>,
    #[serde(default)]
    pub tools: ToolsConfig,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EntityIdParams {
    pub id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UpdateAgentParams {
    pub id: String,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
    #[serde(default)]
    pub idle_timeout_seconds: Option<u64>,
    #[serde(default)]
    pub tools: Option<ToolsConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CreateChannelParams {
    pub id: String,
    pub kind: String,
    pub agent_id: String,
    #[serde(default)]
    pub idle_timeout_seconds: Option<u64>,
    #[serde(default = "default_enabled")]
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

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BindHarnessParams {
    pub id: String,
    pub harness_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SubmitTaskParams {
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub slot_id: Option<String>,
    pub prompt: String,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<TaskInputContent>>,
    #[serde(default)]
    pub tools: Option<ToolsConfig>,
    #[serde(default)]
    pub conflict_policy: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct StoreTargetParams {
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub alias: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct ContextPersistenceParams {
    #[serde(default)]
    pub state: Option<StoreTargetParams>,
    #[serde(default)]
    pub store: Option<StoreTargetParams>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScheduleActionParams {
    pub name: String,
    #[serde(default)]
    pub params: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScheduleCreateParams {
    pub agent_id: String,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<TaskInputContent>>,
    #[serde(default)]
    pub tools: Option<ToolsConfig>,
    #[serde(default)]
    pub conflict_policy: Option<String>,
    #[serde(default)]
    pub action: Option<ScheduleActionParams>,
    pub next_run_unix_ms: i64,
    #[serde(default)]
    pub interval_seconds: Option<u64>,
    #[serde(default)]
    pub recurring_pattern: Option<String>,
    #[serde(default)]
    pub overlap_policy: Option<String>,
    #[serde(default)]
    pub work_key: Option<String>,
    #[serde(default)]
    pub max_concurrency: Option<u32>,
    #[serde(default)]
    pub persistence: Option<ContextPersistenceParams>,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScheduleUpdateParams {
    pub id: String,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<TaskInputContent>>,
    #[serde(default)]
    pub tools: Option<ToolsConfig>,
    #[serde(default)]
    pub conflict_policy: Option<String>,
    #[serde(default)]
    pub action: Option<ScheduleActionParams>,
    #[serde(default)]
    pub next_run_unix_ms: Option<i64>,
    #[serde(default)]
    pub interval_seconds: Option<u64>,
    #[serde(default)]
    pub recurring_pattern: Option<String>,
    #[serde(default)]
    pub overlap_policy: Option<String>,
    #[serde(default)]
    pub work_key: Option<String>,
    #[serde(default)]
    pub max_concurrency: Option<u32>,
    #[serde(default)]
    pub persistence: Option<ContextPersistenceParams>,
    #[serde(default)]
    pub enabled: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScheduleRunsParams {
    pub id: String,
    #[serde(default)]
    pub active_only: bool,
    #[serde(default)]
    pub limit: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorklistListParams {
    #[serde(default)]
    pub persistence: Option<ContextPersistenceParams>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub scope: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorklistTargetParams {
    pub id: String,
    #[serde(default)]
    pub persistence: Option<ContextPersistenceParams>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorkItemTargetParams {
    pub id: String,
    #[serde(default)]
    pub persistence: Option<ContextPersistenceParams>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorklistItemsParams {
    pub id: String,
    #[serde(default)]
    pub persistence: Option<ContextPersistenceParams>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub parent_id: Option<String>,
    #[serde(default)]
    pub r#where: Option<serde_json::Map<String, Value>>,
    #[serde(default)]
    pub claimed_only: bool,
    #[serde(default)]
    pub paused_only: bool,
    #[serde(default)]
    pub due_only: bool,
    #[serde(default)]
    pub limit: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScheduleJobList {
    pub jobs: Vec<ScheduleJobDetail>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScheduleJobRunList {
    pub public_id: String,
    pub runs: Vec<ScheduleJobRunDetail>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorklistList {
    pub worklists: Vec<WorklistDetail>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorkItemList {
    pub worklist_id: String,
    pub items: Vec<WorkItemDetail>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScheduleJobDetail {
    pub id: i64,
    pub public_id: String,
    pub agent_id: String,
    pub kind: String,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub content: Option<Vec<TaskInputContent>>,
    #[serde(default)]
    pub tools: Option<ToolsConfig>,
    #[serde(default)]
    pub conflict_policy: Option<String>,
    #[serde(default)]
    pub action: Option<ScheduleActionParams>,
    #[serde(default)]
    pub persistence: Option<ContextPersistenceParams>,
    pub next_run_unix_ms: i64,
    #[serde(default)]
    pub interval_seconds: Option<u64>,
    #[serde(default)]
    pub recurring_pattern: Option<String>,
    pub overlap_policy: String,
    #[serde(default)]
    pub work_key: Option<String>,
    #[serde(default)]
    pub max_concurrency: Option<u32>,
    pub enabled: bool,
    pub slot_id: String,
    #[serde(default)]
    pub running_task_id: Option<String>,
    #[serde(default)]
    pub active_run_count: u32,
    pub pending_rerun: bool,
    #[serde(default)]
    pub last_run_unix_ms: Option<i64>,
    #[serde(default)]
    pub last_status: Option<String>,
    #[serde(default)]
    pub last_error_code: Option<String>,
    #[serde(default)]
    pub failure_count: u64,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScheduleJobRunDetail {
    pub id: i64,
    pub task_id: String,
    pub started_unix_ms: i64,
    #[serde(default)]
    pub finished_unix_ms: Option<i64>,
    #[serde(default)]
    pub duration_ms: Option<u64>,
    #[serde(default)]
    pub last_status: Option<String>,
    pub active: bool,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorklistDetail {
    pub id: i64,
    pub public_id: String,
    pub name: String,
    pub scope_ref: String,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub persistence: Option<ContextPersistenceParams>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorkItemDetail {
    pub id: i64,
    pub public_id: String,
    pub worklist_id: String,
    #[serde(default)]
    pub parent_id: Option<String>,
    pub title: String,
    pub kind: String,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub content: Option<Vec<TaskInputContent>>,
    #[serde(default)]
    pub tools: Option<ToolsConfig>,
    #[serde(default)]
    pub conflict_policy: Option<String>,
    #[serde(default)]
    pub action: Option<ScheduleActionParams>,
    pub status: String,
    #[serde(default)]
    pub paused: bool,
    #[serde(default)]
    pub pause_reason: Option<String>,
    #[serde(default)]
    pub pause_until_unix_ms: Option<i64>,
    pub priority: i64,
    #[serde(default)]
    pub after: Option<Vec<String>>,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub claim_agent_id: Option<String>,
    #[serde(default)]
    pub claim_session_id: Option<String>,
    #[serde(default)]
    pub claim_execution_id: Option<String>,
    #[serde(default)]
    pub claim_heartbeat_unix_ms: Option<i64>,
    #[serde(default)]
    pub claimed_at: Option<String>,
    #[serde(default)]
    pub completed_at: Option<String>,
    #[serde(default)]
    pub failure_reason: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SidestepContextTargetParams {
    BranchHead { branch_head_id: i64 },
    TurnId { turn_id: i64 },
    SelectedPath { turn_ids: Vec<i64> },
    ExternalReference { reference: String },
    SummarySource { source_turn_id: i64 },
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SidestepModeParams {
    #[default]
    Ephemeral,
    ForkSibling,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SidestepTaskParams {
    pub session_id: String,
    #[serde(default)]
    pub slot_id: Option<String>,
    pub prompt: String,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<TaskInputContent>>,
    #[serde(default)]
    pub tools: Option<ToolsConfig>,
    #[serde(default)]
    pub mode: SidestepModeParams,
    #[serde(default)]
    pub context_target: Option<SidestepContextTargetParams>,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

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
pub struct TaskIdParams {
    pub request_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WaitTaskParams {
    pub request_id: String,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromoteTaskParams {
    pub request_id: String,
    #[serde(default)]
    pub branch_name: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SessionIdParams {
    pub session_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LiveSessionTargetParams {
    pub session_id: String,
    #[serde(default)]
    pub slot_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SessionListParams {
    #[serde(default = "default_session_limit")]
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
    #[serde(default = "default_search_limit")]
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

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "op", content = "params")]
pub enum DaemonRequest {
    #[serde(rename = "daemon.ping")]
    DaemonPing(NoParams),
    #[serde(rename = "daemon.status")]
    DaemonStatus(NoParams),
    #[serde(rename = "daemon.stop")]
    DaemonStop(NoParams),
    #[serde(rename = "runtime.rescan")]
    RuntimeRescan(NoParams),
    #[serde(rename = "runtime.reload")]
    RuntimeReload(NoParams),
    #[serde(rename = "runtime.errors")]
    RuntimeErrors(NoParams),
    #[serde(rename = "runtime.events.subscribe")]
    RuntimeEventsSubscribe(RuntimeEventsSubscribeParams),
    #[serde(rename = "agent.list")]
    AgentList(NoParams),
    #[serde(rename = "agent.get")]
    AgentGet(EntityIdParams),
    #[serde(rename = "agent.status")]
    AgentStatus(EntityIdParams),
    #[serde(rename = "agent.issues")]
    AgentIssues(EntityIdParams),
    #[serde(rename = "agent.create")]
    AgentCreate(CreateAgentParams),
    #[serde(rename = "agent.enable")]
    AgentEnable(EntityIdParams),
    #[serde(rename = "agent.disable")]
    AgentDisable(EntityIdParams),
    #[serde(rename = "agent.update")]
    AgentUpdate(UpdateAgentParams),
    #[serde(rename = "agent.reload")]
    AgentReload(EntityIdParams),
    #[serde(rename = "agent.bind_harness")]
    AgentBindHarness(BindHarnessParams),
    #[serde(rename = "agent.use_local_harness")]
    AgentUseLocalHarness(EntityIdParams),
    #[serde(rename = "agent.delete")]
    AgentDelete(EntityIdParams),
    #[serde(rename = "task.submit")]
    TaskSubmit(SubmitTaskParams),
    #[serde(rename = "task.sidestep")]
    TaskSidestep(SidestepTaskParams),
    #[serde(rename = "task.get")]
    TaskGet(TaskIdParams),
    #[serde(rename = "task.wait")]
    TaskWait(WaitTaskParams),
    #[serde(rename = "task.promote")]
    TaskPromote(PromoteTaskParams),
    #[serde(rename = "task.cancel")]
    TaskCancel(TaskIdParams),
    #[serde(rename = "task.list")]
    TaskList(NoParams),
    #[serde(rename = "schedule.create")]
    ScheduleCreate(ScheduleCreateParams),
    #[serde(rename = "schedule.update")]
    ScheduleUpdate(ScheduleUpdateParams),
    #[serde(rename = "schedule.get")]
    ScheduleGet(EntityIdParams),
    #[serde(rename = "schedule.list")]
    ScheduleList(NoParams),
    #[serde(rename = "schedule.runs")]
    ScheduleRuns(ScheduleRunsParams),
    #[serde(rename = "schedule.enable")]
    ScheduleEnable(EntityIdParams),
    #[serde(rename = "schedule.disable")]
    ScheduleDisable(EntityIdParams),
    #[serde(rename = "schedule.delete")]
    ScheduleDelete(EntityIdParams),
    #[serde(rename = "worklist.list")]
    WorklistList(WorklistListParams),
    #[serde(rename = "worklist.get")]
    WorklistGet(WorklistTargetParams),
    #[serde(rename = "worklist.items")]
    WorklistItems(WorklistItemsParams),
    #[serde(rename = "workitem.get")]
    WorkItemGet(WorkItemTargetParams),
    #[serde(rename = "session.list")]
    SessionList(SessionListParams),
    #[serde(rename = "session.list_live")]
    SessionListLive(NoParams),
    #[serde(rename = "session.search")]
    SessionSearch(SessionSearchParams),
    #[serde(rename = "session.open")]
    SessionOpen(OpenSessionParams),
    #[serde(rename = "session.resume")]
    SessionResume(ResumeSessionParams),
    #[serde(rename = "session.get")]
    SessionGet(SessionIdParams),
    #[serde(rename = "session.set_title")]
    SessionSetTitle(SessionTitleParams),
    #[serde(rename = "session.branch_list")]
    SessionBranchList(SessionIdParams),
    #[serde(rename = "session.branch_create")]
    SessionBranchCreate(SessionBranchCreateParams),
    #[serde(rename = "session.branch_checkout")]
    SessionBranchCheckout(SessionBranchCheckoutParams),
    #[serde(rename = "session.branch_siblings")]
    SessionBranchSiblings(SessionBranchSiblingsParams),
    #[serde(rename = "session.cancel")]
    SessionCancel(LiveSessionTargetParams),
    #[serde(rename = "session.kill")]
    SessionKill(LiveSessionTargetParams),
    #[serde(rename = "harness.list")]
    HarnessList(NoParams),
    #[serde(rename = "harness.create")]
    HarnessCreate(EntityIdParams),
    #[serde(rename = "harness.get")]
    HarnessGet(EntityIdParams),
    #[serde(rename = "harness.issues")]
    HarnessIssues(EntityIdParams),
    #[serde(rename = "harness.reload")]
    HarnessReload(EntityIdParams),
    #[serde(rename = "harness.validate")]
    HarnessValidate(EntityIdParams),
    #[serde(rename = "harness.delete")]
    HarnessDelete(EntityIdParams),
    #[serde(rename = "channel.list")]
    ChannelList(NoParams),
    #[serde(rename = "channel.create")]
    ChannelCreate(CreateChannelParams),
    #[serde(rename = "channel.get")]
    ChannelGet(EntityIdParams),
    #[serde(rename = "channel.status")]
    ChannelStatus(EntityIdParams),
    #[serde(rename = "channel.issues")]
    ChannelIssues(EntityIdParams),
    #[serde(rename = "channel.enable")]
    ChannelEnable(EntityIdParams),
    #[serde(rename = "channel.disable")]
    ChannelDisable(EntityIdParams),
    #[serde(rename = "channel.update")]
    ChannelUpdate(UpdateChannelParams),
    #[serde(rename = "channel.access.get")]
    ChannelAccessGet(ChannelAccessParams),
    #[serde(rename = "channel.access.approve")]
    ChannelAccessApprove(ChannelAccessRoomParams),
    #[serde(rename = "channel.access.reject")]
    ChannelAccessReject(ChannelAccessRoomParams),
    #[serde(rename = "channel.access.revoke")]
    ChannelAccessRevoke(ChannelAccessRoomParams),
    #[serde(rename = "channel.runner.hello")]
    ChannelRunnerHello(ChannelRunnerHelloParams),
    #[serde(rename = "channel.runner.heartbeat")]
    ChannelRunnerHeartbeat(ChannelRunnerHeartbeatParams),
    #[serde(rename = "channel.delete")]
    ChannelDelete(EntityIdParams),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RequestEnvelope {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(flatten)]
    pub request: DaemonRequest,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    InvalidRequest,
    InvalidParams,
    AgentNotFound,
    TaskNotFound,
    ScheduleNotFound,
    WorklistNotFound,
    WorkItemNotFound,
    SessionNotFound,
    HarnessNotFound,
    ChannelNotFound,
    ValidationFailed,
    Conflict,
    ResourceBusy,
    UnsupportedOperation,
    InternalError,
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            ErrorCode::InvalidRequest => "invalid_request",
            ErrorCode::InvalidParams => "invalid_params",
            ErrorCode::AgentNotFound => "agent_not_found",
            ErrorCode::TaskNotFound => "task_not_found",
            ErrorCode::ScheduleNotFound => "schedule_not_found",
            ErrorCode::WorklistNotFound => "worklist_not_found",
            ErrorCode::WorkItemNotFound => "workitem_not_found",
            ErrorCode::SessionNotFound => "session_not_found",
            ErrorCode::HarnessNotFound => "harness_not_found",
            ErrorCode::ChannelNotFound => "channel_not_found",
            ErrorCode::ValidationFailed => "validation_failed",
            ErrorCode::Conflict => "conflict",
            ErrorCode::ResourceBusy => "resource_busy",
            ErrorCode::UnsupportedOperation => "unsupported_operation",
            ErrorCode::InternalError => "internal_error",
        };
        f.write_str(name)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ErrorEnvelope {
    pub code: ErrorCode,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResponseEnvelope {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub ok: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorEnvelope>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EventEnvelope {
    pub event: String,
    #[serde(default)]
    pub data: Value,
}

impl ResponseEnvelope {
    pub fn ok(id: Option<String>, result: Value) -> Self {
        Self {
            id,
            ok: true,
            result: Some(result),
            error: None,
        }
    }

    pub fn err(
        id: Option<String>,
        code: ErrorCode,
        message: impl Into<String>,
        details: Option<Value>,
    ) -> Self {
        Self {
            id,
            ok: false,
            result: None,
            error: Some(ErrorEnvelope {
                code,
                message: message.into(),
                details,
            }),
        }
    }
}

impl EventEnvelope {
    pub fn new(event: impl Into<String>, data: Value) -> Self {
        Self {
            event: event.into(),
            data,
        }
    }
}

impl RequestEnvelope {
    pub fn new(id: Option<String>, request: DaemonRequest) -> Self {
        Self { id, request }
    }
}

fn default_enabled() -> bool {
    true
}

fn default_session_limit() -> usize {
    50
}

fn default_search_limit() -> usize {
    64
}

#[cfg(test)]
mod tests;
