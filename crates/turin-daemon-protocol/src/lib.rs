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
    pub runtime_idle_secs: Option<u64>,
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
    pub runtime_idle_secs: Option<u64>,
    #[serde(default)]
    pub tools: Option<ToolsConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CreateChannelParams {
    pub id: String,
    pub kind: String,
    pub agent_id: String,
    #[serde(default)]
    pub idle_ttl_secs: Option<u64>,
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
    pub idle_ttl_secs: Option<u64>,
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
pub struct ScheduleCreateParams {
    pub agent_id: String,
    pub prompt: String,
    pub next_run_unix_ms: i64,
    #[serde(default)]
    pub interval_seconds: Option<u64>,
    #[serde(default)]
    pub overlap_policy: Option<String>,
    #[serde(default)]
    pub persistence: Option<ContextPersistenceParams>,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScheduleJobDetail {
    pub id: i64,
    pub public_id: String,
    pub agent_id: String,
    pub prompt: String,
    #[serde(default)]
    pub persistence: Option<ContextPersistenceParams>,
    pub next_run_unix_ms: i64,
    #[serde(default)]
    pub interval_seconds: Option<u64>,
    pub overlap_policy: String,
    pub enabled: bool,
    pub slot_id: String,
    #[serde(default)]
    pub running_task_id: Option<String>,
    pub pending_rerun: bool,
    #[serde(default)]
    pub last_run_unix_ms: Option<i64>,
    #[serde(default)]
    pub last_status: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScheduleJobList {
    pub jobs: Vec<ScheduleJobDetail>,
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
    #[serde(rename = "schedule.list")]
    ScheduleList(NoParams),
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
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn request_envelope_round_trips_typed_shape() {
        let request = RequestEnvelope::new(
            Some("req_1".to_string()),
            DaemonRequest::TaskSubmit(SubmitTaskParams {
                agent_id: Some("writer".to_string()),
                session_id: None,
                slot_id: None,
                prompt: "review this".to_string(),
                content: None,
                tools: Default::default(),
                conflict_policy: Some("detached".to_string()),
            }),
        );

        let value = serde_json::to_value(&request).expect("serialize request");
        assert_eq!(value["id"], "req_1");
        assert_eq!(value["op"], "task.submit");
        assert_eq!(value["params"]["agent_id"], "writer");
        assert_eq!(value["params"]["prompt"], "review this");
        assert_eq!(value["params"]["conflict_policy"], "detached");

        let decoded: RequestEnvelope = serde_json::from_value(value).expect("deserialize request");
        match decoded.request {
            DaemonRequest::TaskSubmit(params) => {
                assert_eq!(params.agent_id.as_deref(), Some("writer"));
                assert!(params.session_id.is_none());
                assert_eq!(params.prompt, "review this");
                assert_eq!(params.conflict_policy.as_deref(), Some("detached"));
            }
            other => panic!("unexpected request variant: {other:?}"),
        }
    }

    #[test]
    fn sidestep_request_round_trips_typed_shape() {
        let request = RequestEnvelope::new(
            Some("req_3".to_string()),
            DaemonRequest::TaskSidestep(SidestepTaskParams {
                session_id: "sess_123".to_string(),
                slot_id: Some("sd_manual".to_string()),
                prompt: "What else should we add?".to_string(),
                content: None,
                tools: Default::default(),
                mode: SidestepModeParams::ForkSibling,
                context_target: Some(SidestepContextTargetParams::TurnId { turn_id: 42 }),
                timeout_ms: Some(2_500),
            }),
        );

        let value = serde_json::to_value(&request).expect("serialize request");
        assert_eq!(value["op"], "task.sidestep");
        assert_eq!(value["params"]["session_id"], "sess_123");
        assert_eq!(value["params"]["slot_id"], "sd_manual");
        assert_eq!(value["params"]["mode"], "fork_sibling");
        assert_eq!(value["params"]["context_target"]["kind"], "turn_id");
        assert_eq!(value["params"]["context_target"]["turn_id"], 42);

        let decoded: RequestEnvelope = serde_json::from_value(value).expect("deserialize request");
        match decoded.request {
            DaemonRequest::TaskSidestep(params) => {
                assert_eq!(params.session_id, "sess_123");
                assert_eq!(params.slot_id.as_deref(), Some("sd_manual"));
                assert_eq!(params.prompt, "What else should we add?");
                assert_eq!(params.mode, SidestepModeParams::ForkSibling);
                assert_eq!(params.timeout_ms, Some(2_500));
                assert!(matches!(
                    params.context_target,
                    Some(SidestepContextTargetParams::TurnId { turn_id: 42 })
                ));
            }
            other => panic!("unexpected request variant: {other:?}"),
        }
    }

    #[test]
    fn promote_task_request_round_trips_typed_shape() {
        let request = RequestEnvelope::new(
            Some("req_4".to_string()),
            DaemonRequest::TaskPromote(PromoteTaskParams {
                request_id: "req_task".to_string(),
                branch_name: Some("kept-idea".to_string()),
            }),
        );

        let value = serde_json::to_value(&request).expect("serialize request");
        assert_eq!(value["op"], "task.promote");
        assert_eq!(value["params"]["request_id"], "req_task");
        assert_eq!(value["params"]["branch_name"], "kept-idea");

        let decoded: RequestEnvelope = serde_json::from_value(value).expect("deserialize request");
        match decoded.request {
            DaemonRequest::TaskPromote(params) => {
                assert_eq!(params.request_id, "req_task");
                assert_eq!(params.branch_name.as_deref(), Some("kept-idea"));
            }
            other => panic!("unexpected request variant: {other:?}"),
        }
    }

    #[test]
    fn raw_daemon_wire_shape_deserializes_into_typed_request() {
        let decoded: RequestEnvelope = serde_json::from_value(json!({
            "id": "req_2",
            "op": "agent.disable",
            "params": { "id": "docs-reviewer" }
        }))
        .expect("deserialize request");

        match decoded.request {
            DaemonRequest::AgentDisable(EntityIdParams { id }) => {
                assert_eq!(id, "docs-reviewer");
            }
            other => panic!("unexpected request variant: {other:?}"),
        }
    }

    #[test]
    fn error_code_serializes_as_snake_case() {
        let response = ResponseEnvelope::err(
            Some("req_2".to_string()),
            ErrorCode::AgentNotFound,
            "missing",
            None,
        );

        let value = serde_json::to_value(&response).expect("serialize response");
        assert_eq!(value["error"]["code"], "agent_not_found");
    }

    #[test]
    fn handshake_round_trips_typed_shape() {
        let handshake = DaemonHandshake {
            pong: true,
            version: env!("CARGO_PKG_VERSION").to_string(),
            protocol_version: DAEMON_PROTOCOL_VERSION,
            transport: DAEMON_TRANSPORT_UNIX.to_string(),
            wire_format: DAEMON_WIRE_FORMAT_NDJSON.to_string(),
            capabilities: DaemonCapabilities {
                runtime_snapshot_v1: true,
                scoped_event_snapshots: true,
                lag_resnapshot: true,
                watcher_rescan_failed_events: true,
                channels: true,
            },
        };

        let value = serde_json::to_value(&handshake).expect("serialize handshake");
        assert_eq!(value["protocol_version"], DAEMON_PROTOCOL_VERSION);
        assert_eq!(value["transport"], DAEMON_TRANSPORT_UNIX);

        let decoded: DaemonHandshake =
            serde_json::from_value(value).expect("deserialize handshake");
        assert!(decoded.capabilities.runtime_snapshot_v1);
        assert!(decoded.capabilities.channels);
    }

    #[test]
    fn schedule_create_request_round_trips_typed_shape() {
        let request = RequestEnvelope::new(
            Some("req_sched".to_string()),
            DaemonRequest::ScheduleCreate(ScheduleCreateParams {
                agent_id: "default".to_string(),
                prompt: "Heartbeat".to_string(),
                next_run_unix_ms: 1_700_000_000_000,
                interval_seconds: Some(300),
                overlap_policy: Some("skip".to_string()),
                persistence: Some(ContextPersistenceParams {
                    state: Some(StoreTargetParams {
                        path: None,
                        alias: Some("project-alpha".to_string()),
                    }),
                    store: None,
                }),
                enabled: true,
            }),
        );

        let value = serde_json::to_value(&request).expect("serialize request");
        assert_eq!(value["op"], "schedule.create");
        assert_eq!(value["params"]["agent_id"], "default");
        assert_eq!(value["params"]["prompt"], "Heartbeat");
        assert_eq!(value["params"]["next_run_unix_ms"], 1_700_000_000_000i64);
        assert_eq!(value["params"]["interval_seconds"], 300);
        assert_eq!(value["params"]["overlap_policy"], "skip");
        assert_eq!(
            value["params"]["persistence"]["state"]["alias"],
            "project-alpha"
        );

        let decoded: RequestEnvelope = serde_json::from_value(value).expect("deserialize request");
        match decoded.request {
            DaemonRequest::ScheduleCreate(params) => {
                assert_eq!(params.agent_id, "default");
                assert_eq!(params.prompt, "Heartbeat");
                assert_eq!(params.next_run_unix_ms, 1_700_000_000_000i64);
                assert_eq!(params.interval_seconds, Some(300));
                assert_eq!(params.overlap_policy.as_deref(), Some("skip"));
                assert_eq!(
                    params
                        .persistence
                        .as_ref()
                        .and_then(|p| p.state.as_ref())
                        .and_then(|state| state.alias.as_deref()),
                    Some("project-alpha")
                );
                assert!(params.enabled);
            }
            other => panic!("unexpected request variant: {other:?}"),
        }
    }
}
