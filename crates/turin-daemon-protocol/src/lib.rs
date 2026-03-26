use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fmt;
use turin_types::{AgentMode, ThinkingConfig};

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
    pub mode: Option<AgentMode>,
    #[serde(default)]
    pub harness: Option<String>,
    #[serde(default)]
    pub idle_grace_secs: Option<u64>,
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
    pub mode: Option<AgentMode>,
    #[serde(default)]
    pub idle_grace_secs: Option<u64>,
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
    pub prompt: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenSessionParams {
    pub agent_id: String,
    #[serde(default)]
    pub slot_id: Option<String>,
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
pub struct SessionIdParams {
    pub session_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SessionListParams {
    #[serde(default = "default_session_limit")]
    pub limit: usize,
    #[serde(default)]
    pub offset: usize,
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
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SessionTitleParams {
    pub session_id: String,
    #[serde(default)]
    pub title: Option<String>,
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
    #[serde(rename = "task.get")]
    TaskGet(TaskIdParams),
    #[serde(rename = "task.wait")]
    TaskWait(WaitTaskParams),
    #[serde(rename = "task.cancel")]
    TaskCancel(TaskIdParams),
    #[serde(rename = "task.list")]
    TaskList(NoParams),
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
    #[serde(rename = "session.cancel")]
    SessionCancel(SessionIdParams),
    #[serde(rename = "session.kill")]
    SessionKill(SessionIdParams),
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
                prompt: "review this".to_string(),
            }),
        );

        let value = serde_json::to_value(&request).expect("serialize request");
        assert_eq!(value["id"], "req_1");
        assert_eq!(value["op"], "task.submit");
        assert_eq!(value["params"]["agent_id"], "writer");
        assert_eq!(value["params"]["prompt"], "review this");

        let decoded: RequestEnvelope = serde_json::from_value(value).expect("deserialize request");
        match decoded.request {
            DaemonRequest::TaskSubmit(params) => {
                assert_eq!(params.agent_id.as_deref(), Some("writer"));
                assert!(params.session_id.is_none());
                assert_eq!(params.prompt, "review this");
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
}
