use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::Duration;
use turin_daemon_client::DaemonClient;
use turin_daemon_protocol::{
    ChannelAccessParams, ChannelAccessRoomParams, DaemonHandshake, DaemonRequest, EntityIdParams,
    EventEnvelope, NoParams, OpenSessionParams, RequestEnvelope, ResponseEnvelope,
    ResumeSessionParams, RuntimeEventsSubscribeParams, SessionIdParams, SessionListParams,
    SessionSearchHitKind, SessionSearchParams, SessionSearchScope, SessionTitleParams,
    SubmitTaskParams, TaskIdParams, WaitTaskParams,
};
use turin_remote_client::RemoteClient;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConnectionSpec {
    LocalConfig {
        config_path: PathBuf,
    },
    LocalEndpoint {
        endpoint: PathBuf,
    },
    Remote {
        base_url: String,
        auth_token: String,
    },
    RemoteEnv {
        base_url: String,
        auth_token_env: String,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConnectionKind {
    Local,
    Remote,
}

#[derive(Debug, Clone, Serialize)]
pub struct ControlHealth {
    pub connection_kind: ConnectionKind,
    pub target: String,
    pub ready: bool,
    pub version: String,
    pub protocol_version: u32,
    pub transport: String,
    pub wire_format: String,
    pub issue_count: usize,
    pub agent_count: usize,
    pub harness_count: usize,
    pub channel_count: usize,
    pub running_agent_count: usize,
    pub active_task_count: usize,
    pub queued_task_count: usize,
    pub awaiting_result_count: usize,
    pub channel_runtime_count: usize,
    pub failed_channel_count: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct ManagedSubscribeOptions {
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
}

impl Default for ManagedSubscribeOptions {
    fn default() -> Self {
        Self {
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(1),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ControlClient {
    Local(DaemonClient),
    Remote(RemoteClient),
}

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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRuntime {
    pub agent_id: String,
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
    pub mode: Option<String>,
    pub harness: Option<String>,
    pub idle_grace_secs: Option<u64>,
    pub has_local_harness: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelDetail {
    pub id: String,
    pub directory: String,
    pub enabled: bool,
    pub kind: String,
    pub agent_id: String,
    pub idle_ttl_secs: Option<u64>,
    pub settings: Value,
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
    pub approved_at_unix_secs: u64,
    pub approved_by_user_id: Option<String>,
    pub approved_by_username: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingChannelRoom {
    pub room: ChannelAccessRoom,
    pub first_seen_unix_secs: u64,
    pub last_seen_unix_secs: u64,
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
    pub state: String,
    pub runtime_task_id: Option<String>,
    pub status: Option<String>,
    pub task_turn_count: Option<u32>,
    pub output: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TaskList {
    tasks: Vec<TaskStatus>,
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
struct SessionList {
    sessions: Vec<SessionSummary>,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LiveSessionList {
    sessions: Vec<LiveSession>,
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
    pub turn_index: u32,
    pub role: String,
    pub content: Value,
    pub token_count: Option<u64>,
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
pub struct SessionDetail {
    pub session: SessionSummary,
    #[serde(default)]
    pub events: Vec<SessionEventDetail>,
    #[serde(default)]
    pub messages: Vec<SessionMessageDetail>,
    #[serde(default)]
    pub tool_executions: Vec<SessionToolExecutionDetail>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionActionResult {
    pub agent_id: String,
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
struct SessionSearchResultList {
    hits: Vec<SessionSearchHit>,
}

pub enum ManagedEventStream {
    Local(turin_daemon_client::ManagedEventStream),
    Remote(turin_remote_client::ManagedRemoteEventStream),
}

impl ConnectionSpec {
    pub async fn from_local_config(config_path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::LocalConfig {
            config_path: config_path.as_ref().to_path_buf(),
        })
    }
}

impl ControlClient {
    pub async fn connect(spec: &ConnectionSpec) -> Result<Self> {
        match spec {
            ConnectionSpec::LocalConfig { config_path } => Ok(Self::Local(
                DaemonClient::from_config(config_path)
                    .await
                    .with_context(|| {
                        format!(
                            "Failed to resolve daemon endpoint from '{}'",
                            config_path.display()
                        )
                    })?,
            )),
            ConnectionSpec::LocalEndpoint { endpoint } => {
                Ok(Self::Local(DaemonClient::new(endpoint.clone())))
            }
            ConnectionSpec::Remote {
                base_url,
                auth_token,
            } => Ok(Self::Remote(RemoteClient::new(
                base_url.clone(),
                auth_token.clone(),
            ))),
            ConnectionSpec::RemoteEnv {
                base_url,
                auth_token_env,
            } => {
                let auth_token = std::env::var(auth_token_env).with_context(|| {
                    format!("Remote auth token env var '{}' is not set", auth_token_env)
                })?;
                Ok(Self::Remote(RemoteClient::new(
                    base_url.clone(),
                    auth_token,
                )))
            }
        }
    }

    pub fn kind(&self) -> ConnectionKind {
        match self {
            Self::Local(_) => ConnectionKind::Local,
            Self::Remote(_) => ConnectionKind::Remote,
        }
    }

    pub fn target(&self) -> String {
        match self {
            Self::Local(client) => client.endpoint().display().to_string(),
            Self::Remote(client) => client.base_url().to_string(),
        }
    }

    pub async fn send(&self, request: RequestEnvelope) -> Result<ResponseEnvelope> {
        match self {
            Self::Local(client) => client.send(request).await,
            Self::Remote(client) => client.send(request).await,
        }
    }

    pub async fn request(
        &self,
        id: Option<String>,
        request: DaemonRequest,
    ) -> Result<ResponseEnvelope> {
        self.send(RequestEnvelope::new(id, request)).await
    }

    pub async fn request_ok<T: for<'de> Deserialize<'de>>(
        &self,
        id: Option<String>,
        request: DaemonRequest,
    ) -> Result<T> {
        match self {
            Self::Local(client) => client.request_ok(id, request).await,
            Self::Remote(client) => client.request_ok(id, request).await,
        }
    }

    pub async fn handshake(&self) -> Result<DaemonHandshake> {
        match self {
            Self::Local(client) => client.handshake().await,
            Self::Remote(client) => client.handshake().await,
        }
    }

    pub async fn status(&self) -> Result<DaemonStatus> {
        self.request_ok(None, DaemonRequest::DaemonStatus(NoParams::default()))
            .await
    }

    pub async fn health(&self) -> Result<ControlHealth> {
        let (health, _) = self.health_and_status().await?;
        Ok(health)
    }

    pub async fn health_and_status(&self) -> Result<(ControlHealth, DaemonStatus)> {
        let target = self.target();
        let connection_kind = self.kind();
        let handshake = self.handshake().await?;
        let status = self.status().await?;
        let health = build_health(connection_kind, target, handshake, &status);
        Ok((health, status))
    }

    pub async fn list_live_sessions(&self) -> Result<Vec<LiveSession>> {
        let response: LiveSessionList = self
            .request_ok(None, DaemonRequest::SessionListLive(NoParams::default()))
            .await?;
        Ok(response.sessions)
    }

    pub async fn list_sessions(&self, limit: usize, offset: usize) -> Result<Vec<SessionSummary>> {
        let response: SessionList = self
            .request_ok(
                None,
                DaemonRequest::SessionList(SessionListParams { limit, offset }),
            )
            .await?;
        Ok(response.sessions)
    }

    pub async fn get_session(&self, session_id: &str) -> Result<SessionDetail> {
        self.request_ok(
            None,
            DaemonRequest::SessionGet(SessionIdParams {
                session_id: session_id.to_string(),
            }),
        )
        .await
    }

    pub async fn search_sessions(
        &self,
        query: &str,
        scope: SessionSearchScope,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<SessionSearchHit>> {
        let response: SessionSearchResultList = self
            .request_ok(
                None,
                DaemonRequest::SessionSearch(SessionSearchParams {
                    query: query.to_string(),
                    scope: Some(scope),
                    limit,
                    offset,
                }),
            )
            .await?;
        Ok(response.hits)
    }

    pub async fn set_session_title(
        &self,
        session_id: &str,
        title: Option<String>,
    ) -> Result<SessionSummary> {
        self.request_ok(
            None,
            DaemonRequest::SessionSetTitle(SessionTitleParams {
                session_id: session_id.to_string(),
                title,
            }),
        )
        .await
    }

    pub async fn open_session(
        &self,
        agent_id: &str,
        slot_id: Option<String>,
    ) -> Result<LiveSession> {
        self.request_ok(
            None,
            DaemonRequest::SessionOpen(OpenSessionParams {
                agent_id: agent_id.to_string(),
                slot_id,
            }),
        )
        .await
    }

    pub async fn resume_session(
        &self,
        session_id: &str,
        slot_id: Option<String>,
    ) -> Result<LiveSession> {
        self.request_ok(
            None,
            DaemonRequest::SessionResume(ResumeSessionParams {
                session_id: session_id.to_string(),
                slot_id,
            }),
        )
        .await
    }

    pub async fn cancel_session(&self, session_id: &str) -> Result<SessionActionResult> {
        self.request_ok(
            None,
            DaemonRequest::SessionCancel(SessionIdParams {
                session_id: session_id.to_string(),
            }),
        )
        .await
    }

    pub async fn kill_session(&self, session_id: &str) -> Result<SessionActionResult> {
        self.request_ok(
            None,
            DaemonRequest::SessionKill(SessionIdParams {
                session_id: session_id.to_string(),
            }),
        )
        .await
    }

    pub async fn list_tasks(&self) -> Result<Vec<TaskStatus>> {
        let response: TaskList = self
            .request_ok(None, DaemonRequest::TaskList(NoParams::default()))
            .await?;
        Ok(response.tasks)
    }

    pub async fn get_task(&self, request_id: &str) -> Result<TaskStatus> {
        self.request_ok(
            None,
            DaemonRequest::TaskGet(TaskIdParams {
                request_id: request_id.to_string(),
            }),
        )
        .await
    }

    pub async fn submit_task(
        &self,
        agent_id: Option<String>,
        session_id: Option<String>,
        prompt: String,
    ) -> Result<TaskStatus> {
        self.request_ok(
            None,
            DaemonRequest::TaskSubmit(SubmitTaskParams {
                agent_id,
                session_id,
                prompt,
            }),
        )
        .await
    }

    pub async fn wait_task(&self, request_id: &str, timeout_ms: Option<u64>) -> Result<TaskStatus> {
        self.request_ok(
            None,
            DaemonRequest::TaskWait(WaitTaskParams {
                request_id: request_id.to_string(),
                timeout_ms,
            }),
        )
        .await
    }

    pub async fn cancel_task(&self, request_id: &str) -> Result<TaskStatus> {
        self.request_ok(
            None,
            DaemonRequest::TaskCancel(TaskIdParams {
                request_id: request_id.to_string(),
            }),
        )
        .await
    }

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

    pub async fn subscribe_managed(
        &self,
        filter: RuntimeEventsSubscribeParams,
    ) -> Result<ManagedEventStream> {
        self.subscribe_managed_with_options(filter, ManagedSubscribeOptions::default())
            .await
    }

    pub async fn subscribe_managed_with_options(
        &self,
        filter: RuntimeEventsSubscribeParams,
        options: ManagedSubscribeOptions,
    ) -> Result<ManagedEventStream> {
        match self {
            Self::Local(client) => Ok(ManagedEventStream::Local(
                client
                    .subscribe_managed_with_options(
                        filter,
                        turin_daemon_client::ManagedSubscribeOptions {
                            initial_backoff: options.initial_backoff,
                            max_backoff: options.max_backoff,
                        },
                    )
                    .await?,
            )),
            Self::Remote(client) => Ok(ManagedEventStream::Remote(
                client
                    .subscribe_managed_with_options(
                        filter,
                        turin_remote_client::ManagedSubscribeOptions {
                            initial_backoff: options.initial_backoff,
                            max_backoff: options.max_backoff,
                        },
                    )
                    .await?,
            )),
        }
    }
}

impl ManagedEventStream {
    pub async fn next_event(&mut self) -> Result<EventEnvelope> {
        match self {
            Self::Local(stream) => stream.next_event().await,
            Self::Remote(stream) => stream.next_event().await,
        }
    }
}

fn build_health(
    connection_kind: ConnectionKind,
    target: String,
    handshake: DaemonHandshake,
    status: &DaemonStatus,
) -> ControlHealth {
    let running_agent_count = status
        .agent_runtimes
        .iter()
        .filter(|runtime| runtime.running)
        .count();
    let active_task_count = status
        .agent_runtimes
        .iter()
        .map(|runtime| runtime.active_tasks)
        .sum();
    let queued_task_count = status
        .agent_runtimes
        .iter()
        .map(|runtime| runtime.queued_tasks)
        .sum();
    let awaiting_result_count = status
        .agent_runtimes
        .iter()
        .map(|runtime| runtime.awaiting_results)
        .sum();
    let failed_channel_count = status
        .channel_runtimes
        .iter()
        .filter(|runtime| runtime.state == "failed")
        .count();

    ControlHealth {
        connection_kind,
        target: match connection_kind {
            ConnectionKind::Local => status.endpoint.clone(),
            ConnectionKind::Remote => target,
        },
        ready: status.registry.issues.is_empty() && failed_channel_count == 0,
        version: handshake.version,
        protocol_version: handshake.protocol_version,
        transport: handshake.transport,
        wire_format: handshake.wire_format,
        issue_count: status.registry.issues.len(),
        agent_count: status.registry.agents.len(),
        harness_count: status.registry.shared_harnesses.len(),
        channel_count: status.registry.channels.len(),
        running_agent_count,
        active_task_count,
        queued_task_count,
        awaiting_result_count,
        channel_runtime_count: status.channel_runtimes.len(),
        failed_channel_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn remote_env_requires_set_variable() {
        let spec = ConnectionSpec::RemoteEnv {
            base_url: "http://127.0.0.1:9324".into(),
            auth_token_env: "TURIN_CONTROL_CLIENT_TEST_TOKEN_MISSING".into(),
        };
        let err = ControlClient::connect(&spec)
            .await
            .expect_err("missing env rejected");
        assert!(err.to_string().contains("is not set"));
    }
}
