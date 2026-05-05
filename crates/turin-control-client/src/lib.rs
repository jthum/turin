use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::Duration;
use turin_channel_core::ChannelAdapterManifest;
use turin_daemon_client::DaemonClient;
use turin_daemon_protocol::{
    ChannelAccessParams, ChannelAccessRoomParams, DaemonHandshake, DaemonRequest, EntityIdParams,
    EventEnvelope, LiveSessionTargetParams, NoParams, OpenSessionParams, PromoteTaskParams,
    RequestEnvelope, ResponseEnvelope, ResumeSessionParams, RuntimeEventsSubscribeParams,
    ScheduleCreateParams, ScheduleJobDetail, ScheduleJobList, ScheduleUpdateParams,
    SessionBranchCheckoutParams, SessionBranchCreateParams, SessionBranchSiblingsParams,
    SessionIdParams, SessionListParams, SessionSearchHitKind, SessionSearchParams,
    SessionSearchScope, SessionTitleParams, SidestepContextTargetParams, SidestepModeParams,
    SidestepTaskParams, SubmitTaskParams, TaskIdParams, UpdateChannelParams, WaitTaskParams,
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
    pub harness: Option<String>,
    pub runtime_idle_secs: Option<u64>,
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
    pub execution: LiveExecution,
    pub status: Option<String>,
    pub task_turn_count: Option<u32>,
    pub branch_outcome: Option<serde_json::Value>,
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
pub struct SessionBranchDetail {
    pub branch_id: String,
    pub name: String,
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
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionBranchList {
    branches: Vec<SessionBranchDetail>,
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

    pub async fn create_schedule(&self, params: ScheduleCreateParams) -> Result<ScheduleJobDetail> {
        self.request_ok(None, DaemonRequest::ScheduleCreate(params))
            .await
    }

    pub async fn get_schedule(&self, id: impl Into<String>) -> Result<ScheduleJobDetail> {
        self.request_ok(
            None,
            DaemonRequest::ScheduleGet(EntityIdParams { id: id.into() }),
        )
        .await
    }

    pub async fn update_schedule(&self, params: ScheduleUpdateParams) -> Result<ScheduleJobDetail> {
        self.request_ok(None, DaemonRequest::ScheduleUpdate(params))
            .await
    }

    pub async fn list_schedules(&self) -> Result<Vec<ScheduleJobDetail>> {
        let response: ScheduleJobList = self
            .request_ok(None, DaemonRequest::ScheduleList(NoParams::default()))
            .await?;
        Ok(response.jobs)
    }

    pub async fn enable_schedule(&self, id: impl Into<String>) -> Result<ScheduleJobDetail> {
        self.request_ok(
            None,
            DaemonRequest::ScheduleEnable(EntityIdParams { id: id.into() }),
        )
        .await
    }

    pub async fn disable_schedule(&self, id: impl Into<String>) -> Result<ScheduleJobDetail> {
        self.request_ok(
            None,
            DaemonRequest::ScheduleDisable(EntityIdParams { id: id.into() }),
        )
        .await
    }

    pub async fn delete_schedule(&self, id: impl Into<String>) -> Result<ScheduleJobDetail> {
        self.request_ok(
            None,
            DaemonRequest::ScheduleDelete(EntityIdParams { id: id.into() }),
        )
        .await
    }

    pub async fn list_sessions(&self, limit: usize, offset: usize) -> Result<Vec<SessionSummary>> {
        self.list_sessions_in(limit, offset, None, None).await
    }

    pub async fn list_sessions_in(
        &self,
        limit: usize,
        offset: usize,
        store: Option<&str>,
        path: Option<&str>,
    ) -> Result<Vec<SessionSummary>> {
        let response: SessionList = self
            .request_ok(
                None,
                DaemonRequest::SessionList(SessionListParams {
                    limit,
                    offset,
                    store: store.map(str::to_string),
                    path: path.map(str::to_string),
                }),
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
        self.search_sessions_in(query, scope, limit, offset, None, None)
            .await
    }

    pub async fn search_sessions_in(
        &self,
        query: &str,
        scope: SessionSearchScope,
        limit: usize,
        offset: usize,
        store: Option<&str>,
        path: Option<&str>,
    ) -> Result<Vec<SessionSearchHit>> {
        let response: SessionSearchResultList = self
            .request_ok(
                None,
                DaemonRequest::SessionSearch(SessionSearchParams {
                    query: query.to_string(),
                    scope: Some(scope),
                    limit,
                    offset,
                    store: store.map(str::to_string),
                    path: path.map(str::to_string),
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

    pub async fn list_session_branches(
        &self,
        session_id: &str,
    ) -> Result<Vec<SessionBranchDetail>> {
        let response: SessionBranchList = self
            .request_ok(
                None,
                DaemonRequest::SessionBranchList(SessionIdParams {
                    session_id: session_id.to_string(),
                }),
            )
            .await?;
        Ok(response.branches)
    }

    pub async fn create_session_branch(
        &self,
        session_id: &str,
        name: &str,
        from_turn_index: Option<u32>,
        activate: bool,
    ) -> Result<SessionBranchDetail> {
        self.create_session_branch_in_slot(session_id, None, name, from_turn_index, activate)
            .await
    }

    pub async fn create_session_branch_in_slot(
        &self,
        session_id: &str,
        slot_id: Option<String>,
        name: &str,
        from_turn_index: Option<u32>,
        activate: bool,
    ) -> Result<SessionBranchDetail> {
        self.request_ok(
            None,
            DaemonRequest::SessionBranchCreate(SessionBranchCreateParams {
                session_id: session_id.to_string(),
                name: name.to_string(),
                slot_id,
                from_turn_index,
                activate,
            }),
        )
        .await
    }

    pub async fn checkout_session_branch(
        &self,
        session_id: &str,
        branch: &str,
    ) -> Result<SessionBranchDetail> {
        self.checkout_session_branch_in_slot(session_id, None, branch)
            .await
    }

    pub async fn checkout_session_branch_in_slot(
        &self,
        session_id: &str,
        slot_id: Option<String>,
        branch: &str,
    ) -> Result<SessionBranchDetail> {
        self.request_ok(
            None,
            DaemonRequest::SessionBranchCheckout(SessionBranchCheckoutParams {
                session_id: session_id.to_string(),
                branch: branch.to_string(),
                slot_id,
            }),
        )
        .await
    }

    pub async fn list_session_branch_siblings(
        &self,
        session_id: &str,
        source_turn_id: i64,
    ) -> Result<Vec<SessionBranchDetail>> {
        let response: SessionBranchList = self
            .request_ok(
                None,
                DaemonRequest::SessionBranchSiblings(SessionBranchSiblingsParams {
                    session_id: session_id.to_string(),
                    source_turn_id,
                }),
            )
            .await?;
        Ok(response.branches)
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
                channel_id: None,
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

    pub async fn cancel_live_session(
        &self,
        session_id: &str,
        slot_id: Option<String>,
    ) -> Result<SessionActionResult> {
        self.request_ok(
            None,
            DaemonRequest::SessionCancel(LiveSessionTargetParams {
                session_id: session_id.to_string(),
                slot_id,
            }),
        )
        .await
    }

    pub async fn cancel_session(&self, session_id: &str) -> Result<SessionActionResult> {
        self.cancel_live_session(session_id, None).await
    }

    pub async fn kill_live_session(
        &self,
        session_id: &str,
        slot_id: Option<String>,
    ) -> Result<SessionActionResult> {
        self.request_ok(
            None,
            DaemonRequest::SessionKill(LiveSessionTargetParams {
                session_id: session_id.to_string(),
                slot_id,
            }),
        )
        .await
    }

    pub async fn kill_session(&self, session_id: &str) -> Result<SessionActionResult> {
        self.kill_live_session(session_id, None).await
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

    pub async fn submit_task_in_slot(
        &self,
        agent_id: Option<String>,
        session_id: Option<String>,
        slot_id: Option<String>,
        prompt: String,
    ) -> Result<TaskStatus> {
        self.submit_task_in_slot_with_conflict_policy(agent_id, session_id, slot_id, prompt, None)
            .await
    }

    pub async fn submit_task_in_slot_with_conflict_policy(
        &self,
        agent_id: Option<String>,
        session_id: Option<String>,
        slot_id: Option<String>,
        prompt: String,
        conflict_policy: Option<String>,
    ) -> Result<TaskStatus> {
        self.request_ok(
            None,
            DaemonRequest::TaskSubmit(SubmitTaskParams {
                agent_id,
                session_id,
                slot_id,
                prompt,
                content: None,
                tools: None,
                conflict_policy,
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
        self.submit_task_in_slot(agent_id, session_id, None, prompt)
            .await
    }

    pub async fn sidestep_task(
        &self,
        session_id: String,
        slot_id: Option<String>,
        prompt: String,
        mode: SidestepModeParams,
        context_target: Option<SidestepContextTargetParams>,
        timeout_ms: Option<u64>,
    ) -> Result<TaskStatus> {
        self.request_ok(
            None,
            DaemonRequest::TaskSidestep(SidestepTaskParams {
                session_id,
                slot_id,
                prompt,
                content: None,
                tools: None,
                mode,
                context_target,
                timeout_ms,
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

    pub async fn promote_task(
        &self,
        request_id: &str,
        branch_name: Option<String>,
    ) -> Result<SessionBranchDetail> {
        self.request_ok(
            None,
            DaemonRequest::TaskPromote(PromoteTaskParams {
                request_id: request_id.to_string(),
                branch_name,
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

    pub async fn update_channel_settings(
        &self,
        channel_id: &str,
        settings: Value,
    ) -> Result<ChannelDetail> {
        self.request_ok(
            None,
            DaemonRequest::ChannelUpdate(UpdateChannelParams {
                id: channel_id.to_string(),
                kind: None,
                agent_id: None,
                idle_ttl_secs: None,
                settings: Some(settings),
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
