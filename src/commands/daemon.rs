mod agents;
mod channels;
mod client;
mod control;
mod harnesses;
mod render;
mod sessions;
mod tasks;

pub use self::agents::*;
pub use self::channels::*;
use self::client::*;
pub use self::control::*;
pub use self::harnesses::*;
use self::render::*;
pub use self::sessions::*;
pub use self::tasks::*;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::Path;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use turin_channel_core::ChannelAdapterManifest;
use turin_local_ipc::{connect as connect_local_ipc, split as split_local_ipc};

use turin::daemon::protocol::{
    DaemonRequest, ErrorCode, ErrorEnvelope, EventEnvelope, RequestEnvelope, ResponseEnvelope,
    RuntimeEventsSubscribeParams,
};

#[derive(Debug, Deserialize)]
struct DaemonStatusView {
    config_path: String,
    workspace_root: String,
    endpoint: String,
    registry: RegistrySnapshotView,
    harnesses: Vec<HarnessRuntimeView>,
    agent_runtimes: Vec<AgentRuntimeView>,
}

#[derive(Debug, Deserialize)]
struct RegistrySnapshotView {
    agents: Vec<AgentSummaryView>,
    shared_harnesses: Vec<SharedHarnessView>,
    channels: Vec<ChannelSummaryView>,
    issues: Vec<IssueView>,
}

#[derive(Debug, Deserialize)]
struct AgentSummaryView {
    id: String,
    enabled: bool,
    provider: String,
    model: String,
    harness_ref: String,
}

#[derive(Debug, Deserialize)]
struct SharedHarnessView {
    id: String,
}

#[derive(Debug, Deserialize)]
struct ChannelSummaryView {
    id: String,
    enabled: bool,
    kind: String,
    agent_id: String,
}

#[derive(Debug, Deserialize)]
struct HarnessRuntimeView {
    harness_id: String,
    bound_agents: Vec<String>,
    watched_roots: Vec<String>,
    loaded_scripts: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct AgentRuntimeView {
    agent_id: String,
    running: bool,
    active_tasks: usize,
    queued_tasks: usize,
    awaiting_results: usize,
    current_session_id: Option<String>,
    current_request_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct IssueView {
    path: String,
    message: String,
}

#[derive(Debug, Deserialize)]
struct IssueListView {
    issues: Vec<IssueView>,
}

#[derive(Debug, Deserialize)]
struct AgentDetailView {
    id: String,
    directory: String,
    enabled: bool,
    provider: String,
    model: String,
    system_prompt: Option<String>,
    mode: Option<String>,
    harness: Option<String>,
    idle_grace_secs: Option<u64>,
    has_local_harness: bool,
}

#[derive(Debug, Deserialize)]
struct HarnessDetailView {
    harness_id: String,
    directory: String,
    bound_agents: Vec<String>,
    watched_roots: Vec<String>,
    loaded_scripts: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ChannelDetailView {
    id: String,
    directory: String,
    enabled: bool,
    kind: String,
    agent_id: String,
    idle_ttl_secs: Option<u64>,
    settings: Value,
    #[serde(default)]
    adapter: Option<ChannelAdapterManifest>,
}

#[derive(Debug, Deserialize)]
struct ChannelRunnerHandshakeView {
    display_name: String,
    protocol_version: u32,
    runner_binary: Option<String>,
    runner_version: Option<String>,
    pid: Option<u32>,
    last_handshake_unix_ms: u64,
}

#[derive(Debug, Deserialize)]
struct ChannelRuntimeView {
    id: String,
    kind: String,
    agent_id: String,
    directory: String,
    state: String,
    last_error: Option<String>,
    last_error_code: Option<String>,
    start_count: u64,
    restart_count: u64,
    failure_count: u64,
    last_transition_unix_ms: u64,
    last_started_unix_ms: Option<u64>,
    last_stopped_unix_ms: Option<u64>,
    handshake: Option<ChannelRunnerHandshakeView>,
}

#[derive(Debug, Deserialize)]
struct ChannelAccessRoomView {
    channel: String,
    workspace_id: String,
    room_id: Option<String>,
    thread_id: String,
}

#[derive(Debug, Deserialize)]
struct ApprovedRoomView {
    room: ChannelAccessRoomView,
    approved_at_unix_secs: u64,
    approved_by_user_id: Option<String>,
    approved_by_username: Option<String>,
}

#[derive(Debug, Deserialize)]
struct PendingRoomView {
    room: ChannelAccessRoomView,
    first_seen_unix_secs: u64,
    last_seen_unix_secs: u64,
    sample_user_id: Option<String>,
    sample_username: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ChannelAccessView {
    approved_rooms: Vec<ApprovedRoomView>,
    pending_rooms: Vec<PendingRoomView>,
}

#[derive(Debug, Deserialize)]
struct TaskStatusView {
    request_id: String,
    agent_id: String,
    slot_id: String,
    trace_id: String,
    state: String,
    runtime_task_id: Option<String>,
    execution: LiveExecutionView,
    status: Option<String>,
    task_turn_count: Option<u32>,
    branch_outcome: Option<Value>,
    output: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TaskListView {
    tasks: Vec<TaskStatusView>,
}

#[derive(Debug, Deserialize)]
struct SessionSummaryView {
    internal_id: i64,
    session_id: String,
    agent_id: String,
    metadata: Option<Value>,
    created_at: String,
}

#[derive(Debug, Deserialize)]
struct SessionListView {
    sessions: Vec<SessionSummaryView>,
}

#[derive(Debug, Deserialize)]
struct LiveSessionView {
    agent_id: String,
    slot_id: String,
    session_id: String,
    running: bool,
    active_tasks: usize,
    queued_tasks: usize,
    current_request_id: Option<String>,
    execution: LiveExecutionView,
    conflict_policy: String,
}

#[derive(Debug, Deserialize)]
struct LiveExecutionView {
    execution_id: String,
    context_target: Value,
    visibility: String,
    durability: String,
    write_policy: String,
}

#[derive(Debug, Deserialize)]
struct LiveSessionListView {
    sessions: Vec<LiveSessionView>,
}

#[derive(Debug, Deserialize)]
struct SessionEventDetailView {
    id: i64,
    event_type: String,
    payload: Value,
    created_at: String,
}

#[derive(Debug, Deserialize)]
struct SessionMessageDetailView {
    id: i64,
    turn_index: u32,
    role: String,
    content: Value,
    token_count: Option<u64>,
    created_at: String,
}

#[derive(Debug, Deserialize)]
struct SessionToolExecutionDetailView {
    id: i64,
    turn_index: u32,
    tool_call_id: String,
    tool_name: String,
    args: Value,
    output: Option<Value>,
    is_error: bool,
    duration_ms: Option<u64>,
    verdict: String,
    created_at: String,
}

#[derive(Debug, Deserialize)]
struct SessionDetailView {
    session: SessionSummaryView,
    #[serde(default)]
    branches: Vec<SessionBranchDetailView>,
    events: Vec<SessionEventDetailView>,
    messages: Vec<SessionMessageDetailView>,
    tool_executions: Vec<SessionToolExecutionDetailView>,
}

#[derive(Debug, Deserialize)]
struct SessionBranchDetailView {
    branch_id: String,
    name: String,
    head_turn_index: Option<u32>,
    active: bool,
    created_at: String,
}

#[derive(Debug, Deserialize)]
struct SessionBranchListView {
    branches: Vec<SessionBranchDetailView>,
}

#[derive(Debug, Serialize)]
struct DaemonHealthReport {
    state: String,
    ready: bool,
    endpoint: String,
    error: Option<String>,
    version: Option<String>,
    protocol_version: Option<u32>,
    transport: Option<String>,
    wire_format: Option<String>,
    issue_count: usize,
    agent_count: usize,
    harness_count: usize,
    channel_count: usize,
    running_agent_count: usize,
    active_task_count: usize,
    queued_task_count: usize,
    awaiting_result_count: usize,
    channel_runtime_count: usize,
    failed_channel_count: usize,
}

#[derive(Debug, Serialize)]
struct DaemonStartReport {
    started: bool,
    endpoint: String,
    log_path: String,
    health: DaemonHealthReport,
}

#[derive(Debug, Serialize)]
struct DaemonLogReport {
    path: String,
    exists: bool,
    lines: Vec<String>,
}
