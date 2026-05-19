use anyhow::{Context, Result, anyhow};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::time::sleep;
use turin_daemon_protocol::{
    ChannelRunnerHeartbeatParams, ChannelRunnerHelloParams, DAEMON_PROTOCOL_VERSION,
    DaemonHandshake, DaemonRequest, EntityIdParams, EventEnvelope, NoParams, RequestEnvelope,
    ResponseEnvelope, RuntimeEventsSubscribeParams, ScheduleCreateParams, ScheduleJobDetail,
    ScheduleJobList, ScheduleJobRunList, ScheduleRunsParams, ScheduleUpdateParams, WorkItemDetail,
    WorkItemList, WorkItemTargetParams, WorklistDetail, WorklistItemsParams, WorklistList,
    WorklistListParams, WorklistTargetParams,
};
use turin_local_ipc::{
    LocalIpcReadHalf, connect as connect_local_ipc, current_transport_name,
    resolve_endpoint as resolve_local_ipc_endpoint, split as split_local_ipc,
};
use turin_types::layout::{
    DEFAULT_LAYOUT_DAEMON_SOCKET, config_dir, config_workspace_anchor, resolve_relative_to,
};

#[derive(Debug, Clone)]
pub struct DaemonClient {
    endpoint: PathBuf,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DaemonHealthState {
    Ready,
    Degraded,
}

#[derive(Debug, Clone, Serialize)]
pub struct DaemonHealth {
    pub state: DaemonHealthState,
    pub ready: bool,
    pub endpoint: String,
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

#[derive(Debug, Deserialize)]
struct DaemonStatusSnapshot {
    endpoint: String,
    registry: RegistrySnapshot,
    agent_runtimes: Vec<AgentRuntimeSnapshot>,
    #[serde(default)]
    channel_runtimes: Vec<ChannelRuntimeSnapshot>,
}

#[derive(Debug, Deserialize)]
struct RegistrySnapshot {
    #[serde(default)]
    agents: Vec<Value>,
    #[serde(default)]
    shared_harnesses: Vec<Value>,
    #[serde(default)]
    channels: Vec<Value>,
    #[serde(default)]
    issues: Vec<Value>,
}

#[derive(Debug, Deserialize)]
struct AgentRuntimeSnapshot {
    running: bool,
    active_tasks: usize,
    queued_tasks: usize,
    awaiting_results: usize,
}

#[derive(Debug, Deserialize)]
struct ChannelRuntimeSnapshot {
    state: String,
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

impl DaemonClient {
    pub fn new(endpoint: impl Into<PathBuf>) -> Self {
        Self {
            endpoint: endpoint.into(),
        }
    }

    pub async fn from_config(config_path: impl AsRef<Path>) -> Result<Self> {
        let config_path = config_path.as_ref();
        let raw = tokio::fs::read_to_string(config_path)
            .await
            .with_context(|| format!("Failed to read '{}'", config_path.display()))?;
        let value: toml::Value = toml::from_str(&raw)
            .with_context(|| format!("Failed to parse '{}'", config_path.display()))?;
        let workspace_root = value
            .get("kernel")
            .and_then(|k| k.get("workspace_root"))
            .and_then(|v| v.as_str())
            .unwrap_or(".");
        if let Some(endpoint) = value
            .get("daemon")
            .and_then(|d| d.get("endpoint"))
            .and_then(|v| v.as_str())
        {
            return Ok(Self::new(resolve_local_ipc_endpoint(
                config_path.parent().unwrap_or(Path::new(".")),
                workspace_root,
                endpoint,
            )));
        }

        let config_dir = config_dir(config_path);
        let layout_root = value
            .get("layout")
            .and_then(|layout| layout.get("root"))
            .and_then(|v| v.as_str())
            .map(Path::new)
            .map(|path| resolve_relative_to(&config_workspace_anchor(&config_dir), path))
            .unwrap_or(config_dir);
        let daemon_socket = value
            .get("layout")
            .and_then(|layout| layout.get("daemon_socket"))
            .and_then(|v| v.as_str())
            .unwrap_or(DEFAULT_LAYOUT_DAEMON_SOCKET);
        Ok(Self::new(resolve_relative_to(
            &layout_root,
            Path::new(daemon_socket),
        )))
    }

    pub fn endpoint(&self) -> &Path {
        &self.endpoint
    }

    pub async fn send(&self, request: RequestEnvelope) -> Result<ResponseEnvelope> {
        let mut stream = connect_local_ipc(&self.endpoint)
            .await
            .with_context(|| format!("Failed to connect to '{}'", self.endpoint.display()))?;
        let body = serde_json::to_string(&request)?;
        stream.write_all(body.as_bytes()).await?;
        stream.write_all(b"\n").await?;

        let (reader, _) = split_local_ipc(stream);
        let mut lines = BufReader::new(reader).lines();
        let line = lines
            .next_line()
            .await?
            .ok_or_else(|| anyhow!("Daemon closed connection before response"))?;
        serde_json::from_str(&line).context("Failed to decode daemon response")
    }

    pub async fn request(
        &self,
        id: Option<String>,
        request: DaemonRequest,
    ) -> Result<ResponseEnvelope> {
        self.send(RequestEnvelope::new(id, request)).await
    }

    pub async fn request_ok<T: DeserializeOwned>(
        &self,
        id: Option<String>,
        request: DaemonRequest,
    ) -> Result<T> {
        let response = self.request(id, request).await?;
        decode_ok(response)
    }

    pub async fn handshake(&self) -> Result<DaemonHandshake> {
        let handshake: DaemonHandshake = self
            .request_ok(None, DaemonRequest::DaemonPing(NoParams::default()))
            .await?;
        ensure_compatible_handshake(&handshake)?;
        Ok(handshake)
    }

    pub async fn channel_runner_hello(&self, params: ChannelRunnerHelloParams) -> Result<()> {
        let _: Value = self
            .request_ok(None, DaemonRequest::ChannelRunnerHello(params))
            .await?;
        Ok(())
    }

    pub async fn channel_runner_heartbeat(
        &self,
        params: ChannelRunnerHeartbeatParams,
    ) -> Result<()> {
        let _: Value = self
            .request_ok(None, DaemonRequest::ChannelRunnerHeartbeat(params))
            .await?;
        Ok(())
    }

    pub async fn health(&self) -> Result<DaemonHealth> {
        let handshake = self.handshake().await?;
        let status: DaemonStatusSnapshot = self
            .request_ok(None, DaemonRequest::DaemonStatus(NoParams::default()))
            .await?;

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
        let state = if status.registry.issues.is_empty() && failed_channel_count == 0 {
            DaemonHealthState::Ready
        } else {
            DaemonHealthState::Degraded
        };

        Ok(DaemonHealth {
            ready: true,
            endpoint: status.endpoint,
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
            state,
        })
    }

    pub async fn schedule_create(&self, params: ScheduleCreateParams) -> Result<ScheduleJobDetail> {
        self.request_ok(None, DaemonRequest::ScheduleCreate(params))
            .await
    }

    pub async fn schedule_get(&self, id: impl Into<String>) -> Result<ScheduleJobDetail> {
        self.request_ok(
            None,
            DaemonRequest::ScheduleGet(EntityIdParams { id: id.into() }),
        )
        .await
    }

    pub async fn schedule_update(&self, params: ScheduleUpdateParams) -> Result<ScheduleJobDetail> {
        self.request_ok(None, DaemonRequest::ScheduleUpdate(params))
            .await
    }

    pub async fn schedule_list(&self) -> Result<Vec<ScheduleJobDetail>> {
        let response: ScheduleJobList = self
            .request_ok(None, DaemonRequest::ScheduleList(NoParams::default()))
            .await?;
        Ok(response.jobs)
    }

    pub async fn schedule_runs(
        &self,
        id: impl Into<String>,
        active_only: bool,
        limit: Option<u32>,
    ) -> Result<ScheduleJobRunList> {
        self.request_ok(
            None,
            DaemonRequest::ScheduleRuns(ScheduleRunsParams {
                id: id.into(),
                active_only,
                limit,
            }),
        )
        .await
    }

    pub async fn schedule_enable(&self, id: impl Into<String>) -> Result<ScheduleJobDetail> {
        self.request_ok(
            None,
            DaemonRequest::ScheduleEnable(EntityIdParams { id: id.into() }),
        )
        .await
    }

    pub async fn schedule_disable(&self, id: impl Into<String>) -> Result<ScheduleJobDetail> {
        self.request_ok(
            None,
            DaemonRequest::ScheduleDisable(EntityIdParams { id: id.into() }),
        )
        .await
    }

    pub async fn schedule_delete(&self, id: impl Into<String>) -> Result<ScheduleJobDetail> {
        self.request_ok(
            None,
            DaemonRequest::ScheduleDelete(EntityIdParams { id: id.into() }),
        )
        .await
    }

    pub async fn worklist_list(&self, params: WorklistListParams) -> Result<Vec<WorklistDetail>> {
        let response: WorklistList = self
            .request_ok(None, DaemonRequest::WorklistList(params))
            .await?;
        Ok(response.worklists)
    }

    pub async fn worklist_get(
        &self,
        id: impl Into<String>,
        persistence: Option<turin_daemon_protocol::ContextPersistenceParams>,
    ) -> Result<WorklistDetail> {
        self.request_ok(
            None,
            DaemonRequest::WorklistGet(WorklistTargetParams {
                id: id.into(),
                persistence,
            }),
        )
        .await
    }

    pub async fn worklist_items(&self, params: WorklistItemsParams) -> Result<WorkItemList> {
        self.request_ok(None, DaemonRequest::WorklistItems(params))
            .await
    }

    pub async fn workitem_get(
        &self,
        id: impl Into<String>,
        persistence: Option<turin_daemon_protocol::ContextPersistenceParams>,
    ) -> Result<WorkItemDetail> {
        self.request_ok(
            None,
            DaemonRequest::WorkItemGet(WorkItemTargetParams {
                id: id.into(),
                persistence,
            }),
        )
        .await
    }

    pub async fn wait_until_ready(
        &self,
        timeout: Duration,
        poll_interval: Duration,
    ) -> Result<DaemonHandshake> {
        let deadline = tokio::time::Instant::now() + timeout;
        let poll_interval = poll_interval.max(Duration::from_millis(10));

        loop {
            match self.handshake().await {
                Ok(handshake) => return Ok(handshake),
                Err(err) if is_recoverable_subscription_error(&err) => {
                    if tokio::time::Instant::now() >= deadline {
                        return Err(err);
                    }
                }
                Err(err) => return Err(err),
            }

            sleep(poll_interval).await;
        }
    }

    pub async fn subscribe(
        &self,
        id: Option<String>,
        filter: RuntimeEventsSubscribeParams,
    ) -> Result<EventStream> {
        let mut stream = connect_local_ipc(&self.endpoint)
            .await
            .with_context(|| format!("Failed to connect to '{}'", self.endpoint.display()))?;
        let request = RequestEnvelope::new(id, DaemonRequest::RuntimeEventsSubscribe(filter));
        let body = serde_json::to_string(&request)?;
        stream.write_all(body.as_bytes()).await?;
        stream.write_all(b"\n").await?;

        let (reader, _) = split_local_ipc(stream);
        let mut lines = BufReader::new(reader).lines();
        let ack_line = lines
            .next_line()
            .await?
            .ok_or_else(|| anyhow!("Daemon closed connection before subscription ack"))?;
        let ack: ResponseEnvelope =
            serde_json::from_str(&ack_line).context("Failed to decode subscription ack")?;
        if !ack.ok {
            return Err(anyhow!(format_error(&ack)));
        }
        Ok(EventStream { lines })
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
        let stream = self.subscribe(None, filter.clone()).await?;
        Ok(ManagedEventStream {
            client: self.clone(),
            filter,
            options,
            stream: Some(stream),
        })
    }
}

pub struct EventStream {
    lines: tokio::io::Lines<BufReader<LocalIpcReadHalf>>,
}

impl EventStream {
    pub async fn next(&mut self) -> Result<Option<EventEnvelope>> {
        match self.lines.next_line().await? {
            Some(line) => Ok(Some(
                serde_json::from_str(&line).context("Failed to decode daemon event")?,
            )),
            None => Ok(None),
        }
    }
}

pub struct ManagedEventStream {
    client: DaemonClient,
    filter: RuntimeEventsSubscribeParams,
    options: ManagedSubscribeOptions,
    stream: Option<EventStream>,
}

impl ManagedEventStream {
    pub async fn next_event(&mut self) -> Result<EventEnvelope> {
        loop {
            if self.stream.is_none() {
                self.stream =
                    Some(reconnect(self.client.clone(), self.filter.clone(), self.options).await?);
            }

            match self
                .stream
                .as_mut()
                .expect("managed stream set before polling")
                .next()
                .await
            {
                Ok(Some(event)) => return Ok(event),
                Ok(None) => {
                    self.stream = None;
                }
                Err(err) if is_recoverable_subscription_error(&err) => {
                    self.stream = None;
                }
                Err(err) => return Err(err),
            }
        }
    }
}

async fn reconnect(
    client: DaemonClient,
    filter: RuntimeEventsSubscribeParams,
    options: ManagedSubscribeOptions,
) -> Result<EventStream> {
    let mut delay = options.initial_backoff;
    loop {
        match client.handshake().await {
            Ok(_) => {}
            Err(err) if is_recoverable_subscription_error(&err) => {
                sleep(delay).await;
                delay = next_backoff(delay, options.max_backoff);
                continue;
            }
            Err(err) => return Err(err),
        }

        match client.subscribe(None, filter.clone()).await {
            Ok(stream) => return Ok(stream),
            Err(err) if is_recoverable_subscription_error(&err) => {
                sleep(delay).await;
                delay = next_backoff(delay, options.max_backoff);
            }
            Err(err) => return Err(err),
        }
    }
}

pub fn decode_ok<T: DeserializeOwned>(response: ResponseEnvelope) -> Result<T> {
    if !response.ok {
        return Err(anyhow!(format_error(&response)));
    }
    let result = response
        .result
        .ok_or_else(|| anyhow!("Daemon response missing result payload"))?;
    serde_json::from_value(result).context("Failed to decode daemon result payload")
}

pub fn encode_params<T: Serialize>(value: T) -> Value {
    serde_json::to_value(value).expect("daemon params must serialize")
}

pub fn ensure_compatible_handshake(handshake: &DaemonHandshake) -> Result<()> {
    if handshake.protocol_version != DAEMON_PROTOCOL_VERSION {
        return Err(anyhow!(
            "Unsupported daemon protocol version {} (client expects {})",
            handshake.protocol_version,
            DAEMON_PROTOCOL_VERSION
        ));
    }
    if handshake.transport != current_transport_name() {
        return Err(anyhow!(
            "Unsupported daemon transport '{}' (client expects '{}')",
            handshake.transport,
            current_transport_name()
        ));
    }
    Ok(())
}

fn is_recoverable_subscription_error(err: &anyhow::Error) -> bool {
    if err.chain().any(|cause| cause.is::<std::io::Error>()) {
        return true;
    }

    let message = err.to_string();
    message.contains("Failed to connect to")
        || message.contains("Daemon closed connection before response")
        || message.contains("Daemon closed connection before subscription ack")
}

fn next_backoff(current: Duration, max: Duration) -> Duration {
    current.checked_mul(2).unwrap_or(max).min(max)
}

fn format_error(response: &ResponseEnvelope) -> String {
    match &response.error {
        Some(err) => match &err.details {
            Some(details) => format!("{}: {} ({})", err.code, err.message, details),
            None => format!("{}: {}", err.code, err.message),
        },
        None => "daemon request failed without error envelope".to_string(),
    }
}

#[cfg(test)]
mod tests;
