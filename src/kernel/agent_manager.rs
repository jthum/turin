mod peer_runtime;
mod runtime_registry;

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::RwLock as StdRwLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::inference::embeddings::EmbeddingProvider;
use crate::inference::provider::ProviderClient;
use crate::kernel::config::TurinConfig;
use crate::kernel::event::TaskTerminalStatus;
use crate::kernel::governance::GovernanceManager;
use crate::kernel::harness_manager::HarnessManager;
use crate::kernel::policy::RuntimePolicyManager;
use crate::kernel::session::QueuedTask;
use crate::persistence::manager::StoreManager;
use crate::tools::registry::ToolRegistry;
use anyhow::Result;
use tokio::sync::{Notify, RwLock, oneshot};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, serde::Serialize)]
pub struct PeerAgentTaskResult {
    pub request_id: String,
    pub agent_id: String,
    pub runtime_task_id: String,
    pub status: TaskTerminalStatus,
    pub task_turn_count: u32,
    pub output: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AgentStatusSnapshot {
    pub agent_id: String,
    pub running: bool,
    pub active_tasks: usize,
    pub queued_tasks: usize,
    pub awaiting_results: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct TaskStatusSnapshot {
    pub request_id: String,
    pub agent_id: String,
    pub state: String,
    pub runtime_task_id: Option<String>,
    pub status: Option<TaskTerminalStatus>,
    pub task_turn_count: Option<u32>,
    pub output: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingTaskState {
    Queued,
    Running,
    Cancelling,
}

#[derive(Debug, Clone)]
struct PendingTaskRecord {
    agent_id: String,
    state: PendingTaskState,
    runtime_task_id: Option<String>,
}

#[derive(Default)]
pub(crate) struct RuntimeControl {
    current_session_id: StdRwLock<Option<String>>,
    current_request_id: StdRwLock<Option<String>>,
    current_runtime_task_id: StdRwLock<Option<String>>,
    current_cancel_token: Mutex<Option<CancellationToken>>,
    session_reset_requested: AtomicBool,
}

impl RuntimeControl {
    fn set_current_session_id(&self, session_id: Option<String>) {
        *self
            .current_session_id
            .write()
            .expect("runtime control session lock poisoned") = session_id;
    }

    fn current_session_id(&self) -> Option<String> {
        self.current_session_id
            .read()
            .expect("runtime control session lock poisoned")
            .clone()
    }

    fn activate_task(
        &self,
        request_id: Option<String>,
        runtime_task_id: String,
        cancel_token: CancellationToken,
    ) {
        *self
            .current_request_id
            .write()
            .expect("runtime control request lock poisoned") = request_id;
        *self
            .current_runtime_task_id
            .write()
            .expect("runtime control task id lock poisoned") = Some(runtime_task_id);
        *self
            .current_cancel_token
            .lock()
            .expect("runtime control cancel lock poisoned") = Some(cancel_token);
    }

    fn clear_active_task(&self) {
        *self
            .current_request_id
            .write()
            .expect("runtime control request lock poisoned") = None;
        *self
            .current_runtime_task_id
            .write()
            .expect("runtime control task id lock poisoned") = None;
        *self
            .current_cancel_token
            .lock()
            .expect("runtime control cancel lock poisoned") = None;
    }

    fn current_request_id(&self) -> Option<String> {
        self.current_request_id
            .read()
            .expect("runtime control request lock poisoned")
            .clone()
    }

    fn current_runtime_task_id(&self) -> Option<String> {
        self.current_runtime_task_id
            .read()
            .expect("runtime control task id lock poisoned")
            .clone()
    }

    fn request_task_cancel(&self) -> bool {
        let token = self
            .current_cancel_token
            .lock()
            .expect("runtime control cancel lock poisoned")
            .clone();
        if let Some(token) = token {
            token.cancel();
            true
        } else {
            false
        }
    }

    fn request_session_cancel(&self) -> bool {
        self.session_reset_requested.store(true, Ordering::Relaxed);
        self.request_task_cancel()
    }

    fn take_session_reset_requested(&self) -> bool {
        self.session_reset_requested.swap(false, Ordering::Relaxed)
    }
}

struct PeerAgentTaskEnvelope {
    task: QueuedTask,
    request_id: Option<String>,
    result_tx: Option<oneshot::Sender<PeerAgentTaskResult>>,
    delegated_capabilities: Option<BTreeMap<String, bool>>,
}

/// A handle to a running peer agent.
pub struct AgentRuntimeHandle {
    /// Explicit queued envelopes awaiting execution.
    queue: Arc<Mutex<VecDeque<PeerAgentTaskEnvelope>>>,
    /// Notification used to wake the background runtime when new work arrives.
    notify: Arc<Notify>,
    /// Shared execution/session control state for the runtime.
    control: Arc<RuntimeControl>,
    /// The background task running the agent's event loop.
    task: Option<JoinHandle<()>>,
    /// Approximate number of tasks currently queued in the runtime channel.
    queued_tasks: Arc<AtomicUsize>,
    /// Number of tasks currently executing inside the runtime loop.
    active_tasks: Arc<AtomicUsize>,
}

impl AgentRuntimeHandle {
    fn is_running(&self) -> bool {
        self.task
            .as_ref()
            .map(|jh| !jh.is_finished())
            .unwrap_or(false)
    }
}

#[derive(Clone)]
pub(crate) struct SharedPeerRuntimeContext {
    pub(crate) json: bool,
    pub(crate) tool_registry: ToolRegistry,
    pub(crate) policy_manager: Arc<RuntimePolicyManager>,
    pub(crate) governance_manager: Arc<GovernanceManager>,
    pub(crate) harness_manager: Arc<HarnessManager>,
}

#[derive(Clone, Default)]
pub(crate) struct SharedInferenceState {
    pub(crate) clients: HashMap<String, ProviderClient>,
    pub(crate) embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
}

#[derive(Default)]
struct CompletedTaskCache {
    order: VecDeque<String>,
    results: HashMap<String, PeerAgentTaskResult>,
}

impl CompletedTaskCache {
    const MAX_ENTRIES: usize = 1024;

    fn insert(&mut self, result: PeerAgentTaskResult) {
        let request_id = result.request_id.clone();
        if !self.results.contains_key(&request_id) {
            self.order.push_back(request_id.clone());
        }
        self.results.insert(request_id, result);
        while self.order.len() > Self::MAX_ENTRIES {
            if let Some(evicted) = self.order.pop_front() {
                self.results.remove(&evicted);
            }
        }
    }
}

/// Orchestrates peer agents, spinning them up on demand and routing tasks to their independent runtimes.
pub struct AgentManager {
    /// The full configuration, used to look up agent profiles and instantiate kernels.
    config: Arc<TurinConfig>,
    /// Reference to the shared StoreManager for database operations.
    store_manager: Arc<StoreManager>,
    /// The list of active, running agent handles, keyed by agent_id.
    runtimes: RwLock<HashMap<String, Arc<AgentRuntimeHandle>>>,
    /// Task result receivers keyed by runtime request id (consumed by `await_result`).
    pending_results: RwLock<HashMap<String, oneshot::Receiver<PeerAgentTaskResult>>>,
    /// Mapping of request id -> current non-terminal task state for status accounting.
    pending_task_states: RwLock<HashMap<String, PendingTaskRecord>>,
    /// Bounded cache of completed task results for daemon/control-plane inspection.
    completed_results: RwLock<CompletedTaskCache>,
    /// Shared runtime pieces used to fork peer kernels without cloning the whole kernel topology.
    shared_runtime: OnceLock<SharedPeerRuntimeContext>,
    /// Live inference state copied from the root kernel after provider initialization.
    shared_inference: Mutex<SharedInferenceState>,
}

impl AgentManager {
    /// Create a new AgentManager.
    pub fn new(config: Arc<TurinConfig>, store_manager: Arc<StoreManager>) -> Self {
        Self {
            config,
            store_manager,
            runtimes: RwLock::new(HashMap::new()),
            pending_results: RwLock::new(HashMap::new()),
            pending_task_states: RwLock::new(HashMap::new()),
            completed_results: RwLock::new(CompletedTaskCache::default()),
            shared_runtime: OnceLock::new(),
            shared_inference: Mutex::new(SharedInferenceState::default()),
        }
    }

    pub(crate) fn bind_shared_runtime(&self, runtime: SharedPeerRuntimeContext) {
        let _ = self.shared_runtime.set(runtime);
    }

    pub(crate) fn shared_runtime(&self) -> Option<&SharedPeerRuntimeContext> {
        self.shared_runtime.get()
    }

    pub(crate) fn bind_inference_state(
        &self,
        clients: HashMap<String, ProviderClient>,
        embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    ) {
        *self
            .shared_inference
            .lock()
            .expect("agent manager shared inference mutex poisoned") = SharedInferenceState {
            clients,
            embedding_provider,
        };
    }

    /// Dispatch a task to an agent by ID. If the agent isn't running, it will be started automatically.
    pub async fn send(
        self: &Arc<Self>,
        agent_id: &str,
        task: QueuedTask,
        delegated_capabilities: Option<BTreeMap<String, bool>>,
    ) -> Result<()> {
        let handle = self.ensure_runtime(agent_id).await?;
        self.enqueue_runtime_task(
            &handle,
            PeerAgentTaskEnvelope {
                task,
                request_id: None,
                result_tx: None,
                delegated_capabilities,
            },
        )
        .await?;
        Ok(())
    }

    /// Submit a task to a peer agent and return a request ID for later `await_result`.
    pub async fn submit(
        self: &Arc<Self>,
        agent_id: &str,
        task: QueuedTask,
        delegated_capabilities: Option<BTreeMap<String, bool>>,
    ) -> Result<String> {
        let request_id = uuid::Uuid::now_v7().simple().to_string();
        let (tx_result, rx_result) = oneshot::channel();
        {
            let mut pending = self.pending_results.write().await;
            pending.insert(request_id.clone(), rx_result);
        }
        {
            let mut pending = self.pending_task_states.write().await;
            pending.insert(
                request_id.clone(),
                PendingTaskRecord {
                    agent_id: agent_id.to_string(),
                    state: PendingTaskState::Queued,
                    runtime_task_id: None,
                },
            );
        }

        let handle = match self.ensure_runtime(agent_id).await {
            Ok(handle) => handle,
            Err(e) => {
                self.remove_pending_request(&request_id).await;
                return Err(e);
            }
        };

        if let Err(e) = self
            .enqueue_runtime_task(
                &handle,
                PeerAgentTaskEnvelope {
                    task,
                    request_id: Some(request_id.clone()),
                    result_tx: Some(tx_result),
                    delegated_capabilities,
                },
            )
            .await
        {
            self.remove_pending_request(&request_id).await;
            return Err(e);
        }

        Ok(request_id)
    }

    /// Await a previously submitted peer task result.
    pub async fn await_result(
        &self,
        request_id: &str,
        timeout_ms: Option<u64>,
    ) -> Result<PeerAgentTaskResult> {
        if let Some(result) = self.completed_result(request_id).await {
            return Ok(result);
        }

        let mut rx = {
            let mut pending = self.pending_results.write().await;
            pending.remove(request_id).ok_or_else(|| {
                anyhow::anyhow!("Unknown or already-awaited peer task '{}'", request_id)
            })?
        };

        let result = if let Some(ms) = timeout_ms {
            match tokio::time::timeout(std::time::Duration::from_millis(ms), &mut rx).await {
                Ok(Ok(res)) => Ok(res),
                Ok(Err(_)) => Err(anyhow::anyhow!(
                    "Peer task '{}' result channel closed",
                    request_id
                )),
                Err(_) => {
                    self.pending_results
                        .write()
                        .await
                        .insert(request_id.to_string(), rx);
                    Err(anyhow::anyhow!(
                        "Timed out waiting for peer task '{}'",
                        request_id
                    ))
                }
            }
        } else {
            match rx.await {
                Ok(res) => Ok(res),
                Err(_) => Err(anyhow::anyhow!(
                    "Peer task '{}' result channel closed",
                    request_id
                )),
            }
        };

        let result = result?;
        self.record_completed_result(result.clone()).await;
        Ok(result)
    }

    /// List configured agents with runtime status.
    pub async fn list_statuses(&self) -> Vec<AgentStatusSnapshot> {
        let runtimes = self.runtimes.read().await;
        let pending = self.pending_task_states.read().await;
        let mut awaiting_by_agent: HashMap<&str, usize> = HashMap::new();
        for pending in pending.values() {
            *awaiting_by_agent
                .entry(pending.agent_id.as_str())
                .or_default() += 1;
        }

        let mut ids = vec![self.config.agent.id.clone()];
        ids.extend(self.config.agents.keys().cloned());
        ids.sort();
        ids.dedup();

        ids.into_iter()
            .map(|agent_id| {
                let handle = runtimes.get(&agent_id);
                let running = handle.map(|h| h.is_running()).unwrap_or(false);
                let awaiting_results = *awaiting_by_agent.get(agent_id.as_str()).unwrap_or(&0);
                let queued_tasks = handle
                    .map(|h| h.queued_tasks.load(Ordering::Relaxed))
                    .unwrap_or(0);
                let active_tasks = handle
                    .map(|h| h.active_tasks.load(Ordering::Relaxed))
                    .unwrap_or(0);
                AgentStatusSnapshot {
                    agent_id,
                    running,
                    active_tasks,
                    queued_tasks,
                    awaiting_results,
                }
            })
            .collect()
    }

    /// Get status for a single agent.
    pub async fn get_status(&self, agent_id: &str) -> Option<AgentStatusSnapshot> {
        self.list_statuses()
            .await
            .into_iter()
            .find(|s| s.agent_id == agent_id)
    }

    pub async fn list_tasks(&self) -> Vec<TaskStatusSnapshot> {
        let pending = self.pending_task_states.read().await;
        let mut snapshots: Vec<_> = pending
            .iter()
            .map(|(request_id, pending)| TaskStatusSnapshot {
                request_id: request_id.clone(),
                agent_id: pending.agent_id.clone(),
                state: match pending.state {
                    PendingTaskState::Queued => "queued".to_string(),
                    PendingTaskState::Running => "running".to_string(),
                    PendingTaskState::Cancelling => "cancelling".to_string(),
                },
                runtime_task_id: pending.runtime_task_id.clone(),
                status: None,
                task_turn_count: None,
                output: None,
                error: None,
            })
            .collect();
        drop(pending);

        let completed = self.completed_results.read().await;
        snapshots.extend(
            completed
                .results
                .values()
                .cloned()
                .map(|result| TaskStatusSnapshot {
                    request_id: result.request_id,
                    agent_id: result.agent_id,
                    state: "completed".to_string(),
                    runtime_task_id: Some(result.runtime_task_id),
                    status: Some(result.status),
                    task_turn_count: Some(result.task_turn_count),
                    output: result.output,
                    error: result.error,
                }),
        );
        snapshots.sort_by(|a, b| a.request_id.cmp(&b.request_id));
        snapshots
    }

    pub async fn get_task(&self, request_id: &str) -> Option<TaskStatusSnapshot> {
        self.list_tasks()
            .await
            .into_iter()
            .find(|task| task.request_id == request_id)
    }

    async fn enqueue_runtime_task(
        &self,
        handle: &Arc<AgentRuntimeHandle>,
        envelope: PeerAgentTaskEnvelope,
    ) -> Result<()> {
        {
            let mut queue = handle
                .queue
                .lock()
                .expect("agent runtime queue mutex poisoned");
            queue.push_back(envelope);
        }
        handle.queued_tasks.fetch_add(1, Ordering::Relaxed);
        handle.notify.notify_one();
        Ok(())
    }

    async fn remove_pending_request(&self, request_id: &str) {
        self.pending_results.write().await.remove(request_id);
        self.pending_task_states.write().await.remove(request_id);
    }

    pub(crate) async fn record_completed_result(&self, result: PeerAgentTaskResult) {
        self.pending_results
            .write()
            .await
            .remove(&result.request_id);
        self.pending_task_states
            .write()
            .await
            .remove(&result.request_id);
        let mut completed = self.completed_results.write().await;
        completed.insert(result);
    }

    async fn completed_result(&self, request_id: &str) -> Option<PeerAgentTaskResult> {
        self.completed_results
            .read()
            .await
            .results
            .get(request_id)
            .cloned()
    }

    pub(crate) async fn mark_task_running(&self, request_id: &str, runtime_task_id: String) {
        if let Some(pending) = self.pending_task_states.write().await.get_mut(request_id) {
            pending.state = PendingTaskState::Running;
            pending.runtime_task_id = Some(runtime_task_id);
        }
    }

    pub async fn cancel_task(&self, request_id: &str) -> Result<TaskStatusSnapshot> {
        if let Some(result) = self.completed_result(request_id).await {
            anyhow::bail!(
                "Task '{}' is already terminal ({:?})",
                request_id,
                result.status
            );
        }

        let pending = self
            .pending_task_states
            .read()
            .await
            .get(request_id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Task '{}' not found", request_id))?;

        if pending.state == PendingTaskState::Running
            || pending.state == PendingTaskState::Cancelling
        {
            let handle = {
                let runtimes = self.runtimes.read().await;
                runtimes.get(&pending.agent_id).cloned().ok_or_else(|| {
                    anyhow::anyhow!("Agent runtime '{}' is not available", pending.agent_id)
                })?
            };

            let current_request_id = handle.control.current_request_id();
            if current_request_id.as_deref() != Some(request_id) {
                anyhow::bail!("Task '{}' is no longer the active running task", request_id);
            }
            if !handle.control.request_task_cancel() {
                anyhow::bail!(
                    "Task '{}' is running without a cancellable execution token",
                    request_id
                );
            }
            if let Some(pending) = self.pending_task_states.write().await.get_mut(request_id) {
                pending.state = PendingTaskState::Cancelling;
            }
            let snapshot = self.get_task(request_id).await.ok_or_else(|| {
                anyhow::anyhow!("Task '{}' disappeared after cancellation", request_id)
            })?;
            return Ok(snapshot);
        }

        let handle = {
            let runtimes = self.runtimes.read().await;
            runtimes.get(&pending.agent_id).cloned().ok_or_else(|| {
                anyhow::anyhow!("Agent runtime '{}' is not available", pending.agent_id)
            })?
        };

        let removed = {
            let mut queue = handle
                .queue
                .lock()
                .expect("agent runtime queue mutex poisoned");
            let Some(index) = queue
                .iter()
                .position(|envelope| envelope.request_id.as_deref() == Some(request_id))
            else {
                anyhow::bail!("Task '{}' is no longer queued", request_id);
            };
            queue.remove(index)
        };

        let Some(envelope) = removed else {
            anyhow::bail!("Task '{}' is no longer queued", request_id);
        };

        handle.queued_tasks.fetch_sub(1, Ordering::Relaxed);

        let completed = PeerAgentTaskResult {
            request_id: request_id.to_string(),
            agent_id: pending.agent_id.clone(),
            runtime_task_id: String::new(),
            status: TaskTerminalStatus::Cancelled,
            task_turn_count: 0,
            output: None,
            error: Some("Task cancelled before execution".to_string()),
        };

        if let Some(tx_result) = envelope.result_tx {
            let _ = tx_result.send(completed.clone());
        }

        self.record_completed_result(completed.clone()).await;

        Ok(TaskStatusSnapshot {
            request_id: completed.request_id,
            agent_id: completed.agent_id,
            state: "completed".to_string(),
            runtime_task_id: Some(completed.runtime_task_id),
            status: Some(completed.status),
            task_turn_count: Some(completed.task_turn_count),
            output: completed.output,
            error: completed.error,
        })
    }

    pub async fn cancel_session(&self, session_id: &str) -> Result<(String, String)> {
        let (agent_id, handle) =
            self.find_runtime_by_session(session_id)
                .await
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Session '{}' is not an active managed runtime session",
                        session_id
                    )
                })?;

        self.cancel_queued_requests_for_agent(&agent_id, "Session cancelled before execution")
            .await;

        if handle.control.current_request_id().is_none() {
            handle.control.request_session_cancel();
            handle.notify.notify_one();
        } else if !handle.control.request_session_cancel() {
            anyhow::bail!(
                "Session '{}' has no cancellable active execution",
                session_id
            );
        }

        Ok((agent_id, session_id.to_string()))
    }

    pub async fn kill_session(&self, session_id: &str) -> Result<(String, String)> {
        let (agent_id, handle) =
            self.find_runtime_by_session(session_id)
                .await
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "Session '{}' is not an active managed runtime session",
                        session_id
                    )
                })?;

        self.kill_runtime_requests(&agent_id, &handle, "Session killed")
            .await;

        if let Some(task) = &handle.task {
            task.abort();
        }

        self.runtimes.write().await.remove(&agent_id);

        Ok((agent_id, session_id.to_string()))
    }

    async fn find_runtime_by_session(
        &self,
        session_id: &str,
    ) -> Option<(String, Arc<AgentRuntimeHandle>)> {
        let runtimes = self.runtimes.read().await;
        runtimes.iter().find_map(|(agent_id, handle)| {
            if handle.control.current_session_id().as_deref() == Some(session_id) {
                Some((agent_id.clone(), Arc::clone(handle)))
            } else {
                None
            }
        })
    }

    async fn cancel_queued_requests_for_agent(&self, agent_id: &str, reason: &str) {
        let Some(handle) = self.runtimes.read().await.get(agent_id).cloned() else {
            return;
        };

        let drained: Vec<_> = {
            let mut queue = handle
                .queue
                .lock()
                .expect("agent runtime queue mutex poisoned");
            queue.drain(..).collect()
        };
        if drained.is_empty() {
            return;
        }
        handle.queued_tasks.store(0, Ordering::Relaxed);

        for envelope in drained {
            let request_id = envelope
                .request_id
                .unwrap_or_else(|| uuid::Uuid::now_v7().simple().to_string());
            let completed = PeerAgentTaskResult {
                request_id: request_id.clone(),
                agent_id: agent_id.to_string(),
                runtime_task_id: String::new(),
                status: TaskTerminalStatus::Cancelled,
                task_turn_count: 0,
                output: None,
                error: Some(reason.to_string()),
            };
            if let Some(tx_result) = envelope.result_tx {
                let _ = tx_result.send(completed.clone());
            }
            self.record_completed_result(completed).await;
        }
    }

    async fn kill_runtime_requests(
        &self,
        agent_id: &str,
        handle: &Arc<AgentRuntimeHandle>,
        reason: &str,
    ) {
        let drained: Vec<_> = {
            let mut queue = handle
                .queue
                .lock()
                .expect("agent runtime queue mutex poisoned");
            queue.drain(..).collect()
        };
        handle.queued_tasks.store(0, Ordering::Relaxed);

        for envelope in drained {
            let request_id = envelope
                .request_id
                .unwrap_or_else(|| uuid::Uuid::now_v7().simple().to_string());
            let completed = PeerAgentTaskResult {
                request_id: request_id.clone(),
                agent_id: agent_id.to_string(),
                runtime_task_id: String::new(),
                status: TaskTerminalStatus::Killed,
                task_turn_count: 0,
                output: None,
                error: Some(reason.to_string()),
            };
            if let Some(tx_result) = envelope.result_tx {
                let _ = tx_result.send(completed.clone());
            }
            self.record_completed_result(completed).await;
        }

        if let Some(request_id) = handle.control.current_request_id() {
            let completed = PeerAgentTaskResult {
                request_id: request_id.clone(),
                agent_id: agent_id.to_string(),
                runtime_task_id: handle.control.current_runtime_task_id().unwrap_or_default(),
                status: TaskTerminalStatus::Killed,
                task_turn_count: 0,
                output: None,
                error: Some(reason.to_string()),
            };
            self.record_completed_result(completed).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::Kernel;
    use crate::kernel::config::{
        AgentConfig, EmbeddingConfig, GovernanceConfig, HarnessConfig, KernelConfig,
        PersistenceConfig, ProviderConfig, TurinConfig,
    };
    use crate::tools::{Tool, ToolContext, ToolEffect, ToolError};
    use async_trait::async_trait;
    use serde_json::json;
    use std::collections::HashMap;
    use tempfile::tempdir;

    struct TestTool;

    #[async_trait]
    impl Tool for TestTool {
        fn name(&self) -> &str {
            "test_tool"
        }

        fn description(&self) -> &str {
            "test tool"
        }

        fn parameters_schema(&self) -> serde_json::Value {
            json!({
                "type": "object",
                "properties": {}
            })
        }

        async fn execute(
            &self,
            _params: serde_json::Value,
            _ctx: &ToolContext,
        ) -> Result<ToolEffect, ToolError> {
            Ok(ToolEffect::Output(crate::tools::ToolOutput::new(
                "ok".to_string(),
            )))
        }
    }

    fn test_config(workspace_root: &std::path::Path, harness_dir: &std::path::Path) -> TurinConfig {
        let mut providers = HashMap::new();
        providers.insert(
            "mock".to_string(),
            ProviderConfig {
                kind: "mock".to_string(),
                api_key_env: None,
                base_url: Some("Mock response".to_string()),
                ..ProviderConfig::default()
            },
        );

        TurinConfig {
            agent: AgentConfig {
                id: "default".to_string(),
                model: "mock-model".to_string(),
                provider: "mock".to_string(),
                system_prompt: "test".to_string(),
                thinking: None,
                mode: crate::kernel::config::AgentMode::Auto,
                harness: None,
                idle_grace_secs: None,
            },
            agents: HashMap::new(),
            kernel: KernelConfig {
                workspace_root: workspace_root.to_string_lossy().to_string(),
                max_turns: 4,
                heartbeat_interval_secs: 30,
                initial_spawn_depth: 0,
            },
            persistence: PersistenceConfig {
                database_path: workspace_root.join("test.db").to_string_lossy().to_string(),
            },
            harness: HarnessConfig {
                directory: harness_dir.to_string_lossy().to_string(),
                fs_root: ".".to_string(),
            },
            harnesses: HashMap::new(),
            providers,
            embeddings: Some(EmbeddingConfig::NoOp),
            governance: GovernanceConfig::default(),
            daemon: Default::default(),
        }
    }

    #[tokio::test]
    async fn build_shared_peer_kernel_reuses_configured_tool_registry() -> Result<()> {
        let tmp = tempdir()?;
        let harness_dir = tmp.path().join("harness");
        std::fs::create_dir_all(&harness_dir)?;

        let mut registry = ToolRegistry::new();
        registry.register(Box::new(TestTool))?;

        let kernel = Kernel::builder(test_config(tmp.path(), &harness_dir))
            .with_tool_registry(registry.clone())
            .build()?;

        let peer_kernel = super::peer_runtime::fork_peer_kernel(&kernel.agent_manager);

        assert_eq!(peer_kernel.tool_registry.len(), registry.len());
        assert!(peer_kernel.tool_registry.get("test_tool").is_some());

        Ok(())
    }

    #[tokio::test]
    async fn cancel_task_removes_queued_work_and_records_terminal_result() -> Result<()> {
        let tmp = tempdir()?;
        let harness_dir = tmp.path().join("harness");
        std::fs::create_dir_all(&harness_dir)?;

        let kernel = Kernel::builder(test_config(tmp.path(), &harness_dir)).build()?;
        let manager = kernel.agent_manager();
        let request_id = "req_cancelled".to_string();
        let (tx_result, rx_result) = oneshot::channel();

        manager
            .pending_results
            .write()
            .await
            .insert(request_id.clone(), rx_result);
        manager.pending_task_states.write().await.insert(
            request_id.clone(),
            PendingTaskRecord {
                agent_id: "default".to_string(),
                state: PendingTaskState::Queued,
                runtime_task_id: None,
            },
        );

        let mut queue = VecDeque::new();
        queue.push_back(PeerAgentTaskEnvelope {
            task: QueuedTask::ad_hoc("cancel me".to_string()),
            request_id: Some(request_id.clone()),
            result_tx: Some(tx_result),
            delegated_capabilities: None,
        });

        manager.runtimes.write().await.insert(
            "default".to_string(),
            Arc::new(AgentRuntimeHandle {
                queue: Arc::new(Mutex::new(queue)),
                notify: Arc::new(Notify::new()),
                control: Arc::new(RuntimeControl::default()),
                task: None,
                queued_tasks: Arc::new(AtomicUsize::new(1)),
                active_tasks: Arc::new(AtomicUsize::new(0)),
            }),
        );

        let snapshot = manager.cancel_task(&request_id).await?;
        assert_eq!(snapshot.state, "completed");
        assert_eq!(snapshot.status, Some(TaskTerminalStatus::Cancelled));
        assert_eq!(
            snapshot.error.as_deref(),
            Some("Task cancelled before execution")
        );
        assert!(
            manager
                .pending_task_states
                .read()
                .await
                .get(&request_id)
                .is_none()
        );
        assert!(
            manager
                .pending_results
                .read()
                .await
                .get(&request_id)
                .is_none()
        );

        let completed = manager
            .get_task(&request_id)
            .await
            .expect("cancelled task should be visible");
        assert_eq!(completed.status, Some(TaskTerminalStatus::Cancelled));

        Ok(())
    }

    #[tokio::test]
    async fn cancel_task_marks_running_work_cancelling() -> Result<()> {
        let tmp = tempdir()?;
        let harness_dir = tmp.path().join("harness");
        std::fs::create_dir_all(&harness_dir)?;

        let kernel = Kernel::builder(test_config(tmp.path(), &harness_dir)).build()?;
        let manager = kernel.agent_manager();
        let request_id = "req_running".to_string();
        let cancel_token = CancellationToken::new();
        let control = Arc::new(RuntimeControl::default());
        control.activate_task(
            Some(request_id.clone()),
            "t_1".to_string(),
            cancel_token.clone(),
        );

        manager.pending_task_states.write().await.insert(
            request_id.clone(),
            PendingTaskRecord {
                agent_id: "default".to_string(),
                state: PendingTaskState::Running,
                runtime_task_id: Some("t_1".to_string()),
            },
        );

        manager.runtimes.write().await.insert(
            "default".to_string(),
            Arc::new(AgentRuntimeHandle {
                queue: Arc::new(Mutex::new(VecDeque::new())),
                notify: Arc::new(Notify::new()),
                control,
                task: None,
                queued_tasks: Arc::new(AtomicUsize::new(0)),
                active_tasks: Arc::new(AtomicUsize::new(1)),
            }),
        );

        let snapshot = manager.cancel_task(&request_id).await?;
        assert_eq!(snapshot.state, "cancelling");
        assert!(snapshot.status.is_none());
        assert!(cancel_token.is_cancelled());

        Ok(())
    }

    #[tokio::test]
    async fn cancel_session_cancels_queued_work_and_requests_reset() -> Result<()> {
        let tmp = tempdir()?;
        let harness_dir = tmp.path().join("harness");
        std::fs::create_dir_all(&harness_dir)?;

        let kernel = Kernel::builder(test_config(tmp.path(), &harness_dir)).build()?;
        let manager = kernel.agent_manager();
        let session_id = "s_cancel";
        let request_id = "req_session_cancel".to_string();
        let (tx_result, rx_result) = oneshot::channel();
        let control = Arc::new(RuntimeControl::default());
        control.set_current_session_id(Some(session_id.to_string()));

        manager
            .pending_results
            .write()
            .await
            .insert(request_id.clone(), rx_result);
        manager.pending_task_states.write().await.insert(
            request_id.clone(),
            PendingTaskRecord {
                agent_id: "default".to_string(),
                state: PendingTaskState::Queued,
                runtime_task_id: None,
            },
        );

        let mut queue = VecDeque::new();
        queue.push_back(PeerAgentTaskEnvelope {
            task: QueuedTask::ad_hoc("queued".to_string()),
            request_id: Some(request_id.clone()),
            result_tx: Some(tx_result),
            delegated_capabilities: None,
        });

        manager.runtimes.write().await.insert(
            "default".to_string(),
            Arc::new(AgentRuntimeHandle {
                queue: Arc::new(Mutex::new(queue)),
                notify: Arc::new(Notify::new()),
                control: Arc::clone(&control),
                task: None,
                queued_tasks: Arc::new(AtomicUsize::new(1)),
                active_tasks: Arc::new(AtomicUsize::new(0)),
            }),
        );

        let (agent_id, returned_session_id) = manager.cancel_session(session_id).await?;
        assert_eq!(agent_id, "default");
        assert_eq!(returned_session_id, session_id);
        assert!(control.take_session_reset_requested());

        let completed = manager
            .get_task(&request_id)
            .await
            .expect("cancelled queued task should be visible");
        assert_eq!(completed.status, Some(TaskTerminalStatus::Cancelled));
        assert_eq!(
            completed.error.as_deref(),
            Some("Session cancelled before execution")
        );

        Ok(())
    }

    #[tokio::test]
    async fn kill_session_marks_running_and_queued_work_killed() -> Result<()> {
        let tmp = tempdir()?;
        let harness_dir = tmp.path().join("harness");
        std::fs::create_dir_all(&harness_dir)?;

        let kernel = Kernel::builder(test_config(tmp.path(), &harness_dir)).build()?;
        let manager = kernel.agent_manager();
        let session_id = "s_kill";
        let running_request_id = "req_running_kill".to_string();
        let queued_request_id = "req_queued_kill".to_string();
        let (tx_result, rx_result) = oneshot::channel();
        let control = Arc::new(RuntimeControl::default());
        control.set_current_session_id(Some(session_id.to_string()));
        control.activate_task(
            Some(running_request_id.clone()),
            "t_running".to_string(),
            CancellationToken::new(),
        );

        manager
            .pending_results
            .write()
            .await
            .insert(queued_request_id.clone(), rx_result);
        manager.pending_task_states.write().await.insert(
            running_request_id.clone(),
            PendingTaskRecord {
                agent_id: "default".to_string(),
                state: PendingTaskState::Running,
                runtime_task_id: Some("t_running".to_string()),
            },
        );
        manager.pending_task_states.write().await.insert(
            queued_request_id.clone(),
            PendingTaskRecord {
                agent_id: "default".to_string(),
                state: PendingTaskState::Queued,
                runtime_task_id: None,
            },
        );

        let mut queue = VecDeque::new();
        queue.push_back(PeerAgentTaskEnvelope {
            task: QueuedTask::ad_hoc("queued".to_string()),
            request_id: Some(queued_request_id.clone()),
            result_tx: Some(tx_result),
            delegated_capabilities: None,
        });

        manager.runtimes.write().await.insert(
            "default".to_string(),
            Arc::new(AgentRuntimeHandle {
                queue: Arc::new(Mutex::new(queue)),
                notify: Arc::new(Notify::new()),
                control,
                task: Some(tokio::spawn(async {
                    tokio::time::sleep(std::time::Duration::from_secs(60)).await;
                })),
                queued_tasks: Arc::new(AtomicUsize::new(1)),
                active_tasks: Arc::new(AtomicUsize::new(1)),
            }),
        );

        let (agent_id, returned_session_id) = manager.kill_session(session_id).await?;
        assert_eq!(agent_id, "default");
        assert_eq!(returned_session_id, session_id);
        assert!(manager.runtimes.read().await.get("default").is_none());

        let running = manager
            .get_task(&running_request_id)
            .await
            .expect("killed running task should be visible");
        assert_eq!(running.status, Some(TaskTerminalStatus::Killed));

        let queued = manager
            .get_task(&queued_request_id)
            .await
            .expect("killed queued task should be visible");
        assert_eq!(queued.status, Some(TaskTerminalStatus::Killed));
        assert_eq!(queued.error.as_deref(), Some("Session killed"));

        Ok(())
    }
}
