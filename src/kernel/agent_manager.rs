mod peer_runtime;
mod runtime_registry;

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, Weak};

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
use tokio::sync::{RwLock, mpsc, oneshot};
use tokio::task::JoinHandle;

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
    pub queued_tasks: usize,
    pub awaiting_results: usize,
}

struct PeerAgentTaskEnvelope {
    task: QueuedTask,
    request_id: Option<String>,
    result_tx: Option<oneshot::Sender<PeerAgentTaskResult>>,
    delegated_capabilities: Option<BTreeMap<String, bool>>,
}

/// A handle to a running peer agent.
pub struct AgentRuntimeHandle {
    /// The channel to send tasks to the background agent loop.
    tx: mpsc::Sender<PeerAgentTaskEnvelope>,
    /// The background task running the agent's event loop.
    task: Option<JoinHandle<()>>,
    /// Approximate number of tasks currently queued in the runtime channel.
    queued_tasks: Arc<AtomicUsize>,
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
    /// Mapping of request id -> agent id for status accounting.
    pending_result_agents: RwLock<HashMap<String, String>>,
    /// Weak self-handle used when constructing shared peer kernels.
    self_handle: OnceLock<Weak<AgentManager>>,
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
            pending_result_agents: RwLock::new(HashMap::new()),
            self_handle: OnceLock::new(),
            shared_runtime: OnceLock::new(),
            shared_inference: Mutex::new(SharedInferenceState::default()),
        }
    }

    pub(crate) fn bind_self_handle(&self, handle: Weak<AgentManager>) {
        let _ = self.self_handle.set(handle);
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

    pub(crate) fn self_arc(&self) -> Result<Arc<AgentManager>> {
        self.self_handle
            .get()
            .and_then(Weak::upgrade)
            .ok_or_else(|| anyhow::anyhow!("AgentManager self handle not bound"))
    }

    pub(crate) fn build_shared_peer_kernel(&self) -> Result<crate::kernel::Kernel> {
        let shared = self
            .shared_runtime()
            .ok_or_else(|| anyhow::anyhow!("AgentManager shared runtime not bound"))?;
        let inference = self
            .shared_inference
            .lock()
            .expect("agent manager shared inference mutex poisoned")
            .clone();

        Ok(crate::kernel::Kernel {
            config: Arc::clone(&self.config),
            json: shared.json,
            tool_registry: shared.tool_registry.clone(),
            store_manager: Arc::clone(&self.store_manager),
            agent_manager: self.self_arc()?,
            policy_manager: Arc::clone(&shared.policy_manager),
            governance_manager: Arc::clone(&shared.governance_manager),
            harness_manager: Arc::clone(&shared.harness_manager),
            check_watcher: None,
            clients: inference.clients,
            embedding_provider: inference.embedding_provider,
            mcp_clients: Vec::new(),
        })
    }

    /// Dispatch a task to an agent by ID. If the agent isn't running, it will be started automatically.
    pub async fn send(
        &self,
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
        &self,
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
            let mut map = self.pending_result_agents.write().await;
            map.insert(request_id.clone(), agent_id.to_string());
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
        let mut rx = {
            let mut pending = self.pending_results.write().await;
            pending.remove(request_id).ok_or_else(|| {
                anyhow::anyhow!("Unknown or already-awaited peer task '{}'", request_id)
            })?
        };

        let mut timed_out = false;
        let result = if let Some(ms) = timeout_ms {
            match tokio::time::timeout(std::time::Duration::from_millis(ms), &mut rx).await {
                Ok(Ok(res)) => Ok(res),
                Ok(Err(_)) => Err(anyhow::anyhow!(
                    "Peer task '{}' result channel closed",
                    request_id
                )),
                Err(_) => {
                    timed_out = true;
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

        if !timed_out {
            self.pending_result_agents.write().await.remove(request_id);
        }

        result
    }

    /// List configured agents with runtime status.
    pub async fn list_statuses(&self) -> Vec<AgentStatusSnapshot> {
        let runtimes = self.runtimes.read().await;
        let pending = self.pending_result_agents.read().await;
        let mut awaiting_by_agent: HashMap<&str, usize> = HashMap::new();
        for agent_id in pending.values() {
            *awaiting_by_agent.entry(agent_id.as_str()).or_default() += 1;
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
                AgentStatusSnapshot {
                    agent_id,
                    running,
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

    async fn enqueue_runtime_task(
        &self,
        handle: &Arc<AgentRuntimeHandle>,
        envelope: PeerAgentTaskEnvelope,
    ) -> Result<()> {
        handle.queued_tasks.fetch_add(1, Ordering::Relaxed);
        if let Err(e) = handle.tx.send(envelope).await {
            handle.queued_tasks.fetch_sub(1, Ordering::Relaxed);
            anyhow::bail!("Failed to route task to agent queue: {}", e);
        }
        Ok(())
    }

    async fn remove_pending_request(&self, request_id: &str) {
        self.pending_results.write().await.remove(request_id);
        self.pending_result_agents.write().await.remove(request_id);
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

        let peer_kernel = kernel.agent_manager.build_shared_peer_kernel()?;

        assert_eq!(peer_kernel.tool_registry.len(), registry.len());
        assert!(peer_kernel.tool_registry.get("test_tool").is_some());

        Ok(())
    }
}
