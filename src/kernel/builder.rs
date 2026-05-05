use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;

use crate::inference::embeddings::EmbeddingProvider;
use crate::kernel::governance::GovernanceManager;
use crate::kernel::policy::RuntimePolicyManager;
use crate::kernel::{
    Kernel, TurinConfig,
    agent_manager::{AgentManager, SharedPeerRuntimeContext},
    execution_host::{ExecutionHost, SessionPersistenceCoordinator},
    harness_manager::HarnessManager,
};
use crate::persistence::manager::StoreManager;
use crate::tools::builtins::create_default_registry;
use crate::tools::registry::ToolRegistry;

/// Builder for constructing a `Kernel` instance.
pub struct RuntimeBuilder {
    config: TurinConfig,
    json: bool,
    tool_registry: ToolRegistry,

    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
}

impl RuntimeBuilder {
    /// Create a new builder with the given configuration.
    pub fn new(config: TurinConfig) -> Self {
        Self {
            config,
            json: false,
            tool_registry: create_default_registry(),

            embedding_provider: None,
        }
    }

    /// Enable JSON output mode (NDJSON).
    pub fn json_mode(mut self, json: bool) -> Self {
        self.json = json;
        self
    }

    /// Register a custom tool registry (overwriting defaults).
    pub fn with_tool_registry(mut self, registry: ToolRegistry) -> Self {
        self.tool_registry = registry;
        self
    }

    /// Build the Kernel.
    pub fn build(self) -> Result<Kernel> {
        let store_manager = Arc::new(StoreManager::new(
            &self.config.kernel.workspace_root,
            &self.config.layout.stores_dir,
        ));
        let config_arc = Arc::new(self.config);
        let agent_manager = Arc::new(AgentManager::new(config_arc.clone(), store_manager.clone()));
        let policy_manager = Arc::new(RuntimePolicyManager::new());
        let governance_manager = Arc::new(GovernanceManager::new(config_arc.governance.clone()));
        let harness_manager = Arc::new(HarnessManager::from_config(config_arc.as_ref())?);
        let persistence_locks = Arc::new(SessionPersistenceCoordinator::default());
        agent_manager.bind_shared_runtime(SharedPeerRuntimeContext {
            json: self.json,
            tool_registry: self.tool_registry.clone(),
            policy_manager: Arc::clone(&policy_manager),
            governance_manager: Arc::clone(&governance_manager),
            harness_manager: Arc::clone(&harness_manager),
            persistence_locks: Arc::clone(&persistence_locks),
        });
        Ok(Kernel {
            host: ExecutionHost {
                config: config_arc,
                json: self.json,
                tool_registry: self.tool_registry,
                store_manager,
                agent_manager,
                policy_manager,
                governance_manager,
                harness_manager,
                scheduler: None,
                persistence_locks,
                clients: HashMap::new(),
                embedding_provider: self.embedding_provider,
                mcp_clients: Vec::new(),
            },
            check_watcher: Arc::new(std::sync::Mutex::new(None)),
        })
    }
}
