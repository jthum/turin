pub mod agent_manager;
pub mod builder;
pub mod config;
pub mod event;
mod event_persistence;
pub mod governance;
mod harness_hooks;
mod harness_manager;
mod harness_runtime;
pub mod identity;
mod init;
mod mcp_runtime;
pub mod policy;
mod run_loop;
pub mod session;
mod session_lifecycle;
mod task_execution;
mod task_lifecycle;
mod task_planning;
mod turn;

use agent_manager::AgentManager;
use anyhow::Result;
use builder::RuntimeBuilder;
use config::TurinConfig;
use event::TaskTerminalStatus;
use harness_manager::HarnessManager;
use std::collections::HashMap;
use std::sync::Arc;

use crate::inference::embeddings::EmbeddingProvider;
use crate::inference::provider::ProviderClient;
use crate::kernel::governance::GovernanceManager;
use crate::kernel::policy::RuntimePolicyManager;
use crate::persistence::manager::StoreManager;

use crate::tools::registry::ToolRegistry;
use mcp_runtime::McpClientEntry;
use notify::RecommendedWatcher;

/// The Turin Kernel — manages the agent loop, event system, and tool execution.
///
/// The Kernel has no opinions about agent behavior. It provides the physics:
/// transport, streaming, tool execution, persistence, and event hooks.
/// Harness scripts define the behavior.
pub struct Kernel {
    pub(crate) config: Arc<TurinConfig>,
    pub(crate) json: bool,
    pub(crate) tool_registry: ToolRegistry,
    pub(crate) store_manager: Arc<StoreManager>,
    pub(crate) agent_manager: Arc<AgentManager>,
    pub(crate) policy_manager: Arc<RuntimePolicyManager>,
    pub(crate) governance_manager: Arc<GovernanceManager>,
    /// First-class harness manager. In the current checkpoint it still resolves to the
    /// default harness runtime, but it replaces the old single-engine kernel slot.
    pub(crate) harness_manager: Arc<HarnessManager>,
    /// Watcher handle to keep it alive
    pub(crate) check_watcher: Option<RecommendedWatcher>,
    pub(crate) clients: HashMap<String, ProviderClient>,
    pub(crate) embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    /// Active session queue for harness interaction
    pub(crate) active_queue: crate::harness::globals::ActiveSessionQueue,

    pub(crate) mcp_clients: Vec<McpClientEntry>,
}

impl Drop for Kernel {
    fn drop(&mut self) {
        // Ensure MCP client Arcs are dropped promptly so stdio transports can tear down
        // subprocesses even when explicit async shutdown was not reached.
        self.mcp_clients.clear();
    }
}

/// A pending tool call collected during streaming.
#[derive(Debug, Clone)]
pub(crate) struct PendingToolCall {
    pub id: String,
    pub name: String,
    pub args: serde_json::Value,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct TaskExecutionResult {
    pub status: TaskTerminalStatus,
    pub task_turn_count: u32,
}

impl Kernel {
    /// Create a new builder for Kernel.
    pub fn builder(config: TurinConfig) -> RuntimeBuilder {
        RuntimeBuilder::new(config)
    }

    /// Access the store manager.
    pub fn store_manager(&self) -> &Arc<StoreManager> {
        &self.store_manager
    }

    /// Access the runtime policy manager.
    pub fn policy_manager(&self) -> &Arc<RuntimePolicyManager> {
        &self.policy_manager
    }

    /// Access the governance manager (profile/capability observability, G1).
    pub fn governance_manager(&self) -> &Arc<GovernanceManager> {
        &self.governance_manager
    }

    /// Access the configuration.
    pub fn config(&self) -> &TurinConfig {
        &self.config
    }

    /// Lock the harness mutex.
    ///
    /// Panics if the mutex is poisoned (previous holder panicked).
    /// A poisoned harness is an unrecoverable state — continuing would
    /// risk executing tool calls with a partially-updated engine.
    ///
    /// Callers must keep the guard's lifetime fully synchronous (no `.await` while held).
    pub fn lock_harness(
        &self,
    ) -> std::sync::MutexGuard<'_, Option<crate::harness::engine::HarnessEngine>> {
        self.harness_manager.lock_default_engine()
    }

    /// Get names of all loaded harness scripts.
    pub fn loaded_scripts(&self) -> Vec<String> {
        self.harness_manager.default_runtime().loaded_scripts()
    }

    /// Add a provider client manually (e.g. for testing).
    pub fn add_client(&mut self, name: String, client: ProviderClient) {
        self.clients.insert(name, client);
    }

    /// Run a Lua script directly in the harness (for testing/verification).
    pub fn run_script(&self, script: &str) -> Result<()> {
        self.harness_manager
            .default_runtime()
            .load_script_str(script)
    }
}
