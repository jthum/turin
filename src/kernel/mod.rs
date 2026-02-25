pub mod agent_manager;
pub mod builder;
pub mod config;
pub mod event;
mod event_persistence;
pub mod governance;
mod harness_hooks;
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
use std::collections::HashMap;
use std::sync::Arc;

use crate::harness::engine::HarnessEngine;
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
    /// Thread-safe harness engine for hot-reloading.
    ///
    /// Uses `std::sync::Mutex` intentionally: harness/Luau execution is synchronous and
    /// the engine is accessed from sync hook/tool paths. Do not hold this lock across `.await`.
    pub(crate) harness: Arc<std::sync::Mutex<Option<HarnessEngine>>>,
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
    pub fn lock_harness(&self) -> std::sync::MutexGuard<'_, Option<HarnessEngine>> {
        self.harness.lock().expect("harness mutex poisoned")
    }

    /// Get names of all loaded harness scripts.
    pub fn loaded_scripts(&self) -> Vec<String> {
        let lock = self.lock_harness();
        if let Some(ref engine) = *lock {
            engine.loaded_scripts().to_vec()
        } else {
            Vec::new()
        }
    }

    /// Add a provider client manually (e.g. for testing).
    pub fn add_client(&mut self, name: String, client: ProviderClient) {
        self.clients.insert(name, client);
    }

    /// Run a Lua script directly in the harness (for testing/verification).
    pub fn run_script(&self, script: &str) -> Result<()> {
        let mut harness_lock = self.lock_harness();
        if let Some(ref mut engine) = *harness_lock {
            engine.load_script_str(script)?;
        } else {
            anyhow::bail!("Harness not initialized");
        }
        Ok(())
    }
}
