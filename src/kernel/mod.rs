pub mod agent_manager;
pub mod builder;
pub mod config;
pub mod event;
mod event_persistence;
mod execution_host;
pub mod governance;
mod harness_hooks;
mod harness_manager;
pub(crate) mod harness_runtime;
pub mod identity;
mod init;
mod mcp_runtime;
pub mod policy;
mod run_loop;
pub mod session;
mod session_lifecycle;
pub mod session_refs;
mod task_execution;
mod task_lifecycle;
mod task_planning;
pub mod task_promotion;
mod turn;

pub(crate) use session_lifecycle::prepare_persisted_session_sidestep;
pub(crate) use turn::context_window::estimate_history_input_tokens;

use crate::inference::provider::ProviderClient;
use crate::kernel::governance::GovernanceManager;
use crate::kernel::policy::RuntimePolicyManager;
use crate::persistence::manager::StoreManager;
use anyhow::Result;
use builder::RuntimeBuilder;
use config::TurinConfig;
use event::TaskBranchOutcome;
use event::TaskTerminalStatus;
use execution_host::ExecutionHost;
use std::collections::HashMap;
use std::sync::Arc;

use notify::RecommendedWatcher;

#[derive(Debug, Clone, serde::Serialize)]
pub struct HarnessRuntimeSnapshot {
    pub harness_id: String,
    pub directory: String,
    pub bound_agents: Vec<String>,
    pub watched_roots: Vec<String>,
    pub loaded_scripts: Vec<String>,
}

/// The Turin Kernel — manages the agent loop, event system, and tool execution.
///
/// The Kernel has no opinions about agent behavior. It provides the physics:
/// transport, streaming, tool execution, persistence, and event hooks.
/// Harness scripts define the behavior.
pub struct Kernel {
    pub(crate) host: ExecutionHost,
    /// Watcher handle to keep it alive
    pub(crate) check_watcher: Arc<std::sync::Mutex<Option<RecommendedWatcher>>>,
}

impl Drop for Kernel {
    fn drop(&mut self) {
        // Ensure MCP client Arcs are dropped promptly so stdio transports can tear down
        // subprocesses even when explicit async shutdown was not reached.
        self.host.mcp_clients.clear();
    }
}

/// A pending tool call collected during streaming.
#[derive(Debug, Clone)]
pub(crate) struct PendingToolCall {
    pub id: String,
    pub name: String,
    pub args: serde_json::Value,
}

#[derive(Debug, Clone)]
pub(crate) struct TaskExecutionResult {
    pub status: TaskTerminalStatus,
    pub task_turn_count: u32,
    pub branch_outcome: Option<TaskBranchOutcome>,
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

    /// Access the agent manager.
    pub fn agent_manager(&self) -> &Arc<crate::kernel::agent_manager::AgentManager> {
        &self.agent_manager
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

    /// Get names of all loaded harness scripts.
    pub fn loaded_scripts(&self) -> Vec<String> {
        self.harness_manager.default_runtime().loaded_scripts()
    }

    pub fn loaded_scripts_for_agent(&self, agent_id: &str) -> Result<Vec<String>> {
        self.agent_config_for(agent_id)?;
        Ok(self.runtime_for_agent(agent_id).loaded_scripts())
    }

    pub fn harness_snapshot(&self, harness_id: &str) -> Option<HarnessRuntimeSnapshot> {
        self.harness_snapshots()
            .into_iter()
            .find(|snapshot| snapshot.harness_id == harness_id)
    }

    pub fn harness_snapshots(&self) -> Vec<HarnessRuntimeSnapshot> {
        let mut bound_agents: HashMap<String, Vec<String>> = HashMap::new();
        for (agent_id, harness_id) in self.harness_manager.agent_bindings() {
            bound_agents
                .entry(harness_id.clone())
                .or_default()
                .push(agent_id.clone());
        }

        let mut snapshots: Vec<_> = self
            .harness_manager
            .runtime_entries()
            .map(|(harness_id, runtime)| {
                let mut agents = bound_agents.remove(harness_id).unwrap_or_default();
                agents.sort();
                let mut loaded_scripts = runtime.loaded_scripts();
                loaded_scripts.sort();
                let mut watched_roots: Vec<_> = runtime
                    .watch_roots()
                    .into_iter()
                    .map(|root| root.path.display().to_string())
                    .collect();
                watched_roots.sort();
                watched_roots.dedup();
                HarnessRuntimeSnapshot {
                    harness_id: harness_id.clone(),
                    directory: runtime.directory().display().to_string(),
                    bound_agents: agents,
                    watched_roots,
                    loaded_scripts,
                }
            })
            .collect();
        snapshots.sort_by(|a, b| a.harness_id.cmp(&b.harness_id));
        snapshots
    }

    /// Add a provider client manually (e.g. for testing).
    pub fn add_client(&mut self, name: String, client: ProviderClient) {
        self.clients.insert(name, client);
    }

    /// Run a Lua script directly in the harness (for testing/verification).
    pub fn run_script(&self, script: &str) -> Result<()> {
        let mut instance = self
            .harness_manager
            .default_runtime()
            .create_instance(self.harness_init_context())?;
        instance.load_script_str(script)
    }
}

impl std::ops::Deref for Kernel {
    type Target = ExecutionHost;

    fn deref(&self) -> &Self::Target {
        &self.host
    }
}

impl std::ops::DerefMut for Kernel {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.host
    }
}
