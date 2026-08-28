pub mod agent_manager;
pub mod builder;
pub mod config;
mod delegation_budget;
mod error;
pub mod event;
mod event_persistence;
mod execution_host;
pub mod governance;
pub mod harness;
pub mod harness_contract;
mod harness_hooks;
mod harness_manager;
#[doc(hidden)]
pub mod harness_runtime;
mod hot_history;
pub mod identity;
mod init;
mod mcp_runtime;
pub mod policy;
mod run_loop;
pub mod session;
mod session_lifecycle;
pub(crate) mod session_metadata;
pub mod session_refs;
mod task_execution;
mod task_lifecycle;
mod task_planning;
pub mod task_promotion;
pub mod tool_authorization;
mod turn;

pub use error::{KernelError, KernelErrorKind, KernelResult};

#[doc(hidden)]
pub use session_lifecycle::prepare_persisted_session_sidestep;
#[doc(hidden)]
pub use turn::context_window::estimate_history_input_tokens;
pub(crate) use turn::context_window::estimate_persisted_message_input_tokens;

use crate::inference::provider::ProviderClient;
use crate::kernel::governance::GovernanceManager;
use crate::kernel::policy::RuntimePolicyManager;
use crate::persistence::manager::StoreManager;
use crate::persistence::manager::StoreSelector;
use anyhow::Result;
use builder::RuntimeBuilder;
use config::TurinConfig;
use event::TaskBranchOutcome;
use event::TaskTerminalStatus;
use execution_host::ExecutionHost;
use std::collections::HashMap;
use std::sync::Arc;
use turin_daemon_protocol::UiIntentMessage;

use notify::RecommendedWatcher;

#[derive(Debug, Clone, serde::Serialize)]
pub struct HarnessRuntimeSnapshot {
    pub harness_id: String,
    pub directory: String,
    pub bound_agents: Vec<String>,
    pub watched_roots: Vec<String>,
    pub loaded_scripts: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ui_intents: Vec<UiIntentMessage>,
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

    pub(crate) fn harness_init_context(&self) -> harness_runtime::HarnessRuntimeInitContext {
        self.host.harness_init_context()
    }

    pub(crate) fn harness_definition_for_agent(
        &self,
        agent_id: &str,
    ) -> Arc<harness_runtime::HarnessDefinition> {
        self.host.harness_definition_for_agent(agent_id)
    }

    pub(crate) fn harness_definition_by_id(
        &self,
        harness_id: &str,
    ) -> Option<Arc<harness_runtime::HarnessDefinition>> {
        self.host.harness_definition_by_id(harness_id)
    }

    pub(crate) fn agent_config_for(&self, agent_id: &str) -> Result<&config::AgentConfig> {
        self.host.agent_config_for(agent_id)
    }

    pub(crate) fn validate_named_harness_sources(
        &self,
        harness_id: &str,
        source_overlay: crate::harness::source::HarnessSourceOverlay,
    ) -> Result<usize> {
        self.host
            .validate_named_harness_sources(harness_id, source_overlay)
    }

    /// Initialize configured provider and embedding clients.
    pub fn init_clients(&mut self) -> KernelResult<()> {
        self.host
            .init_clients()
            .map_err(|error| KernelError::new(KernelErrorKind::Client, error))
    }

    /// Initialize configured persistent state stores.
    pub async fn init_state(&mut self) -> KernelResult<()> {
        self.host
            .init_state()
            .await
            .map_err(|error| KernelError::new(KernelErrorKind::Persistence, error))
    }

    /// Initialize all configured harness definitions.
    pub async fn init_harness(&mut self) -> KernelResult<()> {
        self.host
            .init_harness()
            .await
            .map_err(|error| KernelError::new(KernelErrorKind::Harness, error))
    }

    /// Run state, client, harness, and watcher initialization.
    ///
    /// `build()` stays I/O-free. Tests that inject clients between steps should
    /// keep calling the individual `init_*` methods.
    pub async fn start(&mut self) -> KernelResult<()> {
        self.init_state().await?;
        self.init_clients()?;
        self.init_harness().await?;
        self.start_watcher()
            .map_err(|error| KernelError::new(KernelErrorKind::Harness, error))?;
        Ok(())
    }

    /// Atomically reload all configured harness definitions.
    pub async fn reload_harness(&mut self) -> KernelResult<()> {
        self.host
            .reload_harness()
            .await
            .map_err(|error| KernelError::new(KernelErrorKind::Harness, error))
    }

    /// Atomically reload one configured harness definition.
    pub async fn reload_named_harness(&mut self, harness_id: &str) -> KernelResult<()> {
        self.host
            .reload_named_harness(harness_id)
            .await
            .map_err(|error| KernelError::new(KernelErrorKind::Harness, error))
    }

    /// Validate one configured harness without activating it.
    pub fn validate_named_harness(&self, harness_id: &str) -> KernelResult<usize> {
        self.host
            .validate_named_harness(harness_id)
            .map_err(|error| KernelError::new(KernelErrorKind::Harness, error))
    }

    /// Create a session for the primary configured agent.
    pub async fn create_session(&self) -> KernelResult<session::SessionState> {
        self.host
            .create_session()
            .await
            .map_err(|error| KernelError::new(KernelErrorKind::Session, error))
    }

    /// Create a session for a configured agent.
    pub async fn create_session_for_agent(
        &self,
        agent_id: &str,
    ) -> KernelResult<session::SessionState> {
        self.host
            .create_session_for_agent(agent_id)
            .await
            .map_err(|error| KernelError::new(KernelErrorKind::Session, error))
    }

    /// Create a session for an agent using explicit state and default-store selectors.
    pub async fn create_session_for_agent_in_store(
        &self,
        agent_id: &str,
        state_selector: Option<StoreSelector>,
        default_store_selector: Option<StoreSelector>,
    ) -> KernelResult<session::SessionState> {
        self.host
            .create_session_for_agent_in_store(agent_id, state_selector, default_store_selector)
            .await
            .map_err(|error| KernelError::new(KernelErrorKind::Session, error))
    }

    /// Create a session with explicit storage, provenance, and inference context.
    pub async fn create_session_for_agent_with_context(
        &self,
        agent_id: &str,
        state_selector: Option<StoreSelector>,
        default_store_selector: Option<StoreSelector>,
        origin_id: Option<String>,
        inference: config::InferenceOverrideConfig,
    ) -> KernelResult<session::SessionState> {
        self.host
            .create_session_for_agent_with_context(
                agent_id,
                state_selector,
                default_store_selector,
                origin_id,
                inference,
            )
            .await
            .map_err(|error| KernelError::new(KernelErrorKind::Session, error))
    }

    /// Resume a persisted session for a configured agent.
    pub async fn resume_session_for_agent(
        &self,
        agent_id: &str,
        session_id: &str,
    ) -> KernelResult<session::SessionState> {
        self.host
            .resume_session_for_agent(agent_id, session_id)
            .await
            .map_err(|error| KernelError::new(KernelErrorKind::Session, error))
    }

    /// Resume a persisted session with explicit provenance and inference context.
    pub async fn resume_session_for_agent_with_context(
        &self,
        agent_id: &str,
        session_id: &str,
        origin_id: Option<String>,
        inference: config::InferenceOverrideConfig,
    ) -> KernelResult<session::SessionState> {
        self.host
            .resume_session_for_agent_with_context(agent_id, session_id, origin_id, inference)
            .await
            .map_err(|error| KernelError::new(KernelErrorKind::Session, error))
    }

    /// Refresh a live session from its persisted execution target.
    pub async fn refresh_session_from_persistence(
        &self,
        session: &mut session::SessionState,
    ) -> KernelResult<()> {
        self.host
            .refresh_session_from_persistence(session)
            .await
            .map_err(|error| KernelError::new(KernelErrorKind::Persistence, error))
    }

    /// Select a named branch in a directly managed session.
    pub async fn select_session_branch_by_name_local(
        &self,
        session: &mut session::SessionState,
        branch_name: &str,
    ) -> KernelResult<bool> {
        self.host
            .select_session_branch_by_name_local(session, branch_name)
            .await
            .map_err(|error| KernelError::new(KernelErrorKind::Session, error))
    }

    /// Select an exact turn in a directly managed session.
    pub async fn select_session_turn_local(
        &self,
        session: &mut session::SessionState,
        turn_id: i64,
    ) -> KernelResult<bool> {
        self.host
            .select_session_turn_local(session, turn_id)
            .await
            .map_err(|error| KernelError::new(KernelErrorKind::Session, error))
    }

    /// Select an external persisted session reference as read context.
    pub async fn select_session_external_reference_local(
        &self,
        session: &mut session::SessionState,
        reference: &str,
    ) -> KernelResult<bool> {
        self.host
            .select_session_external_reference_local(session, reference)
            .await
            .map_err(|error| KernelError::new(KernelErrorKind::Session, error))
    }

    /// Start a directly managed session if it is not already active.
    pub async fn start_session(&self, session: &mut session::SessionState) -> KernelResult<()> {
        self.host
            .start_session(session)
            .await
            .map_err(|error| KernelError::new(KernelErrorKind::Session, error))
    }

    /// Run queued work for a directly managed session.
    pub async fn run(
        &mut self,
        session: &mut session::SessionState,
        prompt: Option<String>,
    ) -> KernelResult<()> {
        self.host
            .run(session, prompt)
            .await
            .map_err(|error| KernelError::new(KernelErrorKind::Runtime, error))
    }

    /// Queue a prompt without starting the run loop.
    pub async fn queue_prompt(
        &self,
        session: &mut session::SessionState,
        prompt: String,
    ) -> KernelResult<()> {
        self.host
            .queue_prompt(session, prompt)
            .await
            .map_err(|error| KernelError::new(KernelErrorKind::Task, error))
    }

    /// Enqueue a prepared task, assigning an id and honoring `queue.max_depth`.
    pub async fn enqueue_task(
        &self,
        session: &mut session::SessionState,
        task: session::QueuedTask,
    ) -> KernelResult<()> {
        self.host
            .enqueue_session_task(session, task)
            .await
            .map_err(|error| KernelError::new(KernelErrorKind::Task, error))
    }

    /// End a directly managed session and flush its required durability records.
    pub async fn end_session(&self, session: &mut session::SessionState) -> KernelResult<()> {
        self.host
            .end_session(session)
            .await
            .map_err(|error| KernelError::new(KernelErrorKind::Session, error))
    }

    /// Cooperatively stop peer runtimes and release external runtime resources.
    pub async fn shutdown(&mut self) {
        *self
            .check_watcher
            .lock()
            .expect("kernel watcher mutex poisoned during shutdown") = None;
        self.host.agent_manager.shutdown().await;
        self.host.shutdown_mcp_clients().await;
    }

    /// Access the store manager.
    pub fn store_manager(&self) -> &Arc<StoreManager> {
        &self.host.store_manager
    }

    /// Access the agent manager.
    pub fn agent_manager(&self) -> &Arc<crate::kernel::agent_manager::AgentManager> {
        &self.host.agent_manager
    }

    /// Access the runtime policy manager.
    pub fn policy_manager(&self) -> &Arc<RuntimePolicyManager> {
        &self.host.policy_manager
    }

    /// Access the governance manager (profile/capability observability, G1).
    pub fn governance_manager(&self) -> &Arc<GovernanceManager> {
        &self.host.governance_manager
    }

    /// Access the configuration.
    pub fn config(&self) -> &TurinConfig {
        &self.host.config
    }

    /// Get names of all loaded harness scripts.
    pub fn loaded_scripts(&self) -> Vec<String> {
        self.host
            .harness_manager
            .default_definition()
            .loaded_scripts()
    }

    pub fn loaded_scripts_for_agent(&self, agent_id: &str) -> KernelResult<Vec<String>> {
        self.host
            .agent_config_for(agent_id)
            .map_err(|error| KernelError::new(KernelErrorKind::Agent, error))?;
        Ok(self
            .host
            .harness_definition_for_agent(agent_id)
            .loaded_scripts())
    }

    pub fn harness_snapshot(&self, harness_id: &str) -> Option<HarnessRuntimeSnapshot> {
        self.harness_snapshots()
            .into_iter()
            .find(|snapshot| snapshot.harness_id == harness_id)
    }

    pub fn harness_snapshots(&self) -> Vec<HarnessRuntimeSnapshot> {
        let mut bound_agents: HashMap<String, Vec<String>> = HashMap::new();
        for (agent_id, harness_id) in self.host.harness_manager.agent_bindings() {
            bound_agents
                .entry(harness_id.clone())
                .or_default()
                .push(agent_id.clone());
        }

        let mut snapshots: Vec<_> = self
            .host
            .harness_manager
            .definition_entries()
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
                let ui_intents = runtime
                    .ui_intents()
                    .into_iter()
                    .map(|mut intent| {
                        if intent.source.harness_id.is_none() {
                            intent.source.harness_id = Some(harness_id.clone());
                        }
                        intent
                    })
                    .collect();
                HarnessRuntimeSnapshot {
                    harness_id: harness_id.clone(),
                    directory: runtime.directory().display().to_string(),
                    bound_agents: agents,
                    watched_roots,
                    loaded_scripts,
                    ui_intents,
                }
            })
            .collect();
        snapshots.sort_by(|a, b| a.harness_id.cmp(&b.harness_id));
        snapshots
    }

    /// Add a provider client manually (e.g. for testing).
    pub fn add_client(&mut self, name: String, client: ProviderClient) {
        self.host.clients.insert(name, client);
    }

    /// Run a Lua script directly in the harness (for testing/verification).
    pub fn run_script(&self, script: &str) -> KernelResult<()> {
        self.host
            .harness_manager
            .default_definition()
            .run_source(self.host.harness_init_context(), script)
            .map_err(|error| KernelError::new(KernelErrorKind::Harness, error))
    }
}
