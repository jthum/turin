use std::collections::HashMap;
use std::sync::Arc;

use crate::inference::embeddings::EmbeddingProvider;
use crate::inference::provider::ProviderClient;
use crate::kernel::agent_manager::AgentManager;
use crate::kernel::config::TurinConfig;
use crate::kernel::governance::GovernanceManager;
use crate::kernel::harness_manager::HarnessManager;
use crate::kernel::mcp_runtime::McpClientEntry;
use crate::kernel::policy::RuntimePolicyManager;
use crate::kernel::session::{SessionHarnessEngine, SessionState};
use crate::persistence::manager::StoreManager;
use crate::tools::registry::ToolRegistry;

/// Shared runtime execution state used by both the top-level kernel and peer runtimes.
pub struct ExecutionHost {
    pub(crate) config: Arc<TurinConfig>,
    pub(crate) json: bool,
    pub(crate) tool_registry: ToolRegistry,
    pub(crate) store_manager: Arc<StoreManager>,
    pub(crate) agent_manager: Arc<AgentManager>,
    pub(crate) policy_manager: Arc<RuntimePolicyManager>,
    pub(crate) governance_manager: Arc<GovernanceManager>,
    pub(crate) harness_manager: Arc<HarnessManager>,
    pub(crate) clients: HashMap<String, ProviderClient>,
    pub(crate) embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    pub(crate) mcp_clients: Vec<McpClientEntry>,
}

impl ExecutionHost {
    pub(crate) fn runtime_for_agent(
        &self,
        agent_id: &str,
    ) -> Arc<crate::kernel::harness_runtime::HarnessRuntime> {
        Arc::clone(self.harness_manager.resolve_harness(Some(agent_id)))
    }

    pub(crate) fn runtime_for_session(
        &self,
        session: &SessionState,
    ) -> Arc<crate::kernel::harness_runtime::HarnessRuntime> {
        self.runtime_for_agent(session.identity.agent_id())
    }

    pub(crate) fn session_harness_engine(
        &self,
        session: &SessionState,
    ) -> Option<SessionHarnessEngine> {
        session.harness_engine.clone()
    }

    pub(crate) fn ensure_session_harness_engine(
        &self,
        session: &mut SessionState,
    ) -> anyhow::Result<()> {
        let runtime = self.runtime_for_session(session);
        let generation = runtime.generation();
        if session.harness_engine.is_some() && session.harness_generation == generation {
            return Ok(());
        }

        let engine = runtime.create_engine(self.harness_init_context())?;
        engine.set_active_queue(Some(session.queue.clone()));
        session.harness_engine = Some(Arc::new(std::sync::Mutex::new(engine)));
        session.harness_generation = generation;
        Ok(())
    }

    pub(crate) fn clear_session_harness_engine(&self, session: &mut SessionState) {
        session.harness_engine = None;
        session.harness_generation = 0;
    }

    pub(crate) fn agent_config_for(
        &self,
        agent_id: &str,
    ) -> anyhow::Result<&crate::kernel::config::AgentConfig> {
        if agent_id == self.config.agent.id {
            Ok(&self.config.agent)
        } else {
            self.config
                .agents
                .get(agent_id)
                .ok_or_else(|| anyhow::anyhow!("Unknown agent profile: {}", agent_id))
        }
    }

    pub(crate) fn agent_config_for_session(
        &self,
        session: &SessionState,
    ) -> anyhow::Result<&crate::kernel::config::AgentConfig> {
        self.agent_config_for(session.identity.agent_id())
    }
}
