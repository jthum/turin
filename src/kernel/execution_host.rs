use std::collections::HashMap;
use std::path::PathBuf;
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
use crate::persistence::manager::{StoreManager, StorePathScope};
use crate::tools::registry::ToolRegistry;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PersistedSessionLockKey {
    store_path: PathBuf,
    session_id: i64,
}

#[derive(Default)]
pub(crate) struct SessionPersistenceCoordinator {
    locks:
        std::sync::Mutex<HashMap<PersistedSessionLockKey, std::sync::Weak<tokio::sync::Mutex<()>>>>,
}

impl SessionPersistenceCoordinator {
    pub(crate) fn lock_for(
        &self,
        store_path: PathBuf,
        session_id: i64,
    ) -> Arc<tokio::sync::Mutex<()>> {
        let key = PersistedSessionLockKey {
            store_path,
            session_id,
        };
        let mut locks = self
            .locks
            .lock()
            .expect("session persistence coordinator mutex poisoned");
        if let Some(existing) = locks.get(&key).and_then(std::sync::Weak::upgrade) {
            return existing;
        }

        let lock = Arc::new(tokio::sync::Mutex::new(()));
        locks.insert(key, Arc::downgrade(&lock));
        lock
    }
}

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
    pub(crate) persistence_locks: Arc<SessionPersistenceCoordinator>,
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

        let instance = runtime.create_instance(self.harness_init_context())?;
        instance.set_active_queue(Some(session.queue.clone()));
        session.harness_engine = Some(Arc::new(std::sync::Mutex::new(instance)));
        session.harness_generation = generation;
        Ok(())
    }

    pub(crate) fn clear_session_harness_engine(&self, session: &mut SessionState) {
        session.harness_engine = None;
        session.harness_generation = 0;
    }

    pub(crate) async fn bind_session_persistence_lock(
        &self,
        session: &mut SessionState,
    ) -> anyhow::Result<()> {
        let Some(internal_id) = session.internal_id else {
            return Ok(());
        };
        let store_path = self
            .store_manager
            .resolve_path_for_selector(&session.store_selector, StorePathScope::AllowAny)
            .await?;
        session.persistence_lock = self.persistence_locks.lock_for(store_path, internal_id);
        Ok(())
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
