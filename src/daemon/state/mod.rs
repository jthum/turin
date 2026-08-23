mod harness_actions;
mod harness_sources;
mod helpers;
mod memories;
mod registry_ops;
mod runtime_sessions;
mod runtime_tasks;
mod scheduled_execution;
mod scheduled_jobs;
mod scheduled_worklist_actions;
mod source_revision;
#[cfg(test)]
mod tests;
mod types;
mod worklists;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::Serialize;
use tokio::sync::Notify;
use turin_types::ToolsConfig;

use crate::daemon::registry::{
    RegistryLoad, RegistrySnapshot, build_effective_config, scan_registry, snapshot,
};
use crate::harness::scheduler::HarnessSchedulerAccess;
use crate::kernel::Kernel;
use crate::kernel::config::{ThinkingConfig, TurinConfig};
use crate::kernel::harness_runtime::HarnessAdapterFactory;
use crate::persistence::state::StateStore;
use source_revision::{SourceRevision, calculate_bootstrap_revision, calculate_source_revision};

pub(crate) use harness_sources::HarnessSourceConflict;
pub(crate) use runtime_sessions::SessionDeleteBusy;
pub(crate) use runtime_sessions::session_store_selector_from_filters;
pub(crate) use scheduled_jobs::{
    CreateScheduledJobInput, ScheduledJobOverlapPolicy, UpdateScheduledJobInput,
};
pub use types::{
    AgentDetail, HarnessDetail, SessionBranchDetail, SessionCompactionDetail, SessionDetail,
    SessionEfficiencyDetail, SessionEventDetail, SessionExecutionContextDetail,
    SessionExecutionDetail, SessionFamilyDetail, SessionFamilyMember, SessionGraphDetail,
    SessionGraphTurnDetail, SessionMessageDetail, SessionMessageWindow, SessionPlanExecutionDetail,
    SessionRequestEfficiencyDetail, SessionSearchHit, SessionSummary, SessionTaskExecutionDetail,
    SessionTaskTurnDetail, SessionToolExecutionDetail, SessionTurnEfficiencyDetail,
};
pub(crate) use worklists::WorklistItemsQuery;

#[derive(Debug, Clone)]
pub struct DaemonWatchPaths {
    pub config_path: PathBuf,
    pub agents_dir: PathBuf,
    pub harnesses_dir: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
pub struct DaemonStatus {
    pub config_path: String,
    pub workspace_root: String,
    pub endpoint: String,
    pub registry: RegistrySnapshot,
    pub harnesses: Vec<crate::kernel::HarnessRuntimeSnapshot>,
    pub agent_runtimes: Vec<crate::kernel::agent_manager::AgentStatusSnapshot>,
    pub live_sessions: Vec<crate::kernel::agent_manager::LiveSessionSnapshot>,
}

#[derive(Debug, Clone, Serialize)]
pub struct DaemonRuntimeSnapshot {
    pub config_path: String,
    pub workspace_root: String,
    pub endpoint: String,
    pub registry: RegistrySnapshot,
    pub harnesses: Vec<crate::kernel::HarnessRuntimeSnapshot>,
    pub agent_runtimes: Vec<crate::kernel::agent_manager::AgentStatusSnapshot>,
    pub live_sessions: Vec<crate::kernel::agent_manager::LiveSessionSnapshot>,
}

pub struct DaemonState {
    pub(super) config_path: PathBuf,
    pub(super) config_base: PathBuf,
    pub(super) bootstrap_config: TurinConfig,
    endpoint: PathBuf,
    pub(super) registry_load: RegistryLoad,
    pub(super) kernel: Kernel,
    pub(super) runtime_store: Arc<StateStore>,
    pub(super) scheduler_wake: Option<Arc<Notify>>,
    script_harness_adapter: Arc<dyn HarnessAdapterFactory>,
    bootstrap_revision: SourceRevision,
    source_revision: SourceRevision,
}

#[derive(Debug, Clone)]
pub struct CreateAgentInput {
    pub id: String,
    pub provider: String,
    pub model: String,
    pub system_prompt: Option<String>,
    pub thinking: Option<ThinkingConfig>,
    pub harness: Option<String>,
    pub idle_timeout_seconds: Option<u64>,
    pub enabled: bool,
    pub tools: ToolsConfig,
}

#[derive(Debug, Clone, Default)]
pub struct UpdateAgentInput {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    pub thinking: Option<ThinkingConfig>,
    pub idle_timeout_seconds: Option<u64>,
    pub tools: Option<ToolsConfig>,
}

impl DaemonState {
    pub async fn load(config_path: &Path) -> Result<Self> {
        Self::load_with_harness_adapter(
            config_path,
            crate::kernel::harness_runtime::default_script_adapter_factory()?,
        )
        .await
    }

    pub async fn load_with_harness_adapter(
        config_path: &Path,
        script_harness_adapter: Arc<dyn HarnessAdapterFactory>,
    ) -> Result<Self> {
        let config_path = config_path.to_path_buf();
        let config_base = config_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));

        let mut bootstrap_config = TurinConfig::from_file(&config_path)
            .with_context(|| format!("Failed to load '{}'", config_path.display()))?;
        helpers::normalize_bootstrap_paths(&mut bootstrap_config, &config_base);

        let registry_load = scan_registry(
            &bootstrap_config,
            &config_base,
            Some(&script_harness_adapter),
        )?;
        let effective_config = build_effective_config(&bootstrap_config, &registry_load)?;
        let endpoint = bootstrap_config.resolve_daemon_endpoint(&config_base);
        let runtime_db_path = bootstrap_config.resolve_daemon_runtime_db(&config_base);
        let runtime_store =
            Arc::new(StateStore::open(&runtime_db_path.display().to_string()).await?);

        let mut kernel = Kernel::builder(effective_config)
            .with_harness_adapter(Arc::clone(&script_harness_adapter))
            .build()?;
        kernel.init_state().await?;
        kernel.init_clients()?;
        kernel.host.scheduler = Some(Arc::new(HarnessSchedulerAccess::new(
            Arc::clone(&runtime_store),
            None,
        )));
        kernel
            .agent_manager()
            .bind_scheduler_access(kernel.host.scheduler.clone());
        kernel.init_harness().await?;
        kernel.start_watcher()?;

        let watch_paths = DaemonWatchPaths {
            config_path: config_path.clone(),
            agents_dir: registry_load.agents_dir.clone(),
            harnesses_dir: registry_load.harnesses_dir.clone(),
        };
        let source_revision = calculate_source_revision(&config_path, &watch_paths)?;
        let bootstrap_revision = calculate_bootstrap_revision(&config_path)?;

        Ok(Self {
            config_path,
            config_base,
            bootstrap_config,
            endpoint,
            registry_load,
            kernel,
            runtime_store,
            scheduler_wake: None,
            script_harness_adapter,
            bootstrap_revision,
            source_revision,
        })
    }

    pub fn endpoint(&self) -> &Path {
        &self.endpoint
    }

    pub async fn status(&self) -> DaemonStatus {
        DaemonStatus {
            config_path: self.config_path.display().to_string(),
            workspace_root: self.bootstrap_config.kernel.workspace_root.clone(),
            endpoint: self.endpoint.display().to_string(),
            registry: snapshot(&self.registry_load),
            harnesses: self.kernel.harness_snapshots(),
            agent_runtimes: self.list_agent_runtime_statuses().await,
            live_sessions: self.list_live_sessions().await,
        }
    }

    pub fn watch_paths(&self) -> DaemonWatchPaths {
        DaemonWatchPaths {
            config_path: self.config_path.clone(),
            agents_dir: self
                .bootstrap_config
                .resolve_daemon_agents_dir(&self.config_base),
            harnesses_dir: self
                .bootstrap_config
                .resolve_daemon_harnesses_dir(&self.config_base),
        }
    }

    pub async fn rescan(&mut self) -> Result<DaemonStatus> {
        let bootstrap_revision = calculate_bootstrap_revision(&self.config_path)?;
        if bootstrap_revision == self.bootstrap_revision {
            return self.reconcile_registry().await;
        }
        self.full_rescan(bootstrap_revision).await
    }

    async fn reconcile_registry(&mut self) -> Result<DaemonStatus> {
        self.reconcile_registry_with(&std::collections::HashSet::new())
            .await
    }

    pub(super) async fn reconcile_registry_with(
        &mut self,
        forced_agents: &std::collections::HashSet<String>,
    ) -> Result<DaemonStatus> {
        let registry_load = scan_registry(
            &self.bootstrap_config,
            &self.config_base,
            Some(&self.script_harness_adapter),
        )?;
        let effective_config = build_effective_config(&self.bootstrap_config, &registry_load)?;
        let current_agents = &self.kernel.config().agents;
        let mut affected_agents = std::collections::HashSet::new();
        for agent_id in current_agents.keys().chain(effective_config.agents.keys()) {
            if current_agents.get(agent_id) != effective_config.agents.get(agent_id) {
                affected_agents.insert(agent_id.clone());
            }
        }
        affected_agents.extend(forced_agents.iter().cloned());

        self.kernel
            .reconcile_agent_catalog(effective_config, &affected_agents)
            .await?;
        self.registry_load = registry_load;
        self.source_revision = calculate_source_revision(&self.config_path, &self.watch_paths())?;
        Ok(self.status().await)
    }

    async fn full_rescan(&mut self, bootstrap_revision: SourceRevision) -> Result<DaemonStatus> {
        let active = self.kernel.agent_manager().list_statuses().await;
        if active.iter().any(|status| {
            status.active_tasks > 0 || status.queued_tasks > 0 || status.awaiting_results > 0
        }) {
            anyhow::bail!("Cannot rescan while agent runtimes have active or queued tasks");
        }

        let mut bootstrap_config = TurinConfig::from_file(&self.config_path)
            .with_context(|| format!("Failed to load '{}'", self.config_path.display()))?;
        helpers::normalize_bootstrap_paths(&mut bootstrap_config, &self.config_base);

        let registry_load = scan_registry(
            &bootstrap_config,
            &self.config_base,
            Some(&self.script_harness_adapter),
        )?;
        let effective_config = build_effective_config(&bootstrap_config, &registry_load)?;
        let runtime_db_path = bootstrap_config.resolve_daemon_runtime_db(&self.config_base);
        let runtime_store =
            Arc::new(StateStore::open(&runtime_db_path.display().to_string()).await?);

        let mut new_kernel = Kernel::builder(effective_config)
            .with_harness_adapter(Arc::clone(&self.script_harness_adapter))
            .build()?;
        new_kernel.init_state().await?;
        new_kernel.init_clients()?;
        new_kernel.host.scheduler = Some(Arc::new(HarnessSchedulerAccess::new(
            Arc::clone(&runtime_store),
            self.scheduler_wake.clone(),
        )));
        new_kernel
            .agent_manager()
            .bind_scheduler_access(new_kernel.host.scheduler.clone());
        new_kernel.init_harness().await?;
        new_kernel.start_watcher()?;

        let watch_paths = DaemonWatchPaths {
            config_path: self.config_path.clone(),
            agents_dir: registry_load.agents_dir.clone(),
            harnesses_dir: registry_load.harnesses_dir.clone(),
        };
        let source_revision = calculate_source_revision(&self.config_path, &watch_paths)?;

        let old_kernel = std::mem::replace(&mut self.kernel, new_kernel);
        self.bootstrap_config = bootstrap_config;
        self.registry_load = registry_load;
        self.runtime_store = runtime_store;
        self.bootstrap_revision = bootstrap_revision;
        self.source_revision = source_revision;

        tokio::spawn(async move {
            let mut kernel = old_kernel;
            kernel.shutdown().await;
        });

        Ok(self.status().await)
    }

    pub(super) async fn rescan_if_changed(&mut self) -> Result<DaemonStatus> {
        let revision = calculate_source_revision(&self.config_path, &self.watch_paths())?;
        if revision == self.source_revision {
            return Ok(self.status().await);
        }
        self.rescan().await
    }

    pub async fn reload_runtime(&mut self) -> Result<DaemonStatus> {
        let bootstrap_revision = calculate_bootstrap_revision(&self.config_path)?;
        self.full_rescan(bootstrap_revision).await
    }

    pub async fn shutdown(&mut self) {
        self.scheduler_wake = None;
        self.kernel.shutdown().await;
    }

    pub fn registry_snapshot(&self) -> RegistrySnapshot {
        snapshot(&self.registry_load)
    }

    pub fn runtime_errors(&self) -> Vec<crate::daemon::registry::RegistryIssue> {
        self.registry_snapshot().issues
    }
}

impl From<DaemonStatus> for DaemonRuntimeSnapshot {
    fn from(status: DaemonStatus) -> Self {
        Self {
            config_path: status.config_path,
            workspace_root: status.workspace_root,
            endpoint: status.endpoint,
            registry: status.registry,
            harnesses: status.harnesses,
            agent_runtimes: status.agent_runtimes,
            live_sessions: status.live_sessions,
        }
    }
}
