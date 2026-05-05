mod channel_validation;
mod helpers;
mod registry_ops;
mod runtime_sessions;
mod runtime_tasks;
mod scheduled_jobs;
#[cfg(test)]
mod tests;
mod types;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::Serialize;
use tokio::sync::Notify;
use turin_types::ToolsConfig;

use crate::daemon::channels::ChannelRuntimeSnapshot;
use crate::daemon::registry::{
    RegistryLoad, RegistrySnapshot, build_effective_config, scan_registry, snapshot,
};
use crate::harness::scheduler::HarnessSchedulerAccess;
use crate::kernel::Kernel;
use crate::kernel::config::{ThinkingConfig, TurinConfig};
use crate::persistence::state::StateStore;

pub(crate) use runtime_sessions::session_store_selector_from_filters;
pub(crate) use scheduled_jobs::{
    CreateScheduledJobInput, ScheduledJobOverlapPolicy, UpdateScheduledJobInput,
};
pub use types::{
    AgentDetail, ChannelDetail, HarnessDetail, SessionBranchDetail, SessionDetail,
    SessionEventDetail, SessionMessageDetail, SessionSearchHit, SessionSummary,
    SessionToolExecutionDetail,
};

#[derive(Debug, Clone)]
pub struct DaemonWatchPaths {
    pub config_path: PathBuf,
    pub agents_dir: PathBuf,
    pub harnesses_dir: PathBuf,
    pub channels_dir: PathBuf,
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
    pub channel_runtimes: Vec<ChannelRuntimeSnapshot>,
}

pub struct DaemonState {
    pub(super) config_path: PathBuf,
    pub(super) config_base: PathBuf,
    pub(super) bootstrap_config: TurinConfig,
    endpoint: PathBuf,
    pub(super) registry_load: RegistryLoad,
    pub(super) kernel: Kernel,
    pub(super) jobs_store: Arc<StateStore>,
    pub(super) scheduler_wake: Option<Arc<Notify>>,
}

#[derive(Debug, Clone)]
pub struct CreateAgentInput {
    pub id: String,
    pub provider: String,
    pub model: String,
    pub system_prompt: Option<String>,
    pub thinking: Option<ThinkingConfig>,
    pub harness: Option<String>,
    pub runtime_idle_secs: Option<u64>,
    pub enabled: bool,
    pub tools: ToolsConfig,
}

#[derive(Debug, Clone, Default)]
pub struct UpdateAgentInput {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    pub thinking: Option<ThinkingConfig>,
    pub runtime_idle_secs: Option<u64>,
    pub tools: Option<ToolsConfig>,
}

#[derive(Debug, Clone)]
pub struct CreateChannelInput {
    pub id: String,
    pub kind: String,
    pub agent_id: String,
    pub idle_ttl_secs: Option<u64>,
    pub enabled: bool,
    pub settings: serde_json::Value,
}

#[derive(Debug, Clone, Default)]
pub struct UpdateChannelInput {
    pub kind: Option<String>,
    pub agent_id: Option<String>,
    pub idle_ttl_secs: Option<u64>,
    pub settings: Option<serde_json::Value>,
}

impl DaemonState {
    pub async fn load(config_path: &Path) -> Result<Self> {
        let config_path = config_path.to_path_buf();
        let config_base = config_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));

        let mut bootstrap_config = TurinConfig::from_file(&config_path)
            .with_context(|| format!("Failed to load '{}'", config_path.display()))?;
        helpers::normalize_bootstrap_paths(&mut bootstrap_config, &config_base);

        let registry_load = scan_registry(&bootstrap_config, &config_base)?;
        let effective_config = build_effective_config(&bootstrap_config, &registry_load)?;
        let endpoint = bootstrap_config.resolve_daemon_endpoint(&config_base);
        let jobs_db_path = bootstrap_config.resolve_daemon_jobs_db(&config_base);
        let jobs_store = Arc::new(StateStore::open(&jobs_db_path.display().to_string()).await?);

        let mut kernel = Kernel::builder(effective_config).build()?;
        kernel.init_state().await?;
        kernel.init_clients()?;
        kernel.host.scheduler = Some(Arc::new(HarnessSchedulerAccess::new(
            Arc::clone(&jobs_store),
            None,
        )));
        kernel
            .agent_manager()
            .bind_scheduler_access(kernel.host.scheduler.clone());
        kernel.init_harness().await?;
        kernel.start_watcher()?;

        Ok(Self {
            config_path,
            config_base,
            bootstrap_config,
            endpoint,
            registry_load,
            kernel,
            jobs_store,
            scheduler_wake: None,
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
            channels_dir: self
                .bootstrap_config
                .resolve_daemon_channels_dir(&self.config_base),
        }
    }

    pub async fn rescan(&mut self) -> Result<DaemonStatus> {
        let active = self.kernel.agent_manager().list_statuses().await;
        if active.iter().any(|status| {
            status.active_tasks > 0 || status.queued_tasks > 0 || status.awaiting_results > 0
        }) {
            anyhow::bail!("Cannot rescan while agent runtimes have active or queued tasks");
        }

        let mut bootstrap_config = TurinConfig::from_file(&self.config_path)
            .with_context(|| format!("Failed to load '{}'", self.config_path.display()))?;
        helpers::normalize_bootstrap_paths(&mut bootstrap_config, &self.config_base);

        let registry_load = scan_registry(&bootstrap_config, &self.config_base)?;
        let effective_config = build_effective_config(&bootstrap_config, &registry_load)?;
        let jobs_db_path = bootstrap_config.resolve_daemon_jobs_db(&self.config_base);
        let jobs_store = Arc::new(StateStore::open(&jobs_db_path.display().to_string()).await?);

        let mut new_kernel = Kernel::builder(effective_config).build()?;
        new_kernel.init_state().await?;
        new_kernel.init_clients()?;
        new_kernel.host.scheduler = Some(Arc::new(HarnessSchedulerAccess::new(
            Arc::clone(&jobs_store),
            self.scheduler_wake.clone(),
        )));
        new_kernel
            .agent_manager()
            .bind_scheduler_access(new_kernel.host.scheduler.clone());
        new_kernel.init_harness().await?;
        new_kernel.start_watcher()?;

        let old_kernel = std::mem::replace(&mut self.kernel, new_kernel);
        self.bootstrap_config = bootstrap_config;
        self.registry_load = registry_load;
        self.jobs_store = jobs_store;

        tokio::spawn(async move {
            let mut kernel = old_kernel;
            kernel.shutdown_mcp_clients().await;
        });

        Ok(self.status().await)
    }

    pub async fn reload_runtime(&mut self) -> Result<DaemonStatus> {
        self.rescan().await
    }

    pub fn registry_snapshot(&self) -> RegistrySnapshot {
        snapshot(&self.registry_load)
    }

    pub fn runtime_errors(&self) -> Vec<crate::daemon::registry::RegistryIssue> {
        self.registry_snapshot().issues
    }
}

impl DaemonRuntimeSnapshot {
    pub fn from_parts(status: DaemonStatus, channel_runtimes: Vec<ChannelRuntimeSnapshot>) -> Self {
        Self {
            config_path: status.config_path,
            workspace_root: status.workspace_root,
            endpoint: status.endpoint,
            registry: status.registry,
            harnesses: status.harnesses,
            agent_runtimes: status.agent_runtimes,
            live_sessions: status.live_sessions,
            channel_runtimes,
        }
    }
}
