use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Serialize;

use crate::daemon::registry::{
    RegistryLoad, RegistrySnapshot, build_effective_config, scan_registry, snapshot,
};
use crate::kernel::Kernel;
use crate::kernel::config::TurinConfig;

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
    pub socket_path: String,
    pub registry: RegistrySnapshot,
    pub harnesses: Vec<crate::kernel::HarnessRuntimeSnapshot>,
}

pub struct DaemonState {
    config_path: PathBuf,
    config_base: PathBuf,
    bootstrap_config: TurinConfig,
    socket_path: PathBuf,
    registry_load: RegistryLoad,
    kernel: Kernel,
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
        normalize_bootstrap_paths(&mut bootstrap_config, &config_base);

        let registry_load = scan_registry(&bootstrap_config, &config_base)?;
        let effective_config = build_effective_config(&bootstrap_config, &registry_load)?;
        let socket_path = bootstrap_config.resolve_daemon_socket_path(&config_base);

        let mut kernel = Kernel::builder(effective_config).build()?;
        kernel.init_state().await?;
        kernel.init_clients()?;
        kernel.init_harness().await?;
        kernel.start_watcher()?;

        Ok(Self {
            config_path,
            config_base,
            bootstrap_config,
            socket_path,
            registry_load,
            kernel,
        })
    }

    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    pub fn status(&self) -> DaemonStatus {
        DaemonStatus {
            config_path: self.config_path.display().to_string(),
            workspace_root: self.bootstrap_config.kernel.workspace_root.clone(),
            socket_path: self.socket_path.display().to_string(),
            registry: snapshot(&self.registry_load),
            harnesses: self.kernel.harness_snapshots(),
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
        let mut bootstrap_config = TurinConfig::from_file(&self.config_path)
            .with_context(|| format!("Failed to load '{}'", self.config_path.display()))?;
        normalize_bootstrap_paths(&mut bootstrap_config, &self.config_base);

        let registry_load = scan_registry(&bootstrap_config, &self.config_base)?;
        let effective_config = build_effective_config(&bootstrap_config, &registry_load)?;

        let mut new_kernel = Kernel::builder(effective_config).build()?;
        new_kernel.init_state().await?;
        new_kernel.init_clients()?;
        new_kernel.init_harness().await?;
        new_kernel.start_watcher()?;

        let old_kernel = std::mem::replace(&mut self.kernel, new_kernel);
        self.bootstrap_config = bootstrap_config;
        self.registry_load = registry_load;

        tokio::spawn(async move {
            let mut kernel = old_kernel;
            kernel.shutdown_mcp_clients().await;
        });

        Ok(self.status())
    }

    pub fn registry_snapshot(&self) -> RegistrySnapshot {
        snapshot(&self.registry_load)
    }
}

fn normalize_bootstrap_paths(config: &mut TurinConfig, config_base: &Path) {
    let workspace_root = config.resolve_workspace_root(config_base);
    config.kernel.workspace_root = workspace_root.display().to_string();

    if Path::new(&config.harness.directory).is_relative() {
        config.harness.directory = workspace_root
            .join(&config.harness.directory)
            .display()
            .to_string();
    }

    if Path::new(&config.harness.fs_root).is_relative() && config.harness.fs_root != "." {
        config.harness.fs_root = workspace_root
            .join(&config.harness.fs_root)
            .display()
            .to_string();
    }
}
