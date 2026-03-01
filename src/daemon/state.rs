use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::Serialize;

use crate::daemon::registry::{
    AgentFileConfig, RegistryLoad, RegistrySnapshot, build_effective_config, read_agent_file,
    scan_registry, snapshot, write_agent_file,
};
use crate::kernel::Kernel;
use crate::kernel::config::{AgentMode, ThinkingConfig, TurinConfig};

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

#[derive(Debug, Clone)]
pub struct CreateAgentInput {
    pub id: String,
    pub provider: String,
    pub model: String,
    pub system_prompt: Option<String>,
    pub thinking: Option<ThinkingConfig>,
    pub mode: Option<AgentMode>,
    pub harness: Option<String>,
    pub idle_grace_secs: Option<u64>,
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct UpdateAgentInput {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    pub thinking: Option<ThinkingConfig>,
    pub mode: Option<AgentMode>,
    pub idle_grace_secs: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentDetail {
    pub id: String,
    pub directory: String,
    pub enabled: bool,
    pub provider: String,
    pub model: String,
    pub system_prompt: Option<String>,
    pub mode: Option<String>,
    pub harness: Option<String>,
    pub idle_grace_secs: Option<u64>,
    pub has_local_harness: bool,
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

    pub async fn create_agent(&mut self, input: CreateAgentInput) -> Result<AgentDetail> {
        validate_agent_id(&input.id)?;
        let agent_dir = self.agent_dir(&input.id);
        if agent_dir.exists() {
            anyhow::bail!("Agent '{}' already exists", input.id);
        }

        if let Some(shared_harness) = &input.harness {
            let shared_dir = self.watch_paths().harnesses_dir.join(shared_harness);
            if !shared_dir.is_dir() {
                anyhow::bail!("Shared harness '{}' does not exist", shared_harness);
            }
        } else {
            scaffold_local_harness(&agent_dir)?;
        }

        let file = AgentFileConfig {
            id: None,
            enabled: input.enabled,
            system_prompt: input.system_prompt,
            model: input.model,
            provider: input.provider,
            thinking: input.thinking,
            mode: input.mode,
            harness: input.harness,
            idle_grace_secs: input.idle_grace_secs,
        };
        write_agent_file(&agent_dir, &file)?;
        self.rescan().await?;
        self.agent_detail(&input.id)?
            .ok_or_else(|| anyhow!("Agent '{}' was created but not loaded", input.id))
    }

    pub async fn set_agent_enabled(
        &mut self,
        agent_id: &str,
        enabled: bool,
    ) -> Result<AgentDetail> {
        let agent_dir = self.agent_dir(agent_id);
        let mut file = read_agent_file(&agent_dir)?
            .ok_or_else(|| anyhow!("Agent '{}' does not exist", agent_id))?;
        file.enabled = enabled;
        write_agent_file(&agent_dir, &file)?;
        self.rescan().await?;
        self.agent_detail(agent_id)?
            .ok_or_else(|| anyhow!("Agent '{}' could not be reloaded", agent_id))
    }

    pub async fn update_agent(
        &mut self,
        agent_id: &str,
        input: UpdateAgentInput,
    ) -> Result<AgentDetail> {
        let agent_dir = self.agent_dir(agent_id);
        let mut file = read_agent_file(&agent_dir)?
            .ok_or_else(|| anyhow!("Agent '{}' does not exist", agent_id))?;

        if let Some(provider) = input.provider {
            file.provider = provider;
        }
        if let Some(model) = input.model {
            file.model = model;
        }
        if let Some(system_prompt) = input.system_prompt {
            file.system_prompt = Some(system_prompt);
        }
        if let Some(thinking) = input.thinking {
            file.thinking = Some(thinking);
        }
        if let Some(mode) = input.mode {
            file.mode = Some(mode);
        }
        if let Some(idle_grace_secs) = input.idle_grace_secs {
            file.idle_grace_secs = Some(idle_grace_secs);
        }

        write_agent_file(&agent_dir, &file)?;
        self.rescan().await?;
        self.agent_detail(agent_id)?
            .ok_or_else(|| anyhow!("Agent '{}' could not be reloaded", agent_id))
    }

    pub async fn delete_agent(&mut self, agent_id: &str) -> Result<DaemonStatus> {
        let agent_dir = self.agent_dir(agent_id);
        if !agent_dir.is_dir() {
            anyhow::bail!("Agent '{}' does not exist", agent_id);
        }
        std::fs::remove_dir_all(&agent_dir)
            .with_context(|| format!("Failed to remove '{}'", agent_dir.display()))?;
        self.rescan().await
    }

    pub fn agent_detail(&self, agent_id: &str) -> Result<Option<AgentDetail>> {
        let agent_dir = self.agent_dir(agent_id);
        let Some(file) = read_agent_file(&agent_dir)? else {
            return Ok(None);
        };

        Ok(Some(AgentDetail {
            id: agent_id.to_string(),
            directory: agent_dir.display().to_string(),
            enabled: file.enabled,
            provider: file.provider,
            model: file.model,
            system_prompt: file.system_prompt,
            mode: file.mode.map(|mode| format!("{:?}", mode).to_lowercase()),
            harness: file.harness,
            idle_grace_secs: file.idle_grace_secs,
            has_local_harness: agent_dir.join("harness").is_dir(),
        }))
    }

    fn agent_dir(&self, agent_id: &str) -> PathBuf {
        self.watch_paths().agents_dir.join(agent_id)
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

fn validate_agent_id(agent_id: &str) -> Result<()> {
    if agent_id.trim().is_empty() {
        anyhow::bail!("Agent ID cannot be empty");
    }
    if agent_id == "default" {
        anyhow::bail!("'default' is reserved for the bootstrap agent");
    }
    if agent_id.contains('/') || agent_id.contains('\\') || agent_id.contains("..") {
        anyhow::bail!("Agent ID '{}' contains invalid path characters", agent_id);
    }
    Ok(())
}

fn scaffold_local_harness(agent_dir: &Path) -> Result<()> {
    let harness_dir = agent_dir.join("harness");
    std::fs::create_dir_all(&harness_dir)
        .with_context(|| format!("Failed to create '{}'", harness_dir.display()))?;
    let main_lua = harness_dir.join("main.lua");
    if main_lua.exists() {
        return Ok(());
    }

    let tmp_path = harness_dir.join(format!(".main.lua.{}.tmp", uuid::Uuid::now_v7().simple()));
    std::fs::write(&tmp_path, "-- Turin daemon scaffold\n")
        .with_context(|| format!("Failed to write '{}'", tmp_path.display()))?;
    std::fs::rename(&tmp_path, &main_lua).with_context(|| {
        format!(
            "Failed to atomically replace '{}' from '{}'",
            main_lua.display(),
            tmp_path.display()
        )
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn write_bootstrap(root: &Path) -> Result<PathBuf> {
        std::fs::create_dir_all(root.join("default-harness"))?;
        std::fs::write(
            root.join("default-harness").join("main.lua"),
            "-- bootstrap\n",
        )?;
        let config_path = root.join("turin.toml");
        std::fs::write(
            &config_path,
            r#"[agent]
id = "default"
system_prompt = "bootstrap"
model = "mock-model"
provider = "mock"

[kernel]
workspace_root = "."

[persistence]
database_path = "state.db"

[harness]
directory = "default-harness"
fs_root = "."

[providers.mock]
type = "mock"

[embeddings]
type = "no_op"
"#,
        )?;
        Ok(config_path)
    }

    #[tokio::test]
    async fn create_disable_and_delete_agent_updates_filesystem_state() -> Result<()> {
        let temp = tempdir()?;
        let config_path = write_bootstrap(temp.path())?;
        let mut state = DaemonState::load(&config_path).await?;

        let created = state
            .create_agent(CreateAgentInput {
                id: "docs-reviewer".to_string(),
                provider: "mock".to_string(),
                model: "mock-model".to_string(),
                system_prompt: Some("Review docs".to_string()),
                thinking: None,
                mode: None,
                harness: None,
                idle_grace_secs: None,
                enabled: true,
            })
            .await?;

        assert_eq!(created.id, "docs-reviewer");
        assert!(created.has_local_harness);
        assert!(
            temp.path()
                .join("agents")
                .join("docs-reviewer")
                .join("harness")
                .join("main.lua")
                .exists()
        );

        let disabled = state.set_agent_enabled("docs-reviewer", false).await?;
        assert!(!disabled.enabled);

        let updated = state
            .update_agent(
                "docs-reviewer",
                UpdateAgentInput {
                    model: Some("mock-model-2".to_string()),
                    system_prompt: Some("Review docs carefully".to_string()),
                    ..UpdateAgentInput::default()
                },
            )
            .await?;
        assert_eq!(updated.model, "mock-model-2");
        assert_eq!(
            updated.system_prompt.as_deref(),
            Some("Review docs carefully")
        );

        let status = state.delete_agent("docs-reviewer").await?;
        assert!(
            status
                .registry
                .agents
                .iter()
                .all(|agent| agent.id != "docs-reviewer")
        );
        assert!(!temp.path().join("agents").join("docs-reviewer").exists());

        Ok(())
    }
}
