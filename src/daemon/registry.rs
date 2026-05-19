use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use turin_types::ToolsConfig;

use crate::kernel::agent_manager::AgentManager;
use crate::kernel::config::{
    AgentConfig, ContextPersistenceConfig, HarnessConfig, InferenceOverrideConfig, ThinkingConfig,
    TurinConfig,
};
use crate::kernel::governance::GovernanceManager;
use crate::kernel::harness_runtime::{HarnessRuntime, HarnessRuntimeInitContext};
use crate::kernel::policy::RuntimePolicyManager;
use crate::persistence::manager::StoreManager;

#[derive(Debug, Clone, Serialize)]
pub struct RegistryIssue {
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SharedHarnessSummary {
    pub id: String,
    pub directory: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChannelSummary {
    pub id: String,
    pub directory: String,
    pub enabled: bool,
    pub kind: String,
    pub agent_id: String,
    pub idle_timeout_seconds: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentSummary {
    pub id: String,
    pub directory: String,
    pub enabled: bool,
    pub provider: String,
    pub model: String,
    pub idle_timeout_seconds: Option<u64>,
    pub harness_kind: String,
    pub harness_ref: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RegistrySnapshot {
    pub agents_dir: String,
    pub harnesses_dir: String,
    pub channels_dir: String,
    pub agents: Vec<AgentSummary>,
    pub shared_harnesses: Vec<SharedHarnessSummary>,
    pub channels: Vec<ChannelSummary>,
    pub issues: Vec<RegistryIssue>,
}

#[derive(Debug, Clone)]
pub struct DiscoveredAgent {
    pub id: String,
    pub directory: PathBuf,
    pub enabled: bool,
    pub agent_config: AgentConfig,
    pub harness_id: String,
    pub harness_kind: HarnessKind,
    pub harness_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarnessKind {
    Local,
    Shared,
}

#[derive(Debug, Clone)]
pub struct SharedHarness {
    pub id: String,
    pub directory: PathBuf,
}

#[derive(Debug, Clone)]
pub struct DiscoveredChannel {
    pub id: String,
    pub directory: PathBuf,
    pub enabled: bool,
    pub kind: String,
    pub agent_id: String,
    pub idle_timeout_seconds: Option<u64>,
    pub persistence: ContextPersistenceConfig,
    pub inference: InferenceOverrideConfig,
    pub extra: toml::Table,
}

#[derive(Debug, Clone)]
pub struct RegistryLoad {
    pub agents_dir: PathBuf,
    pub harnesses_dir: PathBuf,
    pub channels_dir: PathBuf,
    pub agents: Vec<DiscoveredAgent>,
    pub shared_harnesses: Vec<SharedHarness>,
    pub channels: Vec<DiscoveredChannel>,
    pub issues: Vec<RegistryIssue>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct AgentFileConfig {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub system_prompt: Option<String>,
    pub model: String,
    pub provider: String,
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
    #[serde(default)]
    pub harness: Option<String>,
    #[serde(default)]
    pub idle_timeout_seconds: Option<u64>,
    #[serde(default)]
    pub tools: ToolsConfig,
    #[serde(default)]
    pub inference: InferenceOverrideConfig,
    #[serde(default)]
    pub persistence: ContextPersistenceConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct ChannelFileConfig {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    pub kind: String,
    pub agent_id: String,
    #[serde(default)]
    pub idle_timeout_seconds: Option<u64>,
    #[serde(default)]
    pub persistence: ContextPersistenceConfig,
    #[serde(default)]
    pub inference: InferenceOverrideConfig,
    #[serde(flatten)]
    pub extra: toml::Table,
}

fn default_enabled() -> bool {
    true
}

pub(crate) fn read_agent_file(agent_dir: &Path) -> Result<Option<AgentFileConfig>> {
    let agent_toml = agent_dir.join("config.toml");
    if !agent_toml.exists() {
        return Ok(None);
    }

    let raw = fs::read_to_string(&agent_toml)
        .with_context(|| format!("Failed to read '{}'", agent_toml.display()))?;
    let parsed: AgentFileConfig = toml::from_str(&raw)
        .with_context(|| format!("Failed to parse '{}'", agent_toml.display()))?;
    Ok(Some(parsed))
}

pub(crate) fn read_channel_file(channel_dir: &Path) -> Result<Option<ChannelFileConfig>> {
    let channel_toml = channel_dir.join("config.toml");
    if !channel_toml.exists() {
        return Ok(None);
    }

    let raw = fs::read_to_string(&channel_toml)
        .with_context(|| format!("Failed to read '{}'", channel_toml.display()))?;
    let parsed: ChannelFileConfig = toml::from_str(&raw)
        .with_context(|| format!("Failed to parse '{}'", channel_toml.display()))?;
    Ok(Some(parsed))
}

pub(crate) fn write_agent_file(agent_dir: &Path, config: &AgentFileConfig) -> Result<()> {
    fs::create_dir_all(agent_dir)
        .with_context(|| format!("Failed to create agent directory '{}'", agent_dir.display()))?;
    let agent_toml = agent_dir.join("config.toml");
    let tmp_path = agent_dir.join(format!(
        ".config.toml.{}.tmp",
        uuid::Uuid::now_v7().simple()
    ));
    let body = toml::to_string_pretty(config)
        .with_context(|| format!("Failed to serialize '{}'", agent_toml.display()))?;
    fs::write(&tmp_path, body)
        .with_context(|| format!("Failed to write '{}'", tmp_path.display()))?;
    fs::rename(&tmp_path, &agent_toml).with_context(|| {
        format!(
            "Failed to atomically replace '{}' from '{}'",
            agent_toml.display(),
            tmp_path.display()
        )
    })?;
    Ok(())
}

pub(crate) fn write_channel_file(channel_dir: &Path, config: &ChannelFileConfig) -> Result<()> {
    fs::create_dir_all(channel_dir).with_context(|| {
        format!(
            "Failed to create channel directory '{}'",
            channel_dir.display()
        )
    })?;
    let channel_toml = channel_dir.join("config.toml");
    let tmp_path = channel_dir.join(format!(
        ".config.toml.{}.tmp",
        uuid::Uuid::now_v7().simple()
    ));
    let body = toml::to_string_pretty(config)
        .with_context(|| format!("Failed to serialize '{}'", channel_toml.display()))?;
    fs::write(&tmp_path, body)
        .with_context(|| format!("Failed to write '{}'", tmp_path.display()))?;
    fs::rename(&tmp_path, &channel_toml).with_context(|| {
        format!(
            "Failed to atomically replace '{}' from '{}'",
            channel_toml.display(),
            tmp_path.display()
        )
    })?;
    Ok(())
}

pub fn scan_registry(config: &TurinConfig, config_base: &Path) -> Result<RegistryLoad> {
    let agents_dir = config.resolve_daemon_agents_dir(config_base);
    let harnesses_dir = config.resolve_daemon_harnesses_dir(config_base);
    let channels_dir = config.resolve_daemon_channels_dir(config_base);

    let mut issues = Vec::new();
    let mut shared_harness_map = scan_shared_harnesses(&harnesses_dir)?;
    shared_harness_map.retain(
        |id, directory| match validate_harness_dir(config, id, directory) {
            Ok(()) => true,
            Err(err) => {
                issues.push(RegistryIssue {
                    path: directory.display().to_string(),
                    message: err.to_string(),
                });
                false
            }
        },
    );
    let mut agents = Vec::new();

    if agents_dir.exists() {
        for entry in fs::read_dir(&agents_dir)
            .with_context(|| format!("Failed to read agents dir '{}'", agents_dir.display()))?
        {
            let entry = match entry {
                Ok(entry) => entry,
                Err(err) => {
                    issues.push(RegistryIssue {
                        path: agents_dir.display().to_string(),
                        message: format!("Failed to read agent directory entry: {}", err),
                    });
                    continue;
                }
            };

            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            match scan_agent_dir(config, &path, &shared_harness_map) {
                Ok(Some(agent)) => agents.push(agent),
                Ok(None) => {}
                Err(err) => issues.push(RegistryIssue {
                    path: path.display().to_string(),
                    message: err.to_string(),
                }),
            }
        }
    }

    let mut shared_harnesses: Vec<_> = shared_harness_map
        .into_iter()
        .map(|(id, directory)| SharedHarness { id, directory })
        .collect();
    let mut channels = scan_channels(config, &channels_dir, &agents, &mut issues)?;
    shared_harnesses.sort_by(|a, b| a.id.cmp(&b.id));
    agents.sort_by(|a, b| a.id.cmp(&b.id));
    channels.sort_by(|a, b| a.id.cmp(&b.id));
    issues.sort_by(|a, b| a.path.cmp(&b.path));

    Ok(RegistryLoad {
        agents_dir,
        harnesses_dir,
        channels_dir,
        agents,
        shared_harnesses,
        channels,
        issues,
    })
}

fn scan_channels(
    bootstrap: &TurinConfig,
    channels_dir: &Path,
    agents: &[DiscoveredAgent],
    issues: &mut Vec<RegistryIssue>,
) -> Result<Vec<DiscoveredChannel>> {
    let mut channels = Vec::new();

    if !channels_dir.exists() {
        return Ok(channels);
    }

    for entry in fs::read_dir(channels_dir)
        .with_context(|| format!("Failed to read channels dir '{}'", channels_dir.display()))?
    {
        let entry = match entry {
            Ok(entry) => entry,
            Err(err) => {
                issues.push(RegistryIssue {
                    path: channels_dir.display().to_string(),
                    message: format!("Failed to read channel directory entry: {}", err),
                });
                continue;
            }
        };

        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        match scan_channel_dir(bootstrap, &path, agents) {
            Ok(Some(channel)) => channels.push(channel),
            Ok(None) => {}
            Err(err) => issues.push(RegistryIssue {
                path: path.display().to_string(),
                message: err.to_string(),
            }),
        }
    }

    Ok(channels)
}

fn scan_shared_harnesses(harnesses_dir: &Path) -> Result<HashMap<String, PathBuf>> {
    let mut shared = HashMap::new();

    if !harnesses_dir.exists() {
        return Ok(shared);
    }

    for entry in fs::read_dir(harnesses_dir)
        .with_context(|| format!("Failed to read harnesses dir '{}'", harnesses_dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if name == "default" {
            continue;
        }

        shared.insert(name.to_string(), path);
    }

    Ok(shared)
}

fn scan_agent_dir(
    bootstrap: &TurinConfig,
    agent_dir: &Path,
    shared_harnesses: &HashMap<String, PathBuf>,
) -> Result<Option<DiscoveredAgent>> {
    let agent_id = agent_dir
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow::anyhow!("Agent directory name is not valid UTF-8"))?
        .to_string();

    if agent_id == "default" {
        anyhow::bail!("'default' is reserved for the bootstrap agent");
    }

    let Some(parsed) = read_agent_file(agent_dir)? else {
        return Ok(None);
    };

    if let Some(explicit_id) = &parsed.id
        && explicit_id != &agent_id
    {
        anyhow::bail!(
            "agent config id '{}' does not match directory name '{}'",
            explicit_id,
            agent_id
        );
    }

    let local_harness_dir = agent_dir.join("harness");
    let local_harness_exists = local_harness_dir.is_dir();

    if parsed.harness.is_some() && local_harness_exists {
        anyhow::bail!(
            "agent declares a shared harness and also has a local harness/ directory; choose one"
        );
    }

    let (harness_id, harness_kind, harness_dir) = if let Some(shared_id) = &parsed.harness {
        let shared_dir = shared_harnesses.get(shared_id).ok_or_else(|| {
            anyhow::anyhow!("agent references unknown shared harness '{}'", shared_id)
        })?;
        (
            shared_id.clone(),
            HarnessKind::Shared,
            Some(shared_dir.clone()),
        )
    } else if local_harness_exists {
        validate_harness_dir(
            bootstrap,
            &format!("agent::{}", agent_id),
            &local_harness_dir,
        )
        .with_context(|| format!("Failed to validate local harness for agent '{}'", agent_id))?;
        (
            format!("agent::{}", agent_id),
            HarnessKind::Local,
            Some(local_harness_dir),
        )
    } else {
        anyhow::bail!(
            "agent requires either a local harness/ directory or a harness = \"<id>\" reference"
        );
    };

    let agent_config = AgentConfig {
        id: agent_id.clone(),
        system_prompt: parsed
            .system_prompt
            .unwrap_or_else(|| bootstrap.agent.system_prompt.clone()),
        model: parsed.model,
        provider: parsed.provider,
        thinking: parsed.thinking,
        harness: Some(harness_id.clone()),
        idle_timeout_seconds: parsed
            .idle_timeout_seconds
            .or(bootstrap.agent.idle_timeout_seconds),
        tools: parsed.tools,
        inference: parsed.inference,
        persistence: parsed.persistence,
    };

    Ok(Some(DiscoveredAgent {
        id: agent_id,
        directory: agent_dir.to_path_buf(),
        enabled: parsed.enabled,
        agent_config,
        harness_id,
        harness_kind,
        harness_dir,
    }))
}

fn scan_channel_dir(
    bootstrap: &TurinConfig,
    channel_dir: &Path,
    agents: &[DiscoveredAgent],
) -> Result<Option<DiscoveredChannel>> {
    let channel_id = channel_dir
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow::anyhow!("Channel directory name is not valid UTF-8"))?
        .to_string();

    let Some(parsed) = read_channel_file(channel_dir)? else {
        return Ok(None);
    };

    if let Some(explicit_id) = &parsed.id
        && explicit_id != &channel_id
    {
        anyhow::bail!(
            "channel config id '{}' does not match directory name '{}'",
            explicit_id,
            channel_id
        );
    }

    anyhow::ensure!(
        !parsed.kind.trim().is_empty(),
        "channel kind must not be empty"
    );

    let known_agent = parsed.agent_id == bootstrap.agent.id
        || agents.iter().any(|agent| agent.id == parsed.agent_id);
    anyhow::ensure!(
        known_agent,
        "channel references unknown agent '{}'",
        parsed.agent_id
    );
    parsed.inference.validate_shallow(
        &bootstrap.providers,
        &format!("channel '{}'.inference", channel_id),
    )?;
    if parsed.persistence.state.is_some() {
        bootstrap
            .persistence
            .resolve_context_state_selector(Some(&parsed.persistence))
            .context("invalid channel persistence.state")?;
    }
    if parsed.persistence.store.is_some() {
        bootstrap
            .persistence
            .resolve_context_store_selector(Some(&parsed.persistence))
            .context("invalid channel persistence.store")?;
    }
    if !parsed.inference.is_empty() {
        let agent_inference = if parsed.agent_id == bootstrap.agent.id {
            &bootstrap.agent.inference
        } else {
            &agents
                .iter()
                .find(|agent| agent.id == parsed.agent_id)
                .expect("known agent already checked")
                .agent_config
                .inference
        };
        let effective = bootstrap
            .inference
            .merged_with(agent_inference)
            .merged_with(&parsed.inference);
        effective.validate_complete(
            &bootstrap.providers,
            &format!("channel '{}'.inference", channel_id),
        )?;
    }

    Ok(Some(DiscoveredChannel {
        id: channel_id,
        directory: channel_dir.to_path_buf(),
        enabled: parsed.enabled,
        kind: parsed.kind,
        agent_id: parsed.agent_id,
        idle_timeout_seconds: parsed.idle_timeout_seconds,
        persistence: parsed.persistence,
        inference: parsed.inference,
        extra: parsed.extra,
    }))
}

pub fn build_effective_config(bootstrap: &TurinConfig, load: &RegistryLoad) -> Result<TurinConfig> {
    let mut effective = bootstrap.clone();
    effective.agents.clear();
    effective.harnesses.clear();

    for shared in &load.shared_harnesses {
        effective.harnesses.insert(
            shared.id.clone(),
            HarnessConfig {
                directory: shared.directory.to_string_lossy().to_string(),
                fs_root: bootstrap.harness.fs_root.clone(),
                memory_limit_mb: bootstrap.harness.memory_limit_mb,
            },
        );
    }

    for agent in &load.agents {
        if !agent.enabled {
            continue;
        }

        if let Some(dir) = &agent.harness_dir {
            effective.harnesses.insert(
                agent.harness_id.clone(),
                HarnessConfig {
                    directory: dir.to_string_lossy().to_string(),
                    fs_root: bootstrap.harness.fs_root.clone(),
                    memory_limit_mb: bootstrap.harness.memory_limit_mb,
                },
            );
        }

        effective
            .agents
            .insert(agent.id.clone(), agent.agent_config.clone());
    }

    effective.validate()?;
    Ok(effective)
}

fn validate_harness_dir(
    bootstrap: &TurinConfig,
    harness_id: &str,
    harness_dir: &Path,
) -> Result<()> {
    let workspace_root = PathBuf::from(&bootstrap.kernel.workspace_root);
    let fs_root = if bootstrap.harness.fs_root == "." {
        workspace_root.clone()
    } else {
        PathBuf::from(&bootstrap.harness.fs_root)
    };

    let config = Arc::new(bootstrap.clone());
    let store_manager = Arc::new(StoreManager::new(
        workspace_root.clone(),
        turin_types::layout::default_stores_dir_for_workspace(&workspace_root),
    ));
    let agent_manager = Arc::new(AgentManager::new(config.clone(), store_manager.clone()));
    let runtime = HarnessRuntime::new(
        harness_id.to_string(),
        harness_dir.to_path_buf(),
        fs_root,
        workspace_root,
        bootstrap.kernel.initial_spawn_depth,
    );

    runtime.validate(HarnessRuntimeInitContext {
        config,
        clients: HashMap::new(),
        store_manager,
        agent_manager,
        policy_manager: Arc::new(RuntimePolicyManager::new()),
        governance_manager: Arc::new(GovernanceManager::new(bootstrap.governance.clone())),
        scheduler: None,
        embedding_provider: None,
    })?;

    Ok(())
}

pub fn snapshot(load: &RegistryLoad) -> RegistrySnapshot {
    RegistrySnapshot {
        agents_dir: load.agents_dir.display().to_string(),
        harnesses_dir: load.harnesses_dir.display().to_string(),
        channels_dir: load.channels_dir.display().to_string(),
        agents: load
            .agents
            .iter()
            .map(|agent| AgentSummary {
                id: agent.id.clone(),
                directory: agent.directory.display().to_string(),
                enabled: agent.enabled,
                provider: agent.agent_config.provider.clone(),
                model: agent.agent_config.model.clone(),
                idle_timeout_seconds: agent.agent_config.idle_timeout_seconds,
                harness_kind: match agent.harness_kind {
                    HarnessKind::Local => "local".to_string(),
                    HarnessKind::Shared => "shared".to_string(),
                },
                harness_ref: agent.harness_id.clone(),
            })
            .collect(),
        shared_harnesses: load
            .shared_harnesses
            .iter()
            .map(|harness| SharedHarnessSummary {
                id: harness.id.clone(),
                directory: harness.directory.display().to_string(),
            })
            .collect(),
        channels: load
            .channels
            .iter()
            .map(|channel| ChannelSummary {
                id: channel.id.clone(),
                directory: channel.directory.display().to_string(),
                enabled: channel.enabled,
                kind: channel.kind.clone(),
                agent_id: channel.agent_id.clone(),
                idle_timeout_seconds: channel.idle_timeout_seconds,
            })
            .collect(),
        issues: load.issues.clone(),
    }
}

#[cfg(test)]
#[path = "tests/registry.rs"]
mod tests;
