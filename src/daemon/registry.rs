use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::kernel::config::{AgentConfig, AgentMode, HarnessConfig, ThinkingConfig, TurinConfig};

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
pub struct AgentSummary {
    pub id: String,
    pub directory: String,
    pub enabled: bool,
    pub provider: String,
    pub model: String,
    pub mode: String,
    pub harness_kind: String,
    pub harness_ref: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RegistrySnapshot {
    pub agents_dir: String,
    pub harnesses_dir: String,
    pub agents: Vec<AgentSummary>,
    pub shared_harnesses: Vec<SharedHarnessSummary>,
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
pub struct RegistryLoad {
    pub agents_dir: PathBuf,
    pub harnesses_dir: PathBuf,
    pub agents: Vec<DiscoveredAgent>,
    pub shared_harnesses: Vec<SharedHarness>,
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
    pub mode: Option<AgentMode>,
    #[serde(default)]
    pub harness: Option<String>,
    #[serde(default)]
    pub idle_grace_secs: Option<u64>,
}

fn default_enabled() -> bool {
    true
}

pub(crate) fn read_agent_file(agent_dir: &Path) -> Result<Option<AgentFileConfig>> {
    let agent_toml = agent_dir.join("agent.toml");
    if !agent_toml.exists() {
        return Ok(None);
    }

    let raw = fs::read_to_string(&agent_toml)
        .with_context(|| format!("Failed to read '{}'", agent_toml.display()))?;
    let parsed: AgentFileConfig = toml::from_str(&raw)
        .with_context(|| format!("Failed to parse '{}'", agent_toml.display()))?;
    Ok(Some(parsed))
}

pub(crate) fn write_agent_file(agent_dir: &Path, config: &AgentFileConfig) -> Result<()> {
    fs::create_dir_all(agent_dir)
        .with_context(|| format!("Failed to create agent directory '{}'", agent_dir.display()))?;
    let agent_toml = agent_dir.join("agent.toml");
    let tmp_path = agent_dir.join(format!(".agent.toml.{}.tmp", uuid::Uuid::now_v7().simple()));
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

pub fn scan_registry(config: &TurinConfig, config_base: &Path) -> Result<RegistryLoad> {
    let agents_dir = config.resolve_daemon_agents_dir(config_base);
    let harnesses_dir = config.resolve_daemon_harnesses_dir(config_base);

    let shared_harness_map = scan_shared_harnesses(&harnesses_dir)?;
    let mut issues = Vec::new();
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
    shared_harnesses.sort_by(|a, b| a.id.cmp(&b.id));
    agents.sort_by(|a, b| a.id.cmp(&b.id));
    issues.sort_by(|a, b| a.path.cmp(&b.path));

    Ok(RegistryLoad {
        agents_dir,
        harnesses_dir,
        agents,
        shared_harnesses,
        issues,
    })
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
            "agent.toml id '{}' does not match directory name '{}'",
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
        mode: parsed.mode.unwrap_or(bootstrap.agent.mode.clone()),
        harness: Some(harness_id.clone()),
        idle_grace_secs: parsed.idle_grace_secs,
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

pub fn snapshot(load: &RegistryLoad) -> RegistrySnapshot {
    RegistrySnapshot {
        agents_dir: load.agents_dir.display().to_string(),
        harnesses_dir: load.harnesses_dir.display().to_string(),
        agents: load
            .agents
            .iter()
            .map(|agent| AgentSummary {
                id: agent.id.clone(),
                directory: agent.directory.display().to_string(),
                enabled: agent.enabled,
                provider: agent.agent_config.provider.clone(),
                model: agent.agent_config.model.clone(),
                mode: format!("{:?}", agent.agent_config.mode).to_lowercase(),
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
        issues: load.issues.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn bootstrap_config(root: &Path) -> TurinConfig {
        let mut config = TurinConfig::default();
        config.agent.model = "mock-model".to_string();
        config.agent.provider = "mock".to_string();
        config.kernel.workspace_root = root.to_string_lossy().to_string();
        config.harness.directory = root.join("default-harness").to_string_lossy().to_string();
        config.providers.insert(
            "mock".to_string(),
            crate::kernel::config::ProviderConfig {
                kind: "mock".to_string(),
                ..crate::kernel::config::ProviderConfig::default()
            },
        );
        config
    }

    #[test]
    fn scans_local_agent_and_builds_effective_config() -> Result<()> {
        let tmp = tempdir()?;
        let root = tmp.path();
        fs::create_dir_all(root.join("default-harness"))?;
        fs::create_dir_all(root.join("agents/docs-reviewer/harness"))?;
        fs::write(
            root.join("agents/docs-reviewer/agent.toml"),
            r#"
model = "mock-model"
provider = "mock"
system_prompt = "Docs reviewer"
"#,
        )?;

        let bootstrap = bootstrap_config(root);
        let load = scan_registry(&bootstrap, root)?;
        assert_eq!(load.agents.len(), 1);
        assert_eq!(load.issues.len(), 0);
        assert_eq!(load.agents[0].harness_kind, HarnessKind::Local);
        assert_eq!(load.agents[0].harness_id, "agent::docs-reviewer");

        let effective = build_effective_config(&bootstrap, &load)?;
        assert!(effective.agents.contains_key("docs-reviewer"));
        assert!(effective.harnesses.contains_key("agent::docs-reviewer"));
        Ok(())
    }

    #[test]
    fn isolates_invalid_agent_toml() -> Result<()> {
        let tmp = tempdir()?;
        let root = tmp.path();
        fs::create_dir_all(root.join("default-harness"))?;
        fs::create_dir_all(root.join("agents/good/harness"))?;
        fs::create_dir_all(root.join("agents/bad/harness"))?;
        fs::write(
            root.join("agents/good/agent.toml"),
            r#"
model = "mock-model"
provider = "mock"
"#,
        )?;
        fs::write(root.join("agents/bad/agent.toml"), "not = [valid")?;

        let bootstrap = bootstrap_config(root);
        let load = scan_registry(&bootstrap, root)?;
        assert_eq!(load.agents.len(), 1);
        assert_eq!(load.agents[0].id, "good");
        assert_eq!(load.issues.len(), 1);
        Ok(())
    }

    #[test]
    fn supports_shared_harness_reference() -> Result<()> {
        let tmp = tempdir()?;
        let root = tmp.path();
        fs::create_dir_all(root.join("default-harness"))?;
        fs::create_dir_all(root.join("harnesses/reviewer"))?;
        fs::create_dir_all(root.join("agents/docs-reviewer"))?;
        fs::write(
            root.join("agents/docs-reviewer/agent.toml"),
            r#"
model = "mock-model"
provider = "mock"
harness = "reviewer"
"#,
        )?;

        let bootstrap = bootstrap_config(root);
        let load = scan_registry(&bootstrap, root)?;
        assert_eq!(load.shared_harnesses.len(), 1);
        assert_eq!(load.agents[0].harness_kind, HarnessKind::Shared);
        assert_eq!(load.agents[0].harness_id, "reviewer");
        Ok(())
    }
}
