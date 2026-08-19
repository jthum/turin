use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};

use super::{
    DiscoveredAgent, DiscoveredChannel, HarnessKind, RegistryIssue, RegistryLoad, SharedHarness,
    read_agent_file, read_channel_file,
};
use crate::kernel::agent_manager::AgentManager;
use crate::kernel::config::{AgentConfig, TurinConfig};
use crate::kernel::governance::GovernanceManager;
use crate::kernel::harness_runtime::{HarnessRuntime, HarnessRuntimeInitContext};
use crate::kernel::policy::RuntimePolicyManager;
use crate::persistence::manager::StoreManager;

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

    let mut agents = scan_agents(config, &agents_dir, &shared_harness_map, &mut issues)?;
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

fn scan_agents(
    bootstrap: &TurinConfig,
    agents_dir: &Path,
    shared_harnesses: &HashMap<String, PathBuf>,
    issues: &mut Vec<RegistryIssue>,
) -> Result<Vec<DiscoveredAgent>> {
    let mut agents = Vec::new();

    if !agents_dir.exists() {
        return Ok(agents);
    }

    for entry in fs::read_dir(agents_dir)
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

        match scan_agent_dir(bootstrap, &path, shared_harnesses) {
            Ok(Some(agent)) => agents.push(agent),
            Ok(None) => {}
            Err(err) => issues.push(RegistryIssue {
                path: path.display().to_string(),
                message: err.to_string(),
            }),
        }
    }

    Ok(agents)
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
    let agent_id = directory_name(agent_dir, "Agent")?;

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
    if let Some(lanes) = parsed.linked_runtime_lanes {
        anyhow::ensure!(
            lanes > 0,
            "agent linked_runtime_lanes must be greater than 0"
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
        linked_runtime_lanes: parsed.linked_runtime_lanes,
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
    let channel_id = directory_name(channel_dir, "Channel")?;

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

    validate_channel_agent(bootstrap, &parsed.agent_id, agents)?;
    parsed.inference.validate_shallow(
        &bootstrap.providers,
        &format!("channel '{}'.inference", channel_id),
    )?;
    validate_channel_persistence(bootstrap, &parsed.persistence)?;
    validate_channel_effective_inference(
        bootstrap,
        &channel_id,
        &parsed.agent_id,
        agents,
        &parsed.inference,
    )?;

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

fn validate_channel_agent(
    bootstrap: &TurinConfig,
    agent_id: &str,
    agents: &[DiscoveredAgent],
) -> Result<()> {
    let known_agent =
        agent_id == bootstrap.agent.id || agents.iter().any(|agent| agent.id == agent_id);
    anyhow::ensure!(
        known_agent,
        "channel references unknown agent '{}'",
        agent_id
    );
    Ok(())
}

fn validate_channel_persistence(
    bootstrap: &TurinConfig,
    persistence: &crate::kernel::config::ContextPersistenceConfig,
) -> Result<()> {
    if persistence.state.is_some() {
        bootstrap
            .persistence
            .resolve_context_state_selector(Some(persistence))
            .context("invalid channel persistence.state")?;
    }
    if persistence.store.is_some() {
        bootstrap
            .persistence
            .resolve_context_store_selector(Some(persistence))
            .context("invalid channel persistence.store")?;
    }
    Ok(())
}

fn validate_channel_effective_inference(
    bootstrap: &TurinConfig,
    channel_id: &str,
    agent_id: &str,
    agents: &[DiscoveredAgent],
    inference: &crate::kernel::config::InferenceOverrideConfig,
) -> Result<()> {
    if inference.is_empty() {
        return Ok(());
    }

    let agent_inference = if agent_id == bootstrap.agent.id {
        &bootstrap.agent.inference
    } else {
        &agents
            .iter()
            .find(|agent| agent.id == agent_id)
            .expect("known agent already checked")
            .agent_config
            .inference
    };
    let effective = bootstrap
        .inference
        .merged_with(agent_inference)
        .merged_with(inference);
    effective.validate_complete(
        &bootstrap.providers,
        &format!("channel '{}'.inference", channel_id),
    )
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

fn directory_name(dir: &Path, label: &str) -> Result<String> {
    dir.file_name()
        .and_then(|s| s.to_str())
        .map(str::to_string)
        .ok_or_else(|| anyhow::anyhow!("{label} directory name is not valid UTF-8"))
}
