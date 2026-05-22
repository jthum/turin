use anyhow::Result;

use super::RegistryLoad;
use crate::kernel::config::{HarnessConfig, TurinConfig};

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
