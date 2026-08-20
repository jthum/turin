use super::{AgentSummary, HarnessKind, RegistryLoad, RegistrySnapshot, SharedHarnessSummary};

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
        issues: load.issues.clone(),
    }
}
