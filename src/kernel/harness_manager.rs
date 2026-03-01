use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use tracing::debug;

use super::config::TurinConfig;
use super::harness_runtime::HarnessRuntime;

pub(crate) struct HarnessManager {
    agent_bindings: HashMap<String, String>,
    runtimes: HashMap<String, Arc<HarnessRuntime>>,
    default_runtime: Arc<HarnessRuntime>,
}

impl HarnessManager {
    pub(crate) fn from_config(config: &TurinConfig) -> Result<Self> {
        let default_harness_id = "default".to_string();
        let default_runtime = Arc::new(HarnessRuntime::from_config(
            default_harness_id.clone(),
            config,
        ));

        let fs_root = if config.harness.fs_root == "." {
            PathBuf::from(&config.kernel.workspace_root)
        } else {
            PathBuf::from(&config.harness.fs_root)
        };
        let workspace_root = PathBuf::from(&config.kernel.workspace_root);
        let spawn_depth = config.kernel.initial_spawn_depth;

        let mut runtimes = HashMap::new();
        runtimes.insert(default_harness_id.clone(), Arc::clone(&default_runtime));

        for (harness_id, harness_cfg) in &config.harnesses {
            let runtime = Arc::new(HarnessRuntime::new(
                harness_id.clone(),
                PathBuf::from(&harness_cfg.directory),
                fs_root.clone(),
                workspace_root.clone(),
                spawn_depth,
            ));
            runtimes.insert(harness_id.clone(), runtime);
        }

        let mut bindings = HashMap::new();
        bindings.insert(
            config.agent.id.clone(),
            config.harness_id_for_agent(&config.agent).to_string(),
        );
        for (agent_id, agent_cfg) in &config.agents {
            bindings.insert(
                agent_id.clone(),
                config.harness_id_for_agent(agent_cfg).to_string(),
            );
        }

        Ok(Self {
            agent_bindings: bindings,
            runtimes,
            default_runtime,
        })
    }

    pub(crate) fn default_runtime(&self) -> &Arc<HarnessRuntime> {
        &self.default_runtime
    }

    pub(crate) fn runtime_id_for_agent(&self, agent_id: Option<&str>) -> &str {
        let Some(agent_id) = agent_id else {
            return "default";
        };
        if let Some(binding) = self.agent_bindings.get(agent_id) {
            binding.as_str()
        } else {
            debug!(
                agent_id = %agent_id,
                "Agent has no named harness binding; falling back to default harness"
            );
            "default"
        }
    }

    pub(crate) fn resolve_harness(&self, agent_id: Option<&str>) -> &Arc<HarnessRuntime> {
        let runtime_id = self.runtime_id_for_agent(agent_id);

        if let Some(runtime) = self.runtimes.get(runtime_id) {
            runtime
        } else {
            debug!(
                requested_harness_id = %runtime_id,
                agent_id = ?agent_id,
                "Named harness binding was missing from registry; falling back to default harness"
            );
            self.default_runtime()
        }
    }

    pub(crate) fn runtimes(&self) -> impl Iterator<Item = &Arc<HarnessRuntime>> {
        self.runtimes.values()
    }

    pub(crate) fn runtime_entries(&self) -> impl Iterator<Item = (&String, &Arc<HarnessRuntime>)> {
        self.runtimes.iter()
    }

    pub(crate) fn agent_bindings(&self) -> impl Iterator<Item = (&String, &String)> {
        self.agent_bindings.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::config::{
        AgentConfig, AgentMode, EmbeddingConfig, HarnessConfig, KernelConfig, PersistenceConfig,
        ProviderConfig, TurinConfig,
    };
    use std::collections::HashMap;
    use tempfile::tempdir;

    #[test]
    fn shared_harness_bindings_resolve_to_same_runtime() -> Result<()> {
        let tmp = tempdir()?;
        let default_harness = tmp.path().join("default-harness");
        let shared_harness = tmp.path().join("shared-harness");
        std::fs::create_dir_all(&default_harness)?;
        std::fs::create_dir_all(&shared_harness)?;

        let mut providers = HashMap::new();
        providers.insert(
            "mock".to_string(),
            ProviderConfig {
                kind: "mock".to_string(),
                ..ProviderConfig::default()
            },
        );

        let config = TurinConfig {
            agent: AgentConfig {
                id: "default".to_string(),
                system_prompt: "Default".to_string(),
                model: "mock-model".to_string(),
                provider: "mock".to_string(),
                thinking: None,
                mode: AgentMode::Auto,
                harness: None,
                idle_grace_secs: None,
            },
            agents: HashMap::from([
                (
                    "writer".to_string(),
                    AgentConfig {
                        id: "writer".to_string(),
                        system_prompt: "Writer".to_string(),
                        model: "mock-model".to_string(),
                        provider: "mock".to_string(),
                        thinking: None,
                        mode: AgentMode::Auto,
                        harness: Some("shared".to_string()),
                        idle_grace_secs: None,
                    },
                ),
                (
                    "reviewer".to_string(),
                    AgentConfig {
                        id: "reviewer".to_string(),
                        system_prompt: "Reviewer".to_string(),
                        model: "mock-model".to_string(),
                        provider: "mock".to_string(),
                        thinking: None,
                        mode: AgentMode::Auto,
                        harness: Some("shared".to_string()),
                        idle_grace_secs: None,
                    },
                ),
            ]),
            kernel: KernelConfig {
                workspace_root: tmp.path().to_string_lossy().to_string(),
                max_turns: 5,
                heartbeat_interval_secs: 30,
                initial_spawn_depth: 0,
            },
            persistence: PersistenceConfig {
                database_path: tmp.path().join("test.db").to_string_lossy().to_string(),
            },
            harness: HarnessConfig {
                directory: default_harness.to_string_lossy().to_string(),
                fs_root: ".".to_string(),
            },
            harnesses: HashMap::from([(
                "shared".to_string(),
                HarnessConfig {
                    directory: shared_harness.to_string_lossy().to_string(),
                    fs_root: ".".to_string(),
                },
            )]),
            providers,
            embeddings: Some(EmbeddingConfig::NoOp),
            governance: crate::kernel::config::GovernanceConfig::default(),
        daemon: Default::default(),
        };

        let manager = HarnessManager::from_config(&config)?;
        let writer = manager.resolve_harness(Some("writer"));
        let reviewer = manager.resolve_harness(Some("reviewer"));
        let default = manager.resolve_harness(Some("default"));

        assert!(Arc::ptr_eq(writer, reviewer));
        assert!(!Arc::ptr_eq(writer, default));
        assert_eq!(manager.runtime_id_for_agent(Some("writer")), "shared");
        assert_eq!(manager.runtime_id_for_agent(Some("reviewer")), "shared");
        assert_eq!(manager.runtime_id_for_agent(Some("default")), "default");

        Ok(())
    }
}
