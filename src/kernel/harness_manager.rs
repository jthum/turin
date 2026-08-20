use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use tracing::debug;

use super::config::TurinConfig;
use super::harness_runtime::HarnessRuntime;
use super::native_harness::NativeHarnessFactory;

pub(crate) struct HarnessManager {
    agent_bindings: HashMap<String, String>,
    runtimes: HashMap<String, Arc<HarnessRuntime>>,
    default_runtime: Arc<HarnessRuntime>,
}

impl HarnessManager {
    #[cfg(test)]
    pub(crate) fn from_config(config: &TurinConfig) -> Result<Self> {
        Self::from_config_with_native(config, None)
    }

    pub(crate) fn from_config_with_native(
        config: &TurinConfig,
        native_factory: Option<Arc<dyn NativeHarnessFactory>>,
    ) -> Result<Self> {
        let default_harness_id = "default".to_string();
        let default_runtime = Arc::new(match native_factory {
            Some(factory) => {
                HarnessRuntime::from_config_with_native(default_harness_id.clone(), config, factory)
            }
            None => HarnessRuntime::from_config(default_harness_id.clone(), config),
        });

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

    pub(crate) fn runtime_by_id(&self, harness_id: &str) -> Option<&Arc<HarnessRuntime>> {
        self.runtimes.get(harness_id)
    }

    pub(crate) fn signal_subscriptions_for_harnesses(
        &self,
        harness_ids: &[String],
    ) -> Vec<(String, String)> {
        let mut out = Vec::new();
        for (agent_id, harness_id) in &self.agent_bindings {
            if !harness_ids.iter().any(|candidate| candidate == harness_id) {
                continue;
            }
            if let Some(runtime) = self.runtimes.get(harness_id) {
                for topic in runtime.runtime_signal_topics() {
                    out.push((agent_id.clone(), topic));
                }
            }
        }
        out.sort();
        out.dedup();
        out
    }

    pub(crate) fn agent_ids_for_harnesses(&self, harness_ids: &[String]) -> Vec<String> {
        let mut out = Vec::new();
        for (agent_id, harness_id) in &self.agent_bindings {
            if harness_ids.iter().any(|candidate| candidate == harness_id) {
                out.push(agent_id.clone());
            }
        }
        out.sort();
        out.dedup();
        out
    }
}

#[cfg(test)]
#[path = "tests/harness_manager.rs"]
mod tests;
