use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use tracing::debug;

use super::config::TurinConfig;
use super::harness::RustHarnessFactories;
use super::harness_runtime::{HarnessAdapterResolver, HarnessDefinition};

pub(crate) struct HarnessManager {
    agent_bindings: HashMap<String, String>,
    definitions: HashMap<String, Arc<HarnessDefinition>>,
    default_definition: Arc<HarnessDefinition>,
}

impl HarnessManager {
    #[cfg(test)]
    pub(crate) fn from_config(config: &TurinConfig) -> Result<Self> {
        Self::from_config_with_harnesses(config, &RustHarnessFactories::new())
    }

    pub(crate) fn from_config_with_harnesses(
        config: &TurinConfig,
        rust_harness_factories: &RustHarnessFactories,
    ) -> Result<Self> {
        let adapters = HarnessAdapterResolver::new(config, rust_harness_factories)?;

        let default_harness_id = "default".to_string();
        let default_definition = Arc::new(HarnessDefinition::from_config(
            default_harness_id.clone(),
            config,
            adapters.resolve(&default_harness_id)?,
        ));

        let fs_root = if config.harness.fs_root == "." {
            PathBuf::from(&config.kernel.workspace_root)
        } else {
            PathBuf::from(&config.harness.fs_root)
        };
        let workspace_root = PathBuf::from(&config.kernel.workspace_root);
        let spawn_depth = config.kernel.initial_spawn_depth;

        let mut definitions = HashMap::new();
        definitions.insert(default_harness_id.clone(), Arc::clone(&default_definition));

        for (harness_id, harness_cfg) in &config.harnesses {
            let definition = Arc::new(HarnessDefinition::new(
                harness_id.clone(),
                PathBuf::from(&harness_cfg.directory),
                fs_root.clone(),
                workspace_root.clone(),
                spawn_depth,
                adapters.resolve(harness_id)?,
            ));
            definitions.insert(harness_id.clone(), definition);
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
            definitions,
            default_definition,
        })
    }

    pub(crate) fn default_definition(&self) -> &Arc<HarnessDefinition> {
        &self.default_definition
    }

    pub(crate) fn harness_id_for_agent(&self, agent_id: Option<&str>) -> &str {
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

    pub(crate) fn resolve_definition(&self, agent_id: Option<&str>) -> &Arc<HarnessDefinition> {
        let harness_id = self.harness_id_for_agent(agent_id);

        if let Some(definition) = self.definitions.get(harness_id) {
            definition
        } else {
            debug!(
                requested_harness_id = %harness_id,
                agent_id = ?agent_id,
                "Named harness binding was missing from registry; falling back to default harness"
            );
            self.default_definition()
        }
    }

    pub(crate) fn definitions(&self) -> impl Iterator<Item = &Arc<HarnessDefinition>> {
        self.definitions.values()
    }

    pub(crate) fn definition_entries(
        &self,
    ) -> impl Iterator<Item = (&String, &Arc<HarnessDefinition>)> {
        self.definitions.iter()
    }

    pub(crate) fn agent_bindings(&self) -> impl Iterator<Item = (&String, &String)> {
        self.agent_bindings.iter()
    }

    pub(crate) fn definition_by_id(&self, harness_id: &str) -> Option<&Arc<HarnessDefinition>> {
        self.definitions.get(harness_id)
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
            if let Some(definition) = self.definitions.get(harness_id) {
                for topic in definition.runtime_signal_topics() {
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
