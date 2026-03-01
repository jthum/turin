use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::harness::engine::HarnessEngine;
use anyhow::Result;

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

    pub(crate) fn lock_default_engine(&self) -> std::sync::MutexGuard<'_, Option<HarnessEngine>> {
        self.resolve_harness(None).lock_engine()
    }

    pub(crate) fn resolve_harness(&self, agent_id: Option<&str>) -> &Arc<HarnessRuntime> {
        let Some(agent_id) = agent_id else {
            return self.default_runtime();
        };

        let Some(runtime_id) = self.agent_bindings.get(agent_id) else {
            return self.default_runtime();
        };

        self.runtimes
            .get(runtime_id)
            .unwrap_or_else(|| self.default_runtime())
    }

    pub(crate) fn runtimes(&self) -> impl Iterator<Item = &Arc<HarnessRuntime>> {
        self.runtimes.values()
    }

    pub(crate) fn explicit_watch_roots(&self) -> Vec<PathBuf> {
        let mut roots = Vec::new();
        for runtime in self.runtimes.values() {
            roots.extend(runtime.explicit_watch_roots());
        }
        roots
    }
}
