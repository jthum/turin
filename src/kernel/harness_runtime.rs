use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use tracing::{debug, info, warn};

use crate::harness::engine::HarnessEngine;
use crate::harness::globals::{HarnessAppData, HarnessExecutionContext};
use crate::harness::scheduler::HarnessSchedulerAccess;
use crate::inference::embeddings::EmbeddingProvider;
use crate::inference::provider::ProviderClient;
use crate::kernel::agent_manager::AgentManager;
use crate::kernel::config::TurinConfig;
use crate::kernel::governance::GovernanceManager;
use crate::kernel::policy::RuntimePolicyManager;
use crate::persistence::manager::StoreManager;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct HarnessWatchRoot {
    pub(crate) path: PathBuf,
    pub(crate) recursive: bool,
}

#[derive(Clone)]
pub(crate) struct HarnessRuntimeInitContext {
    pub(crate) config: Arc<TurinConfig>,
    pub(crate) clients: HashMap<String, ProviderClient>,
    pub(crate) store_manager: Arc<StoreManager>,
    pub(crate) agent_manager: Arc<AgentManager>,
    pub(crate) policy_manager: Arc<RuntimePolicyManager>,
    pub(crate) governance_manager: Arc<GovernanceManager>,
    pub(crate) scheduler: Option<Arc<HarnessSchedulerAccess>>,
    pub(crate) embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
}

pub struct HarnessInstance {
    engine: HarnessEngine,
}

impl HarnessInstance {
    fn new(engine: HarnessEngine) -> Self {
        Self { engine }
    }

    pub(crate) fn loaded_scripts(&self) -> Vec<String> {
        self.engine.loaded_scripts()
    }

    pub(crate) fn explicit_watch_roots(&self) -> Vec<PathBuf> {
        self.engine.explicit_watch_roots()
    }

    pub(crate) fn load_script_str(&mut self, script: &str) -> Result<()> {
        self.engine.load_script_str(script)
    }
}

impl Deref for HarnessInstance {
    type Target = HarnessEngine;

    fn deref(&self) -> &Self::Target {
        &self.engine
    }
}

impl DerefMut for HarnessInstance {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.engine
    }
}

#[derive(Clone, Default)]
struct HarnessLoadedState {
    loaded_scripts: Vec<String>,
    explicit_watch_roots: Vec<PathBuf>,
}

// Despite the legacy name, this is the shared harness definition and metadata cache.
// Live executions use fresh `HarnessInstance` values built from this definition.
pub(crate) struct HarnessRuntime {
    harness_id: String,
    directory: PathBuf,
    fs_root: PathBuf,
    workspace_root: PathBuf,
    spawn_depth: u32,
    loaded_state: std::sync::Mutex<HarnessLoadedState>,
    generation: AtomicU64,
}

impl HarnessRuntime {
    pub(crate) fn new(
        harness_id: impl Into<String>,
        directory: impl Into<PathBuf>,
        fs_root: impl Into<PathBuf>,
        workspace_root: impl Into<PathBuf>,
        spawn_depth: u32,
    ) -> Self {
        Self {
            harness_id: harness_id.into(),
            directory: directory.into(),
            fs_root: fs_root.into(),
            workspace_root: workspace_root.into(),
            spawn_depth,
            loaded_state: std::sync::Mutex::new(HarnessLoadedState::default()),
            generation: AtomicU64::new(0),
        }
    }

    pub(crate) fn from_config(harness_id: impl Into<String>, config: &TurinConfig) -> Self {
        let fs_root = if config.harness.fs_root == "." {
            PathBuf::from(&config.kernel.workspace_root)
        } else {
            PathBuf::from(&config.harness.fs_root)
        };

        Self::new(
            harness_id,
            PathBuf::from(&config.harness.directory),
            fs_root,
            PathBuf::from(&config.kernel.workspace_root),
            config.kernel.initial_spawn_depth,
        )
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }

    pub(crate) fn directory(&self) -> &Path {
        &self.directory
    }

    pub(crate) fn harness_id(&self) -> &str {
        &self.harness_id
    }

    pub(crate) fn explicit_watch_roots(&self) -> Vec<PathBuf> {
        self.loaded_state
            .lock()
            .expect("harness loaded-state mutex poisoned")
            .explicit_watch_roots
            .clone()
    }

    pub(crate) fn watch_roots(&self) -> Vec<HarnessWatchRoot> {
        let mut roots = vec![HarnessWatchRoot {
            path: absolutize_path(&self.directory),
            recursive: false,
        }];

        for root in self.explicit_watch_roots() {
            let watch_root = HarnessWatchRoot {
                path: absolutize_path(&root),
                recursive: root.is_dir(),
            };
            if !roots.contains(&watch_root) {
                roots.push(watch_root);
            }
        }

        roots
    }

    pub(crate) fn owns_path(&self, path: &Path) -> bool {
        let path = absolutize_path(path);
        self.watch_roots().into_iter().any(|root| {
            if root.recursive {
                path == root.path || path.starts_with(&root.path)
            } else {
                path == root.path || path.parent() == Some(root.path.as_path())
            }
        })
    }

    pub(crate) fn loaded_scripts(&self) -> Vec<String> {
        self.loaded_state
            .lock()
            .expect("harness loaded-state mutex poisoned")
            .loaded_scripts
            .clone()
    }

    pub(crate) fn init(&self, ctx: HarnessRuntimeInitContext) -> Result<usize> {
        let instance = self.create_instance(ctx)?;
        let loaded_scripts = instance.loaded_scripts();
        let explicit_watch_roots = instance.explicit_watch_roots();
        let script_count = loaded_scripts.len();
        if script_count > 0 {
            info!(
                harness_id = %self.harness_id,
                count = script_count,
                directory = %self.directory.display(),
                "Harness scripts loaded"
            );
            for name in &loaded_scripts {
                debug!(harness_id = %self.harness_id, script = %name, "Loaded harness script");
            }
        } else {
            warn!(
                harness_id = %self.harness_id,
                directory = %self.directory.display(),
                "No harness scripts found"
            );
        }

        let mut state = self
            .loaded_state
            .lock()
            .expect("harness loaded-state mutex poisoned");
        state.loaded_scripts = loaded_scripts;
        state.explicit_watch_roots = explicit_watch_roots;
        self.generation.fetch_add(1, Ordering::Relaxed);
        Ok(script_count)
    }

    pub(crate) fn reload(&self, ctx: HarnessRuntimeInitContext) -> Result<usize> {
        self.init(ctx)
    }

    pub(crate) fn validate(&self, ctx: HarnessRuntimeInitContext) -> Result<usize> {
        let instance = self.create_instance(ctx)?;
        Ok(instance.loaded_scripts().len())
    }

    pub(crate) fn create_instance(
        &self,
        ctx: HarnessRuntimeInitContext,
    ) -> Result<HarnessInstance> {
        self.build_instance(ctx)
    }

    fn build_app_data(&self, ctx: HarnessRuntimeInitContext) -> HarnessAppData {
        HarnessAppData {
            fs_root: self.fs_root.clone(),
            workspace_root: self.workspace_root.clone(),
            harness_directory: self.directory.clone(),
            store_manager: ctx.store_manager,
            agent_manager: ctx.agent_manager,
            policy_manager: ctx.policy_manager,
            governance_manager: ctx.governance_manager,
            scheduler: ctx.scheduler,
            execution_ctx: Arc::new(std::sync::Mutex::new(HarnessExecutionContext::default())),
            clients: ctx.clients,
            embedding_provider: ctx.embedding_provider,
            config: ctx.config,
            spawn_depth: self.spawn_depth,
            active_modules: Arc::new(std::sync::Mutex::new(Vec::new())),
            watch_roots: Arc::new(std::sync::Mutex::new(Vec::new())),
            loading_phase: Arc::new(std::sync::Mutex::new(true)),
        }
    }

    fn build_instance(&self, ctx: HarnessRuntimeInitContext) -> Result<HarnessInstance> {
        let mut engine = HarnessEngine::new(self.build_app_data(ctx))
            .context("Failed to create harness engine")?;
        engine.load_dir(&self.directory).with_context(|| {
            format!(
                "Failed to load harness scripts from '{}'",
                self.directory.display()
            )
        })?;
        engine.set_loading_phase(false);
        Ok(HarnessInstance::new(engine))
    }
}

fn absolutize_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(path)
    }
}
