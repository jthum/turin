use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use tracing::{debug, info, warn};

use crate::harness::engine::HarnessEngine;
use crate::harness::globals::{HarnessAppData, HarnessExecutionContext};
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
    pub(crate) embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
}

pub(crate) struct HarnessRuntime {
    harness_id: String,
    directory: PathBuf,
    fs_root: PathBuf,
    workspace_root: PathBuf,
    spawn_depth: u32,
    engine: std::sync::Mutex<Option<HarnessEngine>>,
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
            engine: std::sync::Mutex::new(None),
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

    pub(crate) fn lock_engine(&self) -> std::sync::MutexGuard<'_, Option<HarnessEngine>> {
        self.engine.lock().expect("harness mutex poisoned")
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
        let engine = self.lock_engine();
        engine
            .as_ref()
            .map(HarnessEngine::explicit_watch_roots)
            .unwrap_or_default()
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
        let engine = self.lock_engine();
        engine
            .as_ref()
            .map(HarnessEngine::loaded_scripts)
            .unwrap_or_default()
    }

    pub(crate) fn load_script_str(&self, script: &str) -> Result<()> {
        let mut engine = self.lock_engine();
        if let Some(ref mut harness) = *engine {
            harness.load_script_str(script)?;
            Ok(())
        } else {
            anyhow::bail!("Harness not initialized");
        }
    }

    pub(crate) fn init(&self, ctx: HarnessRuntimeInitContext) -> Result<usize> {
        let engine = self.build_engine(ctx)?;
        let script_count = engine.loaded_scripts().len();
        if script_count > 0 {
            info!(
                harness_id = %self.harness_id,
                count = script_count,
                directory = %self.directory.display(),
                "Harness scripts loaded"
            );
            for name in engine.loaded_scripts() {
                debug!(harness_id = %self.harness_id, script = %name, "Loaded harness script");
            }
        } else {
            warn!(
                harness_id = %self.harness_id,
                directory = %self.directory.display(),
                "No harness scripts found"
            );
        }

        let mut current = self.lock_engine();
        *current = Some(engine);
        self.generation.fetch_add(1, Ordering::Relaxed);
        Ok(script_count)
    }

    pub(crate) fn reload(&self, ctx: HarnessRuntimeInitContext) -> Result<usize> {
        self.init(ctx)
    }

    pub(crate) fn validate(&self, ctx: HarnessRuntimeInitContext) -> Result<usize> {
        let engine = self.build_engine(ctx)?;
        Ok(engine.loaded_scripts().len())
    }

    pub(crate) fn create_engine(&self, ctx: HarnessRuntimeInitContext) -> Result<HarnessEngine> {
        self.build_engine(ctx)
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

    fn build_engine(&self, ctx: HarnessRuntimeInitContext) -> Result<HarnessEngine> {
        let mut engine = HarnessEngine::new(self.build_app_data(ctx))
            .context("Failed to create harness engine")?;
        engine.load_dir(&self.directory).with_context(|| {
            format!(
                "Failed to load harness scripts from '{}'",
                self.directory.display()
            )
        })?;
        engine.set_loading_phase(false);
        Ok(engine)
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
