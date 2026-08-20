use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use tracing::{debug, info, warn};
use turin_daemon_protocol::UiIntentMessage;

use crate::harness::scheduler::HarnessSchedulerAccess;
use crate::harness::source::HarnessSourceOverlay;
use crate::harness::verdict::Verdict;
use crate::harness::virtual_tools::{
    DeclaredVirtualTool, VirtualToolFollowUp, VirtualToolResultResolution,
};
use crate::inference::embeddings::EmbeddingProvider;
use crate::inference::provider::ProviderClient;
use crate::kernel::agent_manager::AgentManager;
use crate::kernel::config::TurinConfig;
use crate::kernel::governance::GovernanceManager;
use crate::kernel::harness::HarnessFactory;
use crate::kernel::harness_contract::{
    HarnessActionRequest, HarnessExecutionBinding, HarnessHook, HarnessSignal, HarnessTurnRequest,
    HarnessTurnServices, SessionQueue,
};
use crate::kernel::policy::RuntimePolicyManager;
use crate::persistence::manager::StoreManager;

#[cfg(feature = "lua")]
mod lua_adapter;
mod resolver;
mod rust_adapter;

pub(crate) use resolver::HarnessAdapterResolver;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct HarnessWatchRoot {
    pub(crate) path: PathBuf,
    pub(crate) recursive: bool,
}

#[derive(Clone)]
#[cfg_attr(not(feature = "lua"), allow(dead_code))]
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

pub(crate) trait HarnessAdapterFactory: Send + Sync {
    fn name(&self) -> &'static str;

    fn watches_sources(&self) -> bool {
        false
    }

    fn create(
        &self,
        definition: &HarnessDefinition,
        ctx: HarnessRuntimeInitContext,
        source_overlay: Option<Arc<HarnessSourceOverlay>>,
    ) -> Result<Box<dyn HarnessInstance>>;
}

pub(crate) trait HarnessInstance: Send {
    fn loaded_scripts(&self) -> Vec<String> {
        Vec::new()
    }
    fn explicit_watch_roots(&self) -> Vec<PathBuf> {
        Vec::new()
    }
    fn runtime_signal_topics(&self) -> Vec<String> {
        Vec::new()
    }
    fn ui_intents(&self) -> Vec<UiIntentMessage> {
        Vec::new()
    }
    fn ui_intent_count(&self) -> Result<usize> {
        Ok(0)
    }
    fn ui_intents_from(&self, _start_index: usize) -> Result<Vec<UiIntentMessage>> {
        Ok(Vec::new())
    }
    fn load_script_str(&mut self, _script: &str) -> Result<()> {
        anyhow::bail!("this harness adapter does not support source loading")
    }
    fn evaluate_hook(&self, hook: HarnessHook<'_>) -> Result<Verdict>;
    fn has_hook(&self, hook_name: &str) -> bool;
    fn prepare_turn(
        &self,
        request: &mut HarnessTurnRequest,
        services: HarnessTurnServices<'_>,
    ) -> Result<Verdict>;
    fn bind_execution_context(&self, _binding: HarnessExecutionBinding) {}
    fn unbind_execution_context(&self) {}
    fn set_active_queue(&self, _queue: Option<SessionQueue>) {}
    fn set_active_capability_delegation(
        &self,
        _capabilities: Option<std::collections::BTreeMap<String, bool>>,
    ) {
    }
    fn take_pending_session_branch_checkout(&self) -> Option<String> {
        None
    }
    fn invoke_action(&self, request: HarnessActionRequest<'_>)
    -> Result<Option<serde_json::Value>>;
    fn declared_virtual_tools(&self) -> Result<Vec<DeclaredVirtualTool>> {
        Ok(Vec::new())
    }
    fn invoke_virtual_tool(
        &self,
        _name: &str,
        _args: serde_json::Value,
    ) -> Result<Option<VirtualToolResultResolution>> {
        Ok(None)
    }
    fn virtual_tool_follow_up(&self, _name: &str) -> Result<Option<VirtualToolFollowUp>> {
        Ok(None)
    }
    fn invoke_virtual_tool_result_handler(
        &self,
        key: &str,
        _payload: serde_json::Value,
        _default_is_error: bool,
    ) -> Result<VirtualToolResultResolution> {
        anyhow::bail!("harness has no virtual result handler '{key}'")
    }
    fn discard_virtual_tool_result_handler(&self, _key: &str) -> Result<()> {
        Ok(())
    }
    fn dispatch_runtime_signal(&self, signal: HarnessSignal<'_>) -> Result<usize>;
}

#[derive(Clone, Default)]
struct HarnessLoadedState {
    loaded_scripts: Vec<String>,
    explicit_watch_roots: Vec<PathBuf>,
    runtime_signal_topics: Vec<String>,
    ui_intents: Vec<UiIntentMessage>,
}

/// Shared harness configuration, adapter factory, and loaded metadata.
///
/// Live sessions use fresh `HarnessInstance` values created from this definition.
pub(crate) struct HarnessDefinition {
    harness_id: String,
    directory: PathBuf,
    #[cfg_attr(not(feature = "lua"), allow(dead_code))]
    fs_root: PathBuf,
    #[cfg_attr(not(feature = "lua"), allow(dead_code))]
    workspace_root: PathBuf,
    #[cfg_attr(not(feature = "lua"), allow(dead_code))]
    spawn_depth: u32,
    loaded_state: std::sync::Mutex<HarnessLoadedState>,
    generation: AtomicU64,
    adapter: Arc<dyn HarnessAdapterFactory>,
}

impl HarnessDefinition {
    pub(crate) fn new(
        harness_id: impl Into<String>,
        directory: impl Into<PathBuf>,
        fs_root: impl Into<PathBuf>,
        workspace_root: impl Into<PathBuf>,
        spawn_depth: u32,
        adapter: Arc<dyn HarnessAdapterFactory>,
    ) -> Self {
        Self {
            harness_id: harness_id.into(),
            directory: directory.into(),
            fs_root: fs_root.into(),
            workspace_root: workspace_root.into(),
            spawn_depth,
            loaded_state: std::sync::Mutex::new(HarnessLoadedState::default()),
            generation: AtomicU64::new(0),
            adapter,
        }
    }

    pub(crate) fn from_config(
        harness_id: impl Into<String>,
        config: &TurinConfig,
        adapter: Arc<dyn HarnessAdapterFactory>,
    ) -> Self {
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
            adapter,
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
        if !self.adapter.watches_sources() {
            return Vec::new();
        }
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

    pub(crate) fn runtime_signal_topics(&self) -> Vec<String> {
        self.loaded_state
            .lock()
            .expect("harness loaded-state mutex poisoned")
            .runtime_signal_topics
            .clone()
    }

    pub(crate) fn ui_intents(&self) -> Vec<UiIntentMessage> {
        self.loaded_state
            .lock()
            .expect("harness loaded-state mutex poisoned")
            .ui_intents
            .clone()
    }

    pub(crate) fn init(&self, ctx: HarnessRuntimeInitContext) -> Result<usize> {
        let instance = self.create_instance(ctx)?;
        let loaded_scripts = instance.loaded_scripts();
        let explicit_watch_roots = instance.explicit_watch_roots();
        let runtime_signal_topics = instance.runtime_signal_topics();
        let ui_intents = instance.ui_intents();
        let script_count = loaded_scripts.len();
        if script_count > 0 {
            info!(
                harness_id = %self.harness_id,
                adapter = self.adapter.name(),
                count = script_count,
                directory = %self.directory.display(),
                "Harness scripts loaded"
            );
            for name in &loaded_scripts {
                debug!(harness_id = %self.harness_id, script = %name, "Loaded harness script");
            }
        } else if self.adapter.watches_sources() {
            warn!(
                harness_id = %self.harness_id,
                adapter = self.adapter.name(),
                directory = %self.directory.display(),
                "No harness scripts found"
            );
        } else {
            info!(
                harness_id = %self.harness_id,
                adapter = self.adapter.name(),
                "Harness initialized"
            );
        }

        let mut state = self
            .loaded_state
            .lock()
            .expect("harness loaded-state mutex poisoned");
        state.loaded_scripts = loaded_scripts;
        state.explicit_watch_roots = explicit_watch_roots;
        state.runtime_signal_topics = runtime_signal_topics;
        state.ui_intents = ui_intents;
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

    pub(crate) fn validate_sources(
        &self,
        ctx: HarnessRuntimeInitContext,
        source_overlay: HarnessSourceOverlay,
    ) -> Result<usize> {
        let instance = self
            .adapter
            .create(self, ctx, Some(Arc::new(source_overlay)))?;
        Ok(instance.loaded_scripts().len())
    }

    pub(crate) fn create_instance(
        &self,
        ctx: HarnessRuntimeInitContext,
    ) -> Result<Box<dyn HarnessInstance>> {
        self.adapter.create(self, ctx, None)
    }
}

fn rust_adapter_factory(factory: Arc<dyn HarnessFactory>) -> Arc<dyn HarnessAdapterFactory> {
    rust_adapter::factory(factory)
}

#[cfg(feature = "lua")]
fn default_script_adapter_factory() -> Result<Arc<dyn HarnessAdapterFactory>> {
    Ok(lua_adapter::factory())
}

#[cfg(not(feature = "lua"))]
fn default_script_adapter_factory() -> Result<Arc<dyn HarnessAdapterFactory>> {
    anyhow::bail!("No script harness adapter is enabled in this Turin build")
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
