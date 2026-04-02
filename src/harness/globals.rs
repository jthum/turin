//! Turin-SL canonical globals injected into the Luau harness VM.

use mlua::{Lua, Result as LuaResult};
use std::future::Future;
use std::path::PathBuf;
use tokio::sync::Mutex;

use crate::harness::dx;
use crate::harness::stdlib::{
    agent_bindings, memory_kv_bindings, runtime_bindings, session_user_aliases, system_globals,
    tool_bindings,
};
use crate::inference::embeddings::EmbeddingProvider;
use crate::inference::provider::ProviderClient;
use crate::kernel::event::KernelEvent;
use crate::kernel::session::{PersistedKernelRecord, QueuedTask};
use crate::persistence::manager::StoreManager;
use crate::persistence::manager::StoreSelector;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;

const MAX_HARNESS_FILE_SIZE: usize = 10 * 1024 * 1024;

pub type SessionQueue = Arc<Mutex<VecDeque<QueuedTask>>>;
pub type ActiveHarnessModuleList = Arc<std::sync::Mutex<Vec<String>>>;
pub type ExplicitWatchRoots = Arc<std::sync::Mutex<Vec<PathBuf>>>;
pub type HarnessLoadPhase = Arc<std::sync::Mutex<bool>>;

#[derive(Clone)]
pub struct HarnessEventContext {
    pub json: bool,
    pub internal_id: Option<i64>,
    pub event_tx: tokio::sync::broadcast::Sender<(Option<i64>, KernelEvent)>,
    pub durability_tx: Option<tokio::sync::mpsc::UnboundedSender<PersistedKernelRecord>>,
}

#[derive(Clone, Default)]
pub struct HarnessExecutionContext {
    pub session_id: Option<String>,
    pub session_store_selector: Option<StoreSelector>,
    pub default_store_selector: Option<StoreSelector>,
    pub pending_branch_checkout: Option<String>,
    pub session_mode: Option<crate::kernel::config::AgentMode>,
    pub trace_id: Option<String>,
    pub queue: Option<SessionQueue>,
    pub harness_module: Option<String>,
    pub harness_root: Option<String>,
    pub import_capabilities: Option<BTreeMap<String, bool>>,
    pub governance_grant: Option<String>,
    pub event_context: Option<HarnessEventContext>,
}

pub type ActiveHarnessExecutionContext = Arc<std::sync::Mutex<HarnessExecutionContext>>;

/// Shared state passed to async Lua callbacks via app data.
#[derive(Clone)]
pub struct HarnessAppData {
    pub fs_root: PathBuf,
    pub workspace_root: PathBuf,
    pub harness_directory: PathBuf,
    pub store_manager: Arc<StoreManager>,
    pub agent_manager: Arc<crate::kernel::agent_manager::AgentManager>,
    pub policy_manager: Arc<crate::kernel::policy::RuntimePolicyManager>,
    pub governance_manager: Arc<crate::kernel::governance::GovernanceManager>,
    pub execution_ctx: ActiveHarnessExecutionContext,
    pub clients: HashMap<String, ProviderClient>,
    pub embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    pub config: Arc<crate::kernel::config::TurinConfig>,
    pub spawn_depth: u32,
    pub active_modules: ActiveHarnessModuleList,
    pub watch_roots: ExplicitWatchRoots,
    pub loading_phase: HarnessLoadPhase,
}

pub(crate) fn block_on_current<F>(fut: F) -> F::Output
where
    F: Future,
{
    tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(fut))
}

// -----------------------------------------------------------------------------
// CORE ENTRY
// -----------------------------------------------------------------------------

pub fn register_globals(lua: &Lua, app_data: HarnessAppData) -> LuaResult<()> {
    register_verdict_constants(lua)?;

    system_globals::register_system_globals(lua, &app_data.fs_root, MAX_HARNESS_FILE_SIZE)?;

    runtime_bindings::register_runtime_namespace(lua, &app_data)?;
    memory_kv_bindings::register_memory_module(lua, &app_data)?;
    memory_kv_bindings::register_kv_module(lua, &app_data)?;
    session_user_aliases::register_session_user_aliases(lua, &app_data)?;
    agent_bindings::register_agent_bindings(lua, &app_data)?;
    tool_bindings::register_tool_globals(lua)?;
    system_globals::register_import_global(lua)?;
    dx::register_dx_globals(lua, &app_data)?;

    lua.set_app_data(app_data);
    Ok(())
}

fn register_verdict_constants(lua: &Lua) -> LuaResult<()> {
    let globals = lua.globals();
    globals.set("ALLOW", 1)?;
    globals.set("REJECT", 2)?;
    globals.set("ESCALATE", 3)?;
    globals.set("MODIFY", 4)?;
    Ok(())
}
