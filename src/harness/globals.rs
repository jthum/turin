//! Turin-SL canonical globals injected into the Luau harness VM.

use mlua::{Lua, Result as LuaResult};
use std::future::Future;
use std::path::PathBuf;
use tokio::sync::Mutex;

use crate::harness::stdlib::{
    agent_bindings, memory_kv_bindings, runtime_bindings, session_user_aliases, system_globals,
};
use crate::inference::embeddings::EmbeddingProvider;
use crate::inference::provider::ProviderClient;
use crate::kernel::event::KernelEvent;
use crate::kernel::session::QueuedTask;
use crate::persistence::manager::StoreManager;

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;

const MAX_HARNESS_FILE_SIZE: usize = 10 * 1024 * 1024;

pub type SessionQueue = Arc<Mutex<VecDeque<QueuedTask>>>;
pub type ActiveSessionQueue = Arc<Mutex<Option<SessionQueue>>>;

#[derive(Clone)]
pub struct HarnessEventContext {
    pub json: bool,
    pub internal_id: Option<i64>,
    pub event_tx: tokio::sync::broadcast::Sender<(Option<i64>, KernelEvent)>,
    pub durability_tx: Option<tokio::sync::mpsc::UnboundedSender<(Option<i64>, KernelEvent)>>,
}

pub type ActiveHarnessEventContext = Arc<std::sync::Mutex<Option<HarnessEventContext>>>;

/// Shared state passed to async Lua callbacks via app data.
#[derive(Clone)]
pub struct HarnessAppData {
    pub fs_root: PathBuf,
    pub workspace_root: PathBuf,
    pub store_manager: Arc<StoreManager>,
    pub agent_manager: Arc<crate::kernel::agent_manager::AgentManager>,
    pub policy_manager: Arc<crate::kernel::policy::RuntimePolicyManager>,
    pub governance_manager: Arc<crate::kernel::governance::GovernanceManager>,
    pub active_session_id: Arc<std::sync::Mutex<Option<String>>>,
    pub active_session_mode: Arc<std::sync::Mutex<Option<crate::kernel::config::AgentMode>>>,
    pub active_harness_module: Arc<std::sync::Mutex<Option<String>>>,
    pub active_harness_root: Arc<std::sync::Mutex<Option<String>>>,
    pub active_import_capabilities: Arc<std::sync::Mutex<Option<BTreeMap<String, bool>>>>,
    pub active_governance_grant: Arc<std::sync::Mutex<Option<String>>>,
    pub active_event_context: ActiveHarnessEventContext,
    pub clients: HashMap<String, ProviderClient>,
    pub embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    pub queue: ActiveSessionQueue,
    pub config: Arc<crate::kernel::config::TurinConfig>,
    pub spawn_depth: u32,
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
    system_globals::register_import_global(lua)?;

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
