//! Turin-SL canonical globals injected into the Luau harness VM.

use mlua::{Function, Lua, MultiValue, Result as LuaResult, Table, Value};
use std::future::Future;
use std::path::PathBuf;
use tokio_util::sync::CancellationToken;

use crate::harness::dx;
use crate::harness::scheduler::HarnessSchedulerAccess;
use crate::harness::stdlib::{
    action_bindings, agent_bindings, event_bindings, memory_kv_bindings, runtime_bindings,
    session_user_aliases, system_globals, tool_bindings, ui_bindings,
};
use crate::inference::embeddings::EmbeddingProvider;
pub(crate) use crate::kernel::harness_contract::{
    HarnessEventContext, HarnessExecutionBinding, SessionQueue,
};
use crate::kernel::session::{
    CompletedLocalTaskResultsHandle, ExecutionConflictPolicy, ExecutionContextTarget,
    ExecutionDurability, ExecutionVisibility, ExecutionWritePolicy,
};
use crate::persistence::manager::StoreManager;
use crate::persistence::manager::StoreSelector;

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::harness::source::HarnessSourceOverlay;

const MAX_HARNESS_FILE_SIZE: usize = 10 * 1024 * 1024;

pub type ActiveHarnessModuleList = Arc<std::sync::Mutex<Vec<String>>>;
pub type ExplicitWatchRoots = Arc<std::sync::Mutex<Vec<PathBuf>>>;
pub type HarnessLoadPhase = Arc<std::sync::Mutex<bool>>;

#[derive(Clone, Default)]
pub struct HarnessExecutionContext {
    pub agent_id: Option<String>,
    pub execution_id: Option<String>,
    pub execution_context_target: Option<ExecutionContextTarget>,
    pub execution_visibility: Option<ExecutionVisibility>,
    pub execution_durability: Option<ExecutionDurability>,
    pub execution_write_policy: Option<ExecutionWritePolicy>,
    pub execution_conflict_policy: Option<ExecutionConflictPolicy>,
    pub session_id: Option<String>,
    pub runtime_slot_id: Option<String>,
    pub session_store_selector: Option<StoreSelector>,
    pub default_store_selector: Option<StoreSelector>,
    pub pending_branch_checkout: Option<String>,
    pub trace_id: Option<String>,
    pub queue: Option<SessionQueue>,
    pub harness_module: Option<String>,
    pub harness_root: Option<String>,
    pub import_capabilities: Option<BTreeMap<String, bool>>,
    pub governance_grant: Option<String>,
    pub event_context: Option<HarnessEventContext>,
    pub completed_task_results: Option<CompletedLocalTaskResultsHandle>,
    pub cancel_token: Option<CancellationToken>,
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
    pub scheduler: Option<Arc<HarnessSchedulerAccess>>,
    pub execution_ctx: ActiveHarnessExecutionContext,
    pub embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    pub config: Arc<crate::kernel::config::TurinConfig>,
    pub spawn_depth: u32,
    pub active_modules: ActiveHarnessModuleList,
    pub watch_roots: ExplicitWatchRoots,
    pub loading_phase: HarnessLoadPhase,
    pub source_overlay: Option<Arc<HarnessSourceOverlay>>,
}

pub fn block_on_current<F>(fut: F) -> F::Output
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
    action_bindings::register_action_globals(lua)?;
    crate::harness::stdlib::object_refs::register_ref_and_target_globals(lua)?;
    event_bindings::register_event_globals(lua)?;
    ui_bindings::register_ui_globals(lua)?;
    tool_bindings::register_tool_globals(lua)?;
    system_globals::register_import_global(lua)?;
    dx::register_dx_globals(lua, &app_data)?;
    install_public_error_contract(lua)?;

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

fn install_public_error_contract(lua: &Lua) -> LuaResult<()> {
    let globals = lua.globals();
    for root in [
        "runtime", "memory", "kv", "agent", "action", "session", "user", "fs", "json", "time", "ui",
    ] {
        if let Ok(table) = globals.get::<Table>(root) {
            wrap_public_table(lua, &table, root)?;
        }
    }
    Ok(())
}

fn wrap_public_table(lua: &Lua, table: &Table, path: &str) -> LuaResult<()> {
    let mut keys = Vec::<(String, Value)>::new();
    for pair in table.pairs::<Value, Value>() {
        let (key, value) = pair?;
        if let Value::String(key_str) = key {
            keys.push((key_str.to_str()?.to_string(), value));
        }
    }

    for (key, value) in keys {
        let entry_path = format!("{path}.{key}");
        match value {
            Value::Function(func) => {
                table.set(key, wrap_public_function(lua, func, entry_path)?)?;
            }
            Value::Table(child) => {
                wrap_public_table(lua, &child, &entry_path)?;
            }
            _ => {}
        }
    }
    Ok(())
}

fn wrap_public_function(lua: &Lua, func: Function, path: String) -> LuaResult<Function> {
    lua.create_function(move |lua, args: MultiValue| {
        let values = func.call::<MultiValue>(args)?;
        coerce_public_result(lua, &path, values)
    })
}

fn coerce_public_result(_lua: &Lua, path: &str, values: MultiValue) -> LuaResult<MultiValue> {
    let items: Vec<Value> = values.into_iter().collect();
    if items.is_empty() {
        return Ok(MultiValue::new());
    }

    if items.len() >= 2 {
        match &items[1] {
            Value::Nil => {
                let mut out = MultiValue::new();
                out.push_back(items[0].clone());
                for value in items.into_iter().skip(2) {
                    out.push_back(value);
                }
                return Ok(out);
            }
            Value::String(s) => {
                let err = match s.to_str() {
                    Ok(value) => value.to_string(),
                    Err(_) => "<invalid utf-8 error string>".to_string(),
                };
                return Err(mlua::Error::runtime(format!("[{}] {}", path, err)));
            }
            Value::Boolean(false) => {
                return Err(mlua::Error::runtime(format!(
                    "[{}] operation returned false",
                    path
                )));
            }
            other => {
                return Err(mlua::Error::runtime(format!("[{}] {:?}", path, other)));
            }
        }
    }

    let mut out = MultiValue::new();
    out.push_back(items[0].clone());
    Ok(out)
}
