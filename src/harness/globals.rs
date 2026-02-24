//! Turin-SL canonical globals injected into the Luau harness VM.

use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};
use std::future::Future;
use std::path::PathBuf;
use tokio::sync::Mutex;

use crate::harness::stdlib::{
    agent_bindings, memory_kv_bindings, runtime_agent, runtime_context, runtime_data, runtime_db,
    runtime_policy, session_user_aliases, system_globals,
};
use crate::inference::embeddings::EmbeddingProvider;
use crate::inference::provider::ProviderClient;
use crate::kernel::identity::RuntimeIdentity;
use crate::kernel::session::QueuedTask;
use crate::persistence::manager::StoreManager;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

const MAX_HARNESS_FILE_SIZE: usize = 10 * 1024 * 1024;

pub type SessionQueue = Arc<Mutex<VecDeque<QueuedTask>>>;
pub type ActiveSessionQueue = Arc<Mutex<Option<SessionQueue>>>;

/// Shared state passed to async Lua callbacks via app data.
#[derive(Clone)]
pub struct HarnessAppData {
    pub fs_root: PathBuf,
    pub workspace_root: PathBuf,
    pub store_manager: Arc<StoreManager>,
    pub agent_manager: Arc<crate::kernel::agent_manager::AgentManager>,
    pub policy_manager: Arc<crate::kernel::policy::RuntimePolicyManager>,
    pub active_session_id: Arc<std::sync::Mutex<Option<String>>>,
    pub active_session_mode: Arc<std::sync::Mutex<Option<crate::kernel::config::AgentMode>>>,
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

    register_runtime_module(lua, &app_data)?;
    register_memory_module(lua, &app_data)?;
    register_kv_module(lua, &app_data)?;
    session_user_aliases::register_session_user_aliases(lua, &app_data)?;
    register_agent_module(lua, &app_data)?;
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

// -----------------------------------------------------------------------------
// IDENTITY & DELEGATION
// -----------------------------------------------------------------------------

pub(crate) fn get_active_identity(app_data: &HarnessAppData) -> anyhow::Result<RuntimeIdentity> {
    let session_id = app_data
        .active_session_id
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| anyhow::anyhow!("No active session context"))?;

    Ok(RuntimeIdentity::new(
        session_id,
        app_data.config.agent.id.clone(),
    ))
}

pub(crate) fn identity_to_lua_table(lua: &Lua, identity: &RuntimeIdentity) -> LuaResult<Table> {
    let tbl = lua.create_table()?;
    tbl.set("session_id", identity.session_id())?;
    tbl.set("agent_id", identity.agent_id())?;
    match identity.user_id() {
        Some(v) => tbl.set("user_id", v)?,
        None => tbl.set("user_id", Value::Nil)?,
    }
    match identity.channel_id() {
        Some(v) => tbl.set("channel_id", v)?,
        None => tbl.set("channel_id", Value::Nil)?,
    }
    match identity.tenant_id() {
        Some(v) => tbl.set("tenant_id", v)?,
        None => tbl.set("tenant_id", Value::Nil)?,
    }
    match identity.run_id() {
        Some(v) => tbl.set("run_id", v)?,
        None => tbl.set("run_id", Value::Nil)?,
    }
    let extra = lua.create_table()?;
    for (k, v) in identity.extra() {
        extra.set(k.as_str(), v.as_str())?;
    }
    tbl.set("extra", extra)?;
    Ok(tbl)
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{:02x}", b);
    }
    out
}

fn format_uuid_bytes_simple(bytes: &[u8]) -> Option<String> {
    if bytes.len() != 16 {
        return None;
    }
    let uuid = uuid::Uuid::from_slice(bytes).ok()?;
    Some(uuid.simple().to_string())
}

pub(crate) fn session_row_to_lua_table(
    lua: &Lua,
    row: &crate::persistence::schema::SessionRow,
) -> LuaResult<Table> {
    let t = lua.create_table()?;
    t.set("internal_id", row.id)?;
    t.set(
        "session_id",
        format_uuid_bytes_simple(&row.public_id).unwrap_or_else(|| bytes_to_hex(&row.public_id)),
    )?;
    t.set("agent_id", row.agent_id.clone())?;
    if let Some(m) = &row.metadata {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(m) {
            t.set("metadata", lua.to_value(&json)?)?;
        } else {
            t.set("metadata", m.clone())?;
        }
    } else {
        t.set("metadata", Value::Nil)?;
    }
    t.set("created_at", row.created_at.clone())?;
    Ok(t)
}

fn register_runtime_module(lua: &Lua, app_data: &HarnessAppData) -> LuaResult<()> {
    let runtime_table = lua.create_table()?;
    runtime_context::register_runtime_context_namespace(lua, &runtime_table, app_data)?;

    runtime_data::register_runtime_data_namespaces(lua, &runtime_table, app_data)?;

    runtime_db::register_runtime_db_namespace(lua, &runtime_table, app_data)?;

    runtime_agent::register_runtime_agent_namespace(lua, &runtime_table, app_data)?;
    runtime_policy::register_runtime_policy_namespace(lua, &runtime_table, app_data)?;

    lua.globals().set("runtime", runtime_table)?;
    Ok(())
}

// -----------------------------------------------------------------------------
// TIER 1: MEMORY.*
// -----------------------------------------------------------------------------

fn register_memory_module(lua: &Lua, app_data: &HarnessAppData) -> LuaResult<()> {
    memory_kv_bindings::register_memory_module(lua, app_data)
}

// -----------------------------------------------------------------------------
// TIER 1: KV.*
// -----------------------------------------------------------------------------

fn register_kv_module(lua: &Lua, app_data: &HarnessAppData) -> LuaResult<()> {
    memory_kv_bindings::register_kv_module(lua, app_data)
}

// -----------------------------------------------------------------------------
// SYSTEM: AGENT.*
// -----------------------------------------------------------------------------

fn register_agent_module(lua: &Lua, app_data: &HarnessAppData) -> LuaResult<()> {
    agent_bindings::register_agent_bindings(lua, app_data)
}
