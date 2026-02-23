//! Turin-SL canonical globals injected into the Luau harness VM.

use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};
use std::path::{Path, PathBuf};
use tokio::sync::Mutex;

use crate::inference::embeddings::EmbeddingProvider;
use crate::inference::provider::ProviderClient;
use crate::kernel::session::QueuedTask;
use crate::persistence::manager::StoreManager;
use crate::kernel::identity::{ContextSelector, RuntimeIdentity};

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

const MAX_SPAWN_DEPTH: u32 = 3;
const MAX_HARNESS_FILE_SIZE: usize = 10 * 1024 * 1024;

pub type SessionQueue = Arc<Mutex<VecDeque<QueuedTask>>>;
pub type ActiveSessionQueue = Arc<Mutex<Option<SessionQueue>>>;

/// Shared state passed to async Lua callbacks via app data.
pub struct HarnessAppData {
    pub fs_root: PathBuf,
    pub workspace_root: PathBuf,
    pub store_manager: Arc<StoreManager>,
    pub agent_manager: Arc<crate::kernel::agent_manager::AgentManager>,
    pub active_session_id: Arc<std::sync::Mutex<Option<String>>>,
    pub active_session_mode: Arc<std::sync::Mutex<Option<crate::kernel::config::AgentMode>>>,
    pub clients: HashMap<String, ProviderClient>,
    pub embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
    pub queue: ActiveSessionQueue,
    pub config: Arc<crate::kernel::config::TurinConfig>,
    pub spawn_depth: u32,
}

// -----------------------------------------------------------------------------
// CORE ENTRY
// -----------------------------------------------------------------------------

pub fn register_globals(lua: &Lua, app_data: HarnessAppData) -> LuaResult<()> {
    register_verdict_constants(lua)?;
    
    register_fs_module(lua, &app_data)?;
    register_json_module(lua)?;
    register_time_module(lua)?;
    register_log_function(lua)?;
    
    register_runtime_module(lua, &app_data)?;
    register_memory_module(lua, &app_data)?;
    register_kv_module(lua, &app_data)?;
    register_tier2_aliases(lua, &app_data)?;
    register_agent_module(lua, &app_data)?;

    let globals = lua.globals();
    globals.set(
        "import",
        lua.create_function(|lua, name: String| {
            let globals = lua.globals();
            let modules: Table = globals.get("__harness_modules")?;
            modules.get::<Value>(name)
        })?,
    )?;

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

#[allow(dead_code)]
fn get_active_identity(app_data: &HarnessAppData) -> anyhow::Result<RuntimeIdentity> {
     let session_id_str = app_data.active_session_id.lock().unwrap()
         .clone()
         .ok_or_else(|| anyhow::anyhow!("No active session context"))?;
         
     let agent_id = app_data.config.agent.id.clone();
     let mut identity = RuntimeIdentity::new(session_id_str, agent_id);
     identity.user_id = Some("default_auth_user".to_string()); // mocked for generic demo scenarios
     
     Ok(identity)
}

fn resolve_safe_path(root: &Path, path_str: &str) -> Option<PathBuf> {
    crate::tools::is_safe_path(root, Path::new(path_str)).ok()
}

fn table_to_selector(lua: &Lua, ctx_tbl: Table) -> LuaResult<ContextSelector> {
    let mut tags = Vec::new();
    if let Ok(Value::Table(tags_tbl)) = ctx_tbl.get("tags") {
        for pair in tags_tbl.pairs::<i64, String>() {
            let (_, val) = pair?;
            tags.push(val);
        }
    }
    let namespace = ctx_tbl.get::<String>("namespace").unwrap_or_else(|_| "default".to_string());
    let visibility = ctx_tbl.get::<String>("visibility").unwrap_or_else(|_| "private".to_string());
    
    Ok(ContextSelector { tags, namespace, visibility })
}

// -----------------------------------------------------------------------------
// RUNTIME MODULE
// -----------------------------------------------------------------------------

fn register_runtime_module(lua: &Lua, _app_data: &HarnessAppData) -> LuaResult<()> {
    let runtime_table = lua.create_table()?;
    
    runtime_table.set(
        "context",
        lua.create_function(|lua, (arg1, arg2, _opts): (Value, Option<String>, Option<Table>)| {
            match arg1 {
                Value::Table(tbl) => Ok(Value::Table(tbl)),
                Value::String(scope) => {
                    let id = arg2.unwrap_or_else(|| "default".to_string());
                    let tag = format!("{}:{}", scope.to_str()?, id);
                    
                    let ctx = lua.create_table()?;
                    let tags = lua.create_sequence_from(vec![tag])?;
                    ctx.set("tags", tags)?;
                    ctx.set("namespace", "default")?;
                    ctx.set("visibility", "private")?;
                    Ok(Value::Table(ctx))
                }
                _ => Err(mlua::Error::runtime("runtime.context missing valid signature")),
            }
        })?,
    )?;

    // Stub canonical delegates for the STDLIB surface validation
    let r_memory = lua.create_table()?;
    runtime_table.set("memory", r_memory)?;
    let r_kv = lua.create_table()?;
    runtime_table.set("kv", r_kv)?;

    lua.globals().set("runtime", runtime_table)?;
    Ok(())
}

// -----------------------------------------------------------------------------
// TIER 1: MEMORY.*
// -----------------------------------------------------------------------------

fn register_memory_module(lua: &Lua, app_data: &HarnessAppData) -> LuaResult<()> {
    let memory_table = lua.create_table()?;
    
    let manager = app_data.store_manager.clone();
    let embedding_provider = app_data.embedding_provider.clone();
    let session_state = app_data.active_session_id.clone();
    
    // As context clones for the .as factory
    let m_as = manager.clone();
    let ep_as = embedding_provider.clone();

    // Setup implicit tier-1 (Session bounded `memory.search` / `memory.store`)
    {
        let m = manager.clone();
        let e = embedding_provider.clone();
        let s = session_state.clone();
        
        memory_table.set("search", lua.create_function(move |lua, (query, limit_opt): (String, Option<usize>)| {
            let limit = limit_opt.unwrap_or(5);
            let s_id = s.lock().unwrap().clone().unwrap_or_default();
            
            let result = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    let mut vector = None;
                    if let Some(p) = &e { if let Ok(emb) = p.embed(&query).await { vector = Some(emb.vector); } }
                    
                    if let Ok(store) = m.get_default().await {
                        if let Ok(pub_id) = uuid::Uuid::parse_str(&s_id) {
                            if let Ok(Some(int_id)) = store.get_session_by_public_id(pub_id).await {
                                return store.search_memories(int_id, vector.as_deref(), Some(&query), limit).await
                                    .map_err(|err| err.to_string());
                            }
                        }
                    }
                    Err("No active session or store".to_string())
                })
            });
            match result {
                Ok(rows) => {
                    let tbl = lua.create_table()?;
                    for (i, row) in rows.into_iter().enumerate() {
                        let rt = lua.create_table()?;
                        rt.set("content", row.content)?;
                        rt.set("score", row.score)?;
                        tbl.set(i + 1, rt)?;
                    }
                    Ok((Value::Table(tbl), Value::Nil))
                },
                Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
            }
        })?)?;
    }
    
    {
        let m = manager.clone();
        let e = embedding_provider.clone();
        let s = session_state.clone();
        
        memory_table.set("store", lua.create_function(move |lua, (content, _metadata): (String, Option<Table>)| {
            let s_id = s.lock().unwrap().clone().unwrap_or_default();
            let result = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    if let Some(p) = &e {
                        if let Ok(emb) = p.embed(&content).await {
                            if let Ok(store) = m.get_default().await {
                                if let Ok(pub_id) = uuid::Uuid::parse_str(&s_id) {
                                    if let Ok(Some(int_id)) = store.get_session_by_public_id(pub_id).await {
                                        return store.insert_memory(int_id, &content, &emb.vector, &serde_json::json!({})).await
                                            .map_err(|err| err.to_string());
                                    }
                                }
                            }
                        }
                    }
                    Err("No active session or store".to_string())
                })
            });
            match result {
                Ok(_) => Ok((Value::Boolean(true), Value::Nil)),
                Err(err) => Ok((Value::Boolean(false), Value::String(lua.create_string(&err)?))),
            }
        })?)?;
    }

    // memory.as(ctx) Proxy pattern
    let as_func = lua.create_function(move |lua, ctx: Table| {
        let proxy = lua.create_table()?;
        let selector = table_to_selector(lua, ctx)?;
        
        // memory.as(ctx).search
        let _m_sc = m_as.clone();
        let _e_sc = ep_as.clone();
        let _sel_sc = selector.clone();
        proxy.set("search", lua.create_function(move |lua, (_query, limit_opt): (String, Option<usize>)| {
            let limit = limit_opt.unwrap_or(5);
            let result: Result<Vec<crate::persistence::schema::MemoryRow>, String> = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    // MOCK DELEGATE: In production, route to manager.get(&sel_sc.to_alias()), wait search_memories_global
                    // Since search_memories_global doesn't exist yet, we just mock success.
                    Ok(vec![])
                })
            });
            match result {
                Ok(rows) => {
                    let tbl = lua.create_table()?;
                    for (i, row) in rows.into_iter().enumerate() {
                        let rt = lua.create_table()?;
                        rt.set("content", row.content)?;
                        tbl.set(i + 1, rt)?;
                    }
                    Ok((Value::Table(tbl), Value::Nil))
                },
                Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
            }
        })?)?;
        
        // memory.as(ctx).store
        let _m_st = m_as.clone();
        let _e_st = ep_as.clone();
        let _sel_st = selector.clone();
        proxy.set("store", lua.create_function(move |lua, (_content, _meta): (String, Option<Table>)| {
            // MOCK DELEGATE 
            Ok((Value::Boolean(true), Value::Nil))
        })?)?;
        
        Ok(proxy)
    })?;
    
    memory_table.set("as", as_func)?;
    lua.globals().set("memory", memory_table)?;
    Ok(())
}

// -----------------------------------------------------------------------------
// TIER 1: KV.*
// -----------------------------------------------------------------------------

fn register_kv_module(lua: &Lua, app_data: &HarnessAppData) -> LuaResult<()> {
    let kv_table = lua.create_table()?;
    let manager = app_data.store_manager.clone();
    let m_as = manager.clone();

    // kv.get
    {
        let m = manager.clone();
        kv_table.set("get", lua.create_function(move |lua, key: String| {
            let result = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    if let Ok(store) = m.get_default().await {
                        store.kv_get(&key).await.ok().flatten()
                    } else { None }
                })
            });
            match result {
                Some(val) => Ok((Value::String(lua.create_string(&val)?), Value::Nil)),
                None => Ok((Value::Nil, Value::Nil)),
            }
        })?)?;
    }

    // kv.set
    {
        let m = manager.clone();
        kv_table.set("set", lua.create_function(move |lua, (key, value): (String, String)| {
            let result: anyhow::Result<()> = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    let store = m.get_default().await?;
                    store.kv_set(&key, &value).await?; Ok(())
                })
            });
            match result {
                Ok(_) => Ok((Value::Boolean(true), Value::Nil)),
                Err(e) => Ok((Value::Boolean(false), Value::String(lua.create_string(&e.to_string())?))),
            }
        })?)?;
    }

    // kv.delete
    {
        let m = manager.clone();
        kv_table.set("delete", lua.create_function(move |lua, key: String| {
            let result: anyhow::Result<()> = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    let store = m.get_default().await?;
                    store.kv_delete(&key).await?; Ok(())
                })
            });
            match result {
                Ok(_) => Ok((Value::Boolean(true), Value::Nil)),
                Err(e) => Ok((Value::Boolean(false), Value::String(lua.create_string(&e.to_string())?))),
            }
        })?)?;
    }

    // kv.as(ctx) Proxy
    let as_func = lua.create_function(move |lua, ctx: Table| {
        let proxy = lua.create_table()?;
        let selector = table_to_selector(lua, ctx)?;
        
        let m_get = m_as.clone();
        let sel_get = selector.clone();
        proxy.set("get", lua.create_function(move |lua, key: String| {
            let result = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    if let Ok(store) = m_get.open(&crate::persistence::manager::StoreSelector::Alias(sel_get.to_alias())).await {
                        store.kv_get(&key).await.ok().flatten()
                    } else { None }
                })
            });
            match result {
                Some(val) => Ok((Value::String(lua.create_string(&val)?), Value::Nil)),
                None => Ok((Value::Nil, Value::Nil)),
            }
        })?)?;
        
        let m_set = m_as.clone();
        let sel_set = selector.clone();
        proxy.set("set", lua.create_function(move |lua, (key, value): (String, String)| {
            let result: anyhow::Result<()> = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    let store = m_set.open(&crate::persistence::manager::StoreSelector::Alias(sel_set.to_alias())).await?;
                    store.kv_set(&key, &value).await?; Ok(())
                })
            });
            match result {
                Ok(_) => Ok((Value::Boolean(true), Value::Nil)),
                Err(e) => Ok((Value::Boolean(false), Value::String(lua.create_string(&e.to_string())?))),
            }
        })?)?;
        
        let m_del = m_as.clone();
        let sel_del = selector.clone();
        proxy.set("delete", lua.create_function(move |lua, key: String| {
            let result: anyhow::Result<()> = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async {
                    let store = m_del.open(&crate::persistence::manager::StoreSelector::Alias(sel_del.to_alias())).await?;
                    store.kv_delete(&key).await?; Ok(())
                })
            });
            match result {
                Ok(_) => Ok((Value::Boolean(true), Value::Nil)),
                Err(e) => Ok((Value::Boolean(false), Value::String(lua.create_string(&e.to_string())?))),
            }
        })?)?;
        
        Ok(proxy)
    })?;
    
    kv_table.set("as", as_func)?;
    lua.globals().set("kv", kv_table)?;
    Ok(())
}

// -----------------------------------------------------------------------------
// TIER 2: SESSION.* and USER.* STUBS
// -----------------------------------------------------------------------------

fn register_tier2_aliases(lua: &Lua, _app_data: &HarnessAppData) -> LuaResult<()> {
    let session_table = lua.create_table()?;
    let user_table = lua.create_table()?;
    
    let set_proxy = |t: &Table| -> LuaResult<()> {
        let mem = lua.create_table()?;
        mem.set("search", lua.create_function(|lua, _args: Value| Ok((Value::Table(lua.create_table()?), Value::Nil)))?)?;
        mem.set("store", lua.create_function(|_lua, _args: Value| Ok((Value::Boolean(true), Value::Nil)))?)?;
        t.set("memory", mem)?;

        let kv = lua.create_table()?;
        kv.set("get", lua.create_function(|_lua, _k: String| Ok((Value::Nil, Value::Nil)))?)?;
        kv.set("set", lua.create_function(|_lua, _args: Value| Ok((Value::Boolean(true), Value::Nil)))?)?;
        t.set("kv", kv)?;
        Ok(())
    };
    
    set_proxy(&session_table)?;
    set_proxy(&user_table)?;

    lua.globals().set("session", session_table)?;
    lua.globals().set("user", user_table)?;
    Ok(())
}

// -----------------------------------------------------------------------------
// SYSTEM: AGENT.*
// -----------------------------------------------------------------------------

fn register_agent_module(lua: &Lua, app_data: &HarnessAppData) -> LuaResult<()> {
    let agent_table = lua.create_table()?;
    let session_ns = lua.create_table()?;
    let mode_ns = lua.create_table()?;
    
    let agent_manager = app_data.agent_manager.clone();
    
    // agent.spawn
    agent_table.set("spawn", lua.create_function(|lua, _args: Value| {
        Ok((Value::String(lua.create_string("dummy_spawn_resp")?), Value::Nil))
    })?)?;

    // agent.complete
    agent_table.set("complete", lua.create_function(|lua, _args: Value| {
        Ok((Value::String(lua.create_string("dummy_complete_resp")?), Value::Nil))
    })?)?;

    // agent.session.identity()
    session_ns.set("identity", lua.create_function(|lua, ()| {
        let tbl = lua.create_table()?;
        tbl.set("agent_id", "default")?;
        tbl.set("session_id", "local_123")?;
        Ok(tbl)
    })?)?;
    
    // agent.session.queue
    let aq = app_data.queue.clone();
    session_ns.set("queue", lua.create_function(move |_lua, cmd: String| {
        let aq = aq.clone();
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                if let Some(q) = &*aq.lock().await {
                    q.lock().await.push_back(QueuedTask::ad_hoc(cmd));
                }
            })
        });
        Ok((Value::Boolean(true), Value::Nil))
    })?)?;
    
    // agent.session.queue_next
    let aq2 = app_data.queue.clone();
    session_ns.set("queue_next", lua.create_function(move |_lua, cmd: String| {
        let aq = aq2.clone();
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                if let Some(q) = &*aq.lock().await {
                    q.lock().await.push_front(QueuedTask::ad_hoc(cmd));
                }
            })
        });
        Ok((Value::Boolean(true), Value::Nil))
    })?)?;

    let sm1 = app_data.active_session_mode.clone();
    mode_ns.set("get", lua.create_function(move |lua, ()| {
        let mode = sm1.lock().unwrap().clone().unwrap_or(crate::kernel::config::AgentMode::Auto);
        let mode_str = match mode {
            crate::kernel::config::AgentMode::Auto => "auto",
            crate::kernel::config::AgentMode::Stateful => "stateful",
            crate::kernel::config::AgentMode::Stateless => "stateless",
        };
        Ok(Value::String(lua.create_string(mode_str)?))
    })?)?;

    let sm2 = app_data.active_session_mode.clone();
    mode_ns.set("set", lua.create_function(move |_lua, m: String| {
        let mode = match m.as_str() {
            "stateful" => crate::kernel::config::AgentMode::Stateful,
            "stateless" => crate::kernel::config::AgentMode::Stateless,
            _ => crate::kernel::config::AgentMode::Auto,
        };
        if let Ok(mut lock) = sm2.lock() {
            *lock = Some(mode);
        }
        Ok((Value::Boolean(true), Value::Nil))
    })?)?;

    agent_table.set("session", session_ns)?;
    agent_table.set("mode", mode_ns)?;
    
    // Deprecated send
    agent_table.set("send", lua.create_function(move |_lua, (id, prompt): (String, String)| {
        let m = agent_manager.clone();
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                let _ = m.send(&id, QueuedTask::ad_hoc(prompt)).await;
            })
        });
        Ok((Value::Boolean(true), Value::Nil))
    })?)?;

    lua.globals().set("agent", agent_table)?;
    Ok(())
}

// -----------------------------------------------------------------------------
// GENERIC MODULES
// -----------------------------------------------------------------------------

fn register_fs_module(lua: &Lua, app_data: &HarnessAppData) -> LuaResult<()> {
    let fs_table = lua.create_table()?;
    let root = app_data.fs_root.clone();

    let r1 = root.clone();
    fs_table.set("read", lua.create_function(move |lua, path: String| {
        match resolve_safe_path(&r1, &path) {
            Some(p) => match std::fs::read_to_string(&p) {
                Ok(c) => Ok((Value::String(lua.create_string(&c)?), Value::Nil)),
                Err(e) => Ok((Value::Nil, Value::String(lua.create_string(&e.to_string())?))),
            },
            None => Ok((Value::Nil, Value::String(lua.create_string("Unsafe path traversal")?))),
        }
    })?)?;

    let r2 = root.clone();
    fs_table.set("write", lua.create_function(move |lua, (path, content): (String, String)| {
        if content.len() > MAX_HARNESS_FILE_SIZE {
            return Ok((Value::Boolean(false), Value::String(lua.create_string("File exceeds max size")?)));
        }
        match resolve_safe_path(&r2, &path) {
            Some(p) => {
                if let Some(parent) = p.parent() { let _ = std::fs::create_dir_all(parent); }
                match std::fs::write(&p, content) {
                    Ok(_) => Ok((Value::Boolean(true), Value::Nil)),
                    Err(e) => Ok((Value::Boolean(false), Value::String(lua.create_string(&e.to_string())?))),
                }
            }
            None => Ok((Value::Boolean(false), Value::String(lua.create_string("Unsafe path traversal")?))),
        }
    })?)?;

    let r3 = root.clone();
    fs_table.set("exists", lua.create_function(move |_lua, path: String| {
        match resolve_safe_path(&r3, &path) {
            Some(p) => Ok(p.exists()),
            None => Ok(false),
        }
    })?)?;

    let r4 = root.clone();
    fs_table.set("is_safe_path", lua.create_function(move |_lua, path: String| {
        Ok(resolve_safe_path(&r4, &path).is_some())
    })?)?;

    lua.globals().set("fs", fs_table)?;
    Ok(())
}

fn register_json_module(lua: &Lua) -> LuaResult<()> {
    let json_table = lua.create_table()?;
    json_table.set("encode", lua.create_function(|lua, val: Value| {
        match serde_json::to_string(&val) {
            Ok(s) => Ok((Value::String(lua.create_string(&s)?), Value::Nil)),
            Err(e) => Ok((Value::Nil, Value::String(lua.create_string(&e.to_string())?))),
        }
    })?)?;
    json_table.set("decode", lua.create_function(|lua, s: String| {
        match serde_json::from_str::<serde_json::Value>(&s) {
            Ok(j) => Ok((lua.to_value(&j)?, Value::Nil)),
            Err(e) => Ok((Value::Nil, Value::String(lua.create_string(&e.to_string())?))),
        }
    })?)?;
    lua.globals().set("json", json_table)?;
    Ok(())
}

fn register_time_module(lua: &Lua) -> LuaResult<()> {
    let time_table = lua.create_table()?;
    time_table.set("epoch_seconds", lua.create_function(|_lua, ()| {
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
        Ok(ts)
    })?)?;
    time_table.set("now_utc", lua.create_function(|lua, ()| {
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs().to_string();
        Ok(Value::String(lua.create_string(&ts)?))
    })?)?;
    lua.globals().set("time", time_table)?;
    Ok(())
}

fn register_log_function(lua: &Lua) -> LuaResult<()> {
    lua.globals().set("log", lua.create_function(|_lua, msg: String| {
        eprintln!("[harness] {}", msg);
        Ok(())
    })?)?;
    Ok(())
}
