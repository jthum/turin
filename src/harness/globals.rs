//! Turin-SL canonical globals injected into the Luau harness VM.

use mlua::{Lua, LuaSerdeExt, MultiValue, Result as LuaResult, Table, Value};
use std::future::Future;
use std::path::PathBuf;
use tokio::sync::Mutex;

use crate::harness::stdlib::{
    memory_kv_bindings, runtime_agent, runtime_data, runtime_db, runtime_policy,
    session_user_aliases, system_globals,
};
use crate::inference::embeddings::EmbeddingProvider;
use crate::inference::provider::ProviderClient;
use crate::kernel::identity::{ContextSelector, RuntimeIdentity};
use crate::kernel::policy::PolicyScope;
use crate::kernel::session::QueuedTask;
use crate::persistence::manager::{StoreManager, StoreSelector};

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

pub(crate) enum SqlParams {
    None,
    Positional(Vec<turso::Value>),
    Named(Vec<(String, turso::Value)>),
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

fn get_active_identity(app_data: &HarnessAppData) -> anyhow::Result<RuntimeIdentity> {
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

fn identity_to_lua_table(lua: &Lua, identity: &RuntimeIdentity) -> LuaResult<Table> {
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

pub(crate) fn normalize_selector(mut selector: ContextSelector) -> anyhow::Result<ContextSelector> {
    if selector.namespace.trim().is_empty() {
        selector.namespace = "default".to_string();
    }
    if selector.visibility.trim().is_empty() {
        selector.visibility = "private".to_string();
    }

    let mut tags = Vec::new();
    for tag in selector.tags {
        let t = tag.trim();
        if t.is_empty() {
            continue;
        }
        if !t.contains(':') {
            anyhow::bail!("invalid tag '{}': expected 'dimension:value'", t);
        }
        tags.push(t.to_string());
    }
    tags.sort();
    tags.dedup();
    selector.tags = tags;
    Ok(selector)
}

pub(crate) fn table_to_selector(ctx_tbl: Table) -> LuaResult<ContextSelector> {
    let mut tags = Vec::new();
    if let Ok(Value::Table(tags_tbl)) = ctx_tbl.get::<Value>("tags") {
        for pair in tags_tbl.sequence_values::<String>() {
            tags.push(pair?);
        }
    }

    let namespace = ctx_tbl
        .get::<String>("namespace")
        .unwrap_or_else(|_| "default".to_string());
    let visibility = ctx_tbl
        .get::<String>("visibility")
        .unwrap_or_else(|_| "private".to_string());

    let selector = ContextSelector {
        tags,
        namespace,
        visibility,
    };
    normalize_selector(selector).map_err(mlua::Error::runtime)
}

fn selector_to_lua_table(lua: &Lua, selector: &ContextSelector) -> LuaResult<Table> {
    let ctx = lua.create_table()?;
    ctx.set("tags", lua.create_sequence_from(selector.tags.clone())?)?;
    ctx.set("namespace", selector.namespace.clone())?;
    ctx.set("visibility", selector.visibility.clone())?;
    Ok(ctx)
}

fn context_opts_to_selector(
    _lua: &Lua,
    scope: &str,
    id: Option<String>,
    opts: Option<Table>,
) -> LuaResult<ContextSelector> {
    let tag = if scope == "global" {
        format!("global:{}", id.unwrap_or_else(|| "*".to_string()))
    } else {
        let id = id.ok_or_else(|| {
            mlua::Error::runtime(format!("runtime.context('{}', ...) requires an id", scope))
        })?;
        format!("{}:{}", scope, id)
    };

    let mut selector = ContextSelector {
        tags: vec![tag],
        namespace: "default".to_string(),
        visibility: "private".to_string(),
    };

    if let Some(opts_tbl) = opts {
        if let Ok(ns) = opts_tbl.get::<String>("namespace") {
            selector.namespace = ns;
        }
        if let Ok(vis) = opts_tbl.get::<String>("visibility") {
            selector.visibility = vis;
        }
    }

    normalize_selector(selector).map_err(mlua::Error::runtime)
}

fn parse_context_args(lua: &Lua, args: MultiValue) -> LuaResult<ContextSelector> {
    let mut it = args.into_iter();
    let first = it.next().unwrap_or(Value::Nil);
    match first {
        Value::Table(tbl) => table_to_selector(tbl),
        Value::String(scope) => {
            let scope = scope.to_str()?.to_string();
            let second = it.next().unwrap_or(Value::Nil);
            let third = it.next().unwrap_or(Value::Nil);

            let (id, opts) = match (second, third) {
                (Value::Nil, Value::Nil) => (None, None),
                (Value::Nil, Value::Table(opts)) => (None, Some(opts)),
                (Value::String(id), Value::Nil) => (Some(id.to_str()?.to_string()), None),
                (Value::String(id), Value::Table(opts)) => {
                    (Some(id.to_str()?.to_string()), Some(opts))
                }
                (Value::Table(opts), Value::Nil) => (None, Some(opts)),
                _ => {
                    return Err(mlua::Error::runtime(
                        "runtime.context invalid signature; expected (scope, id?, opts?) or ({tags=...})",
                    ));
                }
            };
            context_opts_to_selector(lua, &scope, id, opts)
        }
        _ => Err(mlua::Error::runtime(
            "runtime.context invalid signature; expected (scope, id?, opts?) or ({tags=...})",
        )),
    }
}

fn selector_from_active_scope(
    app_data: &HarnessAppData,
    scope: &str,
) -> anyhow::Result<ContextSelector> {
    let identity = get_active_identity(app_data)?;
    let selector = match scope {
        "agent" => ContextSelector {
            tags: vec![format!("agent:{}", identity.agent_id())],
            namespace: "default".to_string(),
            visibility: "private".to_string(),
        },
        "session" => ContextSelector {
            tags: vec![format!("session:{}", identity.session_id())],
            namespace: "default".to_string(),
            visibility: "private".to_string(),
        },
        "user" => {
            let user_id = identity
                .user_id()
                .ok_or_else(|| anyhow::anyhow!("user.* requires RuntimeIdentity.user_id"))?;
            ContextSelector {
                tags: vec![format!("user:{}", user_id)],
                namespace: "default".to_string(),
                visibility: "private".to_string(),
            }
        }
        _ => anyhow::bail!("Unsupported implicit scope: {}", scope),
    };
    normalize_selector(selector)
}

pub(crate) fn selector_from_active_scope_lua(
    app_data: &HarnessAppData,
    scope: &'static str,
) -> LuaResult<ContextSelector> {
    selector_from_active_scope(app_data, scope).map_err(|e| mlua::Error::runtime(e.to_string()))
}

pub(crate) fn search_limit_from_opt(arg: Option<Value>) -> LuaResult<usize> {
    match arg {
        None | Some(Value::Nil) => Ok(5),
        Some(Value::Integer(i)) => Ok(i.max(0) as usize),
        Some(Value::Number(n)) => Ok(n.max(0.0) as usize),
        Some(Value::Table(t)) => {
            let limit = t.get::<usize>("limit").unwrap_or(5);
            Ok(limit)
        }
        Some(_) => Err(mlua::Error::runtime(
            "invalid opts; expected number limit or options table",
        )),
    }
}

pub(crate) fn runtime_policy_snapshot(
    app_data: &HarnessAppData,
) -> anyhow::Result<HashMap<String, serde_json::Value>> {
    let mut scope = PolicyScope::default();
    if let Ok(identity) = get_active_identity(app_data) {
        scope.agent_id = Some(identity.agent_id().to_string());
        scope.session_id = Some(identity.session_id().to_string());
        scope.run_id = identity.run_id().map(ToString::to_string);
    }

    let policy_manager = app_data.policy_manager.clone();
    Ok(block_on_current(async move {
        policy_manager.snapshot(&scope).await
    }))
}

pub(crate) fn policy_bool(
    snapshot: &HashMap<String, serde_json::Value>,
    key: &str,
    default: bool,
) -> bool {
    snapshot
        .get(key)
        .and_then(|v| v.as_bool())
        .unwrap_or(default)
}

pub(crate) fn policy_u64(
    snapshot: &HashMap<String, serde_json::Value>,
    key: &str,
    default: u64,
) -> u64 {
    snapshot
        .get(key)
        .and_then(|v| v.as_u64())
        .unwrap_or(default)
}

pub(crate) fn policy_string<'a>(
    snapshot: &'a HashMap<String, serde_json::Value>,
    key: &str,
    default: &'a str,
) -> &'a str {
    snapshot
        .get(key)
        .and_then(|v| v.as_str())
        .unwrap_or(default)
}

fn parse_store_selector_string(s: &str) -> StoreSelector {
    if s.contains('/')
        || s.contains('\\')
        || s.starts_with('.')
        || s.ends_with(".db")
        || s.starts_with('~')
    {
        StoreSelector::Path(s.to_string())
    } else {
        StoreSelector::Alias(s.to_string())
    }
}

pub(crate) fn selector_from_db_value(value: Value) -> LuaResult<StoreSelector> {
    match value {
        Value::String(s) => Ok(parse_store_selector_string(&s.to_str()?)),
        Value::Table(t) => {
            if let Ok(handle) = t.get::<String>("handle") {
                return Ok(StoreSelector::Handle(handle));
            }
            if let Ok(path) = t.get::<String>("path") {
                return Ok(StoreSelector::Path(path));
            }
            if let Ok(store) = t.get::<String>("store") {
                return Ok(StoreSelector::Alias(store));
            }
            if let Ok(alias) = t.get::<String>("alias") {
                return Ok(StoreSelector::Alias(alias));
            }
            if let Ok(Value::Table(selector_tbl)) = t.get::<Value>("selector") {
                let selector = table_to_selector(selector_tbl)?;
                return Ok(StoreSelector::Alias(selector.to_alias()));
            }
            // Allow raw context selector table directly.
            let selector = table_to_selector(t)?;
            Ok(StoreSelector::Alias(selector.to_alias()))
        }
        _ => Err(mlua::Error::runtime(
            "invalid db selector; expected string or table",
        )),
    }
}

pub(crate) fn selector_from_db_opts(opts: Option<Table>) -> LuaResult<StoreSelector> {
    match opts {
        Some(t) => selector_from_db_value(Value::Table(t)),
        None => Ok(StoreSelector::Alias("state".to_string())),
    }
}

fn lua_value_to_sql_param(value: Value) -> LuaResult<turso::Value> {
    match value {
        Value::Nil => Ok(turso::Value::Null),
        Value::Boolean(b) => Ok(turso::Value::Integer(if b { 1 } else { 0 })),
        Value::Integer(i) => Ok(turso::Value::Integer(i)),
        Value::Number(n) => Ok(turso::Value::Real(n)),
        Value::String(s) => Ok(turso::Value::Text(s.to_str()?.to_string())),
        _ => Err(mlua::Error::runtime(
            "invalid SQL param type; expected nil/boolean/number/string",
        )),
    }
}

pub(crate) fn lua_table_to_sql_params(tbl: Option<Table>) -> LuaResult<SqlParams> {
    let Some(tbl) = tbl else {
        return Ok(SqlParams::None);
    };

    let mut positional = Vec::<(usize, turso::Value)>::new();
    let mut named = Vec::<(String, turso::Value)>::new();
    let mut saw_integer_key = false;
    let mut saw_string_key = false;

    for pair in tbl.pairs::<Value, Value>() {
        let (k, v) = pair?;
        match k {
            Value::Integer(i) if i >= 1 => {
                saw_integer_key = true;
                positional.push((i as usize, lua_value_to_sql_param(v)?));
            }
            Value::String(s) => {
                saw_string_key = true;
                let mut name = s.to_str()?.to_string();
                if !name.starts_with(':') && !name.starts_with('@') && !name.starts_with('$') {
                    name = format!(":{}", name);
                }
                named.push((name, lua_value_to_sql_param(v)?));
            }
            _ => {
                return Err(mlua::Error::runtime(
                    "invalid SQL params key; expected array indices or string keys",
                ));
            }
        }
    }

    if saw_integer_key && saw_string_key {
        return Err(mlua::Error::runtime(
            "mixed positional and named SQL params are not supported",
        ));
    }

    if saw_integer_key {
        positional.sort_by_key(|(i, _)| *i);
        let mut out = Vec::with_capacity(positional.len());
        for (idx, val) in positional {
            if idx != out.len() + 1 {
                return Err(mlua::Error::runtime(
                    "positional SQL params must be a dense 1-based array",
                ));
            }
            out.push(val);
        }
        Ok(SqlParams::Positional(out))
    } else {
        Ok(SqlParams::Named(named))
    }
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{:02x}", b);
    }
    out
}

pub(crate) fn sql_value_to_json(value: turso::Value) -> serde_json::Value {
    match value {
        turso::Value::Null => serde_json::Value::Null,
        turso::Value::Integer(i) => serde_json::json!(i),
        turso::Value::Real(n) => serde_json::json!(n),
        turso::Value::Text(s) => serde_json::json!(s),
        turso::Value::Blob(b) => serde_json::json!({
            "__type": "blob",
            "hex": bytes_to_hex(&b),
        }),
    }
}

fn format_uuid_bytes_simple(bytes: &[u8]) -> Option<String> {
    if bytes.len() != 16 {
        return None;
    }
    let uuid = uuid::Uuid::from_slice(bytes).ok()?;
    Some(uuid.simple().to_string())
}

fn session_row_to_lua_table(
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

fn wildcard_match(pattern: &str, text: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    let p = pattern.as_bytes();
    let s = text.as_bytes();
    let (mut pi, mut si, mut star_idx, mut match_idx) = (0usize, 0usize, None, 0usize);
    while si < s.len() {
        if pi < p.len() && (p[pi] == s[si]) {
            pi += 1;
            si += 1;
        } else if pi < p.len() && p[pi] == b'*' {
            star_idx = Some(pi);
            match_idx = si;
            pi += 1;
        } else if let Some(star) = star_idx {
            pi = star + 1;
            match_idx += 1;
            si = match_idx;
        } else {
            return false;
        }
    }
    while pi < p.len() && p[pi] == b'*' {
        pi += 1;
    }
    pi == p.len()
}

pub(crate) fn policy_scope_from_value(
    app_data: &HarnessAppData,
    scope: Option<Value>,
) -> LuaResult<PolicyScope> {
    let mut out = PolicyScope::default();

    match scope {
        None | Some(Value::Nil) => {
            out.scope = Some("global".to_string());
            return Ok(out);
        }
        Some(Value::String(s)) => {
            out.scope = Some(s.to_str()?.to_string());
        }
        Some(Value::Table(t)) => {
            if let Ok(s) = t.get::<String>("scope") {
                out.scope = Some(s);
            }
            if let Ok(agent_id) = t.get::<String>("agent_id") {
                out.agent_id = Some(agent_id);
            }
            if let Ok(session_id) = t.get::<String>("session_id") {
                out.session_id = Some(session_id);
            }
            if let Ok(run_id) = t.get::<String>("run_id") {
                out.run_id = Some(run_id);
            }
        }
        _ => {
            return Err(mlua::Error::runtime(
                "invalid policy scope; expected nil, string, or table",
            ));
        }
    }

    if out.scope.is_none() {
        out.scope = Some("global".to_string());
    }

    // Fill common defaults from active runtime identity when available.
    if (out.agent_id.is_none() || out.session_id.is_none() || out.run_id.is_none())
        && let Ok(identity) = get_active_identity(app_data)
    {
        if out.agent_id.is_none() {
            out.agent_id = Some(identity.agent_id().to_string());
        }
        if out.session_id.is_none() {
            out.session_id = Some(identity.session_id().to_string());
        }
        if out.run_id.is_none() {
            out.run_id = identity.run_id().map(ToString::to_string);
        }
    }

    Ok(out)
}

fn visibility_allowed(selector: &ContextSelector) -> anyhow::Result<()> {
    match selector.visibility.as_str() {
        "private" => Ok(()),
        "children" | "agent_group" | "all_agents" => {
            anyhow::bail!(
                "Policy denial: visibility '{}' not enabled",
                selector.visibility
            )
        }
        other => anyhow::bail!("Invalid visibility: {}", other),
    }
}

async fn open_selector_store(
    manager: &StoreManager,
    selector: &ContextSelector,
) -> anyhow::Result<Arc<crate::persistence::state::StateStore>> {
    visibility_allowed(selector)?;
    manager
        .open(&StoreSelector::Alias(selector.to_alias()))
        .await
        .map_err(|e| anyhow::anyhow!(e.to_string()))
}

async fn ensure_context_memory_session(
    store: &crate::persistence::state::StateStore,
    selector: &ContextSelector,
) -> anyhow::Result<i64> {
    const KEY: &str = "__turin_context_session_public_id";

    let public_id = if let Some(existing) = store.kv_get(KEY).await? {
        uuid::Uuid::parse_str(&existing)
            .map_err(|e| anyhow::anyhow!("Invalid stored context session UUID: {}", e))?
    } else {
        let new_id = uuid::Uuid::now_v7();
        store.kv_set(KEY, &new_id.simple().to_string()).await?;
        new_id
    };

    if let Some(id) = store.get_session_by_public_id(public_id).await? {
        return Ok(id);
    }

    let agent_id = selector
        .tags
        .iter()
        .find_map(|t| t.strip_prefix("agent:").map(ToOwned::to_owned))
        .unwrap_or_else(|| "context".to_string());
    let metadata = serde_json::to_string(selector).ok();
    store
        .create_session(public_id, &agent_id, metadata.as_deref())
        .await
}

pub(crate) async fn kv_get_backend(
    manager: &StoreManager,
    selector: &ContextSelector,
    key: &str,
) -> anyhow::Result<Option<String>> {
    let store = open_selector_store(manager, selector).await?;
    store.kv_get(key).await
}

pub(crate) async fn kv_set_backend(
    manager: &StoreManager,
    selector: &ContextSelector,
    key: &str,
    value: &str,
) -> anyhow::Result<()> {
    let store = open_selector_store(manager, selector).await?;
    store.kv_set(key, value).await
}

pub(crate) async fn kv_delete_backend(
    manager: &StoreManager,
    selector: &ContextSelector,
    key: &str,
) -> anyhow::Result<()> {
    let store = open_selector_store(manager, selector).await?;
    store.kv_delete(key).await
}

pub(crate) async fn memory_store_backend(
    manager: &StoreManager,
    embedding_provider: Option<&Arc<dyn EmbeddingProvider>>,
    selector: &ContextSelector,
    content: &str,
    metadata: &serde_json::Value,
) -> anyhow::Result<()> {
    let store = open_selector_store(manager, selector).await?;
    let session_id = ensure_context_memory_session(&store, selector).await?;

    let provider =
        embedding_provider.ok_or_else(|| anyhow::anyhow!("No embedding provider configured"))?;
    let emb = provider.embed(content).await?;
    store
        .insert_memory(session_id, content, &emb.vector, metadata)
        .await
}

pub(crate) async fn memory_search_backend(
    manager: &StoreManager,
    embedding_provider: Option<&Arc<dyn EmbeddingProvider>>,
    selector: &ContextSelector,
    query: &str,
    limit: usize,
) -> anyhow::Result<Vec<crate::persistence::schema::MemoryRow>> {
    let store = open_selector_store(manager, selector).await?;
    let session_id = ensure_context_memory_session(&store, selector).await?;

    let vector = if let Some(provider) = embedding_provider {
        provider.embed(query).await.ok().map(|emb| emb.vector)
    } else {
        None
    };

    store
        .search_memories(session_id, vector.as_deref(), Some(query), limit)
        .await
}

fn register_runtime_module(lua: &Lua, app_data: &HarnessAppData) -> LuaResult<()> {
    let runtime_table = lua.create_table()?;
    let context_ns = lua.create_table()?;
    {
        let manager = app_data.store_manager.clone();
        context_ns.set(
            "glob",
            lua.create_function(move |lua, pattern: String| {
                let manager = manager.clone();
                let aliases = block_on_current(async move { manager.list_aliases().await });
                let out = lua.create_table()?;
                let mut idx = 1;
                for alias in aliases {
                    if wildcard_match(&pattern, &alias) {
                        out.set(idx, alias)?;
                        idx += 1;
                    }
                }
                Ok((Value::Table(out), Value::Nil))
            })?,
        )?;
    }
    let context_meta = lua.create_table()?;
    context_meta.set(
        "__call",
        lua.create_function(|lua, (_self, args): (Value, MultiValue)| {
            let selector = parse_context_args(lua, args)?;
            Ok(Value::Table(selector_to_lua_table(lua, &selector)?))
        })?,
    )?;
    let _ = context_ns.set_metatable(Some(context_meta));
    runtime_table.set("context", context_ns)?;

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
    let agent_table = lua.create_table()?;
    let session_ns = lua.create_table()?;
    let mode_ns = lua.create_table()?;

    let agent_manager = app_data.agent_manager.clone();

    // agent.spawn (local subtask enqueue for current session queue)
    let spawn_q = app_data.queue.clone();
    let spawn_policy_snapshot = app_data.clone();
    let spawn_depth = app_data.spawn_depth;
    agent_table.set(
        "spawn",
        lua.create_function(move |lua, (prompt, _opts): (String, Option<Table>)| {
            let snapshot =
                runtime_policy_snapshot(&spawn_policy_snapshot).map_err(mlua::Error::runtime)?;
            if !policy_bool(&snapshot, "spawn.enabled", true) {
                return Ok((
                    Value::Nil,
                    Value::String(lua.create_string("Policy denial: spawn.enabled=false")?),
                ));
            }
            let max_depth = policy_u64(&snapshot, "spawn.max_depth", 3) as u32;
            if spawn_depth >= max_depth {
                return Ok((
                    Value::Nil,
                    Value::String(lua.create_string("Policy denial: spawn.max_depth exceeded")?),
                ));
            }
            let spawn_q = spawn_q.clone();
            let enqueue_res = block_on_current(async {
                if let Some(q) = &*spawn_q.lock().await {
                    let queue_max = policy_u64(&snapshot, "queue.max_depth", 1024) as usize;
                    let mut q = q.lock().await;
                    if q.len() >= queue_max {
                        return Err(format!(
                            "Policy denial: queue.max_depth={} reached",
                            queue_max
                        ));
                    }
                    q.push_back(QueuedTask::ad_hoc(prompt.clone()));
                    Ok(())
                } else {
                    Err("No active session queue".to_string())
                }
            });
            match enqueue_res {
                Ok(()) => {
                    let token = format!("q_{}", uuid::Uuid::now_v7().simple());
                    Ok((Value::String(lua.create_string(&token)?), Value::Nil))
                }
                Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
            }
        })?,
    )?;

    // agent.complete
    {
        let manager = app_data.agent_manager.clone();
        let default_agent = app_data.config.agent.id.clone();
        let complete_policy_snapshot = app_data.clone();
        agent_table.set(
            "complete",
            lua.create_function(move |lua, (prompt, opts): (String, Option<Table>)| {
                let snapshot = runtime_policy_snapshot(&complete_policy_snapshot)
                    .map_err(mlua::Error::runtime)?;
                if !policy_bool(&snapshot, "spawn.enabled", true) {
                    return Ok((
                        Value::Nil,
                        Value::String(lua.create_string("Policy denial: spawn.enabled=false")?),
                    ));
                }
                let target_agent = opts
                    .as_ref()
                    .and_then(|t| t.get::<String>("agent_id").ok())
                    .unwrap_or_else(|| default_agent.clone());
                let timeout_ms = opts.as_ref().and_then(|t| t.get::<u64>("timeout_ms").ok());

                let manager_submit = manager.clone();
                let request_id = block_on_current(async move {
                    manager_submit
                        .submit(&target_agent, QueuedTask::ad_hoc(prompt))
                        .await
                        .map_err(|e| e.to_string())
                });
                let request_id = match request_id {
                    Ok(id) => id,
                    Err(err) => return Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                };

                let manager_await = manager.clone();
                let result = block_on_current(async move {
                    manager_await
                        .await_result(&request_id, timeout_ms)
                        .await
                        .map_err(|e| e.to_string())
                });
                match result {
                    Ok(res) => {
                        if let Some(err) = res.error {
                            Ok((Value::Nil, Value::String(lua.create_string(&err)?)))
                        } else if let Some(output) = res.output {
                            Ok((Value::String(lua.create_string(&output)?), Value::Nil))
                        } else {
                            Ok((Value::String(lua.create_string("")?), Value::Nil))
                        }
                    }
                    Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                }
            })?,
        )?;
    }

    // agent.session.identity()
    session_ns.set(
        "identity",
        lua.create_function(move |lua, ()| {
            let app_data = lua
                .app_data_ref::<HarnessAppData>()
                .ok_or_else(|| mlua::Error::runtime("missing harness app data"))?;
            let identity = get_active_identity(&app_data).map_err(mlua::Error::runtime)?;
            identity_to_lua_table(lua, &identity)
        })?,
    )?;

    // agent.session.queue
    let aq = app_data.queue.clone();
    let queue_policy_snapshot = app_data.clone();
    session_ns.set(
        "queue",
        lua.create_function(move |lua, cmd: String| {
            let aq = aq.clone();
            let snapshot =
                runtime_policy_snapshot(&queue_policy_snapshot).map_err(mlua::Error::runtime)?;
            let queue_max = policy_u64(&snapshot, "queue.max_depth", 1024) as usize;
            let res = block_on_current(async {
                if let Some(q) = &*aq.lock().await {
                    let mut q = q.lock().await;
                    if q.len() >= queue_max {
                        return Err(format!(
                            "Policy denial: queue.max_depth={} reached",
                            queue_max
                        ));
                    }
                    q.push_back(QueuedTask::ad_hoc(cmd));
                    Ok(())
                } else {
                    Err("No active session queue".to_string())
                }
            });
            match res {
                Ok(()) => Ok((Value::Boolean(true), Value::Nil)),
                Err(err) => Ok((
                    Value::Boolean(false),
                    Value::String(lua.create_string(&err)?),
                )),
            }
        })?,
    )?;

    // agent.session.queue_next
    let aq2 = app_data.queue.clone();
    let queue_next_policy_snapshot = app_data.clone();
    session_ns.set(
        "queue_next",
        lua.create_function(move |lua, cmd: String| {
            let aq = aq2.clone();
            let snapshot = runtime_policy_snapshot(&queue_next_policy_snapshot)
                .map_err(mlua::Error::runtime)?;
            let queue_max = policy_u64(&snapshot, "queue.max_depth", 1024) as usize;
            let res = block_on_current(async {
                if let Some(q) = &*aq.lock().await {
                    let mut q = q.lock().await;
                    if q.len() >= queue_max {
                        return Err(format!(
                            "Policy denial: queue.max_depth={} reached",
                            queue_max
                        ));
                    }
                    q.push_front(QueuedTask::ad_hoc(cmd));
                    Ok(())
                } else {
                    Err("No active session queue".to_string())
                }
            });
            match res {
                Ok(()) => Ok((Value::Boolean(true), Value::Nil)),
                Err(err) => Ok((
                    Value::Boolean(false),
                    Value::String(lua.create_string(&err)?),
                )),
            }
        })?,
    )?;

    // agent.session.queue_all
    let aq3 = app_data.queue.clone();
    let queue_all_policy_snapshot = app_data.clone();
    session_ns.set(
        "queue_all",
        lua.create_function(move |lua, commands: Table| {
            let mut items = Vec::new();
            for v in commands.sequence_values::<String>() {
                items.push(v?);
            }
            let snapshot = runtime_policy_snapshot(&queue_all_policy_snapshot)
                .map_err(mlua::Error::runtime)?;
            let queue_max = policy_u64(&snapshot, "queue.max_depth", 1024) as usize;
            let aq = aq3.clone();
            let res = block_on_current(async {
                if let Some(q) = &*aq.lock().await {
                    let mut q = q.lock().await;
                    if q.len().saturating_add(items.len()) > queue_max {
                        return Err(format!(
                            "Policy denial: queue.max_depth={} would be exceeded",
                            queue_max
                        ));
                    }
                    for cmd in &items {
                        q.push_back(QueuedTask::ad_hoc(cmd.clone()));
                    }
                    Ok(())
                } else {
                    Err("No active session queue".to_string())
                }
            });
            match res {
                Ok(()) => Ok((Value::Boolean(true), Value::Nil)),
                Err(err) => Ok((
                    Value::Boolean(false),
                    Value::String(lua.create_string(&err)?),
                )),
            }
        })?,
    )?;

    // agent.session.load(session_id)
    {
        let manager = app_data.store_manager.clone();
        session_ns.set(
            "load",
            lua.create_function(move |lua, session_id: String| {
                let manager = manager.clone();
                let result = block_on_current(async move {
                    let store = manager.get_default().await.map_err(|e| e.to_string())?;
                    let uuid = uuid::Uuid::parse_str(&session_id).map_err(|e| e.to_string())?;
                    let row = store
                        .get_session_row_by_public_id(uuid)
                        .await
                        .map_err(|e| e.to_string())?;
                    Ok::<_, String>(row)
                });
                match result {
                    Ok(Some(row)) => Ok((
                        Value::Table(session_row_to_lua_table(lua, &row)?),
                        Value::Nil,
                    )),
                    Ok(None) => Ok((Value::Nil, Value::Nil)),
                    Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                }
            })?,
        )?;
    }

    // agent.session.list(limit?, offset?)
    {
        let manager = app_data.store_manager.clone();
        session_ns.set(
            "list",
            lua.create_function(
                move |lua, (limit, offset): (Option<usize>, Option<usize>)| {
                    let limit = limit.unwrap_or(20);
                    let offset = offset.unwrap_or(0);
                    let manager = manager.clone();
                    let result = block_on_current(async move {
                        let store = manager.get_default().await.map_err(|e| e.to_string())?;
                        store
                            .list_session_rows(limit, offset)
                            .await
                            .map_err(|e| e.to_string())
                    });
                    match result {
                        Ok(rows) => {
                            let out = lua.create_table()?;
                            for (i, row) in rows.iter().enumerate() {
                                out.set(i + 1, session_row_to_lua_table(lua, row)?)?;
                            }
                            Ok((Value::Table(out), Value::Nil))
                        }
                        Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                    }
                },
            )?,
        )?;
    }

    let sm1 = app_data.active_session_mode.clone();
    mode_ns.set(
        "get",
        lua.create_function(move |lua, ()| {
            let mode = sm1
                .lock()
                .unwrap()
                .clone()
                .unwrap_or(crate::kernel::config::AgentMode::Auto);
            let mode_str = match mode {
                crate::kernel::config::AgentMode::Auto => "auto",
                crate::kernel::config::AgentMode::Stateful => "stateful",
                crate::kernel::config::AgentMode::Stateless => "stateless",
            };
            Ok(Value::String(lua.create_string(mode_str)?))
        })?,
    )?;

    let sm2 = app_data.active_session_mode.clone();
    mode_ns.set(
        "set",
        lua.create_function(move |lua, m: String| {
            let mode = match m.as_str() {
                "stateful" => crate::kernel::config::AgentMode::Stateful,
                "stateless" => crate::kernel::config::AgentMode::Stateless,
                "auto" => crate::kernel::config::AgentMode::Auto,
                _ => {
                    return Ok((
                        Value::Boolean(false),
                        Value::String(
                            lua.create_string("invalid mode; expected auto|stateful|stateless")?,
                        ),
                    ));
                }
            };
            if let Ok(mut lock) = sm2.lock() {
                *lock = Some(mode);
            }
            Ok((Value::Boolean(true), Value::Nil))
        })?,
    )?;

    agent_table.set("session", session_ns)?;
    agent_table.set("mode", mode_ns)?;

    // Deprecated send
    agent_table.set(
        "send",
        lua.create_function(move |_lua, (id, prompt): (String, String)| {
            let m = agent_manager.clone();
            block_on_current(async {
                let _ = m.send(&id, QueuedTask::ad_hoc(prompt)).await;
            });
            Ok((Value::Boolean(true), Value::Nil))
        })?,
    )?;

    lua.globals().set("agent", agent_table)?;
    Ok(())
}
