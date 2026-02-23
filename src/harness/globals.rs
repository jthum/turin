//! Turin-SL canonical globals injected into the Luau harness VM.

use mlua::{Lua, LuaSerdeExt, MultiValue, Result as LuaResult, Table, Value};
use std::path::{Path, PathBuf};
use tokio::sync::Mutex;

use crate::inference::embeddings::EmbeddingProvider;
use crate::inference::provider::ProviderClient;
use crate::kernel::policy::PolicyScope;
use crate::kernel::session::QueuedTask;
use crate::persistence::manager::{StoreManager, StorePathScope, StoreSelector};
use crate::kernel::identity::{ContextSelector, RuntimeIdentity};

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

const MAX_HARNESS_FILE_SIZE: usize = 10 * 1024 * 1024;

pub type SessionQueue = Arc<Mutex<VecDeque<QueuedTask>>>;
pub type ActiveSessionQueue = Arc<Mutex<Option<SessionQueue>>>;

/// Shared state passed to async Lua callbacks via app data.
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

enum SqlParams {
    None,
    Positional(Vec<turso::Value>),
    Named(Vec<(String, turso::Value)>),
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

fn get_active_identity(app_data: &HarnessAppData) -> anyhow::Result<RuntimeIdentity> {
    let session_id = app_data
        .active_session_id
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| anyhow::anyhow!("No active session context"))?;

    Ok(RuntimeIdentity::new(session_id, app_data.config.agent.id.clone()))
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

fn resolve_safe_path(root: &Path, path_str: &str) -> Option<PathBuf> {
    crate::tools::is_safe_path(root, Path::new(path_str)).ok()
}

fn normalize_selector(mut selector: ContextSelector) -> anyhow::Result<ContextSelector> {
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

fn table_to_selector(ctx_tbl: Table) -> LuaResult<ContextSelector> {
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

fn context_opts_to_selector(_lua: &Lua, scope: &str, id: Option<String>, opts: Option<Table>) -> LuaResult<ContextSelector> {
    let tag = if scope == "global" {
        format!("global:{}", id.unwrap_or_else(|| "*".to_string()))
    } else {
        let id = id.ok_or_else(|| mlua::Error::runtime(format!("runtime.context('{}', ...) requires an id", scope)))?;
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
                (Value::String(id), Value::Table(opts)) => (Some(id.to_str()?.to_string()), Some(opts)),
                (Value::Table(opts), Value::Nil) => (None, Some(opts)),
                _ => {
                    return Err(mlua::Error::runtime(
                        "runtime.context invalid signature; expected (scope, id?, opts?) or ({tags=...})",
                    ))
                }
            };
            context_opts_to_selector(lua, &scope, id, opts)
        }
        _ => Err(mlua::Error::runtime(
            "runtime.context invalid signature; expected (scope, id?, opts?) or ({tags=...})",
        )),
    }
}

fn selector_from_active_scope(app_data: &HarnessAppData, scope: &str) -> anyhow::Result<ContextSelector> {
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

fn search_limit_from_opt(arg: Option<Value>) -> LuaResult<usize> {
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

fn runtime_policy_snapshot(app_data: &HarnessAppData) -> anyhow::Result<HashMap<String, serde_json::Value>> {
    let mut scope = PolicyScope::default();
    if let Ok(identity) = get_active_identity(app_data) {
        scope.agent_id = Some(identity.agent_id().to_string());
        scope.session_id = Some(identity.session_id().to_string());
        scope.run_id = identity.run_id().map(ToString::to_string);
    }

    let policy_manager = app_data.policy_manager.clone();
    Ok(tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(async move { policy_manager.snapshot(&scope).await })
    }))
}

fn policy_bool(snapshot: &HashMap<String, serde_json::Value>, key: &str, default: bool) -> bool {
    snapshot.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
}

fn policy_u64(snapshot: &HashMap<String, serde_json::Value>, key: &str, default: u64) -> u64 {
    snapshot.get(key).and_then(|v| v.as_u64()).unwrap_or(default)
}

fn policy_string<'a>(
    snapshot: &'a HashMap<String, serde_json::Value>,
    key: &str,
    default: &'a str,
) -> &'a str {
    snapshot.get(key).and_then(|v| v.as_str()).unwrap_or(default)
}

fn parse_store_selector_string(s: &str) -> StoreSelector {
    if s.contains('/') || s.contains('\\') || s.starts_with('.') || s.ends_with(".db") || s.starts_with('~') {
        StoreSelector::Path(s.to_string())
    } else {
        StoreSelector::Alias(s.to_string())
    }
}

fn selector_from_db_value(value: Value) -> LuaResult<StoreSelector> {
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

fn selector_from_db_opts(opts: Option<Table>) -> LuaResult<StoreSelector> {
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

fn lua_table_to_sql_params(tbl: Option<Table>) -> LuaResult<SqlParams> {
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
                ))
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

fn sql_value_to_json(value: turso::Value) -> serde_json::Value {
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

fn policy_scope_from_value(app_data: &HarnessAppData, scope: Option<Value>) -> LuaResult<PolicyScope> {
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
            ))
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
            anyhow::bail!("Policy denial: visibility '{}' not enabled", selector.visibility)
        }
        other => anyhow::bail!("Invalid visibility: {}", other),
    }
}

async fn open_selector_store(manager: &StoreManager, selector: &ContextSelector) -> anyhow::Result<Arc<crate::persistence::state::StateStore>> {
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
    store.create_session(public_id, &agent_id, metadata.as_deref()).await
}

async fn kv_get_backend(
    manager: &StoreManager,
    selector: &ContextSelector,
    key: &str,
) -> anyhow::Result<Option<String>> {
    let store = open_selector_store(manager, selector).await?;
    store.kv_get(key).await
}

async fn kv_set_backend(
    manager: &StoreManager,
    selector: &ContextSelector,
    key: &str,
    value: &str,
) -> anyhow::Result<()> {
    let store = open_selector_store(manager, selector).await?;
    store.kv_set(key, value).await
}

async fn kv_delete_backend(
    manager: &StoreManager,
    selector: &ContextSelector,
    key: &str,
) -> anyhow::Result<()> {
    let store = open_selector_store(manager, selector).await?;
    store.kv_delete(key).await
}

async fn memory_store_backend(
    manager: &StoreManager,
    embedding_provider: Option<&Arc<dyn EmbeddingProvider>>,
    selector: &ContextSelector,
    content: &str,
    metadata: &serde_json::Value,
) -> anyhow::Result<()> {
    let store = open_selector_store(manager, selector).await?;
    let session_id = ensure_context_memory_session(&store, selector).await?;

    let provider = embedding_provider
        .ok_or_else(|| anyhow::anyhow!("No embedding provider configured"))?;
    let emb = provider.embed(content).await?;
    store.insert_memory(session_id, content, &emb.vector, metadata).await
}

async fn memory_search_backend(
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
                let aliases = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move { manager.list_aliases().await })
                });
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

    // runtime.memory.* canonical backend delegates
    let runtime_memory = lua.create_table()?;
    {
        let manager = app_data.store_manager.clone();
        let embedding = app_data.embedding_provider.clone();
        runtime_memory.set(
            "search",
            lua.create_function(move |lua, (query, ctx, opts): (String, Table, Option<Value>)| {
                let selector = table_to_selector(ctx)?;
                let limit = search_limit_from_opt(opts)?;
                let manager = manager.clone();
                let embedding = embedding.clone();
                let result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move {
                        memory_search_backend(&manager, embedding.as_ref(), &selector, &query, limit)
                            .await
                            .map_err(|e| e.to_string())
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
                    }
                    Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                }
            })?,
        )?;
    }
    {
        let manager = app_data.store_manager.clone();
        let embedding = app_data.embedding_provider.clone();
        runtime_memory.set(
            "store",
            lua.create_function(move |lua, (content, ctx, metadata, _opts): (String, Table, Option<Table>, Option<Table>)| {
                let selector = table_to_selector(ctx)?;
                let metadata_json = if let Some(tbl) = metadata {
                    lua.from_value::<serde_json::Value>(Value::Table(tbl))
                        .map_err(|e| mlua::Error::runtime(format!("invalid metadata table: {}", e)))?
                } else {
                    serde_json::json!({})
                };
                let manager = manager.clone();
                let embedding = embedding.clone();
                let result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move {
                        memory_store_backend(&manager, embedding.as_ref(), &selector, &content, &metadata_json)
                            .await
                            .map_err(|e| e.to_string())
                    })
                });
                match result {
                    Ok(_) => Ok((Value::Boolean(true), Value::Nil)),
                    Err(err) => Ok((Value::Boolean(false), Value::String(lua.create_string(&err)?))),
                }
            })?,
        )?;
    }
    runtime_table.set("memory", runtime_memory)?;

    // runtime.kv.* canonical backend delegates
    let runtime_kv = lua.create_table()?;
    {
        let manager = app_data.store_manager.clone();
        runtime_kv.set(
            "get",
            lua.create_function(move |lua, (key, ctx): (String, Table)| {
                let selector = table_to_selector(ctx)?;
                let manager = manager.clone();
                let result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move {
                        kv_get_backend(&manager, &selector, &key)
                            .await
                            .map_err(|e| e.to_string())
                    })
                });
                match result {
                    Ok(Some(val)) => Ok((Value::String(lua.create_string(&val)?), Value::Nil)),
                    Ok(None) => Ok((Value::Nil, Value::Nil)),
                    Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                }
            })?,
        )?;
    }
    {
        let manager = app_data.store_manager.clone();
        runtime_kv.set(
            "set",
            lua.create_function(move |lua, (key, value, ctx): (String, String, Table)| {
                let selector = table_to_selector(ctx)?;
                let manager = manager.clone();
                let result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move {
                        kv_set_backend(&manager, &selector, &key, &value)
                            .await
                            .map_err(|e| e.to_string())
                    })
                });
                match result {
                    Ok(_) => Ok((Value::Boolean(true), Value::Nil)),
                    Err(err) => Ok((Value::Boolean(false), Value::String(lua.create_string(&err)?))),
                }
            })?,
        )?;
    }
    {
        let manager = app_data.store_manager.clone();
        runtime_kv.set(
            "delete",
            lua.create_function(move |lua, (key, ctx): (String, Table)| {
                let selector = table_to_selector(ctx)?;
                let manager = manager.clone();
                let result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move {
                        kv_delete_backend(&manager, &selector, &key)
                            .await
                            .map_err(|e| e.to_string())
                    })
                });
                match result {
                    Ok(_) => Ok((Value::Boolean(true), Value::Nil)),
                    Err(err) => Ok((Value::Boolean(false), Value::String(lua.create_string(&err)?))),
                }
            })?,
        )?;
    }
    runtime_table.set("kv", runtime_kv)?;

    // runtime.db.* dynamic store operations
    let runtime_db = lua.create_table()?;
    {
        let manager = app_data.store_manager.clone();
        let app_data_snapshot = HarnessAppData {
            fs_root: app_data.fs_root.clone(),
            workspace_root: app_data.workspace_root.clone(),
            store_manager: app_data.store_manager.clone(),
            agent_manager: app_data.agent_manager.clone(),
            policy_manager: app_data.policy_manager.clone(),
            active_session_id: app_data.active_session_id.clone(),
            active_session_mode: app_data.active_session_mode.clone(),
            clients: app_data.clients.clone(),
            embedding_provider: app_data.embedding_provider.clone(),
            queue: app_data.queue.clone(),
            config: app_data.config.clone(),
            spawn_depth: app_data.spawn_depth,
        };
        runtime_db.set(
            "open",
            lua.create_function(move |lua, arg: Value| {
                let selector = selector_from_db_value(arg)?;
                let snapshot =
                    runtime_policy_snapshot(&app_data_snapshot).map_err(mlua::Error::runtime)?;
                if matches!(selector, StoreSelector::Path(_))
                    && !policy_bool(&snapshot, "db.allow_dynamic_open", true)
                {
                    return Ok((
                        Value::Nil,
                        Value::String(
                            lua.create_string("Policy denial: db.allow_dynamic_open=false")?,
                        ),
                    ));
                }

                let path_scope = StorePathScope::from_policy(policy_string(
                    &snapshot,
                    "db.path_scope",
                    "workspace_only",
                ));
                let max_open_handles =
                    policy_u64(&snapshot, "db.max_open_handles", 128).clamp(1, u64::MAX) as usize;
                let idle_close_secs = policy_u64(&snapshot, "db.idle_close_secs", 300);

                let manager = manager.clone();
                let result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move {
                        manager
                            .open_handle(&selector, path_scope, max_open_handles, idle_close_secs)
                            .await
                            .map_err(|e| e.to_string())
                    })
                });

                match result {
                    Ok(info) => {
                        let t = lua.create_table()?;
                        t.set("handle", info.handle)?;
                        t.set("path", info.path.to_string_lossy().to_string())?;
                        if let Some(alias) = info.alias {
                            t.set("alias", alias)?;
                        } else {
                            t.set("alias", Value::Nil)?;
                        }
                        t.set("open_count", info.open_count)?;
                        t.set("idle_ms", info.idle_ms)?;
                        Ok((Value::Table(t), Value::Nil))
                    }
                    Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                }
            })?,
        )?;
    }
    {
        let manager = app_data.store_manager.clone();
        runtime_db.set(
            "close",
            lua.create_function(move |lua, handle: Value| {
                let handle_id = match handle {
                    Value::String(s) => s.to_str()?.to_string(),
                    Value::Table(t) => t.get::<String>("handle")?,
                    _ => {
                        return Ok((
                            Value::Boolean(false),
                            Value::String(lua.create_string(
                                "invalid handle; expected string or {handle=...}",
                            )?),
                        ))
                    }
                };
                let manager = manager.clone();
                let result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move {
                        manager.close_handle(&handle_id).await.map_err(|e| e.to_string())
                    })
                });
                match result {
                    Ok(closed) => Ok((Value::Boolean(closed), Value::Nil)),
                    Err(err) => Ok((Value::Boolean(false), Value::String(lua.create_string(&err)?))),
                }
            })?,
        )?;
    }
    {
        let manager = app_data.store_manager.clone();
        runtime_db.set(
            "list",
            lua.create_function(move |lua, ()| {
                let manager = manager.clone();
                let handles = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move { manager.list_handles().await })
                });
                let out = lua.create_table()?;
                for (i, h) in handles.into_iter().enumerate() {
                    let t = lua.create_table()?;
                    t.set("handle", h.handle)?;
                    t.set("path", h.path.to_string_lossy().to_string())?;
                    if let Some(alias) = h.alias {
                        t.set("alias", alias)?;
                    } else {
                        t.set("alias", Value::Nil)?;
                    }
                    t.set("open_count", h.open_count)?;
                    t.set("idle_ms", h.idle_ms)?;
                    out.set(i + 1, t)?;
                }
                Ok((Value::Table(out), Value::Nil))
            })?,
        )?;
    }
    {
        let manager = app_data.store_manager.clone();
        let app_data_snapshot = HarnessAppData {
            fs_root: app_data.fs_root.clone(),
            workspace_root: app_data.workspace_root.clone(),
            store_manager: app_data.store_manager.clone(),
            agent_manager: app_data.agent_manager.clone(),
            policy_manager: app_data.policy_manager.clone(),
            active_session_id: app_data.active_session_id.clone(),
            active_session_mode: app_data.active_session_mode.clone(),
            clients: app_data.clients.clone(),
            embedding_provider: app_data.embedding_provider.clone(),
            queue: app_data.queue.clone(),
            config: app_data.config.clone(),
            spawn_depth: app_data.spawn_depth,
        };
        runtime_db.set(
            "query",
            lua.create_function(move |lua, (sql, params, opts): (String, Option<Table>, Option<Table>)| {
                let selector = selector_from_db_opts(opts)?;
                let sql_params = lua_table_to_sql_params(params)?;
                let snapshot = runtime_policy_snapshot(&app_data_snapshot).map_err(mlua::Error::runtime)?;
                if matches!(selector, StoreSelector::Path(_)) && !policy_bool(&snapshot, "db.allow_dynamic_open", true) {
                    return Ok((Value::Nil, Value::String(lua.create_string("Policy denial: db.allow_dynamic_open=false")?)));
                }
                let path_scope = StorePathScope::from_policy(policy_string(&snapshot, "db.path_scope", "workspace_only"));
                let max_open_handles = policy_u64(&snapshot, "db.max_open_handles", 128).clamp(1, u64::MAX) as usize;
                let idle_close_secs = policy_u64(&snapshot, "db.idle_close_secs", 300);
                let manager = manager.clone();
                let result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move {
                        let _ = manager.trim_cache(max_open_handles, idle_close_secs).await;
                        let store = manager.open_with_path_scope(&selector, path_scope).await.map_err(|e| e.to_string())?;
                        let conn = store.get_connection().await.map_err(|e| e.to_string())?;
                        let mut stmt = conn.prepare(&sql).await.map_err(|e| e.to_string())?;
                        let cols = stmt.columns().into_iter().map(|c| c.name().to_string()).collect::<Vec<_>>();
                        let mut rows = match sql_params {
                            SqlParams::None => stmt.query(()).await.map_err(|e| e.to_string())?,
                            SqlParams::Positional(v) => stmt.query(v).await.map_err(|e| e.to_string())?,
                            SqlParams::Named(v) => stmt.query(v).await.map_err(|e| e.to_string())?,
                        };
                        let mut out_rows = Vec::<serde_json::Value>::new();
                        while let Some(row) = rows.next().await.map_err(|e| e.to_string())? {
                            let mut obj = serde_json::Map::new();
                            for (idx, col) in cols.iter().enumerate() {
                                let v = row.get_value(idx).map_err(|e| e.to_string())?;
                                obj.insert(col.clone(), sql_value_to_json(v));
                            }
                            out_rows.push(serde_json::Value::Object(obj));
                        }
                        Ok::<_, String>(out_rows)
                    })
                });
                match result {
                    Ok(rows) => {
                        let lua_v = lua.to_value(&rows).map_err(|e| mlua::Error::runtime(e.to_string()))?;
                        Ok((lua_v, Value::Nil))
                    }
                    Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                }
            })?,
        )?;
    }
    {
        let manager = app_data.store_manager.clone();
        let app_data_snapshot = HarnessAppData {
            fs_root: app_data.fs_root.clone(),
            workspace_root: app_data.workspace_root.clone(),
            store_manager: app_data.store_manager.clone(),
            agent_manager: app_data.agent_manager.clone(),
            policy_manager: app_data.policy_manager.clone(),
            active_session_id: app_data.active_session_id.clone(),
            active_session_mode: app_data.active_session_mode.clone(),
            clients: app_data.clients.clone(),
            embedding_provider: app_data.embedding_provider.clone(),
            queue: app_data.queue.clone(),
            config: app_data.config.clone(),
            spawn_depth: app_data.spawn_depth,
        };
        runtime_db.set(
            "exec",
            lua.create_function(move |lua, (sql, params, opts): (String, Option<Table>, Option<Table>)| {
                let selector = selector_from_db_opts(opts)?;
                let sql_params = lua_table_to_sql_params(params)?;
                let snapshot = runtime_policy_snapshot(&app_data_snapshot).map_err(mlua::Error::runtime)?;
                if matches!(selector, StoreSelector::Path(_)) && !policy_bool(&snapshot, "db.allow_dynamic_open", true) {
                    return Ok((Value::Nil, Value::String(lua.create_string("Policy denial: db.allow_dynamic_open=false")?)));
                }
                let path_scope = StorePathScope::from_policy(policy_string(&snapshot, "db.path_scope", "workspace_only"));
                let max_open_handles = policy_u64(&snapshot, "db.max_open_handles", 128).clamp(1, u64::MAX) as usize;
                let idle_close_secs = policy_u64(&snapshot, "db.idle_close_secs", 300);
                let manager = manager.clone();
                let result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move {
                        let _ = manager.trim_cache(max_open_handles, idle_close_secs).await;
                        let store = manager.open_with_path_scope(&selector, path_scope).await.map_err(|e| e.to_string())?;
                        let conn = store.get_connection().await.map_err(|e| e.to_string())?;
                        let changed = match sql_params {
                            SqlParams::None => conn.execute(&sql, ()).await.map_err(|e| e.to_string())?,
                            SqlParams::Positional(v) => conn.execute(&sql, v).await.map_err(|e| e.to_string())?,
                            SqlParams::Named(v) => conn.execute(&sql, v).await.map_err(|e| e.to_string())?,
                        };
                        Ok::<_, String>(changed)
                    })
                });
                match result {
                    Ok(changed) => Ok((Value::Integer(changed as i64), Value::Nil)),
                    Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                }
            })?,
        )?;
    }
    runtime_table.set("db", runtime_db)?;

    // runtime.agent.* peer agent orchestration
    let runtime_agent = lua.create_table()?;
    {
        let manager = app_data.agent_manager.clone();
        runtime_agent.set(
            "list",
            lua.create_function(move |lua, ()| {
                let manager = manager.clone();
                let statuses = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move { manager.list_statuses().await })
                });
                let lua_v = lua
                    .to_value(&statuses)
                    .map_err(|e| mlua::Error::runtime(e.to_string()))?;
                Ok((lua_v, Value::Nil))
            })?,
        )?;
    }
    {
        let manager = app_data.agent_manager.clone();
        runtime_agent.set(
            "get_status",
            lua.create_function(move |lua, agent_id: String| {
                let manager = manager.clone();
                let status = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move { manager.get_status(&agent_id).await })
                });
                match status {
                    Some(s) => {
                        let lua_v = lua.to_value(&s).map_err(|e| mlua::Error::runtime(e.to_string()))?;
                        Ok((lua_v, Value::Nil))
                    }
                    None => Ok((Value::Nil, Value::String(lua.create_string("unknown agent")?))),
                }
            })?,
        )?;
    }
    {
        let manager = app_data.agent_manager.clone();
        let app_data_snapshot = HarnessAppData {
            fs_root: app_data.fs_root.clone(),
            workspace_root: app_data.workspace_root.clone(),
            store_manager: app_data.store_manager.clone(),
            agent_manager: app_data.agent_manager.clone(),
            policy_manager: app_data.policy_manager.clone(),
            active_session_id: app_data.active_session_id.clone(),
            active_session_mode: app_data.active_session_mode.clone(),
            clients: app_data.clients.clone(),
            embedding_provider: app_data.embedding_provider.clone(),
            queue: app_data.queue.clone(),
            config: app_data.config.clone(),
            spawn_depth: app_data.spawn_depth,
        };
        runtime_agent.set(
            "submit",
            lua.create_function(move |lua, (agent_id, task_val, _opts): (String, Value, Option<Table>)| {
                let snapshot = runtime_policy_snapshot(&app_data_snapshot).map_err(mlua::Error::runtime)?;
                if !policy_bool(&snapshot, "spawn.enabled", true) {
                    return Ok((Value::Nil, Value::String(lua.create_string("Policy denial: spawn.enabled=false")?)));
                }

                let task = match task_val {
                    Value::String(s) => QueuedTask::ad_hoc(s.to_str()?.to_string()),
                    Value::Table(t) => {
                        let prompt = t.get::<String>("prompt")
                            .map_err(|_| mlua::Error::runtime("runtime.agent.submit task table requires prompt"))?;
                        let mut task = QueuedTask::ad_hoc(prompt);
                        if let Ok(title) = t.get::<String>("title") {
                            task.title = Some(title);
                        }
                        task
                    }
                    _ => return Ok((Value::Nil, Value::String(lua.create_string("invalid task; expected string or {prompt=...}")?))),
                };

                let manager = manager.clone();
                let result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move {
                        manager.submit(&agent_id, task).await.map_err(|e| e.to_string())
                    })
                });
                match result {
                    Ok(task_id) => Ok((Value::String(lua.create_string(&task_id)?), Value::Nil)),
                    Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                }
            })?,
        )?;
    }
    {
        let manager = app_data.agent_manager.clone();
        runtime_agent.set(
            "await",
            lua.create_function(move |lua, (task_id, opts): (String, Option<Table>)| {
                let timeout_ms = opts
                    .as_ref()
                    .and_then(|t| t.get::<u64>("timeout_ms").ok());
                let manager = manager.clone();
                let result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move {
                        manager.await_result(&task_id, timeout_ms).await.map_err(|e| e.to_string())
                    })
                });
                match result {
                    Ok(res) => {
                        let lua_v = lua.to_value(&res).map_err(|e| mlua::Error::runtime(e.to_string()))?;
                        Ok((lua_v, Value::Nil))
                    }
                    Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                }
            })?,
        )?;
    }
    runtime_table.set("agent", runtime_agent)?;
    let policy_table = lua.create_table()?;
    {
        let policy_manager = app_data.policy_manager.clone();
        let app_data_snapshot = HarnessAppData {
            fs_root: app_data.fs_root.clone(),
            workspace_root: app_data.workspace_root.clone(),
            store_manager: app_data.store_manager.clone(),
            agent_manager: app_data.agent_manager.clone(),
            policy_manager: app_data.policy_manager.clone(),
            active_session_id: app_data.active_session_id.clone(),
            active_session_mode: app_data.active_session_mode.clone(),
            clients: app_data.clients.clone(),
            embedding_provider: app_data.embedding_provider.clone(),
            queue: app_data.queue.clone(),
            config: app_data.config.clone(),
            spawn_depth: app_data.spawn_depth,
        };
        policy_table.set(
            "get",
            lua.create_function(move |lua, (key, scope): (String, Option<Value>)| {
                let scope = policy_scope_from_value(&app_data_snapshot, scope)?;
                let policy_manager = policy_manager.clone();
                let result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move {
                        policy_manager
                            .get(&key, &scope)
                            .await
                            .map_err(|e| e.to_string())
                    })
                });

                match result {
                    Ok(Some(v)) => {
                        let lua_v = lua
                            .to_value(&v)
                            .map_err(|e| mlua::Error::runtime(e.to_string()))?;
                        Ok((lua_v, Value::Nil))
                    }
                    Ok(None) => Ok((Value::Nil, Value::Nil)),
                    Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                }
            })?,
        )?;
    }
    {
        let policy_manager = app_data.policy_manager.clone();
        let app_data_snapshot = HarnessAppData {
            fs_root: app_data.fs_root.clone(),
            workspace_root: app_data.workspace_root.clone(),
            store_manager: app_data.store_manager.clone(),
            agent_manager: app_data.agent_manager.clone(),
            policy_manager: app_data.policy_manager.clone(),
            active_session_id: app_data.active_session_id.clone(),
            active_session_mode: app_data.active_session_mode.clone(),
            clients: app_data.clients.clone(),
            embedding_provider: app_data.embedding_provider.clone(),
            queue: app_data.queue.clone(),
            config: app_data.config.clone(),
            spawn_depth: app_data.spawn_depth,
        };
        policy_table.set(
            "set",
            lua.create_function(move |lua, (key, value, scope): (String, Value, Option<Value>)| {
                let scope = policy_scope_from_value(&app_data_snapshot, scope)?;
                let json_value = lua
                    .from_value::<serde_json::Value>(value)
                    .map_err(|e| mlua::Error::runtime(format!("invalid policy value: {}", e)))?;
                let policy_manager = policy_manager.clone();
                let result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move {
                        policy_manager
                            .set(&key, json_value, &scope)
                            .await
                            .map_err(|e| e.to_string())
                    })
                });
                match result {
                    Ok(()) => Ok((Value::Boolean(true), Value::Nil)),
                    Err(err) => Ok((Value::Boolean(false), Value::String(lua.create_string(&err)?))),
                }
            })?,
        )?;
    }
    runtime_table.set("policy", policy_table)?;

    lua.globals().set("runtime", runtime_table)?;
    Ok(())
}

// -----------------------------------------------------------------------------
// TIER 1: MEMORY.*
// -----------------------------------------------------------------------------

fn register_memory_module(lua: &Lua, app_data: &HarnessAppData) -> LuaResult<()> {
    let memory_table = lua.create_table()?;

    // memory.search (Tier 1 -> agent default selector)
    {
        let manager = app_data.store_manager.clone();
        let embedding = app_data.embedding_provider.clone();
        let app_config = app_data.config.clone();
        let active_session = app_data.active_session_id.clone();
        memory_table.set(
            "search",
            lua.create_function(move |lua, (query, opts): (String, Option<Value>)| {
                let limit = search_limit_from_opt(opts)?;
                let selector = normalize_selector(ContextSelector {
                    tags: vec![format!("agent:{}", app_config.agent.id)],
                    namespace: "default".to_string(),
                    visibility: "private".to_string(),
                })
                .map_err(mlua::Error::runtime)?;
                // Require active session context so harness calls outside a session fail clearly.
                if active_session.lock().unwrap().is_none() {
                    return Ok((Value::Nil, Value::String(lua.create_string("No active session context")?)));
                }
                let manager = manager.clone();
                let embedding = embedding.clone();
                let result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move {
                        memory_search_backend(&manager, embedding.as_ref(), &selector, &query, limit)
                            .await
                            .map_err(|e| e.to_string())
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
                    }
                    Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                }
            })?,
        )?;
    }

    // memory.store (Tier 1 -> agent default selector)
    {
        let manager = app_data.store_manager.clone();
        let embedding = app_data.embedding_provider.clone();
        let app_config = app_data.config.clone();
        let active_session = app_data.active_session_id.clone();
        memory_table.set(
            "store",
            lua.create_function(move |lua, (content, metadata, _opts): (String, Option<Table>, Option<Table>)| {
                let selector = normalize_selector(ContextSelector {
                    tags: vec![format!("agent:{}", app_config.agent.id)],
                    namespace: "default".to_string(),
                    visibility: "private".to_string(),
                })
                .map_err(mlua::Error::runtime)?;
                if active_session.lock().unwrap().is_none() {
                    return Ok((Value::Boolean(false), Value::String(lua.create_string("No active session context")?)));
                }
                let metadata_json = if let Some(tbl) = metadata {
                    lua.from_value::<serde_json::Value>(Value::Table(tbl))
                        .map_err(|e| mlua::Error::runtime(format!("invalid metadata table: {}", e)))?
                } else {
                    serde_json::json!({})
                };
                let manager = manager.clone();
                let embedding = embedding.clone();
                let result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move {
                        memory_store_backend(&manager, embedding.as_ref(), &selector, &content, &metadata_json)
                            .await
                            .map_err(|e| e.to_string())
                    })
                });
                match result {
                    Ok(_) => Ok((Value::Boolean(true), Value::Nil)),
                    Err(err) => Ok((Value::Boolean(false), Value::String(lua.create_string(&err)?))),
                }
            })?,
        )?;
    }

    // memory.as(ctx) proxy
    {
        let manager = app_data.store_manager.clone();
        let embedding = app_data.embedding_provider.clone();
        memory_table.set(
            "as",
            lua.create_function(move |lua, ctx: Table| {
                let selector = table_to_selector(ctx)?;
                let proxy = lua.create_table()?;

                let sel_search = selector.clone();
                let m_search = manager.clone();
                let e_search = embedding.clone();
                proxy.set(
                    "search",
                    lua.create_function(move |lua, (query, opts): (String, Option<Value>)| {
                        let limit = search_limit_from_opt(opts)?;
                        let selector = sel_search.clone();
                        let manager = m_search.clone();
                        let embedding = e_search.clone();
                        let result = tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(async move {
                                memory_search_backend(&manager, embedding.as_ref(), &selector, &query, limit)
                                    .await
                                    .map_err(|e| e.to_string())
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
                            }
                            Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                        }
                    })?,
                )?;

                let sel_store = selector.clone();
                let m_store = manager.clone();
                let e_store = embedding.clone();
                proxy.set(
                    "store",
                    lua.create_function(move |lua, (content, metadata, _opts): (String, Option<Table>, Option<Table>)| {
                        let metadata_json = if let Some(tbl) = metadata {
                            lua.from_value::<serde_json::Value>(Value::Table(tbl))
                                .map_err(|e| mlua::Error::runtime(format!("invalid metadata table: {}", e)))?
                        } else {
                            serde_json::json!({})
                        };
                        let selector = sel_store.clone();
                        let manager = m_store.clone();
                        let embedding = e_store.clone();
                        let result = tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(async move {
                                memory_store_backend(&manager, embedding.as_ref(), &selector, &content, &metadata_json)
                                    .await
                                    .map_err(|e| e.to_string())
                            })
                        });
                        match result {
                            Ok(_) => Ok((Value::Boolean(true), Value::Nil)),
                            Err(err) => Ok((Value::Boolean(false), Value::String(lua.create_string(&err)?))),
                        }
                    })?,
                )?;

                Ok(proxy)
            })?,
        )?;
    }

    lua.globals().set("memory", memory_table)?;
    Ok(())
}

// -----------------------------------------------------------------------------
// TIER 1: KV.*
// -----------------------------------------------------------------------------

fn register_kv_module(lua: &Lua, app_data: &HarnessAppData) -> LuaResult<()> {
    let kv_table = lua.create_table()?;

    // kv.get
    {
        let manager = app_data.store_manager.clone();
        let app_config = app_data.config.clone();
        let active_session = app_data.active_session_id.clone();
        kv_table.set(
            "get",
            lua.create_function(move |lua, key: String| {
                if active_session.lock().unwrap().is_none() {
                    return Ok((Value::Nil, Value::String(lua.create_string("No active session context")?)));
                }
                let selector = normalize_selector(ContextSelector {
                    tags: vec![format!("agent:{}", app_config.agent.id)],
                    namespace: "default".to_string(),
                    visibility: "private".to_string(),
                })
                .map_err(mlua::Error::runtime)?;
                let manager = manager.clone();
                let result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move {
                        kv_get_backend(&manager, &selector, &key)
                            .await
                            .map_err(|e| e.to_string())
                    })
                });
                match result {
                    Ok(Some(val)) => Ok((Value::String(lua.create_string(&val)?), Value::Nil)),
                    Ok(None) => Ok((Value::Nil, Value::Nil)),
                    Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                }
            })?,
        )?;
    }

    // kv.set
    {
        let manager = app_data.store_manager.clone();
        let app_config = app_data.config.clone();
        let active_session = app_data.active_session_id.clone();
        kv_table.set(
            "set",
            lua.create_function(move |lua, (key, value): (String, String)| {
                if active_session.lock().unwrap().is_none() {
                    return Ok((Value::Boolean(false), Value::String(lua.create_string("No active session context")?)));
                }
                let selector = normalize_selector(ContextSelector {
                    tags: vec![format!("agent:{}", app_config.agent.id)],
                    namespace: "default".to_string(),
                    visibility: "private".to_string(),
                })
                .map_err(mlua::Error::runtime)?;
                let manager = manager.clone();
                let result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move {
                        kv_set_backend(&manager, &selector, &key, &value)
                            .await
                            .map_err(|e| e.to_string())
                    })
                });
                match result {
                    Ok(_) => Ok((Value::Boolean(true), Value::Nil)),
                    Err(err) => Ok((Value::Boolean(false), Value::String(lua.create_string(&err)?))),
                }
            })?,
        )?;
    }

    // kv.delete
    {
        let manager = app_data.store_manager.clone();
        let app_config = app_data.config.clone();
        let active_session = app_data.active_session_id.clone();
        kv_table.set(
            "delete",
            lua.create_function(move |lua, key: String| {
                if active_session.lock().unwrap().is_none() {
                    return Ok((Value::Boolean(false), Value::String(lua.create_string("No active session context")?)));
                }
                let selector = normalize_selector(ContextSelector {
                    tags: vec![format!("agent:{}", app_config.agent.id)],
                    namespace: "default".to_string(),
                    visibility: "private".to_string(),
                })
                .map_err(mlua::Error::runtime)?;
                let manager = manager.clone();
                let result = tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move {
                        kv_delete_backend(&manager, &selector, &key)
                            .await
                            .map_err(|e| e.to_string())
                    })
                });
                match result {
                    Ok(_) => Ok((Value::Boolean(true), Value::Nil)),
                    Err(err) => Ok((Value::Boolean(false), Value::String(lua.create_string(&err)?))),
                }
            })?,
        )?;
    }

    // kv.as(ctx) proxy
    {
        let manager = app_data.store_manager.clone();
        kv_table.set(
            "as",
            lua.create_function(move |lua, ctx: Table| {
                let selector = table_to_selector(ctx)?;
                let proxy = lua.create_table()?;

                let sel_get = selector.clone();
                let m_get = manager.clone();
                proxy.set(
                    "get",
                    lua.create_function(move |lua, key: String| {
                        let selector = sel_get.clone();
                        let manager = m_get.clone();
                        let result = tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(async move {
                                kv_get_backend(&manager, &selector, &key)
                                    .await
                                    .map_err(|e| e.to_string())
                            })
                        });
                        match result {
                            Ok(Some(val)) => Ok((Value::String(lua.create_string(&val)?), Value::Nil)),
                            Ok(None) => Ok((Value::Nil, Value::Nil)),
                            Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                        }
                    })?,
                )?;

                let sel_set = selector.clone();
                let m_set = manager.clone();
                proxy.set(
                    "set",
                    lua.create_function(move |lua, (key, value): (String, String)| {
                        let selector = sel_set.clone();
                        let manager = m_set.clone();
                        let result = tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(async move {
                                kv_set_backend(&manager, &selector, &key, &value)
                                    .await
                                    .map_err(|e| e.to_string())
                            })
                        });
                        match result {
                            Ok(_) => Ok((Value::Boolean(true), Value::Nil)),
                            Err(err) => Ok((Value::Boolean(false), Value::String(lua.create_string(&err)?))),
                        }
                    })?,
                )?;

                let sel_del = selector.clone();
                let m_del = manager.clone();
                proxy.set(
                    "delete",
                    lua.create_function(move |lua, key: String| {
                        let selector = sel_del.clone();
                        let manager = m_del.clone();
                        let result = tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(async move {
                                kv_delete_backend(&manager, &selector, &key)
                                    .await
                                    .map_err(|e| e.to_string())
                            })
                        });
                        match result {
                            Ok(_) => Ok((Value::Boolean(true), Value::Nil)),
                            Err(err) => Ok((Value::Boolean(false), Value::String(lua.create_string(&err)?))),
                        }
                    })?,
                )?;

                Ok(proxy)
            })?,
        )?;
    }

    lua.globals().set("kv", kv_table)?;
    Ok(())
}

// -----------------------------------------------------------------------------
// TIER 2: SESSION.* and USER.*
// -----------------------------------------------------------------------------

fn register_tier2_aliases(lua: &Lua, app_data: &HarnessAppData) -> LuaResult<()> {
    fn attach_alias_memory(
        lua: &Lua,
        t: &Table,
        app_data: &HarnessAppData,
        scope: &'static str,
    ) -> LuaResult<()> {
        let mem = lua.create_table()?;
        {
            let manager = app_data.store_manager.clone();
            let embedding = app_data.embedding_provider.clone();
            let active_session = app_data.active_session_id.clone();
            let config = app_data.config.clone();
            let policy_manager = app_data.policy_manager.clone();
            mem.set(
                "search",
                lua.create_function(move |lua, (query, opts): (String, Option<Value>)| {
                    let limit = search_limit_from_opt(opts)?;
                    let temp = HarnessAppData {
                        fs_root: PathBuf::new(),
                        workspace_root: PathBuf::new(),
                        store_manager: manager.clone(),
                        agent_manager: Arc::new(crate::kernel::agent_manager::AgentManager::new(config.clone(), manager.clone())),
                        policy_manager: policy_manager.clone(),
                        active_session_id: active_session.clone(),
                        active_session_mode: Arc::new(std::sync::Mutex::new(None)),
                        clients: HashMap::new(),
                        embedding_provider: embedding.clone(),
                        queue: Arc::new(tokio::sync::Mutex::new(None)),
                        config: config.clone(),
                        spawn_depth: 0,
                    };
                    let selector = selector_from_active_scope(&temp, scope)
                        .map_err(|e| mlua::Error::runtime(e.to_string()))?;
                    let manager = manager.clone();
                    let embedding = embedding.clone();
                    let result = tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async move {
                            memory_search_backend(&manager, embedding.as_ref(), &selector, &query, limit)
                                .await
                                .map_err(|e| e.to_string())
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
                        }
                        Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                    }
                })?,
            )?;
        }
        {
            let manager = app_data.store_manager.clone();
            let embedding = app_data.embedding_provider.clone();
            let active_session = app_data.active_session_id.clone();
            let config = app_data.config.clone();
            let policy_manager = app_data.policy_manager.clone();
            mem.set(
                "store",
                lua.create_function(move |lua, (content, metadata, _opts): (String, Option<Table>, Option<Table>)| {
                    let temp = HarnessAppData {
                        fs_root: PathBuf::new(),
                        workspace_root: PathBuf::new(),
                        store_manager: manager.clone(),
                        agent_manager: Arc::new(crate::kernel::agent_manager::AgentManager::new(config.clone(), manager.clone())),
                        policy_manager: policy_manager.clone(),
                        active_session_id: active_session.clone(),
                        active_session_mode: Arc::new(std::sync::Mutex::new(None)),
                        clients: HashMap::new(),
                        embedding_provider: embedding.clone(),
                        queue: Arc::new(tokio::sync::Mutex::new(None)),
                        config: config.clone(),
                        spawn_depth: 0,
                    };
                    let selector = selector_from_active_scope(&temp, scope)
                        .map_err(|e| mlua::Error::runtime(e.to_string()))?;
                    let metadata_json = if let Some(tbl) = metadata {
                        lua.from_value::<serde_json::Value>(Value::Table(tbl))
                            .map_err(|e| mlua::Error::runtime(format!("invalid metadata table: {}", e)))?
                    } else {
                        serde_json::json!({})
                    };
                    let manager = manager.clone();
                    let embedding = embedding.clone();
                    let result = tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async move {
                            memory_store_backend(&manager, embedding.as_ref(), &selector, &content, &metadata_json)
                                .await
                                .map_err(|e| e.to_string())
                        })
                    });
                    match result {
                        Ok(_) => Ok((Value::Boolean(true), Value::Nil)),
                        Err(err) => Ok((Value::Boolean(false), Value::String(lua.create_string(&err)?))),
                    }
                })?,
            )?;
        }
        t.set("memory", mem)?;
        Ok(())
    }

    fn attach_alias_kv(
        lua: &Lua,
        t: &Table,
        app_data: &HarnessAppData,
        scope: &'static str,
    ) -> LuaResult<()> {
        let kv = lua.create_table()?;
        {
            let manager = app_data.store_manager.clone();
            let active_session = app_data.active_session_id.clone();
            let config = app_data.config.clone();
            let policy_manager = app_data.policy_manager.clone();
            kv.set(
                "get",
                lua.create_function(move |lua, key: String| {
                    let temp = HarnessAppData {
                        fs_root: PathBuf::new(),
                        workspace_root: PathBuf::new(),
                        store_manager: manager.clone(),
                        agent_manager: Arc::new(crate::kernel::agent_manager::AgentManager::new(config.clone(), manager.clone())),
                        policy_manager: policy_manager.clone(),
                        active_session_id: active_session.clone(),
                        active_session_mode: Arc::new(std::sync::Mutex::new(None)),
                        clients: HashMap::new(),
                        embedding_provider: None,
                        queue: Arc::new(tokio::sync::Mutex::new(None)),
                        config: config.clone(),
                        spawn_depth: 0,
                    };
                    let selector = selector_from_active_scope(&temp, scope)
                        .map_err(|e| mlua::Error::runtime(e.to_string()))?;
                    let manager = manager.clone();
                    let result = tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async move {
                            kv_get_backend(&manager, &selector, &key).await.map_err(|e| e.to_string())
                        })
                    });
                    match result {
                        Ok(Some(val)) => Ok((Value::String(lua.create_string(&val)?), Value::Nil)),
                        Ok(None) => Ok((Value::Nil, Value::Nil)),
                        Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                    }
                })?,
            )?;
        }
        {
            let manager = app_data.store_manager.clone();
            let active_session = app_data.active_session_id.clone();
            let config = app_data.config.clone();
            let policy_manager = app_data.policy_manager.clone();
            kv.set(
                "set",
                lua.create_function(move |lua, (key, value): (String, String)| {
                    let temp = HarnessAppData {
                        fs_root: PathBuf::new(),
                        workspace_root: PathBuf::new(),
                        store_manager: manager.clone(),
                        agent_manager: Arc::new(crate::kernel::agent_manager::AgentManager::new(config.clone(), manager.clone())),
                        policy_manager: policy_manager.clone(),
                        active_session_id: active_session.clone(),
                        active_session_mode: Arc::new(std::sync::Mutex::new(None)),
                        clients: HashMap::new(),
                        embedding_provider: None,
                        queue: Arc::new(tokio::sync::Mutex::new(None)),
                        config: config.clone(),
                        spawn_depth: 0,
                    };
                    let selector = selector_from_active_scope(&temp, scope)
                        .map_err(|e| mlua::Error::runtime(e.to_string()))?;
                    let manager = manager.clone();
                    let result = tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async move {
                            kv_set_backend(&manager, &selector, &key, &value)
                                .await
                                .map_err(|e| e.to_string())
                        })
                    });
                    match result {
                        Ok(_) => Ok((Value::Boolean(true), Value::Nil)),
                        Err(err) => Ok((Value::Boolean(false), Value::String(lua.create_string(&err)?))),
                    }
                })?,
            )?;
        }
        {
            let manager = app_data.store_manager.clone();
            let active_session = app_data.active_session_id.clone();
            let config = app_data.config.clone();
            let policy_manager = app_data.policy_manager.clone();
            kv.set(
                "delete",
                lua.create_function(move |lua, key: String| {
                    let temp = HarnessAppData {
                        fs_root: PathBuf::new(),
                        workspace_root: PathBuf::new(),
                        store_manager: manager.clone(),
                        agent_manager: Arc::new(crate::kernel::agent_manager::AgentManager::new(config.clone(), manager.clone())),
                        policy_manager: policy_manager.clone(),
                        active_session_id: active_session.clone(),
                        active_session_mode: Arc::new(std::sync::Mutex::new(None)),
                        clients: HashMap::new(),
                        embedding_provider: None,
                        queue: Arc::new(tokio::sync::Mutex::new(None)),
                        config: config.clone(),
                        spawn_depth: 0,
                    };
                    let selector = selector_from_active_scope(&temp, scope)
                        .map_err(|e| mlua::Error::runtime(e.to_string()))?;
                    let manager = manager.clone();
                    let result = tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async move {
                            kv_delete_backend(&manager, &selector, &key)
                                .await
                                .map_err(|e| e.to_string())
                        })
                    });
                    match result {
                        Ok(_) => Ok((Value::Boolean(true), Value::Nil)),
                        Err(err) => Ok((Value::Boolean(false), Value::String(lua.create_string(&err)?))),
                    }
                })?,
            )?;
        }
        t.set("kv", kv)?;
        Ok(())
    }

    let session_table = lua.create_table()?;
    let user_table = lua.create_table()?;

    attach_alias_memory(lua, &session_table, app_data, "session")?;
    attach_alias_kv(lua, &session_table, app_data, "session")?;

    attach_alias_memory(lua, &user_table, app_data, "user")?;
    attach_alias_kv(lua, &user_table, app_data, "user")?;

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

    // agent.spawn (local subtask enqueue for current session queue)
    let spawn_q = app_data.queue.clone();
    let spawn_policy_snapshot = HarnessAppData {
        fs_root: app_data.fs_root.clone(),
        workspace_root: app_data.workspace_root.clone(),
        store_manager: app_data.store_manager.clone(),
        agent_manager: app_data.agent_manager.clone(),
        policy_manager: app_data.policy_manager.clone(),
        active_session_id: app_data.active_session_id.clone(),
        active_session_mode: app_data.active_session_mode.clone(),
        clients: app_data.clients.clone(),
        embedding_provider: app_data.embedding_provider.clone(),
        queue: app_data.queue.clone(),
        config: app_data.config.clone(),
        spawn_depth: app_data.spawn_depth,
    };
    let spawn_depth = app_data.spawn_depth;
    agent_table.set("spawn", lua.create_function(move |lua, (prompt, _opts): (String, Option<Table>)| {
        let snapshot = runtime_policy_snapshot(&spawn_policy_snapshot).map_err(mlua::Error::runtime)?;
        if !policy_bool(&snapshot, "spawn.enabled", true) {
            return Ok((Value::Nil, Value::String(lua.create_string("Policy denial: spawn.enabled=false")?)));
        }
        let max_depth = policy_u64(&snapshot, "spawn.max_depth", 3) as u32;
        if spawn_depth >= max_depth {
            return Ok((Value::Nil, Value::String(lua.create_string("Policy denial: spawn.max_depth exceeded")?)));
        }
        let spawn_q = spawn_q.clone();
        let enqueue_res = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                if let Some(q) = &*spawn_q.lock().await {
                    let queue_max = policy_u64(&snapshot, "queue.max_depth", 1024) as usize;
                    let mut q = q.lock().await;
                    if q.len() >= queue_max {
                        return Err(format!("Policy denial: queue.max_depth={} reached", queue_max));
                    }
                    q.push_back(QueuedTask::ad_hoc(prompt.clone()));
                    Ok(())
                } else {
                    Err("No active session queue".to_string())
                }
            })
        });
        match enqueue_res {
            Ok(()) => {
                let token = format!("q_{}", uuid::Uuid::now_v7().simple());
                Ok((Value::String(lua.create_string(&token)?), Value::Nil))
            }
            Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
        }
    })?)?;

    // agent.complete
    {
        let manager = app_data.agent_manager.clone();
        let default_agent = app_data.config.agent.id.clone();
        let complete_policy_snapshot = HarnessAppData {
            fs_root: app_data.fs_root.clone(),
            workspace_root: app_data.workspace_root.clone(),
            store_manager: app_data.store_manager.clone(),
            agent_manager: app_data.agent_manager.clone(),
            policy_manager: app_data.policy_manager.clone(),
            active_session_id: app_data.active_session_id.clone(),
            active_session_mode: app_data.active_session_mode.clone(),
            clients: app_data.clients.clone(),
            embedding_provider: app_data.embedding_provider.clone(),
            queue: app_data.queue.clone(),
            config: app_data.config.clone(),
            spawn_depth: app_data.spawn_depth,
        };
        agent_table.set("complete", lua.create_function(move |lua, (prompt, opts): (String, Option<Table>)| {
            let snapshot = runtime_policy_snapshot(&complete_policy_snapshot).map_err(mlua::Error::runtime)?;
            if !policy_bool(&snapshot, "spawn.enabled", true) {
                return Ok((Value::Nil, Value::String(lua.create_string("Policy denial: spawn.enabled=false")?)));
            }
            let target_agent = opts
                .as_ref()
                .and_then(|t| t.get::<String>("agent_id").ok())
                .unwrap_or_else(|| default_agent.clone());
            let timeout_ms = opts
                .as_ref()
                .and_then(|t| t.get::<u64>("timeout_ms").ok());

            let manager_submit = manager.clone();
            let request_id = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async move {
                    manager_submit
                        .submit(&target_agent, QueuedTask::ad_hoc(prompt))
                        .await
                        .map_err(|e| e.to_string())
                })
            });
            let request_id = match request_id {
                Ok(id) => id,
                Err(err) => return Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
            };

            let manager_await = manager.clone();
            let result = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async move {
                    manager_await
                        .await_result(&request_id, timeout_ms)
                        .await
                        .map_err(|e| e.to_string())
                })
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
        })?)?;
    }

    // agent.session.identity()
    session_ns.set("identity", lua.create_function(move |lua, ()| {
        let app_data = lua
            .app_data_ref::<HarnessAppData>()
            .ok_or_else(|| mlua::Error::runtime("missing harness app data"))?;
        let identity = get_active_identity(&app_data).map_err(mlua::Error::runtime)?;
        identity_to_lua_table(lua, &identity)
    })?)?;

    // agent.session.queue
    let aq = app_data.queue.clone();
    let queue_policy_snapshot = HarnessAppData {
        fs_root: app_data.fs_root.clone(),
        workspace_root: app_data.workspace_root.clone(),
        store_manager: app_data.store_manager.clone(),
        agent_manager: app_data.agent_manager.clone(),
        policy_manager: app_data.policy_manager.clone(),
        active_session_id: app_data.active_session_id.clone(),
        active_session_mode: app_data.active_session_mode.clone(),
        clients: app_data.clients.clone(),
        embedding_provider: app_data.embedding_provider.clone(),
        queue: app_data.queue.clone(),
        config: app_data.config.clone(),
        spawn_depth: app_data.spawn_depth,
    };
    session_ns.set("queue", lua.create_function(move |lua, cmd: String| {
        let aq = aq.clone();
        let snapshot = runtime_policy_snapshot(&queue_policy_snapshot).map_err(mlua::Error::runtime)?;
        let queue_max = policy_u64(&snapshot, "queue.max_depth", 1024) as usize;
        let res = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                if let Some(q) = &*aq.lock().await {
                    let mut q = q.lock().await;
                    if q.len() >= queue_max {
                        return Err(format!("Policy denial: queue.max_depth={} reached", queue_max));
                    }
                    q.push_back(QueuedTask::ad_hoc(cmd));
                    Ok(())
                } else {
                    Err("No active session queue".to_string())
                }
            })
        });
        match res {
            Ok(()) => Ok((Value::Boolean(true), Value::Nil)),
            Err(err) => Ok((Value::Boolean(false), Value::String(lua.create_string(&err)?))),
        }
    })?)?;

    // agent.session.queue_next
    let aq2 = app_data.queue.clone();
    let queue_next_policy_snapshot = HarnessAppData {
        fs_root: app_data.fs_root.clone(),
        workspace_root: app_data.workspace_root.clone(),
        store_manager: app_data.store_manager.clone(),
        agent_manager: app_data.agent_manager.clone(),
        policy_manager: app_data.policy_manager.clone(),
        active_session_id: app_data.active_session_id.clone(),
        active_session_mode: app_data.active_session_mode.clone(),
        clients: app_data.clients.clone(),
        embedding_provider: app_data.embedding_provider.clone(),
        queue: app_data.queue.clone(),
        config: app_data.config.clone(),
        spawn_depth: app_data.spawn_depth,
    };
    session_ns.set("queue_next", lua.create_function(move |lua, cmd: String| {
        let aq = aq2.clone();
        let snapshot = runtime_policy_snapshot(&queue_next_policy_snapshot).map_err(mlua::Error::runtime)?;
        let queue_max = policy_u64(&snapshot, "queue.max_depth", 1024) as usize;
        let res = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                if let Some(q) = &*aq.lock().await {
                    let mut q = q.lock().await;
                    if q.len() >= queue_max {
                        return Err(format!("Policy denial: queue.max_depth={} reached", queue_max));
                    }
                    q.push_front(QueuedTask::ad_hoc(cmd));
                    Ok(())
                } else {
                    Err("No active session queue".to_string())
                }
            })
        });
        match res {
            Ok(()) => Ok((Value::Boolean(true), Value::Nil)),
            Err(err) => Ok((Value::Boolean(false), Value::String(lua.create_string(&err)?))),
        }
    })?)?;

    // agent.session.queue_all
    let aq3 = app_data.queue.clone();
    let queue_all_policy_snapshot = HarnessAppData {
        fs_root: app_data.fs_root.clone(),
        workspace_root: app_data.workspace_root.clone(),
        store_manager: app_data.store_manager.clone(),
        agent_manager: app_data.agent_manager.clone(),
        policy_manager: app_data.policy_manager.clone(),
        active_session_id: app_data.active_session_id.clone(),
        active_session_mode: app_data.active_session_mode.clone(),
        clients: app_data.clients.clone(),
        embedding_provider: app_data.embedding_provider.clone(),
        queue: app_data.queue.clone(),
        config: app_data.config.clone(),
        spawn_depth: app_data.spawn_depth,
    };
    session_ns.set("queue_all", lua.create_function(move |lua, commands: Table| {
        let mut items = Vec::new();
        for v in commands.sequence_values::<String>() {
            items.push(v?);
        }
        let snapshot = runtime_policy_snapshot(&queue_all_policy_snapshot).map_err(mlua::Error::runtime)?;
        let queue_max = policy_u64(&snapshot, "queue.max_depth", 1024) as usize;
        let aq = aq3.clone();
        let res = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                if let Some(q) = &*aq.lock().await {
                    let mut q = q.lock().await;
                    if q.len().saturating_add(items.len()) > queue_max {
                        return Err(format!("Policy denial: queue.max_depth={} would be exceeded", queue_max));
                    }
                    for cmd in &items {
                        q.push_back(QueuedTask::ad_hoc(cmd.clone()));
                    }
                    Ok(())
                } else {
                    Err("No active session queue".to_string())
                }
            })
        });
        match res {
            Ok(()) => Ok((Value::Boolean(true), Value::Nil)),
            Err(err) => Ok((Value::Boolean(false), Value::String(lua.create_string(&err)?))),
        }
    })?)?;

    // agent.session.load(session_id)
    {
        let manager = app_data.store_manager.clone();
        session_ns.set("load", lua.create_function(move |lua, session_id: String| {
            let manager = manager.clone();
            let result = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async move {
                    let store = manager.get_default().await.map_err(|e| e.to_string())?;
                    let uuid = uuid::Uuid::parse_str(&session_id).map_err(|e| e.to_string())?;
                    let row = store
                        .get_session_row_by_public_id(uuid)
                        .await
                        .map_err(|e| e.to_string())?;
                    Ok::<_, String>(row)
                })
            });
            match result {
                Ok(Some(row)) => Ok((Value::Table(session_row_to_lua_table(lua, &row)?), Value::Nil)),
                Ok(None) => Ok((Value::Nil, Value::Nil)),
                Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
            }
        })?)?;
    }

    // agent.session.list(limit?, offset?)
    {
        let manager = app_data.store_manager.clone();
        session_ns.set("list", lua.create_function(move |lua, (limit, offset): (Option<usize>, Option<usize>)| {
            let limit = limit.unwrap_or(20);
            let offset = offset.unwrap_or(0);
            let manager = manager.clone();
            let result = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async move {
                    let store = manager.get_default().await.map_err(|e| e.to_string())?;
                    store
                        .list_session_rows(limit, offset)
                        .await
                        .map_err(|e| e.to_string())
                })
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
        })?)?;
    }

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
    mode_ns.set("set", lua.create_function(move |lua, m: String| {
        let mode = match m.as_str() {
            "stateful" => crate::kernel::config::AgentMode::Stateful,
            "stateless" => crate::kernel::config::AgentMode::Stateless,
            "auto" => crate::kernel::config::AgentMode::Auto,
            _ => {
                return Ok((
                    Value::Boolean(false),
                    Value::String(lua.create_string("invalid mode; expected auto|stateful|stateless")?),
                ))
            }
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
                Err(e) => Ok((Value::Nil, Value::String(lua.create_string(e.to_string())?))),
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
                    Err(e) => Ok((Value::Boolean(false), Value::String(lua.create_string(e.to_string())?))),
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
            Err(e) => Ok((Value::Nil, Value::String(lua.create_string(e.to_string())?))),
        }
    })?)?;
    json_table.set("decode", lua.create_function(|lua, s: String| {
        match serde_json::from_str::<serde_json::Value>(&s) {
            Ok(j) => Ok((lua.to_value(&j)?, Value::Nil)),
            Err(e) => Ok((Value::Nil, Value::String(lua.create_string(e.to_string())?))),
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
