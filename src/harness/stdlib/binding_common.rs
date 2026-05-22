use std::collections::BTreeMap;
use std::fmt::Display;
use std::future::Future;

use mlua::{Function, Lua, LuaSerdeExt, MultiValue, Result as LuaResult, Table, Value};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use uuid::Uuid;

use crate::harness::globals::HarnessAppData;
use crate::harness::globals::block_on_current;
use crate::harness::stdlib::db_support::{
    selector_denied_by_dynamic_open, store_path_scope_from_snapshot, store_selector_from_fields,
};
use crate::harness::stdlib::object_refs;
use crate::harness::stdlib::policy_support::runtime_policy_snapshot;
use crate::harness::stdlib::scoped_data_backend::{
    MemoryFeedbackRequest, MemoryFeedbackSignal, MemoryPurgeRequest, MemorySearchMode,
    MemorySearchRequest, MemorySearchSource, MemoryStoreMode, MemoryStoreRequest, encode_scope_key,
    selector_scope_ref,
};
use crate::kernel::identity::ContextSelector;
use crate::kernel::session_refs::parse_session_reference;
use crate::persistence::manager::{StorePathScope, StoreSelector};

#[derive(Clone, Default)]
pub struct RegisteredCallbackContext {
    pub harness_module: Option<String>,
    pub harness_root: Option<String>,
    pub import_capabilities: Option<BTreeMap<String, bool>>,
}

pub fn bridge_async<F>(fut: F) -> F::Output
where
    F: Future,
{
    block_on_current(fut)
}

pub fn bridge_async_result<F, T>(fut: F) -> Result<T, String>
where
    F: Future<Output = Result<T, String>>,
{
    block_on_current(fut)
}

pub fn bridge_async_display_err<F, T, E>(fut: F) -> Result<T, String>
where
    F: Future<Output = Result<T, E>>,
    E: Display,
{
    block_on_current(async move { fut.await.map_err(|e| e.to_string()) })
}

pub fn bridge_async_lua<F, T, E>(fut: F) -> LuaResult<T>
where
    F: Future<Output = Result<T, E>>,
    E: Display,
{
    bridge_async_display_err(fut).map_err(mlua::Error::runtime)
}

pub fn bridge_async_anyhow<F, T, E>(fut: F) -> anyhow::Result<T>
where
    F: Future<Output = Result<T, E>>,
    E: Display,
{
    bridge_async_display_err(fut).map_err(anyhow::Error::msg)
}

pub fn ok_bool() -> (Value, Value) {
    (Value::Boolean(true), Value::Nil)
}

pub fn bool_value_ok(value: bool) -> (Value, Value) {
    (Value::Boolean(value), Value::Nil)
}

pub fn ok_value(value: Value) -> (Value, Value) {
    (value, Value::Nil)
}

pub fn nil_ok() -> (Value, Value) {
    (Value::Nil, Value::Nil)
}

pub fn string_ok(lua: &Lua, value: &str) -> LuaResult<(Value, Value)> {
    Ok((Value::String(lua.create_string(value)?), Value::Nil))
}

pub fn string_value(lua: &Lua, value: &str) -> LuaResult<Value> {
    Ok(Value::String(lua.create_string(value)?))
}

pub fn nil_err(lua: &Lua, err: &str) -> LuaResult<(Value, Value)> {
    Ok((Value::Nil, Value::String(lua.create_string(err)?)))
}

pub fn bool_err(lua: &Lua, err: &str) -> LuaResult<(Value, Value)> {
    Ok((
        Value::Boolean(false),
        Value::String(lua.create_string(err)?),
    ))
}

pub fn json_ok<T>(lua: &Lua, value: &T) -> LuaResult<(Value, Value)>
where
    T: Serialize + ?Sized,
{
    let lua_v = lua
        .to_value(value)
        .map_err(|e| mlua::Error::runtime(e.to_string()))?;
    Ok((lua_v, Value::Nil))
}

pub fn lua_table_result<T, F>(
    lua: &Lua,
    result: Result<T, String>,
    to_table: F,
) -> LuaResult<(Value, Value)>
where
    F: FnOnce(&Lua, T) -> LuaResult<Table>,
{
    match result {
        Ok(value) => Ok(ok_value(Value::Table(to_table(lua, value)?))),
        Err(err) => nil_err(lua, &err),
    }
}

pub fn lua_value_result<T, F>(
    lua: &Lua,
    result: Result<T, String>,
    to_value: F,
) -> LuaResult<(Value, Value)>
where
    F: FnOnce(&Lua, T) -> LuaResult<Value>,
{
    match result {
        Ok(value) => Ok(ok_value(to_value(lua, value)?)),
        Err(err) => nil_err(lua, &err),
    }
}

pub fn metadata_json_or_empty(lua: &Lua, metadata: Option<Table>) -> LuaResult<serde_json::Value> {
    if let Some(tbl) = metadata {
        lua.from_value::<serde_json::Value>(Value::Table(tbl))
            .map_err(|e| mlua::Error::runtime(format!("invalid metadata table: {}", e)))
    } else {
        Ok(serde_json::json!({}))
    }
}

pub fn optional_lua_json(lua: &Lua, value: Value) -> LuaResult<Option<serde_json::Value>> {
    match value {
        Value::Nil => Ok(None),
        value => object_refs::encode_lua_payload(lua, value)
            .map(Some)
            .map_err(mlua::Error::runtime),
    }
}

pub fn optional_lua_object_json(
    lua: &Lua,
    value: Option<Value>,
    context: &str,
) -> LuaResult<serde_json::Map<String, serde_json::Value>> {
    match value {
        None | Some(Value::Nil) => Ok(serde_json::Map::new()),
        Some(Value::Table(table)) => {
            match object_refs::encode_lua_payload(lua, Value::Table(table))? {
                serde_json::Value::Object(map) => Ok(map),
                _ => Err(mlua::Error::runtime(format!(
                    "{} must be an object-like table",
                    context
                ))),
            }
        }
        Some(other) => Err(mlua::Error::runtime(format!(
            "{} must be an object-like table, got {:?}",
            context, other
        ))),
    }
}

pub fn parse_lua_table<T>(lua: &Lua, table: &Table, context: &str) -> LuaResult<T>
where
    T: DeserializeOwned,
{
    lua.from_value::<T>(Value::Table(table.clone()))
        .map_err(|err| mlua::Error::runtime(format!("invalid {}: {}", context, err)))
}

pub fn parse_optional_lua_table<T>(lua: &Lua, table: Option<&Table>, context: &str) -> LuaResult<T>
where
    T: DeserializeOwned + Default,
{
    match table {
        Some(table) => parse_lua_table(lua, table, context),
        None => Ok(T::default()),
    }
}

pub fn active_registered_callback_context(lua: &Lua) -> RegisteredCallbackContext {
    lua.app_data_ref::<HarnessAppData>()
        .and_then(|app_data| {
            app_data
                .execution_ctx
                .lock()
                .ok()
                .map(|ctx| RegisteredCallbackContext {
                    harness_module: ctx.harness_module.clone(),
                    harness_root: ctx.harness_root.clone(),
                    import_capabilities: ctx.import_capabilities.clone(),
                })
        })
        .unwrap_or_default()
}

pub fn wrap_registered_callback(lua: &Lua, func: Function) -> LuaResult<Function> {
    let registered_ctx = active_registered_callback_context(lua);
    lua.create_function(move |lua, args: MultiValue| {
        let prev_ctx = active_registered_callback_context(lua);
        set_active_registered_callback_context(lua, &registered_ctx);
        let result = func.call::<MultiValue>(args);
        set_active_registered_callback_context(lua, &prev_ctx);
        result
    })
}

fn set_active_registered_callback_context(lua: &Lua, ctx: &RegisteredCallbackContext) {
    if let Some(app_data) = lua.app_data_ref::<HarnessAppData>()
        && let Ok(mut lock) = app_data.execution_ctx.lock()
    {
        lock.harness_module = ctx.harness_module.clone();
        lock.harness_root = ctx.harness_root.clone();
        lock.import_capabilities = ctx.import_capabilities.clone();
    }
}

#[derive(Debug, Default, Deserialize)]
struct LuaMemorySearchOpts {
    limit: Option<i64>,
    mode: Option<String>,
    min_score: Option<f64>,
    include_metadata: Option<bool>,
    include_superseded: Option<bool>,
    strict: Option<bool>,
    trace: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
struct LuaMemoryStoreOpts {
    source_task: Option<String>,
    tags: Option<Vec<String>>,
    storage: Option<String>,
    trace: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
struct LuaMemoryFeedbackOpts {
    reason: Option<String>,
    task_id: Option<String>,
    step: Option<f64>,
    clamp: Option<LuaMemoryFeedbackClamp>,
    trace: Option<bool>,
}

#[derive(Debug, Default, Deserialize)]
struct LuaMemoryFeedbackClamp {
    min: Option<f64>,
    max: Option<f64>,
}

#[derive(Debug, Default, Deserialize)]
struct LuaMemoryPurgeOpts {
    older_than_days: Option<u64>,
    min_weight: Option<f64>,
    max_retrieval_count: Option<u64>,
    only_superseded: Option<bool>,
    all: Option<bool>,
    dry_run: Option<bool>,
    trace: Option<bool>,
}

fn parse_memory_search_source(source: Value) -> LuaResult<MemorySearchSource> {
    let table = match source {
        Value::Table(table) => table,
        _ => {
            return Err(mlua::Error::runtime(
                "invalid memory search source; expected a table",
            ));
        }
    };
    let scope_kind = table
        .get::<String>("scope_kind")
        .map_err(|_| mlua::Error::runtime("memory search source requires 'scope_kind'"))?;
    let namespace = match table.get::<Value>("namespace")? {
        Value::Nil => "default".to_string(),
        Value::String(value) => value.to_str()?.to_string(),
        _ => {
            return Err(mlua::Error::runtime(
                "invalid memory search source namespace; expected string",
            ));
        }
    };
    let raw_scope_key = match table.get::<Value>("scope_key")? {
        Value::Nil if scope_kind == "global" => "*".to_string(),
        Value::Nil => {
            return Err(mlua::Error::runtime(
                "memory search source requires 'scope_key' unless scope_kind='global'",
            ));
        }
        Value::String(value) => value.to_str()?.to_string(),
        _ => {
            return Err(mlua::Error::runtime(
                "invalid memory search source scope_key; expected string",
            ));
        }
    };
    let store_selector = store_selector_from_fields(&table)?;
    Ok(MemorySearchSource {
        scope_kind,
        scope_key: encode_scope_key(&raw_scope_key, &namespace),
        raw_scope_key,
        namespace,
        store_selector,
    })
}

fn memory_search_sources_from_table(opts: &Table) -> LuaResult<Vec<MemorySearchSource>> {
    match opts.get::<Value>("sources")? {
        Value::Nil => Ok(Vec::new()),
        Value::Table(values) => {
            let mut out = Vec::new();
            for source in values.sequence_values::<Value>() {
                out.push(parse_memory_search_source(source?)?);
            }
            Ok(out)
        }
        _ => Err(mlua::Error::runtime(
            "invalid memory search opts: sources must be an array of tables",
        )),
    }
}

pub(crate) fn memory_search_request_from_opt(
    lua: &Lua,
    arg: Option<Value>,
) -> LuaResult<MemorySearchRequest> {
    match arg {
        None | Some(Value::Nil) => Ok(MemorySearchRequest::default()),
        Some(Value::Integer(i)) => Ok(MemorySearchRequest {
            limit: i.max(0) as usize,
            ..MemorySearchRequest::default()
        }),
        Some(Value::Number(n)) => Ok(MemorySearchRequest {
            limit: n.max(0.0) as usize,
            ..MemorySearchRequest::default()
        }),
        Some(Value::Table(t)) => {
            let parsed = parse_lua_table::<LuaMemorySearchOpts>(lua, &t, "memory search opts")?;
            let _ = parsed.trace;
            let store_selector = store_selector_from_fields(&t)?;
            let sources = memory_search_sources_from_table(&t)?;
            Ok(MemorySearchRequest {
                limit: parsed.limit.unwrap_or(5).max(0) as usize,
                mode: parse_memory_search_mode(parsed.mode.as_deref())?,
                min_score: parsed.min_score.unwrap_or(0.0),
                include_metadata: parsed.include_metadata.unwrap_or(false),
                include_superseded: parsed.include_superseded.unwrap_or(false),
                strict: parsed.strict.unwrap_or(false),
                store_selector,
                sources,
            })
        }
        Some(_) => Err(mlua::Error::runtime(
            "invalid opts; expected number limit or options table",
        )),
    }
}

pub(crate) fn memory_store_request_from_opts(
    lua: &Lua,
    opts: Option<Table>,
) -> LuaResult<MemoryStoreRequest> {
    match opts {
        None => Ok(MemoryStoreRequest::default()),
        Some(t) => {
            let parsed = parse_lua_table::<LuaMemoryStoreOpts>(lua, &t, "memory store opts")?;
            let _ = parsed.trace;
            let store_selector = store_selector_from_fields(&t)?;
            Ok(MemoryStoreRequest {
                source_task: parsed.source_task,
                tags: parsed.tags.unwrap_or_default(),
                storage: parse_memory_store_mode(parsed.storage.as_deref())?,
                store_selector,
            })
        }
    }
}

pub(crate) fn memory_store_row_to_lua_value(
    lua: &Lua,
    row: crate::persistence::schema::StoredMemoryRow,
) -> LuaResult<Value> {
    let tbl = lua.create_table()?;
    tbl.set("id", public_id_to_simple_string(&row.public_id)?)?;
    tbl.set("stored_at", row.stored_at)?;
    tbl.set("storage", row.storage.as_str())?;
    Ok(Value::Table(tbl))
}

pub(crate) fn memory_feedback_signal_from_value(signal: Value) -> LuaResult<MemoryFeedbackSignal> {
    match signal {
        Value::String(s) => match s.to_str()?.as_ref() {
            "up" => Ok(MemoryFeedbackSignal::Up),
            "down" => Ok(MemoryFeedbackSignal::Down),
            other => Err(mlua::Error::runtime(format!(
                "invalid memory feedback signal: {}",
                other
            ))),
        },
        Value::Integer(i) => Ok(MemoryFeedbackSignal::Delta(i as f64)),
        Value::Number(n) => Ok(MemoryFeedbackSignal::Delta(n)),
        _ => Err(mlua::Error::runtime(
            "invalid memory feedback signal; expected \"up\", \"down\", or numeric delta",
        )),
    }
}

pub(crate) fn memory_feedback_request_from_opts(
    lua: &Lua,
    opts: Option<Table>,
) -> LuaResult<MemoryFeedbackRequest> {
    match opts {
        None => Ok(MemoryFeedbackRequest::default()),
        Some(t) => {
            let parsed = parse_lua_table::<LuaMemoryFeedbackOpts>(lua, &t, "memory feedback opts")?;
            let _ = parsed.trace;
            let store_selector = store_selector_from_fields(&t)?;
            let clamp = parsed.clamp.unwrap_or_default();
            Ok(MemoryFeedbackRequest {
                reason: parsed.reason,
                task_id: parsed.task_id,
                step: parsed.step.unwrap_or(0.1),
                clamp_min: clamp.min.unwrap_or(0.1),
                clamp_max: clamp.max.unwrap_or(5.0),
                store_selector,
            })
        }
    }
}

pub(crate) fn memory_feedback_state_to_lua_value(
    lua: &Lua,
    state: crate::persistence::schema::MemoryFeedbackState,
) -> LuaResult<Value> {
    let tbl = lua.create_table()?;
    tbl.set("id", public_id_to_simple_string(&state.public_id)?)?;
    tbl.set("weight", state.weight)?;
    tbl.set("updated_at", state.updated_at)?;
    Ok(Value::Table(tbl))
}

pub(crate) fn memory_correction_row_to_lua_value(
    lua: &Lua,
    row: crate::persistence::schema::MemoryCorrectionRow,
) -> LuaResult<Value> {
    let tbl = lua.create_table()?;
    tbl.set(
        "superseded_id",
        public_id_to_simple_string(&row.superseded_public_id)?,
    )?;
    tbl.set(
        "replacement_id",
        public_id_to_simple_string(&row.replacement_public_id)?,
    )?;
    tbl.set("corrected_at", row.corrected_at)?;
    Ok(Value::Table(tbl))
}

pub(crate) fn memory_purge_request_from_opts(
    lua: &Lua,
    opts: Option<Table>,
) -> LuaResult<MemoryPurgeRequest> {
    match opts {
        None => Ok(MemoryPurgeRequest::default()),
        Some(t) => {
            let parsed = parse_lua_table::<LuaMemoryPurgeOpts>(lua, &t, "memory purge opts")?;
            let _ = parsed.trace;
            let store_selector = store_selector_from_fields(&t)?;
            Ok(MemoryPurgeRequest {
                older_than_days: parsed.older_than_days,
                min_weight: parsed.min_weight,
                max_retrieval_count: parsed.max_retrieval_count,
                only_superseded: parsed.only_superseded.unwrap_or(false),
                all: parsed.all.unwrap_or(false),
                dry_run: parsed.dry_run.unwrap_or(true),
                store_selector,
            })
        }
    }
}

pub(crate) fn scoped_state_path_scope(
    app_data: &HarnessAppData,
    selector: Option<&StoreSelector>,
) -> LuaResult<StorePathScope> {
    scoped_state_path_scope_for_selectors(app_data, selector)
}

pub(crate) fn scoped_state_path_scope_for_selectors<'a>(
    app_data: &HarnessAppData,
    selectors: impl IntoIterator<Item = &'a StoreSelector>,
) -> LuaResult<StorePathScope> {
    let snapshot = runtime_policy_snapshot(app_data).map_err(mlua::Error::runtime)?;
    for selector in selectors {
        if selector_denied_by_dynamic_open(&snapshot, selector) {
            return Err(mlua::Error::runtime(
                "Policy denial: db.allow_dynamic_open=false",
            ));
        }
    }
    Ok(store_path_scope_from_snapshot(&snapshot))
}

pub(crate) fn resolve_scoped_store_selector(
    app_data: &HarnessAppData,
    selector: &ContextSelector,
    explicit: Option<StoreSelector>,
) -> LuaResult<Option<StoreSelector>> {
    let scope = selector_scope_ref(selector).map_err(mlua::Error::runtime)?;
    resolve_contextual_store_selector(
        app_data,
        &scope.scope_kind,
        scope.raw_scope_key.as_deref(),
        &scope.namespace,
        explicit,
    )
}

pub(crate) fn resolve_contextual_store_selector(
    app_data: &HarnessAppData,
    scope_kind: &str,
    raw_scope_key: Option<&str>,
    namespace: &str,
    explicit: Option<StoreSelector>,
) -> LuaResult<Option<StoreSelector>> {
    if explicit.is_some() {
        return Ok(explicit);
    }
    if scope_kind == "session"
        && let Some(raw_scope_key) = raw_scope_key
        && let Ok(lock) = app_data.execution_ctx.lock()
        && lock
            .session_id
            .as_deref()
            .and_then(|session_id| parse_session_reference(session_id).ok())
            .is_some_and(|session_ref| session_ref.public_id == raw_scope_key)
        && let Some(selector) = lock.session_store_selector.clone()
    {
        return Ok(Some(selector));
    }
    if let Ok(lock) = app_data.execution_ctx.lock()
        && let Some(selector) = lock.default_store_selector.clone()
    {
        return Ok(Some(selector));
    }
    Ok(resolve_scope_store_selector(
        &app_data.config,
        scope_kind,
        raw_scope_key,
        namespace,
    ))
}

pub(crate) fn resolve_scope_store_selector(
    config: &crate::kernel::config::TurinConfig,
    scope_kind: &str,
    raw_scope_key: Option<&str>,
    namespace: &str,
) -> Option<StoreSelector> {
    config
        .persistence
        .resolve_store_selector_for_scope(scope_kind, raw_scope_key, namespace)
}

pub(crate) fn resolve_memory_search_request(
    app_data: &HarnessAppData,
    selector: &ContextSelector,
    request: &MemorySearchRequest,
) -> LuaResult<MemorySearchRequest> {
    let mut resolved = request.clone();
    if resolved.sources.is_empty() {
        resolved.store_selector =
            resolve_scoped_store_selector(app_data, selector, resolved.store_selector.clone())?;
    } else {
        let common_store_selector = resolved.store_selector.clone();
        for source in &mut resolved.sources {
            if source.store_selector.is_none() {
                source.store_selector = common_store_selector.clone().or_else(|| {
                    resolve_contextual_store_selector(
                        app_data,
                        &source.scope_kind,
                        Some(source.raw_scope_key.as_str()),
                        &source.namespace,
                        None,
                    )
                    .ok()
                    .flatten()
                });
            }
        }
        resolved.store_selector = None;
    }
    Ok(resolved)
}

pub(crate) fn store_selector_from_opts_table(
    opts: Option<Table>,
) -> LuaResult<Option<StoreSelector>> {
    match opts {
        Some(table) => store_selector_from_fields(&table),
        None => Ok(None),
    }
}

pub(crate) fn memory_purge_report_to_lua_value(
    lua: &Lua,
    report: crate::persistence::schema::MemoryPurgeReport,
) -> LuaResult<Value> {
    let tbl = lua.create_table()?;
    tbl.set("matched", report.matched)?;
    tbl.set("deleted", report.deleted)?;
    tbl.set("dry_run", report.dry_run)?;
    Ok(Value::Table(tbl))
}

pub fn memory_rows_to_lua_table(
    lua: &Lua,
    rows: Vec<crate::persistence::schema::MemoryRow>,
) -> LuaResult<Table> {
    let tbl = lua.create_table()?;
    for (i, row) in rows.into_iter().enumerate() {
        let rt = lua.create_table()?;
        rt.set("id", public_id_to_simple_string(&row.public_id)?)?;
        rt.set("content", row.content)?;
        rt.set("score", row.score)?;
        if let Some(lexical_score) = row.lexical_score {
            rt.set("lexical_score", lexical_score)?;
        }
        if let Some(semantic_score) = row.semantic_score {
            rt.set("semantic_score", semantic_score)?;
        }
        rt.set("weight", row.weight)?;
        rt.set("retrieval_count", row.retrieval_count)?;
        if let Some(last_retrieved_at) = row.last_retrieved_at {
            rt.set("last_retrieved_at", last_retrieved_at)?;
        }
        if let Some(metadata) = row.metadata {
            let parsed: serde_json::Value = serde_json::from_str(&metadata)
                .map_err(|e| mlua::Error::runtime(format!("invalid memory metadata: {}", e)))?;
            rt.set("metadata", lua.to_value(&parsed)?)?;
        }
        tbl.set(i + 1, rt)?;
    }
    Ok(tbl)
}

fn public_id_to_simple_string(bytes: &[u8]) -> LuaResult<String> {
    Uuid::from_slice(bytes)
        .map(|id| id.simple().to_string())
        .map_err(|e| mlua::Error::runtime(format!("invalid memory public id: {}", e)))
}

fn parse_memory_store_mode(value: Option<&str>) -> LuaResult<MemoryStoreMode> {
    match value.unwrap_or("auto") {
        "auto" => Ok(MemoryStoreMode::Auto),
        "lexical_only" => Ok(MemoryStoreMode::LexicalOnly),
        "embedded" => Ok(MemoryStoreMode::Embedded),
        other => Err(mlua::Error::runtime(format!(
            "invalid memory storage mode: {}",
            other
        ))),
    }
}

fn parse_memory_search_mode(value: Option<&str>) -> LuaResult<MemorySearchMode> {
    match value.unwrap_or("auto") {
        "auto" => Ok(MemorySearchMode::Auto),
        "lexical" => Ok(MemorySearchMode::Lexical),
        "semantic" => Ok(MemorySearchMode::Semantic),
        "hybrid" => Ok(MemorySearchMode::Hybrid),
        other => Err(mlua::Error::runtime(format!(
            "invalid memory search mode: {}",
            other
        ))),
    }
}
