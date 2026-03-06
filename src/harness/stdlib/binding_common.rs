use std::fmt::Display;
use std::future::Future;

use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::harness::globals::block_on_current;
use crate::harness::stdlib::scoped_data_backend::{
    MemorySearchMode, MemorySearchRequest, MemoryStoreMode, MemoryStoreRequest,
};

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

pub fn metadata_json_or_empty(lua: &Lua, metadata: Option<Table>) -> LuaResult<serde_json::Value> {
    if let Some(tbl) = metadata {
        lua.from_value::<serde_json::Value>(Value::Table(tbl))
            .map_err(|e| mlua::Error::runtime(format!("invalid metadata table: {}", e)))
    } else {
        Ok(serde_json::json!({}))
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
            let parsed = lua
                .from_value::<LuaMemorySearchOpts>(Value::Table(t))
                .map_err(|e| mlua::Error::runtime(format!("invalid memory search opts: {}", e)))?;
            let _ = parsed.trace;
            Ok(MemorySearchRequest {
                limit: parsed.limit.unwrap_or(5).max(0) as usize,
                mode: parse_memory_search_mode(parsed.mode.as_deref())?,
                min_score: parsed.min_score.unwrap_or(0.0),
                include_metadata: parsed.include_metadata.unwrap_or(false),
                include_superseded: parsed.include_superseded.unwrap_or(false),
                strict: parsed.strict.unwrap_or(false),
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
            let parsed = lua
                .from_value::<LuaMemoryStoreOpts>(Value::Table(t))
                .map_err(|e| mlua::Error::runtime(format!("invalid memory store opts: {}", e)))?;
            let _ = parsed.trace;
            Ok(MemoryStoreRequest {
                source_task: parsed.source_task,
                tags: parsed.tags.unwrap_or_default(),
                storage: parse_memory_store_mode(parsed.storage.as_deref())?,
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
