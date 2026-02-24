use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};
use serde::Serialize;

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

pub fn memory_rows_to_lua_table(
    lua: &Lua,
    rows: Vec<crate::persistence::schema::MemoryRow>,
) -> LuaResult<Table> {
    let tbl = lua.create_table()?;
    for (i, row) in rows.into_iter().enumerate() {
        let rt = lua.create_table()?;
        rt.set("content", row.content)?;
        rt.set("score", row.score)?;
        tbl.set(i + 1, rt)?;
    }
    Ok(tbl)
}
