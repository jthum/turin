use mlua::{Function, Lua, Result as LuaResult, Table, Value};

use crate::harness::dx::common::call_and_raise_on_err;

fn parse_counter_value(value: Value) -> LuaResult<i64> {
    match value {
        Value::Nil => Ok(0),
        Value::Integer(i) => Ok(i),
        Value::Number(n) => {
            if n.is_finite() && n.fract() == 0.0 {
                let i = n as i64;
                if (i as f64) == n {
                    Ok(i)
                } else {
                    Err(mlua::Error::runtime("counter value out of i64 range"))
                }
            } else {
                Err(mlua::Error::runtime(
                    "counter value must be an integer-compatible number",
                ))
            }
        }
        Value::String(s) => s
            .to_str()?
            .trim()
            .parse::<i64>()
            .map_err(|_| mlua::Error::runtime("counter value is not a valid integer")),
        other => Err(mlua::Error::runtime(format!(
            "counter value has unsupported type: {:?}",
            other
        ))),
    }
}

fn register_scope_helpers(lua: &Lua, scope: &str) -> LuaResult<()> {
    let globals = lua.globals();
    let scope_table: Table = globals.get(scope)?;
    let memory: Table = scope_table.get("memory")?;
    let kv: Table = scope_table.get("kv")?;

    let memory_store: Function = memory.get("store")?;
    let memory_search: Function = memory.get("search")?;
    let kv_get: Function = kv.get("get")?;
    let kv_set: Function = kv.get("set")?;
    let kv_delete: Function = kv.get("delete")?;

    {
        let memory_store = memory_store.clone();
        let op_name = format!("{scope}.remember");
        scope_table.set(
            "remember",
            lua.create_function(move |lua, (content, metadata): (String, Option<Table>)| {
                call_and_raise_on_err(
                    lua,
                    &memory_store,
                    (content, metadata, Option::<Table>::None),
                    &op_name,
                )
            })?,
        )?;
    }

    {
        let memory_search = memory_search.clone();
        let op_name = format!("{scope}.recall");
        scope_table.set(
            "recall",
            lua.create_function(move |lua, (query, opts): (String, Option<Value>)| {
                call_and_raise_on_err(lua, &memory_search, (query, opts), &op_name)
            })?,
        )?;
    }

    {
        let kv_get = kv_get.clone();
        let op_name = format!("{scope}.get");
        scope_table.set(
            "get",
            lua.create_function(move |lua, key: String| {
                call_and_raise_on_err(lua, &kv_get, key, &op_name)
            })?,
        )?;
    }

    {
        let kv_set = kv_set.clone();
        let op_name = format!("{scope}.set");
        scope_table.set(
            "set",
            lua.create_function(move |lua, (key, value): (String, String)| {
                call_and_raise_on_err(lua, &kv_set, (key, value), &op_name)
            })?,
        )?;
    }

    {
        let kv_delete = kv_delete.clone();
        let op_name = format!("{scope}.del");
        scope_table.set(
            "del",
            lua.create_function(move |lua, key: String| {
                call_and_raise_on_err(lua, &kv_delete, key, &op_name)
            })?,
        )?;
    }

    {
        let kv_get = kv_get.clone();
        let kv_set = kv_set.clone();
        let op_get = format!("{scope}.incr.get");
        let op_set = format!("{scope}.incr.set");
        scope_table.set(
            "incr",
            lua.create_function(move |lua, (key, by): (String, Option<i64>)| {
                let current = call_and_raise_on_err(lua, &kv_get, key.clone(), &op_get)?;
                let base = parse_counter_value(current)?;
                let delta = by.unwrap_or(1);
                let next = base
                    .checked_add(delta)
                    .ok_or_else(|| mlua::Error::runtime("counter overflow"))?;
                let _ = call_and_raise_on_err(lua, &kv_set, (key, next.to_string()), &op_set)?;
                Ok(Value::Integer(next))
            })?,
        )?;
    }

    Ok(())
}

pub fn register_data_globals(lua: &Lua) -> LuaResult<()> {
    let globals = lua.globals();
    let memory: Table = globals.get("memory")?;
    let memory_store: Function = memory.get("store")?;
    let memory_search: Function = memory.get("search")?;

    {
        let memory_store = memory_store.clone();
        globals.set(
            "remember",
            lua.create_function(
                move |lua, (content, metadata, opts): (String, Option<Table>, Option<Table>)| {
                    call_and_raise_on_err(lua, &memory_store, (content, metadata, opts), "remember")
                },
            )?,
        )?;
    }

    {
        let memory_search = memory_search.clone();
        globals.set(
            "recall",
            lua.create_function(move |lua, (query, opts): (String, Option<Value>)| {
                call_and_raise_on_err(lua, &memory_search, (query, opts), "recall")
            })?,
        )?;
    }

    register_scope_helpers(lua, "session")?;
    register_scope_helpers(lua, "user")?;
    Ok(())
}
