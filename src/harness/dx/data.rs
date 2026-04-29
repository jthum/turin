use mlua::{Function, Lua, Result as LuaResult, Table, Value};

use crate::harness::dx::common::call_and_raise_on_err;
use crate::harness::stdlib::context_selectors::{normalize_selector, selector_to_lua_table};
use crate::kernel::identity::ContextSelector;

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

fn attach_scope_methods(
    lua: &Lua,
    target: &Table,
    op_prefix: &str,
    memory_proxy: &Table,
    kv_proxy: &Table,
) -> LuaResult<()> {
    let memory_store: Function = memory_proxy.get("store")?;
    let memory_search: Function = memory_proxy.get("search")?;
    let kv_get: Function = kv_proxy.get("get")?;
    let kv_set: Function = kv_proxy.get("set")?;
    let kv_delete: Function = kv_proxy.get("delete")?;

    {
        let memory_store = memory_store.clone();
        let op_name = format!("{op_prefix}.remember");
        target.set(
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
        let op_name = format!("{op_prefix}.recall");
        target.set(
            "recall",
            lua.create_function(move |lua, (query, opts): (String, Option<Value>)| {
                call_and_raise_on_err(lua, &memory_search, (query, opts), &op_name)
            })?,
        )?;
    }

    {
        let kv_get = kv_get.clone();
        let op_name = format!("{op_prefix}.get");
        target.set(
            "get",
            lua.create_function(move |lua, key: String| {
                call_and_raise_on_err(lua, &kv_get, key, &op_name)
            })?,
        )?;
    }

    {
        let kv_set = kv_set.clone();
        let op_name = format!("{op_prefix}.set");
        target.set(
            "set",
            lua.create_function(move |lua, (key, value): (String, String)| {
                call_and_raise_on_err(lua, &kv_set, (key, value), &op_name)
            })?,
        )?;
    }

    {
        let kv_delete = kv_delete.clone();
        let op_name = format!("{op_prefix}.del");
        target.set(
            "del",
            lua.create_function(move |lua, key: String| {
                call_and_raise_on_err(lua, &kv_delete, key, &op_name)
            })?,
        )?;
    }

    {
        let kv_get = kv_get.clone();
        let kv_set = kv_set.clone();
        let op_get = format!("{op_prefix}.incr.get");
        let op_set = format!("{op_prefix}.incr.set");
        target.set(
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

fn register_scope_helpers(lua: &Lua, scope: &str) -> LuaResult<()> {
    let globals = lua.globals();
    let scope_table: Table = globals.get(scope)?;
    let memory: Table = scope_table.get("memory")?;
    let kv: Table = scope_table.get("kv")?;

    attach_scope_methods(lua, &scope_table, scope, &memory, &kv)
}

fn scope_selector_table(
    lua: &Lua,
    kind: String,
    key_or_opts: Value,
    opts: Option<Table>,
) -> LuaResult<Table> {
    let (raw_key, opts) = match (key_or_opts, opts) {
        (Value::Nil, opts) => (None, opts),
        (Value::String(value), opts) => (Some(value.to_str()?.to_string()), opts),
        (Value::Table(table), None) => (None, Some(table)),
        (Value::Table(_), Some(_)) => {
            return Err(mlua::Error::runtime(
                "scope(kind, key, opts?) received both table key and opts".to_string(),
            ));
        }
        (other, _) => {
            return Err(mlua::Error::runtime(format!(
                "scope(kind, key, opts?) expected string key or opts table, got {:?}",
                other
            )));
        }
    };

    let tag = if kind == "global" {
        format!("global:{}", raw_key.unwrap_or_else(|| "*".to_string()))
    } else {
        let raw_key = raw_key.ok_or_else(|| {
            mlua::Error::runtime(format!("scope('{}', ...) requires a key", kind))
        })?;
        format!("{kind}:{raw_key}")
    };

    let namespace = opts
        .as_ref()
        .and_then(|table| table.get::<String>("namespace").ok())
        .unwrap_or_else(|| "default".to_string());
    let visibility = opts
        .as_ref()
        .and_then(|table| table.get::<String>("visibility").ok())
        .unwrap_or_else(|| "private".to_string());

    let selector = normalize_selector(ContextSelector {
        tags: vec![tag],
        namespace,
        visibility,
    })
    .map_err(mlua::Error::runtime)?;

    selector_to_lua_table(lua, &selector)
}

pub fn register_data_globals(lua: &Lua) -> LuaResult<()> {
    let globals = lua.globals();
    let memory: Table = globals.get("memory")?;
    let memory_store: Function = memory.get("store")?;
    let memory_search: Function = memory.get("search")?;
    let memory_as: Function = memory.get("as")?;
    let kv: Table = globals.get("kv")?;
    let kv_as: Function = kv.get("as")?;

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

    {
        let memory_as = memory_as.clone();
        let kv_as = kv_as.clone();
        globals.set(
            "scope",
            lua.create_function(
                move |lua, (kind, key_or_opts, opts): (String, Value, Option<Table>)| {
                    let selector = scope_selector_table(lua, kind, key_or_opts, opts)?;
                    let memory_proxy =
                        call_and_raise_on_err(lua, &memory_as, selector.clone(), "memory.as")?;
                    let memory_proxy = match memory_proxy {
                        Value::Table(table) => table,
                        other => {
                            return Err(mlua::Error::runtime(format!(
                                "[scope] expected table from memory.as, got {:?}",
                                other
                            )));
                        }
                    };

                    let kv_proxy = call_and_raise_on_err(lua, &kv_as, selector.clone(), "kv.as")?;
                    let kv_proxy = match kv_proxy {
                        Value::Table(table) => table,
                        other => {
                            return Err(mlua::Error::runtime(format!(
                                "[scope] expected table from kv.as, got {:?}",
                                other
                            )));
                        }
                    };

                    let proxy = lua.create_table()?;
                    attach_scope_methods(lua, &proxy, "scope", &memory_proxy, &kv_proxy)?;
                    Ok(Value::Table(proxy))
                },
            )?,
        )?;
    }

    Ok(())
}
