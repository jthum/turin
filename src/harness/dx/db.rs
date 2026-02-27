use mlua::{Function, Lua, MultiValue, Result as LuaResult, Table, Value};

use crate::harness::dx::common::call_and_raise_on_err;

fn is_path_like(selector: &str) -> bool {
    selector.contains('/')
        || selector.contains('\\')
        || selector.starts_with('.')
        || selector.ends_with(".db")
        || selector.starts_with('~')
}

fn selector_to_open_arg(lua: &Lua, selector: Value) -> LuaResult<Value> {
    match selector {
        Value::Nil => {
            let table = lua.create_table()?;
            table.set("store", "state")?;
            Ok(Value::Table(table))
        }
        Value::String(s) => {
            let s = s.to_str()?.to_string();
            let table = lua.create_table()?;
            if is_path_like(&s) {
                table.set("path", s)?;
            } else {
                table.set("store", s)?;
            }
            Ok(Value::Table(table))
        }
        Value::Table(t) => Ok(Value::Table(t)),
        _ => Err(mlua::Error::runtime(
            "runtime.db(...) expects selector string/table",
        )),
    }
}

fn open_handle_id(lua: &Lua, open_fn: &Function, selector: Value) -> LuaResult<String> {
    let arg = selector_to_open_arg(lua, selector)?;
    let opened = call_and_raise_on_err(lua, open_fn, arg, "runtime.db.open")?;
    let table = match opened {
        Value::Table(t) => t,
        other => {
            return Err(mlua::Error::runtime(format!(
                "[runtime.db.open] expected table result, got {:?}",
                other
            )));
        }
    };
    table.get::<String>("handle")
}

fn close_handle_checked(
    lua: &Lua,
    close_fn: &Function,
    handle: &str,
    op_name: &str,
) -> LuaResult<Value> {
    let value = call_and_raise_on_err(lua, close_fn, handle.to_string(), op_name)?;
    match value {
        Value::Boolean(true) => Ok(Value::Boolean(true)),
        Value::Boolean(false) => Err(mlua::Error::runtime(format!(
            "[{}] close returned false for handle '{}'",
            op_name, handle
        ))),
        other => Ok(other),
    }
}

fn create_handle_opts(lua: &Lua, handle: &str) -> LuaResult<Table> {
    let opts = lua.create_table()?;
    opts.set("handle", handle)?;
    Ok(opts)
}

fn create_db_proxy(
    lua: &Lua,
    handle: String,
    query_fn: Function,
    exec_fn: Function,
    close_fn: Function,
) -> LuaResult<Table> {
    let proxy = lua.create_table()?;

    {
        let query_fn = query_fn.clone();
        let handle = handle.clone();
        proxy.set(
            "all",
            lua.create_function(
                move |lua,
                      (_self, sql, params, _opts): (
                    Table,
                    String,
                    Option<Table>,
                    Option<Table>,
                )| {
                    let opts = create_handle_opts(lua, &handle)?;
                    call_and_raise_on_err(
                        lua,
                        &query_fn,
                        (sql, params, Some(opts)),
                        "runtime.db.query",
                    )
                },
            )?,
        )?;
    }

    {
        let query_fn = query_fn.clone();
        let handle = handle.clone();
        proxy.set(
            "one",
            lua.create_function(
                move |lua,
                      (_self, sql, params, _opts): (
                    Table,
                    String,
                    Option<Table>,
                    Option<Table>,
                )| {
                    let opts = create_handle_opts(lua, &handle)?;
                    let rows = call_and_raise_on_err(
                        lua,
                        &query_fn,
                        (sql, params, Some(opts)),
                        "runtime.db.query",
                    )?;
                    match rows {
                        Value::Nil => Ok(Value::Nil),
                        Value::Table(rows_table) => rows_table.get::<Value>(1),
                        other => Err(mlua::Error::runtime(format!(
                            "[runtime.db.one] expected table rows, got {:?}",
                            other
                        ))),
                    }
                },
            )?,
        )?;
    }

    {
        let exec_fn = exec_fn.clone();
        let handle = handle.clone();
        proxy.set(
            "exec",
            lua.create_function(
                move |lua,
                      (_self, sql, params, _opts): (
                    Table,
                    String,
                    Option<Table>,
                    Option<Table>,
                )| {
                    let opts = create_handle_opts(lua, &handle)?;
                    call_and_raise_on_err(
                        lua,
                        &exec_fn,
                        (sql, params, Some(opts)),
                        "runtime.db.exec",
                    )
                },
            )?,
        )?;
    }

    {
        let close_fn = close_fn.clone();
        let handle = handle.clone();
        proxy.set(
            "close",
            lua.create_function(move |lua, _self: Table| {
                close_handle_checked(lua, &close_fn, &handle, "runtime.db.close")
            })?,
        )?;
    }

    Ok(proxy)
}

pub fn register_db_dx(lua: &Lua) -> LuaResult<()> {
    let globals = lua.globals();
    let runtime: Table = globals.get("runtime")?;
    let runtime_db: Table = runtime.get("db")?;

    let open_fn: Function = runtime_db.get("open")?;
    let query_fn: Function = runtime_db.get("query")?;
    let exec_fn: Function = runtime_db.get("exec")?;
    let close_fn: Function = runtime_db.get("close")?;

    {
        let open_fn = open_fn.clone();
        let query_fn = query_fn.clone();
        let exec_fn = exec_fn.clone();
        let close_fn = close_fn.clone();
        runtime_db.set(
            "with",
            lua.create_function(
                move |lua, (selector, callback, _opts): (Value, Function, Option<Table>)| {
                    let handle = open_handle_id(lua, &open_fn, selector)?;
                    let proxy = create_db_proxy(
                        lua,
                        handle.clone(),
                        query_fn.clone(),
                        exec_fn.clone(),
                        close_fn.clone(),
                    )?;

                    let callback_result = callback.call::<MultiValue>(proxy);
                    let close_result =
                        close_handle_checked(lua, &close_fn, &handle, "runtime.db.with.close");

                    match (callback_result, close_result) {
                        (Err(callback_err), _) => Err(callback_err),
                        (Ok(_), Err(close_err)) => Err(close_err),
                        (Ok(values), Ok(_)) => Ok(values),
                    }
                },
            )?,
        )?;
    }

    {
        let open_fn = open_fn.clone();
        let query_fn = query_fn.clone();
        let exec_fn = exec_fn.clone();
        let close_fn = close_fn.clone();
        let mt = lua.create_table()?;
        mt.set(
            "__call",
            lua.create_function(move |lua, (_self, selector): (Value, Value)| {
                let handle = open_handle_id(lua, &open_fn, selector)?;
                create_db_proxy(
                    lua,
                    handle,
                    query_fn.clone(),
                    exec_fn.clone(),
                    close_fn.clone(),
                )
            })?,
        )?;
        let _ = runtime_db.set_metatable(Some(mt));
    }

    Ok(())
}
