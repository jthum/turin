use mlua::{Function, Lua, Result as LuaResult, Table, Value};

use crate::harness::dx::common::call_and_raise_on_err;

pub fn register_code_cache_dx(lua: &Lua) -> LuaResult<()> {
    let globals = lua.globals();
    let runtime: Table = globals.get("runtime")?;

    let runtime_cache: Table = runtime.get("cache")?;
    let runtime_code: Table = runtime.get("code")?;
    let runtime_search: Table = runtime_code.get("search")?;

    let cache_read: Function = runtime_cache.get("read")?;
    let hybrid_search: Function = runtime_search.get("hybrid")?;

    let cache_table = match globals.get::<Value>("cache")? {
        Value::Table(table) => table,
        _ => lua.create_table()?,
    };
    {
        let cache_read = cache_read.clone();
        cache_table.set(
            "file",
            lua.create_function(move |lua, (path, opts): (String, Option<Table>)| {
                call_and_raise_on_err(lua, &cache_read, (path, opts), "cache.file")
            })?,
        )?;
    }
    globals.set("cache", cache_table)?;

    let code_table = match globals.get::<Value>("code")? {
        Value::Table(table) => table,
        _ => lua.create_table()?,
    };
    {
        let hybrid_search = hybrid_search.clone();
        code_table.set(
            "find",
            lua.create_function(move |lua, (query, opts): (String, Option<Table>)| {
                let selector = code_selector_from_opts(lua, opts.clone())?;
                call_and_raise_on_err(lua, &hybrid_search, (selector, query, opts), "code.find")
            })?,
        )?;
    }
    globals.set("code", code_table)?;

    Ok(())
}

fn code_selector_from_opts(lua: &Lua, opts: Option<Table>) -> LuaResult<Value> {
    let Some(opts) = opts else {
        return Ok(Value::String(lua.create_string(".")?));
    };

    let root = opts.get::<Option<String>>("root")?;
    let index_path = opts.get::<Option<String>>("index_path")?;
    if root.is_none() && index_path.is_none() {
        return Ok(Value::String(lua.create_string(".")?));
    }

    let selector = lua.create_table()?;
    selector.set("root", root.unwrap_or_else(|| ".".to_string()))?;
    if let Some(index_path) = index_path {
        selector.set("index_path", index_path)?;
    }
    Ok(Value::Table(selector))
}
