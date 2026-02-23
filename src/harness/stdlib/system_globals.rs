use std::path::{Path, PathBuf};

use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};

pub fn register_system_globals(lua: &Lua, fs_root: &Path, max_file_size: usize) -> LuaResult<()> {
    register_fs_module(lua, fs_root, max_file_size)?;
    register_json_module(lua)?;
    register_time_module(lua)?;
    register_log_function(lua)?;
    Ok(())
}

pub fn register_import_global(lua: &Lua) -> LuaResult<()> {
    let globals = lua.globals();
    globals.set(
        "import",
        lua.create_function(|lua, name: String| {
            let globals = lua.globals();
            let modules: Table = globals.get("__harness_modules")?;
            modules.get::<Value>(name)
        })?,
    )?;
    Ok(())
}

fn resolve_safe_path(root: &Path, path_str: &str) -> Option<PathBuf> {
    crate::tools::is_safe_path(root, Path::new(path_str)).ok()
}

fn register_fs_module(lua: &Lua, fs_root: &Path, max_file_size: usize) -> LuaResult<()> {
    let fs_table = lua.create_table()?;
    let root = fs_root.to_path_buf();

    let r1 = root.clone();
    fs_table.set(
        "read",
        lua.create_function(
            move |lua, path: String| match resolve_safe_path(&r1, &path) {
                Some(p) => match std::fs::read_to_string(&p) {
                    Ok(c) => Ok((Value::String(lua.create_string(&c)?), Value::Nil)),
                    Err(e) => Ok((Value::Nil, Value::String(lua.create_string(e.to_string())?))),
                },
                None => Ok((
                    Value::Nil,
                    Value::String(lua.create_string("Unsafe path traversal")?),
                )),
            },
        )?,
    )?;

    let r2 = root.clone();
    fs_table.set(
        "write",
        lua.create_function(move |lua, (path, content): (String, String)| {
            if content.len() > max_file_size {
                return Ok((
                    Value::Boolean(false),
                    Value::String(lua.create_string("File exceeds max size")?),
                ));
            }
            match resolve_safe_path(&r2, &path) {
                Some(p) => {
                    if let Some(parent) = p.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    match std::fs::write(&p, content) {
                        Ok(_) => Ok((Value::Boolean(true), Value::Nil)),
                        Err(e) => Ok((
                            Value::Boolean(false),
                            Value::String(lua.create_string(e.to_string())?),
                        )),
                    }
                }
                None => Ok((
                    Value::Boolean(false),
                    Value::String(lua.create_string("Unsafe path traversal")?),
                )),
            }
        })?,
    )?;

    let r3 = root.clone();
    fs_table.set(
        "exists",
        lua.create_function(
            move |_lua, path: String| match resolve_safe_path(&r3, &path) {
                Some(p) => Ok(p.exists()),
                None => Ok(false),
            },
        )?,
    )?;

    let r4 = root.clone();
    fs_table.set(
        "is_safe_path",
        lua.create_function(move |_lua, path: String| Ok(resolve_safe_path(&r4, &path).is_some()))?,
    )?;

    lua.globals().set("fs", fs_table)?;
    Ok(())
}

fn register_json_module(lua: &Lua) -> LuaResult<()> {
    let json_table = lua.create_table()?;
    json_table.set(
        "encode",
        lua.create_function(|lua, val: Value| match serde_json::to_string(&val) {
            Ok(s) => Ok((Value::String(lua.create_string(&s)?), Value::Nil)),
            Err(e) => Ok((Value::Nil, Value::String(lua.create_string(e.to_string())?))),
        })?,
    )?;
    json_table.set(
        "decode",
        lua.create_function(|lua, s: String| {
            match serde_json::from_str::<serde_json::Value>(&s) {
                Ok(j) => Ok((lua.to_value(&j)?, Value::Nil)),
                Err(e) => Ok((Value::Nil, Value::String(lua.create_string(e.to_string())?))),
            }
        })?,
    )?;
    lua.globals().set("json", json_table)?;
    Ok(())
}

fn register_time_module(lua: &Lua) -> LuaResult<()> {
    let time_table = lua.create_table()?;
    time_table.set(
        "epoch_seconds",
        lua.create_function(|_lua, ()| {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();
            Ok(ts)
        })?,
    )?;
    time_table.set(
        "now_utc",
        lua.create_function(|lua, ()| {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs()
                .to_string();
            Ok(Value::String(lua.create_string(&ts)?))
        })?,
    )?;
    lua.globals().set("time", time_table)?;
    Ok(())
}

fn register_log_function(lua: &Lua) -> LuaResult<()> {
    lua.globals().set(
        "log",
        lua.create_function(|_lua, msg: String| {
            eprintln!("[harness] {}", msg);
            Ok(())
        })?,
    )?;
    Ok(())
}
