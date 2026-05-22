use std::path::{Path, PathBuf};

use mlua::{Lua, LuaSerdeExt, MultiValue, Result as LuaResult, Value};
use sha2::{Digest, Sha256};

mod fs;
mod imports;

pub(crate) use imports::ensure_load_time;
pub use imports::register_import_global;

use crate::harness::stdlib::binding_common::{lua_value_result, nil_err, string_ok, string_value};
use crate::harness::stdlib::governance_support::require_capability as require_governance_capability;

pub fn register_system_globals(lua: &Lua, fs_root: &Path, max_file_size: usize) -> LuaResult<()> {
    fs::register_fs_module(lua, fs_root, max_file_size)?;
    register_hash_module(lua)?;
    register_json_module(lua)?;
    register_time_module(lua)?;
    register_log_function(lua)?;
    register_try_function(lua)?;
    Ok(())
}

fn register_try_function(lua: &Lua) -> LuaResult<()> {
    let globals = lua.globals();
    globals.set(
        "try",
        lua.create_function(|lua, args: MultiValue| {
            let mut iter = args.into_iter();
            let func = match iter.next() {
                Some(Value::Function(func)) => func,
                Some(_) => {
                    return Err(mlua::Error::runtime(
                        "try expects a function as its first argument",
                    ));
                }
                None => return Err(mlua::Error::runtime("try expects a function argument")),
            };
            let rest: MultiValue = iter.collect();
            match func.call::<MultiValue>(rest) {
                Ok(values) => Ok(values),
                Err(err) => {
                    let mut out = MultiValue::new();
                    out.push_back(Value::Nil);
                    out.push_back(Value::String(lua.create_string(err.to_string())?));
                    Ok(out)
                }
            }
        })?,
    )?;
    Ok(())
}

pub(crate) fn resolve_safe_path(root: &Path, path_str: &str) -> Option<PathBuf> {
    crate::tools::is_safe_path(root, Path::new(path_str)).ok()
}

pub(crate) fn require_capability_for_lua(lua: &Lua, capability: &str) -> LuaResult<()> {
    if let Some(app_data) = lua.app_data_ref::<crate::harness::globals::HarnessAppData>() {
        require_governance_capability(&app_data, capability).map_err(mlua::Error::runtime)?;
    }
    Ok(())
}

fn hash_sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

fn register_hash_module(lua: &Lua) -> LuaResult<()> {
    let hash_table = lua.create_table()?;
    hash_table.set(
        "sha256",
        lua.create_function(|lua, text: String| {
            string_value(lua, &hash_sha256_hex(text.as_bytes()))
        })?,
    )?;
    lua.globals().set("hash", hash_table)?;
    Ok(())
}

fn register_json_module(lua: &Lua) -> LuaResult<()> {
    let json_table = lua.create_table()?;
    json_table.set(
        "encode",
        lua.create_function(|lua, val: Value| match serde_json::to_string(&val) {
            Ok(s) => string_ok(lua, &s),
            Err(e) => nil_err(lua, &e.to_string()),
        })?,
    )?;
    json_table.set(
        "decode",
        lua.create_function(|lua, s: String| {
            let result = serde_json::from_str::<serde_json::Value>(&s).map_err(|e| e.to_string());
            lua_value_result(lua, result, |lua, json| lua.to_value(&json))
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
            string_value(lua, &ts)
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
