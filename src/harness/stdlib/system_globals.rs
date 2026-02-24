use std::path::{Path, PathBuf};

use mlua::{Function, Lua, LuaSerdeExt, MultiValue, Result as LuaResult, Table, Value};

use crate::harness::stdlib::binding_common::{
    bool_err, nil_err, ok_bool, ok_value, string_ok, string_value,
};

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
        lua.create_function(|lua, name: String| import_module(lua, &name, None))?,
    )?;
    globals.set(
        "import_scoped",
        lua.create_function(|lua, (name, opts): (String, Option<Table>)| {
            import_module(lua, &name, opts)
        })?,
    )?;
    Ok(())
}

fn import_module(lua: &Lua, name: &str, opts: Option<Table>) -> LuaResult<Value> {
    let globals = lua.globals();
    let modules: Table = globals.get("__harness_modules")?;
    let module_value: Value = modules.get(name)?;
    if matches!(module_value, Value::Nil) {
        return Err(mlua::Error::runtime(format!(
            "import failed: module '{}' not found",
            name
        )));
    }

    let meta_value = globals
        .get::<Table>("__harness_module_meta")
        .ok()
        .and_then(|t| t.get::<Value>(name).ok())
        .unwrap_or(Value::Nil);

    if let Some(opts) = opts
        && let Ok(expected_root) = opts.get::<String>("root")
    {
        let actual_root = match &meta_value {
            Value::Table(t) => t.get::<String>("root").ok(),
            _ => None,
        };
        if actual_root.as_deref() != Some(expected_root.as_str()) {
            return Err(mlua::Error::runtime(format!(
                "import_scoped root mismatch for '{}': expected '{}', got '{}'",
                name,
                expected_root,
                actual_root.unwrap_or_else(|| "<none>".to_string())
            )));
        }
    }

    wrap_imported_module(lua, name, module_value, meta_value)
}

fn wrap_imported_module(
    lua: &Lua,
    module_name: &str,
    module_value: Value,
    meta_value: Value,
) -> LuaResult<Value> {
    let Value::Table(module_table) = module_value else {
        return Ok(module_value);
    };

    let proxy = lua.create_table()?;
    if !matches!(meta_value, Value::Nil) {
        proxy.set("__meta", meta_value)?;
    }

    for pair in module_table.clone().pairs::<Value, Value>() {
        let (key, value) = pair?;
        match value {
            Value::Function(func) => {
                let wrapped = wrap_module_function(lua, module_name, func)?;
                proxy.set(key, wrapped)?;
            }
            other => {
                proxy.set(key, other)?;
            }
        }
    }

    Ok(Value::Table(proxy))
}

fn wrap_module_function(lua: &Lua, module_name: &str, func: Function) -> LuaResult<Function> {
    let module_name = module_name.to_string();
    lua.create_function(move |lua, args: MultiValue| {
        let prev_module = get_active_harness_module(lua);
        set_active_harness_module(lua, Some(module_name.as_str()));
        let result = func.call::<MultiValue>(args);
        set_active_harness_module(lua, prev_module.as_deref());
        result
    })
}

fn set_active_harness_module(lua: &Lua, module_name: Option<&str>) {
    if let Some(app_data) = lua.app_data_ref::<crate::harness::globals::HarnessAppData>()
        && let Ok(mut lock) = app_data.active_harness_module.lock()
    {
        *lock = module_name.map(|s| s.to_string());
    }
}

fn get_active_harness_module(lua: &Lua) -> Option<String> {
    lua.app_data_ref::<crate::harness::globals::HarnessAppData>()
        .and_then(|app_data| {
            app_data
                .active_harness_module
                .lock()
                .ok()
                .and_then(|l| l.clone())
        })
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
                    Ok(c) => string_ok(lua, &c),
                    Err(e) => nil_err(lua, &e.to_string()),
                },
                None => nil_err(lua, "Unsafe path traversal"),
            },
        )?,
    )?;

    let r2 = root.clone();
    fs_table.set(
        "write",
        lua.create_function(move |lua, (path, content): (String, String)| {
            if content.len() > max_file_size {
                return bool_err(lua, "File exceeds max size");
            }
            match resolve_safe_path(&r2, &path) {
                Some(p) => {
                    if let Some(parent) = p.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    match std::fs::write(&p, content) {
                        Ok(_) => Ok(ok_bool()),
                        Err(e) => bool_err(lua, &e.to_string()),
                    }
                }
                None => bool_err(lua, "Unsafe path traversal"),
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
            Ok(s) => string_ok(lua, &s),
            Err(e) => nil_err(lua, &e.to_string()),
        })?,
    )?;
    json_table.set(
        "decode",
        lua.create_function(|lua, s: String| {
            match serde_json::from_str::<serde_json::Value>(&s) {
                Ok(j) => Ok(ok_value(lua.to_value(&j)?)),
                Err(e) => nil_err(lua, &e.to_string()),
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
