use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mlua::{Function, Lua, LuaSerdeExt, MultiValue, Result as LuaResult, Table, Value};

use crate::harness::stdlib::binding_common::{
    bool_err, nil_err, ok_bool, ok_value, string_ok, string_value,
};
use crate::harness::stdlib::governance_support::current_subject;
use crate::kernel::config::{GovernanceImportMode, GovernanceProfile};

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
        lua.create_function(|lua, name: String| import_module(lua, &name, None, false))?,
    )?;
    globals.set(
        "import_scoped",
        lua.create_function(|lua, (name, opts): (String, Option<Table>)| {
            import_module(lua, &name, opts, true)
        })?,
    )?;
    Ok(())
}

fn import_module(
    lua: &Lua,
    name: &str,
    opts: Option<Table>,
    is_scoped_call: bool,
) -> LuaResult<Value> {
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

    let requested_root = effective_import_root(lua, opts.as_ref(), is_scoped_call);
    enforce_import_policy(
        lua,
        name,
        &meta_value,
        requested_root.as_deref(),
        is_scoped_call,
    )?;

    if let Some(expected_root) = requested_root {
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

    let delegated_capabilities = delegated_import_capabilities(opts.as_ref())?;
    wrap_imported_module(lua, name, module_value, meta_value, delegated_capabilities)
}

fn enforce_import_policy(
    lua: &Lua,
    module_name: &str,
    meta_value: &Value,
    requested_root: Option<&str>,
    is_scoped_call: bool,
) -> LuaResult<()> {
    let Some(app_data) = lua.app_data_ref::<crate::harness::globals::HarnessAppData>() else {
        return Ok(());
    };

    let gov_cfg = app_data.governance_manager.config().clone();
    let subject = current_subject(&app_data);

    if gov_cfg.enforcement_enabled {
        let cap = if is_scoped_call {
            "harness.import.scoped"
        } else {
            "harness.import.unscoped"
        };
        app_data
            .governance_manager
            .require_capability_for_subject(&subject, cap)
            .map_err(mlua::Error::runtime)?;
    }

    if !gov_cfg.enforcement_enabled {
        return Ok(());
    }

    let allow_unscoped_open_override =
        matches!(gov_cfg.profile, GovernanceProfile::Open) && gov_cfg.import.allow_unscoped_in_open;

    match gov_cfg.import.mode {
        GovernanceImportMode::Legacy => {
            if is_scoped_call {
                return Err(mlua::Error::runtime(
                    "import_scoped is disabled when governance.import.mode=legacy".to_string(),
                ));
            }
        }
        GovernanceImportMode::Mixed => {}
        GovernanceImportMode::Scoped => {
            if !is_scoped_call && !allow_unscoped_open_override {
                return Err(mlua::Error::runtime(
                    "unscoped import() is disabled when governance.import.mode=scoped; use import_scoped(...)"
                        .to_string(),
                ));
            }
            if is_scoped_call && requested_root.is_none() {
                return Err(mlua::Error::runtime(
                    "import_scoped(...) requires opts.root or governance.import.default_root when governance.import.mode=scoped"
                        .to_string(),
                ));
            }
        }
    }

    if is_scoped_call {
        // In scoped mode / governed usage, importing a module without root attribution is suspicious.
        // Keep this as a runtime error only when a root is explicitly requested.
        if let Some(expected_root) = requested_root {
            let actual_root = match meta_value {
                Value::Table(t) => t.get::<String>("root").ok(),
                _ => None,
            };
            if actual_root.is_none() {
                return Err(mlua::Error::runtime(format!(
                    "import_scoped root '{}' requested for '{}', but module has no attributed governance root",
                    expected_root, module_name
                )));
            }
        }
    }

    Ok(())
}

fn effective_import_root(lua: &Lua, opts: Option<&Table>, is_scoped_call: bool) -> Option<String> {
    if let Some(root) = opts.and_then(|t| t.get::<String>("root").ok()) {
        return Some(root);
    }
    if !is_scoped_call {
        return None;
    }
    lua.app_data_ref::<crate::harness::globals::HarnessAppData>()
        .and_then(|app_data| {
            app_data
                .governance_manager
                .config()
                .import
                .default_root
                .clone()
        })
}

fn wrap_imported_module(
    lua: &Lua,
    module_name: &str,
    module_value: Value,
    meta_value: Value,
    delegated_capabilities: Option<BTreeMap<String, bool>>,
) -> LuaResult<Value> {
    let Value::Table(module_table) = module_value else {
        return Ok(module_value);
    };

    let proxy = lua.create_table()?;
    let module_root = match &meta_value {
        Value::Table(t) => t.get::<String>("root").ok(),
        _ => None,
    };
    if !matches!(meta_value, Value::Nil) {
        proxy.set("__meta", meta_value)?;
    }

    for pair in module_table.clone().pairs::<Value, Value>() {
        let (key, value) = pair?;
        match value {
            Value::Function(func) => {
                let wrapped = wrap_module_function(
                    lua,
                    module_name,
                    module_root.clone(),
                    delegated_capabilities.clone(),
                    func,
                )?;
                proxy.set(key, wrapped)?;
            }
            other => {
                proxy.set(key, other)?;
            }
        }
    }

    Ok(Value::Table(proxy))
}

fn wrap_module_function(
    lua: &Lua,
    module_name: &str,
    module_root: Option<String>,
    delegated_capabilities: Option<BTreeMap<String, bool>>,
    func: Function,
) -> LuaResult<Function> {
    let module_name = module_name.to_string();
    let module_root = module_root.clone();
    let delegated_capabilities = delegated_capabilities.clone();
    lua.create_function(move |lua, args: MultiValue| {
        let prev_module = get_active_harness_module(lua);
        let prev_root = get_active_harness_root(lua);
        let prev_caps = get_active_import_capabilities(lua);
        set_active_harness_module(lua, Some(module_name.as_str()));
        set_active_harness_root(lua, module_root.as_deref());
        set_active_import_capabilities(lua, delegated_capabilities.clone());
        let result = func.call::<MultiValue>(args);
        set_active_harness_module(lua, prev_module.as_deref());
        set_active_harness_root(lua, prev_root.as_deref());
        set_active_import_capabilities(lua, prev_caps);
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

fn set_active_harness_root(lua: &Lua, root_name: Option<&str>) {
    if let Some(app_data) = lua.app_data_ref::<crate::harness::globals::HarnessAppData>()
        && let Ok(mut lock) = app_data.active_harness_root.lock()
    {
        *lock = root_name.map(|s| s.to_string());
    }
}

fn get_active_harness_root(lua: &Lua) -> Option<String> {
    lua.app_data_ref::<crate::harness::globals::HarnessAppData>()
        .and_then(|app_data| {
            app_data
                .active_harness_root
                .lock()
                .ok()
                .and_then(|l| l.clone())
        })
}

fn set_active_import_capabilities(lua: &Lua, caps: Option<BTreeMap<String, bool>>) {
    if let Some(app_data) = lua.app_data_ref::<crate::harness::globals::HarnessAppData>()
        && let Ok(mut lock) = app_data.active_import_capabilities.lock()
    {
        *lock = caps;
    }
}

fn get_active_import_capabilities(lua: &Lua) -> Option<BTreeMap<String, bool>> {
    lua.app_data_ref::<crate::harness::globals::HarnessAppData>()
        .and_then(|app_data| {
            app_data
                .active_import_capabilities
                .lock()
                .ok()
                .and_then(|l| l.clone())
        })
}

fn delegated_import_capabilities(
    opts: Option<&Table>,
) -> LuaResult<Option<BTreeMap<String, bool>>> {
    let Some(opts) = opts else {
        return Ok(None);
    };
    let caps_value = opts.get::<Value>("capabilities").unwrap_or(Value::Nil);
    match caps_value {
        Value::Nil => Ok(None),
        Value::Table(t) => {
            let mut caps = BTreeMap::new();
            for pair in t.pairs::<String, Value>() {
                let (key, value) = pair?;
                match value {
                    Value::Boolean(b) => {
                        caps.insert(key, b);
                    }
                    _ => {
                        return Err(mlua::Error::runtime(format!(
                            "import_scoped opts.capabilities values must be booleans (key '{}')",
                            key
                        )));
                    }
                }
            }
            Ok(Some(caps))
        }
        _ => Err(mlua::Error::runtime(
            "import_scoped opts.capabilities must be a table".to_string(),
        )),
    }
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
