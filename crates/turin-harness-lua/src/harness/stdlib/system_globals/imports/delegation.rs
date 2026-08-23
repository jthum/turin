use std::collections::BTreeMap;

use mlua::{Function, Lua, MultiValue, Result as LuaResult, Table, Value};

use crate::kernel::governance::capability_allowed_by_bool_rules;

const MAX_IMPORT_PROXY_WRAP_DEPTH: usize = 16;

pub(super) fn wrap_imported_module(
    lua: &Lua,
    module_name: &str,
    module_value: Value,
    meta_value: Value,
    delegated_capabilities: Option<BTreeMap<String, bool>>,
) -> LuaResult<Value> {
    let Value::Table(module_table) = module_value else {
        return Ok(module_value);
    };
    let module_root = module_root(&meta_value);
    let proxy = wrap_imported_table(
        lua,
        module_name,
        module_root,
        delegated_capabilities,
        module_table,
        Some(meta_value),
        0,
    )?;
    Ok(Value::Table(proxy))
}

fn wrap_imported_table(
    lua: &Lua,
    module_name: &str,
    module_root: Option<String>,
    delegated_capabilities: Option<BTreeMap<String, bool>>,
    source_table: Table,
    meta_value: Option<Value>,
    depth: usize,
) -> LuaResult<Table> {
    if depth > MAX_IMPORT_PROXY_WRAP_DEPTH {
        return Err(mlua::Error::runtime(format!(
            "imported module '{}' nested export depth exceeds limit {}",
            module_name, MAX_IMPORT_PROXY_WRAP_DEPTH
        )));
    }

    let proxy = lua.create_table()?;
    if let Some(meta_value) = meta_value
        && !matches!(meta_value, Value::Nil)
    {
        proxy.set("__meta", meta_value)?;
    }

    for pair in source_table.pairs::<Value, Value>() {
        let (key, value) = pair?;
        let wrapped = wrap_imported_value(
            lua,
            module_name,
            module_root.clone(),
            delegated_capabilities.clone(),
            value,
            depth + 1,
        )?;
        proxy.set(key, wrapped)?;
    }

    Ok(proxy)
}

fn wrap_imported_value(
    lua: &Lua,
    module_name: &str,
    module_root: Option<String>,
    delegated_capabilities: Option<BTreeMap<String, bool>>,
    value: Value,
    depth: usize,
) -> LuaResult<Value> {
    match value {
        Value::Function(func) => Ok(Value::Function(wrap_module_function(
            lua,
            module_name,
            module_root,
            delegated_capabilities,
            func,
        )?)),
        Value::Table(table) => Ok(Value::Table(wrap_imported_table(
            lua,
            module_name,
            module_root,
            delegated_capabilities,
            table,
            None,
            depth,
        )?)),
        other => Ok(other),
    }
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
        let applied_caps = delegated_capabilities.clone().or_else(|| prev_caps.clone());
        set_active_harness_module(lua, Some(module_name.as_str()));
        set_active_harness_root(lua, module_root.as_deref());
        set_active_import_capabilities(lua, applied_caps);
        let result = func.call::<MultiValue>(args);
        set_active_harness_module(lua, prev_module.as_deref());
        set_active_harness_root(lua, prev_root.as_deref());
        set_active_import_capabilities(lua, prev_caps);
        result
    })
}

fn module_root(meta_value: &Value) -> Option<String> {
    match meta_value {
        Value::Table(t) => t.get::<String>("root").ok(),
        _ => None,
    }
}

fn set_active_harness_module(lua: &Lua, module_name: Option<&str>) {
    if let Some(app_data) = lua.app_data_ref::<crate::harness::globals::HarnessAppData>()
        && let Ok(mut lock) = app_data.execution_ctx.lock()
    {
        lock.harness_module = module_name.map(|s| s.to_string());
    }
}

fn get_active_harness_module(lua: &Lua) -> Option<String> {
    lua.app_data_ref::<crate::harness::globals::HarnessAppData>()
        .and_then(|app_data| {
            app_data
                .execution_ctx
                .lock()
                .ok()
                .and_then(|l| l.harness_module.clone())
        })
}

fn set_active_harness_root(lua: &Lua, root_name: Option<&str>) {
    if let Some(app_data) = lua.app_data_ref::<crate::harness::globals::HarnessAppData>()
        && let Ok(mut lock) = app_data.execution_ctx.lock()
    {
        lock.harness_root = root_name.map(|s| s.to_string());
    }
}

fn get_active_harness_root(lua: &Lua) -> Option<String> {
    lua.app_data_ref::<crate::harness::globals::HarnessAppData>()
        .and_then(|app_data| {
            app_data
                .execution_ctx
                .lock()
                .ok()
                .and_then(|l| l.harness_root.clone())
        })
}

fn set_active_import_capabilities(lua: &Lua, caps: Option<BTreeMap<String, bool>>) {
    if let Some(app_data) = lua.app_data_ref::<crate::harness::globals::HarnessAppData>()
        && let Ok(mut lock) = app_data.execution_ctx.lock()
    {
        lock.import_capabilities = caps;
    }
}

pub(super) fn get_active_import_capabilities(lua: &Lua) -> Option<BTreeMap<String, bool>> {
    lua.app_data_ref::<crate::harness::globals::HarnessAppData>()
        .and_then(|app_data| {
            app_data
                .execution_ctx
                .lock()
                .ok()
                .and_then(|l| l.import_capabilities.clone())
        })
}

pub(super) fn delegated_import_capabilities(
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

pub(super) fn enforce_delegated_capability_subset(
    lua: &Lua,
    requested_caps: Option<&BTreeMap<String, bool>>,
) -> LuaResult<()> {
    let Some(requested_caps) = requested_caps else {
        return Ok(());
    };
    let Some(parent_caps) = get_active_import_capabilities(lua) else {
        return Ok(());
    };

    for (capability, allowed) in requested_caps {
        if !*allowed {
            continue;
        }
        if !capability_allowed_by_bool_rules(&parent_caps, capability) {
            return Err(mlua::Error::runtime(format!(
                "import_scoped capability delegation cannot grant '{}' beyond importer delegation",
                capability
            )));
        }
    }

    Ok(())
}
