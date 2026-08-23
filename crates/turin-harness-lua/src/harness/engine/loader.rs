use anyhow::Result;
use mlua::{Function, Lua, LuaSerdeExt, MultiValue, Table, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::display;
use crate::harness::globals::HarnessAppData;

use super::KNOWN_HOOKS;

pub(crate) fn format_lua_error(e: &mlua::Error) -> String {
    let err_str = e.to_string();
    let ansi = display::stderr_ansi();

    if let Some(first_colon) = err_str.find(':') {
        let prefix = &err_str[..first_colon];
        let rest = &err_str[first_colon + 1..];

        if let Some(second_colon) = rest.find(':') {
            let line_num = rest[..second_colon].trim();
            let message = rest[second_colon + 1..].trim();

            if line_num.chars().all(|c| c.is_ascii_digit()) {
                let cleaned_prefix = prefix
                    .strip_prefix("[string \"@")
                    .and_then(|s| s.strip_suffix("\"]"))
                    .or_else(|| prefix.strip_prefix('@'))
                    .unwrap_or(prefix);

                let header = format!(
                    "{} {} {}",
                    display::paint("Script Error", "31;1", ansi),
                    display::paint("in", "31", ansi),
                    display::paint(cleaned_prefix, "31", ansi)
                );
                let line = display::paint(&format!("  Line {line_num}: {message}"), "31", ansi);
                return format!("{header}\n{line}");
            }
        }
    }

    format!("{} {}", display::paint("Lua Error:", "31;1", ansi), err_str)
}

pub(crate) fn ensure_module_registry(lua: &Lua) -> mlua::Result<(Table, Table, Table)> {
    let globals = lua.globals();
    if !globals.contains_key("__harness_modules")? {
        globals.set("__harness_modules", lua.create_table()?)?;
    }
    if !globals.contains_key("__harness_module_meta")? {
        globals.set("__harness_module_meta", lua.create_table()?)?;
    }
    if !globals.contains_key("__harness_module_path_index")? {
        globals.set("__harness_module_path_index", lua.create_table()?)?;
    }

    Ok((
        globals.get("__harness_modules")?,
        globals.get("__harness_module_meta")?,
        globals.get("__harness_module_path_index")?,
    ))
}

pub(crate) fn active_module_names(lua: &Lua) -> Vec<String> {
    lua.app_data_ref::<HarnessAppData>()
        .and_then(|app_data| app_data.active_modules.lock().ok().map(|v| v.clone()))
        .unwrap_or_default()
}

pub(crate) fn clear_active_modules(lua: &Lua) {
    if let Some(app_data) = lua.app_data_ref::<HarnessAppData>()
        && let Ok(mut lock) = app_data.active_modules.lock()
    {
        lock.clear();
    }
}

pub(crate) fn push_active_module(lua: &Lua, module_name: &str) {
    if let Some(app_data) = lua.app_data_ref::<HarnessAppData>()
        && let Ok(mut lock) = app_data.active_modules.lock()
        && !lock.iter().any(|name| name == module_name)
    {
        lock.push(module_name.to_string());
    }
}

pub(crate) fn explicit_watch_roots(lua: &Lua) -> Vec<PathBuf> {
    lua.app_data_ref::<HarnessAppData>()
        .and_then(|app_data| app_data.watch_roots.lock().ok().map(|v| v.clone()))
        .unwrap_or_default()
}

pub(crate) fn register_watch_root(lua: &Lua, path: PathBuf) {
    if let Some(app_data) = lua.app_data_ref::<HarnessAppData>()
        && let Ok(mut lock) = app_data.watch_roots.lock()
        && !lock.iter().any(|p| p == &path)
    {
        lock.push(path);
    }
}

pub(crate) fn set_loading_phase(lua: &Lua, is_loading: bool) {
    if let Some(app_data) = lua.app_data_ref::<HarnessAppData>()
        && let Ok(mut lock) = app_data.loading_phase.lock()
    {
        *lock = is_loading;
    }
}

pub(crate) fn is_loading_phase(lua: &Lua) -> bool {
    lua.app_data_ref::<HarnessAppData>()
        .and_then(|app_data| app_data.loading_phase.lock().ok().map(|v| *v))
        .unwrap_or(false)
}

pub(crate) fn lookup_loaded_module_by_canonical_path(lua: &Lua, path: &Path) -> Option<String> {
    let (_, _, path_index) = ensure_module_registry(lua).ok()?;
    let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    path_index
        .get::<String>(canon.to_string_lossy().to_string())
        .ok()
}

pub(crate) fn register_module_path(lua: &Lua, module_name: &str, path: &Path) -> mlua::Result<()> {
    let (_, _, path_index) = ensure_module_registry(lua)?;
    let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    path_index.set(canon.to_string_lossy().to_string(), module_name)
}

pub(crate) fn resolve_governance_root_name(lua: &Lua, script_path: &Path) -> Option<String> {
    let app_data = lua.app_data_ref::<HarnessAppData>()?;
    let script_canon =
        std::fs::canonicalize(script_path).unwrap_or_else(|_| PathBuf::from(script_path));

    let mut best: Option<(usize, String)> = None;
    for (root_name, root_cfg) in &app_data.config.governance.roots {
        let configured = PathBuf::from(&root_cfg.path);
        let root_path = if configured.is_absolute() {
            configured
        } else {
            app_data.workspace_root.join(configured)
        };
        let root_canon = std::fs::canonicalize(&root_path).unwrap_or(root_path);
        if script_canon.starts_with(&root_canon) {
            let score = root_canon.components().count();
            match &best {
                Some((best_score, _)) if *best_score >= score => {}
                _ => best = Some((score, root_name.clone())),
            }
        }
    }

    best.map(|(_, name)| name)
}

fn build_block_table(
    lua: &Lua,
    module_name: &str,
    path: &Path,
    block_name: Option<&str>,
    block_config: Option<&Table>,
) -> mlua::Result<Table> {
    let block = lua.create_table()?;
    block.set("name", block_name.unwrap_or(module_name))?;
    block.set("path", path.to_string_lossy().to_string())?;
    if let Some(config) = block_config {
        block.set("config", config.clone())?;
    } else {
        block.set("config", lua.create_table()?)?;
    }
    Ok(block)
}

fn wrap_hook_with_when(
    lua: &Lua,
    hook_name: &str,
    when_fn: Function,
    func: Function,
) -> mlua::Result<Function> {
    let hook_name = hook_name.to_string();
    lua.create_function(move |_lua, args: MultiValue| {
        let payload = args.front().cloned().unwrap_or(Value::Nil);
        let allowed = when_fn.call::<bool>((hook_name.clone(), payload))?;
        if !allowed {
            return Ok(MultiValue::new());
        }
        func.call::<MultiValue>(args)
    })
}

pub(crate) struct ModuleLoadOptions {
    pub activate: bool,
    pub block_name: Option<String>,
    pub block_config: Option<Table>,
    pub when_fn: Option<Function>,
    pub delegated_capabilities: Option<BTreeMap<String, bool>>,
    pub cache_by_path: bool,
}

pub(crate) fn load_module_from_source(
    lua: &Lua,
    module_name: &str,
    source: &str,
    path: &Path,
    opts: ModuleLoadOptions,
) -> Result<()> {
    let (modules, module_meta, _) = ensure_module_registry(lua)?;
    let globals = lua.globals();
    let module_root = resolve_governance_root_name(lua, path);
    let prev_module = lua.app_data_ref::<HarnessAppData>().and_then(|app_data| {
        app_data
            .execution_ctx
            .lock()
            .ok()
            .and_then(|ctx| ctx.harness_module.clone())
    });
    let prev_root = lua.app_data_ref::<HarnessAppData>().and_then(|app_data| {
        app_data
            .execution_ctx
            .lock()
            .ok()
            .and_then(|ctx| ctx.harness_root.clone())
    });
    let prev_caps = lua.app_data_ref::<HarnessAppData>().and_then(|app_data| {
        app_data
            .execution_ctx
            .lock()
            .ok()
            .and_then(|ctx| ctx.import_capabilities.clone())
    });
    if let Some(app_data) = lua.app_data_ref::<HarnessAppData>()
        && let Ok(mut lock) = app_data.execution_ctx.lock()
    {
        lock.harness_module = Some(module_name.to_string());
        lock.harness_root = module_root.clone();
        lock.import_capabilities = opts.delegated_capabilities.clone().or(prev_caps.clone());
    }

    let env = lua.create_table()?;
    let meta = lua.create_table()?;
    meta.set("__index", globals)?;
    let _ = env.set_metatable(Some(meta));
    env.set(
        "block",
        build_block_table(
            lua,
            module_name,
            path,
            opts.block_name.as_deref(),
            opts.block_config.as_ref(),
        )?,
    )?;

    let eval_result = lua
        .load(source)
        .set_name(format!("@{}", path.display()))
        .set_environment(env.clone())
        .eval();
    if let Some(app_data) = lua.app_data_ref::<HarnessAppData>()
        && let Ok(mut lock) = app_data.execution_ctx.lock()
    {
        lock.harness_module = prev_module;
        lock.harness_root = prev_root;
        lock.import_capabilities = prev_caps;
    }
    let retval: Value = eval_result.map_err(|e| {
        anyhow::anyhow!(format!(
            "Failed to load harness script '{}':\n{}",
            path.display(),
            format_lua_error(&e)
        ))
    })?;

    let module_exports = match retval {
        Value::Table(t) => t,
        _ => lua.create_table()?,
    };

    for hook in KNOWN_HOOKS {
        if !module_exports.contains_key(*hook)?
            && let Ok(func) = env.get::<Function>(*hook)
        {
            let wrapped = if let Some(ref when_fn) = opts.when_fn {
                wrap_hook_with_when(lua, hook, when_fn.clone(), func)?
            } else {
                func
            };
            module_exports.set(*hook, wrapped)?;
        } else if let Some(ref when_fn) = opts.when_fn
            && let Ok(func) = module_exports.get::<Function>(*hook)
        {
            module_exports.set(
                *hook,
                wrap_hook_with_when(lua, hook, when_fn.clone(), func)?,
            )?;
        }
    }

    modules.set(module_name, module_exports)?;
    let meta = lua.create_table()?;
    meta.set("name", module_name)?;
    meta.set("path", path.to_string_lossy().to_string())?;
    meta.set("spec", opts.block_name.as_deref().unwrap_or(module_name))?;
    if let Some(root_name) = module_root {
        meta.set("root", root_name)?;
    }
    if let Some(caps) = opts.delegated_capabilities {
        let caps_value = lua.to_value(&caps)?;
        meta.set("delegated_capabilities", caps_value)?;
    }
    module_meta.set(module_name, meta)?;
    if opts.cache_by_path {
        register_module_path(lua, module_name, path)?;
    }
    if opts.activate {
        push_active_module(lua, module_name);
    }
    Ok(())
}
