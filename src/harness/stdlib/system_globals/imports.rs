use std::path::{Path, PathBuf};

use mlua::{Function, Lua, Result as LuaResult, Table, Value};

use crate::harness::engine::{
    ModuleLoadOptions, is_loading_phase, load_module_from_source,
    lookup_loaded_module_by_canonical_path, register_watch_root, resolve_governance_root_name,
};
use crate::harness::stdlib::system_globals::resolve_safe_path;
use crate::kernel::config::GovernanceImportMode;
use delegation::{
    delegated_import_capabilities, enforce_delegated_capability_subset,
    get_active_import_capabilities, wrap_imported_module,
};

mod delegation;

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
    globals.set(
        "use",
        lua.create_function(|lua, (name, opts): (String, Option<Table>)| {
            use_module(lua, &name, opts, false)
        })?,
    )?;
    globals.set(
        "use_scoped",
        lua.create_function(|lua, (name, opts): (String, Option<Table>)| {
            use_module(lua, &name, opts, true)
        })?,
    )?;
    globals.set(
        "watch",
        lua.create_function(|lua, path: String| watch_path(lua, &path))?,
    )?;
    Ok(())
}

#[derive(Clone, Copy)]
enum ModulePolicyOp {
    Import,
    Use,
}

impl ModulePolicyOp {
    fn unscoped_call(self) -> &'static str {
        match self {
            Self::Import => "import",
            Self::Use => "use",
        }
    }

    fn scoped_call(self) -> &'static str {
        match self {
            Self::Import => "import_scoped",
            Self::Use => "use_scoped",
        }
    }

    fn capability(self, is_scoped_call: bool) -> &'static str {
        match (self, is_scoped_call) {
            (Self::Import, true) => "harness.import.scoped",
            (Self::Import, false) => "harness.import.unscoped",
            (Self::Use, true) => "harness.use.scoped",
            (Self::Use, false) => "harness.use.unscoped",
        }
    }
}

fn watch_path(lua: &Lua, path_str: &str) -> LuaResult<()> {
    ensure_load_time(lua, "watch")?;
    let path = resolve_watch_path(lua, path_str)?;
    register_watch_root(lua, path);
    Ok(())
}

fn import_module(
    lua: &Lua,
    name: &str,
    opts: Option<Table>,
    is_scoped_call: bool,
) -> LuaResult<Value> {
    let requested_root = effective_import_root(lua, opts.as_ref(), is_scoped_call);
    let requested_capabilities = delegated_import_capabilities(opts.as_ref())?;
    enforce_delegated_capability_subset(lua, requested_capabilities.as_ref())?;

    let (module_name, module_value, meta_value) =
        ensure_importable_module_loaded(lua, name, requested_root.as_deref(), is_scoped_call)?;

    enforce_import_policy(
        lua,
        name,
        &meta_value,
        requested_root.as_deref(),
        is_scoped_call,
    )?;
    ensure_root_match(
        name,
        &meta_value,
        requested_root.as_deref(),
        "import_scoped",
    )?;

    let effective_capabilities =
        requested_capabilities.or_else(|| get_active_import_capabilities(lua));
    wrap_imported_module(
        lua,
        &module_name,
        module_value,
        meta_value,
        effective_capabilities,
    )
}

fn use_module(lua: &Lua, name: &str, opts: Option<Table>, is_scoped_call: bool) -> LuaResult<()> {
    ensure_load_time(lua, if is_scoped_call { "use_scoped" } else { "use" })?;

    let requested_root = effective_import_root(lua, opts.as_ref(), is_scoped_call);
    let requested_capabilities = delegated_import_capabilities(opts.as_ref())?;
    enforce_delegated_capability_subset(lua, requested_capabilities.as_ref())?;
    let block_config = delegated_block_config(opts.as_ref())?;
    let when_fn = delegated_when_fn(opts.as_ref())?;
    let effective_capabilities =
        requested_capabilities.or_else(|| get_active_import_capabilities(lua));

    let path = resolve_module_path(lua, name)?;
    let meta_value = provisional_module_meta(lua, name, &path)?;
    enforce_use_policy(
        lua,
        name,
        &meta_value,
        requested_root.as_deref(),
        is_scoped_call,
    )?;
    ensure_root_match(name, &meta_value, requested_root.as_deref(), "use_scoped")?;

    let source = read_module_source("use", &path)?;
    let module_name = next_used_module_name(lua, name)?;
    load_module_from_source(
        lua,
        &module_name,
        &source,
        &path,
        ModuleLoadOptions {
            activate: true,
            block_name: Some(name.to_string()),
            block_config,
            when_fn,
            delegated_capabilities: effective_capabilities,
            cache_by_path: false,
        },
    )
    .map_err(mlua::Error::runtime)?;
    Ok(())
}

pub(crate) fn ensure_load_time(lua: &Lua, op_name: &str) -> LuaResult<()> {
    if !is_loading_phase(lua) {
        return Err(mlua::Error::runtime(format!(
            "{}(...) can only be called during harness load",
            op_name
        )));
    }
    Ok(())
}

fn resolve_watch_path(lua: &Lua, path_str: &str) -> LuaResult<PathBuf> {
    let Some(app_data) = lua.app_data_ref::<crate::harness::globals::HarnessAppData>() else {
        return Err(mlua::Error::runtime(
            "watch failed: harness app data unavailable".to_string(),
        ));
    };
    resolve_safe_path(&app_data.harness_directory, path_str)
        .ok_or_else(|| mlua::Error::runtime(format!("watch failed: unsafe path '{}'", path_str)))
}

fn resolve_module_path(lua: &Lua, module_name: &str) -> LuaResult<PathBuf> {
    let Some(app_data) = lua.app_data_ref::<crate::harness::globals::HarnessAppData>() else {
        return Err(mlua::Error::runtime(
            "module resolution failed: harness app data unavailable".to_string(),
        ));
    };

    let root = &app_data.harness_directory;
    let spec_path = Path::new(module_name);
    if spec_path.as_os_str().is_empty() {
        return Err(mlua::Error::runtime(
            "module resolution failed: empty module name".to_string(),
        ));
    }

    let mut candidates = Vec::new();
    if spec_path.extension().is_some() {
        candidates.push(spec_path.to_path_buf());
    } else {
        candidates.push(spec_path.with_extension("lua"));
        candidates.push(spec_path.join("init.lua"));
    }

    for rel in candidates {
        if let Some(path) = resolve_safe_path(root, &rel.to_string_lossy())
            && path.is_file()
        {
            return Ok(path);
        }
    }

    Err(mlua::Error::runtime(format!(
        "module '{}' not found under harness directory '{}'",
        module_name,
        root.display()
    )))
}

fn provisional_module_meta(lua: &Lua, module_name: &str, path: &Path) -> LuaResult<Value> {
    let meta = lua.create_table()?;
    meta.set("name", module_name)?;
    meta.set("path", path.to_string_lossy().to_string())?;
    if let Some(root_name) = resolve_governance_root_name(lua, path) {
        meta.set("root", root_name)?;
    }
    Ok(Value::Table(meta))
}

fn ensure_importable_module_loaded(
    lua: &Lua,
    name: &str,
    requested_root: Option<&str>,
    is_scoped_call: bool,
) -> LuaResult<(String, Value, Value)> {
    if let Some(found) = get_module_and_meta(lua, name)? {
        return Ok((name.to_string(), found.0, found.1));
    }

    let path = resolve_module_path(lua, name)?;
    if let Some(existing_name) = lookup_loaded_module_by_canonical_path(lua, &path)
        && let Some(found) = get_module_and_meta(lua, &existing_name)?
    {
        return Ok((existing_name, found.0, found.1));
    }

    let meta_value = provisional_module_meta(lua, name, &path)?;
    enforce_import_policy(lua, name, &meta_value, requested_root, is_scoped_call)?;
    ensure_root_match(name, &meta_value, requested_root, "import_scoped")?;

    let source = read_module_source("import", &path)?;
    load_module_from_source(
        lua,
        name,
        &source,
        &path,
        ModuleLoadOptions {
            activate: false,
            block_name: Some(name.to_string()),
            block_config: None,
            when_fn: None,
            delegated_capabilities: None,
            cache_by_path: true,
        },
    )
    .map_err(mlua::Error::runtime)?;

    let Some(found) = get_module_and_meta(lua, name)? else {
        return Err(mlua::Error::runtime(format!(
            "import failed: module '{}' did not register",
            name
        )));
    };

    Ok((name.to_string(), found.0, found.1))
}

fn get_module_and_meta(lua: &Lua, name: &str) -> LuaResult<Option<(Value, Value)>> {
    let globals = lua.globals();
    let modules: Table = globals.get("__harness_modules")?;
    let module_value: Value = modules.get(name)?;
    if matches!(module_value, Value::Nil) {
        return Ok(None);
    }
    let meta_value = globals
        .get::<Table>("__harness_module_meta")
        .ok()
        .and_then(|t| t.get::<Value>(name).ok())
        .unwrap_or(Value::Nil);
    Ok(Some((module_value, meta_value)))
}

fn read_module_source(op_name: &str, path: &Path) -> LuaResult<String> {
    std::fs::read_to_string(path).map_err(|e| {
        mlua::Error::runtime(format!(
            "{} failed: could not read '{}' ({})",
            op_name,
            path.display(),
            e
        ))
    })
}

fn next_used_module_name(lua: &Lua, spec: &str) -> LuaResult<String> {
    let globals = lua.globals();
    let modules: Table = globals.get("__harness_modules")?;
    for idx in 1.. {
        let candidate = format!("{spec}#use{idx}");
        if !modules.contains_key(candidate.as_str())? {
            return Ok(candidate);
        }
    }
    Err(mlua::Error::runtime(format!(
        "use failed: exhausted module slots for '{}'",
        spec
    )))
}

fn delegated_block_config(opts: Option<&Table>) -> LuaResult<Option<Table>> {
    let Some(opts) = opts else {
        return Ok(None);
    };
    match opts.get::<Value>("config").unwrap_or(Value::Nil) {
        Value::Nil => Ok(None),
        Value::Table(t) => Ok(Some(t)),
        _ => Err(mlua::Error::runtime(
            "use opts.config must be a table".to_string(),
        )),
    }
}

fn delegated_when_fn(opts: Option<&Table>) -> LuaResult<Option<Function>> {
    let Some(opts) = opts else {
        return Ok(None);
    };
    match opts.get::<Value>("when").unwrap_or(Value::Nil) {
        Value::Nil => Ok(None),
        Value::Function(f) => Ok(Some(f)),
        _ => Err(mlua::Error::runtime(
            "use opts.when must be a function".to_string(),
        )),
    }
}

fn ensure_root_match(
    module_name: &str,
    meta_value: &Value,
    expected_root: Option<&str>,
    op_name: &str,
) -> LuaResult<()> {
    let Some(expected_root) = expected_root else {
        return Ok(());
    };
    let actual_root = match meta_value {
        Value::Table(t) => t.get::<String>("root").ok(),
        _ => None,
    };
    if actual_root.as_deref() != Some(expected_root) {
        return Err(mlua::Error::runtime(format!(
            "{} root mismatch for '{}': expected '{}', got '{}'",
            op_name,
            module_name,
            expected_root,
            actual_root.unwrap_or_else(|| "<none>".to_string())
        )));
    }
    Ok(())
}

fn enforce_import_policy(
    lua: &Lua,
    module_name: &str,
    meta_value: &Value,
    requested_root: Option<&str>,
    is_scoped_call: bool,
) -> LuaResult<()> {
    enforce_module_policy(
        lua,
        module_name,
        meta_value,
        requested_root,
        is_scoped_call,
        ModulePolicyOp::Import,
    )
}

fn enforce_use_policy(
    lua: &Lua,
    module_name: &str,
    meta_value: &Value,
    requested_root: Option<&str>,
    is_scoped_call: bool,
) -> LuaResult<()> {
    enforce_module_policy(
        lua,
        module_name,
        meta_value,
        requested_root,
        is_scoped_call,
        ModulePolicyOp::Use,
    )
}

fn enforce_module_policy(
    lua: &Lua,
    module_name: &str,
    meta_value: &Value,
    requested_root: Option<&str>,
    is_scoped_call: bool,
    op: ModulePolicyOp,
) -> LuaResult<()> {
    let Some(app_data) = lua.app_data_ref::<crate::harness::globals::HarnessAppData>() else {
        return Ok(());
    };

    let gov_cfg = app_data.governance_manager.config().clone();
    let subject = crate::harness::stdlib::governance_support::current_subject(&app_data);

    if gov_cfg.enforcement_enabled {
        app_data
            .governance_manager
            .require_capability_for_subject(&subject, op.capability(is_scoped_call))
            .map_err(mlua::Error::runtime)?;
    }

    if !gov_cfg.enforcement_enabled {
        return Ok(());
    }

    let allow_unscoped_open_override = gov_cfg.import.allow_unscoped_in_open;

    match gov_cfg.import.mode {
        GovernanceImportMode::Legacy => {
            if is_scoped_call {
                return Err(mlua::Error::runtime(format!(
                    "{} is disabled when governance.import.mode=legacy",
                    op.scoped_call()
                )));
            }
        }
        GovernanceImportMode::Mixed => {}
        GovernanceImportMode::Scoped => {
            if !is_scoped_call && !allow_unscoped_open_override {
                return Err(mlua::Error::runtime(format!(
                    "unscoped {}() is disabled when governance.import.mode=scoped; use {}(...)",
                    op.unscoped_call(),
                    op.scoped_call()
                )));
            }
            if is_scoped_call && requested_root.is_none() {
                return Err(mlua::Error::runtime(format!(
                    "{}(...) requires opts.root or governance.import.default_root when governance.import.mode=scoped",
                    op.scoped_call()
                )));
            }
        }
    }

    if is_scoped_call
        && let Some(expected_root) = requested_root
        && module_root(meta_value).is_none()
    {
        return Err(mlua::Error::runtime(format!(
            "{} root '{}' requested for '{}', but module has no attributed governance root",
            op.scoped_call(),
            expected_root,
            module_name
        )));
    }

    Ok(())
}

fn module_root(meta_value: &Value) -> Option<String> {
    match meta_value {
        Value::Table(t) => t.get::<String>("root").ok(),
        _ => None,
    }
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
