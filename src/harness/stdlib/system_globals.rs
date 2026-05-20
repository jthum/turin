use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use mlua::{Function, Lua, LuaSerdeExt, MultiValue, Result as LuaResult, Table, Value};
use sha2::{Digest, Sha256};

use crate::harness::engine::{
    ModuleLoadOptions, is_loading_phase, load_module_from_source,
    lookup_loaded_module_by_canonical_path, register_watch_root, resolve_governance_root_name,
};
use crate::harness::globals::HarnessAppData;
use crate::harness::globals::block_on_current;
use crate::harness::stdlib::binding_common::{
    bool_err, nil_err, ok_bool, ok_value, string_ok, string_value,
};
use crate::harness::stdlib::governance_support::{
    current_subject, require_capability as require_governance_capability,
};
use crate::harness::stdlib::scoped_data_backend::encode_scope_key;
use crate::kernel::config::{GovernanceImportMode, GovernanceProfile};
use crate::persistence::manager::{StorePathScope, StoreSelector};

pub fn register_system_globals(lua: &Lua, fs_root: &Path, max_file_size: usize) -> LuaResult<()> {
    register_fs_module(lua, fs_root, max_file_size)?;
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

const MAX_IMPORT_PROXY_WRAP_DEPTH: usize = 16;

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

    let source = std::fs::read_to_string(&path).map_err(|e| {
        mlua::Error::runtime(format!(
            "use failed: could not read '{}' ({})",
            path.display(),
            e
        ))
    })?;
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

    let source = std::fs::read_to_string(&path).map_err(|e| {
        mlua::Error::runtime(format!(
            "import failed: could not read '{}' ({})",
            path.display(),
            e
        ))
    })?;
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
    let subject = current_subject(&app_data);

    if gov_cfg.enforcement_enabled {
        app_data
            .governance_manager
            .require_capability_for_subject(&subject, op.capability(is_scoped_call))
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

fn get_active_import_capabilities(lua: &Lua) -> Option<BTreeMap<String, bool>> {
    lua.app_data_ref::<crate::harness::globals::HarnessAppData>()
        .and_then(|app_data| {
            app_data
                .execution_ctx
                .lock()
                .ok()
                .and_then(|l| l.import_capabilities.clone())
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

fn enforce_delegated_capability_subset(
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
        if !delegated_capability_allowed_by_parent(&parent_caps, capability) {
            return Err(mlua::Error::runtime(format!(
                "import_scoped capability delegation cannot grant '{}' beyond importer delegation",
                capability
            )));
        }
    }

    Ok(())
}

fn delegated_capability_allowed_by_parent(
    parent_caps: &BTreeMap<String, bool>,
    capability: &str,
) -> bool {
    if let Some(allowed) = parent_caps.get(capability) {
        return *allowed;
    }

    let mut best: Option<(&str, bool)> = None;
    for (rule, allowed) in parent_caps {
        let Some(prefix) = rule.strip_suffix(".*") else {
            continue;
        };
        if capability == prefix || capability.starts_with(&format!("{prefix}.")) {
            match best {
                Some((best_rule, _)) if best_rule.len() >= rule.len() => {}
                _ => best = Some((rule.as_str(), *allowed)),
            }
        }
    }

    best.map(|(_, allowed)| allowed).unwrap_or(false)
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

const FS_STAT_HASH_KEY_PREFIX: &str = "_turin:fs_stat_hash:";

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

fn normalize_tracking_path(root: &Path, resolved: &Path) -> String {
    resolved
        .strip_prefix(root)
        .unwrap_or(resolved)
        .to_string_lossy()
        .replace('\\', "/")
}

fn current_session_tracking_context(
    lua: &Lua,
) -> Result<
    Option<(
        std::sync::Arc<crate::persistence::state::StateStore>,
        String,
    )>,
    String,
> {
    let Some(app_data) = lua.app_data_ref::<HarnessAppData>() else {
        return Ok(None);
    };

    let store_manager = app_data.store_manager.clone();
    let execution_ctx = app_data.execution_ctx.clone();
    let _ = app_data;

    let (session_id, store_selector) = execution_ctx
        .lock()
        .map_err(|_| "execution context mutex poisoned".to_string())
        .map(|lock| {
            (
                lock.session_id.clone(),
                lock.session_store_selector
                    .clone()
                    .unwrap_or_else(|| StoreSelector::Alias("state".to_string())),
            )
        })?;

    let Some(session_id) = session_id else {
        return Ok(None);
    };

    let store = block_on_current(async move {
        store_manager
            .open_with_path_scope(&store_selector, StorePathScope::AllowAny)
            .await
            .map_err(|err| err.to_string())
    })?;

    Ok(Some((store, encode_scope_key(&session_id, "default"))))
}

fn load_previous_session_hash(lua: &Lua, tracking_key: &str) -> Result<Option<String>, String> {
    let Some((store, session_scope_key)) = current_session_tracking_context(lua)? else {
        return Ok(None);
    };
    block_on_current(async move {
        store
            .kv_get("session", &session_scope_key, tracking_key)
            .await
            .map_err(|err| err.to_string())
    })
}

fn store_current_session_hash(lua: &Lua, tracking_key: &str, hash: &str) -> Result<(), String> {
    let Some((store, session_scope_key)) = current_session_tracking_context(lua)? else {
        return Ok(());
    };
    let tracking_key = tracking_key.to_string();
    let hash = hash.to_string();
    block_on_current(async move {
        store
            .kv_set("session", &session_scope_key, &tracking_key, &hash)
            .await
            .map_err(|err| err.to_string())
    })
}

fn register_fs_module(lua: &Lua, fs_root: &Path, max_file_size: usize) -> LuaResult<()> {
    let fs_table = lua.create_table()?;
    let root = fs_root.to_path_buf();

    let r1 = root.clone();
    fs_table.set(
        "read",
        lua.create_function(move |lua, path: String| {
            if let Err(err) = require_capability_for_lua(lua, "fs.read") {
                return nil_err(lua, &err.to_string());
            }
            match resolve_safe_path(&r1, &path) {
                Some(p) => match std::fs::read_to_string(&p) {
                    Ok(c) => string_ok(lua, &c),
                    Err(e) => nil_err(lua, &e.to_string()),
                },
                None => nil_err(lua, "Unsafe path traversal"),
            }
        })?,
    )?;

    let r2 = root.clone();
    fs_table.set(
        "write",
        lua.create_function(move |lua, (path, content): (String, String)| {
            if let Err(err) = require_capability_for_lua(lua, "fs.write") {
                return bool_err(lua, &err.to_string());
            }
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

    let r5 = root.clone();
    fs_table.set(
        "stat",
        lua.create_function(move |lua, path: String| {
            require_capability_for_lua(lua, "fs.read")?;

            let resolved = resolve_safe_path(&r5, &path)
                .ok_or_else(|| mlua::Error::runtime("Unsafe path traversal".to_string()))?;
            let bytes =
                std::fs::read(&resolved).map_err(|err| mlua::Error::runtime(err.to_string()))?;
            let metadata = std::fs::metadata(&resolved)
                .map_err(|err| mlua::Error::runtime(err.to_string()))?;
            let hash = hash_sha256_hex(&bytes);
            let tracking_path = normalize_tracking_path(&r5, &resolved);
            let tracking_key = format!("{FS_STAT_HASH_KEY_PREFIX}{tracking_path}");
            let previous_hash =
                load_previous_session_hash(lua, &tracking_key).map_err(mlua::Error::runtime)?;
            store_current_session_hash(lua, &tracking_key, &hash).map_err(mlua::Error::runtime)?;

            let stat = lua.create_table()?;
            stat.set("path", tracking_path)?;
            stat.set("bytes", bytes.len() as i64)?;
            stat.set("hash", hash.clone())?;
            stat.set("previous_hash", previous_hash.clone())?;
            stat.set("seen_before", previous_hash.is_some())?;
            stat.set(
                "changed",
                previous_hash
                    .as_ref()
                    .map(|previous| previous != &hash)
                    .unwrap_or(true),
            )?;

            if let Ok(modified) = metadata.modified()
                && let Ok(duration) = modified.duration_since(std::time::UNIX_EPOCH)
            {
                stat.set("modified_at", duration.as_secs() as i64)?;
            } else {
                stat.set("modified_at", Value::Nil)?;
            }

            Ok(Value::Table(stat))
        })?,
    )?;

    lua.globals().set("fs", fs_table)?;
    Ok(())
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
