//! Harness Engine — loads and evaluates Luau harness scripts.
//!
//! The engine manages a sandboxed Luau VM, loads `.lua` files from a directory,
//! and evaluates hook functions against incoming events. Results are composed
//! using first-REJECT-wins semantics.

use anyhow::{Context, Result};
use mlua::{Function, Lua, LuaOptions, LuaSerdeExt, MultiValue, StdLib, Table, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use tracing::error;

use crate::harness::globals::{self, HarnessAppData};
use crate::harness::stdlib::tool_bindings;
use crate::harness::verdict::{Verdict, compose_verdicts};
use crate::harness::virtual_tools::{
    DeclaredVirtualTool, VirtualToolPlan, VirtualToolResultResolution,
};

mod loader;
#[cfg(test)]
mod tests;

pub(crate) use loader::{
    ModuleLoadOptions, is_loading_phase, load_module_from_source,
    lookup_loaded_module_by_canonical_path, register_watch_root, resolve_governance_root_name,
};
use loader::{
    active_module_names, clear_active_modules, explicit_watch_roots, format_lua_error,
    set_loading_phase,
};

/// The harness engine manages script loading and hook evaluation.
pub struct HarnessEngine {
    lua: Lua,
}

pub(crate) const KNOWN_HOOKS: &[&str] = &[
    "on_tool_call",
    "on_tool_result",
    "on_token_usage",
    "on_session_start",
    "on_session_end",
    "on_task_start",
    "on_turn_start",
    "on_turn_prepare",
    "on_turn_end",
    "on_inference_error",
    "on_plan_submit",
    "on_task_complete",
    "on_plan_complete",
    "on_all_tasks_complete",
    "on_kernel_event",
];

impl HarnessEngine {
    /// Create a new harness engine with sandboxed Luau VM.
    ///
    /// `app_data` provides the globals context (fs root, state store, etc.).
    pub fn new(app_data: HarnessAppData) -> Result<Self> {
        let max_lua_memory = app_data.config.harness.memory_limit_mb as usize * 1024 * 1024;

        // Defense-in-depth: exclude IO, OS, FFI, PACKAGE standard library modules.
        // Even though sandbox() removes access to dangerous functions, excluding
        // them at VM creation ensures they cannot be reached even if sandbox is
        // bypassed by a future mlua/Luau vulnerability.
        let lua = Lua::new_with(StdLib::ALL_SAFE, LuaOptions::default())
            .map_err(|e| anyhow::anyhow!("Failed to create Luau VM: {}", e))?;

        // Register all Turin-SL globals before sandboxing.
        // This makes them available but read-only once sandbox is enabled.
        globals::register_globals(&lua, app_data)
            .map_err(|e| anyhow::anyhow!("Failed to register harness globals: {}", e))?;

        // Enable Luau sandboxing:
        // - All libraries and built-in metatables become read-only
        // - Globals become read-only
        // - Removes access to dangerous functions (os, io, loadfile, etc.)
        lua.sandbox(true)
            .map_err(|e| anyhow::anyhow!("Failed to enable Luau sandbox: {}", e))?;

        // Defense-in-depth: cap Lua memory to prevent OOM from runaway scripts.
        lua.set_memory_limit(max_lua_memory)?;

        Ok(Self { lua })
    }

    /// Load all `.lua` files from the given directory.
    ///
    /// Scripts are loaded in alphabetical order. Each script's hook functions
    /// are registered in the Lua environment. If the directory doesn't exist,
    /// no scripts are loaded (harness-free operation).
    pub fn load_dir(&mut self, dir: &Path) -> Result<()> {
        set_loading_phase(&self.lua, true);
        if !dir.exists() {
            set_loading_phase(&self.lua, false);
            return Ok(());
        }

        clear_active_modules(&self.lua);

        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .with_context(|| format!("Failed to read harness directory: {}", dir.display()))?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "lua")
                    .unwrap_or(false)
            })
            .collect();

        // Sort alphabetically for deterministic evaluation order
        entries.sort_by_key(|e| e.file_name());

        for entry in entries {
            let path = entry.path();
            let name = path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            let source = std::fs::read_to_string(&path)
                .with_context(|| format!("Failed to read harness script: {}", path.display()))?;

            self.load_script(&name, &source, &path)?;
        }

        set_loading_phase(&self.lua, false);
        Ok(())
    }

    /// Load a single harness script by name.
    ///
    /// Each script's hook functions are captured into a per-module table
    /// (`__harness_modules[name]`). In sandboxed Luau, `function NAME(...)`
    /// writes to the chunk's local environment, so we reference hooks directly
    /// by name after executing the script source.
    fn load_script(&mut self, name: &str, source: &str, path: &Path) -> Result<()> {
        load_module_from_source(
            &self.lua,
            name,
            source,
            path,
            ModuleLoadOptions {
                activate: true,
                block_name: Some(name.to_string()),
                block_config: None,
                when_fn: None,
                delegated_capabilities: None,
                cache_by_path: true,
            },
        )
    }

    /// Call a hook function across all loaded scripts and compose the verdicts.
    ///
    /// The hook receives a Lua table with the event payload. Each script's
    /// implementation of the hook (if any) is called in load order.
    /// Verdicts are composed using first-REJECT-wins semantics.
    pub fn evaluate(&self, hook_name: &str, payload: serde_json::Value) -> Result<Verdict> {
        let verdicts = self.call_hook(hook_name, payload)?;
        Ok(compose_verdicts(&verdicts))
    }

    /// Load and execute a Lua script string (for testing/verification).
    pub fn load_script_str(&mut self, script: &str) -> Result<()> {
        self.lua
            .load(script)
            .exec()
            .map_err(|e| anyhow::anyhow!(format_lua_error(&e)))?;
        Ok(())
    }

    /// Call a hook with a UserData argument (e.g. ContextWrapper).
    pub fn evaluate_userdata(
        &self,
        hook_name: &str,
        data: impl mlua::UserData + Clone + Send + 'static,
    ) -> Result<Verdict> {
        let verdicts = self.call_hook_userdata(hook_name, data)?;
        Ok(compose_verdicts(&verdicts))
    }

    /// Set the active session ID for the current execution context.
    /// This is used by global functions (e.g. turin.memory) to isolate data.
    pub fn set_active_session(
        &self,
        session_id: Option<&str>,
        store_selector: Option<crate::persistence::manager::StoreSelector>,
        default_store_selector: Option<crate::persistence::manager::StoreSelector>,
        mode: Option<crate::kernel::config::AgentMode>,
    ) {
        if let Some(app_data) = self.lua.app_data_ref::<HarnessAppData>()
            && let Ok(mut lock) = app_data.execution_ctx.lock()
        {
            lock.session_id = session_id.map(|s| s.to_string());
            lock.session_store_selector = store_selector;
            lock.default_store_selector = default_store_selector;
            lock.session_mode = mode;
        }
    }

    pub fn set_active_trace_id(&self, trace_id: Option<&str>) {
        if let Some(app_data) = self.lua.app_data_ref::<HarnessAppData>()
            && let Ok(mut lock) = app_data.execution_ctx.lock()
        {
            lock.trace_id = trace_id.map(|s| s.to_string());
        }
    }

    pub fn set_active_execution_metadata(
        &self,
        execution_id: Option<&str>,
        visibility: Option<crate::kernel::session::ExecutionVisibility>,
        durability: Option<crate::kernel::session::ExecutionDurability>,
        write_policy: Option<crate::kernel::session::ExecutionWritePolicy>,
    ) {
        if let Some(app_data) = self.lua.app_data_ref::<HarnessAppData>()
            && let Ok(mut lock) = app_data.execution_ctx.lock()
        {
            lock.execution_id = execution_id.map(|s| s.to_string());
            lock.execution_visibility = visibility;
            lock.execution_durability = durability;
            lock.execution_write_policy = write_policy;
        }
    }

    pub fn get_active_trace_id(&self) -> Option<String> {
        self.lua
            .app_data_ref::<HarnessAppData>()
            .and_then(|app_data| {
                app_data
                    .execution_ctx
                    .lock()
                    .ok()
                    .and_then(|lock| lock.trace_id.clone())
            })
    }

    pub fn set_active_queue(&self, queue: Option<crate::harness::globals::SessionQueue>) {
        if let Some(app_data) = self.lua.app_data_ref::<HarnessAppData>()
            && let Ok(mut lock) = app_data.execution_ctx.lock()
        {
            lock.queue = queue;
        }
    }

    pub fn request_active_session_branch_checkout(&self, branch: &str) {
        if let Some(app_data) = self.lua.app_data_ref::<HarnessAppData>()
            && let Ok(mut lock) = app_data.execution_ctx.lock()
        {
            lock.pending_branch_checkout = Some(branch.to_string());
        }
    }

    pub fn take_pending_session_branch_checkout(&self) -> Option<String> {
        self.lua
            .app_data_ref::<HarnessAppData>()
            .and_then(|app_data| {
                app_data
                    .execution_ctx
                    .lock()
                    .ok()
                    .and_then(|mut lock| lock.pending_branch_checkout.take())
            })
    }

    pub fn get_active_session_mode(&self) -> Option<crate::kernel::config::AgentMode> {
        if let Some(app_data) = self.lua.app_data_ref::<HarnessAppData>()
            && let Ok(lock) = app_data.execution_ctx.lock()
        {
            return lock.session_mode.clone();
        }
        None
    }

    pub fn set_active_capability_delegation(&self, caps: Option<BTreeMap<String, bool>>) {
        if let Some(app_data) = self.lua.app_data_ref::<HarnessAppData>()
            && let Ok(mut lock) = app_data.execution_ctx.lock()
        {
            lock.import_capabilities = caps;
        }
    }

    pub fn get_active_capability_delegation(&self) -> Option<BTreeMap<String, bool>> {
        self.lua
            .app_data_ref::<HarnessAppData>()
            .and_then(|app_data| {
                app_data
                    .execution_ctx
                    .lock()
                    .ok()
                    .and_then(|lock| lock.import_capabilities.clone())
            })
    }

    pub fn set_active_event_context(
        &self,
        ctx: Option<crate::harness::globals::HarnessEventContext>,
    ) {
        if let Some(app_data) = self.lua.app_data_ref::<HarnessAppData>()
            && let Ok(mut lock) = app_data.execution_ctx.lock()
        {
            lock.event_context = ctx;
        }
    }

    fn set_active_harness_module(&self, module_name: Option<&str>) {
        let root_name = module_name.and_then(|name| self.lookup_module_root_name(name));
        if let Some(app_data) = self.lua.app_data_ref::<HarnessAppData>()
            && let Ok(mut lock) = app_data.execution_ctx.lock()
        {
            lock.harness_module = module_name.map(|s| s.to_string());
            lock.harness_root = root_name;
        }
    }

    fn lookup_module_root_name(&self, module_name: &str) -> Option<String> {
        let globals = self.lua.globals();
        let module_meta: Table = globals.get("__harness_module_meta").ok()?;
        let meta: Table = module_meta.get(module_name).ok()?;
        meta.get::<String>("root").ok()
    }

    fn lookup_module_import_capabilities(
        &self,
        module_name: &str,
    ) -> Option<BTreeMap<String, bool>> {
        let globals = self.lua.globals();
        let module_meta: Table = globals.get("__harness_module_meta").ok()?;
        let meta: Table = module_meta.get(module_name).ok()?;
        let caps_value: Value = meta.get("delegated_capabilities").ok()?;
        self.lua.from_value(caps_value).ok()
    }

    /// Call a hook across all loaded scripts, returning individual verdicts.
    fn call_hook(&self, hook_name: &str, payload: serde_json::Value) -> Result<Vec<Verdict>> {
        let mut verdicts = Vec::new();

        let modules: Value = self
            .lua
            .globals()
            .get("__harness_modules")
            .unwrap_or(Value::Nil);

        let modules_table = match modules {
            Value::Table(t) => t,
            _ => return Ok(verdicts),
        };

        // Convert payload to Lua value
        let lua_payload = self
            .lua
            .to_value(&payload)
            .map_err(|e| anyhow::anyhow!("Failed to convert payload to Lua: {}", e))?;

        for script_name in active_module_names(&self.lua) {
            let module: Value = modules_table
                .get(script_name.as_str())
                .unwrap_or(Value::Nil);

            let module_table = match module {
                Value::Table(t) => t,
                _ => continue,
            };

            let hook_fn: Value = module_table.get(hook_name).unwrap_or(Value::Nil);

            match hook_fn {
                Value::Function(func) => {
                    let prev_caps = self.get_active_capability_delegation();
                    let module_caps = self.lookup_module_import_capabilities(&script_name);
                    let effective_caps = module_caps.or_else(|| prev_caps.clone());
                    self.set_active_harness_module(Some(&script_name));
                    self.set_active_capability_delegation(effective_caps);
                    let result = func.call::<MultiValue>(lua_payload.clone()).map_err(|e| {
                        self.set_active_harness_module(None);
                        self.set_active_capability_delegation(prev_caps.clone());
                        anyhow::anyhow!(
                            "Harness '{}' hook '{}' failed:\n{}",
                            script_name,
                            hook_name,
                            format_lua_error(&e)
                        )
                    })?;
                    self.set_active_harness_module(None);
                    self.set_active_capability_delegation(prev_caps);

                    let verdict = parse_verdict(&self.lua, result)?;
                    verdicts.push(verdict);
                }
                _ => {
                    // Hook not defined in this script — skip (implicit ALLOW)
                    continue;
                }
            }
        }

        Ok(verdicts)
    }

    /// Call a hook with UserData, returning individual verdicts.
    fn call_hook_userdata(
        &self,
        hook_name: &str,
        data: impl mlua::UserData + Clone + Send + 'static,
    ) -> Result<Vec<Verdict>> {
        let mut verdicts = Vec::new();

        let modules: Value = self
            .lua
            .globals()
            .get("__harness_modules")
            .unwrap_or(Value::Nil);

        let modules_table = match modules {
            Value::Table(t) => t,
            _ => return Ok(verdicts),
        };

        for name in active_module_names(&self.lua) {
            if let Ok(module) = modules_table.get::<Table>(name.as_str())
                && let Ok(func) = module.get::<Function>(hook_name)
            {
                let ud = self.lua.create_userdata(data.clone()).map_err(|e| {
                    anyhow::anyhow!("Failed to create userdata for hook '{}': {}", hook_name, e)
                })?;

                let prev_caps = self.get_active_capability_delegation();
                let module_caps = self.lookup_module_import_capabilities(&name);
                let effective_caps = module_caps.or_else(|| prev_caps.clone());
                self.set_active_harness_module(Some(&name));
                self.set_active_capability_delegation(effective_caps);
                match func.call::<MultiValue>(ud) {
                    Ok(result) => {
                        self.set_active_harness_module(None);
                        self.set_active_capability_delegation(prev_caps);
                        if let Ok(v) = parse_verdict(&self.lua, result) {
                            verdicts.push(v);
                        }
                    }
                    Err(e) => {
                        self.set_active_harness_module(None);
                        self.set_active_capability_delegation(prev_caps);
                        error!(hook = %hook_name, script = %name, "Error in harness hook:\n{}", format_lua_error(&e));
                    }
                }
            }
        }

        Ok(verdicts)
    }

    /// Get the names of active hook-contributing scripts/modules.
    pub fn loaded_scripts(&self) -> Vec<String> {
        active_module_names(&self.lua)
    }

    pub fn explicit_watch_roots(&self) -> Vec<PathBuf> {
        explicit_watch_roots(&self.lua)
    }

    pub fn declared_virtual_tools(&self) -> Result<Vec<DeclaredVirtualTool>> {
        tool_bindings::declared_virtual_tools(&self.lua)
    }

    pub fn invoke_virtual_tool(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<Option<VirtualToolPlan>> {
        tool_bindings::invoke_declared_virtual_tool(&self.lua, name, args)
    }

    pub fn invoke_virtual_tool_result_handler(
        &self,
        key: &str,
        payload: serde_json::Value,
        default_is_error: bool,
    ) -> Result<VirtualToolResultResolution> {
        tool_bindings::invoke_virtual_result_handler(&self.lua, key, payload, default_is_error)
    }

    pub fn discard_virtual_tool_result_handler(&self, key: &str) -> Result<()> {
        tool_bindings::discard_virtual_result_handler(&self.lua, key)
    }

    pub fn set_loading_phase(&self, is_loading: bool) {
        set_loading_phase(&self.lua, is_loading);
    }
}

/// Parse a Lua return value into a Verdict.
///
/// Convention:
///   return ALLOW              → Verdict::Allow
///   return REJECT, "reason"   → Verdict::Reject(reason)
///   return ESCALATE, "reason" → Verdict::Escalate(reason)
///   return MODIFY, {new_data} → Verdict::Modify(json_data)
fn parse_verdict(lua: &Lua, values: MultiValue) -> Result<Verdict> {
    let mut iter = values.into_iter();

    let first = iter.next();

    let (verdict_code, first_payload) = match first {
        Some(Value::Integer(n)) => (n, iter.next()),
        Some(Value::Table(t)) => {
            let verdict_code = t
                .get::<i64>("code")
                .map_err(|_| anyhow::anyhow!("Harness verdict table is missing integer 'code'"))?;
            let reason = t.get::<Option<String>>("reason").ok().flatten();
            let payload = if let Some(v) = t
                .get::<Value>("value")
                .ok()
                .filter(|v| !matches!(v, Value::Nil))
            {
                Some(v)
            } else if let Some(reason) = reason {
                Some(Value::String(lua.create_string(&reason)?))
            } else {
                None
            };
            (verdict_code, payload)
        }
        Some(Value::Nil) | None => return Ok(Verdict::Allow), // No return = ALLOW
        other => {
            return Err(anyhow::anyhow!(
                "Harness hook returned non-integer verdict: {:?}",
                other
            ));
        }
    };

    match verdict_code {
        1 => Ok(Verdict::Allow),
        2 | 3 => {
            let reason = match first_payload {
                Some(Value::String(s)) => s
                    .to_str()
                    .map_err(|e| anyhow::anyhow!("Invalid UTF-8 in verdict reason: {}", e))?
                    .to_string(),
                _ => String::new(),
            };
            if verdict_code == 2 {
                Ok(Verdict::Reject(reason))
            } else {
                Ok(Verdict::Escalate(reason))
            }
        }
        4 => {
            let val = match first_payload {
                Some(v) => lua.from_value::<serde_json::Value>(v).map_err(|e| {
                    anyhow::anyhow!("Failed to convert MODIFY value to JSON: {}", e)
                })?,
                None => serde_json::Value::Null,
            };
            Ok(Verdict::Modify(val))
        }
        _ => Err(anyhow::anyhow!("Unknown verdict code: {}", verdict_code)),
    }
}
