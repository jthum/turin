//! Harness Engine — loads and evaluates Luau harness scripts.
//!
//! The engine manages a sandboxed Luau VM, loads `.lua` files from a directory,
//! and evaluates hook functions against incoming events. Results are composed
//! using first-REJECT-wins semantics.

use anyhow::{Context, Result};
use mlua::{Function, Lua, LuaOptions, StdLib, Table, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::harness::globals::{self, HarnessAppData, HarnessExecutionBinding};
use crate::harness::stdlib::{action_bindings, runtime_signal, tool_bindings, ui_bindings};
use crate::harness::verdict::{Verdict, compose_verdicts};
use crate::harness::virtual_tools::{
    DeclaredVirtualTool, VirtualToolFollowUp, VirtualToolResultResolution,
};

#[path = "engine/hook_dispatch.rs"]
mod hook_dispatch;
#[path = "engine/loader.rs"]
mod loader;
#[cfg(test)]
#[path = "engine/tests.rs"]
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
        let source_overlay = self
            .lua
            .app_data_ref::<HarnessAppData>()
            .and_then(|app_data| app_data.source_overlay.clone());
        if !dir.exists() && source_overlay.is_none() {
            set_loading_phase(&self.lua, false);
            return Ok(());
        }

        clear_active_modules(&self.lua);

        let mut paths = BTreeSet::new();
        if dir.exists()
            && !source_overlay
                .as_ref()
                .is_some_and(|overlay| overlay.is_authoritative())
        {
            for entry in std::fs::read_dir(dir)
                .with_context(|| format!("Failed to read harness directory: {}", dir.display()))?
                .filter_map(|entry| entry.ok())
            {
                let path = entry.path();
                if path.extension().is_some_and(|extension| extension == "lua") {
                    paths.insert(PathBuf::from(entry.file_name()));
                }
            }
        }
        if let Some(overlay) = &source_overlay {
            for (path, present) in overlay.root_lua_paths() {
                if present {
                    paths.insert(path.to_path_buf());
                } else {
                    paths.remove(path);
                }
            }
        }

        for relative_path in paths {
            let path = dir.join(&relative_path);
            let name = path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            let source = match &source_overlay {
                Some(overlay) => overlay.read_to_string(dir, &path),
                None => std::fs::read_to_string(&path),
            }
            .with_context(|| format!("Failed to read harness script: {}", path.display()))?;

            if let Some(capture) = self
                .lua
                .app_data_ref::<HarnessAppData>()
                .and_then(|app_data| app_data.source_capture.clone())
            {
                let relative_path = path.strip_prefix(dir).with_context(|| {
                    format!("Harness source escaped its root: {}", path.display())
                })?;
                capture
                    .lock()
                    .map_err(|_| anyhow::anyhow!("Harness source capture mutex poisoned"))?
                    .insert(relative_path.to_path_buf(), Some(source.clone()));
            }

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
        data: impl mlua::UserData + Clone + Send + Sync + 'static,
    ) -> Result<Verdict> {
        let verdicts = self.call_hook_userdata(hook_name, data)?;
        Ok(compose_verdicts(&verdicts))
    }

    pub fn has_hook(&self, hook_name: &str) -> bool {
        let modules: Value = self
            .lua
            .globals()
            .get("__harness_modules")
            .unwrap_or(Value::Nil);
        let Value::Table(modules_table) = modules else {
            return false;
        };

        active_module_names(&self.lua).into_iter().any(|name| {
            modules_table
                .get::<Table>(name.as_str())
                .and_then(|module| module.get::<Function>(hook_name))
                .is_ok()
        })
    }

    /// Bind the full active execution context for the current task.
    pub fn bind_execution_context(&self, binding: HarnessExecutionBinding) {
        if let Some(app_data) = self.lua.app_data_ref::<HarnessAppData>()
            && let Ok(mut lock) = app_data.execution_ctx.lock()
        {
            lock.agent_id = Some(binding.agent_id);
            lock.session_id = Some(binding.session_id);
            lock.session_store_selector = Some(binding.store_selector);
            lock.default_store_selector = binding.default_store_selector;
            lock.execution_id = Some(binding.execution.execution_id);
            lock.execution_context_target = Some(binding.execution.context_target);
            lock.execution_visibility = Some(binding.execution.visibility);
            lock.execution_durability = Some(binding.execution.durability);
            lock.execution_write_policy = Some(binding.execution.write_policy);
            lock.execution_conflict_policy = Some(binding.execution.conflict_policy);
            lock.runtime_slot_id = binding.runtime_slot_id;
            lock.trace_id = Some(binding.trace_id);
            lock.completed_task_results = Some(binding.completed_task_results);
            lock.event_context = Some(binding.event_context);
            lock.cancel_token = Some(binding.cancel_token);
        }
    }

    /// Clear the active execution context after a task completes.
    pub fn unbind_execution_context(&self) {
        if let Some(app_data) = self.lua.app_data_ref::<HarnessAppData>()
            && let Ok(mut lock) = app_data.execution_ctx.lock()
        {
            lock.agent_id = None;
            lock.session_id = None;
            lock.session_store_selector = None;
            lock.default_store_selector = None;
            lock.execution_id = None;
            lock.execution_context_target = None;
            lock.execution_visibility = None;
            lock.execution_durability = None;
            lock.execution_write_policy = None;
            lock.execution_conflict_policy = None;
            lock.runtime_slot_id = None;
            lock.trace_id = None;
            lock.completed_task_results = None;
            lock.event_context = None;
            lock.cancel_token = None;
        }
    }

    pub fn set_active_queue(&self, queue: Option<crate::harness::globals::SessionQueue>) {
        if let Some(app_data) = self.lua.app_data_ref::<HarnessAppData>()
            && let Ok(mut lock) = app_data.execution_ctx.lock()
        {
            lock.queue = queue;
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

    /// Get the names of active hook-contributing scripts/modules.
    pub fn loaded_scripts(&self) -> Vec<String> {
        active_module_names(&self.lua)
    }

    pub fn explicit_watch_roots(&self) -> Vec<PathBuf> {
        explicit_watch_roots(&self.lua)
    }

    pub fn runtime_signal_topics(&self) -> Result<Vec<String>> {
        runtime_signal::runtime_signal_topics(&self.lua).map_err(anyhow::Error::from)
    }

    pub fn ui_intents(&self) -> Result<Vec<turin_daemon_protocol::UiIntentMessage>> {
        ui_bindings::ui_intents(&self.lua).map_err(anyhow::Error::from)
    }

    pub fn ui_intent_count(&self) -> Result<usize> {
        ui_bindings::ui_intent_count(&self.lua).map_err(anyhow::Error::from)
    }

    pub fn ui_intents_from(
        &self,
        start_index: usize,
    ) -> Result<Vec<turin_daemon_protocol::UiIntentMessage>> {
        ui_bindings::ui_intents_from(&self.lua, start_index).map_err(anyhow::Error::from)
    }

    pub fn declared_virtual_tools(&self) -> Result<Vec<DeclaredVirtualTool>> {
        tool_bindings::declared_virtual_tools(&self.lua)
    }

    pub fn invoke_declared_action_for_agent(
        &self,
        agent_id: &str,
        name: &str,
        params: serde_json::Value,
    ) -> Result<Option<serde_json::Value>> {
        let previous_agent = self
            .lua
            .app_data_ref::<HarnessAppData>()
            .and_then(|app_data| {
                app_data
                    .execution_ctx
                    .lock()
                    .ok()
                    .and_then(|lock| lock.agent_id.clone())
            });
        self.set_active_action_agent(Some(agent_id));
        let app_data = self
            .lua
            .app_data_ref::<HarnessAppData>()
            .map(|app_data| app_data.clone())
            .ok_or_else(|| anyhow::anyhow!("Harness app data missing"))?;
        let result = action_bindings::invoke_declared_action(
            &self.lua,
            name,
            params.clone(),
            action_bindings::ActionInvocationContext {
                app_data,
                action_name: name.to_string(),
                params,
                work_item: None,
            },
        );
        self.set_active_action_agent(previous_agent.as_deref());
        result
    }

    pub fn invoke_virtual_tool(
        &self,
        name: &str,
        args: serde_json::Value,
    ) -> Result<Option<VirtualToolResultResolution>> {
        tool_bindings::invoke_declared_virtual_tool(&self.lua, name, args)
    }

    pub fn virtual_tool_follow_up(&self, name: &str) -> Result<Option<VirtualToolFollowUp>> {
        Ok(self
            .declared_virtual_tools()?
            .into_iter()
            .find(|tool| tool.name == name)
            .map(|tool| tool.follow_up))
    }

    pub fn dispatch_runtime_signal(
        &self,
        signal: crate::kernel::harness_contract::HarnessSignal<'_>,
    ) -> Result<usize> {
        runtime_signal::dispatch_runtime_signal(&self.lua, signal).map_err(anyhow::Error::from)
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

    fn set_active_action_agent(&self, agent_id: Option<&str>) {
        if let Some(app_data) = self.lua.app_data_ref::<HarnessAppData>()
            && let Ok(mut lock) = app_data.execution_ctx.lock()
        {
            lock.agent_id = agent_id.map(|value| value.to_string());
        }
    }

    pub fn set_loading_phase(&self, is_loading: bool) {
        set_loading_phase(&self.lua, is_loading);
    }
}
