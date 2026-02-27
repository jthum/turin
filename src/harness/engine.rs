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

use crate::display;
use crate::harness::globals::{self, HarnessAppData};
use crate::harness::verdict::{Verdict, compose_verdicts};

fn format_lua_error(e: &mlua::Error) -> String {
    let err_str = e.to_string();
    let ansi = display::stderr_ansi();

    // Attempt to parse standard Lua error format: "@path:line: message"
    // or mlua's "[string \"@path\"]:line: message"
    if let Some(first_colon) = err_str.find(':') {
        let prefix = &err_str[..first_colon];
        let rest = &err_str[first_colon + 1..];

        if let Some(second_colon) = rest.find(':') {
            let line_num = rest[..second_colon].trim();
            let message = rest[second_colon + 1..].trim();

            if line_num.chars().all(|c| c.is_ascii_digit()) {
                // Clean up the prefix (remove [string "@..."] wrapper if present)
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

/// The harness engine manages script loading and hook evaluation.
pub struct HarnessEngine {
    lua: Lua,
    /// Names of loaded scripts (in evaluation order)
    scripts: Vec<String>,
}

impl HarnessEngine {
    /// Create a new harness engine with sandboxed Luau VM.
    ///
    /// `app_data` provides the globals context (fs root, state store, etc.).
    pub fn new(app_data: HarnessAppData) -> Result<Self> {
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

        // Defense-in-depth: cap Lua memory at 32MB to prevent OOM from runaway scripts.
        const MAX_LUA_MEMORY: usize = 32 * 1024 * 1024;
        lua.set_memory_limit(MAX_LUA_MEMORY)?;

        Ok(Self {
            lua,
            scripts: Vec::new(),
        })
    }

    /// Load all `.lua` files from the given directory.
    ///
    /// Scripts are loaded in alphabetical order. Each script's hook functions
    /// are registered in the Lua environment. If the directory doesn't exist,
    /// no scripts are loaded (harness-free operation).
    pub fn load_dir(&mut self, dir: &Path) -> Result<()> {
        if !dir.exists() {
            return Ok(());
        }

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

        Ok(())
    }

    /// Load a single harness script by name.
    ///
    /// Each script's hook functions are captured into a per-module table
    /// (`__harness_modules[name]`). In sandboxed Luau, `function NAME(...)`
    /// writes to the chunk's local environment, so we reference hooks directly
    /// by name after executing the script source.
    fn load_script(&mut self, name: &str, source: &str, path: &Path) -> Result<()> {
        let globals = self.lua.globals();

        // Ensure __harness_modules exists in global registry if not already
        if !globals.contains_key("__harness_modules")? {
            globals.set("__harness_modules", self.lua.create_table()?)?;
        }
        if !globals.contains_key("__harness_module_meta")? {
            globals.set("__harness_module_meta", self.lua.create_table()?)?;
        }
        let modules: Table = globals.get("__harness_modules")?;
        let module_meta: Table = globals.get("__harness_module_meta")?;

        // Create a sandboxed environment for this script.
        // Writes go to 'env', reads fall back to 'globals' (via __index).
        let env = self.lua.create_table()?;
        let meta = self.lua.create_table()?;
        meta.set("__index", globals)?;
        let _ = env.set_metatable(Some(meta));

        // Load and execute string in the sandboxed environment, capturing return value
        let retval: Value = self
            .lua
            .load(source)
            .set_name(format!("@{}", path.display()))
            .set_environment(env.clone())
            .eval()
            .map_err(|e| {
                anyhow::anyhow!(format!(
                    "Failed to load harness script '{}':\n{}",
                    path.display(),
                    format_lua_error(&e)
                ))
            })?;

        // Extract known hooks: priority to return value (module table), fallback to env (globals)
        let module_exports = match retval {
            Value::Table(t) => t,
            _ => self.lua.create_table()?,
        };

        let known_hooks = [
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

        for hook in known_hooks {
            // If hook is already in exports (from return table), keep it.
            // Otherwise, check if it exists in the script's global env.
            if !module_exports.contains_key(hook)?
                && let Ok(func) = env.get::<Function>(hook)
            {
                module_exports.set(hook, func)?;
            }
        }

        // Register the module
        modules.set(name, module_exports)?;
        let meta = self.lua.create_table()?;
        meta.set("name", name)?;
        meta.set("path", path.to_string_lossy().to_string())?;
        if let Some(root_name) = self.resolve_governance_root_name(path) {
            meta.set("root", root_name)?;
        }
        module_meta.set(name, meta)?;
        self.scripts.push(name.to_string());
        Ok(())
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
        mode: Option<crate::kernel::config::AgentMode>,
    ) {
        if let Some(app_data) = self.lua.app_data_ref::<HarnessAppData>()
            && let Ok(mut lock) = app_data.execution_ctx.lock()
        {
            lock.session_id = session_id.map(|s| s.to_string());
            lock.session_mode = mode;
        }
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

    fn resolve_governance_root_name(&self, script_path: &Path) -> Option<String> {
        let app_data = self.lua.app_data_ref::<HarnessAppData>()?;
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

        for script_name in &self.scripts {
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
                    self.set_active_harness_module(Some(script_name));
                    let result = func.call::<MultiValue>(lua_payload.clone()).map_err(|e| {
                        self.set_active_harness_module(None);
                        anyhow::anyhow!(
                            "Harness '{}' hook '{}' failed:\n{}",
                            script_name,
                            hook_name,
                            format_lua_error(&e)
                        )
                    })?;
                    self.set_active_harness_module(None);

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

        for name in &self.scripts {
            if let Ok(module) = modules_table.get::<Table>(name.as_str())
                && let Ok(func) = module.get::<Function>(hook_name)
            {
                let ud = self.lua.create_userdata(data.clone()).map_err(|e| {
                    anyhow::anyhow!("Failed to create userdata for hook '{}': {}", hook_name, e)
                })?;

                self.set_active_harness_module(Some(name));
                match func.call::<MultiValue>(ud) {
                    Ok(result) => {
                        self.set_active_harness_module(None);
                        if let Ok(v) = parse_verdict(&self.lua, result) {
                            verdicts.push(v);
                        }
                    }
                    Err(e) => {
                        self.set_active_harness_module(None);
                        error!(hook = %hook_name, script = %name, "Error in harness hook:\n{}", format_lua_error(&e));
                    }
                }
            }
        }

        Ok(verdicts)
    }

    /// Get the names of loaded scripts.
    pub fn loaded_scripts(&self) -> &[String] {
        &self.scripts
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::manager::StoreManager;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tempfile::TempDir;

    fn test_app_data_for_root(root: PathBuf) -> HarnessAppData {
        HarnessAppData {
            fs_root: root.clone(),
            workspace_root: root.clone(),
            store_manager: Arc::new(StoreManager::new(root.clone())),
            agent_manager: Arc::new(crate::kernel::agent_manager::AgentManager::new(
                std::sync::Arc::new(crate::kernel::config::TurinConfig::default()),
                Arc::new(StoreManager::new(root)),
            )),
            policy_manager: Arc::new(crate::kernel::policy::RuntimePolicyManager::new()),
            governance_manager: Arc::new(crate::kernel::governance::GovernanceManager::new(
                crate::kernel::config::GovernanceConfig::default(),
            )),
            clients: std::collections::HashMap::new(),
            embedding_provider: None,
            queue: std::sync::Arc::new(tokio::sync::Mutex::new(Some(std::sync::Arc::new(
                tokio::sync::Mutex::new(std::collections::VecDeque::new()),
            )))),
            execution_ctx: std::sync::Arc::new(std::sync::Mutex::new(
                crate::harness::globals::HarnessExecutionContext {
                    session_id: Some("test-session".to_string()),
                    ..Default::default()
                },
            )),
            config: std::sync::Arc::new(crate::kernel::config::TurinConfig::default()),
            spawn_depth: 0,
        }
    }

    fn test_app_data() -> HarnessAppData {
        test_app_data_for_root(PathBuf::from("."))
    }

    #[test]
    fn test_engine_no_scripts() {
        let engine = HarnessEngine::new(test_app_data()).unwrap();
        assert!(engine.loaded_scripts().is_empty());

        let verdict = engine
            .evaluate("on_tool_call", serde_json::json!({}))
            .unwrap();
        assert_eq!(verdict, Verdict::Allow);
    }

    #[test]
    fn test_engine_load_empty_dir() {
        let dir = TempDir::new().unwrap();
        let mut engine = HarnessEngine::new(test_app_data()).unwrap();
        engine.load_dir(dir.path()).unwrap();
        assert!(engine.loaded_scripts().is_empty());
    }

    #[test]
    fn test_engine_load_nonexistent_dir() {
        let mut engine = HarnessEngine::new(test_app_data()).unwrap();
        engine.load_dir(Path::new("/nonexistent/path")).unwrap();
        assert!(engine.loaded_scripts().is_empty());
    }

    #[test]
    fn test_engine_allow_verdict() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("allow.lua"),
            r#"
            function on_tool_call(call)
                return ALLOW
            end
            "#,
        )
        .unwrap();

        let mut engine = HarnessEngine::new(test_app_data()).unwrap();
        engine.load_dir(dir.path()).unwrap();

        let verdict = engine
            .evaluate(
                "on_tool_call",
                serde_json::json!({"name": "read_file", "args": {}}),
            )
            .unwrap();
        assert_eq!(verdict, Verdict::Allow);
    }

    #[test]
    fn test_engine_reject_verdict() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("safety.lua"),
            r#"
            function on_tool_call(call)
                if call.name == "shell_exec" then
                    return REJECT, "Shell commands are not allowed"
                end
                return ALLOW
            end
            "#,
        )
        .unwrap();

        let mut engine = HarnessEngine::new(test_app_data()).unwrap();
        engine.load_dir(dir.path()).unwrap();

        // shell_exec should be rejected
        let verdict = engine
            .evaluate(
                "on_tool_call",
                serde_json::json!({"name": "shell_exec", "args": {"command": "ls"}}),
            )
            .unwrap();
        assert!(verdict.is_rejected());
        assert_eq!(verdict.reason(), Some("Shell commands are not allowed"));

        // read_file should be allowed
        let verdict = engine
            .evaluate(
                "on_tool_call",
                serde_json::json!({"name": "read_file", "args": {"path": "foo.txt"}}),
            )
            .unwrap();
        assert_eq!(verdict, Verdict::Allow);
    }

    #[test]
    fn test_engine_escalate_verdict() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("escalation.lua"),
            r#"
            function on_tool_call(call)
                if call.name == "write_file" then
                    return ESCALATE, "File writes need human approval"
                end
                return ALLOW
            end
            "#,
        )
        .unwrap();

        let mut engine = HarnessEngine::new(test_app_data()).unwrap();
        engine.load_dir(dir.path()).unwrap();

        let verdict = engine
            .evaluate(
                "on_tool_call",
                serde_json::json!({"name": "write_file", "args": {}}),
            )
            .unwrap();
        assert!(verdict.is_escalated());
        assert_eq!(verdict.reason(), Some("File writes need human approval"));
    }

    #[test]
    fn test_engine_composition_reject_wins() {
        let dir = TempDir::new().unwrap();
        // Script "a" allows everything
        std::fs::write(
            dir.path().join("a_permissive.lua"),
            r#"
            function on_tool_call(call)
                return ALLOW
            end
            "#,
        )
        .unwrap();

        // Script "b" rejects shell_exec
        std::fs::write(
            dir.path().join("b_safety.lua"),
            r#"
            function on_tool_call(call)
                if call.name == "shell_exec" then
                    return REJECT, "Blocked by safety harness"
                end
                return ALLOW
            end
            "#,
        )
        .unwrap();

        let mut engine = HarnessEngine::new(test_app_data()).unwrap();
        engine.load_dir(dir.path()).unwrap();

        assert_eq!(engine.loaded_scripts(), &["a_permissive", "b_safety"]);

        let verdict = engine
            .evaluate(
                "on_tool_call",
                serde_json::json!({"name": "shell_exec", "args": {}}),
            )
            .unwrap();
        assert_eq!(
            verdict,
            Verdict::Reject("Blocked by safety harness".to_string())
        );
    }

    #[test]
    fn test_engine_rm_rf_blocked() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("safety.lua"),
            r#"
            function on_tool_call(call)
                if call.name == "shell_exec" then
                    local cmd = call.args.command
                    if cmd and cmd:find("rm %-rf") then
                        return REJECT, "Destructive command 'rm -rf' is not allowed"
                    end
                end
                return ALLOW
            end
            "#,
        )
        .unwrap();

        let mut engine = HarnessEngine::new(test_app_data()).unwrap();
        engine.load_dir(dir.path()).unwrap();

        // rm -rf should be blocked
        let verdict = engine
            .evaluate(
                "on_tool_call",
                serde_json::json!({"name": "shell_exec", "args": {"command": "rm -rf /"}}),
            )
            .unwrap();
        assert_eq!(
            verdict,
            Verdict::Reject("Destructive command 'rm -rf' is not allowed".to_string())
        );

        // Safe commands should pass
        let verdict = engine
            .evaluate(
                "on_tool_call",
                serde_json::json!({"name": "shell_exec", "args": {"command": "ls -la"}}),
            )
            .unwrap();
        assert_eq!(verdict, Verdict::Allow);
    }

    #[test]
    fn test_engine_undefined_hook_returns_allow() {
        let dir = TempDir::new().unwrap();
        // Script only defines on_tool_call, not on_token_usage
        std::fs::write(
            dir.path().join("partial.lua"),
            r#"
            function on_tool_call(call)
                return ALLOW
            end
            "#,
        )
        .unwrap();

        let mut engine = HarnessEngine::new(test_app_data()).unwrap();
        engine.load_dir(dir.path()).unwrap();

        // Calling undefined hook should return ALLOW (no opinions)
        let verdict = engine
            .evaluate("on_token_usage", serde_json::json!({}))
            .unwrap();
        assert_eq!(verdict, Verdict::Allow);
    }

    #[test]
    fn test_engine_token_usage_hook() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("budget.lua"),
            r#"
            function on_token_usage(usage)
                if usage.total_cost_usd and usage.total_cost_usd > 1.0 then
                    return REJECT, "Budget exceeded: $" .. tostring(usage.total_cost_usd)
                end
                return ALLOW
            end
            "#,
        )
        .unwrap();

        let mut engine = HarnessEngine::new(test_app_data()).unwrap();
        engine.load_dir(dir.path()).unwrap();

        // Under budget
        let verdict = engine
            .evaluate(
                "on_token_usage",
                serde_json::json!({"total_cost_usd": 0.5, "input_tokens": 100, "output_tokens": 50}),
            )
            .unwrap();
        assert_eq!(verdict, Verdict::Allow);

        // Over budget
        let verdict = engine
            .evaluate(
                "on_token_usage",
                serde_json::json!({"total_cost_usd": 1.5, "input_tokens": 100, "output_tokens": 50}),
            )
            .unwrap();
        assert!(verdict.is_rejected());
    }

    #[test]
    fn test_engine_modify_verdict() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("modify.lua"),
            r#"
            function on_plan_submit(payload)
                return MODIFY, { "Modified Task 1", "Modified Task 2" }
            end
            "#,
        )
        .unwrap();

        let mut engine = HarnessEngine::new(test_app_data()).unwrap();
        engine.load_dir(dir.path()).unwrap();

        let verdict = engine
            .evaluate(
                "on_plan_submit",
                serde_json::json!({"action": "submit_plan"}),
            )
            .unwrap();

        match verdict {
            Verdict::Modify(val) => {
                let arr = val.as_array().unwrap();
                assert_eq!(arr.len(), 2);
                assert_eq!(arr[0].as_str().unwrap(), "Modified Task 1");
                assert_eq!(arr[1].as_str().unwrap(), "Modified Task 2");
            }
            _ => panic!("Expected Modify verdict, got {:?}", verdict),
        }
    }

    #[derive(Clone)]
    struct MockContext;
    impl mlua::UserData for MockContext {}

    #[test]
    fn test_on_turn_prepare_reject() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("reject.lua"),
            r#"
            function on_turn_prepare(ctx)
                return REJECT, "Blocked by harness"
            end
            "#,
        )
        .unwrap();

        let mut engine = HarnessEngine::new(test_app_data()).unwrap();
        engine.load_dir(dir.path()).unwrap();

        let verdict = engine
            .evaluate_userdata("on_turn_prepare", MockContext)
            .unwrap();
        assert!(verdict.is_rejected());
        assert_eq!(verdict.reason(), Some("Blocked by harness"));
    }

    #[test]
    fn test_verdict_helpers_support_or_chains() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("verdict_dx.lua"),
            r#"
            function on_tool_call(call)
                return verdict.reject_if(call.name == "shell_exec", "blocked by dx")
                    or verdict.allow()
            end
            "#,
        )
        .unwrap();

        let mut engine = HarnessEngine::new(test_app_data()).unwrap();
        engine.load_dir(dir.path()).unwrap();

        let verdict = engine
            .evaluate(
                "on_tool_call",
                serde_json::json!({"name": "shell_exec", "args": {}}),
            )
            .unwrap();
        assert!(verdict.is_rejected());
        assert_eq!(verdict.reason(), Some("blocked by dx"));
    }

    #[test]
    fn test_verdict_modify_helper() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("verdict_modify_dx.lua"),
            r#"
            function on_plan_submit(payload)
                return verdict.modify({ "A", "B" })
            end
            "#,
        )
        .unwrap();

        let mut engine = HarnessEngine::new(test_app_data()).unwrap();
        engine.load_dir(dir.path()).unwrap();

        let verdict = engine
            .evaluate(
                "on_plan_submit",
                serde_json::json!({ "action": "submit_plan" }),
            )
            .unwrap();

        match verdict {
            Verdict::Modify(val) => {
                let arr = val.as_array().unwrap();
                assert_eq!(arr.len(), 2);
                assert_eq!(arr[0].as_str().unwrap(), "A");
                assert_eq!(arr[1].as_str().unwrap(), "B");
            }
            other => panic!("Expected Modify verdict, got {:?}", other),
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_dx_access_helpers() {
        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("access_dx.lua"),
            r#"
            function on_turn_prepare(ctx)
                if not allowed("runtime.db.exec") then
                    return REJECT, "expected allowed in default config"
                end
                local decision = access.check("runtime.db.exec")
                if type(decision) ~= "table" then
                    return REJECT, "access.check did not return table"
                end
                needs("runtime.db.exec")
                return ALLOW
            end
            "#,
        )
        .unwrap();

        let mut engine = HarnessEngine::new(test_app_data()).unwrap();
        engine.load_dir(dir.path()).unwrap();
        let verdict = engine
            .evaluate_userdata("on_turn_prepare", MockContext)
            .unwrap();
        assert!(verdict.is_allowed());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_dx_session_user_kv_helpers() {
        let root = TempDir::new().unwrap();
        let mut engine =
            HarnessEngine::new(test_app_data_for_root(root.path().to_path_buf())).unwrap();

        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("data_dx.lua"),
            r#"
            function on_turn_prepare(ctx)
                session.set("counter", "0")
                local a = session.incr("counter")
                local b = session.incr("counter", 2)
                if a ~= 1 or b ~= 3 then
                    return REJECT, "session.incr mismatch"
                end

                if session.get("counter") ~= "3" then
                    return REJECT, "session.get mismatch"
                end

                user.set("tz", "UTC")
                if user.get("tz") ~= "UTC" then
                    return REJECT, "user.set/user.get mismatch"
                end
                user.del("tz")
                if user.get("tz") ~= nil then
                    return REJECT, "user.del mismatch"
                end
                return ALLOW
            end
            "#,
        )
        .unwrap();

        engine.load_dir(dir.path()).unwrap();
        let verdict = engine
            .evaluate_userdata("on_turn_prepare", MockContext)
            .unwrap();
        assert!(verdict.is_allowed());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_dx_runtime_db_proxy_one_and_with_error_precedence() {
        let root = TempDir::new().unwrap();
        let mut engine =
            HarnessEngine::new(test_app_data_for_root(root.path().to_path_buf())).unwrap();

        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("db_dx.lua"),
            r#"
            function on_turn_prepare(ctx)
                runtime.db.with("state", function(db)
                    db:exec("CREATE TABLE IF NOT EXISTS dx_users(id INTEGER PRIMARY KEY, name TEXT)")
                    db:exec("DELETE FROM dx_users")
                    db:exec("INSERT INTO dx_users(name) VALUES (?)", {"alice"})

                    local missing = db:one("SELECT name FROM dx_users WHERE id = ?", { 999 })
                    if missing ~= nil then
                        error("runtime.db:one should return nil when no rows")
                    end

                    local first = db:one("SELECT name FROM dx_users ORDER BY id LIMIT 1")
                    if first == nil or first.name ~= "alice" then
                        error("runtime.db:one returned wrong row")
                    end
                end)

                local ok, err = pcall(function()
                    runtime.db.with("state", function(db)
                        db:close()
                        error("callback error sentinel")
                    end)
                end)
                if ok then
                    return REJECT, "runtime.db.with should have failed"
                end
                if not tostring(err):find("callback error sentinel", 1, true) then
                    return REJECT, "runtime.db.with should prioritize callback error"
                end

                return ALLOW
            end
            "#,
        )
        .unwrap();

        engine.load_dir(dir.path()).unwrap();
        let verdict = engine
            .evaluate_userdata("on_turn_prepare", MockContext)
            .unwrap();
        assert!(verdict.is_allowed());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_dx_runtime_agent_status_proxy_and_fs_json_helpers() {
        let root = TempDir::new().unwrap();
        let mut engine =
            HarnessEngine::new(test_app_data_for_root(root.path().to_path_buf())).unwrap();

        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("agent_fs_dx.lua"),
            r#"
            function on_turn_prepare(ctx)
                local status = runtime.agent("default"):status()
                if status == nil or status.agent_id ~= "default" then
                    return REJECT, "runtime.agent(...):status() mismatch"
                end

                fs.write_json("dx-config.json", { enabled = true, count = 3 }, { pretty = true })
                local cfg = fs.read_json("dx-config.json")
                if cfg.enabled ~= true or cfg.count ~= 3 then
                    return REJECT, "fs.read_json/fs.write_json mismatch"
                end

                return ALLOW
            end
            "#,
        )
        .unwrap();

        engine.load_dir(dir.path()).unwrap();
        let verdict = engine
            .evaluate_userdata("on_turn_prepare", MockContext)
            .unwrap();
        assert!(verdict.is_allowed());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_dx_runtime_governance_grant_wrapper() {
        let mut engine = HarnessEngine::new(test_app_data()).unwrap();

        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("governance_dx.lua"),
            r#"
            function on_turn_prepare(ctx)
                local gid = nil
                local result = runtime.governance.grant({
                    ttl_ms = 5000,
                    capabilities = {
                        ["runtime.db.query"] = true,
                        ["runtime.governance.grant.get"] = true,
                    }
                }, function()
                    local query_dec = access.check("runtime.db.query")
                    if query_dec == nil or not query_dec.allowed then
                        error("runtime.db.query should be allowed inside grant")
                    end

                    local policy_dec = access.check("runtime.policy.set")
                    if policy_dec == nil then
                        error("runtime.policy.set decision missing")
                    end
                    if policy_dec.allowed then
                        error("runtime.policy.set should be denied by grant ceiling")
                    end

                    gid = query_dec.subject_grant_id
                    if gid == nil or gid == "" then
                        error("missing subject_grant_id in access decision")
                    end

                    return "grant_wrapper_ok"
                end)

                if result ~= "grant_wrapper_ok" then
                    return REJECT, "runtime.governance.grant result mismatch"
                end

                local grant, ge = runtime.governance.grant_get(gid)
                if grant ~= nil then
                    return REJECT, "grant should be revoked after callback returns"
                end
                if ge == nil then
                    return REJECT, "grant_get should report missing grant after revoke"
                end

                local ok, err = pcall(function()
                    runtime.governance.grant({
                        ttl_ms = 5000,
                        capabilities = {
                            ["runtime.db.query"] = true,
                            ["runtime.governance.grant.revoke"] = true,
                        }
                    }, function()
                        local dec = access.check("runtime.db.query")
                        local inner_gid = dec.subject_grant_id
                        local revoked, re = runtime.governance.grant_revoke(inner_gid)
                        if not revoked then
                            error("inner grant_revoke failed: " .. tostring(re))
                        end
                        error("grant callback sentinel")
                    end)
                end)
                if ok then
                    return REJECT, "runtime.governance.grant should fail when callback errors"
                end
                if not tostring(err):find("grant callback sentinel", 1, true) then
                    return REJECT, "runtime.governance.grant should prioritize callback error"
                end

                return ALLOW
            end
            "#,
        )
        .unwrap();

        engine.load_dir(dir.path()).unwrap();
        let verdict = engine
            .evaluate_userdata("on_turn_prepare", MockContext)
            .unwrap();
        assert!(verdict.is_allowed());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_dx_time_helpers() {
        let mut engine = HarnessEngine::new(test_app_data()).unwrap();

        let dir = TempDir::new().unwrap();
        std::fs::write(
            dir.path().join("time_dx.lua"),
            r#"
            function on_turn_prepare(ctx)
                local now = tonumber(time.now_utc())
                if now == nil then
                    return REJECT, "time.now_utc should be numeric string"
                end

                local since_num = time.since(now - 2)
                if since_num < 1 then
                    return REJECT, "time.since(number) should be positive"
                end

                local since_str = time.since(tostring(now - 1))
                if since_str < 0 then
                    return REJECT, "time.since(string) should parse numeric string"
                end

                if not time.after(now - 1, 0.5) then
                    return REJECT, "time.after should be true when elapsed >= threshold"
                end

                if time.after(now - 1, 10) then
                    return REJECT, "time.after should be false for large threshold"
                end

                return ALLOW
            end
            "#,
        )
        .unwrap();

        engine.load_dir(dir.path()).unwrap();
        let verdict = engine
            .evaluate_userdata("on_turn_prepare", MockContext)
            .unwrap();
        assert!(verdict.is_allowed());
    }
}
