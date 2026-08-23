use std::collections::BTreeMap;

use anyhow::Result;
use mlua::{Function, Lua, LuaSerdeExt, MultiValue, Table, Value};
use tracing::error;

use crate::harness::globals::HarnessAppData;
use crate::harness::stdlib::ui_bindings;
use crate::harness::verdict::Verdict;
use crate::kernel::event::{KernelEvent, UiEvent};

use super::HarnessEngine;
use super::loader::{active_module_names, format_lua_error};

impl HarnessEngine {
    fn set_active_hook_context(&self, context: Option<mlua::AnyUserData>) {
        let globals = self.lua.globals();
        let value = match context {
            Some(context) => Value::UserData(context),
            None => Value::Nil,
        };
        let _ = globals.set("__current_hook_context", value);
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
    pub(super) fn call_hook(
        &self,
        hook_name: &str,
        payload: serde_json::Value,
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
                    let ui_start = ui_bindings::ui_intent_count(&self.lua)?;
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
                    self.emit_ui_intents_since(ui_start)?;
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
    pub(super) fn call_hook_userdata(
        &self,
        hook_name: &str,
        data: impl mlua::UserData + Clone + Send + Sync + 'static,
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
                let ui_start = ui_bindings::ui_intent_count(&self.lua)?;
                self.set_active_harness_module(Some(&name));
                self.set_active_capability_delegation(effective_caps);
                self.set_active_hook_context(Some(ud.clone()));
                match func.call::<MultiValue>(ud) {
                    Ok(result) => {
                        self.set_active_harness_module(None);
                        self.set_active_capability_delegation(prev_caps);
                        self.set_active_hook_context(None);
                        if let Ok(v) = parse_verdict(&self.lua, result) {
                            self.emit_ui_intents_since(ui_start)?;
                            verdicts.push(v);
                        }
                    }
                    Err(e) => {
                        self.set_active_harness_module(None);
                        self.set_active_capability_delegation(prev_caps);
                        self.set_active_hook_context(None);
                        error!(hook = %hook_name, script = %name, "Error in harness hook:\n{}", format_lua_error(&e));
                    }
                }
            }
        }

        Ok(verdicts)
    }

    fn emit_ui_intents_since(&self, start_index: usize) -> Result<()> {
        let intents = ui_bindings::ui_intents_from(&self.lua, start_index)?;
        if intents.is_empty() {
            return Ok(());
        }

        let Some(app_data) = self.lua.app_data_ref::<HarnessAppData>() else {
            return Ok(());
        };

        let Some((agent_id, internal_id, event_tx)) =
            app_data.execution_ctx.lock().ok().and_then(|lock| {
                lock.event_context
                    .as_ref()
                    .map(|ctx| (lock.agent_id.clone(), ctx.internal_id, ctx.event_tx.clone()))
            })
        else {
            return Ok(());
        };

        for mut intent in intents {
            if intent.source.agent_id.is_none() {
                intent.source.agent_id.clone_from(&agent_id);
            }
            let _ = event_tx.send((internal_id, KernelEvent::Ui(UiEvent::Intent { intent })));
        }

        Ok(())
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
