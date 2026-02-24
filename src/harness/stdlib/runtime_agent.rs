use std::collections::BTreeMap;

use mlua::{Lua, Result as LuaResult, Table, Value};

use crate::harness::globals::{HarnessAppData, block_on_current};
use crate::harness::stdlib::binding_common::{json_ok, nil_err, string_ok};
use crate::harness::stdlib::governance_support::{
    current_subject, require_capability as require_governance_capability,
    require_child_agent as require_child_agent_governance,
};
use crate::harness::stdlib::policy_support::{policy_bool, runtime_policy_snapshot};
use crate::kernel::session::QueuedTask;

fn parse_submit_task(task_val: Value) -> LuaResult<QueuedTask> {
    match task_val {
        Value::String(s) => Ok(QueuedTask::ad_hoc(s.to_str()?.to_string())),
        Value::Table(t) => {
            let prompt = t.get::<String>("prompt").map_err(|_| {
                mlua::Error::runtime("runtime.agent.submit task table requires prompt")
            })?;
            let mut task = QueuedTask::ad_hoc(prompt);
            if let Ok(title) = t.get::<String>("title") {
                task.title = Some(title);
            }
            Ok(task)
        }
        _ => Err(mlua::Error::runtime(
            "invalid task; expected string or {prompt=...}",
        )),
    }
}

fn parse_submit_delegated_capabilities(
    app_data: &HarnessAppData,
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
            let subject = current_subject(app_data);
            for pair in t.pairs::<String, Value>() {
                let (key, value) = pair?;
                if key.ends_with(".*") {
                    return Err(mlua::Error::runtime(format!(
                        "runtime.agent.submit opts.capabilities wildcard rules are not yet supported (key '{}')",
                        key
                    )));
                }
                let allowed = match value {
                    Value::Boolean(b) => b,
                    _ => {
                        return Err(mlua::Error::runtime(format!(
                            "runtime.agent.submit opts.capabilities values must be booleans (key '{}')",
                            key
                        )));
                    }
                };
                if allowed {
                    app_data
                        .governance_manager
                        .require_capability_for_subject(&subject, &key)
                        .map_err(mlua::Error::runtime)?;
                }
                caps.insert(key, allowed);
            }
            Ok(Some(caps))
        }
        _ => Err(mlua::Error::runtime(
            "runtime.agent.submit opts.capabilities must be a table".to_string(),
        )),
    }
}

pub fn register_runtime_agent_namespace(
    lua: &Lua,
    runtime_table: &Table,
    app_data: &HarnessAppData,
) -> LuaResult<()> {
    let runtime_agent = lua.create_table()?;
    {
        let manager = app_data.agent_manager.clone();
        let app_data_snapshot = app_data.clone();
        runtime_agent.set(
            "list",
            lua.create_function(move |lua, ()| {
                if let Err(err) =
                    require_governance_capability(&app_data_snapshot, "runtime.agent.status")
                {
                    return nil_err(lua, &err);
                }
                let manager = manager.clone();
                let statuses = block_on_current(async move { manager.list_statuses().await });
                json_ok(lua, &statuses)
            })?,
        )?;
    }
    {
        let manager = app_data.agent_manager.clone();
        let app_data_snapshot = app_data.clone();
        runtime_agent.set(
            "get_status",
            lua.create_function(move |lua, agent_id: String| {
                if let Err(err) =
                    require_governance_capability(&app_data_snapshot, "runtime.agent.status")
                {
                    return nil_err(lua, &err);
                }
                let manager = manager.clone();
                let status = block_on_current(async move { manager.get_status(&agent_id).await });
                match status {
                    Some(s) => json_ok(lua, &s),
                    None => nil_err(lua, "unknown agent"),
                }
            })?,
        )?;
    }
    {
        let manager = app_data.agent_manager.clone();
        let app_data_snapshot = app_data.clone();
        runtime_agent.set(
            "submit",
            lua.create_function(
                move |lua, (agent_id, task_val, opts): (String, Value, Option<Table>)| {
                    if let Err(err) =
                        require_governance_capability(&app_data_snapshot, "runtime.agent.submit")
                    {
                        return nil_err(lua, &err);
                    }
                    if let Err(err) = require_child_agent_governance(&app_data_snapshot, &agent_id)
                    {
                        return nil_err(lua, &err);
                    }
                    let snapshot = runtime_policy_snapshot(&app_data_snapshot)
                        .map_err(mlua::Error::runtime)?;
                    if !policy_bool(&snapshot, "spawn.enabled", true) {
                        return nil_err(lua, "Policy denial: spawn.enabled=false");
                    }

                    let task = match task_val {
                        v @ Value::String(_) | v @ Value::Table(_) => parse_submit_task(v)?,
                        _ => {
                            return nil_err(lua, "invalid task; expected string or {prompt=...}");
                        }
                    };
                    let delegated_capabilities =
                        parse_submit_delegated_capabilities(&app_data_snapshot, opts.as_ref())?;

                    let manager = manager.clone();
                    let result = block_on_current(async move {
                        manager
                            .submit(&agent_id, task, delegated_capabilities)
                            .await
                            .map_err(|e| e.to_string())
                    });
                    match result {
                        Ok(task_id) => string_ok(lua, &task_id),
                        Err(err) => nil_err(lua, &err),
                    }
                },
            )?,
        )?;
    }
    {
        let manager = app_data.agent_manager.clone();
        let app_data_snapshot = app_data.clone();
        runtime_agent.set(
            "await",
            lua.create_function(move |lua, (task_id, opts): (String, Option<Table>)| {
                if let Err(err) =
                    require_governance_capability(&app_data_snapshot, "runtime.agent.await")
                {
                    return nil_err(lua, &err);
                }
                let timeout_ms = opts.as_ref().and_then(|t| t.get::<u64>("timeout_ms").ok());
                let manager = manager.clone();
                let result = block_on_current(async move {
                    manager
                        .await_result(&task_id, timeout_ms)
                        .await
                        .map_err(|e| e.to_string())
                });
                match result {
                    Ok(res) => json_ok(lua, &res),
                    Err(err) => nil_err(lua, &err),
                }
            })?,
        )?;
    }
    runtime_table.set("agent", runtime_agent)?;
    Ok(())
}
