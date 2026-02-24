use mlua::{Lua, Result as LuaResult, Table, Value};

use crate::harness::globals::{HarnessAppData, block_on_current};
use crate::harness::stdlib::binding_common::{json_ok, nil_err, string_ok};
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

pub fn register_runtime_agent_namespace(
    lua: &Lua,
    runtime_table: &Table,
    app_data: &HarnessAppData,
) -> LuaResult<()> {
    let runtime_agent = lua.create_table()?;
    {
        let manager = app_data.agent_manager.clone();
        runtime_agent.set(
            "list",
            lua.create_function(move |lua, ()| {
                let manager = manager.clone();
                let statuses = block_on_current(async move { manager.list_statuses().await });
                json_ok(lua, &statuses)
            })?,
        )?;
    }
    {
        let manager = app_data.agent_manager.clone();
        runtime_agent.set(
            "get_status",
            lua.create_function(move |lua, agent_id: String| {
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
                move |lua, (agent_id, task_val, _opts): (String, Value, Option<Table>)| {
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

                    let manager = manager.clone();
                    let result = block_on_current(async move {
                        manager
                            .submit(&agent_id, task)
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
        runtime_agent.set(
            "await",
            lua.create_function(move |lua, (task_id, opts): (String, Option<Table>)| {
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
