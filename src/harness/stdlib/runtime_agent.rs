use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};

use crate::harness::globals::{
    HarnessAppData, block_on_current, policy_bool, runtime_policy_snapshot,
};
use crate::kernel::session::QueuedTask;

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
                let lua_v = lua
                    .to_value(&statuses)
                    .map_err(|e| mlua::Error::runtime(e.to_string()))?;
                Ok((lua_v, Value::Nil))
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
                    Some(s) => {
                        let lua_v = lua
                            .to_value(&s)
                            .map_err(|e| mlua::Error::runtime(e.to_string()))?;
                        Ok((lua_v, Value::Nil))
                    }
                    None => Ok((
                        Value::Nil,
                        Value::String(lua.create_string("unknown agent")?),
                    )),
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
                        return Ok((
                            Value::Nil,
                            Value::String(lua.create_string("Policy denial: spawn.enabled=false")?),
                        ));
                    }

                    let task = match task_val {
                        Value::String(s) => QueuedTask::ad_hoc(s.to_str()?.to_string()),
                        Value::Table(t) => {
                            let prompt = t.get::<String>("prompt").map_err(|_| {
                                mlua::Error::runtime(
                                    "runtime.agent.submit task table requires prompt",
                                )
                            })?;
                            let mut task = QueuedTask::ad_hoc(prompt);
                            if let Ok(title) = t.get::<String>("title") {
                                task.title = Some(title);
                            }
                            task
                        }
                        _ => {
                            return Ok((
                                Value::Nil,
                                Value::String(lua.create_string(
                                    "invalid task; expected string or {prompt=...}",
                                )?),
                            ));
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
                        Ok(task_id) => {
                            Ok((Value::String(lua.create_string(&task_id)?), Value::Nil))
                        }
                        Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
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
                    Ok(res) => {
                        let lua_v = lua
                            .to_value(&res)
                            .map_err(|e| mlua::Error::runtime(e.to_string()))?;
                        Ok((lua_v, Value::Nil))
                    }
                    Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                }
            })?,
        )?;
    }
    runtime_table.set("agent", runtime_agent)?;
    Ok(())
}
