use mlua::{Function, Lua, Result as LuaResult, Table, Value};

use crate::harness::dx::common::call_and_raise_on_err;

fn create_agent_proxy(
    lua: &Lua,
    agent_id: String,
    submit_fn: Function,
    await_fn: Function,
    status_fn: Function,
) -> LuaResult<Table> {
    let proxy = lua.create_table()?;

    {
        let submit_fn = submit_fn.clone();
        let agent_id = agent_id.clone();
        proxy.set(
            "submit",
            lua.create_function(
                move |lua, (_self, task, opts): (Table, Value, Option<Table>)| {
                    call_and_raise_on_err(
                        lua,
                        &submit_fn,
                        (agent_id.clone(), task, opts),
                        "runtime.agent.submit",
                    )
                },
            )?,
        )?;
    }

    {
        let await_fn = await_fn.clone();
        proxy.set(
            "await",
            lua.create_function(
                move |lua, (_self, task_id, opts): (Table, String, Option<Table>)| {
                    call_and_raise_on_err(lua, &await_fn, (task_id, opts), "runtime.agent.await")
                },
            )?,
        )?;
    }

    {
        let status_fn = status_fn.clone();
        let agent_id = agent_id.clone();
        proxy.set(
            "status",
            lua.create_function(move |lua, _self: Table| {
                call_and_raise_on_err(lua, &status_fn, agent_id.clone(), "runtime.agent.status")
            })?,
        )?;
    }

    {
        let submit_fn = submit_fn.clone();
        let await_fn = await_fn.clone();
        let agent_id = agent_id.clone();
        proxy.set(
            "complete",
            lua.create_function(move |lua, (_self, prompt, opts): (Table, String, Option<Table>)| {
                let task = lua.create_table()?;
                task.set("prompt", prompt)?;
                if let Some(opts_tbl) = opts.as_ref()
                    && let Ok(title) = opts_tbl.get::<String>("title")
                {
                    task.set("title", title)?;
                }

                let request_id_value = call_and_raise_on_err(
                    lua,
                    &submit_fn,
                    (agent_id.clone(), Value::Table(task), opts.clone()),
                    "runtime.agent.submit",
                )?;
                let request_id = match request_id_value {
                    Value::String(s) => s.to_str()?.to_string(),
                    other => {
                        return Err(mlua::Error::runtime(format!(
                            "[runtime.agent.complete] submit returned non-string request id: {:?}",
                            other
                        )))
                    }
                };

                let awaited = call_and_raise_on_err(
                    lua,
                    &await_fn,
                    (request_id, opts),
                    "runtime.agent.await",
                )?;
                let result = match awaited {
                    Value::Table(t) => t,
                    other => {
                        return Err(mlua::Error::runtime(format!(
                            "[runtime.agent.complete] await returned non-table result: {:?}",
                            other
                        )))
                    }
                };

                if let Some(err) = result.get::<Option<String>>("error")? {
                    return Err(mlua::Error::runtime(format!(
                        "[runtime.agent.complete] {}",
                        err
                    )));
                }

                if let Some(output) = result.get::<Option<String>>("output")? {
                    Ok(Value::String(lua.create_string(&output)?))
                } else {
                    Ok(Value::String(lua.create_string("")?))
                }
            })?,
        )?;
    }

    Ok(proxy)
}

pub fn register_agent_dx(lua: &Lua) -> LuaResult<()> {
    let globals = lua.globals();
    let runtime: Table = globals.get("runtime")?;
    let runtime_agent: Table = runtime.get("agent")?;

    let submit_fn: Function = runtime_agent.get("submit")?;
    let await_fn: Function = runtime_agent.get("await")?;
    let status_fn: Function = runtime_agent.get("get_status")?;

    let mt = lua.create_table()?;
    mt.set(
        "__call",
        lua.create_function(move |lua, (_self, agent_id): (Value, String)| {
            create_agent_proxy(
                lua,
                agent_id,
                submit_fn.clone(),
                await_fn.clone(),
                status_fn.clone(),
            )
        })?,
    )?;
    let _ = runtime_agent.set_metatable(Some(mt));

    Ok(())
}
