use mlua::{Function, Lua, Result as LuaResult, Table, Value};

use crate::harness::dx::common::call_and_raise_on_err;

fn create_agent_proxy(
    lua: &Lua,
    agent_id: String,
    submit_fn: Function,
    await_fn: Function,
    status_fn: Function,
    complete_fn: Function,
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
        let complete_fn = complete_fn.clone();
        let agent_id = agent_id.clone();
        proxy.set(
            "complete",
            lua.create_function(
                move |lua, (_self, prompt, opts): (Table, String, Option<Table>)| {
                    call_and_raise_on_err(
                        lua,
                        &complete_fn,
                        (agent_id.clone(), prompt, opts),
                        "runtime.agent.complete",
                    )
                },
            )?,
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
    let complete_fn: Function = runtime_agent.get("complete")?;

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
                complete_fn.clone(),
            )
        })?,
    )?;
    let _ = runtime_agent.set_metatable(Some(mt));

    Ok(())
}
