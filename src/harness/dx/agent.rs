use mlua::{Function, Lua, Result as LuaResult, Table, Value};

use crate::harness::dx::common::{call_and_raise_on_err, normalize_capabilities_field};

#[derive(Clone)]
struct AgentRuntimeFns {
    submit: Function,
    sidestep: Function,
    await_task: Function,
    promote: Function,
    status: Function,
    complete: Function,
}

fn create_agent_proxy(lua: &Lua, agent_id: String, fns: AgentRuntimeFns) -> LuaResult<Table> {
    let proxy = lua.create_table()?;

    {
        let submit_fn = fns.submit.clone();
        let agent_id = agent_id.clone();
        proxy.set(
            "submit",
            lua.create_function(
                move |lua, (_self, task, opts): (Table, Value, Option<Table>)| {
                    if let Some(opts) = opts.as_ref() {
                        normalize_capabilities_field(lua, opts)?;
                    }
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
        let sidestep_fn = fns.sidestep.clone();
        let agent_id = agent_id.clone();
        proxy.set(
            "sidestep",
            lua.create_function(
                move |lua, (_self, prompt, opts): (Table, String, Option<Table>)| {
                    call_and_raise_on_err(
                        lua,
                        &sidestep_fn,
                        (agent_id.clone(), prompt, opts),
                        "runtime.agent.sidestep",
                    )
                },
            )?,
        )?;
    }

    {
        let await_fn = fns.await_task.clone();
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
        let promote_fn = fns.promote.clone();
        proxy.set(
            "promote",
            lua.create_function(
                move |lua, (_self, task_id, opts): (Table, String, Option<Table>)| {
                    call_and_raise_on_err(
                        lua,
                        &promote_fn,
                        (task_id, opts),
                        "runtime.agent.promote",
                    )
                },
            )?,
        )?;
    }

    {
        let status_fn = fns.status.clone();
        let agent_id = agent_id.clone();
        proxy.set(
            "status",
            lua.create_function(move |lua, _self: Table| {
                call_and_raise_on_err(lua, &status_fn, agent_id.clone(), "runtime.agent.status")
            })?,
        )?;
    }

    {
        let complete_fn = fns.complete.clone();
        let agent_id = agent_id.clone();
        proxy.set(
            "complete",
            lua.create_function(
                move |lua, (_self, prompt, opts): (Table, String, Option<Table>)| {
                    if let Some(opts) = opts.as_ref() {
                        normalize_capabilities_field(lua, opts)?;
                    }
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

    let fns = AgentRuntimeFns {
        submit: runtime_agent.get("submit")?,
        sidestep: runtime_agent.get("sidestep")?,
        await_task: runtime_agent.get("await")?,
        promote: runtime_agent.get("promote")?,
        status: runtime_agent.get("get_status")?,
        complete: runtime_agent.get("complete")?,
    };

    let mt = lua.create_table()?;
    mt.set(
        "__call",
        lua.create_function(move |lua, (_self, agent_id): (Value, String)| {
            create_agent_proxy(lua, agent_id, fns.clone())
        })?,
    )?;
    let _ = runtime_agent.set_metatable(Some(mt));

    Ok(())
}
