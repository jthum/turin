use mlua::{Function, Lua, Result as LuaResult, Table, Value};

use crate::harness::dx::common::call_and_raise_on_err;

fn merge_schedule_opts(
    lua: &Lua,
    prompt: String,
    time_field: &str,
    time_value: Value,
    opts: Option<Table>,
) -> LuaResult<Table> {
    let merged = lua.create_table()?;
    if let Some(opts) = opts {
        for pair in opts.pairs::<Value, Value>() {
            let (key, value) = pair?;
            merged.set(key, value)?;
        }
    }
    merged.set("prompt", prompt)?;
    merged.set(time_field, time_value)?;

    if let Value::String(overlap) = merged.get::<Value>("overlap")? {
        merged.set("overlap_policy", overlap)?;
    }

    if let Value::Nil = merged.get::<Value>("persistence")? {
        let persistence = lua.create_table()?;
        let mut has_persistence = false;

        match merged.get::<Value>("state")? {
            Value::Nil => {}
            value => {
                persistence.set("state", value)?;
                has_persistence = true;
            }
        }
        match merged.get::<Value>("store")? {
            Value::Nil => {}
            value => {
                persistence.set("store", value)?;
                has_persistence = true;
            }
        }

        if has_persistence {
            merged.set("persistence", persistence)?;
        }
    }

    Ok(merged)
}

pub fn register_schedule_dx(lua: &Lua) -> LuaResult<()> {
    let globals = lua.globals();
    let runtime: Table = globals.get("runtime")?;
    let runtime_schedule: Table = runtime.get("schedule")?;

    let create_fn: Function = runtime_schedule.get("create")?;
    let list_fn: Function = runtime_schedule.get("list")?;

    let schedule = lua.create_table()?;

    {
        let create_fn = create_fn.clone();
        schedule.set(
            "create",
            lua.create_function(move |lua, opts: Table| {
                call_and_raise_on_err(lua, &create_fn, opts, "runtime.schedule.create")
            })?,
        )?;
    }

    {
        let create_fn = create_fn.clone();
        schedule.set(
            "after",
            lua.create_function(
                move |lua, (seconds, prompt, opts): (Value, String, Option<Table>)| {
                    let merged = merge_schedule_opts(lua, prompt, "after_seconds", seconds, opts)?;
                    call_and_raise_on_err(lua, &create_fn, merged, "runtime.schedule.create")
                },
            )?,
        )?;
    }

    {
        let create_fn = create_fn.clone();
        schedule.set(
            "every",
            lua.create_function(
                move |lua, (seconds, prompt, opts): (Value, String, Option<Table>)| {
                    let merged =
                        merge_schedule_opts(lua, prompt, "interval_seconds", seconds, opts)?;
                    call_and_raise_on_err(lua, &create_fn, merged, "runtime.schedule.create")
                },
            )?,
        )?;
    }

    {
        let create_fn = create_fn.clone();
        schedule.set(
            "at",
            lua.create_function(
                move |lua, (unix_ms, prompt, opts): (Value, String, Option<Table>)| {
                    let merged =
                        merge_schedule_opts(lua, prompt, "next_run_unix_ms", unix_ms, opts)?;
                    call_and_raise_on_err(lua, &create_fn, merged, "runtime.schedule.create")
                },
            )?,
        )?;
    }

    {
        let list_fn = list_fn.clone();
        schedule.set(
            "list",
            lua.create_function(move |lua, opts: Option<Table>| {
                call_and_raise_on_err(lua, &list_fn, opts, "runtime.schedule.list")
            })?,
        )?;
    }

    globals.set("schedule", schedule)?;
    Ok(())
}
