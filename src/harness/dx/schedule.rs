use mlua::{Function, Lua, Result as LuaResult, Table, Value};

use crate::harness::dx::common::call_and_raise_on_err;

fn normalize_schedule_opts_in_place(lua: &Lua, merged: &Table) -> LuaResult<()> {
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

    Ok(())
}

fn merge_schedule_opts(
    lua: &Lua,
    payload: Value,
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
    match payload {
        Value::String(prompt) => merged.set("prompt", prompt)?,
        Value::Table(table) => {
            for pair in table.pairs::<Value, Value>() {
                let (key, value) = pair?;
                merged.set(key, value)?;
            }
        }
        other => {
            return Err(mlua::Error::runtime(format!(
                "schedule payload must be string or table, got {:?}",
                other.type_name()
            )));
        }
    }
    merged.set(time_field, time_value)?;
    normalize_schedule_opts_in_place(lua, &merged)?;
    Ok(merged)
}

pub fn register_schedule_dx(lua: &Lua) -> LuaResult<()> {
    let globals = lua.globals();
    let runtime: Table = globals.get("runtime")?;
    let runtime_schedule: Table = runtime.get("schedule")?;

    let create_fn: Function = runtime_schedule.get("create")?;
    let update_fn: Function = runtime_schedule.get("update")?;
    let get_fn: Function = runtime_schedule.get("get")?;
    let list_fn: Function = runtime_schedule.get("list")?;
    let enable_fn: Function = runtime_schedule.get("enable")?;
    let disable_fn: Function = runtime_schedule.get("disable")?;
    let delete_fn: Function = runtime_schedule.get("delete")?;

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
        let update_fn = update_fn.clone();
        schedule.set(
            "update",
            lua.create_function(move |lua, (public_id, opts): (String, Option<Table>)| {
                let merged = lua.create_table()?;
                if let Some(opts) = opts {
                    for pair in opts.pairs::<Value, Value>() {
                        let (key, value) = pair?;
                        merged.set(key, value)?;
                    }
                }
                merged.set("id", public_id)?;
                normalize_schedule_opts_in_place(lua, &merged)?;
                call_and_raise_on_err(lua, &update_fn, merged, "runtime.schedule.update")
            })?,
        )?;
    }

    {
        let create_fn = create_fn.clone();
        schedule.set(
            "after",
            lua.create_function(
                move |lua, (seconds, payload, opts): (Value, Value, Option<Table>)| {
                    let merged = merge_schedule_opts(lua, payload, "after_seconds", seconds, opts)?;
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
                move |lua, (seconds, payload, opts): (Value, Value, Option<Table>)| {
                    let merged =
                        merge_schedule_opts(lua, payload, "interval_seconds", seconds, opts)?;
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
                move |lua, (next_run, payload, opts): (Value, Value, Option<Table>)| {
                    let merged = merge_schedule_opts(lua, payload, "next_run", next_run, opts)?;
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

    {
        let get_fn = get_fn.clone();
        schedule.set(
            "get",
            lua.create_function(move |lua, public_id: String| {
                call_and_raise_on_err(lua, &get_fn, public_id, "runtime.schedule.get")
            })?,
        )?;
    }

    {
        let enable_fn = enable_fn.clone();
        schedule.set(
            "enable",
            lua.create_function(move |lua, public_id: String| {
                call_and_raise_on_err(lua, &enable_fn, public_id, "runtime.schedule.enable")
            })?,
        )?;
    }

    {
        let disable_fn = disable_fn.clone();
        schedule.set(
            "disable",
            lua.create_function(move |lua, public_id: String| {
                call_and_raise_on_err(lua, &disable_fn, public_id, "runtime.schedule.disable")
            })?,
        )?;
    }

    {
        let delete_fn = delete_fn.clone();
        schedule.set(
            "delete",
            lua.create_function(move |lua, public_id: String| {
                call_and_raise_on_err(lua, &delete_fn, public_id, "runtime.schedule.delete")
            })?,
        )?;
    }

    globals.set("schedule", schedule)?;
    Ok(())
}
