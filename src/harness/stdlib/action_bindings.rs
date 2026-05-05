use anyhow::Result;
use mlua::{Function, Lua, LuaSerdeExt, Result as LuaResult, Table, Value};

use crate::harness::stdlib::system_globals::ensure_load_time;

const DECLARED_ACTION_REGISTRY_KEY: &str = "__harness_declared_actions";

pub fn register_action_globals(lua: &Lua) -> LuaResult<()> {
    let action_table = lua.create_table()?;

    action_table.set(
        "define",
        lua.create_function(|lua, (name, handler): (String, Function)| {
            ensure_load_time(lua, "action.define")?;

            let registry = ensure_declared_action_registry(lua)?;
            if registry.contains_key(name.clone())? {
                return Err(mlua::Error::runtime(format!(
                    "action.define('{}') conflicts with an existing declared action",
                    name
                )));
            }

            registry.set(name, handler)?;
            Ok(())
        })?,
    )?;

    lua.globals().set("action", action_table)?;
    Ok(())
}

fn ensure_declared_action_registry(lua: &Lua) -> LuaResult<Table> {
    let globals = lua.globals();
    if !globals.contains_key(DECLARED_ACTION_REGISTRY_KEY)? {
        globals.set(DECLARED_ACTION_REGISTRY_KEY, lua.create_table()?)?;
    }
    globals.get(DECLARED_ACTION_REGISTRY_KEY)
}

pub(crate) fn invoke_declared_action(
    lua: &Lua,
    name: &str,
    params: serde_json::Value,
) -> Result<Option<serde_json::Value>> {
    let registry = ensure_declared_action_registry(lua)?;
    let handler = match registry.get::<Value>(name)? {
        Value::Nil => return Ok(None),
        Value::Function(function) => function,
        other => anyhow::bail!(
            "declared action registry entry '{}' has invalid type {:?}",
            name,
            other
        ),
    };

    let lua_args = lua.to_value(&params)?;
    let result = handler.call::<Value>(lua_args)?;
    let result = lua.from_value::<serde_json::Value>(result).map_err(|err| {
        anyhow::anyhow!(
            "declared action '{}' handler returned a non-JSON value: {}",
            name,
            err
        )
    })?;

    Ok(Some(result))
}
