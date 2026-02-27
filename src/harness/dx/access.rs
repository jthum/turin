use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};

use crate::harness::globals::HarnessAppData;
use crate::harness::stdlib::governance_support::{capability_decision, require_capability};

fn read_agent_id(opts: &Option<Table>) -> LuaResult<Option<String>> {
    if let Some(opts) = opts {
        Ok(opts.get::<Option<String>>("agent_id")?)
    } else {
        Ok(None)
    }
}

pub fn register_access_globals(lua: &Lua, app_data: &HarnessAppData) -> LuaResult<()> {
    {
        let app_data_snapshot = app_data.clone();
        lua.globals().set(
            "allowed",
            lua.create_function(move |_lua, (capability, opts): (String, Option<Table>)| {
                if capability.trim().is_empty() {
                    return Err(mlua::Error::runtime("capability must not be empty"));
                }
                let decision = if let Some(agent_id) = read_agent_id(&opts)? {
                    app_data_snapshot
                        .governance_manager
                        .capability_decision(Some(agent_id.as_str()), &capability)
                } else {
                    capability_decision(&app_data_snapshot, &capability)
                };
                Ok(decision.allowed)
            })?,
        )?;
    }

    {
        let app_data_snapshot = app_data.clone();
        lua.globals().set(
            "needs",
            lua.create_function(move |_lua, (capability, opts): (String, Option<Table>)| {
                if capability.trim().is_empty() {
                    return Err(mlua::Error::runtime("capability must not be empty"));
                }
                if let Some(agent_id) = read_agent_id(&opts)? {
                    app_data_snapshot
                        .governance_manager
                        .require_capability(Some(agent_id.as_str()), &capability)
                        .map_err(mlua::Error::runtime)?;
                } else {
                    require_capability(&app_data_snapshot, &capability)
                        .map_err(mlua::Error::runtime)?;
                }
                Ok(true)
            })?,
        )?;
    }

    {
        let app_data_snapshot = app_data.clone();
        let access_table = lua.create_table()?;
        access_table.set(
            "check",
            lua.create_function(move |lua, (capability, opts): (String, Option<Table>)| {
                if capability.trim().is_empty() {
                    return Err(mlua::Error::runtime("capability must not be empty"));
                }
                let decision = if let Some(agent_id) = read_agent_id(&opts)? {
                    app_data_snapshot
                        .governance_manager
                        .capability_decision(Some(agent_id.as_str()), &capability)
                } else {
                    capability_decision(&app_data_snapshot, &capability)
                };
                lua.to_value(&decision)
            })?,
        )?;
        lua.globals().set("access", Value::Table(access_table))?;
    }

    Ok(())
}
