use mlua::{Lua, Result as LuaResult, Table};

use crate::harness::globals::HarnessAppData;
use crate::harness::stdlib::binding_common::{json_ok, nil_err, string_value};

pub fn register_runtime_governance_namespace(
    lua: &Lua,
    runtime_table: &Table,
    app_data: &HarnessAppData,
) -> LuaResult<()> {
    let governance_table = lua.create_table()?;

    {
        let governance_manager = app_data.governance_manager.clone();
        governance_table.set(
            "profile",
            lua.create_function(move |lua, ()| {
                let snapshot = governance_manager.snapshot();
                match serde_json::to_string(&snapshot.profile) {
                    Ok(serialized) => {
                        let profile = serialized.trim_matches('"').to_string();
                        string_value(lua, &profile)
                    }
                    Err(e) => Err(mlua::Error::runtime(format!(
                        "failed to serialize governance profile: {}",
                        e
                    ))),
                }
            })?,
        )?;
    }

    {
        let governance_manager = app_data.governance_manager.clone();
        governance_table.set(
            "snapshot",
            lua.create_function(move |lua, agent_id: Option<String>| {
                let snapshot = governance_manager.snapshot_for_agent(agent_id.as_deref());
                json_ok(lua, &snapshot)
            })?,
        )?;
    }

    {
        let governance_manager = app_data.governance_manager.clone();
        governance_table.set(
            "agent",
            lua.create_function(move |lua, agent_id: String| {
                if agent_id.trim().is_empty() {
                    return nil_err(lua, "agent_id must not be empty");
                }
                let snapshot = governance_manager.snapshot_for_agent(Some(agent_id.as_str()));
                json_ok(lua, &snapshot)
            })?,
        )?;
    }

    runtime_table.set("governance", governance_table)?;
    Ok(())
}
