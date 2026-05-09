use mlua::{Lua, Result as LuaResult};

use crate::harness::globals::HarnessAppData;
use crate::harness::stdlib::{
    runtime_agent, runtime_code, runtime_context, runtime_data, runtime_db, runtime_governance,
    runtime_graph, runtime_inference, runtime_policy, runtime_schedule, runtime_signal,
    runtime_worklist,
};

pub fn register_runtime_namespace(lua: &Lua, app_data: &HarnessAppData) -> LuaResult<()> {
    let runtime_table = lua.create_table()?;
    runtime_context::register_runtime_context_namespace(lua, &runtime_table, app_data)?;
    runtime_data::register_runtime_data_namespaces(lua, &runtime_table, app_data)?;
    runtime_code::register_runtime_code_namespace(lua, &runtime_table, app_data)?;
    runtime_db::register_runtime_db_namespace(lua, &runtime_table, app_data)?;
    runtime_inference::register_runtime_inference_namespace(lua, &runtime_table, app_data)?;
    runtime_agent::register_runtime_agent_namespace(lua, &runtime_table, app_data)?;
    runtime_policy::register_runtime_policy_namespace(lua, &runtime_table, app_data)?;
    runtime_governance::register_runtime_governance_namespace(lua, &runtime_table, app_data)?;
    runtime_graph::register_runtime_graph_namespace(lua, &runtime_table, app_data)?;
    runtime_schedule::register_runtime_schedule_namespace(lua, &runtime_table, app_data)?;
    runtime_signal::register_runtime_signal_namespace(lua, &runtime_table, app_data)?;
    runtime_worklist::register_runtime_worklist_namespace(lua, &runtime_table, app_data)?;
    lua.globals().set("runtime", runtime_table)?;
    Ok(())
}
