use mlua::{Lua, Result as LuaResult};

use crate::harness::globals::HarnessAppData;

mod access;
mod agent;
mod code_helpers;
mod common;
mod data;
mod db;
mod fs_json;
mod governance;
mod graph;
mod schedule;
mod time;
mod verdict;
mod worklist;

pub(crate) use data::build_scope_proxy;

pub fn register_dx_globals(lua: &Lua, app_data: &HarnessAppData) -> LuaResult<()> {
    verdict::register_verdict_globals(lua)?;
    access::register_access_globals(lua, app_data)?;
    data::register_data_globals(lua)?;
    code_helpers::register_code_helpers_dx(lua)?;
    db::register_db_dx(lua)?;
    agent::register_agent_dx(lua)?;
    graph::register_graph_dx(lua)?;
    schedule::register_schedule_dx(lua)?;
    worklist::register_worklist_dx(lua)?;
    governance::register_governance_dx(lua)?;
    time::register_time_dx(lua)?;
    fs_json::register_fs_json_globals(lua)?;
    Ok(())
}
