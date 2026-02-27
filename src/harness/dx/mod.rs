use mlua::{Lua, Result as LuaResult};

use crate::harness::globals::HarnessAppData;

mod access;
mod agent;
mod common;
mod data;
mod db;
mod fs_json;
mod verdict;

pub fn register_dx_globals(lua: &Lua, app_data: &HarnessAppData) -> LuaResult<()> {
    verdict::register_verdict_globals(lua)?;
    access::register_access_globals(lua, app_data)?;
    data::register_data_globals(lua)?;
    db::register_db_dx(lua)?;
    agent::register_agent_dx(lua)?;
    fs_json::register_fs_json_globals(lua)?;
    Ok(())
}
