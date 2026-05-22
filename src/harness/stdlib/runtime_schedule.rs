mod params;

use mlua::{Lua, Result as LuaResult, Table, Value};
use serde::{Deserialize, Serialize};

use crate::harness::globals::HarnessAppData;
use crate::harness::stdlib::binding_common::{
    bridge_async_result, json_ok, lua_json_result, nil_err, parse_optional_lua_table,
};
use crate::harness::stdlib::governance_support::{current_agent_id, require_capability};
use params::{schedule_create_params, schedule_update_params};

#[derive(Debug, Deserialize, Default)]
struct LuaScheduleListOpts {
    #[serde(default)]
    agent: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct LuaScheduleRunsOpts {
    #[serde(default)]
    active_only: bool,
    #[serde(default)]
    limit: Option<u32>,
}

fn scheduler_access(
    app_data: &HarnessAppData,
) -> Result<std::sync::Arc<crate::harness::scheduler::HarnessSchedulerAccess>, String> {
    app_data
        .scheduler
        .clone()
        .ok_or_else(|| "runtime.schedule requires a daemon-managed runtime".to_string())
}

fn optional_schedule_result<T: Serialize>(
    lua: &Lua,
    result: Result<Option<T>, String>,
    not_found: &str,
) -> LuaResult<(Value, Value)> {
    match result {
        Ok(Some(value)) => json_ok(lua, &value),
        Ok(None) => nil_err(lua, not_found),
        Err(err) => nil_err(lua, &err),
    }
}

fn validate_scheduled_agent(app_data: &HarnessAppData, agent_id: &str) -> Option<String> {
    if agent_id.eq(&current_agent_id(app_data)) || app_data.config.agents.contains_key(agent_id) {
        None
    } else {
        Some(format!("Unknown scheduled agent '{}'", agent_id))
    }
}

pub fn register_runtime_schedule_namespace(
    lua: &Lua,
    runtime_table: &Table,
    app_data: &HarnessAppData,
) -> LuaResult<()> {
    let schedule_ns = lua.create_table()?;

    {
        let app_data = app_data.clone();
        schedule_ns.set(
            "create",
            lua.create_function(move |lua, opts: Table| {
                require_capability(&app_data, "runtime.schedule.create")
                    .map_err(mlua::Error::runtime)?;
                let params = schedule_create_params(lua, &app_data, opts)?;
                let scheduler = match scheduler_access(&app_data) {
                    Ok(scheduler) => scheduler,
                    Err(err) => return nil_err(lua, &err),
                };

                if let Some(err) = validate_scheduled_agent(&app_data, &params.agent_id) {
                    return nil_err(lua, &err);
                }

                let result = bridge_async_result(async move {
                    scheduler
                        .create_job(params)
                        .await
                        .map_err(|e| e.to_string())
                });
                lua_json_result(lua, result)
            })?,
        )?;
    }

    {
        let app_data = app_data.clone();
        schedule_ns.set(
            "update",
            lua.create_function(move |lua, opts: Table| {
                require_capability(&app_data, "runtime.schedule.update")
                    .map_err(mlua::Error::runtime)?;
                let params = schedule_update_params(lua, opts)?;
                let scheduler = match scheduler_access(&app_data) {
                    Ok(scheduler) => scheduler,
                    Err(err) => return nil_err(lua, &err),
                };

                if let Some(agent_id) = params.agent_id.as_ref()
                    && let Some(err) = validate_scheduled_agent(&app_data, agent_id)
                {
                    return nil_err(lua, &err);
                }

                let result = bridge_async_result(async move {
                    scheduler
                        .update_job(params)
                        .await
                        .map_err(|e| e.to_string())
                });
                optional_schedule_result(lua, result, "Scheduled job not found")
            })?,
        )?;
    }

    {
        let app_data = app_data.clone();
        schedule_ns.set(
            "list",
            lua.create_function(move |lua, opts: Option<Table>| {
                require_capability(&app_data, "runtime.schedule.list")
                    .map_err(mlua::Error::runtime)?;
                let scheduler = match scheduler_access(&app_data) {
                    Ok(scheduler) => scheduler,
                    Err(err) => return nil_err(lua, &err),
                };
                let parsed = parse_optional_lua_table::<LuaScheduleListOpts>(
                    lua,
                    opts.as_ref(),
                    "runtime.schedule.list opts",
                )?;
                let result = bridge_async_result(async move {
                    scheduler.list_jobs().await.map_err(|e| e.to_string())
                });
                let result = result.map(|jobs| match parsed.agent {
                    Some(agent) => jobs
                        .into_iter()
                        .filter(|job| job.agent_id == agent)
                        .collect(),
                    None => jobs,
                });
                lua_json_result(lua, result)
            })?,
        )?;
    }

    {
        let app_data = app_data.clone();
        schedule_ns.set(
            "get",
            lua.create_function(move |lua, public_id: String| {
                require_capability(&app_data, "runtime.schedule.get")
                    .map_err(mlua::Error::runtime)?;
                let scheduler = match scheduler_access(&app_data) {
                    Ok(scheduler) => scheduler,
                    Err(err) => return nil_err(lua, &err),
                };
                let result = bridge_async_result(async move {
                    scheduler
                        .get_job(&public_id)
                        .await
                        .map_err(|e| e.to_string())
                });
                optional_schedule_result(lua, result, "scheduled job not found")
            })?,
        )?;
    }

    {
        let app_data = app_data.clone();
        schedule_ns.set(
            "runs",
            lua.create_function(move |lua, (public_id, opts): (String, Option<Table>)| {
                require_capability(&app_data, "runtime.schedule.runs")
                    .map_err(mlua::Error::runtime)?;
                let scheduler = match scheduler_access(&app_data) {
                    Ok(scheduler) => scheduler,
                    Err(err) => return nil_err(lua, &err),
                };
                let parsed = parse_optional_lua_table::<LuaScheduleRunsOpts>(
                    lua,
                    opts.as_ref(),
                    "runtime.schedule.runs opts",
                )?;
                let result = bridge_async_result(async move {
                    scheduler
                        .list_job_runs(&public_id, parsed.active_only, parsed.limit)
                        .await
                        .map_err(|e| e.to_string())
                });
                optional_schedule_result(lua, result, "scheduled job not found")
            })?,
        )?;
    }

    {
        let app_data = app_data.clone();
        schedule_ns.set(
            "enable",
            lua.create_function(move |lua, public_id: String| {
                require_capability(&app_data, "runtime.schedule.enable")
                    .map_err(mlua::Error::runtime)?;
                let scheduler = match scheduler_access(&app_data) {
                    Ok(scheduler) => scheduler,
                    Err(err) => return nil_err(lua, &err),
                };
                let result = bridge_async_result(async move {
                    scheduler
                        .set_job_enabled(&public_id, true)
                        .await
                        .map_err(|e| e.to_string())
                });
                optional_schedule_result(lua, result, "scheduled job not found")
            })?,
        )?;
    }

    {
        let app_data = app_data.clone();
        schedule_ns.set(
            "disable",
            lua.create_function(move |lua, public_id: String| {
                require_capability(&app_data, "runtime.schedule.disable")
                    .map_err(mlua::Error::runtime)?;
                let scheduler = match scheduler_access(&app_data) {
                    Ok(scheduler) => scheduler,
                    Err(err) => return nil_err(lua, &err),
                };
                let result = bridge_async_result(async move {
                    scheduler
                        .set_job_enabled(&public_id, false)
                        .await
                        .map_err(|e| e.to_string())
                });
                optional_schedule_result(lua, result, "scheduled job not found")
            })?,
        )?;
    }

    {
        let app_data = app_data.clone();
        schedule_ns.set(
            "delete",
            lua.create_function(move |lua, public_id: String| {
                require_capability(&app_data, "runtime.schedule.delete")
                    .map_err(mlua::Error::runtime)?;
                let scheduler = match scheduler_access(&app_data) {
                    Ok(scheduler) => scheduler,
                    Err(err) => return nil_err(lua, &err),
                };
                let result = bridge_async_result(async move {
                    scheduler
                        .delete_job(&public_id)
                        .await
                        .map_err(|e| e.to_string())
                });
                optional_schedule_result(lua, result, "scheduled job not found")
            })?,
        )?;
    }

    runtime_table.set("schedule", schedule_ns)?;
    Ok(())
}
