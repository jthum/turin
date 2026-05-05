use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};
use serde::Deserialize;
use turin_daemon_protocol::{
    ContextPersistenceParams, ScheduleCreateParams, ScheduleUpdateParams, StoreTargetParams,
};
use turin_types::{TaskInputContent, ToolsConfig};

use crate::harness::globals::HarnessAppData;
use crate::harness::stdlib::binding_common::{bridge_async_result, nil_err, ok_value};
use crate::harness::stdlib::governance_support::{current_agent_id, require_capability};

#[derive(Debug, Deserialize)]
struct LuaScheduleCreateOpts {
    prompt: String,
    #[serde(default)]
    content: Option<serde_json::Value>,
    #[serde(default)]
    tools: Option<serde_json::Value>,
    #[serde(default)]
    conflict_policy: Option<String>,
    #[serde(default)]
    agent: Option<String>,
    #[serde(default)]
    next_run_unix_ms: Option<i64>,
    #[serde(default)]
    after_seconds: Option<f64>,
    #[serde(default)]
    interval_seconds: Option<f64>,
    #[serde(default)]
    overlap_policy: Option<String>,
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    persistence: Option<LuaSchedulePersistenceOpts>,
}

#[derive(Debug, Deserialize)]
struct LuaScheduleUpdateOpts {
    id: String,
    #[serde(default)]
    agent: Option<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    content: Option<serde_json::Value>,
    #[serde(default)]
    tools: Option<serde_json::Value>,
    #[serde(default)]
    conflict_policy: Option<String>,
    #[serde(default)]
    next_run_unix_ms: Option<i64>,
    #[serde(default)]
    after_seconds: Option<f64>,
    #[serde(default)]
    interval_seconds: Option<f64>,
    #[serde(default)]
    overlap_policy: Option<String>,
    #[serde(default)]
    enabled: Option<bool>,
    #[serde(default)]
    persistence: Option<LuaSchedulePersistenceOpts>,
}

#[derive(Debug, Deserialize)]
struct LuaSchedulePersistenceOpts {
    #[serde(default)]
    state: Option<serde_json::Value>,
    #[serde(default)]
    store: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize, Default)]
struct LuaScheduleListOpts {
    #[serde(default)]
    agent: Option<String>,
}

fn parse_store_target(value: serde_json::Value, field_name: &str) -> LuaResult<StoreTargetParams> {
    match value {
        serde_json::Value::String(value) => {
            if is_path_like(&value) {
                Ok(StoreTargetParams {
                    path: Some(value),
                    alias: None,
                })
            } else {
                Ok(StoreTargetParams {
                    path: None,
                    alias: Some(value),
                })
            }
        }
        serde_json::Value::Object(map) => {
            let path = map
                .get("path")
                .and_then(|value| value.as_str())
                .map(|value| value.to_string());
            let alias = map
                .get("alias")
                .or_else(|| map.get("store"))
                .and_then(|value| value.as_str())
                .map(|value| value.to_string());
            if path.is_none() && alias.is_none() {
                return Err(mlua::Error::runtime(format!(
                    "runtime.schedule {} target requires 'path' or 'alias/store'",
                    field_name
                )));
            }
            Ok(StoreTargetParams { path, alias })
        }
        _ => Err(mlua::Error::runtime(format!(
            "runtime.schedule {} target must be string or table",
            field_name
        ))),
    }
}

fn parse_persistence(
    persistence: Option<LuaSchedulePersistenceOpts>,
) -> LuaResult<Option<ContextPersistenceParams>> {
    let Some(persistence) = persistence else {
        return Ok(None);
    };
    let state = persistence
        .state
        .map(|value| parse_store_target(value, "persistence.state"))
        .transpose()?;
    let store = persistence
        .store
        .map(|value| parse_store_target(value, "persistence.store"))
        .transpose()?;
    if state.is_none() && store.is_none() {
        return Ok(None);
    }
    Ok(Some(ContextPersistenceParams { state, store }))
}

fn parse_schedule_content(
    content: Option<serde_json::Value>,
) -> LuaResult<Option<Vec<TaskInputContent>>> {
    content
        .map(serde_json::from_value)
        .transpose()
        .map_err(|e| mlua::Error::runtime(format!("invalid runtime.schedule content: {}", e)))
}

fn parse_schedule_tools(content: Option<serde_json::Value>) -> LuaResult<Option<ToolsConfig>> {
    content
        .map(serde_json::from_value)
        .transpose()
        .map_err(|e| mlua::Error::runtime(format!("invalid runtime.schedule tools: {}", e)))
}

fn is_path_like(selector: &str) -> bool {
    selector.contains('/')
        || selector.contains('\\')
        || selector.starts_with('.')
        || selector.ends_with(".db")
        || selector.starts_with('~')
}

fn now_unix_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_else(|_| std::time::Duration::from_secs(0))
        .as_millis() as i64
}

fn parse_non_negative_seconds(value: f64, field_name: &str) -> LuaResult<u64> {
    if !value.is_finite() || value < 0.0 {
        return Err(mlua::Error::runtime(format!(
            "runtime.schedule {} must be a non-negative number",
            field_name
        )));
    }
    Ok(value.round() as u64)
}

fn schedule_create_params(
    lua: &Lua,
    app_data: &HarnessAppData,
    opts: Table,
) -> LuaResult<ScheduleCreateParams> {
    let parsed = lua
        .from_value::<LuaScheduleCreateOpts>(Value::Table(opts))
        .map_err(|e| {
            mlua::Error::runtime(format!("invalid runtime.schedule.create opts: {}", e))
        })?;

    let next_run_unix_ms = match (parsed.next_run_unix_ms, parsed.after_seconds) {
        (Some(at), None) => at,
        (None, Some(after)) => now_unix_ms()
            .saturating_add((parse_non_negative_seconds(after, "after_seconds")? as i64) * 1000),
        (Some(_), Some(_)) => {
            return Err(mlua::Error::runtime(
                "runtime.schedule.create opts cannot specify both next_run_unix_ms and after_seconds",
            ));
        }
        (None, None) => {
            if let Some(interval_seconds) = parsed.interval_seconds {
                now_unix_ms().saturating_add(
                    (parse_non_negative_seconds(interval_seconds, "interval_seconds")? as i64)
                        * 1000,
                )
            } else {
                return Err(mlua::Error::runtime(
                    "runtime.schedule.create opts require next_run_unix_ms, after_seconds, or interval_seconds",
                ));
            }
        }
    };

    let interval_seconds = parsed
        .interval_seconds
        .map(|value| parse_non_negative_seconds(value, "interval_seconds"))
        .transpose()?;

    Ok(ScheduleCreateParams {
        agent_id: parsed.agent.unwrap_or_else(|| current_agent_id(app_data)),
        prompt: parsed.prompt,
        content: parse_schedule_content(parsed.content)?,
        tools: parse_schedule_tools(parsed.tools)?,
        conflict_policy: parsed.conflict_policy,
        persistence: parse_persistence(parsed.persistence)?,
        next_run_unix_ms,
        interval_seconds,
        overlap_policy: Some(parsed.overlap_policy.unwrap_or_else(|| "skip".to_string())),
        enabled: parsed.enabled.unwrap_or(true),
    })
}

fn schedule_update_params(lua: &Lua, opts: Table) -> LuaResult<ScheduleUpdateParams> {
    let parsed = lua
        .from_value::<LuaScheduleUpdateOpts>(Value::Table(opts))
        .map_err(|e| {
            mlua::Error::runtime(format!("invalid runtime.schedule.update opts: {}", e))
        })?;

    let next_run_unix_ms = match (parsed.next_run_unix_ms, parsed.after_seconds) {
        (Some(at), None) => Some(at),
        (None, Some(after)) => {
            Some(now_unix_ms().saturating_add(
                (parse_non_negative_seconds(after, "after_seconds")? as i64) * 1000,
            ))
        }
        (Some(_), Some(_)) => {
            return Err(mlua::Error::runtime(
                "runtime.schedule.update opts cannot specify both next_run_unix_ms and after_seconds",
            ));
        }
        (None, None) => None,
    };

    let interval_seconds = parsed
        .interval_seconds
        .map(|value| parse_non_negative_seconds(value, "interval_seconds"))
        .transpose()?;

    Ok(ScheduleUpdateParams {
        id: parsed.id,
        agent_id: parsed.agent,
        prompt: parsed.prompt,
        content: parse_schedule_content(parsed.content)?,
        tools: parse_schedule_tools(parsed.tools)?,
        conflict_policy: parsed.conflict_policy,
        persistence: parse_persistence(parsed.persistence)?,
        next_run_unix_ms,
        interval_seconds,
        overlap_policy: parsed.overlap_policy,
        enabled: parsed.enabled,
    })
}

fn scheduler_access(
    app_data: &HarnessAppData,
) -> Result<std::sync::Arc<crate::harness::scheduler::HarnessSchedulerAccess>, String> {
    app_data
        .scheduler
        .clone()
        .ok_or_else(|| "runtime.schedule requires a daemon-managed runtime".to_string())
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

                if !params.agent_id.eq(&current_agent_id(&app_data))
                    && !app_data.config.agents.contains_key(&params.agent_id)
                {
                    return nil_err(
                        lua,
                        &format!("Unknown scheduled agent '{}'", params.agent_id),
                    );
                }

                let result = bridge_async_result(async move {
                    scheduler
                        .create_job(params)
                        .await
                        .map_err(|e| e.to_string())
                });
                match result {
                    Ok(job) => Ok(ok_value(lua.to_value(&job)?)),
                    Err(err) => nil_err(lua, &err),
                }
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
                    && !agent_id.eq(&current_agent_id(&app_data))
                    && !app_data.config.agents.contains_key(agent_id)
                {
                    return nil_err(lua, &format!("Unknown scheduled agent '{}'", agent_id));
                }

                let result = bridge_async_result(async move {
                    scheduler
                        .update_job(params)
                        .await
                        .map_err(|e| e.to_string())
                });
                match result {
                    Ok(Some(job)) => Ok(ok_value(lua.to_value(&job)?)),
                    Ok(None) => nil_err(lua, "Scheduled job not found"),
                    Err(err) => nil_err(lua, &err),
                }
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
                let parsed = match opts {
                    Some(opts) => lua
                        .from_value::<LuaScheduleListOpts>(Value::Table(opts))
                        .map_err(|e| {
                            mlua::Error::runtime(format!(
                                "invalid runtime.schedule.list opts: {}",
                                e
                            ))
                        })?,
                    None => LuaScheduleListOpts::default(),
                };
                let result = bridge_async_result(async move {
                    scheduler.list_jobs().await.map_err(|e| e.to_string())
                });
                match result {
                    Ok(jobs) => {
                        let jobs: Vec<_> = match parsed.agent {
                            Some(agent) => jobs
                                .into_iter()
                                .filter(|job| job.agent_id == agent)
                                .collect(),
                            None => jobs,
                        };
                        Ok(ok_value(lua.to_value(&jobs)?))
                    }
                    Err(err) => nil_err(lua, &err),
                }
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
                match result {
                    Ok(Some(job)) => Ok(ok_value(lua.to_value(&job)?)),
                    Ok(None) => nil_err(lua, "scheduled job not found"),
                    Err(err) => nil_err(lua, &err),
                }
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
                match result {
                    Ok(Some(job)) => Ok(ok_value(lua.to_value(&job)?)),
                    Ok(None) => nil_err(lua, "scheduled job not found"),
                    Err(err) => nil_err(lua, &err),
                }
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
                match result {
                    Ok(Some(job)) => Ok(ok_value(lua.to_value(&job)?)),
                    Ok(None) => nil_err(lua, "scheduled job not found"),
                    Err(err) => nil_err(lua, &err),
                }
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
                match result {
                    Ok(Some(job)) => Ok(ok_value(lua.to_value(&job)?)),
                    Ok(None) => nil_err(lua, "scheduled job not found"),
                    Err(err) => nil_err(lua, &err),
                }
            })?,
        )?;
    }

    runtime_table.set("schedule", schedule_ns)?;
    Ok(())
}
