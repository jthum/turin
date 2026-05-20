use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};
use serde::{Deserialize, Serialize};
use time::format_description::well_known::Rfc3339;
use time::{Duration as TimeDuration, OffsetDateTime, Time, UtcOffset};
use turin_daemon_protocol::{
    ContextPersistenceParams, ScheduleActionParams, ScheduleCreateParams, ScheduleUpdateParams,
    StoreTargetParams,
};
use turin_types::{TaskInputContent, ToolsConfig};

use crate::harness::globals::HarnessAppData;
use crate::harness::stdlib::binding_common::{bridge_async_result, json_ok, nil_err};
use crate::harness::stdlib::governance_support::{current_agent_id, require_capability};
use crate::harness::stdlib::object_refs;

#[derive(Debug, Deserialize)]
struct LuaScheduleCreateOpts {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    content: Option<serde_json::Value>,
    #[serde(default)]
    tools: Option<serde_json::Value>,
    #[serde(default)]
    conflict_policy: Option<String>,
    #[serde(default)]
    action: Option<serde_json::Value>,
    #[serde(default)]
    params: Option<serde_json::Value>,
    #[serde(default)]
    agent: Option<String>,
    #[serde(default)]
    next_run: Option<serde_json::Value>,
    #[serde(default)]
    next_run_unix_ms: Option<i64>,
    #[serde(default)]
    after_seconds: Option<f64>,
    #[serde(default)]
    interval_seconds: Option<f64>,
    #[serde(default)]
    recurring: Option<String>,
    #[serde(default)]
    overlap_policy: Option<String>,
    #[serde(default)]
    work_key: Option<String>,
    #[serde(default)]
    max_concurrency: Option<u32>,
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
    action: Option<serde_json::Value>,
    #[serde(default)]
    params: Option<serde_json::Value>,
    #[serde(default)]
    next_run_unix_ms: Option<i64>,
    #[serde(default)]
    next_run: Option<serde_json::Value>,
    #[serde(default)]
    after_seconds: Option<f64>,
    #[serde(default)]
    interval_seconds: Option<f64>,
    #[serde(default)]
    recurring: Option<String>,
    #[serde(default)]
    overlap_policy: Option<String>,
    #[serde(default)]
    work_key: Option<String>,
    #[serde(default)]
    max_concurrency: Option<u32>,
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

#[derive(Debug, Deserialize, Default)]
struct LuaScheduleRunsOpts {
    #[serde(default)]
    active_only: bool,
    #[serde(default)]
    limit: Option<u32>,
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

fn parse_schedule_action(
    action: Option<serde_json::Value>,
    params: Option<serde_json::Value>,
) -> LuaResult<Option<ScheduleActionParams>> {
    let Some(action) = action else {
        return Ok(None);
    };
    match action {
        serde_json::Value::String(name) => Ok(Some(ScheduleActionParams { name, params })),
        serde_json::Value::Object(mut map) => {
            if let Some(value) = map
                .remove("action")
                .and_then(|value| value.as_str().map(|s| s.to_string()))
            {
                let params = map.remove("params").or(params);
                return Ok(Some(ScheduleActionParams {
                    name: value,
                    params,
                }));
            }
            serde_json::from_value(serde_json::Value::Object(map))
                .map(Some)
                .map_err(|e| {
                    mlua::Error::runtime(format!("invalid runtime.schedule action: {}", e))
                })
        }
        _ => Err(mlua::Error::runtime(
            "runtime.schedule action must be string or table".to_string(),
        )),
    }
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

fn parse_recurring_pattern(recurring: Option<String>) -> LuaResult<Option<String>> {
    match recurring.as_deref() {
        None => Ok(None),
        Some("daily" | "weekly") => Ok(recurring),
        Some(other) => Err(mlua::Error::runtime(format!(
            "runtime.schedule recurring must be 'daily' or 'weekly', got '{}'",
            other
        ))),
    }
}

fn parse_local_time_shorthand(raw: &str) -> LuaResult<Time> {
    let parts: Vec<_> = raw.split(':').collect();
    let [hour, minute, second] = match parts.as_slice() {
        [hour, minute] => [*hour, *minute, "0"],
        [hour, minute, second] => [*hour, *minute, *second],
        _ => {
            return Err(mlua::Error::runtime(
                "runtime.schedule next_run time shorthand must be 'HH:MM' or 'HH:MM:SS'"
                    .to_string(),
            ));
        }
    };
    let hour = hour.parse::<u8>().map_err(|_| {
        mlua::Error::runtime("runtime.schedule next_run hour must be an integer".to_string())
    })?;
    let minute = minute.parse::<u8>().map_err(|_| {
        mlua::Error::runtime("runtime.schedule next_run minute must be an integer".to_string())
    })?;
    let second = second.parse::<u8>().map_err(|_| {
        mlua::Error::runtime("runtime.schedule next_run second must be an integer".to_string())
    })?;
    Time::from_hms(hour, minute, second).map_err(|_| {
        mlua::Error::runtime(format!(
            "runtime.schedule next_run shorthand '{}' is not a valid local time",
            raw
        ))
    })
}

fn parse_string_next_run(raw: &str, recurring_pattern: Option<&str>) -> LuaResult<i64> {
    if let Ok(parsed) = OffsetDateTime::parse(raw, &Rfc3339) {
        return Ok(parsed.unix_timestamp_nanos() as i64 / 1_000_000);
    }

    let parsed_time = parse_local_time_shorthand(raw)?;
    if matches!(recurring_pattern, Some("weekly")) {
        return Err(mlua::Error::runtime(
            "runtime.schedule next_run weekly shorthand requires an anchored timestamp (for example RFC3339)"
                .to_string(),
        ));
    }

    let local_offset = UtcOffset::current_local_offset().map_err(|_| {
        mlua::Error::runtime(
            "runtime.schedule next_run local-time shorthand requires a detectable local offset"
                .to_string(),
        )
    })?;
    let local_now = OffsetDateTime::now_utc().to_offset(local_offset);
    let mut next = local_now
        .date()
        .with_time(parsed_time)
        .assume_offset(local_offset);
    if next <= local_now {
        next = next.checked_add(TimeDuration::days(1)).ok_or_else(|| {
            mlua::Error::runtime("runtime.schedule next_run overflow".to_string())
        })?;
    }
    Ok(next.unix_timestamp_nanos() as i64 / 1_000_000)
}

fn parse_next_run_value(
    next_run: Option<serde_json::Value>,
    next_run_unix_ms: Option<i64>,
    recurring_pattern: Option<&str>,
) -> LuaResult<Option<i64>> {
    match (next_run_unix_ms, next_run) {
        (Some(at), None) => Ok(Some(at)),
        (None, Some(serde_json::Value::String(raw))) => {
            Ok(Some(parse_string_next_run(&raw, recurring_pattern)?))
        }
        (None, Some(serde_json::Value::Number(number))) => {
            if let Some(value) = number.as_i64() {
                Ok(Some(value))
            } else {
                Err(mlua::Error::runtime(
                    "runtime.schedule next_run numeric value must be an integer timestamp in unix ms"
                        .to_string(),
                ))
            }
        }
        (None, Some(_)) => Err(mlua::Error::runtime(
            "runtime.schedule next_run must be a unix-ms number or timestamp string".to_string(),
        )),
        (Some(_), Some(_)) => Err(mlua::Error::runtime(
            "runtime.schedule opts cannot specify both next_run and next_run_unix_ms".to_string(),
        )),
        (None, None) => Ok(None),
    }
}

fn schedule_create_params(
    lua: &Lua,
    app_data: &HarnessAppData,
    opts: Table,
) -> LuaResult<ScheduleCreateParams> {
    let encoded = object_refs::encode_lua_payload(lua, Value::Table(opts))?;
    let parsed = serde_json::from_value::<LuaScheduleCreateOpts>(encoded).map_err(|e| {
        mlua::Error::runtime(format!("invalid runtime.schedule.create opts: {}", e))
    })?;

    let recurring_pattern = parse_recurring_pattern(parsed.recurring)?;
    let explicit_next_run = parse_next_run_value(
        parsed.next_run,
        parsed.next_run_unix_ms,
        recurring_pattern.as_deref(),
    )?;
    let next_run_unix_ms = match (explicit_next_run, parsed.after_seconds) {
        (Some(at), None) => at,
        (None, Some(after)) => now_unix_ms()
            .saturating_add((parse_non_negative_seconds(after, "after_seconds")? as i64) * 1000),
        (Some(_), Some(_)) => {
            return Err(mlua::Error::runtime(
                "runtime.schedule.create opts cannot specify both next_run/next_run_unix_ms and after_seconds",
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
                    "runtime.schedule.create opts require next_run, next_run_unix_ms, after_seconds, or interval_seconds",
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
        action: parse_schedule_action(parsed.action, parsed.params)?,
        persistence: parse_persistence(parsed.persistence)?,
        next_run_unix_ms,
        interval_seconds,
        recurring_pattern,
        overlap_policy: Some(parsed.overlap_policy.unwrap_or_else(|| "skip".to_string())),
        work_key: parsed.work_key,
        max_concurrency: parsed.max_concurrency,
        enabled: parsed.enabled.unwrap_or(true),
    })
}

fn schedule_update_params(lua: &Lua, opts: Table) -> LuaResult<ScheduleUpdateParams> {
    let encoded = object_refs::encode_lua_payload(lua, Value::Table(opts))?;
    let parsed = serde_json::from_value::<LuaScheduleUpdateOpts>(encoded).map_err(|e| {
        mlua::Error::runtime(format!("invalid runtime.schedule.update opts: {}", e))
    })?;

    let recurring_pattern = parse_recurring_pattern(parsed.recurring)?;
    let explicit_next_run = parse_next_run_value(
        parsed.next_run,
        parsed.next_run_unix_ms,
        recurring_pattern.as_deref(),
    )?;
    let next_run_unix_ms = match (explicit_next_run, parsed.after_seconds) {
        (Some(at), None) => Some(at),
        (None, Some(after)) => {
            Some(now_unix_ms().saturating_add(
                (parse_non_negative_seconds(after, "after_seconds")? as i64) * 1000,
            ))
        }
        (Some(_), Some(_)) => {
            return Err(mlua::Error::runtime(
                "runtime.schedule.update opts cannot specify both next_run/next_run_unix_ms and after_seconds",
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
        action: parse_schedule_action(parsed.action, parsed.params)?,
        persistence: parse_persistence(parsed.persistence)?,
        next_run_unix_ms,
        interval_seconds,
        recurring_pattern,
        overlap_policy: parsed.overlap_policy,
        work_key: parsed.work_key,
        max_concurrency: parsed.max_concurrency,
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

fn schedule_result<T: Serialize>(
    lua: &Lua,
    result: Result<T, String>,
) -> LuaResult<(Value, Value)> {
    match result {
        Ok(value) => json_ok(lua, &value),
        Err(err) => nil_err(lua, &err),
    }
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
                schedule_result(lua, result)
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
                let result = result.map(|jobs| match parsed.agent {
                    Some(agent) => jobs
                        .into_iter()
                        .filter(|job| job.agent_id == agent)
                        .collect(),
                    None => jobs,
                });
                schedule_result(lua, result)
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
                let parsed = match opts {
                    Some(opts) => lua
                        .from_value::<LuaScheduleRunsOpts>(Value::Table(opts))
                        .map_err(|e| {
                            mlua::Error::runtime(format!(
                                "invalid runtime.schedule.runs opts: {}",
                                e
                            ))
                        })?,
                    None => LuaScheduleRunsOpts::default(),
                };
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
