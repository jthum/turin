use std::collections::BTreeMap;

use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};

use crate::harness::globals::HarnessAppData;
use crate::harness::stdlib::binding_common::{
    bridge_async, bridge_async_display_err, json_ok, lua_json_result, lua_string_result, nil_err,
    nil_ok,
};
use crate::harness::stdlib::governance_support::{
    apply_active_grant_ceiling_to_peer_delegation, parse_delegated_capabilities,
    require_capability as require_governance_capability,
    require_child_agent as require_child_agent_governance,
};
use crate::harness::stdlib::policy_support::{policy_bool, runtime_policy_snapshot};
use crate::kernel::prepare_persisted_session_sidestep;
use crate::kernel::session::{
    ExecutionConflictPolicy, ExecutionContextTarget, QueuedTask, SidestepMode,
    TaskExecutionOverrides,
};

fn active_trace_id(app_data: &HarnessAppData) -> Option<String> {
    app_data
        .execution_ctx
        .lock()
        .ok()
        .and_then(|ctx| ctx.trace_id.clone())
}

fn parse_submit_task(
    lua: &Lua,
    task_val: Value,
    app_data: &HarnessAppData,
) -> LuaResult<QueuedTask> {
    let trace_id = active_trace_id(app_data);
    match task_val {
        Value::String(s) => {
            Ok(QueuedTask::ad_hoc(s.to_str()?.to_string())
                .with_inherited_trace(trace_id.as_deref()))
        }
        Value::Table(t) => {
            let prompt = t.get::<String>("prompt").map_err(|_| {
                mlua::Error::runtime("runtime.agent.submit task table requires prompt")
            })?;
            let mut task = QueuedTask::ad_hoc(prompt).with_inherited_trace(trace_id.as_deref());
            if let Ok(title) = t.get::<String>("title") {
                task.title = Some(title);
            }
            if let Ok(conflict_policy) = t.get::<String>("conflict_policy") {
                task.conflict_policy = Some(parse_conflict_policy(&conflict_policy)?);
            }
            if let Ok(execution) = t.get::<Value>("execution") {
                task.execution = Some(parse_execution_overrides_with_lua(lua, execution)?);
            }
            Ok(task)
        }
        _ => Err(mlua::Error::runtime(
            "invalid task; expected string or {prompt=...}",
        )),
    }
}

fn title_from_opts(opts: Option<&Table>) -> Option<String> {
    opts.and_then(|t| t.get::<String>("title").ok())
}

fn timeout_ms_from_opts(opts: Option<&Table>) -> Option<u64> {
    opts.and_then(|t| t.get::<u64>("timeout_ms").ok())
}

fn conflict_policy_from_opts(opts: Option<&Table>) -> LuaResult<Option<ExecutionConflictPolicy>> {
    let Some(opts) = opts else {
        return Ok(None);
    };
    let Ok(conflict_policy) = opts.get::<String>("conflict_policy") else {
        return Ok(None);
    };
    Ok(Some(parse_conflict_policy(&conflict_policy)?))
}

fn execution_overrides_from_opts(
    lua: &Lua,
    opts: Option<&Table>,
) -> LuaResult<Option<TaskExecutionOverrides>> {
    let Some(opts) = opts else {
        return Ok(None);
    };
    let Ok(execution) = opts.get::<Value>("execution") else {
        return Ok(None);
    };
    if matches!(execution, Value::Nil) {
        return Ok(None);
    }
    Ok(Some(parse_execution_overrides_with_lua(lua, execution)?))
}

fn sidestep_mode_from_opts(opts: Option<&Table>) -> LuaResult<SidestepMode> {
    let Some(opts) = opts else {
        return Ok(SidestepMode::Ephemeral);
    };
    let Ok(raw) = opts.get::<String>("mode") else {
        return Ok(SidestepMode::Ephemeral);
    };
    raw.parse().map_err(mlua::Error::runtime)
}

fn sidestep_opts_table_from_value(lua: &Lua, value: Option<Value>) -> LuaResult<Option<Table>> {
    match value {
        None | Some(Value::Nil) => Ok(None),
        Some(Value::Table(table)) => Ok(Some(table)),
        Some(Value::String(mode)) => {
            let table = lua.create_table()?;
            table.set("mode", mode.to_str()?.to_string())?;
            Ok(Some(table))
        }
        Some(_) => Err(mlua::Error::runtime(
            "invalid sidestep opts; expected nil, mode string, or options table",
        )),
    }
}

fn sidestep_target_from_opts(
    lua: &Lua,
    opts: Option<&Table>,
) -> LuaResult<Option<ExecutionContextTarget>> {
    let Some(opts) = opts else {
        return Ok(None);
    };
    let value = match opts.get::<Value>("target") {
        Ok(Value::Nil) | Err(_) => match opts.get::<Value>("context_target") {
            Ok(value) => value,
            Err(_) => Value::Nil,
        },
        Ok(value) => value,
    };
    if matches!(value, Value::Nil) {
        return Ok(None);
    }
    lua.from_value::<ExecutionContextTarget>(value).map(Some)
}

async fn resolve_runtime_agent_sidestep_session(
    app_data: &HarnessAppData,
    agent_id: &str,
    requested_session_id: Option<String>,
    requested_slot_id: Option<String>,
) -> Result<crate::kernel::agent_manager::LiveSessionSnapshot, String> {
    let live = app_data
        .agent_manager
        .list_live_sessions(Some(agent_id))
        .await;
    if live.is_empty() {
        return Err(format!(
            "Agent '{}' has no live session to sidestep",
            agent_id
        ));
    }

    let filtered: Vec<_> = live
        .into_iter()
        .filter(|snapshot| {
            requested_session_id
                .as_ref()
                .is_none_or(|session_id| &snapshot.session_id == session_id)
                && requested_slot_id
                    .as_ref()
                    .is_none_or(|slot_id| &snapshot.slot_id == slot_id)
        })
        .collect();

    match filtered.as_slice() {
        [] => Err(format!(
            "No live session matched agent='{}' session_id={:?} slot_id={:?}",
            agent_id, requested_session_id, requested_slot_id
        )),
        [snapshot] => Ok(snapshot.clone()),
        _ => Err(format!(
            "Agent '{}' has multiple matching live sessions; specify session_id or slot_id",
            agent_id
        )),
    }
}

fn parse_execution_overrides_with_lua(
    lua: &Lua,
    value: Value,
) -> LuaResult<TaskExecutionOverrides> {
    let overrides = lua.from_value::<TaskExecutionOverrides>(value)?;
    if overrides.is_empty() {
        return Err(mlua::Error::runtime(
            "execution overrides must not be an empty table",
        ));
    }
    Ok(overrides)
}

fn parse_conflict_policy(raw: &str) -> LuaResult<ExecutionConflictPolicy> {
    raw.parse().map_err(mlua::Error::runtime)
}

fn require_submit_to_agent(
    app_data: &HarnessAppData,
    agent_id: &str,
) -> LuaResult<Result<(), String>> {
    if let Err(err) = require_governance_capability(app_data, "runtime.agent.submit") {
        return Ok(Err(err));
    }
    if let Err(err) = require_child_agent_governance(app_data, agent_id) {
        return Ok(Err(err));
    }
    let snapshot = runtime_policy_snapshot(app_data).map_err(mlua::Error::runtime)?;
    if !policy_bool(&snapshot, "spawn.enabled", true) {
        return Ok(Err("Policy denial: spawn.enabled=false".to_string()));
    }
    Ok(Ok(()))
}

fn delegated_capabilities_from_opts(
    app_data: &HarnessAppData,
    opts: Option<&Table>,
    caller_label: &str,
) -> LuaResult<Option<BTreeMap<String, bool>>> {
    let delegated_capabilities =
        parse_delegated_capabilities(app_data, opts, "capabilities", caller_label)?;
    apply_active_grant_ceiling_to_peer_delegation(app_data, delegated_capabilities, caller_label)
}

pub fn register_runtime_agent_namespace(
    lua: &Lua,
    runtime_table: &Table,
    app_data: &HarnessAppData,
) -> LuaResult<()> {
    let runtime_agent = lua.create_table()?;
    {
        let manager = app_data.agent_manager.clone();
        let app_data_snapshot = app_data.clone();
        runtime_agent.set(
            "list",
            lua.create_function(move |lua, ()| {
                if let Err(err) =
                    require_governance_capability(&app_data_snapshot, "runtime.agent.status")
                {
                    return nil_err(lua, &err);
                }
                let manager = manager.clone();
                let statuses = bridge_async(async move { manager.list_statuses().await });
                json_ok(lua, &statuses)
            })?,
        )?;
    }
    {
        let manager = app_data.agent_manager.clone();
        let app_data_snapshot = app_data.clone();
        runtime_agent.set(
            "get_status",
            lua.create_function(move |lua, agent_id: String| {
                if let Err(err) =
                    require_governance_capability(&app_data_snapshot, "runtime.agent.status")
                {
                    return nil_err(lua, &err);
                }
                let manager = manager.clone();
                let status = bridge_async(async move { manager.get_status(&agent_id).await });
                match status {
                    Some(s) => json_ok(lua, &s),
                    None => Ok(nil_ok()),
                }
            })?,
        )?;
    }
    {
        let manager = app_data.agent_manager.clone();
        let app_data_snapshot = app_data.clone();
        runtime_agent.set(
            "submit",
            lua.create_function(
                move |lua, (agent_id, task_val, opts): (String, Value, Option<Table>)| {
                    if let Err(err) = require_submit_to_agent(&app_data_snapshot, &agent_id)? {
                        return nil_err(lua, &err);
                    }

                    let task = match task_val {
                        v @ Value::String(_) | v @ Value::Table(_) => {
                            parse_submit_task(lua, v, &app_data_snapshot)?
                        }
                        _ => {
                            return nil_err(lua, "invalid task; expected string or {prompt=...}");
                        }
                    };
                    let mut task = task;
                    if task.conflict_policy.is_none() {
                        task.conflict_policy = conflict_policy_from_opts(opts.as_ref())?;
                    }
                    if task.execution.is_none() {
                        task.execution = execution_overrides_from_opts(lua, opts.as_ref())?;
                    }
                    let delegated_capabilities = delegated_capabilities_from_opts(
                        &app_data_snapshot,
                        opts.as_ref(),
                        "runtime.agent.submit",
                    )?;

                    let manager = manager.clone();
                    let result = bridge_async_display_err(async move {
                        manager
                            .submit(&agent_id, task, delegated_capabilities)
                            .await
                    });
                    lua_string_result(lua, result)
                },
            )?,
        )?;
    }
    {
        let manager = app_data.agent_manager.clone();
        let app_data_snapshot = app_data.clone();
        runtime_agent.set(
            "sidestep",
            lua.create_function(
                move |lua, (agent_id, prompt, opts_value): (String, String, Option<Value>)| {
                    let opts = sidestep_opts_table_from_value(lua, opts_value)?;
                    if let Err(err) = require_submit_to_agent(&app_data_snapshot, &agent_id)? {
                        return nil_err(lua, &err);
                    }

                    let sidestep_mode = sidestep_mode_from_opts(opts.as_ref())?;
                    let sidestep_target = sidestep_target_from_opts(lua, opts.as_ref())?;
                    let requested_session_id = opts
                        .as_ref()
                        .and_then(|table| table.get::<String>("session_id").ok());
                    let requested_slot_id = opts
                        .as_ref()
                        .and_then(|table| table.get::<String>("slot_id").ok());
                    let title = title_from_opts(opts.as_ref());
                    let delegated_capabilities = delegated_capabilities_from_opts(
                        &app_data_snapshot,
                        opts.as_ref(),
                        "runtime.agent.submit",
                    )?;
                    let trace_id = active_trace_id(&app_data_snapshot);

                    let manager = manager.clone();
                    let store_manager = app_data_snapshot.store_manager.clone();
                    let app_data_snapshot = app_data_snapshot.clone();
                    let result = bridge_async_display_err(async move {
                        let live = resolve_runtime_agent_sidestep_session(
                            &app_data_snapshot,
                            &agent_id,
                            requested_session_id,
                            requested_slot_id,
                        )
                        .await
                        .map_err(anyhow::Error::msg)?;
                        let prepared = prepare_persisted_session_sidestep(
                            &store_manager,
                            &live.session_id,
                            &live.execution.context_target,
                            sidestep_mode,
                            sidestep_target,
                        )
                        .await?;

                        let mut task = QueuedTask::ad_hoc(prompt)
                            .with_inherited_trace(trace_id.as_deref())
                            .with_conflict_policy(Some(prepared.conflict_policy))
                            .with_execution(Some(prepared.execution))
                            .with_branch_outcome(prepared.branch_outcome);
                        task.title = title;

                        manager
                            .submit_to_session(
                                &live.session_id,
                                Some(&live.slot_id),
                                task,
                                delegated_capabilities,
                            )
                            .await
                    });
                    lua_string_result(lua, result)
                },
            )?,
        )?;
    }
    {
        let manager = app_data.agent_manager.clone();
        let app_data_snapshot = app_data.clone();
        runtime_agent.set(
            "await",
            lua.create_function(move |lua, (task_id, opts): (String, Option<Table>)| {
                if let Err(err) =
                    require_governance_capability(&app_data_snapshot, "runtime.agent.await")
                {
                    return nil_err(lua, &err);
                }
                let timeout_ms = opts.as_ref().and_then(|t| t.get::<u64>("timeout_ms").ok());
                let manager = manager.clone();
                let result = bridge_async_display_err(async move {
                    manager.await_result(&task_id, timeout_ms).await
                });
                lua_json_result(lua, result)
            })?,
        )?;
    }
    {
        let manager = app_data.agent_manager.clone();
        let app_data_snapshot = app_data.clone();
        runtime_agent.set(
            "promote",
            lua.create_function(move |lua, (task_id, opts): (String, Option<Table>)| {
                if let Err(err) =
                    require_governance_capability(&app_data_snapshot, "runtime.agent.submit")
                {
                    return nil_err(lua, &err);
                }

                let branch_name = opts
                    .as_ref()
                    .and_then(|t| t.get::<String>("branch_name").ok());
                let manager = manager.clone();
                let app_data_snapshot = app_data_snapshot.clone();
                let result = bridge_async_display_err(async move {
                    let snapshot = manager
                        .get_task(&task_id)
                        .await
                        .ok_or_else(|| anyhow::anyhow!("Task '{}' not found", task_id))?;
                    require_child_agent_governance(&app_data_snapshot, &snapshot.agent_id)
                        .map_err(anyhow::Error::msg)?;
                    manager
                        .promote_completed_task(&task_id, branch_name.as_deref())
                        .await
                });
                lua_json_result(lua, result)
            })?,
        )?;
    }
    {
        let manager = app_data.agent_manager.clone();
        let app_data_snapshot = app_data.clone();
        runtime_agent.set(
            "ask",
            lua.create_function(
                move |lua, (agent_id, prompt, opts): (String, String, Option<Table>)| {
                    if let Err(err) =
                        require_governance_capability(&app_data_snapshot, "runtime.agent.await")
                    {
                        return nil_err(lua, &err);
                    }
                    if let Err(err) = require_submit_to_agent(&app_data_snapshot, &agent_id)? {
                        return nil_err(lua, &err);
                    }

                    let mut task = QueuedTask::ad_hoc(prompt)
                        .with_inherited_trace(active_trace_id(&app_data_snapshot).as_deref());
                    task.title = title_from_opts(opts.as_ref());
                    task.conflict_policy = conflict_policy_from_opts(opts.as_ref())?;
                    task.execution = execution_overrides_from_opts(lua, opts.as_ref())?;
                    let timeout_ms = timeout_ms_from_opts(opts.as_ref());
                    let delegated_capabilities = delegated_capabilities_from_opts(
                        &app_data_snapshot,
                        opts.as_ref(),
                        "runtime.agent.ask",
                    )?;

                    let manager = manager.clone();
                    let result = bridge_async_display_err(async move {
                        let request_id = manager
                            .submit(&agent_id, task, delegated_capabilities)
                            .await?;
                        let result = manager.await_result(&request_id, timeout_ms).await?;
                        if let Some(err) = result.error {
                            anyhow::bail!(err);
                        }
                        Ok(result.output.unwrap_or_default())
                    });
                    lua_string_result(lua, result)
                },
            )?,
        )?;
    }
    runtime_table.set("agent", runtime_agent)?;
    Ok(())
}
