use mlua::{Lua, Result as LuaResult, Table, Value};

use crate::harness::globals::{ActiveHarnessExecutionContext, HarnessAppData};
use crate::harness::stdlib::binding_common::{
    bridge_async_result, lua_bool_result, lua_table_result, nil_err, nil_ok, ok_value,
};
use crate::harness::stdlib::identity_support::{
    get_active_identity, identity_to_lua_table, session_row_to_lua_table,
};
use crate::harness::stdlib::policy_support::runtime_policy_snapshot;
use crate::kernel::session::QueuedTask;
use crate::kernel::session_refs::format_session_reference;

use super::branch_lua::{branch_row_to_lua_table, branch_rows_to_lua_table};
use super::options::{opt_activate, opt_from_turn_index, opt_session_id, opt_slot_id};
use super::queue::{active_trace_id, queue_max, queue_push_many, queue_push_one};
use super::session_store::{
    current_session_matches, current_session_store_selector, lookup_session_store,
    require_session_store,
};

fn queue_command_result(
    lua: &Lua,
    execution_ctx: ActiveHarnessExecutionContext,
    app_data: &HarnessAppData,
    cmd: String,
    push_front: bool,
) -> LuaResult<(Value, Value)> {
    let snapshot = runtime_policy_snapshot(app_data).map_err(mlua::Error::runtime)?;
    let queue_max = queue_max(&snapshot);
    let trace_id = active_trace_id(app_data);
    let res = bridge_async_result(async move {
        queue_push_one(
            &execution_ctx,
            QueuedTask::ad_hoc(cmd).with_inherited_trace(trace_id.as_deref()),
            queue_max,
            push_front,
        )
        .await
    });
    lua_bool_result(lua, res)
}

pub(super) fn register_session_bindings(
    lua: &Lua,
    app_data: &HarnessAppData,
    agent_table: &Table,
) -> LuaResult<()> {
    let session_ns = lua.create_table()?;

    // agent.session.identity()
    session_ns.set(
        "identity",
        lua.create_function(move |lua, ()| {
            let app_data = lua
                .app_data_ref::<HarnessAppData>()
                .ok_or_else(|| mlua::Error::runtime("missing harness app data"))?;
            let identity = get_active_identity(&app_data).map_err(mlua::Error::runtime)?;
            identity_to_lua_table(lua, &identity)
        })?,
    )?;

    // agent.session.queue
    let aq = app_data.execution_ctx.clone();
    let queue_policy_snapshot = app_data.clone();
    session_ns.set(
        "queue",
        lua.create_function(move |lua, cmd: String| {
            queue_command_result(lua, aq.clone(), &queue_policy_snapshot, cmd, false)
        })?,
    )?;

    // agent.session.queue_next
    let aq2 = app_data.execution_ctx.clone();
    let queue_next_policy_snapshot = app_data.clone();
    session_ns.set(
        "queue_next",
        lua.create_function(move |lua, cmd: String| {
            queue_command_result(lua, aq2.clone(), &queue_next_policy_snapshot, cmd, true)
        })?,
    )?;

    // agent.session.queue_all
    let aq3 = app_data.execution_ctx.clone();
    let queue_all_policy_snapshot = app_data.clone();
    session_ns.set(
        "queue_all",
        lua.create_function(move |lua, commands: Table| {
            let mut items = Vec::new();
            for v in commands.sequence_values::<String>() {
                items.push(v?);
            }
            let snapshot = runtime_policy_snapshot(&queue_all_policy_snapshot)
                .map_err(mlua::Error::runtime)?;
            let queue_max = queue_max(&snapshot);
            let aq = aq3.clone();
            let trace_id = active_trace_id(&queue_all_policy_snapshot);
            let tasks = items
                .into_iter()
                .map(|cmd| QueuedTask::ad_hoc(cmd).with_inherited_trace(trace_id.as_deref()))
                .collect::<Vec<_>>();
            let res =
                bridge_async_result(async move { queue_push_many(&aq, tasks, queue_max).await });
            lua_bool_result(lua, res)
        })?,
    )?;

    // agent.session.load(session_id)
    {
        let manager = app_data.store_manager.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        session_ns.set(
            "load",
            lua.create_function(move |lua, session_id: String| {
                let manager = manager.clone();
                let execution_ctx = execution_ctx.clone();
                let result = bridge_async_result(async move {
                    lookup_session_store(&manager, &execution_ctx, Some(session_id))
                        .await
                        .map(|lookup| lookup.row)
                });
                match result {
                    Ok(Some(row)) => {
                        Ok(ok_value(Value::Table(session_row_to_lua_table(lua, &row)?)))
                    }
                    Ok(None) => Ok(nil_ok()),
                    Err(err) => nil_err(lua, &err),
                }
            })?,
        )?;
    }

    // agent.session.set_title(title, opts?)
    {
        let manager = app_data.store_manager.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        session_ns.set(
            "set_title",
            lua.create_function(move |lua, (title, opts): (String, Option<Table>)| {
                let title = title.trim().to_string();
                if title.is_empty() {
                    return nil_err(lua, "Session title must not be empty");
                }
                if title.chars().count() > 120 {
                    return nil_err(lua, "Session title must not exceed 120 characters");
                }

                let manager = manager.clone();
                let execution_ctx = execution_ctx.clone();
                let requested_session = opt_session_id(opts.as_ref());
                let if_empty = opts
                    .as_ref()
                    .and_then(|table| table.get::<bool>("if_empty").ok())
                    .unwrap_or(false);
                let result = bridge_async_result(async move {
                    let session =
                        require_session_store(&manager, &execution_ctx, requested_session).await?;
                    let public_id = uuid::Uuid::parse_str(&session.session_ref.public_id)
                        .map_err(|err| err.to_string())?;
                    let updated = if if_empty {
                        session
                            .store
                            .update_session_title_if_empty(public_id, &title)
                            .await
                    } else {
                        session
                            .store
                            .update_session_title(public_id, Some(&title))
                            .await
                    }
                    .map_err(|err| err.to_string())?;
                    updated.ok_or_else(|| "Session not found".to_string())
                });
                lua_table_result(lua, result, |lua, row| session_row_to_lua_table(lua, &row))
            })?,
        )?;
    }

    // agent.session.branch_list(opts?)
    {
        let manager = app_data.store_manager.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        session_ns.set(
            "branch_list",
            lua.create_function(move |lua, opts: Option<Table>| {
                let manager = manager.clone();
                let execution_ctx = execution_ctx.clone();
                let requested_session = opt_session_id(opts.as_ref());
                let result = bridge_async_result(async move {
                    let session =
                        require_session_store(&manager, &execution_ctx, requested_session).await?;
                    session
                        .store
                        .list_branch_heads(session.row.id)
                        .await
                        .map_err(|e| e.to_string())
                });
                lua_table_result(lua, result, |lua, rows| {
                    branch_rows_to_lua_table(lua, &rows)
                })
            })?,
        )?;
    }

    // agent.session.branch_create(name, opts?)
    {
        let manager = app_data.store_manager.clone();
        let agent_manager = app_data.agent_manager.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        session_ns.set(
            "branch_create",
            lua.create_function(move |lua, (name, opts): (String, Option<Table>)| {
                let manager = manager.clone();
                let agent_manager = agent_manager.clone();
                let execution_ctx = execution_ctx.clone();
                let requested_session = opt_session_id(opts.as_ref());
                let requested_slot = opt_slot_id(opts.as_ref());
                let from_turn_index = opt_from_turn_index(opts.as_ref());
                let activate = opt_activate(opts.as_ref(), false);
                let result = bridge_async_result(async move {
                    let session =
                        require_session_store(&manager, &execution_ctx, requested_session).await?;
                    let is_current_session = current_session_matches(
                        &execution_ctx,
                        &session.session_ref,
                        requested_slot.as_deref(),
                    )?;
                    let branch = session
                        .store
                        .create_branch_head_from_turn_index(
                            session.row.id,
                            &name,
                            from_turn_index,
                            activate && !is_current_session,
                        )
                        .await
                        .map_err(|e| e.to_string())?;
                    if activate && is_current_session {
                        if let Ok(mut lock) = execution_ctx.lock() {
                            lock.pending_branch_checkout = Some(name.clone());
                        }
                    } else if activate {
                        let session_ref_str = format_session_reference(
                            &session.session_ref.public_id,
                            &session.selector,
                        );
                        let _ = agent_manager
                            .reload_session_if_live(&session_ref_str, requested_slot.as_deref())
                            .await
                            .map_err(|e| e.to_string())?;
                    }
                    Ok::<_, String>((branch, activate && is_current_session))
                });
                lua_table_result(lua, result, |lua, (branch, deferred)| {
                    branch_row_to_lua_table(lua, &branch, deferred)
                })
            })?,
        )?;
    }

    // agent.session.branch_siblings(source_turn_id, opts?)
    {
        let manager = app_data.store_manager.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        session_ns.set(
            "branch_siblings",
            lua.create_function(move |lua, (source_turn_id, opts): (i64, Option<Table>)| {
                let manager = manager.clone();
                let execution_ctx = execution_ctx.clone();
                let requested_session = opt_session_id(opts.as_ref());
                let result = bridge_async_result(async move {
                    let session =
                        require_session_store(&manager, &execution_ctx, requested_session).await?;
                    session
                        .store
                        .list_branch_heads_from_source_turn(session.row.id, source_turn_id)
                        .await
                        .map_err(|e| e.to_string())
                });
                lua_table_result(lua, result, |lua, rows| {
                    branch_rows_to_lua_table(lua, &rows)
                })
            })?,
        )?;
    }

    // agent.session.branch_checkout(branch, opts?)
    {
        let manager = app_data.store_manager.clone();
        let agent_manager = app_data.agent_manager.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        session_ns.set(
            "branch_checkout",
            lua.create_function(move |lua, (branch, opts): (String, Option<Table>)| {
                let manager = manager.clone();
                let agent_manager = agent_manager.clone();
                let execution_ctx = execution_ctx.clone();
                let requested_session = opt_session_id(opts.as_ref());
                let requested_slot = opt_slot_id(opts.as_ref());
                let result = bridge_async_result(async move {
                    let session =
                        require_session_store(&manager, &execution_ctx, requested_session).await?;
                    let is_current_session = current_session_matches(
                        &execution_ctx,
                        &session.session_ref,
                        requested_slot.as_deref(),
                    )?;
                    if is_current_session {
                        let branch_row = session
                            .store
                            .list_branch_heads(session.row.id)
                            .await
                            .map_err(|e| e.to_string())?
                            .into_iter()
                            .find(|head| head.name == branch)
                            .ok_or_else(|| format!("Branch '{}' not found", branch))?;
                        if let Ok(mut lock) = execution_ctx.lock() {
                            lock.pending_branch_checkout = Some(branch.clone());
                        }
                        return Ok::<_, String>((branch_row, true));
                    }

                    let branch_row = session
                        .store
                        .checkout_branch_head_by_name(session.row.id, &branch)
                        .await
                        .map_err(|e| e.to_string())?
                        .ok_or_else(|| format!("Branch '{}' not found", branch))?;
                    let session_ref_str =
                        format_session_reference(&session.session_ref.public_id, &session.selector);
                    let _ = agent_manager
                        .reload_session_if_live(&session_ref_str, requested_slot.as_deref())
                        .await
                        .map_err(|e| e.to_string())?;
                    Ok::<_, String>((branch_row, false))
                });
                lua_table_result(lua, result, |lua, (branch, deferred)| {
                    branch_row_to_lua_table(lua, &branch, deferred)
                })
            })?,
        )?;
    }

    // agent.session.list(limit?, offset?)
    {
        let manager = app_data.store_manager.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        session_ns.set(
            "list",
            lua.create_function(
                move |lua, (limit, offset): (Option<usize>, Option<usize>)| {
                    let limit = limit.unwrap_or(20);
                    let offset = offset.unwrap_or(0);
                    let manager = manager.clone();
                    let execution_ctx = execution_ctx.clone();
                    let result = bridge_async_result(async move {
                        let selector = current_session_store_selector(&execution_ctx)?;
                        let store = manager.open(&selector).await.map_err(|e| e.to_string())?;
                        store
                            .list_session_rows(limit, offset, None)
                            .await
                            .map_err(|e| e.to_string())
                    });
                    match result {
                        Ok(rows) => {
                            let out = lua.create_table()?;
                            for (i, row) in rows.iter().enumerate() {
                                out.set(i + 1, session_row_to_lua_table(lua, row)?)?;
                            }
                            Ok(ok_value(Value::Table(out)))
                        }
                        Err(err) => nil_err(lua, &err),
                    }
                },
            )?,
        )?;
    }

    agent_table.set("session", session_ns)?;
    Ok(())
}
