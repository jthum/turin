use mlua::{Lua, Result as LuaResult, Table, Value};

use crate::harness::globals::{ActiveHarnessExecutionContext, HarnessAppData};
use crate::harness::stdlib::binding_common::{
    bool_err, bridge_async, bridge_async_display_err, bridge_async_result, nil_err, nil_ok,
    ok_bool, ok_value, string_ok, string_value,
};
use crate::harness::stdlib::governance_support::{
    apply_active_grant_ceiling_to_peer_delegation, parse_delegated_capabilities,
    require_capability as require_governance_capability,
    require_child_agent as require_child_agent_governance,
};
use crate::harness::stdlib::identity_support::{
    get_active_identity, identity_to_lua_table, session_row_to_lua_table,
};
use crate::harness::stdlib::policy_support::{policy_bool, policy_u64, runtime_policy_snapshot};
use crate::kernel::session::ExecutionConflictPolicy;
use crate::kernel::session::QueuedTask;
use crate::kernel::session_refs::{
    SessionReference, format_session_reference, parse_session_reference,
};
use crate::persistence::manager::StoreSelector;

fn active_trace_id(app_data: &HarnessAppData) -> Option<String> {
    app_data
        .execution_ctx
        .lock()
        .ok()
        .and_then(|ctx| ctx.trace_id.clone())
}

fn queue_max(snapshot: &std::collections::HashMap<String, serde_json::Value>) -> usize {
    policy_u64(snapshot, "queue.max_depth", 1024) as usize
}

async fn queue_push_one(
    execution_ctx: &ActiveHarnessExecutionContext,
    task: QueuedTask,
    queue_max: usize,
    push_front: bool,
) -> Result<(), String> {
    let queue = execution_ctx
        .lock()
        .ok()
        .and_then(|lock| lock.queue.clone())
        .ok_or_else(|| "No active session queue".to_string())?;
    let mut q = queue.lock().await;
    if q.len() >= queue_max {
        return Err(format!(
            "Policy denial: queue.max_depth={} reached",
            queue_max
        ));
    }
    if push_front {
        q.push_front(task);
    } else {
        q.push_back(task);
    }
    Ok(())
}

async fn queue_push_many(
    execution_ctx: &ActiveHarnessExecutionContext,
    tasks: Vec<QueuedTask>,
    queue_max: usize,
) -> Result<(), String> {
    let queue = execution_ctx
        .lock()
        .ok()
        .and_then(|lock| lock.queue.clone())
        .ok_or_else(|| "No active session queue".to_string())?;
    let mut q = queue.lock().await;
    if q.len().saturating_add(tasks.len()) > queue_max {
        return Err(format!(
            "Policy denial: queue.max_depth={} would be exceeded",
            queue_max
        ));
    }
    for task in tasks {
        q.push_back(task);
    }
    Ok(())
}

fn current_session_store_selector(
    execution_ctx: &ActiveHarnessExecutionContext,
) -> Result<StoreSelector, String> {
    execution_ctx
        .lock()
        .map_err(|_| "execution context mutex poisoned".to_string())
        .map(|lock| {
            lock.session_store_selector
                .clone()
                .unwrap_or_else(|| StoreSelector::Alias("state".to_string()))
        })
}

fn resolve_session_reference(
    execution_ctx: &ActiveHarnessExecutionContext,
    requested: Option<String>,
) -> Result<SessionReference, String> {
    let implicit_selector = current_session_store_selector(execution_ctx)?;
    let raw = match requested {
        Some(session_id) => session_id,
        None => execution_ctx
            .lock()
            .map_err(|_| "execution context mutex poisoned".to_string())?
            .session_id
            .clone()
            .ok_or_else(|| "No active session context".to_string())?,
    };
    let mut session_ref = parse_session_reference(&raw).map_err(|e| e.to_string())?;
    if session_ref.store_selector.is_none() {
        session_ref.store_selector = Some(implicit_selector);
    }
    Ok(session_ref)
}

fn current_session_matches(
    execution_ctx: &ActiveHarnessExecutionContext,
    target: &SessionReference,
    target_slot_id: Option<&str>,
) -> Result<bool, String> {
    let (current, current_slot_id) = {
        let lock = execution_ctx
            .lock()
            .map_err(|_| "execution context mutex poisoned".to_string())?;
        (lock.session_id.clone(), lock.runtime_slot_id.clone())
    };
    let Some(current) = current else {
        return Ok(false);
    };
    let current_ref = resolve_session_reference(execution_ctx, Some(current))?;
    Ok(current_ref.public_id == target.public_id
        && current_ref.store_selector == target.store_selector
        && match target_slot_id {
            Some(slot_id) => current_slot_id.as_deref() == Some(slot_id),
            None => true,
        })
}

fn opt_session_id(opts: Option<&Table>) -> Option<String> {
    opts.and_then(|table| table.get::<String>("session_id").ok())
}

fn opt_slot_id(opts: Option<&Table>) -> Option<String> {
    opts.and_then(|table| table.get::<String>("slot_id").ok())
}

fn opt_from_turn_index(opts: Option<&Table>) -> Option<u32> {
    opts.and_then(|table| table.get::<u32>("from_turn_index").ok())
}

fn opt_activate(opts: Option<&Table>, default: bool) -> bool {
    opts.and_then(|table| table.get::<bool>("activate").ok())
        .unwrap_or(default)
}

fn opt_conflict_policy(
    opts: Option<&Table>,
) -> std::result::Result<Option<ExecutionConflictPolicy>, String> {
    let Some(opts) = opts else {
        return Ok(None);
    };
    let Ok(raw) = opts.get::<String>("conflict_policy") else {
        return Ok(None);
    };
    raw.parse().map(Some)
}

fn bytes_to_simple_uuid(bytes: &[u8]) -> String {
    uuid::Uuid::from_slice(bytes)
        .map(|uuid| uuid.simple().to_string())
        .unwrap_or_else(|_| {
            let mut out = String::with_capacity(bytes.len() * 2);
            for byte in bytes {
                use std::fmt::Write as _;
                let _ = write!(&mut out, "{:02x}", byte);
            }
            out
        })
}

fn branch_row_to_lua_table(
    lua: &Lua,
    row: &crate::persistence::schema::BranchHeadRow,
    deferred: bool,
) -> LuaResult<Table> {
    let table = lua.create_table()?;
    table.set("branch_id", bytes_to_simple_uuid(&row.public_id))?;
    table.set("name", row.name.clone())?;
    match row.head_turn_depth {
        Some(depth) => table.set("head_turn_index", depth)?,
        None => table.set("head_turn_index", Value::Nil)?,
    }
    table.set("active", row.is_active)?;
    table.set("deferred", deferred)?;
    table.set("created_at", row.created_at.clone())?;
    Ok(table)
}

pub fn register_agent_bindings(lua: &Lua, app_data: &HarnessAppData) -> LuaResult<()> {
    let agent_table = lua.create_table()?;
    let session_ns = lua.create_table()?;
    let mode_ns = lua.create_table()?;

    let agent_manager = app_data.agent_manager.clone();

    // agent.spawn(prompt, opts?)
    let spawn_q = app_data.execution_ctx.clone();
    let spawn_policy_snapshot = app_data.clone();
    let spawn_depth = app_data.spawn_depth;
    agent_table.set(
        "spawn",
        lua.create_function(move |lua, (prompt, opts): (String, Option<Table>)| {
            if let Err(err) =
                require_governance_capability(&spawn_policy_snapshot, "runtime.agent.spawn")
            {
                return nil_err(lua, &err);
            }
            let snapshot =
                runtime_policy_snapshot(&spawn_policy_snapshot).map_err(mlua::Error::runtime)?;
            if !policy_bool(&snapshot, "spawn.enabled", true) {
                return nil_err(lua, "Policy denial: spawn.enabled=false");
            }
            let max_depth = policy_u64(&snapshot, "spawn.max_depth", 3) as u32;
            if spawn_depth >= max_depth {
                return nil_err(lua, "Policy denial: spawn.max_depth exceeded");
            }
            let conflict_policy = match opt_conflict_policy(opts.as_ref()) {
                Ok(conflict_policy) => conflict_policy,
                Err(err) => return nil_err(lua, &err),
            };
            let spawn_q = spawn_q.clone();
            let queue_max = queue_max(&snapshot);
            let trace_id = active_trace_id(&spawn_policy_snapshot);
            let enqueue_res = bridge_async_result(async move {
                queue_push_one(
                    &spawn_q,
                    QueuedTask::ad_hoc(prompt.clone())
                        .with_inherited_trace(trace_id.as_deref())
                        .with_conflict_policy(conflict_policy),
                    queue_max,
                    false,
                )
                .await
            });
            match enqueue_res {
                Ok(()) => {
                    let token = format!("q_{}", uuid::Uuid::now_v7().simple());
                    string_ok(lua, &token)
                }
                Err(err) => nil_err(lua, &err),
            }
        })?,
    )?;

    // agent.complete
    {
        let manager = app_data.agent_manager.clone();
        let default_agent = app_data.config.agent.id.clone();
        let complete_policy_snapshot = app_data.clone();
        agent_table.set(
            "complete",
            lua.create_function(move |lua, (prompt, opts): (String, Option<Table>)| {
                if let Err(err) =
                    require_governance_capability(&complete_policy_snapshot, "runtime.agent.submit")
                {
                    return nil_err(lua, &err);
                }
                if let Err(err) =
                    require_governance_capability(&complete_policy_snapshot, "runtime.agent.await")
                {
                    return nil_err(lua, &err);
                }
                let snapshot = runtime_policy_snapshot(&complete_policy_snapshot)
                    .map_err(mlua::Error::runtime)?;
                if !policy_bool(&snapshot, "spawn.enabled", true) {
                    return nil_err(lua, "Policy denial: spawn.enabled=false");
                }
                let target_agent = opts
                    .as_ref()
                    .and_then(|t| t.get::<String>("agent_id").ok())
                    .unwrap_or_else(|| default_agent.clone());
                if let Err(err) =
                    require_child_agent_governance(&complete_policy_snapshot, &target_agent)
                {
                    return nil_err(lua, &err);
                }
                let delegated_capabilities = parse_delegated_capabilities(
                    &complete_policy_snapshot,
                    opts.as_ref(),
                    "capabilities",
                    "agent.complete",
                )?;
                let delegated_capabilities = apply_active_grant_ceiling_to_peer_delegation(
                    &complete_policy_snapshot,
                    delegated_capabilities,
                    "agent.complete",
                )?;
                let timeout_ms = opts.as_ref().and_then(|t| t.get::<u64>("timeout_ms").ok());
                let trace_id = active_trace_id(&complete_policy_snapshot);

                let manager_submit = manager.clone();
                let request_id = bridge_async_display_err(async move {
                    manager_submit
                        .submit(
                            &target_agent,
                            QueuedTask::ad_hoc(prompt).with_inherited_trace(trace_id.as_deref()),
                            delegated_capabilities,
                        )
                        .await
                });
                let request_id = match request_id {
                    Ok(id) => id,
                    Err(err) => return nil_err(lua, &err),
                };

                let manager_await = manager.clone();
                let result = bridge_async_display_err(async move {
                    manager_await.await_result(&request_id, timeout_ms).await
                });
                match result {
                    Ok(res) => {
                        if let Some(err) = res.error {
                            nil_err(lua, &err)
                        } else if let Some(output) = res.output {
                            string_ok(lua, &output)
                        } else {
                            string_ok(lua, "")
                        }
                    }
                    Err(err) => nil_err(lua, &err),
                }
            })?,
        )?;
    }

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
            let aq = aq.clone();
            let snapshot =
                runtime_policy_snapshot(&queue_policy_snapshot).map_err(mlua::Error::runtime)?;
            let queue_max = queue_max(&snapshot);
            let trace_id = active_trace_id(&queue_policy_snapshot);
            let res = bridge_async_result(async move {
                queue_push_one(
                    &aq,
                    QueuedTask::ad_hoc(cmd).with_inherited_trace(trace_id.as_deref()),
                    queue_max,
                    false,
                )
                .await
            });
            match res {
                Ok(()) => Ok(ok_bool()),
                Err(err) => bool_err(lua, &err),
            }
        })?,
    )?;

    // agent.session.queue_next
    let aq2 = app_data.execution_ctx.clone();
    let queue_next_policy_snapshot = app_data.clone();
    session_ns.set(
        "queue_next",
        lua.create_function(move |lua, cmd: String| {
            let aq = aq2.clone();
            let snapshot = runtime_policy_snapshot(&queue_next_policy_snapshot)
                .map_err(mlua::Error::runtime)?;
            let queue_max = queue_max(&snapshot);
            let trace_id = active_trace_id(&queue_next_policy_snapshot);
            let res = bridge_async_result(async move {
                queue_push_one(
                    &aq,
                    QueuedTask::ad_hoc(cmd).with_inherited_trace(trace_id.as_deref()),
                    queue_max,
                    true,
                )
                .await
            });
            match res {
                Ok(()) => Ok(ok_bool()),
                Err(err) => bool_err(lua, &err),
            }
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
            match res {
                Ok(()) => Ok(ok_bool()),
                Err(err) => bool_err(lua, &err),
            }
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
                    let session_ref = resolve_session_reference(&execution_ctx, Some(session_id))?;
                    let selector = session_ref.store_selector.clone().ok_or_else(|| {
                        "Session reference store could not be resolved".to_string()
                    })?;
                    let store = manager.open(&selector).await.map_err(|e| e.to_string())?;
                    let uuid =
                        uuid::Uuid::parse_str(&session_ref.public_id).map_err(|e| e.to_string())?;
                    let row = store
                        .get_session_row_by_public_id(uuid)
                        .await
                        .map_err(|e| e.to_string())?;
                    Ok::<_, String>(row)
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
                    let session_ref = resolve_session_reference(&execution_ctx, requested_session)?;
                    let selector = session_ref.store_selector.clone().ok_or_else(|| {
                        "Session reference store could not be resolved".to_string()
                    })?;
                    let store = manager.open(&selector).await.map_err(|e| e.to_string())?;
                    let uuid =
                        uuid::Uuid::parse_str(&session_ref.public_id).map_err(|e| e.to_string())?;
                    let row = store
                        .get_session_row_by_public_id(uuid)
                        .await
                        .map_err(|e| e.to_string())?
                        .ok_or_else(|| "Session not found".to_string())?;
                    store
                        .list_branch_heads(row.id)
                        .await
                        .map_err(|e| e.to_string())
                });
                match result {
                    Ok(rows) => {
                        let out = lua.create_table()?;
                        for (i, row) in rows.iter().enumerate() {
                            out.set(i + 1, branch_row_to_lua_table(lua, row, false)?)?;
                        }
                        Ok(ok_value(Value::Table(out)))
                    }
                    Err(err) => nil_err(lua, &err),
                }
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
                    let session_ref = resolve_session_reference(&execution_ctx, requested_session)?;
                    let is_current_session = current_session_matches(
                        &execution_ctx,
                        &session_ref,
                        requested_slot.as_deref(),
                    )?;
                    let selector = session_ref.store_selector.clone().ok_or_else(|| {
                        "Session reference store could not be resolved".to_string()
                    })?;
                    let store = manager.open(&selector).await.map_err(|e| e.to_string())?;
                    let uuid =
                        uuid::Uuid::parse_str(&session_ref.public_id).map_err(|e| e.to_string())?;
                    let row = store
                        .get_session_row_by_public_id(uuid)
                        .await
                        .map_err(|e| e.to_string())?
                        .ok_or_else(|| "Session not found".to_string())?;
                    let branch = store
                        .create_branch_head_from_turn_index(
                            row.id,
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
                        let session_ref_str =
                            format_session_reference(&session_ref.public_id, &selector);
                        let _ = agent_manager
                            .reload_session_if_live(
                                &session_ref_str,
                                requested_slot.as_deref(),
                            )
                            .await
                            .map_err(|e| e.to_string())?;
                    }
                    Ok::<_, String>((branch, activate && is_current_session))
                });
                match result {
                    Ok((branch, deferred)) => Ok(ok_value(Value::Table(branch_row_to_lua_table(
                        lua, &branch, deferred,
                    )?))),
                    Err(err) => nil_err(lua, &err),
                }
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
                    let session_ref = resolve_session_reference(&execution_ctx, requested_session)?;
                    let is_current_session = current_session_matches(
                        &execution_ctx,
                        &session_ref,
                        requested_slot.as_deref(),
                    )?;
                    let selector = session_ref.store_selector.clone().ok_or_else(|| {
                        "Session reference store could not be resolved".to_string()
                    })?;
                    let store = manager.open(&selector).await.map_err(|e| e.to_string())?;
                    let uuid =
                        uuid::Uuid::parse_str(&session_ref.public_id).map_err(|e| e.to_string())?;
                    let row = store
                        .get_session_row_by_public_id(uuid)
                        .await
                        .map_err(|e| e.to_string())?
                        .ok_or_else(|| "Session not found".to_string())?;
                    if is_current_session {
                        let branch_row = store
                            .list_branch_heads(row.id)
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

                    let branch_row = store
                        .checkout_branch_head_by_name(row.id, &branch)
                        .await
                        .map_err(|e| e.to_string())?
                        .ok_or_else(|| format!("Branch '{}' not found", branch))?;
                    let session_ref_str =
                        format_session_reference(&session_ref.public_id, &selector);
                    let _ = agent_manager
                        .reload_session_if_live(&session_ref_str, requested_slot.as_deref())
                        .await
                        .map_err(|e| e.to_string())?;
                    Ok::<_, String>((branch_row, false))
                });
                match result {
                    Ok((branch, deferred)) => Ok(ok_value(Value::Table(branch_row_to_lua_table(
                        lua, &branch, deferred,
                    )?))),
                    Err(err) => nil_err(lua, &err),
                }
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
                            .list_session_rows(limit, offset)
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

    let execution_ctx_get = app_data.execution_ctx.clone();
    mode_ns.set(
        "get",
        lua.create_function(move |lua, ()| {
            let mode = execution_ctx_get
                .lock()
                .map_err(|_| mlua::Error::runtime("harness execution context mutex poisoned"))?
                .session_mode
                .clone()
                .unwrap_or(crate::kernel::config::AgentMode::Auto);
            let mode_str = match mode {
                crate::kernel::config::AgentMode::Auto => "auto",
                crate::kernel::config::AgentMode::Stateful => "stateful",
                crate::kernel::config::AgentMode::Stateless => "stateless",
            };
            string_value(lua, mode_str)
        })?,
    )?;

    let execution_ctx_set = app_data.execution_ctx.clone();
    mode_ns.set(
        "set",
        lua.create_function(move |lua, m: String| {
            let mode = match m.as_str() {
                "stateful" => crate::kernel::config::AgentMode::Stateful,
                "stateless" => crate::kernel::config::AgentMode::Stateless,
                "auto" => crate::kernel::config::AgentMode::Auto,
                _ => {
                    return bool_err(lua, "invalid mode; expected auto|stateful|stateless");
                }
            };
            if let Ok(mut lock) = execution_ctx_set.lock() {
                lock.session_mode = Some(mode);
            }
            Ok(ok_bool())
        })?,
    )?;

    agent_table.set("session", session_ns)?;
    agent_table.set("mode", mode_ns)?;

    // Deprecated send
    let send_policy_snapshot = app_data.clone();
    agent_table.set(
        "send",
        lua.create_function(move |_lua, (id, prompt): (String, String)| {
            if let Err(err) =
                require_governance_capability(&send_policy_snapshot, "runtime.agent.submit")
            {
                return Err(mlua::Error::runtime(err));
            }
            if let Err(err) = require_child_agent_governance(&send_policy_snapshot, &id) {
                return Err(mlua::Error::runtime(err));
            }
            let delegated_capabilities = apply_active_grant_ceiling_to_peer_delegation(
                &send_policy_snapshot,
                None,
                "agent.send",
            )?;
            let m = agent_manager.clone();
            bridge_async(async {
                let _ = m
                    .send(&id, QueuedTask::ad_hoc(prompt), delegated_capabilities)
                    .await;
            });
            Ok(ok_bool())
        })?,
    )?;

    lua.globals().set("agent", agent_table)?;
    Ok(())
}
