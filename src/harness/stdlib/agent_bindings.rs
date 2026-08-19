mod branch_lua;
mod options;
mod queue;
mod session_store;
mod sidestep_graph;

use std::collections::BTreeMap;

use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};

use crate::harness::globals::{ActiveHarnessExecutionContext, HarnessAppData};
use crate::harness::stdlib::binding_common::{
    bridge_async, bridge_async_display_err, bridge_async_result, lua_bool_result,
    lua_string_result, lua_table_result, lua_value_result, nil_err, nil_ok, ok_value, string_ok,
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
use crate::kernel::prepare_persisted_session_sidestep;
use crate::kernel::session::QueuedTask;
use crate::kernel::session_refs::format_session_reference;
use crate::kernel::task_promotion::promote_task_result;
use branch_lua::{branch_row_to_lua_table, branch_rows_to_lua_table};
use options::{
    opt_activate, opt_conflict_policy, opt_execution_overrides, opt_from_turn_index,
    opt_peer_agent_id, opt_session_id, opt_sidestep_context_target, opt_sidestep_mode, opt_slot_id,
    peer_prompt_task, sidestep_opts_table_from_value,
};
use queue::queue_push_many;
pub(crate) use queue::{active_trace_id, queue_max, queue_push_one};
use session_store::{
    current_completed_task_results, current_session_matches, current_session_store_selector,
    lookup_session_store, require_session_store,
};
use sidestep_graph::{attach_sidestep_graph_relation, opt_sidestep_graph_relation};

struct PreparedPeerSubmission {
    target_agent: String,
    origin_session_id: String,
    origin_turn_id: Option<i64>,
    thread_key: String,
    task: QueuedTask,
    delegated_capabilities: Option<BTreeMap<String, bool>>,
}

fn prepare_peer_submission(
    lua: &Lua,
    app_data: &HarnessAppData,
    default_agent: &str,
    prompt: String,
    opts: Option<&Table>,
    caller_label: &str,
) -> LuaResult<Result<PreparedPeerSubmission, String>> {
    if let Err(err) = require_governance_capability(app_data, "runtime.agent.submit") {
        return Ok(Err(err));
    }

    let snapshot = runtime_policy_snapshot(app_data).map_err(mlua::Error::runtime)?;
    if !policy_bool(&snapshot, "spawn.enabled", true) {
        return Ok(Err("Policy denial: spawn.enabled=false".to_string()));
    }

    let target_agent = opt_peer_agent_id(opts, default_agent);
    if let Err(err) = require_child_agent_governance(app_data, &target_agent) {
        return Ok(Err(err));
    }

    let delegated_capabilities =
        parse_delegated_capabilities(app_data, opts, "capabilities", caller_label)?;
    let delegated_capabilities = apply_active_grant_ceiling_to_peer_delegation(
        app_data,
        delegated_capabilities,
        caller_label,
    )?;
    let task = match peer_prompt_task(lua, app_data, prompt, opts) {
        Ok(task) => task,
        Err(err) => return Ok(Err(err)),
    };
    let (origin_session_id, origin_turn_id) = {
        let execution = app_data
            .execution_ctx
            .lock()
            .map_err(|_| mlua::Error::runtime("execution context mutex poisoned"))?;
        let session_id = execution
            .session_id
            .clone()
            .ok_or_else(|| mlua::Error::runtime("No active session context"))?;
        let turn_id = execution
            .event_context
            .as_ref()
            .and_then(|context| context.turn_id);
        (session_id, turn_id)
    };
    let thread_key = opts
        .and_then(|table| table.get::<String>("thread").ok())
        .unwrap_or_else(|| "default".to_string());

    Ok(Ok(PreparedPeerSubmission {
        target_agent,
        origin_session_id,
        origin_turn_id,
        thread_key,
        task,
        delegated_capabilities,
    }))
}

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

pub fn register_agent_bindings(lua: &Lua, app_data: &HarnessAppData) -> LuaResult<()> {
    let agent_table = lua.create_table()?;
    let session_ns = lua.create_table()?;

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
            let execution = match opt_execution_overrides(lua, opts.as_ref()) {
                Ok(execution) => execution,
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
                        .with_conflict_policy(conflict_policy)
                        .with_execution(execution),
                    queue_max,
                    false,
                )
                .await
            });
            lua_string_result(lua, enqueue_res)
        })?,
    )?;

    // agent.sidestep(prompt, opts?)
    let sidestep_q = app_data.execution_ctx.clone();
    let sidestep_policy_snapshot = app_data.clone();
    let sidestep_store_manager = app_data.store_manager.clone();
    agent_table.set(
        "sidestep",
        lua.create_function(move |lua, (prompt, opts_value): (String, Option<Value>)| {
            let opts = match sidestep_opts_table_from_value(lua, opts_value) {
                Ok(opts) => opts,
                Err(err) => return nil_err(lua, &err),
            };
            if let Err(err) =
                require_governance_capability(&sidestep_policy_snapshot, "runtime.agent.submit")
            {
                return nil_err(lua, &err);
            }
            let snapshot =
                runtime_policy_snapshot(&sidestep_policy_snapshot).map_err(mlua::Error::runtime)?;
            if !policy_bool(&snapshot, "spawn.enabled", true) {
                return nil_err(lua, "Policy denial: spawn.enabled=false");
            }

            let sidestep_mode = match opt_sidestep_mode(opts.as_ref()) {
                Ok(mode) => mode,
                Err(err) => return nil_err(lua, &err),
            };
            let requested_target = match opt_sidestep_context_target(lua, opts.as_ref()) {
                Ok(target) => target,
                Err(err) => return nil_err(lua, &err),
            };
            let graph_relation = match opt_sidestep_graph_relation(lua, opts.as_ref()) {
                Ok(relation) => relation,
                Err(err) => return nil_err(lua, &err),
            };
            if graph_relation.is_some()
                && let Err(err) =
                    require_governance_capability(&sidestep_policy_snapshot, "runtime.graph.write")
            {
                return nil_err(lua, &err);
            }
            let queue_max = queue_max(&snapshot);
            let trace_id = active_trace_id(&sidestep_policy_snapshot);
            let title = opts.as_ref().and_then(|t| t.get::<String>("title").ok());
            let sidestep_q = sidestep_q.clone();
            let sidestep_store_manager = sidestep_store_manager.clone();
            let enqueue_res = bridge_async_result(async move {
                let (session_id, default_target) = {
                    let lock = sidestep_q
                        .lock()
                        .map_err(|_| "execution context mutex poisoned".to_string())?;
                    (
                        lock.session_id
                            .clone()
                            .ok_or_else(|| "No active session context".to_string())?,
                        lock.execution_context_target
                            .clone()
                            .ok_or_else(|| "No active execution context target".to_string())?,
                    )
                };
                let prepared = prepare_persisted_session_sidestep(
                    &sidestep_store_manager,
                    &session_id,
                    &default_target,
                    sidestep_mode,
                    requested_target,
                )
                .await
                .map_err(|err| err.to_string())?;
                if let Some(relation) = graph_relation {
                    let branch_outcome = prepared.branch_outcome.as_ref().ok_or_else(|| {
                        "sidestep graph relation requires mode='fork_sibling'".to_string()
                    })?;
                    attach_sidestep_graph_relation(
                        &sidestep_store_manager,
                        &session_id,
                        branch_outcome,
                        relation,
                    )
                    .await?;
                }
                let mut task = QueuedTask::ad_hoc(prompt)
                    .with_inherited_trace(trace_id.as_deref())
                    .with_conflict_policy(Some(prepared.conflict_policy))
                    .with_execution(Some(prepared.execution))
                    .with_branch_outcome(prepared.branch_outcome);
                task.title = title;
                queue_push_one(&sidestep_q, task, queue_max, false).await
            });
            lua_string_result(lua, enqueue_res)
        })?,
    )?;

    // agent.promote(task_id, opts?)
    let promote_execution_ctx = app_data.execution_ctx.clone();
    let promote_store_manager = app_data.store_manager.clone();
    let promote_policy_snapshot = app_data.clone();
    agent_table.set(
        "promote",
        lua.create_function(move |lua, (task_id, opts): (String, Option<Table>)| {
            if let Err(err) =
                require_governance_capability(&promote_policy_snapshot, "runtime.agent.submit")
            {
                return nil_err(lua, &err);
            }
            let branch_name = opts
                .as_ref()
                .and_then(|table| table.get::<String>("branch_name").ok());
            let completed_task_results =
                match current_completed_task_results(&promote_execution_ctx) {
                    Ok(results) => results,
                    Err(err) => return nil_err(lua, &err),
                };
            let store_manager = promote_store_manager.clone();
            let result = bridge_async_display_err(async move {
                let completed = {
                    let lock = completed_task_results.read().await;
                    lock.get(&task_id).cloned()
                }
                .ok_or_else(|| anyhow::anyhow!("Task '{}' not found", task_id))?;
                if let Some(branch) = completed.promoted_branch {
                    return Ok::<_, anyhow::Error>(branch);
                }
                let promotion = completed
                    .promotion_candidate
                    .clone()
                    .ok_or_else(|| anyhow::anyhow!("Task '{}' is not promotable", task_id))?;
                let assistant_content = completed
                    .assistant_content
                    .as_ref()
                    .filter(|content| !content.is_empty())
                    .ok_or_else(|| {
                        anyhow::anyhow!("Task '{}' has no promotable assistant output", task_id)
                    })?;
                let input_content = completed
                    .promotion_input_content
                    .as_ref()
                    .filter(|content| !content.is_empty())
                    .ok_or_else(|| {
                        anyhow::anyhow!("Task '{}' is missing promotable task input", task_id)
                    })?;
                let branch = promote_task_result(
                    &store_manager,
                    &promotion,
                    input_content,
                    assistant_content,
                    Some(&task_id),
                    branch_name.as_deref(),
                )
                .await?;
                completed_task_results
                    .write()
                    .await
                    .mark_promoted(&task_id, branch.clone());
                Ok::<_, anyhow::Error>(branch)
            });
            lua_value_result(lua, result, |lua, branch| lua.to_value(&branch))
        })?,
    )?;

    // agent.task(task_id)
    let task_execution_ctx = app_data.execution_ctx.clone();
    let task_policy_snapshot = app_data.clone();
    agent_table.set(
        "task",
        lua.create_function(move |lua, task_id: String| {
            if let Err(err) =
                require_governance_capability(&task_policy_snapshot, "runtime.agent.status")
            {
                return nil_err(lua, &err);
            }
            let completed_task_results = match current_completed_task_results(&task_execution_ctx) {
                Ok(results) => results,
                Err(err) => return nil_err(lua, &err),
            };
            let lookup_task_id = task_id.clone();
            let result = bridge_async(async move {
                let lock = completed_task_results.read().await;
                lock.get(&lookup_task_id).cloned()
            });
            match result {
                Some(result) => Ok(ok_value(lua.to_value(&result)?)),
                None => Ok(nil_ok()),
            }
        })?,
    )?;

    // agent.submit
    {
        let manager = app_data.agent_manager.clone();
        let default_agent = app_data.config.agent.id.clone();
        let submit_policy_snapshot = app_data.clone();
        agent_table.set(
            "submit",
            lua.create_function(move |lua, (prompt, opts): (String, Option<Table>)| {
                let prepared = match prepare_peer_submission(
                    lua,
                    &submit_policy_snapshot,
                    &default_agent,
                    prompt,
                    opts.as_ref(),
                    "agent.submit",
                )? {
                    Ok(prepared) => prepared,
                    Err(err) => return nil_err(lua, &err),
                };

                let manager = manager.clone();
                let result = bridge_async_display_err(async move {
                    manager
                        .submit_linked(
                            &prepared.origin_session_id,
                            prepared.origin_turn_id,
                            &prepared.target_agent,
                            &prepared.thread_key,
                            prepared.task,
                            prepared.delegated_capabilities,
                        )
                        .await
                });
                lua_string_result(lua, result)
            })?,
        )?;
    }

    // agent.ask
    {
        let manager = app_data.agent_manager.clone();
        let default_agent = app_data.config.agent.id.clone();
        let complete_policy_snapshot = app_data.clone();
        agent_table.set(
            "ask",
            lua.create_function(move |lua, (prompt, opts): (String, Option<Table>)| {
                if let Err(err) =
                    require_governance_capability(&complete_policy_snapshot, "runtime.agent.await")
                {
                    return nil_err(lua, &err);
                }
                let prepared = match prepare_peer_submission(
                    lua,
                    &complete_policy_snapshot,
                    &default_agent,
                    prompt,
                    opts.as_ref(),
                    "agent.ask",
                )? {
                    Ok(prepared) => prepared,
                    Err(err) => return nil_err(lua, &err),
                };
                let timeout_ms = opts.as_ref().and_then(|t| t.get::<u64>("timeout_ms").ok());

                let manager_submit = manager.clone();
                let request_id = bridge_async_display_err(async move {
                    manager_submit
                        .submit_linked(
                            &prepared.origin_session_id,
                            prepared.origin_turn_id,
                            &prepared.target_agent,
                            &prepared.thread_key,
                            prepared.task,
                            prepared.delegated_capabilities,
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

    agent_table.set("session", session_ns)?;

    lua.globals().set("agent", agent_table)?;
    Ok(())
}
