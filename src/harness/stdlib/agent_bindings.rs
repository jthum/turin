use mlua::{Lua, Result as LuaResult, Table, Value};

use crate::harness::globals::{ActiveSessionQueue, HarnessAppData, block_on_current};
use crate::harness::stdlib::binding_common::{bool_err, nil_err, ok_bool, string_ok};
use crate::harness::stdlib::identity_support::{
    get_active_identity, identity_to_lua_table, session_row_to_lua_table,
};
use crate::harness::stdlib::policy_support::{policy_bool, policy_u64, runtime_policy_snapshot};
use crate::kernel::session::QueuedTask;

fn queue_max(snapshot: &std::collections::HashMap<String, serde_json::Value>) -> usize {
    policy_u64(snapshot, "queue.max_depth", 1024) as usize
}

async fn queue_push_one(
    aq: &ActiveSessionQueue,
    task: QueuedTask,
    queue_max: usize,
    push_front: bool,
) -> Result<(), String> {
    if let Some(q) = &*aq.lock().await {
        let mut q = q.lock().await;
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
    } else {
        Err("No active session queue".to_string())
    }
}

async fn queue_push_many(
    aq: &ActiveSessionQueue,
    tasks: Vec<QueuedTask>,
    queue_max: usize,
) -> Result<(), String> {
    if let Some(q) = &*aq.lock().await {
        let mut q = q.lock().await;
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
    } else {
        Err("No active session queue".to_string())
    }
}

pub fn register_agent_bindings(lua: &Lua, app_data: &HarnessAppData) -> LuaResult<()> {
    let agent_table = lua.create_table()?;
    let session_ns = lua.create_table()?;
    let mode_ns = lua.create_table()?;

    let agent_manager = app_data.agent_manager.clone();

    // agent.spawn (local subtask enqueue for current session queue)
    let spawn_q = app_data.queue.clone();
    let spawn_policy_snapshot = app_data.clone();
    let spawn_depth = app_data.spawn_depth;
    agent_table.set(
        "spawn",
        lua.create_function(move |lua, (prompt, _opts): (String, Option<Table>)| {
            let snapshot =
                runtime_policy_snapshot(&spawn_policy_snapshot).map_err(mlua::Error::runtime)?;
            if !policy_bool(&snapshot, "spawn.enabled", true) {
                return Ok((
                    Value::Nil,
                    Value::String(lua.create_string("Policy denial: spawn.enabled=false")?),
                ));
            }
            let max_depth = policy_u64(&snapshot, "spawn.max_depth", 3) as u32;
            if spawn_depth >= max_depth {
                return nil_err(lua, "Policy denial: spawn.max_depth exceeded");
            }
            let spawn_q = spawn_q.clone();
            let queue_max = queue_max(&snapshot);
            let enqueue_res = block_on_current(async move {
                queue_push_one(
                    &spawn_q,
                    QueuedTask::ad_hoc(prompt.clone()),
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
                let snapshot = runtime_policy_snapshot(&complete_policy_snapshot)
                    .map_err(mlua::Error::runtime)?;
                if !policy_bool(&snapshot, "spawn.enabled", true) {
                    return Ok((
                        Value::Nil,
                        Value::String(lua.create_string("Policy denial: spawn.enabled=false")?),
                    ));
                }
                let target_agent = opts
                    .as_ref()
                    .and_then(|t| t.get::<String>("agent_id").ok())
                    .unwrap_or_else(|| default_agent.clone());
                let timeout_ms = opts.as_ref().and_then(|t| t.get::<u64>("timeout_ms").ok());

                let manager_submit = manager.clone();
                let request_id = block_on_current(async move {
                    manager_submit
                        .submit(&target_agent, QueuedTask::ad_hoc(prompt))
                        .await
                        .map_err(|e| e.to_string())
                });
                let request_id = match request_id {
                    Ok(id) => id,
                    Err(err) => return nil_err(lua, &err),
                };

                let manager_await = manager.clone();
                let result = block_on_current(async move {
                    manager_await
                        .await_result(&request_id, timeout_ms)
                        .await
                        .map_err(|e| e.to_string())
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
    let aq = app_data.queue.clone();
    let queue_policy_snapshot = app_data.clone();
    session_ns.set(
        "queue",
        lua.create_function(move |lua, cmd: String| {
            let aq = aq.clone();
            let snapshot =
                runtime_policy_snapshot(&queue_policy_snapshot).map_err(mlua::Error::runtime)?;
            let queue_max = queue_max(&snapshot);
            let res = block_on_current(async move {
                queue_push_one(&aq, QueuedTask::ad_hoc(cmd), queue_max, false).await
            });
            match res {
                Ok(()) => Ok(ok_bool()),
                Err(err) => bool_err(lua, &err),
            }
        })?,
    )?;

    // agent.session.queue_next
    let aq2 = app_data.queue.clone();
    let queue_next_policy_snapshot = app_data.clone();
    session_ns.set(
        "queue_next",
        lua.create_function(move |lua, cmd: String| {
            let aq = aq2.clone();
            let snapshot = runtime_policy_snapshot(&queue_next_policy_snapshot)
                .map_err(mlua::Error::runtime)?;
            let queue_max = queue_max(&snapshot);
            let res = block_on_current(async move {
                queue_push_one(&aq, QueuedTask::ad_hoc(cmd), queue_max, true).await
            });
            match res {
                Ok(()) => Ok(ok_bool()),
                Err(err) => bool_err(lua, &err),
            }
        })?,
    )?;

    // agent.session.queue_all
    let aq3 = app_data.queue.clone();
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
            let tasks = items
                .into_iter()
                .map(QueuedTask::ad_hoc)
                .collect::<Vec<_>>();
            let res = block_on_current(async move { queue_push_many(&aq, tasks, queue_max).await });
            match res {
                Ok(()) => Ok(ok_bool()),
                Err(err) => bool_err(lua, &err),
            }
        })?,
    )?;

    // agent.session.load(session_id)
    {
        let manager = app_data.store_manager.clone();
        session_ns.set(
            "load",
            lua.create_function(move |lua, session_id: String| {
                let manager = manager.clone();
                let result = block_on_current(async move {
                    let store = manager.get_default().await.map_err(|e| e.to_string())?;
                    let uuid = uuid::Uuid::parse_str(&session_id).map_err(|e| e.to_string())?;
                    let row = store
                        .get_session_row_by_public_id(uuid)
                        .await
                        .map_err(|e| e.to_string())?;
                    Ok::<_, String>(row)
                });
                match result {
                    Ok(Some(row)) => Ok((
                        Value::Table(session_row_to_lua_table(lua, &row)?),
                        Value::Nil,
                    )),
                    Ok(None) => Ok((Value::Nil, Value::Nil)),
                    Err(err) => nil_err(lua, &err),
                }
            })?,
        )?;
    }

    // agent.session.list(limit?, offset?)
    {
        let manager = app_data.store_manager.clone();
        session_ns.set(
            "list",
            lua.create_function(
                move |lua, (limit, offset): (Option<usize>, Option<usize>)| {
                    let limit = limit.unwrap_or(20);
                    let offset = offset.unwrap_or(0);
                    let manager = manager.clone();
                    let result = block_on_current(async move {
                        let store = manager.get_default().await.map_err(|e| e.to_string())?;
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
                            Ok((Value::Table(out), Value::Nil))
                        }
                        Err(err) => nil_err(lua, &err),
                    }
                },
            )?,
        )?;
    }

    let sm1 = app_data.active_session_mode.clone();
    mode_ns.set(
        "get",
        lua.create_function(move |lua, ()| {
            let mode = sm1
                .lock()
                .unwrap()
                .clone()
                .unwrap_or(crate::kernel::config::AgentMode::Auto);
            let mode_str = match mode {
                crate::kernel::config::AgentMode::Auto => "auto",
                crate::kernel::config::AgentMode::Stateful => "stateful",
                crate::kernel::config::AgentMode::Stateless => "stateless",
            };
            Ok(Value::String(lua.create_string(mode_str)?))
        })?,
    )?;

    let sm2 = app_data.active_session_mode.clone();
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
            if let Ok(mut lock) = sm2.lock() {
                *lock = Some(mode);
            }
            Ok(ok_bool())
        })?,
    )?;

    agent_table.set("session", session_ns)?;
    agent_table.set("mode", mode_ns)?;

    // Deprecated send
    agent_table.set(
        "send",
        lua.create_function(move |_lua, (id, prompt): (String, String)| {
            let m = agent_manager.clone();
            block_on_current(async {
                let _ = m.send(&id, QueuedTask::ad_hoc(prompt)).await;
            });
            Ok(ok_bool())
        })?,
    )?;

    lua.globals().set("agent", agent_table)?;
    Ok(())
}
