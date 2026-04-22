use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};

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
use crate::kernel::event::TaskBranchOutcome;
use crate::kernel::prepare_persisted_session_sidestep;
use crate::kernel::session::{
    ExecutionConflictPolicy, ExecutionContextTarget, QueuedTask, SidestepMode,
    TaskExecutionOverrides,
};
use crate::kernel::session_refs::{
    SessionReference, format_session_reference, parse_session_reference,
};
use crate::kernel::task_promotion::promote_task_result;
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::{GraphEdgeCreate, GraphProvenance, GraphRef};

#[derive(Debug, Clone)]
struct SidestepGraphRelation {
    source: GraphRef,
    relation_kind: String,
    source_role: Option<String>,
    target_role: Option<String>,
    origin_task_id: Option<String>,
    origin_execution_id: Option<String>,
    metadata: Option<serde_json::Value>,
}

fn ensure_local_task_id(task: &mut QueuedTask) -> String {
    if task.task_id.is_empty() {
        task.task_id = format!("t_{}", uuid::Uuid::now_v7().simple());
    }
    task.task_id.clone()
}

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
    mut task: QueuedTask,
    queue_max: usize,
    push_front: bool,
) -> Result<String, String> {
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
    let task_id = ensure_local_task_id(&mut task);
    if push_front {
        q.push_front(task);
    } else {
        q.push_back(task);
    }
    Ok(task_id)
}

async fn queue_push_many(
    execution_ctx: &ActiveHarnessExecutionContext,
    mut tasks: Vec<QueuedTask>,
    queue_max: usize,
) -> Result<Vec<String>, String> {
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
    let mut task_ids = Vec::with_capacity(tasks.len());
    for task in &mut tasks {
        task_ids.push(ensure_local_task_id(task));
    }
    for task in tasks {
        q.push_back(task);
    }
    Ok(task_ids)
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

fn current_completed_task_results(
    execution_ctx: &ActiveHarnessExecutionContext,
) -> Result<crate::kernel::session::CompletedLocalTaskResultsHandle, String> {
    execution_ctx
        .lock()
        .map_err(|_| "execution context mutex poisoned".to_string())?
        .completed_task_results
        .clone()
        .ok_or_else(|| "No active session completed-task cache".to_string())
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

fn opt_execution_overrides(
    lua: &Lua,
    opts: Option<&Table>,
) -> std::result::Result<Option<TaskExecutionOverrides>, String> {
    let Some(opts) = opts else {
        return Ok(None);
    };
    let Ok(value) = opts.get::<Value>("execution") else {
        return Ok(None);
    };
    if matches!(value, Value::Nil) {
        return Ok(None);
    }
    let overrides = lua
        .from_value::<TaskExecutionOverrides>(value)
        .map_err(|err| err.to_string())?;
    if overrides.is_empty() {
        return Err("execution overrides must not be an empty table".to_string());
    }
    Ok(Some(overrides))
}

fn opt_sidestep_mode(opts: Option<&Table>) -> std::result::Result<SidestepMode, String> {
    let Some(opts) = opts else {
        return Ok(SidestepMode::Ephemeral);
    };
    let Ok(raw) = opts.get::<String>("mode") else {
        return Ok(SidestepMode::Ephemeral);
    };
    raw.parse()
}

fn opt_sidestep_context_target(
    lua: &Lua,
    opts: Option<&Table>,
) -> std::result::Result<Option<ExecutionContextTarget>, String> {
    let Some(opts) = opts else {
        return Ok(None);
    };
    let Ok(value) = opts.get::<Value>("context_target") else {
        return Ok(None);
    };
    if matches!(value, Value::Nil) {
        return Ok(None);
    }
    lua.from_value::<ExecutionContextTarget>(value)
        .map(Some)
        .map_err(|err| err.to_string())
}

fn graph_ref_from_lua_table(table: Table) -> std::result::Result<GraphRef, String> {
    let kind = table.get::<String>("kind").map_err(|err| err.to_string())?;
    let id = table.get::<String>("id").map_err(|err| err.to_string())?;
    if kind.is_empty() || id.is_empty() {
        return Err("graph ref requires non-empty kind and id".to_string());
    }
    Ok(GraphRef::new(kind, id))
}

fn opt_sidestep_graph_relation(
    lua: &Lua,
    opts: Option<&Table>,
) -> std::result::Result<Option<SidestepGraphRelation>, String> {
    let Some(opts) = opts else {
        return Ok(None);
    };
    let Ok(graph) = opts.get::<Table>("graph") else {
        return Ok(None);
    };
    let source = graph
        .get::<Table>("source")
        .map_err(|err| err.to_string())
        .and_then(graph_ref_from_lua_table)?;
    let relation_kind = graph
        .get::<String>("relation_kind")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "contains".to_string());
    let metadata = match graph.get::<Value>("metadata") {
        Ok(Value::Nil) | Err(_) => None,
        Ok(value) => Some(
            lua.from_value::<serde_json::Value>(value)
                .map_err(|err| err.to_string())?,
        ),
    };
    Ok(Some(SidestepGraphRelation {
        source,
        relation_kind,
        source_role: graph
            .get::<String>("source_role")
            .ok()
            .filter(|value| !value.is_empty()),
        target_role: graph
            .get::<String>("target_role")
            .ok()
            .filter(|value| !value.is_empty()),
        origin_task_id: graph
            .get::<String>("origin_task_id")
            .ok()
            .filter(|value| !value.is_empty()),
        origin_execution_id: graph
            .get::<String>("origin_execution_id")
            .ok()
            .filter(|value| !value.is_empty()),
        metadata,
    }))
}

async fn attach_sidestep_graph_relation(
    store_manager: &std::sync::Arc<crate::persistence::manager::StoreManager>,
    session_id: &str,
    branch_outcome: &TaskBranchOutcome,
    relation: SidestepGraphRelation,
) -> std::result::Result<(), String> {
    let TaskBranchOutcome::SidestepSibling {
        branch_public_id, ..
    } = branch_outcome
    else {
        return Err("sidestep graph relation requires a sidestep sibling branch".to_string());
    };
    let session_ref = parse_session_reference(session_id).map_err(|err| err.to_string())?;
    let public_id = uuid::Uuid::parse_str(&session_ref.public_id).map_err(|err| err.to_string())?;
    let store_selector = session_ref
        .store_selector
        .unwrap_or_else(|| StoreSelector::Alias("state".to_string()));
    let store = store_manager
        .open(&store_selector)
        .await
        .map_err(|err| err.to_string())?;
    let row = store
        .get_session_row_by_public_id(public_id)
        .await
        .map_err(|err| err.to_string())?
        .ok_or_else(|| format!("Session '{}' not found", session_ref.public_id))?;
    let branch_ref_id = uuid::Uuid::parse_str(branch_public_id)
        .map(|uuid| uuid.simple().to_string())
        .unwrap_or_else(|_| branch_public_id.clone());
    let mut edge = GraphEdgeCreate::new(
        relation.source,
        GraphRef::new("branch_head", branch_ref_id),
        relation.relation_kind,
    );
    edge.session_id = Some(row.id);
    edge.source_role = relation.source_role;
    edge.target_role = relation.target_role;
    edge.provenance = GraphProvenance::new(relation.origin_task_id, relation.origin_execution_id);
    edge.metadata = relation.metadata;
    store
        .create_graph_edge(edge)
        .await
        .map(|_| ())
        .map_err(|err| err.to_string())
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
    match row.created_from_turn_id {
        Some(turn_id) => table.set("source_turn_id", turn_id)?,
        None => table.set("source_turn_id", Value::Nil)?,
    }
    table.set("origin_kind", row.origin_kind.clone())?;
    match row.origin_task_id.as_deref() {
        Some(task_id) => table.set("origin_task_id", task_id)?,
        None => table.set("origin_task_id", Value::Nil)?,
    }
    match row.origin_execution_id.as_deref() {
        Some(execution_id) => table.set("origin_execution_id", execution_id)?,
        None => table.set("origin_execution_id", Value::Nil)?,
    }
    let metadata = row
        .origin_metadata
        .as_deref()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok());
    match metadata {
        Some(metadata) => table.set("origin_metadata", lua.to_value(&metadata)?)?,
        None => table.set("origin_metadata", Value::Nil)?,
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
            match enqueue_res {
                Ok(task_id) => string_ok(lua, &task_id),
                Err(err) => nil_err(lua, &err),
            }
        })?,
    )?;

    // agent.sidestep(prompt, opts?)
    let sidestep_q = app_data.execution_ctx.clone();
    let sidestep_policy_snapshot = app_data.clone();
    let sidestep_store_manager = app_data.store_manager.clone();
    agent_table.set(
        "sidestep",
        lua.create_function(move |lua, (prompt, opts): (String, Option<Table>)| {
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
            match enqueue_res {
                Ok(task_id) => string_ok(lua, &task_id),
                Err(err) => nil_err(lua, &err),
            }
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
            match result {
                Ok(branch) => Ok(ok_value(lua.to_value(&branch)?)),
                Err(err) => nil_err(lua, &err),
            }
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
                None => nil_err(lua, &format!("Task '{}' not found", task_id)),
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
                let execution = match opt_execution_overrides(lua, opts.as_ref()) {
                    Ok(execution) => execution,
                    Err(err) => return nil_err(lua, &err),
                };

                let manager_submit = manager.clone();
                let request_id = bridge_async_display_err(async move {
                    manager_submit
                        .submit(
                            &target_agent,
                            QueuedTask::ad_hoc(prompt)
                                .with_inherited_trace(trace_id.as_deref())
                                .with_execution(execution),
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
                Ok(_) => Ok(ok_bool()),
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
                Ok(_) => Ok(ok_bool()),
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
                Ok(_) => Ok(ok_bool()),
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
                            .reload_session_if_live(&session_ref_str, requested_slot.as_deref())
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
                        .list_branch_heads_from_source_turn(row.id, source_turn_id)
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
