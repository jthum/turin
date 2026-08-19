use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};
use serde_json::{Map as JsonMap, Value as JsonValue};
use turin_types::{TaskInputContent, ToolsConfig};

use crate::harness::globals::HarnessAppData;
use crate::harness::stdlib::action_bindings;
use crate::harness::stdlib::agent_bindings::{active_trace_id, queue_max, queue_push_one};
use crate::harness::stdlib::binding_common::{
    bridge_async_anyhow, bridge_async_lua, optional_lua_json, resolve_scoped_store_selector,
};
use crate::harness::stdlib::db_support::{
    selector_denied_by_dynamic_open, store_path_scope_from_snapshot, store_selector_from_fields,
};
use crate::harness::stdlib::governance_support::{current_agent_id, require_capability};
use crate::harness::stdlib::object_refs;
use crate::harness::stdlib::policy_support::runtime_policy_snapshot;
use crate::harness::stdlib::runtime_worklist::params::{
    ScopeValue, parse_bool_flag, parse_i64_opt, parse_json_opt, parse_payload,
    parse_present_json_raw, parse_present_string_array_raw, parse_present_string_opt,
    parse_scope_value, parse_stale_after_ms, parse_string_opt, parse_where_map,
    parse_work_item_query, serialize_json_opt,
};
use crate::harness::stdlib::runtime_worklist_selection as selection;
use crate::persistence::manager::{StoreManager, StoreSelector};
use crate::persistence::schema::{WorkItemRow, WorklistRow};
use crate::persistence::state::{StateStore, WorkItemInsert, WorkItemUpdate};
pub(crate) use crate::work_items::public_id_string;
use crate::work_items::{
    work_item_pause_reason as row_pause_reason,
    work_item_pause_until_unix_ms as row_pause_until_unix_ms, work_item_paused as row_paused,
    work_item_prompt_task,
};

mod item_proxy;
mod list_proxy;
mod params;

use item_proxy::item_proxy;
use list_proxy::worklist_proxy;

#[derive(Clone)]
struct WorklistHandle {
    app_data: HarnessAppData,
    store: Arc<StateStore>,
    store_selector: StoreSelector,
    worklist: WorklistRow,
    parent_item_id: Option<i64>,
}

fn now_unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis() as i64
}

fn runtime_store_settings(
    app_data: &HarnessAppData,
) -> LuaResult<(crate::persistence::manager::StorePathScope, usize, u64)> {
    let snapshot = runtime_policy_snapshot(app_data).map_err(mlua::Error::runtime)?;
    let path_scope = store_path_scope_from_snapshot(&snapshot);
    let max_open_handles = snapshot
        .get("db.max_open_handles")
        .and_then(|value| value.as_u64())
        .unwrap_or(128)
        .clamp(1, u64::MAX) as usize;
    let idle_close_seconds = snapshot
        .get("db.idle_close_seconds")
        .and_then(|value| value.as_u64())
        .unwrap_or(300);
    Ok((path_scope, max_open_handles, idle_close_seconds))
}

async fn open_store(
    manager: Arc<StoreManager>,
    selector: StoreSelector,
    path_scope: crate::persistence::manager::StorePathScope,
    max_open_handles: usize,
    idle_close_seconds: u64,
) -> anyhow::Result<Arc<StateStore>> {
    let _ = manager
        .trim_cache(max_open_handles, idle_close_seconds)
        .await;
    manager.open_with_path_scope(&selector, path_scope).await
}

fn resolve_store_selector(
    app_data: &HarnessAppData,
    opts: &Table,
    scope: Option<&ScopeValue>,
) -> LuaResult<StoreSelector> {
    let snapshot = runtime_policy_snapshot(app_data).map_err(mlua::Error::runtime)?;
    let explicit = store_selector_from_fields(opts)?;
    let selector = match (explicit, scope) {
        (Some(explicit), _) => explicit,
        (None, Some(ScopeValue::Selector(selector))) => {
            resolve_scoped_store_selector(app_data, selector, None)?
                .unwrap_or(StoreSelector::Alias("state".to_string()))
        }
        (None, _) => app_data
            .execution_ctx
            .lock()
            .ok()
            .and_then(|lock| lock.default_store_selector.clone())
            .unwrap_or(StoreSelector::Alias("state".to_string())),
    };
    if selector_denied_by_dynamic_open(&snapshot, &selector) {
        return Err(mlua::Error::runtime(
            "Policy denial: db.allow_dynamic_open=false",
        ));
    }
    Ok(selector)
}

fn scope_ref(scope: Option<&ScopeValue>) -> String {
    match scope {
        Some(ScopeValue::Ref(value)) => value.clone(),
        Some(ScopeValue::Selector(selector)) => selector.to_alias(),
        None => String::new(),
    }
}

fn row_json_value<T>(lua: &Lua, raw: Option<&str>) -> LuaResult<Value>
where
    T: serde::de::DeserializeOwned + serde::Serialize,
{
    match parse_json_opt::<T>(raw).ok().flatten() {
        Some(value) => lua.to_value(&value),
        None => Ok(Value::Nil),
    }
}

fn row_json_payload_value(lua: &Lua, raw: Option<&str>) -> LuaResult<Value> {
    match parse_json_opt::<JsonValue>(raw).ok().flatten() {
        Some(value) => object_refs::decode_json_payload(lua, &value),
        None => Ok(Value::Nil),
    }
}

fn item_payload_value(lua: &Lua, row: &WorkItemRow) -> LuaResult<Value> {
    let payload = lua.create_table()?;
    payload.set("kind", row.item_kind.clone())?;
    match row.item_kind.as_str() {
        "action" => {
            let action = lua.create_table()?;
            action.set("name", row.action_name.clone())?;
            action.set(
                "params",
                row_json_value::<JsonValue>(lua, row.action_params.as_deref())?,
            )?;
            payload.set("action", action)?;
        }
        _ => {
            payload.set("prompt", row.prompt.clone())?;
            payload.set(
                "content",
                row_json_value::<Vec<TaskInputContent>>(lua, row.content.as_deref())?,
            )?;
            payload.set(
                "tools",
                row_json_value::<ToolsConfig>(lua, row.tools.as_deref())?,
            )?;
            payload.set("conflict_policy", row.conflict_policy.clone())?;
        }
    }
    Ok(Value::Table(payload))
}

fn dispatch_prompt_item(
    row: &WorkItemRow,
    app_data: &HarnessAppData,
) -> anyhow::Result<serde_json::Value> {
    let trace_id = active_trace_id(app_data);
    let task = work_item_prompt_task(row, trace_id.as_deref())?;

    let snapshot = runtime_policy_snapshot(app_data).map_err(anyhow::Error::msg)?;
    let task_id = bridge_async_lua(async {
        queue_push_one(&app_data.execution_ctx, task, queue_max(&snapshot), false).await
    })?;

    Ok(serde_json::json!({
        "dispatched": "task",
        "task_id": task_id,
    }))
}

fn dispatch_action_item(
    lua: &Lua,
    row: &WorkItemRow,
    handle: &WorklistHandle,
) -> anyhow::Result<serde_json::Value> {
    let action_name = row
        .action_name
        .as_deref()
        .ok_or_else(|| anyhow::anyhow!("action work item '{}' is missing action", row.title))?;
    let params =
        parse_json_opt::<JsonValue>(row.action_params.as_deref())?.unwrap_or(JsonValue::Null);
    let result = action_bindings::invoke_declared_action(
        lua,
        action_name,
        params.clone(),
        action_bindings::ActionInvocationContext {
            app_data: handle.app_data.clone(),
            action_name: action_name.to_string(),
            params,
            work_item: Some(action_bindings::ActionWorkItemContext {
                store: handle.store.clone(),
                store_selector: handle.store_selector.clone(),
                worklist: handle.worklist.clone(),
                row: row.clone(),
            }),
        },
    )?
    .ok_or_else(|| anyhow::anyhow!("declared action '{}' is not defined", action_name))?;
    Ok(serde_json::json!({
        "dispatched": "action",
        "action": action_name,
        "result": result,
    }))
}

fn dispatch_item_result(lua: &Lua, row: &WorkItemRow, handle: &WorklistHandle) -> LuaResult<Value> {
    let result = match row.item_kind.as_str() {
        "action" => dispatch_action_item(lua, row, handle),
        _ => dispatch_prompt_item(row, &handle.app_data),
    }
    .map_err(mlua::Error::runtime)?;
    lua.to_value(&result)
}

fn current_claim_identity(app_data: &HarnessAppData) -> (String, Option<String>, Option<String>) {
    let agent_id = current_agent_id(app_data);
    let lock = app_data.execution_ctx.lock().ok();
    let session_id = lock.as_ref().and_then(|lock| lock.session_id.clone());
    let execution_id = lock
        .as_ref()
        .and_then(|lock| lock.execution_id.clone())
        .or_else(|| session_id.clone())
        .or_else(|| Some(format!("agent:{}", agent_id)));
    (agent_id, session_id, execution_id)
}

fn load_work_items(handle: &WorklistHandle) -> LuaResult<Vec<WorkItemRow>> {
    bridge_async_lua(async { handle.store.list_work_items(handle.worklist.id).await })
}

fn work_items_table(
    lua: &Lua,
    handle: &WorklistHandle,
    rows: Vec<WorkItemRow>,
) -> LuaResult<Table> {
    let out = lua.create_table()?;
    for (index, row) in rows.into_iter().enumerate() {
        out.set(index + 1, item_proxy(lua, handle.clone(), row)?)?;
    }
    Ok(out)
}

fn work_items_value(
    lua: &Lua,
    handle: &WorklistHandle,
    rows: Vec<WorkItemRow>,
) -> LuaResult<Value> {
    Ok(Value::Table(work_items_table(lua, handle, rows)?))
}

fn queried_work_items<F>(
    lua: &Lua,
    handle: &WorklistHandle,
    opts: Option<&Table>,
    select: F,
) -> LuaResult<Vec<WorkItemRow>>
where
    F: FnOnce(Vec<WorkItemRow>, selection::WorkItemSelection<'_>) -> Vec<WorkItemRow>,
{
    let query = parse_work_item_query(lua, opts)?;
    Ok(select(
        load_work_items(handle)?,
        query.selection(handle.parent_item_id),
    ))
}

fn item_value(lua: &Lua, handle: &WorklistHandle, row: WorkItemRow) -> LuaResult<Value> {
    Ok(Value::Table(item_proxy(lua, handle.clone(), row)?))
}

fn load_required_item(
    handle: &WorklistHandle,
    item_id: i64,
    missing_message: &'static str,
) -> LuaResult<WorkItemRow> {
    bridge_async_lua(async { handle.store.get_work_item_by_id(item_id).await })?
        .ok_or_else(|| mlua::Error::runtime(missing_message.to_string()))
}

pub(crate) fn build_work_item_proxy(
    lua: &Lua,
    store: Arc<StateStore>,
    store_selector: StoreSelector,
    worklist: WorklistRow,
    row: WorkItemRow,
    app_data: HarnessAppData,
) -> LuaResult<Table> {
    item_proxy(
        lua,
        WorklistHandle {
            app_data,
            store,
            store_selector,
            worklist,
            parent_item_id: Some(row.id),
        },
        row,
    )
}

fn proxy_method_call(
    _lua: &Lua,
    table: &Table,
    method: &str,
    arg: Option<Table>,
) -> LuaResult<Value> {
    let method_fn = table.get::<mlua::Function>(method)?;
    match arg {
        Some(arg) => method_fn.call((table.clone(), arg)),
        None => method_fn.call(table.clone()),
    }
}

pub fn register_runtime_worklist_namespace(
    lua: &Lua,
    runtime_table: &Table,
    app_data: &HarnessAppData,
) -> LuaResult<()> {
    let runtime_worklist = lua.create_table()?;

    {
        let app_data = app_data.clone();
        runtime_worklist.set(
            "open",
            lua.create_function(move |lua, opts: Table| {
                require_capability(&app_data, "runtime.worklist.open")
                    .map_err(mlua::Error::runtime)?;
                let name = parse_string_opt(&opts, "name")?.ok_or_else(|| {
                    mlua::Error::runtime("runtime.worklist.open requires opts.name".to_string())
                })?;
                let scope = parse_scope_value(opts.get::<Value>("scope")?)?;
                let selector = resolve_store_selector(&app_data, &opts, scope.as_ref())?;
                let (path_scope, max_open_handles, idle_close_seconds) =
                    runtime_store_settings(&app_data)?;
                let manager = app_data.store_manager.clone();
                let store_selector = selector.clone();
                let store = bridge_async_lua(async move {
                    open_store(
                        manager,
                        selector,
                        path_scope,
                        max_open_handles,
                        idle_close_seconds,
                    )
                    .await
                })?;
                let metadata_json = optional_lua_json(lua, opts.get::<Value>("metadata")?)?;
                let metadata_raw =
                    serialize_json_opt(metadata_json.as_ref()).map_err(mlua::Error::runtime)?;
                let worklist = bridge_async_lua(async {
                    store
                        .open_worklist(&name, &scope_ref(scope.as_ref()), metadata_raw.as_deref())
                        .await
                })?;
                Ok(Value::Table(worklist_proxy(
                    lua,
                    WorklistHandle {
                        app_data: app_data.clone(),
                        store,
                        store_selector,
                        worklist,
                        parent_item_id: None,
                    },
                )?))
            })?,
        )?;
    }

    runtime_table.set("worklist", runtime_worklist)?;
    Ok(())
}

fn open_ref_store(
    app_data: &HarnessAppData,
    selector: StoreSelector,
) -> anyhow::Result<Arc<StateStore>> {
    let (path_scope, max_open_handles, idle_close_seconds) =
        runtime_store_settings(app_data).map_err(anyhow::Error::msg)?;
    bridge_async_anyhow(async {
        open_store(
            app_data.store_manager.clone(),
            selector,
            path_scope,
            max_open_handles,
            idle_close_seconds,
        )
        .await
    })
}

pub(crate) fn hydrate_worklist_ref(
    lua: &Lua,
    app_data: &HarnessAppData,
    ref_obj: &JsonMap<String, JsonValue>,
) -> anyhow::Result<Table> {
    let selector = object_refs::store_selector_from_json(ref_obj.get("store"));
    let store = open_ref_store(app_data, selector.clone())?;

    let row = if let Some(id) = ref_obj.get("id").and_then(|value| value.as_str()) {
        let uuid = uuid::Uuid::parse_str(id)?;
        bridge_async_anyhow(async { store.get_worklist_by_public_id(uuid).await })?
            .ok_or_else(|| anyhow::anyhow!("worklist ref '{}' not found", id))?
    } else {
        let name = ref_obj
            .get("name")
            .and_then(|value| value.as_str())
            .ok_or_else(|| anyhow::anyhow!("worklist ref requires id or name"))?;
        let scope_ref = ref_obj
            .get("scope_ref")
            .and_then(|value| value.as_str())
            .unwrap_or("");
        bridge_async_anyhow(async { store.open_worklist(name, scope_ref, None).await })?
    };

    worklist_proxy(
        lua,
        WorklistHandle {
            app_data: app_data.clone(),
            store,
            store_selector: selector,
            worklist: row,
            parent_item_id: None,
        },
    )
    .map_err(anyhow::Error::from)
}

pub(crate) fn hydrate_workitem_ref(
    lua: &Lua,
    app_data: &HarnessAppData,
    ref_obj: &JsonMap<String, JsonValue>,
) -> anyhow::Result<Table> {
    let selector = object_refs::store_selector_from_json(ref_obj.get("store"));
    let store = open_ref_store(app_data, selector.clone())?;
    let id = ref_obj
        .get("id")
        .and_then(|value| value.as_str())
        .ok_or_else(|| anyhow::anyhow!("workitem ref requires id"))?;
    let uuid = uuid::Uuid::parse_str(id)?;
    let row = bridge_async_anyhow(async { store.get_work_item_by_public_id(uuid).await })?
        .ok_or_else(|| anyhow::anyhow!("workitem ref '{}' not found", id))?;
    let worklist = bridge_async_anyhow(async { store.get_worklist_by_id(row.worklist_id).await })?
        .ok_or_else(|| anyhow::anyhow!("worklist for workitem ref '{}' not found", id))?;
    item_proxy(
        lua,
        WorklistHandle {
            app_data: app_data.clone(),
            store,
            store_selector: selector,
            worklist,
            parent_item_id: row.parent_item_id,
        },
        row,
    )
    .map_err(anyhow::Error::from)
}
