use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};
use serde_json::{Map as JsonMap, Value as JsonValue};
use turin_types::{TaskInputContent, ToolsConfig};

use crate::harness::globals::HarnessAppData;
use crate::harness::stdlib::action_bindings;
use crate::harness::stdlib::agent_bindings::{active_trace_id, queue_max, queue_push_one};
use crate::harness::stdlib::binding_common::resolve_scoped_store_selector;
use crate::harness::stdlib::context_selectors::table_to_selector;
use crate::harness::stdlib::db_support::{
    selector_denied_by_dynamic_open, store_path_scope_from_snapshot, store_selector_from_fields,
};
use crate::harness::stdlib::governance_support::{current_agent_id, require_capability};
use crate::harness::stdlib::policy_support::runtime_policy_snapshot;
use crate::kernel::identity::ContextSelector;
use crate::kernel::session::{ExecutionConflictPolicy, QueuedTask};
use crate::persistence::manager::{StoreManager, StoreSelector};
use crate::persistence::schema::{WorkItemRow, WorklistRow};
use crate::persistence::state::{StateStore, WorkItemInsert, WorkItemUpdate};

#[derive(Clone)]
struct WorklistHandle {
    app_data: HarnessAppData,
    store: Arc<StateStore>,
    store_selector: StoreSelector,
    worklist: WorklistRow,
    parent_item_id: Option<i64>,
}

enum ScopeValue {
    Ref(String),
    Selector(ContextSelector),
}

struct ParsedPayload {
    title: String,
    item_kind: String,
    prompt: Option<String>,
    content: Option<Vec<TaskInputContent>>,
    tools: Option<ToolsConfig>,
    conflict_policy: Option<String>,
    action_name: Option<String>,
    action_params: Option<JsonValue>,
    priority: i64,
    after_ids: Option<Vec<String>>,
    metadata: Option<JsonValue>,
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

fn parse_scope_value(value: Value) -> LuaResult<Option<ScopeValue>> {
    match value {
        Value::Nil => Ok(None),
        Value::String(s) => Ok(Some(ScopeValue::Ref(s.to_str()?.to_string()))),
        Value::Table(table) => Ok(Some(ScopeValue::Selector(table_to_selector(table)?))),
        other => Err(mlua::Error::runtime(format!(
            "runtime.worklist scope must be string or selector table, got {:?}",
            other
        ))),
    }
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

fn parse_string_opt(table: &Table, key: &str) -> LuaResult<Option<String>> {
    match table.get::<Value>(key)? {
        Value::Nil => Ok(None),
        Value::String(s) => Ok(Some(s.to_str()?.to_string())),
        other => Err(mlua::Error::runtime(format!(
            "runtime.worklist field '{}' must be a string, got {:?}",
            key, other
        ))),
    }
}

fn parse_i64_opt(table: &Table, key: &str) -> LuaResult<Option<i64>> {
    match table.get::<Value>(key)? {
        Value::Nil => Ok(None),
        Value::Integer(i) => Ok(Some(i)),
        Value::Number(n) if n.is_finite() && n.fract() == 0.0 => Ok(Some(n as i64)),
        other => Err(mlua::Error::runtime(format!(
            "runtime.worklist field '{}' must be an integer, got {:?}",
            key, other
        ))),
    }
}

fn parse_json_array_strings(value: Value, field: &str) -> LuaResult<Option<Vec<String>>> {
    match value {
        Value::Nil => Ok(None),
        Value::Table(table) => {
            let mut out = Vec::new();
            for value in table.sequence_values::<String>() {
                out.push(value?);
            }
            Ok(Some(out))
        }
        other => Err(mlua::Error::runtime(format!(
            "runtime.worklist '{}' must be an array of strings, got {:?}",
            field, other
        ))),
    }
}

fn parse_payload(lua: &Lua, payload: Value, opts: Option<Table>) -> LuaResult<ParsedPayload> {
    let (title, item_kind, prompt);
    let mut content = None;
    let mut tools = None;
    let mut conflict_policy = None;
    let mut action_name = None;
    let mut action_params = None;
    let mut priority = None;
    let mut after_ids = None;
    let mut metadata = None;

    match payload {
        Value::String(s) => {
            let text = s.to_str()?.to_string();
            title = Some(text.clone());
            item_kind = Some("prompt".to_string());
            prompt = Some(text);
        }
        Value::Table(table) => {
            title = parse_string_opt(&table, "title")?;
            item_kind = parse_string_opt(&table, "kind")?;
            prompt = parse_string_opt(&table, "prompt")?;
            conflict_policy = parse_string_opt(&table, "conflict_policy")?;
            priority = parse_i64_opt(&table, "priority")?;
            action_name = parse_string_opt(&table, "action")?;
            content = match table.get::<Value>("content")? {
                Value::Nil => None,
                value => Some(serde_json::from_value(lua.from_value(value)?).map_err(|e| {
                    mlua::Error::runtime(format!("invalid worklist content: {}", e))
                })?),
            };
            tools =
                match table.get::<Value>("tools")? {
                    Value::Nil => None,
                    value => Some(serde_json::from_value(lua.from_value(value)?).map_err(|e| {
                        mlua::Error::runtime(format!("invalid worklist tools: {}", e))
                    })?),
                };
            action_params = match table.get::<Value>("params")? {
                Value::Nil => None,
                value => Some(lua.from_value(value)?),
            };
            after_ids = parse_json_array_strings(table.get::<Value>("after")?, "after")?;
            metadata = match table.get::<Value>("metadata")? {
                Value::Nil => None,
                value => Some(lua.from_value(value)?),
            };
        }
        other => {
            return Err(mlua::Error::runtime(format!(
                "worklist payload must be string or table, got {:?}",
                other
            )));
        }
    }

    if let Some(opts) = opts {
        if priority.is_none() {
            priority = parse_i64_opt(&opts, "priority")?;
        }
        if after_ids.is_none() {
            after_ids = parse_json_array_strings(opts.get::<Value>("after")?, "after")?;
        }
        if metadata.is_none() {
            metadata = match opts.get::<Value>("metadata")? {
                Value::Nil => None,
                value => Some(lua.from_value(value)?),
            };
        }
    }

    let item_kind = match (item_kind, action_name.as_ref(), prompt.as_ref()) {
        (Some(kind), _, _) => kind,
        (None, Some(_), None) => "action".to_string(),
        (None, None, Some(_)) => "prompt".to_string(),
        (None, Some(_), Some(_)) => {
            return Err(mlua::Error::runtime(
                "worklist payload cannot define both prompt and action".to_string(),
            ));
        }
        (None, None, None) => {
            return Err(mlua::Error::runtime(
                "worklist payload requires prompt or action".to_string(),
            ));
        }
    };

    if item_kind == "prompt" && action_name.is_some() {
        return Err(mlua::Error::runtime(
            "prompt worklist payload cannot also define action".to_string(),
        ));
    }
    if item_kind == "action" && prompt.is_some() {
        return Err(mlua::Error::runtime(
            "action worklist payload cannot also define prompt".to_string(),
        ));
    }

    let title = title
        .or_else(|| prompt.clone())
        .or_else(|| action_name.clone())
        .ok_or_else(|| {
            mlua::Error::runtime("worklist payload requires title or prompt/action".to_string())
        })?;

    Ok(ParsedPayload {
        title,
        item_kind,
        prompt,
        content,
        tools,
        conflict_policy,
        action_name,
        action_params,
        priority: priority.unwrap_or(0),
        after_ids,
        metadata,
    })
}

fn parse_where_map(
    lua: &Lua,
    opts: Option<Table>,
) -> LuaResult<Option<JsonMap<String, JsonValue>>> {
    let Some(opts) = opts else {
        return Ok(None);
    };
    match opts.get::<Value>("where")? {
        Value::Nil => Ok(None),
        Value::Table(table) => match lua.from_value::<JsonValue>(Value::Table(table))? {
            JsonValue::Object(map) => Ok(Some(map)),
            _ => Err(mlua::Error::runtime(
                "worklist where filter must be an object-like table".to_string(),
            )),
        },
        other => Err(mlua::Error::runtime(format!(
            "worklist where filter must be a table, got {:?}",
            other
        ))),
    }
}

fn parse_limit(opts: Option<Table>) -> LuaResult<Option<usize>> {
    let Some(opts) = opts else {
        return Ok(None);
    };
    match opts.get::<Value>("limit")? {
        Value::Nil => Ok(None),
        Value::Integer(i) if i >= 0 => Ok(Some(i as usize)),
        Value::Number(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 => Ok(Some(n as usize)),
        other => Err(mlua::Error::runtime(format!(
            "worklist limit must be a non-negative integer, got {:?}",
            other
        ))),
    }
}

fn parse_stale_after_ms(opts: Option<Table>) -> LuaResult<i64> {
    let Some(opts) = opts else {
        return Ok(300_000);
    };
    match opts.get::<Value>("stale_after_seconds")? {
        Value::Nil => Ok(300_000),
        Value::Integer(i) if i >= 0 => Ok(i.saturating_mul(1000)),
        Value::Number(n) if n.is_finite() && n >= 0.0 => {
            Ok((n * 1000.0).round().clamp(0.0, i64::MAX as f64) as i64)
        }
        other => Err(mlua::Error::runtime(format!(
            "worklist stale_after_seconds must be a non-negative number, got {:?}",
            other
        ))),
    }
}

fn parse_bool_flag(opts: Option<Table>, key: &str) -> LuaResult<bool> {
    let Some(opts) = opts else {
        return Ok(false);
    };
    match opts.get::<Value>(key)? {
        Value::Nil => Ok(false),
        Value::Boolean(value) => Ok(value),
        other => Err(mlua::Error::runtime(format!(
            "worklist {} must be a boolean, got {:?}",
            key, other
        ))),
    }
}

pub(crate) fn public_id_string(bytes: &[u8]) -> String {
    uuid::Uuid::from_slice(bytes)
        .map(|uuid| uuid.to_string())
        .unwrap_or_else(|_| {
            let mut out = String::with_capacity(bytes.len() * 2);
            for byte in bytes {
                use std::fmt::Write as _;
                let _ = write!(&mut out, "{:02x}", byte);
            }
            out
        })
}

fn parse_json_opt<T>(raw: Option<&str>) -> anyhow::Result<Option<T>>
where
    T: serde::de::DeserializeOwned,
{
    raw.map(serde_json::from_str)
        .transpose()
        .map_err(anyhow::Error::from)
}

pub(crate) fn serialize_json_opt<T>(value: Option<&T>) -> anyhow::Result<Option<String>>
where
    T: serde::Serialize,
{
    value
        .map(serde_json::to_string)
        .transpose()
        .map_err(anyhow::Error::from)
}

fn row_filter_value(row: &WorkItemRow, metadata: &JsonValue, key: &str) -> Option<JsonValue> {
    match key {
        "id" | "public_id" => Some(JsonValue::String(public_id_string(&row.public_id))),
        "title" => Some(JsonValue::String(row.title.clone())),
        "kind" => Some(JsonValue::String(row.item_kind.clone())),
        "status" => Some(JsonValue::String(row.status.clone())),
        "priority" => Some(JsonValue::Number(row.priority.into())),
        "parent_id" => Some(
            row.parent_item_id
                .map(|v| JsonValue::Number(v.into()))
                .unwrap_or(JsonValue::Null),
        ),
        "paused" => Some(JsonValue::Bool(row_pause_flag(Some(metadata)))),
        "pause_reason" => row_pause_reason(Some(metadata)).map(JsonValue::String),
        "pause_until_unix_ms" => {
            row_pause_until_unix_ms(Some(metadata)).map(|value| JsonValue::Number(value.into()))
        }
        _ => metadata.get(key).cloned(),
    }
}

fn row_matches_where(row: &WorkItemRow, where_map: Option<&JsonMap<String, JsonValue>>) -> bool {
    let Some(where_map) = where_map else {
        return true;
    };
    let metadata = row
        .metadata
        .as_deref()
        .and_then(|raw| serde_json::from_str::<JsonValue>(raw).ok())
        .unwrap_or(JsonValue::Null);
    where_map
        .iter()
        .all(|(key, expected)| row_filter_value(row, &metadata, key).as_ref() == Some(expected))
}

fn dependencies_satisfied(row: &WorkItemRow, status_map: &HashMap<String, String>) -> bool {
    let deps = row
        .after_ids
        .as_deref()
        .and_then(|raw| serde_json::from_str::<Vec<String>>(raw).ok())
        .unwrap_or_default();
    deps.into_iter()
        .all(|dep| status_map.get(&dep).is_some_and(|status| status == "done"))
}

fn row_is_orphaned(row: &WorkItemRow, stale_before_unix_ms: i64) -> bool {
    row.status == "active"
        && match row.claim_heartbeat_unix_ms {
            Some(heartbeat) => heartbeat <= stale_before_unix_ms,
            None => true,
        }
}

fn row_is_paused(row: &WorkItemRow, now_unix_ms: i64) -> bool {
    let metadata = row_metadata(row);
    if !row_pause_flag(metadata.as_ref()) {
        return false;
    }
    match row_pause_until_unix_ms(metadata.as_ref()) {
        Some(pause_until_unix_ms) => pause_until_unix_ms > now_unix_ms,
        None => true,
    }
}

fn row_metadata(row: &WorkItemRow) -> Option<JsonValue> {
    row.metadata
        .as_deref()
        .and_then(|raw| serde_json::from_str::<JsonValue>(raw).ok())
}

fn row_pause_flag(metadata: Option<&JsonValue>) -> bool {
    let Some(JsonValue::Object(map)) = metadata else {
        return false;
    };
    map.get("paused")
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
}

fn row_pause_reason(metadata: Option<&JsonValue>) -> Option<String> {
    let Some(JsonValue::Object(map)) = metadata else {
        return None;
    };
    map.get("pause_reason")
        .and_then(|value| value.as_str())
        .map(ToString::to_string)
}

fn row_pause_until_unix_ms(metadata: Option<&JsonValue>) -> Option<i64> {
    let Some(JsonValue::Object(map)) = metadata else {
        return None;
    };
    map.get("pause_until_unix_ms")
        .and_then(|value| value.as_i64())
}

fn row_pause_due(row: &WorkItemRow, now_unix_ms: i64) -> bool {
    let metadata = row_metadata(row);
    if !row_pause_flag(metadata.as_ref()) {
        return false;
    }
    match row_pause_until_unix_ms(metadata.as_ref()) {
        Some(pause_until_unix_ms) => pause_until_unix_ms <= now_unix_ms,
        None => false,
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
                match parse_json_opt::<JsonValue>(row.action_params.as_deref())
                    .ok()
                    .flatten()
                {
                    Some(value) => lua.to_value(&value)?,
                    None => Value::Nil,
                },
            )?;
            payload.set("action", action)?;
        }
        _ => {
            payload.set("prompt", row.prompt.clone())?;
            payload.set(
                "content",
                match parse_json_opt::<Vec<TaskInputContent>>(row.content.as_deref())
                    .ok()
                    .flatten()
                {
                    Some(value) => lua.to_value(&value)?,
                    None => Value::Nil,
                },
            )?;
            payload.set(
                "tools",
                match parse_json_opt::<ToolsConfig>(row.tools.as_deref())
                    .ok()
                    .flatten()
                {
                    Some(value) => lua.to_value(&value)?,
                    None => Value::Nil,
                },
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
    let prompt = row
        .prompt
        .clone()
        .ok_or_else(|| anyhow::anyhow!("prompt work item '{}' is missing prompt", row.title))?;
    let trace_id = active_trace_id(app_data);
    let mut task = QueuedTask::ad_hoc(prompt).with_inherited_trace(trace_id.as_deref());
    task.title = Some(row.title.clone());
    task.content = parse_json_opt::<Vec<TaskInputContent>>(row.content.as_deref())?;
    task.tools = parse_json_opt::<ToolsConfig>(row.tools.as_deref())?;
    task.conflict_policy = row
        .conflict_policy
        .as_deref()
        .map(str::parse::<ExecutionConflictPolicy>)
        .transpose()
        .map_err(anyhow::Error::msg)?;

    let snapshot = runtime_policy_snapshot(app_data).map_err(anyhow::Error::msg)?;
    let task_id = crate::harness::globals::block_on_current(async {
        queue_push_one(&app_data.execution_ctx, task, queue_max(&snapshot), false).await
    })
    .map_err(anyhow::Error::msg)?;

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

fn item_proxy(lua: &Lua, handle: WorklistHandle, row: WorkItemRow) -> LuaResult<Table> {
    let proxy = worklist_proxy(
        lua,
        WorklistHandle {
            parent_item_id: Some(row.id),
            ..handle.clone()
        },
    )?;
    let metadata_json = parse_json_opt::<JsonValue>(row.metadata.as_deref())
        .ok()
        .flatten();
    proxy.set("id", public_id_string(&row.public_id))?;
    proxy.set("internal_id", row.id)?;
    proxy.set("title", row.title.clone())?;
    proxy.set("kind", row.item_kind.clone())?;
    proxy.set("status", row.status.clone())?;
    proxy.set("priority", row.priority)?;
    proxy.set("parent_internal_id", row.parent_item_id)?;
    proxy.set("prompt", row.prompt.clone())?;
    proxy.set(
        "content",
        match parse_json_opt::<Vec<TaskInputContent>>(row.content.as_deref())
            .ok()
            .flatten()
        {
            Some(value) => lua.to_value(&value)?,
            None => Value::Nil,
        },
    )?;
    proxy.set(
        "tools",
        match parse_json_opt::<ToolsConfig>(row.tools.as_deref())
            .ok()
            .flatten()
        {
            Some(value) => lua.to_value(&value)?,
            None => Value::Nil,
        },
    )?;
    proxy.set("conflict_policy", row.conflict_policy.clone())?;
    proxy.set("action", row.action_name.clone())?;
    proxy.set(
        "params",
        match parse_json_opt::<JsonValue>(row.action_params.as_deref())
            .ok()
            .flatten()
        {
            Some(value) => lua.to_value(&value)?,
            None => Value::Nil,
        },
    )?;
    proxy.set(
        "metadata",
        match metadata_json.clone() {
            Some(value) => lua.to_value(&value)?,
            None => Value::Nil,
        },
    )?;
    proxy.set("paused", row_pause_flag(metadata_json.as_ref()))?;
    proxy.set("pause_reason", row_pause_reason(metadata_json.as_ref()))?;
    proxy.set(
        "pause_until_unix_ms",
        row_pause_until_unix_ms(metadata_json.as_ref()),
    )?;
    proxy.set(
        "after",
        match parse_json_opt::<Vec<String>>(row.after_ids.as_deref())
            .ok()
            .flatten()
        {
            Some(values) => lua.to_value(&values)?,
            None => Value::Nil,
        },
    )?;
    proxy.set("claim_agent_id", row.claim_agent_id.clone())?;
    proxy.set("claim_session_id", row.claim_session_id.clone())?;
    proxy.set("claim_execution_id", row.claim_execution_id.clone())?;
    proxy.set("failure_reason", row.failure_reason.clone())?;
    proxy.set("payload", item_payload_value(lua, &row)?)?;

    {
        let handle = handle.clone();
        let item_id = row.id;
        proxy.set(
            "claim",
            lua.create_function(move |lua, _self: Table| {
                require_capability(&handle.app_data, "runtime.worklist.claim")
                    .map_err(mlua::Error::runtime)?;
                let (agent_id, session_id, execution_id) = current_claim_identity(&handle.app_data);
                let claimed = crate::harness::globals::block_on_current(async {
                    handle
                        .store
                        .try_claim_work_item(
                            item_id,
                            &agent_id,
                            session_id.as_deref(),
                            execution_id.as_deref(),
                            now_unix_ms(),
                        )
                        .await
                })
                .map_err(mlua::Error::runtime)?;
                let row = crate::harness::globals::block_on_current(async {
                    handle.store.get_work_item_by_id(item_id).await
                })
                .map_err(mlua::Error::runtime)?
                .ok_or_else(|| mlua::Error::runtime("work item not found".to_string()))?;
                if !claimed && row.claim_execution_id.as_deref() != execution_id.as_deref() {
                    return Ok(Value::Nil);
                }
                Ok(Value::Table(item_proxy(lua, handle.clone(), row)?))
            })?,
        )?;
    }

    {
        let handle = handle.clone();
        let item_id = row.id;
        proxy.set(
            "heartbeat",
            lua.create_function(move |lua, _self: Table| {
                require_capability(&handle.app_data, "runtime.worklist.heartbeat")
                    .map_err(mlua::Error::runtime)?;
                let (_agent_id, _session_id, execution_id) =
                    current_claim_identity(&handle.app_data);
                let execution_id = execution_id.ok_or_else(|| {
                    mlua::Error::runtime(
                        "runtime.worklist.heartbeat requires an active execution identity"
                            .to_string(),
                    )
                })?;
                let row = crate::harness::globals::block_on_current(async {
                    handle
                        .store
                        .heartbeat_work_item_claim(item_id, &execution_id, now_unix_ms())
                        .await
                })
                .map_err(mlua::Error::runtime)?
                .ok_or_else(|| {
                    mlua::Error::runtime(
                        "work item is not actively claimed by this execution".to_string(),
                    )
                })?;
                Ok(Value::Table(item_proxy(lua, handle.clone(), row)?))
            })?,
        )?;
    }

    {
        let handle = handle.clone();
        let item_id = row.id;
        proxy.set(
            "dispatch",
            lua.create_function(move |lua, (_self, _opts): (Table, Option<Table>)| {
                require_capability(&handle.app_data, "runtime.worklist.dispatch")
                    .map_err(mlua::Error::runtime)?;
                let row = crate::harness::globals::block_on_current(async {
                    handle.store.get_work_item_by_id(item_id).await
                })
                .map_err(mlua::Error::runtime)?
                .ok_or_else(|| mlua::Error::runtime("work item not found".to_string()))?;
                dispatch_item_result(lua, &row, &handle)
            })?,
        )?;
    }

    {
        let handle = handle.clone();
        let item_id = row.id;
        proxy.set(
            "done",
            lua.create_function(move |lua, (_self, meta): (Table, Option<Value>)| {
                require_capability(&handle.app_data, "runtime.worklist.done")
                    .map_err(mlua::Error::runtime)?;
                let metadata_json = match meta {
                    Some(Value::Nil) | None => None,
                    Some(value) => Some(lua.from_value::<JsonValue>(value)?),
                };
                let metadata_raw =
                    serialize_json_opt(metadata_json.as_ref()).map_err(mlua::Error::runtime)?;
                let row = crate::harness::globals::block_on_current(async {
                    handle
                        .store
                        .complete_work_item(item_id, metadata_raw.as_deref())
                        .await
                })
                .map_err(mlua::Error::runtime)?;
                Ok(Value::Table(item_proxy(lua, handle.clone(), row)?))
            })?,
        )?;
    }

    {
        let handle = handle.clone();
        let item_id = row.id;
        proxy.set(
            "fail",
            lua.create_function(move |lua, (_self, reason): (Table, Option<String>)| {
                require_capability(&handle.app_data, "runtime.worklist.fail")
                    .map_err(mlua::Error::runtime)?;
                let row = crate::harness::globals::block_on_current(async {
                    handle
                        .store
                        .fail_work_item(item_id, reason.as_deref())
                        .await
                })
                .map_err(mlua::Error::runtime)?;
                Ok(Value::Table(item_proxy(lua, handle.clone(), row)?))
            })?,
        )?;
    }

    {
        let handle = handle.clone();
        let item_id = row.id;
        proxy.set(
            "requeue",
            lua.create_function(move |lua, _self: Table| {
                require_capability(&handle.app_data, "runtime.worklist.requeue")
                    .map_err(mlua::Error::runtime)?;
                let row = crate::harness::globals::block_on_current(async {
                    handle.store.release_work_item(item_id).await
                })
                .map_err(mlua::Error::runtime)?;
                Ok(Value::Table(item_proxy(lua, handle.clone(), row)?))
            })?,
        )?;
    }

    {
        let handle = handle.clone();
        let item_id = row.id;
        proxy.set(
            "update",
            lua.create_function(move |lua, (_self, fields): (Table, Table)| {
                require_capability(&handle.app_data, "runtime.worklist.update")
                    .map_err(mlua::Error::runtime)?;
                let title = parse_string_opt(&fields, "title")?;
                let prompt = if fields.contains_key("prompt")? {
                    Some(parse_string_opt(&fields, "prompt")?)
                } else {
                    None
                };
                let conflict_policy = if fields.contains_key("conflict_policy")? {
                    Some(parse_string_opt(&fields, "conflict_policy")?)
                } else {
                    None
                };
                let action_name = if fields.contains_key("action")? {
                    Some(parse_string_opt(&fields, "action")?)
                } else {
                    None
                };
                let action_params = if fields.contains_key("params")? {
                    Some(match fields.get::<Value>("params")? {
                        Value::Nil => None,
                        value => Some(lua.from_value::<JsonValue>(value)?),
                    })
                } else {
                    None
                };
                let content = if fields.contains_key("content")? {
                    Some(match fields.get::<Value>("content")? {
                        Value::Nil => None,
                        value => Some(
                            serde_json::to_string(&lua.from_value::<JsonValue>(value)?)
                                .map_err(mlua::Error::runtime)?,
                        ),
                    })
                } else {
                    None
                };
                let tools = if fields.contains_key("tools")? {
                    Some(match fields.get::<Value>("tools")? {
                        Value::Nil => None,
                        value => Some(
                            serde_json::to_string(&lua.from_value::<JsonValue>(value)?)
                                .map_err(mlua::Error::runtime)?,
                        ),
                    })
                } else {
                    None
                };
                let metadata = if fields.contains_key("metadata")? {
                    Some(match fields.get::<Value>("metadata")? {
                        Value::Nil => None,
                        value => Some(
                            serde_json::to_string(&lua.from_value::<JsonValue>(value)?)
                                .map_err(mlua::Error::runtime)?,
                        ),
                    })
                } else {
                    None
                };
                let after_ids = if fields.contains_key("after")? {
                    Some(
                        parse_json_array_strings(fields.get::<Value>("after")?, "after")?
                            .map(|values| {
                                serde_json::to_string(&values).map_err(mlua::Error::runtime)
                            })
                            .transpose()?,
                    )
                } else {
                    None
                };
                let status = if fields.contains_key("status")? {
                    Some(
                        parse_string_opt(&fields, "status")?
                            .unwrap_or_else(|| "pending".to_string()),
                    )
                } else {
                    None
                };
                let failure_reason = if fields.contains_key("failure_reason")? {
                    Some(parse_string_opt(&fields, "failure_reason")?)
                } else {
                    None
                };
                let priority = parse_i64_opt(&fields, "priority")?;
                let action_params_raw = action_params
                    .map(|value| serialize_json_opt(value.as_ref()).map_err(mlua::Error::runtime))
                    .transpose()?;
                let row = crate::harness::globals::block_on_current(async {
                    handle
                        .store
                        .update_work_item(WorkItemUpdate {
                            id: item_id,
                            title: title.as_deref(),
                            prompt: prompt.as_ref().map(|value| value.as_deref()),
                            content: content.as_ref().map(|value| value.as_deref()),
                            tools: tools.as_ref().map(|value| value.as_deref()),
                            conflict_policy: conflict_policy.as_ref().map(|value| value.as_deref()),
                            action_name: action_name.as_ref().map(|value| value.as_deref()),
                            action_params: action_params_raw.as_ref().map(|value| value.as_deref()),
                            priority,
                            after_ids: after_ids.as_ref().map(|value| value.as_deref()),
                            metadata: metadata.as_ref().map(|value| value.as_deref()),
                            status: status.as_deref(),
                            failure_reason: failure_reason.as_ref().map(|value| value.as_deref()),
                        })
                        .await
                })
                .map_err(mlua::Error::runtime)?;
                Ok(Value::Table(item_proxy(lua, handle.clone(), row)?))
            })?,
        )?;
    }

    {
        let handle = handle.clone();
        let parent_id = row.id;
        proxy.set(
            "children",
            lua.create_function(move |lua, _self: Table| {
                let rows = crate::harness::globals::block_on_current(async {
                    handle.store.list_work_items(handle.worklist.id).await
                })
                .map_err(mlua::Error::runtime)?;
                let out = lua.create_table()?;
                for (index, row) in rows
                    .into_iter()
                    .filter(|row| row.parent_item_id == Some(parent_id))
                    .enumerate()
                {
                    out.set(index + 1, item_proxy(lua, handle.clone(), row)?)?;
                }
                Ok(Value::Table(out))
            })?,
        )?;
    }

    Ok(proxy)
}

fn worklist_proxy(lua: &Lua, handle: WorklistHandle) -> LuaResult<Table> {
    let proxy = lua.create_table()?;
    proxy.set("name", handle.worklist.name.clone())?;
    proxy.set("scope_ref", handle.worklist.scope_ref.clone())?;
    proxy.set("id", public_id_string(&handle.worklist.public_id))?;
    proxy.set("internal_id", handle.worklist.id)?;

    {
        let handle = handle.clone();
        proxy.set(
            "add",
            lua.create_function(
                move |lua, (_self, payload, opts): (Table, Value, Option<Table>)| {
                    require_capability(&handle.app_data, "runtime.worklist.add")
                        .map_err(mlua::Error::runtime)?;
                    let parsed = parse_payload(lua, payload, opts)?;
                    let content = serialize_json_opt(parsed.content.as_ref())
                        .map_err(mlua::Error::runtime)?;
                    let tools =
                        serialize_json_opt(parsed.tools.as_ref()).map_err(mlua::Error::runtime)?;
                    let action_params = serialize_json_opt(parsed.action_params.as_ref())
                        .map_err(mlua::Error::runtime)?;
                    let after_ids = serialize_json_opt(parsed.after_ids.as_ref())
                        .map_err(mlua::Error::runtime)?;
                    let metadata = serialize_json_opt(parsed.metadata.as_ref())
                        .map_err(mlua::Error::runtime)?;
                    let row = crate::harness::globals::block_on_current(async {
                        handle
                            .store
                            .create_work_item(WorkItemInsert {
                                public_id: uuid::Uuid::now_v7(),
                                worklist_id: handle.worklist.id,
                                parent_item_id: handle.parent_item_id,
                                title: &parsed.title,
                                item_kind: &parsed.item_kind,
                                prompt: parsed.prompt.as_deref(),
                                content: content.as_deref(),
                                tools: tools.as_deref(),
                                conflict_policy: parsed.conflict_policy.as_deref(),
                                action_name: parsed.action_name.as_deref(),
                                action_params: action_params.as_deref(),
                                priority: parsed.priority,
                                after_ids: after_ids.as_deref(),
                                metadata: metadata.as_deref(),
                            })
                            .await
                    })
                    .map_err(mlua::Error::runtime)?;
                    Ok(Value::Table(item_proxy(lua, handle.clone(), row)?))
                },
            )?,
        )?;
    }

    {
        let handle = handle.clone();
        proxy.set(
            "all",
            lua.create_function(move |lua, (_self, opts): (Table, Option<Table>)| {
                let limit = parse_limit(opts.clone())?;
                let where_map = parse_where_map(lua, opts)?;
                let rows = crate::harness::globals::block_on_current(async {
                    handle.store.list_work_items(handle.worklist.id).await
                })
                .map_err(mlua::Error::runtime)?;
                let out = lua.create_table()?;
                for (index, row) in rows
                    .into_iter()
                    .filter(|row| row.parent_item_id == handle.parent_item_id)
                    .filter(|row| row_matches_where(row, where_map.as_ref()))
                    .take(limit.unwrap_or(usize::MAX))
                    .enumerate()
                {
                    out.set(index + 1, item_proxy(lua, handle.clone(), row)?)?;
                }
                Ok(Value::Table(out))
            })?,
        )?;
    }

    {
        let handle = handle.clone();
        proxy.set(
            "pending",
            lua.create_function(move |lua, (_self, opts): (Table, Option<Table>)| {
                let limit = parse_limit(opts.clone())?;
                let where_map = parse_where_map(lua, opts)?;
                let rows = crate::harness::globals::block_on_current(async {
                    handle.store.list_work_items(handle.worklist.id).await
                })
                .map_err(mlua::Error::runtime)?;
                let status_map = rows
                    .iter()
                    .map(|row| (public_id_string(&row.public_id), row.status.clone()))
                    .collect::<HashMap<_, _>>();
                let now = now_unix_ms();
                let out = lua.create_table()?;
                for (index, row) in rows
                    .into_iter()
                    .filter(|row| row.parent_item_id == handle.parent_item_id)
                    .filter(|row| row.status == "pending")
                    .filter(|row| row.claim_execution_id.is_none())
                    .filter(|row| !row_is_paused(row, now))
                    .filter(|row| dependencies_satisfied(row, &status_map))
                    .filter(|row| row_matches_where(row, where_map.as_ref()))
                    .take(limit.unwrap_or(usize::MAX))
                    .enumerate()
                {
                    out.set(index + 1, item_proxy(lua, handle.clone(), row)?)?;
                }
                Ok(Value::Table(out))
            })?,
        )?;
    }

    {
        let handle = handle.clone();
        proxy.set(
            "orphaned",
            lua.create_function(move |lua, (_self, opts): (Table, Option<Table>)| {
                let limit = parse_limit(opts.clone())?;
                let stale_after_ms = parse_stale_after_ms(opts.clone())?;
                let where_map = parse_where_map(lua, opts)?;
                let stale_before = now_unix_ms().saturating_sub(stale_after_ms);
                let rows = crate::harness::globals::block_on_current(async {
                    handle.store.list_work_items(handle.worklist.id).await
                })
                .map_err(mlua::Error::runtime)?;
                let out = lua.create_table()?;
                for (index, row) in rows
                    .into_iter()
                    .filter(|row| row.parent_item_id == handle.parent_item_id)
                    .filter(|row| row_is_orphaned(row, stale_before))
                    .filter(|row| row_matches_where(row, where_map.as_ref()))
                    .take(limit.unwrap_or(usize::MAX))
                    .enumerate()
                {
                    out.set(index + 1, item_proxy(lua, handle.clone(), row)?)?;
                }
                Ok(Value::Table(out))
            })?,
        )?;
    }

    {
        let handle = handle.clone();
        proxy.set(
            "release_stale",
            lua.create_function(move |lua, (_self, opts): (Table, Option<Table>)| {
                require_capability(&handle.app_data, "runtime.worklist.release_stale")
                    .map_err(mlua::Error::runtime)?;
                let limit = parse_limit(opts.clone())?;
                let stale_after_ms = parse_stale_after_ms(opts.clone())?;
                let where_map = parse_where_map(lua, opts)?;
                let stale_before = now_unix_ms().saturating_sub(stale_after_ms);
                let rows = crate::harness::globals::block_on_current(async {
                    handle.store.list_work_items(handle.worklist.id).await
                })
                .map_err(mlua::Error::runtime)?;
                let candidates = rows
                    .into_iter()
                    .filter(|row| row.parent_item_id == handle.parent_item_id)
                    .filter(|row| row_is_orphaned(row, stale_before))
                    .filter(|row| row_matches_where(row, where_map.as_ref()))
                    .take(limit.unwrap_or(usize::MAX))
                    .collect::<Vec<_>>();
                let out = lua.create_table()?;
                for (index, row) in candidates.into_iter().enumerate() {
                    let released = crate::harness::globals::block_on_current(async {
                        handle.store.release_work_item(row.id).await
                    })
                    .map_err(mlua::Error::runtime)?;
                    out.set(index + 1, item_proxy(lua, handle.clone(), released)?)?;
                }
                Ok(Value::Table(out))
            })?,
        )?;
    }

    {
        let handle = handle.clone();
        proxy.set(
            "dispatch_next",
            lua.create_function(move |lua, (self_tbl, opts): (Table, Option<Table>)| {
                require_capability(&handle.app_data, "runtime.worklist.dispatch")
                    .map_err(mlua::Error::runtime)?;
                let next_value = proxy_method_call(lua, &self_tbl, "next", opts)?;
                let item_table = match next_value {
                    Value::Nil => return Ok(Value::Nil),
                    Value::Table(table) => table,
                    other => {
                        return Err(mlua::Error::runtime(format!(
                            "worklist.next returned unexpected value {:?}",
                            other
                        )));
                    }
                };
                let dispatch_fn = item_table.get::<mlua::Function>("dispatch")?;
                let result: Value = dispatch_fn.call((item_table.clone(), Value::Nil))?;
                let out = lua.create_table()?;
                out.set("item", item_table)?;
                out.set("result", result)?;
                Ok(Value::Table(out))
            })?,
        )?;
    }

    {
        let handle = handle.clone();
        proxy.set(
            "paused",
            lua.create_function(move |lua, (_self, opts): (Table, Option<Table>)| {
                let limit = parse_limit(opts.clone())?;
                let due_only = parse_bool_flag(opts.clone(), "due_only")?;
                let where_map = parse_where_map(lua, opts)?;
                let rows = crate::harness::globals::block_on_current(async {
                    handle.store.list_work_items(handle.worklist.id).await
                })
                .map_err(mlua::Error::runtime)?;
                let now = now_unix_ms();
                let out = lua.create_table()?;
                for (index, row) in rows
                    .into_iter()
                    .filter(|row| row.parent_item_id == handle.parent_item_id)
                    .filter(|row| row_pause_flag(row_metadata(row).as_ref()))
                    .filter(|row| !due_only || row_pause_due(row, now))
                    .filter(|row| row_matches_where(row, where_map.as_ref()))
                    .take(limit.unwrap_or(usize::MAX))
                    .enumerate()
                {
                    out.set(index + 1, item_proxy(lua, handle.clone(), row)?)?;
                }
                Ok(Value::Table(out))
            })?,
        )?;
    }

    {
        let handle = handle.clone();
        proxy.set(
            "active",
            lua.create_function(move |lua, _self: Table| {
                let (_agent, session_id, execution_id) = current_claim_identity(&handle.app_data);
                let rows = crate::harness::globals::block_on_current(async {
                    handle.store.list_work_items(handle.worklist.id).await
                })
                .map_err(mlua::Error::runtime)?;
                let row = rows.into_iter().find(|row| {
                    row.parent_item_id == handle.parent_item_id
                        && row.status == "active"
                        && (row.claim_execution_id.as_deref() == execution_id.as_deref()
                            || row.claim_session_id.as_deref() == session_id.as_deref())
                });
                match row {
                    Some(row) => Ok(Value::Table(item_proxy(lua, handle.clone(), row)?)),
                    None => Ok(Value::Nil),
                }
            })?,
        )?;
    }

    {
        let handle = handle.clone();
        proxy.set(
            "next",
            lua.create_function(move |lua, (_self, opts): (Table, Option<Table>)| {
                require_capability(&handle.app_data, "runtime.worklist.next")
                    .map_err(mlua::Error::runtime)?;
                let where_map = parse_where_map(lua, opts)?;
                let rows = crate::harness::globals::block_on_current(async {
                    handle.store.list_work_items(handle.worklist.id).await
                })
                .map_err(mlua::Error::runtime)?;
                let status_map = rows
                    .iter()
                    .map(|row| (public_id_string(&row.public_id), row.status.clone()))
                    .collect::<HashMap<_, _>>();
                let now = now_unix_ms();
                let (agent_id, session_id, execution_id) = current_claim_identity(&handle.app_data);
                for row in rows
                    .iter()
                    .filter(|row| row.parent_item_id == handle.parent_item_id)
                    .filter(|row| row.status == "pending")
                    .filter(|row| row.claim_execution_id.is_none())
                    .filter(|row| !row_is_paused(row, now))
                    .filter(|row| dependencies_satisfied(row, &status_map))
                    .filter(|row| row_matches_where(row, where_map.as_ref()))
                {
                    let claimed = crate::harness::globals::block_on_current(async {
                        handle
                            .store
                            .try_claim_work_item(
                                row.id,
                                &agent_id,
                                session_id.as_deref(),
                                execution_id.as_deref(),
                                now_unix_ms(),
                            )
                            .await
                    })
                    .map_err(mlua::Error::runtime)?;
                    if claimed {
                        let refreshed = crate::harness::globals::block_on_current(async {
                            handle.store.get_work_item_by_id(row.id).await
                        })
                        .map_err(mlua::Error::runtime)?
                        .ok_or_else(|| {
                            mlua::Error::runtime("claimed work item vanished".to_string())
                        })?;
                        return Ok(Value::Table(item_proxy(lua, handle.clone(), refreshed)?));
                    }
                }
                Ok(Value::Nil)
            })?,
        )?;
    }

    {
        proxy.set(
            "current",
            lua.create_function(move |lua, (self_tbl, opts): (Table, Option<Table>)| {
                let active: Value = proxy_method_call(lua, &self_tbl, "active", None)?;
                if !matches!(active, Value::Nil) {
                    return Ok(active);
                }
                proxy_method_call(lua, &self_tbl, "next", opts)
            })?,
        )?;
    }

    {
        let handle = handle.clone();
        proxy.set(
            "find",
            lua.create_function(move |lua, (_self, opts): (Table, Table)| {
                let where_map = parse_where_map(lua, Some(opts))?.ok_or_else(|| {
                    mlua::Error::runtime("worklist.find requires opts.where".to_string())
                })?;
                let rows = crate::harness::globals::block_on_current(async {
                    handle.store.list_work_items(handle.worklist.id).await
                })
                .map_err(mlua::Error::runtime)?;
                match rows
                    .into_iter()
                    .filter(|row| row.parent_item_id == handle.parent_item_id)
                    .find(|row| row_matches_where(row, Some(&where_map)))
                {
                    Some(row) => Ok(Value::Table(item_proxy(lua, handle.clone(), row)?)),
                    None => Ok(Value::Nil),
                }
            })?,
        )?;
    }

    {
        let handle = handle.clone();
        proxy.set(
            "progress",
            lua.create_function(move |lua, _self: Table| {
                let rows = crate::harness::globals::block_on_current(async {
                    handle.store.list_work_items(handle.worklist.id).await
                })
                .map_err(mlua::Error::runtime)?;
                let total = rows
                    .iter()
                    .filter(|row| row.parent_item_id == handle.parent_item_id)
                    .count();
                let done = rows
                    .iter()
                    .filter(|row| {
                        row.parent_item_id == handle.parent_item_id && row.status == "done"
                    })
                    .count();
                let out = lua.create_table()?;
                out.set("done", done)?;
                out.set("total", total)?;
                Ok(Value::Table(out))
            })?,
        )?;
    }

    {
        let handle = handle.clone();
        proxy.set(
            "empty",
            lua.create_function(move |_lua, _self: Table| {
                let rows = crate::harness::globals::block_on_current(async {
                    handle.store.list_work_items(handle.worklist.id).await
                })
                .map_err(mlua::Error::runtime)?;
                let status_map = rows
                    .iter()
                    .map(|row| (public_id_string(&row.public_id), row.status.clone()))
                    .collect::<HashMap<_, _>>();
                let now = now_unix_ms();
                let has_pending = rows.iter().any(|row| {
                    row.parent_item_id == handle.parent_item_id
                        && row.status == "pending"
                        && row.claim_execution_id.is_none()
                        && !row_is_paused(row, now)
                        && dependencies_satisfied(row, &status_map)
                });
                Ok(Value::Boolean(!has_pending))
            })?,
        )?;
    }

    Ok(proxy)
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
                let store = crate::harness::globals::block_on_current(async move {
                    open_store(
                        manager,
                        selector,
                        path_scope,
                        max_open_handles,
                        idle_close_seconds,
                    )
                    .await
                })
                .map_err(mlua::Error::runtime)?;
                let metadata_json = match opts.get::<Value>("metadata")? {
                    Value::Nil => None,
                    value => Some(lua.from_value::<JsonValue>(value)?),
                };
                let metadata_raw =
                    serialize_json_opt(metadata_json.as_ref()).map_err(mlua::Error::runtime)?;
                let worklist = crate::harness::globals::block_on_current(async {
                    store
                        .open_worklist(&name, &scope_ref(scope.as_ref()), metadata_raw.as_deref())
                        .await
                })
                .map_err(mlua::Error::runtime)?;
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
