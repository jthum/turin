use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use mlua::{Function, Lua, Result as LuaResult, Table, Value};
use serde_json::{Map as JsonMap, Value as JsonValue};
use turin_daemon_protocol::{ScheduleActionParams, ScheduleCreateParams};

use crate::harness::globals::HarnessAppData;
use crate::harness::scheduler::HarnessSchedulerAccess;
use crate::harness::stdlib::binding_common::wrap_registered_callback;
use crate::harness::stdlib::object_refs;
use crate::harness::stdlib::runtime_worklist::{build_work_item_proxy, public_id_string};
use crate::harness::stdlib::system_globals::ensure_load_time;
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::{WorkItemRow, WorklistRow};
use crate::persistence::state::{StateStore, WorkItemUpdate};

const DECLARED_ACTION_REGISTRY_KEY: &str = "__harness_declared_actions";

#[derive(Clone)]
pub(crate) struct ActionInvocationContext {
    pub app_data: HarnessAppData,
    pub action_name: String,
    pub params: JsonValue,
    pub work_item: Option<ActionWorkItemContext>,
}

#[derive(Clone)]
pub(crate) struct ActionWorkItemContext {
    pub store: Arc<StateStore>,
    pub store_selector: StoreSelector,
    pub worklist: WorklistRow,
    pub row: WorkItemRow,
}

fn now_unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis() as i64
}

pub fn register_action_globals(lua: &Lua) -> LuaResult<()> {
    let action_table = lua.create_table()?;

    action_table.set(
        "define",
        lua.create_function(|lua, (name, handler): (String, Function)| {
            ensure_load_time(lua, "action.define")?;

            let registry = ensure_declared_action_registry(lua)?;
            if registry.contains_key(name.clone())? {
                return Err(mlua::Error::runtime(format!(
                    "action.define('{}') conflicts with an existing declared action",
                    name
                )));
            }

            registry.set(name, wrap_registered_callback(lua, handler)?)?;
            Ok(())
        })?,
    )?;

    action_table.set(
        "define_on",
        lua.create_function(
            |lua, (target_value, method, handler): (Value, String, Function)| {
                ensure_load_time(lua, "action.define_on")?;
                let target = object_refs::parse_target(target_value)?;
                let action_name = action_name_for_target_method(&target, &method);

                let registry = ensure_declared_action_registry(lua)?;
                if registry.contains_key(action_name.clone())? {
                    return Err(mlua::Error::runtime(format!(
                        "action.define_on('{}', '{}') conflicts with existing action '{}'",
                        target.key(),
                        method,
                        action_name
                    )));
                }

                let handler = wrap_registered_callback(lua, handler)?;
                let method_name = method.clone();
                let action_name_for_error = action_name.clone();
                let wrapper =
                    lua.create_function(move |_lua, (ctx, envelope): (Table, Value)| {
                        let (subject, params) = match envelope {
                            Value::Table(table) if table.contains_key("subject")? => {
                                let subject = table.get::<Value>("subject")?;
                                let params = table.get::<Value>("params")?;
                                (subject, params)
                            }
                            Value::Nil => {
                                return Err(mlua::Error::runtime(format!(
                                    "action '{}' requires a subject",
                                    action_name_for_error
                                )));
                            }
                            other => (other, Value::Nil),
                        };
                        if matches!(subject, Value::Nil) {
                            return Err(mlua::Error::runtime(format!(
                                "action '{}' requires a subject",
                                action_name_for_error
                            )));
                        }
                        handler.call::<Value>((ctx, subject, params))
                    })?;

                registry.set(action_name.clone(), wrapper)?;
                object_refs::register_proxy_method(lua, target, &method_name, &action_name)?;
                Ok(())
            },
        )?,
    )?;

    action_table.set(
        "run",
        lua.create_function(|lua, (name, params): (String, Option<Value>)| {
            let params = match params {
                None | Some(Value::Nil) => JsonValue::Object(JsonMap::new()),
                Some(value) => object_refs::encode_lua_payload(lua, value)?,
            };

            let app_data = lua
                .app_data_ref::<HarnessAppData>()
                .map(|app_data| app_data.clone())
                .ok_or_else(|| mlua::Error::runtime("Harness app data missing"))?;
            let action_name = name.clone();
            let result = invoke_declared_action(
                lua,
                &name,
                params.clone(),
                ActionInvocationContext {
                    app_data,
                    action_name,
                    params,
                    work_item: None,
                },
            )
            .map_err(mlua::Error::runtime)?;

            match result {
                Some(value) => object_refs::decode_json_payload(lua, &value),
                None => Ok(Value::Nil),
            }
        })?,
    )?;

    lua.globals().set("action", action_table)?;
    Ok(())
}

fn action_name_for_target_method(target: &object_refs::ProxyTarget, method: &str) -> String {
    match target.kind.as_str() {
        "scope" => target
            .name
            .as_ref()
            .map(|kind| format!("{kind}.{method}"))
            .unwrap_or_else(|| format!("scope.{method}")),
        "worklist" => target
            .name
            .as_ref()
            .map(|name| format!("worklist.{name}.{method}"))
            .unwrap_or_else(|| format!("worklist.{method}")),
        "workitem" => target
            .name
            .as_ref()
            .map(|name| format!("workitem.{name}.{method}"))
            .unwrap_or_else(|| format!("workitem.{method}")),
        _ => method.to_string(),
    }
}

fn ensure_declared_action_registry(lua: &Lua) -> LuaResult<Table> {
    let globals = lua.globals();
    if !globals.contains_key(DECLARED_ACTION_REGISTRY_KEY)? {
        globals.set(DECLARED_ACTION_REGISTRY_KEY, lua.create_table()?)?;
    }
    globals.get(DECLARED_ACTION_REGISTRY_KEY)
}

pub(crate) fn invoke_declared_action(
    lua: &Lua,
    name: &str,
    params: JsonValue,
    invocation: ActionInvocationContext,
) -> Result<Option<JsonValue>> {
    let registry = ensure_declared_action_registry(lua)?;
    let handler = match registry.get::<Value>(name)? {
        Value::Nil => return invoke_builtin_action(lua, name, params),
        Value::Function(function) => function,
        other => anyhow::bail!(
            "declared action registry entry '{}' has invalid type {:?}",
            name,
            other
        ),
    };

    let ctx = build_action_context(lua, &invocation)?;
    let lua_args = object_refs::decode_json_payload(lua, &params)?;
    let result = handler.call::<Value>((ctx, lua_args))?;
    let result = object_refs::encode_lua_payload(lua, result).map_err(|err| {
        anyhow::anyhow!(
            "declared action '{}' handler returned invalid value: {}",
            name,
            err
        )
    })?;

    Ok(Some(result))
}

fn build_action_context(lua: &Lua, invocation: &ActionInvocationContext) -> LuaResult<Table> {
    let ctx = lua.create_table()?;
    ctx.set("action", invocation.action_name.clone())?;
    ctx.set(
        "params",
        object_refs::decode_json_payload(lua, &invocation.params)?,
    )?;

    let checkpoint = invocation
        .work_item
        .as_ref()
        .and_then(|item| item.row.metadata.as_deref())
        .and_then(|raw| serde_json::from_str::<JsonValue>(raw).ok())
        .and_then(|value| value.get("checkpoint").cloned())
        .unwrap_or(JsonValue::Object(JsonMap::new()));
    ctx.set("checkpoint", checkpoint_proxy(lua, checkpoint)?)?;

    match invocation.work_item.as_ref() {
        Some(item) => {
            ctx.set(
                "item",
                build_work_item_proxy(
                    lua,
                    item.store.clone(),
                    item.store_selector.clone(),
                    item.worklist.clone(),
                    item.row.clone(),
                    invocation.app_data.clone(),
                )?,
            )?;
        }
        None => ctx.set("item", Value::Nil)?,
    }

    {
        let invocation = invocation.clone();
        ctx.set(
            "complete",
            lua.create_function(move |lua, (_self, value): (Table, Value)| {
                let value_json = match value {
                    Value::Nil => None,
                    value => Some(
                        object_refs::encode_lua_payload(lua, value)
                            .map_err(mlua::Error::runtime)?,
                    ),
                };
                let result =
                    complete_action(&invocation, value_json).map_err(mlua::Error::runtime)?;
                object_refs::decode_json_payload(lua, &result)
            })?,
        )?;
    }

    {
        let invocation = invocation.clone();
        ctx.set(
            "fail",
            lua.create_function(move |lua, (_self, value): (Table, Value)| {
                let value_json = match value {
                    Value::Nil => None,
                    value => Some(
                        object_refs::encode_lua_payload(lua, value)
                            .map_err(mlua::Error::runtime)?,
                    ),
                };
                let result = fail_action(&invocation, value_json).map_err(mlua::Error::runtime)?;
                object_refs::decode_json_payload(lua, &result)
            })?,
        )?;
    }

    {
        let invocation = invocation.clone();
        ctx.set(
            "cancel",
            lua.create_function(move |lua, (_self, value): (Table, Value)| {
                let value_json = match value {
                    Value::Nil => None,
                    value => Some(
                        object_refs::encode_lua_payload(lua, value)
                            .map_err(mlua::Error::runtime)?,
                    ),
                };
                let result =
                    cancel_action(&invocation, value_json).map_err(mlua::Error::runtime)?;
                object_refs::decode_json_payload(lua, &result)
            })?,
        )?;
    }

    {
        let invocation = invocation.clone();
        ctx.set(
            "is_cancelled",
            lua.create_function(move |_lua, _self: Table| Ok(action_is_cancelled(&invocation)))?,
        )?;
    }

    {
        let invocation = invocation.clone();
        ctx.set(
            "pause",
            lua.create_function(move |lua, (_self, value): (Table, Value)| {
                let opts = match value {
                    Value::Table(table) => {
                        object_refs::encode_lua_payload(lua, Value::Table(table))?
                    }
                    other => {
                        return Err(mlua::Error::runtime(format!(
                            "ctx:pause requires an object-like table, got {:?}",
                            other
                        )));
                    }
                };
                let result = pause_action(&invocation, opts).map_err(mlua::Error::runtime)?;
                object_refs::decode_json_payload(lua, &result)
            })?,
        )?;
    }

    {
        let invocation = invocation.clone();
        ctx.set(
            "pause_for",
            lua.create_function(
                move |lua, (_self, seconds, opts): (Table, u64, Option<Value>)| {
                    let mut opts = match opts {
                        None | Some(Value::Nil) => JsonMap::new(),
                        Some(Value::Table(table)) => {
                            match object_refs::encode_lua_payload(lua, Value::Table(table))? {
                                JsonValue::Object(map) => map,
                                _ => {
                                    return Err(mlua::Error::runtime(
                                        "ctx:pause_for opts must be an object-like table"
                                            .to_string(),
                                    ));
                                }
                            }
                        }
                        Some(other) => {
                            return Err(mlua::Error::runtime(format!(
                                "ctx:pause_for opts must be an object-like table, got {:?}",
                                other
                            )));
                        }
                    };
                    opts.insert("resume_in_seconds".to_string(), JsonValue::from(seconds));
                    let result = pause_action(&invocation, JsonValue::Object(opts))
                        .map_err(mlua::Error::runtime)?;
                    object_refs::decode_json_payload(lua, &result)
                },
            )?,
        )?;
    }

    Ok(ctx)
}

fn checkpoint_proxy(lua: &Lua, checkpoint: JsonValue) -> LuaResult<Table> {
    let table = match object_refs::decode_json_payload(lua, &checkpoint)? {
        Value::Table(table) => table,
        _ => lua.create_table()?,
    };
    table.set("get", {
        let checkpoint = checkpoint.clone();
        lua.create_function(
            move |lua, (_self, key, default): (Table, String, Option<Value>)| {
                let value = checkpoint
                    .as_object()
                    .and_then(|map| map.get(&key))
                    .cloned();
                match value {
                    Some(value) => object_refs::decode_json_payload(lua, &value),
                    None => Ok(default.unwrap_or(Value::Nil)),
                }
            },
        )?
    })?;
    table.set(
        "all",
        lua.create_function(move |lua, self_tbl: Table| {
            let out = lua.create_table()?;
            for pair in self_tbl.pairs::<Value, Value>() {
                let (key, value) = pair?;
                if let Value::String(key_str) = &key {
                    let name = key_str.to_str()?;
                    if name == "get" || name == "all" {
                        continue;
                    }
                }
                out.set(key, value)?;
            }
            Ok(Value::Table(out))
        })?,
    )?;
    Ok(table)
}

fn action_agent_id(app_data: &HarnessAppData) -> String {
    app_data
        .execution_ctx
        .lock()
        .ok()
        .and_then(|lock| lock.agent_id.clone())
        .unwrap_or_else(|| app_data.config.agent.id.clone())
}

fn action_is_cancelled(invocation: &ActionInvocationContext) -> bool {
    invocation
        .app_data
        .execution_ctx
        .lock()
        .ok()
        .and_then(|lock| lock.cancel_token.clone())
        .is_some_and(|token| token.is_cancelled())
}

fn merge_metadata_patch(existing: Option<&str>, patch: JsonValue) -> Result<Option<String>> {
    if patch.is_null() {
        return Ok(existing.map(ToString::to_string));
    }
    let merged = match (
        existing.and_then(|raw| serde_json::from_str::<JsonValue>(raw).ok()),
        patch,
    ) {
        (Some(JsonValue::Object(mut current)), JsonValue::Object(patch)) => {
            for (key, value) in patch {
                current.insert(key, value);
            }
            JsonValue::Object(current)
        }
        (_, patch) => patch,
    };
    Ok(Some(serde_json::to_string(&merged)?))
}

fn value_reason(value: Option<&JsonValue>) -> Option<String> {
    match value {
        Some(JsonValue::String(s)) if !s.is_empty() => Some(s.clone()),
        Some(JsonValue::Object(map)) => map
            .get("because")
            .or_else(|| map.get("reason"))
            .or_else(|| map.get("message"))
            .and_then(|value| value.as_str())
            .map(ToString::to_string),
        _ => None,
    }
}

fn complete_action(
    invocation: &ActionInvocationContext,
    value: Option<JsonValue>,
) -> Result<JsonValue> {
    if let Some(item) = invocation.work_item.as_ref() {
        let metadata = value
            .clone()
            .map(|value| {
                let mut map = JsonMap::new();
                map.insert("output".to_string(), value);
                JsonValue::Object(map)
            })
            .map(|patch| merge_metadata_patch(item.row.metadata.as_deref(), patch))
            .transpose()?
            .flatten();
        crate::harness::globals::block_on_current(async {
            item.store
                .complete_work_item(item.row.id, metadata.as_deref())
                .await
        })?;
    }

    let mut out = JsonMap::new();
    out.insert(
        "status".to_string(),
        JsonValue::String("completed".to_string()),
    );
    if let Some(value) = value {
        out.insert("result".to_string(), value);
    }
    Ok(JsonValue::Object(out))
}

fn fail_action(
    invocation: &ActionInvocationContext,
    value: Option<JsonValue>,
) -> Result<JsonValue> {
    if let Some(item) = invocation.work_item.as_ref() {
        let reason = value_reason(value.as_ref());
        let metadata = value
            .clone()
            .map(|value| {
                let mut map = JsonMap::new();
                map.insert("failure".to_string(), value);
                JsonValue::Object(map)
            })
            .map(|patch| merge_metadata_patch(item.row.metadata.as_deref(), patch))
            .transpose()?
            .flatten();
        crate::harness::globals::block_on_current(async {
            if let Some(metadata) = metadata.as_deref() {
                item.store
                    .update_work_item(WorkItemUpdate {
                        id: item.row.id,
                        metadata: Some(Some(metadata)),
                        ..Default::default()
                    })
                    .await
                    .context("failed to persist work item failure metadata")?;
            }
            item.store
                .fail_work_item(item.row.id, reason.as_deref())
                .await
        })?;
    }

    let mut out = JsonMap::new();
    out.insert(
        "status".to_string(),
        JsonValue::String("failed".to_string()),
    );
    if let Some(value) = value {
        out.insert("error".to_string(), value);
    }
    Ok(JsonValue::Object(out))
}

fn cancel_action(
    invocation: &ActionInvocationContext,
    value: Option<JsonValue>,
) -> Result<JsonValue> {
    if let Some(item) = invocation.work_item.as_ref() {
        let mut patch = match value.clone() {
            Some(JsonValue::Object(map)) => map,
            Some(other) => {
                let mut map = JsonMap::new();
                map.insert("cancel".to_string(), other);
                map
            }
            None => JsonMap::new(),
        };
        patch.insert(
            "cancelled_at_unix_ms".to_string(),
            JsonValue::from(now_unix_ms()),
        );
        let metadata =
            merge_metadata_patch(item.row.metadata.as_deref(), JsonValue::Object(patch))?;
        crate::harness::globals::block_on_current(async {
            item.store
                .update_work_item(WorkItemUpdate {
                    id: item.row.id,
                    metadata: Some(metadata.as_deref()),
                    ..Default::default()
                })
                .await
                .context("failed to persist work item cancel metadata")?;
            item.store.release_work_item(item.row.id).await
        })?;
    }

    let mut out = JsonMap::new();
    out.insert(
        "status".to_string(),
        JsonValue::String("cancelled".to_string()),
    );
    if let Some(value) = value {
        out.insert("cancel".to_string(), value);
    }
    Ok(JsonValue::Object(out))
}

fn pause_action(invocation: &ActionInvocationContext, opts: JsonValue) -> Result<JsonValue> {
    let opts = match opts {
        JsonValue::Object(map) => map,
        _ => anyhow::bail!("ctx:pause requires an object-like table"),
    };

    let reason = opts
        .get("because")
        .or_else(|| opts.get("reason"))
        .and_then(|value| value.as_str())
        .map(ToString::to_string);
    let note = opts
        .get("note")
        .and_then(|value| value.as_str())
        .map(ToString::to_string);
    let resume_in_seconds = opts
        .get("resume_in_seconds")
        .and_then(|value| value.as_u64());
    let checkpoint = opts.get("checkpoint").cloned();

    let scheduled_job_id = if let Some(item) = invocation.work_item.as_ref() {
        let mut patch = JsonMap::new();
        if let Some(reason) = reason.as_ref() {
            patch.insert(
                "pause_reason".to_string(),
                JsonValue::String(reason.clone()),
            );
        }
        if let Some(note) = note.as_ref() {
            patch.insert("pause_note".to_string(), JsonValue::String(note.clone()));
        }
        if let Some(checkpoint) = checkpoint.clone() {
            patch.insert("checkpoint".to_string(), checkpoint);
        }
        patch.insert("paused".to_string(), JsonValue::Bool(true));
        let paused_at_unix_ms = now_unix_ms();
        patch.insert(
            "paused_at_unix_ms".to_string(),
            JsonValue::from(paused_at_unix_ms),
        );
        if let Some(after_seconds) = resume_in_seconds {
            patch.insert(
                "pause_until_unix_ms".to_string(),
                JsonValue::from(
                    paused_at_unix_ms.saturating_add((after_seconds.saturating_mul(1000)) as i64),
                ),
            );
        }
        let metadata =
            merge_metadata_patch(item.row.metadata.as_deref(), JsonValue::Object(patch))?;
        crate::harness::globals::block_on_current(async {
            item.store
                .pause_work_item(item.row.id, metadata.as_deref())
                .await
                .context("failed to persist paused work item state")
        })?;

        match resume_in_seconds {
            Some(after_seconds) => schedule_worklist_resume(
                invocation.app_data.scheduler.as_ref(),
                action_agent_id(&invocation.app_data),
                item,
                after_seconds,
            )?,
            None => None,
        }
    } else {
        match resume_in_seconds {
            Some(after_seconds) => schedule_action_resume(
                invocation.app_data.scheduler.as_ref(),
                action_agent_id(&invocation.app_data),
                &invocation.action_name,
                invocation.params.clone(),
                after_seconds,
            )?,
            None => None,
        }
    };

    let mut out = JsonMap::new();
    out.insert(
        "status".to_string(),
        JsonValue::String("paused".to_string()),
    );
    if let Some(reason) = reason {
        out.insert("reason".to_string(), JsonValue::String(reason));
    }
    if let Some(note) = note {
        out.insert("note".to_string(), JsonValue::String(note));
    }
    if let Some(checkpoint) = checkpoint {
        out.insert("checkpoint".to_string(), checkpoint);
    }
    if let Some(seconds) = resume_in_seconds {
        out.insert("resume_in_seconds".to_string(), JsonValue::from(seconds));
    }
    if let Some(job_id) = scheduled_job_id {
        out.insert("resume_job_id".to_string(), JsonValue::String(job_id));
    }
    Ok(JsonValue::Object(out))
}

fn schedule_action_resume(
    scheduler: Option<&Arc<HarnessSchedulerAccess>>,
    agent_id: String,
    action_name: &str,
    params: JsonValue,
    after_seconds: u64,
) -> Result<Option<String>> {
    let Some(scheduler) = scheduler else {
        return Ok(None);
    };
    let next_run_unix_ms =
        now_unix_ms().saturating_add((after_seconds.saturating_mul(1000)) as i64);
    let job = crate::harness::globals::block_on_current(async {
        scheduler
            .create_job(ScheduleCreateParams {
                agent_id,
                prompt: None,
                content: None,
                tools: None,
                conflict_policy: None,
                action: Some(ScheduleActionParams {
                    name: action_name.to_string(),
                    params: Some(params),
                }),
                persistence: None,
                next_run_unix_ms,
                interval_seconds: None,
                recurring_pattern: None,
                overlap_policy: Some("skip".to_string()),
                work_key: None,
                max_concurrency: None,
                enabled: true,
            })
            .await
    })?;
    Ok(Some(job.public_id))
}

fn store_selector_json(selector: &StoreSelector) -> JsonValue {
    match selector {
        StoreSelector::Alias(alias) => JsonValue::String(alias.clone()),
        StoreSelector::Path(path) => serde_json::json!({ "path": path }),
        StoreSelector::Handle(handle) => serde_json::json!({ "store": handle }),
    }
}

fn schedule_worklist_resume(
    scheduler: Option<&Arc<HarnessSchedulerAccess>>,
    agent_id: String,
    item: &ActionWorkItemContext,
    after_seconds: u64,
) -> Result<Option<String>> {
    let Some(scheduler) = scheduler else {
        return Ok(None);
    };
    let next_run_unix_ms =
        now_unix_ms().saturating_add((after_seconds.saturating_mul(1000)) as i64);
    let scope = if item.worklist.scope_ref.is_empty() {
        JsonValue::Null
    } else {
        JsonValue::String(item.worklist.scope_ref.clone())
    };
    let job = crate::harness::globals::block_on_current(async {
        scheduler
            .create_job(ScheduleCreateParams {
                agent_id,
                prompt: None,
                content: None,
                tools: None,
                conflict_policy: None,
                action: Some(ScheduleActionParams {
                    name: "worklist.dispatch_next".to_string(),
                    params: Some(serde_json::json!({
                        "name": item.worklist.name,
                        "scope": scope,
                        "store": store_selector_json(&item.store_selector),
                        "where": {
                            "id": public_id_string(&item.row.public_id)
                        }
                    })),
                }),
                persistence: None,
                next_run_unix_ms,
                interval_seconds: None,
                recurring_pattern: None,
                overlap_policy: Some("skip".to_string()),
                work_key: None,
                max_concurrency: None,
                enabled: true,
            })
            .await
    })?;
    Ok(Some(job.public_id))
}

fn invoke_builtin_action(lua: &Lua, name: &str, params: JsonValue) -> Result<Option<JsonValue>> {
    match name {
        "worklist.dispatch_next" => Ok(Some(invoke_builtin_worklist_method(
            lua,
            "dispatch_next",
            params,
        )?)),
        "worklist.release_stale" => Ok(Some(invoke_builtin_worklist_method(
            lua,
            "release_stale",
            params,
        )?)),
        _ => Ok(None),
    }
}

fn invoke_builtin_worklist_method(
    lua: &Lua,
    method_name: &str,
    params: JsonValue,
) -> Result<JsonValue> {
    let params_value = object_refs::decode_json_payload(lua, &params)?;
    let params_table = match params_value {
        Value::Nil => lua.create_table()?,
        Value::Table(table) => table,
        other => anyhow::bail!(
            "built-in action 'worklist.{}' requires object-like params, got {:?}",
            method_name,
            other
        ),
    };
    if !params_table.contains_key("name")? {
        anyhow::bail!(
            "built-in action 'worklist.{}' requires params.name",
            method_name
        );
    }

    let globals = lua.globals();
    let runtime: Table = globals.get("runtime")?;
    let runtime_worklist: Table = runtime.get("worklist")?;
    let open_fn: Function = runtime_worklist.get("open")?;
    let list_proxy: Table = open_fn.call(params_table.clone())?;
    let method: Function = list_proxy.get(method_name)?;
    let result: Value = method.call((list_proxy, params_table))?;
    worklist_action_result_to_json(lua, method_name, result)
}

fn worklist_action_result_to_json(
    lua: &Lua,
    method_name: &str,
    result: Value,
) -> Result<JsonValue> {
    match method_name {
        "dispatch_next" => match result {
            Value::Nil => Ok(JsonValue::Null),
            Value::Table(table) => {
                let item = summarize_worklist_item_table(&table.get::<Table>("item")?)?;
                let dispatch_result = object_refs::encode_lua_payload(lua, table.get("result")?)?;
                Ok(serde_json::json!({
                    "item": item,
                    "result": dispatch_result,
                }))
            }
            other => anyhow::bail!(
                "built-in action 'worklist.dispatch_next' returned unexpected value {:?}",
                other
            ),
        },
        "release_stale" => match result {
            Value::Table(table) => {
                let mut items = Vec::new();
                for value in table.sequence_values::<Table>() {
                    items.push(summarize_worklist_item_table(&value?)?);
                }
                Ok(JsonValue::Array(items))
            }
            other => anyhow::bail!(
                "built-in action 'worklist.release_stale' returned unexpected value {:?}",
                other
            ),
        },
        _ => object_refs::encode_lua_payload(lua, result).map_err(Into::into),
    }
}

fn summarize_worklist_item_table(table: &Table) -> Result<JsonValue> {
    Ok(serde_json::json!({
        "id": table.get::<Option<String>>("id")?,
        "title": table.get::<Option<String>>("title")?,
        "kind": table.get::<Option<String>>("kind")?,
        "status": table.get::<Option<String>>("status")?,
        "priority": table.get::<Option<i64>>("priority")?,
        "prompt": table.get::<Option<String>>("prompt")?,
        "action": table.get::<Option<String>>("action_name")?,
        "claim_execution_id": table.get::<Option<String>>("claim_execution_id")?,
        "failure_reason": table.get::<Option<String>>("failure_reason")?,
    }))
}
