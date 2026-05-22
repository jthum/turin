use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use mlua::{Lua, Result as LuaResult, Table, Value};
use serde_json::{Map as JsonMap, Value as JsonValue};
use turin_daemon_protocol::{ScheduleActionParams, ScheduleCreateParams};

use crate::harness::globals::HarnessAppData;
use crate::harness::scheduler::HarnessSchedulerAccess;
use crate::harness::stdlib::binding_common::{optional_lua_json, optional_lua_object_json};
use crate::harness::stdlib::object_refs;
use crate::harness::stdlib::runtime_worklist::{build_work_item_proxy, public_id_string};
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::{WorkItemRow, WorklistRow};
use crate::persistence::state::{StateStore, WorkItemUpdate};

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

pub(crate) fn build_action_context(
    lua: &Lua,
    invocation: &ActionInvocationContext,
) -> LuaResult<Table> {
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

    set_value_action_method(lua, &ctx, "complete", invocation, complete_action)?;
    set_value_action_method(lua, &ctx, "fail", invocation, fail_action)?;
    set_value_action_method(lua, &ctx, "cancel", invocation, cancel_action)?;

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
                let opts = optional_lua_object_json(lua, Some(value), "ctx:pause")?;
                let result = pause_action(&invocation, JsonValue::Object(opts))
                    .map_err(mlua::Error::runtime)?;
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
                    let mut opts = optional_lua_object_json(lua, opts, "ctx:pause_for opts")?;
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

fn set_value_action_method(
    lua: &Lua,
    ctx: &Table,
    name: &str,
    invocation: &ActionInvocationContext,
    action: fn(&ActionInvocationContext, Option<JsonValue>) -> Result<JsonValue>,
) -> LuaResult<()> {
    let invocation = invocation.clone();
    ctx.set(
        name,
        lua.create_function(move |lua, (_self, value): (Table, Value)| {
            let value_json = optional_lua_json(lua, value)?;
            let result = action(&invocation, value_json).map_err(mlua::Error::runtime)?;
            object_refs::decode_json_payload(lua, &result)
        })?,
    )
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

fn metadata_value_patch(
    existing: Option<&str>,
    key: &str,
    value: Option<JsonValue>,
) -> Result<Option<String>> {
    value
        .map(|value| {
            let mut map = JsonMap::new();
            map.insert(key.to_string(), value);
            merge_metadata_patch(existing, JsonValue::Object(map))
        })
        .transpose()
        .map(Option::flatten)
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

fn action_status_result(status: &str, field: Option<(&str, JsonValue)>) -> JsonValue {
    let mut out = JsonMap::new();
    out.insert("status".to_string(), JsonValue::String(status.to_string()));
    if let Some((key, value)) = field {
        out.insert(key.to_string(), value);
    }
    JsonValue::Object(out)
}

fn complete_action(
    invocation: &ActionInvocationContext,
    value: Option<JsonValue>,
) -> Result<JsonValue> {
    if let Some(item) = invocation.work_item.as_ref() {
        let metadata = metadata_value_patch(item.row.metadata.as_deref(), "output", value.clone())?;
        crate::harness::globals::block_on_current(async {
            item.store
                .complete_work_item(item.row.id, metadata.as_deref())
                .await
        })?;
    }

    Ok(action_status_result(
        "completed",
        value.map(|value| ("result", value)),
    ))
}

fn fail_action(
    invocation: &ActionInvocationContext,
    value: Option<JsonValue>,
) -> Result<JsonValue> {
    if let Some(item) = invocation.work_item.as_ref() {
        let reason = value_reason(value.as_ref());
        let metadata =
            metadata_value_patch(item.row.metadata.as_deref(), "failure", value.clone())?;
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

    Ok(action_status_result(
        "failed",
        value.map(|value| ("error", value)),
    ))
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

    Ok(action_status_result(
        "cancelled",
        value.map(|value| ("cancel", value)),
    ))
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
    let job_id = create_resume_job(
        scheduler,
        agent_id,
        ScheduleActionParams {
            name: action_name.to_string(),
            params: Some(params),
        },
        after_seconds,
    )?;
    Ok(Some(job_id))
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
    let scope = if item.worklist.scope_ref.is_empty() {
        JsonValue::Null
    } else {
        JsonValue::String(item.worklist.scope_ref.clone())
    };
    let action = ScheduleActionParams {
        name: "worklist.dispatch_next".to_string(),
        params: Some(serde_json::json!({
            "name": item.worklist.name,
            "scope": scope,
            "store": store_selector_json(&item.store_selector),
            "where": {
                "id": public_id_string(&item.row.public_id)
            }
        })),
    };
    Ok(Some(create_resume_job(
        scheduler,
        agent_id,
        action,
        after_seconds,
    )?))
}

fn create_resume_job(
    scheduler: &HarnessSchedulerAccess,
    agent_id: String,
    action: ScheduleActionParams,
    after_seconds: u64,
) -> Result<String> {
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
                action: Some(action),
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
    Ok(job.public_id)
}
