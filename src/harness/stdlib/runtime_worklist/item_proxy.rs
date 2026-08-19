use mlua::{Lua, Result as LuaResult, Table, Value};
use serde_json::Value as JsonValue;
use turin_types::{TaskInputContent, ToolsConfig};

use super::{
    WorkItemRow, WorkItemUpdate, WorklistHandle, bridge_async_lua, current_claim_identity,
    dispatch_item_result, item_payload_value, item_value, load_required_item, load_work_items,
    now_unix_ms, object_refs, optional_lua_json, parse_i64_opt, parse_json_opt,
    parse_present_json_raw, parse_present_string_array_raw, parse_present_string_opt,
    parse_string_opt, public_id_string, require_capability, row_json_payload_value, row_json_value,
    row_pause_reason, row_pause_until_unix_ms, row_paused, selection, serialize_json_opt,
    work_items_value, worklist_proxy,
};

pub(super) fn item_proxy(lua: &Lua, handle: WorklistHandle, row: WorkItemRow) -> LuaResult<Table> {
    let proxy = worklist_proxy(
        lua,
        WorklistHandle {
            parent_item_id: Some(row.id),
            ..handle.clone()
        },
    )?;
    let item_public_id = public_id_string(&row.public_id);
    object_refs::annotate_workitem_proxy(&proxy, &item_public_id)?;
    let metadata_json = parse_json_opt::<JsonValue>(row.metadata.as_deref())
        .ok()
        .flatten();
    proxy.set("id", item_public_id)?;
    proxy.set("internal_id", row.id)?;
    proxy.set("title", row.title.clone())?;
    proxy.set("kind", row.item_kind.clone())?;
    proxy.set("status", row.status.clone())?;
    proxy.set("priority", row.priority)?;
    proxy.set("parent_internal_id", row.parent_item_id)?;
    proxy.set("prompt", row.prompt.clone())?;
    proxy.set(
        "content",
        row_json_value::<Vec<TaskInputContent>>(lua, row.content.as_deref())?,
    )?;
    proxy.set(
        "tools",
        row_json_value::<ToolsConfig>(lua, row.tools.as_deref())?,
    )?;
    proxy.set("conflict_policy", row.conflict_policy.clone())?;
    proxy.set("action_name", row.action_name.clone())?;
    proxy.set(
        "params",
        row_json_payload_value(lua, row.action_params.as_deref())?,
    )?;
    proxy.set(
        "metadata",
        match metadata_json.clone() {
            Some(value) => object_refs::decode_json_payload(lua, &value)?,
            None => Value::Nil,
        },
    )?;
    proxy.set("paused", row_paused(&row))?;
    proxy.set("pause_reason", row_pause_reason(metadata_json.as_ref()))?;
    proxy.set(
        "pause_until_unix_ms",
        row_pause_until_unix_ms(metadata_json.as_ref()),
    )?;
    proxy.set(
        "after",
        row_json_value::<Vec<String>>(lua, row.after_ids.as_deref())?,
    )?;
    proxy.set("claim_agent_id", row.claim_agent_id.clone())?;
    proxy.set("claim_session_id", row.claim_session_id.clone())?;
    proxy.set("claim_execution_id", row.claim_execution_id.clone())?;
    proxy.set("failure_reason", row.failure_reason.clone())?;
    proxy.set("payload", item_payload_value(lua, &row)?)?;
    object_refs::attach_proxy_action(
        lua,
        &proxy,
        object_refs::ProxyTarget::workitem(Some(handle.worklist.name.clone())),
    )?;

    {
        let handle = handle.clone();
        let item_id = row.id;
        proxy.set(
            "claim",
            lua.create_function(move |lua, _self: Table| {
                require_capability(&handle.app_data, "runtime.worklist.claim")
                    .map_err(mlua::Error::runtime)?;
                let (agent_id, session_id, execution_id) = current_claim_identity(&handle.app_data);
                let claimed = bridge_async_lua(async {
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
                })?;
                let row = load_required_item(&handle, item_id, "work item not found")?;
                if !claimed && row.claim_execution_id.as_deref() != execution_id.as_deref() {
                    return Ok(Value::Nil);
                }
                item_value(lua, &handle, row)
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
                let row = bridge_async_lua(async {
                    handle
                        .store
                        .heartbeat_work_item_claim(item_id, &execution_id, now_unix_ms())
                        .await
                })?
                .ok_or_else(|| {
                    mlua::Error::runtime(
                        "work item is not actively claimed by this execution".to_string(),
                    )
                })?;
                item_value(lua, &handle, row)
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
                let row = load_required_item(&handle, item_id, "work item not found")?;
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
                let metadata_json = meta
                    .map(|value| optional_lua_json(lua, value))
                    .transpose()?
                    .flatten();
                let metadata_raw =
                    serialize_json_opt(metadata_json.as_ref()).map_err(mlua::Error::runtime)?;
                let row = bridge_async_lua(async {
                    handle
                        .store
                        .complete_work_item(item_id, metadata_raw.as_deref())
                        .await
                })?;
                item_value(lua, &handle, row)
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
                let row = bridge_async_lua(async {
                    handle
                        .store
                        .fail_work_item(item_id, reason.as_deref())
                        .await
                })?;
                item_value(lua, &handle, row)
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
                let row =
                    bridge_async_lua(async { handle.store.release_work_item(item_id).await })?;
                item_value(lua, &handle, row)
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
                let prompt = parse_present_string_opt(&fields, "prompt")?;
                let conflict_policy = parse_present_string_opt(&fields, "conflict_policy")?;
                let action_name = parse_present_string_opt(&fields, "action")?;
                let action_params = parse_present_json_raw(lua, &fields, "params")?;
                let content = parse_present_json_raw(lua, &fields, "content")?;
                let tools = parse_present_json_raw(lua, &fields, "tools")?;
                let metadata = parse_present_json_raw(lua, &fields, "metadata")?;
                let after_ids = parse_present_string_array_raw(&fields, "after")?;
                let status = parse_present_string_opt(&fields, "status")?
                    .map(|value| value.unwrap_or_else(|| "pending".to_string()));
                let failure_reason = parse_present_string_opt(&fields, "failure_reason")?;
                let priority = parse_i64_opt(&fields, "priority")?;
                let row = bridge_async_lua(async {
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
                            action_params: action_params.as_ref().map(|value| value.as_deref()),
                            priority,
                            after_ids: after_ids.as_ref().map(|value| value.as_deref()),
                            metadata: metadata.as_ref().map(|value| value.as_deref()),
                            status: status.as_deref(),
                            failure_reason: failure_reason.as_ref().map(|value| value.as_deref()),
                        })
                        .await
                })?;
                item_value(lua, &handle, row)
            })?,
        )?;
    }

    {
        let handle = handle.clone();
        let parent_id = row.id;
        proxy.set(
            "children",
            lua.create_function(move |lua, _self: Table| {
                let rows = selection::children(load_work_items(&handle)?, parent_id);
                work_items_value(lua, &handle, rows)
            })?,
        )?;
    }

    Ok(proxy)
}
