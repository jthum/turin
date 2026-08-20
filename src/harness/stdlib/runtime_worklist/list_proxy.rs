use mlua::{Lua, Result as LuaResult, Table, Value};

use super::{
    WorkItemInsert, WorklistHandle, bridge_async_lua, current_claim_identity, item_value,
    load_required_item, load_work_items, now_unix_ms, object_refs, parse_bool_flag, parse_payload,
    parse_stale_after_ms, parse_where_map, parse_work_item_query, proxy_method_call,
    public_id_string, queried_work_items, require_capability, selection, serialize_json_opt,
    work_items_value,
};

pub(super) fn worklist_proxy(lua: &Lua, handle: WorklistHandle) -> LuaResult<Table> {
    let proxy = lua.create_table()?;
    let public_id = public_id_string(&handle.worklist.public_id);
    proxy.set("name", handle.worklist.name.clone())?;
    proxy.set("scope_ref", handle.worklist.scope_ref.clone())?;
    proxy.set("id", public_id.clone())?;
    proxy.set("internal_id", handle.worklist.id)?;
    object_refs::annotate_worklist_proxy(
        lua,
        &proxy,
        &handle.store_selector,
        &handle.worklist.name,
        &handle.worklist.scope_ref,
        &public_id,
    )?;
    object_refs::attach_proxy_action(
        lua,
        &proxy,
        object_refs::ProxyTarget::worklist(Some(handle.worklist.name.clone())),
    )?;

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
                    let row = bridge_async_lua(async {
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
                    })?;
                    item_value(lua, &handle, row)
                },
            )?,
        )?;
    }

    {
        let handle = handle.clone();
        proxy.set(
            "all",
            lua.create_function(move |lua, (_self, opts): (Table, Option<Table>)| {
                let rows = queried_work_items(lua, &handle, opts.as_ref(), selection::all_rows)?;
                work_items_value(lua, &handle, rows)
            })?,
        )?;
    }

    {
        let handle = handle.clone();
        proxy.set(
            "pending",
            lua.create_function(move |lua, (_self, opts): (Table, Option<Table>)| {
                let rows =
                    queried_work_items(lua, &handle, opts.as_ref(), selection::pending_rows)?;
                work_items_value(lua, &handle, rows)
            })?,
        )?;
    }

    {
        let handle = handle.clone();
        proxy.set(
            "orphaned",
            lua.create_function(move |lua, (_self, opts): (Table, Option<Table>)| {
                let stale_after_ms = parse_stale_after_ms(opts.as_ref())?;
                let stale_before = now_unix_ms().saturating_sub(stale_after_ms);
                let rows = queried_work_items(lua, &handle, opts.as_ref(), |rows, query| {
                    selection::orphaned_rows(rows, query, stale_before)
                })?;
                work_items_value(lua, &handle, rows)
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
                let stale_after_ms = parse_stale_after_ms(opts.as_ref())?;
                let stale_before = now_unix_ms().saturating_sub(stale_after_ms);
                let candidates = queried_work_items(lua, &handle, opts.as_ref(), |rows, query| {
                    selection::orphaned_rows(rows, query, stale_before)
                })?;
                let mut released_rows = Vec::with_capacity(candidates.len());
                for row in candidates {
                    let released = bridge_async_lua(async {
                        handle
                            .store
                            .release_stale_work_item(row.id, stale_before)
                            .await
                    })?;
                    if let Some(released) = released {
                        released_rows.push(released);
                    }
                }
                work_items_value(lua, &handle, released_rows)
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
                let due_only = parse_bool_flag(opts.as_ref(), "due_only")?;
                let now = now_unix_ms();
                let rows = queried_work_items(lua, &handle, opts.as_ref(), |rows, query| {
                    selection::paused_rows(rows, query, due_only, now)
                })?;
                work_items_value(lua, &handle, rows)
            })?,
        )?;
    }

    {
        let handle = handle.clone();
        proxy.set(
            "active",
            lua.create_function(move |lua, _self: Table| {
                let (_agent, session_id, execution_id) = current_claim_identity(&handle.app_data);
                match selection::active_for_current_claim(
                    load_work_items(&handle)?,
                    handle.parent_item_id,
                    session_id.as_deref(),
                    execution_id.as_deref(),
                ) {
                    Some(row) => item_value(lua, &handle, row),
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
                let query = parse_work_item_query(lua, opts.as_ref())?;
                let rows = load_work_items(&handle)?;
                let now = now_unix_ms();
                let (agent_id, session_id, execution_id) = current_claim_identity(&handle.app_data);
                for row in selection::next_candidates(
                    &rows,
                    handle.parent_item_id,
                    query.where_map.as_ref(),
                    now,
                ) {
                    let claimed = bridge_async_lua(async {
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
                    })?;
                    if claimed {
                        let refreshed =
                            load_required_item(&handle, row.id, "claimed work item vanished")?;
                        return item_value(lua, &handle, refreshed);
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
                let where_map = parse_where_map(lua, Some(&opts))?.ok_or_else(|| {
                    mlua::Error::runtime("worklist.find requires opts.where".to_string())
                })?;
                match selection::find_matching(
                    load_work_items(&handle)?,
                    handle.parent_item_id,
                    &where_map,
                ) {
                    Some(row) => item_value(lua, &handle, row),
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
                let rows = load_work_items(&handle)?;
                let (done, total) = selection::progress_counts(&rows, handle.parent_item_id);
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
                let rows = load_work_items(&handle)?;
                let now = now_unix_ms();
                let has_pending = selection::has_pending_work(&rows, handle.parent_item_id, now);
                Ok(Value::Boolean(!has_pending))
            })?,
        )?;
    }

    Ok(proxy)
}
