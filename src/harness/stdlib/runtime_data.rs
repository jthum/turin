use mlua::{Lua, Result as LuaResult, Table, Value};

use crate::harness::globals::HarnessAppData;
use crate::harness::stdlib::binding_common::{
    memory_feedback_request_from_opts, memory_feedback_signal_from_value,
    memory_purge_request_from_opts, memory_search_request_from_opt, memory_store_request_from_opts,
    metadata_json_or_empty, resolve_memory_search_request, resolve_scoped_store_and_path_scope,
    scoped_state_path_scope_for_selectors, store_selector_from_opts_table,
};
use crate::harness::stdlib::context_selectors::table_to_selector;
use crate::harness::stdlib::memory_kv_bindings::{
    kv_delete_result, kv_get_result, kv_set_result, memory_correct_result, memory_feedback_result,
    memory_purge_result, memory_search_result, memory_store_result,
};

pub fn register_runtime_data_namespaces(
    lua: &Lua,
    runtime_table: &Table,
    app_data: &HarnessAppData,
) -> LuaResult<()> {
    // runtime.memory.* canonical backend delegates
    let runtime_memory = lua.create_table()?;
    {
        let manager = app_data.store_manager.clone();
        let embedding = app_data.embedding_provider.clone();
        let app_data_snapshot = app_data.clone();
        runtime_memory.set(
            "search",
            lua.create_function(
                move |lua, (query, ctx, opts): (String, Table, Option<Value>)| {
                    let selector = table_to_selector(ctx)?;
                    let request = resolve_memory_search_request(
                        &app_data_snapshot,
                        &selector,
                        &memory_search_request_from_opt(lua, opts)?,
                    )?;
                    let source_selectors = request
                        .sources
                        .iter()
                        .filter_map(|source| source.store_selector.as_ref());
                    let path_scope = scoped_state_path_scope_for_selectors(
                        &app_data_snapshot,
                        request
                            .store_selector
                            .as_ref()
                            .into_iter()
                            .chain(source_selectors),
                    )?;
                    memory_search_result(
                        lua,
                        manager.clone(),
                        embedding.clone(),
                        selector,
                        query,
                        request,
                        path_scope,
                    )
                },
            )?,
        )?;
    }
    {
        let manager = app_data.store_manager.clone();
        let embedding = app_data.embedding_provider.clone();
        let app_data_snapshot = app_data.clone();
        runtime_memory.set(
            "store",
            lua.create_function(
                move |lua,
                      (content, ctx, metadata, opts): (
                    String,
                    Table,
                    Option<Table>,
                    Option<Table>,
                )| {
                    let selector = table_to_selector(ctx)?;
                    let metadata_json = metadata_json_or_empty(lua, metadata)?;
                    let mut request = memory_store_request_from_opts(lua, opts)?;
                    let (store_selector, path_scope) = resolve_scoped_store_and_path_scope(
                        &app_data_snapshot,
                        &selector,
                        request.store_selector.clone(),
                    )?;
                    request.store_selector = store_selector;
                    memory_store_result(
                        lua,
                        manager.clone(),
                        embedding.clone(),
                        selector,
                        content,
                        metadata_json,
                        request,
                        path_scope,
                    )
                },
            )?,
        )?;
    }
    {
        let manager = app_data.store_manager.clone();
        let app_data_snapshot = app_data.clone();
        runtime_memory.set(
            "feedback",
            lua.create_function(
                move |lua, (memory_id, signal, ctx, opts): (String, Value, Table, Option<Table>)| {
                    let selector = table_to_selector(ctx)?;
                    let signal = memory_feedback_signal_from_value(signal)?;
                    let mut request = memory_feedback_request_from_opts(lua, opts)?;
                    let (store_selector, path_scope) = resolve_scoped_store_and_path_scope(
                        &app_data_snapshot,
                        &selector,
                        request.store_selector.clone(),
                    )?;
                    request.store_selector = store_selector;
                    memory_feedback_result(
                        lua,
                        manager.clone(),
                        selector,
                        memory_id,
                        signal,
                        request,
                        path_scope,
                    )
                },
            )?,
        )?;
    }
    {
        let manager = app_data.store_manager.clone();
        let embedding = app_data.embedding_provider.clone();
        let app_data_snapshot = app_data.clone();
        runtime_memory.set(
            "correct",
            lua.create_function(
                move |lua,
                      (memory_id, content, ctx, metadata, opts): (
                    String,
                    String,
                    Table,
                    Option<Table>,
                    Option<Table>,
                )| {
                    let selector = table_to_selector(ctx)?;
                    let metadata_json = metadata_json_or_empty(lua, metadata)?;
                    let mut request = memory_store_request_from_opts(lua, opts)?;
                    let (store_selector, path_scope) = resolve_scoped_store_and_path_scope(
                        &app_data_snapshot,
                        &selector,
                        request.store_selector.clone(),
                    )?;
                    request.store_selector = store_selector;
                    memory_correct_result(
                        lua,
                        manager.clone(),
                        embedding.clone(),
                        selector,
                        memory_id,
                        content,
                        metadata_json,
                        request,
                        path_scope,
                    )
                },
            )?,
        )?;
    }
    {
        let manager = app_data.store_manager.clone();
        let app_data_snapshot = app_data.clone();
        runtime_memory.set(
            "purge",
            lua.create_function(move |lua, (ctx, opts): (Table, Option<Table>)| {
                let selector = table_to_selector(ctx)?;
                let mut request = memory_purge_request_from_opts(lua, opts)?;
                let (store_selector, path_scope) = resolve_scoped_store_and_path_scope(
                    &app_data_snapshot,
                    &selector,
                    request.store_selector.clone(),
                )?;
                request.store_selector = store_selector;
                memory_purge_result(lua, manager.clone(), selector, request, path_scope)
            })?,
        )?;
    }
    runtime_table.set("memory", runtime_memory)?;

    // runtime.kv.* canonical backend delegates
    let runtime_kv = lua.create_table()?;
    {
        let manager = app_data.store_manager.clone();
        let app_data_snapshot = app_data.clone();
        runtime_kv.set(
            "get",
            lua.create_function(
                move |lua, (key, ctx, opts): (String, Table, Option<Table>)| {
                    let selector = table_to_selector(ctx)?;
                    let (store_selector, path_scope) = resolve_scoped_store_and_path_scope(
                        &app_data_snapshot,
                        &selector,
                        store_selector_from_opts_table(opts)?,
                    )?;
                    kv_get_result(
                        lua,
                        manager.clone(),
                        selector,
                        key,
                        store_selector,
                        path_scope,
                    )
                },
            )?,
        )?;
    }
    {
        let manager = app_data.store_manager.clone();
        let app_data_snapshot = app_data.clone();
        runtime_kv.set(
            "set",
            lua.create_function(
                move |lua, (key, value, ctx, opts): (String, String, Table, Option<Table>)| {
                    let selector = table_to_selector(ctx)?;
                    let (store_selector, path_scope) = resolve_scoped_store_and_path_scope(
                        &app_data_snapshot,
                        &selector,
                        store_selector_from_opts_table(opts)?,
                    )?;
                    kv_set_result(
                        lua,
                        manager.clone(),
                        selector,
                        key,
                        value,
                        store_selector,
                        path_scope,
                    )
                },
            )?,
        )?;
    }
    {
        let manager = app_data.store_manager.clone();
        let app_data_snapshot = app_data.clone();
        runtime_kv.set(
            "delete",
            lua.create_function(
                move |lua, (key, ctx, opts): (String, Table, Option<Table>)| {
                    let selector = table_to_selector(ctx)?;
                    let (store_selector, path_scope) = resolve_scoped_store_and_path_scope(
                        &app_data_snapshot,
                        &selector,
                        store_selector_from_opts_table(opts)?,
                    )?;
                    kv_delete_result(
                        lua,
                        manager.clone(),
                        selector,
                        key,
                        store_selector,
                        path_scope,
                    )
                },
            )?,
        )?;
    }
    runtime_table.set("kv", runtime_kv)?;

    Ok(())
}
