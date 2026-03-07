use mlua::{Lua, Result as LuaResult, Table, Value};

use crate::harness::globals::HarnessAppData;
use crate::harness::stdlib::binding_common::{
    bool_err, bridge_async_result, memory_correction_row_to_lua_value,
    memory_feedback_request_from_opts, memory_feedback_signal_from_value,
    memory_feedback_state_to_lua_value, memory_purge_report_to_lua_value,
    memory_purge_request_from_opts, memory_rows_to_lua_table, memory_search_request_from_opt,
    memory_store_request_from_opts, memory_store_row_to_lua_value, metadata_json_or_empty,
    nil_err, nil_ok, ok_bool, ok_value, string_ok,
};
use crate::harness::stdlib::context_selectors::table_to_selector;
use crate::harness::stdlib::scoped_data_backend::{
    kv_delete_backend, kv_get_backend, kv_set_backend, memory_correct_backend_with_request,
    memory_feedback_backend_with_request, memory_purge_backend_with_request,
    memory_search_backend_with_request, memory_store_backend_with_request,
};
use crate::inference::embeddings::EmbeddingProvider;
use crate::kernel::identity::ContextSelector;
use crate::persistence::manager::StoreManager;
use std::sync::Arc;

fn memory_search_result(
    lua: &Lua,
    manager: Arc<StoreManager>,
    embedding: Option<Arc<dyn EmbeddingProvider>>,
    selector: ContextSelector,
    query: String,
    request: crate::harness::stdlib::scoped_data_backend::MemorySearchRequest,
) -> LuaResult<(Value, Value)> {
    let result = bridge_async_result(async move {
        memory_search_backend_with_request(&manager, embedding.as_ref(), &selector, &query, &request)
            .await
            .map_err(|e| e.to_string())
    });
    match result {
        Ok(rows) => Ok(ok_value(Value::Table(memory_rows_to_lua_table(lua, rows)?))),
        Err(err) => nil_err(lua, &err),
    }
}

fn memory_store_result(
    lua: &Lua,
    manager: Arc<StoreManager>,
    embedding: Option<Arc<dyn EmbeddingProvider>>,
    selector: ContextSelector,
    content: String,
    metadata_json: serde_json::Value,
    request: crate::harness::stdlib::scoped_data_backend::MemoryStoreRequest,
) -> LuaResult<(Value, Value)> {
    let result = bridge_async_result(async move {
        memory_store_backend_with_request(
            &manager,
            embedding.as_ref(),
            &selector,
            &content,
            &metadata_json,
            &request,
        )
        .await
        .map_err(|e| e.to_string())
    });
    match result {
        Ok(row) => Ok(ok_value(memory_store_row_to_lua_value(lua, row)?)),
        Err(err) => nil_err(lua, &err),
    }
}

fn memory_feedback_result(
    lua: &Lua,
    manager: Arc<StoreManager>,
    selector: ContextSelector,
    memory_id: String,
    signal: crate::harness::stdlib::scoped_data_backend::MemoryFeedbackSignal,
    request: crate::harness::stdlib::scoped_data_backend::MemoryFeedbackRequest,
) -> LuaResult<(Value, Value)> {
    let result = bridge_async_result(async move {
        memory_feedback_backend_with_request(&manager, &selector, &memory_id, signal, &request)
            .await
            .map_err(|e| e.to_string())
    });
    match result {
        Ok(state) => Ok(ok_value(memory_feedback_state_to_lua_value(lua, state)?)),
        Err(err) => nil_err(lua, &err),
    }
}

fn memory_correct_result(
    lua: &Lua,
    manager: Arc<StoreManager>,
    embedding: Option<Arc<dyn EmbeddingProvider>>,
    selector: ContextSelector,
    memory_id: String,
    content: String,
    metadata_json: serde_json::Value,
    request: crate::harness::stdlib::scoped_data_backend::MemoryStoreRequest,
) -> LuaResult<(Value, Value)> {
    let result = bridge_async_result(async move {
        memory_correct_backend_with_request(
            &manager,
            embedding.as_ref(),
            &selector,
            &memory_id,
            &content,
            &metadata_json,
            &request,
        )
        .await
        .map_err(|e| e.to_string())
    });
    match result {
        Ok(row) => Ok(ok_value(memory_correction_row_to_lua_value(lua, row)?)),
        Err(err) => nil_err(lua, &err),
    }
}

fn memory_purge_result(
    lua: &Lua,
    manager: Arc<StoreManager>,
    selector: ContextSelector,
    request: crate::harness::stdlib::scoped_data_backend::MemoryPurgeRequest,
) -> LuaResult<(Value, Value)> {
    let result = bridge_async_result(async move {
        memory_purge_backend_with_request(&manager, &selector, &request)
            .await
            .map_err(|e| e.to_string())
    });
    match result {
        Ok(report) => Ok(ok_value(memory_purge_report_to_lua_value(lua, report)?)),
        Err(err) => nil_err(lua, &err),
    }
}

fn kv_get_result(
    lua: &Lua,
    manager: Arc<StoreManager>,
    selector: ContextSelector,
    key: String,
) -> LuaResult<(Value, Value)> {
    let result = bridge_async_result(async move {
        kv_get_backend(&manager, &selector, &key)
            .await
            .map_err(|e| e.to_string())
    });
    match result {
        Ok(Some(val)) => string_ok(lua, &val),
        Ok(None) => Ok(nil_ok()),
        Err(err) => nil_err(lua, &err),
    }
}

fn kv_set_result(
    lua: &Lua,
    manager: Arc<StoreManager>,
    selector: ContextSelector,
    key: String,
    value: String,
) -> LuaResult<(Value, Value)> {
    let result = bridge_async_result(async move {
        kv_set_backend(&manager, &selector, &key, &value)
            .await
            .map_err(|e| e.to_string())
    });
    match result {
        Ok(_) => Ok(ok_bool()),
        Err(err) => bool_err(lua, &err),
    }
}

fn kv_delete_result(
    lua: &Lua,
    manager: Arc<StoreManager>,
    selector: ContextSelector,
    key: String,
) -> LuaResult<(Value, Value)> {
    let result = bridge_async_result(async move {
        kv_delete_backend(&manager, &selector, &key)
            .await
            .map_err(|e| e.to_string())
    });
    match result {
        Ok(_) => Ok(ok_bool()),
        Err(err) => bool_err(lua, &err),
    }
}

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
        runtime_memory.set(
            "search",
            lua.create_function(
                move |lua, (query, ctx, opts): (String, Table, Option<Value>)| {
                    let selector = table_to_selector(ctx)?;
                    let request = memory_search_request_from_opt(lua, opts)?;
                    memory_search_result(
                        lua,
                        manager.clone(),
                        embedding.clone(),
                        selector,
                        query,
                        request,
                    )
                },
            )?,
        )?;
    }
    {
        let manager = app_data.store_manager.clone();
        let embedding = app_data.embedding_provider.clone();
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
                    let request = memory_store_request_from_opts(lua, opts)?;
                    memory_store_result(
                        lua,
                        manager.clone(),
                        embedding.clone(),
                        selector,
                        content,
                        metadata_json,
                        request,
                    )
                },
            )?,
        )?;
    }
    {
        let manager = app_data.store_manager.clone();
        runtime_memory.set(
            "feedback",
            lua.create_function(
                move |lua, (memory_id, signal, ctx, opts): (String, Value, Table, Option<Table>)| {
                    let selector = table_to_selector(ctx)?;
                    let signal = memory_feedback_signal_from_value(signal)?;
                    let request = memory_feedback_request_from_opts(lua, opts)?;
                    memory_feedback_result(
                        lua,
                        manager.clone(),
                        selector,
                        memory_id,
                        signal,
                        request,
                    )
                },
            )?,
        )?;
    }
    {
        let manager = app_data.store_manager.clone();
        let embedding = app_data.embedding_provider.clone();
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
                    let request = memory_store_request_from_opts(lua, opts)?;
                    memory_correct_result(
                        lua,
                        manager.clone(),
                        embedding.clone(),
                        selector,
                        memory_id,
                        content,
                        metadata_json,
                        request,
                    )
                },
            )?,
        )?;
    }
    {
        let manager = app_data.store_manager.clone();
        runtime_memory.set(
            "purge",
            lua.create_function(move |lua, (ctx, opts): (Table, Option<Table>)| {
                let selector = table_to_selector(ctx)?;
                let request = memory_purge_request_from_opts(lua, opts)?;
                memory_purge_result(lua, manager.clone(), selector, request)
            })?,
        )?;
    }
    runtime_table.set("memory", runtime_memory)?;

    // runtime.kv.* canonical backend delegates
    let runtime_kv = lua.create_table()?;
    {
        let manager = app_data.store_manager.clone();
        runtime_kv.set(
            "get",
            lua.create_function(move |lua, (key, ctx): (String, Table)| {
                let selector = table_to_selector(ctx)?;
                kv_get_result(lua, manager.clone(), selector, key)
            })?,
        )?;
    }
    {
        let manager = app_data.store_manager.clone();
        runtime_kv.set(
            "set",
            lua.create_function(move |lua, (key, value, ctx): (String, String, Table)| {
                let selector = table_to_selector(ctx)?;
                kv_set_result(lua, manager.clone(), selector, key, value)
            })?,
        )?;
    }
    {
        let manager = app_data.store_manager.clone();
        runtime_kv.set(
            "delete",
            lua.create_function(move |lua, (key, ctx): (String, Table)| {
                let selector = table_to_selector(ctx)?;
                kv_delete_result(lua, manager.clone(), selector, key)
            })?,
        )?;
    }
    runtime_table.set("kv", runtime_kv)?;

    Ok(())
}
