use mlua::{Lua, Result as LuaResult, Table, Value};

use crate::harness::globals::{ActiveHarnessExecutionContext, HarnessAppData};
use crate::harness::stdlib::binding_common::{
    bool_err, bridge_async_result, memory_correction_row_to_lua_value,
    memory_feedback_request_from_opts, memory_feedback_signal_from_value,
    memory_feedback_state_to_lua_value, memory_purge_report_to_lua_value,
    memory_purge_request_from_opts, memory_rows_to_lua_table, memory_search_request_from_opt,
    memory_store_request_from_opts, memory_store_row_to_lua_value, metadata_json_or_empty, nil_err,
    nil_ok, ok_bool, ok_value, scoped_state_path_scope, store_selector_from_opts_table, string_ok,
};
use crate::harness::stdlib::context_selectors::{normalize_selector, table_to_selector};
use crate::harness::stdlib::scoped_data_backend::{
    MemoryFeedbackRequest, MemoryFeedbackSignal, MemoryPurgeRequest, MemorySearchRequest,
    MemoryStoreRequest, kv_delete_backend, kv_get_backend, kv_set_backend,
    memory_correct_backend_with_request, memory_feedback_backend_with_request,
    memory_purge_backend_with_request, memory_search_backend_with_request,
    memory_store_backend_with_request,
};
use crate::inference::embeddings::EmbeddingProvider;
use crate::kernel::identity::ContextSelector;
use crate::persistence::manager::StoreManager;
use std::sync::Arc;

fn default_agent_selector(app_data: &HarnessAppData) -> LuaResult<ContextSelector> {
    normalize_selector(ContextSelector {
        tags: vec![format!("agent:{}", app_data.config.agent.id)],
        namespace: "default".to_string(),
        visibility: "private".to_string(),
    })
    .map_err(mlua::Error::runtime)
}

fn has_active_session(execution_ctx: &ActiveHarnessExecutionContext) -> bool {
    execution_ctx
        .lock()
        .map(|lock| lock.session_id.is_some())
        .unwrap_or(false)
}

fn memory_search_result(
    lua: &Lua,
    manager: Arc<StoreManager>,
    embedding: Option<Arc<dyn EmbeddingProvider>>,
    selector: ContextSelector,
    query: String,
    request: MemorySearchRequest,
    path_scope: crate::persistence::manager::StorePathScope,
) -> LuaResult<(Value, Value)> {
    let result = bridge_async_result(async move {
        memory_search_backend_with_request(
            &manager,
            embedding.as_ref(),
            &selector,
            &query,
            &request,
            path_scope,
        )
        .await
        .map_err(|e| e.to_string())
    });
    match result {
        Ok(rows) => Ok(ok_value(Value::Table(memory_rows_to_lua_table(lua, rows)?))),
        Err(err) => nil_err(lua, &err),
    }
}

#[allow(clippy::too_many_arguments)]
fn memory_store_result(
    lua: &Lua,
    manager: Arc<StoreManager>,
    embedding: Option<Arc<dyn EmbeddingProvider>>,
    selector: ContextSelector,
    content: String,
    metadata_json: serde_json::Value,
    request: MemoryStoreRequest,
    path_scope: crate::persistence::manager::StorePathScope,
) -> LuaResult<(Value, Value)> {
    let result = bridge_async_result(async move {
        memory_store_backend_with_request(
            &manager,
            embedding.as_ref(),
            &selector,
            &content,
            &metadata_json,
            &request,
            path_scope,
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
    signal: MemoryFeedbackSignal,
    request: MemoryFeedbackRequest,
    path_scope: crate::persistence::manager::StorePathScope,
) -> LuaResult<(Value, Value)> {
    let result = bridge_async_result(async move {
        memory_feedback_backend_with_request(
            &manager, &selector, &memory_id, signal, &request, path_scope,
        )
        .await
        .map_err(|e| e.to_string())
    });
    match result {
        Ok(state) => Ok(ok_value(memory_feedback_state_to_lua_value(lua, state)?)),
        Err(err) => nil_err(lua, &err),
    }
}

#[allow(clippy::too_many_arguments)]
fn memory_correct_result(
    lua: &Lua,
    manager: Arc<StoreManager>,
    embedding: Option<Arc<dyn EmbeddingProvider>>,
    selector: ContextSelector,
    memory_id: String,
    content: String,
    metadata_json: serde_json::Value,
    request: MemoryStoreRequest,
    path_scope: crate::persistence::manager::StorePathScope,
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
            path_scope,
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
    request: MemoryPurgeRequest,
    path_scope: crate::persistence::manager::StorePathScope,
) -> LuaResult<(Value, Value)> {
    let result = bridge_async_result(async move {
        memory_purge_backend_with_request(&manager, &selector, &request, path_scope)
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
    store_selector: Option<crate::persistence::manager::StoreSelector>,
    path_scope: crate::persistence::manager::StorePathScope,
) -> LuaResult<(Value, Value)> {
    let result = bridge_async_result(async move {
        kv_get_backend(
            &manager,
            &selector,
            &key,
            store_selector.as_ref(),
            path_scope,
        )
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
    store_selector: Option<crate::persistence::manager::StoreSelector>,
    path_scope: crate::persistence::manager::StorePathScope,
) -> LuaResult<(Value, Value)> {
    let result = bridge_async_result(async move {
        kv_set_backend(
            &manager,
            &selector,
            &key,
            &value,
            store_selector.as_ref(),
            path_scope,
        )
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
    store_selector: Option<crate::persistence::manager::StoreSelector>,
    path_scope: crate::persistence::manager::StorePathScope,
) -> LuaResult<(Value, Value)> {
    let result = bridge_async_result(async move {
        kv_delete_backend(
            &manager,
            &selector,
            &key,
            store_selector.as_ref(),
            path_scope,
        )
        .await
        .map_err(|e| e.to_string())
    });
    match result {
        Ok(_) => Ok(ok_bool()),
        Err(err) => bool_err(lua, &err),
    }
}

pub fn register_memory_module(lua: &Lua, app_data: &HarnessAppData) -> LuaResult<()> {
    let memory_table = lua.create_table()?;

    // memory.search -> agent default selector
    {
        let manager = app_data.store_manager.clone();
        let embedding = app_data.embedding_provider.clone();
        let app_data_snapshot = app_data.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        memory_table.set(
            "search",
            lua.create_function(move |lua, (query, opts): (String, Option<Value>)| {
                let request = memory_search_request_from_opt(lua, opts)?;
                let selector = default_agent_selector(&app_data_snapshot)?;
                if !has_active_session(&execution_ctx) {
                    return nil_err(lua, "No active session context");
                }
                let path_scope =
                    scoped_state_path_scope(&app_data_snapshot, request.store_selector.as_ref())?;
                memory_search_result(
                    lua,
                    manager.clone(),
                    embedding.clone(),
                    selector,
                    query,
                    request,
                    path_scope,
                )
            })?,
        )?;
    }

    // memory.store -> agent default selector
    {
        let manager = app_data.store_manager.clone();
        let embedding = app_data.embedding_provider.clone();
        let app_data_snapshot = app_data.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        memory_table.set(
            "store",
            lua.create_function(
                move |lua, (content, metadata, opts): (String, Option<Table>, Option<Table>)| {
                    let selector = default_agent_selector(&app_data_snapshot)?;
                    if !has_active_session(&execution_ctx) {
                        return nil_err(lua, "No active session context");
                    }
                    let metadata_json = metadata_json_or_empty(lua, metadata)?;
                    let request = memory_store_request_from_opts(lua, opts)?;
                    let path_scope = scoped_state_path_scope(
                        &app_data_snapshot,
                        request.store_selector.as_ref(),
                    )?;
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

    // memory.as(ctx) proxy
    {
        let manager = app_data.store_manager.clone();
        let embedding = app_data.embedding_provider.clone();
        let app_data_snapshot = app_data.clone();
        memory_table.set(
            "as",
            lua.create_function(move |lua, ctx: Table| {
                let selector = table_to_selector(ctx)?;
                let proxy = lua.create_table()?;

                let sel_search = selector.clone();
                let m_search = manager.clone();
                let e_search = embedding.clone();
                let app_data_search = app_data_snapshot.clone();
                proxy.set(
                    "search",
                    lua.create_function(move |lua, (query, opts): (String, Option<Value>)| {
                        let request = memory_search_request_from_opt(lua, opts)?;
                        let path_scope =
                            scoped_state_path_scope(&app_data_search, request.store_selector.as_ref())?;
                        memory_search_result(
                            lua,
                            m_search.clone(),
                            e_search.clone(),
                            sel_search.clone(),
                            query,
                            request,
                            path_scope,
                        )
                    })?,
                )?;

                let sel_store = selector.clone();
                let m_store = manager.clone();
                let e_store = embedding.clone();
                let app_data_store = app_data_snapshot.clone();
                proxy.set(
                    "store",
                    lua.create_function(
                        move |lua, (content, metadata, opts): (String, Option<Table>, Option<Table>)| {
                            let metadata_json = metadata_json_or_empty(lua, metadata)?;
                            let request = memory_store_request_from_opts(lua, opts)?;
                            let path_scope =
                                scoped_state_path_scope(&app_data_store, request.store_selector.as_ref())?;
                            memory_store_result(
                                lua,
                                m_store.clone(),
                                e_store.clone(),
                                sel_store.clone(),
                                content,
                                metadata_json,
                                request,
                                path_scope,
                            )
                        },
                    )?,
                )?;

                let sel_feedback = selector.clone();
                let m_feedback = manager.clone();
                let app_data_feedback = app_data_snapshot.clone();
                proxy.set(
                    "feedback",
                    lua.create_function(
                        move |lua, (memory_id, signal, opts): (String, Value, Option<Table>)| {
                            let signal = memory_feedback_signal_from_value(signal)?;
                            let request = memory_feedback_request_from_opts(lua, opts)?;
                            let path_scope =
                                scoped_state_path_scope(&app_data_feedback, request.store_selector.as_ref())?;
                            memory_feedback_result(
                                lua,
                                m_feedback.clone(),
                                sel_feedback.clone(),
                                memory_id,
                                signal,
                                request,
                                path_scope,
                            )
                        },
                    )?,
                )?;

                let sel_correct = selector.clone();
                let m_correct = manager.clone();
                let e_correct = embedding.clone();
                let app_data_correct = app_data_snapshot.clone();
                proxy.set(
                    "correct",
                    lua.create_function(
                        move |lua,
                              (memory_id, content, metadata, opts): (
                            String,
                            String,
                            Option<Table>,
                            Option<Table>,
                        )| {
                            let metadata_json = metadata_json_or_empty(lua, metadata)?;
                            let request = memory_store_request_from_opts(lua, opts)?;
                            let path_scope =
                                scoped_state_path_scope(&app_data_correct, request.store_selector.as_ref())?;
                            memory_correct_result(
                                lua,
                                m_correct.clone(),
                                e_correct.clone(),
                                sel_correct.clone(),
                                memory_id,
                                content,
                                metadata_json,
                                request,
                                path_scope,
                            )
                        },
                    )?,
                )?;

                let sel_purge = selector.clone();
                let m_purge = manager.clone();
                let app_data_purge = app_data_snapshot.clone();
                proxy.set(
                    "purge",
                    lua.create_function(move |lua, opts: Option<Table>| {
                        let request = memory_purge_request_from_opts(lua, opts)?;
                        let path_scope =
                            scoped_state_path_scope(&app_data_purge, request.store_selector.as_ref())?;
                        memory_purge_result(
                            lua,
                            m_purge.clone(),
                            sel_purge.clone(),
                            request,
                            path_scope,
                        )
                    })?,
                )?;

                Ok(proxy)
            })?,
        )?;
    }

    // memory.feedback -> agent default selector
    {
        let manager = app_data.store_manager.clone();
        let app_data_snapshot = app_data.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        memory_table.set(
            "feedback",
            lua.create_function(
                move |lua, (memory_id, signal, opts): (String, Value, Option<Table>)| {
                    let selector = default_agent_selector(&app_data_snapshot)?;
                    if !has_active_session(&execution_ctx) {
                        return nil_err(lua, "No active session context");
                    }
                    let signal = memory_feedback_signal_from_value(signal)?;
                    let request = memory_feedback_request_from_opts(lua, opts)?;
                    let path_scope = scoped_state_path_scope(
                        &app_data_snapshot,
                        request.store_selector.as_ref(),
                    )?;
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

    // memory.correct -> agent default selector
    {
        let manager = app_data.store_manager.clone();
        let embedding = app_data.embedding_provider.clone();
        let app_data_snapshot = app_data.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        memory_table.set(
            "correct",
            lua.create_function(
                move |lua,
                      (memory_id, content, metadata, opts): (
                    String,
                    String,
                    Option<Table>,
                    Option<Table>,
                )| {
                    let selector = default_agent_selector(&app_data_snapshot)?;
                    if !has_active_session(&execution_ctx) {
                        return nil_err(lua, "No active session context");
                    }
                    let metadata_json = metadata_json_or_empty(lua, metadata)?;
                    let request = memory_store_request_from_opts(lua, opts)?;
                    let path_scope = scoped_state_path_scope(
                        &app_data_snapshot,
                        request.store_selector.as_ref(),
                    )?;
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

    // memory.purge -> agent default selector
    {
        let manager = app_data.store_manager.clone();
        let app_data_snapshot = app_data.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        memory_table.set(
            "purge",
            lua.create_function(move |lua, opts: Option<Table>| {
                let selector = default_agent_selector(&app_data_snapshot)?;
                if !has_active_session(&execution_ctx) {
                    return nil_err(lua, "No active session context");
                }
                let request = memory_purge_request_from_opts(lua, opts)?;
                let path_scope =
                    scoped_state_path_scope(&app_data_snapshot, request.store_selector.as_ref())?;
                memory_purge_result(lua, manager.clone(), selector, request, path_scope)
            })?,
        )?;
    }

    lua.globals().set("memory", memory_table)?;
    Ok(())
}

pub fn register_kv_module(lua: &Lua, app_data: &HarnessAppData) -> LuaResult<()> {
    let kv_table = lua.create_table()?;

    // kv.get
    {
        let manager = app_data.store_manager.clone();
        let app_data_snapshot = app_data.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        kv_table.set(
            "get",
            lua.create_function(move |lua, (key, opts): (String, Option<Table>)| {
                if !has_active_session(&execution_ctx) {
                    return nil_err(lua, "No active session context");
                }
                let selector = default_agent_selector(&app_data_snapshot)?;
                let store_selector = store_selector_from_opts_table(opts)?;
                let path_scope =
                    scoped_state_path_scope(&app_data_snapshot, store_selector.as_ref())?;
                kv_get_result(
                    lua,
                    manager.clone(),
                    selector,
                    key,
                    store_selector,
                    path_scope,
                )
            })?,
        )?;
    }

    // kv.set
    {
        let manager = app_data.store_manager.clone();
        let app_data_snapshot = app_data.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        kv_table.set(
            "set",
            lua.create_function(
                move |lua, (key, value, opts): (String, String, Option<Table>)| {
                    if !has_active_session(&execution_ctx) {
                        return bool_err(lua, "No active session context");
                    }
                    let selector = default_agent_selector(&app_data_snapshot)?;
                    let store_selector = store_selector_from_opts_table(opts)?;
                    let path_scope =
                        scoped_state_path_scope(&app_data_snapshot, store_selector.as_ref())?;
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

    // kv.delete
    {
        let manager = app_data.store_manager.clone();
        let app_data_snapshot = app_data.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        kv_table.set(
            "delete",
            lua.create_function(move |lua, (key, opts): (String, Option<Table>)| {
                if !has_active_session(&execution_ctx) {
                    return bool_err(lua, "No active session context");
                }
                let selector = default_agent_selector(&app_data_snapshot)?;
                let store_selector = store_selector_from_opts_table(opts)?;
                let path_scope =
                    scoped_state_path_scope(&app_data_snapshot, store_selector.as_ref())?;
                kv_delete_result(
                    lua,
                    manager.clone(),
                    selector,
                    key,
                    store_selector,
                    path_scope,
                )
            })?,
        )?;
    }

    // kv.as(ctx) proxy
    {
        let manager = app_data.store_manager.clone();
        let app_data_snapshot = app_data.clone();
        kv_table.set(
            "as",
            lua.create_function(move |lua, ctx: Table| {
                let selector = table_to_selector(ctx)?;
                let proxy = lua.create_table()?;

                let sel_get = selector.clone();
                let m_get = manager.clone();
                let app_data_get = app_data_snapshot.clone();
                proxy.set(
                    "get",
                    lua.create_function(move |lua, (key, opts): (String, Option<Table>)| {
                        let store_selector = store_selector_from_opts_table(opts)?;
                        let path_scope =
                            scoped_state_path_scope(&app_data_get, store_selector.as_ref())?;
                        kv_get_result(
                            lua,
                            m_get.clone(),
                            sel_get.clone(),
                            key,
                            store_selector,
                            path_scope,
                        )
                    })?,
                )?;

                let sel_set = selector.clone();
                let m_set = manager.clone();
                let app_data_set = app_data_snapshot.clone();
                proxy.set(
                    "set",
                    lua.create_function(
                        move |lua, (key, value, opts): (String, String, Option<Table>)| {
                            let store_selector = store_selector_from_opts_table(opts)?;
                            let path_scope =
                                scoped_state_path_scope(&app_data_set, store_selector.as_ref())?;
                            kv_set_result(
                                lua,
                                m_set.clone(),
                                sel_set.clone(),
                                key,
                                value,
                                store_selector,
                                path_scope,
                            )
                        },
                    )?,
                )?;

                let sel_del = selector.clone();
                let m_del = manager.clone();
                let app_data_delete = app_data_snapshot.clone();
                proxy.set(
                    "delete",
                    lua.create_function(move |lua, (key, opts): (String, Option<Table>)| {
                        let store_selector = store_selector_from_opts_table(opts)?;
                        let path_scope =
                            scoped_state_path_scope(&app_data_delete, store_selector.as_ref())?;
                        kv_delete_result(
                            lua,
                            m_del.clone(),
                            sel_del.clone(),
                            key,
                            store_selector,
                            path_scope,
                        )
                    })?,
                )?;

                Ok(proxy)
            })?,
        )?;
    }

    lua.globals().set("kv", kv_table)?;
    Ok(())
}
