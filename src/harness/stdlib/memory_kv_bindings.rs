use mlua::{Lua, Result as LuaResult, Table, Value};

use crate::harness::globals::{ActiveHarnessExecutionContext, HarnessAppData};
use crate::harness::stdlib::binding_common::{
    bool_err, bridge_async_result, memory_rows_to_lua_table, metadata_json_or_empty, nil_err,
    nil_ok, ok_bool, ok_value, string_ok,
};
use crate::harness::stdlib::context_selectors::{
    normalize_selector, search_limit_from_opt, table_to_selector,
};
use crate::harness::stdlib::scoped_data_backend::{
    kv_delete_backend, kv_get_backend, kv_set_backend, memory_search_backend, memory_store_backend,
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
    limit: usize,
) -> LuaResult<(Value, Value)> {
    let result = bridge_async_result(async move {
        memory_search_backend(&manager, embedding.as_ref(), &selector, &query, limit)
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
) -> LuaResult<(Value, Value)> {
    let result = bridge_async_result(async move {
        memory_store_backend(
            &manager,
            embedding.as_ref(),
            &selector,
            &content,
            &metadata_json,
        )
        .await
        .map_err(|e| e.to_string())
    });
    match result {
        Ok(_) => Ok(ok_bool()),
        Err(err) => bool_err(lua, &err),
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
                let limit = search_limit_from_opt(opts)?;
                let selector = default_agent_selector(&app_data_snapshot)?;
                if !has_active_session(&execution_ctx) {
                    return nil_err(lua, "No active session context");
                }
                memory_search_result(
                    lua,
                    manager.clone(),
                    embedding.clone(),
                    selector,
                    query,
                    limit,
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
                move |lua, (content, metadata, _opts): (String, Option<Table>, Option<Table>)| {
                    let selector = default_agent_selector(&app_data_snapshot)?;
                    if !has_active_session(&execution_ctx) {
                        return bool_err(lua, "No active session context");
                    }
                    let metadata_json = metadata_json_or_empty(lua, metadata)?;
                    memory_store_result(
                        lua,
                        manager.clone(),
                        embedding.clone(),
                        selector,
                        content,
                        metadata_json,
                    )
                },
            )?,
        )?;
    }

    // memory.as(ctx) proxy
    {
        let manager = app_data.store_manager.clone();
        let embedding = app_data.embedding_provider.clone();
        memory_table.set(
            "as",
            lua.create_function(move |lua, ctx: Table| {
                let selector = table_to_selector(ctx)?;
                let proxy = lua.create_table()?;

                let sel_search = selector.clone();
                let m_search = manager.clone();
                let e_search = embedding.clone();
                proxy.set(
                    "search",
                    lua.create_function(move |lua, (query, opts): (String, Option<Value>)| {
                        let limit = search_limit_from_opt(opts)?;
                        memory_search_result(
                            lua,
                            m_search.clone(),
                            e_search.clone(),
                            sel_search.clone(),
                            query,
                            limit,
                        )
                    })?,
                )?;

                let sel_store = selector.clone();
                let m_store = manager.clone();
                let e_store = embedding.clone();
                proxy.set(
                    "store",
                    lua.create_function(
                        move |lua,
                              (content, metadata, _opts): (
                            String,
                            Option<Table>,
                            Option<Table>,
                        )| {
                            let metadata_json = metadata_json_or_empty(lua, metadata)?;
                            memory_store_result(
                                lua,
                                m_store.clone(),
                                e_store.clone(),
                                sel_store.clone(),
                                content,
                                metadata_json,
                            )
                        },
                    )?,
                )?;

                Ok(proxy)
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
            lua.create_function(move |lua, key: String| {
                if !has_active_session(&execution_ctx) {
                    return nil_err(lua, "No active session context");
                }
                let selector = default_agent_selector(&app_data_snapshot)?;
                kv_get_result(lua, manager.clone(), selector, key)
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
            lua.create_function(move |lua, (key, value): (String, String)| {
                if !has_active_session(&execution_ctx) {
                    return bool_err(lua, "No active session context");
                }
                let selector = default_agent_selector(&app_data_snapshot)?;
                kv_set_result(lua, manager.clone(), selector, key, value)
            })?,
        )?;
    }

    // kv.delete
    {
        let manager = app_data.store_manager.clone();
        let app_data_snapshot = app_data.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        kv_table.set(
            "delete",
            lua.create_function(move |lua, key: String| {
                if !has_active_session(&execution_ctx) {
                    return bool_err(lua, "No active session context");
                }
                let selector = default_agent_selector(&app_data_snapshot)?;
                kv_delete_result(lua, manager.clone(), selector, key)
            })?,
        )?;
    }

    // kv.as(ctx) proxy
    {
        let manager = app_data.store_manager.clone();
        kv_table.set(
            "as",
            lua.create_function(move |lua, ctx: Table| {
                let selector = table_to_selector(ctx)?;
                let proxy = lua.create_table()?;

                let sel_get = selector.clone();
                let m_get = manager.clone();
                proxy.set(
                    "get",
                    lua.create_function(move |lua, key: String| {
                        kv_get_result(lua, m_get.clone(), sel_get.clone(), key)
                    })?,
                )?;

                let sel_set = selector.clone();
                let m_set = manager.clone();
                proxy.set(
                    "set",
                    lua.create_function(move |lua, (key, value): (String, String)| {
                        kv_set_result(lua, m_set.clone(), sel_set.clone(), key, value)
                    })?,
                )?;

                let sel_del = selector.clone();
                let m_del = manager.clone();
                proxy.set(
                    "delete",
                    lua.create_function(move |lua, key: String| {
                        kv_delete_result(lua, m_del.clone(), sel_del.clone(), key)
                    })?,
                )?;

                Ok(proxy)
            })?,
        )?;
    }

    lua.globals().set("kv", kv_table)?;
    Ok(())
}
