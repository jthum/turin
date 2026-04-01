mod kv;
mod memory;

pub use kv::register_kv_module;
pub use memory::register_memory_module;

use mlua::{Lua, Result as LuaResult, Value};

use crate::harness::globals::{ActiveHarnessExecutionContext, HarnessAppData};
use crate::harness::stdlib::binding_common::{
    bool_err, bridge_async_result, memory_correction_row_to_lua_value,
    memory_feedback_state_to_lua_value, memory_purge_report_to_lua_value,
    memory_rows_to_lua_table, memory_store_row_to_lua_value, nil_err, nil_ok, ok_bool, ok_value,
    string_ok,
};
use crate::harness::stdlib::scoped_data_backend::{
    MemoryFeedbackRequest, MemoryFeedbackSignal, MemoryPurgeRequest, MemorySearchRequest,
    MemoryStoreRequest, kv_delete_backend, kv_get_backend, kv_set_backend,
    memory_correct_backend_with_request, memory_feedback_backend_with_request,
    memory_purge_backend_with_request, memory_search_backend_with_request,
    memory_store_backend_with_request,
};
use crate::inference::embeddings::EmbeddingProvider;
use crate::kernel::identity::ContextSelector;
use crate::persistence::manager::{StoreManager, StorePathScope, StoreSelector};
use std::sync::Arc;

fn default_agent_selector(app_data: &HarnessAppData) -> LuaResult<ContextSelector> {
    super::context_selectors::normalize_selector(ContextSelector {
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
    path_scope: StorePathScope,
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
    path_scope: StorePathScope,
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
    path_scope: StorePathScope,
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
    path_scope: StorePathScope,
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
    path_scope: StorePathScope,
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
    store_selector: Option<StoreSelector>,
    path_scope: StorePathScope,
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
    store_selector: Option<StoreSelector>,
    path_scope: StorePathScope,
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
    store_selector: Option<StoreSelector>,
    path_scope: StorePathScope,
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
