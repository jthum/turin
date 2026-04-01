use mlua::{Lua, Result as LuaResult, Table, Value};

use crate::harness::globals::HarnessAppData;
use crate::harness::stdlib::binding_common::{
    bool_err, bridge_async_result, memory_correction_row_to_lua_value,
    memory_feedback_request_from_opts, memory_feedback_signal_from_value,
    memory_feedback_state_to_lua_value, memory_purge_report_to_lua_value,
    memory_purge_request_from_opts, memory_rows_to_lua_table, memory_search_request_from_opt,
    memory_store_request_from_opts, memory_store_row_to_lua_value, metadata_json_or_empty, nil_err,
    nil_ok, ok_bool, ok_value, resolve_memory_search_request, resolve_scoped_store_selector,
    scoped_state_path_scope, scoped_state_path_scope_for_selectors, store_selector_from_opts_table,
    string_ok,
};
use crate::harness::stdlib::context_selectors::selector_from_active_scope_lua;
use crate::harness::stdlib::scoped_data_backend::{
    kv_delete_backend, kv_get_backend, kv_set_backend, memory_correct_backend_with_request,
    memory_feedback_backend_with_request, memory_purge_backend_with_request,
    memory_search_backend_with_request, memory_store_backend_with_request,
};

pub fn register_session_user_aliases(lua: &Lua, app_data: &HarnessAppData) -> LuaResult<()> {
    fn attach_alias_memory(
        lua: &Lua,
        t: &Table,
        app_data: &HarnessAppData,
        scope: &'static str,
    ) -> LuaResult<()> {
        let mem = lua.create_table()?;
        {
            let manager = app_data.store_manager.clone();
            let embedding = app_data.embedding_provider.clone();
            let selector_app = app_data.clone();
            mem.set(
                "search",
                lua.create_function(move |lua, (query, opts): (String, Option<Value>)| {
                    let selector = selector_from_active_scope_lua(&selector_app, scope)?;
                    let request = resolve_memory_search_request(
                        &selector_app,
                        &selector,
                        &memory_search_request_from_opt(lua, opts)?,
                    )?;
                    let path_scope = scoped_state_path_scope_for_selectors(
                        &selector_app,
                        request.store_selector.as_ref().into_iter().chain(
                            request
                                .sources
                                .iter()
                                .filter_map(|source| source.store_selector.as_ref()),
                        ),
                    )?;
                    let manager = manager.clone();
                    let embedding = embedding.clone();
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
                        Ok(rows) => {
                            Ok(ok_value(Value::Table(memory_rows_to_lua_table(lua, rows)?)))
                        }
                        Err(err) => nil_err(lua, &err),
                    }
                })?,
            )?;
        }
        {
            let manager = app_data.store_manager.clone();
            let embedding = app_data.embedding_provider.clone();
            let selector_app = app_data.clone();
            mem.set(
                "store",
                lua.create_function(
                    move |lua, (content, metadata, opts): (String, Option<Table>, Option<Table>)| {
                        let selector = selector_from_active_scope_lua(&selector_app, scope)?;
                        let metadata_json = metadata_json_or_empty(lua, metadata)?;
                        let mut request = memory_store_request_from_opts(lua, opts)?;
                        request.store_selector = resolve_scoped_store_selector(
                            &selector_app,
                            &selector,
                            request.store_selector.clone(),
                        )?;
                        let path_scope =
                            scoped_state_path_scope(&selector_app, request.store_selector.as_ref())?;
                        let manager = manager.clone();
                        let embedding = embedding.clone();
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
                    },
                )?,
                )?;
        }
        {
            let manager = app_data.store_manager.clone();
            let selector_app = app_data.clone();
            mem.set(
                "feedback",
                lua.create_function(
                    move |lua, (memory_id, signal, opts): (String, Value, Option<Table>)| {
                        let selector = selector_from_active_scope_lua(&selector_app, scope)?;
                        let signal = memory_feedback_signal_from_value(signal)?;
                        let mut request = memory_feedback_request_from_opts(lua, opts)?;
                        request.store_selector = resolve_scoped_store_selector(
                            &selector_app,
                            &selector,
                            request.store_selector.clone(),
                        )?;
                        let path_scope = scoped_state_path_scope(
                            &selector_app,
                            request.store_selector.as_ref(),
                        )?;
                        let manager = manager.clone();
                        let result = bridge_async_result(async move {
                            memory_feedback_backend_with_request(
                                &manager, &selector, &memory_id, signal, &request, path_scope,
                            )
                            .await
                            .map_err(|e| e.to_string())
                        });
                        match result {
                            Ok(state) => {
                                Ok(ok_value(memory_feedback_state_to_lua_value(lua, state)?))
                            }
                            Err(err) => nil_err(lua, &err),
                        }
                    },
                )?,
            )?;
        }
        {
            let manager = app_data.store_manager.clone();
            let embedding = app_data.embedding_provider.clone();
            let selector_app = app_data.clone();
            mem.set(
                "correct",
                lua.create_function(
                    move |lua,
                          (memory_id, content, metadata, opts): (
                        String,
                        String,
                        Option<Table>,
                        Option<Table>,
                    )| {
                        let selector = selector_from_active_scope_lua(&selector_app, scope)?;
                        let metadata_json = metadata_json_or_empty(lua, metadata)?;
                        let mut request = memory_store_request_from_opts(lua, opts)?;
                        request.store_selector = resolve_scoped_store_selector(
                            &selector_app,
                            &selector,
                            request.store_selector.clone(),
                        )?;
                        let path_scope = scoped_state_path_scope(
                            &selector_app,
                            request.store_selector.as_ref(),
                        )?;
                        let manager = manager.clone();
                        let embedding = embedding.clone();
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
                    },
                )?,
            )?;
        }
        {
            let manager = app_data.store_manager.clone();
            let selector_app = app_data.clone();
            mem.set(
                "purge",
                lua.create_function(move |lua, opts: Option<Table>| {
                    let selector = selector_from_active_scope_lua(&selector_app, scope)?;
                    let mut request = memory_purge_request_from_opts(lua, opts)?;
                    request.store_selector = resolve_scoped_store_selector(
                        &selector_app,
                        &selector,
                        request.store_selector.clone(),
                    )?;
                    let path_scope =
                        scoped_state_path_scope(&selector_app, request.store_selector.as_ref())?;
                    let manager = manager.clone();
                    let result = bridge_async_result(async move {
                        memory_purge_backend_with_request(&manager, &selector, &request, path_scope)
                            .await
                            .map_err(|e| e.to_string())
                    });
                    match result {
                        Ok(report) => Ok(ok_value(memory_purge_report_to_lua_value(lua, report)?)),
                        Err(err) => nil_err(lua, &err),
                    }
                })?,
            )?;
        }
        t.set("memory", mem)?;
        Ok(())
    }

    fn attach_alias_kv(
        lua: &Lua,
        t: &Table,
        app_data: &HarnessAppData,
        scope: &'static str,
    ) -> LuaResult<()> {
        let kv = lua.create_table()?;
        {
            let manager = app_data.store_manager.clone();
            let selector_app = app_data.clone();
            kv.set(
                "get",
                lua.create_function(move |lua, (key, opts): (String, Option<Table>)| {
                    let selector = selector_from_active_scope_lua(&selector_app, scope)?;
                    let store_selector = resolve_scoped_store_selector(
                        &selector_app,
                        &selector,
                        store_selector_from_opts_table(opts)?,
                    )?;
                    let path_scope =
                        scoped_state_path_scope(&selector_app, store_selector.as_ref())?;
                    let manager = manager.clone();
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
                })?,
            )?;
        }
        {
            let manager = app_data.store_manager.clone();
            let selector_app = app_data.clone();
            kv.set(
                "set",
                lua.create_function(
                    move |lua, (key, value, opts): (String, String, Option<Table>)| {
                        let selector = selector_from_active_scope_lua(&selector_app, scope)?;
                        let store_selector = resolve_scoped_store_selector(
                            &selector_app,
                            &selector,
                            store_selector_from_opts_table(opts)?,
                        )?;
                        let path_scope =
                            scoped_state_path_scope(&selector_app, store_selector.as_ref())?;
                        let manager = manager.clone();
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
                    },
                )?,
            )?;
        }
        {
            let manager = app_data.store_manager.clone();
            let selector_app = app_data.clone();
            kv.set(
                "delete",
                lua.create_function(move |lua, (key, opts): (String, Option<Table>)| {
                    let selector = selector_from_active_scope_lua(&selector_app, scope)?;
                    let store_selector = resolve_scoped_store_selector(
                        &selector_app,
                        &selector,
                        store_selector_from_opts_table(opts)?,
                    )?;
                    let path_scope =
                        scoped_state_path_scope(&selector_app, store_selector.as_ref())?;
                    let manager = manager.clone();
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
                })?,
            )?;
        }
        t.set("kv", kv)?;
        Ok(())
    }

    let session_table = lua.create_table()?;
    let user_table = lua.create_table()?;

    attach_alias_memory(lua, &session_table, app_data, "session")?;
    attach_alias_kv(lua, &session_table, app_data, "session")?;

    attach_alias_memory(lua, &user_table, app_data, "user")?;
    attach_alias_kv(lua, &user_table, app_data, "user")?;

    lua.globals().set("session", session_table)?;
    lua.globals().set("user", user_table)?;
    Ok(())
}
