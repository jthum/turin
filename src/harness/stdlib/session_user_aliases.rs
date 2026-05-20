use mlua::{Lua, Result as LuaResult, Table, Value};

use crate::harness::globals::HarnessAppData;
use crate::harness::stdlib::binding_common::{
    memory_feedback_request_from_opts, memory_feedback_signal_from_value,
    memory_purge_request_from_opts, memory_search_request_from_opt, memory_store_request_from_opts,
    metadata_json_or_empty, resolve_memory_search_request, resolve_scoped_store_selector,
    scoped_state_path_scope, scoped_state_path_scope_for_selectors, store_selector_from_opts_table,
};
use crate::harness::stdlib::context_selectors::selector_from_active_scope_lua;
use crate::harness::stdlib::memory_kv_bindings::{
    kv_delete_result, kv_get_result, kv_set_result, memory_correct_result, memory_feedback_result,
    memory_purge_result, memory_search_result, memory_store_result,
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
                    memory_purge_result(lua, manager.clone(), selector, request, path_scope)
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
