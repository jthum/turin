use mlua::{Lua, Result as LuaResult, Table, Value};

use super::{
    default_agent_selector, has_active_session, memory_correct_result, memory_feedback_result,
    memory_purge_result, memory_search_result, memory_store_result,
};
use crate::harness::globals::HarnessAppData;
use crate::harness::stdlib::binding_common::{
    memory_feedback_request_from_opts, memory_feedback_signal_from_value,
    memory_purge_request_from_opts, memory_search_request_from_opt, memory_store_request_from_opts,
    metadata_json_or_empty, nil_err, resolve_memory_search_request, resolve_scoped_store_selector,
    scoped_state_path_scope, scoped_state_path_scope_for_selectors,
};
use crate::harness::stdlib::context_selectors::table_to_selector;

pub fn register_memory_module(lua: &Lua, app_data: &HarnessAppData) -> LuaResult<()> {
    let memory_table = lua.create_table()?;

    {
        let manager = app_data.store_manager.clone();
        let embedding = app_data.embedding_provider.clone();
        let app_data_snapshot = app_data.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        memory_table.set(
            "search",
            lua.create_function(move |lua, (query, opts): (String, Option<Value>)| {
                let selector = default_agent_selector(&app_data_snapshot)?;
                let request = resolve_memory_search_request(
                    &app_data_snapshot,
                    &selector,
                    &memory_search_request_from_opt(lua, opts)?,
                )?;
                if !has_active_session(&execution_ctx) {
                    return nil_err(lua, "No active session context");
                }
                let path_scope = scoped_state_path_scope_for_selectors(
                    &app_data_snapshot,
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
                    let mut request = memory_store_request_from_opts(lua, opts)?;
                    request.store_selector = resolve_scoped_store_selector(
                        &app_data_snapshot,
                        &selector,
                        request.store_selector.clone(),
                    )?;
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
                        let request = resolve_memory_search_request(
                            &app_data_search,
                            &sel_search,
                            &memory_search_request_from_opt(lua, opts)?,
                        )?;
                        let path_scope = scoped_state_path_scope_for_selectors(
                            &app_data_search,
                            request.store_selector.as_ref().into_iter().chain(
                                request
                                    .sources
                                    .iter()
                                    .filter_map(|source| source.store_selector.as_ref()),
                            ),
                        )?;
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
                            let mut request = memory_store_request_from_opts(lua, opts)?;
                            request.store_selector = resolve_scoped_store_selector(
                                &app_data_store,
                                &sel_store,
                                request.store_selector.clone(),
                            )?;
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
                            let mut request = memory_feedback_request_from_opts(lua, opts)?;
                            request.store_selector = resolve_scoped_store_selector(
                                &app_data_feedback,
                                &sel_feedback,
                                request.store_selector.clone(),
                            )?;
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
                            let mut request = memory_store_request_from_opts(lua, opts)?;
                            request.store_selector = resolve_scoped_store_selector(
                                &app_data_correct,
                                &sel_correct,
                                request.store_selector.clone(),
                            )?;
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
                        let mut request = memory_purge_request_from_opts(lua, opts)?;
                        request.store_selector = resolve_scoped_store_selector(
                            &app_data_purge,
                            &sel_purge,
                            request.store_selector.clone(),
                        )?;
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
                    let mut request = memory_feedback_request_from_opts(lua, opts)?;
                    request.store_selector = resolve_scoped_store_selector(
                        &app_data_snapshot,
                        &selector,
                        request.store_selector.clone(),
                    )?;
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
                    let mut request = memory_store_request_from_opts(lua, opts)?;
                    request.store_selector = resolve_scoped_store_selector(
                        &app_data_snapshot,
                        &selector,
                        request.store_selector.clone(),
                    )?;
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
                let mut request = memory_purge_request_from_opts(lua, opts)?;
                request.store_selector = resolve_scoped_store_selector(
                    &app_data_snapshot,
                    &selector,
                    request.store_selector.clone(),
                )?;
                let path_scope =
                    scoped_state_path_scope(&app_data_snapshot, request.store_selector.as_ref())?;
                memory_purge_result(lua, manager.clone(), selector, request, path_scope)
            })?,
        )?;
    }

    lua.globals().set("memory", memory_table)?;
    Ok(())
}
