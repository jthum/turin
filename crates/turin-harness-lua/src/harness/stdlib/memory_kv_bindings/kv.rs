use mlua::{Lua, Result as LuaResult, Table};

use super::{
    default_agent_selector, has_active_session, kv_delete_result, kv_get_result, kv_set_result,
};
use crate::harness::globals::HarnessAppData;
use crate::harness::stdlib::binding_common::{
    bool_err, nil_err, resolve_scoped_store_and_path_scope, store_selector_from_opts_table,
};
use crate::harness::stdlib::context_selectors::table_to_selector;

pub fn register_kv_module(lua: &Lua, app_data: &HarnessAppData) -> LuaResult<()> {
    let kv_table = lua.create_table()?;

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
            })?,
        )?;
    }

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
        let execution_ctx = app_data.execution_ctx.clone();
        kv_table.set(
            "delete",
            lua.create_function(move |lua, (key, opts): (String, Option<Table>)| {
                if !has_active_session(&execution_ctx) {
                    return bool_err(lua, "No active session context");
                }
                let selector = default_agent_selector(&app_data_snapshot)?;
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
            })?,
        )?;
    }

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
                        let (store_selector, path_scope) = resolve_scoped_store_and_path_scope(
                            &app_data_get,
                            &sel_get,
                            store_selector_from_opts_table(opts)?,
                        )?;
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
                            let (store_selector, path_scope) = resolve_scoped_store_and_path_scope(
                                &app_data_set,
                                &sel_set,
                                store_selector_from_opts_table(opts)?,
                            )?;
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
                        let (store_selector, path_scope) = resolve_scoped_store_and_path_scope(
                            &app_data_delete,
                            &sel_del,
                            store_selector_from_opts_table(opts)?,
                        )?;
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
