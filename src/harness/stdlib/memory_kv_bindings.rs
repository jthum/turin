use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};

use crate::harness::globals::{
    HarnessAppData, block_on_current, kv_delete_backend, kv_get_backend, kv_set_backend,
    memory_search_backend, memory_store_backend, normalize_selector, search_limit_from_opt,
    table_to_selector,
};
use crate::harness::stdlib::binding_common::memory_rows_to_lua_table;
use crate::kernel::identity::ContextSelector;

fn default_agent_selector(app_data: &HarnessAppData) -> LuaResult<ContextSelector> {
    normalize_selector(ContextSelector {
        tags: vec![format!("agent:{}", app_data.config.agent.id)],
        namespace: "default".to_string(),
        visibility: "private".to_string(),
    })
    .map_err(mlua::Error::runtime)
}

pub fn register_memory_module(lua: &Lua, app_data: &HarnessAppData) -> LuaResult<()> {
    let memory_table = lua.create_table()?;

    // memory.search -> agent default selector
    {
        let manager = app_data.store_manager.clone();
        let embedding = app_data.embedding_provider.clone();
        let app_data_snapshot = app_data.clone();
        let active_session = app_data.active_session_id.clone();
        memory_table.set(
            "search",
            lua.create_function(move |lua, (query, opts): (String, Option<Value>)| {
                let limit = search_limit_from_opt(opts)?;
                let selector = default_agent_selector(&app_data_snapshot)?;
                if active_session.lock().unwrap().is_none() {
                    return Ok((
                        Value::Nil,
                        Value::String(lua.create_string("No active session context")?),
                    ));
                }
                let manager = manager.clone();
                let embedding = embedding.clone();
                let result = block_on_current(async move {
                    memory_search_backend(&manager, embedding.as_ref(), &selector, &query, limit)
                        .await
                        .map_err(|e| e.to_string())
                });
                match result {
                    Ok(rows) => Ok((
                        Value::Table(memory_rows_to_lua_table(lua, rows)?),
                        Value::Nil,
                    )),
                    Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                }
            })?,
        )?;
    }

    // memory.store -> agent default selector
    {
        let manager = app_data.store_manager.clone();
        let embedding = app_data.embedding_provider.clone();
        let app_data_snapshot = app_data.clone();
        let active_session = app_data.active_session_id.clone();
        memory_table.set(
            "store",
            lua.create_function(
                move |lua, (content, metadata, _opts): (String, Option<Table>, Option<Table>)| {
                    let selector = default_agent_selector(&app_data_snapshot)?;
                    if active_session.lock().unwrap().is_none() {
                        return Ok((
                            Value::Boolean(false),
                            Value::String(lua.create_string("No active session context")?),
                        ));
                    }
                    let metadata_json = if let Some(tbl) = metadata {
                        lua.from_value::<serde_json::Value>(Value::Table(tbl))
                            .map_err(|e| {
                                mlua::Error::runtime(format!("invalid metadata table: {}", e))
                            })?
                    } else {
                        serde_json::json!({})
                    };
                    let manager = manager.clone();
                    let embedding = embedding.clone();
                    let result = block_on_current(async move {
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
                        Ok(_) => Ok((Value::Boolean(true), Value::Nil)),
                        Err(err) => Ok((
                            Value::Boolean(false),
                            Value::String(lua.create_string(&err)?),
                        )),
                    }
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
                        let selector = sel_search.clone();
                        let manager = m_search.clone();
                        let embedding = e_search.clone();
                        let result = block_on_current(async move {
                            memory_search_backend(
                                &manager,
                                embedding.as_ref(),
                                &selector,
                                &query,
                                limit,
                            )
                            .await
                            .map_err(|e| e.to_string())
                        });
                        match result {
                            Ok(rows) => Ok((
                                Value::Table(memory_rows_to_lua_table(lua, rows)?),
                                Value::Nil,
                            )),
                            Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                        }
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
                            let metadata_json = if let Some(tbl) = metadata {
                                lua.from_value::<serde_json::Value>(Value::Table(tbl))
                                    .map_err(|e| {
                                        mlua::Error::runtime(format!(
                                            "invalid metadata table: {}",
                                            e
                                        ))
                                    })?
                            } else {
                                serde_json::json!({})
                            };
                            let selector = sel_store.clone();
                            let manager = m_store.clone();
                            let embedding = e_store.clone();
                            let result = block_on_current(async move {
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
                                Ok(_) => Ok((Value::Boolean(true), Value::Nil)),
                                Err(err) => Ok((
                                    Value::Boolean(false),
                                    Value::String(lua.create_string(&err)?),
                                )),
                            }
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
        let active_session = app_data.active_session_id.clone();
        kv_table.set(
            "get",
            lua.create_function(move |lua, key: String| {
                if active_session.lock().unwrap().is_none() {
                    return Ok((
                        Value::Nil,
                        Value::String(lua.create_string("No active session context")?),
                    ));
                }
                let selector = default_agent_selector(&app_data_snapshot)?;
                let manager = manager.clone();
                let result = block_on_current(async move {
                    kv_get_backend(&manager, &selector, &key)
                        .await
                        .map_err(|e| e.to_string())
                });
                match result {
                    Ok(Some(val)) => Ok((Value::String(lua.create_string(&val)?), Value::Nil)),
                    Ok(None) => Ok((Value::Nil, Value::Nil)),
                    Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                }
            })?,
        )?;
    }

    // kv.set
    {
        let manager = app_data.store_manager.clone();
        let app_data_snapshot = app_data.clone();
        let active_session = app_data.active_session_id.clone();
        kv_table.set(
            "set",
            lua.create_function(move |lua, (key, value): (String, String)| {
                if active_session.lock().unwrap().is_none() {
                    return Ok((
                        Value::Boolean(false),
                        Value::String(lua.create_string("No active session context")?),
                    ));
                }
                let selector = default_agent_selector(&app_data_snapshot)?;
                let manager = manager.clone();
                let result = block_on_current(async move {
                    kv_set_backend(&manager, &selector, &key, &value)
                        .await
                        .map_err(|e| e.to_string())
                });
                match result {
                    Ok(_) => Ok((Value::Boolean(true), Value::Nil)),
                    Err(err) => Ok((
                        Value::Boolean(false),
                        Value::String(lua.create_string(&err)?),
                    )),
                }
            })?,
        )?;
    }

    // kv.delete
    {
        let manager = app_data.store_manager.clone();
        let app_data_snapshot = app_data.clone();
        let active_session = app_data.active_session_id.clone();
        kv_table.set(
            "delete",
            lua.create_function(move |lua, key: String| {
                if active_session.lock().unwrap().is_none() {
                    return Ok((
                        Value::Boolean(false),
                        Value::String(lua.create_string("No active session context")?),
                    ));
                }
                let selector = default_agent_selector(&app_data_snapshot)?;
                let manager = manager.clone();
                let result = block_on_current(async move {
                    kv_delete_backend(&manager, &selector, &key)
                        .await
                        .map_err(|e| e.to_string())
                });
                match result {
                    Ok(_) => Ok((Value::Boolean(true), Value::Nil)),
                    Err(err) => Ok((
                        Value::Boolean(false),
                        Value::String(lua.create_string(&err)?),
                    )),
                }
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
                        let selector = sel_get.clone();
                        let manager = m_get.clone();
                        let result = block_on_current(async move {
                            kv_get_backend(&manager, &selector, &key)
                                .await
                                .map_err(|e| e.to_string())
                        });
                        match result {
                            Ok(Some(val)) => {
                                Ok((Value::String(lua.create_string(&val)?), Value::Nil))
                            }
                            Ok(None) => Ok((Value::Nil, Value::Nil)),
                            Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                        }
                    })?,
                )?;

                let sel_set = selector.clone();
                let m_set = manager.clone();
                proxy.set(
                    "set",
                    lua.create_function(move |lua, (key, value): (String, String)| {
                        let selector = sel_set.clone();
                        let manager = m_set.clone();
                        let result = block_on_current(async move {
                            kv_set_backend(&manager, &selector, &key, &value)
                                .await
                                .map_err(|e| e.to_string())
                        });
                        match result {
                            Ok(_) => Ok((Value::Boolean(true), Value::Nil)),
                            Err(err) => Ok((
                                Value::Boolean(false),
                                Value::String(lua.create_string(&err)?),
                            )),
                        }
                    })?,
                )?;

                let sel_del = selector.clone();
                let m_del = manager.clone();
                proxy.set(
                    "delete",
                    lua.create_function(move |lua, key: String| {
                        let selector = sel_del.clone();
                        let manager = m_del.clone();
                        let result = block_on_current(async move {
                            kv_delete_backend(&manager, &selector, &key)
                                .await
                                .map_err(|e| e.to_string())
                        });
                        match result {
                            Ok(_) => Ok((Value::Boolean(true), Value::Nil)),
                            Err(err) => Ok((
                                Value::Boolean(false),
                                Value::String(lua.create_string(&err)?),
                            )),
                        }
                    })?,
                )?;

                Ok(proxy)
            })?,
        )?;
    }

    lua.globals().set("kv", kv_table)?;
    Ok(())
}
