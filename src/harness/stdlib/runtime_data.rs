use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};

use crate::harness::globals::{HarnessAppData, block_on_current};
use crate::harness::stdlib::binding_common::{
    bool_err, memory_rows_to_lua_table, nil_err, ok_bool, ok_value, string_ok,
};
use crate::harness::stdlib::context_selectors::{search_limit_from_opt, table_to_selector};
use crate::harness::stdlib::scoped_data_backend::{
    kv_delete_backend, kv_get_backend, kv_set_backend, memory_search_backend, memory_store_backend,
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
        runtime_memory.set(
            "search",
            lua.create_function(
                move |lua, (query, ctx, opts): (String, Table, Option<Value>)| {
                    let selector = table_to_selector(ctx)?;
                    let limit = search_limit_from_opt(opts)?;
                    let manager = manager.clone();
                    let embedding = embedding.clone();
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
                        Ok(rows) => {
                            Ok(ok_value(Value::Table(memory_rows_to_lua_table(lua, rows)?)))
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
        runtime_memory.set(
            "store",
            lua.create_function(
                move |lua,
                      (content, ctx, metadata, _opts): (
                    String,
                    Table,
                    Option<Table>,
                    Option<Table>,
                )| {
                    let selector = table_to_selector(ctx)?;
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
                        Ok(_) => Ok(ok_bool()),
                        Err(err) => bool_err(lua, &err),
                    }
                },
            )?,
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
                let manager = manager.clone();
                let result = block_on_current(async move {
                    kv_get_backend(&manager, &selector, &key)
                        .await
                        .map_err(|e| e.to_string())
                });
                match result {
                    Ok(Some(val)) => string_ok(lua, &val),
                    Ok(None) => Ok((Value::Nil, Value::Nil)),
                    Err(err) => nil_err(lua, &err),
                }
            })?,
        )?;
    }
    {
        let manager = app_data.store_manager.clone();
        runtime_kv.set(
            "set",
            lua.create_function(move |lua, (key, value, ctx): (String, String, Table)| {
                let selector = table_to_selector(ctx)?;
                let manager = manager.clone();
                let result = block_on_current(async move {
                    kv_set_backend(&manager, &selector, &key, &value)
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
    {
        let manager = app_data.store_manager.clone();
        runtime_kv.set(
            "delete",
            lua.create_function(move |lua, (key, ctx): (String, Table)| {
                let selector = table_to_selector(ctx)?;
                let manager = manager.clone();
                let result = block_on_current(async move {
                    kv_delete_backend(&manager, &selector, &key)
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
    runtime_table.set("kv", runtime_kv)?;

    Ok(())
}
