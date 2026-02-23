use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};

use crate::harness::globals::{HarnessAppData, block_on_current};
use crate::harness::stdlib::context_selectors::{
    search_limit_from_opt, selector_from_active_scope_lua,
};
use crate::harness::stdlib::scoped_data_backend::{
    kv_delete_backend, kv_get_backend, kv_set_backend, memory_search_backend, memory_store_backend,
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
                    let limit = search_limit_from_opt(opts)?;
                    let selector = selector_from_active_scope_lua(&selector_app, scope)?;
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
                            let tbl = lua.create_table()?;
                            for (i, row) in rows.into_iter().enumerate() {
                                let rt = lua.create_table()?;
                                rt.set("content", row.content)?;
                                rt.set("score", row.score)?;
                                tbl.set(i + 1, rt)?;
                            }
                            Ok((Value::Table(tbl), Value::Nil))
                        }
                        Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
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
                    move |lua, (content, metadata, _opts): (String, Option<Table>, Option<Table>)| {
                        let selector = selector_from_active_scope_lua(&selector_app, scope)?;
                        let metadata_json = if let Some(tbl) = metadata {
                            lua.from_value::<serde_json::Value>(Value::Table(tbl)).map_err(|e| {
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
                lua.create_function(move |lua, key: String| {
                    let selector = selector_from_active_scope_lua(&selector_app, scope)?;
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
        {
            let manager = app_data.store_manager.clone();
            let selector_app = app_data.clone();
            kv.set(
                "set",
                lua.create_function(move |lua, (key, value): (String, String)| {
                    let selector = selector_from_active_scope_lua(&selector_app, scope)?;
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
        {
            let manager = app_data.store_manager.clone();
            let selector_app = app_data.clone();
            kv.set(
                "delete",
                lua.create_function(move |lua, key: String| {
                    let selector = selector_from_active_scope_lua(&selector_app, scope)?;
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
