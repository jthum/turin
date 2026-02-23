use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};

use crate::harness::globals::{
    HarnessAppData, SqlParams, block_on_current, lua_table_to_sql_params, policy_bool,
    policy_string, policy_u64, runtime_policy_snapshot, selector_from_db_opts,
    selector_from_db_value, sql_value_to_json,
};
use crate::persistence::manager::{StorePathScope, StoreSelector};

pub fn register_runtime_db_namespace(
    lua: &Lua,
    runtime_table: &Table,
    app_data: &HarnessAppData,
) -> LuaResult<()> {
    let runtime_db = lua.create_table()?;
    {
        let manager = app_data.store_manager.clone();
        let app_data_snapshot = app_data.clone();
        runtime_db.set(
            "open",
            lua.create_function(move |lua, arg: Value| {
                let selector = selector_from_db_value(arg)?;
                let snapshot =
                    runtime_policy_snapshot(&app_data_snapshot).map_err(mlua::Error::runtime)?;
                if matches!(selector, StoreSelector::Path(_))
                    && !policy_bool(&snapshot, "db.allow_dynamic_open", true)
                {
                    return Ok((
                        Value::Nil,
                        Value::String(
                            lua.create_string("Policy denial: db.allow_dynamic_open=false")?,
                        ),
                    ));
                }

                let path_scope = StorePathScope::from_policy(policy_string(
                    &snapshot,
                    "db.path_scope",
                    "workspace_only",
                ));
                let max_open_handles =
                    policy_u64(&snapshot, "db.max_open_handles", 128).clamp(1, u64::MAX) as usize;
                let idle_close_secs = policy_u64(&snapshot, "db.idle_close_secs", 300);

                let manager = manager.clone();
                let result = block_on_current(async move {
                    manager
                        .open_handle(&selector, path_scope, max_open_handles, idle_close_secs)
                        .await
                        .map_err(|e| e.to_string())
                });

                match result {
                    Ok(info) => {
                        let t = lua.create_table()?;
                        t.set("handle", info.handle)?;
                        t.set("path", info.path.to_string_lossy().to_string())?;
                        if let Some(alias) = info.alias {
                            t.set("alias", alias)?;
                        } else {
                            t.set("alias", Value::Nil)?;
                        }
                        t.set("open_count", info.open_count)?;
                        t.set("idle_ms", info.idle_ms)?;
                        Ok((Value::Table(t), Value::Nil))
                    }
                    Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                }
            })?,
        )?;
    }
    {
        let manager = app_data.store_manager.clone();
        runtime_db.set(
            "close",
            lua.create_function(move |lua, handle: Value| {
                let handle_id = match handle {
                    Value::String(s) => s.to_str()?.to_string(),
                    Value::Table(t) => t.get::<String>("handle")?,
                    _ => {
                        return Ok((
                            Value::Boolean(false),
                            Value::String(lua.create_string(
                                "invalid handle; expected string or {handle=...}",
                            )?),
                        ));
                    }
                };
                let manager = manager.clone();
                let result = block_on_current(async move {
                    manager
                        .close_handle(&handle_id)
                        .await
                        .map_err(|e| e.to_string())
                });
                match result {
                    Ok(closed) => Ok((Value::Boolean(closed), Value::Nil)),
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
        runtime_db.set(
            "list",
            lua.create_function(move |lua, ()| {
                let manager = manager.clone();
                let handles = block_on_current(async move { manager.list_handles().await });
                let out = lua.create_table()?;
                for (i, h) in handles.into_iter().enumerate() {
                    let t = lua.create_table()?;
                    t.set("handle", h.handle)?;
                    t.set("path", h.path.to_string_lossy().to_string())?;
                    if let Some(alias) = h.alias {
                        t.set("alias", alias)?;
                    } else {
                        t.set("alias", Value::Nil)?;
                    }
                    t.set("open_count", h.open_count)?;
                    t.set("idle_ms", h.idle_ms)?;
                    out.set(i + 1, t)?;
                }
                Ok((Value::Table(out), Value::Nil))
            })?,
        )?;
    }
    {
        let manager = app_data.store_manager.clone();
        let app_data_snapshot = app_data.clone();
        runtime_db.set(
            "query",
            lua.create_function(
                move |lua, (sql, params, opts): (String, Option<Table>, Option<Table>)| {
                    let selector = selector_from_db_opts(opts)?;
                    let sql_params = lua_table_to_sql_params(params)?;
                    let snapshot = runtime_policy_snapshot(&app_data_snapshot)
                        .map_err(mlua::Error::runtime)?;
                    if matches!(selector, StoreSelector::Path(_))
                        && !policy_bool(&snapshot, "db.allow_dynamic_open", true)
                    {
                        return Ok((
                            Value::Nil,
                            Value::String(
                                lua.create_string("Policy denial: db.allow_dynamic_open=false")?,
                            ),
                        ));
                    }
                    let path_scope = StorePathScope::from_policy(policy_string(
                        &snapshot,
                        "db.path_scope",
                        "workspace_only",
                    ));
                    let max_open_handles = policy_u64(&snapshot, "db.max_open_handles", 128)
                        .clamp(1, u64::MAX) as usize;
                    let idle_close_secs = policy_u64(&snapshot, "db.idle_close_secs", 300);
                    let manager = manager.clone();
                    let result = block_on_current(async move {
                        let _ = manager.trim_cache(max_open_handles, idle_close_secs).await;
                        let store = manager
                            .open_with_path_scope(&selector, path_scope)
                            .await
                            .map_err(|e| e.to_string())?;
                        let conn = store.get_connection().await.map_err(|e| e.to_string())?;
                        let mut stmt = conn.prepare(&sql).await.map_err(|e| e.to_string())?;
                        let cols = stmt
                            .columns()
                            .into_iter()
                            .map(|c| c.name().to_string())
                            .collect::<Vec<_>>();
                        let mut rows = match sql_params {
                            SqlParams::None => stmt.query(()).await.map_err(|e| e.to_string())?,
                            SqlParams::Positional(v) => {
                                stmt.query(v).await.map_err(|e| e.to_string())?
                            }
                            SqlParams::Named(v) => {
                                stmt.query(v).await.map_err(|e| e.to_string())?
                            }
                        };
                        let mut out_rows = Vec::<serde_json::Value>::new();
                        while let Some(row) = rows.next().await.map_err(|e| e.to_string())? {
                            let mut obj = serde_json::Map::new();
                            for (idx, col) in cols.iter().enumerate() {
                                let v = row.get_value(idx).map_err(|e| e.to_string())?;
                                obj.insert(col.clone(), sql_value_to_json(v));
                            }
                            out_rows.push(serde_json::Value::Object(obj));
                        }
                        Ok::<_, String>(out_rows)
                    });
                    match result {
                        Ok(rows) => {
                            let lua_v = lua
                                .to_value(&rows)
                                .map_err(|e| mlua::Error::runtime(e.to_string()))?;
                            Ok((lua_v, Value::Nil))
                        }
                        Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                    }
                },
            )?,
        )?;
    }
    {
        let manager = app_data.store_manager.clone();
        let app_data_snapshot = app_data.clone();
        runtime_db.set(
            "exec",
            lua.create_function(
                move |lua, (sql, params, opts): (String, Option<Table>, Option<Table>)| {
                    let selector = selector_from_db_opts(opts)?;
                    let sql_params = lua_table_to_sql_params(params)?;
                    let snapshot = runtime_policy_snapshot(&app_data_snapshot)
                        .map_err(mlua::Error::runtime)?;
                    if matches!(selector, StoreSelector::Path(_))
                        && !policy_bool(&snapshot, "db.allow_dynamic_open", true)
                    {
                        return Ok((
                            Value::Nil,
                            Value::String(
                                lua.create_string("Policy denial: db.allow_dynamic_open=false")?,
                            ),
                        ));
                    }
                    let path_scope = StorePathScope::from_policy(policy_string(
                        &snapshot,
                        "db.path_scope",
                        "workspace_only",
                    ));
                    let max_open_handles = policy_u64(&snapshot, "db.max_open_handles", 128)
                        .clamp(1, u64::MAX) as usize;
                    let idle_close_secs = policy_u64(&snapshot, "db.idle_close_secs", 300);
                    let manager = manager.clone();
                    let result = block_on_current(async move {
                        let _ = manager.trim_cache(max_open_handles, idle_close_secs).await;
                        let store = manager
                            .open_with_path_scope(&selector, path_scope)
                            .await
                            .map_err(|e| e.to_string())?;
                        let conn = store.get_connection().await.map_err(|e| e.to_string())?;
                        let changed = match sql_params {
                            SqlParams::None => {
                                conn.execute(&sql, ()).await.map_err(|e| e.to_string())?
                            }
                            SqlParams::Positional(v) => {
                                conn.execute(&sql, v).await.map_err(|e| e.to_string())?
                            }
                            SqlParams::Named(v) => {
                                conn.execute(&sql, v).await.map_err(|e| e.to_string())?
                            }
                        };
                        Ok::<_, String>(changed)
                    });
                    match result {
                        Ok(changed) => Ok((Value::Integer(changed as i64), Value::Nil)),
                        Err(err) => Ok((Value::Nil, Value::String(lua.create_string(&err)?))),
                    }
                },
            )?,
        )?;
    }
    runtime_table.set("db", runtime_db)?;
    Ok(())
}
