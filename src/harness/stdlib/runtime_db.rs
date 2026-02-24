use mlua::{Lua, Result as LuaResult, Table, Value};

use crate::harness::globals::{HarnessAppData, block_on_current};
use crate::harness::stdlib::binding_common::{bool_err, bool_value_ok, json_ok, nil_err, ok_value};
use crate::harness::stdlib::db_support::{
    SqlParams, lua_table_to_sql_params, selector_from_db_opts, selector_from_db_value,
    sql_value_to_json,
};
use crate::harness::stdlib::policy_support::{
    policy_bool, policy_string, policy_u64, runtime_policy_snapshot,
};
use crate::persistence::manager::{StoreHandleInfo, StoreManager, StorePathScope, StoreSelector};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Copy)]
struct DbRuntimeSettings {
    path_scope: StorePathScope,
    max_open_handles: usize,
    idle_close_secs: u64,
}

fn db_runtime_settings(snapshot: &HashMap<String, serde_json::Value>) -> DbRuntimeSettings {
    DbRuntimeSettings {
        path_scope: StorePathScope::from_policy(policy_string(
            snapshot,
            "db.path_scope",
            "workspace_only",
        )),
        max_open_handles: policy_u64(snapshot, "db.max_open_handles", 128).clamp(1, u64::MAX)
            as usize,
        idle_close_secs: policy_u64(snapshot, "db.idle_close_secs", 300),
    }
}

fn selector_denied_by_dynamic_open(
    snapshot: &HashMap<String, serde_json::Value>,
    selector: &StoreSelector,
) -> bool {
    matches!(selector, StoreSelector::Path(_))
        && !policy_bool(snapshot, "db.allow_dynamic_open", true)
}

fn store_handle_info_to_lua_table(lua: &Lua, h: StoreHandleInfo) -> LuaResult<Table> {
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
    Ok(t)
}

fn store_handle_infos_to_lua_table(lua: &Lua, handles: Vec<StoreHandleInfo>) -> LuaResult<Table> {
    let out = lua.create_table()?;
    for (i, h) in handles.into_iter().enumerate() {
        out.set(i + 1, store_handle_info_to_lua_table(lua, h)?)?;
    }
    Ok(out)
}

async fn open_store_for_query_exec(
    manager: Arc<StoreManager>,
    selector: StoreSelector,
    settings: DbRuntimeSettings,
) -> Result<Arc<crate::persistence::state::StateStore>, String> {
    let _ = manager
        .trim_cache(settings.max_open_handles, settings.idle_close_secs)
        .await;
    manager
        .open_with_path_scope(&selector, settings.path_scope)
        .await
        .map_err(|e| e.to_string())
}

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
                if selector_denied_by_dynamic_open(&snapshot, &selector) {
                    return nil_err(lua, "Policy denial: db.allow_dynamic_open=false");
                }
                let settings = db_runtime_settings(&snapshot);

                let manager = manager.clone();
                let result = block_on_current(async move {
                    manager
                        .open_handle(
                            &selector,
                            settings.path_scope,
                            settings.max_open_handles,
                            settings.idle_close_secs,
                        )
                        .await
                        .map_err(|e| e.to_string())
                });

                match result {
                    Ok(info) => Ok(ok_value(Value::Table(store_handle_info_to_lua_table(
                        lua, info,
                    )?))),
                    Err(err) => nil_err(lua, &err),
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
                        return bool_err(lua, "invalid handle; expected string or {handle=...}");
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
                    Ok(closed) => Ok(bool_value_ok(closed)),
                    Err(err) => bool_err(lua, &err),
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
                Ok(ok_value(Value::Table(store_handle_infos_to_lua_table(
                    lua, handles,
                )?)))
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
                    if selector_denied_by_dynamic_open(&snapshot, &selector) {
                        return nil_err(lua, "Policy denial: db.allow_dynamic_open=false");
                    }
                    let settings = db_runtime_settings(&snapshot);
                    let manager = manager.clone();
                    let result = block_on_current(async move {
                        let store = open_store_for_query_exec(manager, selector, settings).await?;
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
                        Ok(rows) => json_ok(lua, &rows),
                        Err(err) => nil_err(lua, &err),
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
                    if selector_denied_by_dynamic_open(&snapshot, &selector) {
                        return nil_err(lua, "Policy denial: db.allow_dynamic_open=false");
                    }
                    let settings = db_runtime_settings(&snapshot);
                    let manager = manager.clone();
                    let result = block_on_current(async move {
                        let store = open_store_for_query_exec(manager, selector, settings).await?;
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
                        Ok(changed) => Ok(ok_value(Value::Integer(changed as i64))),
                        Err(err) => nil_err(lua, &err),
                    }
                },
            )?,
        )?;
    }
    runtime_table.set("db", runtime_db)?;
    Ok(())
}
