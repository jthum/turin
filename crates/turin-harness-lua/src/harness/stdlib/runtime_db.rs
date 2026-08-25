use mlua::{Lua, Result as LuaResult, Table, Value};

use crate::harness::globals::HarnessAppData;
use crate::harness::stdlib::binding_common::{
    bool_err, bool_value_ok, bridge_async, bridge_async_display_err, bridge_async_result,
    lua_json_result, lua_table_result, lua_value_result, nil_err, ok_value,
};
use crate::harness::stdlib::db_support::{
    SqlParams, lua_table_to_sql_params, selector_denied_by_dynamic_open, selector_from_db_opts,
    selector_from_db_value, sql_value_to_json, store_path_scope_from_snapshot,
};
use crate::harness::stdlib::governance_support::require_capability as require_governance_capability;
use crate::harness::stdlib::policy_support::{policy_u64, runtime_policy_snapshot};
use crate::persistence::manager::{StoreHandleInfo, StoreManager, StorePathScope, StoreSelector};
use std::collections::HashMap;
use std::sync::Arc;
use turso::Connection;

#[derive(Clone, Copy)]
struct DbRuntimeSettings {
    path_scope: StorePathScope,
    max_open_handles: usize,
    idle_close_seconds: u64,
}

fn db_runtime_settings(snapshot: &HashMap<String, serde_json::Value>) -> DbRuntimeSettings {
    DbRuntimeSettings {
        path_scope: store_path_scope_from_snapshot(snapshot),
        max_open_handles: policy_u64(snapshot, "db.max_open_handles", 128).clamp(1, u64::MAX)
            as usize,
        idle_close_seconds: policy_u64(snapshot, "db.idle_close_seconds", 300),
    }
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

fn resolve_db_target(
    app_data: &HarnessAppData,
    opts: Option<Table>,
) -> LuaResult<(StoreSelector, DbRuntimeSettings)> {
    let selector = selector_from_db_opts(opts)?;
    let snapshot = runtime_policy_snapshot(app_data).map_err(mlua::Error::runtime)?;
    if selector_denied_by_dynamic_open(&snapshot, &selector) {
        return Err(mlua::Error::runtime(
            "Policy denial: db.allow_dynamic_open=false",
        ));
    }
    Ok((selector, db_runtime_settings(&snapshot)))
}

async fn open_connection_for_query_exec(
    manager: Arc<StoreManager>,
    selector: StoreSelector,
    settings: DbRuntimeSettings,
) -> Result<Connection, String> {
    let _ = manager
        .trim_cache(settings.max_open_handles, settings.idle_close_seconds)
        .await;
    let store = manager
        .open_with_path_scope(&selector, settings.path_scope)
        .await
        .map_err(|e| e.to_string())?;
    store.get_connection().await.map_err(|e| e.to_string())
}

async fn query_sql_rows(
    manager: Arc<StoreManager>,
    selector: StoreSelector,
    settings: DbRuntimeSettings,
    sql: String,
    sql_params: SqlParams,
) -> Result<Vec<serde_json::Value>, String> {
    let conn = open_connection_for_query_exec(manager, selector, settings).await?;
    let mut stmt = conn.prepare(&sql).await.map_err(|e| e.to_string())?;
    let cols = stmt
        .columns()
        .into_iter()
        .map(|c| c.name().to_string())
        .collect::<Vec<_>>();
    let mut rows = match sql_params {
        SqlParams::None => stmt.query(()).await.map_err(|e| e.to_string())?,
        SqlParams::Positional(v) => stmt.query(v).await.map_err(|e| e.to_string())?,
        SqlParams::Named(v) => stmt.query(v).await.map_err(|e| e.to_string())?,
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
    Ok(out_rows)
}

fn rejects_kernel_schema_ddl(selector: &StoreSelector, sql: &str) -> bool {
    if !matches!(selector, StoreSelector::Alias(alias) if alias == "state") {
        return false;
    }
    let trimmed = sql.trim_start();
    let head = trimmed
        .get(..12)
        .unwrap_or(trimmed)
        .to_ascii_uppercase();
    head.starts_with("CREATE")
        || head.starts_with("DROP")
        || head.starts_with("ALTER")
        || head.starts_with("ATTACH")
        || head.starts_with("DETACH")
}

async fn exec_sql(
    manager: Arc<StoreManager>,
    selector: StoreSelector,
    settings: DbRuntimeSettings,
    sql: String,
    sql_params: SqlParams,
) -> Result<u64, String> {
    if rejects_kernel_schema_ddl(&selector, &sql) {
        return Err(
            "kernel state store does not allow DDL; open a harness-owned database instead"
                .to_string(),
        );
    }
    let conn = open_connection_for_query_exec(manager, selector, settings).await?;
    match sql_params {
        SqlParams::None => conn.execute(&sql, ()).await.map_err(|e| e.to_string()),
        SqlParams::Positional(v) => conn.execute(&sql, v).await.map_err(|e| e.to_string()),
        SqlParams::Named(v) => conn.execute(&sql, v).await.map_err(|e| e.to_string()),
    }
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
                if let Err(err) =
                    require_governance_capability(&app_data_snapshot, "runtime.db.open")
                {
                    return nil_err(lua, &err);
                }
                let selector = selector_from_db_value(arg)?;
                let snapshot =
                    runtime_policy_snapshot(&app_data_snapshot).map_err(mlua::Error::runtime)?;
                if selector_denied_by_dynamic_open(&snapshot, &selector) {
                    return nil_err(lua, "Policy denial: db.allow_dynamic_open=false");
                }
                let settings = db_runtime_settings(&snapshot);

                let manager = manager.clone();
                let result = bridge_async_display_err(async move {
                    manager
                        .open_handle(
                            &selector,
                            settings.path_scope,
                            settings.max_open_handles,
                            settings.idle_close_seconds,
                        )
                        .await
                });

                lua_table_result(lua, result, store_handle_info_to_lua_table)
            })?,
        )?;
    }
    {
        let manager = app_data.store_manager.clone();
        let app_data_snapshot = app_data.clone();
        runtime_db.set(
            "close",
            lua.create_function(move |lua, handle: Value| {
                if let Err(err) =
                    require_governance_capability(&app_data_snapshot, "runtime.db.close")
                {
                    return bool_err(lua, &err);
                }
                let handle_id = match handle {
                    Value::String(s) => s.to_str()?.to_string(),
                    Value::Table(t) => t.get::<String>("handle")?,
                    _ => {
                        return bool_err(lua, "invalid handle; expected string or {handle=...}");
                    }
                };
                let manager = manager.clone();
                let result =
                    bridge_async_display_err(async move { manager.close_handle(&handle_id).await });
                match result {
                    Ok(closed) => Ok(bool_value_ok(closed)),
                    Err(err) => bool_err(lua, &err),
                }
            })?,
        )?;
    }
    {
        let manager = app_data.store_manager.clone();
        let app_data_snapshot = app_data.clone();
        runtime_db.set(
            "list",
            lua.create_function(move |lua, ()| {
                if let Err(err) =
                    require_governance_capability(&app_data_snapshot, "runtime.db.list_handles")
                {
                    return nil_err(lua, &err);
                }
                let manager = manager.clone();
                let handles = bridge_async(async move { manager.list_handles().await });
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
                    if let Err(err) =
                        require_governance_capability(&app_data_snapshot, "runtime.db.query")
                    {
                        return nil_err(lua, &err);
                    }
                    let sql_params = lua_table_to_sql_params(params)?;
                    let (selector, settings) = match resolve_db_target(&app_data_snapshot, opts) {
                        Ok(target) => target,
                        Err(err) => return nil_err(lua, &err.to_string()),
                    };
                    let manager = manager.clone();
                    let result = bridge_async_result(async move {
                        query_sql_rows(manager, selector, settings, sql, sql_params).await
                    });
                    lua_json_result(lua, result)
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
                    if let Err(err) =
                        require_governance_capability(&app_data_snapshot, "runtime.db.exec")
                    {
                        return nil_err(lua, &err);
                    }
                    let sql_params = lua_table_to_sql_params(params)?;
                    let (selector, settings) = match resolve_db_target(&app_data_snapshot, opts) {
                        Ok(target) => target,
                        Err(err) => return nil_err(lua, &err.to_string()),
                    };
                    let manager = manager.clone();
                    let result = bridge_async_result(async move {
                        exec_sql(manager, selector, settings, sql, sql_params).await
                    });
                    lua_value_result(lua, result, |_lua, changed| {
                        Ok(Value::Integer(changed as i64))
                    })
                },
            )?,
        )?;
    }
    runtime_table.set("db", runtime_db)?;
    Ok(())
}
