use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};

use crate::harness::globals::{HarnessAppData, block_on_current};
use crate::harness::stdlib::binding_common::{bool_err, json_ok, nil_err, nil_ok, ok_bool};
use crate::harness::stdlib::policy_support::policy_scope_from_value;

pub fn register_runtime_policy_namespace(
    lua: &Lua,
    runtime_table: &Table,
    app_data: &HarnessAppData,
) -> LuaResult<()> {
    let policy_table = lua.create_table()?;
    {
        let policy_manager = app_data.policy_manager.clone();
        let app_data_snapshot = app_data.clone();
        policy_table.set(
            "get",
            lua.create_function(move |lua, (key, scope): (String, Option<Value>)| {
                let scope = policy_scope_from_value(&app_data_snapshot, scope)?;
                let policy_manager = policy_manager.clone();
                let result = block_on_current(async move {
                    policy_manager
                        .get(&key, &scope)
                        .await
                        .map_err(|e| e.to_string())
                });

                match result {
                    Ok(Some(v)) => json_ok(lua, &v),
                    Ok(None) => Ok(nil_ok()),
                    Err(err) => nil_err(lua, &err),
                }
            })?,
        )?;
    }
    {
        let policy_manager = app_data.policy_manager.clone();
        let app_data_snapshot = app_data.clone();
        policy_table.set(
            "set",
            lua.create_function(
                move |lua, (key, value, scope): (String, Value, Option<Value>)| {
                    let scope = policy_scope_from_value(&app_data_snapshot, scope)?;
                    let json_value = lua.from_value::<serde_json::Value>(value).map_err(|e| {
                        mlua::Error::runtime(format!("invalid policy value: {}", e))
                    })?;
                    let policy_manager = policy_manager.clone();
                    let result = block_on_current(async move {
                        policy_manager
                            .set(&key, json_value, &scope)
                            .await
                            .map_err(|e| e.to_string())
                    });
                    match result {
                        Ok(()) => Ok(ok_bool()),
                        Err(err) => bool_err(lua, &err),
                    }
                },
            )?,
        )?;
    }
    runtime_table.set("policy", policy_table)?;
    Ok(())
}
