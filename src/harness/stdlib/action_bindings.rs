use anyhow::Result;
use mlua::{Function, Lua, LuaSerdeExt, Result as LuaResult, Table, Value};

use crate::harness::stdlib::system_globals::ensure_load_time;

const DECLARED_ACTION_REGISTRY_KEY: &str = "__harness_declared_actions";

pub fn register_action_globals(lua: &Lua) -> LuaResult<()> {
    let action_table = lua.create_table()?;

    action_table.set(
        "define",
        lua.create_function(|lua, (name, handler): (String, Function)| {
            ensure_load_time(lua, "action.define")?;

            let registry = ensure_declared_action_registry(lua)?;
            if registry.contains_key(name.clone())? {
                return Err(mlua::Error::runtime(format!(
                    "action.define('{}') conflicts with an existing declared action",
                    name
                )));
            }

            registry.set(name, handler)?;
            Ok(())
        })?,
    )?;

    lua.globals().set("action", action_table)?;
    Ok(())
}

fn ensure_declared_action_registry(lua: &Lua) -> LuaResult<Table> {
    let globals = lua.globals();
    if !globals.contains_key(DECLARED_ACTION_REGISTRY_KEY)? {
        globals.set(DECLARED_ACTION_REGISTRY_KEY, lua.create_table()?)?;
    }
    globals.get(DECLARED_ACTION_REGISTRY_KEY)
}

pub(crate) fn invoke_declared_action(
    lua: &Lua,
    name: &str,
    params: serde_json::Value,
) -> Result<Option<serde_json::Value>> {
    let registry = ensure_declared_action_registry(lua)?;
    let handler = match registry.get::<Value>(name)? {
        Value::Nil => return invoke_builtin_action(lua, name, params),
        Value::Function(function) => function,
        other => anyhow::bail!(
            "declared action registry entry '{}' has invalid type {:?}",
            name,
            other
        ),
    };

    let lua_args = lua.to_value(&params)?;
    let result = handler.call::<Value>(lua_args)?;
    let result = lua.from_value::<serde_json::Value>(result).map_err(|err| {
        anyhow::anyhow!(
            "declared action '{}' handler returned a non-JSON value: {}",
            name,
            err
        )
    })?;

    Ok(Some(result))
}

fn invoke_builtin_action(
    lua: &Lua,
    name: &str,
    params: serde_json::Value,
) -> Result<Option<serde_json::Value>> {
    match name {
        "worklist.dispatch_next" => Ok(Some(invoke_builtin_worklist_method(
            lua,
            "dispatch_next",
            params,
        )?)),
        "worklist.release_stale" => Ok(Some(invoke_builtin_worklist_method(
            lua,
            "release_stale",
            params,
        )?)),
        _ => Ok(None),
    }
}

fn invoke_builtin_worklist_method(
    lua: &Lua,
    method_name: &str,
    params: serde_json::Value,
) -> Result<serde_json::Value> {
    let params_value = lua.to_value(&params)?;
    let params_table = match params_value {
        Value::Nil => lua.create_table()?,
        Value::Table(table) => table,
        other => anyhow::bail!(
            "built-in action 'worklist.{}' requires object-like params, got {:?}",
            method_name,
            other
        ),
    };
    if !params_table.contains_key("name")? {
        anyhow::bail!(
            "built-in action 'worklist.{}' requires params.name",
            method_name
        );
    }

    let globals = lua.globals();
    let runtime: Table = globals.get("runtime")?;
    let runtime_worklist: Table = runtime.get("worklist")?;
    let open_fn: Function = runtime_worklist.get("open")?;
    let list_proxy: Table = open_fn.call(params_table.clone())?;
    let method: Function = list_proxy.get(method_name)?;
    let result: Value = method.call((list_proxy, params_table))?;
    worklist_action_result_to_json(lua, method_name, result)
}

fn worklist_action_result_to_json(
    lua: &Lua,
    method_name: &str,
    result: Value,
) -> Result<serde_json::Value> {
    match method_name {
        "dispatch_next" => match result {
            Value::Nil => Ok(serde_json::Value::Null),
            Value::Table(table) => {
                let item = summarize_worklist_item_table(&table.get::<Table>("item")?)?;
                let dispatch_result = lua.from_value::<serde_json::Value>(table.get("result")?)?;
                Ok(serde_json::json!({
                    "item": item,
                    "result": dispatch_result,
                }))
            }
            other => anyhow::bail!(
                "built-in action 'worklist.dispatch_next' returned unexpected value {:?}",
                other
            ),
        },
        "release_stale" => match result {
            Value::Table(table) => {
                let mut items = Vec::new();
                for value in table.sequence_values::<Table>() {
                    items.push(summarize_worklist_item_table(&value?)?);
                }
                Ok(serde_json::Value::Array(items))
            }
            other => anyhow::bail!(
                "built-in action 'worklist.release_stale' returned unexpected value {:?}",
                other
            ),
        },
        _ => lua.from_value(result).map_err(Into::into),
    }
}

fn summarize_worklist_item_table(table: &Table) -> Result<serde_json::Value> {
    Ok(serde_json::json!({
        "id": table.get::<Option<String>>("id")?,
        "title": table.get::<Option<String>>("title")?,
        "kind": table.get::<Option<String>>("kind")?,
        "status": table.get::<Option<String>>("status")?,
        "priority": table.get::<Option<i64>>("priority")?,
        "prompt": table.get::<Option<String>>("prompt")?,
        "action": table.get::<Option<String>>("action")?,
        "claim_execution_id": table.get::<Option<String>>("claim_execution_id")?,
        "failure_reason": table.get::<Option<String>>("failure_reason")?,
    }))
}
