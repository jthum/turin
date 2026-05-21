use anyhow::Result;
use mlua::{Function, Lua, Table, Value};
use serde_json::Value as JsonValue;

use crate::harness::stdlib::object_refs;

pub(crate) fn invoke_builtin_action(
    lua: &Lua,
    name: &str,
    params: JsonValue,
) -> Result<Option<JsonValue>> {
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
    params: JsonValue,
) -> Result<JsonValue> {
    let params_value = object_refs::decode_json_payload(lua, &params)?;
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
) -> Result<JsonValue> {
    match method_name {
        "dispatch_next" => match result {
            Value::Nil => Ok(JsonValue::Null),
            Value::Table(table) => {
                let item = summarize_worklist_item_table(&table.get::<Table>("item")?)?;
                let dispatch_result = object_refs::encode_lua_payload(lua, table.get("result")?)?;
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
                Ok(JsonValue::Array(items))
            }
            other => anyhow::bail!(
                "built-in action 'worklist.release_stale' returned unexpected value {:?}",
                other
            ),
        },
        _ => object_refs::encode_lua_payload(lua, result).map_err(Into::into),
    }
}

fn summarize_worklist_item_table(table: &Table) -> Result<JsonValue> {
    Ok(serde_json::json!({
        "id": table.get::<Option<String>>("id")?,
        "title": table.get::<Option<String>>("title")?,
        "kind": table.get::<Option<String>>("kind")?,
        "status": table.get::<Option<String>>("status")?,
        "priority": table.get::<Option<i64>>("priority")?,
        "prompt": table.get::<Option<String>>("prompt")?,
        "action": table.get::<Option<String>>("action_name")?,
        "claim_execution_id": table.get::<Option<String>>("claim_execution_id")?,
        "failure_reason": table.get::<Option<String>>("failure_reason")?,
    }))
}
