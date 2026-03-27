use anyhow::Result;
use mlua::{Function, Lua, LuaSerdeExt, Result as LuaResult, Table, Value};

use crate::harness::stdlib::system_globals::ensure_load_time;
use crate::harness::virtual_tools::{
    DeclaredVirtualTool, VirtualToolPlan, VirtualToolResultResolution, normalize_tool_declaration,
    parse_handler_plan, parse_result_handler_output, shell_quote,
};

const DECLARED_TOOL_REGISTRY_KEY: &str = "__harness_declared_tools";
const DECLARED_TOOL_CALLBACKS_KEY: &str = "__harness_declared_tool_callbacks";
const DECLARED_TOOL_CALLBACK_SEQ_KEY: &str = "__harness_declared_tool_callback_seq";

pub fn register_tool_globals(lua: &Lua) -> LuaResult<()> {
    let tool_table = lua.create_table()?;

    tool_table.set(
        "declare",
        lua.create_function(|lua, (name, spec): (String, Table)| {
            ensure_load_time(lua, "tool.declare")?;

            let description: String = spec
                .get("description")
                .map_err(|_| mlua::Error::runtime("tool.declare requires description"))?;
            let handler: Function = spec.get("handler").map_err(|_| {
                mlua::Error::runtime("tool.declare requires handler = function(...) end")
            })?;

            let params = read_optional_json_field(lua, &spec, "params")?;
            let input_schema = read_optional_json_field(lua, &spec, "input_schema")?;
            let normalized = normalize_tool_declaration(&name, &description, params, input_schema)
                .map_err(mlua::Error::runtime)?;

            let registry = ensure_declared_tool_registry(lua)?;
            if registry.contains_key(name.clone())? {
                return Err(mlua::Error::runtime(format!(
                    "tool.declare('{}') conflicts with an existing declared tool",
                    name
                )));
            }

            let entry = lua.create_table()?;
            entry.set("description", normalized.description)?;
            entry.set("input_schema", lua.to_value(&normalized.input_schema)?)?;
            entry.set("handler", handler)?;
            registry.set(name, entry)?;
            Ok(())
        })?,
    )?;

    tool_table.set(
        "call",
        lua.create_function(
            |lua, (name, args, callback): (String, Option<Value>, Option<Function>)| {
                let out = lua.create_table()?;
                out.set("__kind", "tool_call")?;
                out.set("name", name)?;
                let args = match args {
                    Some(Value::Nil) | None => Value::Table(lua.create_table()?),
                    Some(value) => value,
                };
                out.set("args", args)?;
                if let Some(callback) = callback {
                    out.set(
                        "__result_handler_key",
                        register_result_handler(lua, callback)?,
                    )?;
                }
                Ok(out)
            },
        )?,
    )?;

    tool_table.set(
        "sequence",
        lua.create_function(|lua, (calls, callback): (Table, Option<Function>)| {
            let out = lua.create_table()?;
            out.set("__kind", "tool_sequence")?;
            out.set("calls", calls)?;
            if let Some(callback) = callback {
                out.set(
                    "__result_handler_key",
                    register_result_handler(lua, callback)?,
                )?;
            }
            Ok(out)
        })?,
    )?;

    lua.globals().set("tool", tool_table)?;

    let shell_table = lua.create_table()?;
    shell_table.set(
        "quote",
        lua.create_function(|_lua, input: String| Ok(shell_quote(&input)))?,
    )?;
    lua.globals().set("shell", shell_table)?;

    Ok(())
}

fn read_optional_json_field(
    lua: &Lua,
    spec: &Table,
    key: &str,
) -> LuaResult<Option<serde_json::Value>> {
    let value: Value = spec.get(key).unwrap_or(Value::Nil);
    if matches!(value, Value::Nil) {
        return Ok(None);
    }
    lua.from_value(value).map(Some).map_err(|err| {
        mlua::Error::runtime(format!(
            "tool.declare {} is not JSON-compatible: {}",
            key, err
        ))
    })
}

fn ensure_declared_tool_registry(lua: &Lua) -> LuaResult<Table> {
    let globals = lua.globals();
    if !globals.contains_key(DECLARED_TOOL_REGISTRY_KEY)? {
        globals.set(DECLARED_TOOL_REGISTRY_KEY, lua.create_table()?)?;
    }
    globals.get(DECLARED_TOOL_REGISTRY_KEY)
}

pub(crate) fn declared_virtual_tools(lua: &Lua) -> Result<Vec<DeclaredVirtualTool>> {
    let registry = ensure_declared_tool_registry(lua)?;
    let mut tools = Vec::new();
    for pair in registry.pairs::<String, Table>() {
        let (name, entry) = pair?;
        let description: String = entry.get("description")?;
        let input_schema = entry.get::<Value>("input_schema")?;
        let input_schema = lua.from_value::<serde_json::Value>(input_schema)?;
        tools.push(DeclaredVirtualTool {
            name,
            description,
            input_schema,
        });
    }
    tools.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(tools)
}

pub(crate) fn invoke_declared_virtual_tool(
    lua: &Lua,
    name: &str,
    args: serde_json::Value,
) -> Result<Option<VirtualToolPlan>> {
    let registry = ensure_declared_tool_registry(lua)?;
    let entry = match registry.get::<Value>(name)? {
        Value::Nil => return Ok(None),
        Value::Table(table) => table,
        other => anyhow::bail!(
            "declared tool registry entry '{}' has invalid type {:?}",
            name,
            other
        ),
    };

    let handler: Function = entry.get("handler")?;
    let lua_args = lua.to_value(&args)?;
    let result = handler.call::<Value>(lua_args)?;
    let result = lua.from_value::<serde_json::Value>(result).map_err(|err| {
        anyhow::anyhow!(
            "virtual tool '{}' handler returned a non-JSON value: {}",
            name,
            err
        )
    })?;

    Ok(Some(parse_handler_plan(&result)?))
}

pub(crate) fn invoke_virtual_result_handler(
    lua: &Lua,
    key: &str,
    payload: serde_json::Value,
    default_is_error: bool,
) -> Result<VirtualToolResultResolution> {
    let registry = ensure_result_handler_registry(lua)?;
    let callback = match registry.get::<Value>(key)? {
        Value::Nil => anyhow::bail!("virtual tool result handler '{}' was not found", key),
        Value::Function(function) => function,
        other => anyhow::bail!(
            "virtual tool result handler '{}' has invalid type {:?}",
            key,
            other
        ),
    };
    registry.set(key, Value::Nil)?;

    let lua_payload = lua.to_value(&payload)?;
    let result = callback.call::<Value>(lua_payload)?;
    let result = lua.from_value::<serde_json::Value>(result).map_err(|err| {
        anyhow::anyhow!(
            "virtual tool result handler '{}' returned a non-JSON value: {}",
            key,
            err
        )
    })?;

    parse_result_handler_output(&result, default_is_error)
}

pub(crate) fn discard_virtual_result_handler(lua: &Lua, key: &str) -> Result<()> {
    let registry = ensure_result_handler_registry(lua)?;
    registry.set(key, Value::Nil)?;
    Ok(())
}

fn ensure_result_handler_registry(lua: &Lua) -> LuaResult<Table> {
    let globals = lua.globals();
    if !globals.contains_key(DECLARED_TOOL_CALLBACKS_KEY)? {
        globals.set(DECLARED_TOOL_CALLBACKS_KEY, lua.create_table()?)?;
    }
    if !globals.contains_key(DECLARED_TOOL_CALLBACK_SEQ_KEY)? {
        globals.set(DECLARED_TOOL_CALLBACK_SEQ_KEY, 0_i64)?;
    }
    globals.get(DECLARED_TOOL_CALLBACKS_KEY)
}

fn register_result_handler(lua: &Lua, callback: Function) -> LuaResult<String> {
    let registry = ensure_result_handler_registry(lua)?;
    let globals = lua.globals();
    let seq = globals
        .get::<i64>(DECLARED_TOOL_CALLBACK_SEQ_KEY)
        .unwrap_or(0)
        + 1;
    globals.set(DECLARED_TOOL_CALLBACK_SEQ_KEY, seq)?;
    let key = format!("cb_{}", seq);
    registry.set(key.clone(), callback)?;
    Ok(key)
}
