use anyhow::Result;
use mlua::{Function, Lua, LuaSerdeExt, Result as LuaResult, Table, Value};

use crate::harness::stdlib::system_globals::ensure_load_time;
use crate::harness::virtual_tools::{
    DeclaredVirtualTool, VirtualToolPlan, normalize_tool_declaration, parse_handler_plan,
    shell_quote,
};

const DECLARED_TOOL_REGISTRY_KEY: &str = "__harness_declared_tools";

pub fn register_tool_globals(lua: &Lua) -> LuaResult<()> {
    let tool_table = lua.create_table()?;

    tool_table.set(
        "declare",
        lua.create_function(|lua, (name, spec): (String, Table)| {
            ensure_load_time(lua, "tool.declare")?;

            let description: String = spec
                .get("description")
                .map_err(|_| mlua::Error::runtime("tool.declare requires description"))?;
            let handler: Function = spec
                .get("handler")
                .map_err(|_| mlua::Error::runtime("tool.declare requires handler = function(...) end"))?;

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
        lua.create_function(|lua, (name, args): (String, Option<Value>)| {
            let out = lua.create_table()?;
            out.set("__kind", "tool_call")?;
            out.set("name", name)?;
            let args = match args {
                Some(Value::Nil) | None => Value::Table(lua.create_table()?),
                Some(value) => value,
            };
            out.set("args", args)?;
            Ok(out)
        })?,
    )?;

    tool_table.set(
        "sequence",
        lua.create_function(|lua, calls: Table| {
            let out = lua.create_table()?;
            out.set("__kind", "tool_sequence")?;
            out.set("calls", calls)?;
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

fn read_optional_json_field(lua: &Lua, spec: &Table, key: &str) -> LuaResult<Option<serde_json::Value>> {
    let value: Value = spec.get(key).unwrap_or(Value::Nil);
    if matches!(value, Value::Nil) {
        return Ok(None);
    }
    lua.from_value(value)
        .map(Some)
        .map_err(|err| mlua::Error::runtime(format!("tool.declare {} is not JSON-compatible: {}", key, err)))
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
    let result = lua
        .from_value::<serde_json::Value>(result)
        .map_err(|err| anyhow::anyhow!("virtual tool '{}' handler returned a non-JSON value: {}", name, err))?;

    Ok(Some(parse_handler_plan(&result)?))
}
