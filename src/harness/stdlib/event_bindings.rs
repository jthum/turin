use mlua::{Function, Lua, LuaSerdeExt, Result as LuaResult, Table, Value};
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::harness::stdlib::binding_common::wrap_registered_callback;
use crate::harness::stdlib::system_globals::ensure_load_time;

const EVENT_LISTENER_REGISTRY_KEY: &str = "__harness_event_listeners";

pub fn register_event_globals(lua: &Lua) -> LuaResult<()> {
    lua.globals().set(
        "on",
        lua.create_function(|lua, (name, handler): (String, Function)| {
            ensure_load_time(lua, "on")?;

            let registry = ensure_event_listener_registry(lua)?;
            let listeners = match registry.get::<Value>(name.clone())? {
                Value::Nil => {
                    let table = lua.create_table()?;
                    registry.set(name.clone(), table.clone())?;
                    table
                }
                Value::Table(table) => table,
                other => {
                    return Err(mlua::Error::runtime(format!(
                        "event listener registry entry '{}' has invalid type {:?}",
                        name, other
                    )));
                }
            };

            let next_index = listeners.raw_len() + 1;
            listeners.raw_set(next_index, wrap_registered_callback(lua, handler)?)?;
            Ok(())
        })?,
    )?;

    lua.globals().set(
        "emit",
        lua.create_function(|lua, (name, payload): (String, Option<Value>)| {
            let registry = ensure_event_listener_registry(lua)?;
            let listeners = match registry.get::<Value>(name.clone())? {
                Value::Nil => return Ok(0usize),
                Value::Table(table) => table,
                other => {
                    return Err(mlua::Error::runtime(format!(
                        "event listener registry entry '{}' has invalid type {:?}",
                        name, other
                    )));
                }
            };

            let payload = match payload {
                None | Some(Value::Nil) => JsonValue::Object(JsonMap::new()),
                Some(value) => lua
                    .from_value::<JsonValue>(value)
                    .map_err(|err| mlua::Error::runtime(err.to_string()))?,
            };

            let event_ctx = lua.create_table()?;
            event_ctx.set("name", name)?;

            let count = listeners.raw_len();
            for index in 1..=count {
                let listener: Function = listeners.raw_get(index)?;
                let payload_value = lua.to_value(&payload)?;
                listener.call::<()>((event_ctx.clone(), payload_value))?;
            }

            Ok(count)
        })?,
    )?;

    Ok(())
}

fn ensure_event_listener_registry(lua: &Lua) -> LuaResult<Table> {
    let globals = lua.globals();
    if !globals.contains_key(EVENT_LISTENER_REGISTRY_KEY)? {
        globals.set(EVENT_LISTENER_REGISTRY_KEY, lua.create_table()?)?;
    }
    globals.get(EVENT_LISTENER_REGISTRY_KEY)
}
