use mlua::{Function, Lua, Result as LuaResult, Table, Value};
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::harness::stdlib::binding_common::wrap_registered_callback;
use crate::harness::stdlib::object_refs;
use crate::harness::stdlib::system_globals::ensure_load_time;
use crate::signal_topics::{signal_topic_subscription_candidates, validate_signal_topic_pattern};

const EVENT_LISTENER_REGISTRY_KEY: &str = "__harness_event_listeners";

pub fn register_event_globals(lua: &Lua) -> LuaResult<()> {
    lua.globals().set(
        "on",
        lua.create_function(|lua, (name, handler): (String, Function)| {
            ensure_load_time(lua, "on")?;
            validate_signal_topic_pattern(&name).map_err(mlua::Error::runtime)?;

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
            let payload = match payload {
                None | Some(Value::Nil) => JsonValue::Object(JsonMap::new()),
                Some(value) => object_refs::encode_lua_payload(lua, value)?,
            };

            let event_ctx = lua.create_table()?;
            event_ctx.set("name", name.clone())?;

            let mut invoked = 0usize;
            for pattern in signal_topic_subscription_candidates(&name) {
                let listeners = match registry.get::<Value>(pattern.clone())? {
                    Value::Nil => continue,
                    Value::Table(table) => table,
                    other => {
                        return Err(mlua::Error::runtime(format!(
                            "event listener registry entry '{}' has invalid type {:?}",
                            pattern, other
                        )));
                    }
                };
                for index in 1..=listeners.raw_len() {
                    let listener: Function = listeners.raw_get(index)?;
                    let payload_value = object_refs::decode_json_payload(lua, &payload)?;
                    listener.call::<()>((payload_value, event_ctx.clone()))?;
                    invoked += 1;
                }
            }

            Ok(invoked)
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
