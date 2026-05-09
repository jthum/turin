use mlua::{Function, Lua, LuaSerdeExt, Result as LuaResult, Table, Value};
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::harness::globals::HarnessAppData;
use crate::harness::stdlib::binding_common::{bridge_async_display_err, wrap_registered_callback};
use crate::harness::stdlib::governance_support::{
    current_agent_id, require_capability, require_child_agent,
};
use crate::harness::stdlib::system_globals::ensure_load_time;
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::SignalDeliveryRow;
use crate::persistence::state::SignalDeliveryInsert;

const RUNTIME_SIGNAL_LISTENER_REGISTRY_KEY: &str = "__harness_runtime_signal_listeners";
const RUNTIME_SIGNAL_TOPIC_REGISTRY_KEY: &str = "__harness_runtime_signal_topics";

pub fn register_runtime_signal_namespace(
    lua: &Lua,
    runtime_table: &Table,
    app_data: &HarnessAppData,
) -> LuaResult<()> {
    runtime_table.set(
        "on",
        lua.create_function(|lua, (topic, handler): (String, Function)| {
            ensure_load_time(lua, "runtime.on")?;
            register_runtime_signal_listener(lua, &topic, handler)
        })?,
    )?;

    {
        let app_data = app_data.clone();
        runtime_table.set(
            "emit",
            lua.create_function(move |lua, (topic, payload): (String, Option<Value>)| {
                if let Err(err) = require_capability(&app_data, "runtime.agent.submit") {
                    return Err(mlua::Error::runtime(err));
                }

                let payload = match payload {
                    None | Some(Value::Nil) => JsonValue::Object(JsonMap::new()),
                    Some(value) => lua
                        .from_value::<JsonValue>(value)
                        .map_err(|err| mlua::Error::runtime(err.to_string()))?,
                };

                let Some(shared_runtime) = app_data.agent_manager.shared_runtime() else {
                    return Err(mlua::Error::runtime(
                        "runtime.emit requires a live kernel runtime context".to_string(),
                    ));
                };

                let subscribers = shared_runtime
                    .harness_manager
                    .agent_ids_for_runtime_signal_topic(&topic);
                let source_agent_id = current_agent_id(&app_data);

                let mut eligible = Vec::new();
                for agent_id in subscribers {
                    if let Err(err) = require_child_agent(&app_data, &agent_id) {
                        return Err(mlua::Error::runtime(err));
                    }
                    eligible.push(agent_id);
                }

                let store_manager = app_data.store_manager.clone();
                let agent_manager = app_data.agent_manager.clone();
                let emitted_count = bridge_async_display_err(async move {
                    let store = store_manager
                        .open(&StoreSelector::Alias("state".to_string()))
                        .await
                        .map_err(|e| e.to_string())?;
                    for target_agent_id in &eligible {
                        store
                            .insert_signal_delivery(SignalDeliveryInsert {
                                public_id: uuid::Uuid::now_v7().into_bytes().to_vec(),
                                topic: topic.clone(),
                                source_agent_id: source_agent_id.clone(),
                                target_agent_id: target_agent_id.clone(),
                                payload: serde_json::to_string(&payload)
                                    .map_err(|e| e.to_string())?,
                            })
                            .await
                            .map_err(|e| e.to_string())?;
                        agent_manager
                            .wake_agent(target_agent_id)
                            .await
                            .map_err(|e| e.to_string())?;
                    }
                    Ok::<usize, String>(eligible.len())
                })
                .map_err(mlua::Error::runtime)?;

                Ok(emitted_count)
            })?,
        )?;
    }

    Ok(())
}

pub(crate) fn runtime_signal_topics(lua: &Lua) -> LuaResult<Vec<String>> {
    let topics = ensure_runtime_signal_topic_registry(lua)?;
    let mut out = Vec::new();
    for pair in topics.pairs::<Value, Value>() {
        let (key, _) = pair?;
        if let Value::String(key_str) = key {
            out.push(key_str.to_str()?.to_string());
        }
    }
    out.sort();
    Ok(out)
}

pub(crate) fn dispatch_runtime_signal(
    lua: &Lua,
    delivery: &SignalDeliveryRow,
) -> Result<usize, mlua::Error> {
    let registry = ensure_runtime_signal_listener_registry(lua)?;
    let listeners = match registry.get::<Value>(delivery.topic.clone())? {
        Value::Nil => return Ok(0),
        Value::Table(table) => table,
        other => {
            return Err(mlua::Error::runtime(format!(
                "runtime signal listener registry entry '{}' has invalid type {:?}",
                delivery.topic, other
            )));
        }
    };

    let event_ctx = lua.create_table()?;
    event_ctx.set("name", delivery.topic.clone())?;
    event_ctx.set(
        "delivery_id",
        uuid::Uuid::from_slice(&delivery.public_id)
            .map(|uuid| uuid.to_string())
            .unwrap_or_else(|_| "<invalid>".to_string()),
    )?;
    event_ctx.set("source_agent_id", delivery.source_agent_id.clone())?;
    event_ctx.set("target_agent_id", delivery.target_agent_id.clone())?;
    event_ctx.set("created_at", delivery.created_at.clone())?;

    let event_payload = build_runtime_signal_event_payload(lua, delivery)?;

    let count = listeners.raw_len();
    for index in 1..=count {
        let listener: Function = listeners.raw_get(index)?;
        listener.call::<()>((event_ctx.clone(), event_payload.clone()))?;
    }
    Ok(count)
}

fn register_runtime_signal_listener(lua: &Lua, topic: &str, handler: Function) -> LuaResult<()> {
    let registry = ensure_runtime_signal_listener_registry(lua)?;
    let listeners = match registry.get::<Value>(topic)? {
        Value::Nil => {
            let table = lua.create_table()?;
            registry.set(topic, table.clone())?;
            table
        }
        Value::Table(table) => table,
        other => {
            return Err(mlua::Error::runtime(format!(
                "runtime signal listener registry entry '{}' has invalid type {:?}",
                topic, other
            )));
        }
    };

    let next_index = listeners.raw_len() + 1;
    listeners.raw_set(next_index, wrap_registered_callback(lua, handler)?)?;

    let topics = ensure_runtime_signal_topic_registry(lua)?;
    topics.set(topic, true)?;
    Ok(())
}

fn ensure_runtime_signal_listener_registry(lua: &Lua) -> LuaResult<Table> {
    let globals = lua.globals();
    if !globals.contains_key(RUNTIME_SIGNAL_LISTENER_REGISTRY_KEY)? {
        globals.set(RUNTIME_SIGNAL_LISTENER_REGISTRY_KEY, lua.create_table()?)?;
    }
    globals.get(RUNTIME_SIGNAL_LISTENER_REGISTRY_KEY)
}

fn ensure_runtime_signal_topic_registry(lua: &Lua) -> LuaResult<Table> {
    let globals = lua.globals();
    if !globals.contains_key(RUNTIME_SIGNAL_TOPIC_REGISTRY_KEY)? {
        globals.set(RUNTIME_SIGNAL_TOPIC_REGISTRY_KEY, lua.create_table()?)?;
    }
    globals.get(RUNTIME_SIGNAL_TOPIC_REGISTRY_KEY)
}

fn build_runtime_signal_event_payload(
    lua: &Lua,
    delivery: &SignalDeliveryRow,
) -> Result<Value, mlua::Error> {
    let payload = serde_json::from_str::<JsonValue>(&delivery.payload).unwrap_or(JsonValue::Null);
    let mut base = match payload {
        JsonValue::Object(map) => map,
        other => {
            let mut map = JsonMap::new();
            map.insert("payload".to_string(), other);
            map
        }
    };
    base.insert(
        "delivery_id".to_string(),
        JsonValue::String(
            uuid::Uuid::from_slice(&delivery.public_id)
                .map(|uuid| uuid.to_string())
                .unwrap_or_else(|_| "<invalid>".to_string()),
        ),
    );
    base.insert(
        "topic".to_string(),
        JsonValue::String(delivery.topic.clone()),
    );
    base.insert(
        "source".to_string(),
        JsonValue::String(delivery.source_agent_id.clone()),
    );
    base.insert(
        "source_agent_id".to_string(),
        JsonValue::String(delivery.source_agent_id.clone()),
    );
    base.insert(
        "emitted_at".to_string(),
        JsonValue::String(delivery.created_at.clone()),
    );
    lua.to_value(&JsonValue::Object(base))
}
