use mlua::{Function, Lua, LuaSerdeExt, Result as LuaResult, Table, Value};
use serde::Deserialize;
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::harness::globals::HarnessAppData;
use crate::harness::stdlib::binding_common::{bridge_async_display_err, wrap_registered_callback};
use crate::harness::stdlib::governance_support::{
    current_agent_id, require_capability, require_child_agent,
};
use crate::harness::stdlib::system_globals::ensure_load_time;
use crate::persistence::schema::SignalRow;
use crate::persistence::state::SignalInsert;

const RUNTIME_SIGNAL_LISTENER_REGISTRY_KEY: &str = "__harness_runtime_signal_listeners";
const RUNTIME_SIGNAL_TOPIC_REGISTRY_KEY: &str = "__harness_runtime_signal_topics";

#[derive(Debug, Deserialize, Default)]
struct LuaRuntimeSignalListOpts {
    #[serde(default)]
    topic: Option<String>,
    #[serde(default)]
    source_agent: Option<String>,
    #[serde(default)]
    target_agent: Option<String>,
    #[serde(default)]
    agent: Option<String>,
    #[serde(default)]
    limit: Option<u32>,
}

pub fn register_runtime_signal_namespace(
    lua: &Lua,
    runtime_table: &Table,
    app_data: &HarnessAppData,
) -> LuaResult<()> {
    let signals_ns = lua.create_table()?;

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

                let source_agent_id = current_agent_id(&app_data);

                let Some(runtime_scheduler) = app_data.scheduler.clone() else {
                    return Err(mlua::Error::runtime(
                        "runtime.emit requires daemon runtime coordination".to_string(),
                    ));
                };
                let runtime_store = runtime_scheduler.runtime_store();
                let subscriber_topic = topic.clone();
                let subscribers = bridge_async_display_err(async move {
                    runtime_store
                        .list_signal_subscriber_agent_ids(&subscriber_topic)
                        .await
                        .map_err(|e| e.to_string())
                })
                .map_err(mlua::Error::runtime)?;

                let mut eligible = Vec::new();
                for agent_id in subscribers {
                    if let Err(err) = require_child_agent(&app_data, &agent_id) {
                        return Err(mlua::Error::runtime(err));
                    }
                    eligible.push(agent_id);
                }

                let agent_manager = app_data.agent_manager.clone();
                let emitted_count = bridge_async_display_err(async move {
                    let store = runtime_scheduler.runtime_store();
                    for target_agent_id in &eligible {
                        store
                            .insert_signal(SignalInsert {
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

    {
        let app_data = app_data.clone();
        signals_ns.set(
            "subscribers",
            lua.create_function(move |lua, topic: String| {
                require_capability(&app_data, "runtime.db.query").map_err(mlua::Error::runtime)?;

                let Some(runtime_scheduler) = app_data.scheduler.clone() else {
                    return Err(mlua::Error::runtime(
                        "runtime.signals.subscribers requires daemon runtime coordination"
                            .to_string(),
                    ));
                };

                let runtime_store = runtime_scheduler.runtime_store();
                let rows = bridge_async_display_err(async move {
                    runtime_store
                        .list_signal_subscriber_agent_ids(&topic)
                        .await
                        .map_err(|e| e.to_string())
                })
                .map_err(mlua::Error::runtime)?;

                lua.to_value(&rows)
            })?,
        )?;
    }

    {
        let app_data = app_data.clone();
        signals_ns.set(
            "list",
            lua.create_function(move |lua, opts: Option<Table>| {
                require_capability(&app_data, "runtime.db.query").map_err(mlua::Error::runtime)?;

                let Some(runtime_scheduler) = app_data.scheduler.clone() else {
                    return Err(mlua::Error::runtime(
                        "runtime.signals.list requires daemon runtime coordination".to_string(),
                    ));
                };

                let parsed = match opts {
                    Some(opts) => lua
                        .from_value::<LuaRuntimeSignalListOpts>(Value::Table(opts))
                        .map_err(|e| {
                            mlua::Error::runtime(format!(
                                "invalid runtime.signals.list opts: {}",
                                e
                            ))
                        })?,
                    None => LuaRuntimeSignalListOpts::default(),
                };

                let target_agent = parsed.target_agent.or(parsed.agent);
                let limit = parsed.limit.unwrap_or(100) as usize;
                let runtime_store = runtime_scheduler.runtime_store();
                let rows = bridge_async_display_err(async move {
                    runtime_store
                        .list_signals(
                            parsed.topic.as_deref(),
                            parsed.source_agent.as_deref(),
                            target_agent.as_deref(),
                            limit,
                        )
                        .await
                        .map_err(|e| e.to_string())
                })
                .map_err(mlua::Error::runtime)?;

                let out: Vec<JsonValue> = rows.iter().map(signal_row_to_json).collect();
                lua.to_value(&out)
            })?,
        )?;
    }

    runtime_table.set("signals", signals_ns)?;
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

pub(crate) fn dispatch_runtime_signal(lua: &Lua, signal: &SignalRow) -> Result<usize, mlua::Error> {
    let registry = ensure_runtime_signal_listener_registry(lua)?;
    let listeners = match registry.get::<Value>(signal.topic.clone())? {
        Value::Nil => return Ok(0),
        Value::Table(table) => table,
        other => {
            return Err(mlua::Error::runtime(format!(
                "runtime signal listener registry entry '{}' has invalid type {:?}",
                signal.topic, other
            )));
        }
    };

    let event_ctx = lua.create_table()?;
    event_ctx.set("name", signal.topic.clone())?;
    event_ctx.set(
        "signal_id",
        uuid::Uuid::from_slice(&signal.public_id)
            .map(|uuid| uuid.to_string())
            .unwrap_or_else(|_| "<invalid>".to_string()),
    )?;
    event_ctx.set("source_agent_id", signal.source_agent_id.clone())?;
    event_ctx.set("target_agent_id", signal.target_agent_id.clone())?;
    event_ctx.set("created_at", signal.created_at.clone())?;

    let event_payload = build_runtime_signal_event_payload(lua, signal)?;

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

fn build_runtime_signal_event_payload(lua: &Lua, signal: &SignalRow) -> Result<Value, mlua::Error> {
    let payload = serde_json::from_str::<JsonValue>(&signal.payload).unwrap_or(JsonValue::Null);
    let mut base = match payload {
        JsonValue::Object(map) => map,
        other => {
            let mut map = JsonMap::new();
            map.insert("payload".to_string(), other);
            map
        }
    };
    base.insert(
        "signal_id".to_string(),
        JsonValue::String(
            uuid::Uuid::from_slice(&signal.public_id)
                .map(|uuid| uuid.to_string())
                .unwrap_or_else(|_| "<invalid>".to_string()),
        ),
    );
    base.insert("topic".to_string(), JsonValue::String(signal.topic.clone()));
    base.insert(
        "source".to_string(),
        JsonValue::String(signal.source_agent_id.clone()),
    );
    base.insert(
        "source_agent_id".to_string(),
        JsonValue::String(signal.source_agent_id.clone()),
    );
    base.insert(
        "emitted_at".to_string(),
        JsonValue::String(signal.created_at.clone()),
    );
    lua.to_value(&JsonValue::Object(base))
}

fn signal_row_to_json(signal: &SignalRow) -> JsonValue {
    let payload = serde_json::from_str::<JsonValue>(&signal.payload)
        .unwrap_or(JsonValue::String(signal.payload.clone()));
    JsonValue::Object(JsonMap::from_iter([
        (
            "signal_id".to_string(),
            JsonValue::String(
                uuid::Uuid::from_slice(&signal.public_id)
                    .map(|uuid| uuid.to_string())
                    .unwrap_or_else(|_| "<invalid>".to_string()),
            ),
        ),
        ("topic".to_string(), JsonValue::String(signal.topic.clone())),
        (
            "source_agent_id".to_string(),
            JsonValue::String(signal.source_agent_id.clone()),
        ),
        (
            "target_agent_id".to_string(),
            JsonValue::String(signal.target_agent_id.clone()),
        ),
        ("payload".to_string(), payload),
        (
            "attempt_count".to_string(),
            JsonValue::Number(serde_json::Number::from(signal.attempt_count)),
        ),
        (
            "last_attempted_at".to_string(),
            signal
                .last_attempted_at
                .clone()
                .map(JsonValue::String)
                .unwrap_or(JsonValue::Null),
        ),
        (
            "last_error".to_string(),
            signal
                .last_error
                .clone()
                .map(JsonValue::String)
                .unwrap_or(JsonValue::Null),
        ),
        (
            "created_at".to_string(),
            JsonValue::String(signal.created_at.clone()),
        ),
    ]))
}
