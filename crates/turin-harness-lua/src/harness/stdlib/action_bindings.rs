mod builtins;
mod context;

use anyhow::Result;
use mlua::{Function, Lua, Result as LuaResult, Table, Value};
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::harness::globals::HarnessAppData;
use crate::harness::stdlib::binding_common::wrap_registered_callback;
use crate::harness::stdlib::object_refs;
use crate::harness::stdlib::system_globals::ensure_load_time;

use builtins::invoke_builtin_action;
use context::build_action_context;
pub(crate) use context::{ActionInvocationContext, ActionWorkItemContext};

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

            registry.set(name, wrap_registered_callback(lua, handler)?)?;
            Ok(())
        })?,
    )?;

    action_table.set(
        "define_on",
        lua.create_function(
            |lua, (target_value, method, handler): (Value, String, Function)| {
                ensure_load_time(lua, "action.define_on")?;
                let target = object_refs::parse_target(target_value)?;
                let action_name = action_name_for_target_method(&target, &method);

                let registry = ensure_declared_action_registry(lua)?;
                if registry.contains_key(action_name.clone())? {
                    return Err(mlua::Error::runtime(format!(
                        "action.define_on('{}', '{}') conflicts with existing action '{}'",
                        target.key(),
                        method,
                        action_name
                    )));
                }

                let handler = wrap_registered_callback(lua, handler)?;
                let method_name = method.clone();
                let action_name_for_error = action_name.clone();
                let wrapper =
                    lua.create_function(move |_lua, (ctx, envelope): (Table, Value)| {
                        let (subject, params) = match envelope {
                            Value::Table(table) if table.contains_key("subject")? => {
                                let subject = table.get::<Value>("subject")?;
                                let params = table.get::<Value>("params")?;
                                (subject, params)
                            }
                            Value::Nil => {
                                return Err(mlua::Error::runtime(format!(
                                    "action '{}' requires a subject",
                                    action_name_for_error
                                )));
                            }
                            other => (other, Value::Nil),
                        };
                        if matches!(subject, Value::Nil) {
                            return Err(mlua::Error::runtime(format!(
                                "action '{}' requires a subject",
                                action_name_for_error
                            )));
                        }
                        handler.call::<Value>((ctx, subject, params))
                    })?;

                registry.set(action_name.clone(), wrapper)?;
                object_refs::register_proxy_method(lua, target, &method_name, &action_name)?;
                Ok(())
            },
        )?,
    )?;

    action_table.set(
        "run",
        lua.create_function(|lua, (name, params): (String, Option<Value>)| {
            let params = match params {
                None | Some(Value::Nil) => JsonValue::Object(JsonMap::new()),
                Some(value) => object_refs::encode_lua_payload(lua, value)?,
            };

            let app_data = lua
                .app_data_ref::<HarnessAppData>()
                .map(|app_data| app_data.clone())
                .ok_or_else(|| mlua::Error::runtime("Harness app data missing"))?;
            let action_name = name.clone();
            let result = invoke_declared_action(
                lua,
                &name,
                params.clone(),
                ActionInvocationContext {
                    app_data,
                    action_name,
                    params,
                    work_item: None,
                },
            )
            .map_err(mlua::Error::runtime)?;

            match result {
                Some(value) => object_refs::decode_json_payload(lua, &value),
                None => Ok(Value::Nil),
            }
        })?,
    )?;

    lua.globals().set("action", action_table)?;
    Ok(())
}

fn action_name_for_target_method(target: &object_refs::ProxyTarget, method: &str) -> String {
    match target.kind.as_str() {
        "scope" => target
            .name
            .as_ref()
            .map(|kind| format!("{kind}.{method}"))
            .unwrap_or_else(|| format!("scope.{method}")),
        "worklist" => target
            .name
            .as_ref()
            .map(|name| format!("worklist.{name}.{method}"))
            .unwrap_or_else(|| format!("worklist.{method}")),
        "workitem" => target
            .name
            .as_ref()
            .map(|name| format!("workitem.{name}.{method}"))
            .unwrap_or_else(|| format!("workitem.{method}")),
        _ => method.to_string(),
    }
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
    params: JsonValue,
    invocation: ActionInvocationContext,
) -> Result<Option<JsonValue>> {
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

    let ctx = build_action_context(lua, &invocation)?;
    let lua_args = object_refs::decode_json_payload(lua, &params)?;
    let result = handler.call::<Value>((ctx, lua_args))?;
    let result = object_refs::encode_lua_payload(lua, result).map_err(|err| {
        anyhow::anyhow!(
            "declared action '{}' handler returned invalid value: {}",
            name,
            err
        )
    })?;

    Ok(Some(result))
}
