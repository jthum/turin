use mlua::{Function, Lua, LuaSerdeExt, Result as LuaResult, Table, Value};
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::harness::dx;
use crate::harness::globals::HarnessAppData;
use crate::harness::stdlib::runtime_worklist;
use crate::kernel::identity::ContextSelector;
use crate::persistence::manager::StoreSelector;

pub(crate) const REF_FIELD: &str = "_ref";
pub(crate) const PROXY_TYPE_KEY: &str = "__turin_proxy_type";
pub(crate) const REF_ONLY_KEY: &str = "__turin_ref_only";
pub(crate) const REF_TARGET_KEY: &str = "__turin_ref_target";
pub(crate) const TARGET_KIND_KEY: &str = "__turin_target_kind";
pub(crate) const TARGET_NAME_KEY: &str = "__turin_target_name";
pub(crate) const SCOPE_SELECTOR_KEY: &str = "__turin_scope_selector";
pub(crate) const SCOPE_KIND_KEY: &str = "__turin_scope_kind";
pub(crate) const STORE_SELECTOR_KEY: &str = "__turin_store_selector";
pub(crate) const WORKLIST_NAME_KEY: &str = "__turin_worklist_name";
pub(crate) const WORKLIST_SCOPE_REF_KEY: &str = "__turin_worklist_scope_ref";
pub(crate) const WORKLIST_PUBLIC_ID_KEY: &str = "__turin_worklist_public_id";
pub(crate) const WORKITEM_PUBLIC_ID_KEY: &str = "__turin_workitem_public_id";
pub(crate) const PROXY_METHOD_REGISTRY_KEY: &str = "__harness_proxy_methods";

const PROXY_SCOPE: &str = "scope";
const PROXY_WORKLIST: &str = "worklist";
const PROXY_WORKITEM: &str = "workitem";

#[derive(Debug, Clone)]
pub(crate) struct ProxyTarget {
    pub kind: String,
    pub name: Option<String>,
}

impl ProxyTarget {
    pub(crate) fn scope(kind: Option<String>) -> Self {
        Self {
            kind: "scope".to_string(),
            name: kind,
        }
    }

    pub(crate) fn worklist(name: Option<String>) -> Self {
        Self {
            kind: "worklist".to_string(),
            name,
        }
    }

    pub(crate) fn workitem(name: Option<String>) -> Self {
        Self {
            kind: "workitem".to_string(),
            name,
        }
    }

    pub(crate) fn key(&self) -> String {
        format!("{}:{}", self.kind, self.name.as_deref().unwrap_or("*"))
    }
}

pub(crate) fn register_ref_and_target_globals(lua: &Lua) -> LuaResult<()> {
    lua.globals().set(
        "ref",
        lua.create_function(|lua, value: Value| {
            let table = lua.create_table()?;
            table.set(REF_ONLY_KEY, true)?;
            table.set(REF_TARGET_KEY, value)?;
            Ok(Value::Table(table))
        })?,
    )?;

    let target = lua.create_table()?;
    target.set(
        "scope",
        lua.create_function(|lua, kind: Option<String>| target_table(lua, "scope", kind))?,
    )?;
    target.set(
        "worklist",
        lua.create_function(|lua, name: Option<String>| target_table(lua, "worklist", name))?,
    )?;
    target.set(
        "workitem",
        lua.create_function(|lua, name: Option<String>| target_table(lua, "workitem", name))?,
    )?;
    lua.globals().set("target", target)?;
    Ok(())
}

fn target_table(lua: &Lua, kind: &str, name: Option<String>) -> LuaResult<Table> {
    let table = lua.create_table()?;
    table.set(TARGET_KIND_KEY, kind)?;
    match name {
        Some(name) => table.set(TARGET_NAME_KEY, name)?,
        None => table.set(TARGET_NAME_KEY, Value::Nil)?,
    }
    Ok(table)
}

pub(crate) fn parse_target(value: Value) -> LuaResult<ProxyTarget> {
    match value {
        Value::String(kind) => Ok(ProxyTarget::scope(Some(kind.to_str()?.to_string()))),
        Value::Table(table) => {
            let kind = table.get::<String>(TARGET_KIND_KEY)?;
            let name = match table.get::<Value>(TARGET_NAME_KEY)? {
                Value::Nil => None,
                Value::String(name) => Some(name.to_str()?.to_string()),
                other => {
                    return Err(mlua::Error::runtime(format!(
                        "target name must be a string, got {:?}",
                        other
                    )));
                }
            };
            match kind.as_str() {
                "scope" => Ok(ProxyTarget::scope(name)),
                "worklist" => Ok(ProxyTarget::worklist(name)),
                "workitem" => Ok(ProxyTarget::workitem(name)),
                _ => Err(mlua::Error::runtime(format!(
                    "unknown action target kind '{}'",
                    kind
                ))),
            }
        }
        other => Err(mlua::Error::runtime(format!(
            "action.define_on target must be a scope string or target.* value, got {:?}",
            other
        ))),
    }
}

pub(crate) fn register_proxy_method(
    lua: &Lua,
    target: ProxyTarget,
    method: &str,
    action_name: &str,
) -> LuaResult<()> {
    let registry = ensure_proxy_method_registry(lua)?;
    let key = target.key();
    let methods = match registry.get::<Value>(key.clone())? {
        Value::Nil => {
            let methods = lua.create_table()?;
            registry.set(key, methods.clone())?;
            methods
        }
        Value::Table(methods) => methods,
        other => {
            return Err(mlua::Error::runtime(format!(
                "proxy method registry entry '{}' has invalid type {:?}",
                key, other
            )));
        }
    };
    methods.set(method, action_name)?;
    Ok(())
}

fn ensure_proxy_method_registry(lua: &Lua) -> LuaResult<Table> {
    let globals = lua.globals();
    if !globals.contains_key(PROXY_METHOD_REGISTRY_KEY)? {
        globals.set(PROXY_METHOD_REGISTRY_KEY, lua.create_table()?)?;
    }
    globals.get(PROXY_METHOD_REGISTRY_KEY)
}

pub(crate) fn annotate_scope_proxy(
    lua: &Lua,
    proxy: &Table,
    selector: &ContextSelector,
    scope_kind: Option<&str>,
) -> LuaResult<()> {
    proxy.set(PROXY_TYPE_KEY, PROXY_SCOPE)?;
    proxy.set(
        SCOPE_SELECTOR_KEY,
        lua.to_value(&selector_to_json(selector))?,
    )?;
    if let Some(kind) = scope_kind {
        proxy.set(SCOPE_KIND_KEY, kind)?;
    }
    Ok(())
}

pub(crate) fn annotate_worklist_proxy(
    lua: &Lua,
    proxy: &Table,
    store_selector: &StoreSelector,
    name: &str,
    scope_ref: &str,
    public_id: &str,
) -> LuaResult<()> {
    proxy.set(PROXY_TYPE_KEY, PROXY_WORKLIST)?;
    proxy.set(
        STORE_SELECTOR_KEY,
        lua.to_value(&store_selector_json(store_selector))?,
    )?;
    proxy.set(WORKLIST_NAME_KEY, name)?;
    proxy.set(WORKLIST_SCOPE_REF_KEY, scope_ref)?;
    proxy.set(WORKLIST_PUBLIC_ID_KEY, public_id)?;
    Ok(())
}

pub(crate) fn annotate_workitem_proxy(proxy: &Table, public_id: &str) -> LuaResult<()> {
    proxy.set(PROXY_TYPE_KEY, PROXY_WORKITEM)?;
    proxy.set(WORKITEM_PUBLIC_ID_KEY, public_id)?;
    Ok(())
}

pub(crate) fn attach_proxy_action(lua: &Lua, proxy: &Table, target: ProxyTarget) -> LuaResult<()> {
    let target_for_action = target.clone();
    proxy.set(
        "action",
        lua.create_function(
            move |lua, (self_tbl, name, params): (Table, String, Option<Value>)| {
                let action_name = resolve_contextual_action_name(&target_for_action, &name);
                run_contextual_action(lua, self_tbl, action_name, params)
            },
        )?,
    )?;
    attach_declared_methods(lua, proxy, target)
}

fn resolve_contextual_action_name(target: &ProxyTarget, name: &str) -> String {
    if let Some(stripped) = name.strip_prefix('/') {
        return stripped.to_string();
    }
    if name.contains('.') {
        return name.to_string();
    }
    match target.kind.as_str() {
        "scope" => target
            .name
            .as_ref()
            .map(|kind| format!("{kind}.{name}"))
            .unwrap_or_else(|| name.to_string()),
        "worklist" => target
            .name
            .as_ref()
            .map(|list| format!("worklist.{list}.{name}"))
            .unwrap_or_else(|| format!("worklist.{name}")),
        "workitem" => target
            .name
            .as_ref()
            .map(|list| format!("workitem.{list}.{name}"))
            .unwrap_or_else(|| format!("workitem.{name}")),
        _ => name.to_string(),
    }
}

fn run_contextual_action(
    lua: &Lua,
    subject: Table,
    action_name: String,
    params: Option<Value>,
) -> LuaResult<Value> {
    let action: Table = lua.globals().get("action")?;
    let run: Function = action.get("run")?;
    let envelope = lua.create_table()?;
    envelope.set("subject", subject)?;
    envelope.set("params", params.unwrap_or(Value::Nil))?;
    run.call((action_name, envelope))
}

fn attach_declared_methods(lua: &Lua, proxy: &Table, target: ProxyTarget) -> LuaResult<()> {
    let registry = ensure_proxy_method_registry(lua)?;
    let mut keys = Vec::new();
    keys.push(format!("{}:*", target.kind));
    if let Some(name) = target.name.as_ref() {
        keys.push(format!("{}:{}", target.kind, name));
    }

    let mut methods = std::collections::BTreeMap::<String, String>::new();
    for key in keys {
        if let Value::Table(table) = registry.get::<Value>(key)? {
            for pair in table.pairs::<String, String>() {
                let (method, action_name) = pair?;
                methods.insert(method, action_name);
            }
        }
    }

    for (method, action_name) in methods {
        let existing = proxy.get::<Value>(method.as_str())?;
        if matches!(existing, Value::Function(_)) {
            return Err(mlua::Error::runtime(format!(
                "action.define_on cannot attach '{}' because the proxy already has a method with that name",
                method
            )));
        }
        let action_name = action_name.clone();
        proxy.set(
            method,
            lua.create_function(move |lua, (self_tbl, params): (Table, Option<Value>)| {
                run_contextual_action(lua, self_tbl, action_name.clone(), params)
            })?,
        )?;
    }
    Ok(())
}

pub(crate) fn encode_lua_payload(lua: &Lua, value: Value) -> LuaResult<JsonValue> {
    encode_lua_value(lua, value)
}

fn encode_lua_value(lua: &Lua, value: Value) -> LuaResult<JsonValue> {
    match value {
        Value::Nil => Ok(JsonValue::Null),
        Value::Boolean(value) => Ok(JsonValue::Bool(value)),
        Value::Integer(value) => Ok(JsonValue::Number(value.into())),
        Value::Number(value) => serde_json::Number::from_f64(value)
            .map(JsonValue::Number)
            .ok_or_else(|| mlua::Error::runtime("cannot serialize non-finite number")),
        Value::String(value) => Ok(JsonValue::String(value.to_str()?.to_string())),
        Value::Table(table) => {
            if table.get::<bool>(REF_ONLY_KEY).unwrap_or(false) {
                let target = table.get::<Value>(REF_TARGET_KEY)?;
                return encode_ref_only_target(lua, target);
            }
            if let Some(ref_obj) = encode_proxy_ref(lua, &table)? {
                let mut object = encode_lua_table(lua, &table)?;
                object.insert(REF_FIELD.to_string(), ref_obj);
                return Ok(JsonValue::Object(object));
            }
            encode_lua_table_or_array(lua, &table)
        }
        Value::Function(_)
        | Value::Thread(_)
        | Value::UserData(_)
        | Value::LightUserData(_)
        | Value::Error(_)
        | Value::Vector(_)
        | Value::Buffer(_)
        | Value::Other(_) => Err(mlua::Error::runtime(format!(
            "value is not JSON-serializable: {:?}",
            value
        ))),
    }
}

fn encode_ref_only_target(lua: &Lua, target: Value) -> LuaResult<JsonValue> {
    match target {
        Value::Table(table) => match encode_proxy_ref(lua, &table)? {
            Some(ref_obj) => Ok(JsonValue::Object(JsonMap::from_iter([(
                REF_FIELD.to_string(),
                ref_obj,
            )]))),
            None => Err(mlua::Error::runtime(
                "ref(...) requires a reference-aware runtime proxy",
            )),
        },
        other => Err(mlua::Error::runtime(format!(
            "ref(...) requires a reference-aware runtime proxy, got {:?}",
            other
        ))),
    }
}

fn encode_lua_table(lua: &Lua, table: &Table) -> LuaResult<JsonMap<String, JsonValue>> {
    let mut map = JsonMap::new();
    for pair in table.pairs::<Value, Value>() {
        let (key, value) = pair?;
        let Value::String(key) = key else {
            continue;
        };
        let key = key.to_str()?.to_string();
        if key.starts_with("__turin_") || key == REF_FIELD {
            continue;
        }
        if matches!(
            value,
            Value::Function(_)
                | Value::Thread(_)
                | Value::UserData(_)
                | Value::LightUserData(_)
                | Value::Error(_)
                | Value::Vector(_)
                | Value::Buffer(_)
                | Value::Other(_)
        ) {
            continue;
        }
        map.insert(key, encode_lua_value(lua, value)?);
    }
    Ok(map)
}

fn encode_lua_table_or_array(lua: &Lua, table: &Table) -> LuaResult<JsonValue> {
    if is_sequence_table(table)? {
        let len = table.raw_len();
        let mut values = Vec::with_capacity(len);
        for index in 1..=len {
            values.push(encode_lua_value(lua, table.raw_get::<Value>(index)?)?);
        }
        Ok(JsonValue::Array(values))
    } else {
        Ok(JsonValue::Object(encode_lua_table(lua, table)?))
    }
}

fn is_sequence_table(table: &Table) -> LuaResult<bool> {
    let len = table.raw_len();
    if len == 0 {
        return Ok(false);
    }
    for pair in table.pairs::<Value, Value>() {
        let (key, _) = pair?;
        match key {
            Value::Integer(i) if i >= 1 && (i as usize) <= len => {}
            _ => return Ok(false),
        }
    }
    Ok(true)
}

fn encode_proxy_ref(lua: &Lua, table: &Table) -> LuaResult<Option<JsonValue>> {
    let proxy_type = match table.get::<Value>(PROXY_TYPE_KEY)? {
        Value::String(value) => value.to_str()?.to_string(),
        _ => return Ok(None),
    };
    let mut ref_obj = JsonMap::new();
    ref_obj.insert("type".to_string(), JsonValue::String(proxy_type.clone()));
    match proxy_type.as_str() {
        PROXY_SCOPE => {
            insert_encoded_lua_field(lua, &mut ref_obj, "selector", table, SCOPE_SELECTOR_KEY)?;
            if let Value::String(kind) = table.get::<Value>(SCOPE_KIND_KEY)? {
                ref_obj.insert(
                    "kind".to_string(),
                    JsonValue::String(kind.to_str()?.to_string()),
                );
            }
        }
        PROXY_WORKLIST => {
            insert_encoded_lua_field(lua, &mut ref_obj, "store", table, STORE_SELECTOR_KEY)?;
            insert_lua_string_field(&mut ref_obj, "id", table, WORKLIST_PUBLIC_ID_KEY)?;
            insert_lua_string_field(&mut ref_obj, "name", table, WORKLIST_NAME_KEY)?;
            insert_lua_string_field(&mut ref_obj, "scope_ref", table, WORKLIST_SCOPE_REF_KEY)?;
        }
        PROXY_WORKITEM => {
            insert_encoded_lua_field(lua, &mut ref_obj, "store", table, STORE_SELECTOR_KEY)?;
            insert_lua_string_field(&mut ref_obj, "id", table, WORKITEM_PUBLIC_ID_KEY)?;
            insert_lua_string_field(&mut ref_obj, "worklist_id", table, WORKLIST_PUBLIC_ID_KEY)?;
            insert_lua_string_field(&mut ref_obj, "worklist", table, WORKLIST_NAME_KEY)?;
        }
        _ => return Ok(None),
    }
    Ok(Some(JsonValue::Object(ref_obj)))
}

fn insert_encoded_lua_field(
    lua: &Lua,
    object: &mut JsonMap<String, JsonValue>,
    key: &str,
    table: &Table,
    table_key: &str,
) -> LuaResult<()> {
    let value: Value = table.get(table_key)?;
    object.insert(key.to_string(), encode_lua_value(lua, value)?);
    Ok(())
}

fn insert_lua_string_field(
    object: &mut JsonMap<String, JsonValue>,
    key: &str,
    table: &Table,
    table_key: &str,
) -> LuaResult<()> {
    object.insert(key.to_string(), JsonValue::String(table.get(table_key)?));
    Ok(())
}

pub(crate) fn decode_json_payload(lua: &Lua, value: &JsonValue) -> LuaResult<Value> {
    match value {
        JsonValue::Null => Ok(Value::Nil),
        JsonValue::Bool(value) => Ok(Value::Boolean(*value)),
        JsonValue::Number(value) => {
            if let Some(i) = value.as_i64() {
                Ok(Value::Integer(i))
            } else if let Some(f) = value.as_f64() {
                Ok(Value::Number(f))
            } else {
                Ok(Value::Nil)
            }
        }
        JsonValue::String(value) => Ok(Value::String(lua.create_string(value)?)),
        JsonValue::Array(values) => {
            let table = lua.create_table()?;
            for (index, value) in values.iter().enumerate() {
                table.set(index + 1, decode_json_payload(lua, value)?)?;
            }
            Ok(Value::Table(table))
        }
        JsonValue::Object(map) => decode_json_object(lua, map),
    }
}

fn decode_json_object(lua: &Lua, map: &JsonMap<String, JsonValue>) -> LuaResult<Value> {
    if let Some(JsonValue::Object(ref_obj)) = map.get(REF_FIELD)
        && let Some(proxy) = hydrate_ref(lua, ref_obj)?
    {
        for (key, value) in map {
            if key == REF_FIELD {
                continue;
            }
            let existing = proxy.get::<Value>(key.as_str())?;
            if matches!(existing, Value::Function(_)) {
                return Err(mlua::Error::runtime(format!(
                    "ref overlay cannot replace proxy method '{}'",
                    key
                )));
            }
            proxy.set(key.as_str(), decode_json_payload(lua, value)?)?;
        }
        return Ok(Value::Table(proxy));
    }

    let table = lua.create_table()?;
    for (key, value) in map {
        table.set(key.as_str(), decode_json_payload(lua, value)?)?;
    }
    Ok(Value::Table(table))
}

fn hydrate_ref(lua: &Lua, ref_obj: &JsonMap<String, JsonValue>) -> LuaResult<Option<Table>> {
    let Some(ref_type) = ref_obj.get("type").and_then(|value| value.as_str()) else {
        return Ok(None);
    };
    let app_data = lua
        .app_data_ref::<HarnessAppData>()
        .map(|app_data| app_data.clone())
        .ok_or_else(|| mlua::Error::runtime("Harness app data missing"))?;
    match ref_type {
        PROXY_SCOPE => {
            let Some(selector) = ref_obj.get("selector") else {
                return Ok(None);
            };
            let selector = selector_from_json(selector)?;
            let scope_kind = ref_obj.get("kind").and_then(|value| value.as_str());
            let proxy = dx::build_scope_proxy(lua, &selector, scope_kind)?;
            Ok(Some(proxy))
        }
        PROXY_WORKLIST => runtime_worklist::hydrate_worklist_ref(lua, &app_data, ref_obj)
            .map(Some)
            .map_err(mlua::Error::runtime),
        PROXY_WORKITEM => runtime_worklist::hydrate_workitem_ref(lua, &app_data, ref_obj)
            .map(Some)
            .map_err(mlua::Error::runtime),
        _ => Ok(None),
    }
}

pub(crate) fn store_selector_json(selector: &StoreSelector) -> JsonValue {
    match selector {
        StoreSelector::Alias(alias) => JsonValue::String(alias.clone()),
        StoreSelector::Path(path) => serde_json::json!({ "path": path }),
        StoreSelector::Handle(handle) => serde_json::json!({ "store": handle }),
    }
}

pub(crate) fn store_selector_from_json(value: Option<&JsonValue>) -> StoreSelector {
    match value {
        Some(JsonValue::String(alias)) => StoreSelector::Alias(alias.clone()),
        Some(JsonValue::Object(map)) => {
            if let Some(path) = map.get("path").and_then(|value| value.as_str()) {
                StoreSelector::Path(path.to_string())
            } else if let Some(store) = map
                .get("store")
                .or_else(|| map.get("handle"))
                .and_then(|value| value.as_str())
            {
                StoreSelector::Handle(store.to_string())
            } else if let Some(alias) = map.get("alias").and_then(|value| value.as_str()) {
                StoreSelector::Alias(alias.to_string())
            } else {
                StoreSelector::Alias("state".to_string())
            }
        }
        _ => StoreSelector::Alias("state".to_string()),
    }
}

fn selector_to_json(selector: &ContextSelector) -> JsonValue {
    serde_json::json!({
        "tags": selector.tags,
        "namespace": selector.namespace,
        "visibility": selector.visibility,
    })
}

fn selector_from_json(value: &JsonValue) -> LuaResult<ContextSelector> {
    let JsonValue::Object(map) = value else {
        return Err(mlua::Error::runtime("scope ref selector must be an object"));
    };
    let tags = match map.get("tags") {
        Some(JsonValue::Array(values)) => values
            .iter()
            .filter_map(|value| value.as_str().map(ToString::to_string))
            .collect(),
        _ => Vec::new(),
    };
    let namespace = map
        .get("namespace")
        .and_then(|value| value.as_str())
        .unwrap_or("default")
        .to_string();
    let visibility = map
        .get("visibility")
        .and_then(|value| value.as_str())
        .unwrap_or("private")
        .to_string();
    crate::harness::stdlib::context_selectors::normalize_selector(ContextSelector {
        tags,
        namespace,
        visibility,
    })
    .map_err(mlua::Error::runtime)
}
