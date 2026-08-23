use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};
use serde_json::{Map as JsonMap, Value as JsonValue};
use turin_types::{TaskInputContent, ToolsConfig};

use crate::harness::stdlib::binding_common::optional_lua_json;
use crate::harness::stdlib::context_selectors::table_to_selector;
use crate::harness::stdlib::object_refs;
use crate::harness::stdlib::runtime_worklist_selection as selection;
use crate::kernel::identity::ContextSelector;

pub(super) enum ScopeValue {
    Ref(String),
    Selector(ContextSelector),
}

pub(super) struct ParsedPayload {
    pub title: String,
    pub item_kind: String,
    pub prompt: Option<String>,
    pub content: Option<Vec<TaskInputContent>>,
    pub tools: Option<ToolsConfig>,
    pub conflict_policy: Option<String>,
    pub action_name: Option<String>,
    pub action_params: Option<JsonValue>,
    pub priority: i64,
    pub after_ids: Option<Vec<String>>,
    pub metadata: Option<JsonValue>,
}

pub(super) struct ParsedWorkItemQuery {
    pub where_map: Option<JsonMap<String, JsonValue>>,
    limit: Option<usize>,
}

impl ParsedWorkItemQuery {
    pub fn selection(&self, parent_item_id: Option<i64>) -> selection::WorkItemSelection<'_> {
        selection::WorkItemSelection::new(parent_item_id, self.where_map.as_ref(), self.limit)
    }
}

pub(super) fn parse_scope_value(value: Value) -> LuaResult<Option<ScopeValue>> {
    match value {
        Value::Nil => Ok(None),
        Value::String(s) => Ok(Some(ScopeValue::Ref(s.to_str()?.to_string()))),
        Value::Table(table) => {
            if let Value::Table(selector) = table.get::<Value>(object_refs::SCOPE_SELECTOR_KEY)? {
                return Ok(Some(ScopeValue::Selector(table_to_selector(selector)?)));
            }
            Ok(Some(ScopeValue::Selector(table_to_selector(table)?)))
        }
        other => Err(mlua::Error::runtime(format!(
            "runtime.worklist scope must be string or selector table, got {:?}",
            other
        ))),
    }
}

pub(super) fn parse_string_opt(table: &Table, key: &str) -> LuaResult<Option<String>> {
    match table.get::<Value>(key)? {
        Value::Nil => Ok(None),
        Value::String(s) => Ok(Some(s.to_str()?.to_string())),
        other => Err(mlua::Error::runtime(format!(
            "runtime.worklist field '{}' must be a string, got {:?}",
            key, other
        ))),
    }
}

pub(super) fn parse_i64_opt(table: &Table, key: &str) -> LuaResult<Option<i64>> {
    match table.get::<Value>(key)? {
        Value::Nil => Ok(None),
        Value::Integer(i) => Ok(Some(i)),
        Value::Number(n) if n.is_finite() && n.fract() == 0.0 => Ok(Some(n as i64)),
        other => Err(mlua::Error::runtime(format!(
            "runtime.worklist field '{}' must be an integer, got {:?}",
            key, other
        ))),
    }
}

pub(super) fn parse_present_string_opt(
    table: &Table,
    key: &str,
) -> LuaResult<Option<Option<String>>> {
    if table.contains_key(key)? {
        return parse_string_opt(table, key).map(Some);
    }
    Ok(None)
}

pub(super) fn parse_present_json_raw(
    lua: &Lua,
    table: &Table,
    key: &str,
) -> LuaResult<Option<Option<String>>> {
    if !table.contains_key(key)? {
        return Ok(None);
    }
    match table.get::<Value>(key)? {
        Value::Nil => Ok(Some(None)),
        value => serialize_json_opt(optional_lua_json(lua, value)?.as_ref())
            .map_err(mlua::Error::runtime)
            .map(Some),
    }
}

pub(super) fn parse_present_string_array_raw(
    table: &Table,
    key: &str,
) -> LuaResult<Option<Option<String>>> {
    if !table.contains_key(key)? {
        return Ok(None);
    }
    parse_json_array_strings(table.get::<Value>(key)?, key)?
        .map(|values| serde_json::to_string(&values).map_err(mlua::Error::runtime))
        .transpose()
        .map(Some)
}

fn parse_json_array_strings(value: Value, field: &str) -> LuaResult<Option<Vec<String>>> {
    match value {
        Value::Nil => Ok(None),
        Value::Table(table) => {
            let mut out = Vec::new();
            for value in table.sequence_values::<String>() {
                out.push(value?);
            }
            Ok(Some(out))
        }
        other => Err(mlua::Error::runtime(format!(
            "runtime.worklist '{}' must be an array of strings, got {:?}",
            field, other
        ))),
    }
}

pub(super) fn parse_payload(
    lua: &Lua,
    payload: Value,
    opts: Option<Table>,
) -> LuaResult<ParsedPayload> {
    let (title, item_kind, prompt);
    let mut content = None;
    let mut tools = None;
    let mut conflict_policy = None;
    let mut action_name = None;
    let mut action_params = None;
    let mut priority = None;
    let mut after_ids = None;
    let mut metadata = None;

    match payload {
        Value::String(s) => {
            let text = s.to_str()?.to_string();
            title = Some(text.clone());
            item_kind = Some("prompt".to_string());
            prompt = Some(text);
        }
        Value::Table(table) => {
            title = parse_string_opt(&table, "title")?;
            item_kind = parse_string_opt(&table, "kind")?;
            prompt = parse_string_opt(&table, "prompt")?;
            conflict_policy = parse_string_opt(&table, "conflict_policy")?;
            priority = parse_i64_opt(&table, "priority")?;
            action_name = parse_string_opt(&table, "action")?;
            content = match table.get::<Value>("content")? {
                Value::Nil => None,
                value => Some(serde_json::from_value(lua.from_value(value)?).map_err(|e| {
                    mlua::Error::runtime(format!("invalid worklist content: {}", e))
                })?),
            };
            tools =
                match table.get::<Value>("tools")? {
                    Value::Nil => None,
                    value => Some(serde_json::from_value(lua.from_value(value)?).map_err(|e| {
                        mlua::Error::runtime(format!("invalid worklist tools: {}", e))
                    })?),
                };
            action_params = match table.get::<Value>("params")? {
                Value::Nil => None,
                value => optional_lua_json(lua, value)?,
            };
            after_ids = parse_json_array_strings(table.get::<Value>("after")?, "after")?;
            metadata = match table.get::<Value>("metadata")? {
                Value::Nil => None,
                value => optional_lua_json(lua, value)?,
            };
        }
        other => {
            return Err(mlua::Error::runtime(format!(
                "worklist payload must be string or table, got {:?}",
                other
            )));
        }
    }

    if let Some(opts) = opts {
        if priority.is_none() {
            priority = parse_i64_opt(&opts, "priority")?;
        }
        if after_ids.is_none() {
            after_ids = parse_json_array_strings(opts.get::<Value>("after")?, "after")?;
        }
        if metadata.is_none() {
            metadata = match opts.get::<Value>("metadata")? {
                Value::Nil => None,
                value => optional_lua_json(lua, value)?,
            };
        }
    }

    let item_kind = match (item_kind, action_name.as_ref(), prompt.as_ref()) {
        (Some(kind), _, _) => kind,
        (None, Some(_), None) => "action".to_string(),
        (None, None, Some(_)) => "prompt".to_string(),
        (None, Some(_), Some(_)) => {
            return Err(mlua::Error::runtime(
                "worklist payload cannot define both prompt and action".to_string(),
            ));
        }
        (None, None, None) => {
            return Err(mlua::Error::runtime(
                "worklist payload requires prompt or action".to_string(),
            ));
        }
    };

    if item_kind == "prompt" && action_name.is_some() {
        return Err(mlua::Error::runtime(
            "prompt worklist payload cannot also define action".to_string(),
        ));
    }
    if item_kind == "action" && prompt.is_some() {
        return Err(mlua::Error::runtime(
            "action worklist payload cannot also define prompt".to_string(),
        ));
    }

    let title = title
        .or_else(|| prompt.clone())
        .or_else(|| action_name.clone())
        .ok_or_else(|| {
            mlua::Error::runtime("worklist payload requires title or prompt/action".to_string())
        })?;

    Ok(ParsedPayload {
        title,
        item_kind,
        prompt,
        content,
        tools,
        conflict_policy,
        action_name,
        action_params,
        priority: priority.unwrap_or(0),
        after_ids,
        metadata,
    })
}

pub(super) fn parse_where_map(
    lua: &Lua,
    opts: Option<&Table>,
) -> LuaResult<Option<JsonMap<String, JsonValue>>> {
    let Some(opts) = opts else {
        return Ok(None);
    };
    match opts.get::<Value>("where")? {
        Value::Nil => Ok(None),
        Value::Table(table) => match lua.from_value::<JsonValue>(Value::Table(table))? {
            JsonValue::Object(map) => Ok(Some(map)),
            _ => Err(mlua::Error::runtime(
                "worklist where filter must be an object-like table".to_string(),
            )),
        },
        other => Err(mlua::Error::runtime(format!(
            "worklist where filter must be a table, got {:?}",
            other
        ))),
    }
}

fn parse_limit(opts: Option<&Table>) -> LuaResult<Option<usize>> {
    let Some(opts) = opts else {
        return Ok(None);
    };
    match opts.get::<Value>("limit")? {
        Value::Nil => Ok(None),
        Value::Integer(i) if i >= 0 => Ok(Some(i as usize)),
        Value::Number(n) if n.is_finite() && n >= 0.0 && n.fract() == 0.0 => Ok(Some(n as usize)),
        other => Err(mlua::Error::runtime(format!(
            "worklist limit must be a non-negative integer, got {:?}",
            other
        ))),
    }
}

pub(super) fn parse_work_item_query(
    lua: &Lua,
    opts: Option<&Table>,
) -> LuaResult<ParsedWorkItemQuery> {
    Ok(ParsedWorkItemQuery {
        where_map: parse_where_map(lua, opts)?,
        limit: parse_limit(opts)?,
    })
}

pub(super) fn parse_stale_after_ms(opts: Option<&Table>) -> LuaResult<i64> {
    let Some(opts) = opts else {
        return Ok(300_000);
    };
    match opts.get::<Value>("stale_after_seconds")? {
        Value::Nil => Ok(300_000),
        Value::Integer(i) if i >= 0 => Ok(i.saturating_mul(1000)),
        Value::Number(n) if n.is_finite() && n >= 0.0 => {
            Ok((n * 1000.0).round().clamp(0.0, i64::MAX as f64) as i64)
        }
        other => Err(mlua::Error::runtime(format!(
            "worklist stale_after_seconds must be a non-negative number, got {:?}",
            other
        ))),
    }
}

pub(super) fn parse_bool_flag(opts: Option<&Table>, key: &str) -> LuaResult<bool> {
    let Some(opts) = opts else {
        return Ok(false);
    };
    match opts.get::<Value>(key)? {
        Value::Nil => Ok(false),
        Value::Boolean(value) => Ok(value),
        other => Err(mlua::Error::runtime(format!(
            "worklist {} must be a boolean, got {:?}",
            key, other
        ))),
    }
}

pub(super) fn parse_json_opt<T>(raw: Option<&str>) -> anyhow::Result<Option<T>>
where
    T: serde::de::DeserializeOwned,
{
    raw.map(serde_json::from_str)
        .transpose()
        .map_err(anyhow::Error::from)
}

pub(super) fn serialize_json_opt<T>(value: Option<&T>) -> anyhow::Result<Option<String>>
where
    T: serde::Serialize,
{
    value
        .map(serde_json::to_string)
        .transpose()
        .map_err(anyhow::Error::from)
}
