use mlua::{Result as LuaResult, Table, Value};
use std::collections::HashMap;

use crate::harness::stdlib::context_selectors::table_to_selector;
use crate::persistence::manager::{StorePathScope, StoreSelector};

pub(crate) enum SqlParams {
    None,
    Positional(Vec<turso::Value>),
    Named(Vec<(String, turso::Value)>),
}

fn parse_store_selector_string(s: &str) -> StoreSelector {
    if s.contains('/')
        || s.contains('\\')
        || s.starts_with('.')
        || s.ends_with(".db")
        || s.starts_with('~')
    {
        StoreSelector::Path(s.to_string())
    } else {
        StoreSelector::Alias(s.to_string())
    }
}

pub(crate) fn store_path_scope_from_snapshot(
    snapshot: &HashMap<String, serde_json::Value>,
) -> StorePathScope {
    StorePathScope::from_policy(
        snapshot
            .get("db.path_scope")
            .and_then(|value| value.as_str())
            .unwrap_or("workspace_only"),
    )
}

pub(crate) fn selector_denied_by_dynamic_open(
    snapshot: &HashMap<String, serde_json::Value>,
    selector: &StoreSelector,
) -> bool {
    matches!(selector, StoreSelector::Path(_))
        && !snapshot
            .get("db.allow_dynamic_open")
            .and_then(|value| value.as_bool())
            .unwrap_or(true)
}

pub(crate) fn selector_from_db_value(value: Value) -> LuaResult<StoreSelector> {
    match value {
        Value::String(s) => Ok(parse_store_selector_string(&s.to_str()?)),
        Value::Table(t) => {
            if let Ok(handle) = t.get::<String>("handle") {
                return Ok(StoreSelector::Handle(handle));
            }
            if let Ok(path) = t.get::<String>("path") {
                return Ok(StoreSelector::Path(path));
            }
            if let Ok(store) = t.get::<String>("store") {
                return Ok(StoreSelector::Alias(store));
            }
            if let Ok(alias) = t.get::<String>("alias") {
                return Ok(StoreSelector::Alias(alias));
            }
            if let Ok(Value::Table(selector_tbl)) = t.get::<Value>("selector") {
                let selector = table_to_selector(selector_tbl)?;
                return Ok(StoreSelector::Alias(selector.to_alias()));
            }
            let selector = table_to_selector(t)?;
            Ok(StoreSelector::Alias(selector.to_alias()))
        }
        _ => Err(mlua::Error::runtime(
            "invalid db selector; expected string or table",
        )),
    }
}

pub(crate) fn selector_from_db_opts(opts: Option<Table>) -> LuaResult<StoreSelector> {
    match opts {
        Some(t) => selector_from_db_value(Value::Table(t)),
        // Bare DB access defaults to the primary `state` store. Harnesses that need another
        // store must opt in explicitly with `store`, `path`, or a selector table.
        None => Ok(StoreSelector::Alias("state".to_string())),
    }
}

pub(crate) fn store_selector_from_fields(opts: &Table) -> LuaResult<Option<StoreSelector>> {
    let store = opts.get::<Value>("store")?;
    let path = opts.get::<Value>("path")?;
    if !matches!(store, Value::Nil) && !matches!(path, Value::Nil) {
        return Err(mlua::Error::runtime(
            "invalid opts: only one of 'store' or 'path' may be set",
        ));
    }
    if !matches!(store, Value::Nil) {
        return selector_from_db_value(store).map(Some);
    }
    if !matches!(path, Value::Nil) {
        return match path {
            Value::String(s) => Ok(Some(StoreSelector::Path(s.to_str()?.to_string()))),
            other => selector_from_db_value(other).map(Some),
        };
    }
    Ok(None)
}

fn lua_value_to_sql_param(value: Value) -> LuaResult<turso::Value> {
    match value {
        Value::Nil => Ok(turso::Value::Null),
        Value::Boolean(b) => Ok(turso::Value::Integer(if b { 1 } else { 0 })),
        Value::Integer(i) => Ok(turso::Value::Integer(i)),
        Value::Number(n) => Ok(turso::Value::Real(n)),
        Value::String(s) => Ok(turso::Value::Text(s.to_str()?.to_string())),
        _ => Err(mlua::Error::runtime(
            "invalid SQL param type; expected nil/boolean/number/string",
        )),
    }
}

pub(crate) fn lua_table_to_sql_params(tbl: Option<Table>) -> LuaResult<SqlParams> {
    let Some(tbl) = tbl else {
        return Ok(SqlParams::None);
    };

    let mut positional = Vec::<(usize, turso::Value)>::new();
    let mut named = Vec::<(String, turso::Value)>::new();
    let mut saw_integer_key = false;
    let mut saw_string_key = false;

    for pair in tbl.pairs::<Value, Value>() {
        let (k, v) = pair?;
        match k {
            Value::Integer(i) if i >= 1 => {
                saw_integer_key = true;
                positional.push((i as usize, lua_value_to_sql_param(v)?));
            }
            Value::String(s) => {
                saw_string_key = true;
                let mut name = s.to_str()?.to_string();
                if !name.starts_with(':') && !name.starts_with('@') && !name.starts_with('$') {
                    name = format!(":{}", name);
                }
                named.push((name, lua_value_to_sql_param(v)?));
            }
            _ => {
                return Err(mlua::Error::runtime(
                    "invalid SQL params key; expected array indices or string keys",
                ));
            }
        }
    }

    if saw_integer_key && saw_string_key {
        return Err(mlua::Error::runtime(
            "mixed positional and named SQL params are not supported",
        ));
    }

    if saw_integer_key {
        positional.sort_by_key(|(i, _)| *i);
        let mut out = Vec::with_capacity(positional.len());
        for (idx, val) in positional {
            if idx != out.len() + 1 {
                return Err(mlua::Error::runtime(
                    "positional SQL params must be a dense 1-based array",
                ));
            }
            out.push(val);
        }
        Ok(SqlParams::Positional(out))
    } else {
        Ok(SqlParams::Named(named))
    }
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{:02x}", b);
    }
    out
}

pub(crate) fn sql_value_to_json(value: turso::Value) -> serde_json::Value {
    match value {
        turso::Value::Null => serde_json::Value::Null,
        turso::Value::Integer(i) => serde_json::json!(i),
        turso::Value::Real(n) => serde_json::json!(n),
        turso::Value::Text(s) => serde_json::json!(s),
        turso::Value::Blob(b) => serde_json::json!({
            "__type": "blob",
            "hex": bytes_to_hex(&b),
        }),
    }
}
