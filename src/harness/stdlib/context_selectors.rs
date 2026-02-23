use mlua::{Lua, MultiValue, Result as LuaResult, Table, Value};

use crate::harness::globals::{HarnessAppData, get_active_identity};
use crate::kernel::identity::ContextSelector;

pub(crate) fn normalize_selector(mut selector: ContextSelector) -> anyhow::Result<ContextSelector> {
    if selector.namespace.trim().is_empty() {
        selector.namespace = "default".to_string();
    }
    if selector.visibility.trim().is_empty() {
        selector.visibility = "private".to_string();
    }

    let mut tags = Vec::new();
    for tag in selector.tags {
        let t = tag.trim();
        if t.is_empty() {
            continue;
        }
        if !t.contains(':') {
            anyhow::bail!("invalid tag '{}': expected 'dimension:value'", t);
        }
        tags.push(t.to_string());
    }
    tags.sort();
    tags.dedup();
    selector.tags = tags;
    Ok(selector)
}

pub(crate) fn table_to_selector(ctx_tbl: Table) -> LuaResult<ContextSelector> {
    let mut tags = Vec::new();
    if let Ok(Value::Table(tags_tbl)) = ctx_tbl.get::<Value>("tags") {
        for pair in tags_tbl.sequence_values::<String>() {
            tags.push(pair?);
        }
    }

    let namespace = ctx_tbl
        .get::<String>("namespace")
        .unwrap_or_else(|_| "default".to_string());
    let visibility = ctx_tbl
        .get::<String>("visibility")
        .unwrap_or_else(|_| "private".to_string());

    let selector = ContextSelector {
        tags,
        namespace,
        visibility,
    };
    normalize_selector(selector).map_err(mlua::Error::runtime)
}

pub(crate) fn selector_to_lua_table(lua: &Lua, selector: &ContextSelector) -> LuaResult<Table> {
    let ctx = lua.create_table()?;
    ctx.set("tags", lua.create_sequence_from(selector.tags.clone())?)?;
    ctx.set("namespace", selector.namespace.clone())?;
    ctx.set("visibility", selector.visibility.clone())?;
    Ok(ctx)
}

fn context_opts_to_selector(
    scope: &str,
    id: Option<String>,
    opts: Option<Table>,
) -> LuaResult<ContextSelector> {
    let tag = if scope == "global" {
        format!("global:{}", id.unwrap_or_else(|| "*".to_string()))
    } else {
        let id = id.ok_or_else(|| {
            mlua::Error::runtime(format!("runtime.context('{}', ...) requires an id", scope))
        })?;
        format!("{}:{}", scope, id)
    };

    let mut selector = ContextSelector {
        tags: vec![tag],
        namespace: "default".to_string(),
        visibility: "private".to_string(),
    };

    if let Some(opts_tbl) = opts {
        if let Ok(ns) = opts_tbl.get::<String>("namespace") {
            selector.namespace = ns;
        }
        if let Ok(vis) = opts_tbl.get::<String>("visibility") {
            selector.visibility = vis;
        }
    }

    normalize_selector(selector).map_err(mlua::Error::runtime)
}

pub(crate) fn parse_context_args(_lua: &Lua, args: MultiValue) -> LuaResult<ContextSelector> {
    let mut it = args.into_iter();
    let first = it.next().unwrap_or(Value::Nil);
    match first {
        Value::Table(tbl) => table_to_selector(tbl),
        Value::String(scope) => {
            let scope = scope.to_str()?.to_string();
            let second = it.next().unwrap_or(Value::Nil);
            let third = it.next().unwrap_or(Value::Nil);

            let (id, opts) = match (second, third) {
                (Value::Nil, Value::Nil) => (None, None),
                (Value::Nil, Value::Table(opts)) => (None, Some(opts)),
                (Value::String(id), Value::Nil) => (Some(id.to_str()?.to_string()), None),
                (Value::String(id), Value::Table(opts)) => {
                    (Some(id.to_str()?.to_string()), Some(opts))
                }
                (Value::Table(opts), Value::Nil) => (None, Some(opts)),
                _ => {
                    return Err(mlua::Error::runtime(
                        "runtime.context invalid signature; expected (scope, id?, opts?) or ({tags=...})",
                    ));
                }
            };
            context_opts_to_selector(&scope, id, opts)
        }
        _ => Err(mlua::Error::runtime(
            "runtime.context invalid signature; expected (scope, id?, opts?) or ({tags=...})",
        )),
    }
}

fn selector_from_active_scope(
    app_data: &HarnessAppData,
    scope: &str,
) -> anyhow::Result<ContextSelector> {
    let identity = get_active_identity(app_data)?;
    let selector = match scope {
        "agent" => ContextSelector {
            tags: vec![format!("agent:{}", identity.agent_id())],
            namespace: "default".to_string(),
            visibility: "private".to_string(),
        },
        "session" => ContextSelector {
            tags: vec![format!("session:{}", identity.session_id())],
            namespace: "default".to_string(),
            visibility: "private".to_string(),
        },
        "user" => {
            let user_id = identity
                .user_id()
                .ok_or_else(|| anyhow::anyhow!("user.* requires RuntimeIdentity.user_id"))?;
            ContextSelector {
                tags: vec![format!("user:{}", user_id)],
                namespace: "default".to_string(),
                visibility: "private".to_string(),
            }
        }
        _ => anyhow::bail!("Unsupported implicit scope: {}", scope),
    };
    normalize_selector(selector)
}

pub(crate) fn selector_from_active_scope_lua(
    app_data: &HarnessAppData,
    scope: &'static str,
) -> LuaResult<ContextSelector> {
    selector_from_active_scope(app_data, scope).map_err(|e| mlua::Error::runtime(e.to_string()))
}

pub(crate) fn search_limit_from_opt(arg: Option<Value>) -> LuaResult<usize> {
    match arg {
        None | Some(Value::Nil) => Ok(5),
        Some(Value::Integer(i)) => Ok(i.max(0) as usize),
        Some(Value::Number(n)) => Ok(n.max(0.0) as usize),
        Some(Value::Table(t)) => {
            let limit = t.get::<usize>("limit").unwrap_or(5);
            Ok(limit)
        }
        Some(_) => Err(mlua::Error::runtime(
            "invalid opts; expected number limit or options table",
        )),
    }
}
