use mlua::{Function, Lua, Result as LuaResult, Table, Value};

use crate::harness::dx::common::call_and_raise_on_err;

#[derive(Clone)]
struct GraphRuntimeFns {
    node_create: Function,
    edge_create: Function,
    path_select: Function,
}

fn friendly_graph_kind(kind: &str) -> &str {
    match kind {
        "node" => "graph_node",
        "branch" => "branch_head",
        other => other,
    }
}

fn graph_ref_table(lua: &Lua, kind: &str, id: &str) -> LuaResult<Table> {
    if id.is_empty() {
        return Err(mlua::Error::runtime("graph ref id must not be empty"));
    }
    let table = lua.create_table()?;
    table.set("kind", friendly_graph_kind(kind))?;
    table.set("id", id)?;
    Ok(table)
}

fn graph_ref_from_value(lua: &Lua, value: Value) -> LuaResult<Table> {
    match value {
        Value::Table(table) => {
            if let Ok(kind) = table.get::<String>("kind")
                && let Ok(id) = table.get::<String>("id")
                && !kind.is_empty()
                && !id.is_empty()
            {
                return Ok(table);
            }

            if let Ok(node_id) = table.get::<String>("node_id")
                && !node_id.is_empty()
            {
                return graph_ref_table(lua, "graph_node", &node_id);
            }

            if let Ok(branch_id) = table.get::<String>("branch_id")
                && !branch_id.is_empty()
            {
                let ref_table = graph_ref_table(lua, "branch_head", &branch_id)?;
                ref_table.set("branch_id", branch_id)?;
                return Ok(ref_table);
            }

            if let Ok(turn_id) = table.get::<i64>("turn_id") {
                let turn_str = turn_id.to_string();
                let ref_table = graph_ref_table(lua, "turn", &turn_str)?;
                ref_table.set("turn_id", turn_id)?;
                return Ok(ref_table);
            }

            Err(mlua::Error::runtime(
                "graph ref requires kind/id, node_id, branch_id, or turn_id".to_string(),
            ))
        }
        Value::String(text) => graph_ref_table(lua, "external_path", text.to_str()?.as_ref()),
        Value::Integer(turn_id) => {
            let turn_str = turn_id.to_string();
            let ref_table = graph_ref_table(lua, "turn", &turn_str)?;
            ref_table.set("turn_id", turn_id)?;
            Ok(ref_table)
        }
        Value::Number(number) if number.is_finite() && number.fract() == 0.0 => {
            let turn_id = number as i64;
            if (turn_id as f64) != number {
                return Err(mlua::Error::runtime(
                    "turn ref number is out of i64 range".to_string(),
                ));
            }
            let turn_str = turn_id.to_string();
            let ref_table = graph_ref_table(lua, "turn", &turn_str)?;
            ref_table.set("turn_id", turn_id)?;
            Ok(ref_table)
        }
        other => Err(mlua::Error::runtime(format!(
            "unsupported graph ref value: {:?}",
            other
        ))),
    }
}

fn parse_graph_new_args(
    kind: String,
    label_or_opts: Option<Value>,
    opts: Option<Table>,
) -> LuaResult<(String, Option<String>, Option<Table>)> {
    let kind = friendly_graph_kind(&kind).to_string();
    if kind == "graph_node" || kind == "branch_head" || kind == "turn" {
        return Err(mlua::Error::runtime(
            "graph.new(kind, ...) expects a semantic node kind such as 'experiment'".to_string(),
        ));
    }
    match label_or_opts {
        None | Some(Value::Nil) => Ok((kind, None, opts)),
        Some(Value::String(label)) => Ok((kind, Some(label.to_str()?.to_string()), opts)),
        Some(Value::Table(table)) => {
            if opts.is_some() {
                return Err(mlua::Error::runtime(
                    "graph.new(kind, label?, opts?) received both table label and opts".to_string(),
                ));
            }
            Ok((kind, None, Some(table)))
        }
        Some(other) => Err(mlua::Error::runtime(format!(
            "graph.new(kind, label?, opts?) expected string label or opts table, got {:?}",
            other
        ))),
    }
}

fn parse_role_or_opts(lua: &Lua, role_or_opts: Option<Value>) -> LuaResult<Option<Table>> {
    match role_or_opts {
        None | Some(Value::Nil) => Ok(None),
        Some(Value::String(role)) => {
            let table = lua.create_table()?;
            table.set("role", role.to_str()?.to_string())?;
            Ok(Some(table))
        }
        Some(Value::Table(table)) => Ok(Some(table)),
        Some(other) => Err(mlua::Error::runtime(format!(
            "expected role string or opts table, got {:?}",
            other
        ))),
    }
}

fn normalize_path_select_opts(lua: &Lua, source: &Table, opts: Option<Table>) -> LuaResult<Table> {
    let out = opts.unwrap_or(lua.create_table()?);

    if !matches!(out.get::<Value>("source")?, Value::Nil) {
        return Err(mlua::Error::runtime(
            "graph node path helpers do not accept source; it is implied by the node".to_string(),
        ));
    }
    if !matches!(out.get::<Value>("refs")?, Value::Nil) {
        return Err(mlua::Error::runtime(
            "graph node path helpers do not accept refs; use runtime.graph.path.select for explicit refs"
                .to_string(),
        ));
    }

    if matches!(out.get::<Value>("target_kind")?, Value::Nil) {
        match out.get::<Value>("target")? {
            Value::String(kind) => {
                out.set("target_kind", friendly_graph_kind(kind.to_str()?.as_ref()))?
            }
            Value::Nil => out.set("target_kind", "branch_head")?,
            other => {
                return Err(mlua::Error::runtime(format!(
                    "graph path helper target must be a string kind, got {:?}",
                    other
                )));
            }
        }
    }

    if matches!(out.get::<Value>("relation_kind")?, Value::Nil)
        && let Value::String(relation) = out.get::<Value>("relation")?
    {
        out.set("relation_kind", relation.to_str()?.to_string())?;
    }

    if matches!(out.get::<Value>("target_role")?, Value::Nil)
        && let Value::String(role) = out.get::<Value>("role")?
    {
        out.set("target_role", role.to_str()?.to_string())?;
    }

    out.set("source", source.clone())?;
    Ok(out)
}

fn create_node_proxy(lua: &Lua, seed: Table, fns: GraphRuntimeFns) -> LuaResult<Table> {
    let proxy = lua.create_table()?;

    let node_id = if let Ok(node_id) = seed.get::<String>("node_id") {
        node_id
    } else if let Ok(id) = seed.get::<String>("id") {
        id
    } else {
        return Err(mlua::Error::runtime(
            "graph node proxy requires node_id or id".to_string(),
        ));
    };

    proxy.set("kind", "graph_node")?;
    proxy.set("id", node_id.clone())?;
    proxy.set("node_id", node_id.clone())?;

    if let Ok(node_kind) = seed.get::<String>("kind")
        && node_kind != "graph_node"
    {
        proxy.set("node_kind", node_kind)?;
    }
    if let Ok(row_id) = seed.get::<Value>("id") {
        match row_id {
            Value::Integer(_) | Value::Number(_) => proxy.set("row_id", row_id)?,
            _ => {}
        }
    }
    for key in [
        "label",
        "metadata",
        "origin_task_id",
        "origin_execution_id",
        "created_at",
        "session_internal_id",
    ] {
        let value = seed.get::<Value>(key)?;
        if !matches!(value, Value::Nil) {
            proxy.set(key, value)?;
        }
    }

    {
        let edge_create = fns.edge_create.clone();
        proxy.set(
            "link",
            lua.create_function(
                move |lua,
                      (self_table, target, relation, opts): (
                    Table,
                    Value,
                    String,
                    Option<Table>,
                )| {
                    let target = graph_ref_from_value(lua, target)?;
                    let out = opts.unwrap_or(lua.create_table()?);
                    out.set("source", self_table)?;
                    out.set("target", target)?;
                    out.set("relation_kind", relation)?;
                    if matches!(out.get::<Value>("target_role")?, Value::Nil)
                        && let Value::String(role) = out.get::<Value>("role")?
                    {
                        out.set("target_role", role.to_str()?.to_string())?;
                    }
                    call_and_raise_on_err(lua, &edge_create, out, "graph.node.link")
                },
            )?,
        )?;
    }

    {
        let edge_create = fns.edge_create.clone();
        proxy.set(
            "add",
            lua.create_function(
                move |lua, (self_table, target, role_or_opts): (Table, Value, Option<Value>)| {
                    let target = graph_ref_from_value(lua, target)?;
                    let out = parse_role_or_opts(lua, role_or_opts)?.unwrap_or(lua.create_table()?);
                    out.set("source", self_table)?;
                    out.set("target", target)?;
                    if matches!(out.get::<Value>("relation_kind")?, Value::Nil) {
                        out.set("relation_kind", "contains")?;
                    }
                    if matches!(out.get::<Value>("target_role")?, Value::Nil)
                        && let Value::String(role) = out.get::<Value>("role")?
                    {
                        out.set("target_role", role.to_str()?.to_string())?;
                    }
                    call_and_raise_on_err(lua, &edge_create, out, "graph.node.add")
                },
            )?,
        )?;
    }

    {
        let path_select = fns.path_select.clone();
        proxy.set(
            "find",
            lua.create_function(
                move |lua, (self_table, role_or_opts): (Table, Option<Value>)| {
                    let opts = normalize_path_select_opts(
                        lua,
                        &self_table,
                        parse_role_or_opts(lua, role_or_opts)?,
                    )?;
                    call_and_raise_on_err(lua, &path_select, opts, "graph.node.find")
                },
            )?,
        )?;
    }

    {
        let path_select = fns.path_select.clone();
        proxy.set(
            "newest",
            lua.create_function(
                move |lua, (self_table, role_or_opts): (Table, Option<Value>)| {
                    let opts = normalize_path_select_opts(
                        lua,
                        &self_table,
                        parse_role_or_opts(lua, role_or_opts)?,
                    )?;
                    opts.set("order", "newest_first")?;
                    if matches!(opts.get::<Value>("limit")?, Value::Nil) {
                        opts.set("limit", 1)?;
                    }
                    call_and_raise_on_err(lua, &path_select, opts, "graph.node.newest")
                },
            )?,
        )?;
    }

    {
        let path_select = fns.path_select.clone();
        proxy.set(
            "oldest",
            lua.create_function(
                move |lua, (self_table, role_or_opts): (Table, Option<Value>)| {
                    let opts = normalize_path_select_opts(
                        lua,
                        &self_table,
                        parse_role_or_opts(lua, role_or_opts)?,
                    )?;
                    opts.set("order", "oldest_first")?;
                    if matches!(opts.get::<Value>("limit")?, Value::Nil) {
                        opts.set("limit", 1)?;
                    }
                    call_and_raise_on_err(lua, &path_select, opts, "graph.node.oldest")
                },
            )?,
        )?;
    }

    {
        let path_select = fns.path_select.clone();
        proxy.set(
            "all",
            lua.create_function(
                move |lua, (self_table, role_or_opts): (Table, Option<Value>)| {
                    let opts = normalize_path_select_opts(
                        lua,
                        &self_table,
                        parse_role_or_opts(lua, role_or_opts)?,
                    )?;
                    if matches!(opts.get::<Value>("order")?, Value::Nil) {
                        opts.set("order", "oldest_first")?;
                    }
                    call_and_raise_on_err(lua, &path_select, opts, "graph.node.all")
                },
            )?,
        )?;
    }

    Ok(proxy)
}

pub fn register_graph_dx(lua: &Lua) -> LuaResult<()> {
    let globals = lua.globals();
    let runtime: Table = globals.get("runtime")?;
    let runtime_graph: Table = runtime.get("graph")?;
    let runtime_graph_node: Table = runtime_graph.get("node")?;
    let runtime_graph_edge: Table = runtime_graph.get("edge")?;
    let runtime_graph_path: Table = runtime_graph.get("path")?;

    let fns = GraphRuntimeFns {
        node_create: runtime_graph_node.get("create")?,
        edge_create: runtime_graph_edge.get("create")?,
        path_select: runtime_graph_path.get("select")?,
    };

    let graph = lua.create_table()?;

    {
        let fns = fns.clone();
        graph.set(
            "new",
            lua.create_function(
                move |lua, (kind, label_or_opts, opts): (String, Option<Value>, Option<Table>)| {
                    let (kind, label, opts) = parse_graph_new_args(kind, label_or_opts, opts)?;
                    let out = opts.unwrap_or(lua.create_table()?);
                    out.set("kind", kind)?;
                    if let Some(label) = label {
                        out.set("label", label)?;
                    }
                    let created = call_and_raise_on_err(lua, &fns.node_create, out, "graph.new")?;
                    match created {
                        Value::Table(table) => {
                            Ok(Value::Table(create_node_proxy(lua, table, fns.clone())?))
                        }
                        other => Err(mlua::Error::runtime(format!(
                            "[graph.new] expected table from runtime.graph.node.create, got {:?}",
                            other
                        ))),
                    }
                },
            )?,
        )?;
    }

    {
        let fns = fns.clone();
        graph.set(
            "node",
            lua.create_function(move |lua, value: Value| {
                let seed = match value {
                    Value::Table(table) => table,
                    Value::String(text) => {
                        let table = lua.create_table()?;
                        table.set("node_id", text.to_str()?.to_string())?;
                        table
                    }
                    other => {
                        return Err(mlua::Error::runtime(format!(
                            "graph.node(...) expected node_id string or node table, got {:?}",
                            other
                        )));
                    }
                };
                Ok(Value::Table(create_node_proxy(lua, seed, fns.clone())?))
            })?,
        )?;
    }

    graph.set(
        "branch",
        lua.create_function(move |lua, value: Value| {
            let branch = match value {
                Value::Table(table) => table,
                Value::String(text) => {
                    let table = lua.create_table()?;
                    table.set("branch_id", text.to_str()?.to_string())?;
                    table
                }
                other => {
                    return Err(mlua::Error::runtime(format!(
                        "graph.branch(...) expected branch_id string or branch table, got {:?}",
                        other
                    )));
                }
            };
            let ref_table = graph_ref_from_value(lua, Value::Table(branch))?;
            if !matches!(ref_table.get::<Value>("branch_id")?, Value::Nil) {
                return Ok(Value::Table(ref_table));
            }
            let branch_id = ref_table.get::<String>("id")?;
            ref_table.set("branch_id", branch_id)?;
            Ok(Value::Table(ref_table))
        })?,
    )?;

    graph.set(
        "turn",
        lua.create_function(move |lua, value: Value| {
            let ref_table = graph_ref_from_value(lua, value)?;
            if !matches!(ref_table.get::<Value>("turn_id")?, Value::Nil) {
                return Ok(Value::Table(ref_table));
            }
            let turn_id = ref_table.get::<String>("id")?;
            ref_table.set("turn_id", turn_id)?;
            Ok(Value::Table(ref_table))
        })?,
    )?;

    graph.set(
        "ref",
        lua.create_function(move |lua, (kind, value): (String, Value)| match value {
            Value::String(text) => Ok(Value::Table(graph_ref_table(
                lua,
                &kind,
                text.to_str()?.as_ref(),
            )?)),
            Value::Integer(i) => {
                let ref_table = graph_ref_table(lua, &kind, &i.to_string())?;
                Ok(Value::Table(ref_table))
            }
            Value::Number(n) if n.is_finite() && n.fract() == 0.0 => {
                let i = n as i64;
                if (i as f64) != n {
                    return Err(mlua::Error::runtime(
                        "graph.ref number is out of i64 range".to_string(),
                    ));
                }
                let ref_table = graph_ref_table(lua, &kind, &i.to_string())?;
                Ok(Value::Table(ref_table))
            }
            Value::Table(table) => {
                let id = if let Ok(id) = table.get::<String>("id") {
                    id
                } else if let Ok(id) = table.get::<String>("node_id") {
                    id
                } else if let Ok(id) = table.get::<String>("branch_id") {
                    id
                } else if let Ok(id) = table.get::<i64>("turn_id") {
                    id.to_string()
                } else {
                    return Err(mlua::Error::runtime(
                        "graph.ref(kind, table) requires id, node_id, branch_id, or turn_id"
                            .to_string(),
                    ));
                };
                Ok(Value::Table(graph_ref_table(lua, &kind, &id)?))
            }
            other => Err(mlua::Error::runtime(format!(
                "graph.ref(kind, id) expected string, integer, or table, got {:?}",
                other
            ))),
        })?,
    )?;

    globals.set("graph", graph)?;
    Ok(())
}
