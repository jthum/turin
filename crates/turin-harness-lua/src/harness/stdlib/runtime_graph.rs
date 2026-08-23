use std::sync::Arc;

use anyhow::{Context, Result};
use mlua::{IntoLua, Lua, LuaSerdeExt, Result as LuaResult, Table, Value};

use crate::harness::globals::{ActiveHarnessExecutionContext, HarnessAppData};
use crate::harness::stdlib::binding_common::{bridge_async_display_err, lua_table_result, nil_err};
use crate::harness::stdlib::governance_support::require_capability as require_governance_capability;
use crate::kernel::session_refs::{SessionReference, parse_session_reference};
use crate::persistence::manager::{StoreManager, StoreSelector};
use crate::persistence::schema::{
    GraphEdgeCreate, GraphEdgeRow, GraphNodeRow, GraphProvenance, GraphRef,
};
use crate::persistence::state::StateStore;

struct GraphSession {
    store: Arc<StateStore>,
    internal_id: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SelectedPathOrder {
    OldestFirst,
    NewestFirst,
}

#[derive(Clone, Debug)]
struct SelectedPathSourceOptions {
    relation_kind: Option<String>,
    target_kind: Option<String>,
    target_role: Option<String>,
    order: SelectedPathOrder,
    limit: Option<usize>,
}

enum SelectedPathRequest {
    Refs(Vec<GraphRef>),
    Source(GraphRef, SelectedPathSourceOptions),
}

fn bytes_to_simple_uuid(bytes: &[u8]) -> String {
    uuid::Uuid::from_slice(bytes)
        .map(|uuid| uuid.simple().to_string())
        .unwrap_or_else(|_| {
            let mut out = String::with_capacity(bytes.len() * 2);
            for byte in bytes {
                use std::fmt::Write as _;
                let _ = write!(&mut out, "{:02x}", byte);
            }
            out
        })
}

fn opt_string(table: &Table, key: &str) -> Option<String> {
    table
        .get::<String>(key)
        .ok()
        .filter(|value| !value.is_empty())
}

fn opt_positive_usize(table: &Table, key: &str) -> LuaResult<Option<usize>> {
    match table.get::<Value>(key) {
        Ok(Value::Nil) | Err(_) => Ok(None),
        Ok(Value::Integer(value)) => {
            let value = usize::try_from(value)
                .map_err(|_| mlua::Error::runtime(format!("{key} must be a positive integer")))?;
            if value == 0 {
                return Err(mlua::Error::runtime(format!(
                    "{key} must be greater than zero"
                )));
            }
            Ok(Some(value))
        }
        Ok(_) => Err(mlua::Error::runtime(format!(
            "{key} must be a positive integer"
        ))),
    }
}

fn selected_path_order(table: &Table) -> LuaResult<SelectedPathOrder> {
    match opt_string(table, "order").as_deref() {
        None | Some("oldest_first") => Ok(SelectedPathOrder::OldestFirst),
        Some("newest_first") => Ok(SelectedPathOrder::NewestFirst),
        Some(other) => Err(mlua::Error::runtime(format!(
            "selected_path order must be 'oldest_first' or 'newest_first', got '{other}'"
        ))),
    }
}

fn selected_path_request(opts: &Table) -> LuaResult<SelectedPathRequest> {
    let refs = match opts.get::<Value>("refs") {
        Ok(Value::Nil) | Err(_) => None,
        Ok(Value::Table(table)) => Some(graph_refs_from_sequence(table)?),
        Ok(_) => {
            return Err(mlua::Error::runtime(
                "selected_path refs must be an array of graph refs",
            ));
        }
    };
    let source = match opts.get::<Value>("source") {
        Ok(Value::Nil) | Err(_) => None,
        Ok(Value::Table(table)) => Some(graph_ref_from_table(table)?),
        Ok(_) => {
            return Err(mlua::Error::runtime(
                "selected_path source must be a graph ref table",
            ));
        }
    };
    let relation_kind = opt_string(opts, "relation_kind");
    let target_kind = opt_string(opts, "target_kind");
    let target_role = opt_string(opts, "target_role");
    let order = selected_path_order(opts)?;
    let limit = opt_positive_usize(opts, "limit")?;

    match (refs, source) {
        (Some(_), Some(_)) => Err(mlua::Error::runtime(
            "selected_path accepts either refs or source, not both",
        )),
        (None, None) => Err(mlua::Error::runtime(
            "selected_path requires either refs or source",
        )),
        (Some(refs), None) => {
            if relation_kind.is_some()
                || target_kind.is_some()
                || target_role.is_some()
                || !matches!(order, SelectedPathOrder::OldestFirst)
                || limit.is_some()
            {
                return Err(mlua::Error::runtime(
                    "selected_path refs mode does not accept relation_kind, target_kind, target_role, order, or limit",
                ));
            }
            Ok(SelectedPathRequest::Refs(refs))
        }
        (None, Some(source)) => Ok(SelectedPathRequest::Source(
            source,
            SelectedPathSourceOptions {
                relation_kind,
                target_kind,
                target_role,
                order,
                limit,
            },
        )),
    }
}

fn graph_metadata(lua: &Lua, table: &Table) -> LuaResult<Option<serde_json::Value>> {
    match table.get::<Value>("metadata") {
        Ok(Value::Nil) | Err(_) => Ok(None),
        Ok(value) => lua.from_value(value).map(Some),
    }
}

fn graph_provenance(table: &Table) -> GraphProvenance {
    GraphProvenance::new(
        opt_string(table, "origin_task_id"),
        opt_string(table, "origin_execution_id"),
    )
}

fn graph_ref_from_table(table: Table) -> LuaResult<GraphRef> {
    let kind = table.get::<String>("kind")?;
    let id = table.get::<String>("id")?;
    if kind.is_empty() || id.is_empty() {
        return Err(mlua::Error::runtime(
            "graph ref requires non-empty kind and id",
        ));
    }
    Ok(GraphRef::new(kind, id))
}

fn graph_refs_from_sequence(table: Table) -> LuaResult<Vec<GraphRef>> {
    let mut refs = Vec::new();
    for value in table.sequence_values::<Table>() {
        refs.push(graph_ref_from_table(value?)?);
    }
    if refs.is_empty() {
        return Err(mlua::Error::runtime(
            "graph refs list must include at least one ref",
        ));
    }
    Ok(refs)
}

fn graph_node_to_lua(lua: &Lua, row: GraphNodeRow) -> LuaResult<Table> {
    let table = lua.create_table()?;
    table.set("id", row.id)?;
    table.set("node_id", bytes_to_simple_uuid(&row.public_id))?;
    set_optional(&table, "session_internal_id", row.session_id)?;
    table.set("kind", row.kind)?;
    set_optional(&table, "label", row.label)?;
    set_optional(&table, "origin_task_id", row.origin_task_id)?;
    set_optional(&table, "origin_execution_id", row.origin_execution_id)?;
    set_json_metadata(lua, &table, row.metadata.as_deref())?;
    table.set("created_at", row.created_at)?;
    Ok(table)
}

fn graph_ref_to_lua(lua: &Lua, graph_ref: GraphRef) -> LuaResult<Table> {
    let table = lua.create_table()?;
    table.set("kind", graph_ref.kind)?;
    table.set("id", graph_ref.id)?;
    Ok(table)
}

fn graph_edge_to_lua(lua: &Lua, row: GraphEdgeRow) -> LuaResult<Table> {
    let table = lua.create_table()?;
    table.set("id", row.id)?;
    table.set("edge_id", bytes_to_simple_uuid(&row.public_id))?;
    set_optional(&table, "session_internal_id", row.session_id)?;
    table.set("source", graph_ref_to_lua(lua, row.source)?)?;
    table.set("target", graph_ref_to_lua(lua, row.target)?)?;
    table.set("relation_kind", row.relation_kind)?;
    set_optional(&table, "source_role", row.source_role)?;
    set_optional(&table, "target_role", row.target_role)?;
    set_optional(&table, "origin_task_id", row.origin_task_id)?;
    set_optional(&table, "origin_execution_id", row.origin_execution_id)?;
    set_json_metadata(lua, &table, row.metadata.as_deref())?;
    table.set("created_at", row.created_at)?;
    Ok(table)
}

fn set_optional<T>(table: &Table, key: &str, value: Option<T>) -> LuaResult<()>
where
    T: IntoLua,
{
    match value {
        Some(value) => table.set(key, value),
        None => table.set(key, Value::Nil),
    }
}

fn set_json_metadata(lua: &Lua, table: &Table, raw: Option<&str>) -> LuaResult<()> {
    match raw.and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok()) {
        Some(metadata) => table.set("metadata", lua.to_value(&metadata)?),
        None => table.set("metadata", Value::Nil),
    }
}

fn graph_nodes_to_lua(lua: &Lua, rows: Vec<GraphNodeRow>) -> LuaResult<Table> {
    let out = lua.create_table()?;
    for (idx, row) in rows.into_iter().enumerate() {
        out.set(idx + 1, graph_node_to_lua(lua, row)?)?;
    }
    Ok(out)
}

fn graph_edges_to_lua(lua: &Lua, rows: Vec<GraphEdgeRow>) -> LuaResult<Table> {
    let out = lua.create_table()?;
    for (idx, row) in rows.into_iter().enumerate() {
        out.set(idx + 1, graph_edge_to_lua(lua, row)?)?;
    }
    Ok(out)
}

fn selected_path_to_lua(lua: &Lua, turn_ids: Vec<i64>) -> LuaResult<Table> {
    let out = lua.create_table()?;
    out.set("kind", "selected_path")?;
    let ids = lua.create_table()?;
    for (idx, turn_id) in turn_ids.into_iter().enumerate() {
        ids.set(idx + 1, turn_id)?;
    }
    out.set("turn_ids", ids)?;
    Ok(out)
}

async fn graph_ref_to_turn_id(
    store: &StateStore,
    session_id: i64,
    graph_ref: &GraphRef,
) -> Result<i64> {
    match graph_ref.kind.as_str() {
        "turn" => {
            let turn_id = graph_ref
                .id
                .parse::<i64>()
                .with_context(|| format!("Invalid turn graph ref id '{}'", graph_ref.id))?;
            let turn = store
                .get_turn_row(turn_id)
                .await?
                .with_context(|| format!("Turn graph ref '{}' was not found", graph_ref.id))?;
            if turn.session_id != session_id {
                anyhow::bail!(
                    "Turn graph ref '{}' belongs to another session",
                    graph_ref.id
                );
            }
            Ok(turn.id)
        }
        "branch_head" => {
            let branch_public_id = uuid::Uuid::parse_str(&graph_ref.id)
                .with_context(|| format!("Invalid branch_head graph ref id '{}'", graph_ref.id))?;
            let branch = store
                .get_branch_head_by_public_id(session_id, branch_public_id)
                .await?
                .with_context(|| {
                    format!("Branch head graph ref '{}' was not found", graph_ref.id)
                })?;
            branch.head_turn_id.with_context(|| {
                format!(
                    "Branch head graph ref '{}' does not have a materialized head turn",
                    graph_ref.id
                )
            })
        }
        other => anyhow::bail!(
            "Graph ref kind '{}' cannot be materialized into a selected path turn",
            other
        ),
    }
}

async fn selected_path_from_graph_edges(
    store: &StateStore,
    session_id: i64,
    source: GraphRef,
    opts: SelectedPathSourceOptions,
) -> Result<Vec<i64>> {
    let mut edges = store.list_graph_edges_from(&source).await?;
    if matches!(opts.order, SelectedPathOrder::NewestFirst) {
        edges.reverse();
    }
    if let Some(limit) = opts.limit {
        edges.truncate(limit);
    }
    let mut turn_ids = Vec::new();
    for edge in edges {
        if edge
            .session_id
            .is_some_and(|edge_session| edge_session != session_id)
        {
            continue;
        }
        if opts
            .relation_kind
            .as_ref()
            .is_some_and(|expected| expected != &edge.relation_kind)
        {
            continue;
        }
        if opts
            .target_kind
            .as_ref()
            .is_some_and(|expected| expected != &edge.target.kind)
        {
            continue;
        }
        if opts
            .target_role
            .as_ref()
            .is_some_and(|expected| edge.target_role.as_ref() != Some(expected))
        {
            continue;
        }
        let turn_id = graph_ref_to_turn_id(store, session_id, &edge.target).await?;
        push_unique_selected_turn(&mut turn_ids, turn_id)?;
    }
    if turn_ids.is_empty() {
        let mut detail = format!("source={}#{}", source.kind, source.id);
        if let Some(relation_kind) = opts.relation_kind.as_deref() {
            detail.push_str(&format!(", relation_kind={relation_kind}"));
        }
        if let Some(target_kind) = opts.target_kind.as_deref() {
            detail.push_str(&format!(", target_kind={target_kind}"));
        }
        if let Some(target_role) = opts.target_role.as_deref() {
            detail.push_str(&format!(", target_role={target_role}"));
        }
        anyhow::bail!("Graph selected path produced no materializable turns ({detail})");
    }
    Ok(turn_ids)
}

fn push_unique_selected_turn(turn_ids: &mut Vec<i64>, turn_id: i64) -> Result<()> {
    if turn_ids.contains(&turn_id) {
        anyhow::bail!("Graph selected path contains duplicate turn {}", turn_id);
    }
    turn_ids.push(turn_id);
    Ok(())
}

async fn selected_path_from_graph_refs(
    store: &StateStore,
    session_id: i64,
    refs: Vec<GraphRef>,
) -> Result<Vec<i64>> {
    let mut turn_ids = Vec::with_capacity(refs.len());
    for graph_ref in refs {
        let turn_id = graph_ref_to_turn_id(store, session_id, &graph_ref).await?;
        push_unique_selected_turn(&mut turn_ids, turn_id)?;
    }
    Ok(turn_ids)
}

fn resolve_session_reference(
    execution_ctx: &ActiveHarnessExecutionContext,
    requested: Option<String>,
) -> Result<SessionReference, String> {
    let (current_session_id, implicit_selector) = {
        let lock = execution_ctx
            .lock()
            .map_err(|_| "execution context mutex poisoned".to_string())?;
        (
            lock.session_id.clone(),
            lock.session_store_selector
                .clone()
                .unwrap_or_else(|| StoreSelector::Alias("state".to_string())),
        )
    };
    let raw = requested
        .or(current_session_id)
        .ok_or_else(|| "No active session context".to_string())?;
    let mut session_ref = parse_session_reference(&raw).map_err(|err| err.to_string())?;
    if session_ref.store_selector.is_none() {
        session_ref.store_selector = Some(implicit_selector);
    }
    Ok(session_ref)
}

async fn resolve_graph_session(
    store_manager: Arc<StoreManager>,
    execution_ctx: ActiveHarnessExecutionContext,
    requested: Option<String>,
) -> Result<GraphSession> {
    let session_ref =
        resolve_session_reference(&execution_ctx, requested).map_err(anyhow::Error::msg)?;
    let public_id = uuid::Uuid::parse_str(&session_ref.public_id)
        .with_context(|| format!("Invalid session id '{}'", session_ref.public_id))?;
    let selector = session_ref
        .store_selector
        .unwrap_or_else(|| StoreSelector::Alias("state".to_string()));
    let store = store_manager.open(&selector).await?;
    let row = store
        .get_session_row_by_public_id(public_id)
        .await?
        .with_context(|| format!("Session '{}' not found", session_ref.public_id))?;
    Ok(GraphSession {
        store,
        internal_id: row.id,
    })
}

pub fn register_runtime_graph_namespace(
    lua: &Lua,
    runtime_table: &Table,
    app_data: &HarnessAppData,
) -> LuaResult<()> {
    let graph_table = lua.create_table()?;
    let node_table = lua.create_table()?;
    let edge_table = lua.create_table()?;
    let path_table = lua.create_table()?;

    {
        let store_manager = app_data.store_manager.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        let app_data_snapshot = app_data.clone();
        node_table.set(
            "create",
            lua.create_function(move |lua, opts: Table| {
                if let Err(err) =
                    require_governance_capability(&app_data_snapshot, "runtime.graph.write")
                {
                    return nil_err(lua, &err);
                }
                let kind = opts.get::<String>("kind")?;
                if kind.is_empty() {
                    return nil_err(lua, "graph node kind must not be empty");
                }
                let label = opt_string(&opts, "label");
                let metadata = match graph_metadata(lua, &opts) {
                    Ok(metadata) => metadata,
                    Err(err) => return nil_err(lua, &err.to_string()),
                };
                let provenance = graph_provenance(&opts);
                let session_id = opt_string(&opts, "session_id");
                let store_manager = store_manager.clone();
                let execution_ctx = execution_ctx.clone();
                let result = bridge_async_display_err(async move {
                    let session =
                        resolve_graph_session(store_manager, execution_ctx, session_id).await?;
                    session
                        .store
                        .create_graph_node(
                            Some(session.internal_id),
                            &kind,
                            label.as_deref(),
                            provenance,
                            metadata.as_ref(),
                        )
                        .await
                });
                lua_table_result(lua, result, graph_node_to_lua)
            })?,
        )?;
    }

    {
        let store_manager = app_data.store_manager.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        let app_data_snapshot = app_data.clone();
        path_table.set(
            "select",
            lua.create_function(move |lua, opts: Table| {
                if let Err(err) =
                    require_governance_capability(&app_data_snapshot, "runtime.graph.query")
                {
                    return nil_err(lua, &err);
                }
                let request = match selected_path_request(&opts) {
                    Ok(request) => request,
                    Err(err) => return nil_err(lua, &err.to_string()),
                };
                let session_id = opt_string(&opts, "session_id");
                let store_manager = store_manager.clone();
                let execution_ctx = execution_ctx.clone();
                let result = bridge_async_display_err(async move {
                    let session =
                        resolve_graph_session(store_manager, execution_ctx, session_id).await?;
                    match request {
                        SelectedPathRequest::Refs(refs) => {
                            selected_path_from_graph_refs(&session.store, session.internal_id, refs)
                                .await
                        }
                        SelectedPathRequest::Source(source, source_opts) => {
                            selected_path_from_graph_edges(
                                &session.store,
                                session.internal_id,
                                source,
                                source_opts,
                            )
                            .await
                        }
                    }
                });
                lua_table_result(lua, result, selected_path_to_lua)
            })?,
        )?;
    }

    {
        let store_manager = app_data.store_manager.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        let app_data_snapshot = app_data.clone();
        edge_table.set(
            "create",
            lua.create_function(move |lua, opts: Table| {
                if let Err(err) =
                    require_governance_capability(&app_data_snapshot, "runtime.graph.write")
                {
                    return nil_err(lua, &err);
                }
                let source = match opts.get::<Table>("source").and_then(graph_ref_from_table) {
                    Ok(source) => source,
                    Err(err) => return nil_err(lua, &err.to_string()),
                };
                let target = match opts.get::<Table>("target").and_then(graph_ref_from_table) {
                    Ok(target) => target,
                    Err(err) => return nil_err(lua, &err.to_string()),
                };
                let relation_kind = opts.get::<String>("relation_kind")?;
                if relation_kind.is_empty() {
                    return nil_err(lua, "graph edge relation_kind must not be empty");
                }
                let metadata = match graph_metadata(lua, &opts) {
                    Ok(metadata) => metadata,
                    Err(err) => return nil_err(lua, &err.to_string()),
                };
                let session_id = opt_string(&opts, "session_id");
                let mut edge = GraphEdgeCreate::new(source, target, relation_kind);
                edge.source_role = opt_string(&opts, "source_role");
                edge.target_role = opt_string(&opts, "target_role");
                edge.provenance = graph_provenance(&opts);
                edge.metadata = metadata;
                let store_manager = store_manager.clone();
                let execution_ctx = execution_ctx.clone();
                let result = bridge_async_display_err(async move {
                    let session =
                        resolve_graph_session(store_manager, execution_ctx, session_id).await?;
                    edge.session_id = Some(session.internal_id);
                    session.store.create_graph_edge(edge).await
                });
                lua_table_result(lua, result, graph_edge_to_lua)
            })?,
        )?;
    }

    {
        let store_manager = app_data.store_manager.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        let app_data_snapshot = app_data.clone();
        node_table.set(
            "list",
            lua.create_function(move |lua, opts: Option<Table>| {
                if let Err(err) =
                    require_governance_capability(&app_data_snapshot, "runtime.graph.query")
                {
                    return nil_err(lua, &err);
                }
                let session_id = opts
                    .as_ref()
                    .and_then(|opts| opt_string(opts, "session_id"));
                let store_manager = store_manager.clone();
                let execution_ctx = execution_ctx.clone();
                let result = bridge_async_display_err(async move {
                    let session =
                        resolve_graph_session(store_manager, execution_ctx, session_id).await?;
                    session
                        .store
                        .list_graph_nodes_for_session(session.internal_id)
                        .await
                });
                lua_table_result(lua, result, graph_nodes_to_lua)
            })?,
        )?;
    }

    {
        let store_manager = app_data.store_manager.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        let app_data_snapshot = app_data.clone();
        edge_table.set(
            "list",
            lua.create_function(move |lua, opts: Option<Table>| {
                if let Err(err) =
                    require_governance_capability(&app_data_snapshot, "runtime.graph.query")
                {
                    return nil_err(lua, &err);
                }
                let source = opts
                    .as_ref()
                    .and_then(|opts| opts.get::<Table>("source").ok())
                    .map(graph_ref_from_table)
                    .transpose()?;
                let target = opts
                    .as_ref()
                    .and_then(|opts| opts.get::<Table>("target").ok())
                    .map(graph_ref_from_table)
                    .transpose()?;
                let session_id = opts
                    .as_ref()
                    .and_then(|opts| opt_string(opts, "session_id"));
                let store_manager = store_manager.clone();
                let execution_ctx = execution_ctx.clone();
                let result = bridge_async_display_err(async move {
                    let session =
                        resolve_graph_session(store_manager, execution_ctx, session_id).await?;
                    match (source, target) {
                        (Some(source), None) => session.store.list_graph_edges_from(&source).await,
                        (None, Some(target)) => session.store.list_graph_edges_to(&target).await,
                        _ => {
                            session
                                .store
                                .list_graph_edges_for_session(session.internal_id)
                                .await
                        }
                    }
                });
                lua_table_result(lua, result, graph_edges_to_lua)
            })?,
        )?;
    }

    graph_table.set("node", node_table)?;
    graph_table.set("edge", edge_table)?;
    graph_table.set("path", path_table)?;
    runtime_table.set("graph", graph_table)?;
    Ok(())
}

#[cfg(test)]
#[path = "tests/runtime_graph.rs"]
mod tests;
