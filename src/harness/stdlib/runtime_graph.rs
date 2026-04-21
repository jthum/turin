use std::sync::Arc;

use anyhow::{Context, Result};
use mlua::{Lua, LuaSerdeExt, Result as LuaResult, Table, Value};

use crate::harness::globals::{ActiveHarnessExecutionContext, HarnessAppData};
use crate::harness::stdlib::binding_common::{bridge_async_display_err, nil_err, ok_value};
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

fn graph_node_to_lua(lua: &Lua, row: GraphNodeRow) -> LuaResult<Table> {
    let table = lua.create_table()?;
    table.set("id", row.id)?;
    table.set("node_id", bytes_to_simple_uuid(&row.public_id))?;
    match row.session_id {
        Some(session_id) => table.set("session_internal_id", session_id)?,
        None => table.set("session_internal_id", Value::Nil)?,
    }
    table.set("kind", row.kind)?;
    match row.label {
        Some(label) => table.set("label", label)?,
        None => table.set("label", Value::Nil)?,
    }
    match row.origin_task_id {
        Some(task_id) => table.set("origin_task_id", task_id)?,
        None => table.set("origin_task_id", Value::Nil)?,
    }
    match row.origin_execution_id {
        Some(execution_id) => table.set("origin_execution_id", execution_id)?,
        None => table.set("origin_execution_id", Value::Nil)?,
    }
    match row
        .metadata
        .as_deref()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
    {
        Some(metadata) => table.set("metadata", lua.to_value(&metadata)?)?,
        None => table.set("metadata", Value::Nil)?,
    }
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
    match row.session_id {
        Some(session_id) => table.set("session_internal_id", session_id)?,
        None => table.set("session_internal_id", Value::Nil)?,
    }
    table.set("source", graph_ref_to_lua(lua, row.source)?)?;
    table.set("target", graph_ref_to_lua(lua, row.target)?)?;
    table.set("relation_kind", row.relation_kind)?;
    match row.source_role {
        Some(role) => table.set("source_role", role)?,
        None => table.set("source_role", Value::Nil)?,
    }
    match row.target_role {
        Some(role) => table.set("target_role", role)?,
        None => table.set("target_role", Value::Nil)?,
    }
    match row.origin_task_id {
        Some(task_id) => table.set("origin_task_id", task_id)?,
        None => table.set("origin_task_id", Value::Nil)?,
    }
    match row.origin_execution_id {
        Some(execution_id) => table.set("origin_execution_id", execution_id)?,
        None => table.set("origin_execution_id", Value::Nil)?,
    }
    match row
        .metadata
        .as_deref()
        .and_then(|raw| serde_json::from_str::<serde_json::Value>(raw).ok())
    {
        Some(metadata) => table.set("metadata", lua.to_value(&metadata)?)?,
        None => table.set("metadata", Value::Nil)?,
    }
    table.set("created_at", row.created_at)?;
    Ok(table)
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

    {
        let store_manager = app_data.store_manager.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        let app_data_snapshot = app_data.clone();
        graph_table.set(
            "node_create",
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
                match result {
                    Ok(row) => Ok(ok_value(Value::Table(graph_node_to_lua(lua, row)?))),
                    Err(err) => nil_err(lua, &err),
                }
            })?,
        )?;
    }

    {
        let store_manager = app_data.store_manager.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        let app_data_snapshot = app_data.clone();
        graph_table.set(
            "edge_create",
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
                match result {
                    Ok(row) => Ok(ok_value(Value::Table(graph_edge_to_lua(lua, row)?))),
                    Err(err) => nil_err(lua, &err),
                }
            })?,
        )?;
    }

    {
        let store_manager = app_data.store_manager.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        let app_data_snapshot = app_data.clone();
        graph_table.set(
            "nodes",
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
                match result {
                    Ok(rows) => Ok(ok_value(Value::Table(graph_nodes_to_lua(lua, rows)?))),
                    Err(err) => nil_err(lua, &err),
                }
            })?,
        )?;
    }

    {
        let store_manager = app_data.store_manager.clone();
        let execution_ctx = app_data.execution_ctx.clone();
        let app_data_snapshot = app_data.clone();
        graph_table.set(
            "edges",
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
                match result {
                    Ok(rows) => Ok(ok_value(Value::Table(graph_edges_to_lua(lua, rows)?))),
                    Err(err) => nil_err(lua, &err),
                }
            })?,
        )?;
    }

    runtime_table.set("graph", graph_table)?;
    Ok(())
}
