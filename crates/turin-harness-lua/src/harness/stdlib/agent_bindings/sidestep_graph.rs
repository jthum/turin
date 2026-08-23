use std::sync::Arc;

use mlua::{Lua, Table, Value};

use crate::harness::stdlib::object_refs;
use crate::kernel::event::TaskBranchOutcome;
use crate::kernel::session_refs::parse_session_reference;
use crate::persistence::manager::{StoreManager, StoreSelector};
use crate::persistence::schema::{GraphEdgeCreate, GraphProvenance, GraphRef};

pub(super) struct SidestepGraphRelation {
    source: GraphRef,
    relation_kind: String,
    source_role: Option<String>,
    target_role: Option<String>,
    origin_task_id: Option<String>,
    origin_execution_id: Option<String>,
    metadata: Option<serde_json::Value>,
}

fn graph_ref_from_lua_table(table: Table) -> std::result::Result<GraphRef, String> {
    let kind = table.get::<String>("kind").map_err(|err| err.to_string())?;
    let id = table.get::<String>("id").map_err(|err| err.to_string())?;
    if kind.is_empty() || id.is_empty() {
        return Err("graph ref requires non-empty kind and id".to_string());
    }
    Ok(GraphRef::new(kind, id))
}

pub(super) fn opt_sidestep_graph_relation(
    lua: &Lua,
    opts: Option<&Table>,
) -> std::result::Result<Option<SidestepGraphRelation>, String> {
    let Some(opts) = opts else {
        return Ok(None);
    };
    let Ok(graph) = opts.get::<Table>("graph") else {
        return Ok(None);
    };
    let source = graph
        .get::<Table>("source")
        .map_err(|err| err.to_string())
        .and_then(graph_ref_from_lua_table)?;
    let relation_kind = graph
        .get::<String>("relation_kind")
        .ok()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "contains".to_string());
    let metadata = match graph.get::<Value>("metadata") {
        Ok(Value::Nil) | Err(_) => None,
        Ok(value) => {
            Some(object_refs::encode_lua_payload(lua, value).map_err(|err| err.to_string())?)
        }
    };
    Ok(Some(SidestepGraphRelation {
        source,
        relation_kind,
        source_role: graph
            .get::<String>("source_role")
            .ok()
            .filter(|value| !value.is_empty()),
        target_role: graph
            .get::<String>("target_role")
            .ok()
            .filter(|value| !value.is_empty()),
        origin_task_id: graph
            .get::<String>("origin_task_id")
            .ok()
            .filter(|value| !value.is_empty()),
        origin_execution_id: graph
            .get::<String>("origin_execution_id")
            .ok()
            .filter(|value| !value.is_empty()),
        metadata,
    }))
}

pub(super) async fn attach_sidestep_graph_relation(
    store_manager: &Arc<StoreManager>,
    session_id: &str,
    branch_outcome: &TaskBranchOutcome,
    relation: SidestepGraphRelation,
) -> std::result::Result<(), String> {
    let TaskBranchOutcome::SidestepSibling {
        branch_public_id, ..
    } = branch_outcome
    else {
        return Err("sidestep graph relation requires a sidestep sibling branch".to_string());
    };
    let session_ref = parse_session_reference(session_id).map_err(|err| err.to_string())?;
    let public_id = uuid::Uuid::parse_str(&session_ref.public_id).map_err(|err| err.to_string())?;
    let store_selector = session_ref
        .store_selector
        .unwrap_or_else(|| StoreSelector::Alias("state".to_string()));
    let store = store_manager
        .open(&store_selector)
        .await
        .map_err(|err| err.to_string())?;
    let row = store
        .get_session_row_by_public_id(public_id)
        .await
        .map_err(|err| err.to_string())?
        .ok_or_else(|| format!("Session '{}' not found", session_ref.public_id))?;
    let branch_ref_id = uuid::Uuid::parse_str(branch_public_id)
        .map(|uuid| uuid.simple().to_string())
        .unwrap_or_else(|_| branch_public_id.clone());
    let mut edge = GraphEdgeCreate::new(
        relation.source,
        GraphRef::new("branch_head", branch_ref_id),
        relation.relation_kind,
    );
    edge.session_id = Some(row.id);
    edge.source_role = relation.source_role;
    edge.target_role = relation.target_role;
    edge.provenance = GraphProvenance::new(relation.origin_task_id, relation.origin_execution_id);
    edge.metadata = relation.metadata;
    store
        .create_graph_edge(edge)
        .await
        .map(|_| ())
        .map_err(|err| err.to_string())
}
