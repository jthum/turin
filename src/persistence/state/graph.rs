use anyhow::{Context, Result};

use super::{GraphEdgeCreate, GraphEdgeRow, GraphNodeRow, GraphProvenance, GraphRef, StateStore};

impl StateStore {
    pub async fn create_graph_node(
        &self,
        session_id: Option<i64>,
        kind: &str,
        label: Option<&str>,
        provenance: GraphProvenance,
        metadata: Option<&serde_json::Value>,
    ) -> Result<GraphNodeRow> {
        let conn = self.connect().await?;
        let public_id = uuid::Uuid::now_v7().into_bytes().to_vec();
        let metadata = metadata.map(serde_json::to_string).transpose()?;

        conn.execute(
            r#"
            INSERT INTO graph_nodes (
                public_id,
                session_id,
                kind,
                label,
                origin_task_id,
                origin_execution_id,
                metadata
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
            "#,
            turso::params![
                public_id,
                session_id,
                kind,
                label,
                provenance.origin_task_id,
                provenance.origin_execution_id,
                metadata
            ],
        )
        .await
        .with_context(|| format!("Failed to create graph node '{}'", kind))?;

        let node_id = conn.last_insert_rowid();
        self.get_graph_node(node_id)
            .await?
            .with_context(|| format!("Created graph node '{}' was not readable", kind))
    }

    pub async fn get_graph_node(&self, node_id: i64) -> Result<Option<GraphNodeRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                r#"
                SELECT id,
                       public_id,
                       session_id,
                       kind,
                       label,
                       origin_task_id,
                       origin_execution_id,
                       metadata,
                       created_at
                FROM graph_nodes
                WHERE id = ?1
                "#,
                [node_id],
            )
            .await?;

        if let Some(row) = rows.next().await? {
            Ok(Some(graph_node_from_row(&row)?))
        } else {
            Ok(None)
        }
    }

    pub async fn list_graph_nodes_for_session(&self, session_id: i64) -> Result<Vec<GraphNodeRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                r#"
                SELECT id,
                       public_id,
                       session_id,
                       kind,
                       label,
                       origin_task_id,
                       origin_execution_id,
                       metadata,
                       created_at
                FROM graph_nodes
                WHERE session_id = ?1
                ORDER BY created_at, id
                "#,
                [session_id],
            )
            .await?;

        let mut nodes = Vec::new();
        while let Some(row) = rows.next().await? {
            nodes.push(graph_node_from_row(&row)?);
        }
        Ok(nodes)
    }

    pub async fn create_graph_edge(&self, edge: GraphEdgeCreate) -> Result<GraphEdgeRow> {
        let conn = self.connect().await?;
        let public_id = uuid::Uuid::now_v7().into_bytes().to_vec();
        let relation_kind = edge.relation_kind;
        let metadata = edge
            .metadata
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;

        conn.execute(
            r#"
            INSERT INTO graph_edges (
                public_id,
                session_id,
                source_kind,
                source_id,
                target_kind,
                target_id,
                relation_kind,
                source_role,
                target_role,
                origin_task_id,
                origin_execution_id,
                metadata
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
            "#,
            turso::params![
                public_id,
                edge.session_id,
                edge.source.kind,
                edge.source.id,
                edge.target.kind,
                edge.target.id,
                relation_kind.clone(),
                edge.source_role,
                edge.target_role,
                edge.provenance.origin_task_id,
                edge.provenance.origin_execution_id,
                metadata
            ],
        )
        .await
        .with_context(|| format!("Failed to create graph edge '{}'", relation_kind))?;

        let edge_id = conn.last_insert_rowid();
        self.get_graph_edge(edge_id)
            .await?
            .with_context(|| format!("Created graph edge '{}' was not readable", relation_kind))
    }

    pub async fn get_graph_edge(&self, edge_id: i64) -> Result<Option<GraphEdgeRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                r#"
                SELECT id,
                       public_id,
                       session_id,
                       source_kind,
                       source_id,
                       target_kind,
                       target_id,
                       relation_kind,
                       source_role,
                       target_role,
                       origin_task_id,
                       origin_execution_id,
                       metadata,
                       created_at
                FROM graph_edges
                WHERE id = ?1
                "#,
                [edge_id],
            )
            .await?;

        if let Some(row) = rows.next().await? {
            Ok(Some(graph_edge_from_row(&row)?))
        } else {
            Ok(None)
        }
    }

    pub async fn list_graph_edges_for_session(&self, session_id: i64) -> Result<Vec<GraphEdgeRow>> {
        self.query_graph_edges("session_id = ?1", [session_id])
            .await
    }

    pub async fn list_graph_edges_from(&self, source: &GraphRef) -> Result<Vec<GraphEdgeRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                r#"
                SELECT id,
                       public_id,
                       session_id,
                       source_kind,
                       source_id,
                       target_kind,
                       target_id,
                       relation_kind,
                       source_role,
                       target_role,
                       origin_task_id,
                       origin_execution_id,
                       metadata,
                       created_at
                FROM graph_edges
                WHERE source_kind = ?1 AND source_id = ?2
                ORDER BY created_at, id
                "#,
                turso::params![source.kind.clone(), source.id.clone()],
            )
            .await?;

        let mut edges = Vec::new();
        while let Some(row) = rows.next().await? {
            edges.push(graph_edge_from_row(&row)?);
        }
        Ok(edges)
    }

    pub async fn list_graph_edges_to(&self, target: &GraphRef) -> Result<Vec<GraphEdgeRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                r#"
                SELECT id,
                       public_id,
                       session_id,
                       source_kind,
                       source_id,
                       target_kind,
                       target_id,
                       relation_kind,
                       source_role,
                       target_role,
                       origin_task_id,
                       origin_execution_id,
                       metadata,
                       created_at
                FROM graph_edges
                WHERE target_kind = ?1 AND target_id = ?2
                ORDER BY created_at, id
                "#,
                turso::params![target.kind.clone(), target.id.clone()],
            )
            .await?;

        let mut edges = Vec::new();
        while let Some(row) = rows.next().await? {
            edges.push(graph_edge_from_row(&row)?);
        }
        Ok(edges)
    }

    async fn query_graph_edges<const N: usize>(
        &self,
        where_clause: &str,
        params: [i64; N],
    ) -> Result<Vec<GraphEdgeRow>> {
        let conn = self.connect().await?;
        let sql = format!(
            r#"
            SELECT id,
                   public_id,
                   session_id,
                   source_kind,
                   source_id,
                   target_kind,
                   target_id,
                   relation_kind,
                   source_role,
                   target_role,
                   origin_task_id,
                   origin_execution_id,
                   metadata,
                   created_at
            FROM graph_edges
            WHERE {where_clause}
            ORDER BY created_at, id
            "#
        );
        let mut rows = conn.query(&sql, params).await?;

        let mut edges = Vec::new();
        while let Some(row) = rows.next().await? {
            edges.push(graph_edge_from_row(&row)?);
        }
        Ok(edges)
    }
}

fn graph_node_from_row(row: &turso::Row) -> Result<GraphNodeRow> {
    Ok(GraphNodeRow {
        id: row.get::<i64>(0)?,
        public_id: row.get::<Vec<u8>>(1)?,
        session_id: row.get::<Option<i64>>(2)?,
        kind: row.get::<String>(3)?,
        label: row.get::<Option<String>>(4)?,
        origin_task_id: row.get::<Option<String>>(5)?,
        origin_execution_id: row.get::<Option<String>>(6)?,
        metadata: row.get::<Option<String>>(7)?,
        created_at: row.get::<String>(8)?,
    })
}

fn graph_edge_from_row(row: &turso::Row) -> Result<GraphEdgeRow> {
    Ok(GraphEdgeRow {
        id: row.get::<i64>(0)?,
        public_id: row.get::<Vec<u8>>(1)?,
        session_id: row.get::<Option<i64>>(2)?,
        source: GraphRef {
            kind: row.get::<String>(3)?,
            id: row.get::<String>(4)?,
        },
        target: GraphRef {
            kind: row.get::<String>(5)?,
            id: row.get::<String>(6)?,
        },
        relation_kind: row.get::<String>(7)?,
        source_role: row.get::<Option<String>>(8)?,
        target_role: row.get::<Option<String>>(9)?,
        origin_task_id: row.get::<Option<String>>(10)?,
        origin_execution_id: row.get::<Option<String>>(11)?,
        metadata: row.get::<Option<String>>(12)?,
        created_at: row.get::<String>(13)?,
    })
}
