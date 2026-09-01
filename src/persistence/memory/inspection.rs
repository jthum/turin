use anyhow::{Context, Result};
use turso::Value as SqlValue;

use crate::persistence::schema::{
    MemoryInspectionPage, MemoryInspectionRow, MemoryInspectionScopeKindRow,
};
use crate::persistence::state::StateStore;

impl StateStore {
    /// Browse persisted memories without recording a retrieval.
    pub async fn inspect_memories(
        &self,
        scope_kind: Option<&str>,
        scope_key: Option<&str>,
        query: Option<&str>,
        include_superseded: bool,
        limit: u32,
        offset: u32,
    ) -> Result<MemoryInspectionPage> {
        let conn = self.connect().await?;
        let (where_clause, params) =
            memory_inspection_filter(scope_kind, scope_key, query, include_superseded);

        let count_sql = format!("SELECT COUNT(*) FROM memories m {where_clause}");
        let mut count_rows = conn
            .prepare(&count_sql)
            .await?
            .query(params.clone())
            .await
            .context("Failed to count memories for inspection")?;
        let total = crate::persistence::state::persisted_u64(
            "memory inspection aggregate",
            "total count",
            count_rows
                .next()
                .await?
                .map(|row| row.get::<i64>(0))
                .transpose()?
                .unwrap_or(0),
        )?;

        let scope_where = if include_superseded {
            String::new()
        } else {
            "WHERE superseded_at IS NULL".to_string()
        };
        let scope_sql = format!(
            "SELECT scope_kind, COUNT(*) FROM memories {scope_where} \
             GROUP BY scope_kind ORDER BY scope_kind"
        );
        let mut scope_rows = conn
            .query(&scope_sql, ())
            .await
            .context("Failed to list memory scopes for inspection")?;
        let mut scope_kinds = Vec::new();
        while let Some(row) = scope_rows.next().await? {
            let scope_kind = row.get::<String>(0)?;
            scope_kinds.push(MemoryInspectionScopeKindRow {
                count: crate::persistence::state::persisted_u64(
                    &format!("memory scope kind {scope_kind}"),
                    "count",
                    row.get::<i64>(1)?,
                )?,
                scope_kind,
            });
        }

        let mut page_params = params;
        page_params.push(SqlValue::Integer(limit as i64));
        let limit_index = page_params.len();
        page_params.push(SqlValue::Integer(offset as i64));
        let offset_index = page_params.len();
        let page_sql = format!(
            "SELECT m.public_id, m.scope_kind, m.scope_key, m.content, m.metadata, \
                    m.embedding IS NOT NULL, m.embedding_key, m.embedding_dimensions, m.weight, \
                    m.retrieval_count, m.last_retrieved_at, m.superseded_at, m.created_at, \
                    replacement.public_id \
             FROM memories m \
             LEFT JOIN memories replacement ON replacement.id = m.superseded_by_memory_id \
             {where_clause} ORDER BY m.id DESC \
             LIMIT ?{limit_index} OFFSET ?{offset_index}"
        );
        let mut rows = conn
            .prepare(&page_sql)
            .await?
            .query(page_params)
            .await
            .context("Failed to list memories for inspection")?;
        let mut memories = Vec::new();
        while let Some(row) = rows.next().await? {
            let public_id = row.get::<Vec<u8>>(0)?;
            let record = "memory inspection row";
            memories.push(MemoryInspectionRow {
                public_id,
                scope_kind: row.get(1)?,
                scope_key: row.get(2)?,
                content: row.get(3)?,
                metadata: row.get(4)?,
                embedded: row.get::<i64>(5)? != 0,
                embedding_key: row.get(6)?,
                embedding_dimensions: crate::persistence::state::persisted_optional_u32(
                    record,
                    "embedding dimensions",
                    row.get::<Option<i64>>(7)?,
                )?,
                weight: row.get(8)?,
                retrieval_count: crate::persistence::state::persisted_u64(
                    record,
                    "retrieval count",
                    row.get::<i64>(9)?,
                )?,
                last_retrieved_at: row.get(10)?,
                superseded_at: row.get(11)?,
                created_at: row.get(12)?,
                superseded_by_public_id: row.get(13)?,
            });
        }

        Ok(MemoryInspectionPage {
            rows: memories,
            scope_kinds,
            total,
        })
    }

    pub async fn inspect_memory(
        &self,
        public_id: uuid::Uuid,
    ) -> Result<Option<MemoryInspectionRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                r#"
				SELECT m.public_id, m.scope_kind, m.scope_key, m.content, m.metadata,
				       m.embedding IS NOT NULL, m.embedding_key, m.embedding_dimensions, m.weight,
				       m.retrieval_count, m.last_retrieved_at, m.superseded_at, m.created_at,
				       replacement.public_id
				FROM memories m
				LEFT JOIN memories replacement ON replacement.id = m.superseded_by_memory_id
				WHERE m.public_id = ?1
				LIMIT 1
				"#,
                turso::params![public_id.into_bytes().to_vec()],
            )
            .await
            .context("Failed to inspect memory")?;
        let Some(row) = rows.next().await? else {
            return Ok(None);
        };
        Ok(Some(MemoryInspectionRow {
            public_id: row.get(0)?,
            scope_kind: row.get(1)?,
            scope_key: row.get(2)?,
            content: row.get(3)?,
            metadata: row.get(4)?,
            embedded: row.get::<i64>(5)? != 0,
            embedding_key: row.get(6)?,
            embedding_dimensions: crate::persistence::state::persisted_optional_u32(
                "memory inspection row",
                "embedding dimensions",
                row.get::<Option<i64>>(7)?,
            )?,
            weight: row.get(8)?,
            retrieval_count: crate::persistence::state::persisted_u64(
                "memory inspection row",
                "retrieval count",
                row.get::<i64>(9)?,
            )?,
            last_retrieved_at: row.get(10)?,
            superseded_at: row.get(11)?,
            created_at: row.get(12)?,
            superseded_by_public_id: row.get(13)?,
        }))
    }
}

fn memory_inspection_filter(
    scope_kind: Option<&str>,
    scope_key: Option<&str>,
    query: Option<&str>,
    include_superseded: bool,
) -> (String, Vec<SqlValue>) {
    let mut clauses = Vec::new();
    let mut params = Vec::new();
    if let Some(scope_kind) = scope_kind {
        params.push(SqlValue::Text(scope_kind.to_string()));
        clauses.push(format!("m.scope_kind = ?{}", params.len()));
    }
    if let Some(scope_key) = scope_key {
        params.push(SqlValue::Text(scope_key.to_string()));
        clauses.push(format!("m.scope_key = ?{}", params.len()));
    }
    if let Some(query) = query.map(str::trim).filter(|query| !query.is_empty()) {
        params.push(SqlValue::Text(query.to_string()));
        clauses.push(format!("fts_match(m.content, ?{})", params.len()));
    }
    if !include_superseded {
        clauses.push("m.superseded_at IS NULL".to_string());
    }
    let where_clause = if clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", clauses.join(" AND "))
    };
    (where_clause, params)
}
