use anyhow::{Context, Result};
use turso::Value as SqlValue;

use crate::persistence::schema::{
    MemoryInspectionPage, MemoryInspectionRow, MemoryInspectionScopeRow,
};
use crate::persistence::state::StateStore;

impl StateStore {
    /// Browse persisted memories without recording a retrieval.
    pub async fn inspect_memories(
        &self,
        scope_kind: Option<&str>,
        scope_key: Option<&str>,
        include_superseded: bool,
        limit: u32,
        offset: u32,
    ) -> Result<MemoryInspectionPage> {
        let conn = self.connect().await?;
        let (where_clause, params) =
            memory_inspection_filter(scope_kind, scope_key, include_superseded);

        let count_sql = format!("SELECT COUNT(*) FROM memories {where_clause}");
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
            "SELECT scope_kind, scope_key, COUNT(*) FROM memories {scope_where} \
             GROUP BY scope_kind, scope_key ORDER BY scope_kind, scope_key"
        );
        let mut scope_rows = conn
            .query(&scope_sql, ())
            .await
            .context("Failed to list memory scopes for inspection")?;
        let mut scopes = Vec::new();
        while let Some(row) = scope_rows.next().await? {
            let scope_kind = row.get::<String>(0)?;
            let scope_key = row.get::<String>(1)?;
            scopes.push(MemoryInspectionScopeRow {
                count: crate::persistence::state::persisted_u64(
                    &format!("memory scope {scope_kind}:{scope_key}"),
                    "count",
                    row.get::<i64>(2)?,
                )?,
                scope_kind,
                scope_key,
            });
        }

        let mut page_params = params;
        page_params.push(SqlValue::Integer(limit as i64));
        let limit_index = page_params.len();
        page_params.push(SqlValue::Integer(offset as i64));
        let offset_index = page_params.len();
        let page_sql = format!(
            "SELECT public_id, scope_kind, scope_key, content, metadata, \
                    embedding IS NOT NULL, embedding_key, embedding_dimensions, weight, \
                    retrieval_count, last_retrieved_at, superseded_at, created_at \
             FROM memories {where_clause} ORDER BY id DESC \
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
            });
        }

        Ok(MemoryInspectionPage {
            rows: memories,
            scopes,
            total,
        })
    }
}

fn memory_inspection_filter(
    scope_kind: Option<&str>,
    scope_key: Option<&str>,
    include_superseded: bool,
) -> (String, Vec<SqlValue>) {
    let mut clauses = Vec::new();
    let mut params = Vec::new();
    if let Some(scope_kind) = scope_kind {
        params.push(SqlValue::Text(scope_kind.to_string()));
        clauses.push(format!("scope_kind = ?{}", params.len()));
    }
    if let Some(scope_key) = scope_key {
        params.push(SqlValue::Text(scope_key.to_string()));
        clauses.push(format!("scope_key = ?{}", params.len()));
    }
    if !include_superseded {
        clauses.push("superseded_at IS NULL".to_string());
    }
    let where_clause = if clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", clauses.join(" AND "))
    };
    (where_clause, params)
}
