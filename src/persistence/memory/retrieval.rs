use std::collections::HashMap;

use anyhow::{Context, Result};

use super::vector_to_le_bytes;
use crate::persistence::schema::MemoryRow;
use crate::persistence::state::StateStore;

impl StateStore {
    /// Search memories using lexical, vector, or hybrid retrieval.
    #[allow(clippy::too_many_arguments)]
    pub async fn search_memories(
        &self,
        scope_kind: &str,
        scope_key: &str,
        vector: Option<&[f32]>,
        query_embedding_key: Option<&str>,
        query_embedding_dimensions: Option<usize>,
        content_query: Option<&str>,
        limit: usize,
        min_score: f64,
        include_metadata: bool,
        include_superseded: bool,
    ) -> Result<Vec<MemoryRow>> {
        const RRF_K: f64 = 60.0;
        let mut scores: HashMap<i64, f64> = HashMap::new();
        let mut rows_data: HashMap<i64, MemoryRow> = HashMap::new();
        let mut recency_boosts: HashMap<i64, f64> = HashMap::new();
        let conn = self.connect().await?;

        if let Some(vector) = vector {
            let query_embedding_key =
                query_embedding_key.context("Missing embedding key for semantic memory search")?;
            let query_embedding_dimensions = query_embedding_dimensions
                .context("Missing embedding dimensions for semantic memory search")?;
            let vector_bytes = vector_to_le_bytes(vector);
            let (sql, params) = if include_superseded {
                (
                    "SELECT id, public_id, scope_kind, scope_key, content, metadata, created_at, weight, retrieval_count, last_retrieved_at, superseded_at,
                            CAST((julianday('now') - julianday(created_at)) * 86400.0 AS REAL) AS age_seconds,
                            vector_distance_cos(embedding, ?1) AS distance
                     FROM memories
                     WHERE scope_kind = ?2 AND scope_key = ?3 AND embedding IS NOT NULL AND embedding_key = ?4 AND embedding_dimensions = ?5
                     ORDER BY distance ASC
                     LIMIT ?6",
                    turso::params![
                        vector_bytes,
                        scope_kind,
                        scope_key,
                        query_embedding_key.to_string(),
                        query_embedding_dimensions as i64,
                        limit as i64
                    ],
                )
            } else {
                (
                    "SELECT id, public_id, scope_kind, scope_key, content, metadata, created_at, weight, retrieval_count, last_retrieved_at, superseded_at,
                            CAST((julianday('now') - julianday(created_at)) * 86400.0 AS REAL) AS age_seconds,
                            vector_distance_cos(embedding, ?1) AS distance
                     FROM memories
                     WHERE scope_kind = ?2 AND scope_key = ?3 AND embedding IS NOT NULL AND embedding_key = ?4 AND embedding_dimensions = ?5 AND superseded_at IS NULL
                     ORDER BY distance ASC
                     LIMIT ?6",
                    turso::params![
                        vector_bytes,
                        scope_kind,
                        scope_key,
                        query_embedding_key.to_string(),
                        query_embedding_dimensions as i64,
                        limit as i64
                    ],
                )
            };
            let mut rows = conn
                .query(sql, params)
                .await
                .context("Failed to search memories (vector)")?;
            let mut rank = 1;
            while let Some(row) = rows.next().await? {
                let id = row.get::<i64>(0)?;
                let age_seconds = row.get::<Option<f64>>(11)?.unwrap_or(0.0);
                let distance = row.get::<f64>(12)?;
                if let std::collections::hash_map::Entry::Vacant(entry) = rows_data.entry(id) {
                    let mut memory = memory_row_from_search_row(&row, id, include_metadata)?;
                    memory.semantic_score = Some(1.0 / (1.0 + distance.max(0.0)));
                    entry.insert(memory);
                    recency_boosts.insert(id, recency_boost(age_seconds));
                } else if let Some(existing) = rows_data.get_mut(&id) {
                    existing.semantic_score = Some(1.0 / (1.0 + distance.max(0.0)));
                }
                *scores.entry(id).or_default() += 1.0 / (RRF_K + rank as f64);
                rank += 1;
            }
        }

        if let Some(query) = content_query
            .map(str::trim)
            .filter(|query| !query.is_empty())
        {
            let (sql, params) = if include_superseded {
                (
                    "SELECT id, public_id, scope_kind, scope_key, content, metadata, created_at, weight, retrieval_count, last_retrieved_at, superseded_at,
                            CAST((julianday('now') - julianday(created_at)) * 86400.0 AS REAL) AS age_seconds,
                            fts_score(content, ?1) AS lexical_score
                     FROM memories
                     WHERE scope_kind = ?2 AND scope_key = ?3
                     AND fts_match(content, ?1)
                     ORDER BY lexical_score DESC
                     LIMIT ?4",
                    turso::params![query, scope_kind, scope_key, limit as i64],
                )
            } else {
                (
                    "SELECT id, public_id, scope_kind, scope_key, content, metadata, created_at, weight, retrieval_count, last_retrieved_at, superseded_at,
                            CAST((julianday('now') - julianday(created_at)) * 86400.0 AS REAL) AS age_seconds,
                            fts_score(content, ?1) AS lexical_score
                     FROM memories
                     WHERE scope_kind = ?2 AND scope_key = ?3
                     AND superseded_at IS NULL
                     AND fts_match(content, ?1)
                     ORDER BY lexical_score DESC
                     LIMIT ?4",
                    turso::params![query, scope_kind, scope_key, limit as i64],
                )
            };
            let mut rows = conn
                .query(sql, params)
                .await
                .context("Failed to search memories (lexical)")?;
            let mut rank = 1;
            while let Some(row) = rows.next().await? {
                let id = row.get::<i64>(0)?;
                let age_seconds = row.get::<Option<f64>>(11)?.unwrap_or(0.0);
                let lexical_score = row.get::<f64>(12)?;
                if let std::collections::hash_map::Entry::Vacant(entry) = rows_data.entry(id) {
                    let mut memory = memory_row_from_search_row(&row, id, include_metadata)?;
                    memory.lexical_score = Some(lexical_score);
                    entry.insert(memory);
                    recency_boosts.insert(id, recency_boost(age_seconds));
                } else if let Some(existing) = rows_data.get_mut(&id) {
                    existing.lexical_score = Some(lexical_score);
                }
                *scores.entry(id).or_default() += 1.0 / (RRF_K + rank as f64);
                rank += 1;
            }
        }

        let mut results: Vec<MemoryRow> = scores
            .into_iter()
            .filter_map(|(id, score)| {
                rows_data.remove(&id).map(|mut row| {
                    row.score =
                        score * row.weight * recency_boosts.get(&id).copied().unwrap_or(1.0);
                    row
                })
            })
            .collect();
        results.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.retain(|row| row.score >= min_score);
        results.truncate(limit);

        if !results.is_empty() {
            let retrieved_at = current_utc_timestamp(&conn).await?;
            for row in &mut results {
                conn.execute(
                    "UPDATE memories SET retrieval_count = retrieval_count + 1, last_retrieved_at = ?2 WHERE id = ?1",
                    turso::params![row.id, retrieved_at.clone()],
                )
                .await
                .with_context(|| format!("Failed to update retrieval stats for memory {}", row.id))?;
                row.retrieval_count += 1;
                row.last_retrieved_at = Some(retrieved_at.clone());
            }
        }
        Ok(results)
    }
}

fn recency_boost(age_seconds: f64) -> f64 {
    let age_days = age_seconds.max(0.0) / 86_400.0;
    1.0 + (0.05 / (1.0 + age_days))
}

fn memory_row_from_search_row(
    row: &turso::Row,
    id: i64,
    include_metadata: bool,
) -> Result<MemoryRow> {
    let record = format!("memory {id}");
    Ok(MemoryRow {
        id,
        public_id: row.get(1)?,
        scope_kind: row.get(2)?,
        scope_key: row.get(3)?,
        content: row.get(4)?,
        metadata: include_metadata
            .then(|| row.get::<Option<String>>(5))
            .transpose()?
            .flatten(),
        created_at: row.get(6)?,
        score: 0.0,
        lexical_score: None,
        semantic_score: None,
        weight: row.get(7)?,
        retrieval_count: crate::persistence::state::persisted_u64(
            &record,
            "retrieval count",
            row.get::<i64>(8)?,
        )?,
        last_retrieved_at: row.get(9)?,
        superseded_at: row.get(10)?,
    })
}

async fn current_utc_timestamp(conn: &turso::Connection) -> Result<String> {
    let mut rows = conn
        .query("SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')", ())
        .await
        .context("Failed to fetch current UTC timestamp")?;
    rows.next()
        .await?
        .map(|row| row.get::<String>(0))
        .transpose()?
        .context("Timestamp query returned no row")
}
