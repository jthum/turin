//! Cognitive memory storage and lexical/vector/hybrid search.

use anyhow::{Context, Result};
use std::collections::HashMap;
use uuid::Uuid;

use super::schema::{MemoryRow, MemoryStorageKind, StoredMemoryRow};
use super::state::StateStore;

impl StateStore {
    // ─── Memories (Vector + Native FTS Hybrid Store) ──────────────

    /// Insert a memory, optionally with an embedding vector.
    pub async fn insert_memory(
        &self,
        session_id: i64,
        content: &str,
        vector: Option<&[f32]>,
        metadata: &serde_json::Value,
    ) -> Result<StoredMemoryRow> {
        let metadata_str = serde_json::to_string(metadata)?;
        let public_id = Uuid::now_v7();
        let public_id_bytes = public_id.into_bytes().to_vec();
        let storage = if vector.is_some() {
            MemoryStorageKind::Embedded
        } else {
            MemoryStorageKind::LexicalOnly
        };

        let conn = self.connect().await?;
        match vector {
            Some(vector) => {
                let mut vector_bytes = Vec::with_capacity(vector.len() * 4);
                for &val in vector {
                    vector_bytes.extend_from_slice(&val.to_le_bytes());
                }

                conn.execute(
                    "INSERT INTO memories (public_id, session_id, content, embedding, metadata) VALUES (?1, ?2, ?3, ?4, ?5)",
                    turso::params![public_id_bytes.clone(), session_id, content, vector_bytes, metadata_str],
                )
                .await
                .with_context(|| format!("Failed to insert memory for session: {}", session_id))?;
            }
            None => {
                conn.execute(
                    "INSERT INTO memories (public_id, session_id, content, metadata) VALUES (?1, ?2, ?3, ?4)",
                    turso::params![public_id_bytes.clone(), session_id, content, metadata_str],
                )
                .await
                .with_context(|| format!("Failed to insert memory for session: {}", session_id))?;
            }
        }

        let row_id = conn.last_insert_rowid();
        let mut rows = conn
            .query(
                "SELECT created_at FROM memories WHERE id = ?1",
                turso::params![row_id],
            )
            .await
            .context("Failed to fetch inserted memory timestamp")?;
        let stored_at = rows
            .next()
            .await?
            .map(|row| row.get::<String>(0))
            .transpose()?
            .context("Inserted memory row missing timestamp")?;

        Ok(StoredMemoryRow {
            public_id: public_id_bytes,
            stored_at,
            storage,
        })
    }

    /// Search memories using lexical, vector, or hybrid retrieval.
    ///
    /// Uses Reciprocal Rank Fusion (RRF) to combine results.
    /// - `vector`: Optional embedding for semantic search.
    /// - `content_query`: Optional keyword string for lexical search. If `None`, relies only on vector.
    pub async fn search_memories(
        &self,
        session_id: i64,
        vector: Option<&[f32]>,
        content_query: Option<&str>,
        limit: usize,
        min_score: f64,
        include_metadata: bool,
        include_superseded: bool,
    ) -> Result<Vec<MemoryRow>> {
        // RRF constant k (usually 60)
        const RRF_K: f64 = 60.0;
        let mut scores: HashMap<i64, f64> = HashMap::new();
        let mut rows_data: HashMap<i64, MemoryRow> = HashMap::new();
        let mut recency_boosts: HashMap<i64, f64> = HashMap::new();

        let conn = self.connect().await?;

        // 1. Vector Search
        if let Some(vec) = vector {
            // Convert to bytes
            let mut vector_bytes = Vec::with_capacity(vec.len() * 4);
            for &val in vec {
                vector_bytes.extend_from_slice(&val.to_le_bytes());
            }

            let (sql, params) = if include_superseded {
                (
                    "SELECT id, public_id, session_id, content, metadata, created_at, weight, retrieval_count, last_retrieved_at, superseded_at,
                            CAST((julianday('now') - julianday(created_at)) * 86400.0 AS REAL) AS age_seconds,
                            vector_distance_cos(embedding, ?1) AS distance
                     FROM memories
                     WHERE session_id = ?2 AND embedding IS NOT NULL
                     ORDER BY distance ASC
                     LIMIT ?3",
                    turso::params![vector_bytes, session_id, limit as i64],
                )
            } else {
                (
                    "SELECT id, public_id, session_id, content, metadata, created_at, weight, retrieval_count, last_retrieved_at, superseded_at,
                            CAST((julianday('now') - julianday(created_at)) * 86400.0 AS REAL) AS age_seconds,
                            vector_distance_cos(embedding, ?1) AS distance
                     FROM memories
                     WHERE session_id = ?2 AND embedding IS NOT NULL AND superseded_at IS NULL
                     ORDER BY distance ASC
                     LIMIT ?3",
                    turso::params![vector_bytes, session_id, limit as i64],
                )
            };
            let mut rows = conn
                .query(sql, params)
                .await
                .context("Failed to search memories (vector)")?;

            let mut rank = 1;
            while let Some(row) = rows.next().await? {
                let id: i64 = row.get(0)?;
                let distance: f64 = row.get(11)?;
                let age_seconds: f64 = row.get::<Option<f64>>(10)?.unwrap_or(0.0);

                // Track row data if not seen
                if let std::collections::hash_map::Entry::Vacant(e) = rows_data.entry(id) {
                    e.insert(MemoryRow {
                        id,
                        public_id: row.get(1)?,
                        session_id: row.get(2)?,
                        content: row.get(3)?,
                        metadata: include_metadata.then(|| row.get::<Option<String>>(4)).transpose()?.flatten(),
                        created_at: row.get(5)?,
                        score: 0.0,
                        lexical_score: None,
                        semantic_score: Some(1.0 / (1.0 + distance.max(0.0))),
                        weight: row.get(6)?,
                        retrieval_count: row.get::<i64>(7)? as u64,
                        last_retrieved_at: row.get(8)?,
                        superseded_at: row.get(9)?,
                    });
                    recency_boosts.insert(id, recency_boost(age_seconds));
                } else if let Some(existing) = rows_data.get_mut(&id) {
                    existing.semantic_score = Some(1.0 / (1.0 + distance.max(0.0)));
                }

                // RRF score addition
                let rrf = 1.0 / (RRF_K + rank as f64);
                *scores.entry(id).or_default() += rrf;
                rank += 1;
            }
        }

        // 2. Native FTS Search
        if let Some(query) = content_query {
            // Trim and verify query isn't empty
            let query = query.trim();
            if !query.is_empty() {
                let (sql, params) = if include_superseded {
                    (
                        "SELECT id, public_id, session_id, content, metadata, created_at, weight, retrieval_count, last_retrieved_at, superseded_at,
                                CAST((julianday('now') - julianday(created_at)) * 86400.0 AS REAL) AS age_seconds,
                                fts_score(content, ?1) AS lexical_score
                         FROM memories
                         WHERE session_id = ?2
                         AND fts_match(content, ?1)
                         ORDER BY lexical_score DESC
                         LIMIT ?3",
                        turso::params![query, session_id, limit as i64],
                    )
                } else {
                    (
                        "SELECT id, public_id, session_id, content, metadata, created_at, weight, retrieval_count, last_retrieved_at, superseded_at,
                                CAST((julianday('now') - julianday(created_at)) * 86400.0 AS REAL) AS age_seconds,
                                fts_score(content, ?1) AS lexical_score
                         FROM memories
                         WHERE session_id = ?2
                         AND superseded_at IS NULL
                         AND fts_match(content, ?1)
                         ORDER BY lexical_score DESC
                         LIMIT ?3",
                        turso::params![query, session_id, limit as i64],
                    )
                };
                let mut rows = conn
                    .query(sql, params)
                    .await
                    .context("Failed to search memories (lexical)")?;

                let mut rank = 1;
                while let Some(row) = rows.next().await? {
                    let id: i64 = row.get(0)?;
                    let lexical_score: f64 = row.get(11)?;
                    let age_seconds: f64 = row.get::<Option<f64>>(10)?.unwrap_or(0.0);

                    if let std::collections::hash_map::Entry::Vacant(e) = rows_data.entry(id) {
                        e.insert(MemoryRow {
                            id,
                            public_id: row.get(1)?,
                            session_id: row.get(2)?,
                            content: row.get(3)?,
                            metadata: include_metadata.then(|| row.get::<Option<String>>(4)).transpose()?.flatten(),
                            created_at: row.get(5)?,
                            score: 0.0,
                            lexical_score: Some(lexical_score),
                            semantic_score: None,
                            weight: row.get(6)?,
                            retrieval_count: row.get::<i64>(7)? as u64,
                            last_retrieved_at: row.get(8)?,
                            superseded_at: row.get(9)?,
                        });
                        recency_boosts.insert(id, recency_boost(age_seconds));
                    } else if let Some(existing) = rows_data.get_mut(&id) {
                        existing.lexical_score = Some(lexical_score);
                    }

                    let rrf = 1.0 / (RRF_K + rank as f64);
                    *scores.entry(id).or_default() += rrf;
                    rank += 1;
                }
            }
        }

        // 3. Sort by final score
        let mut results: Vec<MemoryRow> = scores
            .into_iter()
            .filter_map(|(id, score)| {
                if let Some(mut row) = rows_data.remove(&id) {
                    let recency = recency_boosts.get(&id).copied().unwrap_or(1.0);
                    row.score = score * row.weight * recency;
                    Some(row)
                } else {
                    None
                }
            })
            .collect();

        // Sort descending by score
        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
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
