//! Cognitive memory storage and lexical/vector/hybrid search.

use anyhow::{Context, Result};

use super::schema::MemoryRow;
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
    ) -> Result<()> {
        let metadata_str = serde_json::to_string(metadata)?;

        let conn = self.connect().await?;
        match vector {
            Some(vector) => {
                let mut vector_bytes = Vec::with_capacity(vector.len() * 4);
                for &val in vector {
                    vector_bytes.extend_from_slice(&val.to_le_bytes());
                }

                conn.execute(
                    "INSERT INTO memories (session_id, content, embedding, metadata) VALUES (?1, ?2, ?3, ?4)",
                    turso::params![session_id, content, vector_bytes, metadata_str],
                )
                .await
                .with_context(|| format!("Failed to insert memory for session: {}", session_id))?;
            }
            None => {
                conn.execute(
                    "INSERT INTO memories (session_id, content, metadata) VALUES (?1, ?2, ?3)",
                    turso::params![session_id, content, metadata_str],
                )
                .await
                .with_context(|| format!("Failed to insert memory for session: {}", session_id))?;
            }
        }
        Ok(())
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
    ) -> Result<Vec<MemoryRow>> {
        use std::collections::HashMap;

        // RRF constant k (usually 60)
        const RRF_K: f64 = 60.0;
        let mut scores: HashMap<i64, f64> = HashMap::new();
        let mut rows_data: HashMap<i64, MemoryRow> = HashMap::new();

        let conn = self.connect().await?;

        // 1. Vector Search
        if let Some(vec) = vector {
            // Convert to bytes
            let mut vector_bytes = Vec::with_capacity(vec.len() * 4);
            for &val in vec {
                vector_bytes.extend_from_slice(&val.to_le_bytes());
            }

            let mut rows = conn.query(
                "SELECT id, session_id, content, metadata, created_at, vector_distance_cos(embedding, ?1) as distance 
                 FROM memories 
                 WHERE session_id = ?2 AND embedding IS NOT NULL
                 ORDER BY distance ASC 
                 LIMIT ?3",
                turso::params![vector_bytes, session_id, limit as i64],
            ).await.context("Failed to search memories (vector)")?;

            let mut rank = 1;
            while let Some(row) = rows.next().await? {
                let id: i64 = row.get(0)?;

                // Track row data if not seen
                if let std::collections::hash_map::Entry::Vacant(e) = rows_data.entry(id) {
                    e.insert(MemoryRow {
                        id,
                        session_id: row.get(1)?,
                        content: row.get(2)?,
                        metadata: row.get(3)?,
                        created_at: row.get(4)?,
                        score: 0.0, // Re-calculated later
                    });
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
                let mut rows = conn
                    .query(
                        "SELECT id, session_id, content, metadata, created_at, fts_score(content, ?1) AS lexical_score
                         FROM memories
                         WHERE session_id = ?2
                         AND content MATCH ?1
                         ORDER BY lexical_score DESC
                         LIMIT ?3",
                        turso::params![query, session_id, limit as i64],
                    )
                    .await
                    .context("Failed to search memories (lexical)")?;

                let mut rank = 1;
                while let Some(row) = rows.next().await? {
                    let id: i64 = row.get(0)?;

                    if let std::collections::hash_map::Entry::Vacant(e) = rows_data.entry(id) {
                        e.insert(MemoryRow {
                            id,
                            session_id: row.get(1)?,
                            content: row.get(2)?,
                            metadata: row.get(3)?,
                            created_at: row.get(4)?,
                            score: 0.0,
                        });
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
                    row.score = score;
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

        Ok(results)
    }
}
