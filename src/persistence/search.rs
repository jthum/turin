//! Cognitive memory storage and hybrid search (Vector + FTS5 + LIKE fallback).

use anyhow::{Context, Result};

use super::schema::MemoryRow;
use super::state::StateStore;

impl StateStore {
    // ─── Memories (Vector + FTS Hybrid Store) ─────────────────────

    /// Insert a memory with an embedding vector.
    pub async fn insert_memory(
        &self,
        session_id: &str,
        content: &str,
        vector: &[f32],
        metadata: &serde_json::Value,
    ) -> Result<()> {
        // Convert vector to raw bytes (little endian)
        let mut vector_bytes = Vec::with_capacity(vector.len() * 4);
        for &val in vector {
            vector_bytes.extend_from_slice(&val.to_le_bytes());
        }

        let metadata_str = serde_json::to_string(metadata)?;
        
        let conn = self.db.connect()?;
        conn
            .execute(
                "INSERT INTO memories (session_id, content, embedding, metadata) VALUES (?1, ?2, ?3, ?4)",
                turso::params![
                    session_id,
                    content,
                    vector_bytes,
                    metadata_str,
                ],
            )
            .await
            .with_context(|| format!("Failed to insert memory for session: {}", session_id))?;
        Ok(())
    }

    /// Search memories using Hybrid Search (Vector + FTS5).
    /// 
    /// Uses Reciprocal Rank Fusion (RRF) to combine results.
    /// - `vector`: Optional embedding for semantic search.
    /// - `content_query`: Optional keyword string for FTS search. if None, relies only on vector.
    pub async fn search_memories(
        &self,
        session_id: &str,
        vector: Option<&[f32]>,
        content_query: Option<&str>,
        limit: usize,
    ) -> Result<Vec<MemoryRow>> {
        use std::collections::HashMap;

        // RRF constant k (usually 60)
        const RRF_K: f64 = 60.0;
        let mut scores: HashMap<i64, f64> = HashMap::new();
        let mut rows_data: HashMap<i64, MemoryRow> = HashMap::new();

        let conn = self.db.connect()?;

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
                 WHERE session_id = ?2 
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

        // 2. FTS Search
        if let Some(query) = content_query {
            // Trim and verify query isn't empty
            let query = query.trim();
            if !query.is_empty() {
                 match conn.query(
                    "SELECT rowid, rank FROM memories_fts 
                     WHERE memories_fts MATCH ?1 
                     ORDER BY rank 
                     LIMIT ?2",
                    turso::params![query, limit as i64],
                ).await {
                    Ok(mut rows) => {
                        let mut rank = 1;
                        while let Some(row) = rows.next().await? {
                            let id: i64 = row.get(0)?;
        
                            // If we haven't fetched this row's data yet (from vector search), we need to fetch it
                            if let std::collections::hash_map::Entry::Vacant(e) = rows_data.entry(id) {
                                    // Fetch full row data
                                let mut full_row_q = conn.query(
                                    "SELECT session_id, content, metadata, created_at FROM memories WHERE id = ?1", 
                                    [id]
                                ).await?;
                                if let Some(full_row) = full_row_q.next().await? {
                                    e.insert(MemoryRow {
                                        id,
                                        session_id: full_row.get(0)?,
                                        content: full_row.get(1)?,
                                        metadata: full_row.get(2)?,
                                        created_at: full_row.get(3)?,
                                        score: 0.0, 
                                    });
                                }
                            }

                            // RRF score addition
                            let rrf = 1.0 / (RRF_K + rank as f64);
                            *scores.entry(id).or_default() += rrf;
                            rank += 1;
                        }
                    },
                    Err(e) => {
                        // Check if error is due to missing FTS table
                        let err_str = e.to_string();
                        if !err_str.contains("no such table") && !err_str.contains("no such module") {
                             eprintln!("[WARN] FTS search failed: {}", e);
                        }
                    }
                }
            }
        }

        // 3. Fallback: Tokenized LIKE (Scenario D)
        // Runs when both vector and FTS produced no results but we have a keyword query.
        if scores.is_empty()
             && let Some(query) = content_query {
                let query = query.trim();
                // Tokenize by whitespace
                let terms: Vec<&str> = query.split_whitespace().collect();
                if !terms.is_empty() {
                    let mut sql = "SELECT id, session_id, content, metadata, created_at FROM memories WHERE session_id = ?1 AND (".to_string();
                    let mut params = vec![turso::Value::from(session_id.to_string())];
                    
                    for (i, term) in terms.iter().enumerate() {
                        if i > 0 {
                            sql.push_str(" OR ");
                        }
                        sql.push_str(&format!("content LIKE ?{}", i + 2));
                        params.push(turso::Value::from(format!("%{}%", term)));
                    }
                    sql.push_str(") ORDER BY id DESC LIMIT ?");
                    sql.push_str(&(terms.len() + 2).to_string());
                    params.push(turso::Value::from(limit as i64));

                    let mut rows = conn.query(&sql, params).await.context("Failed to execute fallback LIKE search")?;
                    
                     while let Some(row) = rows.next().await? {
                         let id: i64 = row.get(0)?;
                        rows_data.insert(id, MemoryRow {
                            id,
                            session_id: row.get(1)?,
                            content: row.get(2)?,
                            metadata: row.get(3)?,
                            created_at: row.get(4)?,
                            score: 0.1, // Low score to indicate fallback
                        });
                        scores.insert(id, 0.1);
                     }
                }
             }

        // 4. Sort by final score
        let mut results: Vec<MemoryRow> = scores.into_iter().filter_map(|(id, score)| {
            if let Some(mut row) = rows_data.remove(&id) {
                row.score = score;
                Some(row)
            } else {
                None
            }
        }).collect();

        // Sort descending by score
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(results)
    }
}
