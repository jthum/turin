//! Cognitive memory storage, lifecycle, inspection, and retrieval.

use anyhow::{Context, Result};
use uuid::Uuid;

use super::schema::{
    MemoryCorrectionRow, MemoryFeedbackState, MemoryPurgeReport, MemoryStorageKind, StoredMemoryRow,
};
use super::state::StateStore;

mod inspection;
mod retrieval;

impl StateStore {
    // ─── Memories (Vector + Native FTS Hybrid Store) ──────────────

    /// Insert a memory, optionally with an embedding vector.
    #[allow(clippy::too_many_arguments)]
    pub async fn insert_memory(
        &self,
        scope_kind: &str,
        scope_key: &str,
        content: &str,
        vector: Option<&[f32]>,
        embedding_key: Option<&str>,
        embedding_dimensions: Option<usize>,
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
                let embedding_key =
                    embedding_key.context("Missing embedding key for embedded memory")?;
                let embedding_dimensions = embedding_dimensions
                    .context("Missing embedding dimensions for embedded memory")?;
                let vector_bytes = vector_to_le_bytes(vector);

                conn.execute(
                    "INSERT INTO memories (public_id, scope_kind, scope_key, content, embedding, embedding_key, embedding_dimensions, metadata) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    turso::params![
                        public_id_bytes.clone(),
                        scope_kind,
                        scope_key,
                        content,
                        vector_bytes,
                        embedding_key.to_string(),
                        embedding_dimensions as i64,
                        metadata_str
                    ],
                )
                .await
                .with_context(|| {
                    format!(
                        "Failed to insert memory for scope {}:{}",
                        scope_kind, scope_key
                    )
                })?;
            }
            None => {
                conn.execute(
                    "INSERT INTO memories (public_id, scope_kind, scope_key, content, metadata) VALUES (?1, ?2, ?3, ?4, ?5)",
                    turso::params![
                        public_id_bytes.clone(),
                        scope_kind,
                        scope_key,
                        content,
                        metadata_str
                    ],
                )
                .await
                .with_context(|| {
                    format!(
                        "Failed to insert memory for scope {}:{}",
                        scope_kind, scope_key
                    )
                })?;
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

    #[allow(clippy::too_many_arguments)]
    pub async fn apply_memory_feedback(
        &self,
        scope_kind: &str,
        scope_key: &str,
        public_id: Uuid,
        delta: f64,
        clamp_min: f64,
        clamp_max: f64,
        reason: Option<&str>,
        task_id: Option<&str>,
    ) -> Result<MemoryFeedbackState> {
        if clamp_min > clamp_max {
            anyhow::bail!("invalid feedback clamp: min cannot exceed max");
        }

        let mut conn = self.connect().await?;
        let tx = conn
            .transaction()
            .await
            .context("Failed to start memory feedback transaction")?;
        let public_id_bytes = public_id.into_bytes().to_vec();
        let mut rows = tx
            .query(
                r#"
                UPDATE memories
                SET weight = min(max(weight + ?4, ?5), ?6)
                WHERE scope_kind = ?1 AND scope_key = ?2 AND public_id = ?3
                RETURNING id, weight
                "#,
                turso::params![
                    scope_kind,
                    scope_key,
                    public_id_bytes.clone(),
                    delta,
                    clamp_min,
                    clamp_max
                ],
            )
            .await
            .context("Failed to apply memory feedback")?;
        let Some(row) = rows.next().await? else {
            anyhow::bail!("runtime.memory.feedback: memory not found");
        };
        let row_id = row.get::<i64>(0)?;
        let updated_weight = row.get::<f64>(1)?;
        drop(rows);
        let updated_at = current_utc_timestamp_in_transaction(&tx).await?;

        tx.execute(
            "INSERT INTO memory_feedback_events (memory_id, delta, reason, task_id, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            turso::params![row_id, delta, reason, task_id, updated_at.clone()],
        )
        .await
        .with_context(|| format!("Failed to insert feedback event for memory {}", row_id))?;
        tx.commit()
            .await
            .context("Failed to commit memory feedback transaction")?;

        Ok(MemoryFeedbackState {
            public_id: public_id_bytes,
            weight: updated_weight,
            updated_at,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn correct_memory(
        &self,
        scope_kind: &str,
        scope_key: &str,
        public_id: Uuid,
        content: &str,
        vector: Option<&[f32]>,
        embedding_key: Option<&str>,
        embedding_dimensions: Option<usize>,
        metadata: &serde_json::Value,
    ) -> Result<MemoryCorrectionRow> {
        let metadata_str = serde_json::to_string(metadata)?;
        let (vector_bytes, embedding_key, embedding_dimensions) = match vector {
            Some(vector) => (
                Some(vector_to_le_bytes(vector)),
                Some(
                    embedding_key
                        .context("Missing embedding key for embedded memory")?
                        .to_string(),
                ),
                Some(
                    i64::try_from(
                        embedding_dimensions
                            .context("Missing embedding dimensions for embedded memory")?,
                    )
                    .context("Embedding dimensions exceed the persisted integer range")?,
                ),
            ),
            None => (None, None, None),
        };
        let mut conn = self.connect().await?;
        let tx = conn
            .transaction()
            .await
            .context("Failed to start memory correction transaction")?;
        let public_id_bytes = public_id.into_bytes().to_vec();
        let mut rows = tx
            .query(
                "SELECT id, superseded_at FROM memories WHERE scope_kind = ?1 AND scope_key = ?2 AND public_id = ?3",
                turso::params![scope_kind, scope_key, public_id_bytes.clone()],
            )
            .await
            .context("Failed to look up memory for correction")?;
        let Some(row) = rows.next().await? else {
            anyhow::bail!("runtime.memory.correct: memory not found");
        };
        let old_row_id = row.get::<i64>(0)?;
        let already_superseded = row.get::<Option<String>>(1)?.is_some();
        drop(rows);
        if already_superseded {
            anyhow::bail!("runtime.memory.correct: memory is already superseded");
        }

        let replacement_public_id = Uuid::now_v7().into_bytes().to_vec();
        tx.execute(
            r#"
            INSERT INTO memories (
                public_id, scope_kind, scope_key, content, embedding,
                embedding_key, embedding_dimensions, metadata
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
            "#,
            turso::params![
                replacement_public_id.clone(),
                scope_kind,
                scope_key,
                content,
                vector_bytes,
                embedding_key,
                embedding_dimensions,
                metadata_str
            ],
        )
        .await
        .context("Failed to store corrected memory")?;
        let replacement_row_id = tx.last_insert_rowid();
        let corrected_at = current_utc_timestamp_in_transaction(&tx).await?;

        let changed = tx
            .execute(
                r#"
                UPDATE memories
                SET superseded_at = ?2, superseded_by_memory_id = ?3
                WHERE id = ?1 AND superseded_at IS NULL
                "#,
                turso::params![old_row_id, corrected_at.clone(), replacement_row_id],
            )
            .await
            .with_context(|| format!("Failed to mark memory {} as superseded", old_row_id))?;
        anyhow::ensure!(
            changed == 1,
            "runtime.memory.correct: memory was concurrently superseded"
        );
        tx.commit()
            .await
            .context("Failed to commit memory correction transaction")?;

        Ok(MemoryCorrectionRow {
            superseded_public_id: public_id_bytes,
            replacement_public_id,
            corrected_at,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn purge_memories(
        &self,
        scope_kind: &str,
        scope_key: &str,
        older_than_days: Option<u64>,
        min_weight: Option<f64>,
        max_retrieval_count: Option<u64>,
        only_superseded: bool,
        all: bool,
        dry_run: bool,
    ) -> Result<MemoryPurgeReport> {
        let mut conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT id,
                        CAST((julianday('now') - julianday(created_at)) AS REAL) AS age_days,
                        weight,
                        retrieval_count,
                        superseded_at
                 FROM memories
                 WHERE scope_kind = ?1 AND scope_key = ?2",
                turso::params![scope_kind, scope_key],
            )
            .await
            .context("Failed to enumerate memories for purge")?;

        let has_filters = all
            || older_than_days.is_some()
            || min_weight.is_some()
            || max_retrieval_count.is_some()
            || only_superseded;
        if !has_filters {
            return Ok(MemoryPurgeReport {
                matched: 0,
                deleted: 0,
                dry_run,
            });
        }

        let mut matched_ids = Vec::new();
        while let Some(row) = rows.next().await? {
            let row_id: i64 = row.get(0)?;
            let age_days = row.get::<Option<f64>>(1)?.unwrap_or(0.0);
            let weight: f64 = row.get(2)?;
            let retrieval_count = super::state::persisted_u64(
                &format!("memory {row_id}"),
                "retrieval count",
                row.get::<i64>(3)?,
            )?;
            let is_superseded = row.get::<Option<String>>(4)?.is_some();

            let matches = if all {
                !only_superseded || is_superseded
            } else {
                let mut matched = true;
                if let Some(days) = older_than_days {
                    matched &= age_days >= days as f64;
                }
                if let Some(weight_ceiling) = min_weight {
                    matched &= weight <= weight_ceiling;
                }
                if let Some(retrieval_ceiling) = max_retrieval_count {
                    matched &= retrieval_count <= retrieval_ceiling;
                }
                if only_superseded {
                    matched &= is_superseded;
                }
                matched
            };

            if matches {
                matched_ids.push(row_id);
            }
        }
        drop(rows);

        if dry_run || matched_ids.is_empty() {
            return Ok(MemoryPurgeReport {
                matched: matched_ids.len(),
                deleted: 0,
                dry_run,
            });
        }

        let tx = conn
            .transaction()
            .await
            .context("Failed to start memory purge transaction")?;
        for row_id in &matched_ids {
            tx.execute(
                "DELETE FROM memory_feedback_events WHERE memory_id = ?1",
                turso::params![row_id],
            )
            .await
            .with_context(|| format!("Failed to delete feedback events for memory {}", row_id))?;
            tx.execute("DELETE FROM memories WHERE id = ?1", turso::params![row_id])
                .await
                .with_context(|| format!("Failed to delete memory {}", row_id))?;
        }
        tx.commit()
            .await
            .context("Failed to commit memory purge transaction")?;

        Ok(MemoryPurgeReport {
            matched: matched_ids.len(),
            deleted: matched_ids.len(),
            dry_run,
        })
    }
}

pub(super) fn vector_to_le_bytes(vector: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(vector.len() * 4);
    for &val in vector {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

async fn current_utc_timestamp_in_transaction(
    tx: &turso::transaction::Transaction<'_>,
) -> Result<String> {
    let mut rows = tx
        .query("SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')", ())
        .await
        .context("Failed to fetch current UTC timestamp")?;
    rows.next()
        .await?
        .map(|row| row.get::<String>(0))
        .transpose()?
        .context("Timestamp query returned no row")
}
