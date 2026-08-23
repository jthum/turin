use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};

use super::{
    LinkedSessionCreate, LinkedSessionFamilyStats, SessionRow, StateStore,
    update_session_title_metadata, validate_session_metadata,
};

mod search;

pub(super) const SESSION_SELECT: &str = "SELECT id, public_id, agent_id, origin_id, metadata, active_branch_head_id, parent_session_id, root_session_id, origin_turn_id, relation_kind, thread_key, visibility, created_at FROM sessions";

impl StateStore {
    pub async fn create_session(
        &self,
        public_id: uuid::Uuid,
        agent_id: &str,
        metadata: Option<&str>,
    ) -> Result<i64> {
        self.create_session_with_origin(public_id, agent_id, None, metadata)
            .await
    }

    pub async fn create_session_with_origin(
        &self,
        public_id: uuid::Uuid,
        agent_id: &str,
        origin_id: Option<&str>,
        metadata: Option<&str>,
    ) -> Result<i64> {
        validate_session_metadata("new session", metadata)?;
        let mut conn = self.connect().await?;
        let tx = conn
            .transaction()
            .await
            .context("Failed to start session creation transaction")?;
        let public_id_bytes = public_id.into_bytes().to_vec();

        tx.execute(
            "INSERT INTO sessions (public_id, agent_id, origin_id, metadata) VALUES (?1, ?2, ?3, ?4)",
            turso::params![public_id_bytes, agent_id, origin_id, metadata],
        )
        .await
        .context("Failed to insert into sessions table")?;

        let session_id = tx.last_insert_rowid();
        let branch_public_id = uuid::Uuid::now_v7().into_bytes().to_vec();
        tx.execute(
            "INSERT INTO branch_heads (public_id, session_id, name, origin_kind) VALUES (?1, ?2, 'main', 'main')",
            turso::params![branch_public_id, session_id],
        )
        .await
        .context("Failed to insert initial main branch head")?;
        let branch_id = tx.last_insert_rowid();
        let changed = tx
            .execute(
                "UPDATE sessions SET active_branch_head_id = ?1 WHERE id = ?2",
                turso::params![branch_id, session_id],
            )
            .await
            .context("Failed to activate main branch head")?;
        anyhow::ensure!(changed == 1, "New session disappeared before activation");
        tx.commit()
            .await
            .context("Failed to commit session creation transaction")?;
        Ok(session_id)
    }

    pub async fn create_linked_session(
        &self,
        public_id: uuid::Uuid,
        agent_id: &str,
        metadata: Option<&str>,
        link: &LinkedSessionCreate,
    ) -> Result<i64> {
        validate_session_metadata("new linked session", metadata)?;
        anyhow::ensure!(
            matches!(link.visibility.as_str(), "contextual" | "hidden"),
            "Linked session visibility must be 'contextual' or 'hidden'"
        );
        anyhow::ensure!(
            !link.relation_kind.trim().is_empty(),
            "Linked session relation kind must not be empty"
        );
        anyhow::ensure!(
            !link.thread_key.trim().is_empty(),
            "Linked session thread key must not be empty"
        );

        let parent = self
            .get_session_row(link.parent_session_id)
            .await?
            .with_context(|| format!("Parent session '{}' not found", link.parent_session_id))?;
        if let Some(origin_turn_id) = link.origin_turn_id {
            let origin = self
                .get_turn_row(origin_turn_id)
                .await?
                .with_context(|| format!("Origin turn '{}' not found", origin_turn_id))?;
            anyhow::ensure!(
                origin.session_id == parent.id,
                "Origin turn '{}' does not belong to parent session '{}'",
                origin_turn_id,
                parent.id
            );
        }

        let root_session_id = parent.root_session_id.unwrap_or(parent.id);
        let mut conn = self.connect().await?;
        let tx = conn
            .transaction()
            .await
            .context("Failed to start linked session creation transaction")?;
        let public_id_bytes = public_id.into_bytes().to_vec();
        tx.execute(
            r#"
            INSERT INTO sessions (
                public_id, agent_id, origin_id, metadata, parent_session_id, root_session_id,
                origin_turn_id, relation_kind, thread_key, visibility
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
            "#,
            turso::params![
                public_id_bytes,
                agent_id,
                parent.origin_id.as_deref(),
                metadata,
                link.parent_session_id,
                root_session_id,
                link.origin_turn_id,
                link.relation_kind.trim(),
                link.thread_key.trim(),
                link.visibility.as_str(),
            ],
        )
        .await
        .context("Failed to insert linked session")?;

        let session_id = tx.last_insert_rowid();
        let branch_public_id = uuid::Uuid::now_v7().into_bytes().to_vec();
        tx.execute(
            "INSERT INTO branch_heads (public_id, session_id, name, origin_kind) VALUES (?1, ?2, 'main', 'main')",
            turso::params![branch_public_id, session_id],
        )
        .await
        .context("Failed to insert initial main branch head")?;
        let branch_id = tx.last_insert_rowid();
        let changed = tx
            .execute(
                "UPDATE sessions SET active_branch_head_id = ?1 WHERE id = ?2",
                turso::params![branch_id, session_id],
            )
            .await
            .context("Failed to activate main branch head")?;
        anyhow::ensure!(
            changed == 1,
            "New linked session disappeared before activation"
        );
        tx.commit()
            .await
            .context("Failed to commit linked session creation transaction")?;
        Ok(session_id)
    }

    pub async fn find_linked_session(
        &self,
        parent_session_id: i64,
        agent_id: &str,
        thread_key: &str,
    ) -> Result<Option<SessionRow>> {
        let conn = self.connect().await?;
        let sql = format!(
            "{SESSION_SELECT} WHERE parent_session_id = ?1 AND agent_id = ?2 AND thread_key = ?3"
        );
        let mut rows = conn
            .query(
                &sql,
                turso::params![parent_session_id, agent_id, thread_key],
            )
            .await?;
        rows.next()
            .await?
            .map(|row| map_session_row(&row))
            .transpose()
    }

    pub async fn list_linked_session_rows(
        &self,
        parent_session_id: i64,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<SessionRow>> {
        self.list_linked_session_rows_with_archived(parent_session_id, limit, offset, false)
            .await
    }

    pub async fn list_linked_session_rows_with_archived(
        &self,
        parent_session_id: i64,
        limit: usize,
        offset: usize,
        include_archived: bool,
    ) -> Result<Vec<SessionRow>> {
        let conn = self.connect().await?;
        let archived_filter = if include_archived {
            ""
        } else {
            " AND visibility != 'archived'"
        };
        let sql = format!(
            "{SESSION_SELECT} WHERE parent_session_id = ?1{archived_filter} ORDER BY created_at DESC, id DESC LIMIT ?2 OFFSET ?3"
        );
        let mut rows = conn
            .query(
                &sql,
                turso::params![parent_session_id, limit as i64, offset as i64],
            )
            .await?;
        let mut sessions = Vec::new();
        while let Some(row) = rows.next().await? {
            sessions.push(map_session_row(&row)?);
        }
        Ok(sessions)
    }

    pub async fn archive_linked_session_family(&self, session_id: i64) -> Result<usize> {
        let Some(session) = self.get_session_row(session_id).await? else {
            return Ok(0);
        };
        anyhow::ensure!(
            session.parent_session_id.is_some(),
            "Top-level sessions cannot be archived as linked work"
        );
        let mut family = self.list_linked_session_descendants(session_id).await?;
        family.push(session);
        let mut conn = self.connect().await?;
        let tx = conn.transaction().await?;
        for row in &family {
            tx.execute(
                "UPDATE sessions SET visibility = 'archived' WHERE id = ?1",
                [row.id],
            )
            .await?;
        }
        tx.commit().await?;
        Ok(family.len())
    }

    pub async fn restore_linked_session(&self, session_id: i64) -> Result<()> {
        let conn = self.connect().await?;
        conn.execute(
            "UPDATE sessions SET visibility = 'contextual' WHERE id = ?1 AND visibility = 'archived'",
            [session_id],
        )
        .await?;
        Ok(())
    }

    /// Count one session's children and descendants without loading transcripts or session rows.
    pub async fn linked_session_family_stats(
        &self,
        session_id: i64,
    ) -> Result<Option<LinkedSessionFamilyStats>> {
        let conn = self.connect().await?;
        let mut target_rows = conn
            .query(
                "SELECT root_session_id FROM sessions WHERE id = ?1",
                [session_id],
            )
            .await?;
        let Some(target) = target_rows.next().await? else {
            return Ok(None);
        };
        let root_session_id = target.get::<Option<i64>>(0)?.unwrap_or(session_id);
        drop(target_rows);

        let mut rows = conn
            .query(
                "SELECT id, parent_session_id FROM sessions WHERE id = ?1 OR root_session_id = ?1",
                [root_session_id],
            )
            .await?;
        let mut family_size = 0usize;
        let mut children: HashMap<i64, Vec<i64>> = HashMap::new();
        let mut parents = HashMap::new();
        while let Some(row) = rows.next().await? {
            family_size += 1;
            let id = row.get::<i64>(0)?;
            if let Some(parent_id) = row.get::<Option<i64>>(1)? {
                children.entry(parent_id).or_default().push(id);
                parents.insert(id, parent_id);
            }
        }

        let mut depth = 0usize;
        let mut ancestor_id = session_id;
        let mut ancestors = HashSet::from([session_id]);
        while let Some(parent_id) = parents.get(&ancestor_id).copied() {
            anyhow::ensure!(
                ancestors.insert(parent_id),
                "Linked session ancestry contains a cycle at session '{}'",
                parent_id
            );
            depth += 1;
            ancestor_id = parent_id;
        }

        let direct_child_count = children.get(&session_id).map_or(0, Vec::len);
        let mut visited = HashSet::from([session_id]);
        let mut stack = children.get(&session_id).cloned().unwrap_or_default();
        let mut descendant_count = 0usize;
        while let Some(current_id) = stack.pop() {
            anyhow::ensure!(
                visited.insert(current_id),
                "Linked session ancestry contains a cycle at session '{}'",
                current_id
            );
            descendant_count += 1;
            if let Some(child_ids) = children.get(&current_id) {
                stack.extend(child_ids);
            }
        }

        Ok(Some(LinkedSessionFamilyStats {
            depth,
            direct_child_count,
            descendant_count,
            root_family_size: family_size,
        }))
    }

    /// Return descendants in deletion order (deepest children first).
    pub async fn list_linked_session_descendants(
        &self,
        session_id: i64,
    ) -> Result<Vec<SessionRow>> {
        let Some(target) = self.get_session_row(session_id).await? else {
            return Ok(Vec::new());
        };
        let root_session_id = target.root_session_id.unwrap_or(target.id);
        let conn = self.connect().await?;
        let sql = format!("{SESSION_SELECT} WHERE id = ?1 OR root_session_id = ?1 ORDER BY id ASC");
        let mut rows = conn.query(&sql, [root_session_id]).await?;
        let mut family = HashMap::new();
        while let Some(row) = rows.next().await? {
            let row = map_session_row(&row)?;
            family.insert(row.id, row);
        }

        let mut children: HashMap<i64, Vec<i64>> = HashMap::new();
        for row in family.values() {
            if let Some(parent_id) = row.parent_session_id {
                children.entry(parent_id).or_default().push(row.id);
            }
        }

        let mut descendants = Vec::new();
        let mut stack = vec![(session_id, false)];
        while let Some((current_id, expanded)) = stack.pop() {
            if expanded {
                if current_id != session_id
                    && let Some(row) = family.get(&current_id)
                {
                    descendants.push(row.clone());
                }
                continue;
            }
            stack.push((current_id, true));
            if let Some(child_ids) = children.get(&current_id) {
                stack.extend(child_ids.iter().map(|child_id| (*child_id, false)));
            }
        }
        Ok(descendants)
    }

    pub async fn get_session_by_public_id(&self, public_id: uuid::Uuid) -> Result<Option<i64>> {
        let conn = self.connect().await?;
        let public_id_bytes = public_id.into_bytes().to_vec();

        let mut rows = conn
            .query(
                "SELECT id FROM sessions WHERE public_id = ?1",
                turso::params![public_id_bytes],
            )
            .await?;

        if let Some(row) = rows.next().await? {
            Ok(Some(row.get(0)?))
        } else {
            Ok(None)
        }
    }

    pub async fn get_session_row(&self, session_id: i64) -> Result<Option<SessionRow>> {
        let conn = self.connect().await?;
        let sql = format!("{SESSION_SELECT} WHERE id = ?1");
        let mut rows = conn.query(&sql, [session_id]).await?;

        if let Some(row) = rows.next().await? {
            Ok(Some(map_session_row(&row)?))
        } else {
            Ok(None)
        }
    }

    pub async fn get_session_row_by_public_id(
        &self,
        public_id: uuid::Uuid,
    ) -> Result<Option<SessionRow>> {
        let conn = self.connect().await?;
        let public_id_bytes = public_id.into_bytes().to_vec();
        let sql = format!("{SESSION_SELECT} WHERE public_id = ?1");
        let mut rows = conn.query(&sql, turso::params![public_id_bytes]).await?;

        if let Some(row) = rows.next().await? {
            Ok(Some(map_session_row(&row)?))
        } else {
            Ok(None)
        }
    }

    pub async fn update_session_title(
        &self,
        public_id: uuid::Uuid,
        title: Option<&str>,
    ) -> Result<Option<SessionRow>> {
        let Some(mut row) = self.get_session_row_by_public_id(public_id).await? else {
            return Ok(None);
        };

        let metadata = update_session_title_metadata(row.metadata.as_deref(), title)?;
        let conn = self.connect().await?;
        let public_id_bytes = public_id.into_bytes().to_vec();
        conn.execute(
            "UPDATE sessions SET metadata = ?1 WHERE public_id = ?2",
            turso::params![metadata.clone(), public_id_bytes],
        )
        .await
        .context("Failed to update session metadata title")?;
        row.metadata = metadata;
        Ok(Some(row))
    }

    pub async fn update_session_title_if_empty(
        &self,
        public_id: uuid::Uuid,
        title: &str,
    ) -> Result<Option<SessionRow>> {
        let title = title.trim();
        let Some(existing) = self.get_session_row_by_public_id(public_id).await? else {
            return Ok(None);
        };
        let updated_metadata =
            update_session_title_metadata(existing.metadata.as_deref(), Some(title))?;
        if existing
            .metadata
            .as_deref()
            .and_then(|metadata| serde_json::from_str::<serde_json::Value>(metadata).ok())
            .and_then(|metadata| metadata.get("title").cloned())
            .and_then(|title| title.as_str().map(str::to_owned))
            .is_some_and(|title| !title.trim().is_empty())
        {
            return Ok(Some(existing));
        }

        let conn = self.connect().await?;
        let public_id_bytes = public_id.into_bytes().to_vec();
        conn.execute(
            r#"
UPDATE sessions
SET metadata = ?1
WHERE public_id = ?2
  AND metadata IS ?3
"#,
            turso::params![updated_metadata, public_id_bytes, existing.metadata],
        )
        .await
        .context("Failed to set empty session metadata title")?;
        self.get_session_row_by_public_id(public_id).await
    }

    pub async fn delete_session_by_public_id(&self, public_id: uuid::Uuid) -> Result<bool> {
        let Some(session) = self.get_session_row_by_public_id(public_id).await? else {
            return Ok(false);
        };
        let mut family_public_ids = Vec::new();
        for descendant in self.list_linked_session_descendants(session.id).await? {
            let descendant_public_id = uuid::Uuid::from_slice(&descendant.public_id)
                .context("Linked session has an invalid public id")?;
            family_public_ids.push(descendant_public_id);
        }
        family_public_ids.push(public_id);

        let mut conn = self.connect().await?;
        let tx = conn
            .transaction()
            .await
            .context("Failed to start session family deletion transaction")?;
        let mut root_deleted = false;
        for family_public_id in family_public_ids {
            match delete_single_session_in_transaction(&tx, family_public_id).await {
                Ok(deleted) => root_deleted = deleted,
                Err(error) => {
                    tx.rollback()
                        .await
                        .context("Failed to roll back session family deletion")?;
                    return Err(error);
                }
            }
        }
        tx.commit()
            .await
            .context("Failed to commit session family deletion transaction")?;
        Ok(root_deleted)
    }
}

async fn delete_single_session_in_transaction(
    tx: &turso::transaction::Transaction<'_>,
    public_id: uuid::Uuid,
) -> Result<bool> {
    let public_id_bytes = public_id.into_bytes().to_vec();
    let mut rows = tx
        .query(
            "SELECT id FROM sessions WHERE public_id = ?1",
            turso::params![public_id_bytes.clone()],
        )
        .await?;
    let Some(row) = rows.next().await? else {
        return Ok(false);
    };
    let session_id = row.get::<i64>(0)?;
    drop(rows);

    let bare_id = public_id.simple().to_string();
    let hyphenated_id = public_id.to_string();
    let scoped_data_predicate = r#"
scope_kind = 'session' AND (
    scope_key = ?1 OR scope_key = ?2 OR
    (json_valid(scope_key) AND json_extract(scope_key, '$.key') IN (?1, ?2))
)"#;

    tx.execute(
        r#"
            UPDATE work_items
            SET status = 'pending',
                claim_agent_id = NULL,
                claim_session_id = NULL,
                claim_execution_id = NULL,
                claim_heartbeat_unix_ms = NULL,
                claimed_at = NULL,
                completed_at = NULL,
                failure_reason = NULL,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE claim_session_id IN (?1, ?2)
               OR claim_session_id LIKE ?3
               OR claim_session_id LIKE ?4
            "#,
        turso::params![
            bare_id.clone(),
            hyphenated_id.clone(),
            format!("{}@%", bare_id),
            format!("{}@%", hyphenated_id),
        ],
    )
    .await?;
    tx.execute(
            &format!(
                "DELETE FROM memory_feedback_events WHERE memory_id IN (SELECT id FROM memories WHERE {scoped_data_predicate})"
            ),
            turso::params![bare_id.clone(), hyphenated_id.clone()],
        )
        .await?;
    tx.execute(
            &format!(
                "UPDATE memories SET superseded_by_memory_id = NULL WHERE superseded_by_memory_id IN (SELECT id FROM memories WHERE {scoped_data_predicate})"
            ),
            turso::params![bare_id.clone(), hyphenated_id.clone()],
        )
        .await?;
    tx.execute(
        &format!("DELETE FROM memories WHERE {scoped_data_predicate}"),
        turso::params![bare_id.clone(), hyphenated_id.clone()],
    )
    .await?;
    tx.execute(
        &format!("DELETE FROM kv WHERE {scoped_data_predicate}"),
        turso::params![bare_id, hyphenated_id],
    )
    .await?;

    tx.execute(
        "UPDATE sessions SET active_branch_head_id = NULL WHERE id = ?1",
        [session_id],
    )
    .await?;
    tx.execute(
        "DELETE FROM messages WHERE turn_id IN (SELECT id FROM turns WHERE session_id = ?1)",
        [session_id],
    )
    .await?;
    tx.execute(
        "DELETE FROM tool_executions WHERE turn_id IN (SELECT id FROM turns WHERE session_id = ?1)",
        [session_id],
    )
    .await?;
    tx.execute("DELETE FROM events WHERE session_id = ?1", [session_id])
        .await?;
    tx.execute(
        "DELETE FROM graph_edges WHERE session_id = ?1",
        [session_id],
    )
    .await?;
    tx.execute(
        "DELETE FROM graph_nodes WHERE session_id = ?1",
        [session_id],
    )
    .await?;
    tx.execute(
        "DELETE FROM branch_heads WHERE session_id = ?1",
        [session_id],
    )
    .await?;
    tx.execute("DELETE FROM turns WHERE session_id = ?1", [session_id])
        .await?;
    tx.execute("DELETE FROM sessions WHERE id = ?1", [session_id])
        .await?;
    Ok(true)
}

pub(super) fn map_session_row(row: &turso::Row) -> Result<SessionRow> {
    let id = row.get::<i64>(0)?;
    let metadata = row.get::<Option<String>>(4)?;
    Ok(SessionRow {
        id,
        public_id: row.get::<Vec<u8>>(1)?,
        agent_id: row.get::<String>(2)?,
        origin_id: row.get::<Option<String>>(3)?,
        metadata,
        active_branch_head_id: row.get::<Option<i64>>(5)?,
        parent_session_id: row.get::<Option<i64>>(6)?,
        root_session_id: row.get::<Option<i64>>(7)?,
        origin_turn_id: row.get::<Option<i64>>(8)?,
        relation_kind: row.get::<Option<String>>(9)?,
        thread_key: row.get::<Option<String>>(10)?,
        visibility: row.get::<String>(11)?,
        created_at: row.get::<String>(12)?,
    })
}
