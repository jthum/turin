use anyhow::{Context, Result};

use super::{StateStore, WorkItemRow, WorklistRow};

pub struct WorkItemInsert<'a> {
    pub public_id: uuid::Uuid,
    pub worklist_id: i64,
    pub parent_item_id: Option<i64>,
    pub title: &'a str,
    pub item_kind: &'a str,
    pub prompt: Option<&'a str>,
    pub content: Option<&'a str>,
    pub tools: Option<&'a str>,
    pub conflict_policy: Option<&'a str>,
    pub action_name: Option<&'a str>,
    pub action_params: Option<&'a str>,
    pub priority: i64,
    pub after_ids: Option<&'a str>,
    pub metadata: Option<&'a str>,
}

#[derive(Default)]
pub struct WorkItemUpdate<'a> {
    pub id: i64,
    pub title: Option<&'a str>,
    pub prompt: Option<Option<&'a str>>,
    pub content: Option<Option<&'a str>>,
    pub tools: Option<Option<&'a str>>,
    pub conflict_policy: Option<Option<&'a str>>,
    pub action_name: Option<Option<&'a str>>,
    pub action_params: Option<Option<&'a str>>,
    pub priority: Option<i64>,
    pub after_ids: Option<Option<&'a str>>,
    pub metadata: Option<Option<&'a str>>,
    pub status: Option<&'a str>,
    pub failure_reason: Option<Option<&'a str>>,
}

const WORKLIST_SELECT: &str =
    "SELECT id, public_id, name, scope_ref, metadata, created_at, updated_at FROM worklists";

const WORK_ITEM_SELECT: &str = r#"
SELECT id, public_id, worklist_id, parent_item_id, title, item_kind, prompt, content, tools,
       conflict_policy, action_name, action_params, status, priority, after_ids, metadata,
       claim_agent_id, claim_session_id, claim_execution_id, claim_heartbeat_unix_ms,
       claimed_at, completed_at, failure_reason, created_at, updated_at
FROM work_items
"#;

impl StateStore {
    pub async fn list_worklists(&self) -> Result<Vec<WorklistRow>> {
        let conn = self.connect().await?;
        let sql = format!("{WORKLIST_SELECT} ORDER BY updated_at DESC, id DESC");
        let rows = conn.query(&sql, ()).await?;
        collect_mapped_rows(rows, map_worklist_row).await
    }

    pub async fn open_worklist(
        &self,
        name: &str,
        scope_ref: &str,
        metadata: Option<&str>,
    ) -> Result<WorklistRow> {
        let conn = self.connect().await?;
        let sql = format!("{WORKLIST_SELECT} WHERE name = ?1 AND scope_ref = ?2 LIMIT 1");
        let mut rows = conn.query(&sql, turso::params![name, scope_ref]).await?;
        if let Some(row) = rows.next().await? {
            return map_worklist_row(&row);
        }

        let public_id = uuid::Uuid::now_v7().into_bytes().to_vec();
        conn.execute(
            r#"
            INSERT INTO worklists (public_id, name, scope_ref, metadata, updated_at)
            VALUES (?1, ?2, ?3, ?4, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            "#,
            turso::params![public_id, name, scope_ref, metadata],
        )
        .await
        .context("Failed to insert worklist")?;

        let mut rows = conn.query(&sql, turso::params![name, scope_ref]).await?;
        let row = rows
            .next()
            .await?
            .ok_or_else(|| anyhow::anyhow!("Worklist '{}' was created but not visible", name))?;
        map_worklist_row(&row)
    }

    pub async fn get_worklist_by_public_id(
        &self,
        public_id: uuid::Uuid,
    ) -> Result<Option<WorklistRow>> {
        let conn = self.connect().await?;
        let sql = format!("{WORKLIST_SELECT} WHERE public_id = ?1 LIMIT 1");
        let mut rows = conn
            .query(&sql, turso::params![public_id.into_bytes().to_vec()])
            .await?;
        next_mapped_row(&mut rows, map_worklist_row).await
    }

    pub async fn get_worklist_by_id(&self, id: i64) -> Result<Option<WorklistRow>> {
        let conn = self.connect().await?;
        let sql = format!("{WORKLIST_SELECT} WHERE id = ?1 LIMIT 1");
        let mut rows = conn.query(&sql, turso::params![id]).await?;
        next_mapped_row(&mut rows, map_worklist_row).await
    }

    pub async fn list_work_items(&self, worklist_id: i64) -> Result<Vec<WorkItemRow>> {
        let conn = self.connect().await?;
        let sql =
            format!("{WORK_ITEM_SELECT} WHERE worklist_id = ?1 ORDER BY priority DESC, id ASC");
        let rows = conn.query(&sql, turso::params![worklist_id]).await?;
        collect_mapped_rows(rows, map_work_item_row).await
    }

    pub async fn get_work_item_by_public_id(
        &self,
        public_id: uuid::Uuid,
    ) -> Result<Option<WorkItemRow>> {
        let conn = self.connect().await?;
        let sql = format!("{WORK_ITEM_SELECT} WHERE public_id = ?1 LIMIT 1");
        let mut rows = conn
            .query(&sql, turso::params![public_id.into_bytes().to_vec()])
            .await?;
        next_mapped_row(&mut rows, map_work_item_row).await
    }

    pub async fn get_work_item_by_id(&self, id: i64) -> Result<Option<WorkItemRow>> {
        let conn = self.connect().await?;
        let sql = format!("{WORK_ITEM_SELECT} WHERE id = ?1 LIMIT 1");
        let mut rows = conn.query(&sql, turso::params![id]).await?;
        next_mapped_row(&mut rows, map_work_item_row).await
    }

    pub async fn create_work_item(&self, item: WorkItemInsert<'_>) -> Result<WorkItemRow> {
        let conn = self.connect().await?;
        conn.execute(
            r#"
            INSERT INTO work_items (
                public_id, worklist_id, parent_item_id, title, item_kind, prompt, content, tools,
                conflict_policy, action_name, action_params, priority, after_ids, metadata, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14,
                      strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            "#,
            turso::params![
                item.public_id.into_bytes().to_vec(),
                item.worklist_id,
                item.parent_item_id,
                item.title,
                item.item_kind,
                item.prompt,
                item.content,
                item.tools,
                item.conflict_policy,
                item.action_name,
                item.action_params,
                item.priority,
                item.after_ids,
                item.metadata
            ],
        )
        .await
        .context("Failed to insert work item")?;
        let row_id = conn.last_insert_rowid();
        self.require_visible_work_item_after(row_id, "created")
            .await
    }

    pub async fn update_work_item(&self, update: WorkItemUpdate<'_>) -> Result<WorkItemRow> {
        let current = self.require_work_item(update.id).await?;
        let conn = self.connect().await?;
        conn.execute(
            r#"
            UPDATE work_items
            SET title = ?2,
                prompt = ?3,
                content = ?4,
                tools = ?5,
                conflict_policy = ?6,
                action_name = ?7,
                action_params = ?8,
                priority = ?9,
                after_ids = ?10,
                metadata = ?11,
                status = ?12,
                failure_reason = ?13,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?1
            "#,
            turso::params![
                update.id,
                update.title.unwrap_or(&current.title),
                update.prompt.unwrap_or(current.prompt.as_deref()),
                update.content.unwrap_or(current.content.as_deref()),
                update.tools.unwrap_or(current.tools.as_deref()),
                update
                    .conflict_policy
                    .unwrap_or(current.conflict_policy.as_deref()),
                update.action_name.unwrap_or(current.action_name.as_deref()),
                update
                    .action_params
                    .unwrap_or(current.action_params.as_deref()),
                update.priority.unwrap_or(current.priority),
                update.after_ids.unwrap_or(current.after_ids.as_deref()),
                update.metadata.unwrap_or(current.metadata.as_deref()),
                update.status.unwrap_or(&current.status),
                update
                    .failure_reason
                    .unwrap_or(current.failure_reason.as_deref()),
            ],
        )
        .await
        .context("Failed to update work item")?;
        self.require_visible_work_item_after(update.id, "updated")
            .await
    }

    pub async fn try_claim_work_item(
        &self,
        id: i64,
        claim_agent_id: &str,
        claim_session_id: Option<&str>,
        claim_execution_id: Option<&str>,
        heartbeat_unix_ms: i64,
    ) -> Result<bool> {
        let conn = self.connect().await?;
        let changed = conn
            .execute(
                r#"
                UPDATE work_items
                SET status = 'active',
                    claim_agent_id = ?2,
                    claim_session_id = ?3,
                    claim_execution_id = ?4,
                    claim_heartbeat_unix_ms = ?5,
                    claimed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    completed_at = NULL,
                    failure_reason = NULL,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id = ?1
                  AND status IN ('pending', 'paused')
                  AND claim_execution_id IS NULL
                "#,
                turso::params![
                    id,
                    claim_agent_id,
                    claim_session_id,
                    claim_execution_id,
                    heartbeat_unix_ms
                ],
            )
            .await
            .context("Failed to claim work item")?;
        Ok(changed > 0)
    }

    pub async fn release_work_item(&self, id: i64) -> Result<WorkItemRow> {
        let current = self.require_work_item(id).await?;
        let metadata = clear_pause_metadata(current.metadata.as_deref())?;
        let conn = self.connect().await?;
        conn.execute(
            r#"
            UPDATE work_items
            SET status = 'pending',
                metadata = ?2,
                claim_agent_id = NULL,
                claim_session_id = NULL,
                claim_execution_id = NULL,
                claim_heartbeat_unix_ms = NULL,
                claimed_at = NULL,
                completed_at = NULL,
                failure_reason = NULL,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?1
            "#,
            turso::params![id, metadata],
        )
        .await
        .context("Failed to release work item")?;
        self.require_visible_work_item_after(id, "released").await
    }

    pub async fn pause_work_item(&self, id: i64, metadata: Option<&str>) -> Result<WorkItemRow> {
        let current = self.require_work_item(id).await?;
        let conn = self.connect().await?;
        conn.execute(
            r#"
            UPDATE work_items
            SET status = 'paused',
                metadata = ?2,
                claim_agent_id = NULL,
                claim_session_id = NULL,
                claim_execution_id = NULL,
                claim_heartbeat_unix_ms = NULL,
                claimed_at = NULL,
                completed_at = NULL,
                failure_reason = NULL,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?1
            "#,
            turso::params![id, metadata.or(current.metadata.as_deref())],
        )
        .await
        .context("Failed to pause work item")?;
        self.require_visible_work_item_after(id, "paused").await
    }

    pub async fn heartbeat_work_item_claim(
        &self,
        id: i64,
        claim_execution_id: &str,
        heartbeat_unix_ms: i64,
    ) -> Result<Option<WorkItemRow>> {
        let conn = self.connect().await?;
        let changed = conn
            .execute(
                r#"
                UPDATE work_items
                SET claim_heartbeat_unix_ms = ?3,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id = ?1
                  AND status = 'active'
                  AND claim_execution_id = ?2
                "#,
                turso::params![id, claim_execution_id, heartbeat_unix_ms],
            )
            .await
            .context("Failed to heartbeat work item claim")?;
        if changed == 0 {
            return Ok(None);
        }
        self.get_work_item_by_id(id).await
    }

    pub async fn complete_work_item(&self, id: i64, metadata: Option<&str>) -> Result<WorkItemRow> {
        let current = self.require_work_item(id).await?;
        let conn = self.connect().await?;
        conn.execute(
            r#"
            UPDATE work_items
            SET status = 'done',
                metadata = ?2,
                claim_agent_id = NULL,
                claim_session_id = NULL,
                claim_execution_id = NULL,
                claim_heartbeat_unix_ms = NULL,
                claimed_at = NULL,
                completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                failure_reason = NULL,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?1
            "#,
            turso::params![id, metadata.or(current.metadata.as_deref())],
        )
        .await
        .context("Failed to complete work item")?;
        self.require_visible_work_item_after(id, "completed").await
    }

    pub async fn fail_work_item(&self, id: i64, reason: Option<&str>) -> Result<WorkItemRow> {
        let conn = self.connect().await?;
        conn.execute(
            r#"
            UPDATE work_items
            SET status = 'failed',
                claim_agent_id = NULL,
                claim_session_id = NULL,
                claim_execution_id = NULL,
                claim_heartbeat_unix_ms = NULL,
                claimed_at = NULL,
                completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                failure_reason = ?2,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?1
            "#,
            turso::params![id, reason],
        )
        .await
        .context("Failed to fail work item")?;
        self.require_visible_work_item_after(id, "failed").await
    }

    async fn require_work_item(&self, id: i64) -> Result<WorkItemRow> {
        self.get_work_item_by_id(id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Work item {} not found", id))
    }

    async fn require_visible_work_item_after(&self, id: i64, action: &str) -> Result<WorkItemRow> {
        self.get_work_item_by_id(id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("Work item {} {} but not visible", id, action))
    }
}

fn clear_pause_metadata(metadata: Option<&str>) -> Result<Option<String>> {
    let Some(metadata) = metadata else {
        return Ok(None);
    };
    let mut value: serde_json::Value = serde_json::from_str(metadata)
        .context("Failed to parse work item metadata while clearing pause state")?;
    let serde_json::Value::Object(map) = &mut value else {
        return Ok(Some(metadata.to_string()));
    };
    map.remove("paused");
    map.remove("pause_reason");
    map.remove("pause_note");
    map.remove("pause_until_unix_ms");
    map.remove("paused_at_unix_ms");
    if map.is_empty() {
        return Ok(None);
    }
    Ok(Some(serde_json::to_string(&value)?))
}

async fn next_mapped_row<T>(
    rows: &mut turso::Rows,
    mapper: fn(&turso::Row) -> Result<T>,
) -> Result<Option<T>> {
    rows.next().await?.map(|row| mapper(&row)).transpose()
}

async fn collect_mapped_rows<T>(
    mut rows: turso::Rows,
    mapper: fn(&turso::Row) -> Result<T>,
) -> Result<Vec<T>> {
    let mut result = Vec::new();
    while let Some(row) = rows.next().await? {
        result.push(mapper(&row)?);
    }
    Ok(result)
}

fn map_worklist_row(row: &turso::Row) -> Result<WorklistRow> {
    Ok(WorklistRow {
        id: row.get::<i64>(0)?,
        public_id: row.get::<Vec<u8>>(1)?,
        name: row.get::<String>(2)?,
        scope_ref: row.get::<String>(3)?,
        metadata: row.get::<Option<String>>(4)?,
        created_at: row.get::<String>(5)?,
        updated_at: row.get::<String>(6)?,
    })
}

fn map_work_item_row(row: &turso::Row) -> Result<WorkItemRow> {
    Ok(WorkItemRow {
        id: row.get::<i64>(0)?,
        public_id: row.get::<Vec<u8>>(1)?,
        worklist_id: row.get::<i64>(2)?,
        parent_item_id: row.get::<Option<i64>>(3)?,
        title: row.get::<String>(4)?,
        item_kind: row.get::<String>(5)?,
        prompt: row.get::<Option<String>>(6)?,
        content: row.get::<Option<String>>(7)?,
        tools: row.get::<Option<String>>(8)?,
        conflict_policy: row.get::<Option<String>>(9)?,
        action_name: row.get::<Option<String>>(10)?,
        action_params: row.get::<Option<String>>(11)?,
        status: row.get::<String>(12)?,
        priority: row.get::<i64>(13)?,
        after_ids: row.get::<Option<String>>(14)?,
        metadata: row.get::<Option<String>>(15)?,
        claim_agent_id: row.get::<Option<String>>(16)?,
        claim_session_id: row.get::<Option<String>>(17)?,
        claim_execution_id: row.get::<Option<String>>(18)?,
        claim_heartbeat_unix_ms: row.get::<Option<i64>>(19)?,
        claimed_at: row.get::<Option<String>>(20)?,
        completed_at: row.get::<Option<String>>(21)?,
        failure_reason: row.get::<Option<String>>(22)?,
        created_at: row.get::<String>(23)?,
        updated_at: row.get::<String>(24)?,
    })
}
