use anyhow::{Context, Result, anyhow};

use super::super::{BranchHeadRow, BranchProvenance, StateStore};

const BRANCH_HEAD_SELECT: &str = r#"
SELECT bh.id,
       bh.public_id,
       bh.session_id,
       bh.name,
       bh.head_turn_id,
       t.branch_depth,
       bh.created_from_turn_id,
       bh.origin_kind,
       bh.origin_task_id,
       bh.origin_execution_id,
       bh.origin_metadata,
       bh.created_at,
       CASE WHEN s.active_branch_head_id = bh.id THEN 1 ELSE 0 END AS is_active
FROM branch_heads bh
JOIN sessions s ON s.id = bh.session_id
LEFT JOIN turns t ON t.id = bh.head_turn_id
"#;

impl StateStore {
    pub async fn initialize_main_branch(&self, session_id: i64) -> Result<BranchHeadRow> {
        let conn = self.connect().await?;
        let public_id = uuid::Uuid::now_v7().into_bytes().to_vec();

        let provenance = BranchProvenance::main();
        conn.execute(
            "INSERT INTO branch_heads (public_id, session_id, name, origin_kind) VALUES (?1, ?2, 'main', ?3)",
            turso::params![public_id, session_id, provenance.origin_kind],
        )
        .await
        .context("Failed to insert initial main branch head")?;

        let branch_id = conn.last_insert_rowid();
        conn.execute(
            "UPDATE sessions SET active_branch_head_id = ?1 WHERE id = ?2",
            turso::params![branch_id, session_id],
        )
        .await
        .context("Failed to activate main branch head")?;

        self.get_active_branch_head(session_id)
            .await?
            .ok_or_else(|| anyhow!("Initial main branch head was not readable after creation"))
    }

    pub async fn get_active_branch_head(&self, session_id: i64) -> Result<Option<BranchHeadRow>> {
        let conn = self.connect().await?;
        let sql = format!(
            "{BRANCH_HEAD_SELECT}
             WHERE s.id = ?1 AND bh.id = s.active_branch_head_id"
        );
        let mut rows = conn.query(&sql, [session_id]).await?;

        if let Some(row) = rows.next().await? {
            Ok(Some(branch_head_from_row(&row)?))
        } else {
            Ok(None)
        }
    }

    pub async fn get_branch_head(
        &self,
        session_id: i64,
        branch_id: i64,
    ) -> Result<Option<BranchHeadRow>> {
        let conn = self.connect().await?;
        let sql = format!("{BRANCH_HEAD_SELECT} WHERE bh.session_id = ?1 AND bh.id = ?2");
        let mut rows = conn
            .query(&sql, turso::params![session_id, branch_id])
            .await?;

        if let Some(row) = rows.next().await? {
            Ok(Some(branch_head_from_row(&row)?))
        } else {
            Ok(None)
        }
    }

    pub async fn get_branch_head_by_name(
        &self,
        session_id: i64,
        name: &str,
    ) -> Result<Option<BranchHeadRow>> {
        let conn = self.connect().await?;
        let sql = format!("{BRANCH_HEAD_SELECT} WHERE bh.session_id = ?1 AND bh.name = ?2");
        let mut rows = conn.query(&sql, turso::params![session_id, name]).await?;

        if let Some(row) = rows.next().await? {
            Ok(Some(branch_head_from_row(&row)?))
        } else {
            Ok(None)
        }
    }

    pub async fn get_branch_head_by_public_id(
        &self,
        session_id: i64,
        public_id: uuid::Uuid,
    ) -> Result<Option<BranchHeadRow>> {
        let conn = self.connect().await?;
        let sql = format!("{BRANCH_HEAD_SELECT} WHERE bh.session_id = ?1 AND bh.public_id = ?2");
        let mut rows = conn
            .query(
                &sql,
                turso::params![session_id, public_id.into_bytes().to_vec()],
            )
            .await?;
        if let Some(row) = rows.next().await? {
            Ok(Some(branch_head_from_row(&row)?))
        } else {
            Ok(None)
        }
    }

    pub async fn list_branch_heads(&self, session_id: i64) -> Result<Vec<BranchHeadRow>> {
        let conn = self.connect().await?;
        let sql =
            format!("{BRANCH_HEAD_SELECT} WHERE bh.session_id = ?1 ORDER BY bh.created_at, bh.id");
        let mut rows = conn.query(&sql, [session_id]).await?;

        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(branch_head_from_row(&row)?);
        }
        Ok(out)
    }

    pub async fn list_branch_heads_from_source_turn(
        &self,
        session_id: i64,
        source_turn_id: i64,
    ) -> Result<Vec<BranchHeadRow>> {
        let conn = self.connect().await?;
        let sql = format!(
            "{BRANCH_HEAD_SELECT}
             WHERE bh.session_id = ?1 AND bh.created_from_turn_id = ?2
             ORDER BY bh.created_at, bh.id"
        );
        let mut rows = conn
            .query(&sql, turso::params![session_id, source_turn_id])
            .await?;

        let mut out = Vec::new();
        while let Some(row) = rows.next().await? {
            out.push(branch_head_from_row(&row)?);
        }
        Ok(out)
    }

    pub async fn create_branch_head_from_turn_index(
        &self,
        session_id: i64,
        name: &str,
        from_turn_index: Option<u32>,
        activate: bool,
    ) -> Result<BranchHeadRow> {
        self.create_branch_head_from_turn_index_with_provenance(
            session_id,
            name,
            from_turn_index,
            activate,
            BranchProvenance::manual(),
        )
        .await
    }

    pub async fn create_branch_head_from_turn_index_with_provenance(
        &self,
        session_id: i64,
        name: &str,
        from_turn_index: Option<u32>,
        activate: bool,
        provenance: BranchProvenance,
    ) -> Result<BranchHeadRow> {
        let conn = self.connect().await?;
        let public_id = uuid::Uuid::now_v7().into_bytes().to_vec();
        let source_turn_id = self
            .resolve_branch_source_turn(session_id, from_turn_index)
            .await?;

        let BranchProvenance {
            origin_kind,
            origin_task_id,
            origin_execution_id,
            origin_metadata,
        } = provenance;
        conn.execute(
            r#"
            INSERT INTO branch_heads (
                public_id,
                session_id,
                name,
                head_turn_id,
                created_from_turn_id,
                origin_kind,
                origin_task_id,
                origin_execution_id,
                origin_metadata
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            "#,
            turso::params![
                public_id,
                session_id,
                name,
                source_turn_id,
                source_turn_id,
                origin_kind,
                origin_task_id,
                origin_execution_id,
                origin_metadata
            ],
        )
        .await
        .with_context(|| format!("Failed to create branch head '{}'", name))?;

        let branch_id = conn.last_insert_rowid();
        if activate {
            conn.execute(
                "UPDATE sessions SET active_branch_head_id = ?1 WHERE id = ?2",
                turso::params![branch_id, session_id],
            )
            .await
            .with_context(|| format!("Failed to activate branch head '{}'", name))?;
        }

        let heads = self.list_branch_heads(session_id).await?;
        heads
            .into_iter()
            .find(|head| head.id == branch_id)
            .ok_or_else(|| anyhow!("Created branch head '{}' was not readable", name))
    }

    pub async fn create_branch_head_from_turn_id(
        &self,
        session_id: i64,
        name: &str,
        source_turn_id: i64,
        activate: bool,
    ) -> Result<BranchHeadRow> {
        self.create_branch_head_from_turn_id_with_provenance(
            session_id,
            name,
            source_turn_id,
            activate,
            BranchProvenance::manual(),
        )
        .await
    }

    pub async fn create_branch_head_from_turn_id_with_provenance(
        &self,
        session_id: i64,
        name: &str,
        source_turn_id: i64,
        activate: bool,
        provenance: BranchProvenance,
    ) -> Result<BranchHeadRow> {
        let source_turn = self
            .get_turn_row(source_turn_id)
            .await?
            .ok_or_else(|| anyhow!("Source turn '{}' not found", source_turn_id))?;
        if source_turn.session_id != session_id {
            anyhow::bail!(
                "Source turn '{}' does not belong to session '{}'",
                source_turn_id,
                session_id
            );
        }

        let conn = self.connect().await?;
        let public_id = uuid::Uuid::now_v7().into_bytes().to_vec();

        let BranchProvenance {
            origin_kind,
            origin_task_id,
            origin_execution_id,
            origin_metadata,
        } = provenance;
        conn.execute(
            r#"
            INSERT INTO branch_heads (
                public_id,
                session_id,
                name,
                head_turn_id,
                created_from_turn_id,
                origin_kind,
                origin_task_id,
                origin_execution_id,
                origin_metadata
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)
            "#,
            turso::params![
                public_id,
                session_id,
                name,
                source_turn_id,
                source_turn_id,
                origin_kind,
                origin_task_id,
                origin_execution_id,
                origin_metadata
            ],
        )
        .await
        .with_context(|| format!("Failed to create branch head '{}' from turn", name))?;

        let branch_id = conn.last_insert_rowid();
        if activate {
            conn.execute(
                "UPDATE sessions SET active_branch_head_id = ?1 WHERE id = ?2",
                turso::params![branch_id, session_id],
            )
            .await
            .with_context(|| format!("Failed to activate branch head '{}'", name))?;
        }

        let heads = self.list_branch_heads(session_id).await?;
        heads
            .into_iter()
            .find(|head| head.id == branch_id)
            .ok_or_else(|| anyhow!("Created branch head '{}' was not readable", name))
    }

    pub async fn checkout_branch_head_by_name(
        &self,
        session_id: i64,
        name: &str,
    ) -> Result<Option<BranchHeadRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT id FROM branch_heads WHERE session_id = ?1 AND name = ?2",
                turso::params![session_id, name],
            )
            .await?;
        let Some(row) = rows.next().await? else {
            return Ok(None);
        };
        let branch_id = row.get::<i64>(0)?;
        conn.execute(
            "UPDATE sessions SET active_branch_head_id = ?1 WHERE id = ?2",
            turso::params![branch_id, session_id],
        )
        .await
        .with_context(|| format!("Failed to check out branch head '{}'", name))?;
        self.get_active_branch_head(session_id).await
    }

    pub async fn checkout_branch_head_by_public_id(
        &self,
        session_id: i64,
        public_id: uuid::Uuid,
    ) -> Result<Option<BranchHeadRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT id FROM branch_heads WHERE session_id = ?1 AND public_id = ?2",
                turso::params![session_id, public_id.into_bytes().to_vec()],
            )
            .await?;
        let Some(row) = rows.next().await? else {
            return Ok(None);
        };
        let branch_id = row.get::<i64>(0)?;
        conn.execute(
            "UPDATE sessions SET active_branch_head_id = ?1 WHERE id = ?2",
            turso::params![branch_id, session_id],
        )
        .await
        .context("Failed to check out branch head by id")?;
        self.get_active_branch_head(session_id).await
    }

    pub(super) async fn resolve_branch_head(
        &self,
        session_id: i64,
        branch_head_id: Option<i64>,
    ) -> Result<Option<BranchHeadRow>> {
        match branch_head_id {
            Some(branch_id) => self.get_branch_head(session_id, branch_id).await,
            None => self.get_active_branch_head(session_id).await,
        }
    }

    async fn resolve_branch_source_turn(
        &self,
        session_id: i64,
        from_turn_index: Option<u32>,
    ) -> Result<Option<i64>> {
        let conn = self.connect().await?;
        match from_turn_index {
            None => {
                let mut rows = conn
                    .query(
                        r#"
                        SELECT bh.head_turn_id
                        FROM sessions s
                        JOIN branch_heads bh ON bh.id = s.active_branch_head_id
                        WHERE s.id = ?1
                        "#,
                        [session_id],
                    )
                    .await?;
                Ok(rows
                    .next()
                    .await?
                    .map(|row| row.get::<Option<i64>>(0))
                    .transpose()?
                    .flatten())
            }
            Some(branch_depth) => Ok(self
                .active_branch_path_turns(session_id)
                .await?
                .into_iter()
                .find(|turn| turn.branch_depth == branch_depth)
                .map(|turn| turn.id)),
        }
    }
}

fn branch_head_from_row(row: &turso::Row) -> Result<BranchHeadRow> {
    Ok(BranchHeadRow {
        id: row.get::<i64>(0)?,
        public_id: row.get::<Vec<u8>>(1)?,
        session_id: row.get::<i64>(2)?,
        name: row.get::<String>(3)?,
        head_turn_id: row.get::<Option<i64>>(4)?,
        head_turn_depth: row.get::<Option<i64>>(5)?.map(|value| value as u32),
        created_from_turn_id: row.get::<Option<i64>>(6)?,
        origin_kind: row.get::<String>(7)?,
        origin_task_id: row.get::<Option<String>>(8)?,
        origin_execution_id: row.get::<Option<String>>(9)?,
        origin_metadata: row.get::<Option<String>>(10)?,
        created_at: row.get::<String>(11)?,
        is_active: row.get::<i64>(12)? != 0,
    })
}
