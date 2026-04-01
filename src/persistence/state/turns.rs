use anyhow::{Context, Result, anyhow};

use super::{BranchHeadRow, StateStore, TurnRow};

impl StateStore {
    pub async fn initialize_main_branch(&self, session_id: i64) -> Result<BranchHeadRow> {
        let conn = self.connect().await?;
        let public_id = uuid::Uuid::now_v7().into_bytes().to_vec();

        conn.execute(
            "INSERT INTO branch_heads (public_id, session_id, name) VALUES (?1, ?2, 'main')",
            turso::params![public_id, session_id],
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
        let mut rows = conn
            .query(
                r#"
                SELECT bh.id,
                       bh.public_id,
                       bh.session_id,
                       bh.name,
                       bh.head_turn_id,
                       t.branch_depth,
                       bh.created_from_turn_id,
                       bh.created_at,
                       1 AS is_active
                FROM sessions s
                JOIN branch_heads bh ON bh.id = s.active_branch_head_id
                LEFT JOIN turns t ON t.id = bh.head_turn_id
                WHERE s.id = ?1
                "#,
                [session_id],
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
        let mut rows = conn
            .query(
                r#"
                SELECT bh.id,
                       bh.public_id,
                       bh.session_id,
                       bh.name,
                       bh.head_turn_id,
                       t.branch_depth,
                       bh.created_from_turn_id,
                       bh.created_at,
                       CASE WHEN s.active_branch_head_id = bh.id THEN 1 ELSE 0 END AS is_active
                FROM branch_heads bh
                JOIN sessions s ON s.id = bh.session_id
                LEFT JOIN turns t ON t.id = bh.head_turn_id
                WHERE bh.session_id = ?1
                ORDER BY bh.created_at, bh.id
                "#,
                [session_id],
            )
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
        let conn = self.connect().await?;
        let public_id = uuid::Uuid::now_v7().into_bytes().to_vec();
        let source_turn_id = self
            .resolve_branch_source_turn(session_id, from_turn_index)
            .await?;

        conn.execute(
            r#"
            INSERT INTO branch_heads (
                public_id,
                session_id,
                name,
                head_turn_id,
                created_from_turn_id
            ) VALUES (?1, ?2, ?3, ?4, ?5)
            "#,
            turso::params![public_id, session_id, name, source_turn_id, source_turn_id],
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

    pub async fn active_branch_turn_count(&self, session_id: i64) -> Result<u32> {
        Ok(self.active_branch_path_turns(session_id).await?.len() as u32)
    }

    pub(crate) async fn ensure_turn_for_active_branch(
        &self,
        session_id: i64,
        turn_index: u32,
    ) -> Result<Option<TurnRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                r#"
                SELECT bh.id,
                       bh.head_turn_id,
                       t.public_id,
                       t.parent_turn_id,
                       t.branch_depth,
                       t.created_at
                FROM sessions s
                JOIN branch_heads bh ON bh.id = s.active_branch_head_id
                LEFT JOIN turns t ON t.id = bh.head_turn_id
                WHERE s.id = ?1
                "#,
                [session_id],
            )
            .await?;

        let Some(row) = rows.next().await? else {
            return Ok(None);
        };

        let branch_id = row.get::<i64>(0)?;
        let head_turn_id = row.get::<Option<i64>>(1)?;
        let head_turn = head_turn_id.map(|id| TurnRow {
            id,
            public_id: row.get::<Vec<u8>>(2).unwrap_or_default(),
            session_id,
            parent_turn_id: row.get::<Option<i64>>(3).ok().flatten(),
            branch_depth: row
                .get::<Option<i64>>(4)
                .ok()
                .flatten()
                .map(|value| value as u32)
                .unwrap_or(0),
            created_at: row.get::<String>(5).unwrap_or_default(),
        });

        if let Some(existing) = head_turn {
            if existing.branch_depth == turn_index {
                return Ok(Some(existing));
            }
            if existing.branch_depth + 1 != turn_index {
                anyhow::bail!(
                    "Invalid active branch turn progression: active depth {}, requested {}",
                    existing.branch_depth,
                    turn_index
                );
            }
            return self
                .create_turn_for_branch(&conn, session_id, branch_id, Some(existing.id), turn_index)
                .await
                .map(Some);
        }

        if turn_index != 0 {
            anyhow::bail!(
                "Invalid active branch turn progression: empty branch requested turn {}",
                turn_index
            );
        }

        self.create_turn_for_branch(&conn, session_id, branch_id, None, 0)
            .await
            .map(Some)
    }

    pub(crate) async fn active_branch_path_turns(&self, session_id: i64) -> Result<Vec<TurnRow>> {
        let conn = self.connect().await?;
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
        let Some(row) = rows.next().await? else {
            return Ok(Vec::new());
        };
        let mut current_turn_id = row.get::<Option<i64>>(0)?;
        let mut turns = Vec::new();
        while let Some(turn_id) = current_turn_id {
            let Some(turn) = self.get_turn_row(turn_id).await? else {
                anyhow::bail!("Turn {} on active branch path could not be loaded", turn_id);
            };
            current_turn_id = turn.parent_turn_id;
            turns.push(turn);
        }
        turns.reverse();
        Ok(turns)
    }

    pub(crate) async fn get_turn_row(&self, turn_id: i64) -> Result<Option<TurnRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT id, public_id, session_id, parent_turn_id, branch_depth, created_at FROM turns WHERE id = ?1",
                [turn_id],
            )
            .await?;
        if let Some(row) = rows.next().await? {
            Ok(Some(TurnRow {
                id: row.get::<i64>(0)?,
                public_id: row.get::<Vec<u8>>(1)?,
                session_id: row.get::<i64>(2)?,
                parent_turn_id: row.get::<Option<i64>>(3)?,
                branch_depth: row.get::<i64>(4)? as u32,
                created_at: row.get::<String>(5)?,
            }))
        } else {
            Ok(None)
        }
    }

    async fn create_turn_for_branch(
        &self,
        conn: &turso::Connection,
        session_id: i64,
        branch_id: i64,
        parent_turn_id: Option<i64>,
        branch_depth: u32,
    ) -> Result<TurnRow> {
        let public_id = uuid::Uuid::now_v7().into_bytes().to_vec();
        conn.execute(
            r#"
            INSERT INTO turns (public_id, session_id, parent_turn_id, branch_depth)
            VALUES (?1, ?2, ?3, ?4)
            "#,
            turso::params![public_id, session_id, parent_turn_id, branch_depth as i64],
        )
        .await
        .context("Failed to create turn row")?;
        let turn_id = conn.last_insert_rowid();
        conn.execute(
            "UPDATE branch_heads SET head_turn_id = ?1 WHERE id = ?2",
            turso::params![turn_id, branch_id],
        )
        .await
        .context("Failed to advance active branch head")?;

        let mut rows = conn
            .query(
                "SELECT public_id, created_at FROM turns WHERE id = ?1",
                [turn_id],
            )
            .await?;
        let row = rows
            .next()
            .await?
            .ok_or_else(|| anyhow!("Turn row missing immediately after insert"))?;
        Ok(TurnRow {
            id: turn_id,
            public_id: row.get::<Vec<u8>>(0)?,
            session_id,
            parent_turn_id,
            branch_depth,
            created_at: row.get::<String>(1)?,
        })
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
        created_at: row.get::<String>(7)?,
        is_active: row.get::<i64>(8)? != 0,
    })
}
