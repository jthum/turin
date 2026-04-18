use anyhow::{Context, Result, anyhow};

use super::{BranchHeadRow, StateStore, TurnRow, TurnWriteError, TurnWriteTarget};

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

    pub async fn get_branch_head(
        &self,
        session_id: i64,
        branch_id: i64,
    ) -> Result<Option<BranchHeadRow>> {
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
                WHERE bh.session_id = ?1 AND bh.id = ?2
                "#,
                turso::params![session_id, branch_id],
            )
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
                WHERE bh.session_id = ?1 AND bh.name = ?2
                "#,
                turso::params![session_id, name],
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

    pub async fn list_branch_heads_from_source_turn(
        &self,
        session_id: i64,
        source_turn_id: i64,
    ) -> Result<Vec<BranchHeadRow>> {
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
                WHERE bh.session_id = ?1 AND bh.created_from_turn_id = ?2
                ORDER BY bh.created_at, bh.id
                "#,
                turso::params![session_id, source_turn_id],
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
        Ok(self.branch_path_turns(session_id, None).await?.len() as u32)
    }

    pub async fn prepare_turn_write_target(
        &self,
        session_id: i64,
        target: TurnWriteTarget,
    ) -> Result<Option<TurnWriteTarget>> {
        let turn = self
            .resolve_turn_for_write_target(session_id, target)
            .await?;
        Ok(turn.map(|turn| TurnWriteTarget::existing_turn(turn.id, turn.branch_depth)))
    }

    pub(crate) async fn ensure_turn_for_branch_head(
        &self,
        session_id: i64,
        branch_head_id: Option<i64>,
        turn_index: u32,
    ) -> Result<Option<TurnRow>> {
        let conn = self.connect().await?;
        let Some(branch) = self.resolve_branch_head(session_id, branch_head_id).await? else {
            return Ok(None);
        };

        let branch_id = branch.id;
        let head_turn = match branch.head_turn_id {
            Some(turn_id) => Some(self.get_turn_row(turn_id).await?.ok_or_else(|| {
                anyhow!(
                    "Turn {} on branch {} could not be loaded",
                    turn_id,
                    branch_id
                )
            })?),
            None => None,
        };

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

    pub(crate) async fn resolve_turn_for_write_target(
        &self,
        session_id: i64,
        target: TurnWriteTarget,
    ) -> Result<Option<TurnRow>> {
        match target {
            TurnWriteTarget::ExistingTurn {
                turn_id,
                turn_index,
            } => {
                let Some(turn) = self.get_turn_row(turn_id).await? else {
                    anyhow::bail!("Turn {} could not be loaded", turn_id);
                };
                if turn.session_id != session_id {
                    anyhow::bail!("Turn {} does not belong to session {}", turn_id, session_id);
                }
                if turn.branch_depth != turn_index {
                    anyhow::bail!(
                        "Turn {} depth mismatch: expected {}, found {}",
                        turn_id,
                        turn_index,
                        turn.branch_depth
                    );
                }
                Ok(Some(turn))
            }
            TurnWriteTarget::BranchAdvance {
                branch_head_id,
                expected_head_turn_id: Some(expected_head_turn_id),
                turn_index,
            } => {
                let conn = self.connect().await?;
                let Some(branch) = self.resolve_branch_head(session_id, branch_head_id).await?
                else {
                    return Ok(None);
                };
                if branch.head_turn_id != Some(expected_head_turn_id) {
                    return Err(TurnWriteError::BranchHeadChanged {
                        expected_head_turn_id,
                        found_head_turn_id: branch.head_turn_id,
                    }
                    .into());
                }
                let expected_parent =
                    self.get_turn_row(expected_head_turn_id)
                        .await?
                        .ok_or_else(|| {
                            anyhow!(
                                "Turn {} on branch {} could not be loaded",
                                expected_head_turn_id,
                                branch.id
                            )
                        })?;
                if expected_parent.session_id != session_id {
                    anyhow::bail!(
                        "Turn {} does not belong to session {}",
                        expected_head_turn_id,
                        session_id
                    );
                }
                if expected_parent.branch_depth + 1 != turn_index {
                    anyhow::bail!(
                        "Invalid branch advance progression: parent depth {}, requested {}",
                        expected_parent.branch_depth,
                        turn_index
                    );
                }
                self.create_turn_for_branch(
                    &conn,
                    session_id,
                    branch.id,
                    Some(expected_parent.id),
                    turn_index,
                )
                .await
                .map(Some)
            }
            TurnWriteTarget::BranchAdvance {
                branch_head_id,
                expected_head_turn_id: None,
                turn_index,
            } => {
                let conn = self.connect().await?;
                let Some(branch) = self.resolve_branch_head(session_id, branch_head_id).await?
                else {
                    return Ok(None);
                };
                if branch.head_turn_id.is_some() {
                    return self
                        .ensure_turn_for_branch_head(session_id, branch_head_id, turn_index)
                        .await;
                }
                if turn_index != 0 {
                    anyhow::bail!(
                        "Invalid branch advance progression: empty branch requested turn {}",
                        turn_index
                    );
                }
                self.create_turn_for_branch(&conn, session_id, branch.id, None, 0)
                    .await
                    .map(Some)
            }
        }
    }

    pub(crate) async fn active_branch_path_turns(&self, session_id: i64) -> Result<Vec<TurnRow>> {
        self.branch_path_turns(session_id, None).await
    }

    pub(crate) async fn branch_path_turns(
        &self,
        session_id: i64,
        branch_head_id: Option<i64>,
    ) -> Result<Vec<TurnRow>> {
        let Some(branch) = self.resolve_branch_head(session_id, branch_head_id).await? else {
            return Ok(Vec::new());
        };
        let mut current_turn_id = branch.head_turn_id;
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

    pub(crate) async fn turn_path_to_turn_id(
        &self,
        session_id: i64,
        turn_id: i64,
    ) -> Result<Vec<TurnRow>> {
        let mut current_turn_id = Some(turn_id);
        let mut turns = Vec::new();
        while let Some(turn_id) = current_turn_id {
            let Some(turn) = self.get_turn_row(turn_id).await? else {
                anyhow::bail!("Turn {} could not be loaded", turn_id);
            };
            if turn.session_id != session_id {
                anyhow::bail!("Turn {} does not belong to session {}", turn_id, session_id);
            }
            current_turn_id = turn.parent_turn_id;
            turns.push(turn);
        }
        turns.reverse();
        Ok(turns)
    }

    pub(crate) async fn turn_rows_for_selected_path(
        &self,
        session_id: i64,
        turn_ids: &[i64],
    ) -> Result<Vec<TurnRow>> {
        let mut turns = Vec::with_capacity(turn_ids.len());
        for turn_id in turn_ids {
            let Some(turn) = self.get_turn_row(*turn_id).await? else {
                anyhow::bail!("Turn {} on selected path could not be loaded", turn_id);
            };
            if turn.session_id != session_id {
                anyhow::bail!(
                    "Turn {} on selected path does not belong to session {}",
                    turn_id,
                    session_id
                );
            }
            turns.push(turn);
        }
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

    async fn resolve_branch_head(
        &self,
        session_id: i64,
        branch_head_id: Option<i64>,
    ) -> Result<Option<BranchHeadRow>> {
        match branch_head_id {
            Some(branch_id) => self.get_branch_head(session_id, branch_id).await,
            None => self.get_active_branch_head(session_id).await,
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
