mod branch_heads;

use anyhow::{Context, Result, anyhow};

use super::{StateStore, TurnRow, TurnWriteError, TurnWriteTarget};

const TURN_SELECT: &str =
    "SELECT id, public_id, session_id, parent_turn_id, branch_depth, created_at FROM turns";

impl StateStore {
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
        if turn_ids.is_empty() {
            anyhow::bail!("Selected path must include at least one turn");
        }
        let mut turns = Vec::with_capacity(turn_ids.len());
        for turn_id in turn_ids {
            if turns.iter().any(|turn: &TurnRow| turn.id == *turn_id) {
                anyhow::bail!("Selected path contains duplicate turn {}", turn_id);
            }
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
        let sql = format!("{TURN_SELECT} WHERE id = ?1");
        let mut rows = conn.query(&sql, [turn_id]).await?;
        if let Some(row) = rows.next().await? {
            Ok(Some(turn_row_from_row(&row)?))
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
}

fn turn_row_from_row(row: &turso::Row) -> Result<TurnRow> {
    Ok(TurnRow {
        id: row.get::<i64>(0)?,
        public_id: row.get::<Vec<u8>>(1)?,
        session_id: row.get::<i64>(2)?,
        parent_turn_id: row.get::<Option<i64>>(3)?,
        branch_depth: row.get::<i64>(4)? as u32,
        created_at: row.get::<String>(5)?,
    })
}
