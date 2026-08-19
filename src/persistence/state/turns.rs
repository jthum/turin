mod branch_heads;

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};

use super::{SessionGraphTurnRow, StateStore, TurnRow, TurnWriteError, TurnWriteTarget};
use crate::perf_diagnostics::{perf_stage, perf_stage_finish};

const TURN_SELECT_BY_ID: &str = concat!(
    "SELECT id, public_id, session_id, parent_turn_id, branch_depth, created_at ",
    "FROM turns WHERE id = ?1"
);
const ANCESTRY_DEPTH_CHUNK: u32 = 256;

impl StateStore {
    pub async fn list_session_graph_turns(
        &self,
        session_id: i64,
    ) -> Result<Vec<SessionGraphTurnRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                r#"
                SELECT t.id,
                       t.public_id,
                       t.session_id,
                       t.parent_turn_id,
                       t.branch_depth,
                       t.created_at,
                       (SELECT COUNT(*) FROM messages m WHERE m.turn_id = t.id),
                       (SELECT COUNT(*) FROM tool_executions te WHERE te.turn_id = t.id),
                       (
                           SELECT substr(m.content, 1, 320)
                           FROM messages m
                           WHERE m.turn_id = t.id AND m.role = 'user'
                           ORDER BY m.id
                           LIMIT 1
                       )
                FROM turns t
                WHERE t.session_id = ?1
                ORDER BY t.branch_depth, t.created_at, t.id
                "#,
                [session_id],
            )
            .await?;
        let mut turns = Vec::new();
        while let Some(row) = rows.next().await? {
            turns.push(SessionGraphTurnRow {
                turn: turn_row_from_row(&row)?,
                message_count: row.get::<i64>(6)? as usize,
                tool_execution_count: row.get::<i64>(7)? as usize,
                preview: row.get::<Option<String>>(8)?,
            });
        }
        Ok(turns)
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
        perf_stage!(
            path_stage,
            "persistence.branch_path",
            None,
            serde_json::json!({
                "internal_session_id": session_id,
                "branch_head_id": branch_head_id,
            })
        );
        let Some(branch) = self.resolve_branch_head(session_id, branch_head_id).await? else {
            perf_stage_finish!(
                path_stage,
                "empty",
                serde_json::json!({
                    "turns_visited": 0,
                    "turn_row_queries": 0,
                    "connection_setups": 1,
                })
            );
            return Ok(Vec::new());
        };
        let Some(head_turn_id) = branch.head_turn_id else {
            return Ok(Vec::new());
        };
        let (turns, _has_prior_history, _ancestry_queries) = self
            .ancestry_path_turns(session_id, head_turn_id, None)
            .await?;
        perf_stage_finish!(
            path_stage,
            "ok",
            serde_json::json!({
                "turns_visited": turns.len(),
                "turn_row_queries": _ancestry_queries,
                "connection_setups": 2,
            })
        );
        Ok(turns)
    }

    pub(crate) async fn recent_branch_path_turns(
        &self,
        session_id: i64,
        branch_head_id: Option<i64>,
        max_turns: usize,
    ) -> Result<(Vec<TurnRow>, bool)> {
        let Some(branch) = self.resolve_branch_head(session_id, branch_head_id).await? else {
            return Ok((Vec::new(), false));
        };
        let Some(head_turn_id) = branch.head_turn_id else {
            return Ok((Vec::new(), false));
        };
        self.ancestry_path_turns(session_id, head_turn_id, Some(max_turns.max(1)))
            .await
            .map(|(turns, has_prior_history, _)| (turns, has_prior_history))
    }

    pub(crate) async fn turn_path_to_turn_id(
        &self,
        session_id: i64,
        turn_id: i64,
    ) -> Result<Vec<TurnRow>> {
        self.ancestry_path_turns(session_id, turn_id, None)
            .await
            .map(|(turns, _, _)| turns)
    }

    pub(crate) async fn recent_turn_path_to_turn_id(
        &self,
        session_id: i64,
        turn_id: i64,
        max_turns: usize,
    ) -> Result<(Vec<TurnRow>, bool)> {
        self.ancestry_path_turns(session_id, turn_id, Some(max_turns.max(1)))
            .await
            .map(|(turns, has_prior_history, _)| (turns, has_prior_history))
    }

    async fn ancestry_path_turns(
        &self,
        session_id: i64,
        head_turn_id: i64,
        max_turns: Option<usize>,
    ) -> Result<(Vec<TurnRow>, bool, usize)> {
        let conn = self.connect().await?;
        let Some(head) = self.get_turn_row_with_conn(&conn, head_turn_id).await? else {
            anyhow::bail!("Turn {} could not be loaded", head_turn_id);
        };
        if head.session_id != session_id {
            anyhow::bail!(
                "Turn {} does not belong to session {}",
                head_turn_id,
                session_id
            );
        }

        let limit = max_turns.unwrap_or(usize::MAX);
        let mut expected_turn_id = Some(head_turn_id);
        let mut upper_depth = head.branch_depth;
        let mut reverse_path = Vec::with_capacity(limit.min(ANCESTRY_DEPTH_CHUNK as usize));
        let mut ancestry_queries = 0usize;

        while let Some(expected_id) = expected_turn_id {
            let lower_depth = upper_depth.saturating_sub(ANCESTRY_DEPTH_CHUNK - 1);
            let mut stmt = conn
                .prepare_cached(
                    "SELECT id, public_id, session_id, parent_turn_id, branch_depth, created_at
                     FROM turns
                     WHERE session_id = ?1 AND branch_depth BETWEEN ?2 AND ?3",
                )
                .await?;
            let mut rows = stmt
                .query(turso::params![
                    session_id,
                    lower_depth as i64,
                    upper_depth as i64
                ])
                .await?;
            ancestry_queries = ancestry_queries.saturating_add(1);
            let mut chunk = HashMap::new();
            while let Some(row) = rows.next().await? {
                let turn = turn_row_from_row(&row)?;
                chunk.insert(turn.id, turn);
            }

            let mut current_id = expected_id;
            loop {
                let Some(turn) = chunk.remove(&current_id) else {
                    anyhow::bail!("Turn {} on ancestry path could not be loaded", current_id);
                };
                let parent_turn_id = turn.parent_turn_id;
                reverse_path.push(turn);
                if reverse_path.len() >= limit || parent_turn_id.is_none() {
                    expected_turn_id = parent_turn_id;
                    break;
                }
                let parent_id = parent_turn_id.expect("parent checked above");
                if chunk.contains_key(&parent_id) {
                    current_id = parent_id;
                    continue;
                }
                expected_turn_id = Some(parent_id);
                break;
            }

            if reverse_path.len() >= limit || expected_turn_id.is_none() {
                break;
            }
            if lower_depth == 0 {
                anyhow::bail!(
                    "Turn {} on ancestry path could not be loaded",
                    expected_turn_id.expect("missing ancestry turn")
                );
            }
            upper_depth = lower_depth - 1;
        }

        let has_prior_history = reverse_path
            .last()
            .and_then(|oldest_loaded| oldest_loaded.parent_turn_id)
            .is_some();
        reverse_path.reverse();
        Ok((reverse_path, has_prior_history, ancestry_queries))
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
        self.get_turn_row_with_conn(&conn, turn_id).await
    }

    pub(super) async fn get_turn_row_with_conn(
        &self,
        conn: &turso::Connection,
        turn_id: i64,
    ) -> Result<Option<TurnRow>> {
        let mut stmt = conn.prepare_cached(TURN_SELECT_BY_ID).await?;
        let mut rows = stmt.query([turn_id]).await?;
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
