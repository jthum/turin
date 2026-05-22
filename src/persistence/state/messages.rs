use std::collections::HashMap;

use anyhow::{Context, Result};
use turso::Value as SqlValue;

use super::{MessageRow, SessionReadTarget, SessionRow, StateStore, TurnWriteTarget};
use crate::persistence::state::sessions::map_session_row;

const MESSAGE_TURN_QUERY_CHUNK: usize = 500;

impl StateStore {
    pub async fn list_session_rows(&self, limit: usize, offset: usize) -> Result<Vec<SessionRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                r#"
                SELECT s.id, s.public_id, s.agent_id, s.metadata, s.active_branch_head_id, s.created_at
                FROM sessions s
                LEFT JOIN events e ON e.session_id = s.id
                GROUP BY s.id
                ORDER BY COALESCE(MAX(e.id), s.id) DESC
                LIMIT ?1 OFFSET ?2
                "#,
                turso::params![limit as i64, offset as i64],
            )
            .await?;

        let mut sessions = Vec::new();
        while let Some(row) = rows.next().await? {
            sessions.push(map_session_row(&row)?);
        }
        Ok(sessions)
    }

    pub async fn insert_message(
        &self,
        session_id: i64,
        target: TurnWriteTarget,
        role: &str,
        content: &serde_json::Value,
        token_count: Option<u64>,
    ) -> Result<()> {
        let conn = self.connect().await?;
        let content_str = serde_json::to_string(content)?;
        let turn = self
            .resolve_turn_for_write_target(session_id, target)
            .await?
            .ok_or_else(|| {
                anyhow::anyhow!("No active branch head available for session {}", session_id)
            })?;
        conn.execute(
            "INSERT INTO messages (turn_id, role, content, token_count) VALUES (?1, ?2, ?3, ?4)",
            turso::params![turn.id, role, content_str, token_count.map(|t| t as i64),],
        )
        .await
        .with_context(|| format!("Failed to insert turn message for session: {}", session_id))?;
        Ok(())
    }

    pub async fn get_messages(
        &self,
        session_id: i64,
        target: &SessionReadTarget,
    ) -> Result<Vec<MessageRow>> {
        match target {
            SessionReadTarget::ActiveBranch => {
                self.messages_for_turns(
                    session_id,
                    &self.branch_path_turns(session_id, None).await?,
                )
                .await
            }
            SessionReadTarget::BranchHead(branch_head_id) => {
                self.messages_for_turns(
                    session_id,
                    &self
                        .branch_path_turns(session_id, Some(*branch_head_id))
                        .await?,
                )
                .await
            }
            SessionReadTarget::TurnId(turn_id) => {
                self.messages_for_turns(
                    session_id,
                    &self.turn_path_to_turn_id(session_id, *turn_id).await?,
                )
                .await
            }
            SessionReadTarget::SelectedPath(turn_ids) => {
                self.messages_for_turns(
                    session_id,
                    &self
                        .turn_rows_for_selected_path(session_id, turn_ids)
                        .await?,
                )
                .await
            }
        }
    }

    async fn messages_for_turns(
        &self,
        session_id: i64,
        turns: &[super::TurnRow],
    ) -> Result<Vec<MessageRow>> {
        if turns.is_empty() {
            return Ok(Vec::new());
        }

        let conn = self.connect().await?;
        let turn_order = turns
            .iter()
            .enumerate()
            .map(|(index, turn)| (turn.id, (index, turn.branch_depth)))
            .collect::<HashMap<_, _>>();
        let mut messages = Vec::new();
        for chunk in turns.chunks(MESSAGE_TURN_QUERY_CHUNK) {
            let placeholders = (1..=chunk.len())
                .map(|index| format!("?{index}"))
                .collect::<Vec<_>>()
                .join(", ");
            let sql = format!(
                "SELECT turn_id, id, role, content, token_count, created_at
                 FROM messages
                 WHERE turn_id IN ({placeholders})
                 ORDER BY turn_id, id"
            );
            let params = chunk
                .iter()
                .map(|turn| SqlValue::Integer(turn.id))
                .collect::<Vec<_>>();
            let mut stmt = conn.prepare(&sql).await?;
            let mut rows = stmt.query(params).await?;
            while let Some(row) = rows.next().await? {
                let turn_id = row.get::<i64>(0)?;
                let (turn_position, turn_index) = turn_order.get(&turn_id).ok_or_else(|| {
                    anyhow::anyhow!("Message references unexpected turn {}", turn_id)
                })?;
                let message_id = row.get::<i64>(1)?;
                messages.push((
                    *turn_position,
                    message_id,
                    MessageRow {
                        id: message_id,
                        session_id,
                        turn_index: *turn_index,
                        role: row.get::<String>(2)?,
                        content: row.get::<String>(3)?,
                        token_count: row.get::<Option<i64>>(4)?.map(|t| t as u64),
                        created_at: row.get::<String>(5)?,
                    },
                ));
            }
        }
        messages.sort_by_key(|(turn_position, message_id, _)| (*turn_position, *message_id));
        Ok(messages
            .into_iter()
            .map(|(_, _, message)| message)
            .collect())
    }
}
