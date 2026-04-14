use anyhow::{Context, Result};

use super::{MessageRow, SessionReadTarget, SessionRow, StateStore, TurnWriteTarget};

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
            sessions.push(SessionRow {
                id: row.get::<i64>(0)?,
                public_id: row.get::<Vec<u8>>(1)?,
                agent_id: row.get::<String>(2)?,
                metadata: row.get::<Option<String>>(3)?,
                active_branch_head_id: row.get::<Option<i64>>(4)?,
                created_at: row.get::<String>(5)?,
            });
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
            "INSERT INTO messages (session_id, turn_index, role, content, token_count) VALUES (?1, ?2, ?3, ?4, ?5)",
            turso::params![
                session_id,
                target.turn_index() as i64,
                role,
                content_str.clone(),
                token_count.map(|t| t as i64),
            ],
        )
        .await
        .with_context(|| format!("Failed to insert message for session: {}", session_id))?;
        conn.execute(
            "INSERT INTO turn_messages (turn_id, role, content, token_count) VALUES (?1, ?2, ?3, ?4)",
            turso::params![
                turn.id,
                role,
                content_str,
                token_count.map(|t| t as i64),
            ],
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
        let conn = self.connect().await?;
        let mut messages = Vec::new();
        for turn in turns {
            let mut rows = conn
                .query(
                    "SELECT id, role, content, token_count, created_at FROM turn_messages WHERE turn_id = ?1 ORDER BY id",
                    [turn.id],
                )
                .await?;
            while let Some(row) = rows.next().await? {
                messages.push(MessageRow {
                    id: row.get::<i64>(0)?,
                    session_id,
                    turn_index: turn.branch_depth,
                    role: row.get::<String>(1)?,
                    content: row.get::<String>(2)?,
                    token_count: row.get::<Option<i64>>(3)?.map(|t| t as u64),
                    created_at: row.get::<String>(4)?,
                });
            }
        }
        Ok(messages)
    }
}
