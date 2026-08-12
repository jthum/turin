use std::collections::HashMap;

use anyhow::{Context, Result};
use turso::{Connection, Value as SqlValue};

use super::{MessageRow, SessionReadTarget, SessionRow, StateStore, TurnWriteTarget};
use crate::persistence::state::sessions::map_session_row;

const MESSAGE_TURN_QUERY_CHUNK: usize = 500;
const MESSAGE_WINDOW_TURN_QUERY_CHUNK: usize = 32;

type OrderedMessageRow = (usize, i64, MessageRow);

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

    pub async fn get_recent_messages(
        &self,
        session_id: i64,
        target: &SessionReadTarget,
        limit: usize,
    ) -> Result<(Vec<MessageRow>, usize)> {
        let turns = match target {
            SessionReadTarget::ActiveBranch => self.branch_path_turns(session_id, None).await?,
            SessionReadTarget::BranchHead(branch_head_id) => {
                self.branch_path_turns(session_id, Some(*branch_head_id))
                    .await?
            }
            SessionReadTarget::TurnId(turn_id) => {
                self.turn_path_to_turn_id(session_id, *turn_id).await?
            }
            SessionReadTarget::SelectedPath(turn_ids) => {
                self.turn_rows_for_selected_path(session_id, turn_ids)
                    .await?
            }
        };
        self.recent_messages_for_turns(session_id, &turns, limit.max(1))
            .await
    }

    pub async fn get_message_window(
        &self,
        session_id: i64,
        target: &SessionReadTarget,
        offset: usize,
        limit: usize,
    ) -> Result<(Vec<MessageRow>, usize, usize)> {
        let turns = match target {
            SessionReadTarget::ActiveBranch => self.branch_path_turns(session_id, None).await?,
            SessionReadTarget::BranchHead(branch_head_id) => {
                self.branch_path_turns(session_id, Some(*branch_head_id))
                    .await?
            }
            SessionReadTarget::TurnId(turn_id) => {
                self.turn_path_to_turn_id(session_id, *turn_id).await?
            }
            SessionReadTarget::SelectedPath(turn_ids) => {
                self.turn_rows_for_selected_path(session_id, turn_ids)
                    .await?
            }
        };
        self.messages_window_for_turns(session_id, &turns, offset, limit.max(1))
            .await
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
            messages.extend(query_message_chunk(&conn, session_id, chunk, &turn_order).await?);
        }
        Ok(order_messages(messages))
    }

    async fn recent_messages_for_turns(
        &self,
        session_id: i64,
        turns: &[super::TurnRow],
        limit: usize,
    ) -> Result<(Vec<MessageRow>, usize)> {
        if turns.is_empty() {
            return Ok((Vec::new(), 0));
        }

        let conn = self.connect().await?;
        let turn_order = turns
            .iter()
            .enumerate()
            .map(|(index, turn)| (turn.id, (index, turn.branch_depth)))
            .collect::<HashMap<_, _>>();
        let total = count_messages_for_turns(&conn, turns).await?;
        let mut messages = Vec::new();
        for chunk in turns.rchunks(MESSAGE_WINDOW_TURN_QUERY_CHUNK) {
            messages.extend(query_message_chunk(&conn, session_id, chunk, &turn_order).await?);
            if messages.len() >= limit {
                break;
            }
        }
        let mut messages = order_messages(messages);
        let mut retain_from = messages.len().saturating_sub(limit);
        while retain_from > 0
            && messages[retain_from - 1].turn_index == messages[retain_from].turn_index
        {
            retain_from -= 1;
        }
        messages.drain(0..retain_from);
        Ok((messages, total))
    }

    async fn messages_window_for_turns(
        &self,
        session_id: i64,
        turns: &[super::TurnRow],
        offset: usize,
        limit: usize,
    ) -> Result<(Vec<MessageRow>, usize, usize)> {
        if turns.is_empty() {
            return Ok((Vec::new(), 0, 0));
        }

        let conn = self.connect().await?;
        let counts = message_counts_for_turns(&conn, turns).await?;
        let total = counts.values().copied().sum::<usize>();
        let target_offset = offset.min(total);
        let mut preceding = 0usize;
        let mut selected_count = 0usize;
        let mut selected_start = None;
        let mut selected_end = 0usize;

        for (index, turn) in turns.iter().enumerate() {
            let count = counts.get(&turn.id).copied().unwrap_or(0);
            if selected_start.is_none() {
                if preceding.saturating_add(count) <= target_offset {
                    preceding = preceding.saturating_add(count);
                    continue;
                }
                selected_start = Some(index);
            }
            selected_count = selected_count.saturating_add(count);
            selected_end = index + 1;
            if selected_count >= limit {
                break;
            }
        }

        let Some(selected_start) = selected_start else {
            return Ok((Vec::new(), total, total));
        };
        let messages = self
            .messages_for_turns(session_id, &turns[selected_start..selected_end])
            .await?;
        Ok((messages, total, preceding))
    }
}

async fn count_messages_for_turns(conn: &Connection, turns: &[super::TurnRow]) -> Result<usize> {
    let mut total = 0usize;
    for chunk in turns.chunks(MESSAGE_TURN_QUERY_CHUNK) {
        let placeholders = query_placeholders(chunk.len());
        let sql = format!("SELECT COUNT(*) FROM messages WHERE turn_id IN ({placeholders})");
        let params = turn_params(chunk);
        let mut stmt = conn.prepare(&sql).await?;
        let mut rows = stmt.query(params).await?;
        if let Some(row) = rows.next().await? {
            total = total.saturating_add(row.get::<i64>(0)? as usize);
        }
    }
    Ok(total)
}

async fn message_counts_for_turns(
    conn: &Connection,
    turns: &[super::TurnRow],
) -> Result<HashMap<i64, usize>> {
    let mut counts = HashMap::new();
    for chunk in turns.chunks(MESSAGE_TURN_QUERY_CHUNK) {
        let placeholders = query_placeholders(chunk.len());
        let sql = format!(
            "SELECT turn_id, COUNT(*) FROM messages WHERE turn_id IN ({placeholders}) GROUP BY turn_id"
        );
        let mut stmt = conn.prepare(&sql).await?;
        let mut rows = stmt.query(turn_params(chunk)).await?;
        while let Some(row) = rows.next().await? {
            counts.insert(row.get::<i64>(0)?, row.get::<i64>(1)? as usize);
        }
    }
    Ok(counts)
}

async fn query_message_chunk(
    conn: &Connection,
    session_id: i64,
    turns: &[super::TurnRow],
    turn_order: &HashMap<i64, (usize, u32)>,
) -> Result<Vec<OrderedMessageRow>> {
    let placeholders = query_placeholders(turns.len());
    let sql = format!(
        "SELECT turn_id, id, role, content, token_count, created_at
         FROM messages
         WHERE turn_id IN ({placeholders})
         ORDER BY turn_id, id"
    );
    let mut stmt = conn.prepare(&sql).await?;
    let mut rows = stmt.query(turn_params(turns)).await?;
    let mut messages = Vec::new();
    while let Some(row) = rows.next().await? {
        let turn_id = row.get::<i64>(0)?;
        let (turn_position, turn_index) = turn_order
            .get(&turn_id)
            .ok_or_else(|| anyhow::anyhow!("Message references unexpected turn {}", turn_id))?;
        let message_id = row.get::<i64>(1)?;
        messages.push((
            *turn_position,
            message_id,
            MessageRow {
                id: message_id,
                session_id,
                turn_id,
                turn_index: *turn_index,
                role: row.get::<String>(2)?,
                content: row.get::<String>(3)?,
                token_count: row.get::<Option<i64>>(4)?.map(|t| t as u64),
                created_at: row.get::<String>(5)?,
            },
        ));
    }
    Ok(messages)
}

fn query_placeholders(count: usize) -> String {
    (1..=count)
        .map(|index| format!("?{index}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn turn_params(turns: &[super::TurnRow]) -> Vec<SqlValue> {
    turns
        .iter()
        .map(|turn| SqlValue::Integer(turn.id))
        .collect()
}

fn order_messages(mut messages: Vec<OrderedMessageRow>) -> Vec<MessageRow> {
    messages.sort_by_key(|(turn_position, message_id, _)| (*turn_position, *message_id));
    messages
        .into_iter()
        .map(|(_, _, message)| message)
        .collect()
}
