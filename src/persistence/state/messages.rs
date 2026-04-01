use anyhow::{Context, Result};

use super::{MessageRow, SessionRow, StateStore};

impl StateStore {
    pub async fn list_session_rows(&self, limit: usize, offset: usize) -> Result<Vec<SessionRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                r#"
                SELECT s.id, s.public_id, s.agent_id, s.metadata, s.created_at
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
                created_at: row.get::<String>(4)?,
            });
        }
        Ok(sessions)
    }

    pub async fn insert_message(
        &self,
        session_id: i64,
        turn_index: u32,
        role: &str,
        content: &serde_json::Value,
        token_count: Option<u64>,
    ) -> Result<()> {
        let conn = self.connect().await?;
        let content_str = serde_json::to_string(content)?;
        conn.execute(
            "INSERT INTO messages (session_id, turn_index, role, content, token_count) VALUES (?1, ?2, ?3, ?4, ?5)",
            turso::params![
                session_id,
                turn_index as i64,
                role,
                content_str,
                token_count.map(|t| t as i64),
            ],
        )
        .await
        .with_context(|| format!("Failed to insert message for session: {}", session_id))?;
        Ok(())
    }

    pub async fn get_messages(&self, session_id: i64) -> Result<Vec<MessageRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT id, session_id, turn_index, role, content, token_count, created_at FROM messages WHERE session_id = ?1 ORDER BY id",
                [session_id],
            )
            .await?;

        let mut messages = Vec::new();
        while let Some(row) = rows.next().await? {
            messages.push(MessageRow {
                id: row.get::<i64>(0)?,
                session_id: row.get::<i64>(1)?,
                turn_index: row.get::<i64>(2)? as u32,
                role: row.get::<String>(3)?,
                content: row.get::<String>(4)?,
                token_count: row.get::<Option<i64>>(5)?.map(|t| t as u64),
                created_at: row.get::<String>(6)?,
            });
        }
        Ok(messages)
    }
}
