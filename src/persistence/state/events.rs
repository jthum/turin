use anyhow::{Context, Result};

use super::{EventRow, StateStore};

impl StateStore {
    pub async fn insert_event(
        &self,
        session_id: i64,
        event_type: &str,
        payload: &serde_json::Value,
    ) -> Result<()> {
        let conn = self.connect().await?;
        let payload_str = serde_json::to_string(payload)?;
        conn.execute(
            "INSERT INTO events (session_id, event_type, payload) VALUES (?1, ?2, ?3)",
            turso::params![session_id, event_type, payload_str],
        )
        .await
        .with_context(|| format!("Failed to insert event for session: {}", session_id))?;
        Ok(())
    }

    pub async fn get_events(&self, session_id: i64) -> Result<Vec<EventRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT id, session_id, event_type, payload, created_at FROM events WHERE session_id = ?1 ORDER BY id",
                [session_id],
            )
            .await?;

        let mut events = Vec::new();
        while let Some(row) = rows.next().await? {
            events.push(EventRow {
                id: row.get::<i64>(0)?,
                session_id: row.get::<i64>(1)?,
                event_type: row.get::<String>(2)?,
                payload: row.get::<String>(3)?,
                created_at: row.get::<String>(4)?,
            });
        }
        Ok(events)
    }

    pub async fn list_sessions(&self, limit: usize, offset: usize) -> Result<Vec<i64>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT session_id FROM events GROUP BY session_id ORDER BY MAX(id) DESC LIMIT ?1 OFFSET ?2",
                turso::params![limit as i64, offset as i64],
            )
            .await?;

        let mut sessions = Vec::new();
        while let Some(row) = rows.next().await? {
            sessions.push(row.get(0)?);
        }
        Ok(sessions)
    }
}
