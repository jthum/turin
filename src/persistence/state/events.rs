use std::collections::HashSet;

use anyhow::{Context, Result};

use super::{EventRow, StateStore};

impl StateStore {
    pub async fn insert_event(
        &self,
        session_id: i64,
        event_type: &str,
        payload: &serde_json::Value,
    ) -> Result<()> {
        self.insert_event_with_turn_index_for_branch_head(
            session_id, None, None, event_type, payload,
        )
        .await
    }

    pub async fn insert_event_with_turn_index(
        &self,
        session_id: i64,
        turn_index: Option<u32>,
        event_type: &str,
        payload: &serde_json::Value,
    ) -> Result<()> {
        self.insert_event_with_turn_index_for_branch_head(
            session_id, None, turn_index, event_type, payload,
        )
        .await
    }

    pub async fn insert_event_with_turn_index_for_branch_head(
        &self,
        session_id: i64,
        branch_head_id: Option<i64>,
        turn_index: Option<u32>,
        event_type: &str,
        payload: &serde_json::Value,
    ) -> Result<()> {
        let conn = self.connect().await?;
        let payload_str = serde_json::to_string(payload)?;
        let turn_id = match turn_index {
            Some(turn_index) => self
                .ensure_turn_for_branch_head(session_id, branch_head_id, turn_index)
                .await?
                .map(|turn| turn.id),
            None => None,
        };
        conn.execute(
            "INSERT INTO events (session_id, turn_id, event_type, payload) VALUES (?1, ?2, ?3, ?4)",
            turso::params![session_id, turn_id, event_type, payload_str],
        )
        .await
        .with_context(|| format!("Failed to insert event for session: {}", session_id))?;
        Ok(())
    }

    pub async fn get_all_events(&self, session_id: i64) -> Result<Vec<EventRow>> {
        self.query_events(session_id).await
    }

    pub async fn get_events(&self, session_id: i64) -> Result<Vec<EventRow>> {
        let events = self.query_events(session_id).await?;
        self.filter_events_for_branch_head(session_id, None, events)
            .await
    }

    pub async fn get_events_for_branch_head(
        &self,
        session_id: i64,
        branch_head_id: Option<i64>,
    ) -> Result<Vec<EventRow>> {
        let events = self.query_events(session_id).await?;
        self.filter_events_for_branch_head(session_id, branch_head_id, events)
            .await
    }

    pub async fn get_events_for_turn_id(
        &self,
        session_id: i64,
        turn_id: i64,
    ) -> Result<Vec<EventRow>> {
        let events = self.query_events(session_id).await?;
        let turn_ids = self
            .turn_path_to_turn_id(session_id, turn_id)
            .await?
            .into_iter()
            .map(|turn| turn.id)
            .collect::<HashSet<_>>();
        Ok(self.filter_events_for_turn_ids(events, &turn_ids))
    }

    pub async fn get_events_for_selected_path(
        &self,
        session_id: i64,
        turn_ids: &[i64],
    ) -> Result<Vec<EventRow>> {
        let events = self.query_events(session_id).await?;
        let turn_ids = self
            .turn_rows_for_selected_path(session_id, turn_ids)
            .await?
            .into_iter()
            .map(|turn| turn.id)
            .collect::<HashSet<_>>();
        Ok(self.filter_events_for_turn_ids(events, &turn_ids))
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

    async fn query_events(&self, session_id: i64) -> Result<Vec<EventRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                r#"
                SELECT e.id,
                       e.session_id,
                       e.turn_id,
                       e.event_type,
                       e.payload,
                       t.branch_depth,
                       e.created_at
                FROM events e
                LEFT JOIN turns t ON t.id = e.turn_id
                WHERE e.session_id = ?1
                ORDER BY e.id
                "#,
                [session_id],
            )
            .await?;

        let mut events = Vec::new();
        while let Some(row) = rows.next().await? {
            events.push(EventRow {
                id: row.get::<i64>(0)?,
                session_id: row.get::<i64>(1)?,
                turn_id: row.get::<Option<i64>>(2)?,
                event_type: row.get::<String>(3)?,
                payload: row.get::<String>(4)?,
                turn_index: row.get::<Option<i64>>(5)?.map(|value| value as u32),
                created_at: row.get::<String>(6)?,
            });
        }
        Ok(events)
    }

    async fn filter_events_for_branch_head(
        &self,
        session_id: i64,
        branch_head_id: Option<i64>,
        events: Vec<EventRow>,
    ) -> Result<Vec<EventRow>> {
        let active_turn_ids = self
            .branch_path_turns(session_id, branch_head_id)
            .await?
            .into_iter()
            .map(|turn| turn.id)
            .collect::<HashSet<_>>();
        Ok(self.filter_events_for_turn_ids(events, &active_turn_ids))
    }

    fn filter_events_for_turn_ids(
        &self,
        events: Vec<EventRow>,
        turn_ids: &HashSet<i64>,
    ) -> Vec<EventRow> {
        events
            .into_iter()
            .filter(|event| {
                event
                    .turn_id
                    .is_none_or(|turn_id| turn_ids.contains(&turn_id))
            })
            .collect()
    }
}
