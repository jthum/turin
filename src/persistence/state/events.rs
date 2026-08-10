use std::collections::HashSet;

use anyhow::{Context, Result};
use turso::Value as SqlValue;

use super::{EventRow, SessionReadTarget, StateStore, TurnRow, TurnWriteTarget};

const INSERT_EVENT_SQL: &str =
    "INSERT INTO events (session_id, turn_id, event_type, payload) VALUES (?1, ?2, ?3, ?4)";

pub(crate) struct EventWriter {
    store: StateStore,
    conn: turso::Connection,
}

impl EventWriter {
    pub async fn insert_event(
        &self,
        session_id: i64,
        target: Option<TurnWriteTarget>,
        event_type: &str,
        payload: &serde_json::Value,
    ) -> Result<()> {
        self.store
            .insert_event_with_conn(&self.conn, session_id, target, event_type, payload)
            .await
    }
}

impl StateStore {
    pub(crate) async fn event_writer(&self) -> Result<EventWriter> {
        Ok(EventWriter {
            store: self.clone(),
            conn: self.connect().await?,
        })
    }

    pub async fn insert_event(
        &self,
        session_id: i64,
        target: Option<TurnWriteTarget>,
        event_type: &str,
        payload: &serde_json::Value,
    ) -> Result<()> {
        let conn = self.connect().await?;
        self.insert_event_with_conn(&conn, session_id, target, event_type, payload)
            .await
    }

    async fn insert_event_with_conn(
        &self,
        conn: &turso::Connection,
        session_id: i64,
        target: Option<TurnWriteTarget>,
        event_type: &str,
        payload: &serde_json::Value,
    ) -> Result<()> {
        let payload_str = serde_json::to_string(payload)?;
        let turn_id = self
            .resolve_event_turn_with_conn(conn, session_id, target)
            .await?
            .map(|turn| turn.id);
        let mut stmt = conn.prepare_cached(INSERT_EVENT_SQL).await?;
        stmt.execute(turso::params![session_id, turn_id, event_type, payload_str])
            .await
            .with_context(|| format!("Failed to insert event for session: {}", session_id))?;
        Ok(())
    }

    async fn resolve_event_turn_with_conn(
        &self,
        conn: &turso::Connection,
        session_id: i64,
        target: Option<TurnWriteTarget>,
    ) -> Result<Option<TurnRow>> {
        match target {
            None => Ok(None),
            Some(TurnWriteTarget::ExistingTurn {
                turn_id,
                turn_index,
            }) => {
                let Some(turn) = self.get_turn_row_with_conn(conn, turn_id).await? else {
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
            Some(target) => self.resolve_turn_for_write_target(session_id, target).await,
        }
    }

    pub async fn get_all_events(&self, session_id: i64) -> Result<Vec<EventRow>> {
        self.query_events(session_id).await
    }

    pub async fn get_events(
        &self,
        session_id: i64,
        target: &SessionReadTarget,
    ) -> Result<Vec<EventRow>> {
        let events = self.query_events(session_id).await?;
        match target {
            SessionReadTarget::ActiveBranch => {
                self.filter_events_for_branch_head(session_id, None, events)
                    .await
            }
            SessionReadTarget::BranchHead(branch_head_id) => {
                self.filter_events_for_branch_head(session_id, Some(*branch_head_id), events)
                    .await
            }
            SessionReadTarget::TurnId(turn_id) => {
                let turn_ids = self
                    .turn_path_to_turn_id(session_id, *turn_id)
                    .await?
                    .into_iter()
                    .map(|turn| turn.id)
                    .collect::<HashSet<_>>();
                Ok(self.filter_events_for_turn_ids(events, &turn_ids))
            }
            SessionReadTarget::SelectedPath(turn_ids) => {
                let turn_ids = self
                    .turn_rows_for_selected_path(session_id, turn_ids)
                    .await?
                    .into_iter()
                    .map(|turn| turn.id)
                    .collect::<HashSet<_>>();
                Ok(self.filter_events_for_turn_ids(events, &turn_ids))
            }
        }
    }

    pub async fn get_events_by_types(
        &self,
        session_id: i64,
        target: &SessionReadTarget,
        event_types: &[&str],
    ) -> Result<Vec<EventRow>> {
        if event_types.is_empty() {
            return Ok(Vec::new());
        }
        let events = self.query_events_by_types(session_id, event_types).await?;
        match target {
            SessionReadTarget::ActiveBranch => {
                self.filter_events_for_branch_head(session_id, None, events)
                    .await
            }
            SessionReadTarget::BranchHead(branch_head_id) => {
                self.filter_events_for_branch_head(session_id, Some(*branch_head_id), events)
                    .await
            }
            SessionReadTarget::TurnId(turn_id) => {
                let turn_ids = self
                    .turn_path_to_turn_id(session_id, *turn_id)
                    .await?
                    .into_iter()
                    .map(|turn| turn.id)
                    .collect::<HashSet<_>>();
                Ok(self.filter_events_for_turn_ids(events, &turn_ids))
            }
            SessionReadTarget::SelectedPath(turn_ids) => {
                let turn_ids = self
                    .turn_rows_for_selected_path(session_id, turn_ids)
                    .await?
                    .into_iter()
                    .map(|turn| turn.id)
                    .collect::<HashSet<_>>();
                Ok(self.filter_events_for_turn_ids(events, &turn_ids))
            }
        }
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

    async fn query_events_by_types(
        &self,
        session_id: i64,
        event_types: &[&str],
    ) -> Result<Vec<EventRow>> {
        let conn = self.connect().await?;
        let placeholders = (2..event_types.len() + 2)
            .map(|index| format!("?{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
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
                  AND e.event_type IN ({placeholders})
                ORDER BY e.id
                "#
        );
        let mut params = Vec::with_capacity(event_types.len() + 1);
        params.push(SqlValue::Integer(session_id));
        params.extend(
            event_types
                .iter()
                .map(|event_type| SqlValue::Text((*event_type).to_string())),
        );
        let mut stmt = conn.prepare(&sql).await?;
        let mut rows = stmt.query(params).await?;
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
