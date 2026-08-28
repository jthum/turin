use std::collections::HashSet;

use anyhow::{Context, Result};
use turso::Value as SqlValue;

use super::{EventRow, SessionReadTarget, StateStore, TurnRow, TurnWriteTarget};

const INSERT_EVENT_SQL: &str =
    "INSERT INTO events (session_id, turn_id, event_type, payload) VALUES (?1, ?2, ?3, ?4)";

#[derive(Debug, Clone, Copy)]
enum EventTurnClause {
    All,
    SessionLevelOnly,
    Turns,
    SessionLevelOrTurns,
}

fn event_row_from_sql_row(row: &turso::Row) -> Result<EventRow> {
    let id = row.get::<i64>(0)?;
    Ok(EventRow {
        id,
        session_id: row.get::<i64>(1)?,
        turn_id: row.get::<Option<i64>>(2)?,
        event_type: row.get::<String>(3)?,
        payload: row.get::<String>(4)?,
        turn_index: super::persisted_optional_u32(
            &format!("event {id}"),
            "turn index",
            row.get::<Option<i64>>(5)?,
        )?,
        created_at: row.get::<String>(6)?,
    })
}

fn append_event_turn_filter(
    sql: &mut String,
    params: &mut Vec<SqlValue>,
    next: &mut usize,
    turn_ids: Option<&[i64]>,
    turn_clause: EventTurnClause,
) {
    match turn_clause {
        EventTurnClause::All => {}
        EventTurnClause::SessionLevelOnly => sql.push_str(" AND e.turn_id IS NULL"),
        EventTurnClause::Turns | EventTurnClause::SessionLevelOrTurns => {
            let turn_ids = turn_ids.unwrap_or(&[]);
            if turn_ids.is_empty() {
                if matches!(turn_clause, EventTurnClause::SessionLevelOrTurns) {
                    sql.push_str(" AND e.turn_id IS NULL");
                }
            } else {
                let placeholders = (*next..*next + turn_ids.len())
                    .map(|index| format!("?{index}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                if matches!(turn_clause, EventTurnClause::SessionLevelOrTurns) {
                    sql.push_str(&format!(
                        " AND (e.turn_id IS NULL OR e.turn_id IN ({placeholders}))"
                    ));
                } else {
                    sql.push_str(&format!(" AND e.turn_id IN ({placeholders})"));
                }
                params.extend(turn_ids.iter().copied().map(SqlValue::Integer));
                *next += turn_ids.len();
            }
        }
    }
}

fn append_event_type_filter(
    sql: &mut String,
    params: &mut Vec<SqlValue>,
    next: &mut usize,
    event_types: Option<&[&str]>,
) {
    let Some(event_types) = event_types else {
        return;
    };
    let placeholders = (*next..*next + event_types.len())
        .map(|index| format!("?{index}"))
        .collect::<Vec<_>>()
        .join(", ");
    sql.push_str(&format!(" AND e.event_type IN ({placeholders})"));
    params.extend(
        event_types
            .iter()
            .map(|event_type| SqlValue::Text((*event_type).to_string())),
    );
    *next += event_types.len();
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionCounters {
    pub next_task_id: u32,
    pub next_plan_id: u32,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
}

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
        self.query_events_matching(session_id, None, None, EventTurnClause::All, None)
            .await
    }

    pub async fn get_events(
        &self,
        session_id: i64,
        target: &SessionReadTarget,
    ) -> Result<Vec<EventRow>> {
        let turn_ids = self.turn_ids_for_read_target(session_id, target).await?;
        self.query_events_for_turns(session_id, &turn_ids, None, None)
            .await
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
        let turn_ids = self.turn_ids_for_read_target(session_id, target).await?;
        self.query_events_for_turns(session_id, &turn_ids, Some(event_types), None)
            .await
    }

    pub async fn get_event_window(
        &self,
        session_id: i64,
        target: &SessionReadTarget,
        offset: Option<usize>,
        limit: usize,
        event_types: Option<&[&str]>,
    ) -> Result<(Vec<EventRow>, usize, usize)> {
        if event_types.is_some_and(<[&str]>::is_empty) {
            return Ok((Vec::new(), 0, 0));
        }
        let turn_ids = self.turn_ids_for_read_target(session_id, target).await?;
        let total = self
            .count_events_for_turns(session_id, &turn_ids, event_types)
            .await?;
        let offset = offset.unwrap_or_else(|| total.saturating_sub(limit));
        if limit == 0 || offset >= total {
            return Ok((Vec::new(), total, offset));
        }
        let end = offset.saturating_add(limit).min(total);
        let scan_descending = total.saturating_sub(end) < offset;
        let skip = if scan_descending {
            total.saturating_sub(end)
        } else {
            offset
        };

        const MIN_CANDIDATE_PAGE_SIZE: usize = 128;
        const MAX_CANDIDATE_PAGE_SIZE: usize = 1024;
        let page_size = limit.clamp(MIN_CANDIDATE_PAGE_SIZE, MAX_CANDIDATE_PAGE_SIZE);
        let mut candidate_offset = 0usize;
        let mut accepted = 0usize;
        let mut events = Vec::with_capacity(limit.min(total.saturating_sub(offset)));

        loop {
            let candidates = self
                .query_event_candidate_page(
                    session_id,
                    event_types,
                    candidate_offset,
                    page_size,
                    scan_descending,
                )
                .await?;
            let candidate_count = candidates.len();
            for event in candidates {
                if event
                    .turn_id
                    .is_some_and(|turn_id| !turn_ids.contains(&turn_id))
                {
                    continue;
                }
                if accepted < skip {
                    accepted += 1;
                    continue;
                }
                events.push(event);
                if events.len() == limit {
                    if scan_descending {
                        events.reverse();
                    }
                    return Ok((events, total, offset));
                }
            }
            if candidate_count < page_size {
                if scan_descending {
                    events.reverse();
                }
                return Ok((events, total, offset));
            }
            candidate_offset = candidate_offset.saturating_add(candidate_count);
        }
    }

    pub async fn get_latest_session_event_by_type(
        &self,
        session_id: i64,
        event_type: &str,
    ) -> Result<Option<EventRow>> {
        Ok(self
            .query_recent_events_by_types(session_id, &[event_type], 1, 0)
            .await?
            .pop())
    }

    pub(crate) async fn get_recent_session_events_by_type_page(
        &self,
        session_id: i64,
        event_type: &str,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<EventRow>> {
        self.query_recent_events_by_types(session_id, &[event_type], limit, offset)
            .await
    }

    pub async fn get_session_counters(&self, session_id: i64) -> Result<SessionCounters> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                r#"
                SELECT COALESCE(MAX(
                           CASE
                               WHEN json_extract(payload, '$.task_id') GLOB 't_[0-9]*'
                               THEN CAST(substr(json_extract(payload, '$.task_id'), 3) AS INTEGER)
                           END
                       ), 0) + 1,
                       COALESCE(MAX(
                           CASE
                               WHEN json_extract(payload, '$.plan_id') GLOB 'p_[0-9]*'
                               THEN CAST(substr(json_extract(payload, '$.plan_id'), 3) AS INTEGER)
                           END
                       ), 0) + 1,
                       COALESCE(SUM(
                           CASE WHEN event_type = 'message_end'
                               THEN CAST(COALESCE(json_extract(payload, '$.input_tokens'), 0) AS INTEGER)
                               ELSE 0
                           END
                       ), 0),
                       COALESCE(SUM(
                           CASE WHEN event_type = 'message_end'
                               THEN CAST(COALESCE(json_extract(payload, '$.output_tokens'), 0) AS INTEGER)
                               ELSE 0
                           END
                       ), 0)
                FROM events
                WHERE session_id = ?1
                  AND event_type IN ('task_start', 'task_complete', 'plan_complete', 'message_end')
                "#,
                [session_id],
            )
            .await?;
        let row = rows
            .next()
            .await?
            .ok_or_else(|| anyhow::anyhow!("Session counter aggregate returned no row"))?;
        let record = format!("session {session_id} counters");
        Ok(SessionCounters {
            next_task_id: super::persisted_u32(&record, "next task id", row.get::<i64>(0)?)?,
            next_plan_id: super::persisted_u32(&record, "next plan id", row.get::<i64>(1)?)?,
            total_input_tokens: super::persisted_u64(
                &record,
                "total input tokens",
                row.get::<i64>(2)?,
            )?,
            total_output_tokens: super::persisted_u64(
                &record,
                "total output tokens",
                row.get::<i64>(3)?,
            )?,
        })
    }

    pub async fn get_recent_events_by_types(
        &self,
        session_id: i64,
        target: &SessionReadTarget,
        event_types: &[&str],
        limit: usize,
    ) -> Result<Vec<EventRow>> {
        if event_types.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let turn_ids = self.turn_ids_for_read_target(session_id, target).await?;
        self.query_events_for_turns(session_id, &turn_ids, Some(event_types), Some(limit))
            .await
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

    async fn turn_ids_for_read_target(
        &self,
        session_id: i64,
        target: &SessionReadTarget,
    ) -> Result<HashSet<i64>> {
        Ok(match target {
            SessionReadTarget::ActiveBranch => self
                .branch_path_turns(session_id, None)
                .await?
                .into_iter()
                .map(|turn| turn.id)
                .collect(),
            SessionReadTarget::BranchHead(branch_head_id) => self
                .branch_path_turns(session_id, Some(*branch_head_id))
                .await?
                .into_iter()
                .map(|turn| turn.id)
                .collect(),
            SessionReadTarget::TurnId(turn_id) => self
                .turn_path_to_turn_id(session_id, *turn_id)
                .await?
                .into_iter()
                .map(|turn| turn.id)
                .collect(),
            SessionReadTarget::SelectedPath(turn_ids) => self
                .turn_rows_for_selected_path(session_id, turn_ids)
                .await?
                .into_iter()
                .map(|turn| turn.id)
                .collect(),
        })
    }

    async fn query_events_for_turns(
        &self,
        session_id: i64,
        turn_ids: &HashSet<i64>,
        event_types: Option<&[&str]>,
        limit: Option<usize>,
    ) -> Result<Vec<EventRow>> {
        const TURN_QUERY_CHUNK: usize = 500;
        if limit == Some(0) {
            return Ok(Vec::new());
        }
        let turn_list: Vec<i64> = turn_ids.iter().copied().collect();
        if turn_list.len() <= TURN_QUERY_CHUNK {
            return self
                .query_events_matching(
                    session_id,
                    Some(&turn_list),
                    event_types,
                    EventTurnClause::SessionLevelOrTurns,
                    limit,
                )
                .await;
        }

        let mut events = self
            .query_events_matching(
                session_id,
                None,
                event_types,
                EventTurnClause::SessionLevelOnly,
                limit,
            )
            .await?;
        for chunk in turn_list.chunks(TURN_QUERY_CHUNK) {
            events.extend(
                self.query_events_matching(
                    session_id,
                    Some(chunk),
                    event_types,
                    EventTurnClause::Turns,
                    limit,
                )
                .await?,
            );
        }
        events.sort_by_key(|event| event.id);
        events.dedup_by_key(|event| event.id);
        if let Some(limit) = limit {
            let start = events.len().saturating_sub(limit);
            events = events.split_off(start);
        }
        Ok(events)
    }

    async fn count_events_for_turns(
        &self,
        session_id: i64,
        turn_ids: &HashSet<i64>,
        event_types: Option<&[&str]>,
    ) -> Result<usize> {
        const TURN_QUERY_CHUNK: usize = 500;
        let turn_list = turn_ids.iter().copied().collect::<Vec<_>>();
        if turn_list.len() <= TURN_QUERY_CHUNK {
            return self
                .query_event_count_matching(
                    session_id,
                    Some(&turn_list),
                    event_types,
                    EventTurnClause::SessionLevelOrTurns,
                )
                .await;
        }

        let mut total = self
            .query_event_count_matching(
                session_id,
                None,
                event_types,
                EventTurnClause::SessionLevelOnly,
            )
            .await?;
        for chunk in turn_list.chunks(TURN_QUERY_CHUNK) {
            total = total.saturating_add(
                self.query_event_count_matching(
                    session_id,
                    Some(chunk),
                    event_types,
                    EventTurnClause::Turns,
                )
                .await?,
            );
        }
        Ok(total)
    }

    async fn query_event_candidate_page(
        &self,
        session_id: i64,
        event_types: Option<&[&str]>,
        offset: usize,
        limit: usize,
        descending: bool,
    ) -> Result<Vec<EventRow>> {
        let conn = self.connect().await?;
        let mut sql = String::from(
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
            "#,
        );
        let mut params = vec![SqlValue::Integer(session_id)];
        let mut next = 2;
        append_event_type_filter(&mut sql, &mut params, &mut next, event_types);
        let direction = if descending { "DESC" } else { "ASC" };
        sql.push_str(&format!(
            " ORDER BY e.id {direction} LIMIT ?{next} OFFSET ?{}",
            next + 1
        ));
        params.push(SqlValue::Integer(limit as i64));
        params.push(SqlValue::Integer(offset as i64));

        let mut rows = conn.prepare(&sql).await?.query(params).await?;
        let mut events = Vec::new();
        while let Some(row) = rows.next().await? {
            events.push(event_row_from_sql_row(&row)?);
        }
        Ok(events)
    }

    async fn query_event_count_matching(
        &self,
        session_id: i64,
        turn_ids: Option<&[i64]>,
        event_types: Option<&[&str]>,
        turn_clause: EventTurnClause,
    ) -> Result<usize> {
        let conn = self.connect().await?;
        let mut sql = String::from("SELECT COUNT(*) FROM events e WHERE e.session_id = ?1");
        let mut params = vec![SqlValue::Integer(session_id)];
        let mut next = 2;
        append_event_turn_filter(&mut sql, &mut params, &mut next, turn_ids, turn_clause);
        append_event_type_filter(&mut sql, &mut params, &mut next, event_types);
        let mut rows = conn.prepare(&sql).await?.query(params).await?;
        let row = rows
            .next()
            .await?
            .context("Event count query returned no row")?;
        let count = row.get::<i64>(0)?;
        usize::try_from(count).context("Event count cannot be represented as usize")
    }

    async fn query_events_matching(
        &self,
        session_id: i64,
        turn_ids: Option<&[i64]>,
        event_types: Option<&[&str]>,
        turn_clause: EventTurnClause,
        limit: Option<usize>,
    ) -> Result<Vec<EventRow>> {
        if matches!(turn_clause, EventTurnClause::Turns)
            && turn_ids.is_some_and(|ids| ids.is_empty())
        {
            return Ok(Vec::new());
        }
        let conn = self.connect().await?;
        let mut sql = String::from(
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
            "#,
        );
        let mut params = vec![SqlValue::Integer(session_id)];
        let mut next = 2;
        append_event_turn_filter(&mut sql, &mut params, &mut next, turn_ids, turn_clause);
        append_event_type_filter(&mut sql, &mut params, &mut next, event_types);
        if let Some(limit) = limit {
            sql.push_str(&format!(" ORDER BY e.id DESC LIMIT ?{next}"));
            params.push(SqlValue::Integer(limit as i64));
        } else {
            sql.push_str(" ORDER BY e.id");
        }
        let mut stmt = conn.prepare(&sql).await?;
        let mut rows = stmt.query(params).await?;
        let mut events = Vec::new();
        while let Some(row) = rows.next().await? {
            events.push(event_row_from_sql_row(&row)?);
        }
        if limit.is_some() {
            events.reverse();
        }
        Ok(events)
    }

    async fn query_recent_events_by_types(
        &self,
        session_id: i64,
        event_types: &[&str],
        limit: usize,
        offset: usize,
    ) -> Result<Vec<EventRow>> {
        let conn = self.connect().await?;
        let placeholders = (2..event_types.len() + 2)
            .map(|index| format!("?{index}"))
            .collect::<Vec<_>>()
            .join(", ");
        let limit_index = event_types.len() + 2;
        let offset_index = event_types.len() + 3;
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
                ORDER BY e.id DESC
                LIMIT ?{limit_index}
                OFFSET ?{offset_index}
                "#
        );
        let mut params = Vec::with_capacity(event_types.len() + 3);
        params.push(SqlValue::Integer(session_id));
        params.extend(
            event_types
                .iter()
                .map(|event_type| SqlValue::Text((*event_type).to_string())),
        );
        params.push(SqlValue::Integer(limit as i64));
        params.push(SqlValue::Integer(offset as i64));
        let mut stmt = conn.prepare(&sql).await?;
        let mut rows = stmt.query(params).await?;
        let mut events = Vec::new();
        while let Some(row) = rows.next().await? {
            events.push(event_row_from_sql_row(&row)?);
        }
        events.reverse();
        Ok(events)
    }
}
