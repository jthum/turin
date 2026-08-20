use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};
use turin_daemon_protocol::{SessionSearchHitKind, SessionSearchScope};

use crate::persistence::state::{SessionSearchRow, StateStore};

#[derive(Debug)]
struct RankedSessionSearchHit {
    row: SessionSearchRow,
    sort_id: i64,
}

impl StateStore {
    pub async fn search_session_history(
        &self,
        query: &str,
        scope: SessionSearchScope,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<SessionSearchRow>> {
        let normalized = query.trim().to_ascii_lowercase();
        if normalized.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }

        let mut hits = Vec::new();
        if matches!(
            scope,
            SessionSearchScope::All | SessionSearchScope::Sessions
        ) {
            hits.extend(self.search_session_title_hits(&normalized).await?);
        }
        if matches!(
            scope,
            SessionSearchScope::All | SessionSearchScope::Messages
        ) {
            hits.extend(self.search_active_branch_message_hits(&normalized).await?);
        }
        if matches!(
            scope,
            SessionSearchScope::All | SessionSearchScope::ToolExecutions
        ) {
            hits.extend(self.search_active_branch_tool_hits(&normalized).await?);
        }
        if matches!(scope, SessionSearchScope::All | SessionSearchScope::Events) {
            hits.extend(self.search_event_hits(&normalized).await?);
        }

        if hits.is_empty() {
            return Ok(Vec::new());
        }

        hits.sort_by(|left, right| {
            right
                .row
                .score
                .cmp(&left.row.score)
                .then_with(|| right.row.created_at.cmp(&left.row.created_at))
                .then_with(|| right.sort_id.cmp(&left.sort_id))
        });

        Ok(hits
            .into_iter()
            .skip(offset)
            .take(limit)
            .map(|hit| hit.row)
            .collect())
    }
}

impl StateStore {
    async fn search_session_title_hits(
        &self,
        normalized: &str,
    ) -> Result<Vec<RankedSessionSearchHit>> {
        let conn = self.connect().await?;
        let needle = format!("%{normalized}%");
        let mut rows = conn
            .query(
                r#"
                SELECT CASE
                           WHEN LOWER(s.agent_id) = ?2 THEN 1200
                           WHEN LOWER(COALESCE(s.metadata, '')) LIKE (?2 || '%') THEN 1120
                           WHEN LOWER(s.agent_id) LIKE (?2 || '%') THEN 1080
                           ELSE 980
                       END AS score,
                       s.id AS sort_id,
                       s.public_id,
                       s.agent_id,
                       s.metadata,
                       s.created_at,
                       COALESCE(s.metadata, s.agent_id) AS match_text
                FROM sessions s
                WHERE LOWER(s.agent_id) LIKE ?1
                   OR LOWER(COALESCE(s.metadata, '')) LIKE ?1
                "#,
                turso::params![needle, normalized],
            )
            .await
            .context("Failed to search persisted session titles")?;

        let mut hits = Vec::new();
        while let Some(row) = rows.next().await? {
            hits.push(RankedSessionSearchHit {
                sort_id: row.get::<i64>(1)?,
                row: SessionSearchRow {
                    kind: SessionSearchHitKind::Session,
                    score: row.get::<i64>(0)?,
                    public_id: row.get::<Vec<u8>>(2)?,
                    agent_id: row.get::<String>(3)?,
                    metadata: row.get::<Option<String>>(4)?,
                    created_at: row.get::<String>(5)?,
                    turn_index: None,
                    role: None,
                    tool_name: None,
                    event_type: None,
                    match_text: row.get::<String>(6)?,
                },
            });
        }
        Ok(hits)
    }

    async fn search_active_branch_message_hits(
        &self,
        normalized: &str,
    ) -> Result<Vec<RankedSessionSearchHit>> {
        let conn = self.connect().await?;
        let needle = format!("%{normalized}%");
        let mut rows = conn
            .query(
                r#"
                SELECT CASE
                           WHEN LOWER(tm.role) = ?2 THEN 860
                           WHEN instr(LOWER(tm.content), ?2) = 1 THEN 820
                           ELSE 740
                       END AS score,
                       tm.id AS sort_id,
                       s.public_id,
                       s.agent_id,
                       s.metadata,
                       tm.created_at,
                       t.branch_depth,
                       tm.role,
                       tm.content,
                       t.session_id,
                       t.id
                FROM messages tm
                JOIN turns t ON t.id = tm.turn_id
                JOIN sessions s ON s.id = t.session_id
                WHERE LOWER(tm.content) LIKE ?1
                   OR LOWER(tm.role) LIKE ?1
                "#,
                turso::params![needle, normalized],
            )
            .await
            .context("Failed to search active-branch messages")?;

        let mut active_turn_ids_by_session = HashMap::<i64, HashSet<i64>>::new();
        let mut hits = Vec::new();
        while let Some(row) = rows.next().await? {
            let session_id = row.get::<i64>(9)?;
            let turn_id = row.get::<i64>(10)?;
            if !self
                .active_branch_turn_ids_contains(
                    session_id,
                    turn_id,
                    &mut active_turn_ids_by_session,
                )
                .await?
            {
                continue;
            }
            hits.push(RankedSessionSearchHit {
                sort_id: row.get::<i64>(1)?,
                row: SessionSearchRow {
                    kind: SessionSearchHitKind::Message,
                    score: row.get::<i64>(0)?,
                    public_id: row.get::<Vec<u8>>(2)?,
                    agent_id: row.get::<String>(3)?,
                    metadata: row.get::<Option<String>>(4)?,
                    created_at: row.get::<String>(5)?,
                    turn_index: Some(super::super::persisted_u32(
                        &format!("message search turn {turn_id}"),
                        "turn index",
                        row.get::<i64>(6)?,
                    )?),
                    role: Some(row.get::<String>(7)?),
                    tool_name: None,
                    event_type: None,
                    match_text: row.get::<String>(8)?,
                },
            });
        }

        Ok(hits)
    }

    async fn search_active_branch_tool_hits(
        &self,
        normalized: &str,
    ) -> Result<Vec<RankedSessionSearchHit>> {
        let conn = self.connect().await?;
        let needle = format!("%{normalized}%");
        let mut rows = conn
            .query(
                r#"
                SELECT CASE
                           WHEN LOWER(tt.tool_name) = ?2 THEN 900
                           WHEN LOWER(tt.tool_name) LIKE (?2 || '%') THEN 860
                           WHEN instr(LOWER(COALESCE(tt.args, '')), ?2) = 1 THEN 760
                           WHEN instr(LOWER(COALESCE(tt.output, '')), ?2) = 1 THEN 740
                           ELSE 700
                       END AS score,
                       tt.id AS sort_id,
                       s.public_id,
                       s.agent_id,
                       s.metadata,
                       tt.created_at,
                       t.branch_depth,
                       tt.tool_name,
                       TRIM(
                           tt.tool_name || ' ' ||
                           COALESCE(tt.args, '') || ' ' ||
                           COALESCE(tt.output, '') || ' ' ||
                           COALESCE(tt.verdict, '')
                       ) AS match_text,
                       t.session_id,
                       t.id
                FROM tool_executions tt
                JOIN turns t ON t.id = tt.turn_id
                JOIN sessions s ON s.id = t.session_id
                WHERE LOWER(tt.tool_name) LIKE ?1
                   OR LOWER(COALESCE(tt.args, '')) LIKE ?1
                   OR LOWER(COALESCE(tt.output, '')) LIKE ?1
                   OR LOWER(COALESCE(tt.verdict, '')) LIKE ?1
                "#,
                turso::params![needle, normalized],
            )
            .await
            .context("Failed to search active-branch tool executions")?;

        let mut active_turn_ids_by_session = HashMap::<i64, HashSet<i64>>::new();
        let mut hits = Vec::new();
        while let Some(row) = rows.next().await? {
            let session_id = row.get::<i64>(9)?;
            let turn_id = row.get::<i64>(10)?;
            if !self
                .active_branch_turn_ids_contains(
                    session_id,
                    turn_id,
                    &mut active_turn_ids_by_session,
                )
                .await?
            {
                continue;
            }
            hits.push(RankedSessionSearchHit {
                sort_id: row.get::<i64>(1)?,
                row: SessionSearchRow {
                    kind: SessionSearchHitKind::ToolExecution,
                    score: row.get::<i64>(0)?,
                    public_id: row.get::<Vec<u8>>(2)?,
                    agent_id: row.get::<String>(3)?,
                    metadata: row.get::<Option<String>>(4)?,
                    created_at: row.get::<String>(5)?,
                    turn_index: Some(super::super::persisted_u32(
                        &format!("tool search turn {turn_id}"),
                        "turn index",
                        row.get::<i64>(6)?,
                    )?),
                    role: None,
                    tool_name: Some(row.get::<String>(7)?),
                    event_type: None,
                    match_text: row.get::<String>(8)?,
                },
            });
        }

        Ok(hits)
    }

    async fn search_event_hits(&self, normalized: &str) -> Result<Vec<RankedSessionSearchHit>> {
        let conn = self.connect().await?;
        let needle = format!("%{normalized}%");
        let mut rows = conn
            .query(
                r#"
                SELECT CASE
                           WHEN LOWER(e.event_type) = ?2 THEN 820
                           WHEN LOWER(e.event_type) LIKE (?2 || '%') THEN 780
                           ELSE 680
                       END AS score,
                       e.id AS sort_id,
                       s.public_id,
                       s.agent_id,
                       s.metadata,
                       e.created_at,
                       t.branch_depth,
                       e.event_type,
                       e.payload,
                       e.session_id,
                       e.turn_id
                FROM events e
                JOIN sessions s ON s.id = e.session_id
                LEFT JOIN turns t ON t.id = e.turn_id
                WHERE LOWER(e.event_type) LIKE ?1
                   OR LOWER(e.payload) LIKE ?1
                "#,
                turso::params![needle, normalized],
            )
            .await
            .context("Failed to search persisted events")?;

        let mut active_turn_ids_by_session = HashMap::<i64, HashSet<i64>>::new();
        let mut hits = Vec::new();
        while let Some(row) = rows.next().await? {
            let session_id = row.get::<i64>(9)?;
            let turn_id = row.get::<Option<i64>>(10)?;
            if let Some(turn_id) = turn_id
                && !self
                    .active_branch_turn_ids_contains(
                        session_id,
                        turn_id,
                        &mut active_turn_ids_by_session,
                    )
                    .await?
            {
                continue;
            }
            hits.push(RankedSessionSearchHit {
                sort_id: row.get::<i64>(1)?,
                row: SessionSearchRow {
                    kind: SessionSearchHitKind::Event,
                    score: row.get::<i64>(0)?,
                    public_id: row.get::<Vec<u8>>(2)?,
                    agent_id: row.get::<String>(3)?,
                    metadata: row.get::<Option<String>>(4)?,
                    created_at: row.get::<String>(5)?,
                    turn_index: super::super::persisted_optional_u32(
                        &format!("event search session {session_id}"),
                        "turn index",
                        row.get::<Option<i64>>(6)?,
                    )?,
                    role: None,
                    tool_name: None,
                    event_type: Some(row.get::<String>(7)?),
                    match_text: row.get::<String>(8)?,
                },
            });
        }

        Ok(hits)
    }

    async fn active_branch_turn_ids_contains(
        &self,
        session_id: i64,
        turn_id: i64,
        active_turn_ids_by_session: &mut HashMap<i64, HashSet<i64>>,
    ) -> Result<bool> {
        if let Some(turn_ids) = active_turn_ids_by_session.get(&session_id) {
            return Ok(turn_ids.contains(&turn_id));
        }

        let turn_ids = self
            .active_branch_path_turns(session_id)
            .await?
            .into_iter()
            .map(|turn| turn.id)
            .collect::<HashSet<_>>();
        let contains = turn_ids.contains(&turn_id);
        active_turn_ids_by_session.insert(session_id, turn_ids);
        Ok(contains)
    }
}
