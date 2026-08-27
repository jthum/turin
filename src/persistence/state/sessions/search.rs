use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};
use turin_daemon_protocol::{SessionSearchHitKind, SessionSearchScope};

use crate::persistence::state::{SessionSearchRow, StateStore};

const MIN_CANDIDATE_PAGE_SIZE: usize = 128;
const MAX_CANDIDATE_PAGE_SIZE: usize = 1024;

struct SessionSearchCandidate {
    row: SessionSearchRow,
    session_id: i64,
    turn_id: Option<i64>,
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

        let page_size = limit
            .saturating_add(offset)
            .clamp(MIN_CANDIDATE_PAGE_SIZE, MAX_CANDIDATE_PAGE_SIZE);
        let mut candidate_offset = 0usize;
        let mut accepted = 0usize;
        let mut active_turn_ids_by_session = HashMap::<i64, HashSet<i64>>::new();
        let mut results = Vec::with_capacity(limit);

        loop {
            let candidates = self
                .query_ranked_session_search_candidates(
                    &normalized,
                    scope,
                    page_size,
                    candidate_offset,
                )
                .await?;
            let candidate_count = candidates.len();

            for candidate in candidates {
                if !self
                    .search_candidate_is_on_active_path(&candidate, &mut active_turn_ids_by_session)
                    .await?
                {
                    continue;
                }
                if accepted < offset {
                    accepted += 1;
                    continue;
                }
                results.push(candidate.row);
                if results.len() == limit {
                    return Ok(results);
                }
            }

            if candidate_count < page_size {
                return Ok(results);
            }
            candidate_offset = candidate_offset.saturating_add(candidate_count);
        }
    }

    async fn query_ranked_session_search_candidates(
        &self,
        normalized: &str,
        scope: SessionSearchScope,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<SessionSearchCandidate>> {
        let conn = self.connect().await?;
        let sql = ranked_session_search_sql(scope);
        let needle = format!("%{normalized}%");
        let mut stmt = conn.prepare(&sql).await?;
        let mut rows = stmt
            .query(turso::params![
                needle,
                normalized,
                limit as i64,
                offset as i64
            ])
            .await
            .context("Failed to search persisted session history")?;

        let mut hits = Vec::new();
        while let Some(row) = rows.next().await? {
            let sort_id = row.get::<i64>(1)?;
            let kind = search_hit_kind(&row.get::<String>(11)?)?;
            let turn_id = row.get::<Option<i64>>(12)?;
            let session_id = row.get::<i64>(13)?;
            hits.push(SessionSearchCandidate {
                row: SessionSearchRow {
                    kind,
                    score: row.get::<i64>(0)?,
                    public_id: row.get::<Vec<u8>>(2)?,
                    agent_id: row.get::<String>(3)?,
                    metadata: row.get::<Option<String>>(4)?,
                    created_at: row.get::<String>(5)?,
                    turn_index: super::super::persisted_optional_u32(
                        &search_hit_record(kind, sort_id, session_id, turn_id),
                        "turn index",
                        row.get::<Option<i64>>(6)?,
                    )?,
                    role: row.get::<Option<String>>(7)?,
                    tool_name: row.get::<Option<String>>(8)?,
                    event_type: row.get::<Option<String>>(9)?,
                    match_text: row.get::<String>(10)?,
                },
                session_id,
                turn_id,
            });
        }
        Ok(hits)
    }

    async fn search_candidate_is_on_active_path(
        &self,
        candidate: &SessionSearchCandidate,
        active_turn_ids_by_session: &mut HashMap<i64, HashSet<i64>>,
    ) -> Result<bool> {
        let Some(turn_id) = candidate.turn_id else {
            return Ok(true);
        };
        if let std::collections::hash_map::Entry::Vacant(entry) =
            active_turn_ids_by_session.entry(candidate.session_id)
        {
            let turn_ids = self
                .active_branch_path_turns(candidate.session_id)
                .await?
                .into_iter()
                .map(|turn| turn.id)
                .collect();
            entry.insert(turn_ids);
        }
        Ok(active_turn_ids_by_session
            .get(&candidate.session_id)
            .is_some_and(|turn_ids| turn_ids.contains(&turn_id)))
    }
}

fn ranked_session_search_sql(scope: SessionSearchScope) -> String {
    let mut arms = Vec::new();
    if matches!(
        scope,
        SessionSearchScope::All | SessionSearchScope::Sessions
    ) {
        arms.push(
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
                   NULL AS turn_index,
                   NULL AS role,
                   NULL AS tool_name,
                   NULL AS event_type,
                   COALESCE(s.metadata, s.agent_id) AS match_text,
                   'session' AS kind,
                   NULL AS turn_id,
                   s.id AS session_id
            FROM sessions s
            WHERE LOWER(s.agent_id) LIKE ?1
               OR LOWER(COALESCE(s.metadata, '')) LIKE ?1
            "#,
        );
    }
    if matches!(
        scope,
        SessionSearchScope::All | SessionSearchScope::Messages
    ) {
        arms.push(
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
                   t.branch_depth AS turn_index,
                   tm.role,
                   NULL AS tool_name,
                   NULL AS event_type,
                   tm.content AS match_text,
                   'message' AS kind,
                   t.id AS turn_id,
                   t.session_id
            FROM messages tm
            JOIN turns t ON t.id = tm.turn_id
            JOIN sessions s ON s.id = t.session_id
            WHERE LOWER(tm.content) LIKE ?1
               OR LOWER(tm.role) LIKE ?1
            "#,
        );
    }
    if matches!(
        scope,
        SessionSearchScope::All | SessionSearchScope::ToolExecutions
    ) {
        arms.push(
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
                   t.branch_depth AS turn_index,
                   NULL AS role,
                   tt.tool_name,
                   NULL AS event_type,
                   TRIM(
                       tt.tool_name || ' ' ||
                       COALESCE(tt.args, '') || ' ' ||
                       COALESCE(tt.output, '') || ' ' ||
                       COALESCE(tt.verdict, '')
                   ) AS match_text,
                   'tool_execution' AS kind,
                   t.id AS turn_id,
                   t.session_id
            FROM tool_executions tt
            JOIN turns t ON t.id = tt.turn_id
            JOIN sessions s ON s.id = t.session_id
            WHERE LOWER(tt.tool_name) LIKE ?1
               OR LOWER(COALESCE(tt.args, '')) LIKE ?1
               OR LOWER(COALESCE(tt.output, '')) LIKE ?1
               OR LOWER(COALESCE(tt.verdict, '')) LIKE ?1
            "#,
        );
    }
    if matches!(scope, SessionSearchScope::All | SessionSearchScope::Events) {
        arms.push(
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
                   t.branch_depth AS turn_index,
                   NULL AS role,
                   NULL AS tool_name,
                   e.event_type,
                   e.payload AS match_text,
                   'event' AS kind,
                   e.turn_id,
                   e.session_id
            FROM events e
            JOIN sessions s ON s.id = e.session_id
            LEFT JOIN turns t ON t.id = e.turn_id
            WHERE LOWER(e.event_type) LIKE ?1 OR LOWER(e.payload) LIKE ?1
            "#,
        );
    }

    format!(
        "SELECT * FROM ({}) AS ranked\nORDER BY score DESC, created_at DESC, sort_id DESC\nLIMIT ?3 OFFSET ?4",
        arms.join("\nUNION ALL\n")
    )
}

fn search_hit_kind(kind: &str) -> Result<SessionSearchHitKind> {
    match kind {
        "session" => Ok(SessionSearchHitKind::Session),
        "message" => Ok(SessionSearchHitKind::Message),
        "tool_execution" => Ok(SessionSearchHitKind::ToolExecution),
        "event" => Ok(SessionSearchHitKind::Event),
        other => anyhow::bail!("Unknown session search hit kind '{other}'"),
    }
}

fn search_hit_record(
    kind: SessionSearchHitKind,
    sort_id: i64,
    session_id: i64,
    turn_id: Option<i64>,
) -> String {
    match kind {
        SessionSearchHitKind::Session => format!("session search {session_id}"),
        SessionSearchHitKind::Message => {
            format!("message search turn {}", turn_id.unwrap_or(sort_id))
        }
        SessionSearchHitKind::ToolExecution => {
            format!("tool search turn {}", turn_id.unwrap_or(sort_id))
        }
        SessionSearchHitKind::Event => format!("event search session {session_id}"),
    }
}
