use anyhow::{Context, Result};
use turin_daemon_protocol::{SessionSearchHitKind, SessionSearchScope};

use crate::persistence::state::{SessionSearchRow, StateStore};

const ACTIVE_PATH_CTE: &str = r#"
WITH RECURSIVE active_path(session_id, turn_id) AS (
    SELECT s.id, bh.head_turn_id
    FROM sessions s
    JOIN branch_heads bh ON bh.id = s.active_branch_head_id
    WHERE bh.head_turn_id IS NOT NULL
    UNION ALL
    SELECT t.session_id, t.parent_turn_id
    FROM turns t
    JOIN active_path a ON t.id = a.turn_id
    WHERE t.parent_turn_id IS NOT NULL
      AND t.session_id = a.session_id
)
"#;

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

        self.query_ranked_session_search_hits(&normalized, scope, limit, offset)
            .await
    }
}

impl StateStore {
    async fn query_ranked_session_search_hits(
        &self,
        normalized: &str,
        scope: SessionSearchScope,
        limit: usize,
        offset: usize,
    ) -> Result<Vec<SessionSearchRow>> {
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
            hits.push(SessionSearchRow {
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
            });
        }
        Ok(hits)
    }
}

fn ranked_session_search_sql(scope: SessionSearchScope) -> String {
    let mut arms = Vec::new();
    if matches!(scope, SessionSearchScope::All | SessionSearchScope::Sessions) {
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
    if matches!(scope, SessionSearchScope::All | SessionSearchScope::Messages) {
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
            JOIN active_path a ON a.turn_id = tm.turn_id AND a.session_id = t.session_id
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
            JOIN active_path a ON a.turn_id = tt.turn_id AND a.session_id = t.session_id
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
            LEFT JOIN active_path a ON a.turn_id = e.turn_id AND a.session_id = e.session_id
            WHERE (LOWER(e.event_type) LIKE ?1 OR LOWER(e.payload) LIKE ?1)
              AND (e.turn_id IS NULL OR a.turn_id IS NOT NULL)
            "#,
        );
    }

    let body = arms.join("\nUNION ALL\n");
    let ordered = format!("{body}\nORDER BY score DESC, created_at DESC, sort_id DESC\nLIMIT ?3 OFFSET ?4");
    if matches!(scope, SessionSearchScope::Sessions) {
        ordered
    } else {
        format!("{ACTIVE_PATH_CTE} {ordered}")
    }
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
        SessionSearchHitKind::Message => format!("message search turn {}", turn_id.unwrap_or(sort_id)),
        SessionSearchHitKind::ToolExecution => {
            format!("tool search turn {}", turn_id.unwrap_or(sort_id))
        }
        SessionSearchHitKind::Event => format!("event search session {session_id}"),
    }
}
