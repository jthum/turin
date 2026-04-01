use anyhow::{Context, Result};
use turin_daemon_protocol::{SessionSearchHitKind, SessionSearchScope};

use super::{SessionRow, SessionSearchRow, StateStore, update_session_title_metadata};

impl StateStore {
    pub async fn create_session(
        &self,
        public_id: uuid::Uuid,
        agent_id: &str,
        metadata: Option<&str>,
    ) -> Result<i64> {
        let conn = self.connect().await?;
        let public_id_bytes = public_id.into_bytes().to_vec();

        conn.execute(
            "INSERT INTO sessions (public_id, agent_id, metadata) VALUES (?1, ?2, ?3)",
            turso::params![public_id_bytes, agent_id, metadata],
        )
        .await
        .context("Failed to insert into sessions table")?;

        let session_id = conn.last_insert_rowid();
        self.initialize_main_branch(session_id).await?;
        Ok(session_id)
    }

    pub async fn get_session_by_public_id(&self, public_id: uuid::Uuid) -> Result<Option<i64>> {
        let conn = self.connect().await?;
        let public_id_bytes = public_id.into_bytes().to_vec();

        let mut rows = conn
            .query(
                "SELECT id FROM sessions WHERE public_id = ?1",
                turso::params![public_id_bytes],
            )
            .await?;

        if let Some(row) = rows.next().await? {
            Ok(Some(row.get(0)?))
        } else {
            Ok(None)
        }
    }

    pub async fn get_session_row(&self, session_id: i64) -> Result<Option<SessionRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT id, public_id, agent_id, metadata, active_branch_head_id, created_at FROM sessions WHERE id = ?1",
                [session_id],
            )
            .await?;

        if let Some(row) = rows.next().await? {
            Ok(Some(SessionRow {
                id: row.get::<i64>(0)?,
                public_id: row.get::<Vec<u8>>(1)?,
                agent_id: row.get::<String>(2)?,
                metadata: row.get::<Option<String>>(3)?,
                active_branch_head_id: row.get::<Option<i64>>(4)?,
                created_at: row.get::<String>(5)?,
            }))
        } else {
            Ok(None)
        }
    }

    pub async fn get_session_row_by_public_id(
        &self,
        public_id: uuid::Uuid,
    ) -> Result<Option<SessionRow>> {
        let conn = self.connect().await?;
        let public_id_bytes = public_id.into_bytes().to_vec();
        let mut rows = conn
            .query(
                "SELECT id, public_id, agent_id, metadata, active_branch_head_id, created_at FROM sessions WHERE public_id = ?1",
                turso::params![public_id_bytes],
            )
            .await?;

        if let Some(row) = rows.next().await? {
            Ok(Some(SessionRow {
                id: row.get::<i64>(0)?,
                public_id: row.get::<Vec<u8>>(1)?,
                agent_id: row.get::<String>(2)?,
                metadata: row.get::<Option<String>>(3)?,
                active_branch_head_id: row.get::<Option<i64>>(4)?,
                created_at: row.get::<String>(5)?,
            }))
        } else {
            Ok(None)
        }
    }

    pub async fn update_session_title(
        &self,
        public_id: uuid::Uuid,
        title: Option<&str>,
    ) -> Result<Option<SessionRow>> {
        let Some(mut row) = self.get_session_row_by_public_id(public_id).await? else {
            return Ok(None);
        };

        let metadata = update_session_title_metadata(row.metadata.as_deref(), title)?;
        let conn = self.connect().await?;
        let public_id_bytes = public_id.into_bytes().to_vec();
        conn.execute(
            "UPDATE sessions SET metadata = ?1 WHERE public_id = ?2",
            turso::params![metadata.clone(), public_id_bytes],
        )
        .await
        .context("Failed to update session metadata title")?;
        row.metadata = metadata;
        Ok(Some(row))
    }

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

        let mut clauses = Vec::new();
        if matches!(
            scope,
            SessionSearchScope::All | SessionSearchScope::Sessions
        ) {
            clauses.push(
                r#"
                SELECT 'session' AS kind,
                       CASE
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
                       COALESCE(s.metadata, s.agent_id) AS match_text
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
            clauses.push(
                r#"
                SELECT 'message' AS kind,
                       CASE
                           WHEN LOWER(m.role) = ?2 THEN 860
                           WHEN instr(LOWER(m.content), ?2) = 1 THEN 820
                           ELSE 740
                       END AS score,
                       m.id AS sort_id,
                       s.public_id,
                       s.agent_id,
                       s.metadata,
                       m.created_at,
                       m.turn_index,
                       m.role,
                       NULL AS tool_name,
                       NULL AS event_type,
                       m.content AS match_text
                FROM messages m
                JOIN sessions s ON s.id = m.session_id
                WHERE LOWER(m.content) LIKE ?1
                   OR LOWER(m.role) LIKE ?1
                "#,
            );
        }
        if matches!(
            scope,
            SessionSearchScope::All | SessionSearchScope::ToolExecutions
        ) {
            clauses.push(
                r#"
                SELECT 'tool_execution' AS kind,
                       CASE
                           WHEN LOWER(t.tool_name) = ?2 THEN 900
                           WHEN LOWER(t.tool_name) LIKE (?2 || '%') THEN 860
                           WHEN instr(LOWER(COALESCE(t.args, '')), ?2) = 1 THEN 760
                           WHEN instr(LOWER(COALESCE(t.output, '')), ?2) = 1 THEN 740
                           ELSE 700
                       END AS score,
                       t.id AS sort_id,
                       s.public_id,
                       s.agent_id,
                       s.metadata,
                       t.created_at,
                       t.turn_index,
                       NULL AS role,
                       t.tool_name,
                       NULL AS event_type,
                       TRIM(
                           t.tool_name || ' ' ||
                           COALESCE(t.args, '') || ' ' ||
                           COALESCE(t.output, '') || ' ' ||
                           COALESCE(t.verdict, '')
                       ) AS match_text
                FROM tool_executions t
                JOIN sessions s ON s.id = t.session_id
                WHERE LOWER(t.tool_name) LIKE ?1
                   OR LOWER(COALESCE(t.args, '')) LIKE ?1
                   OR LOWER(COALESCE(t.output, '')) LIKE ?1
                   OR LOWER(COALESCE(t.verdict, '')) LIKE ?1
                "#,
            );
        }
        if matches!(scope, SessionSearchScope::All | SessionSearchScope::Events) {
            clauses.push(
                r#"
                SELECT 'event' AS kind,
                       CASE
                           WHEN LOWER(e.event_type) = ?2 THEN 820
                           WHEN LOWER(e.event_type) LIKE (?2 || '%') THEN 780
                           ELSE 680
                       END AS score,
                       e.id AS sort_id,
                       s.public_id,
                       s.agent_id,
                       s.metadata,
                       e.created_at,
                       NULL AS turn_index,
                       NULL AS role,
                       NULL AS tool_name,
                       e.event_type,
                       e.payload AS match_text
                FROM events e
                JOIN sessions s ON s.id = e.session_id
                WHERE LOWER(e.event_type) LIKE ?1
                   OR LOWER(e.payload) LIKE ?1
                "#,
            );
        }

        if clauses.is_empty() {
            return Ok(Vec::new());
        }

        let sql = format!(
            r#"
            SELECT *
            FROM (
                {}
            ) search_hits
            ORDER BY score DESC, created_at DESC, sort_id DESC
            LIMIT ?3 OFFSET ?4
            "#,
            clauses.join("\nUNION ALL\n")
        );

        let conn = self.connect().await?;
        let needle = format!("%{normalized}%");
        let mut rows = conn
            .query(
                &sql,
                turso::params![needle, normalized, limit as i64, offset as i64],
            )
            .await
            .context("Failed to search persisted session history")?;

        let mut hits = Vec::new();
        while let Some(row) = rows.next().await? {
            let kind = match row.get::<String>(0)?.as_str() {
                "session" => SessionSearchHitKind::Session,
                "message" => SessionSearchHitKind::Message,
                "tool_execution" => SessionSearchHitKind::ToolExecution,
                "event" => SessionSearchHitKind::Event,
                other => anyhow::bail!("Unexpected persisted search hit kind '{}'", other),
            };
            hits.push(SessionSearchRow {
                kind,
                score: row.get::<i64>(1)?,
                public_id: row.get::<Vec<u8>>(3)?,
                agent_id: row.get::<String>(4)?,
                metadata: row.get::<Option<String>>(5)?,
                created_at: row.get::<String>(6)?,
                turn_index: row.get::<Option<i64>>(7)?.map(|value| value as u32),
                role: row.get::<Option<String>>(8)?,
                tool_name: row.get::<Option<String>>(9)?,
                event_type: row.get::<Option<String>>(10)?,
                match_text: row.get::<String>(11)?,
            });
        }

        Ok(hits)
    }
}
