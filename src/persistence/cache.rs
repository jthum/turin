//! Session-aware file read cache state and diff tracking.

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use similar::TextDiff;

use super::schema::{
    CacheGlobalStats, CacheReadResult, CacheReadStatus, CacheResetReport, CacheSessionStats,
    CacheStatsReport,
};
use super::state::StateStore;

impl StateStore {
    #[allow(clippy::too_many_arguments)]
    pub async fn cache_read_file(
        &self,
        session_id: i64,
        path: &str,
        content: &str,
        include_content: bool,
        include_previous: bool,
        max_diff_lines: usize,
        token_estimate: bool,
    ) -> Result<CacheReadResult> {
        let conn = self.connect().await?;
        let hash = cache_content_hash(content);
        let content_bytes = content.len() as i64;

        conn.execute(
            "INSERT OR IGNORE INTO file_cache_versions (path, content_hash, content, content_bytes) VALUES (?1, ?2, ?3, ?4)",
            turso::params![path, hash.clone(), content, content_bytes],
        )
        .await
        .with_context(|| format!("Failed to cache file version for '{}'", path))?;

        let mut previous_rows = conn
            .query(
                "SELECT r.content_hash, v.content
                 FROM file_cache_reads r
                 LEFT JOIN file_cache_versions v
                   ON v.path = r.path AND v.content_hash = r.content_hash
                 WHERE r.session_id = ?1 AND r.path = ?2",
                turso::params![session_id, path],
            )
            .await
            .with_context(|| format!("Failed to look up cache read state for '{}'", path))?;

        let previous = if let Some(row) = previous_rows.next().await? {
            Some((row.get::<String>(0)?, row.get::<Option<String>>(1)?))
        } else {
            None
        };

        let previous_hash = previous.as_ref().map(|(prev_hash, _)| prev_hash.clone());
        let previous_content = previous.and_then(|(_, prev_content)| prev_content);

        let status = match previous_hash.as_deref() {
            None => CacheReadStatus::Fresh,
            Some(prev_hash) if prev_hash == hash => CacheReadStatus::Unchanged,
            Some(_) => CacheReadStatus::Changed,
        };

        let should_include_content =
            include_content || !matches!(status, CacheReadStatus::Unchanged);
        let estimated_tokens_saved = if token_estimate && !should_include_content {
            estimate_tokens(content)
        } else {
            0
        };

        let (diff, diff_truncated) =
            if matches!(status, CacheReadStatus::Changed) && max_diff_lines > 0 {
                let old_content = previous_content.as_deref().unwrap_or("");
                build_unified_diff(path, old_content, content, max_diff_lines)
            } else {
                (None, false)
            };

        let read_at = cache_timestamp(&conn).await?;
        conn.execute(
            "INSERT INTO file_cache_reads (session_id, path, content_hash, tokens_saved, last_read_at)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(session_id, path) DO UPDATE SET
               content_hash = excluded.content_hash,
               tokens_saved = file_cache_reads.tokens_saved + excluded.tokens_saved,
               last_read_at = excluded.last_read_at",
            turso::params![session_id, path, hash.clone(), estimated_tokens_saved as i64, read_at.clone()],
        )
        .await
        .with_context(|| format!("Failed to update cache read state for '{}'", path))?;

        Ok(CacheReadResult {
            status,
            path: path.to_string(),
            hash,
            previous_hash,
            content: should_include_content.then(|| content.to_string()),
            previous_content: include_previous.then_some(previous_content).flatten(),
            diff,
            diff_truncated,
            estimated_tokens_saved,
            read_at,
        })
    }

    pub async fn cache_invalidate_file(
        &self,
        path: &str,
        session_id: Option<i64>,
        global: bool,
    ) -> Result<bool> {
        let conn = self.connect().await?;

        let removed_reads = if global {
            conn.execute(
                "DELETE FROM file_cache_reads WHERE path = ?1",
                turso::params![path],
            )
            .await
            .with_context(|| format!("Failed to invalidate cached reads for '{}'", path))?
        } else {
            let session_id =
                session_id.context("session-scoped cache invalidation requires a session id")?;
            conn.execute(
                "DELETE FROM file_cache_reads WHERE session_id = ?1 AND path = ?2",
                turso::params![session_id, path],
            )
            .await
            .with_context(|| format!("Failed to invalidate session cache read for '{}'", path))?
        };

        let removed_versions = if global {
            conn.execute(
                "DELETE FROM file_cache_versions WHERE path = ?1",
                turso::params![path],
            )
            .await
            .with_context(|| format!("Failed to invalidate cached versions for '{}'", path))?
        } else {
            0
        };

        Ok(removed_reads > 0 || removed_versions > 0)
    }

    pub async fn cache_stats(
        &self,
        session_id: Option<i64>,
        include_global: bool,
        include_session: bool,
    ) -> Result<CacheStatsReport> {
        let conn = self.connect().await?;
        let mut report = CacheStatsReport::default();

        if include_global {
            let mut rows = conn
                .query(
                    "SELECT
                        COUNT(DISTINCT path),
                        COUNT(*),
                        COALESCE((SELECT SUM(tokens_saved) FROM file_cache_reads), 0)
                     FROM file_cache_versions",
                    (),
                )
                .await
                .context("Failed to query global cache stats")?;
            if let Some(row) = rows.next().await? {
                report.global = Some(CacheGlobalStats {
                    cached_files: row.get::<i64>(0)? as u64,
                    cached_versions: row.get::<i64>(1)? as u64,
                    tokens_saved: row.get::<i64>(2)? as u64,
                });
            }
        }

        if include_session {
            let session_id = session_id.context("session cache stats require a session id")?;
            let mut rows = conn
                .query(
                    "SELECT s.public_id,
                            COALESCE(COUNT(r.path), 0),
                            COALESCE(SUM(r.tokens_saved), 0)
                     FROM sessions s
                     LEFT JOIN file_cache_reads r ON r.session_id = s.id
                     WHERE s.id = ?1
                     GROUP BY s.id, s.public_id",
                    turso::params![session_id],
                )
                .await
                .context("Failed to query session cache stats")?;
            let row = rows
                .next()
                .await?
                .context("Session cache stats requested for unknown session")?;
            report.session = Some(CacheSessionStats {
                public_id: row.get(0)?,
                files_seen: row.get::<i64>(1)? as u64,
                tokens_saved: row.get::<i64>(2)? as u64,
            });
        }

        Ok(report)
    }

    pub async fn cache_reset(
        &self,
        session_id: Option<i64>,
        global: bool,
        dry_run: bool,
    ) -> Result<CacheResetReport> {
        let conn = self.connect().await?;
        let (removed_reads, removed_versions, scope) = if global {
            let removed_reads = count_table_rows(&conn, CountTable::FileCacheReads, None).await?;
            let removed_versions =
                count_table_rows(&conn, CountTable::FileCacheVersions, None).await?;
            if !dry_run {
                conn.execute("DELETE FROM file_cache_reads", ())
                    .await
                    .context("Failed to reset file cache reads")?;
                conn.execute("DELETE FROM file_cache_versions", ())
                    .await
                    .context("Failed to reset file cache versions")?;
            }
            (removed_reads, removed_versions, "global".to_string())
        } else {
            let session_id =
                session_id.context("session-scoped cache reset requires a session id")?;
            let removed_reads = count_table_rows(
                &conn,
                CountTable::FileCacheReads,
                Some(CountFilter::SessionId(session_id)),
            )
            .await?;
            if !dry_run {
                conn.execute(
                    "DELETE FROM file_cache_reads WHERE session_id = ?1",
                    turso::params![session_id],
                )
                .await
                .context("Failed to reset session cache reads")?;
            }
            (removed_reads, 0, "session".to_string())
        };

        Ok(CacheResetReport {
            scope,
            removed_versions,
            removed_reads,
            reset_stats: true,
            dry_run,
        })
    }
}

fn cache_content_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn estimate_tokens(content: &str) -> u64 {
    content.chars().count().div_ceil(4) as u64
}

fn build_unified_diff(
    path: &str,
    previous_content: &str,
    current_content: &str,
    max_diff_lines: usize,
) -> (Option<String>, bool) {
    let diff = TextDiff::from_lines(previous_content, current_content)
        .unified_diff()
        .header(&format!("a/{path}"), &format!("b/{path}"))
        .to_string();
    truncate_lines(diff, max_diff_lines)
}

fn truncate_lines(diff: String, max_diff_lines: usize) -> (Option<String>, bool) {
    if diff.is_empty() {
        return (None, false);
    }

    let lines = diff.lines().map(ToOwned::to_owned).collect::<Vec<_>>();
    if lines.len() <= max_diff_lines {
        return (Some(diff), false);
    }

    let truncated = lines
        .into_iter()
        .take(max_diff_lines)
        .collect::<Vec<_>>()
        .join("\n");
    (Some(format!("{truncated}\n")), true)
}

async fn cache_timestamp(conn: &turso::Connection) -> Result<String> {
    let mut rows = conn
        .query("SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')", ())
        .await
        .context("Failed to fetch cache timestamp")?;
    rows.next()
        .await?
        .map(|row| row.get::<String>(0))
        .transpose()?
        .context("Cache timestamp query returned no row")
}

#[derive(Clone, Copy)]
enum CountTable {
    FileCacheReads,
    FileCacheVersions,
}

impl CountTable {
    fn as_sql_identifier(self) -> &'static str {
        match self {
            Self::FileCacheReads => "file_cache_reads",
            Self::FileCacheVersions => "file_cache_versions",
        }
    }
}

#[derive(Clone, Copy)]
enum CountFilter {
    SessionId(i64),
}

async fn count_table_rows(
    conn: &turso::Connection,
    table: CountTable,
    filter: Option<CountFilter>,
) -> Result<u64> {
    let table_name = table.as_sql_identifier();
    let mut rows = if let Some(filter) = filter {
        let (column_name, value) = match filter {
            CountFilter::SessionId(value) => ("session_id", value),
        };
        let sql = format!("SELECT COUNT(*) FROM {table_name} WHERE {column_name} = ?1");
        conn.query(&sql, turso::params![value])
            .await
            .with_context(|| format!("Failed to count rows in '{table_name}'"))?
    } else {
        let sql = format!("SELECT COUNT(*) FROM {table_name}");
        conn.query(&sql, ())
            .await
            .with_context(|| format!("Failed to count rows in '{table_name}'"))?
    };
    let row = rows.next().await?.context("Count query returned no row")?;
    Ok(row.get::<i64>(0)? as u64)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[tokio::test]
    async fn file_cache_tracks_fresh_unchanged_changed_and_reset() {
        let store = StateStore::open_memory().await.expect("state store");
        let session_id = store
            .create_session(uuid::Uuid::now_v7(), "default", None)
            .await
            .expect("session row");

        let fresh = store
            .cache_read_file(
                session_id,
                "notes.txt",
                "alpha\nbeta\n",
                false,
                false,
                200,
                true,
            )
            .await
            .expect("fresh cache read");
        assert_eq!(fresh.status, CacheReadStatus::Fresh);
        assert_eq!(fresh.content.as_deref(), Some("alpha\nbeta\n"));
        assert_eq!(fresh.estimated_tokens_saved, 0);

        let unchanged = store
            .cache_read_file(
                session_id,
                "notes.txt",
                "alpha\nbeta\n",
                false,
                false,
                200,
                true,
            )
            .await
            .expect("unchanged cache read");
        assert_eq!(unchanged.status, CacheReadStatus::Unchanged);
        assert!(unchanged.content.is_none());
        assert!(unchanged.estimated_tokens_saved > 0);

        let changed = store
            .cache_read_file(
                session_id,
                "notes.txt",
                "alpha\ngamma\n",
                false,
                true,
                200,
                true,
            )
            .await
            .expect("changed cache read");
        assert_eq!(changed.status, CacheReadStatus::Changed);
        assert_eq!(changed.content.as_deref(), Some("alpha\ngamma\n"));
        assert_eq!(changed.previous_content.as_deref(), Some("alpha\nbeta\n"));
        assert!(changed.previous_hash.is_some());
        assert!(
            changed
                .diff
                .as_deref()
                .is_some_and(|diff| diff.contains("@@"))
        );

        let stats = store
            .cache_stats(Some(session_id), true, true)
            .await
            .expect("cache stats");
        assert_eq!(stats.global.expect("global stats").cached_versions, 2);
        let session = stats.session.expect("session stats");
        assert_eq!(session.files_seen, 1);
        assert!(session.tokens_saved > 0);

        let invalidated = store
            .cache_invalidate_file("notes.txt", Some(session_id), false)
            .await
            .expect("invalidate");
        assert!(invalidated);

        let fresh_again = store
            .cache_read_file(
                session_id,
                "notes.txt",
                "alpha\ngamma\n",
                false,
                false,
                200,
                true,
            )
            .await
            .expect("fresh again");
        assert_eq!(fresh_again.status, CacheReadStatus::Fresh);

        let dry_run = store
            .cache_reset(Some(session_id), false, true)
            .await
            .expect("dry-run reset");
        assert_eq!(dry_run.scope, "session");
        assert_eq!(dry_run.removed_versions, 0);
        assert!(dry_run.removed_reads >= 1);
        assert!(dry_run.dry_run);

        let reset = store
            .cache_reset(Some(session_id), false, false)
            .await
            .expect("session reset");
        assert!(reset.removed_reads >= 1);
        assert_eq!(reset.removed_versions, 0);

        let global_stats = store
            .cache_stats(Some(session_id), true, true)
            .await
            .expect("stats after session reset");
        assert_eq!(global_stats.global.expect("global stats").cached_files, 1);
        assert_eq!(global_stats.session.expect("session stats").files_seen, 0);

        let global_reset = store
            .cache_reset(None, true, false)
            .await
            .expect("global reset");
        assert!(global_reset.removed_versions >= 1);
        assert!(global_reset.removed_reads == 0);

        let empty_stats = store
            .cache_stats(Some(session_id), true, true)
            .await
            .expect("empty stats");
        assert_eq!(empty_stats.global.expect("global stats").cached_versions, 0);
        assert_eq!(empty_stats.session.expect("session stats").files_seen, 0);

        let still_searches = store
            .insert_memory(
                "session",
                &session_id.to_string(),
                "cache reset should not affect memories",
                None,
                None,
                None,
                &json!({}),
            )
            .await
            .expect("memory insert");
        assert!(!still_searches.public_id.is_empty());
    }
}
