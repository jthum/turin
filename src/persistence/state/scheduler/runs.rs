use anyhow::{Context, Result};

use super::super::{ScheduledJobRunRow, StateStore};

const SCHEDULED_JOB_RUN_SELECT: &str = r#"
SELECT id, scheduled_job_id, task_id, started_unix_ms, finished_unix_ms, last_status, created_at, updated_at
FROM scheduled_job_runs
"#;

impl StateStore {
    pub async fn list_active_scheduled_job_runs(&self) -> Result<Vec<ScheduledJobRunRow>> {
        let conn = self.connect().await?;
        let sql = format!(
            "{SCHEDULED_JOB_RUN_SELECT}
             WHERE finished_unix_ms IS NULL
             ORDER BY id ASC"
        );
        let mut rows = conn.query(&sql, ()).await?;
        let mut result = Vec::new();
        while let Some(row) = rows.next().await? {
            result.push(map_scheduled_job_run_row(&row)?);
        }
        Ok(result)
    }

    pub async fn list_scheduled_job_runs(
        &self,
        scheduled_job_id: i64,
        active_only: bool,
        limit: Option<u32>,
    ) -> Result<Vec<ScheduledJobRunRow>> {
        let conn = self.connect().await?;
        let mut query = format!("{SCHEDULED_JOB_RUN_SELECT} WHERE scheduled_job_id = ?1");
        if active_only {
            query.push_str(" AND finished_unix_ms IS NULL");
        }
        query.push_str(" ORDER BY id DESC");
        let mut rows = match limit {
            Some(limit) => {
                query.push_str(" LIMIT ?2");
                conn.query(query, turso::params![scheduled_job_id, limit as i64])
                    .await?
            }
            None => conn.query(query, turso::params![scheduled_job_id]).await?,
        };
        let mut result = Vec::new();
        while let Some(row) = rows.next().await? {
            result.push(map_scheduled_job_run_row(&row)?);
        }
        Ok(result)
    }

    pub async fn count_active_scheduled_job_runs(&self, scheduled_job_id: i64) -> Result<u32> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                r#"
                SELECT COUNT(*)
                FROM scheduled_job_runs
                WHERE scheduled_job_id = ?1 AND finished_unix_ms IS NULL
                "#,
                turso::params![scheduled_job_id],
            )
            .await?;
        if let Some(row) = rows.next().await? {
            Ok(row.get::<i64>(0)? as u32)
        } else {
            Ok(0)
        }
    }

    pub async fn mark_scheduled_job_started(
        &self,
        id: i64,
        task_id: &str,
        next_run_unix_ms: i64,
        enabled: bool,
        last_run_unix_ms: i64,
    ) -> Result<()> {
        let conn = self.connect().await?;
        conn.execute(
            r#"
            INSERT INTO scheduled_job_runs (
                scheduled_job_id, task_id, started_unix_ms, updated_at
            ) VALUES (?1, ?2, ?3, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            "#,
            turso::params![id, task_id, last_run_unix_ms],
        )
        .await
        .context("Failed to insert scheduled job run")?;
        conn.execute(
            r#"
            UPDATE scheduled_jobs
            SET running_task_id = COALESCE(running_task_id, ?2),
                next_run_unix_ms = ?3,
                enabled = ?4,
                last_run_unix_ms = ?5,
                active_run_count = active_run_count + 1,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?1
            "#,
            turso::params![
                id,
                task_id,
                next_run_unix_ms,
                if enabled { 1 } else { 0 },
                last_run_unix_ms
            ],
        )
        .await
        .context("Failed to mark scheduled job started")?;
        Ok(())
    }

    pub async fn finish_scheduled_job_run(
        &self,
        scheduled_job_id: i64,
        task_id: &str,
        finished_unix_ms: i64,
        last_status: Option<&str>,
    ) -> Result<()> {
        let conn = self.connect().await?;
        conn.execute(
            r#"
            UPDATE scheduled_job_runs
            SET finished_unix_ms = ?3,
                last_status = ?4,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE scheduled_job_id = ?1
              AND task_id = ?2
              AND finished_unix_ms IS NULL
            "#,
            turso::params![scheduled_job_id, task_id, finished_unix_ms, last_status],
        )
        .await
        .context("Failed to finish scheduled job run")?;

        let mut rows = conn
            .query(
                r#"
                SELECT task_id
                FROM scheduled_job_runs
                WHERE scheduled_job_id = ?1 AND finished_unix_ms IS NULL
                ORDER BY id ASC
                LIMIT 1
                "#,
                turso::params![scheduled_job_id],
            )
            .await?;
        let running_task_id = if let Some(row) = rows.next().await? {
            Some(row.get::<String>(0)?)
        } else {
            None
        };
        let remaining = self
            .count_active_scheduled_job_runs(scheduled_job_id)
            .await?;

        conn.execute(
            r#"
            UPDATE scheduled_jobs
            SET running_task_id = ?2,
                active_run_count = ?3,
                last_status = ?4,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?1
            "#,
            turso::params![
                scheduled_job_id,
                running_task_id,
                remaining as i64,
                last_status
            ],
        )
        .await
        .context("Failed to refresh scheduled job after run completion")?;
        Ok(())
    }
}

fn map_scheduled_job_run_row(row: &turso::Row) -> Result<ScheduledJobRunRow> {
    Ok(ScheduledJobRunRow {
        id: row.get::<i64>(0)?,
        scheduled_job_id: row.get::<i64>(1)?,
        task_id: row.get::<String>(2)?,
        started_unix_ms: row.get::<i64>(3)?,
        finished_unix_ms: row.get::<Option<i64>>(4)?,
        last_status: row.get::<Option<String>>(5)?,
        created_at: row.get::<String>(6)?,
        updated_at: row.get::<String>(7)?,
    })
}
