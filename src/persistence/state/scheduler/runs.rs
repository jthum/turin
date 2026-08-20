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
            super::super::persisted_u32(
                "scheduled job run aggregate",
                "active run count",
                row.get::<i64>(0)?,
            )
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
        let mut conn = self.connect().await?;
        let tx = conn
            .transaction()
            .await
            .context("Failed to start scheduled job run transaction")?;
        tx.execute(
            r#"
            INSERT INTO scheduled_job_runs (
                scheduled_job_id, task_id, started_unix_ms, updated_at
            ) VALUES (?1, ?2, ?3, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            "#,
            turso::params![id, task_id, last_run_unix_ms],
        )
        .await
        .context("Failed to insert scheduled job run")?;
        let changed = tx
            .execute(
                r#"
            UPDATE scheduled_jobs
            SET running_task_id = (
                    SELECT task_id
                    FROM scheduled_job_runs
                    WHERE scheduled_job_id = ?1 AND finished_unix_ms IS NULL
                    ORDER BY id ASC
                    LIMIT 1
                ),
                next_run_unix_ms = ?2,
                enabled = ?3,
                last_run_unix_ms = ?4,
                active_run_count = (
                    SELECT COUNT(*)
                    FROM scheduled_job_runs
                    WHERE scheduled_job_id = ?1 AND finished_unix_ms IS NULL
                ),
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?1
            "#,
                turso::params![
                    id,
                    next_run_unix_ms,
                    if enabled { 1 } else { 0 },
                    last_run_unix_ms
                ],
            )
            .await
            .context("Failed to mark scheduled job started")?;
        anyhow::ensure!(changed == 1, "Scheduled job '{}' not found", id);
        tx.commit()
            .await
            .context("Failed to commit scheduled job run start")?;
        Ok(())
    }

    pub async fn finish_scheduled_job_run(
        &self,
        scheduled_job_id: i64,
        task_id: &str,
        finished_unix_ms: i64,
        last_status: Option<&str>,
    ) -> Result<()> {
        let mut conn = self.connect().await?;
        let tx = conn
            .transaction()
            .await
            .context("Failed to start scheduled job completion transaction")?;
        let finished = tx
            .execute(
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
        anyhow::ensure!(
            finished == 1,
            "Active scheduled job run '{}' for job '{}' not found",
            task_id,
            scheduled_job_id
        );

        let changed = tx
            .execute(
                r#"
            UPDATE scheduled_jobs
            SET running_task_id = (
                    SELECT task_id
                    FROM scheduled_job_runs
                    WHERE scheduled_job_id = ?1 AND finished_unix_ms IS NULL
                    ORDER BY id ASC
                    LIMIT 1
                ),
                active_run_count = (
                    SELECT COUNT(*)
                    FROM scheduled_job_runs
                    WHERE scheduled_job_id = ?1 AND finished_unix_ms IS NULL
                ),
                last_status = ?2,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?1
            "#,
                turso::params![scheduled_job_id, last_status],
            )
            .await
            .context("Failed to refresh scheduled job after run completion")?;
        anyhow::ensure!(
            changed == 1,
            "Scheduled job '{}' not found while completing run '{}'",
            scheduled_job_id,
            task_id
        );
        tx.commit()
            .await
            .context("Failed to commit scheduled job run completion")?;
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
