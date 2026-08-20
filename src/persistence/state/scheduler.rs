use anyhow::{Context, Result};

mod runs;

use super::{ScheduledJobRow, StateStore};

pub struct ScheduledJobInsert<'a> {
    pub public_id: uuid::Uuid,
    pub agent_id: &'a str,
    pub job_kind: &'a str,
    pub prompt: Option<&'a str>,
    pub content: Option<&'a str>,
    pub tools: Option<&'a str>,
    pub conflict_policy: Option<&'a str>,
    pub action_name: Option<&'a str>,
    pub action_params: Option<&'a str>,
    pub state_target: Option<&'a str>,
    pub store_target: Option<&'a str>,
    pub next_run_unix_ms: i64,
    pub interval_seconds: Option<u64>,
    pub recurring_pattern: Option<&'a str>,
    pub overlap_policy: &'a str,
    pub work_key: Option<&'a str>,
    pub max_concurrency: Option<u32>,
    pub enabled: bool,
}

pub struct ScheduledJobUpdate<'a> {
    pub id: i64,
    pub agent_id: &'a str,
    pub job_kind: &'a str,
    pub prompt: Option<&'a str>,
    pub content: Option<&'a str>,
    pub tools: Option<&'a str>,
    pub conflict_policy: Option<&'a str>,
    pub action_name: Option<&'a str>,
    pub action_params: Option<&'a str>,
    pub state_target: Option<&'a str>,
    pub store_target: Option<&'a str>,
    pub next_run_unix_ms: i64,
    pub interval_seconds: Option<u64>,
    pub recurring_pattern: Option<&'a str>,
    pub overlap_policy: &'a str,
    pub work_key: Option<&'a str>,
    pub max_concurrency: Option<u32>,
    pub enabled: bool,
}

const SCHEDULED_JOB_SELECT: &str = r#"
SELECT id, public_id, agent_id, job_kind, prompt, content, tools, conflict_policy,
       action_name, action_params, state_target, store_target,
       next_run_unix_ms, interval_seconds, recurring_pattern,
       overlap_policy, work_key, max_concurrency, enabled, running_task_id, active_run_count, pending_rerun,
       last_run_unix_ms, last_status, last_error_code, failure_count, created_at, updated_at
FROM scheduled_jobs
"#;

impl StateStore {
    pub async fn get_scheduled_job_by_public_id(
        &self,
        public_id: uuid::Uuid,
    ) -> Result<Option<ScheduledJobRow>> {
        let conn = self.connect().await?;
        let public_id_bytes = public_id.into_bytes().to_vec();
        let sql = format!("{SCHEDULED_JOB_SELECT} WHERE public_id = ?1 LIMIT 1");
        let mut rows = conn.query(&sql, turso::params![public_id_bytes]).await?;
        next_scheduled_job_row(&mut rows).await
    }

    pub async fn create_scheduled_job(&self, job: ScheduledJobInsert<'_>) -> Result<i64> {
        let conn = self.connect().await?;
        let public_id_bytes = job.public_id.into_bytes().to_vec();
        conn.execute(
            r#"
            INSERT INTO scheduled_jobs (
                public_id, agent_id, job_kind, prompt, content, tools, conflict_policy, action_name, action_params, state_target, store_target,
                next_run_unix_ms, interval_seconds, recurring_pattern,
                overlap_policy, work_key, max_concurrency, enabled, active_run_count, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, 0, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            "#,
            turso::params![
                public_id_bytes,
                job.agent_id,
                job.job_kind,
                job.prompt,
                job.content,
                job.tools,
                job.conflict_policy,
                job.action_name,
                job.action_params,
                job.state_target,
                job.store_target,
                job.next_run_unix_ms,
                job.interval_seconds.map(|v| v as i64),
                job.recurring_pattern,
                job.overlap_policy,
                job.work_key,
                job.max_concurrency.map(|v| v as i64),
                if job.enabled { 1 } else { 0 }
            ],
        )
        .await
        .context("Failed to insert scheduled job")?;
        Ok(conn.last_insert_rowid())
    }

    pub async fn list_scheduled_jobs(&self) -> Result<Vec<ScheduledJobRow>> {
        let conn = self.connect().await?;
        let sql = format!("{SCHEDULED_JOB_SELECT} ORDER BY id ASC");
        let rows = conn.query(&sql, ()).await?;
        collect_scheduled_job_rows(rows).await
    }

    pub async fn list_due_scheduled_jobs(
        &self,
        now_unix_ms: i64,
        limit: usize,
    ) -> Result<Vec<ScheduledJobRow>> {
        let conn = self.connect().await?;
        let sql = format!(
            "{SCHEDULED_JOB_SELECT}
             WHERE enabled = 1 AND next_run_unix_ms <= ?1
             ORDER BY next_run_unix_ms ASC, id ASC
             LIMIT ?2"
        );
        let rows = conn
            .query(&sql, turso::params![now_unix_ms, limit as i64])
            .await?;
        collect_scheduled_job_rows(rows).await
    }

    pub async fn next_scheduled_due_unix_ms(&self) -> Result<Option<i64>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                "SELECT next_run_unix_ms FROM scheduled_jobs WHERE enabled = 1 ORDER BY next_run_unix_ms ASC LIMIT 1",
                (),
            )
            .await?;
        if let Some(row) = rows.next().await? {
            Ok(Some(row.get::<i64>(0)?))
        } else {
            Ok(None)
        }
    }

    pub async fn list_running_scheduled_jobs(&self) -> Result<Vec<ScheduledJobRow>> {
        let conn = self.connect().await?;
        let sql = format!("{SCHEDULED_JOB_SELECT} WHERE active_run_count > 0 ORDER BY id ASC");
        let rows = conn.query(&sql, ()).await?;
        collect_scheduled_job_rows(rows).await
    }

    pub async fn count_running_scheduled_jobs_for_work_key(&self, work_key: &str) -> Result<u32> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                r#"
                SELECT COALESCE(SUM(active_run_count), 0)
                FROM scheduled_jobs
                WHERE active_run_count > 0 AND work_key = ?1
                "#,
                turso::params![work_key],
            )
            .await?;
        if let Some(row) = rows.next().await? {
            super::persisted_u32(
                "scheduled job work-key aggregate",
                "active run count",
                row.get::<i64>(0)?,
            )
        } else {
            Ok(0)
        }
    }

    pub async fn get_scheduled_job_by_id(&self, id: i64) -> Result<Option<ScheduledJobRow>> {
        let conn = self.connect().await?;
        let sql = format!("{SCHEDULED_JOB_SELECT} WHERE id = ?1 LIMIT 1");
        let mut rows = conn.query(&sql, turso::params![id]).await?;
        next_scheduled_job_row(&mut rows).await
    }

    pub async fn mark_scheduled_job_overlap(
        &self,
        id: i64,
        next_run_unix_ms: i64,
        pending_rerun: bool,
    ) -> Result<()> {
        let conn = self.connect().await?;
        conn.execute(
            r#"
            UPDATE scheduled_jobs
            SET next_run_unix_ms = ?2,
                pending_rerun = ?3,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?1
            "#,
            turso::params![id, next_run_unix_ms, if pending_rerun { 1 } else { 0 }],
        )
        .await
        .context("Failed to update scheduled job overlap state")?;
        Ok(())
    }

    pub async fn finalize_scheduled_job_after_runs(
        &self,
        id: i64,
        next_run_unix_ms_override: Option<i64>,
        pending_rerun: bool,
    ) -> Result<()> {
        let conn = self.connect().await?;
        match next_run_unix_ms_override {
            Some(next_run_unix_ms) => {
                conn.execute(
                    r#"
                    UPDATE scheduled_jobs
                    SET next_run_unix_ms = ?2,
                        pending_rerun = ?3,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE id = ?1
                    "#,
                    turso::params![id, next_run_unix_ms, if pending_rerun { 1 } else { 0 }],
                )
                .await?;
            }
            None => {
                conn.execute(
                    r#"
                    UPDATE scheduled_jobs
                    SET pending_rerun = ?2,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE id = ?1
                    "#,
                    turso::params![id, if pending_rerun { 1 } else { 0 }],
                )
                .await?;
            }
        }
        Ok(())
    }

    pub async fn mark_scheduled_job_finished(
        &self,
        id: i64,
        last_status: Option<&str>,
        next_run_unix_ms_override: Option<i64>,
        pending_rerun: bool,
    ) -> Result<()> {
        let conn = self.connect().await?;
        match next_run_unix_ms_override {
            Some(next_run_unix_ms) => {
                conn.execute(
                    r#"
                    UPDATE scheduled_jobs
                    SET running_task_id = NULL,
                        active_run_count = 0,
                        last_status = ?2,
                        last_error_code = NULL,
                        failure_count = 0,
                        next_run_unix_ms = ?3,
                        pending_rerun = ?4,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE id = ?1
                    "#,
                    turso::params![
                        id,
                        last_status,
                        next_run_unix_ms,
                        if pending_rerun { 1 } else { 0 }
                    ],
                )
                .await?;
            }
            None => {
                conn.execute(
                    r#"
                    UPDATE scheduled_jobs
                    SET running_task_id = NULL,
                        active_run_count = 0,
                        last_status = ?2,
                        last_error_code = NULL,
                        failure_count = 0,
                        pending_rerun = ?3,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE id = ?1
                    "#,
                    turso::params![id, last_status, if pending_rerun { 1 } else { 0 }],
                )
                .await?;
            }
        }
        Ok(())
    }

    pub async fn mark_scheduled_job_action_completed(
        &self,
        id: i64,
        next_run_unix_ms: i64,
        enabled: bool,
        last_run_unix_ms: i64,
        last_status: &str,
    ) -> Result<()> {
        let conn = self.connect().await?;
        conn.execute(
            r#"
            UPDATE scheduled_jobs
            SET last_status = ?2,
                last_error_code = NULL,
                failure_count = 0,
                next_run_unix_ms = ?3,
                enabled = ?4,
                last_run_unix_ms = ?5,
                pending_rerun = 0,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?1
            "#,
            turso::params![
                id,
                last_status,
                next_run_unix_ms,
                if enabled { 1 } else { 0 },
                last_run_unix_ms
            ],
        )
        .await
        .context("Failed to record scheduled action completion")?;
        Ok(())
    }

    pub async fn mark_scheduled_job_failed(
        &self,
        id: i64,
        retry_at_unix_ms: i64,
        last_error_code: &str,
        last_status: &str,
    ) -> Result<()> {
        let conn = self.connect().await?;
        conn.execute(
            r#"
            UPDATE scheduled_jobs
            SET last_status = ?2,
                last_error_code = ?3,
                failure_count = failure_count + 1,
                next_run_unix_ms = ?4,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?1
            "#,
            turso::params![id, last_status, last_error_code, retry_at_unix_ms],
        )
        .await
        .context("Failed to record scheduled job submit failure")?;
        Ok(())
    }

    pub async fn set_scheduled_job_enabled(&self, id: i64, enabled: bool) -> Result<()> {
        let conn = self.connect().await?;
        conn.execute(
            r#"
            UPDATE scheduled_jobs
            SET enabled = ?2,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?1
            "#,
            turso::params![id, if enabled { 1 } else { 0 }],
        )
        .await
        .context("Failed to update scheduled job enabled state")?;
        Ok(())
    }

    pub async fn mark_scheduled_job_capacity_blocked(
        &self,
        id: i64,
        next_run_unix_ms: i64,
        pending_rerun: bool,
        last_status: &str,
    ) -> Result<()> {
        let conn = self.connect().await?;
        conn.execute(
            r#"
            UPDATE scheduled_jobs
            SET next_run_unix_ms = ?2,
                pending_rerun = ?3,
                last_status = ?4,
                last_error_code = NULL,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?1
            "#,
            turso::params![
                id,
                next_run_unix_ms,
                if pending_rerun { 1 } else { 0 },
                last_status
            ],
        )
        .await
        .context("Failed to mark scheduled job capacity blocked")?;
        Ok(())
    }

    pub async fn wake_pending_scheduled_jobs_for_work_key(
        &self,
        work_key: &str,
        next_run_unix_ms: i64,
    ) -> Result<()> {
        let conn = self.connect().await?;
        conn.execute(
            r#"
            UPDATE scheduled_jobs
            SET next_run_unix_ms = ?2,
                pending_rerun = 0,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE work_key = ?1
              AND active_run_count = 0
              AND enabled = 1
              AND pending_rerun = 1
            "#,
            turso::params![work_key, next_run_unix_ms],
        )
        .await
        .context("Failed to wake pending scheduled jobs for work key")?;
        Ok(())
    }

    pub async fn update_scheduled_job(&self, job: ScheduledJobUpdate<'_>) -> Result<()> {
        let conn = self.connect().await?;
        conn.execute(
            r#"
            UPDATE scheduled_jobs
            SET agent_id = ?2,
                job_kind = ?3,
                prompt = ?4,
                content = ?5,
                tools = ?6,
                conflict_policy = ?7,
                action_name = ?8,
                action_params = ?9,
                state_target = ?10,
                store_target = ?11,
                next_run_unix_ms = ?12,
                interval_seconds = ?13,
                recurring_pattern = ?14,
                overlap_policy = ?15,
                work_key = ?16,
                max_concurrency = ?17,
                enabled = ?18,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?1
            "#,
            turso::params![
                job.id,
                job.agent_id,
                job.job_kind,
                job.prompt,
                job.content,
                job.tools,
                job.conflict_policy,
                job.action_name,
                job.action_params,
                job.state_target,
                job.store_target,
                job.next_run_unix_ms,
                job.interval_seconds.map(|v| v as i64),
                job.recurring_pattern,
                job.overlap_policy,
                job.work_key,
                job.max_concurrency.map(|v| v as i64),
                if job.enabled { 1 } else { 0 }
            ],
        )
        .await
        .context("Failed to update scheduled job")?;
        Ok(())
    }

    pub async fn delete_scheduled_job(&self, id: i64) -> Result<()> {
        let mut conn = self.connect().await?;
        let tx = conn
            .transaction()
            .await
            .context("Failed to start scheduled job deletion transaction")?;
        tx.execute(
            "DELETE FROM scheduled_job_runs WHERE scheduled_job_id = ?1",
            turso::params![id],
        )
        .await
        .context("Failed to delete scheduled job runs")?;
        tx.execute(
            "DELETE FROM scheduled_jobs WHERE id = ?1",
            turso::params![id],
        )
        .await
        .context("Failed to delete scheduled job")?;
        tx.commit()
            .await
            .context("Failed to commit scheduled job deletion")?;
        Ok(())
    }
}

async fn next_scheduled_job_row(rows: &mut turso::Rows) -> Result<Option<ScheduledJobRow>> {
    rows.next()
        .await?
        .map(|row| map_scheduled_job_row(&row))
        .transpose()
}

async fn collect_scheduled_job_rows(mut rows: turso::Rows) -> Result<Vec<ScheduledJobRow>> {
    let mut result = Vec::new();
    while let Some(row) = rows.next().await? {
        result.push(map_scheduled_job_row(&row)?);
    }
    Ok(result)
}

fn map_scheduled_job_row(row: &turso::Row) -> Result<ScheduledJobRow> {
    let id = row.get::<i64>(0)?;
    let record = format!("scheduled job {id}");
    Ok(ScheduledJobRow {
        id,
        public_id: row.get::<Vec<u8>>(1)?,
        agent_id: row.get::<String>(2)?,
        job_kind: row.get::<String>(3)?,
        prompt: row.get::<Option<String>>(4)?,
        content: row.get::<Option<String>>(5)?,
        tools: row.get::<Option<String>>(6)?,
        conflict_policy: row.get::<Option<String>>(7)?,
        action_name: row.get::<Option<String>>(8)?,
        action_params: row.get::<Option<String>>(9)?,
        state_target: row.get::<Option<String>>(10)?,
        store_target: row.get::<Option<String>>(11)?,
        next_run_unix_ms: row.get::<i64>(12)?,
        interval_seconds: super::persisted_optional_u64(
            &record,
            "interval seconds",
            row.get::<Option<i64>>(13)?,
        )?,
        recurring_pattern: row.get::<Option<String>>(14)?,
        overlap_policy: row.get::<String>(15)?,
        work_key: row.get::<Option<String>>(16)?,
        max_concurrency: super::persisted_optional_u32(
            &record,
            "maximum concurrency",
            row.get::<Option<i64>>(17)?,
        )?,
        enabled: row.get::<i64>(18)? != 0,
        running_task_id: row.get::<Option<String>>(19)?,
        active_run_count: super::persisted_u32(&record, "active run count", row.get::<i64>(20)?)?,
        pending_rerun: row.get::<i64>(21)? != 0,
        last_run_unix_ms: row.get::<Option<i64>>(22)?,
        last_status: row.get::<Option<String>>(23)?,
        last_error_code: row.get::<Option<String>>(24)?,
        failure_count: super::persisted_u64(&record, "failure count", row.get::<i64>(25)?)?,
        created_at: row.get::<String>(26)?,
        updated_at: row.get::<String>(27)?,
    })
}
