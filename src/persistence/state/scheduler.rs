use anyhow::{Context, Result};

use super::{ScheduledJobRow, StateStore};

impl StateStore {
    pub async fn create_scheduled_job(
        &self,
        public_id: uuid::Uuid,
        agent_id: &str,
        prompt: &str,
        state_target: Option<&str>,
        store_target: Option<&str>,
        next_run_unix_ms: i64,
        interval_seconds: Option<u64>,
        overlap_policy: &str,
        enabled: bool,
    ) -> Result<i64> {
        let conn = self.connect().await?;
        let public_id_bytes = public_id.into_bytes().to_vec();
        conn.execute(
            r#"
            INSERT INTO scheduled_jobs (
                public_id, agent_id, prompt, state_target, store_target,
                next_run_unix_ms, interval_seconds,
                overlap_policy, enabled, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            "#,
            turso::params![
                public_id_bytes,
                agent_id,
                prompt,
                state_target,
                store_target,
                next_run_unix_ms,
                interval_seconds.map(|v| v as i64),
                overlap_policy,
                if enabled { 1 } else { 0 }
            ],
        )
        .await
        .context("Failed to insert scheduled job")?;
        Ok(conn.last_insert_rowid())
    }

    pub async fn list_scheduled_jobs(&self) -> Result<Vec<ScheduledJobRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                r#"
                SELECT id, public_id, agent_id, prompt, state_target, store_target,
                       next_run_unix_ms, interval_seconds,
                       overlap_policy, enabled, running_task_id, pending_rerun,
                       last_run_unix_ms, last_status, created_at, updated_at
                FROM scheduled_jobs
                ORDER BY id ASC
                "#,
                (),
            )
            .await?;
        let mut result = Vec::new();
        while let Some(row) = rows.next().await? {
            result.push(ScheduledJobRow {
                id: row.get::<i64>(0)?,
                public_id: row.get::<Vec<u8>>(1)?,
                agent_id: row.get::<String>(2)?,
                prompt: row.get::<String>(3)?,
                state_target: row.get::<Option<String>>(4)?,
                store_target: row.get::<Option<String>>(5)?,
                next_run_unix_ms: row.get::<i64>(6)?,
                interval_seconds: row.get::<Option<i64>>(7)?.map(|v| v as u64),
                overlap_policy: row.get::<String>(8)?,
                enabled: row.get::<i64>(9)? != 0,
                running_task_id: row.get::<Option<String>>(10)?,
                pending_rerun: row.get::<i64>(11)? != 0,
                last_run_unix_ms: row.get::<Option<i64>>(12)?,
                last_status: row.get::<Option<String>>(13)?,
                created_at: row.get::<String>(14)?,
                updated_at: row.get::<String>(15)?,
            });
        }
        Ok(result)
    }

    pub async fn list_due_scheduled_jobs(
        &self,
        now_unix_ms: i64,
        limit: usize,
    ) -> Result<Vec<ScheduledJobRow>> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                r#"
                SELECT id, public_id, agent_id, prompt, state_target, store_target,
                       next_run_unix_ms, interval_seconds,
                       overlap_policy, enabled, running_task_id, pending_rerun,
                       last_run_unix_ms, last_status, created_at, updated_at
                FROM scheduled_jobs
                WHERE enabled = 1 AND next_run_unix_ms <= ?1
                ORDER BY next_run_unix_ms ASC, id ASC
                LIMIT ?2
                "#,
                turso::params![now_unix_ms, limit as i64],
            )
            .await?;
        let mut result = Vec::new();
        while let Some(row) = rows.next().await? {
            result.push(ScheduledJobRow {
                id: row.get::<i64>(0)?,
                public_id: row.get::<Vec<u8>>(1)?,
                agent_id: row.get::<String>(2)?,
                prompt: row.get::<String>(3)?,
                state_target: row.get::<Option<String>>(4)?,
                store_target: row.get::<Option<String>>(5)?,
                next_run_unix_ms: row.get::<i64>(6)?,
                interval_seconds: row.get::<Option<i64>>(7)?.map(|v| v as u64),
                overlap_policy: row.get::<String>(8)?,
                enabled: row.get::<i64>(9)? != 0,
                running_task_id: row.get::<Option<String>>(10)?,
                pending_rerun: row.get::<i64>(11)? != 0,
                last_run_unix_ms: row.get::<Option<i64>>(12)?,
                last_status: row.get::<Option<String>>(13)?,
                created_at: row.get::<String>(14)?,
                updated_at: row.get::<String>(15)?,
            });
        }
        Ok(result)
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
        let mut rows = conn
            .query(
                r#"
                SELECT id, public_id, agent_id, prompt, state_target, store_target,
                       next_run_unix_ms, interval_seconds,
                       overlap_policy, enabled, running_task_id, pending_rerun,
                       last_run_unix_ms, last_status, created_at, updated_at
                FROM scheduled_jobs
                WHERE running_task_id IS NOT NULL
                ORDER BY id ASC
                "#,
                (),
            )
            .await?;
        let mut result = Vec::new();
        while let Some(row) = rows.next().await? {
            result.push(ScheduledJobRow {
                id: row.get::<i64>(0)?,
                public_id: row.get::<Vec<u8>>(1)?,
                agent_id: row.get::<String>(2)?,
                prompt: row.get::<String>(3)?,
                state_target: row.get::<Option<String>>(4)?,
                store_target: row.get::<Option<String>>(5)?,
                next_run_unix_ms: row.get::<i64>(6)?,
                interval_seconds: row.get::<Option<i64>>(7)?.map(|v| v as u64),
                overlap_policy: row.get::<String>(8)?,
                enabled: row.get::<i64>(9)? != 0,
                running_task_id: row.get::<Option<String>>(10)?,
                pending_rerun: row.get::<i64>(11)? != 0,
                last_run_unix_ms: row.get::<Option<i64>>(12)?,
                last_status: row.get::<Option<String>>(13)?,
                created_at: row.get::<String>(14)?,
                updated_at: row.get::<String>(15)?,
            });
        }
        Ok(result)
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
            UPDATE scheduled_jobs
            SET running_task_id = ?2,
                next_run_unix_ms = ?3,
                enabled = ?4,
                last_run_unix_ms = ?5,
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
                        last_status = ?2,
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
                        last_status = ?2,
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

    pub async fn mark_scheduled_job_submit_failed(
        &self,
        id: i64,
        retry_at_unix_ms: i64,
        last_status: &str,
    ) -> Result<()> {
        let conn = self.connect().await?;
        conn.execute(
            r#"
            UPDATE scheduled_jobs
            SET running_task_id = NULL,
                last_status = ?2,
                next_run_unix_ms = ?3,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?1
            "#,
            turso::params![id, last_status, retry_at_unix_ms],
        )
        .await
        .context("Failed to record scheduled job submit failure")?;
        Ok(())
    }
}
