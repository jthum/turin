use anyhow::{Context, Result};

use super::{ScheduledJobRow, StateStore};

impl StateStore {
    pub async fn get_scheduled_job_by_public_id(
        &self,
        public_id: uuid::Uuid,
    ) -> Result<Option<ScheduledJobRow>> {
        let conn = self.connect().await?;
        let public_id_bytes = public_id.into_bytes().to_vec();
        let mut rows = conn
            .query(
                r#"
                SELECT id, public_id, agent_id, job_kind, prompt, content, tools, conflict_policy,
                       action_name, action_params, state_target, store_target,
                       next_run_unix_ms, interval_seconds, recurring_pattern,
                       overlap_policy, work_key, max_concurrency, enabled, running_task_id, pending_rerun,
                       last_run_unix_ms, last_status, created_at, updated_at
                FROM scheduled_jobs
                WHERE public_id = ?1
                LIMIT 1
                "#,
                turso::params![public_id_bytes],
            )
            .await?;
        if let Some(row) = rows.next().await? {
            return Ok(Some(ScheduledJobRow {
                id: row.get::<i64>(0)?,
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
                interval_seconds: row.get::<Option<i64>>(13)?.map(|v| v as u64),
                recurring_pattern: row.get::<Option<String>>(14)?,
                overlap_policy: row.get::<String>(15)?,
                work_key: row.get::<Option<String>>(16)?,
                max_concurrency: row.get::<Option<i64>>(17)?.map(|v| v as u32),
                enabled: row.get::<i64>(18)? != 0,
                running_task_id: row.get::<Option<String>>(19)?,
                pending_rerun: row.get::<i64>(20)? != 0,
                last_run_unix_ms: row.get::<Option<i64>>(21)?,
                last_status: row.get::<Option<String>>(22)?,
                created_at: row.get::<String>(23)?,
                updated_at: row.get::<String>(24)?,
            }));
        }
        Ok(None)
    }

    pub async fn create_scheduled_job(
        &self,
        public_id: uuid::Uuid,
        agent_id: &str,
        job_kind: &str,
        prompt: Option<&str>,
        content: Option<&str>,
        tools: Option<&str>,
        conflict_policy: Option<&str>,
        action_name: Option<&str>,
        action_params: Option<&str>,
        state_target: Option<&str>,
        store_target: Option<&str>,
        next_run_unix_ms: i64,
        interval_seconds: Option<u64>,
        recurring_pattern: Option<&str>,
        overlap_policy: &str,
        work_key: Option<&str>,
        max_concurrency: Option<u32>,
        enabled: bool,
    ) -> Result<i64> {
        let conn = self.connect().await?;
        let public_id_bytes = public_id.into_bytes().to_vec();
        conn.execute(
            r#"
            INSERT INTO scheduled_jobs (
                public_id, agent_id, job_kind, prompt, content, tools, conflict_policy, action_name, action_params, state_target, store_target,
                next_run_unix_ms, interval_seconds, recurring_pattern,
                overlap_policy, work_key, max_concurrency, enabled, updated_at
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
            "#,
            turso::params![
                public_id_bytes,
                agent_id,
                job_kind,
                prompt,
                content,
                tools,
                conflict_policy,
                action_name,
                action_params,
                state_target,
                store_target,
                next_run_unix_ms,
                interval_seconds.map(|v| v as i64),
                recurring_pattern,
                overlap_policy,
                work_key,
                max_concurrency.map(|v| v as i64),
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
                SELECT id, public_id, agent_id, job_kind, prompt, content, tools, conflict_policy,
                       action_name, action_params, state_target, store_target,
                       next_run_unix_ms, interval_seconds, recurring_pattern,
                       overlap_policy, work_key, max_concurrency, enabled, running_task_id, pending_rerun,
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
                interval_seconds: row.get::<Option<i64>>(13)?.map(|v| v as u64),
                recurring_pattern: row.get::<Option<String>>(14)?,
                overlap_policy: row.get::<String>(15)?,
                work_key: row.get::<Option<String>>(16)?,
                max_concurrency: row.get::<Option<i64>>(17)?.map(|v| v as u32),
                enabled: row.get::<i64>(18)? != 0,
                running_task_id: row.get::<Option<String>>(19)?,
                pending_rerun: row.get::<i64>(20)? != 0,
                last_run_unix_ms: row.get::<Option<i64>>(21)?,
                last_status: row.get::<Option<String>>(22)?,
                created_at: row.get::<String>(23)?,
                updated_at: row.get::<String>(24)?,
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
                SELECT id, public_id, agent_id, job_kind, prompt, content, tools, conflict_policy,
                       action_name, action_params, state_target, store_target,
                       next_run_unix_ms, interval_seconds, recurring_pattern,
                       overlap_policy, work_key, max_concurrency, enabled, running_task_id, pending_rerun,
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
                interval_seconds: row.get::<Option<i64>>(13)?.map(|v| v as u64),
                recurring_pattern: row.get::<Option<String>>(14)?,
                overlap_policy: row.get::<String>(15)?,
                work_key: row.get::<Option<String>>(16)?,
                max_concurrency: row.get::<Option<i64>>(17)?.map(|v| v as u32),
                enabled: row.get::<i64>(18)? != 0,
                running_task_id: row.get::<Option<String>>(19)?,
                pending_rerun: row.get::<i64>(20)? != 0,
                last_run_unix_ms: row.get::<Option<i64>>(21)?,
                last_status: row.get::<Option<String>>(22)?,
                created_at: row.get::<String>(23)?,
                updated_at: row.get::<String>(24)?,
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
                SELECT id, public_id, agent_id, job_kind, prompt, content, tools, conflict_policy,
                       action_name, action_params, state_target, store_target,
                       next_run_unix_ms, interval_seconds, recurring_pattern,
                       overlap_policy, work_key, max_concurrency, enabled, running_task_id, pending_rerun,
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
                interval_seconds: row.get::<Option<i64>>(13)?.map(|v| v as u64),
                recurring_pattern: row.get::<Option<String>>(14)?,
                overlap_policy: row.get::<String>(15)?,
                work_key: row.get::<Option<String>>(16)?,
                max_concurrency: row.get::<Option<i64>>(17)?.map(|v| v as u32),
                enabled: row.get::<i64>(18)? != 0,
                running_task_id: row.get::<Option<String>>(19)?,
                pending_rerun: row.get::<i64>(20)? != 0,
                last_run_unix_ms: row.get::<Option<i64>>(21)?,
                last_status: row.get::<Option<String>>(22)?,
                created_at: row.get::<String>(23)?,
                updated_at: row.get::<String>(24)?,
            });
        }
        Ok(result)
    }

    pub async fn count_running_scheduled_jobs_for_work_key(&self, work_key: &str) -> Result<u32> {
        let conn = self.connect().await?;
        let mut rows = conn
            .query(
                r#"
                SELECT COUNT(*)
                FROM scheduled_jobs
                WHERE running_task_id IS NOT NULL AND work_key = ?1
                "#,
                turso::params![work_key],
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
              AND running_task_id IS NULL
              AND enabled = 1
              AND pending_rerun = 1
            "#,
            turso::params![work_key, next_run_unix_ms],
        )
        .await
        .context("Failed to wake pending scheduled jobs for work key")?;
        Ok(())
    }

    pub async fn update_scheduled_job(
        &self,
        id: i64,
        agent_id: &str,
        job_kind: &str,
        prompt: Option<&str>,
        content: Option<&str>,
        tools: Option<&str>,
        conflict_policy: Option<&str>,
        action_name: Option<&str>,
        action_params: Option<&str>,
        state_target: Option<&str>,
        store_target: Option<&str>,
        next_run_unix_ms: i64,
        interval_seconds: Option<u64>,
        recurring_pattern: Option<&str>,
        overlap_policy: &str,
        work_key: Option<&str>,
        max_concurrency: Option<u32>,
        enabled: bool,
    ) -> Result<()> {
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
                id,
                agent_id,
                job_kind,
                prompt,
                content,
                tools,
                conflict_policy,
                action_name,
                action_params,
                state_target,
                store_target,
                next_run_unix_ms,
                interval_seconds.map(|v| v as i64),
                recurring_pattern,
                overlap_policy,
                work_key,
                max_concurrency.map(|v| v as i64),
                if enabled { 1 } else { 0 }
            ],
        )
        .await
        .context("Failed to update scheduled job")?;
        Ok(())
    }

    pub async fn delete_scheduled_job(&self, id: i64) -> Result<()> {
        let conn = self.connect().await?;
        conn.execute(
            "DELETE FROM scheduled_jobs WHERE id = ?1",
            turso::params![id],
        )
        .await
        .context("Failed to delete scheduled job")?;
        Ok(())
    }
}
