use std::sync::Arc;

use anyhow::{Result, anyhow};
use tokio::sync::Notify;
use turin_daemon_protocol::{
    ContextPersistenceParams, ScheduleCreateParams, ScheduleJobDetail, ScheduleUpdateParams,
    StoreTargetParams,
};

use crate::persistence::schema::ScheduledJobRow;
use crate::persistence::state::StateStore;

#[derive(Clone)]
pub struct HarnessSchedulerAccess {
    jobs_store: Arc<StateStore>,
    wake: Option<Arc<Notify>>,
}

impl HarnessSchedulerAccess {
    pub fn new(jobs_store: Arc<StateStore>, wake: Option<Arc<Notify>>) -> Self {
        Self { jobs_store, wake }
    }

    pub async fn create_job(&self, params: ScheduleCreateParams) -> Result<ScheduleJobDetail> {
        let public_id = uuid::Uuid::now_v7();
        let state_target = serialize_store_target(
            params
                .persistence
                .as_ref()
                .and_then(|persistence| persistence.state.as_ref()),
        )?;
        let store_target = serialize_store_target(
            params
                .persistence
                .as_ref()
                .and_then(|persistence| persistence.store.as_ref()),
        )?;
        let id = self
            .jobs_store
            .create_scheduled_job(
                public_id,
                &params.agent_id,
                &params.prompt,
                state_target.as_deref(),
                store_target.as_deref(),
                params.next_run_unix_ms,
                params.interval_seconds,
                params.overlap_policy.as_deref().unwrap_or("skip"),
                params.enabled,
            )
            .await?;
        if let Some(wake) = &self.wake {
            wake.notify_one();
        }
        let job = self
            .jobs_store
            .list_scheduled_jobs()
            .await?
            .into_iter()
            .find(|row| row.id == id)
            .ok_or_else(|| anyhow!("Scheduled job '{}' was created but not visible", id))?;
        Ok(map_scheduled_job_detail(job))
    }

    pub async fn list_jobs(&self) -> Result<Vec<ScheduleJobDetail>> {
        Ok(self
            .jobs_store
            .list_scheduled_jobs()
            .await?
            .into_iter()
            .map(map_scheduled_job_detail)
            .collect())
    }

    pub async fn get_job(&self, public_id: &str) -> Result<Option<ScheduleJobDetail>> {
        let public_id = uuid::Uuid::parse_str(public_id)?;
        Ok(self
            .jobs_store
            .get_scheduled_job_by_public_id(public_id)
            .await?
            .map(map_scheduled_job_detail))
    }

    pub async fn update_job(
        &self,
        params: ScheduleUpdateParams,
    ) -> Result<Option<ScheduleJobDetail>> {
        let public_id = uuid::Uuid::parse_str(&params.id)?;
        let Some(row) = self
            .jobs_store
            .get_scheduled_job_by_public_id(public_id)
            .await?
        else {
            return Ok(None);
        };
        let persistence = params
            .persistence
            .or_else(|| scheduled_job_persistence(&row).ok().flatten());
        let state_target = serialize_store_target(
            persistence
                .as_ref()
                .and_then(|persistence| persistence.state.as_ref()),
        )?;
        let store_target = serialize_store_target(
            persistence
                .as_ref()
                .and_then(|persistence| persistence.store.as_ref()),
        )?;
        self.jobs_store
            .update_scheduled_job(
                row.id,
                params.agent_id.as_deref().unwrap_or(&row.agent_id),
                params.prompt.as_deref().unwrap_or(&row.prompt),
                state_target.as_deref(),
                store_target.as_deref(),
                params.next_run_unix_ms.unwrap_or(row.next_run_unix_ms),
                params.interval_seconds.or(row.interval_seconds),
                params
                    .overlap_policy
                    .as_deref()
                    .unwrap_or(row.overlap_policy.as_str()),
                params.enabled.unwrap_or(row.enabled),
            )
            .await?;
        if let Some(wake) = &self.wake {
            wake.notify_one();
        }
        Ok(self
            .jobs_store
            .get_scheduled_job_by_public_id(public_id)
            .await?
            .map(map_scheduled_job_detail))
    }

    pub async fn set_job_enabled(
        &self,
        public_id: &str,
        enabled: bool,
    ) -> Result<Option<ScheduleJobDetail>> {
        let public_id = uuid::Uuid::parse_str(public_id)?;
        let Some(row) = self
            .jobs_store
            .get_scheduled_job_by_public_id(public_id)
            .await?
        else {
            return Ok(None);
        };
        self.jobs_store
            .set_scheduled_job_enabled(row.id, enabled)
            .await?;
        if let Some(wake) = &self.wake {
            wake.notify_one();
        }
        Ok(self
            .jobs_store
            .get_scheduled_job_by_public_id(public_id)
            .await?
            .map(map_scheduled_job_detail))
    }

    pub async fn delete_job(&self, public_id: &str) -> Result<Option<ScheduleJobDetail>> {
        let public_id = uuid::Uuid::parse_str(public_id)?;
        let Some(row) = self
            .jobs_store
            .get_scheduled_job_by_public_id(public_id)
            .await?
        else {
            return Ok(None);
        };
        if row.running_task_id.is_some() {
            anyhow::bail!(
                "Cannot delete scheduled job '{}' while it has an active run",
                public_id
            );
        }
        let detail = map_scheduled_job_detail(row.clone());
        self.jobs_store.delete_scheduled_job(row.id).await?;
        if let Some(wake) = &self.wake {
            wake.notify_one();
        }
        Ok(Some(detail))
    }
}

fn map_scheduled_job_detail(row: ScheduledJobRow) -> ScheduleJobDetail {
    let public_id = uuid::Uuid::from_slice(&row.public_id)
        .map(|id| id.to_string())
        .unwrap_or_else(|_| format_uuid_bytes_simple(&row.public_id));
    let persistence = scheduled_job_persistence(&row).ok().flatten();
    ScheduleJobDetail {
        id: row.id,
        public_id: public_id.clone(),
        agent_id: row.agent_id,
        prompt: row.prompt,
        persistence,
        next_run_unix_ms: row.next_run_unix_ms,
        interval_seconds: row.interval_seconds,
        overlap_policy: row.overlap_policy,
        enabled: row.enabled,
        slot_id: scheduled_job_slot_id(&public_id),
        running_task_id: row.running_task_id,
        pending_rerun: row.pending_rerun,
        last_run_unix_ms: row.last_run_unix_ms,
        last_status: row.last_status,
        created_at: row.created_at,
        updated_at: row.updated_at,
    }
}

fn scheduled_job_slot_id(public_id: &str) -> String {
    format!("sched_{}", public_id.replace('-', ""))
}

fn format_uuid_bytes_simple(bytes: &[u8]) -> String {
    uuid::Uuid::from_slice(bytes)
        .map(|uuid| uuid.simple().to_string())
        .unwrap_or_else(|_| {
            let mut out = String::with_capacity(bytes.len() * 2);
            for byte in bytes {
                use std::fmt::Write as _;
                let _ = write!(&mut out, "{:02x}", byte);
            }
            out
        })
}

fn serialize_store_target(target: Option<&StoreTargetParams>) -> Result<Option<String>> {
    target
        .map(serde_json::to_string)
        .transpose()
        .map_err(anyhow::Error::from)
}

fn parse_store_target(raw: Option<&str>) -> Result<Option<StoreTargetParams>> {
    raw.map(serde_json::from_str)
        .transpose()
        .map_err(anyhow::Error::from)
}

fn scheduled_job_persistence(job: &ScheduledJobRow) -> Result<Option<ContextPersistenceParams>> {
    let state = parse_store_target(job.state_target.as_deref())?;
    let store = parse_store_target(job.store_target.as_deref())?;
    if state.is_none() && store.is_none() {
        return Ok(None);
    }
    Ok(Some(ContextPersistenceParams { state, store }))
}
