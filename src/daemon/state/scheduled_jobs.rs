use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};
use serde::Serialize;
use turin_daemon_protocol::{ContextPersistenceParams, ScheduleJobDetail, StoreTargetParams};

use super::DaemonState;
use crate::kernel::agent_manager::TaskStatusSnapshot;
use crate::kernel::config::{ContextPersistenceConfig, InferenceOverrideConfig, StoreTargetConfig};
use crate::kernel::session::QueuedTask;
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::ScheduledJobRow;

const SCHEDULED_JOB_BATCH_LIMIT: usize = 32;
const SCHEDULED_JOB_FAILURE_RETRY_MS: i64 = 60_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduledJobOverlapPolicy {
    Skip,
    Queue,
}

impl ScheduledJobOverlapPolicy {
    fn as_str(self) -> &'static str {
        match self {
            Self::Skip => "skip",
            Self::Queue => "queue",
        }
    }
}

impl std::str::FromStr for ScheduledJobOverlapPolicy {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "skip" => Ok(Self::Skip),
            "queue" => Ok(Self::Queue),
            _ => Err(anyhow!("Unsupported overlap policy '{}'", value)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CreateScheduledJobInput {
    pub agent_id: String,
    pub prompt: String,
    pub persistence: Option<ContextPersistenceParams>,
    pub next_run_unix_ms: i64,
    pub interval_seconds: Option<u64>,
    pub overlap_policy: ScheduledJobOverlapPolicy,
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct UpdateScheduledJobInput {
    pub agent_id: Option<String>,
    pub prompt: Option<String>,
    pub persistence: Option<ContextPersistenceParams>,
    pub next_run_unix_ms: Option<i64>,
    pub interval_seconds: Option<u64>,
    pub overlap_policy: Option<ScheduledJobOverlapPolicy>,
    pub enabled: Option<bool>,
}

impl DaemonState {
    pub(crate) fn set_scheduler_wake(&mut self, wake: std::sync::Arc<tokio::sync::Notify>) {
        self.scheduler_wake = Some(std::sync::Arc::clone(&wake));
        self.kernel.host.scheduler = Some(std::sync::Arc::new(
            crate::harness::scheduler::HarnessSchedulerAccess::new(
                std::sync::Arc::clone(&self.jobs_store),
                Some(wake),
            ),
        ));
        self.kernel
            .agent_manager()
            .bind_scheduler_access(self.kernel.host.scheduler.clone());
    }

    pub(crate) async fn create_scheduled_job(
        &self,
        input: CreateScheduledJobInput,
    ) -> Result<ScheduleJobDetail> {
        self.ensure_enabled_agent(&input.agent_id)?;
        let _ = self.resolve_scheduled_job_persistence(input.persistence.as_ref())?;
        let store = Arc::clone(&self.jobs_store);
        let public_id = uuid::Uuid::now_v7();
        let state_target = serialize_store_target(
            input
                .persistence
                .as_ref()
                .and_then(|persistence| persistence.state.as_ref()),
        )?;
        let store_target = serialize_store_target(
            input
                .persistence
                .as_ref()
                .and_then(|persistence| persistence.store.as_ref()),
        )?;
        let id = store
            .create_scheduled_job(
                public_id,
                &input.agent_id,
                &input.prompt,
                state_target.as_deref(),
                store_target.as_deref(),
                input.next_run_unix_ms,
                input.interval_seconds,
                input.overlap_policy.as_str(),
                input.enabled,
            )
            .await?;
        if let Some(wake) = &self.scheduler_wake {
            wake.notify_one();
        }
        let job = store
            .list_scheduled_jobs()
            .await?
            .into_iter()
            .find(|row| row.id == id)
            .ok_or_else(|| anyhow!("Scheduled job '{}' was created but not visible", id))?;
        Ok(map_scheduled_job_detail(job))
    }

    pub(crate) async fn list_scheduled_jobs(&self) -> Result<Vec<ScheduleJobDetail>> {
        let store = Arc::clone(&self.jobs_store);
        Ok(store
            .list_scheduled_jobs()
            .await?
            .into_iter()
            .map(map_scheduled_job_detail)
            .collect())
    }

    pub(crate) async fn update_scheduled_job(
        &self,
        public_id: &str,
        input: UpdateScheduledJobInput,
    ) -> Result<Option<ScheduleJobDetail>> {
        let store = Arc::clone(&self.jobs_store);
        let public_id = uuid::Uuid::parse_str(public_id)?;
        let Some(row) = store.get_scheduled_job_by_public_id(public_id).await? else {
            return Ok(None);
        };

        let agent_id = input.agent_id.unwrap_or_else(|| row.agent_id.clone());
        self.ensure_enabled_agent(&agent_id)?;

        let prompt = input.prompt.unwrap_or_else(|| row.prompt.clone());
        let next_run_unix_ms = input.next_run_unix_ms.unwrap_or(row.next_run_unix_ms);
        let interval_seconds = input.interval_seconds.or(row.interval_seconds);
        let overlap_policy = input
            .overlap_policy
            .unwrap_or_else(|| {
                row.overlap_policy
                    .parse::<ScheduledJobOverlapPolicy>()
                    .unwrap_or(ScheduledJobOverlapPolicy::Skip)
            })
            .as_str()
            .to_string();
        let enabled = input.enabled.unwrap_or(row.enabled);

        let persistence = match input.persistence {
            Some(persistence) => Some(persistence),
            None => scheduled_job_persistence(&row)?,
        };
        let _ = self.resolve_scheduled_job_persistence(persistence.as_ref())?;
        let state_target =
            serialize_store_target(persistence.as_ref().and_then(|value| value.state.as_ref()))?;
        let store_target =
            serialize_store_target(persistence.as_ref().and_then(|value| value.store.as_ref()))?;

        store
            .update_scheduled_job(
                row.id,
                &agent_id,
                &prompt,
                state_target.as_deref(),
                store_target.as_deref(),
                next_run_unix_ms,
                interval_seconds,
                &overlap_policy,
                enabled,
            )
            .await?;
        if let Some(wake) = &self.scheduler_wake {
            wake.notify_one();
        }
        Ok(store
            .get_scheduled_job_by_public_id(public_id)
            .await?
            .map(map_scheduled_job_detail))
    }

    pub(crate) async fn scheduled_job_detail(
        &self,
        public_id: &str,
    ) -> Result<Option<ScheduleJobDetail>> {
        let store = Arc::clone(&self.jobs_store);
        let public_id = uuid::Uuid::parse_str(public_id)?;
        Ok(store
            .get_scheduled_job_by_public_id(public_id)
            .await?
            .map(map_scheduled_job_detail))
    }

    pub(crate) async fn set_scheduled_job_enabled(
        &self,
        public_id: &str,
        enabled: bool,
    ) -> Result<Option<ScheduleJobDetail>> {
        let store = Arc::clone(&self.jobs_store);
        let public_id = uuid::Uuid::parse_str(public_id)?;
        let Some(row) = store.get_scheduled_job_by_public_id(public_id).await? else {
            return Ok(None);
        };
        store.set_scheduled_job_enabled(row.id, enabled).await?;
        if let Some(wake) = &self.scheduler_wake {
            wake.notify_one();
        }
        Ok(store
            .get_scheduled_job_by_public_id(public_id)
            .await?
            .map(map_scheduled_job_detail))
    }

    pub(crate) async fn delete_scheduled_job(
        &self,
        public_id: &str,
    ) -> Result<Option<ScheduleJobDetail>> {
        let store = Arc::clone(&self.jobs_store);
        let public_id = uuid::Uuid::parse_str(public_id)?;
        let Some(row) = store.get_scheduled_job_by_public_id(public_id).await? else {
            return Ok(None);
        };
        if row.running_task_id.is_some() {
            anyhow::bail!(
                "Cannot delete scheduled job '{}' while it has an active run",
                public_id
            );
        }
        let detail = map_scheduled_job_detail(row.clone());
        store.delete_scheduled_job(row.id).await?;
        if let Some(wake) = &self.scheduler_wake {
            wake.notify_one();
        }
        Ok(Some(detail))
    }

    pub(crate) async fn scheduler_tick(&self) -> Result<Option<Duration>> {
        let store = Arc::clone(&self.jobs_store);
        let now = now_unix_ms();

        let running_jobs = store.list_running_scheduled_jobs().await?;
        for job in running_jobs {
            self.reconcile_running_scheduled_job(&store, &job, now)
                .await?;
        }

        let due_jobs = store
            .list_due_scheduled_jobs(now, SCHEDULED_JOB_BATCH_LIMIT)
            .await?;
        for job in due_jobs {
            self.process_due_scheduled_job(&store, &job, now).await?;
        }

        let next_due = store.next_scheduled_due_unix_ms().await?;
        Ok(next_due.map(|due| {
            let delay_ms = (due - now).max(0) as u64;
            Duration::from_millis(delay_ms)
        }))
    }

    async fn reconcile_running_scheduled_job(
        &self,
        store: &std::sync::Arc<crate::persistence::state::StateStore>,
        job: &ScheduledJobRow,
        now_unix_ms: i64,
    ) -> Result<()> {
        let Some(task_id) = &job.running_task_id else {
            return Ok(());
        };
        let snapshot = self.get_task(task_id).await;
        if let Some(snapshot) = snapshot.as_ref()
            && matches!(snapshot.state.as_str(), "queued" | "running" | "cancelling")
        {
            return Ok(());
        }

        let last_status = snapshot
            .as_ref()
            .map(scheduled_job_terminal_status)
            .or_else(|| Some("orphaned".to_string()));
        let next_run_override = if job.pending_rerun && job.enabled {
            Some(now_unix_ms)
        } else {
            None
        };
        store
            .mark_scheduled_job_finished(job.id, last_status.as_deref(), next_run_override, false)
            .await
    }

    async fn process_due_scheduled_job(
        &self,
        store: &std::sync::Arc<crate::persistence::state::StateStore>,
        job: &ScheduledJobRow,
        now_unix_ms: i64,
    ) -> Result<()> {
        if job.running_task_id.is_some() {
            let overlap = job
                .overlap_policy
                .parse::<ScheduledJobOverlapPolicy>()
                .unwrap_or(ScheduledJobOverlapPolicy::Skip);
            if let Some(interval_seconds) = job.interval_seconds {
                let advanced =
                    advance_recurring_due(job.next_run_unix_ms, interval_seconds, now_unix_ms);
                store
                    .mark_scheduled_job_overlap(
                        job.id,
                        advanced,
                        matches!(overlap, ScheduledJobOverlapPolicy::Queue),
                    )
                    .await?;
            }
            return Ok(());
        }

        let submit = self.submit_scheduled_job(job).await;
        match submit {
            Ok(task) => {
                let (next_run_unix_ms, enabled) = match job.interval_seconds {
                    Some(interval_seconds) => (
                        advance_recurring_due(job.next_run_unix_ms, interval_seconds, now_unix_ms),
                        true,
                    ),
                    None => (job.next_run_unix_ms, false),
                };
                store
                    .mark_scheduled_job_started(
                        job.id,
                        &task.request_id,
                        next_run_unix_ms,
                        enabled,
                        now_unix_ms,
                    )
                    .await?;
            }
            Err(err) => {
                let message = format!("submit_failed: {}", err);
                store
                    .mark_scheduled_job_submit_failed(
                        job.id,
                        now_unix_ms + SCHEDULED_JOB_FAILURE_RETRY_MS,
                        &message,
                    )
                    .await?;
            }
        }

        Ok(())
    }

    async fn submit_scheduled_job(&self, job: &ScheduledJobRow) -> Result<TaskStatusSnapshot> {
        let public_id = uuid::Uuid::from_slice(&job.public_id)
            .map(|id| id.to_string())
            .unwrap_or_else(|_| super::helpers::format_uuid_bytes_simple(&job.public_id));
        let slot_id = scheduled_job_slot_id(&public_id);
        let persistence = scheduled_job_persistence(job)?;
        let (state_selector, default_store_selector) =
            self.resolve_scheduled_job_persistence(persistence.as_ref())?;
        let live = self
            .kernel
            .agent_manager()
            .open_session(
                &job.agent_id,
                Some(&slot_id),
                state_selector,
                default_store_selector,
                None,
                InferenceOverrideConfig::default(),
            )
            .await?;
        let request_id = self
            .kernel
            .agent_manager()
            .submit_to_session(
                &live.session_id,
                Some(&live.slot_id),
                QueuedTask::ad_hoc(job.prompt.clone()),
                None,
            )
            .await?;
        self.kernel
            .agent_manager()
            .get_task(&request_id)
            .await
            .ok_or_else(|| anyhow!("Task '{}' was submitted but is not visible", request_id))
    }

    fn resolve_scheduled_job_persistence(
        &self,
        persistence: Option<&ContextPersistenceParams>,
    ) -> Result<(Option<StoreSelector>, Option<StoreSelector>)> {
        let Some(persistence) = persistence else {
            return Ok((None, None));
        };
        let context = ContextPersistenceConfig {
            state: persistence
                .state
                .as_ref()
                .map(store_target_config_from_params),
            store: persistence
                .store
                .as_ref()
                .map(store_target_config_from_params),
        };
        Ok((
            Some(
                self.bootstrap_config
                    .persistence
                    .resolve_context_state_selector(Some(&context))?,
            ),
            Some(
                self.bootstrap_config
                    .persistence
                    .resolve_context_store_selector(Some(&context))?,
            ),
        ))
    }
}

fn map_scheduled_job_detail(row: ScheduledJobRow) -> ScheduleJobDetail {
    let public_id = uuid::Uuid::from_slice(&row.public_id)
        .map(|id| id.to_string())
        .unwrap_or_else(|_| super::helpers::format_uuid_bytes_simple(&row.public_id));
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

fn now_unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis() as i64
}

fn advance_recurring_due(next_run_unix_ms: i64, interval_seconds: u64, now_unix_ms: i64) -> i64 {
    let step = (interval_seconds as i64).saturating_mul(1000).max(1_000);
    let mut next = next_run_unix_ms;
    while next <= now_unix_ms {
        next = next.saturating_add(step);
    }
    next
}

fn scheduled_job_terminal_status(snapshot: &TaskStatusSnapshot) -> String {
    snapshot
        .status
        .map(|status| format!("{:?}", status).to_ascii_lowercase())
        .unwrap_or_else(|| snapshot.state.clone())
}

fn scheduled_job_slot_id(public_id: &str) -> String {
    format!("sched_{}", public_id.replace('-', ""))
}

fn store_target_config_from_params(params: &StoreTargetParams) -> StoreTargetConfig {
    StoreTargetConfig {
        path: params.path.clone(),
        alias: params.alias.clone(),
    }
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
