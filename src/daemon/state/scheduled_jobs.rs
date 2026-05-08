use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};
use serde::Serialize;
use serde_json::{Map as JsonMap, Value as JsonValue};
use turin_daemon_protocol::{
    ContextPersistenceParams, ScheduleActionParams, ScheduleJobDetail, ScheduleJobRunDetail,
    ScheduleJobRunList, StoreTargetParams,
};
use turin_types::{TaskInputContent, ToolsConfig};

use super::DaemonState;
use crate::kernel::agent_manager::TaskStatusSnapshot;
use crate::kernel::config::{ContextPersistenceConfig, InferenceOverrideConfig, StoreTargetConfig};
use crate::kernel::session::{ExecutionConflictPolicy, QueuedTask};
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::{ScheduledJobRow, ScheduledJobRunRow, WorkItemRow};
use crate::persistence::state::{ScheduledJobInsert, ScheduledJobUpdate};

const SCHEDULED_JOB_BATCH_LIMIT: usize = 32;
const SCHEDULED_JOB_FAILURE_RETRY_MS: i64 = 60_000;

#[derive(Debug, Clone)]
struct ScheduledJobFailure {
    code: &'static str,
    message: String,
}

impl ScheduledJobFailure {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    fn status(&self) -> String {
        format!("{}: {}", self.code, self.message)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScheduledJobOverlapPolicy {
    Skip,
    Queue,
    Parallel,
}

impl ScheduledJobOverlapPolicy {
    fn as_str(self) -> &'static str {
        match self {
            Self::Skip => "skip",
            Self::Queue => "queue",
            Self::Parallel => "parallel",
        }
    }
}

impl std::str::FromStr for ScheduledJobOverlapPolicy {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "skip" => Ok(Self::Skip),
            "queue" => Ok(Self::Queue),
            "parallel" => Ok(Self::Parallel),
            _ => Err(anyhow!("Unsupported overlap policy '{}'", value)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CreateScheduledJobInput {
    pub agent_id: String,
    pub prompt: Option<String>,
    pub content: Option<Vec<TaskInputContent>>,
    pub tools: Option<ToolsConfig>,
    pub conflict_policy: Option<String>,
    pub action: Option<ScheduleActionParams>,
    pub persistence: Option<ContextPersistenceParams>,
    pub next_run_unix_ms: i64,
    pub interval_seconds: Option<u64>,
    pub recurring_pattern: Option<String>,
    pub overlap_policy: ScheduledJobOverlapPolicy,
    pub work_key: Option<String>,
    pub max_concurrency: Option<u32>,
    pub enabled: bool,
}

#[derive(Debug, Clone, Default)]
pub struct UpdateScheduledJobInput {
    pub agent_id: Option<String>,
    pub prompt: Option<String>,
    pub content: Option<Vec<TaskInputContent>>,
    pub tools: Option<ToolsConfig>,
    pub conflict_policy: Option<String>,
    pub action: Option<ScheduleActionParams>,
    pub persistence: Option<ContextPersistenceParams>,
    pub next_run_unix_ms: Option<i64>,
    pub interval_seconds: Option<u64>,
    pub recurring_pattern: Option<String>,
    pub overlap_policy: Option<ScheduledJobOverlapPolicy>,
    pub work_key: Option<String>,
    pub max_concurrency: Option<u32>,
    pub enabled: Option<bool>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScheduledJobKind {
    Prompt,
    Action,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScheduledJobRecurringPattern {
    Daily,
    Weekly,
}

impl ScheduledJobRecurringPattern {
    fn step_ms(self) -> i64 {
        match self {
            Self::Daily => 86_400_000,
            Self::Weekly => 604_800_000,
        }
    }
}

impl std::str::FromStr for ScheduledJobRecurringPattern {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "daily" => Ok(Self::Daily),
            "weekly" => Ok(Self::Weekly),
            _ => Err(anyhow!("Unsupported recurring pattern '{}'", value)),
        }
    }
}

impl ScheduledJobKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Prompt => "prompt",
            Self::Action => "action",
        }
    }
}

impl std::str::FromStr for ScheduledJobKind {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        match value {
            "prompt" => Ok(Self::Prompt),
            "action" => Ok(Self::Action),
            _ => Err(anyhow!("Unsupported scheduled job kind '{}'", value)),
        }
    }
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
        let job_kind =
            validate_scheduled_job_payload(input.prompt.as_ref(), input.action.as_ref())?;
        validate_scheduled_job_recurrence(
            input.interval_seconds,
            input.recurring_pattern.as_deref(),
        )?;
        let _ = self.resolve_scheduled_job_persistence(input.persistence.as_ref())?;
        let store = Arc::clone(&self.jobs_store);
        let public_id = uuid::Uuid::now_v7();
        let content = serialize_json(input.content.as_ref())?;
        let tools = serialize_json(input.tools.as_ref())?;
        let action_name = input.action.as_ref().map(|action| action.name.clone());
        let action_params = serialize_json(
            input
                .action
                .as_ref()
                .and_then(|action| action.params.as_ref()),
        )?;
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
            .create_scheduled_job(ScheduledJobInsert {
                public_id,
                agent_id: &input.agent_id,
                job_kind: job_kind.as_str(),
                prompt: input.prompt.as_deref(),
                content: content.as_deref(),
                tools: tools.as_deref(),
                conflict_policy: input.conflict_policy.as_deref(),
                action_name: action_name.as_deref(),
                action_params: action_params.as_deref(),
                state_target: state_target.as_deref(),
                store_target: store_target.as_deref(),
                next_run_unix_ms: input.next_run_unix_ms,
                interval_seconds: input.interval_seconds,
                recurring_pattern: input.recurring_pattern.as_deref(),
                overlap_policy: input.overlap_policy.as_str(),
                work_key: input.work_key.as_deref(),
                max_concurrency: input.max_concurrency,
                enabled: input.enabled,
            })
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

        let job_kind = match (input.prompt.as_ref(), input.action.as_ref()) {
            (Some(_), None) => ScheduledJobKind::Prompt,
            (None, Some(_)) => ScheduledJobKind::Action,
            (None, None) => row.job_kind.parse::<ScheduledJobKind>()?,
            (Some(_), Some(_)) => {
                anyhow::bail!("Scheduled job cannot define both prompt and action payloads")
            }
        };
        let prompt = if matches!(job_kind, ScheduledJobKind::Prompt) {
            Some(
                input
                    .prompt
                    .or_else(|| row.prompt.clone())
                    .ok_or_else(|| anyhow!("Prompt jobs require prompt text"))?,
            )
        } else {
            None
        };
        let content = match input.content {
            Some(content) => Some(content),
            None => parse_json(row.content.as_deref())?,
        };
        let tools = match input.tools {
            Some(tools) => Some(tools),
            None => parse_json(row.tools.as_deref())?,
        };
        let conflict_policy = input.conflict_policy.or(row.conflict_policy.clone());
        let action = match input.action {
            Some(action) => Some(action),
            None => scheduled_job_action(&row)?,
        };
        validate_scheduled_job_payload(prompt.as_ref(), action.as_ref())?;
        let next_run_unix_ms = input.next_run_unix_ms.unwrap_or(row.next_run_unix_ms);
        let interval_seconds = input.interval_seconds.or(row.interval_seconds);
        let recurring_pattern = input.recurring_pattern.or(row.recurring_pattern.clone());
        validate_scheduled_job_recurrence(interval_seconds, recurring_pattern.as_deref())?;
        let overlap_policy = input
            .overlap_policy
            .unwrap_or_else(|| {
                row.overlap_policy
                    .parse::<ScheduledJobOverlapPolicy>()
                    .unwrap_or(ScheduledJobOverlapPolicy::Skip)
            })
            .as_str()
            .to_string();
        let work_key = input.work_key.or(row.work_key.clone());
        let max_concurrency = input.max_concurrency.or(row.max_concurrency);
        let enabled = input.enabled.unwrap_or(row.enabled);

        let persistence = match input.persistence {
            Some(persistence) => Some(persistence),
            None => scheduled_job_persistence(&row)?,
        };
        let content_json = serialize_json(content.as_ref())?;
        let tools_json = serialize_json(tools.as_ref())?;
        let action_name = action.as_ref().map(|action| action.name.clone());
        let action_params =
            serialize_json(action.as_ref().and_then(|action| action.params.as_ref()))?;
        let _ = self.resolve_scheduled_job_persistence(persistence.as_ref())?;
        let state_target =
            serialize_store_target(persistence.as_ref().and_then(|value| value.state.as_ref()))?;
        let store_target =
            serialize_store_target(persistence.as_ref().and_then(|value| value.store.as_ref()))?;

        store
            .update_scheduled_job(ScheduledJobUpdate {
                id: row.id,
                agent_id: &agent_id,
                job_kind: job_kind.as_str(),
                prompt: prompt.as_deref(),
                content: content_json.as_deref(),
                tools: tools_json.as_deref(),
                conflict_policy: conflict_policy.as_deref(),
                action_name: action_name.as_deref(),
                action_params: action_params.as_deref(),
                state_target: state_target.as_deref(),
                store_target: store_target.as_deref(),
                next_run_unix_ms,
                interval_seconds,
                recurring_pattern: recurring_pattern.as_deref(),
                overlap_policy: &overlap_policy,
                work_key: work_key.as_deref(),
                max_concurrency,
                enabled,
            })
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

    pub(crate) async fn scheduled_job_runs(
        &self,
        public_id: &str,
        active_only: bool,
        limit: Option<u32>,
    ) -> Result<Option<ScheduleJobRunList>> {
        let store = Arc::clone(&self.jobs_store);
        let public_id = uuid::Uuid::parse_str(public_id)?;
        let Some(row) = store.get_scheduled_job_by_public_id(public_id).await? else {
            return Ok(None);
        };
        let public_id = uuid::Uuid::from_slice(&row.public_id)
            .map(|id| id.to_string())
            .unwrap_or_else(|_| super::helpers::format_uuid_bytes_simple(&row.public_id));
        let runs = store
            .list_scheduled_job_runs(row.id, active_only, limit)
            .await?
            .into_iter()
            .map(map_scheduled_job_run_detail)
            .collect();
        Ok(Some(ScheduleJobRunList { public_id, runs }))
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
        if row.active_run_count > 0 {
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

    pub(crate) async fn scheduler_tick(&mut self) -> Result<Option<Duration>> {
        let store = Arc::clone(&self.jobs_store);
        let now = now_unix_ms();

        let running_runs = store.list_active_scheduled_job_runs().await?;
        for run in running_runs {
            self.reconcile_running_scheduled_job_run(&store, &run, now)
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

    async fn reconcile_running_scheduled_job_run(
        &self,
        store: &std::sync::Arc<crate::persistence::state::StateStore>,
        run: &ScheduledJobRunRow,
        now_unix_ms: i64,
    ) -> Result<()> {
        let Some(job) = store.get_scheduled_job_by_id(run.scheduled_job_id).await? else {
            return Ok(());
        };
        let snapshot = self.get_task(&run.task_id).await;
        if let Some(snapshot) = snapshot.as_ref()
            && matches!(snapshot.state.as_str(), "queued" | "running" | "cancelling")
        {
            return Ok(());
        }

        let last_status = snapshot
            .as_ref()
            .map(scheduled_job_terminal_status)
            .or_else(|| Some("orphaned".to_string()));
        store
            .finish_scheduled_job_run(job.id, &run.task_id, now_unix_ms, last_status.as_deref())
            .await?;
        let Some(updated_job) = store.get_scheduled_job_by_id(job.id).await? else {
            return Ok(());
        };
        if updated_job.active_run_count == 0 {
            let next_run_override = if updated_job.pending_rerun && updated_job.enabled {
                Some(now_unix_ms)
            } else {
                None
            };
            store
                .finalize_scheduled_job_after_runs(job.id, next_run_override, false)
                .await?;
        }
        self.wake_group_pending_jobs(store, &job, now_unix_ms).await
    }

    async fn process_due_scheduled_job(
        &mut self,
        store: &std::sync::Arc<crate::persistence::state::StateStore>,
        job: &ScheduledJobRow,
        now_unix_ms: i64,
    ) -> Result<()> {
        let job_kind = job
            .job_kind
            .parse::<ScheduledJobKind>()
            .unwrap_or(ScheduledJobKind::Prompt);
        if job.active_run_count > 0 {
            let overlap = job
                .overlap_policy
                .parse::<ScheduledJobOverlapPolicy>()
                .unwrap_or(ScheduledJobOverlapPolicy::Skip);
            if !matches!(overlap, ScheduledJobOverlapPolicy::Parallel) {
                if let Some(advanced) = next_recurring_due(job, now_unix_ms)? {
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
        }

        if let Some(work_key) = job.work_key.as_deref() {
            let max_concurrency = job.max_concurrency.unwrap_or(1).max(1);
            let active = store
                .count_running_scheduled_jobs_for_work_key(work_key)
                .await?;
            if active >= max_concurrency {
                self.defer_capacity_blocked_job(store, job, now_unix_ms)
                    .await?;
                return Ok(());
            }
        }

        match job_kind {
            ScheduledJobKind::Prompt => {
                let submit = self.submit_scheduled_job(job).await;
                match submit {
                    Ok(task) => {
                        let (next_run_unix_ms, enabled) =
                            match next_recurring_due(job, now_unix_ms)? {
                                Some(next) => (next, true),
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
                        store
                            .mark_scheduled_job_failed(
                                job.id,
                                now_unix_ms + SCHEDULED_JOB_FAILURE_RETRY_MS,
                                "schedule_submit_failed",
                                &format!("schedule_submit_failed: {}", err),
                            )
                            .await?;
                    }
                }
            }
            ScheduledJobKind::Action => {
                let run = self.execute_scheduled_action(job).await;
                match run {
                    Ok(status) => {
                        let (next_run_unix_ms, enabled) =
                            match next_recurring_due(job, now_unix_ms)? {
                                Some(next) => (next, true),
                                None => (job.next_run_unix_ms, false),
                            };
                        store
                            .mark_scheduled_job_action_completed(
                                job.id,
                                next_run_unix_ms,
                                enabled,
                                now_unix_ms,
                                &status,
                            )
                            .await?;
                        self.wake_group_pending_jobs(store, job, now_unix_ms)
                            .await?;
                    }
                    Err(err) => {
                        store
                            .mark_scheduled_job_failed(
                                job.id,
                                now_unix_ms + SCHEDULED_JOB_FAILURE_RETRY_MS,
                                err.code,
                                &err.status(),
                            )
                            .await?;
                    }
                }
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
                scheduled_job_task(job)?,
                None,
            )
            .await?;
        self.kernel
            .agent_manager()
            .get_task(&request_id)
            .await
            .ok_or_else(|| anyhow!("Task '{}' was submitted but is not visible", request_id))
    }

    async fn defer_capacity_blocked_job(
        &self,
        store: &std::sync::Arc<crate::persistence::state::StateStore>,
        job: &ScheduledJobRow,
        now_unix_ms: i64,
    ) -> Result<()> {
        let overlap = job
            .overlap_policy
            .parse::<ScheduledJobOverlapPolicy>()
            .unwrap_or(ScheduledJobOverlapPolicy::Skip);
        let recurring_next = next_recurring_due(job, now_unix_ms)?;
        let (next_run_unix_ms, pending_rerun) = match (recurring_next, overlap) {
            (Some(next), ScheduledJobOverlapPolicy::Skip) => (next, false),
            (Some(next), ScheduledJobOverlapPolicy::Queue) => (next, true),
            (Some(next), ScheduledJobOverlapPolicy::Parallel) => (next, true),
            (None, ScheduledJobOverlapPolicy::Queue) => {
                (now_unix_ms + SCHEDULED_JOB_FAILURE_RETRY_MS, true)
            }
            (None, ScheduledJobOverlapPolicy::Parallel) => {
                (now_unix_ms + SCHEDULED_JOB_FAILURE_RETRY_MS, true)
            }
            (None, ScheduledJobOverlapPolicy::Skip) => {
                (now_unix_ms + SCHEDULED_JOB_FAILURE_RETRY_MS, false)
            }
        };
        store
            .mark_scheduled_job_capacity_blocked(
                job.id,
                next_run_unix_ms,
                pending_rerun,
                "blocked: concurrency limit reached",
            )
            .await
    }

    async fn wake_group_pending_jobs(
        &self,
        store: &std::sync::Arc<crate::persistence::state::StateStore>,
        job: &ScheduledJobRow,
        now_unix_ms: i64,
    ) -> Result<()> {
        let Some(work_key) = job.work_key.as_deref() else {
            return Ok(());
        };
        store
            .wake_pending_scheduled_jobs_for_work_key(work_key, now_unix_ms)
            .await?;
        if let Some(wake) = &self.scheduler_wake {
            wake.notify_one();
        }
        Ok(())
    }

    async fn execute_scheduled_action(
        &mut self,
        job: &ScheduledJobRow,
    ) -> std::result::Result<String, ScheduledJobFailure> {
        let Some(action) = scheduled_job_action(job).map_err(|err| {
            ScheduledJobFailure::new("schedule_action_invalid_payload", err.to_string())
        })?
        else {
            return Err(ScheduledJobFailure::new(
                "schedule_action_missing_payload",
                format!("action job '{}' is missing action payload", job.id),
            ));
        };
        self.execute_named_scheduled_action(&job.agent_id, &action)
            .await
    }

    async fn execute_named_scheduled_action(
        &mut self,
        agent_id: &str,
        action: &ScheduleActionParams,
    ) -> std::result::Result<String, ScheduledJobFailure> {
        match action.name.as_str() {
            "worklist.dispatch_next" => {
                self.execute_scheduled_worklist_dispatch(agent_id, action)
                    .await
            }
            "worklist.release_stale" => self.execute_scheduled_worklist_release_stale(action).await,
            _ => self.execute_leaf_scheduled_action(agent_id, action).await,
        }
    }

    async fn execute_leaf_scheduled_action(
        &mut self,
        agent_id: &str,
        action: &ScheduleActionParams,
    ) -> std::result::Result<String, ScheduledJobFailure> {
        match action.name.as_str() {
            "agent.enable" => {
                let id = required_action_id(action).map_err(|err| {
                    ScheduledJobFailure::new("schedule_action_invalid_params", err.to_string())
                })?;
                self.set_agent_enabled(&id, true).await.map_err(|err| {
                    ScheduledJobFailure::new("schedule_action_builtin_failed", err.to_string())
                })?;
                Ok("completed: agent enabled".to_string())
            }
            "agent.disable" => {
                let id = required_action_id(action).map_err(|err| {
                    ScheduledJobFailure::new("schedule_action_invalid_params", err.to_string())
                })?;
                self.set_agent_enabled(&id, false).await.map_err(|err| {
                    ScheduledJobFailure::new("schedule_action_builtin_failed", err.to_string())
                })?;
                Ok("completed: agent disabled".to_string())
            }
            "channel.enable" => {
                let id = required_action_id(action).map_err(|err| {
                    ScheduledJobFailure::new("schedule_action_invalid_params", err.to_string())
                })?;
                self.set_channel_enabled(&id, true).await.map_err(|err| {
                    ScheduledJobFailure::new("schedule_action_builtin_failed", err.to_string())
                })?;
                Ok("completed: channel enabled".to_string())
            }
            "channel.disable" => {
                let id = required_action_id(action).map_err(|err| {
                    ScheduledJobFailure::new("schedule_action_invalid_params", err.to_string())
                })?;
                self.set_channel_enabled(&id, false).await.map_err(|err| {
                    ScheduledJobFailure::new("schedule_action_builtin_failed", err.to_string())
                })?;
                Ok("completed: channel disabled".to_string())
            }
            _ => self.execute_harness_scheduled_action(agent_id, action),
        }
    }

    async fn execute_scheduled_worklist_dispatch(
        &mut self,
        agent_id: &str,
        action: &ScheduleActionParams,
    ) -> std::result::Result<String, ScheduledJobFailure> {
        let params = scheduled_worklist_action_params(action).map_err(|err| {
            ScheduledJobFailure::new("schedule_action_invalid_params", err.to_string())
        })?;
        let selector = scheduled_worklist_store_selector(&params).map_err(|err| {
            ScheduledJobFailure::new("schedule_action_invalid_params", err.to_string())
        })?;
        let store = self
            .kernel
            .store_manager()
            .open(&selector)
            .await
            .map_err(|err| {
                ScheduledJobFailure::new("schedule_action_builtin_failed", err.to_string())
            })?;
        let worklist = store
            .open_worklist(&params.name, params.scope.as_deref().unwrap_or(""), None)
            .await
            .map_err(|err| {
                ScheduledJobFailure::new("schedule_action_builtin_failed", err.to_string())
            })?;
        let rows = store.list_work_items(worklist.id).await.map_err(|err| {
            ScheduledJobFailure::new("schedule_action_builtin_failed", err.to_string())
        })?;
        let status_map = rows
            .iter()
            .map(|row| {
                (
                    format_work_item_public_id(&row.public_id),
                    row.status.clone(),
                )
            })
            .collect::<std::collections::HashMap<_, _>>();
        let now_unix_ms = now_unix_ms();
        let execution_id = format!("scheduled:worklist:{}", action.name);
        for row in rows
            .iter()
            .filter(|row| row.parent_item_id.is_none())
            .filter(|row| row.claim_execution_id.is_none())
            .filter(|row| work_item_claimable_now(row, now_unix_ms))
            .filter(|row| work_item_dependencies_satisfied(row, &status_map))
            .filter(|row| work_item_matches_where(row, params.where_filter.as_ref()))
            .take(params.limit.unwrap_or(usize::MAX))
        {
            let claimed = store
                .try_claim_work_item(row.id, agent_id, None, Some(&execution_id), now_unix_ms)
                .await
                .map_err(|err| {
                    ScheduledJobFailure::new("schedule_action_builtin_failed", err.to_string())
                })?;
            if !claimed {
                continue;
            }
            let refreshed = store
                .get_work_item_by_id(row.id)
                .await
                .map_err(|err| {
                    ScheduledJobFailure::new("schedule_action_builtin_failed", err.to_string())
                })?
                .ok_or_else(|| {
                    ScheduledJobFailure::new(
                        "schedule_action_builtin_failed",
                        "claimed work item vanished",
                    )
                })?;
            let status = match refreshed.item_kind.as_str() {
                "action" => {
                    let nested = ScheduleActionParams {
                        name: refreshed.action_name.clone().ok_or_else(|| {
                            ScheduledJobFailure::new(
                                "schedule_action_invalid_payload",
                                "worklist action item missing action",
                            )
                        })?,
                        params: refreshed
                            .action_params
                            .as_deref()
                            .map(serde_json::from_str)
                            .transpose()
                            .map_err(|err| {
                                ScheduledJobFailure::new(
                                    "schedule_action_invalid_payload",
                                    err.to_string(),
                                )
                            })?,
                    };
                    if nested.name.starts_with("worklist.") {
                        return Err(ScheduledJobFailure::new(
                            "schedule_action_invalid_payload",
                            "nested worklist.* actions are not supported inside scheduled worklist dispatch",
                        ));
                    }
                    self.execute_leaf_scheduled_action(agent_id, &nested)
                        .await?
                }
                _ => {
                    let live = self
                        .kernel
                        .agent_manager()
                        .open_session(
                            agent_id,
                            Some("worklist"),
                            Some(selector.clone()),
                            None,
                            None,
                            InferenceOverrideConfig::default(),
                        )
                        .await
                        .map_err(|err| {
                            ScheduledJobFailure::new(
                                "schedule_action_builtin_failed",
                                err.to_string(),
                            )
                        })?;
                    let request_id = self
                        .kernel
                        .agent_manager()
                        .submit_to_session(
                            &live.session_id,
                            Some(&live.slot_id),
                            work_item_task(&refreshed).map_err(|err| {
                                ScheduledJobFailure::new(
                                    "schedule_action_invalid_payload",
                                    err.to_string(),
                                )
                            })?,
                            None,
                        )
                        .await
                        .map_err(|err| {
                            ScheduledJobFailure::new(
                                "schedule_action_builtin_failed",
                                err.to_string(),
                            )
                        })?;
                    format!("completed: queued task {}", request_id)
                }
            };
            return Ok(status);
        }
        Ok("completed: no eligible work item".to_string())
    }

    async fn execute_scheduled_worklist_release_stale(
        &mut self,
        action: &ScheduleActionParams,
    ) -> std::result::Result<String, ScheduledJobFailure> {
        let params = scheduled_worklist_action_params(action).map_err(|err| {
            ScheduledJobFailure::new("schedule_action_invalid_params", err.to_string())
        })?;
        let selector = scheduled_worklist_store_selector(&params).map_err(|err| {
            ScheduledJobFailure::new("schedule_action_invalid_params", err.to_string())
        })?;
        let store = self
            .kernel
            .store_manager()
            .open(&selector)
            .await
            .map_err(|err| {
                ScheduledJobFailure::new("schedule_action_builtin_failed", err.to_string())
            })?;
        let worklist = store
            .open_worklist(&params.name, params.scope.as_deref().unwrap_or(""), None)
            .await
            .map_err(|err| {
                ScheduledJobFailure::new("schedule_action_builtin_failed", err.to_string())
            })?;
        let stale_before =
            now_unix_ms().saturating_sub(params.stale_after_seconds.unwrap_or(300) as i64 * 1000);
        let rows = store.list_work_items(worklist.id).await.map_err(|err| {
            ScheduledJobFailure::new("schedule_action_builtin_failed", err.to_string())
        })?;
        let candidates = rows
            .into_iter()
            .filter(|row| row.parent_item_id.is_none())
            .filter(|row| work_item_is_orphaned(row, stale_before))
            .filter(|row| work_item_matches_where(row, params.where_filter.as_ref()))
            .take(params.limit.unwrap_or(usize::MAX))
            .collect::<Vec<_>>();
        let mut released = 0usize;
        for row in candidates {
            store.release_work_item(row.id).await.map_err(|err| {
                ScheduledJobFailure::new("schedule_action_builtin_failed", err.to_string())
            })?;
            released += 1;
        }
        Ok(format!("completed: released {} stale work items", released))
    }

    fn execute_harness_scheduled_action(
        &self,
        agent_id: &str,
        action: &ScheduleActionParams,
    ) -> std::result::Result<String, ScheduledJobFailure> {
        let runtime = self.kernel.runtime_for_agent(agent_id);
        let instance = runtime
            .create_instance(self.kernel.harness_init_context())
            .map_err(|err| {
                ScheduledJobFailure::new("schedule_action_harness_load_failed", err.to_string())
            })?;
        let result = instance.invoke_declared_action_for_agent(
            agent_id,
            &action.name,
            action.params.clone().unwrap_or(serde_json::Value::Null),
        );
        let result = result.map_err(|err| {
            ScheduledJobFailure::new("schedule_action_handler_failed", err.to_string())
        })?;
        match result {
            Some(serde_json::Value::String(message)) if !message.is_empty() => {
                Ok(format!("completed: {}", message))
            }
            Some(serde_json::Value::Object(map)) => {
                if let Some(status) = map.get("status").and_then(|value| value.as_str())
                    && !status.is_empty()
                {
                    return Ok(format!("completed: {}", status));
                }
                Ok("completed".to_string())
            }
            Some(_) => Ok("completed".to_string()),
            None => Err(ScheduledJobFailure::new(
                "schedule_action_missing_handler",
                format!(
                    "scheduled action '{}' is not defined in the target harness",
                    action.name
                ),
            )),
        }
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
    let action = scheduled_job_action(&row).ok().flatten();
    ScheduleJobDetail {
        id: row.id,
        public_id: public_id.clone(),
        agent_id: row.agent_id,
        kind: row.job_kind,
        prompt: row.prompt,
        content: parse_json(row.content.as_deref()).ok().flatten(),
        tools: parse_json(row.tools.as_deref()).ok().flatten(),
        conflict_policy: row.conflict_policy,
        action,
        persistence,
        next_run_unix_ms: row.next_run_unix_ms,
        interval_seconds: row.interval_seconds,
        recurring_pattern: row.recurring_pattern,
        overlap_policy: row.overlap_policy,
        work_key: row.work_key,
        max_concurrency: row.max_concurrency,
        enabled: row.enabled,
        slot_id: scheduled_job_slot_id(&public_id),
        running_task_id: row.running_task_id,
        active_run_count: row.active_run_count,
        pending_rerun: row.pending_rerun,
        last_run_unix_ms: row.last_run_unix_ms,
        last_status: row.last_status,
        last_error_code: row.last_error_code,
        failure_count: row.failure_count,
        created_at: row.created_at,
        updated_at: row.updated_at,
    }
}

fn map_scheduled_job_run_detail(row: ScheduledJobRunRow) -> ScheduleJobRunDetail {
    ScheduleJobRunDetail {
        id: row.id,
        task_id: row.task_id,
        started_unix_ms: row.started_unix_ms,
        finished_unix_ms: row.finished_unix_ms,
        duration_ms: row
            .finished_unix_ms
            .and_then(|finished| finished.checked_sub(row.started_unix_ms))
            .map(|duration| duration as u64),
        last_status: row.last_status,
        active: row.finished_unix_ms.is_none(),
        created_at: row.created_at,
        updated_at: row.updated_at,
    }
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
struct ScheduledWorklistActionParams {
    name: String,
    scope: Option<String>,
    #[serde(default)]
    store: Option<JsonValue>,
    path: Option<String>,
    #[serde(rename = "where", default)]
    where_filter: Option<JsonMap<String, JsonValue>>,
    stale_after_seconds: Option<u64>,
    limit: Option<usize>,
}

fn scheduled_worklist_action_params(
    action: &ScheduleActionParams,
) -> Result<ScheduledWorklistActionParams> {
    let params = action.params.clone().unwrap_or(JsonValue::Null);
    match params {
        JsonValue::Object(_) | JsonValue::Null => {
            let parsed = serde_json::from_value::<ScheduledWorklistActionParams>(params)?;
            if parsed.name.is_empty() {
                anyhow::bail!("Scheduled action '{}' requires params.name", action.name);
            }
            Ok(parsed)
        }
        _ => anyhow::bail!(
            "Scheduled action '{}' requires object-like params",
            action.name
        ),
    }
}

fn scheduled_worklist_store_selector(
    params: &ScheduledWorklistActionParams,
) -> Result<StoreSelector> {
    if let Some(path) = params.path.as_deref() {
        return Ok(StoreSelector::Path(path.to_string()));
    }
    if let Some(store) = params.store.as_ref() {
        return store_selector_from_json(store);
    }
    Ok(StoreSelector::Alias("state".to_string()))
}

fn store_selector_from_json(value: &JsonValue) -> Result<StoreSelector> {
    match value {
        JsonValue::String(s) => Ok(parse_store_selector_string(s)),
        JsonValue::Object(map) => {
            if let Some(path) = map.get("path").and_then(|value| value.as_str()) {
                return Ok(StoreSelector::Path(path.to_string()));
            }
            if let Some(store) = map.get("store").and_then(|value| value.as_str()) {
                return Ok(StoreSelector::Alias(store.to_string()));
            }
            if let Some(alias) = map.get("alias").and_then(|value| value.as_str()) {
                return Ok(StoreSelector::Alias(alias.to_string()));
            }
            anyhow::bail!("invalid store selector object for worklist action")
        }
        _ => anyhow::bail!("invalid store selector for worklist action"),
    }
}

fn parse_store_selector_string(s: &str) -> StoreSelector {
    if s.contains('/')
        || s.contains('\\')
        || s.starts_with('.')
        || s.ends_with(".db")
        || s.starts_with('~')
    {
        StoreSelector::Path(s.to_string())
    } else {
        StoreSelector::Alias(s.to_string())
    }
}

fn format_work_item_public_id(bytes: &[u8]) -> String {
    uuid::Uuid::from_slice(bytes)
        .map(|uuid| uuid.to_string())
        .unwrap_or_else(|_| super::helpers::format_uuid_bytes_simple(bytes))
}

fn work_item_filter_value(row: &WorkItemRow, metadata: &JsonValue, key: &str) -> Option<JsonValue> {
    match key {
        "id" | "public_id" => Some(JsonValue::String(format_work_item_public_id(
            &row.public_id,
        ))),
        "title" => Some(JsonValue::String(row.title.clone())),
        "kind" => Some(JsonValue::String(row.item_kind.clone())),
        "status" => Some(JsonValue::String(row.status.clone())),
        "priority" => Some(JsonValue::Number(row.priority.into())),
        "parent_id" => Some(
            row.parent_item_id
                .map(|value| JsonValue::Number(value.into()))
                .unwrap_or(JsonValue::Null),
        ),
        _ => metadata.get(key).cloned(),
    }
}

fn work_item_matches_where(
    row: &WorkItemRow,
    where_map: Option<&JsonMap<String, JsonValue>>,
) -> bool {
    let Some(where_map) = where_map else {
        return true;
    };
    let metadata = row
        .metadata
        .as_deref()
        .and_then(|raw| serde_json::from_str::<JsonValue>(raw).ok())
        .unwrap_or(JsonValue::Null);
    where_map.iter().all(|(key, expected)| {
        work_item_filter_value(row, &metadata, key).as_ref() == Some(expected)
    })
}

fn work_item_dependencies_satisfied(
    row: &WorkItemRow,
    status_map: &std::collections::HashMap<String, String>,
) -> bool {
    row.after_ids
        .as_deref()
        .and_then(|raw| serde_json::from_str::<Vec<String>>(raw).ok())
        .unwrap_or_default()
        .into_iter()
        .all(|dep| status_map.get(&dep).is_some_and(|status| status == "done"))
}

fn work_item_is_orphaned(row: &WorkItemRow, stale_before_unix_ms: i64) -> bool {
    row.status == "active"
        && match row.claim_heartbeat_unix_ms {
            Some(heartbeat) => heartbeat <= stale_before_unix_ms,
            None => true,
        }
}

fn work_item_is_paused(row: &WorkItemRow, now_unix_ms: i64) -> bool {
    let Some(metadata_raw) = row.metadata.as_deref() else {
        return false;
    };
    let Ok(JsonValue::Object(map)) = serde_json::from_str::<JsonValue>(metadata_raw) else {
        return false;
    };
    if !work_item_paused(row) {
        return false;
    }
    match map
        .get("pause_until_unix_ms")
        .and_then(|value| value.as_i64())
    {
        Some(pause_until_unix_ms) => pause_until_unix_ms > now_unix_ms,
        None => true,
    }
}

fn work_item_pause_flag(row: &WorkItemRow) -> bool {
    let Some(metadata_raw) = row.metadata.as_deref() else {
        return false;
    };
    let Ok(JsonValue::Object(map)) = serde_json::from_str::<JsonValue>(metadata_raw) else {
        return false;
    };
    map.get("paused")
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
}

fn work_item_paused(row: &WorkItemRow) -> bool {
    row.status == "paused" || (row.status == "pending" && work_item_pause_flag(row))
}

fn work_item_claimable_now(row: &WorkItemRow, now_unix_ms: i64) -> bool {
    match row.status.as_str() {
        "pending" => !work_item_is_paused(row, now_unix_ms),
        "paused" => work_item_pause_due(row, now_unix_ms),
        _ => false,
    }
}

fn work_item_pause_due(row: &WorkItemRow, now_unix_ms: i64) -> bool {
    if !work_item_paused(row) {
        return false;
    }
    let Some(metadata_raw) = row.metadata.as_deref() else {
        return false;
    };
    let Ok(JsonValue::Object(map)) = serde_json::from_str::<JsonValue>(metadata_raw) else {
        return false;
    };
    match map
        .get("pause_until_unix_ms")
        .and_then(|value| value.as_i64())
    {
        Some(pause_until_unix_ms) => pause_until_unix_ms <= now_unix_ms,
        None => false,
    }
}

fn work_item_task(row: &WorkItemRow) -> Result<QueuedTask> {
    let mut task = QueuedTask::ad_hoc(
        row.prompt
            .clone()
            .ok_or_else(|| anyhow!("Prompt work item '{}' is missing prompt", row.title))?,
    );
    task.title = Some(row.title.clone());
    task.content = parse_json(row.content.as_deref())?;
    if let Some(tools) = parse_json::<ToolsConfig>(row.tools.as_deref())?
        && !tools.is_empty()
    {
        task.tools = Some(tools);
    }
    task.conflict_policy = match row.conflict_policy.as_deref() {
        Some(conflict_policy) => Some(
            conflict_policy
                .parse::<ExecutionConflictPolicy>()
                .map_err(anyhow::Error::msg)?,
        ),
        None => None,
    };
    Ok(task)
}

fn scheduled_job_task(job: &ScheduledJobRow) -> Result<QueuedTask> {
    let mut task = QueuedTask::ad_hoc(
        job.prompt
            .clone()
            .ok_or_else(|| anyhow!("Prompt job '{}' is missing prompt text", job.id))?,
    );
    task.content = parse_json(job.content.as_deref())?;
    if let Some(tools) = parse_json::<ToolsConfig>(job.tools.as_deref())?
        && !tools.is_empty()
    {
        task.tools = Some(tools);
    }
    task.conflict_policy = match job.conflict_policy.as_deref() {
        Some(conflict_policy) => Some(conflict_policy.parse().map_err(anyhow::Error::msg)?),
        None => None,
    };
    Ok(task)
}

fn scheduled_job_action(job: &ScheduledJobRow) -> Result<Option<ScheduleActionParams>> {
    let Some(name) = job.action_name.clone() else {
        return Ok(None);
    };
    Ok(Some(ScheduleActionParams {
        name,
        params: parse_json(job.action_params.as_deref())?,
    }))
}

fn validate_scheduled_job_payload(
    prompt: Option<&String>,
    action: Option<&ScheduleActionParams>,
) -> Result<ScheduledJobKind> {
    match (prompt, action) {
        (Some(_), None) => Ok(ScheduledJobKind::Prompt),
        (None, Some(_)) => Ok(ScheduledJobKind::Action),
        (Some(_), Some(_)) => {
            anyhow::bail!("Scheduled job cannot define both prompt and action payloads")
        }
        (None, None) => anyhow::bail!("Scheduled job requires prompt or action payload"),
    }
}

fn validate_scheduled_job_recurrence(
    interval_seconds: Option<u64>,
    recurring_pattern: Option<&str>,
) -> Result<()> {
    if interval_seconds.is_some() && recurring_pattern.is_some() {
        anyhow::bail!("scheduled job cannot define both interval_seconds and recurring_pattern");
    }
    if let Some(pattern) = recurring_pattern {
        let _ = pattern.parse::<ScheduledJobRecurringPattern>()?;
    }
    Ok(())
}

fn required_action_id(action: &ScheduleActionParams) -> Result<String> {
    action
        .params
        .as_ref()
        .and_then(|value| value.get("id"))
        .and_then(|value| value.as_str())
        .map(|value| value.to_string())
        .ok_or_else(|| anyhow!("Scheduled action '{}' requires params.id", action.name))
}

fn now_unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis() as i64
}

fn advance_recurring_due(next_run_unix_ms: i64, step_ms: i64, now_unix_ms: i64) -> i64 {
    let step = step_ms.max(1_000);
    let mut next = next_run_unix_ms;
    while next <= now_unix_ms {
        next = next.saturating_add(step);
    }
    next
}

fn next_recurring_due(job: &ScheduledJobRow, now_unix_ms: i64) -> Result<Option<i64>> {
    if let Some(interval_seconds) = job.interval_seconds {
        return Ok(Some(advance_recurring_due(
            job.next_run_unix_ms,
            (interval_seconds as i64).saturating_mul(1000),
            now_unix_ms,
        )));
    }
    if let Some(pattern) = job.recurring_pattern.as_deref() {
        let step_ms = pattern.parse::<ScheduledJobRecurringPattern>()?.step_ms();
        return Ok(Some(advance_recurring_due(
            job.next_run_unix_ms,
            step_ms,
            now_unix_ms,
        )));
    }
    Ok(None)
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

fn parse_json<T>(raw: Option<&str>) -> Result<Option<T>>
where
    T: serde::de::DeserializeOwned,
{
    raw.map(serde_json::from_str)
        .transpose()
        .map_err(anyhow::Error::from)
}

fn serialize_json<T>(value: Option<&T>) -> Result<Option<String>>
where
    T: serde::Serialize,
{
    value
        .map(serde_json::to_string)
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
