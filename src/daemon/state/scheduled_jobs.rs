use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use serde::Serialize;
use turin_daemon_protocol::{
    ContextPersistenceParams, ScheduleActionParams, ScheduleJobDetail, ScheduleJobRunList,
};
use turin_types::{TaskInputContent, ToolsConfig};

use super::DaemonState;
use crate::persistence::state::{ScheduledJobInsert, ScheduledJobUpdate};
use crate::schedule_support::{
    map_scheduled_job_detail, map_scheduled_job_run_detail, parse_json, scheduled_job_action,
    scheduled_job_persistence, scheduled_job_public_id, serialize_json, serialize_store_target,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ScheduledJobOverlapPolicy {
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
pub(crate) struct CreateScheduledJobInput {
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
pub(crate) struct UpdateScheduledJobInput {
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
pub(super) enum ScheduledJobKind {
    Prompt,
    Action,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ScheduledJobRecurringPattern {
    Daily,
    Weekly,
}

impl ScheduledJobRecurringPattern {
    pub(super) fn step_ms(self) -> i64 {
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
                std::sync::Arc::clone(&self.runtime_store),
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
        let store = Arc::clone(&self.runtime_store);
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
        map_scheduled_job_detail(job)
    }

    pub(crate) async fn list_scheduled_jobs(&self) -> Result<Vec<ScheduleJobDetail>> {
        let store = Arc::clone(&self.runtime_store);
        store
            .list_scheduled_jobs()
            .await?
            .into_iter()
            .map(map_scheduled_job_detail)
            .collect()
    }

    pub(crate) async fn update_scheduled_job(
        &self,
        public_id: &str,
        input: UpdateScheduledJobInput,
    ) -> Result<Option<ScheduleJobDetail>> {
        let store = Arc::clone(&self.runtime_store);
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
        let overlap_policy = match input.overlap_policy {
            Some(policy) => policy,
            None => row
                .overlap_policy
                .parse::<ScheduledJobOverlapPolicy>()
                .with_context(|| {
                    format!(
                        "Scheduled job '{}' has invalid overlap policy '{}'",
                        row.id, row.overlap_policy
                    )
                })?,
        }
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
        store
            .get_scheduled_job_by_public_id(public_id)
            .await?
            .map(map_scheduled_job_detail)
            .transpose()
    }

    pub(crate) async fn scheduled_job_detail(
        &self,
        public_id: &str,
    ) -> Result<Option<ScheduleJobDetail>> {
        let store = Arc::clone(&self.runtime_store);
        let public_id = uuid::Uuid::parse_str(public_id)?;
        store
            .get_scheduled_job_by_public_id(public_id)
            .await?
            .map(map_scheduled_job_detail)
            .transpose()
    }

    pub(crate) async fn set_scheduled_job_enabled(
        &self,
        public_id: &str,
        enabled: bool,
    ) -> Result<Option<ScheduleJobDetail>> {
        let store = Arc::clone(&self.runtime_store);
        let public_id = uuid::Uuid::parse_str(public_id)?;
        let Some(row) = store.get_scheduled_job_by_public_id(public_id).await? else {
            return Ok(None);
        };
        store.set_scheduled_job_enabled(row.id, enabled).await?;
        if let Some(wake) = &self.scheduler_wake {
            wake.notify_one();
        }
        store
            .get_scheduled_job_by_public_id(public_id)
            .await?
            .map(map_scheduled_job_detail)
            .transpose()
    }

    pub(crate) async fn scheduled_job_runs(
        &self,
        public_id: &str,
        active_only: bool,
        limit: Option<u32>,
    ) -> Result<Option<ScheduleJobRunList>> {
        let store = Arc::clone(&self.runtime_store);
        let public_id = uuid::Uuid::parse_str(public_id)?;
        let Some(row) = store.get_scheduled_job_by_public_id(public_id).await? else {
            return Ok(None);
        };
        let public_id = scheduled_job_public_id(&row.public_id);
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
        let store = Arc::clone(&self.runtime_store);
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
        let detail = map_scheduled_job_detail(row.clone())?;
        store.delete_scheduled_job(row.id).await?;
        if let Some(wake) = &self.scheduler_wake {
            wake.notify_one();
        }
        Ok(Some(detail))
    }
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
