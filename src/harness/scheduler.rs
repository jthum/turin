use std::sync::Arc;

use anyhow::{Result, anyhow};
use tokio::sync::Notify;
use turin_daemon_protocol::{
    ScheduleActionParams, ScheduleCreateParams, ScheduleJobDetail, ScheduleJobRunList,
    ScheduleUpdateParams,
};

use crate::persistence::state::{ScheduledJobInsert, ScheduledJobUpdate, StateStore};
use crate::schedule_support::{
    map_scheduled_job_detail, map_scheduled_job_run_detail, parse_json, scheduled_job_action,
    scheduled_job_persistence, scheduled_job_public_id, serialize_json, serialize_store_target,
};

#[derive(Clone)]
pub struct HarnessSchedulerAccess {
    runtime_store: Arc<StateStore>,
    wake: Option<Arc<Notify>>,
}

impl HarnessSchedulerAccess {
    pub fn new(runtime_store: Arc<StateStore>, wake: Option<Arc<Notify>>) -> Self {
        Self {
            runtime_store,
            wake,
        }
    }

    pub fn runtime_store(&self) -> Arc<StateStore> {
        Arc::clone(&self.runtime_store)
    }

    pub async fn create_job(&self, params: ScheduleCreateParams) -> Result<ScheduleJobDetail> {
        let public_id = uuid::Uuid::now_v7();
        let job_kind = validate_schedule_payload(params.prompt.as_ref(), params.action.as_ref())?;
        validate_schedule_recurrence(params.interval_seconds, params.recurring_pattern.as_deref())?;
        let content = serialize_json(params.content.as_ref())?;
        let tools = serialize_json(params.tools.as_ref())?;
        let action_name = params.action.as_ref().map(|action| action.name.as_str());
        let action_params = serialize_json(
            params
                .action
                .as_ref()
                .and_then(|action| action.params.as_ref()),
        )?;
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
            .runtime_store
            .create_scheduled_job(ScheduledJobInsert {
                public_id,
                agent_id: &params.agent_id,
                job_kind,
                prompt: params.prompt.as_deref(),
                content: content.as_deref(),
                tools: tools.as_deref(),
                conflict_policy: params.conflict_policy.as_deref(),
                action_name,
                action_params: action_params.as_deref(),
                state_target: state_target.as_deref(),
                store_target: store_target.as_deref(),
                next_run_unix_ms: params.next_run_unix_ms,
                interval_seconds: params.interval_seconds,
                recurring_pattern: params.recurring_pattern.as_deref(),
                overlap_policy: params.overlap_policy.as_deref().unwrap_or("skip"),
                work_key: params.work_key.as_deref(),
                max_concurrency: params.max_concurrency,
                enabled: params.enabled,
            })
            .await?;
        if let Some(wake) = &self.wake {
            wake.notify_one();
        }
        let job = self
            .runtime_store
            .list_scheduled_jobs()
            .await?
            .into_iter()
            .find(|row| row.id == id)
            .ok_or_else(|| anyhow!("Scheduled job '{}' was created but not visible", id))?;
        map_scheduled_job_detail(job)
    }

    pub async fn list_jobs(&self) -> Result<Vec<ScheduleJobDetail>> {
        self.runtime_store
            .list_scheduled_jobs()
            .await?
            .into_iter()
            .map(map_scheduled_job_detail)
            .collect()
    }

    pub async fn get_job(&self, public_id: &str) -> Result<Option<ScheduleJobDetail>> {
        let public_id = uuid::Uuid::parse_str(public_id)?;
        self.runtime_store
            .get_scheduled_job_by_public_id(public_id)
            .await?
            .map(map_scheduled_job_detail)
            .transpose()
    }

    pub async fn list_job_runs(
        &self,
        public_id: &str,
        active_only: bool,
        limit: Option<u32>,
    ) -> Result<Option<ScheduleJobRunList>> {
        let public_id = uuid::Uuid::parse_str(public_id)?;
        let Some(row) = self
            .runtime_store
            .get_scheduled_job_by_public_id(public_id)
            .await?
        else {
            return Ok(None);
        };
        let public_id = scheduled_job_public_id(&row.public_id);
        let runs = self
            .runtime_store
            .list_scheduled_job_runs(row.id, active_only, limit)
            .await?
            .into_iter()
            .map(map_scheduled_job_run_detail)
            .collect();
        Ok(Some(ScheduleJobRunList { public_id, runs }))
    }

    pub async fn update_job(
        &self,
        params: ScheduleUpdateParams,
    ) -> Result<Option<ScheduleJobDetail>> {
        let public_id = uuid::Uuid::parse_str(&params.id)?;
        let Some(row) = self
            .runtime_store
            .get_scheduled_job_by_public_id(public_id)
            .await?
        else {
            return Ok(None);
        };
        let persistence = params
            .persistence
            .or_else(|| scheduled_job_persistence(&row).ok().flatten());
        let job_kind =
            match validate_schedule_payload(params.prompt.as_ref(), params.action.as_ref()) {
                Ok(kind) => kind,
                Err(_) if params.prompt.is_none() && params.action.is_none() => {
                    row.job_kind.as_str()
                }
                Err(err) => return Err(err),
            };
        validate_schedule_recurrence(
            params.interval_seconds.or(row.interval_seconds),
            params
                .recurring_pattern
                .as_deref()
                .or(row.recurring_pattern.as_deref()),
        )?;
        let prompt = if job_kind == "prompt" {
            params.prompt.or_else(|| row.prompt.clone())
        } else {
            None
        };
        let content = match params.content {
            Some(content) => Some(content),
            None => parse_json(row.content.as_deref())?,
        };
        let tools = match params.tools {
            Some(tools) => Some(tools),
            None => parse_json(row.tools.as_deref())?,
        };
        let conflict_policy = params.conflict_policy.or(row.conflict_policy.clone());
        let action = params
            .action
            .or_else(|| scheduled_job_action(&row).ok().flatten());
        let content_json = serialize_json(content.as_ref())?;
        let tools_json = serialize_json(tools.as_ref())?;
        let action_name = action.as_ref().map(|action| action.name.as_str());
        let action_params =
            serialize_json(action.as_ref().and_then(|action| action.params.as_ref()))?;
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
        self.runtime_store
            .update_scheduled_job(ScheduledJobUpdate {
                id: row.id,
                agent_id: params.agent_id.as_deref().unwrap_or(&row.agent_id),
                job_kind,
                prompt: prompt.as_deref(),
                content: content_json.as_deref(),
                tools: tools_json.as_deref(),
                conflict_policy: conflict_policy.as_deref(),
                action_name,
                action_params: action_params.as_deref(),
                state_target: state_target.as_deref(),
                store_target: store_target.as_deref(),
                next_run_unix_ms: params.next_run_unix_ms.unwrap_or(row.next_run_unix_ms),
                interval_seconds: params.interval_seconds.or(row.interval_seconds),
                recurring_pattern: params
                    .recurring_pattern
                    .as_deref()
                    .or(row.recurring_pattern.as_deref()),
                overlap_policy: params
                    .overlap_policy
                    .as_deref()
                    .unwrap_or(row.overlap_policy.as_str()),
                work_key: params.work_key.as_deref().or(row.work_key.as_deref()),
                max_concurrency: params.max_concurrency.or(row.max_concurrency),
                enabled: params.enabled.unwrap_or(row.enabled),
            })
            .await?;
        if let Some(wake) = &self.wake {
            wake.notify_one();
        }
        self.runtime_store
            .get_scheduled_job_by_public_id(public_id)
            .await?
            .map(map_scheduled_job_detail)
            .transpose()
    }

    pub async fn set_job_enabled(
        &self,
        public_id: &str,
        enabled: bool,
    ) -> Result<Option<ScheduleJobDetail>> {
        let public_id = uuid::Uuid::parse_str(public_id)?;
        let Some(row) = self
            .runtime_store
            .get_scheduled_job_by_public_id(public_id)
            .await?
        else {
            return Ok(None);
        };
        self.runtime_store
            .set_scheduled_job_enabled(row.id, enabled)
            .await?;
        if let Some(wake) = &self.wake {
            wake.notify_one();
        }
        self.runtime_store
            .get_scheduled_job_by_public_id(public_id)
            .await?
            .map(map_scheduled_job_detail)
            .transpose()
    }

    pub async fn delete_job(&self, public_id: &str) -> Result<Option<ScheduleJobDetail>> {
        let public_id = uuid::Uuid::parse_str(public_id)?;
        let Some(row) = self
            .runtime_store
            .get_scheduled_job_by_public_id(public_id)
            .await?
        else {
            return Ok(None);
        };
        if row.active_run_count > 0 {
            anyhow::bail!(
                "Cannot delete scheduled job '{}' while it has an active run",
                public_id
            );
        }
        let detail = map_scheduled_job_detail(row.clone())?;
        self.runtime_store.delete_scheduled_job(row.id).await?;
        if let Some(wake) = &self.wake {
            wake.notify_one();
        }
        Ok(Some(detail))
    }
}

fn validate_schedule_recurrence(
    interval_seconds: Option<u64>,
    recurring_pattern: Option<&str>,
) -> Result<()> {
    if interval_seconds.is_some() && recurring_pattern.is_some() {
        anyhow::bail!("scheduled job cannot define both interval_seconds and recurring_pattern");
    }
    if let Some(pattern) = recurring_pattern {
        match pattern {
            "daily" | "weekly" => {}
            _ => anyhow::bail!(
                "unsupported recurring pattern '{}'; expected 'daily' or 'weekly'",
                pattern
            ),
        }
    }
    Ok(())
}

fn validate_schedule_payload(
    prompt: Option<&String>,
    action: Option<&ScheduleActionParams>,
) -> Result<&'static str> {
    match (prompt, action) {
        (Some(_), None) => Ok("prompt"),
        (None, Some(_)) => Ok("action"),
        (Some(_), Some(_)) => {
            anyhow::bail!("Scheduled job cannot define both prompt and action payloads")
        }
        (None, None) => anyhow::bail!("Scheduled job requires prompt or action payload"),
    }
}
