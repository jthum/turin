use anyhow::{Context, Result};
use turin_daemon_protocol::{
    ContextPersistenceParams, ScheduleActionParams, ScheduleJobDetail, ScheduleJobRunDetail,
    StoreTargetParams,
};

use crate::kernel::config::StoreTargetConfig;
use crate::persistence::schema::{ScheduledJobRow, ScheduledJobRunRow};

pub(crate) fn map_scheduled_job_detail(row: ScheduledJobRow) -> Result<ScheduleJobDetail> {
    anyhow::ensure!(
        matches!(row.job_kind.as_str(), "prompt" | "action"),
        "Scheduled job '{}' has invalid kind '{}'",
        row.id,
        row.job_kind
    );
    anyhow::ensure!(
        matches!(row.overlap_policy.as_str(), "skip" | "queue" | "parallel"),
        "Scheduled job '{}' has invalid overlap policy '{}'",
        row.id,
        row.overlap_policy
    );
    if let Some(pattern) = row.recurring_pattern.as_deref() {
        anyhow::ensure!(
            matches!(pattern, "daily" | "weekly"),
            "Scheduled job '{}' has invalid recurring pattern '{}'",
            row.id,
            pattern
        );
    }

    let public_id = scheduled_job_public_id(&row.public_id);
    let persistence = scheduled_job_persistence(&row)
        .with_context(|| format!("Scheduled job '{}' has invalid persistence", row.id))?;
    let action = scheduled_job_action(&row)
        .with_context(|| format!("Scheduled job '{}' has invalid action", row.id))?;
    let content = parse_json(row.content.as_deref())
        .with_context(|| format!("Scheduled job '{}' has invalid content", row.id))?;
    let tools = parse_json(row.tools.as_deref())
        .with_context(|| format!("Scheduled job '{}' has invalid tools", row.id))?;
    Ok(ScheduleJobDetail {
        id: row.id,
        public_id: public_id.clone(),
        agent_id: row.agent_id,
        kind: row.job_kind,
        prompt: row.prompt,
        content,
        tools,
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
    })
}

pub(crate) fn map_scheduled_job_run_detail(row: ScheduledJobRunRow) -> ScheduleJobRunDetail {
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

pub(crate) fn scheduled_job_public_id(bytes: &[u8]) -> String {
    uuid::Uuid::from_slice(bytes)
        .map(|uuid| uuid.to_string())
        .unwrap_or_else(|_| {
            let mut out = String::with_capacity(bytes.len() * 2);
            for byte in bytes {
                use std::fmt::Write as _;
                let _ = write!(&mut out, "{:02x}", byte);
            }
            out
        })
}

pub(crate) fn scheduled_job_slot_id(public_id: &str) -> String {
    format!("sched_{}", public_id.replace('-', ""))
}

pub(crate) fn store_target_config_from_params(params: &StoreTargetParams) -> StoreTargetConfig {
    StoreTargetConfig {
        path: params.path.clone(),
        alias: params.alias.clone(),
    }
}

pub(crate) fn serialize_store_target(target: Option<&StoreTargetParams>) -> Result<Option<String>> {
    target
        .map(serde_json::to_string)
        .transpose()
        .map_err(anyhow::Error::from)
}

pub(crate) fn parse_store_target(raw: Option<&str>) -> Result<Option<StoreTargetParams>> {
    raw.map(serde_json::from_str)
        .transpose()
        .map_err(anyhow::Error::from)
}

pub(crate) fn parse_json<T>(raw: Option<&str>) -> Result<Option<T>>
where
    T: serde::de::DeserializeOwned,
{
    raw.map(serde_json::from_str)
        .transpose()
        .map_err(anyhow::Error::from)
}

pub(crate) fn serialize_json<T>(value: Option<&T>) -> Result<Option<String>>
where
    T: serde::Serialize,
{
    value
        .map(serde_json::to_string)
        .transpose()
        .map_err(anyhow::Error::from)
}

pub(crate) fn scheduled_job_persistence(
    job: &ScheduledJobRow,
) -> Result<Option<ContextPersistenceParams>> {
    let state = parse_store_target(job.state_target.as_deref())?;
    let store = parse_store_target(job.store_target.as_deref())?;
    if state.is_none() && store.is_none() {
        return Ok(None);
    }
    Ok(Some(ContextPersistenceParams { state, store }))
}

pub(crate) fn scheduled_job_action(job: &ScheduledJobRow) -> Result<Option<ScheduleActionParams>> {
    let Some(name) = job.action_name.clone() else {
        return Ok(None);
    };
    Ok(Some(ScheduleActionParams {
        name,
        params: parse_json(job.action_params.as_deref())?,
    }))
}
