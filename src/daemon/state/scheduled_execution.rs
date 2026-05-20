use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};
use serde_json::{Map as JsonMap, Value as JsonValue};
use turin_daemon_protocol::{ContextPersistenceParams, ScheduleActionParams};
use turin_types::ToolsConfig;

use super::DaemonState;
use super::scheduled_jobs::{
    ScheduledJobKind, ScheduledJobOverlapPolicy, ScheduledJobRecurringPattern,
};
use crate::kernel::agent_manager::TaskStatusSnapshot;
use crate::kernel::config::{ContextPersistenceConfig, InferenceOverrideConfig};
use crate::kernel::session::{ExecutionConflictPolicy, QueuedTask};
use crate::persistence::manager::StoreSelector;
use crate::persistence::schema::{ScheduledJobRow, ScheduledJobRunRow, WorkItemRow};
use crate::persistence::state::StateStore;
use crate::schedule_support::{
    parse_json, scheduled_job_action, scheduled_job_persistence, scheduled_job_public_id,
    scheduled_job_slot_id, store_target_config_from_params,
};
use crate::work_items::{
    WorkItemParentId, public_id_string as format_work_item_public_id, work_item_claimable_now,
    work_item_dependencies_satisfied, work_item_is_orphaned, work_item_matches_where,
};

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

impl DaemonState {
    pub(crate) async fn scheduler_tick(&mut self) -> Result<Option<Duration>> {
        let store = Arc::clone(&self.runtime_store);
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
        store: &Arc<StateStore>,
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
        store: &Arc<StateStore>,
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
        let public_id = scheduled_job_public_id(&job.public_id);
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
        store: &Arc<StateStore>,
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
        store: &Arc<StateStore>,
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
            .filter(|row| {
                work_item_matches_where(
                    row,
                    params.where_filter.as_ref(),
                    WorkItemParentId::DatabaseId,
                )
            })
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
            .filter(|row| {
                work_item_matches_where(
                    row,
                    params.where_filter.as_ref(),
                    WorkItemParentId::DatabaseId,
                )
            })
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

    pub(super) fn resolve_scheduled_job_persistence(
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
