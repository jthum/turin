use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use serde_json::{Map as JsonMap, Value as JsonValue};
use turin_daemon_protocol::ScheduleActionParams;

use super::DaemonState;
use super::scheduled_execution::ScheduledJobFailure;
use crate::kernel::config::InferenceOverrideConfig;
use crate::persistence::manager::StoreSelector;
use crate::persistence::state::StateStore;
use crate::work_items::{
    WorkItemParentId, public_id_string as format_work_item_public_id, work_item_claimable_now,
    work_item_dependencies_satisfied, work_item_is_orphaned, work_item_matches_where,
    work_item_prompt_task,
};

impl DaemonState {
    pub(super) async fn execute_scheduled_worklist_dispatch(
        &mut self,
        agent_id: &str,
        action: &ScheduleActionParams,
    ) -> std::result::Result<String, ScheduledJobFailure> {
        let context = self.open_scheduled_worklist(action).await?;
        let rows = context
            .store
            .list_work_items(context.worklist_id)
            .await
            .map_err(builtin_failed)?;
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
                    context.params.where_filter.as_ref(),
                    WorkItemParentId::DatabaseId,
                )
            })
            .take(context.params.limit.unwrap_or(usize::MAX))
        {
            let claimed = context
                .store
                .try_claim_work_item(row.id, agent_id, None, Some(&execution_id), now_unix_ms)
                .await
                .map_err(builtin_failed)?;
            if !claimed {
                continue;
            }
            let refreshed = context
                .store
                .get_work_item_by_id(row.id)
                .await
                .map_err(builtin_failed)?
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
                            Some(context.selector.clone()),
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
                            work_item_prompt_task(&refreshed, None).map_err(|err| {
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

    pub(super) async fn execute_scheduled_worklist_release_stale(
        &mut self,
        action: &ScheduleActionParams,
    ) -> std::result::Result<String, ScheduledJobFailure> {
        let context = self.open_scheduled_worklist(action).await?;
        let stale_before = now_unix_ms()
            .saturating_sub(context.params.stale_after_seconds.unwrap_or(300) as i64 * 1000);
        let rows = context
            .store
            .list_work_items(context.worklist_id)
            .await
            .map_err(builtin_failed)?;
        let candidates = rows
            .into_iter()
            .filter(|row| row.parent_item_id.is_none())
            .filter(|row| work_item_is_orphaned(row, stale_before))
            .filter(|row| {
                work_item_matches_where(
                    row,
                    context.params.where_filter.as_ref(),
                    WorkItemParentId::DatabaseId,
                )
            })
            .take(context.params.limit.unwrap_or(usize::MAX))
            .collect::<Vec<_>>();
        let mut released = 0usize;
        for row in candidates {
            context
                .store
                .release_work_item(row.id)
                .await
                .map_err(builtin_failed)?;
            released += 1;
        }
        Ok(format!("completed: released {} stale work items", released))
    }

    async fn open_scheduled_worklist(
        &self,
        action: &ScheduleActionParams,
    ) -> std::result::Result<ScheduledWorklistContext, ScheduledJobFailure> {
        let params = scheduled_worklist_action_params(action).map_err(invalid_params)?;
        let selector = scheduled_worklist_store_selector(&params).map_err(invalid_params)?;
        let store = self
            .kernel
            .store_manager()
            .open(&selector)
            .await
            .map_err(builtin_failed)?;
        let worklist = store
            .open_worklist(&params.name, params.scope.as_deref().unwrap_or(""), None)
            .await
            .map_err(builtin_failed)?;
        Ok(ScheduledWorklistContext {
            params,
            selector,
            store,
            worklist_id: worklist.id,
        })
    }
}

struct ScheduledWorklistContext {
    params: ScheduledWorklistActionParams,
    selector: StoreSelector,
    store: Arc<StateStore>,
    worklist_id: i64,
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

fn invalid_params(err: anyhow::Error) -> ScheduledJobFailure {
    ScheduledJobFailure::new("schedule_action_invalid_params", err.to_string())
}

fn builtin_failed(err: anyhow::Error) -> ScheduledJobFailure {
    ScheduledJobFailure::new("schedule_action_builtin_failed", err.to_string())
}

fn now_unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_else(|_| Duration::from_secs(0))
        .as_millis() as i64
}
