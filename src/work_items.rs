use std::collections::HashMap;

use anyhow::{Result, anyhow};
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::kernel::session::{ExecutionConflictPolicy, QueuedTask};
use crate::persistence::schema::WorkItemRow;

pub(crate) enum WorkItemParentId<'a> {
    DatabaseId,
    PublicId(&'a HashMap<i64, String>),
}

pub(crate) fn public_id_string(bytes: &[u8]) -> String {
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

pub(crate) fn work_item_metadata(row: &WorkItemRow) -> Option<JsonValue> {
    row.metadata
        .as_deref()
        .and_then(|raw| serde_json::from_str::<JsonValue>(raw).ok())
}

pub(crate) fn work_item_paused(row: &WorkItemRow) -> bool {
    row.status == "paused"
}

pub(crate) fn work_item_pause_reason(metadata: Option<&JsonValue>) -> Option<String> {
    let Some(JsonValue::Object(map)) = metadata else {
        return None;
    };
    map.get("pause_reason")
        .and_then(|value| value.as_str())
        .map(ToString::to_string)
}

pub(crate) fn work_item_pause_until_unix_ms(metadata: Option<&JsonValue>) -> Option<i64> {
    let Some(JsonValue::Object(map)) = metadata else {
        return None;
    };
    map.get("pause_until_unix_ms")
        .and_then(|value| value.as_i64())
}

pub(crate) fn work_item_pause_due(row: &WorkItemRow, now_unix_ms: i64) -> bool {
    if !work_item_paused(row) {
        return false;
    }
    match work_item_pause_until_unix_ms(work_item_metadata(row).as_ref()) {
        Some(pause_until_unix_ms) => pause_until_unix_ms <= now_unix_ms,
        None => false,
    }
}

pub(crate) fn work_item_is_paused(row: &WorkItemRow, now_unix_ms: i64) -> bool {
    if !work_item_paused(row) {
        return false;
    }
    match work_item_pause_until_unix_ms(work_item_metadata(row).as_ref()) {
        Some(pause_until_unix_ms) => pause_until_unix_ms > now_unix_ms,
        None => true,
    }
}

pub(crate) fn work_item_claimable_now(row: &WorkItemRow, now_unix_ms: i64) -> bool {
    match row.status.as_str() {
        "pending" => !work_item_is_paused(row, now_unix_ms),
        "paused" => work_item_pause_due(row, now_unix_ms),
        _ => false,
    }
}

pub(crate) fn work_item_is_orphaned(row: &WorkItemRow, stale_before_unix_ms: i64) -> bool {
    row.status == "active"
        && match row.claim_heartbeat_unix_ms {
            Some(heartbeat) => heartbeat <= stale_before_unix_ms,
            None => true,
        }
}

pub(crate) fn work_item_dependencies_satisfied(
    row: &WorkItemRow,
    status_map: &HashMap<String, String>,
) -> bool {
    row.after_ids
        .as_deref()
        .and_then(|raw| serde_json::from_str::<Vec<String>>(raw).ok())
        .unwrap_or_default()
        .into_iter()
        .all(|dep| status_map.get(&dep).is_some_and(|status| status == "done"))
}

pub(crate) fn work_item_prompt_task(
    row: &WorkItemRow,
    inherited_trace_id: Option<&str>,
) -> Result<QueuedTask> {
    let mut task = QueuedTask::ad_hoc(
        row.prompt
            .clone()
            .ok_or_else(|| anyhow!("Prompt work item '{}' is missing prompt", row.title))?,
    )
    .with_inherited_trace(inherited_trace_id);
    task.title = Some(row.title.clone());
    task.content = parse_json_opt(row.content.as_deref())?;
    task.tools = parse_json_opt(row.tools.as_deref())?;
    task.conflict_policy = row
        .conflict_policy
        .as_deref()
        .map(str::parse::<ExecutionConflictPolicy>)
        .transpose()
        .map_err(anyhow::Error::msg)?;
    Ok(task)
}

pub(crate) fn work_item_matches_where(
    row: &WorkItemRow,
    where_map: Option<&JsonMap<String, JsonValue>>,
    parent_id: WorkItemParentId<'_>,
) -> bool {
    let Some(where_map) = where_map else {
        return true;
    };
    let metadata = work_item_metadata(row).unwrap_or(JsonValue::Null);
    where_map.iter().all(|(key, expected)| {
        work_item_filter_value(row, &metadata, key, &parent_id).as_ref() == Some(expected)
    })
}

fn work_item_filter_value(
    row: &WorkItemRow,
    metadata: &JsonValue,
    key: &str,
    parent_id: &WorkItemParentId<'_>,
) -> Option<JsonValue> {
    match key {
        "id" | "public_id" => Some(JsonValue::String(public_id_string(&row.public_id))),
        "title" => Some(JsonValue::String(row.title.clone())),
        "kind" => Some(JsonValue::String(row.item_kind.clone())),
        "status" => Some(JsonValue::String(row.status.clone())),
        "paused" => Some(JsonValue::Bool(work_item_paused(row))),
        "pause_reason" => work_item_pause_reason(Some(metadata)).map(JsonValue::String),
        "pause_until_unix_ms" => work_item_pause_until_unix_ms(Some(metadata)).map(JsonValue::from),
        "priority" => Some(JsonValue::Number(row.priority.into())),
        "parent_id" => Some(parent_id_value(row, parent_id)),
        _ => metadata.get(key).cloned(),
    }
}

fn parent_id_value(row: &WorkItemRow, parent_id: &WorkItemParentId<'_>) -> JsonValue {
    match parent_id {
        WorkItemParentId::DatabaseId => row
            .parent_item_id
            .map(JsonValue::from)
            .unwrap_or(JsonValue::Null),
        WorkItemParentId::PublicId(public_ids) => row
            .parent_item_id
            .and_then(|id| public_ids.get(&id).cloned())
            .map(JsonValue::String)
            .unwrap_or(JsonValue::Null),
    }
}

fn parse_json_opt<T>(raw: Option<&str>) -> Result<Option<T>>
where
    T: serde::de::DeserializeOwned,
{
    raw.map(serde_json::from_str)
        .transpose()
        .map_err(anyhow::Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::session::ExecutionConflictPolicy;

    fn prompt_row() -> WorkItemRow {
        WorkItemRow {
            id: 1,
            public_id: uuid::Uuid::now_v7().into_bytes().to_vec(),
            worklist_id: 10,
            parent_item_id: None,
            title: "Review".to_string(),
            item_kind: "prompt".to_string(),
            prompt: Some("Check the report".to_string()),
            content: None,
            tools: Some(r#"{"allow":["shell"]}"#.to_string()),
            conflict_policy: Some("fork_sibling".to_string()),
            action_name: None,
            action_params: None,
            status: "pending".to_string(),
            priority: 0,
            after_ids: None,
            metadata: None,
            claim_agent_id: None,
            claim_session_id: None,
            claim_execution_id: None,
            claim_heartbeat_unix_ms: None,
            claimed_at: None,
            completed_at: None,
            failure_reason: None,
            created_at: "now".to_string(),
            updated_at: "now".to_string(),
        }
    }

    #[test]
    fn work_item_prompt_task_maps_prompt_fields() {
        let task = work_item_prompt_task(&prompt_row(), Some("trace-1")).unwrap();

        assert_eq!(task.title.as_deref(), Some("Review"));
        assert_eq!(task.prompt, "Check the report");
        assert_eq!(task.trace_id, "trace-1");
        assert_eq!(
            task.conflict_policy,
            Some(ExecutionConflictPolicy::ForkSibling)
        );
        assert_eq!(
            task.tools
                .as_ref()
                .and_then(|tools| tools.selection.allow.as_ref())
                .cloned(),
            Some(vec!["shell".to_string()])
        );
    }

    #[test]
    fn work_item_prompt_task_requires_prompt() {
        let mut row = prompt_row();
        row.prompt = None;

        let err = work_item_prompt_task(&row, None).unwrap_err();
        assert!(err.to_string().contains("missing prompt"));
    }
}
