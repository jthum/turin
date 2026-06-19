use std::collections::BTreeMap;

use serde_json::Value;
use turin_daemon_protocol::{WorkItemDetail, WorkItemList};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WorklistStatusCounts {
    pub pending: usize,
    pub claimed: usize,
    pub done: usize,
    pub failed: usize,
    pub other: usize,
}

impl WorklistStatusCounts {
    pub fn total(self) -> usize {
        self.pending + self.claimed + self.done + self.failed + self.other
    }
}

pub fn worklist_status_counts(items: &WorkItemList) -> WorklistStatusCounts {
    let mut counts = WorklistStatusCounts::default();
    for item in &items.items {
        match item.status.as_str() {
            "pending" => counts.pending += 1,
            "claimed" => counts.claimed += 1,
            "done" => counts.done += 1,
            "failed" => counts.failed += 1,
            _ => counts.other += 1,
        }
    }
    counts
}

pub fn worklist_highest_priority_pending_item(items: &WorkItemList) -> Option<&WorkItemDetail> {
    items
        .items
        .iter()
        .filter(|item| item.status == "pending")
        .max_by_key(|item| item.priority)
}

pub fn worklist_chart_group_field(intent: Option<&str>) -> &'static str {
    match intent {
        Some("kind_breakdown") => "kind",
        Some("priority_breakdown") => "priority",
        _ => "status",
    }
}

pub fn worklist_chart_group_label(intent: Option<&str>) -> &'static str {
    match worklist_chart_group_field(intent) {
        "kind" => "Kind",
        "priority" => "Priority",
        _ => "Status",
    }
}

pub fn worklist_group_counts(items: &WorkItemList, field: &str) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for item in &items.items {
        let label = work_item_field_label(item, field);
        let label = if label.is_empty() {
            "unknown".to_string()
        } else {
            label
        };
        *counts.entry(label).or_insert(0) += 1;
    }
    counts
}

pub fn work_item_field_label(item: &WorkItemDetail, field: &str) -> String {
    match field {
        "id" | "public_id" => work_item_key(item),
        "internal_id" => item.id.to_string(),
        "worklist_id" => item.worklist_id.clone(),
        "parent_id" => item.parent_id.clone().unwrap_or_default(),
        "title" => item.title.clone(),
        "kind" => item.kind.clone(),
        "status" => item.status.clone(),
        "paused" => item.paused.to_string(),
        "priority" => item.priority.to_string(),
        "claim_agent_id" | "agent" => item.claim_agent_id.clone().unwrap_or_default(),
        "pause_reason" => item.pause_reason.clone().unwrap_or_default(),
        "prompt" => item.prompt.clone().unwrap_or_default(),
        other => item
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get(other))
            .map(value_label)
            .unwrap_or_default(),
    }
}

pub fn work_item_key(item: &WorkItemDetail) -> String {
    if item.public_id.is_empty() {
        item.id.to_string()
    } else {
        item.public_id.clone()
    }
}

pub fn work_item_matches_key(item: &WorkItemDetail, key: &str) -> bool {
    if item.public_id == key {
        return true;
    }
    key.parse::<i64>().ok() == Some(item.id)
}

pub fn work_item_index_by_key(items: &WorkItemList, key: Option<&str>) -> Option<usize> {
    let key = key?;
    items
        .items
        .iter()
        .position(|item| work_item_matches_key(item, key))
}

fn value_label(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => value.clone(),
        Value::Array(_) | Value::Object(_) => {
            serde_json::to_string(value).unwrap_or_else(|_| String::new())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn status_counts_track_known_and_other_states() {
        let items = WorkItemList {
            worklist_id: "release".to_string(),
            items: vec![
                test_work_item(1, "REL-1", "pending", "approval"),
                test_work_item(2, "REL-2", "claimed", "approval"),
                test_work_item(3, "REL-3", "done", "qa"),
                test_work_item(4, "REL-4", "failed", "qa"),
                test_work_item(5, "REL-5", "paused", "ops"),
            ],
        };

        let counts = worklist_status_counts(&items);

        assert_eq!(counts.pending, 1);
        assert_eq!(counts.claimed, 1);
        assert_eq!(counts.done, 1);
        assert_eq!(counts.failed, 1);
        assert_eq!(counts.other, 1);
        assert_eq!(counts.total(), 5);
    }

    #[test]
    fn highest_priority_pending_item_ignores_non_pending_items() {
        let items = WorkItemList {
            worklist_id: "release".to_string(),
            items: vec![
                test_work_item(1, "REL-1", "done", "approval"),
                test_work_item(2, "REL-2", "pending", "approval"),
                test_work_item(3, "REL-3", "pending", "qa"),
                test_work_item(10, "REL-10", "claimed", "ops"),
            ],
        };

        let item = worklist_highest_priority_pending_item(&items).expect("pending item");

        assert_eq!(item.public_id, "REL-3");
    }

    #[test]
    fn grouping_uses_work_item_fields_and_metadata() {
        let mut first = test_work_item(1, "REL-1", "pending", "approval");
        first.metadata = Some(json!({ "lane": "release" }));
        let mut second = test_work_item(2, "REL-2", "done", "approval");
        second.metadata = Some(json!({ "lane": "release" }));
        let mut third = test_work_item(3, "REL-3", "done", "qa");
        third.metadata = Some(json!({ "lane": "qa" }));
        let items = WorkItemList {
            worklist_id: "release".to_string(),
            items: vec![first, second, third],
        };

        assert_eq!(worklist_group_counts(&items, "kind")["approval"], 2);
        assert_eq!(worklist_group_counts(&items, "status")["done"], 2);
        assert_eq!(worklist_group_counts(&items, "lane")["release"], 2);
        assert_eq!(
            worklist_chart_group_field(Some("priority_breakdown")),
            "priority"
        );
        assert_eq!(
            worklist_chart_group_label(Some("priority_breakdown")),
            "Priority"
        );
    }

    #[test]
    fn field_labels_prefer_public_ids_for_ui_display() {
        let mut item = test_work_item(42, "REL-42", "pending", "approval");
        item.metadata = Some(json!({ "release": "2026.06" }));

        assert_eq!(work_item_key(&item), "REL-42");
        assert_eq!(work_item_field_label(&item, "id"), "REL-42");
        assert_eq!(work_item_field_label(&item, "internal_id"), "42");
        assert_eq!(work_item_field_label(&item, "release"), "2026.06");
    }

    #[test]
    fn work_item_matching_accepts_public_or_numeric_keys() {
        let item = test_work_item(42, "REL-42", "pending", "approval");

        assert!(work_item_matches_key(&item, "REL-42"));
        assert!(work_item_matches_key(&item, "42"));
        assert!(work_item_matches_key(&item, "042"));
        assert!(!work_item_matches_key(&item, "REL-1"));
    }

    #[test]
    fn work_item_index_by_key_preserves_selection_after_reorder() {
        let items = WorkItemList {
            worklist_id: "release".to_string(),
            items: vec![
                test_work_item(2, "REL-2", "pending", "approval"),
                test_work_item(1, "REL-1", "pending", "approval"),
            ],
        };

        assert_eq!(work_item_index_by_key(&items, Some("REL-1")), Some(1));
        assert_eq!(work_item_index_by_key(&items, Some("2")), Some(0));
        assert_eq!(work_item_index_by_key(&items, Some("missing")), None);
        assert_eq!(work_item_index_by_key(&items, None), None);
    }

    fn test_work_item(id: i64, public_id: &str, status: &str, kind: &str) -> WorkItemDetail {
        WorkItemDetail {
            id,
            public_id: public_id.to_string(),
            worklist_id: "release".to_string(),
            parent_id: None,
            title: format!("{kind} item {id}"),
            kind: kind.to_string(),
            prompt: None,
            content: None,
            tools: None,
            conflict_policy: None,
            action: None,
            status: status.to_string(),
            paused: false,
            pause_reason: None,
            pause_until_unix_ms: None,
            priority: id,
            after: None,
            metadata: None,
            claim_agent_id: None,
            claim_session_id: None,
            claim_execution_id: None,
            claim_heartbeat_unix_ms: None,
            claimed_at: None,
            completed_at: None,
            failure_reason: None,
            created_at: "2026-06-18T00:00:00Z".to_string(),
            updated_at: "2026-06-18T00:00:00Z".to_string(),
        }
    }
}
