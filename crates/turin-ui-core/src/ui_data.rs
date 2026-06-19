use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};
use turin_daemon_protocol::UiNode;

use crate::controller::UiListRequest;

pub const DEFAULT_UI_ACTIVITY_LIMIT: u32 = 12;
pub const DEFAULT_UI_DETAIL_LIMIT: u32 = 25;
pub const DEFAULT_UI_REPORT_LIMIT: u32 = 100;
pub const DEFAULT_UI_CHART_LIMIT: u32 = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UiWorklistSourceError {
    Unsupported,
    MissingName,
}

pub fn is_worklist_ui_source(source: &str) -> bool {
    source.starts_with("worklists.")
}

pub fn is_named_worklist_ui_source(source: &str) -> bool {
    ui_worklist_name_from_source(source).is_ok()
}

pub fn ui_worklist_name_from_source(source: &str) -> Result<&str, UiWorklistSourceError> {
    let name = source
        .strip_prefix("worklists.")
        .ok_or(UiWorklistSourceError::Unsupported)?;
    if name.trim().is_empty() {
        Err(UiWorklistSourceError::MissingName)
    } else {
        Ok(name)
    }
}

pub fn ui_worklist_request(source: &str, limit: u32) -> Option<UiListRequest> {
    is_named_worklist_ui_source(source).then(|| UiListRequest {
        source: source.to_string(),
        filter: Map::new(),
        limit: Some(limit),
    })
}

pub fn ui_list_filter_fields(filter: &Map<String, Value>) -> Vec<String> {
    let mut fields = filter.keys().cloned().collect::<Vec<_>>();
    fields.sort();
    fields
}

pub fn ui_list_sort_fields(sort: &[String]) -> Vec<String> {
    sort.iter()
        .map(|entry| ui_sort_entry_field(entry))
        .filter(|field| !field.is_empty())
        .map(str::to_string)
        .collect()
}

pub fn ui_sort_entry_field(entry: &str) -> &str {
    let entry = entry
        .trim()
        .trim_start_matches(|ch| ch == '+' || ch == '-')
        .trim();
    let entry = entry.split_whitespace().next().unwrap_or(entry);
    entry.split(':').next().unwrap_or(entry)
}

pub fn collect_ui_list_requests(nodes: &[UiNode]) -> Vec<UiListRequest> {
    let mut out = Vec::new();
    collect_ui_list_requests_into(nodes, &mut out);
    out
}

pub fn ui_refresh_requests_for_binding(
    binding: &str,
    known_requests: &BTreeMap<String, UiListRequest>,
    visible_requests: Vec<UiListRequest>,
) -> Vec<UiListRequest> {
    let mut requests = Vec::new();
    let mut keys = BTreeSet::new();

    for (key, request) in known_requests {
        if request.source == binding {
            keys.insert(key.clone());
            requests.push(request.clone());
        }
    }

    for request in visible_requests {
        let key = request.cache_key();
        if request.source == binding && keys.insert(key) {
            requests.push(request);
        }
    }

    requests
}

fn collect_ui_list_requests_into(nodes: &[UiNode], out: &mut Vec<UiListRequest>) {
    for node in nodes {
        match node {
            UiNode::Section(section) => collect_ui_list_requests_into(&section.nodes, out),
            UiNode::List(list) if is_named_worklist_ui_source(&list.source) => {
                out.push(UiListRequest {
                    source: list.source.clone(),
                    filter: list.filter.clone(),
                    limit: list.limit,
                });
            }
            UiNode::Activity(activity) if is_named_worklist_ui_source(&activity.source) => {
                out.push(UiListRequest {
                    source: activity.source.clone(),
                    filter: Map::new(),
                    limit: Some(DEFAULT_UI_ACTIVITY_LIMIT),
                });
            }
            UiNode::Detail(detail) if is_named_worklist_ui_source(&detail.source) => {
                out.push(UiListRequest {
                    source: detail.source.clone(),
                    filter: Map::new(),
                    limit: Some(DEFAULT_UI_DETAIL_LIMIT),
                });
            }
            UiNode::Report(report) if is_named_worklist_ui_source(&report.source) => {
                out.push(UiListRequest {
                    source: report.source.clone(),
                    filter: Map::new(),
                    limit: Some(DEFAULT_UI_REPORT_LIMIT),
                });
            }
            UiNode::Chart(chart) if is_named_worklist_ui_source(&chart.source) => {
                out.push(UiListRequest {
                    source: chart.source.clone(),
                    filter: Map::new(),
                    limit: Some(DEFAULT_UI_CHART_LIMIT),
                });
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::{Map, json};
    use turin_daemon_protocol::{
        UiActivityNode, UiChartNode, UiDetailNode, UiListNode, UiNode, UiReportNode, UiSectionNode,
    };

    use crate::controller::UiListRequest;

    use super::{
        DEFAULT_UI_ACTIVITY_LIMIT, DEFAULT_UI_CHART_LIMIT, DEFAULT_UI_DETAIL_LIMIT,
        DEFAULT_UI_REPORT_LIMIT, UiWorklistSourceError, collect_ui_list_requests,
        is_named_worklist_ui_source, is_worklist_ui_source, ui_list_filter_fields,
        ui_list_sort_fields, ui_refresh_requests_for_binding, ui_sort_entry_field,
        ui_worklist_name_from_source, ui_worklist_request,
    };

    #[test]
    fn worklist_source_detection_is_prefix_based() {
        assert!(is_worklist_ui_source("worklists.release"));
        assert!(is_worklist_ui_source("worklists."));
        assert!(!is_worklist_ui_source("tables.release"));
    }

    #[test]
    fn named_worklist_source_requires_a_non_empty_name() {
        assert!(is_named_worklist_ui_source("worklists.release"));
        assert!(!is_named_worklist_ui_source("worklists."));
        assert!(!is_named_worklist_ui_source("worklists. "));
        assert!(!is_named_worklist_ui_source("tables.release"));
    }

    #[test]
    fn worklist_name_reports_missing_name_and_unsupported_source() {
        assert_eq!(
            ui_worklist_name_from_source("worklists.release"),
            Ok("release")
        );
        assert_eq!(
            ui_worklist_name_from_source("worklists."),
            Err(UiWorklistSourceError::MissingName)
        );
        assert_eq!(
            ui_worklist_name_from_source("worklists. "),
            Err(UiWorklistSourceError::MissingName)
        );
        assert_eq!(
            ui_worklist_name_from_source("tables.release"),
            Err(UiWorklistSourceError::Unsupported)
        );
    }

    #[test]
    fn worklist_request_uses_empty_filter_and_explicit_limit() {
        let request = ui_worklist_request("worklists.release", 25).expect("request");
        assert_eq!(request.source, "worklists.release");
        assert!(request.filter.is_empty());
        assert_eq!(request.limit, Some(25));
        assert!(ui_worklist_request("worklists.", 25).is_none());
        assert!(ui_worklist_request("tables.release", 25).is_none());
    }

    #[test]
    fn list_filter_and_sort_fields_are_stable_for_display() {
        let filter = Map::from_iter([
            ("status".to_string(), json!("pending")),
            ("kind".to_string(), json!("approval")),
        ]);
        let sort = vec![
            "-updated_at desc".to_string(),
            "+priority".to_string(),
            "metadata.release:asc".to_string(),
        ];

        assert_eq!(ui_list_filter_fields(&filter), vec!["kind", "status"]);
        assert_eq!(
            ui_list_sort_fields(&sort),
            vec!["updated_at", "priority", "metadata.release"]
        );
        assert_eq!(ui_sort_entry_field("-updated_at desc"), "updated_at");
        assert_eq!(
            ui_sort_entry_field("metadata.release:asc"),
            "metadata.release"
        );
    }

    #[test]
    fn refresh_requests_include_known_and_visible_matching_bindings() {
        let known = BTreeMap::from([(
            list_request("worklists.release", Some(8)).cache_key(),
            list_request("worklists.release", Some(8)),
        )]);
        let visible = vec![
            list_request("worklists.release", Some(25)),
            list_request("worklists.other", Some(10)),
        ];

        let requests = ui_refresh_requests_for_binding("worklists.release", &known, visible);

        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].source, "worklists.release");
        assert_eq!(requests[0].limit, Some(8));
        assert_eq!(requests[1].source, "worklists.release");
        assert_eq!(requests[1].limit, Some(25));
    }

    #[test]
    fn refresh_requests_dedupe_matching_visible_cache_keys() {
        let request = list_request("worklists.release", Some(8));
        let known = BTreeMap::from([(request.cache_key(), request.clone())]);

        let requests = ui_refresh_requests_for_binding("worklists.release", &known, vec![request]);

        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].source, "worklists.release");
        assert_eq!(requests[0].limit, Some(8));
    }

    #[test]
    fn refresh_requests_ignore_other_bindings() {
        let known = BTreeMap::from([(
            list_request("worklists.release", Some(8)).cache_key(),
            list_request("worklists.release", Some(8)),
        )]);
        let visible = vec![list_request("worklists.release", Some(25))];

        let requests = ui_refresh_requests_for_binding("worklists.qa", &known, visible);

        assert!(requests.is_empty());
    }

    #[test]
    fn collect_requests_walks_nested_supported_nodes() {
        let nodes = vec![UiNode::Section(UiSectionNode {
            id: Some("summary".to_string()),
            title: "Summary".to_string(),
            nodes: vec![
                UiNode::List(UiListNode {
                    id: Some("pending".to_string()),
                    title: "Pending".to_string(),
                    source: "worklists.release".to_string(),
                    filter: Map::from_iter([("status".to_string(), json!("pending"))]),
                    fields: Vec::new(),
                    sort: Vec::new(),
                    limit: Some(7),
                    intent: None,
                    render_as: None,
                }),
                UiNode::Activity(UiActivityNode {
                    id: None,
                    title: "Activity".to_string(),
                    source: "worklists.release".to_string(),
                }),
                UiNode::Detail(UiDetailNode {
                    id: None,
                    title: "Detail".to_string(),
                    source: "worklists.release".to_string(),
                    item_id: None,
                }),
                UiNode::Report(UiReportNode {
                    id: None,
                    title: "Report".to_string(),
                    source: "worklists.release".to_string(),
                    prompt: None,
                }),
                UiNode::Chart(UiChartNode {
                    id: None,
                    title: "Chart".to_string(),
                    source: "worklists.release".to_string(),
                    intent: None,
                    render_as: None,
                }),
                UiNode::List(UiListNode {
                    id: Some("custom".to_string()),
                    title: "Custom".to_string(),
                    source: "tables.release".to_string(),
                    filter: Map::new(),
                    fields: Vec::new(),
                    sort: Vec::new(),
                    limit: Some(5),
                    intent: None,
                    render_as: None,
                }),
                UiNode::List(UiListNode {
                    id: Some("missing-worklist".to_string()),
                    title: "Missing Worklist".to_string(),
                    source: "worklists.".to_string(),
                    filter: Map::new(),
                    fields: Vec::new(),
                    sort: Vec::new(),
                    limit: Some(5),
                    intent: None,
                    render_as: None,
                }),
            ],
        })];

        let requests = collect_ui_list_requests(&nodes);

        assert_eq!(requests.len(), 5);
        assert!(
            !requests
                .iter()
                .any(|request| request.source == "worklists.")
        );
        assert_eq!(requests[0].filter["status"], "pending");
        assert_eq!(requests[0].limit, Some(7));
        assert_eq!(requests[1].limit, Some(DEFAULT_UI_ACTIVITY_LIMIT));
        assert_eq!(requests[2].limit, Some(DEFAULT_UI_DETAIL_LIMIT));
        assert_eq!(requests[3].limit, Some(DEFAULT_UI_REPORT_LIMIT));
        assert_eq!(requests[4].limit, Some(DEFAULT_UI_CHART_LIMIT));
    }

    fn list_request(source: &str, limit: Option<u32>) -> UiListRequest {
        UiListRequest {
            source: source.to_string(),
            filter: Map::new(),
            limit,
        }
    }
}
