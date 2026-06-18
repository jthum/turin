use serde_json::Map;
use turin_daemon_protocol::UiNode;

use crate::controller::UiListRequest;

pub const DEFAULT_UI_ACTIVITY_LIMIT: u32 = 12;
pub const DEFAULT_UI_DETAIL_LIMIT: u32 = 25;
pub const DEFAULT_UI_REPORT_LIMIT: u32 = 100;
pub const DEFAULT_UI_CHART_LIMIT: u32 = 100;

pub fn is_worklist_ui_source(source: &str) -> bool {
    source.starts_with("worklists.")
}

pub fn ui_worklist_request(source: &str, limit: u32) -> Option<UiListRequest> {
    is_worklist_ui_source(source).then(|| UiListRequest {
        source: source.to_string(),
        filter: Map::new(),
        limit: Some(limit),
    })
}

pub fn collect_ui_list_requests(nodes: &[UiNode]) -> Vec<UiListRequest> {
    let mut out = Vec::new();
    collect_ui_list_requests_into(nodes, &mut out);
    out
}

fn collect_ui_list_requests_into(nodes: &[UiNode], out: &mut Vec<UiListRequest>) {
    for node in nodes {
        match node {
            UiNode::Section(section) => collect_ui_list_requests_into(&section.nodes, out),
            UiNode::List(list) if is_worklist_ui_source(&list.source) => {
                out.push(UiListRequest {
                    source: list.source.clone(),
                    filter: list.filter.clone(),
                    limit: list.limit,
                });
            }
            UiNode::Activity(activity) if is_worklist_ui_source(&activity.source) => {
                out.push(UiListRequest {
                    source: activity.source.clone(),
                    filter: Map::new(),
                    limit: Some(DEFAULT_UI_ACTIVITY_LIMIT),
                });
            }
            UiNode::Detail(detail) if is_worklist_ui_source(&detail.source) => {
                out.push(UiListRequest {
                    source: detail.source.clone(),
                    filter: Map::new(),
                    limit: Some(DEFAULT_UI_DETAIL_LIMIT),
                });
            }
            UiNode::Report(report) if is_worklist_ui_source(&report.source) => {
                out.push(UiListRequest {
                    source: report.source.clone(),
                    filter: Map::new(),
                    limit: Some(DEFAULT_UI_REPORT_LIMIT),
                });
            }
            UiNode::Chart(chart) if is_worklist_ui_source(&chart.source) => {
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
    use serde_json::{Map, json};
    use turin_daemon_protocol::{
        UiActivityNode, UiChartNode, UiDetailNode, UiListNode, UiNode, UiReportNode, UiSectionNode,
    };

    use super::{
        DEFAULT_UI_ACTIVITY_LIMIT, DEFAULT_UI_CHART_LIMIT, DEFAULT_UI_DETAIL_LIMIT,
        DEFAULT_UI_REPORT_LIMIT, collect_ui_list_requests, is_worklist_ui_source,
        ui_worklist_request,
    };

    #[test]
    fn worklist_source_detection_is_prefix_based() {
        assert!(is_worklist_ui_source("worklists.release"));
        assert!(is_worklist_ui_source("worklists."));
        assert!(!is_worklist_ui_source("tables.release"));
    }

    #[test]
    fn worklist_request_uses_empty_filter_and_explicit_limit() {
        let request = ui_worklist_request("worklists.release", 25).expect("request");
        assert_eq!(request.source, "worklists.release");
        assert!(request.filter.is_empty());
        assert_eq!(request.limit, Some(25));
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
            ],
        })];

        let requests = collect_ui_list_requests(&nodes);

        assert_eq!(requests.len(), 5);
        assert_eq!(requests[0].filter["status"], "pending");
        assert_eq!(requests[0].limit, Some(7));
        assert_eq!(requests[1].limit, Some(DEFAULT_UI_ACTIVITY_LIMIT));
        assert_eq!(requests[2].limit, Some(DEFAULT_UI_DETAIL_LIMIT));
        assert_eq!(requests[3].limit, Some(DEFAULT_UI_REPORT_LIMIT));
        assert_eq!(requests[4].limit, Some(DEFAULT_UI_CHART_LIMIT));
    }
}
