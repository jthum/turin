use turin_daemon_protocol::{UiNode, UiScreenIntent};

use crate::UiAppRecord;

pub fn ui_default_screen_index(app: &UiAppRecord) -> usize {
    app.opens_with
        .as_deref()
        .and_then(|target| ui_screen_index_for_target(app, target))
        .unwrap_or_default()
}

pub fn ui_screen_index_for_target(app: &UiAppRecord, target: &str) -> Option<usize> {
    app.screens
        .values()
        .position(|screen| screen_matches_target(screen, target))
}

fn screen_matches_target(screen: &UiScreenIntent, target: &str) -> bool {
    screen.id == target || screen.title == target
}

pub fn ui_nodes_contain_target(nodes: &[UiNode], target: &str) -> bool {
    nodes
        .iter()
        .any(|node| ui_node_matches_target(node, target))
}

pub fn ui_node_matches_target(node: &UiNode, target: &str) -> bool {
    match node {
        UiNode::Section(section) => {
            ui_node_id_matches(section.id.as_deref(), target)
                || ui_nodes_contain_target(&section.nodes, target)
        }
        UiNode::Text(text) => ui_node_id_matches(text.id.as_deref(), target),
        UiNode::Action(action) => {
            ui_node_id_matches(action.id.as_deref(), target)
                || action.action == target
                || action.label == target
        }
        UiNode::List(list) => ui_node_id_matches(list.id.as_deref(), target),
        UiNode::Activity(activity) => ui_node_id_matches(activity.id.as_deref(), target),
        UiNode::Detail(detail) => ui_node_id_matches(detail.id.as_deref(), target),
        UiNode::Form(form) => {
            ui_node_id_matches(form.id.as_deref(), target)
                || form.action == target
                || form.title == target
        }
        UiNode::Report(report) => ui_node_id_matches(report.id.as_deref(), target),
        UiNode::Chart(chart) => ui_node_id_matches(chart.id.as_deref(), target),
    }
}

pub fn ui_node_id_matches(id: Option<&str>, target: &str) -> bool {
    id == Some(target)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde_json::{Map, Value};
    use turin_daemon_protocol::{
        UiActionNode, UiFormNode, UiIntentSource, UiListNode, UiNode, UiScreenIntent,
        UiSectionNode, UiTextNode,
    };

    use crate::{
        UiAppRecord, ui_default_screen_index, ui_node_matches_target, ui_nodes_contain_target,
        ui_screen_index_for_target,
    };

    fn test_app(opens_with: Option<&str>) -> UiAppRecord {
        UiAppRecord {
            id: "release".to_string(),
            source: UiIntentSource::default(),
            definition: None,
            screens: BTreeMap::from([
                ("approvals".to_string(), screen("approvals", "Approvals")),
                ("home".to_string(), screen("home", "Release Desk")),
                ("intake".to_string(), screen("intake", "Intake")),
            ]),
            panes: BTreeMap::new(),
            menus: Vec::new(),
            opens_with: opens_with.map(str::to_string),
            badges: BTreeMap::new(),
        }
    }

    fn screen(id: &str, title: &str) -> UiScreenIntent {
        UiScreenIntent {
            app_id: "release".to_string(),
            id: id.to_string(),
            title: title.to_string(),
            presentation: None,
            nodes: Vec::new(),
        }
    }

    #[test]
    fn default_screen_uses_opens_with_target_id() {
        let app = test_app(Some("intake"));

        assert_eq!(ui_default_screen_index(&app), 2);
    }

    #[test]
    fn default_screen_uses_opens_with_target_title() {
        let app = test_app(Some("Release Desk"));

        assert_eq!(ui_default_screen_index(&app), 1);
    }

    #[test]
    fn default_screen_falls_back_to_first_screen_for_missing_target() {
        let app = test_app(Some("missing"));

        assert_eq!(ui_default_screen_index(&app), 0);
    }

    #[test]
    fn screen_index_matches_id_or_title() {
        let app = test_app(None);

        assert_eq!(ui_screen_index_for_target(&app, "home"), Some(1));
        assert_eq!(ui_screen_index_for_target(&app, "Approvals"), Some(0));
        assert_eq!(ui_screen_index_for_target(&app, "missing"), None);
    }

    #[test]
    fn node_matching_resolves_ids_actions_forms_and_nested_nodes() {
        let nodes = vec![
            UiNode::Text(UiTextNode {
                id: Some("intro".to_string()),
                text: "Ready".to_string(),
            }),
            UiNode::Action(UiActionNode {
                id: Some("seed-demo-work".to_string()),
                label: "Seed Demo Work".to_string(),
                action: "release.seed_demo_work".to_string(),
                params: Value::Null,
                confirm: false,
            }),
            UiNode::Form(UiFormNode {
                id: Some("seed-demo-form".to_string()),
                title: "Create Demo Approval Batch".to_string(),
                action: "release.seed_demo_work".to_string(),
                fields: Vec::new(),
                params: Value::Null,
            }),
            UiNode::Section(UiSectionNode {
                id: Some("work-section".to_string()),
                title: "Work".to_string(),
                nodes: vec![UiNode::List(UiListNode {
                    id: Some("recent-release-work".to_string()),
                    title: "Recent Release Work".to_string(),
                    source: "worklists.release".to_string(),
                    filter: Map::new(),
                    fields: Vec::new(),
                    sort: Vec::new(),
                    limit: Some(8),
                    intent: Some("tasks".to_string()),
                    render_as: Some("table".to_string()),
                })],
            }),
        ];

        assert!(ui_nodes_contain_target(&nodes, "intro"));
        assert!(ui_nodes_contain_target(&nodes, "seed-demo-work"));
        assert!(ui_nodes_contain_target(&nodes, "Seed Demo Work"));
        assert!(ui_nodes_contain_target(&nodes, "release.seed_demo_work"));
        assert!(ui_nodes_contain_target(&nodes, "seed-demo-form"));
        assert!(ui_nodes_contain_target(
            &nodes,
            "Create Demo Approval Batch"
        ));
        assert!(ui_nodes_contain_target(&nodes, "work-section"));
        assert!(ui_nodes_contain_target(&nodes, "recent-release-work"));
        assert!(!ui_nodes_contain_target(&nodes, "missing"));
        assert!(ui_node_matches_target(&nodes[3], "recent-release-work"));
    }
}
