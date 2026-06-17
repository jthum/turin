use std::collections::{BTreeMap, BTreeSet};

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use serde_json::Value;
use turin_daemon_protocol::{
    UiFormNode, UiListNode, UiMenuItem, UiNode, UiScreenIntent, WorkItemDetail, WorkItemList,
};
use turin_ui_core::{UiAppRecord, UiListRequest};

use crate::app::PendingHarnessAction;
use crate::theme;

#[derive(Debug, Clone)]
pub struct HarnessAction {
    pub app_id: String,
    pub label: String,
    pub action: String,
    pub params: Value,
    pub confirm: bool,
    pub agent_id: Option<String>,
    pub harness_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarnessNavTarget {
    Screen { index: usize },
    Menu { opens: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarnessFocusTarget {
    Screen {
        screen_index: usize,
    },
    Action {
        screen_index: usize,
        action_index: usize,
    },
    Node {
        screen_index: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarnessNavItem {
    pub label: String,
    pub group: String,
    pub badge: Option<String>,
    pub depth: usize,
    pub target: HarnessNavTarget,
}

impl HarnessAction {
    pub fn into_pending(self) -> PendingHarnessAction {
        PendingHarnessAction {
            app_id: self.app_id,
            label: self.label,
            action: self.action,
            agent_id: self.agent_id,
            harness_id: self.harness_id,
            params: self.params,
        }
    }
}

pub fn default_screen_index(app: &UiAppRecord) -> usize {
    app.opens_with
        .as_deref()
        .and_then(|screen_id| screen_index_for_target(app, screen_id))
        .unwrap_or(0)
}

pub fn screen_at(app: &UiAppRecord, index: usize) -> Option<&UiScreenIntent> {
    app.screens.values().nth(index)
}

pub fn screen_index_for_target(app: &UiAppRecord, target: &str) -> Option<usize> {
    app.screens
        .values()
        .position(|screen| screen.id == target || screen.title == target)
}

pub fn collect_nav_items(app: &UiAppRecord) -> Vec<HarnessNavItem> {
    let mut out = Vec::new();

    for (index, screen) in app.screens.values().enumerate() {
        out.push(HarnessNavItem {
            label: screen.title.clone(),
            group: "Screens".to_string(),
            badge: screen.presentation.clone(),
            depth: 0,
            target: HarnessNavTarget::Screen { index },
        });
    }

    for menu in &app.menus {
        collect_menu_nav_items(&menu.title, &menu.items, 0, &mut out);
    }

    out
}

fn collect_menu_nav_items(
    group: &str,
    items: &[UiMenuItem],
    depth: usize,
    out: &mut Vec<HarnessNavItem>,
) {
    for item in items {
        out.push(HarnessNavItem {
            label: item.label.clone(),
            group: group.to_string(),
            badge: item.badge.clone(),
            depth,
            target: HarnessNavTarget::Menu {
                opens: item.opens.clone(),
            },
        });
        if !item.items.is_empty() {
            collect_menu_nav_items(group, &item.items, depth + 1, out);
        }
    }
}

pub fn collect_list_requests(nodes: &[UiNode]) -> Vec<UiListRequest> {
    let mut out = Vec::new();
    collect_list_requests_into(nodes, &mut out);
    out
}

fn collect_list_requests_into(nodes: &[UiNode], out: &mut Vec<UiListRequest>) {
    for node in nodes {
        match node {
            UiNode::Section(section) => collect_list_requests_into(&section.nodes, out),
            UiNode::List(list) if list.source.starts_with("worklists.") => {
                out.push(UiListRequest {
                    source: list.source.clone(),
                    filter: list.filter.clone(),
                    limit: list.limit,
                });
            }
            _ => {}
        }
    }
}

pub fn collect_actions(app: &UiAppRecord, nodes: &[UiNode]) -> Vec<HarnessAction> {
    let mut out = Vec::new();
    collect_actions_into(app, nodes, &mut out);
    out
}

fn collect_actions_into(app: &UiAppRecord, nodes: &[UiNode], out: &mut Vec<HarnessAction>) {
    for node in nodes {
        match node {
            UiNode::Section(section) => collect_actions_into(app, &section.nodes, out),
            UiNode::Action(action) => out.push(HarnessAction {
                app_id: app.id.clone(),
                label: action.label.clone(),
                action: action.action.clone(),
                params: action.params.clone(),
                confirm: action.confirm,
                agent_id: app.source.agent_id.clone(),
                harness_id: app.source.harness_id.clone(),
            }),
            UiNode::Form(form) => {
                out.push(HarnessAction {
                    app_id: app.id.clone(),
                    label: format!("Submit {}", form.title),
                    action: form.action.clone(),
                    params: form.params.clone(),
                    confirm: false,
                    agent_id: app.source.agent_id.clone(),
                    harness_id: app.source.harness_id.clone(),
                });
            }
            _ => {}
        }
    }
}

pub fn find_focus_target(app: &UiAppRecord, target: &str) -> Option<HarnessFocusTarget> {
    if let Some(screen_index) = screen_index_for_target(app, target) {
        return Some(HarnessFocusTarget::Screen { screen_index });
    }

    for (screen_index, screen) in app.screens.values().enumerate() {
        if let Some(target) = find_focus_target_in_nodes(&screen.nodes, target, screen_index) {
            return Some(target);
        }
    }

    None
}

fn find_focus_target_in_nodes(
    nodes: &[UiNode],
    target: &str,
    screen_index: usize,
) -> Option<HarnessFocusTarget> {
    let mut action_index = 0;
    find_focus_target_in_nodes_with_action_index(nodes, target, screen_index, &mut action_index)
}

fn find_focus_target_in_nodes_with_action_index(
    nodes: &[UiNode],
    target: &str,
    screen_index: usize,
    action_index: &mut usize,
) -> Option<HarnessFocusTarget> {
    for node in nodes {
        match node {
            UiNode::Section(section) => {
                if node_id_matches(section.id.as_deref(), target) {
                    return Some(HarnessFocusTarget::Node { screen_index });
                }
                if let Some(found) = find_focus_target_in_nodes_with_action_index(
                    &section.nodes,
                    target,
                    screen_index,
                    action_index,
                ) {
                    return Some(found);
                }
            }
            UiNode::Action(action) => {
                let current_action_index = *action_index;
                *action_index += 1;
                if node_id_matches(action.id.as_deref(), target)
                    || action.action == target
                    || action.label == target
                {
                    return Some(HarnessFocusTarget::Action {
                        screen_index,
                        action_index: current_action_index,
                    });
                }
            }
            UiNode::Form(form) => {
                let current_action_index = *action_index;
                *action_index += 1;
                if node_id_matches(form.id.as_deref(), target)
                    || form.action == target
                    || form.title == target
                {
                    return Some(HarnessFocusTarget::Action {
                        screen_index,
                        action_index: current_action_index,
                    });
                }
            }
            UiNode::Text(text) if node_id_matches(text.id.as_deref(), target) => {
                return Some(HarnessFocusTarget::Node { screen_index });
            }
            UiNode::List(list) if node_id_matches(list.id.as_deref(), target) => {
                return Some(HarnessFocusTarget::Node { screen_index });
            }
            UiNode::Activity(activity) if node_id_matches(activity.id.as_deref(), target) => {
                return Some(HarnessFocusTarget::Node { screen_index });
            }
            UiNode::Detail(detail) if node_id_matches(detail.id.as_deref(), target) => {
                return Some(HarnessFocusTarget::Node { screen_index });
            }
            UiNode::Report(report) if node_id_matches(report.id.as_deref(), target) => {
                return Some(HarnessFocusTarget::Node { screen_index });
            }
            UiNode::Chart(chart) if node_id_matches(chart.id.as_deref(), target) => {
                return Some(HarnessFocusTarget::Node { screen_index });
            }
            _ => {}
        }
    }
    None
}

fn node_id_matches(id: Option<&str>, target: &str) -> bool {
    id == Some(target)
}

pub fn render_harness_screen(
    frame: &mut Frame<'_>,
    area: Rect,
    app: Option<&UiAppRecord>,
    screen_indices: &BTreeMap<String, usize>,
    lists: &BTreeMap<String, WorkItemList>,
    requested_lists: &BTreeSet<String>,
) {
    let Some(app) = app else {
        frame.render_widget(empty_panel("Harness", "No harness UI apps declared"), area);
        return;
    };
    let screen_index = screen_indices
        .get(&app.id)
        .copied()
        .unwrap_or_else(|| default_screen_index(app));
    let Some(screen) = screen_at(app, screen_index) else {
        frame.render_widget(empty_panel("Harness", "Selected app has no screens"), area);
        return;
    };

    let mut lines = vec![
        Line::from(vec![
            Span::styled(screen.title.clone(), theme::title()),
            Span::styled(format!("  {}", screen.id), theme::muted()),
        ]),
        Line::from(""),
    ];
    let max_width = area.width.saturating_sub(4) as usize;
    render_nodes(
        &screen.nodes,
        lists,
        requested_lists,
        &mut lines,
        0,
        max_width,
    );
    frame.render_widget(panel("Screen", lines), area);
}

fn render_nodes(
    nodes: &[UiNode],
    lists: &BTreeMap<String, WorkItemList>,
    requested_lists: &BTreeSet<String>,
    lines: &mut Vec<Line<'static>>,
    depth: usize,
    max_width: usize,
) {
    for node in nodes {
        match node {
            UiNode::Text(text) => {
                lines.push(indent_line(depth, text.text.clone(), theme::base()));
                lines.push(Line::from(""));
            }
            UiNode::Section(section) => {
                lines.push(indent_line(depth, section.title.clone(), theme::accent()));
                render_nodes(
                    &section.nodes,
                    lists,
                    requested_lists,
                    lines,
                    depth + 1,
                    max_width,
                );
            }
            UiNode::Action(action) => {
                let marker = if action.confirm { "!" } else { "→" };
                lines.push(indent_line(
                    depth,
                    format!("{marker} {}", action.label),
                    if action.confirm {
                        theme::warning()
                    } else {
                        theme::base()
                    },
                ));
            }
            UiNode::List(list) => {
                render_list(list, lists, requested_lists, lines, depth, max_width)
            }
            UiNode::Form(form) => render_form(form, lines, depth),
            UiNode::Activity(activity) => lines.push(indent_line(
                depth,
                format!("Activity: {} ({})", activity.title, activity.source),
                theme::muted(),
            )),
            UiNode::Detail(detail) => lines.push(indent_line(
                depth,
                format!("Detail: {} ({})", detail.title, detail.source),
                theme::muted(),
            )),
            UiNode::Report(report) => lines.push(indent_line(
                depth,
                format!("Report: {} ({})", report.title, report.source),
                theme::muted(),
            )),
            UiNode::Chart(chart) => lines.push(indent_line(
                depth,
                format!("Chart: {} ({})", chart.title, chart.source),
                theme::muted(),
            )),
        }
    }
}

fn render_list(
    list: &UiListNode,
    lists: &BTreeMap<String, WorkItemList>,
    requested_lists: &BTreeSet<String>,
    lines: &mut Vec<Line<'static>>,
    depth: usize,
    max_width: usize,
) {
    let mut meta = vec![list.source.clone()];
    if let Some(intent) = &list.intent {
        meta.push(format!("intent={intent}"));
    }
    if let Some(render_as) = &list.render_as {
        meta.push(format!("as={render_as}"));
    }
    if let Some(limit) = list.limit {
        meta.push(format!("limit={limit}"));
    }
    lines.push(indent_line(
        depth,
        format!("{}  {}", list.title, meta.join("  ")),
        theme::accent(),
    ));
    if list
        .render_as
        .as_deref()
        .is_some_and(|render_as| render_as != "table")
    {
        lines.push(indent_line(
            depth + 1,
            "Terminal fallback: rendering this list as a compact table".to_string(),
            theme::muted(),
        ));
    }
    if !list.source.starts_with("worklists.") {
        lines.push(indent_line(
            depth + 1,
            "No terminal data adapter exists for this list source yet".to_string(),
            theme::muted(),
        ));
        lines.push(Line::from(""));
        return;
    }

    let request = UiListRequest {
        source: list.source.clone(),
        filter: list.filter.clone(),
        limit: list.limit,
    };
    let key = request.cache_key();
    match lists.get(&key) {
        Some(items) => render_work_items(list, items, lines, depth + 1, max_width),
        None if requested_lists.contains(&key) => lines.push(indent_line(
            depth + 1,
            "Loading list data...".to_string(),
            theme::muted(),
        )),
        None => lines.push(indent_line(
            depth + 1,
            "List data not requested yet".to_string(),
            theme::muted(),
        )),
    }
    lines.push(Line::from(""));
}

fn render_work_items(
    list: &UiListNode,
    items: &WorkItemList,
    lines: &mut Vec<Line<'static>>,
    depth: usize,
    max_width: usize,
) {
    if items.items.is_empty() {
        lines.push(indent_line(depth, "No items".to_string(), theme::muted()));
        return;
    }

    let fields = if list.fields.is_empty() {
        vec![
            "title".to_string(),
            "status".to_string(),
            "kind".to_string(),
            "priority".to_string(),
        ]
    } else {
        list.fields.clone()
    };
    let widths = table_widths(&fields, max_width.saturating_sub(depth * 2));
    lines.push(indent_line(
        depth,
        table_row(&fields, &widths),
        theme::muted(),
    ));
    for item in items.items.iter().take(12) {
        let values = fields
            .iter()
            .map(|field| work_item_field(item, field))
            .collect::<Vec<_>>();
        lines.push(indent_line(
            depth,
            table_row(&values, &widths),
            theme::base(),
        ));
    }
    if items.items.len() > 12 {
        lines.push(indent_line(
            depth,
            format!("... {} more", items.items.len() - 12),
            theme::muted(),
        ));
    }
}

fn table_widths(fields: &[String], max_width: usize) -> Vec<usize> {
    if fields.is_empty() {
        return Vec::new();
    }
    let separator_width = fields.len().saturating_sub(1) * 3;
    let available = max_width
        .saturating_sub(separator_width)
        .max(fields.len() * 6);
    let width = (available / fields.len()).clamp(6, 28);
    vec![width; fields.len()]
}

fn table_row(values: &[String], widths: &[usize]) -> String {
    values
        .iter()
        .zip(widths.iter())
        .map(|(value, width)| pad_cell(value, *width))
        .collect::<Vec<_>>()
        .join(" | ")
}

fn pad_cell(value: &str, width: usize) -> String {
    let value = truncate(value, width);
    let len = value.chars().count();
    if len >= width {
        return value;
    }
    format!("{value}{}", " ".repeat(width - len))
}

fn render_form(form: &UiFormNode, lines: &mut Vec<Line<'static>>, depth: usize) {
    lines.push(indent_line(
        depth,
        format!("Form: {}", form.title),
        theme::accent(),
    ));
    lines.push(indent_line(
        depth + 1,
        format!("Submit action: {}", form.action),
        theme::muted(),
    ));
    for field in &form.fields {
        let kind = field.kind.as_deref().unwrap_or("value");
        lines.push(indent_line(
            depth + 1,
            format!("{}: {kind}", field.label),
            theme::base(),
        ));
    }
    lines.push(Line::from(""));
}

fn work_item_field(item: &WorkItemDetail, field: &str) -> String {
    match field {
        "id" => item.public_id.clone(),
        "title" => item.title.clone(),
        "kind" => item.kind.clone(),
        "status" => item.status.clone(),
        "priority" => item.priority.to_string(),
        "agent" | "claim_agent_id" => item.claim_agent_id.clone().unwrap_or_default(),
        "paused" => {
            if item.paused {
                "yes".to_string()
            } else {
                "no".to_string()
            }
        }
        other => item
            .metadata
            .as_ref()
            .and_then(|metadata| metadata.get(other))
            .map(json_value)
            .unwrap_or_default(),
    }
}

fn json_value(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(value) => value.clone(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::Array(_) | Value::Object(_) => truncate(&value.to_string(), 48),
    }
}

fn panel(title: &'static str, lines: Vec<Line<'static>>) -> Paragraph<'static> {
    Paragraph::new(lines)
        .block(
            Block::default()
                .title(title)
                .borders(Borders::ALL)
                .border_style(Style::default().fg(theme::PANEL_HOT).bg(theme::BG)),
        )
        .wrap(Wrap { trim: false })
        .style(Style::default().fg(theme::TEXT).bg(theme::BG))
}

fn empty_panel(title: &'static str, message: &'static str) -> Paragraph<'static> {
    panel(
        title,
        vec![Line::from(Span::styled(message, theme::muted()))],
    )
}

fn indent_line(depth: usize, text: String, style: Style) -> Line<'static> {
    Line::from(vec![
        Span::raw("  ".repeat(depth)),
        Span::styled(text, style),
    ])
}

fn truncate(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    if max_chars <= 3 {
        return ".".repeat(max_chars);
    }
    let mut out = String::new();
    let take_chars = max_chars - 3;
    for (index, ch) in value.chars().enumerate() {
        if index >= take_chars {
            out.push_str("...");
            return out;
        }
        out.push(ch);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use turin_daemon_protocol::{
        UiActionNode, UiAppIntent, UiIntent, UiIntentMessage, UiListNode, UiMenuIntent, UiMenuItem,
        UiNode, UiOpensWithIntent, UiScreenIntent,
    };
    use turin_ui_core::UiRegistry;

    fn release_app() -> UiAppRecord {
        let registry = UiRegistry::from_messages([
            UiIntentMessage::new(UiIntent::App(UiAppIntent {
                id: "release".to_string(),
                title: "Release Operator".to_string(),
                about: None,
                icon: None,
            })),
            UiIntentMessage::new(UiIntent::Screen(UiScreenIntent {
                app_id: "release".to_string(),
                id: "home".to_string(),
                title: "Release Desk".to_string(),
                presentation: Some("dashboard".to_string()),
                nodes: vec![
                    UiNode::Text(turin_daemon_protocol::UiTextNode {
                        id: None,
                        text: "Ready".to_string(),
                    }),
                    UiNode::Action(UiActionNode {
                        id: Some("seed-demo-work".to_string()),
                        label: "Seed Demo Work".to_string(),
                        action: "release.seed_demo_work".to_string(),
                        params: Value::Null,
                        confirm: false,
                    }),
                    UiNode::List(UiListNode {
                        id: Some("recent-release-work".to_string()),
                        title: "Recent Release Work".to_string(),
                        source: "worklists.release".to_string(),
                        filter: Default::default(),
                        fields: Vec::new(),
                        sort: Vec::new(),
                        limit: Some(8),
                        intent: Some("tasks".to_string()),
                        render_as: Some("table".to_string()),
                    }),
                ],
            })),
            UiIntentMessage::new(UiIntent::Screen(UiScreenIntent {
                app_id: "release".to_string(),
                id: "approvals".to_string(),
                title: "Approvals".to_string(),
                presentation: None,
                nodes: Vec::new(),
            })),
            UiIntentMessage::new(UiIntent::Screen(UiScreenIntent {
                app_id: "release".to_string(),
                id: "intake".to_string(),
                title: "Intake".to_string(),
                presentation: None,
                nodes: Vec::new(),
            })),
            UiIntentMessage::new(UiIntent::OpensWith(UiOpensWithIntent {
                app_id: "release".to_string(),
                screen_id: "approvals".to_string(),
            })),
            UiIntentMessage::new(UiIntent::Menu(UiMenuIntent {
                app_id: "release".to_string(),
                title: "Main".to_string(),
                items: vec![
                    UiMenuItem {
                        label: "Dashboard".to_string(),
                        opens: "home".to_string(),
                        id: None,
                        icon: None,
                        badge: None,
                        items: Vec::new(),
                    },
                    UiMenuItem {
                        label: "Work".to_string(),
                        opens: "approvals".to_string(),
                        id: None,
                        icon: None,
                        badge: Some("approvals".to_string()),
                        items: vec![UiMenuItem {
                            label: "Intake".to_string(),
                            opens: "intake".to_string(),
                            id: None,
                            icon: None,
                            badge: None,
                            items: Vec::new(),
                        }],
                    },
                ],
            })),
        ]);

        registry.app("release").expect("release app").clone()
    }

    #[test]
    fn default_screen_uses_declared_opens_with() {
        let app = release_app();
        let default = default_screen_index(&app);

        assert_eq!(
            screen_at(&app, default).map(|screen| screen.id.as_str()),
            Some("approvals")
        );
        assert_eq!(
            screen_index_for_target(&app, "home")
                .and_then(|index| screen_at(&app, index))
                .map(|screen| screen.id.as_str()),
            Some("home")
        );
        assert_eq!(
            screen_index_for_target(&app, "Approvals")
                .and_then(|index| screen_at(&app, index))
                .map(|screen| screen.id.as_str()),
            Some("approvals")
        );
        assert_eq!(screen_index_for_target(&app, "missing"), None);
    }

    #[test]
    fn nav_items_include_screens_and_nested_menu_entries() {
        let app = release_app();
        let items = collect_nav_items(&app);

        let home = items
            .iter()
            .find(|item| item.label == "Release Desk")
            .expect("home screen item");
        assert_eq!(home.group, "Screens");
        assert_eq!(home.badge.as_deref(), Some("dashboard"));
        assert!(matches!(
            home.target,
            HarnessNavTarget::Screen { index } if screen_at(&app, index).map(|screen| screen.id.as_str()) == Some("home")
        ));

        let work = items
            .iter()
            .find(|item| item.label == "Work")
            .expect("work menu item");
        assert_eq!(work.group, "Main");
        assert_eq!(work.badge.as_deref(), Some("approvals"));
        assert!(
            matches!(work.target, HarnessNavTarget::Menu { ref opens } if opens == "approvals")
        );

        let intake = items
            .iter()
            .find(|item| item.label == "Intake" && item.group == "Main")
            .expect("nested intake menu item");
        assert_eq!(intake.depth, 1);
        assert!(matches!(intake.target, HarnessNavTarget::Menu { ref opens } if opens == "intake"));
    }

    #[test]
    fn focus_targets_resolve_screens_actions_and_node_ids() {
        let app = release_app();

        assert!(matches!(
            find_focus_target(&app, "home"),
            Some(HarnessFocusTarget::Screen { screen_index })
                if screen_at(&app, screen_index).map(|screen| screen.id.as_str()) == Some("home")
        ));
        assert!(matches!(
            find_focus_target(&app, "seed-demo-work"),
            Some(HarnessFocusTarget::Action {
                screen_index,
                action_index: 0,
            }) if screen_at(&app, screen_index).map(|screen| screen.id.as_str()) == Some("home")
        ));
        assert!(matches!(
            find_focus_target(&app, "recent-release-work"),
            Some(HarnessFocusTarget::Node { screen_index })
                if screen_at(&app, screen_index).map(|screen| screen.id.as_str()) == Some("home")
        ));
        assert_eq!(find_focus_target(&app, "unknown"), None);
    }

    #[test]
    fn table_rows_are_bounded_and_padded() {
        let fields = vec![
            "title".to_string(),
            "status".to_string(),
            "priority".to_string(),
        ];
        let widths = table_widths(&fields, 36);
        let row = table_row(
            &[
                "A very long release approval title".to_string(),
                "pending".to_string(),
                "10".to_string(),
            ],
            &widths,
        );

        assert!(widths.iter().all(|width| (6..=28).contains(width)));
        assert!(row.len() <= 36);
        assert!(row.contains("..."));
        assert!(row.contains("pending"));
    }
}
