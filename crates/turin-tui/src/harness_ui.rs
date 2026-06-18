use std::collections::{BTreeMap, BTreeSet};

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use serde_json::{Map, Number, Value};
use turin_daemon_protocol::{
    UiActivityNode, UiBadgeIntent, UiChartNode, UiDetailNode, UiFormNode, UiListNode, UiMenuItem,
    UiNode, UiNoticeLevel, UiPaneIntent, UiReportNode, UiScreenIntent, WorkItemDetail,
    WorkItemList,
};
use turin_ui_core::{
    DEFAULT_UI_ACTIVITY_LIMIT as ACTIVITY_LIMIT, DEFAULT_UI_CHART_LIMIT as CHART_LIMIT,
    DEFAULT_UI_DETAIL_LIMIT as DETAIL_LIMIT, DEFAULT_UI_REPORT_LIMIT as REPORT_LIMIT, UiAppRecord,
    UiListRequest, collect_ui_list_requests as collect_shared_list_requests, is_worklist_ui_source,
    ui_data_not_loaded_message, ui_worklist_request, unsupported_ui_source_message,
    work_item_field_label, worklist_chart_group_field, worklist_group_counts,
    worklist_highest_priority_pending_item, worklist_status_counts,
};

use crate::app::PendingHarnessAction;
use crate::theme;

const WORK_ITEM_TABLE_VISIBLE_ROWS: usize = 12;
const WORK_ITEM_ROW_MARKER_WIDTH: usize = 5;
const WORK_ITEM_ACTION_MARKER_WIDTH: usize = 6;

#[derive(Debug, Clone)]
pub struct HarnessAction {
    pub app_id: String,
    pub label: String,
    pub action: String,
    pub params: Value,
    pub confirm: bool,
    pub agent_id: Option<String>,
    pub harness_id: Option<String>,
    pub form: Option<UiFormNode>,
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
    pub badge_level: Option<UiNoticeLevel>,
    pub depth: usize,
    pub target: HarnessNavTarget,
}

#[derive(Debug, Clone)]
pub struct HarnessWorkItemSelection {
    pub list_title: String,
    pub list_source: String,
    pub item: WorkItemDetail,
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
        let (badge, badge_level) = nav_badge(app, &screen.id, screen.presentation.as_deref());
        out.push(HarnessNavItem {
            label: screen.title.clone(),
            group: "Screens".to_string(),
            badge,
            badge_level,
            depth: 0,
            target: HarnessNavTarget::Screen { index },
        });
    }

    for menu in &app.menus {
        collect_menu_nav_items(app, &menu.title, &menu.items, 0, &mut out);
    }

    out
}

fn collect_menu_nav_items(
    app: &UiAppRecord,
    group: &str,
    items: &[UiMenuItem],
    depth: usize,
    out: &mut Vec<HarnessNavItem>,
) {
    for item in items {
        let (badge, badge_level) = nav_badge(app, &item.opens, item.badge.as_deref());
        out.push(HarnessNavItem {
            label: item.label.clone(),
            group: group.to_string(),
            badge,
            badge_level,
            depth,
            target: HarnessNavTarget::Menu {
                opens: item.opens.clone(),
            },
        });
        if !item.items.is_empty() {
            collect_menu_nav_items(app, group, &item.items, depth + 1, out);
        }
    }
}

fn nav_badge(
    app: &UiAppRecord,
    target: &str,
    fallback: Option<&str>,
) -> (Option<String>, Option<UiNoticeLevel>) {
    let dynamic = app.badges.get(target);
    (
        badge_text(dynamic, fallback),
        dynamic.and_then(|badge| badge.level),
    )
}

fn badge_text(badge: Option<&UiBadgeIntent>, fallback: Option<&str>) -> Option<String> {
    let label = badge
        .and_then(|badge| badge.label.as_deref())
        .or(fallback)
        .filter(|label| !label.is_empty());
    let count = badge.and_then(|badge| badge.count);
    match (label, count) {
        (Some(label), Some(count)) => Some(format!("{label} {count}")),
        (Some(label), None) => Some(label.to_string()),
        (None, Some(count)) => Some(count.to_string()),
        (None, None) => None,
    }
}

fn title_with_node_badge(app: &UiAppRecord, node_id: Option<&str>, title: &str) -> String {
    let Some(node_id) = node_id else {
        return title.to_string();
    };
    app.badges
        .get(node_id)
        .and_then(|badge| badge_text(Some(badge), None))
        .map(|badge| format!("{title}  [{badge}]"))
        .unwrap_or_else(|| title.to_string())
}

pub fn collect_list_requests(nodes: &[UiNode]) -> Vec<UiListRequest> {
    collect_shared_list_requests(nodes)
}

pub fn collect_work_item_selections(
    nodes: &[UiNode],
    lists: &BTreeMap<String, WorkItemList>,
) -> Vec<HarnessWorkItemSelection> {
    let mut out = Vec::new();
    collect_work_item_selections_into(nodes, lists, &mut out);
    out
}

fn collect_work_item_selections_into(
    nodes: &[UiNode],
    lists: &BTreeMap<String, WorkItemList>,
    out: &mut Vec<HarnessWorkItemSelection>,
) {
    for node in nodes {
        match node {
            UiNode::Section(section) => {
                collect_work_item_selections_into(&section.nodes, lists, out)
            }
            UiNode::List(list) if is_worklist_ui_source(&list.source) => {
                let request = UiListRequest {
                    source: list.source.clone(),
                    filter: list.filter.clone(),
                    limit: list.limit,
                };
                if let Some(items) = lists.get(&request.cache_key()) {
                    out.extend(items.items.iter().take(12).cloned().map(|item| {
                        HarnessWorkItemSelection {
                            list_title: list.title.clone(),
                            list_source: list.source.clone(),
                            item,
                        }
                    }));
                }
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
                form: None,
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
                    form: Some(form.clone()),
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
    selected_work_item_id: Option<&str>,
) {
    let Some(app) = app else {
        frame.render_widget(
            empty_panel(
                "Harness",
                "Default operator console is active. Overview, Tasks, and Events work without custom harness UI; declare ui.app(...) only when a harness needs workflow-specific terminal surfaces.",
            ),
            area,
        );
        return;
    };
    let screen_index = screen_indices
        .get(&app.id)
        .copied()
        .unwrap_or_else(|| default_screen_index(app));
    let Some(screen) = screen_at(app, screen_index) else {
        frame.render_widget(
            empty_panel(
                "Harness",
                "Selected harness app has no screens. Declare app:home(...) or app:screen(...) to render terminal surfaces.",
            ),
            area,
        );
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
    let mut action_index = 0;
    render_nodes(
        app,
        &screen.nodes,
        lists,
        requested_lists,
        &mut lines,
        0,
        max_width,
        selected_work_item_id,
        None,
        &mut action_index,
    );
    frame.render_widget(panel("Screen", lines), area);
}

pub fn render_harness_pane(
    frame: &mut Frame<'_>,
    area: Rect,
    app: Option<&UiAppRecord>,
    pane_id: Option<&str>,
    selected_work_item_id: Option<&str>,
    selected_action_index: Option<usize>,
    lists: &BTreeMap<String, WorkItemList>,
    requested_lists: &BTreeSet<String>,
) {
    let Some(app) = app else {
        return;
    };
    let Some(pane_id) = pane_id else {
        return;
    };
    let Some(pane) = app.panes.get(pane_id) else {
        return;
    };
    frame.render_widget(Clear, area);
    frame.render_widget(
        pane_panel(
            app,
            pane,
            selected_work_item_id,
            selected_action_index,
            lists,
            requested_lists,
            area.width,
        ),
        area,
    );
}

fn pane_panel(
    app: &UiAppRecord,
    pane: &UiPaneIntent,
    selected_work_item_id: Option<&str>,
    selected_action_index: Option<usize>,
    lists: &BTreeMap<String, WorkItemList>,
    requested_lists: &BTreeSet<String>,
    width: u16,
) -> Paragraph<'static> {
    panel(
        "Pane",
        pane_lines(
            app,
            pane,
            selected_work_item_id,
            selected_action_index,
            lists,
            requested_lists,
            width,
        ),
    )
}

fn pane_lines(
    app: &UiAppRecord,
    pane: &UiPaneIntent,
    selected_work_item_id: Option<&str>,
    selected_action_index: Option<usize>,
    lists: &BTreeMap<String, WorkItemList>,
    requested_lists: &BTreeSet<String>,
    width: u16,
) -> Vec<Line<'static>> {
    let mut lines = vec![
        Line::from(vec![
            Span::styled(pane.title.clone(), theme::title()),
            Span::styled(format!("  {}", pane.id), theme::muted()),
        ]),
        Line::from(""),
    ];
    if let Some(presentation) = pane.presentation.as_ref() {
        lines.push(Line::from(Span::styled(
            format!("presentation={presentation}"),
            theme::muted(),
        )));
        lines.push(Line::from(""));
    }
    let max_width = width.saturating_sub(4) as usize;
    if pane.nodes.is_empty() {
        lines.push(Line::from(Span::styled(
            "This pane has no content nodes.",
            theme::muted(),
        )));
    } else {
        let mut action_index = 0;
        render_nodes(
            app,
            &pane.nodes,
            lists,
            requested_lists,
            &mut lines,
            0,
            max_width,
            selected_work_item_id,
            selected_action_index,
            &mut action_index,
        );
    }
    lines.push(Line::from(""));
    let action_count = collect_actions(app, &pane.nodes).len();
    let item_count = collect_work_item_selections(&pane.nodes, lists).len();
    let hint = if item_count > 0 && action_count > 0 {
        "f switches items/actions  j/k moves  Enter selects/runs  Esc/q closes pane"
    } else if item_count > 0 {
        "j/k selects item  Enter queues item action  Esc/q closes pane"
    } else if action_count > 0 {
        "j/k selects pane action  Enter runs selected action  Esc/q closes pane"
    } else {
        "Esc/q closes pane"
    };
    lines.push(Line::from(Span::styled(hint, theme::muted())));
    lines
}

fn render_nodes(
    app: &UiAppRecord,
    nodes: &[UiNode],
    lists: &BTreeMap<String, WorkItemList>,
    requested_lists: &BTreeSet<String>,
    lines: &mut Vec<Line<'static>>,
    depth: usize,
    max_width: usize,
    selected_work_item_id: Option<&str>,
    selected_action_index: Option<usize>,
    action_index: &mut usize,
) {
    for node in nodes {
        match node {
            UiNode::Text(text) => {
                lines.push(indent_line(depth, text.text.clone(), theme::base()));
                lines.push(Line::from(""));
            }
            UiNode::Section(section) => {
                lines.push(indent_line(
                    depth,
                    title_with_node_badge(app, section.id.as_deref(), &section.title),
                    theme::accent(),
                ));
                render_nodes(
                    app,
                    &section.nodes,
                    lists,
                    requested_lists,
                    lines,
                    depth + 1,
                    max_width,
                    selected_work_item_id,
                    selected_action_index,
                    action_index,
                );
            }
            UiNode::Action(action) => {
                let selected = selected_action_index == Some(*action_index);
                *action_index += 1;
                let marker = if selected {
                    "●"
                } else if action.confirm {
                    "!"
                } else {
                    "→"
                };
                lines.push(indent_line(
                    depth,
                    format!(
                        "{marker} {}",
                        title_with_node_badge(app, action.id.as_deref(), &action.label)
                    ),
                    if selected {
                        theme::selected()
                    } else if action.confirm {
                        theme::warning()
                    } else {
                        theme::base()
                    },
                ));
            }
            UiNode::List(list) => render_list(
                app,
                list,
                lists,
                requested_lists,
                lines,
                depth,
                max_width,
                selected_work_item_id,
            ),
            UiNode::Form(form) => {
                let selected = selected_action_index == Some(*action_index);
                *action_index += 1;
                render_form(app, form, lines, depth, selected)
            }
            UiNode::Activity(activity) => {
                render_activity(app, activity, lists, requested_lists, lines, depth)
            }
            UiNode::Detail(detail) => {
                render_detail(app, detail, lists, requested_lists, lines, depth)
            }
            UiNode::Report(report) => {
                render_report(app, report, lists, requested_lists, lines, depth)
            }
            UiNode::Chart(chart) => render_chart(app, chart, lists, requested_lists, lines, depth),
        }
    }
}

fn render_list(
    app: &UiAppRecord,
    list: &UiListNode,
    lists: &BTreeMap<String, WorkItemList>,
    requested_lists: &BTreeSet<String>,
    lines: &mut Vec<Line<'static>>,
    depth: usize,
    max_width: usize,
    selected_work_item_id: Option<&str>,
) {
    let mut meta = vec![list.source.clone()];
    if let Some(intent) = &list.intent {
        meta.push(format!("intent={intent}"));
    }
    if let Some(render_as) = &list.render_as {
        meta.push(format!("as={render_as}"));
    }
    meta.extend(list_metadata_parts(list));
    if let Some(limit) = list.limit {
        meta.push(format!("limit={limit}"));
    }
    lines.push(indent_line(
        depth,
        format!(
            "{}  {}",
            title_with_node_badge(app, list.id.as_deref(), &list.title),
            meta.join("  ")
        ),
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
    if !is_worklist_ui_source(&list.source) {
        lines.push(indent_line(
            depth + 1,
            unsupported_source_line("list", &list.source),
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
        Some(items) => render_work_items(
            list,
            items,
            lines,
            depth + 1,
            max_width,
            selected_work_item_id,
        ),
        None if requested_lists.contains(&key) => lines.push(indent_line(
            depth + 1,
            "Loading list data...".to_string(),
            theme::muted(),
        )),
        None => lines.push(indent_line(
            depth + 1,
            ui_data_not_loaded_message("list"),
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
    selected_work_item_id: Option<&str>,
) {
    if items.items.is_empty() {
        lines.push(indent_line(depth, empty_list_message(list), theme::muted()));
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
    let mut columns = Vec::with_capacity(fields.len() + 2);
    columns.push("#".to_string());
    columns.extend(
        fields
            .iter()
            .map(|field| sorted_field_label(field, &list.sort)),
    );
    columns.push("action".to_string());
    let widths = work_item_table_widths(&fields, max_width.saturating_sub(depth * 2));
    let selected_index = selected_work_item_index(items, selected_work_item_id);
    let (start, end) = work_item_visible_window(items.items.len(), selected_index);
    lines.push(indent_line(
        depth,
        work_item_window_summary(items.items.len(), start, end, selected_index),
        theme::muted(),
    ));
    lines.push(indent_line(
        depth,
        table_row(&columns, &widths),
        theme::muted(),
    ));
    if start > 0 {
        lines.push(indent_line(
            depth,
            format!("... {start} earlier"),
            theme::muted(),
        ));
    }
    for (index, item) in items
        .items
        .iter()
        .enumerate()
        .skip(start)
        .take(end.saturating_sub(start))
    {
        let selected = selected_index == Some(index);
        let mut values = Vec::with_capacity(fields.len() + 2);
        values.push(work_item_row_marker(index, selected));
        values.extend(
            fields
                .iter()
                .map(|field| work_item_field_label(item, field)),
        );
        values.push(work_item_action_marker(item));
        lines.push(indent_line(
            depth,
            table_row(&values, &widths),
            if selected {
                theme::selected()
            } else {
                theme::base()
            },
        ));
    }
    if end < items.items.len() {
        lines.push(indent_line(
            depth,
            format!("... {} more", items.items.len() - end),
            theme::muted(),
        ));
    }
}

fn selected_work_item_index(
    items: &WorkItemList,
    selected_work_item_id: Option<&str>,
) -> Option<usize> {
    let selected_work_item_id = selected_work_item_id?;
    let selected_numeric_id = selected_work_item_id.parse::<i64>().ok();
    items.items.iter().position(|item| {
        item.public_id == selected_work_item_id || selected_numeric_id == Some(item.id)
    })
}

fn work_item_visible_window(item_count: usize, selected_index: Option<usize>) -> (usize, usize) {
    if item_count <= WORK_ITEM_TABLE_VISIBLE_ROWS {
        return (0, item_count);
    }

    let selected_index = selected_index.unwrap_or_default().min(item_count - 1);
    let half_window = WORK_ITEM_TABLE_VISIBLE_ROWS / 2;
    let max_start = item_count - WORK_ITEM_TABLE_VISIBLE_ROWS;
    let start = selected_index.saturating_sub(half_window).min(max_start);
    (start, start + WORK_ITEM_TABLE_VISIBLE_ROWS)
}

fn work_item_window_summary(
    item_count: usize,
    start: usize,
    end: usize,
    selected_index: Option<usize>,
) -> String {
    let visible_start = start.saturating_add(1).min(item_count);
    let visible_end = end.min(item_count);
    let selected = selected_index
        .map(|index| format!(" · selected {}", index.saturating_add(1)))
        .unwrap_or_default();
    format!("Rows {visible_start}-{visible_end} of {item_count}{selected}")
}

fn work_item_row_marker(index: usize, selected: bool) -> String {
    let position = index + 1;
    if selected {
        format!("●{position}")
    } else {
        position.to_string()
    }
}

fn work_item_action_marker(item: &WorkItemDetail) -> String {
    if item.action.is_some() {
        "review".to_string()
    } else {
        "-".to_string()
    }
}

fn work_item_table_widths(fields: &[String], max_width: usize) -> Vec<usize> {
    let column_count = fields.len() + 2;
    let separator_width = column_count.saturating_sub(1) * 3;
    let fixed_width = WORK_ITEM_ROW_MARKER_WIDTH + WORK_ITEM_ACTION_MARKER_WIDTH;
    let field_width = if fields.is_empty() {
        6
    } else {
        max_width
            .saturating_sub(separator_width)
            .saturating_sub(fixed_width)
            .max(fields.len() * 6)
            / fields.len()
    }
    .clamp(6, 28);

    let mut widths = Vec::with_capacity(column_count);
    widths.push(WORK_ITEM_ROW_MARKER_WIDTH);
    widths.extend((0..fields.len()).map(|_| field_width));
    widths.push(WORK_ITEM_ACTION_MARKER_WIDTH);
    widths
}

fn render_activity(
    app: &UiAppRecord,
    activity: &UiActivityNode,
    lists: &BTreeMap<String, WorkItemList>,
    requested_lists: &BTreeSet<String>,
    lines: &mut Vec<Line<'static>>,
    depth: usize,
) {
    lines.push(indent_line(
        depth,
        format!(
            "Activity: {}  {}",
            title_with_node_badge(app, activity.id.as_deref(), &activity.title),
            activity.source
        ),
        theme::accent(),
    ));

    let Some(request) = worklist_request(&activity.source, ACTIVITY_LIMIT) else {
        lines.push(indent_line(
            depth + 1,
            unsupported_source_line("activity", &activity.source),
            theme::muted(),
        ));
        lines.push(Line::from(""));
        return;
    };

    let key = request.cache_key();
    match lists.get(&key) {
        Some(items) => render_worklist_activity(items, lines, depth + 1),
        None if requested_lists.contains(&key) => lines.push(indent_line(
            depth + 1,
            "Loading activity data...".to_string(),
            theme::muted(),
        )),
        None => lines.push(indent_line(
            depth + 1,
            ui_data_not_loaded_message("activity"),
            theme::muted(),
        )),
    }
    lines.push(Line::from(""));
}

fn render_detail(
    app: &UiAppRecord,
    detail: &UiDetailNode,
    lists: &BTreeMap<String, WorkItemList>,
    requested_lists: &BTreeSet<String>,
    lines: &mut Vec<Line<'static>>,
    depth: usize,
) {
    let source = detail
        .item_id
        .as_ref()
        .map(|item_id| format!("{} / {}", detail.source, item_id))
        .unwrap_or_else(|| detail.source.clone());
    lines.push(indent_line(
        depth,
        format!(
            "Detail: {}  {}",
            title_with_node_badge(app, detail.id.as_deref(), &detail.title),
            source
        ),
        theme::accent(),
    ));

    let Some(request) = worklist_request(&detail.source, DETAIL_LIMIT) else {
        lines.push(indent_line(
            depth + 1,
            unsupported_source_line("detail", &detail.source),
            theme::muted(),
        ));
        lines.push(Line::from(""));
        return;
    };

    let key = request.cache_key();
    match lists.get(&key) {
        Some(items) => render_worklist_detail(detail, items, lines, depth + 1),
        None if requested_lists.contains(&key) => lines.push(indent_line(
            depth + 1,
            "Loading detail data...".to_string(),
            theme::muted(),
        )),
        None => lines.push(indent_line(
            depth + 1,
            ui_data_not_loaded_message("detail"),
            theme::muted(),
        )),
    }
    lines.push(Line::from(""));
}

fn render_worklist_activity(items: &WorkItemList, lines: &mut Vec<Line<'static>>, depth: usize) {
    if items.items.is_empty() {
        lines.push(indent_line(
            depth,
            "No worklist activity yet".to_string(),
            theme::muted(),
        ));
        return;
    }

    let mut recent = items.items.iter().collect::<Vec<_>>();
    recent.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));

    for item in recent.into_iter().take(8) {
        lines.push(indent_line(
            depth,
            format!(
                "{}  {}  {}  updated {}",
                item.status,
                truncate(&item.title, 46),
                item.kind,
                item.updated_at
            ),
            theme::base(),
        ));
    }
}

fn render_worklist_detail(
    detail: &UiDetailNode,
    items: &WorkItemList,
    lines: &mut Vec<Line<'static>>,
    depth: usize,
) {
    if items.items.is_empty() {
        lines.push(indent_line(
            depth,
            "No worklist items available for detail".to_string(),
            theme::muted(),
        ));
        return;
    }

    if let Some(item_id) = detail.item_id.as_deref() {
        if let Some(item) = items
            .items
            .iter()
            .find(|item| item.public_id == item_id || item.id.to_string() == item_id)
        {
            render_work_item_detail(item, lines, depth);
        } else {
            lines.push(indent_line(
                depth,
                format!("Work item '{item_id}' was not found in the loaded detail data"),
                theme::muted(),
            ));
        }
        return;
    }

    render_worklist_snapshot(items, lines, depth);
}

fn render_worklist_snapshot(items: &WorkItemList, lines: &mut Vec<Line<'static>>, depth: usize) {
    let pending = items
        .items
        .iter()
        .filter(|item| item.status == "pending")
        .count();
    let claimed = items
        .items
        .iter()
        .filter(|item| item.status == "claimed")
        .count();
    let done = items
        .items
        .iter()
        .filter(|item| item.status == "done")
        .count();
    let failed = items
        .items
        .iter()
        .filter(|item| item.status == "failed")
        .count();

    lines.push(indent_line(
        depth,
        format!(
            "{} loaded  {pending} pending  {claimed} claimed  {done} done  {failed} failed",
            items.items.len()
        ),
        theme::base(),
    ));

    if let Some(next) = worklist_highest_priority_pending_item(items) {
        lines.push(indent_line(
            depth,
            "Highest priority pending item".to_string(),
            theme::muted(),
        ));
        render_work_item_detail(next, lines, depth + 1);
    }
}

fn render_work_item_detail(item: &WorkItemDetail, lines: &mut Vec<Line<'static>>, depth: usize) {
    lines.push(indent_line(
        depth,
        format!(
            "{}  {}  {}  priority {}  worklist {}",
            item.public_id, item.status, item.kind, item.priority, item.worklist_id
        ),
        theme::base(),
    ));
    lines.push(indent_line(depth, truncate(&item.title, 88), theme::base()));
    if item.paused {
        lines.push(indent_line(
            depth,
            "paused: yes".to_string(),
            theme::warning(),
        ));
    }
    if let Some(reason) = item.pause_reason.as_ref() {
        lines.push(indent_line(
            depth,
            format!("pause reason: {}", truncate(reason, 88)),
            theme::muted(),
        ));
    }
    if let Some(agent_id) = item.claim_agent_id.as_ref() {
        lines.push(indent_line(
            depth,
            format!("claimed by: {agent_id}"),
            theme::muted(),
        ));
    }
    if let Some(parent_id) = item.parent_id.as_ref() {
        lines.push(indent_line(
            depth,
            format!("parent: {parent_id}"),
            theme::muted(),
        ));
    }
    if let Some(prompt) = item.prompt.as_ref() {
        lines.push(indent_line(
            depth,
            format!("prompt: {}", truncate(prompt, 88)),
            theme::muted(),
        ));
    }
    if let Some(action) = item.action.as_ref() {
        lines.push(indent_line(
            depth,
            format!("action: {}", action.name),
            theme::muted(),
        ));
    }
    if let Some(reason) = item.failure_reason.as_ref() {
        lines.push(indent_line(
            depth,
            format!("failure: {}", truncate(reason, 88)),
            theme::danger(),
        ));
    }
    if let Some(metadata) = item.metadata.as_ref() {
        lines.push(indent_line(
            depth,
            format!("metadata: {}", truncate(&json_value(metadata), 88)),
            theme::muted(),
        ));
    }
}

fn render_report(
    app: &UiAppRecord,
    report: &UiReportNode,
    lists: &BTreeMap<String, WorkItemList>,
    requested_lists: &BTreeSet<String>,
    lines: &mut Vec<Line<'static>>,
    depth: usize,
) {
    lines.push(indent_line(
        depth,
        format!(
            "Report: {}  {}",
            title_with_node_badge(app, report.id.as_deref(), &report.title),
            report.source
        ),
        theme::accent(),
    ));
    if let Some(prompt) = report.prompt.as_ref() {
        lines.push(indent_line(
            depth + 1,
            format!("prompt: {}", truncate(prompt, 72)),
            theme::muted(),
        ));
    }

    let Some(request) = worklist_request(&report.source, REPORT_LIMIT) else {
        lines.push(indent_line(
            depth + 1,
            unsupported_source_line("report", &report.source),
            theme::muted(),
        ));
        lines.push(Line::from(""));
        return;
    };

    let key = request.cache_key();
    match lists.get(&key) {
        Some(items) => render_worklist_report(items, lines, depth + 1),
        None if requested_lists.contains(&key) => lines.push(indent_line(
            depth + 1,
            "Loading report data...".to_string(),
            theme::muted(),
        )),
        None => lines.push(indent_line(
            depth + 1,
            ui_data_not_loaded_message("report"),
            theme::muted(),
        )),
    }
    lines.push(Line::from(""));
}

fn render_chart(
    app: &UiAppRecord,
    chart: &UiChartNode,
    lists: &BTreeMap<String, WorkItemList>,
    requested_lists: &BTreeSet<String>,
    lines: &mut Vec<Line<'static>>,
    depth: usize,
) {
    let label = chart
        .render_as
        .as_ref()
        .map(|render_as| format!("{} as {}", chart.source, render_as))
        .unwrap_or_else(|| chart.source.clone());
    lines.push(indent_line(
        depth,
        format!(
            "Chart: {}  {}  intent={}",
            title_with_node_badge(app, chart.id.as_deref(), &chart.title),
            label,
            chart.intent.as_deref().unwrap_or("breakdown")
        ),
        theme::accent(),
    ));

    let Some(request) = worklist_request(&chart.source, CHART_LIMIT) else {
        lines.push(indent_line(
            depth + 1,
            unsupported_source_line("chart", &chart.source),
            theme::muted(),
        ));
        lines.push(Line::from(""));
        return;
    };

    let key = request.cache_key();
    match lists.get(&key) {
        Some(items) => render_worklist_chart(chart, items, lines, depth + 1),
        None if requested_lists.contains(&key) => lines.push(indent_line(
            depth + 1,
            "Loading chart data...".to_string(),
            theme::muted(),
        )),
        None => lines.push(indent_line(
            depth + 1,
            ui_data_not_loaded_message("chart"),
            theme::muted(),
        )),
    }
    lines.push(Line::from(""));
}

fn render_worklist_report(items: &WorkItemList, lines: &mut Vec<Line<'static>>, depth: usize) {
    let counts = worklist_status_counts(items);
    lines.push(indent_line(
        depth,
        format!(
            "{} loaded  {} pending  {} claimed  {} done  {} failed",
            items.items.len(),
            counts.pending,
            counts.claimed,
            counts.done,
            counts.failed
        ),
        theme::base(),
    ));
    if let Some(next) = worklist_highest_priority_pending_item(items) {
        lines.push(indent_line(
            depth,
            "Next highest-priority pending item".to_string(),
            theme::muted(),
        ));
        render_work_item_detail(next, lines, depth + 1);
    }
}

fn render_worklist_chart(
    chart: &UiChartNode,
    items: &WorkItemList,
    lines: &mut Vec<Line<'static>>,
    depth: usize,
) {
    let field = worklist_chart_group_field(chart.intent.as_deref());
    let counts = worklist_group_counts(items, field);
    if counts.is_empty() {
        lines.push(indent_line(
            depth,
            "No chart data yet".to_string(),
            theme::muted(),
        ));
        return;
    }
    let max = counts.values().copied().max().unwrap_or(1);
    for (label, count) in counts {
        let width = ((count * 18) / max).max(1);
        lines.push(indent_line(
            depth,
            format!("{:<12} {:<18} {}", label, "█".repeat(width), count),
            theme::base(),
        ));
    }
}

#[cfg(test)]
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

fn render_form(
    app: &UiAppRecord,
    form: &UiFormNode,
    lines: &mut Vec<Line<'static>>,
    depth: usize,
    selected: bool,
) {
    let style = if selected {
        theme::selected()
    } else {
        theme::accent()
    };
    let marker = if selected { "● " } else { "" };
    lines.push(indent_line(
        depth,
        format!(
            "{marker}Form: {}",
            title_with_node_badge(app, form.id.as_deref(), &form.title)
        ),
        style,
    ));
    lines.push(indent_line(
        depth + 1,
        format!("Submit action: {}", form.action),
        theme::muted(),
    ));
    for field in &form.fields {
        let kind = field.kind.as_deref().unwrap_or("value");
        let required = if field.required.unwrap_or(false) {
            " required"
        } else {
            ""
        };
        let default = field
            .default
            .as_ref()
            .or_else(|| form.params.get(&field.name))
            .map(json_value)
            .filter(|value| !value.is_empty())
            .map(|value| format!(" default={value}"))
            .unwrap_or_default();
        lines.push(indent_line(
            depth + 1,
            format!("{}: {kind}{required}{default}", field.label),
            theme::base(),
        ));
    }
    lines.push(Line::from(""));
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

fn field_label(field: &str) -> String {
    field
        .split(['_', '.'])
        .filter(|part| !part.is_empty())
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().chain(chars).collect::<String>(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn sorted_field_label(field: &str, sort: &[String]) -> String {
    let mut label = field_label(field);
    if let Some(index) = sort_field_index(field, sort) {
        label.push_str(&format!(" [sort {}]", index + 1));
    }
    label
}

fn sort_field_index(field: &str, sort: &[String]) -> Option<usize> {
    sort.iter()
        .position(|entry| sort_entry_field(entry) == field)
}

fn sort_entry_field(entry: &str) -> &str {
    let entry = entry
        .trim()
        .trim_start_matches(|ch| ch == '+' || ch == '-')
        .trim();
    let entry = entry.split_whitespace().next().unwrap_or(entry);
    entry.split(':').next().unwrap_or(entry)
}

fn worklist_request(source: &str, limit: u32) -> Option<UiListRequest> {
    ui_worklist_request(source, limit)
}

fn list_metadata_parts(list: &UiListNode) -> Vec<String> {
    let mut meta = Vec::new();
    if !list.filter.is_empty() {
        meta.push(format!("where={}", list.filter.len()));
    }
    if !list.sort.is_empty() {
        meta.push(format!("sort={}", list.sort.len()));
    }
    meta
}

fn empty_list_message(list: &UiListNode) -> String {
    if list.filter.is_empty() {
        "No matching items".to_string()
    } else {
        format!(
            "No matching items after {} declared filter(s)",
            list.filter.len()
        )
    }
}

fn unsupported_source_line(surface: &str, source: &str) -> String {
    unsupported_ui_source_message(surface, source, "the terminal")
}

pub fn form_params(form: &UiFormNode, values: &BTreeMap<String, String>) -> Result<Value, String> {
    let mut params = form.params.as_object().cloned().unwrap_or_else(Map::new);
    for field in &form.fields {
        let value = values
            .get(&field.name)
            .cloned()
            .unwrap_or_else(|| default_form_value(form, field));
        if field.required.unwrap_or(false) && value.trim().is_empty() {
            return Err(format!("Form field '{}' is required", field.label));
        }
        if value.trim().is_empty() && !field.required.unwrap_or(false) {
            continue;
        }
        params.insert(field.name.clone(), parse_form_value(field, &value)?);
    }
    Ok(Value::Object(params))
}

pub fn default_form_value(form: &UiFormNode, field: &turin_daemon_protocol::UiFormField) -> String {
    field
        .default
        .as_ref()
        .or_else(|| form.params.get(&field.name))
        .map(form_value_string)
        .or_else(|| field.options.first().map(form_value_string))
        .unwrap_or_else(|| {
            if is_bool_field(field) {
                "false".to_string()
            } else {
                String::new()
            }
        })
}

pub fn normalized_form_field_kind(field: &turin_daemon_protocol::UiFormField) -> String {
    field.kind.as_deref().unwrap_or("text").to_ascii_lowercase()
}

pub fn form_value_string(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(value) => value.clone(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::Array(_) | Value::Object(_) => value.to_string(),
    }
}

pub fn is_bool_field(field: &turin_daemon_protocol::UiFormField) -> bool {
    matches!(
        normalized_form_field_kind(field).as_str(),
        "bool" | "boolean" | "checkbox" | "switch"
    )
}

pub fn is_multiline_field(field: &turin_daemon_protocol::UiFormField) -> bool {
    matches!(
        normalized_form_field_kind(field).as_str(),
        "textarea" | "multiline" | "markdown"
    )
}

fn parse_form_value(
    field: &turin_daemon_protocol::UiFormField,
    value: &str,
) -> Result<Value, String> {
    if let Some(option) = field
        .options
        .iter()
        .find(|option| form_value_string(option) == value)
    {
        return Ok(option.clone());
    }

    match normalized_form_field_kind(field).as_str() {
        "number" | "float" | "decimal" => {
            let parsed = value
                .trim()
                .parse::<f64>()
                .map_err(|_| format!("Form field '{}' must be a valid number", field.label))?;
            Number::from_f64(parsed)
                .map(Value::Number)
                .ok_or_else(|| format!("Form field '{}' must be a finite number", field.label))
        }
        "int" | "integer" => value
            .trim()
            .parse::<i64>()
            .map(|value| Value::Number(value.into()))
            .map_err(|_| format!("Form field '{}' must be a valid integer", field.label)),
        "bool" | "boolean" | "checkbox" | "switch" => {
            Ok(Value::Bool(matches!(value, "true" | "1" | "yes" | "on")))
        }
        _ => Ok(Value::String(value.to_string())),
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
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::buffer::Buffer;
    use serde_json::{Value, json};
    use turin_daemon_protocol::{
        ScheduleActionParams, UiActionNode, UiActivityNode, UiAppIntent, UiBadgeIntent,
        UiChartNode, UiDetailNode, UiFormField, UiFormNode, UiIntent, UiIntentMessage, UiListNode,
        UiMenuIntent, UiMenuItem, UiNode, UiNoticeLevel, UiOpensWithIntent, UiPaneIntent,
        UiReportNode, UiScreenIntent, UiTextNode,
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
            UiIntentMessage::new(UiIntent::Badge(UiBadgeIntent {
                app_id: "release".to_string(),
                target: "approvals".to_string(),
                count: Some(3),
                label: None,
                level: Some(UiNoticeLevel::Info),
                data: Default::default(),
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
    fn render_smoke_shows_no_app_fallback() {
        let text = rendered_screen_text(None, BTreeMap::new());

        assert!(text.contains("Default operator console is active"));
        assert!(text.contains("Overview, Tasks, and Events"));
    }

    #[test]
    fn unsupported_source_line_names_source_and_adapter_limit() {
        let line = unsupported_source_line("list", "tables.release");

        assert!(line.contains("This list is declared and visible"));
        assert!(line.contains("source 'tables.release'"));
        assert!(line.contains("cannot load in the terminal yet"));
        assert!(line.contains("Only worklists.* sources load today"));
        assert!(line.contains("deliberate adapter for this client"));
    }

    #[test]
    fn list_metadata_parts_name_filters_and_sort() {
        let mut filter = Map::new();
        filter.insert("kind".to_string(), json!("approval"));
        filter.insert("status".to_string(), json!("pending"));
        let list = UiListNode {
            id: None,
            title: "Approvals".to_string(),
            source: "worklists.release".to_string(),
            filter,
            fields: Vec::new(),
            sort: vec!["priority".to_string()],
            limit: Some(25),
            intent: None,
            render_as: None,
        };

        assert_eq!(list_metadata_parts(&list), vec!["where=2", "sort=1"]);
    }

    #[test]
    fn empty_list_message_names_declared_filters() {
        let mut list = UiListNode {
            id: None,
            title: "Approvals".to_string(),
            source: "worklists.release".to_string(),
            filter: Map::new(),
            fields: Vec::new(),
            sort: Vec::new(),
            limit: None,
            intent: None,
            render_as: None,
        };

        assert_eq!(empty_list_message(&list), "No matching items");

        list.filter.insert("kind".to_string(), json!("approval"));
        list.filter.insert("status".to_string(), json!("pending"));

        assert_eq!(
            empty_list_message(&list),
            "No matching items after 2 declared filter(s)"
        );
    }

    #[test]
    fn sorted_field_label_marks_sorted_columns() {
        let sort = vec![
            "-priority".to_string(),
            "updated_at desc".to_string(),
            "+metadata.release".to_string(),
        ];

        assert_eq!(sorted_field_label("priority", &sort), "Priority [sort 1]");
        assert_eq!(
            sorted_field_label("updated_at", &sort),
            "Updated At [sort 2]"
        );
        assert_eq!(
            sorted_field_label("metadata.release", &sort),
            "Metadata Release [sort 3]"
        );
        assert_eq!(sorted_field_label("status", &sort), "Status");
    }

    #[test]
    fn render_smoke_shows_declared_screen_nodes() {
        let app = release_app();
        let home_index = screen_index_for_target(&app, "home").expect("home screen");
        let text = rendered_screen_text(
            Some(&app),
            BTreeMap::from([("release".to_string(), home_index)]),
        );

        assert!(text.contains("Release Desk"));
        assert!(text.contains("Ready"));
        assert!(text.contains("Seed Demo Work"));
        assert!(text.contains("Recent Release Work"));
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
        assert_eq!(work.badge.as_deref(), Some("approvals 3"));
        assert_eq!(work.badge_level, Some(UiNoticeLevel::Info));
        assert!(
            matches!(work.target, HarnessNavTarget::Menu { ref opens } if opens == "approvals")
        );

        let approvals = items
            .iter()
            .find(|item| item.label == "Approvals" && item.group == "Screens")
            .expect("approvals screen item");
        assert_eq!(approvals.badge.as_deref(), Some("3"));
        assert_eq!(approvals.badge_level, Some(UiNoticeLevel::Info));

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

    #[test]
    fn work_item_table_widths_keep_utility_columns_compact() {
        let fields = vec![
            "title".to_string(),
            "status".to_string(),
            "priority".to_string(),
        ];
        let widths = work_item_table_widths(&fields, 48);

        assert_eq!(widths.len(), fields.len() + 2);
        assert_eq!(widths[0], WORK_ITEM_ROW_MARKER_WIDTH);
        assert_eq!(widths[widths.len() - 1], WORK_ITEM_ACTION_MARKER_WIDTH);
        assert!(
            widths[1..widths.len() - 1]
                .iter()
                .all(|width| (6..=28).contains(width))
        );
    }

    #[test]
    fn work_item_table_rows_include_selection_position_and_action_marker() {
        let list = UiListNode {
            id: Some("pending-approvals".to_string()),
            title: "Pending Approvals".to_string(),
            source: "worklists.release".to_string(),
            filter: Default::default(),
            fields: vec!["title".to_string(), "status".to_string()],
            sort: Vec::new(),
            limit: Some(8),
            intent: Some("approvals".to_string()),
            render_as: Some("table".to_string()),
        };
        let mut approval = test_work_item(1, "REL-1", "Approve release");
        approval.action = Some(ScheduleActionParams {
            name: "release.approve_next".to_string(),
            params: Some(json!({ "worklist": "release" })),
        });
        let items = WorkItemList {
            worklist_id: "release".to_string(),
            items: vec![approval, test_work_item(2, "REL-2", "Run QA")],
        };
        let mut lines = Vec::new();

        render_work_items(&list, &items, &mut lines, 0, 72, Some("REL-1"));
        let text = line_text(&lines);

        assert!(text.contains("#"));
        assert!(text.contains("action"));
        assert!(text.contains("Rows 1-2 of 2 · selected 1"));
        assert!(text.contains("●1"));
        assert!(text.contains("review"));
        assert!(text.contains("Run QA"));
        assert_eq!(work_item_row_marker(1, false), "2");
    }

    #[test]
    fn work_item_table_empty_state_names_matching_items() {
        let list = UiListNode {
            id: Some("pending-approvals".to_string()),
            title: "Pending Approvals".to_string(),
            source: "worklists.release".to_string(),
            filter: Default::default(),
            fields: vec!["title".to_string(), "status".to_string()],
            sort: Vec::new(),
            limit: Some(8),
            intent: Some("approvals".to_string()),
            render_as: Some("table".to_string()),
        };
        let items = WorkItemList {
            worklist_id: "release".to_string(),
            items: Vec::new(),
        };
        let mut lines = Vec::new();

        render_work_items(&list, &items, &mut lines, 0, 72, None);
        let text = line_text(&lines);

        assert!(text.contains("No matching items"));
    }

    #[test]
    fn work_item_detail_names_pause_claim_and_parent_context() {
        let mut item = test_work_item(1, "REL-1", "Approve release");
        item.paused = true;
        item.pause_reason = Some("Waiting for sign-off".to_string());
        item.claim_agent_id = Some("release-bot".to_string());
        item.parent_id = Some("REL-0".to_string());
        let mut lines = Vec::new();

        render_work_item_detail(&item, &mut lines, 0);
        let text = line_text(&lines);

        assert!(text.contains("REL-1  pending  approval  priority 10  worklist release"));
        assert!(text.contains("paused: yes"));
        assert!(text.contains("pause reason: Waiting for sign-off"));
        assert!(text.contains("claimed by: release-bot"));
        assert!(text.contains("parent: REL-0"));
    }

    #[test]
    fn work_item_table_window_keeps_selected_row_visible() {
        let list = UiListNode {
            id: Some("pending-approvals".to_string()),
            title: "Pending Approvals".to_string(),
            source: "worklists.release".to_string(),
            filter: Default::default(),
            fields: vec!["title".to_string(), "status".to_string()],
            sort: Vec::new(),
            limit: Some(18),
            intent: Some("approvals".to_string()),
            render_as: Some("table".to_string()),
        };
        let items = WorkItemList {
            worklist_id: "release".to_string(),
            items: (1..=18)
                .map(|id| test_work_item(id, &format!("REL-{id}"), &format!("Release item {id}")))
                .collect(),
        };
        let mut lines = Vec::new();

        render_work_items(&list, &items, &mut lines, 0, 72, Some("REL-15"));
        let text = line_text(&lines);

        assert!(text.contains("Rows 7-18 of 18 · selected 15"));
        assert!(text.contains("... 6 earlier"));
        assert!(text.contains("Release item 15"));
        assert!(text.contains("●15"));
        assert!(!text.contains("Release item 6"));
        assert!(!text.contains("... 0 more"));
    }

    #[test]
    fn work_item_visible_window_starts_at_first_page_without_selection() {
        assert_eq!(work_item_visible_window(18, None), (0, 12));
        assert_eq!(work_item_visible_window(18, Some(17)), (6, 18));
        assert_eq!(work_item_visible_window(3, Some(2)), (0, 3));
        assert_eq!(work_item_window_summary(18, 0, 12, None), "Rows 1-12 of 18");
    }

    #[test]
    fn list_requests_include_worklist_backed_activity_and_detail_nodes() {
        let nodes = vec![
            UiNode::Activity(UiActivityNode {
                id: Some("release-activity".to_string()),
                title: "Release Activity".to_string(),
                source: "worklists.release".to_string(),
            }),
            UiNode::Detail(UiDetailNode {
                id: Some("release-snapshot".to_string()),
                title: "Release Snapshot".to_string(),
                source: "worklists.release".to_string(),
                item_id: None,
            }),
            UiNode::Detail(UiDetailNode {
                id: Some("external-detail".to_string()),
                title: "External Detail".to_string(),
                source: "db.incidents".to_string(),
                item_id: None,
            }),
            UiNode::Report(UiReportNode {
                id: Some("release-readiness".to_string()),
                title: "Release Readiness".to_string(),
                source: "worklists.release".to_string(),
                prompt: Some("Summarize release readiness.".to_string()),
            }),
            UiNode::Chart(UiChartNode {
                id: Some("approval-flow".to_string()),
                title: "Approval Flow".to_string(),
                source: "worklists.release".to_string(),
                intent: Some("status_breakdown".to_string()),
                render_as: Some("bar".to_string()),
            }),
        ];

        let requests = collect_list_requests(&nodes);

        assert_eq!(requests.len(), 4);
        assert_eq!(requests[0].source, "worklists.release");
        assert_eq!(requests[0].limit, Some(ACTIVITY_LIMIT));
        assert_eq!(requests[1].source, "worklists.release");
        assert_eq!(requests[1].limit, Some(DETAIL_LIMIT));
        assert_eq!(requests[2].source, "worklists.release");
        assert_eq!(requests[2].limit, Some(REPORT_LIMIT));
        assert_eq!(requests[3].source, "worklists.release");
        assert_eq!(requests[3].limit, Some(CHART_LIMIT));
    }

    #[test]
    fn rendered_lines_cover_worklist_surfaces_and_item_actions() {
        let mut approval = test_work_item(1, "REL-1", "Approve release");
        approval.action = Some(ScheduleActionParams {
            name: "release.approve_next".to_string(),
            params: Some(json!({ "worklist": "release" })),
        });
        let mut qa = test_work_item(2, "REL-2", "Run QA signoff");
        qa.kind = "qa".to_string();
        qa.status = "done".to_string();
        qa.priority = 4;

        let items = WorkItemList {
            worklist_id: "release".to_string(),
            items: vec![approval, qa],
        };
        let list = UiListNode {
            id: Some("pending-approvals".to_string()),
            title: "Pending Approvals".to_string(),
            source: "worklists.release".to_string(),
            filter: Default::default(),
            fields: vec![
                "title".to_string(),
                "status".to_string(),
                "kind".to_string(),
                "priority".to_string(),
            ],
            sort: Vec::new(),
            limit: Some(8),
            intent: Some("approvals".to_string()),
            render_as: Some("table".to_string()),
        };
        let nodes = vec![
            UiNode::List(list.clone()),
            UiNode::Activity(UiActivityNode {
                id: Some("release-activity".to_string()),
                title: "Release Activity".to_string(),
                source: "worklists.release".to_string(),
            }),
            UiNode::Detail(UiDetailNode {
                id: Some("release-snapshot".to_string()),
                title: "Release Snapshot".to_string(),
                source: "worklists.release".to_string(),
                item_id: None,
            }),
            UiNode::Report(UiReportNode {
                id: Some("release-readiness".to_string()),
                title: "Release Readiness".to_string(),
                source: "worklists.release".to_string(),
                prompt: Some("Summarize release readiness.".to_string()),
            }),
            UiNode::Chart(UiChartNode {
                id: Some("approval-flow".to_string()),
                title: "Approval Flow".to_string(),
                source: "worklists.release".to_string(),
                intent: Some("kind_breakdown".to_string()),
                render_as: Some("bar".to_string()),
            }),
        ];

        let list_request = UiListRequest {
            source: list.source.clone(),
            filter: list.filter.clone(),
            limit: list.limit,
        };
        let mut lists = BTreeMap::from([(list_request.cache_key(), items.clone())]);
        for limit in [ACTIVITY_LIMIT, DETAIL_LIMIT, REPORT_LIMIT, CHART_LIMIT] {
            let request = worklist_request("worklists.release", limit).expect("worklist request");
            lists.insert(request.cache_key(), items.clone());
        }

        let mut app = release_app();
        app.badges.insert(
            "pending-approvals".to_string(),
            UiBadgeIntent {
                app_id: "release".to_string(),
                target: "pending-approvals".to_string(),
                count: Some(2),
                label: Some("hot".to_string()),
                level: Some(UiNoticeLevel::Warning),
                data: Default::default(),
            },
        );

        let mut lines = Vec::new();
        let mut action_index = 0;
        render_nodes(
            &app,
            &nodes,
            &lists,
            &BTreeSet::new(),
            &mut lines,
            0,
            88,
            Some("REL-1"),
            None,
            &mut action_index,
        );
        let text = line_text(&lines);

        assert!(text.contains("Pending Approvals  [hot 2]"));
        assert!(text.contains("Approve release"));
        assert!(text.contains("Activity: Release Activity  worklists.release"));
        assert!(text.contains("Detail: Release Snapshot  worklists.release"));
        assert!(text.contains("action: release.approve_next"));
        assert!(text.contains("Report: Release Readiness  worklists.release"));
        assert!(text.contains("Next highest-priority pending item"));
        assert!(text.contains("2 loaded  1 pending"));
        assert!(
            text.contains("Chart: Approval Flow  worklists.release as bar  intent=kind_breakdown")
        );
        assert!(text.contains("approval"));
        assert!(text.contains("qa"));
    }

    #[test]
    fn pane_lines_cover_context_nodes_and_close_hint() {
        let app = release_app();
        let pane = UiPaneIntent {
            app_id: "release".to_string(),
            id: "release-notes".to_string(),
            title: "Release Notes".to_string(),
            presentation: Some("sheet".to_string()),
            nodes: vec![
                UiNode::Text(UiTextNode {
                    id: None,
                    text: "A lightweight pane can hold contextual workflow surfaces.".to_string(),
                }),
                UiNode::Detail(UiDetailNode {
                    id: Some("pane-release-snapshot".to_string()),
                    title: "Current Release Snapshot".to_string(),
                    source: "worklists.release".to_string(),
                    item_id: None,
                }),
            ],
        };
        let request = worklist_request("worklists.release", DETAIL_LIMIT).expect("detail request");
        let lists = BTreeMap::from([(
            request.cache_key(),
            WorkItemList {
                worklist_id: "release".to_string(),
                items: vec![test_work_item(1, "REL-1", "Approve release")],
            },
        )]);

        let text = line_text(&pane_lines(
            &app,
            &pane,
            Some("REL-1"),
            None,
            &lists,
            &BTreeSet::new(),
            88,
        ));

        assert!(text.contains("Release Notes  release-notes"));
        assert!(text.contains("presentation=sheet"));
        assert!(text.contains("A lightweight pane can hold contextual workflow surfaces."));
        assert!(text.contains("Detail: Current Release Snapshot  worklists.release"));
        assert!(text.contains("1 loaded  1 pending"));
        assert!(text.contains("Approve release"));
        assert!(text.contains("Esc/q closes pane"));
    }

    #[test]
    fn pane_lines_mark_selected_action() {
        let app = release_app();
        let pane = UiPaneIntent {
            app_id: "release".to_string(),
            id: "release-actions".to_string(),
            title: "Release Actions".to_string(),
            presentation: Some("sheet".to_string()),
            nodes: vec![
                UiNode::Action(UiActionNode {
                    id: Some("approve-now".to_string()),
                    label: "Approve now".to_string(),
                    action: "release.approve".to_string(),
                    params: json!({ "force": true }),
                    confirm: true,
                }),
                UiNode::Form(UiFormNode {
                    id: Some("defer-release".to_string()),
                    title: "Defer release".to_string(),
                    action: "release.defer".to_string(),
                    params: json!({}),
                    fields: Vec::new(),
                }),
            ],
        };

        let text = line_text(&pane_lines(
            &app,
            &pane,
            None,
            Some(1),
            &BTreeMap::new(),
            &BTreeSet::new(),
            88,
        ));

        assert!(text.contains("! Approve now"));
        assert!(text.contains("● Form: Defer release"));
        assert!(text.contains("j/k selects pane action"));
        assert!(text.contains("Enter runs selected action"));
    }

    #[test]
    fn pane_lines_cover_worklist_rows_and_item_hint() {
        let app = release_app();
        let list = UiListNode {
            id: Some("pane-list".to_string()),
            title: "Pane List".to_string(),
            source: "worklists.release".to_string(),
            filter: Default::default(),
            fields: vec!["title".to_string(), "status".to_string()],
            sort: Vec::new(),
            limit: Some(5),
            intent: Some("pane".to_string()),
            render_as: Some("table".to_string()),
        };
        let pane = UiPaneIntent {
            app_id: "release".to_string(),
            id: "release-list-pane".to_string(),
            title: "Release List Pane".to_string(),
            presentation: Some("sheet".to_string()),
            nodes: vec![UiNode::List(list.clone())],
        };
        let request = UiListRequest {
            source: list.source,
            filter: list.filter,
            limit: list.limit,
        };
        let lists = BTreeMap::from([(
            request.cache_key(),
            WorkItemList {
                worklist_id: "release".to_string(),
                items: vec![test_work_item(1, "REL-1", "Approve release")],
            },
        )]);

        let text = line_text(&pane_lines(
            &app,
            &pane,
            Some("REL-1"),
            None,
            &lists,
            &BTreeSet::new(),
            88,
        ));

        assert!(text.contains("Pane List"));
        assert!(text.contains("Approve release"));
        assert!(text.contains("j/k selects item"));
        assert!(text.contains("Enter queues item action"));
    }

    #[test]
    fn pane_lines_show_empty_state() {
        let app = release_app();
        let pane = UiPaneIntent {
            app_id: "release".to_string(),
            id: "empty-pane".to_string(),
            title: "Empty Pane".to_string(),
            presentation: None,
            nodes: Vec::new(),
        };

        let text = line_text(&pane_lines(
            &app,
            &pane,
            None,
            None,
            &BTreeMap::new(),
            &BTreeSet::new(),
            88,
        ));

        assert!(text.contains("Empty Pane  empty-pane"));
        assert!(text.contains("This pane has no content nodes."));
        assert!(text.contains("Esc/q closes pane"));
    }

    #[test]
    fn work_item_selections_follow_visible_worklist_list_nodes() {
        let list = UiListNode {
            id: Some("recent-release-work".to_string()),
            title: "Recent Release Work".to_string(),
            source: "worklists.release".to_string(),
            filter: Default::default(),
            fields: Vec::new(),
            sort: Vec::new(),
            limit: Some(8),
            intent: Some("tasks".to_string()),
            render_as: Some("table".to_string()),
        };
        let request = UiListRequest {
            source: list.source.clone(),
            filter: list.filter.clone(),
            limit: list.limit,
        };
        let lists = BTreeMap::from([(
            request.cache_key(),
            WorkItemList {
                worklist_id: "release".to_string(),
                items: vec![
                    test_work_item(1, "item-1", "Ship release"),
                    test_work_item(2, "item-2", "Approve release"),
                ],
            },
        )]);
        let nodes = vec![
            UiNode::List(list),
            UiNode::List(UiListNode {
                id: Some("external".to_string()),
                title: "External".to_string(),
                source: "db.incidents".to_string(),
                filter: Default::default(),
                fields: Vec::new(),
                sort: Vec::new(),
                limit: None,
                intent: None,
                render_as: None,
            }),
        ];

        let selections = collect_work_item_selections(&nodes, &lists);

        assert_eq!(selections.len(), 2);
        assert_eq!(selections[0].list_title, "Recent Release Work");
        assert_eq!(selections[0].list_source, "worklists.release");
        assert_eq!(selections[0].item.public_id, "item-1");
        assert_eq!(selections[1].item.title, "Approve release");
    }

    #[test]
    fn collect_actions_preserves_form_metadata_for_terminal_editing() {
        let registry = UiRegistry::from_messages([
            UiIntentMessage::new(UiIntent::App(UiAppIntent {
                id: "release".to_string(),
                title: "Release Operator".to_string(),
                about: None,
                icon: None,
            })),
            UiIntentMessage::new(UiIntent::Screen(UiScreenIntent {
                app_id: "release".to_string(),
                id: "intake".to_string(),
                title: "Intake".to_string(),
                presentation: None,
                nodes: vec![UiNode::Form(UiFormNode {
                    id: Some("seed-demo-form".to_string()),
                    title: "Create Demo Approval Batch".to_string(),
                    action: "release.seed_demo_work".to_string(),
                    fields: vec![UiFormField {
                        name: "release".to_string(),
                        label: "Release".to_string(),
                        kind: Some("text".to_string()),
                        default: Some(json!("2026.06")),
                        required: Some(true),
                        options: Vec::new(),
                    }],
                    params: json!({ "count": 1 }),
                })],
            })),
        ]);
        let app = registry.app("release").expect("release app");
        let screen = screen_at(app, 0).expect("intake screen");

        let actions = collect_actions(app, &screen.nodes);

        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].label, "Submit Create Demo Approval Batch");
        assert_eq!(actions[0].action, "release.seed_demo_work");
        assert!(actions[0].form.is_some());
    }

    #[test]
    fn form_params_merge_typed_values_over_static_params() {
        let form = UiFormNode {
            id: Some("intake".to_string()),
            title: "Intake".to_string(),
            action: "release.create_item".to_string(),
            fields: vec![
                UiFormField {
                    name: "title".to_string(),
                    label: "Title".to_string(),
                    kind: Some("text".to_string()),
                    default: None,
                    required: Some(true),
                    options: Vec::new(),
                },
                UiFormField {
                    name: "priority".to_string(),
                    label: "Priority".to_string(),
                    kind: Some("integer".to_string()),
                    default: Some(json!(3)),
                    required: None,
                    options: Vec::new(),
                },
                UiFormField {
                    name: "kind".to_string(),
                    label: "Kind".to_string(),
                    kind: Some("text".to_string()),
                    default: None,
                    required: None,
                    options: vec![json!("approval"), json!("qa")],
                },
                UiFormField {
                    name: "urgent".to_string(),
                    label: "Urgent".to_string(),
                    kind: Some("bool".to_string()),
                    default: None,
                    required: None,
                    options: Vec::new(),
                },
            ],
            params: json!({
                "kind": "task",
                "source": "tui",
            }),
        };
        let values = BTreeMap::from([
            ("title".to_string(), "Ship 0.31.0".to_string()),
            ("priority".to_string(), "8".to_string()),
            ("kind".to_string(), "approval".to_string()),
            ("urgent".to_string(), "true".to_string()),
        ]);

        let params = form_params(&form, &values).expect("form params");

        assert_eq!(params["title"], json!("Ship 0.31.0"));
        assert_eq!(params["priority"], json!(8));
        assert_eq!(params["kind"], json!("approval"));
        assert_eq!(params["urgent"], json!(true));
        assert_eq!(params["source"], json!("tui"));
    }

    #[test]
    fn form_params_preserve_static_params_for_optional_blank_fields() {
        let form = UiFormNode {
            id: Some("intake".to_string()),
            title: "Intake".to_string(),
            action: "release.create_item".to_string(),
            fields: vec![UiFormField {
                name: "priority".to_string(),
                label: "Priority".to_string(),
                kind: Some("integer".to_string()),
                default: None,
                required: None,
                options: Vec::new(),
            }],
            params: json!({
                "priority": 3,
                "source": "tui",
            }),
        };
        let values = BTreeMap::from([("priority".to_string(), String::new())]);

        let params = form_params(&form, &values).expect("form params");

        assert_eq!(params["priority"], json!(3));
        assert_eq!(params["source"], json!("tui"));
    }

    #[test]
    fn form_params_validate_required_and_numeric_fields() {
        let form = UiFormNode {
            id: None,
            title: "Schedule".to_string(),
            action: "release.schedule".to_string(),
            fields: vec![
                UiFormField {
                    name: "title".to_string(),
                    label: "Title".to_string(),
                    kind: Some("text".to_string()),
                    default: None,
                    required: Some(true),
                    options: Vec::new(),
                },
                UiFormField {
                    name: "count".to_string(),
                    label: "Count".to_string(),
                    kind: Some("integer".to_string()),
                    default: None,
                    required: None,
                    options: Vec::new(),
                },
            ],
            params: Value::Null,
        };

        let missing_title = BTreeMap::from([("count".to_string(), "2".to_string())]);
        let bad_count = BTreeMap::from([
            ("title".to_string(), "Run checks".to_string()),
            ("count".to_string(), "two".to_string()),
        ]);

        assert!(
            form_params(&form, &missing_title)
                .expect_err("required error")
                .contains("Title")
        );
        assert!(
            form_params(&form, &bad_count)
                .expect_err("integer error")
                .contains("Count")
        );
    }

    fn test_work_item(id: i64, public_id: &str, title: &str) -> WorkItemDetail {
        WorkItemDetail {
            id,
            public_id: public_id.to_string(),
            worklist_id: "release".to_string(),
            parent_id: None,
            title: title.to_string(),
            kind: "approval".to_string(),
            prompt: None,
            content: None,
            tools: None,
            conflict_policy: None,
            action: None,
            status: "pending".to_string(),
            paused: false,
            pause_reason: None,
            pause_until_unix_ms: None,
            priority: 10,
            after: None,
            metadata: Some(json!({ "release": "2026.06" })),
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

    fn line_text(lines: &[Line<'static>]) -> String {
        lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn rendered_screen_text(
        app: Option<&UiAppRecord>,
        screen_indices: BTreeMap<String, usize>,
    ) -> String {
        let backend = TestBackend::new(96, 32);
        let mut terminal = Terminal::new(backend).expect("test terminal");
        terminal
            .draw(|frame| {
                render_harness_screen(
                    frame,
                    frame.area(),
                    app,
                    &screen_indices,
                    &BTreeMap::new(),
                    &BTreeSet::new(),
                    None,
                );
            })
            .expect("draw harness screen");
        buffer_text(terminal.backend().buffer())
    }

    fn buffer_text(buffer: &Buffer) -> String {
        let area = *buffer.area();
        let mut out = String::new();
        for y in area.top()..area.bottom() {
            for x in area.left()..area.right() {
                if let Some(cell) = buffer.cell((x, y)) {
                    out.push_str(cell.symbol());
                }
            }
            out.push('\n');
        }
        out
    }
}
