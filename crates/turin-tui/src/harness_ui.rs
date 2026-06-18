use std::collections::{BTreeMap, BTreeSet};

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use serde_json::{Map, Number, Value};
use turin_daemon_protocol::{
    UiActivityNode, UiDetailNode, UiFormNode, UiListNode, UiMenuItem, UiNode, UiScreenIntent,
    WorkItemDetail, WorkItemList,
};
use turin_ui_core::{UiAppRecord, UiListRequest};

use crate::app::PendingHarnessAction;
use crate::theme;

const ACTIVITY_LIMIT: u32 = 12;
const DETAIL_LIMIT: u32 = 25;

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
            UiNode::Activity(activity) if activity.source.starts_with("worklists.") => {
                out.push(UiListRequest {
                    source: activity.source.clone(),
                    filter: Map::new(),
                    limit: Some(ACTIVITY_LIMIT),
                });
            }
            UiNode::Detail(detail) if detail.source.starts_with("worklists.") => {
                out.push(UiListRequest {
                    source: detail.source.clone(),
                    filter: Map::new(),
                    limit: Some(DETAIL_LIMIT),
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
            UiNode::Activity(activity) => {
                render_activity(activity, lists, requested_lists, lines, depth)
            }
            UiNode::Detail(detail) => render_detail(detail, lists, requested_lists, lines, depth),
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

fn render_activity(
    activity: &UiActivityNode,
    lists: &BTreeMap<String, WorkItemList>,
    requested_lists: &BTreeSet<String>,
    lines: &mut Vec<Line<'static>>,
    depth: usize,
) {
    lines.push(indent_line(
        depth,
        format!("Activity: {}  {}", activity.title, activity.source),
        theme::accent(),
    ));

    let Some(request) = worklist_request(&activity.source, ACTIVITY_LIMIT) else {
        lines.push(indent_line(
            depth + 1,
            "No terminal activity adapter exists for this source yet".to_string(),
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
            "Activity data not requested yet".to_string(),
            theme::muted(),
        )),
    }
    lines.push(Line::from(""));
}

fn render_detail(
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
        format!("Detail: {}  {}", detail.title, source),
        theme::accent(),
    ));

    let Some(request) = worklist_request(&detail.source, DETAIL_LIMIT) else {
        lines.push(indent_line(
            depth + 1,
            "No terminal detail adapter exists for this source yet".to_string(),
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
            "Detail data not requested yet".to_string(),
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

    if let Some(next) = items
        .items
        .iter()
        .filter(|item| item.status == "pending")
        .max_by_key(|item| item.priority)
    {
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
            "{}  {}  {}  priority {}",
            item.public_id, item.status, item.kind, item.priority
        ),
        theme::base(),
    ));
    lines.push(indent_line(depth, truncate(&item.title, 88), theme::base()));
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

fn worklist_request(source: &str, limit: u32) -> Option<UiListRequest> {
    source.starts_with("worklists.").then(|| UiListRequest {
        source: source.to_string(),
        filter: Map::new(),
        limit: Some(limit),
    })
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
    use serde_json::{Value, json};
    use turin_daemon_protocol::{
        UiActionNode, UiActivityNode, UiAppIntent, UiDetailNode, UiFormField, UiFormNode, UiIntent,
        UiIntentMessage, UiListNode, UiMenuIntent, UiMenuItem, UiNode, UiOpensWithIntent,
        UiScreenIntent,
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
        ];

        let requests = collect_list_requests(&nodes);

        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].source, "worklists.release");
        assert_eq!(requests[0].limit, Some(ACTIVITY_LIMIT));
        assert_eq!(requests[1].source, "worklists.release");
        assert_eq!(requests[1].limit, Some(DETAIL_LIMIT));
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
}
