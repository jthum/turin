use std::collections::{BTreeMap, BTreeSet};

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use serde_json::Value;
use turin_daemon_protocol::{
    UiFormNode, UiListNode, UiNode, UiScreenIntent, WorkItemDetail, WorkItemList,
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
        .and_then(|screen_id| {
            app.screens
                .values()
                .position(|screen| screen.id == screen_id || screen.title == screen_id)
        })
        .unwrap_or(0)
}

pub fn screen_at(app: &UiAppRecord, index: usize) -> Option<&UiScreenIntent> {
    app.screens.values().nth(index)
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
    render_nodes(&screen.nodes, lists, requested_lists, &mut lines, 0);
    frame.render_widget(panel("Screen", lines), area);
}

fn render_nodes(
    nodes: &[UiNode],
    lists: &BTreeMap<String, WorkItemList>,
    requested_lists: &BTreeSet<String>,
    lines: &mut Vec<Line<'static>>,
    depth: usize,
) {
    for node in nodes {
        match node {
            UiNode::Text(text) => {
                lines.push(indent_line(depth, text.text.clone(), theme::base()));
                lines.push(Line::from(""));
            }
            UiNode::Section(section) => {
                lines.push(indent_line(depth, section.title.clone(), theme::accent()));
                render_nodes(&section.nodes, lists, requested_lists, lines, depth + 1);
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
            UiNode::List(list) => render_list(list, lists, requested_lists, lines, depth),
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
) {
    lines.push(indent_line(
        depth,
        format!("{}  {}", list.title, list.source),
        theme::accent(),
    ));
    if !list.source.starts_with("worklists.") {
        lines.push(indent_line(
            depth + 1,
            "Unsupported list source in TUI".to_string(),
            theme::muted(),
        ));
        return;
    }

    let request = UiListRequest {
        source: list.source.clone(),
        filter: list.filter.clone(),
        limit: list.limit,
    };
    let key = request.cache_key();
    match lists.get(&key) {
        Some(items) => render_work_items(list, items, lines, depth + 1),
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
    lines.push(indent_line(depth, fields.join("  |  "), theme::muted()));
    for item in items.items.iter().take(10) {
        let row = fields
            .iter()
            .map(|field| work_item_field(item, field))
            .collect::<Vec<_>>()
            .join("  |  ");
        lines.push(indent_line(depth, truncate(&row, 110), theme::base()));
    }
    if items.items.len() > 10 {
        lines.push(indent_line(
            depth,
            format!("… {} more", items.items.len() - 10),
            theme::muted(),
        ));
    }
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
    let mut out = String::new();
    for (index, ch) in value.chars().enumerate() {
        if index >= max_chars {
            out.push_str("...");
            return out;
        }
        out.push(ch);
    }
    out
}
