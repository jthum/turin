use std::collections::{BTreeMap, BTreeSet};

use eframe::egui::{self, RichText};
use serde_json::Value;
use turin_daemon_protocol::{
    UiActionNode, UiActivityNode, UiChartNode, UiDetailNode, UiFormNode, UiListNode, UiMenuItem,
    UiNode, UiReportNode, UiSectionNode, UiTextNode, WorkItemDetail, WorkItemList,
};
use turin_ui_core::{UiAppRecord, UiListRequest};

use crate::presentation::{status_intent, truncate_for_list, ui_app_title};

#[derive(Debug, Clone)]
pub(super) enum HarnessUiEvent {
    OpenScreen(String),
    RunAction {
        label: String,
        action: String,
        params: Value,
        confirm: bool,
    },
}

pub(super) fn default_screen_index(app: &UiAppRecord) -> usize {
    let screens = app.screens.values().collect::<Vec<_>>();
    app.opens_with
        .as_deref()
        .and_then(|target| screens.iter().position(|screen| screen.id == target))
        .unwrap_or_default()
}

pub(super) fn render_harness_app(
    ui: &mut egui::Ui,
    app: &UiAppRecord,
    screen_index: &mut usize,
    lists: &BTreeMap<String, WorkItemList>,
    requested_lists: &BTreeSet<String>,
) -> Option<HarnessUiEvent> {
    let mut event = None;
    let screens = app.screens.values().collect::<Vec<_>>();
    *screen_index = if screens.is_empty() {
        0
    } else {
        (*screen_index).min(screens.len() - 1)
    };

    cast::Panel::new().show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.heading(ui_app_title(app));
            ui.add_space(8.0);
            ui.add(cast::Badge::new(app.id.clone()).variant(cast::Variant::Outline));
            ui.add(cast::Badge::new(format!("{} screens", screens.len())));
            ui.add(cast::Badge::new(format!("{} menus", app.menus.len())));
        });

        if let Some(definition) = &app.definition
            && let Some(about) = &definition.about
        {
            ui.add_space(6.0);
            ui.add(cast::Markdown::new(about.clone()).selectable(true));
        }

        ui.add_space(10.0);
        render_screen_nav(ui, app, &screens, screen_index, &mut event);
        ui.add_space(10.0);

        if let Some(screen) = screens.get(*screen_index) {
            ui.horizontal_wrapped(|ui| {
                ui.heading(screen.title.clone());
                if let Some(presentation) = &screen.presentation {
                    ui.add(
                        cast::Badge::new(presentation.clone())
                            .intent(cast::Intent::Info)
                            .variant(cast::Variant::Outline),
                    );
                }
            });
            ui.add_space(8.0);
            render_nodes(ui, &screen.nodes, lists, requested_lists, &mut event);
        } else {
            ui.label("This app has no declared screens yet.");
        }
    });

    event
}

fn render_screen_nav(
    ui: &mut egui::Ui,
    app: &UiAppRecord,
    screens: &[&turin_daemon_protocol::UiScreenIntent],
    screen_index: &mut usize,
    event: &mut Option<HarnessUiEvent>,
) {
    if !screens.is_empty() {
        let labels = screens
            .iter()
            .map(|screen| screen.title.clone())
            .collect::<Vec<_>>();
        ui.add(cast::Tabs::new(screen_index, labels).size(cast::Size::Small));
    }

    if app.menus.is_empty() {
        return;
    }

    ui.add_space(8.0);
    for menu in &app.menus {
        cast::Panel::new().show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(menu.title.clone()).strong());
                render_menu_items(ui, &menu.items, event);
            });
        });
    }
}

fn render_menu_items(ui: &mut egui::Ui, items: &[UiMenuItem], event: &mut Option<HarnessUiEvent>) {
    for item in items {
        let mut label = item.label.clone();
        if let Some(badge) = &item.badge {
            label.push_str(" (");
            label.push_str(badge);
            label.push(')');
        }
        if ui
            .add(
                cast::Button::new(label)
                    .size(cast::Size::Small)
                    .variant(cast::Variant::Ghost),
            )
            .clicked()
        {
            *event = Some(HarnessUiEvent::OpenScreen(item.opens.clone()));
        }
        if !item.items.is_empty() {
            ui.add(cast::Badge::new("subnav").variant(cast::Variant::Outline));
            render_menu_items(ui, &item.items, event);
        }
    }
}

fn render_nodes(
    ui: &mut egui::Ui,
    nodes: &[UiNode],
    lists: &BTreeMap<String, WorkItemList>,
    requested_lists: &BTreeSet<String>,
    event: &mut Option<HarnessUiEvent>,
) {
    if nodes.is_empty() {
        ui.label("This screen has no content nodes.");
        return;
    }

    for node in nodes {
        match node {
            UiNode::Section(section) => render_section(ui, section, lists, requested_lists, event),
            UiNode::Text(text) => render_text(ui, text),
            UiNode::Action(action) => render_action(ui, action, event),
            UiNode::List(list) => render_list(ui, list, lists, requested_lists),
            UiNode::Activity(activity) => render_activity(ui, activity),
            UiNode::Detail(detail) => render_detail(ui, detail),
            UiNode::Form(form) => render_form(ui, form, event),
            UiNode::Report(report) => render_report(ui, report),
            UiNode::Chart(chart) => render_chart(ui, chart),
        }
        ui.add_space(10.0);
    }
}

fn render_section(
    ui: &mut egui::Ui,
    section: &UiSectionNode,
    lists: &BTreeMap<String, WorkItemList>,
    requested_lists: &BTreeSet<String>,
    event: &mut Option<HarnessUiEvent>,
) {
    cast::Panel::new().show(ui, |ui| {
        ui.heading(section.title.clone());
        ui.add_space(8.0);
        render_nodes(ui, &section.nodes, lists, requested_lists, event);
    });
}

fn render_text(ui: &mut egui::Ui, text: &UiTextNode) {
    ui.add(cast::Markdown::new(text.text.clone()).selectable(true));
}

fn render_action(ui: &mut egui::Ui, action: &UiActionNode, event: &mut Option<HarnessUiEvent>) {
    let response = ui.add(
        cast::Button::new(action.label.clone())
            .intent(if action.confirm {
                cast::Intent::Warning
            } else {
                cast::Intent::Primary
            })
            .variant(if action.confirm {
                cast::Variant::Outline
            } else {
                cast::Variant::Solid
            }),
    );
    if response.clicked() {
        *event = Some(HarnessUiEvent::RunAction {
            label: action.label.clone(),
            action: action.action.clone(),
            params: action.params.clone(),
            confirm: action.confirm,
        });
    }
}

fn render_list(
    ui: &mut egui::Ui,
    list: &UiListNode,
    lists: &BTreeMap<String, WorkItemList>,
    requested_lists: &BTreeSet<String>,
) {
    cast::Panel::new().show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(list.title.clone()).strong());
            ui.add(cast::Badge::new(list.source.clone()).variant(cast::Variant::Outline));
            if let Some(intent) = &list.intent {
                ui.add(cast::Badge::new(intent.clone()).intent(cast::Intent::Info));
            }
            if let Some(render_as) = &list.render_as {
                ui.add(cast::Badge::new(format!("as {render_as}")));
            }
        });
        ui.add_space(8.0);

        if !list.source.starts_with("worklists.") {
            ui.label(format!(
                "List source '{}' is declared but this client only knows worklist-backed lists yet.",
                list.source
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
            Some(items) => render_work_items(ui, list, items),
            None if requested_lists.contains(&key) => {
                ui.horizontal_wrapped(|ui| {
                    ui.add(cast::Loader::new().size(cast::Size::Small));
                    ui.label("Loading list data...");
                });
            }
            None => {
                ui.label("List data has not loaded yet.");
            }
        }
    });
}

fn render_work_items(ui: &mut egui::Ui, list: &UiListNode, items: &WorkItemList) {
    if items.items.is_empty() {
        ui.label("No items.");
        return;
    }

    let fields = if list.fields.is_empty() {
        vec![
            "title".to_string(),
            "status".to_string(),
            "public_id".to_string(),
            "priority".to_string(),
        ]
    } else {
        list.fields.clone()
    };
    let columns = fields
        .iter()
        .map(|field| field_label(field))
        .collect::<Vec<_>>();

    cast::Table::new(columns)
        .size(cast::Size::Small)
        .show(ui, items.items.len(), |row, index| {
            let item = &items.items[index];
            for field in &fields {
                if field == "status" {
                    row.cell(|ui| {
                        ui.add(
                            cast::Badge::new(item.status.clone())
                                .intent(status_intent(&item.status))
                                .status_dot(),
                        );
                    });
                } else {
                    row.text(work_item_field(item, field));
                }
            }
        });
}

fn render_activity(ui: &mut egui::Ui, activity: &UiActivityNode) {
    render_placeholder(
        ui,
        &activity.title,
        "activity",
        &activity.source,
        "Activity streams are declared but not rendered by turin-app yet.",
    );
}

fn render_detail(ui: &mut egui::Ui, detail: &UiDetailNode) {
    let source = detail
        .item_id
        .as_ref()
        .map(|item_id| format!("{} / {}", detail.source, item_id))
        .unwrap_or_else(|| detail.source.clone());
    render_placeholder(
        ui,
        &detail.title,
        "detail",
        &source,
        "Detail views are declared but not rendered by turin-app yet.",
    );
}

fn render_form(ui: &mut egui::Ui, form: &UiFormNode, event: &mut Option<HarnessUiEvent>) {
    cast::Panel::new().show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(form.title.clone()).strong());
            ui.add(cast::Badge::new("form").variant(cast::Variant::Outline));
        });
        ui.add_space(8.0);
        for field in &form.fields {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new(field.label.clone()).strong());
                ui.add(cast::Badge::new(field.name.clone()));
                if let Some(kind) = &field.kind {
                    ui.add(cast::Badge::new(kind.clone()).variant(cast::Variant::Outline));
                }
                if field.required.unwrap_or(false) {
                    ui.add(cast::Badge::new("required").intent(cast::Intent::Warning));
                }
            });
        }
        ui.add_space(8.0);
        if ui
            .add(
                cast::Button::new(format!("Run {}", form.action))
                    .variant(cast::Variant::Outline)
                    .intent(cast::Intent::Primary),
            )
            .clicked()
        {
            *event = Some(HarnessUiEvent::RunAction {
                label: form.title.clone(),
                action: form.action.clone(),
                params: form.params.clone(),
                confirm: false,
            });
        }
    });
}

fn render_report(ui: &mut egui::Ui, report: &UiReportNode) {
    render_placeholder(
        ui,
        &report.title,
        "report",
        &report.source,
        "Reports are declared but not rendered by turin-app yet.",
    );
    if let Some(prompt) = &report.prompt {
        ui.add(cast::Markdown::new(prompt.clone()));
    }
}

fn render_chart(ui: &mut egui::Ui, chart: &UiChartNode) {
    let source = chart
        .render_as
        .as_ref()
        .map(|render_as| format!("{} as {}", chart.source, render_as))
        .unwrap_or_else(|| chart.source.clone());
    render_placeholder(
        ui,
        &chart.title,
        chart.intent.as_deref().unwrap_or("chart"),
        &source,
        "Charts are declared but not rendered by turin-app yet.",
    );
}

fn render_placeholder(ui: &mut egui::Ui, title: &str, kind: &str, source: &str, message: &str) {
    cast::Panel::new().show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(title).strong());
            ui.add(cast::Badge::new(kind.to_string()).variant(cast::Variant::Outline));
            ui.add(cast::Badge::new(source.to_string()));
        });
        ui.add_space(6.0);
        ui.label(message);
    });
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

fn work_item_field(item: &WorkItemDetail, field: &str) -> String {
    match field {
        "id" => item.id.to_string(),
        "public_id" => item.public_id.clone(),
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
            .map(value_preview)
            .unwrap_or_default(),
    }
}

fn value_preview(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => truncate_for_list(value, 80),
        Value::Array(_) | Value::Object(_) => truncate_for_list(
            &serde_json::to_string(value).unwrap_or_else(|_| String::new()),
            80,
        ),
    }
}
