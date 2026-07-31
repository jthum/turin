use std::collections::{BTreeMap, BTreeSet};

use eframe::egui::{self, RichText};
use serde_json::{Map, Value};
use turin_daemon_protocol::{
    UiActionNode, UiActivityNode, UiBadgeIntent, UiChartNode, UiDetailNode, UiFormNode, UiListNode,
    UiMenuItem, UiNode, UiNoticeLevel, UiPaneIntent, UiReportNode, UiSectionNode, UiTextNode,
    WorkItemDetail, WorkItemList,
};
use turin_ui_core::{
    DEFAULT_UI_ACTIVITY_LIMIT as ACTIVITY_LIMIT, DEFAULT_UI_CHART_LIMIT as CHART_LIMIT,
    DEFAULT_UI_DETAIL_LIMIT as DETAIL_LIMIT, DEFAULT_UI_REPORT_LIMIT as REPORT_LIMIT, UiAppRecord,
    UiListRequest, is_named_worklist_ui_source, parse_ui_form_value as parse_form_value,
    ui_badge_text as badge_text, ui_data_load_failed_message, ui_data_not_loaded_message,
    ui_form_default_value as default_form_value, ui_form_field_kind as normalized_field_kind,
    ui_form_is_password_field as is_password_field, ui_form_value_string as form_value_string,
    ui_sorted_field_label as sorted_field_label, ui_worklist_request,
    unsupported_ui_source_message, work_item_field_label, work_item_index_by_key, work_item_key,
    worklist_chart_group_field, worklist_chart_group_label, worklist_count_percent_label,
    worklist_group_counts, worklist_highest_priority_pending_item, worklist_status_counts,
};
pub(super) use turin_ui_core::{
    ui_default_screen_index as default_screen_index,
    ui_nodes_contain_target as nodes_contain_target,
    ui_screen_index_for_target as screen_index_for_target,
};

use crate::presentation::{status_intent, truncate_for_list, ui_app_title};

#[derive(Debug, Clone, PartialEq)]
pub(super) enum HarnessUiEvent {
    OpenScreen(String),
    RunAction {
        label: String,
        action: String,
        params: Value,
        confirm: bool,
    },
    FormError(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum HarnessFocusTarget {
    Screen { screen_index: usize },
    Node { screen_index: usize },
}

pub(super) struct HarnessRenderState<'a> {
    pub(super) lists: &'a BTreeMap<String, WorkItemList>,
    pub(super) requested_lists: &'a BTreeSet<String>,
    pub(super) list_errors: &'a BTreeMap<String, String>,
    pub(super) form_values: &'a mut BTreeMap<String, String>,
    pub(super) selected_list_items: &'a mut BTreeMap<String, String>,
}

struct HarnessRenderContext<'a, 'state> {
    app: &'a UiAppRecord,
    state: &'a mut HarnessRenderState<'state>,
    event: &'a mut Option<HarnessUiEvent>,
}

pub(super) fn find_focus_target(app: &UiAppRecord, target: &str) -> Option<HarnessFocusTarget> {
    if let Some(screen_index) = screen_index_for_target(app, target) {
        return Some(HarnessFocusTarget::Screen { screen_index });
    }

    for (screen_index, screen) in app.screens.values().enumerate() {
        if nodes_contain_target(&screen.nodes, target) {
            return Some(HarnessFocusTarget::Node { screen_index });
        }
    }
    None
}

pub(super) fn render_harness_app(
    ui: &mut egui::Ui,
    app: &UiAppRecord,
    screen_index: &mut usize,
    state: &mut HarnessRenderState<'_>,
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
            });
            ui.add_space(8.0);
            let mut context = HarnessRenderContext {
                app,
                state,
                event: &mut event,
            };
            render_nodes(ui, &screen.nodes, &mut context);
        } else {
            ui.label("This app has no declared screens yet.");
        }
    });

    event
}

pub(super) fn render_harness_screen_content(
    ui: &mut egui::Ui,
    app: &UiAppRecord,
    screen_index: &mut usize,
    state: &mut HarnessRenderState<'_>,
) -> Option<HarnessUiEvent> {
    let mut event = None;
    let screens = app.screens.values().collect::<Vec<_>>();
    *screen_index = if screens.is_empty() {
        0
    } else {
        (*screen_index).min(screens.len() - 1)
    };

    if let Some(screen) = screens.get(*screen_index) {
        cast::Panel::new().show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading(screen.title.clone());
            });
            ui.add_space(12.0);
            let mut context = HarnessRenderContext {
                app,
                state,
                event: &mut event,
            };
            render_nodes(ui, &screen.nodes, &mut context);
        });
    } else {
        cast::EmptyState::new("No screens declared")
            .body("This harness app exists, but it has not declared any screens yet.")
            .intent(cast::Intent::Info)
            .show(ui, |_| {});
    }

    event
}

pub(super) fn render_harness_pane(
    ui: &mut egui::Ui,
    app: &UiAppRecord,
    pane: &UiPaneIntent,
    state: &mut HarnessRenderState<'_>,
) -> Option<HarnessUiEvent> {
    let mut event = None;
    ui.horizontal_wrapped(|ui| {
        ui.heading(pane.title.clone());
    });
    ui.add_space(8.0);
    if pane.nodes.is_empty() {
        ui.label("This pane has no content nodes.");
    } else {
        let mut context = HarnessRenderContext {
            app,
            state,
            event: &mut event,
        };
        render_nodes(ui, &pane.nodes, &mut context);
    }
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
            .map(|screen| screen_nav_label(app, screen))
            .collect::<Vec<_>>();
        ui.add(cast::Tabs::new(screen_index, labels).size(cast::Size::Small));
    }

    if app.menus.is_empty() {
        return;
    }

    ui.add_space(8.0);
    let current_screen_id = screens.get(*screen_index).map(|screen| screen.id.as_str());
    for menu in &app.menus {
        cast::Panel::new().show(ui, |ui| {
            ui.label(RichText::new(menu.title.clone()).strong());
            ui.add_space(6.0);
            render_menu_items(ui, app, &menu.items, current_screen_id, 0, event);
        });
    }
}

fn render_menu_items(
    ui: &mut egui::Ui,
    app: &UiAppRecord,
    items: &[UiMenuItem],
    current_screen_id: Option<&str>,
    depth: usize,
    event: &mut Option<HarnessUiEvent>,
) {
    for item in items {
        let selected = current_screen_id == Some(item.opens.as_str());
        ui.horizontal(|ui| {
            ui.add_space(depth as f32 * 18.0);
            if ui
                .add(
                    cast::MenuItem::new(menu_item_label(app, item))
                        .size(cast::Size::Small)
                        .selected(selected)
                        .intent(if selected {
                            cast::Intent::Info
                        } else {
                            cast::Intent::Neutral
                        }),
                )
                .clicked()
            {
                *event = Some(HarnessUiEvent::OpenScreen(item.opens.clone()));
            }
        });
        if !item.items.is_empty() {
            render_menu_items(ui, app, &item.items, current_screen_id, depth + 1, event);
        }
    }
}

fn menu_item_label(app: &UiAppRecord, item: &UiMenuItem) -> String {
    let mut parts = vec![item.label.clone()];
    if let Some(badge) = badge_text(app.badges.get(&item.opens), item.badge.as_deref()) {
        parts.push(badge);
    }
    parts.join(" · ")
}

fn screen_nav_label(app: &UiAppRecord, screen: &turin_daemon_protocol::UiScreenIntent) -> String {
    badge_text(app.badges.get(&screen.id), screen.presentation.as_deref())
        .map(|badge| format!("{} · {badge}", screen.title))
        .unwrap_or_else(|| screen.title.clone())
}

fn badge_intent(badge: Option<&UiBadgeIntent>) -> cast::Intent {
    match badge.and_then(|badge| badge.level) {
        Some(UiNoticeLevel::Success) => cast::Intent::Success,
        Some(UiNoticeLevel::Warning) => cast::Intent::Warning,
        Some(UiNoticeLevel::Error) => cast::Intent::Danger,
        Some(UiNoticeLevel::Info) => cast::Intent::Info,
        None => cast::Intent::Neutral,
    }
}

fn render_nodes(ui: &mut egui::Ui, nodes: &[UiNode], context: &mut HarnessRenderContext<'_, '_>) {
    if nodes.is_empty() {
        ui.label("This screen has no content nodes.");
        return;
    }

    for node in nodes {
        match node {
            UiNode::Section(section) => render_section(ui, section, context),
            UiNode::Text(text) => render_text(ui, text),
            UiNode::Action(action) => render_action(ui, context.app, action, context.event),
            UiNode::List(list) => render_list(ui, list, context),
            UiNode::Activity(activity) => render_activity(ui, context.app, activity, context.state),
            UiNode::Detail(detail) => {
                render_detail(ui, context.app, detail, context.state, context.event)
            }
            UiNode::Form(form) => render_form(
                ui,
                context.app,
                form,
                context.state.form_values,
                context.event,
            ),
            UiNode::Report(report) => {
                render_report(ui, context.app, report, context.state, context.event)
            }
            UiNode::Chart(chart) => render_chart(ui, context.app, chart, context.state),
        }
        ui.add_space(10.0);
    }
}

fn render_section(
    ui: &mut egui::Ui,
    section: &UiSectionNode,
    context: &mut HarnessRenderContext<'_, '_>,
) {
    cast::Panel::new().show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.heading(section.title.clone());
            render_node_badge(ui, context.app, section.id.as_deref());
        });
        ui.add_space(8.0);
        render_nodes(ui, &section.nodes, context);
    });
}

fn render_text(ui: &mut egui::Ui, text: &UiTextNode) {
    ui.add(cast::Markdown::new(text.text.clone()).selectable(true));
}

fn render_action(
    ui: &mut egui::Ui,
    app: &UiAppRecord,
    action: &UiActionNode,
    event: &mut Option<HarnessUiEvent>,
) {
    ui.horizontal_wrapped(|ui| {
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
        render_node_badge(ui, app, action.id.as_deref());
        if response.clicked() {
            *event = Some(HarnessUiEvent::RunAction {
                label: action.label.clone(),
                action: action.action.clone(),
                params: action.params.clone(),
                confirm: action.confirm,
            });
        }
    });
}

fn render_node_badge(ui: &mut egui::Ui, app: &UiAppRecord, node_id: Option<&str>) {
    let Some(node_id) = node_id else {
        return;
    };
    if let Some(badge) = app.badges.get(node_id)
        && let Some(text) = badge_text(Some(badge), None)
    {
        ui.add(
            cast::Badge::new(text)
                .intent(badge_intent(Some(badge)))
                .variant(cast::Variant::Subtle),
        );
    }
}

fn node_badge_text(app: &UiAppRecord, node_id: Option<&str>) -> Option<String> {
    node_id.and_then(|node_id| {
        app.badges
            .get(node_id)
            .and_then(|badge| badge_text(Some(badge), None))
    })
}

fn node_title_with_badge(app: &UiAppRecord, node_id: Option<&str>, title: &str) -> String {
    node_badge_text(app, node_id)
        .map(|badge| format!("{title} · {badge}"))
        .unwrap_or_else(|| title.to_string())
}

fn render_list(ui: &mut egui::Ui, list: &UiListNode, context: &mut HarnessRenderContext<'_, '_>) {
    cast::Panel::new().show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(list.title.clone()).strong());
            render_node_badge(ui, context.app, list.id.as_deref());
        });
        ui.add_space(8.0);

        if !is_named_worklist_ui_source(&list.source) {
            render_unsupported_source(ui, "list", &list.source);
            return;
        }

        let request = UiListRequest {
            source: list.source.clone(),
            filter: list.filter.clone(),
            limit: list.limit,
        };
        let key = request.cache_key();
        match context.state.lists.get(&key) {
            Some(items) => render_work_items(
                ui,
                list,
                items,
                &key,
                context.state.selected_list_items,
                context.event,
            ),
            None if context.state.requested_lists.contains(&key) => {
                ui.horizontal_wrapped(|ui| {
                    ui.add(cast::Loader::new().size(cast::Size::Small));
                    ui.label("Loading list data...");
                });
            }
            None => {
                if let Some(error) = context.state.list_errors.get(&key) {
                    render_load_failed(ui, "list", error);
                } else {
                    ui.label(ui_data_not_loaded_message("list"));
                }
            }
        }
    });
}

fn render_work_items(
    ui: &mut egui::Ui,
    list: &UiListNode,
    items: &WorkItemList,
    list_key: &str,
    selected_list_items: &mut BTreeMap<String, String>,
    event: &mut Option<HarnessUiEvent>,
) {
    if items.items.is_empty() {
        render_empty_state(ui, "No matching items", &empty_list_message(list));
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
        .map(|field| sorted_field_label(field, &list.sort))
        .collect::<Vec<_>>();
    let mut columns = columns;
    columns.insert(0, String::new());
    columns.push("action".to_string());
    let selected_index =
        work_item_index_by_key(items, selected_list_items.get(list_key).map(String::as_str))
            .unwrap_or(0);
    if let Some(item) = items.items.get(selected_index) {
        selected_list_items.insert(list_key.to_string(), work_item_key(item));
    }

    ui.label(
        RichText::new(work_item_selection_summary(
            items.items.len(),
            selected_index,
        ))
        .weak(),
    );
    ui.add_space(4.0);

    cast::Table::new(columns)
        .size(cast::Size::Small)
        .selected_rows([selected_index])
        .show(ui, items.items.len(), |row, index| {
            let item = &items.items[index];
            row.centered_cell(|ui| {
                let selected = index == selected_index;
                let label = if selected { "Viewing" } else { "View" };
                if ui
                    .add(
                        cast::Button::new(label)
                            .size(cast::Size::Small)
                            .intent(if selected {
                                cast::Intent::Info
                            } else {
                                cast::Intent::Neutral
                            })
                            .variant(cast::Variant::Ghost),
                    )
                    .clicked()
                {
                    selected_list_items.insert(list_key.to_string(), work_item_key(item));
                }
            });
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
                    row.text(truncate_for_list(&work_item_field_label(item, field), 80));
                }
            }
            row.cell(|ui| {
                if item.action.is_some() {
                    ui.add(
                        cast::Badge::new(work_item_action_marker(item))
                            .intent(cast::Intent::Warning)
                            .variant(cast::Variant::Subtle),
                    );
                } else {
                    ui.label(RichText::new(work_item_action_marker(item)).weak());
                }
            });
        });

    if let Some(item) = items.items.get(selected_index) {
        ui.add_space(10.0);
        cast::Panel::new().show(ui, |ui| {
            ui.label(RichText::new("Selected item").strong());
            ui.add_space(6.0);
            render_work_item_detail(ui, item, event);
        });
    }
}

fn render_worklist_activity(ui: &mut egui::Ui, items: &WorkItemList) {
    if items.items.is_empty() {
        render_empty_state(
            ui,
            "No worklist activity yet",
            "Activity will appear after work items are created or updated.",
        );
        return;
    }

    let mut recent = items.items.iter().collect::<Vec<_>>();
    recent.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));

    for item in recent.into_iter().take(8) {
        ui.horizontal_wrapped(|ui| {
            ui.add(
                cast::Badge::new(item.status.clone())
                    .intent(status_intent(&item.status))
                    .status_dot(),
            );
            ui.label(RichText::new(item.title.clone()).strong());
            ui.add(cast::Badge::new(item.kind.clone()).variant(cast::Variant::Outline));
            ui.label(RichText::new(format!("updated {}", item.updated_at)).weak());
        });
    }
}

fn render_worklist_detail(
    ui: &mut egui::Ui,
    detail: &UiDetailNode,
    items: &WorkItemList,
    event: &mut Option<HarnessUiEvent>,
) {
    if items.items.is_empty() {
        render_empty_state(
            ui,
            "No worklist items available",
            "Detail surfaces need at least one loaded work item.",
        );
        return;
    }

    if let Some(item_id) = detail.item_id.as_deref() {
        if let Some(item) = items
            .items
            .iter()
            .find(|item| item.public_id == item_id || item.id.to_string() == item_id)
        {
            render_work_item_detail(ui, item, event);
        } else {
            ui.label(format!(
                "Work item '{item_id}' was not found in the loaded detail data."
            ));
        }
        return;
    }

    render_worklist_snapshot(ui, items, event);
}

fn render_worklist_snapshot(
    ui: &mut egui::Ui,
    items: &WorkItemList,
    event: &mut Option<HarnessUiEvent>,
) {
    let counts = worklist_status_counts(items);

    ui.horizontal_wrapped(|ui| {
        ui.add(cast::Badge::new(format!("{} loaded", items.items.len())));
        ui.add(cast::Badge::new(format!("{} pending", counts.pending)).intent(cast::Intent::Info));
        ui.add(
            cast::Badge::new(format!("{} claimed", counts.claimed)).intent(cast::Intent::Warning),
        );
        ui.add(cast::Badge::new(format!("{} done", counts.done)).intent(cast::Intent::Success));
        if counts.failed > 0 {
            ui.add(
                cast::Badge::new(format!("{} failed", counts.failed)).intent(cast::Intent::Danger),
            );
        }
    });

    if let Some(next) = worklist_highest_priority_pending_item(items) {
        ui.add_space(8.0);
        ui.label(RichText::new("Highest priority pending item").strong());
        render_work_item_detail(ui, next, event);
    }
}

fn render_work_item_detail(
    ui: &mut egui::Ui,
    item: &WorkItemDetail,
    event: &mut Option<HarnessUiEvent>,
) {
    ui.horizontal_wrapped(|ui| {
        ui.add(
            cast::Badge::new(item.status.clone())
                .intent(status_intent(&item.status))
                .status_dot(),
        );
        ui.add(cast::Badge::new(item.kind.clone()));
        ui.add(cast::Badge::new(format!("priority {}", item.priority)));
    });
    ui.add_space(6.0);
    ui.label(RichText::new(item.title.clone()).size(18.0).strong());
    if let Some(reason) = item.pause_reason.as_ref() {
        ui.add_space(6.0);
        ui.label(RichText::new(format!("Pause reason: {reason}")).weak());
    }
    if let Some(prompt) = item.prompt.as_ref() {
        ui.add_space(6.0);
        ui.add(cast::Markdown::new(prompt.clone()).selectable(true));
    }
    if item.action.is_some() {
        ui.add_space(6.0);
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Requires confirmation.").weak());
            if ui
                .add(
                    cast::Button::new("Review Action")
                        .size(cast::Size::Small)
                        .intent(cast::Intent::Warning)
                        .variant(cast::Variant::Outline),
                )
                .clicked()
                && let Some(action_event) = work_item_action_event(item)
            {
                *event = Some(action_event);
            }
        });
    }
    if let Some(reason) = item.failure_reason.as_ref() {
        ui.add_space(6.0);
        ui.label(RichText::new(reason.clone()).color(egui::Color32::from_rgb(255, 171, 145)));
    }
    ui.add_space(6.0);
    ui.collapsing("Details", |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.add(cast::Badge::new(item.public_id.clone()).variant(cast::Variant::Outline));
            for label in work_item_context_badges(item) {
                let badge = cast::Badge::new(label.clone());
                let badge = if label == "paused" {
                    badge.intent(cast::Intent::Warning)
                } else {
                    badge.variant(cast::Variant::Outline)
                };
                ui.add(badge);
            }
        });
        ui.add_space(4.0);
        ui.horizontal_wrapped(|ui| {
            for (label, value) in work_item_timeline_labels(item) {
                ui.label(RichText::new(format!("{label}: {value}")).weak());
            }
        });
        if let Some(metadata) = item.metadata.as_ref() {
            ui.add_space(6.0);
            ui.add(
                cast::CodeOutputPanel::new(
                    "Metadata",
                    serde_json::to_string_pretty(metadata).unwrap_or_else(|_| metadata.to_string()),
                )
                .height(120.0),
            );
        }
    });
}

fn render_activity(
    ui: &mut egui::Ui,
    app: &UiAppRecord,
    activity: &UiActivityNode,
    state: &HarnessRenderState<'_>,
) {
    cast::Panel::new().show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(activity.title.clone()).strong());
            render_node_badge(ui, app, activity.id.as_deref());
        });
        ui.add_space(8.0);

        let Some(request) = worklist_request(&activity.source, ACTIVITY_LIMIT) else {
            render_unsupported_source(ui, "activity", &activity.source);
            return;
        };

        let key = request.cache_key();
        match state.lists.get(&key) {
            Some(items) => render_worklist_activity(ui, items),
            None if state.requested_lists.contains(&key) => {
                ui.horizontal_wrapped(|ui| {
                    ui.add(cast::Loader::new().size(cast::Size::Small));
                    ui.label("Loading activity data...");
                });
            }
            None => {
                if let Some(error) = state.list_errors.get(&key) {
                    render_load_failed(ui, "activity", error);
                } else {
                    ui.label(ui_data_not_loaded_message("activity"));
                }
            }
        }
    });
}

fn render_detail(
    ui: &mut egui::Ui,
    app: &UiAppRecord,
    detail: &UiDetailNode,
    state: &HarnessRenderState<'_>,
    event: &mut Option<HarnessUiEvent>,
) {
    cast::Panel::new().show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(detail.title.clone()).strong());
            render_node_badge(ui, app, detail.id.as_deref());
        });
        ui.add_space(8.0);

        let Some(request) = worklist_request(&detail.source, DETAIL_LIMIT) else {
            render_unsupported_source(ui, "detail", &detail.source);
            return;
        };

        let key = request.cache_key();
        match state.lists.get(&key) {
            Some(items) => render_worklist_detail(ui, detail, items, event),
            None if state.requested_lists.contains(&key) => {
                ui.horizontal_wrapped(|ui| {
                    ui.add(cast::Loader::new().size(cast::Size::Small));
                    ui.label("Loading detail data...");
                });
            }
            None => {
                if let Some(error) = state.list_errors.get(&key) {
                    render_load_failed(ui, "detail", error);
                } else {
                    ui.label(ui_data_not_loaded_message("detail"));
                }
            }
        }
    });
}

fn worklist_request(source: &str, limit: u32) -> Option<UiListRequest> {
    ui_worklist_request(source, limit)
}

fn render_unsupported_source(ui: &mut egui::Ui, surface: &str, source: &str) {
    ui.add(
        cast::Alert::new(format!("Unsupported {surface} source"))
            .body(unsupported_source_message(surface, source))
            .intent(cast::Intent::Warning),
    );
}

fn render_load_failed(ui: &mut egui::Ui, surface: &str, error: &str) {
    ui.add(
        cast::Alert::new(format!("Failed to load {surface} data"))
            .body(ui_data_load_failed_message(surface, error))
            .intent(cast::Intent::Danger),
    );
}

fn render_form(
    ui: &mut egui::Ui,
    app: &UiAppRecord,
    form: &UiFormNode,
    form_values: &mut BTreeMap<String, String>,
    event: &mut Option<HarnessUiEvent>,
) {
    cast::Panel::new().show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new(form.title.clone()).strong());
            render_node_badge(ui, app, form.id.as_deref());
        });
        ui.add_space(8.0);
        for field in &form.fields {
            render_form_field(ui, app, form, field, form_values);
            ui.add_space(6.0);
        }
        ui.add_space(8.0);
        if ui
            .add(
                cast::Button::new("Submit")
                    .variant(cast::Variant::Solid)
                    .intent(cast::Intent::Primary),
            )
            .clicked()
        {
            match form_params(app, form, form_values) {
                Ok(params) => {
                    *event = Some(HarnessUiEvent::RunAction {
                        label: form.title.clone(),
                        action: form.action.clone(),
                        params,
                        confirm: false,
                    });
                }
                Err(message) => *event = Some(HarnessUiEvent::FormError(message)),
            }
        }
    });
}

fn render_form_field(
    ui: &mut egui::Ui,
    app: &UiAppRecord,
    form: &UiFormNode,
    field: &turin_daemon_protocol::UiFormField,
    form_values: &mut BTreeMap<String, String>,
) {
    let key = form_field_key(app, form, field);
    form_values
        .entry(key.clone())
        .or_insert_with(|| default_form_value(form, field));

    let kind = normalized_field_kind(field);
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(field.label.clone()).strong());
        if field.required.unwrap_or(false) {
            ui.add(cast::Badge::new("required").intent(cast::Intent::Warning));
        }
    });

    if !field.options.is_empty() {
        let labels = field
            .options
            .iter()
            .map(form_value_string)
            .collect::<Vec<_>>();
        let current = form_values.get(&key).cloned().unwrap_or_default();
        let mut selected = labels
            .iter()
            .position(|label| *label == current)
            .unwrap_or_default();
        ui.add(cast::Select::new(&mut selected, labels.clone()).width(240.0));
        if let Some(label) = labels.get(selected) {
            form_values.insert(key, label.clone());
        }
        return;
    }

    if matches!(kind.as_str(), "bool" | "boolean" | "checkbox" | "switch") {
        let mut checked = form_values
            .get(&key)
            .is_some_and(|value| matches!(value.as_str(), "true" | "1" | "yes" | "on"));
        ui.add(cast::Checkbox::new(&mut checked, ""));
        form_values.insert(key, checked.to_string());
        return;
    }

    let value = form_values.get_mut(&key).expect("form value initialized");
    if kind == "textarea" || kind == "markdown" {
        ui.add(
            cast::TextArea::new(value)
                .rows(3)
                .width(ui.available_width()),
        );
    } else {
        ui.add(
            cast::TextInput::new(value)
                .hint_text(field.name.clone())
                .password(is_password_field(field))
                .width(260.0),
        );
    }
}

fn form_params(
    app: &UiAppRecord,
    form: &UiFormNode,
    form_values: &BTreeMap<String, String>,
) -> Result<Value, String> {
    let mut params = form.params.as_object().cloned().unwrap_or_else(Map::new);
    for field in &form.fields {
        let key = form_field_key(app, form, field);
        let value = form_values
            .get(&key)
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

fn form_field_key(
    app: &UiAppRecord,
    form: &UiFormNode,
    field: &turin_daemon_protocol::UiFormField,
) -> String {
    format!(
        "{}:{}:{}",
        app.id,
        form.id.as_deref().unwrap_or(&form.title),
        field.name
    )
}

fn render_report(
    ui: &mut egui::Ui,
    app: &UiAppRecord,
    report: &UiReportNode,
    state: &HarnessRenderState<'_>,
    event: &mut Option<HarnessUiEvent>,
) {
    cast::ReportSection::new(node_title_with_badge(
        app,
        report.id.as_deref(),
        &report.title,
    ))
    .description(report.prompt.clone().unwrap_or_default())
    .show(ui, |ui| {
        ui.add_space(8.0);

        let Some(request) = worklist_request(&report.source, REPORT_LIMIT) else {
            render_unsupported_source(ui, "report", &report.source);
            return;
        };

        let key = request.cache_key();
        match state.lists.get(&key) {
            Some(items) => render_worklist_report(ui, items, event),
            None if state.requested_lists.contains(&key) => {
                ui.horizontal_wrapped(|ui| {
                    ui.add(cast::Loader::new().size(cast::Size::Small));
                    ui.label("Loading report data...");
                });
            }
            None => {
                if let Some(error) = state.list_errors.get(&key) {
                    render_load_failed(ui, "report", error);
                } else {
                    ui.label(ui_data_not_loaded_message("report"));
                }
            }
        }
    });
}

fn render_chart(
    ui: &mut egui::Ui,
    app: &UiAppRecord,
    chart: &UiChartNode,
    state: &HarnessRenderState<'_>,
) {
    cast::ReportSection::new(node_title_with_badge(
        app,
        chart.id.as_deref(),
        &chart.title,
    ))
    .description(format!(
        "Grouped by {}",
        worklist_chart_group_label(chart.intent.as_deref())
    ))
    .show(ui, |ui| {
        let Some(request) = worklist_request(&chart.source, CHART_LIMIT) else {
            render_unsupported_source(ui, "chart", &chart.source);
            return;
        };

        let key = request.cache_key();
        match state.lists.get(&key) {
            Some(items) => render_worklist_chart(ui, chart, items),
            None if state.requested_lists.contains(&key) => {
                ui.horizontal_wrapped(|ui| {
                    ui.add(cast::Loader::new().size(cast::Size::Small));
                    ui.label("Loading chart data...");
                });
            }
            None => {
                if let Some(error) = state.list_errors.get(&key) {
                    render_load_failed(ui, "chart", error);
                } else {
                    ui.label(ui_data_not_loaded_message("chart"));
                }
            }
        }
    });
}

fn render_worklist_report(
    ui: &mut egui::Ui,
    items: &WorkItemList,
    event: &mut Option<HarnessUiEvent>,
) {
    let counts = worklist_status_counts(items);
    ui.horizontal_wrapped(|ui| {
        ui.add(cast::Badge::new(format!("{} loaded", items.items.len())));
        ui.add(cast::Badge::new(format!("{} pending", counts.pending)).intent(cast::Intent::Info));
        ui.add(
            cast::Badge::new(format!("{} claimed", counts.claimed)).intent(cast::Intent::Warning),
        );
        ui.add(cast::Badge::new(format!("{} done", counts.done)).intent(cast::Intent::Success));
        if counts.failed > 0 {
            ui.add(
                cast::Badge::new(format!("{} failed", counts.failed)).intent(cast::Intent::Danger),
            );
        }
        if counts.other > 0 {
            ui.add(cast::Badge::new(format!("{} other", counts.other)));
        }
    });

    if items.items.is_empty() {
        ui.add_space(8.0);
        render_empty_state(
            ui,
            "No report data yet",
            "This report will populate when the backing worklist has rows.",
        );
        return;
    }

    if let Some(next) = worklist_highest_priority_pending_item(items) {
        ui.add_space(10.0);
        ui.label(RichText::new("Next highest-priority pending item").strong());
        render_work_item_detail(ui, next, event);
    }
}

fn render_worklist_chart(ui: &mut egui::Ui, chart: &UiChartNode, items: &WorkItemList) {
    let field = worklist_chart_group_field(chart.intent.as_deref());
    let counts = worklist_group_counts(items, field);
    if counts.is_empty() {
        render_empty_state(
            ui,
            "No chart data yet",
            "This chart will populate when the backing worklist has rows.",
        );
        return;
    }

    let total = items.items.len();
    let data = counts
        .iter()
        .map(|(label, count)| {
            cast::BarDatum::new(label.clone(), *count as f32).intent(chart_bar_intent(field, label))
        })
        .collect::<Vec<_>>();
    ui.add(
        cast::BarChart::new(data)
            .height(150.0)
            .width(ui.available_width()),
    );
    ui.add_space(8.0);
    ui.horizontal_wrapped(|ui| {
        for (label, count) in counts {
            ui.add(
                cast::Badge::new(format!(
                    "{label}: {count} ({})",
                    worklist_count_percent_label(count, total)
                ))
                .intent(chart_bar_intent(field, &label))
                .variant(cast::Variant::Subtle),
            );
        }
    });
}

fn chart_bar_intent(field: &str, label: &str) -> cast::Intent {
    if field == "status" {
        status_intent(label)
    } else if field == "priority" {
        cast::Intent::Warning
    } else {
        cast::Intent::Info
    }
}

fn work_item_selection_summary(item_count: usize, selected_index: usize) -> String {
    if item_count == 0 {
        return "Rows 0-0 of 0".to_string();
    }
    let selected = selected_index.saturating_add(1).min(item_count);
    format!("Rows 1-{item_count} of {item_count} · selected {selected}")
}

fn render_empty_state(ui: &mut egui::Ui, title: &str, body: &str) {
    cast::EmptyState::new(title)
        .body(body)
        .intent(cast::Intent::Neutral)
        .show(ui, |_| {});
}

fn work_item_action_event(item: &WorkItemDetail) -> Option<HarnessUiEvent> {
    let action = item.action.as_ref()?;
    Some(HarnessUiEvent::RunAction {
        label: format!("Work item: {}", item.title),
        action: action.name.clone(),
        params: action.params.clone().unwrap_or(Value::Null),
        confirm: true,
    })
}

fn work_item_action_marker(item: &WorkItemDetail) -> &'static str {
    if item.action.is_some() { "action" } else { "-" }
}

fn unsupported_source_message(surface: &str, source: &str) -> String {
    unsupported_ui_source_message(surface, source, "the desktop app")
}

#[cfg(test)]
fn list_metadata_badges(list: &UiListNode) -> Vec<String> {
    let mut meta = Vec::new();
    if !list.filter.is_empty() {
        meta.push(format!(
            "where {}",
            turin_ui_core::ui_list_filter_fields(&list.filter).join(",")
        ));
    }
    if !list.sort.is_empty() {
        meta.push(format!(
            "sort {}",
            turin_ui_core::ui_list_sort_fields(&list.sort).join(",")
        ));
    }
    if let Some(limit) = list.limit {
        meta.push(format!("limit {limit}"));
    }
    meta
}

fn empty_list_message(list: &UiListNode) -> String {
    if list.filter.is_empty() {
        "This worklist query returned no rows.".to_string()
    } else {
        format!(
            "This worklist query returned no rows after applying {} declared filter(s).",
            list.filter.len()
        )
    }
}

fn work_item_context_badges(item: &WorkItemDetail) -> Vec<String> {
    let mut badges = vec![format!("worklist {}", item.worklist_id)];
    if item.paused {
        badges.push("paused".to_string());
    }
    if let Some(agent_id) = item.claim_agent_id.as_ref() {
        badges.push(format!("claimed by {agent_id}"));
    }
    if let Some(parent_id) = item.parent_id.as_ref() {
        badges.push(format!("parent {parent_id}"));
    }
    badges
}

fn work_item_timeline_labels(item: &WorkItemDetail) -> Vec<(&'static str, String)> {
    let mut labels = vec![
        ("Created", item.created_at.clone()),
        ("Updated", item.updated_at.clone()),
    ];
    if let Some(claimed_at) = item.claimed_at.as_ref() {
        labels.push(("Claimed at", claimed_at.clone()));
    }
    if let Some(completed_at) = item.completed_at.as_ref() {
        labels.push(("Completed", completed_at.clone()));
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use turin_daemon_protocol::{
        ScheduleActionParams, UiAppIntent, UiFormField, UiIntentSource, UiScreenIntent,
    };

    fn test_app() -> UiAppRecord {
        UiAppRecord {
            id: "release".to_string(),
            source: UiIntentSource::default(),
            definition: Some(UiAppIntent {
                id: "release".to_string(),
                title: "Release".to_string(),
                about: None,
                icon: None,
            }),
            screens: BTreeMap::new(),
            panes: BTreeMap::new(),
            menus: Vec::new(),
            opens_with: None,
            badges: BTreeMap::new(),
        }
    }

    fn focus_test_app() -> UiAppRecord {
        let mut app = test_app();
        app.screens = BTreeMap::from([
            (
                "home".to_string(),
                UiScreenIntent {
                    app_id: app.id.clone(),
                    id: "home".to_string(),
                    title: "Release Desk".to_string(),
                    presentation: None,
                    nodes: vec![
                        UiNode::Action(UiActionNode {
                            id: Some("seed-demo-work".to_string()),
                            label: "Seed Demo Work".to_string(),
                            action: "release.seed_demo_work".to_string(),
                            params: Value::Null,
                            confirm: false,
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
                    ],
                },
            ),
            (
                "intake".to_string(),
                UiScreenIntent {
                    app_id: app.id.clone(),
                    id: "intake".to_string(),
                    title: "Intake".to_string(),
                    presentation: None,
                    nodes: vec![UiNode::Form(UiFormNode {
                        id: Some("seed-demo-form".to_string()),
                        title: "Create Demo Approval Batch".to_string(),
                        action: "release.seed_demo_work".to_string(),
                        fields: Vec::new(),
                        params: Value::Null,
                    })],
                },
            ),
        ]);
        app
    }

    fn screen_id_at(app: &UiAppRecord, screen_index: usize) -> Option<&str> {
        app.screens
            .values()
            .nth(screen_index)
            .map(|screen| screen.id.as_str())
    }

    #[test]
    fn menu_item_label_includes_badges_without_child_count() {
        let mut app = test_app();
        app.badges = BTreeMap::from([(
            "approvals".to_string(),
            UiBadgeIntent {
                app_id: "release".to_string(),
                target: "approvals".to_string(),
                count: Some(3),
                label: Some("ready".to_string()),
                level: Some(UiNoticeLevel::Info),
                data: Map::new(),
            },
        )]);
        let item = UiMenuItem {
            label: "Work".to_string(),
            opens: "approvals".to_string(),
            id: None,
            icon: None,
            badge: None,
            items: vec![UiMenuItem {
                label: "Review".to_string(),
                opens: "review".to_string(),
                id: None,
                icon: None,
                badge: None,
                items: Vec::new(),
            }],
        };

        assert_eq!(menu_item_label(&app, &item), "Work · ready 3");
    }

    #[test]
    fn screen_nav_label_uses_dynamic_badge_then_presentation_fallback() {
        let mut app = test_app();
        let mut screen = UiScreenIntent {
            app_id: app.id.clone(),
            id: "approvals".to_string(),
            title: "Approvals".to_string(),
            presentation: Some("workflow".to_string()),
            nodes: Vec::new(),
        };

        assert_eq!(screen_nav_label(&app, &screen), "Approvals · workflow");

        app.badges = BTreeMap::from([(
            "approvals".to_string(),
            UiBadgeIntent {
                app_id: "release".to_string(),
                target: "approvals".to_string(),
                count: Some(3),
                label: Some("ready".to_string()),
                level: Some(UiNoticeLevel::Info),
                data: Map::new(),
            },
        )]);
        assert_eq!(screen_nav_label(&app, &screen), "Approvals · ready 3");

        screen.presentation = None;
        assert_eq!(screen_nav_label(&app, &screen), "Approvals · ready 3");
    }

    #[test]
    fn node_title_with_badge_appends_titled_node_badges() {
        let mut app = test_app();
        app.badges = BTreeMap::from([(
            "release-readiness".to_string(),
            UiBadgeIntent {
                app_id: "release".to_string(),
                target: "release-readiness".to_string(),
                count: None,
                label: Some("live".to_string()),
                level: Some(UiNoticeLevel::Info),
                data: Map::new(),
            },
        )]);

        assert_eq!(
            node_title_with_badge(&app, Some("release-readiness"), "Release Readiness"),
            "Release Readiness · live"
        );
        assert_eq!(
            node_title_with_badge(&app, Some("missing"), "Release Readiness"),
            "Release Readiness"
        );
        assert_eq!(
            node_title_with_badge(&app, None, "Release Readiness"),
            "Release Readiness"
        );
    }

    #[test]
    fn unsupported_source_message_names_surface_and_source() {
        let message = unsupported_source_message("list", "tables.release");

        assert!(message.contains("This list is declared and visible"));
        assert!(message.contains("source 'tables.release'"));
        assert!(message.contains("cannot load in the desktop app yet"));
        assert!(message.contains("Only named worklists.<name> sources load today"));
        assert!(message.contains("deliberate adapter for this client"));
    }

    #[test]
    fn worklist_request_rejects_missing_worklist_name() {
        assert!(worklist_request("worklists.release", 25).is_some());
        assert!(worklist_request("worklists.", 25).is_none());
        assert!(worklist_request("worklists. ", 25).is_none());
    }

    #[test]
    fn list_metadata_badges_name_filters_sort_and_limit() {
        let list = UiListNode {
            id: None,
            title: "Approvals".to_string(),
            source: "worklists.release".to_string(),
            filter: Map::from_iter([
                ("kind".to_string(), json!("approval")),
                ("status".to_string(), json!("pending")),
            ]),
            fields: Vec::new(),
            sort: vec!["priority".to_string()],
            limit: Some(25),
            intent: None,
            render_as: None,
        };

        assert_eq!(
            list_metadata_badges(&list),
            vec!["where kind,status", "sort priority", "limit 25"]
        );
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

        assert_eq!(
            empty_list_message(&list),
            "This worklist query returned no rows."
        );

        list.filter.insert("kind".to_string(), json!("approval"));
        list.filter.insert("status".to_string(), json!("pending"));

        assert_eq!(
            empty_list_message(&list),
            "This worklist query returned no rows after applying 2 declared filter(s)."
        );
    }

    #[test]
    fn work_item_context_badges_name_pause_claim_and_parent_context() {
        let mut item = test_work_item(1, "REL-1", "Approve release");
        item.paused = true;
        item.claim_agent_id = Some("release-bot".to_string());
        item.parent_id = Some("REL-0".to_string());

        assert_eq!(
            work_item_context_badges(&item),
            vec![
                "worklist release",
                "paused",
                "claimed by release-bot",
                "parent REL-0"
            ]
        );
    }

    #[test]
    fn work_item_timeline_labels_include_base_and_optional_dates() {
        let mut item = test_work_item(1, "REL-1", "Approve release");
        item.claimed_at = Some("2026-06-18T01:00:00Z".to_string());
        item.completed_at = Some("2026-06-18T02:00:00Z".to_string());

        assert_eq!(
            work_item_timeline_labels(&item),
            vec![
                ("Created", "2026-06-18T00:00:00Z".to_string()),
                ("Updated", "2026-06-18T00:00:00Z".to_string()),
                ("Claimed at", "2026-06-18T01:00:00Z".to_string()),
                ("Completed", "2026-06-18T02:00:00Z".to_string()),
            ]
        );
    }

    #[test]
    fn work_item_action_event_requires_confirmation_and_preserves_params() {
        let mut item = test_work_item(1, "REL-1", "Approve release");
        item.action = Some(ScheduleActionParams {
            name: "release.approve_next".to_string(),
            params: Some(json!({
                "release": "2026.06",
                "mode": "hotfix"
            })),
        });

        assert_eq!(
            work_item_action_event(&item),
            Some(HarnessUiEvent::RunAction {
                label: "Work item: Approve release".to_string(),
                action: "release.approve_next".to_string(),
                params: json!({
                    "release": "2026.06",
                    "mode": "hotfix"
                }),
                confirm: true,
            })
        );
    }

    #[test]
    fn work_item_action_event_defaults_missing_params_to_null() {
        let mut item = test_work_item(1, "REL-1", "Approve release");
        item.action = Some(ScheduleActionParams {
            name: "release.approve_next".to_string(),
            params: None,
        });

        assert_eq!(
            work_item_action_event(&item),
            Some(HarnessUiEvent::RunAction {
                label: "Work item: Approve release".to_string(),
                action: "release.approve_next".to_string(),
                params: Value::Null,
                confirm: true,
            })
        );
    }

    #[test]
    fn work_item_action_marker_names_action_availability() {
        let mut item = test_work_item(1, "REL-1", "Approve release");
        assert_eq!(work_item_action_marker(&item), "-");

        item.action = Some(ScheduleActionParams {
            name: "release.approve_next".to_string(),
            params: None,
        });
        assert_eq!(work_item_action_marker(&item), "action");
    }

    #[test]
    fn work_item_action_event_ignores_items_without_actions() {
        let item = test_work_item(1, "REL-1", "Approve release");

        assert_eq!(work_item_action_event(&item), None);
    }

    #[test]
    fn sorted_field_label_marks_sorted_columns() {
        let sort = vec![
            "-priority".to_string(),
            "updated_at desc".to_string(),
            "+metadata.release".to_string(),
        ];

        assert_eq!(
            sorted_field_label("priority", &sort),
            "Priority [sort 1 desc]"
        );
        assert_eq!(
            sorted_field_label("updated_at", &sort),
            "Updated At [sort 2 desc]"
        );
        assert_eq!(
            sorted_field_label("metadata.release", &sort),
            "Metadata Release [sort 3 asc]"
        );
        assert_eq!(sorted_field_label("status", &sort), "Status");
    }

    #[test]
    fn focus_targets_resolve_screens_actions_forms_and_node_ids() {
        let app = focus_test_app();

        assert!(matches!(
            find_focus_target(&app, "Release Desk"),
            Some(HarnessFocusTarget::Screen { screen_index })
                if screen_id_at(&app, screen_index) == Some("home")
        ));
        assert!(matches!(
            find_focus_target(&app, "seed-demo-work"),
            Some(HarnessFocusTarget::Node { screen_index })
                if screen_id_at(&app, screen_index) == Some("home")
        ));
        assert!(matches!(
            find_focus_target(&app, "recent-release-work"),
            Some(HarnessFocusTarget::Node { screen_index })
                if screen_id_at(&app, screen_index) == Some("home")
        ));
        assert!(matches!(
            find_focus_target(&app, "seed-demo-form"),
            Some(HarnessFocusTarget::Node { screen_index })
                if screen_id_at(&app, screen_index) == Some("intake")
        ));
        assert_eq!(find_focus_target(&app, "unknown"), None);
    }

    #[test]
    fn work_item_selection_summary_names_rows_and_selection() {
        assert_eq!(
            work_item_selection_summary(12, 4),
            "Rows 1-12 of 12 · selected 5"
        );
        assert_eq!(work_item_selection_summary(0, 4), "Rows 0-0 of 0");
    }

    #[test]
    fn default_form_value_uses_static_params_when_field_default_is_absent() {
        let form = UiFormNode {
            id: Some("seed".to_string()),
            title: "Seed".to_string(),
            action: "release.seed".to_string(),
            fields: Vec::new(),
            params: json!({ "count": 3 }),
        };
        let field = UiFormField {
            name: "count".to_string(),
            label: "Count".to_string(),
            kind: Some("integer".to_string()),
            default: None,
            required: None,
            options: Vec::new(),
        };

        assert_eq!(default_form_value(&form, &field), "3");
    }

    #[test]
    fn form_params_preserve_static_params_for_optional_blank_fields() {
        let app = test_app();
        let form = UiFormNode {
            id: Some("seed".to_string()),
            title: "Seed".to_string(),
            action: "release.seed".to_string(),
            fields: vec![UiFormField {
                name: "count".to_string(),
                label: "Count".to_string(),
                kind: Some("integer".to_string()),
                default: None,
                required: None,
                options: Vec::new(),
            }],
            params: json!({ "count": 3, "source": "app" }),
        };
        let key = form_field_key(&app, &form, &form.fields[0]);
        let values = BTreeMap::from([(key, String::new())]);

        let params = form_params(&app, &form, &values).expect("form params");

        assert_eq!(params["count"], json!(3));
        assert_eq!(params["source"], json!("app"));
    }

    #[test]
    fn form_params_preserve_typed_option_values() {
        let app = test_app();
        let form = UiFormNode {
            id: Some("seed".to_string()),
            title: "Seed".to_string(),
            action: "release.seed".to_string(),
            fields: vec![UiFormField {
                name: "count".to_string(),
                label: "Count".to_string(),
                kind: Some("integer".to_string()),
                default: None,
                required: None,
                options: vec![json!(2), json!("3")],
            }],
            params: Value::Null,
        };
        let key = form_field_key(&app, &form, &form.fields[0]);
        let numeric_values = BTreeMap::from([(key.clone(), "2".to_string())]);
        let string_values = BTreeMap::from([(key, "3".to_string())]);

        let numeric_params = form_params(&app, &form, &numeric_values).expect("numeric option");
        let string_params = form_params(&app, &form, &string_values).expect("string option");

        assert_eq!(numeric_params["count"], json!(2));
        assert_eq!(string_params["count"], json!("3"));
    }

    fn test_work_item(id: i64, public_id: &str, title: &str) -> WorkItemDetail {
        WorkItemDetail {
            id,
            public_id: public_id.to_string(),
            worklist_id: "release".to_string(),
            parent_id: None,
            title: title.to_string(),
            kind: "approval".to_string(),
            prompt: Some("Check release gates".to_string()),
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
