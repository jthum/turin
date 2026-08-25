use eframe::egui::{self, Color32, RichText, ScrollArea};
use turin_client::SessionDetail;
use turin_ui_core::{
    ConnectionPreflightOutcome, ConnectionProfileAuth, ConnectionProfileDraftAuthMode,
    ConnectionProfileKind, DashboardFreshness, DashboardNoticeLevel, UiAppRecord,
};

use crate::SessionBranchDetail;
use turin_types::layout::{DEFAULT_BOOTSTRAP_CONFIG_PATH, DEFAULT_BOOTSTRAP_DAEMON_ENDPOINT_PATH};

pub(super) fn clamp_index(current: usize, len: usize) -> usize {
    if len == 0 { 0 } else { current.min(len - 1) }
}

pub(super) fn detail_kv(ui: &mut egui::Ui, key: &str, value: impl ToString) {
    ui.horizontal_wrapped(|ui| {
        themed_strong(ui, key);
        ui.label(value.to_string());
    });
}

pub(super) fn themed_heading(ui: &mut egui::Ui, text: impl Into<String>, size: f32) {
    let text = themed_heading_text(ui, text, size);
    ui.label(text);
}

pub(super) fn themed_heading_text(
    ui: &mut egui::Ui,
    text: impl Into<String>,
    size: f32,
) -> RichText {
    let theme = cast::theme_for_ui(ui);
    let mut font = if size >= theme.typography.heading_lg.size {
        theme.typography.heading_lg.clone()
    } else if size >= theme.typography.heading.size {
        theme.typography.heading.clone()
    } else {
        theme.typography.heading_sm.clone()
    };
    font.size = size;
    RichText::new(text.into())
        .font(font)
        .strong()
        .color(theme.colors.text)
}

pub(super) fn themed_strong(ui: &mut egui::Ui, text: impl Into<String>) {
    let text = themed_strong_text(ui, text);
    ui.label(text);
}

pub(super) fn themed_strong_text(ui: &mut egui::Ui, text: impl Into<String>) -> RichText {
    let theme = cast::theme_for_ui(ui);
    RichText::new(text.into())
        .font(theme.typography.strong.clone())
        .strong()
        .color(theme.colors.text)
}

pub(super) fn themed_muted(ui: &mut egui::Ui, text: impl Into<String>) {
    let text = themed_muted_text(ui, text);
    ui.label(text);
}

pub(super) fn themed_overline(ui: &mut egui::Ui, text: impl Into<String>) {
    let theme = cast::theme_for_ui(ui);
    let mut font = theme.typography.caption.clone();
    font.size = 10.0;
    ui.label(
        RichText::new(text.into().to_ascii_uppercase())
            .font(font)
            .strong()
            .color(theme.colors.text_muted),
    );
}

pub(super) fn themed_muted_text(ui: &mut egui::Ui, text: impl Into<String>) -> RichText {
    let theme = cast::theme_for_ui(ui);
    RichText::new(text.into())
        .font(theme.typography.body.clone())
        .color(theme.colors.text_muted)
}

pub(super) fn themed_danger_text(ui: &mut egui::Ui, text: impl Into<String>) -> RichText {
    let theme = cast::theme_for_ui(ui);
    RichText::new(text.into())
        .font(theme.typography.body.clone())
        .color(theme.colors.danger_family.emphasis)
}

pub(super) fn ui_app_title(app: &UiAppRecord) -> String {
    app.definition
        .as_ref()
        .map(|definition| definition.title.clone())
        .unwrap_or_else(|| app.id.clone())
}

pub(super) fn status_intent(status: &str) -> cast::Intent {
    let normalized = status.to_ascii_lowercase();
    if normalized.contains("fail") || normalized.contains("error") {
        cast::Intent::Danger
    } else if normalized.contains("pause")
        || normalized.contains("blocked")
        || normalized.contains("waiting")
    {
        cast::Intent::Warning
    } else if normalized.contains("done")
        || normalized.contains("complete")
        || normalized.contains("success")
    {
        cast::Intent::Success
    } else if normalized.contains("run") || normalized.contains("active") {
        cast::Intent::Info
    } else {
        cast::Intent::Neutral
    }
}

pub(super) fn chat_role_from_label(role: &str) -> cast::ChatRole {
    match role.to_ascii_lowercase().as_str() {
        "user" => cast::ChatRole::User,
        "system" => cast::ChatRole::System,
        "tool" => cast::ChatRole::Tool,
        _ => cast::ChatRole::Assistant,
    }
}

pub(super) fn tool_status_from_verdict(verdict: &str) -> cast::ToolCallStatus {
    let normalized = verdict.to_ascii_lowercase();
    if normalized.contains("fail") || normalized.contains("error") {
        cast::ToolCallStatus::Failed
    } else if normalized.contains("run") || normalized.contains("active") {
        cast::ToolCallStatus::Running
    } else if normalized.contains("queue") || normalized.contains("pending") {
        cast::ToolCallStatus::Queued
    } else {
        cast::ToolCallStatus::Succeeded
    }
}

pub(super) fn truncate_for_list(value: &str, max_chars: usize) -> String {
    let char_count = value.chars().count();
    if char_count <= max_chars {
        value.to_string()
    } else {
        let prefix: String = value.chars().take(max_chars.saturating_sub(1)).collect();
        format!("{prefix}...")
    }
}

pub(super) fn session_message_text(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter_map(session_message_part_text)
            .collect::<Vec<_>>()
            .join("\n\n"),
        serde_json::Value::Object(_) => session_message_part_text(content).unwrap_or_default(),
        _ => String::new(),
    }
}

fn session_message_part_text(part: &serde_json::Value) -> Option<String> {
    part.get("text")
        .and_then(|text| text.as_str())
        .or_else(|| part.get("content").and_then(|content| content.as_str()))
        .map(str::to_string)
}

pub(super) fn yes_no(value: bool) -> &'static str {
    if value { "Yes" } else { "No" }
}

pub(super) fn connection_kind_label(kind: turin_client::ConnectionKind) -> &'static str {
    match kind {
        turin_client::ConnectionKind::Local => "local",
        turin_client::ConnectionKind::Remote => "remote",
    }
}

pub(super) fn freshness_label(freshness: DashboardFreshness) -> &'static str {
    match freshness {
        DashboardFreshness::Fresh => "fresh",
        DashboardFreshness::Quiet => "quiet",
        DashboardFreshness::Stale => "stale",
    }
}

pub(super) fn freshness_intent(freshness: DashboardFreshness) -> cast::Intent {
    match freshness {
        DashboardFreshness::Fresh => cast::Intent::Success,
        DashboardFreshness::Quiet => cast::Intent::Warning,
        DashboardFreshness::Stale => cast::Intent::Danger,
    }
}

pub(super) fn preflight_outcome_label(outcome: ConnectionPreflightOutcome) -> &'static str {
    match outcome {
        ConnectionPreflightOutcome::Ready => "ready",
        ConnectionPreflightOutcome::Degraded => "degraded",
        ConnectionPreflightOutcome::Invalid => "invalid",
        ConnectionPreflightOutcome::ConnectFailed => "connect-failed",
    }
}

pub(super) fn profile_kind_label(kind: ConnectionProfileKind) -> &'static str {
    match kind {
        ConnectionProfileKind::LocalConfig => "local-config",
        ConnectionProfileKind::LocalEndpoint => "local-endpoint",
        ConnectionProfileKind::Remote => "remote",
    }
}

pub(super) fn profile_kind_index(kind: ConnectionProfileKind) -> usize {
    match kind {
        ConnectionProfileKind::LocalConfig => 0,
        ConnectionProfileKind::LocalEndpoint => 1,
        ConnectionProfileKind::Remote => 2,
    }
}

pub(super) fn profile_kind_from_index(index: usize) -> ConnectionProfileKind {
    match index {
        1 => ConnectionProfileKind::LocalEndpoint,
        2 => ConnectionProfileKind::Remote,
        _ => ConnectionProfileKind::LocalConfig,
    }
}

pub(super) fn profile_target_label(kind: ConnectionProfileKind) -> &'static str {
    match kind {
        ConnectionProfileKind::LocalConfig => "Config Path",
        ConnectionProfileKind::LocalEndpoint => "Endpoint Path",
        ConnectionProfileKind::Remote => "Remote URL",
    }
}

pub(super) fn profile_target_hint(kind: ConnectionProfileKind) -> &'static str {
    match kind {
        ConnectionProfileKind::LocalConfig => DEFAULT_BOOTSTRAP_CONFIG_PATH,
        ConnectionProfileKind::LocalEndpoint => DEFAULT_BOOTSTRAP_DAEMON_ENDPOINT_PATH,
        ConnectionProfileKind::Remote => "http://127.0.0.1:9324",
    }
}

pub(super) fn profile_auth_label(auth: Option<&ConnectionProfileAuth>) -> String {
    match auth {
        Some(ConnectionProfileAuth::TokenEnv(name)) => format!("env:{name}"),
        Some(ConnectionProfileAuth::InlineToken) => "inline token".to_string(),
        None => "none".to_string(),
    }
}

pub(super) fn profile_draft_auth_label(mode: ConnectionProfileDraftAuthMode) -> &'static str {
    match mode {
        ConnectionProfileDraftAuthMode::None => "none",
        ConnectionProfileDraftAuthMode::TokenEnv => "token-env",
        ConnectionProfileDraftAuthMode::InlineToken => "inline-token",
    }
}

pub(super) fn profile_auth_mode_index(mode: ConnectionProfileDraftAuthMode) -> usize {
    match mode {
        ConnectionProfileDraftAuthMode::TokenEnv => 0,
        ConnectionProfileDraftAuthMode::InlineToken => 1,
        ConnectionProfileDraftAuthMode::None => 2,
    }
}

pub(super) fn profile_auth_mode_from_index(index: usize) -> ConnectionProfileDraftAuthMode {
    match index {
        0 => ConnectionProfileDraftAuthMode::TokenEnv,
        1 => ConnectionProfileDraftAuthMode::InlineToken,
        _ => ConnectionProfileDraftAuthMode::None,
    }
}

pub(super) fn profile_auth_value_label(mode: ConnectionProfileDraftAuthMode) -> &'static str {
    match mode {
        ConnectionProfileDraftAuthMode::None => "Auth Value",
        ConnectionProfileDraftAuthMode::TokenEnv => "Token Env Var",
        ConnectionProfileDraftAuthMode::InlineToken => "Inline Token",
    }
}

pub(super) fn profile_auth_value_hint(mode: ConnectionProfileDraftAuthMode) -> &'static str {
    match mode {
        ConnectionProfileDraftAuthMode::None => "remote auth disabled",
        ConnectionProfileDraftAuthMode::TokenEnv => "TURIN_REMOTE_TOKEN",
        ConnectionProfileDraftAuthMode::InlineToken => "paste bearer token",
    }
}

pub(super) fn notice_level_label(level: DashboardNoticeLevel) -> &'static str {
    match level {
        DashboardNoticeLevel::Error => "[error]",
        DashboardNoticeLevel::Info => "[info]",
    }
}

pub(super) fn notice_level_color(level: DashboardNoticeLevel) -> Color32 {
    match level {
        DashboardNoticeLevel::Error => Color32::from_rgb(255, 171, 145),
        DashboardNoticeLevel::Info => Color32::from_rgb(151, 214, 255),
    }
}

pub(super) fn render_session_detail_panel(ui: &mut egui::Ui, detail: Option<&SessionDetail>) {
    cast::Panel::new().show(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.label(RichText::new("Session Detail").strong());
            if let Some(detail) = detail {
                ui.add(cast::Badge::new(format!(
                    "{} messages",
                    detail.messages.len()
                )));
                ui.add(cast::Badge::new(format!("{} events", detail.events.len())));
                ui.add(cast::Badge::new(format!(
                    "{} tool calls",
                    detail.tool_executions.len()
                )));
            }
        });
        ui.add_space(8.0);

        let Some(detail) = detail else {
            ui.label("Loading detailed transcript and tool history...");
            return;
        };

        ScrollArea::vertical().max_height(360.0).show(ui, |ui| {
            if !detail.messages.is_empty() {
                ui.label(RichText::new("Transcript").strong());
                ui.add_space(4.0);
                cast::MessageThread::new().compact(true).show(ui, |thread| {
                    for message in detail.messages.iter().rev().take(8).rev() {
                        thread.message(
                            cast::ChatMessage::new(
                                chat_role_from_label(&message.role),
                                json_preview(&message.content, 360),
                            )
                            .title(format!("{} · turn {}", message.role, message.turn_index)),
                        );
                    }
                });
            }

            if !detail.events.is_empty() {
                ui.add_space(10.0);
                ui.label(RichText::new("Recent Events").strong());
                ui.add_space(4.0);
                for event in detail.events.iter().rev().take(4).rev() {
                    ui.add(
                        cast::CodeOutputPanel::new(
                            event.event_type.clone(),
                            json_preview(&event.payload, 220),
                        )
                        .kind(cast::ToolOutputKind::Json)
                        .height(120.0),
                    );
                    ui.add_space(6.0);
                }
            }

            if !detail.tool_executions.is_empty() {
                ui.add_space(10.0);
                ui.label(RichText::new("Recent Tool Calls").strong());
                ui.add_space(4.0);
                for tool in detail.tool_executions.iter().rev().take(4).rev() {
                    ui.add(
                        cast::ToolCall::new(tool.tool_name.clone())
                            .status(tool_status_from_verdict(&tool.verdict))
                            .metadata(tool.verdict.clone())
                            .body(json_preview(&tool.args, 260)),
                    );
                    if let Some(output) = &tool.output {
                        ui.add(
                            cast::CodeOutputPanel::new("Output", json_preview(output, 260))
                                .kind(cast::ToolOutputKind::Json)
                                .height(120.0),
                        );
                    }
                    ui.add_space(6.0);
                }
            }
        });
    });
}

pub(super) fn branch_descriptor(branch: &SessionBranchDetail) -> String {
    match branch.head_turn_index {
        Some(turn_index) => format!("{} · head {}", branch.name, turn_index),
        None => branch.name.clone(),
    }
}

fn json_preview(value: &serde_json::Value, max_chars: usize) -> String {
    truncate_for_list(
        &serde_json::to_string_pretty(value).unwrap_or_else(|_| "{}".to_string()),
        max_chars,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn session_message_text_reads_structured_text_parts() {
        assert_eq!(
            session_message_text(&json!([
                { "type": "text", "text": "First paragraph" },
                { "type": "image", "source": "ignored" },
                { "type": "text", "text": "Second paragraph" }
            ])),
            "First paragraph\n\nSecond paragraph"
        );
        assert_eq!(session_message_text(&json!("Plain text")), "Plain text");
        assert_eq!(session_message_text(&json!({ "type": "image" })), "");
    }
}
