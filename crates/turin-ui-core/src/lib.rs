mod controller;
mod dashboard;
mod form_values;
mod intents;
mod ui_actions;
mod ui_badges;
mod ui_copy;
mod ui_data;
mod ui_navigation;
mod worklist_view;

pub use controller::{
    ConnectionDraftHistory, ConnectionOptions, ConnectionPreflightOutcome,
    ConnectionPreflightReport, ConnectionProfileActivity, ConnectionProfileActivityBook,
    ConnectionProfileAuth, ConnectionProfileCatalog, ConnectionProfileDraft,
    ConnectionProfileDraftAuthMode, ConnectionProfileDraftDiff, ConnectionProfileDraftFieldDiff,
    ConnectionProfileDraftValidation, ConnectionProfileKind, ConnectionProfileSummary,
    DEFAULT_REFRESH_INTERVAL, HarnessActionFailure, MAX_RECENT_CONNECTION_DRAFTS, OperatorCommand,
    UiController, UiListRequest, UiUpdate, connect_dashboard, ensure_local_daemon_for_draft,
    ensure_local_daemon_for_options, execute_operator_command, preflight_connection_blocking,
    preflight_draft_blocking, spawn_controller, spawn_controller_with_interval,
};
pub use dashboard::{
    DashboardFreshness, DashboardHealth, DashboardNotice, DashboardNoticeLevel, DashboardSnapshot,
    DashboardState, DefaultOperatorConsoleSummary, format_relative_age,
};
pub use form_values::{
    parse_ui_form_value, ui_form_default_value, ui_form_field_kind, ui_form_is_bool_field,
    ui_form_is_multiline_field, ui_form_is_password_field, ui_form_value_string,
};
pub use intents::{DEFAULT_MAX_UI_NOTICES, UiAppRecord, UiRegistry};
pub use ui_actions::{ui_harness_action_failure_matches_app, ui_harness_action_result_matches_app};
pub use ui_badges::ui_badge_text;
pub use ui_copy::{
    ui_data_load_failed_message, ui_data_not_loaded_message, unsupported_ui_source_message,
};
pub use ui_data::{
    DEFAULT_UI_ACTIVITY_LIMIT, DEFAULT_UI_CHART_LIMIT, DEFAULT_UI_DETAIL_LIMIT,
    DEFAULT_UI_REPORT_LIMIT, UiWorklistSourceError, collect_ui_list_requests,
    is_named_worklist_ui_source, is_worklist_ui_source, ui_display_field_label,
    ui_list_filter_fields, ui_list_sort_fields, ui_refresh_requests_for_binding,
    ui_sort_entry_direction, ui_sort_entry_field, ui_sorted_field_label,
    ui_worklist_name_from_source, ui_worklist_request,
};
pub use ui_navigation::{
    UiShowTarget, ui_default_screen_index, ui_node_id_matches, ui_node_matches_target,
    ui_nodes_contain_target, ui_screen_index_for_target, ui_show_target_for,
};
pub use worklist_view::{
    WorklistStatusCounts, work_item_field_label, work_item_index_by_key, work_item_key,
    work_item_matches_key, worklist_chart_group_field, worklist_chart_group_label,
    worklist_group_counts, worklist_highest_priority_pending_item, worklist_status_counts,
};
