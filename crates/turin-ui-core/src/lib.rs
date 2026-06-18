mod controller;
mod dashboard;
mod intents;
mod ui_copy;
mod ui_data;
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
pub use intents::{DEFAULT_MAX_UI_NOTICES, UiAppRecord, UiRegistry};
pub use ui_copy::{ui_data_not_loaded_message, unsupported_ui_source_message};
pub use ui_data::{
    DEFAULT_UI_ACTIVITY_LIMIT, DEFAULT_UI_CHART_LIMIT, DEFAULT_UI_DETAIL_LIMIT,
    DEFAULT_UI_REPORT_LIMIT, collect_ui_list_requests, is_worklist_ui_source, ui_worklist_request,
};
pub use worklist_view::{
    WorklistStatusCounts, work_item_field_label, work_item_key, worklist_chart_group_field,
    worklist_group_counts, worklist_highest_priority_pending_item, worklist_status_counts,
};
