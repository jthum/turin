mod controller;
mod dashboard;
mod intents;

pub use controller::{
    ConnectionDraftHistory, ConnectionOptions, ConnectionPreflightOutcome,
    ConnectionPreflightReport, ConnectionProfileActivity, ConnectionProfileActivityBook,
    ConnectionProfileAuth, ConnectionProfileCatalog, ConnectionProfileDraft,
    ConnectionProfileDraftAuthMode, ConnectionProfileDraftDiff, ConnectionProfileDraftFieldDiff,
    ConnectionProfileDraftValidation, ConnectionProfileKind, ConnectionProfileSummary,
    DEFAULT_REFRESH_INTERVAL, MAX_RECENT_CONNECTION_DRAFTS, OperatorCommand, UiController,
    UiListRequest, UiUpdate, connect_dashboard, ensure_local_daemon_for_draft,
    ensure_local_daemon_for_options, execute_operator_command, preflight_connection_blocking,
    preflight_draft_blocking, spawn_controller, spawn_controller_with_interval,
};
pub use dashboard::{
    DashboardFreshness, DashboardHealth, DashboardNotice, DashboardNoticeLevel, DashboardSnapshot,
    DashboardState, format_relative_age,
};
pub use intents::{DEFAULT_MAX_UI_NOTICES, UiAppRecord, UiRegistry};
