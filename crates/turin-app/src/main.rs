use anyhow::{Result, anyhow};
use clap::Parser;
use eframe::egui::{self, Color32, RichText, ScrollArea, Vec2};
use std::collections::{BTreeMap, BTreeSet};
use std::mem;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;
use turin_control_client::{
    AgentRuntime, ChannelRuntime, ChannelSummary, ConnectionKind, LiveSession, SessionBranchDetail,
    SessionDetail, SessionSummary, TaskStatus,
};
use turin_daemon_protocol::{
    EventEnvelope, HarnessActionRunResult, UiMenuItem, UiNoticeLevel, WorkItemList,
};
use turin_types::layout::DEFAULT_UI_PROFILES_PATH;
use turin_ui_core::{
    ConnectionDraftHistory, ConnectionOptions, ConnectionPreflightReport,
    ConnectionProfileActivityBook, ConnectionProfileCatalog, ConnectionProfileDraft,
    ConnectionProfileDraftAuthMode, ConnectionProfileDraftDiff, ConnectionProfileDraftValidation,
    ConnectionProfileKind, ConnectionProfileSummary, DashboardState, DefaultOperatorConsoleSummary,
    HarnessActionFailure, OperatorCommand, UiAppRecord, UiController, UiListRequest, UiShowTarget,
    UiUpdate, collect_ui_list_requests, connect_dashboard, ensure_local_daemon_for_draft,
    preflight_connection_blocking, preflight_draft_blocking, spawn_controller,
    ui_harness_action_failure_matches_app as harness_action_failure_matches_app,
    ui_harness_action_result_matches_app as harness_action_result_matches_app,
    ui_refresh_requests_for_binding, ui_show_target_for,
};

mod harness_ui;
mod presentation;

use harness_ui::HarnessUiEvent;
use presentation::*;

#[derive(Parser, Debug)]
#[command(name = "turin-app", version, about)]
struct Args {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    endpoint: Option<PathBuf>,
    #[arg(long)]
    remote_url: Option<String>,
    #[arg(long)]
    auth_token: Option<String>,
    #[arg(long)]
    auth_token_env: Option<String>,
    #[arg(long)]
    profile: Option<String>,
    #[arg(long)]
    profiles_file: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TabKind {
    Connections,
    UiApps,
    Agents,
    LiveSessions,
    Sessions,
    Tasks,
    Channels,
    Events,
}

impl TabKind {
    const ALL: [Self; 8] = [
        Self::Connections,
        Self::UiApps,
        Self::Agents,
        Self::LiveSessions,
        Self::Sessions,
        Self::Tasks,
        Self::Channels,
        Self::Events,
    ];

    fn title(self) -> &'static str {
        match self {
            Self::Connections => "Connections",
            Self::UiApps => "UI Apps",
            Self::Agents => "Agents",
            Self::LiveSessions => "Live Sessions",
            Self::Sessions => "Sessions",
            Self::Tasks => "Tasks",
            Self::Channels => "Channels",
            Self::Events => "Events",
        }
    }

    fn as_index(self) -> usize {
        Self::ALL
            .iter()
            .position(|tab| *tab == self)
            .unwrap_or_default()
    }

    fn from_index(index: usize) -> Self {
        Self::ALL.get(index).copied().unwrap_or(Self::Connections)
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let connection_options = connection_options(&args);
    let spec = connection_options.to_spec()?;
    let runtime = Arc::new(Runtime::new()?);
    let (client, dashboard) = runtime.block_on(connect_dashboard(&spec))?;
    let controller = spawn_controller(runtime.handle(), client);
    let profile_catalog = connection_options.load_profiles()?;
    let active_profile = connection_options.resolved_profile_name()?;

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(Vec2::new(1260.0, 820.0))
            .with_min_inner_size(Vec2::new(1040.0, 680.0)),
        ..Default::default()
    };

    eframe::run_native(
        "Turin App",
        native_options,
        Box::new(move |cc| {
            let theme_seed = configure_cast_theme(&cc.egui_ctx);
            Ok(Box::new(TurinDesktopApp::new(
                dashboard,
                controller,
                runtime,
                connection_options,
                profile_catalog,
                active_profile,
                theme_seed,
            )))
        }),
    )
    .map_err(|err| anyhow!(err.to_string()))
}

fn connection_options(args: &Args) -> ConnectionOptions {
    ConnectionOptions {
        config_path: args.config.clone(),
        endpoint: args.endpoint.clone(),
        remote_url: args.remote_url.clone(),
        auth_token: args.auth_token.clone(),
        auth_token_env: args.auth_token_env.clone(),
        profile: args.profile.clone(),
        profiles_file: args.profiles_file.clone(),
        suppress_profile_resolution: false,
    }
}

fn visible_ui_list_requests(
    app: &UiAppRecord,
    screen_index: usize,
    active_pane_id: Option<&str>,
) -> Vec<UiListRequest> {
    let mut requests = Vec::new();
    if !app.screens.is_empty() {
        let screen_index = screen_index.min(app.screens.len() - 1);
        if let Some(screen) = app.screens.values().nth(screen_index) {
            requests.extend(collect_ui_list_requests(&screen.nodes));
        }
    }
    if let Some(pane_id) = active_pane_id
        && let Some(pane) = app.panes.get(pane_id)
    {
        requests.extend(collect_ui_list_requests(&pane.nodes));
    }
    requests
}

fn configure_cast_theme(ctx: &egui::Context) -> cast::ThemeSeed {
    cast::install_cast_fonts(ctx);
    let seed = app_theme_seed(system_theme_mode(ctx));
    cast::set_theme(ctx, seed.clone().resolve());
    seed
}

fn system_theme_mode(ctx: &egui::Context) -> cast::ThemeMode {
    match ctx.system_theme() {
        Some(egui::Theme::Dark) => cast::ThemeMode::Dark,
        _ => cast::ThemeMode::Light,
    }
}

fn app_theme_seed(mode: cast::ThemeMode) -> cast::ThemeSeed {
    cast::ThemeSeed::for_mode(mode)
        .with_primary(Color32::from_rgb(22, 126, 118))
        .with_density(30.0, 8.0)
}

struct TurinDesktopApp {
    dashboard: DashboardState,
    controller: UiController,
    connection_options: ConnectionOptions,
    profile_catalog: Option<ConnectionProfileCatalog>,
    active_profile: Option<String>,
    tab: TabKind,
    profile_index: usize,
    recent_draft_index: usize,
    ui_app_index: usize,
    agent_index: usize,
    live_session_index: usize,
    session_index: usize,
    task_index: usize,
    channel_index: usize,
    event_index: usize,
    profile_name_input: String,
    profile_draft: ConnectionProfileDraft,
    draft_baseline: ConnectionProfileDraft,
    draft_baseline_label: String,
    recent_drafts: ConnectionDraftHistory,
    profile_activity: ConnectionProfileActivityBook,
    pending_discard_action: Option<PendingDraftAction>,
    pending_harness_ui_action: Option<PendingHarnessUiAction>,
    last_preflight_report: Option<ConnectionPreflightReport>,
    save_profile_as_default: bool,
    pending_delete_profile: Option<String>,
    prompt_input: String,
    branch_name_input: String,
    activate_new_branch: bool,
    requested_session_detail: Option<String>,
    task_filter: String,
    channel_filter: String,
    event_filter: String,
    events_paused: bool,
    events_follow_latest: bool,
    paused_events: Vec<EventEnvelope>,
    ui_screen_indices: BTreeMap<String, usize>,
    ui_active_pane: Option<String>,
    ui_form_values: BTreeMap<String, String>,
    ui_selected_list_items: BTreeMap<String, String>,
    ui_list_requests: BTreeMap<String, UiListRequest>,
    ui_lists: BTreeMap<String, WorkItemList>,
    requested_ui_lists: BTreeSet<String>,
    ui_list_errors: BTreeMap<String, String>,
    latest_harness_action_result: Option<HarnessActionRunResult>,
    latest_harness_action_failure: Option<HarnessActionFailure>,
    latest_harness_action_label: Option<(String, String, String)>,
    harness_feedback_revision: u64,
    dismissed_operator_feedback: BTreeSet<String>,
    theme_seed: cast::ThemeSeed,
    follows_system_theme: bool,
    sidebar_settings_open: bool,
    runtime_tools_open: bool,
    _runtime: Arc<Runtime>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PendingDraftAction {
    CurrentConnection,
    SelectedProfile(String),
    LatestRecentDraft,
    SelectedRecentDraft,
    BlankDraft,
}

#[derive(Debug, Clone)]
struct PendingHarnessUiAction {
    app_id: String,
    label: String,
    action: String,
    agent_id: Option<String>,
    harness_id: Option<String>,
    params: serde_json::Value,
}

const DEFAULT_OPERATOR_EMPTY_TITLE: &str = "Turin is ready";
const DEFAULT_OPERATOR_EMPTY_BODY: &str = "No custom app surface is loaded. Use the standard console for agents, sessions, tasks, and events.";
const DEFAULT_OPERATOR_INTRO: &str =
    "The standard console keeps the runtime usable without requiring a harness-defined app.";
const DEFAULT_OPERATOR_GUIDANCE_TITLE: &str = "Custom Apps";
const DEFAULT_OPERATOR_GUIDANCE_BODY: &str = "A harness can add focused screens when a workflow needs dedicated lists, forms, reports, panes, badges, or action buttons.";

#[derive(Debug, Clone, PartialEq, Eq)]
struct DefaultOperatorMetricGroup {
    title: &'static str,
    metrics: Vec<(&'static str, usize)>,
}

impl PendingHarnessUiAction {
    fn new(app: &UiAppRecord, label: String, action: String, params: serde_json::Value) -> Self {
        Self {
            app_id: app.id.clone(),
            label,
            action,
            agent_id: app.source.agent_id.clone(),
            harness_id: app.source.harness_id.clone(),
            params,
        }
    }
}

impl PendingDraftAction {
    fn description(&self) -> String {
        match self {
            Self::CurrentConnection => {
                "load the current live connection into the editor".to_string()
            }
            Self::SelectedProfile(name) => {
                format!("load the saved profile '{name}' into the editor")
            }
            Self::LatestRecentDraft => "load the latest recent draft into the editor".to_string(),
            Self::SelectedRecentDraft => {
                "load the selected recent draft into the editor".to_string()
            }
            Self::BlankDraft => "reset the profile editor to a blank draft".to_string(),
        }
    }
}

impl TurinDesktopApp {
    fn new(
        dashboard: DashboardState,
        controller: UiController,
        runtime: Arc<Runtime>,
        connection_options: ConnectionOptions,
        profile_catalog: Option<ConnectionProfileCatalog>,
        active_profile: Option<String>,
        theme_seed: cast::ThemeSeed,
    ) -> Self {
        let profile_draft = connection_options
            .current_profile_draft()
            .unwrap_or_else(|_| ConnectionProfileDraft::default());
        let tab = if dashboard.ui.apps().next().is_some() {
            TabKind::UiApps
        } else {
            TabKind::Connections
        };
        Self {
            dashboard,
            controller,
            connection_options,
            profile_catalog,
            active_profile,
            tab,
            profile_index: 0,
            recent_draft_index: 0,
            ui_app_index: 0,
            agent_index: 0,
            live_session_index: 0,
            session_index: 0,
            task_index: 0,
            channel_index: 0,
            event_index: 0,
            profile_name_input: String::new(),
            draft_baseline: profile_draft.clone(),
            draft_baseline_label: "current connection".to_string(),
            profile_draft,
            recent_drafts: ConnectionDraftHistory::default(),
            profile_activity: ConnectionProfileActivityBook::default(),
            pending_discard_action: None,
            pending_harness_ui_action: None,
            last_preflight_report: None,
            save_profile_as_default: false,
            pending_delete_profile: None,
            prompt_input: String::new(),
            branch_name_input: String::new(),
            activate_new_branch: true,
            requested_session_detail: None,
            task_filter: String::new(),
            channel_filter: String::new(),
            event_filter: String::new(),
            events_paused: false,
            events_follow_latest: true,
            paused_events: Vec::new(),
            ui_screen_indices: BTreeMap::new(),
            ui_active_pane: None,
            ui_form_values: BTreeMap::new(),
            ui_selected_list_items: BTreeMap::new(),
            ui_list_requests: BTreeMap::new(),
            ui_lists: BTreeMap::new(),
            requested_ui_lists: BTreeSet::new(),
            ui_list_errors: BTreeMap::new(),
            latest_harness_action_result: None,
            latest_harness_action_failure: None,
            latest_harness_action_label: None,
            harness_feedback_revision: 0,
            dismissed_operator_feedback: BTreeSet::new(),
            theme_seed,
            follows_system_theme: true,
            sidebar_settings_open: false,
            runtime_tools_open: false,
            _runtime: runtime,
        }
    }

    fn apply_update(&mut self, update: UiUpdate) {
        let auto_follow_event = matches!(&update, UiUpdate::Event(_))
            && !self.events_paused
            && self.events_follow_latest;
        let harness_action_ran =
            matches!(&update, UiUpdate::Event(event) if event.event == "harness.action_ran");
        if matches!(&update, UiUpdate::SessionEvent(_)) {
            self.clamp_selection_indices();
            return;
        }
        if let UiUpdate::UiListLoaded { request, items } = &update {
            let key = request.cache_key();
            self.ui_list_requests
                .insert(key.clone(), request.as_ref().clone());
            self.requested_ui_lists.remove(&key);
            self.ui_list_errors.remove(&key);
            self.ui_lists.insert(key, items.as_ref().clone());
        }
        if let UiUpdate::UiListFailed { request, message } = &update {
            let key = request.cache_key();
            self.ui_list_requests
                .insert(key.clone(), request.as_ref().clone());
            self.requested_ui_lists.remove(&key);
            self.ui_lists.remove(&key);
            self.ui_list_errors.insert(key, message.clone());
        }
        if let UiUpdate::HarnessActionCompleted(result) = &update {
            self.latest_harness_action_result = Some(result.as_ref().clone());
            self.latest_harness_action_failure = None;
            self.harness_feedback_revision = self.harness_feedback_revision.wrapping_add(1);
        }
        if let UiUpdate::HarnessActionFailed(failure) = &update {
            self.latest_harness_action_failure = Some(failure.as_ref().clone());
            self.latest_harness_action_result = None;
            self.harness_feedback_revision = self.harness_feedback_revision.wrapping_add(1);
        }
        self.dashboard.apply_update(update);
        self.apply_ui_navigation_intents();
        let refreshed = self.apply_ui_refresh_intents();
        if harness_action_ran && refreshed == 0 {
            self.request_selected_ui_lists(true);
        }
        if auto_follow_event {
            self.event_index = 0;
        }
        self.clamp_selection_indices();
    }

    fn clamp_selection_indices(&mut self) {
        self.profile_index = clamp_index(
            self.profile_index,
            self.profile_catalog
                .as_ref()
                .map(|catalog| catalog.profiles().len())
                .unwrap_or(0),
        );
        self.recent_draft_index =
            clamp_index(self.recent_draft_index, self.recent_drafts.drafts().len());
        self.ui_app_index = clamp_index(self.ui_app_index, self.dashboard.ui.apps().count());
        self.agent_index = clamp_index(self.agent_index, self.dashboard.agents().len());
        self.live_session_index =
            clamp_index(self.live_session_index, self.dashboard.live_sessions.len());
        self.session_index = clamp_index(self.session_index, self.dashboard.sessions.len());
        self.task_index = clamp_index(self.task_index, self.filtered_tasks().len());
        self.channel_index = clamp_index(self.channel_index, self.filtered_channels().len());
        self.event_index = clamp_index(self.event_index, self.filtered_events().len());
    }

    fn send_command(&mut self, command: OperatorCommand) {
        if let Err(err) = self.controller.command_tx.send(command) {
            self.dashboard
                .record_error(format!("Failed to dispatch operator command: {err}"));
        }
    }

    fn selected_profile(&self) -> Option<&ConnectionProfileSummary> {
        self.profile_catalog
            .as_ref()?
            .profiles()
            .get(self.profile_index)
    }

    fn select_profile_by_name(&mut self, name: &str) {
        if let Some(catalog) = &self.profile_catalog
            && let Some(index) = catalog
                .profiles()
                .iter()
                .position(|profile| profile.name == name)
        {
            self.profile_index = index;
        }
        self.pending_delete_profile = None;
    }

    fn selected_recent_draft(&self) -> Option<&ConnectionProfileDraft> {
        self.recent_drafts.drafts().get(self.recent_draft_index)
    }

    fn selected_ui_app(&self) -> Option<UiAppRecord> {
        self.dashboard.ui.apps().nth(self.ui_app_index).cloned()
    }

    fn selected_ui_list_requests(&self) -> Vec<UiListRequest> {
        let Some(app) = self.selected_ui_app() else {
            return Vec::new();
        };
        let screen_index = self
            .ui_screen_indices
            .get(&app.id)
            .copied()
            .unwrap_or_else(|| harness_ui::default_screen_index(&app));
        visible_ui_list_requests(&app, screen_index, self.ui_active_pane.as_deref())
    }

    fn request_selected_ui_lists(&mut self, force: bool) {
        for request in self.selected_ui_list_requests() {
            self.request_ui_list(request, force);
        }
    }

    fn request_ui_list(&mut self, request: UiListRequest, force: bool) {
        let key = request.cache_key();
        self.ui_list_requests.insert(key.clone(), request.clone());
        if force {
            self.ui_lists.remove(&key);
            self.requested_ui_lists.remove(&key);
            self.ui_list_errors.remove(&key);
        } else if self.ui_lists.contains_key(&key)
            || self.requested_ui_lists.contains(&key)
            || self.ui_list_errors.contains_key(&key)
        {
            return;
        }
        self.requested_ui_lists.insert(key);
        self.send_command(OperatorCommand::LoadUiList {
            request: Box::new(request),
        });
    }

    fn apply_ui_refresh_intents(&mut self) -> usize {
        let refreshes = self.dashboard.ui.take_refreshes();
        let mut reloads = 0;
        for refresh in refreshes {
            reloads += self.refresh_ui_binding(&refresh.binding);
        }
        reloads
    }

    fn refresh_ui_binding(&mut self, binding: &str) -> usize {
        let requests = ui_refresh_requests_for_binding(
            binding,
            &self.ui_list_requests,
            self.selected_ui_list_requests(),
        );

        for request in &requests {
            let key = request.cache_key();
            self.ui_lists.remove(&key);
            self.requested_ui_lists.remove(&key);
            self.ui_list_errors.remove(&key);
        }

        let count = requests.len();
        for request in requests {
            self.request_ui_list(request, true);
        }
        count
    }

    fn apply_ui_navigation_intents(&mut self) {
        for open in self.dashboard.ui.take_opens() {
            self.apply_ui_open_request(&open.app_id, &open.target);
        }
        for show in self.dashboard.ui.take_shows() {
            self.apply_ui_show_request(&show.app_id, &show.target);
        }
        for focus in self.dashboard.ui.take_focuses() {
            self.apply_ui_focus_request(&focus.app_id, &focus.target);
        }
    }

    fn apply_ui_open_request(&mut self, app_id: &str, target: &str) {
        let Some(app) = self.select_ui_app_by_id(app_id) else {
            return;
        };
        let Some(screen_index) = harness_ui::screen_index_for_target(&app, target) else {
            self.dashboard.record_error(format!(
                "The requested screen could not be opened in {}.",
                ui_app_title(&app)
            ));
            return;
        };
        self.open_harness_screen(&app, screen_index);
    }

    fn apply_ui_show_request(&mut self, app_id: &str, target: &str) {
        let Some(app) = self.select_ui_app_by_id(app_id) else {
            return;
        };
        match ui_show_target_for(&app, target) {
            Some(UiShowTarget::Screen { screen_index }) => {
                self.open_harness_screen(&app, screen_index);
            }
            Some(UiShowTarget::Pane { pane_id }) => {
                self.tab = TabKind::UiApps;
                self.ui_active_pane = Some(pane_id.to_string());
                self.request_selected_ui_lists(false);
            }
            None => {
                self.dashboard.record_error(format!(
                    "The requested view could not be shown in {}.",
                    ui_app_title(&app)
                ));
            }
        }
    }

    fn apply_ui_focus_request(&mut self, app_id: &str, target: &str) {
        let Some(app) = self.select_ui_app_by_id(app_id) else {
            return;
        };
        let Some(target) = harness_ui::find_focus_target(&app, target) else {
            self.dashboard.record_error(format!(
                "The requested focus target was not found in {}.",
                ui_app_title(&app)
            ));
            return;
        };
        match target {
            harness_ui::HarnessFocusTarget::Screen { screen_index }
            | harness_ui::HarnessFocusTarget::Node { screen_index } => {
                self.open_harness_screen(&app, screen_index);
            }
        }
    }

    fn select_ui_app_by_id(&mut self, app_id: &str) -> Option<UiAppRecord> {
        let Some((index, app)) = self
            .dashboard
            .ui
            .apps()
            .cloned()
            .enumerate()
            .find(|(_, app)| app.id == app_id)
        else {
            self.dashboard
                .record_error(format!("UI app '{app_id}' is not declared"));
            return None;
        };
        self.ui_app_index = index;
        Some(app)
    }

    fn open_harness_screen(&mut self, app: &UiAppRecord, screen_index: usize) {
        self.tab = TabKind::UiApps;
        self.ui_screen_indices.insert(app.id.clone(), screen_index);
        self.ui_active_pane = None;
        self.request_selected_ui_lists(false);
        self.clamp_selection_indices();
    }

    fn set_profile_draft(
        &mut self,
        draft: ConnectionProfileDraft,
        baseline_label: impl Into<String>,
    ) {
        self.profile_draft = draft.clone();
        self.draft_baseline = draft;
        self.draft_baseline_label = baseline_label.into();
        self.pending_discard_action = None;
    }

    fn editor_diff(&self) -> ConnectionProfileDraftDiff {
        self.profile_draft.diff_against(&self.draft_baseline)
    }

    fn editor_is_dirty(&self) -> bool {
        !self.editor_diff().is_empty()
    }

    fn selected_profile_diff(&self) -> Option<ConnectionProfileDraftDiff> {
        let selected = self.selected_profile()?;
        self.connection_options
            .draft_diff_against_profile(&self.profile_draft, &selected.name)
            .ok()
    }

    fn queue_or_apply_draft_action(&mut self, action: PendingDraftAction) {
        if self.editor_is_dirty() {
            self.pending_discard_action = Some(action.clone());
            self.dashboard.record_info(format!(
                "The profile editor has unsaved changes ({}). Use Discard Pending Action to {} or Cancel Pending Action to keep editing.",
                self.editor_diff().summary(),
                action.description()
            ));
            return;
        }

        self.apply_draft_action(action);
    }

    fn confirm_pending_discard_action(&mut self) {
        let Some(action) = self.pending_discard_action.take() else {
            self.dashboard
                .record_error("No pending editor action is waiting for discard confirmation");
            return;
        };
        self.apply_draft_action(action);
    }

    fn cancel_pending_discard_action(&mut self) {
        self.pending_discard_action = None;
        self.dashboard
            .record_info("Kept the current editor draft and cancelled the pending discard");
    }

    fn request_harness_ui_action_confirmation(&mut self, action: PendingHarnessUiAction) {
        self.dashboard.record_info(format!(
            "Harness UI action '{}' requires confirmation before running",
            action.label
        ));
        self.pending_harness_ui_action = Some(action);
    }

    fn confirm_pending_harness_ui_action(&mut self) {
        let Some(action) = self.pending_harness_ui_action.take() else {
            self.dashboard
                .record_error("No harness UI action is waiting for confirmation");
            return;
        };
        self.run_harness_ui_action(action);
    }

    fn cancel_pending_harness_ui_action(&mut self) {
        let Some(action) = self.pending_harness_ui_action.take() else {
            self.dashboard
                .record_error("No harness UI action is waiting for cancellation");
            return;
        };
        self.dashboard.record_info(format!(
            "Cancelled harness UI action '{}' ({})",
            action.label, action.action
        ));
    }

    fn run_harness_ui_action(&mut self, action: PendingHarnessUiAction) {
        self.dashboard.record_info(format!(
            "Running harness UI action '{}' ({})",
            action.label, action.action
        ));
        self.latest_harness_action_label = Some((
            action.app_id.clone(),
            action.action.clone(),
            action.label.clone(),
        ));
        self.send_command(OperatorCommand::RunHarnessAction {
            agent_id: action.agent_id,
            harness_id: action.harness_id,
            action: action.action,
            params: action.params,
        });
    }

    fn apply_draft_action(&mut self, action: PendingDraftAction) {
        match action {
            PendingDraftAction::CurrentConnection => {
                match self.connection_options.current_profile_draft() {
                    Ok(draft) => {
                        self.set_profile_draft(draft, "current connection");
                        self.dashboard
                            .record_info("Loaded current connection into the profile editor");
                    }
                    Err(err) => self.dashboard.record_error(format!(
                        "Failed to load current connection into editor: {err}"
                    )),
                }
            }
            PendingDraftAction::SelectedProfile(profile_name) => {
                match self.connection_options.load_profile_draft(&profile_name) {
                    Ok(draft) => {
                        self.set_profile_draft(draft, format!("saved profile '{profile_name}'"));
                        self.dashboard.record_info(format!(
                            "Loaded connection profile '{}' into the editor",
                            profile_name
                        ));
                    }
                    Err(err) => self.dashboard.record_error(format!(
                        "Failed to load connection profile into editor: {err}"
                    )),
                }
            }
            PendingDraftAction::LatestRecentDraft => {
                let Some(draft) = self.recent_drafts.latest().cloned() else {
                    self.dashboard
                        .record_error("No recent draft connections have been recorded yet");
                    return;
                };
                self.recent_draft_index = 0;
                self.set_profile_draft(draft, "latest recent draft");
                self.dashboard
                    .record_info("Loaded the latest successful draft connection into the editor");
            }
            PendingDraftAction::SelectedRecentDraft => {
                let Some(draft) = self.selected_recent_draft().cloned() else {
                    self.dashboard
                        .record_error("No recent draft connection is currently selected");
                    return;
                };
                self.set_profile_draft(draft, "selected recent draft");
                self.dashboard
                    .record_info("Loaded the selected recent draft into the editor");
            }
            PendingDraftAction::BlankDraft => {
                self.set_profile_draft(ConnectionProfileDraft::default(), "blank draft");
                self.profile_name_input.clear();
                self.save_profile_as_default = false;
                self.dashboard
                    .record_info("Reset the connection profile editor");
            }
        }
    }

    fn load_selected_recent_draft(&mut self) {
        self.profile_name_input.clear();
        self.pending_delete_profile = None;
        self.queue_or_apply_draft_action(PendingDraftAction::SelectedRecentDraft);
    }

    fn load_latest_recent_draft(&mut self) {
        self.profile_name_input.clear();
        self.pending_delete_profile = None;
        self.queue_or_apply_draft_action(PendingDraftAction::LatestRecentDraft);
    }

    fn typed_profile_name(&self) -> Option<String> {
        let trimmed = self.profile_name_input.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    }

    fn profile_draft_validation(&self) -> ConnectionProfileDraftValidation {
        self.profile_draft.validate()
    }

    fn reload_profiles(&mut self) {
        match self.connection_options.load_profiles() {
            Ok(catalog) => {
                self.profile_catalog = catalog;
                self.clamp_selection_indices();
                self.pending_delete_profile = None;
                self.dashboard
                    .record_info("Reloaded UI connection profiles");
            }
            Err(err) => self
                .dashboard
                .record_error(format!("Failed to load UI profiles: {err}")),
        }
    }

    fn selected_profile_options(&self) -> Option<ConnectionOptions> {
        let selected = self.selected_profile()?;
        self.profile_catalog
            .as_ref()?
            .connection_options(&selected.name)
    }

    fn load_current_connection_into_editor(&mut self) {
        self.pending_delete_profile = None;
        self.queue_or_apply_draft_action(PendingDraftAction::CurrentConnection);
    }

    fn load_selected_profile_into_editor(&mut self) {
        let Some(profile_name) = self.selected_profile().map(|profile| profile.name.clone()) else {
            self.dashboard
                .record_error("No connection profile is currently selected");
            return;
        };
        self.pending_delete_profile = None;
        self.queue_or_apply_draft_action(PendingDraftAction::SelectedProfile(profile_name));
    }

    fn reset_profile_editor(&mut self) {
        self.profile_name_input.clear();
        self.pending_delete_profile = None;
        self.queue_or_apply_draft_action(PendingDraftAction::BlankDraft);
    }

    fn preflight_selected_profile(&mut self) {
        let Some(selected) = self.selected_profile().cloned() else {
            self.dashboard
                .record_error("No connection profile is currently selected");
            return;
        };
        let Some(options) = self.selected_profile_options() else {
            self.dashboard
                .record_error("Failed to resolve the selected connection profile");
            return;
        };

        let report = preflight_connection_blocking(&options);
        self.profile_activity.record_preflight_result(
            selected.name.clone(),
            report.is_success(),
            report.summary_label(),
        );
        if report.is_success() {
            self.dashboard
                .record_info(format!("Preflight for '{}' succeeded", selected.name));
        } else {
            self.dashboard.record_error(format!(
                "Preflight for '{}' failed: {}",
                selected.name, report.message
            ));
        }
        self.last_preflight_report = Some(report);
    }

    fn preflight_draft(&mut self) {
        let report = preflight_draft_blocking(&self.connection_options, &self.profile_draft);
        if report.is_success() {
            self.dashboard.record_info("Draft preflight succeeded");
        } else {
            self.dashboard
                .record_error(format!("Draft preflight failed: {}", report.message));
        }
        self.last_preflight_report = Some(report);
    }

    fn ensure_local_daemon_for_draft(&mut self) {
        match ensure_local_daemon_for_draft(&self.connection_options, &self.profile_draft) {
            Ok(message) => self.dashboard.record_info(message),
            Err(err) => self
                .dashboard
                .record_error(format!("Failed to ensure local daemon: {err}")),
        }
    }

    fn reconnect_current(&mut self) {
        self.switch_connection(self.connection_options.clone(), None);
    }

    fn connect_selected_profile(&mut self) {
        if let Some(options) = self.selected_profile_options() {
            self.switch_connection(options, None);
        } else {
            self.dashboard
                .record_error("No connection profile is currently selected");
        }
    }

    fn connect_profile_draft(&mut self) {
        let validation = self.profile_draft_validation();
        if !validation.is_valid() {
            self.dashboard.record_error(format!(
                "Cannot connect invalid connection profile draft: {}",
                validation.summary()
            ));
            return;
        }

        match self
            .connection_options
            .connection_options_for_draft(&self.profile_draft)
        {
            Ok(options) => self.switch_connection(options, Some(self.profile_draft.clone())),
            Err(err) => self
                .dashboard
                .record_error(format!("Failed to build connection from draft: {err}")),
        }
    }

    fn update_selected_profile(&mut self) {
        let Some(profile_name) = self.selected_profile().map(|profile| profile.name.clone()) else {
            self.dashboard
                .record_error("No connection profile is currently selected");
            return;
        };
        let validation = self.profile_draft_validation();
        if !validation.is_valid() {
            self.dashboard.record_error(format!(
                "Cannot update selected connection profile: {}",
                validation.summary()
            ));
            return;
        }
        if self
            .selected_profile_diff()
            .is_some_and(|diff| diff.is_empty())
        {
            self.dashboard.record_info(format!(
                "The editor draft already matches saved profile '{}'",
                profile_name
            ));
            return;
        }

        match self.connection_options.save_profile_draft(
            &profile_name,
            &self.profile_draft,
            self.save_profile_as_default,
        ) {
            Ok(catalog) => {
                let active_profile = self.active_profile.as_deref() == Some(profile_name.as_str());
                self.profile_catalog = Some(catalog);
                self.select_profile_by_name(&profile_name);
                self.set_profile_draft(
                    self.profile_draft.clone(),
                    format!("saved profile '{}'", profile_name),
                );
                self.clamp_selection_indices();
                self.dashboard.record_info(if active_profile {
                    format!(
                        "Updated connection profile '{}'. Reconnect current to apply the saved changes.",
                        profile_name
                    )
                } else {
                    format!("Updated connection profile '{}'", profile_name)
                });
            }
            Err(err) => self.dashboard.record_error(format!(
                "Failed to update selected connection profile: {err}"
            )),
        }
    }

    fn save_current_profile(&mut self) {
        let Some(profile_name) = self.typed_profile_name() else {
            self.dashboard
                .record_error("Enter a profile name before saving the draft as a profile");
            return;
        };
        let validation = self.profile_draft_validation();
        if !validation.is_valid() {
            self.dashboard.record_error(format!(
                "Cannot save invalid connection profile draft: {}",
                validation.summary()
            ));
            return;
        }

        match self.connection_options.save_profile_draft(
            &profile_name,
            &self.profile_draft,
            self.save_profile_as_default,
        ) {
            Ok(catalog) => {
                self.profile_catalog = Some(catalog);
                self.select_profile_by_name(&profile_name);
                self.set_profile_draft(
                    self.profile_draft.clone(),
                    format!("saved profile '{}'", profile_name),
                );
                self.profile_name_input = String::new();
                self.clamp_selection_indices();
                self.dashboard.record_info(format!(
                    "Saved draft to profile '{}' in '{}'",
                    profile_name,
                    self.connection_options.profiles_path().display()
                ));
            }
            Err(err) => self
                .dashboard
                .record_error(format!("Failed to save connection profile: {err}")),
        }
    }

    fn duplicate_selected_profile(&mut self) {
        let Some(source_name) = self.selected_profile().map(|profile| profile.name.clone()) else {
            self.dashboard
                .record_error("No connection profile is currently selected");
            return;
        };
        let Some(target_name) = self.typed_profile_name() else {
            self.dashboard
                .record_error("Enter a new profile name before duplicating the selected profile");
            return;
        };

        match self.connection_options.duplicate_profile(
            &source_name,
            &target_name,
            self.save_profile_as_default,
        ) {
            Ok(catalog) => {
                self.profile_catalog = Some(catalog);
                self.select_profile_by_name(&target_name);
                self.profile_name_input = target_name.clone();
                self.clamp_selection_indices();
                self.dashboard.record_info(format!(
                    "Duplicated connection profile '{}' to '{}'",
                    source_name, target_name
                ));
            }
            Err(err) => self
                .dashboard
                .record_error(format!("Failed to duplicate connection profile: {err}")),
        }
    }

    fn rename_selected_profile(&mut self) {
        let Some(source_name) = self.selected_profile().map(|profile| profile.name.clone()) else {
            self.dashboard
                .record_error("No connection profile is currently selected");
            return;
        };
        let Some(target_name) = self.typed_profile_name() else {
            self.dashboard
                .record_error("Enter a new profile name before renaming the selected profile");
            return;
        };

        match self.connection_options.rename_profile(
            &source_name,
            &target_name,
            self.save_profile_as_default,
        ) {
            Ok(catalog) => {
                if self.connection_options.profile.as_deref() == Some(source_name.as_str()) {
                    self.connection_options.profile = Some(target_name.clone());
                }
                if self.active_profile.as_deref() == Some(source_name.as_str()) {
                    self.active_profile = Some(target_name.clone());
                }
                self.profile_catalog = Some(catalog);
                self.select_profile_by_name(&target_name);
                self.profile_name_input = target_name.clone();
                self.draft_baseline_label = format!("saved profile '{}'", target_name);
                self.clamp_selection_indices();
                self.dashboard.record_info(format!(
                    "Renamed connection profile '{}' to '{}'",
                    source_name, target_name
                ));
            }
            Err(err) => self
                .dashboard
                .record_error(format!("Failed to rename connection profile: {err}")),
        }
    }

    fn is_delete_armed_for_selected(&self) -> bool {
        let Some(selected_name) = self.selected_profile().map(|profile| profile.name.as_str())
        else {
            return false;
        };
        self.pending_delete_profile.as_deref() == Some(selected_name)
    }

    fn arm_delete_selected_profile(&mut self) {
        let Some(profile_name) = self.selected_profile().map(|profile| profile.name.clone()) else {
            self.dashboard
                .record_error("No connection profile is currently selected");
            return;
        };
        self.pending_delete_profile = Some(profile_name.clone());
        self.dashboard.record_info(format!(
            "Delete armed for profile '{}'. Confirm to remove it from '{}'",
            profile_name,
            self.connection_options.profiles_path().display()
        ));
    }

    fn cancel_delete_selected_profile(&mut self) {
        self.pending_delete_profile = None;
        self.dashboard
            .record_info("Cancelled connection profile delete");
    }

    fn delete_selected_profile(&mut self) {
        let Some(profile_name) = self.selected_profile().map(|profile| profile.name.clone()) else {
            self.dashboard
                .record_error("No connection profile is currently selected");
            return;
        };
        let fallback_connection =
            if self.connection_options.profile.as_deref() == Some(profile_name.as_str()) {
                self.connection_options.materialized().ok()
            } else {
                None
            };

        match self.connection_options.delete_profile(&profile_name) {
            Ok(catalog) => {
                if let Some(options) = fallback_connection {
                    self.connection_options = options;
                }
                if self.active_profile.as_deref() == Some(profile_name.as_str()) {
                    self.active_profile = None;
                }
                self.pending_delete_profile = None;
                self.profile_catalog = Some(catalog);
                self.clamp_selection_indices();
                self.profile_name_input.clear();
                self.dashboard
                    .record_info(format!("Deleted connection profile '{}'", profile_name));
            }
            Err(err) => self
                .dashboard
                .record_error(format!("Failed to delete connection profile: {err}")),
        }
    }

    fn switch_connection(
        &mut self,
        connection_options: ConnectionOptions,
        connected_draft: Option<ConnectionProfileDraft>,
    ) {
        let spec = match connection_options.to_spec() {
            Ok(spec) => spec,
            Err(err) => {
                self.dashboard
                    .record_error(format!("Failed to resolve connection: {err}"));
                return;
            }
        };

        match self._runtime.block_on(connect_dashboard(&spec)) {
            Ok((client, dashboard)) => {
                self.controller.shutdown();
                self.controller = spawn_controller(self._runtime.handle(), client);
                self.connection_options = connection_options.clone();
                self.active_profile = connection_options.resolved_profile_name().ok().flatten();
                self.profile_catalog = connection_options.load_profiles().ok().flatten();
                if let Some(draft) = connected_draft.as_ref() {
                    self.recent_drafts.record_success(draft);
                    self.recent_draft_index = 0;
                }
                if let Some(profile_name) = self.active_profile.clone() {
                    self.profile_activity.record_connect_result(
                        profile_name,
                        true,
                        format!("Connected to {}", dashboard.connection_target),
                    );
                }
                self.dashboard = dashboard;
                self.profile_name_input.clear();
                self.set_profile_draft(
                    self.connection_options
                        .current_profile_draft()
                        .unwrap_or_else(|_| ConnectionProfileDraft::default()),
                    "current connection",
                );
                self.pending_delete_profile = None;
                self.prompt_input.clear();
                self.requested_session_detail = None;
                self.last_preflight_report = None;
                self.clamp_selection_indices();
                let target = self.dashboard.connection_target.clone();
                self.dashboard
                    .record_info(format!("Connected UI client to {target}"));
            }
            Err(err) => {
                if let Ok(Some(profile_name)) = connection_options.resolved_profile_name() {
                    self.profile_activity.record_connect_result(
                        profile_name,
                        false,
                        format!("Failed to connect UI client: {err}"),
                    );
                }
                self.dashboard
                    .record_error(format!("Failed to connect UI client: {err}"));
            }
        }
    }

    fn selected_agent_runtime(&self, agent_id: &str) -> Option<AgentRuntime> {
        self.dashboard
            .status
            .as_ref()?
            .agent_runtimes
            .iter()
            .find(|runtime| runtime.agent_id == agent_id)
            .cloned()
    }

    fn selected_live_session(&self) -> Option<LiveSession> {
        self.dashboard
            .live_sessions
            .get(self.live_session_index)
            .cloned()
    }

    fn selected_session(&self) -> Option<SessionSummary> {
        self.dashboard.sessions.get(self.session_index).cloned()
    }

    fn selected_channel_runtime(&self, channel_id: &str) -> Option<ChannelRuntime> {
        self.dashboard
            .status
            .as_ref()?
            .channel_runtimes
            .iter()
            .find(|runtime| runtime.id == channel_id)
            .cloned()
    }

    fn filtered_tasks(&self) -> Vec<TaskStatus> {
        let filter = self.task_filter.trim().to_ascii_lowercase();
        self.dashboard
            .tasks
            .iter()
            .filter(|task| {
                filter.is_empty()
                    || task.request_id.to_ascii_lowercase().contains(&filter)
                    || task.agent_id.to_ascii_lowercase().contains(&filter)
                    || task.state.to_ascii_lowercase().contains(&filter)
            })
            .cloned()
            .collect()
    }

    fn filtered_channels(&self) -> Vec<ChannelSummary> {
        let filter = self.channel_filter.trim().to_ascii_lowercase();
        self.dashboard
            .channels()
            .iter()
            .filter(|channel| {
                filter.is_empty()
                    || channel.id.to_ascii_lowercase().contains(&filter)
                    || channel.kind.to_ascii_lowercase().contains(&filter)
                    || channel.agent_id.to_ascii_lowercase().contains(&filter)
            })
            .cloned()
            .collect()
    }

    fn event_source(&self) -> &[EventEnvelope] {
        if self.events_paused {
            &self.paused_events
        } else {
            &self.dashboard.recent_events
        }
    }

    fn filtered_events(&self) -> Vec<EventEnvelope> {
        let filter = self.event_filter.trim().to_ascii_lowercase();
        self.event_source()
            .iter()
            .rev()
            .filter(|event| {
                if filter.is_empty() {
                    return true;
                }
                event.event.to_ascii_lowercase().contains(&filter)
                    || serde_json::to_string(&event.data)
                        .unwrap_or_else(|_| "{}".to_string())
                        .to_ascii_lowercase()
                        .contains(&filter)
            })
            .cloned()
            .collect()
    }

    fn set_events_paused(&mut self, paused: bool) {
        if paused == self.events_paused {
            return;
        }
        self.events_paused = paused;
        if paused {
            self.paused_events = self.dashboard.recent_events.clone();
            self.dashboard
                .record_info("Paused the event list at the current snapshot");
        } else {
            self.paused_events.clear();
            self.dashboard.record_info("Resumed live event updates");
        }
        if self.events_follow_latest {
            self.event_index = 0;
        }
        self.clamp_selection_indices();
    }

    fn selected_session_detail(&self) -> Option<&SessionDetail> {
        self.current_detail_session_id()
            .as_deref()
            .and_then(|session_id| self.dashboard.session_detail(session_id))
    }

    fn active_connection_label(&self) -> String {
        if self.connection_options.suppress_profile_resolution {
            "Unsaved Draft".to_string()
        } else {
            self.active_profile
                .clone()
                .unwrap_or_else(|| "Direct CLI/config".to_string())
        }
    }

    fn sync_theme(&mut self, ctx: &egui::Context) {
        if self.follows_system_theme {
            let system_mode = system_theme_mode(ctx);
            if system_mode != self.theme_seed.mode {
                self.theme_seed = app_theme_seed(system_mode);
                self.apply_theme(ctx);
            }
        }
    }

    fn apply_theme(&self, ctx: &egui::Context) {
        cast::set_theme(ctx, self.theme_seed.clone().resolve());
    }

    fn toggle_theme(&mut self, ctx: &egui::Context) {
        let mode = match self.theme_seed.mode {
            cast::ThemeMode::Light => cast::ThemeMode::Dark,
            cast::ThemeMode::Dark => cast::ThemeMode::Light,
        };
        self.theme_seed = app_theme_seed(mode);
        self.follows_system_theme = false;
        self.apply_theme(ctx);
    }

    fn follow_system_theme(&mut self, ctx: &egui::Context) {
        self.theme_seed = app_theme_seed(system_theme_mode(ctx));
        self.follows_system_theme = true;
        self.apply_theme(ctx);
    }

    fn theme_toggle_label(&self) -> &'static str {
        match self.theme_seed.mode {
            cast::ThemeMode::Light => "Dark Mode",
            cast::ThemeMode::Dark => "Light Mode",
        }
    }

    fn current_detail_session_id(&self) -> Option<String> {
        match self.tab {
            TabKind::LiveSessions => self
                .selected_live_session()
                .map(|session| session.session_id),
            TabKind::Sessions => self.selected_session().map(|session| session.session_id),
            _ => None,
        }
    }

    fn ensure_session_detail_loaded(&mut self) {
        let Some(session_id) = self.current_detail_session_id() else {
            self.requested_session_detail = None;
            return;
        };

        if self.dashboard.session_detail(&session_id).is_some() {
            self.requested_session_detail = Some(session_id);
            return;
        }

        if self.requested_session_detail.as_deref() == Some(session_id.as_str()) {
            return;
        }

        self.requested_session_detail = Some(session_id.clone());
        self.send_command(OperatorCommand::LoadSessionDetail { session_id });
    }

    fn render_default_shell(&mut self, ui: &mut egui::Ui) {
        egui::Panel::top("default_shell_top").show_inside(ui, |ui| {
            ui.add_space(8.0);
            ui.horizontal_wrapped(|ui| {
                themed_heading(ui, "Turin", 26.0);
                ui.add_space(10.0);
                self.render_connection_status_inline(ui);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    self.render_theme_controls(ui);
                    if ui
                        .add(
                            cast::Button::new(if self.runtime_tools_open {
                                "Hide Tools"
                            } else {
                                "Runtime Tools"
                            })
                            .size(cast::Size::Small)
                            .variant(cast::Variant::Outline),
                        )
                        .clicked()
                    {
                        self.runtime_tools_open = !self.runtime_tools_open;
                    }
                    if ui
                        .add(
                            cast::Button::new("Refresh")
                                .size(cast::Size::Small)
                                .variant(cast::Variant::Ghost),
                        )
                        .clicked()
                    {
                        self.send_command(OperatorCommand::Refresh);
                    }
                });
            });
            ui.add_space(8.0);
        });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ScrollArea::vertical().show(ui, |ui| {
                self.render_default_operator_console(ui);
                if self.runtime_tools_open {
                    ui.add_space(14.0);
                    self.render_runtime_tools_panel(ui);
                }
            });
        });
    }

    fn render_connection_status_inline(&self, ui: &mut egui::Ui) {
        let ready = self
            .dashboard
            .health
            .as_ref()
            .is_some_and(|health| health.ready);
        ui.add(
            cast::Badge::new(if ready { "Connected" } else { "Degraded" })
                .intent(if ready {
                    cast::Intent::Success
                } else {
                    cast::Intent::Warning
                })
                .status_dot(),
        );
        ui.add(cast::Badge::new(match self.dashboard.connection_kind {
            ConnectionKind::Local => "Local",
            ConnectionKind::Remote => "Remote",
        }));
    }

    fn render_operator_shell(&mut self, ui: &mut egui::Ui) {
        self.request_selected_ui_lists(false);
        let apps = self.dashboard.ui.apps().cloned().collect::<Vec<_>>();
        self.ui_app_index = clamp_index(self.ui_app_index, apps.len());
        let Some(app) = apps.get(self.ui_app_index).cloned() else {
            self.render_default_operator_console(ui);
            return;
        };

        let theme = cast::theme_for_ui(ui);
        egui::Panel::left("operator_shell_nav")
            .resizable(false)
            .exact_size(252.0)
            .frame(
                egui::Frame::new()
                    .fill(theme.colors.surface)
                    .stroke(egui::Stroke::new(theme.stroke.sm, theme.colors.border))
                    .inner_margin(egui::Margin::symmetric(16, 18)),
            )
            .show_inside(ui, |ui| {
                self.render_operator_sidebar(ui, &apps, &app);
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            ScrollArea::vertical().show(ui, |ui| {
                ui.add_space(24.0);
                let content_width = (ui.available_width() - 48.0).clamp(640.0, 1120.0);
                let inset = ((ui.available_width() - content_width) / 2.0).max(24.0);
                ui.horizontal_top(|ui| {
                    ui.add_space(inset);
                    ui.vertical(|ui| {
                        ui.set_width(content_width);
                        self.render_operator_stage(ui, &app);
                    });
                });
                ui.add_space(32.0);
            });
        });
    }

    fn render_operator_sidebar(
        &mut self,
        ui: &mut egui::Ui,
        apps: &[UiAppRecord],
        app: &UiAppRecord,
    ) {
        themed_heading(ui, ui_app_title(app), 24.0);
        if let Some(definition) = &app.definition
            && let Some(about) = &definition.about
        {
            ui.add_space(3.0);
            themed_muted(ui, about.clone());
        }
        ui.add_space(20.0);

        if apps.len() > 1 {
            themed_muted(ui, "Switch app");
            ui.add_space(6.0);
            for (index, candidate) in apps.iter().enumerate() {
                let selected = candidate.id == app.id;
                if ui
                    .add(
                        cast::MenuItem::new(ui_app_title(candidate))
                            .selected(selected)
                            .intent(if selected {
                                cast::Intent::Primary
                            } else {
                                cast::Intent::Neutral
                            }),
                    )
                    .clicked()
                {
                    self.ui_app_index = index;
                    self.open_harness_screen(
                        candidate,
                        harness_ui::default_screen_index(candidate),
                    );
                }
            }
            ui.add_space(16.0);
        }

        let current_screen_id = self.active_ui_screen_id(app);
        if app.menus.is_empty() {
            for (index, screen) in app.screens.values().enumerate() {
                let selected = Some(screen.id.as_str()) == current_screen_id.as_deref();
                if ui
                    .add(
                        cast::MenuItem::new(screen.title.clone())
                            .selected(selected)
                            .intent(if selected {
                                cast::Intent::Primary
                            } else {
                                cast::Intent::Neutral
                            }),
                    )
                    .clicked()
                {
                    self.open_harness_screen(app, index);
                }
            }
        } else {
            let show_menu_titles = app.menus.len() > 1
                || app
                    .menus
                    .first()
                    .is_some_and(|menu| !menu.title.eq_ignore_ascii_case("main"));
            for menu in &app.menus {
                if show_menu_titles {
                    themed_muted(ui, menu.title.clone());
                    ui.add_space(5.0);
                }
                self.render_operator_menu_items(
                    ui,
                    app,
                    &menu.items,
                    current_screen_id.as_deref(),
                    0,
                );
                ui.add_space(8.0);
            }
        }

        ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
            ui.vertical(|ui| self.render_operator_settings(ui));
        });
    }

    fn render_operator_menu_items(
        &mut self,
        ui: &mut egui::Ui,
        app: &UiAppRecord,
        items: &[UiMenuItem],
        current_screen_id: Option<&str>,
        depth: usize,
    ) {
        for item in items {
            let selected = Some(item.opens.as_str()) == current_screen_id;
            ui.horizontal(|ui| {
                ui.add_space(depth as f32 * 14.0);
                if ui
                    .add(
                        cast::MenuItem::new(item.label.clone())
                            .selected(selected)
                            .intent(if selected {
                                cast::Intent::Primary
                            } else {
                                cast::Intent::Neutral
                            }),
                    )
                    .clicked()
                    && let Some(screen_index) =
                        harness_ui::screen_index_for_target(app, &item.opens)
                {
                    self.open_harness_screen(app, screen_index);
                }
            });
            if !item.items.is_empty() {
                self.render_operator_menu_items(ui, app, &item.items, current_screen_id, depth + 1);
            }
        }
    }

    fn render_operator_stage(&mut self, ui: &mut egui::Ui, app: &UiAppRecord) {
        self.render_operator_feedback(ui, app);

        let mut screen_index = self
            .ui_screen_indices
            .get(&app.id)
            .copied()
            .unwrap_or_else(|| harness_ui::default_screen_index(app));
        let mut render_state = harness_ui::HarnessRenderState {
            lists: &self.ui_lists,
            requested_lists: &self.requested_ui_lists,
            list_errors: &self.ui_list_errors,
            form_values: &mut self.ui_form_values,
            selected_list_items: &mut self.ui_selected_list_items,
        };
        let event = harness_ui::render_harness_screen_content(
            ui,
            app,
            &mut screen_index,
            &mut render_state,
        );
        self.ui_screen_indices.insert(app.id.clone(), screen_index);
        if let Some(event) = event {
            self.handle_harness_ui_event(app, event);
        }

        self.render_pending_harness_ui_action(ui, &app.id);
        self.render_active_harness_pane(ui, app);

        if self.runtime_tools_open {
            ui.add_space(14.0);
            self.render_runtime_tools_panel(ui);
        }
    }

    fn operator_connection_summary(&self) -> (&'static str, cast::Intent) {
        let ready = self
            .dashboard
            .health
            .as_ref()
            .is_some_and(|health| health.ready);
        match (ready, self.dashboard.connection_kind) {
            (true, ConnectionKind::Local) => ("Connected locally", cast::Intent::Success),
            (true, ConnectionKind::Remote) => ("Connected remotely", cast::Intent::Success),
            (false, _) => ("Connection degraded", cast::Intent::Warning),
        }
    }

    fn render_operator_settings(&mut self, ui: &mut egui::Ui) {
        ui.add(cast::Separator::new());
        ui.add_space(8.0);
        let (connection, intent) = self.operator_connection_summary();
        let mut open = self.sidebar_settings_open;
        cast::Disclosure::new(&mut open, "Settings")
            .trailing_status_dot(connection, intent)
            .size(cast::Size::Small)
            .show(ui, |ui| {
                self.render_theme_controls(ui);
                ui.add_space(6.0);
                if ui
                    .add(
                        cast::Button::new("Refresh data")
                            .size(cast::Size::Small)
                            .variant(cast::Variant::Ghost),
                    )
                    .clicked()
                {
                    self.request_selected_ui_lists(true);
                }
                if ui
                    .add(
                        cast::Button::new(if self.runtime_tools_open {
                            "Close runtime tools"
                        } else {
                            "Open runtime tools"
                        })
                        .size(cast::Size::Small)
                        .variant(cast::Variant::Ghost),
                    )
                    .clicked()
                {
                    self.runtime_tools_open = !self.runtime_tools_open;
                }
            });
        self.sidebar_settings_open = open;
    }

    fn render_theme_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui
                .add(
                    cast::Button::new(self.theme_toggle_label())
                        .size(cast::Size::Small)
                        .variant(cast::Variant::Outline),
                )
                .clicked()
            {
                self.toggle_theme(ui.ctx());
            }
            if !self.follows_system_theme
                && ui
                    .add(
                        cast::Button::new("Use System")
                            .size(cast::Size::Small)
                            .variant(cast::Variant::Ghost),
                    )
                    .clicked()
            {
                self.follow_system_theme(ui.ctx());
            }
        });
    }

    fn render_operator_feedback(&mut self, ui: &mut egui::Ui, app: &UiAppRecord) {
        let mut feedback = Vec::new();

        if let Some(error) = &self.dashboard.last_error {
            feedback.push((
                format!("runtime-error:{error}"),
                cast::Toast::new("Something went wrong")
                    .body(error.clone())
                    .intent(cast::Intent::Danger),
            ));
        }

        if let Some(result) = self
            .latest_harness_action_result
            .as_ref()
            .filter(|result| harness_action_result_matches_app(result, app))
        {
            let label = self.harness_action_label(&app.id, &result.action);
            feedback.push((
                format!("action-success:{}", self.harness_feedback_revision),
                cast::Toast::new(format!("{label} completed"))
                    .body("The workflow has been updated.")
                    .intent(cast::Intent::Success),
            ));
        } else if let Some(failure) = self
            .latest_harness_action_failure
            .as_ref()
            .filter(|failure| harness_action_failure_matches_app(failure, app))
        {
            let label = self.harness_action_label(&app.id, &failure.action);
            feedback.push((
                format!("action-failure:{}", self.harness_feedback_revision),
                cast::Toast::new(format!("{label} failed"))
                    .body("The action could not be completed. Runtime Tools contains the technical details.")
                    .intent(cast::Intent::Danger),
            ));
        }

        feedback.extend(
            self.dashboard
                .ui
                .notices()
                .iter()
                .rev()
                .filter(|notice| notice.app_id == app.id)
                .take(3)
                .map(|notice| {
                    let body = notice.body.clone().unwrap_or_default();
                    (
                        format!("notice:{}:{}:{body}", notice.app_id, notice.title),
                        cast::Toast::new(notice.title.clone())
                            .body(body)
                            .intent(ui_notice_intent(notice.level)),
                    )
                }),
        );

        feedback.retain(|(key, _)| !self.dismissed_operator_feedback.contains(key));
        if feedback.is_empty() {
            return;
        }

        let keys = feedback
            .iter()
            .map(|(key, _)| key.clone())
            .collect::<Vec<_>>();
        let toasts = feedback
            .into_iter()
            .map(|(_, toast)| toast)
            .collect::<Vec<_>>();
        if let Some(response) = cast::ToastStack::new("operator_feedback", &toasts)
            .width(340.0)
            .show(ui.ctx())
        {
            for index in response.inner.dismissed_indices {
                if let Some(key) = keys.get(index) {
                    self.dismissed_operator_feedback.insert(key.clone());
                }
            }
        }
    }

    fn harness_action_label(&self, app_id: &str, action: &str) -> String {
        self.latest_harness_action_label
            .as_ref()
            .filter(|(candidate_app, candidate_action, _)| {
                candidate_app == app_id && candidate_action == action
            })
            .map(|(_, _, label)| label.clone())
            .unwrap_or_else(|| "Action".to_string())
    }

    fn render_runtime_tools_panel(&mut self, ui: &mut egui::Ui) {
        cast::Panel::new().show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                themed_heading(ui, "Runtime Tools", 24.0);
                ui.add_space(8.0);
                if ui
                    .add(
                        cast::Button::new("Close")
                            .size(cast::Size::Small)
                            .variant(cast::Variant::Ghost),
                    )
                    .clicked()
                {
                    self.runtime_tools_open = false;
                }
            });
            ui.add_space(8.0);
            self.render_tab_bar(ui);
            ui.add_space(10.0);
            self.render_active_tab(ui);
        });
    }

    fn active_ui_screen_id(&self, app: &UiAppRecord) -> Option<String> {
        let screen_index = self
            .ui_screen_indices
            .get(&app.id)
            .copied()
            .unwrap_or_else(|| harness_ui::default_screen_index(app));
        app.screens
            .values()
            .nth(screen_index)
            .map(|screen| screen.id.clone())
    }

    fn render_tab_bar(&mut self, ui: &mut egui::Ui) {
        let mut selected = self.tab.as_index();
        let labels = TabKind::ALL.map(|tab| tab.title());
        if ui
            .add(cast::Tabs::new(&mut selected, labels).size(cast::Size::Small))
            .changed()
        {
            self.tab = TabKind::from_index(selected);
        }
    }

    fn render_active_tab(&mut self, ui: &mut egui::Ui) {
        match self.tab {
            TabKind::Connections => self.render_connections_tab(ui),
            TabKind::UiApps => self.render_ui_apps_tab(ui),
            TabKind::Agents => self.render_agents_tab(ui),
            TabKind::LiveSessions => self.render_live_sessions_tab(ui),
            TabKind::Sessions => self.render_sessions_tab(ui),
            TabKind::Tasks => self.render_tasks_tab(ui),
            TabKind::Channels => self.render_channels_tab(ui),
            TabKind::Events => self.render_events_tab(ui),
        }
    }

    fn render_connections_tab(&mut self, ui: &mut egui::Ui) {
        let profiles = self
            .profile_catalog
            .as_ref()
            .map(|catalog| catalog.profiles().to_vec())
            .unwrap_or_default();
        let recent_drafts = self.recent_drafts.drafts().to_vec();
        let selected = self.selected_profile().cloned();
        let profiles_source = self.connection_options.profiles_path();
        let typed_name_ready = self.typed_profile_name().is_some();
        let editor_diff = self.editor_diff();
        let selected_diff = self.selected_profile_diff();
        let update_selected_ready = selected_diff
            .as_ref()
            .is_some_and(|diff| !diff.is_empty() && self.profile_draft_validation().is_valid());
        let selected_activity = selected
            .as_ref()
            .and_then(|profile| self.profile_activity.entry(&profile.name))
            .cloned();
        let profile_name_hint = selected
            .as_ref()
            .map(|profile| profile.name.clone())
            .unwrap_or_else(|| "new-profile".to_string());

        ui.columns(2, |columns| {
            cast::Panel::new().show(&mut columns[0], |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.heading("Connection Profiles");
                    ui.add_space(8.0);
                    ui.add(cast::Badge::new(format!("{} profiles", profiles.len())));
                });
                ui.add_space(6.0);
                ui.label(format!("Source: {}", profiles_source.display()));
                ui.add_space(8.0);
                if profiles.is_empty() {
                    ui.label(format!(
                        "No profiles loaded. Add {} or pass --profiles-file.",
                        DEFAULT_UI_PROFILES_PATH
                    ));
                } else {
                    let labels = profiles
                        .iter()
                        .map(|profile| {
                            format!(
                                "{}{} · {} · {}",
                                profile.name,
                                if profile.is_default { " (default)" } else { "" },
                                profile_kind_label(profile.kind),
                                profile_auth_label(profile.auth.as_ref())
                            )
                        })
                        .collect::<Vec<_>>();
                    let mut profile_index = self.profile_index;
                    ScrollArea::vertical().show(ui, |ui| {
                        ui.add(
                            cast::NavList::new(&mut profile_index, labels).size(cast::Size::Small),
                        );
                    });
                    if profile_index != self.profile_index {
                        self.profile_index = profile_index;
                        self.pending_delete_profile = None;
                    }
                }

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(8.0);
                ui.label(RichText::new("Manage Profiles").strong());
                ui.add_space(6.0);
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .add(cast::Button::new("Load Current").size(cast::Size::Small))
                        .clicked()
                    {
                        self.load_current_connection_into_editor();
                    }
                    if ui
                        .add(cast::Button::new("Load Selected").size(cast::Size::Small))
                        .clicked()
                    {
                        self.load_selected_profile_into_editor();
                    }
                    if ui
                        .add(cast::Button::new("Load Latest Recent").size(cast::Size::Small))
                        .clicked()
                    {
                        self.load_latest_recent_draft();
                    }
                    if ui
                        .add(
                            cast::Button::new("New Draft")
                                .size(cast::Size::Small)
                                .variant(cast::Variant::Outline),
                        )
                        .clicked()
                    {
                        self.reset_profile_editor();
                    }
                });
                ui.add_space(8.0);
                ui.label(RichText::new("Recent Drafts").strong());
                ui.add_space(4.0);
                if recent_drafts.is_empty() {
                    ui.label("No successful draft connections yet.");
                } else {
                    let labels = recent_drafts
                        .iter()
                        .map(ConnectionProfileDraft::summary_label)
                        .collect::<Vec<_>>();
                    ScrollArea::vertical().max_height(132.0).show(ui, |ui| {
                        ui.add(
                            cast::NavList::new(&mut self.recent_draft_index, labels)
                                .size(cast::Size::Small),
                        );
                    });
                    ui.add_space(6.0);
                    if ui
                        .add(
                            cast::Button::new("Load Selected Recent")
                                .size(cast::Size::Small)
                                .variant(cast::Variant::Outline),
                        )
                        .clicked()
                    {
                        self.load_selected_recent_draft();
                    }
                }
                ui.add_space(6.0);
                ui.add(
                    cast::TextInput::new(&mut self.profile_name_input)
                        .label("Save As Name")
                        .hint_text(profile_name_hint.clone()),
                );
                ui.add_space(8.0);
                ui.label(RichText::new("Kind").strong());
                let mut kind_index = profile_kind_index(self.profile_draft.kind);
                if ui
                    .add(
                        cast::SegmentedControl::new(
                            &mut kind_index,
                            ["Local Config", "Local Endpoint", "Remote"],
                        )
                        .size(cast::Size::Small),
                    )
                    .changed()
                {
                    self.profile_draft.kind = profile_kind_from_index(kind_index);
                }
                ui.add_space(6.0);
                ui.add(
                    cast::TextInput::new(&mut self.profile_draft.target)
                        .label(profile_target_label(self.profile_draft.kind))
                        .hint_text(profile_target_hint(self.profile_draft.kind)),
                );
                let target_validation = self.profile_draft.validate();
                if let Some(message) = target_validation.target_error.as_ref() {
                    ui.add(
                        cast::Badge::new(message.clone())
                            .intent(cast::Intent::Danger)
                            .variant(cast::Variant::Subtle),
                    );
                } else if let Some(message) = target_validation.target_notice.as_ref() {
                    ui.add(
                        cast::Badge::new(message.clone())
                            .intent(cast::Intent::Warning)
                            .variant(cast::Variant::Subtle),
                    );
                }
                if self.profile_draft.kind == ConnectionProfileKind::Remote {
                    ui.add_space(8.0);
                    ui.label(RichText::new("Auth").strong());
                    let mut auth_index = profile_auth_mode_index(self.profile_draft.auth_mode);
                    if ui
                        .add(
                            cast::SegmentedControl::new(
                                &mut auth_index,
                                ["Token Env", "Inline Token", "None"],
                            )
                            .size(cast::Size::Small),
                        )
                        .changed()
                    {
                        self.profile_draft.auth_mode = profile_auth_mode_from_index(auth_index);
                    }
                    ui.add_space(6.0);
                    if self.profile_draft.auth_mode == ConnectionProfileDraftAuthMode::InlineToken {
                        ui.add(
                            cast::TextInput::new(&mut self.profile_draft.auth_value)
                                .label(profile_auth_value_label(self.profile_draft.auth_mode))
                                .password(true)
                                .hint_text(profile_auth_value_hint(self.profile_draft.auth_mode)),
                        );
                    } else {
                        ui.add(
                            cast::TextInput::new(&mut self.profile_draft.auth_value)
                                .label(profile_auth_value_label(self.profile_draft.auth_mode))
                                .hint_text(profile_auth_value_hint(self.profile_draft.auth_mode)),
                        );
                    }
                    let auth_validation = self.profile_draft.validate();
                    if let Some(message) = auth_validation.auth_error.as_ref() {
                        ui.add(
                            cast::Badge::new(message.clone())
                                .intent(cast::Intent::Danger)
                                .variant(cast::Variant::Subtle),
                        );
                    } else if let Some(message) = auth_validation.auth_notice.as_ref() {
                        ui.add(
                            cast::Badge::new(message.clone())
                                .intent(cast::Intent::Warning)
                                .variant(cast::Variant::Subtle),
                        );
                    }
                } else {
                    self.profile_draft.auth_mode = ConnectionProfileDraftAuthMode::None;
                    self.profile_draft.auth_value.clear();
                }
                let draft_validation = self.profile_draft_validation();
                ui.add_space(8.0);
                ui.label(RichText::new("Draft Validation").strong());
                ui.add(
                    cast::Badge::new(draft_validation.summary())
                        .intent(if draft_validation.is_valid() {
                            cast::Intent::Success
                        } else {
                            cast::Intent::Danger
                        })
                        .status_dot(),
                );
                if self.editor_is_dirty() {
                    ui.add(
                        cast::Badge::new(format!(
                            "Unsaved editor changes vs {}",
                            self.draft_baseline_label
                        ))
                        .intent(cast::Intent::Warning)
                        .variant(cast::Variant::Subtle),
                    );
                }
                ui.add(
                    cast::Checkbox::new(&mut self.save_profile_as_default, "Set as default")
                        .size(cast::Size::Small),
                );
                ui.add_space(8.0);
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .add(
                            cast::Button::new("Test Draft")
                                .size(cast::Size::Small)
                                .enabled(draft_validation.is_valid()),
                        )
                        .clicked()
                    {
                        self.preflight_draft();
                    }
                    if ui
                        .add(
                            cast::Button::new("Connect Draft")
                                .size(cast::Size::Small)
                                .enabled(draft_validation.is_valid()),
                        )
                        .clicked()
                    {
                        self.connect_profile_draft();
                    }
                    if ui
                        .add(
                            cast::Button::new("Test Selected")
                                .size(cast::Size::Small)
                                .enabled(selected.is_some()),
                        )
                        .clicked()
                    {
                        self.preflight_selected_profile();
                    }
                    if ui
                        .add(
                            cast::Button::new("Ensure Draft Local")
                                .size(cast::Size::Small)
                                .enabled(
                                    self.profile_draft.kind == ConnectionProfileKind::LocalConfig,
                                ),
                        )
                        .clicked()
                    {
                        self.ensure_local_daemon_for_draft();
                    }
                    if ui
                        .add(
                            cast::Button::new("Update Selected")
                                .size(cast::Size::Small)
                                .enabled(update_selected_ready),
                        )
                        .clicked()
                    {
                        self.update_selected_profile();
                    }
                    if ui
                        .add(
                            cast::Button::new("Save As Name")
                                .size(cast::Size::Small)
                                .enabled(draft_validation.is_valid() && typed_name_ready),
                        )
                        .clicked()
                    {
                        self.save_current_profile();
                    }
                    if ui
                        .add(
                            cast::Button::new("Duplicate Selected")
                                .size(cast::Size::Small)
                                .variant(cast::Variant::Outline),
                        )
                        .clicked()
                    {
                        self.duplicate_selected_profile();
                    }
                    if ui
                        .add(
                            cast::Button::new("Rename Selected")
                                .size(cast::Size::Small)
                                .variant(cast::Variant::Outline),
                        )
                        .clicked()
                    {
                        self.rename_selected_profile();
                    }
                });
                if let Some(action) = self.pending_discard_action.as_ref() {
                    let action_description = action.description();
                    ui.add_space(6.0);
                    ui.horizontal_wrapped(|ui| {
                        ui.add(
                            cast::Badge::new(format!("Pending: {}", action_description))
                                .intent(cast::Intent::Warning)
                                .variant(cast::Variant::Subtle),
                        );
                        if ui
                            .add(
                                cast::Button::new("Discard Pending Action")
                                    .size(cast::Size::Small)
                                    .intent(cast::Intent::Warning)
                                    .variant(cast::Variant::Outline),
                            )
                            .clicked()
                        {
                            self.confirm_pending_discard_action();
                        }
                        if ui
                            .add(
                                cast::Button::new("Cancel Pending Action")
                                    .size(cast::Size::Small)
                                    .variant(cast::Variant::Ghost),
                            )
                            .clicked()
                        {
                            self.cancel_pending_discard_action();
                        }
                    });
                }
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    let armed = self.is_delete_armed_for_selected();
                    if ui
                        .add(
                            cast::Button::new(if armed {
                                "Confirm Delete"
                            } else {
                                "Arm Delete"
                            })
                            .size(cast::Size::Small)
                            .intent(cast::Intent::Danger)
                            .variant(cast::Variant::Outline),
                        )
                        .clicked()
                    {
                        if armed {
                            self.delete_selected_profile();
                        } else {
                            self.arm_delete_selected_profile();
                        }
                    }
                    if armed
                        && ui
                            .add(
                                cast::Button::new("Cancel Delete")
                                    .size(cast::Size::Small)
                                    .variant(cast::Variant::Ghost),
                            )
                            .clicked()
                    {
                        self.cancel_delete_selected_profile();
                    }
                });
                if let Some(profile_name) = self.pending_delete_profile.as_deref() {
                    ui.add_space(6.0);
                    ui.add(
                        cast::Badge::new(format!(
                            "Delete armed for '{}'. Confirm to remove it from the profiles file.",
                            profile_name
                        ))
                        .intent(cast::Intent::Danger)
                        .variant(cast::Variant::Subtle),
                    );
                }
            });

            cast::Panel::new().show(&mut columns[1], |ui| {
                let draft_validation = self.profile_draft_validation();
                ui.horizontal_wrapped(|ui| {
                    ui.heading("Connection Detail");
                    ui.add(
                        cast::Badge::new(connection_kind_label(self.dashboard.connection_kind))
                            .intent(cast::Intent::Info)
                            .variant(cast::Variant::Outline),
                    );
                    ui.add(
                        cast::Badge::new(freshness_label(self.dashboard.snapshot_freshness()))
                            .intent(freshness_intent(self.dashboard.snapshot_freshness()))
                            .status_dot(),
                    );
                });
                ui.add_space(8.0);
                detail_kv(
                    ui,
                    "Current Target",
                    self.dashboard.connection_target.clone(),
                );
                detail_kv(
                    ui,
                    "Connection Kind",
                    connection_kind_label(self.dashboard.connection_kind),
                );
                detail_kv(
                    ui,
                    "Snapshot Freshness",
                    format!(
                        "{} ({})",
                        freshness_label(self.dashboard.snapshot_freshness()),
                        self.dashboard.snapshot_age_label()
                    ),
                );
                detail_kv(ui, "Last Event", self.dashboard.event_age_label());
                detail_kv(ui, "Last Notice", self.dashboard.notice_age_label());
                detail_kv(
                    ui,
                    "Events Observed",
                    self.dashboard.total_event_count.to_string(),
                );
                detail_kv(
                    ui,
                    "Refresh Successes",
                    self.dashboard.refresh_success_count.to_string(),
                );
                detail_kv(
                    ui,
                    "Refresh Failures",
                    self.dashboard.refresh_failure_count.to_string(),
                );
                detail_kv(
                    ui,
                    "Last Refresh",
                    format!(
                        "{} ({})",
                        self.dashboard.last_refresh_status_label(),
                        self.dashboard.last_refresh_latency_label()
                    ),
                );
                detail_kv(ui, "Active Profile", self.active_connection_label());
                detail_kv(ui, "Profiles File", profiles_source.display().to_string());
                detail_kv(ui, "Available Profiles", profiles.len().to_string());
                detail_kv(ui, "Recent Drafts", recent_drafts.len().to_string());
                if let Some(health) = self.dashboard.health.as_ref() {
                    detail_kv(ui, "Transport", health.transport.clone());
                    detail_kv(ui, "Wire Format", health.wire_format.clone());
                    detail_kv(ui, "Ready", yes_no(health.ready));
                }
                detail_kv(
                    ui,
                    "Last Error",
                    self.dashboard
                        .last_error
                        .clone()
                        .unwrap_or_else(|| "None".to_string()),
                );
                detail_kv(
                    ui,
                    "Last Info",
                    self.dashboard
                        .last_info
                        .clone()
                        .unwrap_or_else(|| "None".to_string()),
                );

                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    if ui
                        .add(cast::Button::new("Reconnect Current").size(cast::Size::Small))
                        .clicked()
                    {
                        self.reconnect_current();
                    }
                    if ui
                        .add(
                            cast::Button::new("Reload Profiles")
                                .size(cast::Size::Small)
                                .variant(cast::Variant::Outline),
                        )
                        .clicked()
                    {
                        self.reload_profiles();
                    }
                });

                if let Some(profile) = selected.as_ref() {
                    ui.add_space(12.0);
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new("Selected Profile").strong());
                        ui.add(
                            cast::Badge::new(profile_kind_label(profile.kind))
                                .variant(cast::Variant::Outline),
                        );
                    });
                    detail_kv(ui, "Name", profile.name.clone());
                    detail_kv(ui, "Kind", profile_kind_label(profile.kind));
                    detail_kv(ui, "Target", profile.target.clone());
                    detail_kv(ui, "Default", yes_no(profile.is_default));
                    detail_kv(ui, "Auth", profile_auth_label(profile.auth.as_ref()));
                    ui.add_space(10.0);
                    if ui
                        .add(cast::Button::new("Connect Selected Profile").size(cast::Size::Small))
                        .clicked()
                    {
                        self.connect_selected_profile();
                    }
                }

                ui.add_space(12.0);
                ui.label(RichText::new("Editor Draft").strong());
                detail_kv(
                    ui,
                    "Draft Kind",
                    profile_kind_label(self.profile_draft.kind),
                );
                detail_kv(ui, "Draft Target", self.profile_draft.target.clone());
                detail_kv(
                    ui,
                    "Draft Auth",
                    profile_draft_auth_label(self.profile_draft.auth_mode),
                );
                detail_kv(
                    ui,
                    "Draft Status",
                    if draft_validation.is_valid() {
                        "valid".to_string()
                    } else {
                        "invalid".to_string()
                    },
                );
                detail_kv(ui, "Draft Summary", draft_validation.summary());
                detail_kv(
                    ui,
                    "Draft Dirty",
                    if self.editor_is_dirty() {
                        "yes".to_string()
                    } else {
                        "no".to_string()
                    },
                );
                detail_kv(ui, "Draft Baseline", self.draft_baseline_label.clone());
                detail_kv(ui, "Draft Changes", editor_diff.summary());
                detail_kv(
                    ui,
                    "Selected Update Ready",
                    if update_selected_ready {
                        "yes".to_string()
                    } else {
                        "no".to_string()
                    },
                );
                detail_kv(
                    ui,
                    "Named Save Ready",
                    if draft_validation.is_valid() && typed_name_ready {
                        "yes".to_string()
                    } else {
                        "no".to_string()
                    },
                );
                if let Some(draft) = self.selected_recent_draft() {
                    detail_kv(ui, "Recent Draft Selection", draft.summary_label());
                }
                if let Some(diff) = selected_diff.as_ref() {
                    ui.add_space(10.0);
                    ui.label(RichText::new("Selected Profile Diff").strong());
                    detail_kv(ui, "Summary", diff.summary());
                    if diff.is_empty() {
                        ui.label("The editor draft already matches the selected saved profile.");
                    } else {
                        for field in &diff.changed_fields {
                            detail_kv(
                                ui,
                                &field.field,
                                format!("{} -> {}", field.comparison_value, field.draft_value),
                            );
                        }
                    }
                }
                if let Some(activity) = selected_activity.as_ref() {
                    ui.add_space(10.0);
                    ui.label(RichText::new("Selected Profile Activity").strong());
                    detail_kv(ui, "Connect Attempts", activity.connect_count.to_string());
                    detail_kv(
                        ui,
                        "Successful Connects",
                        activity.successful_connect_count.to_string(),
                    );
                    detail_kv(ui, "Preflights", activity.preflight_count.to_string());
                    detail_kv(ui, "Failures", activity.failure_count.to_string());
                    detail_kv(
                        ui,
                        "Last Connect",
                        activity
                            .last_connect_result
                            .clone()
                            .unwrap_or_else(|| "None".to_string()),
                    );
                    detail_kv(
                        ui,
                        "Last Preflight",
                        activity
                            .last_preflight_result
                            .clone()
                            .unwrap_or_else(|| "None".to_string()),
                    );
                }
                if let Some(report) = self.last_preflight_report.as_ref() {
                    ui.add_space(10.0);
                    ui.label(RichText::new("Latest Preflight").strong());
                    detail_kv(ui, "Outcome", preflight_outcome_label(report.outcome));
                    detail_kv(ui, "Target", report.target.clone());
                    detail_kv(ui, "Auth", report.auth.clone());
                    detail_kv(ui, "Message", report.message.clone());
                    detail_kv(
                        ui,
                        "Latency",
                        report
                            .latency_ms
                            .map(|latency| format!("{latency}ms"))
                            .unwrap_or_else(|| "None".to_string()),
                    );
                    if let Some(ready) = report.ready {
                        detail_kv(ui, "Ready", yes_no(ready));
                    }
                    if let Some(transport) = report.transport.as_ref() {
                        detail_kv(ui, "Transport", transport.clone());
                    }
                    if let Some(wire_format) = report.wire_format.as_ref() {
                        detail_kv(ui, "Wire Format", wire_format.clone());
                    }
                }

                if !self.dashboard.recent_notices.is_empty() {
                    ui.add_space(12.0);
                    ui.label(RichText::new("Recent Notices").strong());
                    ui.add_space(6.0);
                    ScrollArea::vertical().max_height(150.0).show(ui, |ui| {
                        for notice in self.dashboard.recent_notices.iter().rev().take(6) {
                            ui.label(
                                RichText::new(format!(
                                    "{} {}",
                                    notice_level_label(notice.level),
                                    notice.message
                                ))
                                .color(notice_level_color(notice.level)),
                            );
                        }
                    });
                }
            });
        });
    }

    fn render_ui_apps_tab(&mut self, ui: &mut egui::Ui) {
        self.request_selected_ui_lists(false);
        let apps = self.dashboard.ui.apps().cloned().collect::<Vec<_>>();
        self.ui_app_index = clamp_index(self.ui_app_index, apps.len());
        let selected = apps.get(self.ui_app_index).cloned();

        ui.columns(2, |columns| {
            cast::Panel::new().show(&mut columns[0], |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.heading("Harness UI Apps");
                    ui.add_space(8.0);
                    ui.add(cast::Badge::new(format!("{} apps", apps.len())));
                });
                ui.add_space(8.0);
                if apps.is_empty() {
                    cast::EmptyState::new(DEFAULT_OPERATOR_EMPTY_TITLE)
                        .body(DEFAULT_OPERATOR_EMPTY_BODY)
                        .icon("UI")
                        .intent(cast::Intent::Info)
                        .show(ui, |_| {});
                } else {
                    let labels = apps
                        .iter()
                        .map(|app| {
                            format!(
                                "{} · {} screens · {} panes",
                                ui_app_title(app),
                                app.screens.len(),
                                app.panes.len()
                            )
                        })
                        .collect::<Vec<_>>();
                    ScrollArea::vertical().show(ui, |ui| {
                        ui.add(
                            cast::NavList::new(&mut self.ui_app_index, labels)
                                .size(cast::Size::Small),
                        );
                    });
                }

                ui.add_space(12.0);
                if ui
                    .add(
                        cast::Button::new("Refresh Selected Lists")
                            .size(cast::Size::Small)
                            .variant(cast::Variant::Outline),
                    )
                    .clicked()
                {
                    self.request_selected_ui_lists(true);
                }
                ui.add_space(8.0);
                ui.label(RichText::new("Dynamic UI Signals").strong());
                ui.horizontal_wrapped(|ui| {
                    ui.add(cast::Badge::new(format!(
                        "Notices: {}",
                        self.dashboard.ui.notices().len()
                    )));
                    ui.add(cast::Badge::new(format!(
                        "Opens: {}",
                        self.dashboard.ui.opens().len()
                    )));
                    ui.add(cast::Badge::new(format!(
                        "Shows: {}",
                        self.dashboard.ui.shows().len()
                    )));
                    ui.add(cast::Badge::new(format!(
                        "Focuses: {}",
                        self.dashboard.ui.focuses().len()
                    )));
                    ui.add(cast::Badge::new(format!(
                        "Refreshes: {}",
                        self.dashboard.ui.refreshes().len()
                    )));
                });
            });

            ScrollArea::vertical().show(&mut columns[1], |ui| {
                let Some(app) = selected else {
                    self.render_default_operator_console(ui);
                    return;
                };

                let mut screen_index = self
                    .ui_screen_indices
                    .get(&app.id)
                    .copied()
                    .unwrap_or_else(|| harness_ui::default_screen_index(&app));
                let mut render_state = harness_ui::HarnessRenderState {
                    lists: &self.ui_lists,
                    requested_lists: &self.requested_ui_lists,
                    list_errors: &self.ui_list_errors,
                    form_values: &mut self.ui_form_values,
                    selected_list_items: &mut self.ui_selected_list_items,
                };
                let event =
                    harness_ui::render_harness_app(ui, &app, &mut screen_index, &mut render_state);
                self.ui_screen_indices.insert(app.id.clone(), screen_index);
                if let Some(event) = event {
                    self.handle_harness_ui_event(&app, event);
                }

                self.render_pending_harness_ui_action(ui, &app.id);
                self.render_latest_harness_action_result(ui, &app);
                self.render_latest_harness_action_failure(ui, &app);
                self.render_active_harness_pane(ui, &app);

                if !self.dashboard.ui.notices().is_empty() {
                    ui.add_space(10.0);
                    ui.label(RichText::new("Recent UI Notices").strong());
                    for notice in self.dashboard.ui.notices().iter().rev().take(5) {
                        if notice.app_id == app.id {
                            ui.horizontal_wrapped(|ui| {
                                ui.add(
                                    cast::Badge::new(notice.app_id.clone())
                                        .variant(cast::Variant::Outline),
                                );
                                ui.label(notice.title.clone());
                            });
                        }
                    }
                }
            });
        });
    }

    fn render_default_operator_console(&mut self, ui: &mut egui::Ui) {
        let summary = DefaultOperatorConsoleSummary::from_dashboard(&self.dashboard);

        cast::Panel::new().show(ui, |ui| {
            themed_heading(ui, DEFAULT_OPERATOR_EMPTY_TITLE, 34.0);
            ui.add_space(8.0);
            ui.label(DEFAULT_OPERATOR_EMPTY_BODY);
            ui.add_space(12.0);
            ui.horizontal_wrapped(|ui| {
                ui.add(
                    cast::Badge::new(summary.freshness.clone())
                        .intent(freshness_intent(self.dashboard.snapshot_freshness()))
                        .status_dot(),
                );
                themed_muted(ui, summary.target.clone());
            });
        });

        ui.add_space(12.0);
        let groups = default_operator_metric_groups(&summary);
        ui.columns(groups.len(), |columns| {
            for (column, group) in columns.iter_mut().zip(groups) {
                cast::Panel::new().show(column, |ui| {
                    themed_heading(ui, group.title, 20.0);
                    ui.add_space(6.0);
                    for (label, value) in group.metrics {
                        ui.horizontal_wrapped(|ui| {
                            themed_heading(ui, value.to_string(), 24.0);
                            themed_muted(ui, label);
                        });
                    }
                });
            }
        });

        ui.add_space(12.0);
        cast::Panel::new().show(ui, |ui| {
            themed_heading(ui, DEFAULT_OPERATOR_GUIDANCE_TITLE, 20.0);
            ui.add_space(6.0);
            ui.label(DEFAULT_OPERATOR_GUIDANCE_BODY);
            ui.add_space(8.0);
            themed_muted(ui, DEFAULT_OPERATOR_INTRO);
        });
    }

    fn render_active_harness_pane(&mut self, ui: &mut egui::Ui, app: &UiAppRecord) {
        let Some(pane_id) = self.ui_active_pane.clone() else {
            return;
        };
        let Some(pane) = app.panes.get(&pane_id).cloned() else {
            self.ui_active_pane = None;
            return;
        };

        let mut close = false;
        let mut pane_event = None;
        let response = egui::Modal::new(egui::Id::new(format!(
            "harness_ui_pane:{}:{}",
            app.id, pane.id
        )))
        .show(ui.ctx(), |ui| {
            ui.set_min_width(560.0);
            let mut render_state = harness_ui::HarnessRenderState {
                lists: &self.ui_lists,
                requested_lists: &self.requested_ui_lists,
                list_errors: &self.ui_list_errors,
                form_values: &mut self.ui_form_values,
                selected_list_items: &mut self.ui_selected_list_items,
            };
            pane_event = harness_ui::render_harness_pane(ui, app, &pane, &mut render_state);
            ui.add_space(8.0);
            ui.separator();
            ui.add_space(8.0);
            if ui
                .add(
                    cast::Button::new("Close Pane")
                        .size(cast::Size::Small)
                        .variant(cast::Variant::Ghost),
                )
                .clicked()
            {
                close = true;
            }
        });

        if response.should_close() {
            close = true;
        }
        if let Some(event) = pane_event {
            self.handle_harness_ui_event(app, event);
        }
        if close {
            self.ui_active_pane = None;
        }
    }

    fn render_latest_harness_action_result(&self, ui: &mut egui::Ui, app: &UiAppRecord) {
        let Some(result) = self.latest_harness_action_result.as_ref() else {
            return;
        };
        if !harness_action_result_matches_app(result, app) {
            return;
        }

        ui.add_space(10.0);
        cast::Panel::new().show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Latest Action Result").strong());
                ui.add(
                    cast::Badge::new(result.action.clone())
                        .intent(cast::Intent::Success)
                        .variant(cast::Variant::Subtle),
                );
                ui.add(
                    cast::Badge::new(format!("Agent: {}", result.agent_id))
                        .variant(cast::Variant::Outline),
                );
                if let Some(harness_id) = result.harness_id.as_ref() {
                    ui.add(
                        cast::Badge::new(format!("Harness: {harness_id}"))
                            .variant(cast::Variant::Outline),
                    );
                }
            });
            if result.result.is_null() {
                ui.add_space(6.0);
                ui.label("Action completed without a result payload.");
            } else {
                ui.add_space(8.0);
                let rendered = serde_json::to_string_pretty(&result.result)
                    .unwrap_or_else(|_| result.result.to_string());
                ui.add(cast::CodeOutputPanel::new("Result", rendered).height(140.0));
            }
        });
    }

    fn render_latest_harness_action_failure(&self, ui: &mut egui::Ui, app: &UiAppRecord) {
        let Some(failure) = self.latest_harness_action_failure.as_ref() else {
            return;
        };
        if !harness_action_failure_matches_app(failure, app) {
            return;
        }

        ui.add_space(10.0);
        cast::Panel::new().show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Latest Action Failure").strong());
                ui.add(
                    cast::Badge::new(failure.action.clone())
                        .intent(cast::Intent::Danger)
                        .variant(cast::Variant::Subtle),
                );
                if let Some(agent_id) = failure.agent_id.as_ref() {
                    ui.add(
                        cast::Badge::new(format!("Agent: {agent_id}"))
                            .variant(cast::Variant::Outline),
                    );
                }
                if let Some(harness_id) = failure.harness_id.as_ref() {
                    ui.add(
                        cast::Badge::new(format!("Harness: {harness_id}"))
                            .variant(cast::Variant::Outline),
                    );
                }
            });
            ui.add_space(8.0);
            ui.add(cast::CodeOutputPanel::new("Error", failure.message.clone()).height(120.0));
        });
    }

    fn render_pending_harness_ui_action(&mut self, ui: &mut egui::Ui, app_id: &str) {
        let Some(pending) = self
            .pending_harness_ui_action
            .as_ref()
            .filter(|pending| pending.app_id == app_id)
        else {
            return;
        };

        let label = pending.label.clone();
        let title = if label.ends_with(['?', '!', '.']) {
            label.clone()
        } else {
            format!("{label}?")
        };
        let mut open = true;
        let response = cast::ConfirmDialog::new(&mut open, "harness_ui_action_confirmation")
            .title(title)
            .description("This will update workflow data. Continue with this action?")
            .confirm_label(label)
            .cancel_label("Cancel")
            .intent(cast::Intent::Primary)
            .width(440.0)
            .show(ui.ctx());

        match response {
            Some(cast::ConfirmDialogResponse::Confirmed) => {
                self.confirm_pending_harness_ui_action();
            }
            Some(cast::ConfirmDialogResponse::Cancelled) => {
                self.cancel_pending_harness_ui_action();
            }
            None if !open => {
                self.cancel_pending_harness_ui_action();
            }
            None => {}
        }
    }

    fn handle_harness_ui_event(&mut self, app: &UiAppRecord, event: HarnessUiEvent) {
        match event {
            HarnessUiEvent::OpenScreen(target) => {
                if let Some(target_index) = harness_ui::screen_index_for_target(app, &target) {
                    self.ui_screen_indices.insert(app.id.clone(), target_index);
                    self.ui_active_pane = None;
                } else {
                    self.dashboard.record_error(format!(
                        "Harness UI app '{}' requested unknown screen '{}'",
                        app.id, target
                    ));
                }
            }
            HarnessUiEvent::RunAction {
                label,
                action,
                params,
                confirm,
            } => {
                let pending = PendingHarnessUiAction::new(app, label, action, params);
                if confirm {
                    self.request_harness_ui_action_confirmation(pending);
                } else {
                    self.pending_harness_ui_action = None;
                    self.run_harness_ui_action(pending);
                }
            }
            HarnessUiEvent::FormError(message) => {
                self.dashboard.record_error(message);
            }
        }
    }

    fn render_agents_tab(&mut self, ui: &mut egui::Ui) {
        let agents = self.dashboard.agents().to_vec();
        self.agent_index = clamp_index(self.agent_index, agents.len());
        let selected_agent = agents.get(self.agent_index).cloned();
        let selected_runtime = selected_agent
            .as_ref()
            .and_then(|agent| self.selected_agent_runtime(&agent.id));

        ui.columns(2, |columns| {
            cast::Panel::new().show(&mut columns[0], |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.heading("Agents");
                    ui.add_space(8.0);
                    ui.add(cast::Badge::new(format!("{} agents", agents.len())));
                });
                ui.add_space(8.0);
                ScrollArea::vertical().show(ui, |ui| {
                    let labels = agents
                        .iter()
                        .map(|agent| format!("{} · {} / {}", agent.id, agent.provider, agent.model))
                        .collect::<Vec<_>>();
                    ui.add(
                        cast::NavList::new(&mut self.agent_index, labels).size(cast::Size::Small),
                    );
                });
            });

            cast::Panel::new().show(&mut columns[1], |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.heading("Agent Detail");
                    if let Some(agent) = &selected_agent {
                        ui.add(
                            cast::Badge::new(if agent.enabled { "Enabled" } else { "Disabled" })
                                .intent(if agent.enabled {
                                    cast::Intent::Success
                                } else {
                                    cast::Intent::Warning
                                })
                                .status_dot(),
                        );
                    }
                });
                ui.add_space(8.0);
                if let Some(agent) = selected_agent {
                    detail_kv(ui, "Agent", &agent.id);
                    detail_kv(ui, "Enabled", yes_no(agent.enabled));
                    detail_kv(ui, "Provider", &agent.provider);
                    detail_kv(ui, "Model", &agent.model);
                    detail_kv(ui, "Harness", &agent.harness_ref);

                    if let Some(runtime) = selected_runtime {
                        ui.add_space(8.0);
                        ui.horizontal_wrapped(|ui| {
                            ui.label(RichText::new("Runtime").strong());
                            ui.add(
                                cast::Badge::new(if runtime.running { "Running" } else { "Idle" })
                                    .intent(if runtime.running {
                                        cast::Intent::Success
                                    } else {
                                        cast::Intent::Neutral
                                    })
                                    .status_dot(),
                            );
                        });
                        detail_kv(ui, "Running", yes_no(runtime.running));
                        detail_kv(ui, "Active Tasks", runtime.active_tasks.to_string());
                        detail_kv(ui, "Queued Tasks", runtime.queued_tasks.to_string());
                        detail_kv(ui, "Awaiting Results", runtime.awaiting_results.to_string());
                        detail_kv(
                            ui,
                            "Current Session",
                            runtime
                                .current_session_id
                                .unwrap_or_else(|| "None".to_string()),
                        );
                    }

                    ui.add_space(12.0);
                    if ui.add(cast::Button::new("Open Live Session")).clicked() {
                        self.send_command(OperatorCommand::OpenSession {
                            agent_id: agent.id.clone(),
                        });
                        self.tab = TabKind::LiveSessions;
                    }
                } else {
                    ui.label("No agents are currently registered.");
                }
            });
        });
    }

    fn render_live_sessions_tab(&mut self, ui: &mut egui::Ui) {
        let live_sessions = self.dashboard.live_sessions.clone();
        self.live_session_index = clamp_index(self.live_session_index, live_sessions.len());
        let selected = live_sessions.get(self.live_session_index).cloned();
        let selected_detail = self.selected_session_detail().cloned();

        ui.columns(2, |columns| {
            cast::Panel::new().show(&mut columns[0], |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.heading("Live Sessions");
                    ui.add_space(8.0);
                    ui.add(cast::Badge::new(format!("{} live", live_sessions.len())));
                });
                ui.add_space(8.0);
                ScrollArea::vertical().show(ui, |ui| {
                    let labels = live_sessions
                        .iter()
                        .map(|session| {
                            format!(
                                "{} · {} · slot {}",
                                session.session_id, session.agent_id, session.slot_id
                            )
                        })
                        .collect::<Vec<_>>();
                    ui.add(
                        cast::NavList::new(&mut self.live_session_index, labels)
                            .size(cast::Size::Small),
                    );
                });
            });

            cast::Panel::new().show(&mut columns[1], |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.heading("Live Session Detail");
                    if let Some(session) = &selected {
                        ui.add(
                            cast::Badge::new(if session.running {
                                "Running"
                            } else {
                                "Stopped"
                            })
                            .intent(if session.running {
                                cast::Intent::Success
                            } else {
                                cast::Intent::Warning
                            })
                            .status_dot(),
                        );
                    }
                });
                ui.add_space(8.0);
                if let Some(session) = selected {
                    detail_kv(ui, "Session", &session.session_id);
                    detail_kv(ui, "Agent", &session.agent_id);
                    detail_kv(ui, "Slot", &session.slot_id);
                    detail_kv(ui, "Running", yes_no(session.running));
                    detail_kv(ui, "Active Tasks", session.active_tasks.to_string());
                    detail_kv(ui, "Queued Tasks", session.queued_tasks.to_string());
                    detail_kv(
                        ui,
                        "Current Request",
                        session
                            .current_request_id
                            .clone()
                            .unwrap_or_else(|| "None".to_string()),
                    );

                    ui.add_space(12.0);
                    ui.label(RichText::new("Prompt Composer").strong());
                    let composer = cast::AgentComposer::new(&mut self.prompt_input)
                        .placeholder("Write a prompt for the selected live session")
                        .send_label("Submit Prompt")
                        .tool_label("Tools")
                        .rows(8)
                        .enabled(session.running)
                        .show(ui)
                        .inner;
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if composer.submitted && !self.prompt_input.trim().is_empty() {
                            let prompt = mem::take(&mut self.prompt_input);
                            self.send_command(OperatorCommand::SubmitPrompt {
                                session_id: session.session_id.clone(),
                                prompt,
                            });
                        }
                        if ui
                            .add(
                                cast::Button::new("Clear")
                                    .size(cast::Size::Small)
                                    .variant(cast::Variant::Ghost),
                            )
                            .clicked()
                        {
                            self.prompt_input.clear();
                        }
                    });

                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui
                            .add(
                                cast::Button::new("Cancel Session")
                                    .intent(cast::Intent::Warning)
                                    .variant(cast::Variant::Outline),
                            )
                            .clicked()
                        {
                            self.send_command(OperatorCommand::CancelSession {
                                session_id: session.session_id.clone(),
                            });
                        }
                        if ui
                            .add(
                                cast::Button::new("Kill Session")
                                    .intent(cast::Intent::Danger)
                                    .variant(cast::Variant::Outline),
                            )
                            .clicked()
                        {
                            self.send_command(OperatorCommand::KillSession {
                                session_id: session.session_id.clone(),
                            });
                        }
                    });

                    ui.add_space(12.0);
                    self.render_session_branch_controls(
                        ui,
                        &session.session_id,
                        selected_detail.as_ref(),
                    );

                    ui.add_space(12.0);
                    render_session_detail_panel(ui, selected_detail.as_ref());
                } else {
                    ui.label("No live sessions are running right now.");
                }
            });
        });
    }

    fn render_sessions_tab(&mut self, ui: &mut egui::Ui) {
        let sessions = self.dashboard.sessions.clone();
        self.session_index = clamp_index(self.session_index, sessions.len());
        let selected = sessions.get(self.session_index).cloned();
        let selected_detail = self.selected_session_detail().cloned();

        ui.columns(2, |columns| {
            cast::Panel::new().show(&mut columns[0], |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.heading("Stored Sessions");
                    ui.add_space(8.0);
                    ui.add(cast::Badge::new(format!("{} stored", sessions.len())));
                });
                ui.add_space(8.0);
                ScrollArea::vertical().show(ui, |ui| {
                    let labels = sessions
                        .iter()
                        .map(|session| {
                            format!(
                                "{} · {}",
                                session.session_id,
                                truncate_for_list(&session.created_at, 22)
                            )
                        })
                        .collect::<Vec<_>>();
                    ui.add(
                        cast::NavList::new(&mut self.session_index, labels).size(cast::Size::Small),
                    );
                });
            });

            cast::Panel::new().show(&mut columns[1], |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.heading("Stored Session Detail");
                    if let Some(session) = &selected {
                        ui.add(
                            cast::Badge::new(session.agent_id.clone())
                                .intent(cast::Intent::Info)
                                .variant(cast::Variant::Outline),
                        );
                    }
                });
                ui.add_space(8.0);
                if let Some(session) = selected {
                    detail_kv(ui, "Session", &session.session_id);
                    detail_kv(ui, "Agent", &session.agent_id);
                    detail_kv(ui, "Created", &session.created_at);
                    detail_kv(ui, "Internal ID", session.internal_id.to_string());
                    ui.add_space(8.0);
                    ui.label(RichText::new("Metadata").strong());
                    ui.code(
                        session
                            .metadata
                            .as_ref()
                            .and_then(|value| serde_json::to_string_pretty(value).ok())
                            .unwrap_or_else(|| "null".to_string()),
                    );
                    ui.add_space(12.0);
                    if ui
                        .add(cast::Button::new("Resume Into Live Session"))
                        .clicked()
                    {
                        self.send_command(OperatorCommand::ResumeSession {
                            session_id: session.session_id.clone(),
                        });
                        self.tab = TabKind::LiveSessions;
                    }

                    ui.add_space(12.0);
                    self.render_session_branch_controls(
                        ui,
                        &session.session_id,
                        selected_detail.as_ref(),
                    );

                    ui.add_space(12.0);
                    render_session_detail_panel(ui, selected_detail.as_ref());
                } else {
                    ui.label("No persisted sessions found.");
                }
            });
        });
    }

    fn render_tasks_tab(&mut self, ui: &mut egui::Ui) {
        let tasks = self.filtered_tasks();
        self.task_index = clamp_index(self.task_index, tasks.len());
        let selected = tasks.get(self.task_index).cloned();

        ui.columns(2, |columns| {
            cast::Panel::new().show(&mut columns[0], |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.heading("Tasks");
                    ui.add_space(8.0);
                    ui.add(cast::Badge::new(format!("{} visible", tasks.len())));
                });
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.add(
                        cast::SearchInput::new(&mut self.task_filter)
                            .hint_text("request id, agent, or state")
                            .size(cast::Size::Small),
                    );
                    if ui
                        .add(
                            cast::Button::new("Clear")
                                .size(cast::Size::Small)
                                .variant(cast::Variant::Ghost),
                        )
                        .clicked()
                    {
                        self.task_filter.clear();
                    }
                });
                ui.add_space(8.0);
                ScrollArea::vertical().show(ui, |ui| {
                    let labels = tasks
                        .iter()
                        .map(|task| {
                            format!(
                                "{} · {} · {}",
                                truncate_for_list(&task.request_id, 18),
                                task.agent_id,
                                task.state
                            )
                        })
                        .collect::<Vec<_>>();
                    ui.add(
                        cast::NavList::new(&mut self.task_index, labels).size(cast::Size::Small),
                    );
                });
            });

            cast::Panel::new().show(&mut columns[1], |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.heading("Task Detail");
                    if let Some(task) = &selected {
                        ui.add(
                            cast::Badge::new(task.state.clone())
                                .intent(status_intent(&task.state))
                                .status_dot(),
                        );
                    }
                });
                ui.add_space(8.0);
                if let Some(task) = selected {
                    detail_kv(ui, "Request", &task.request_id);
                    detail_kv(ui, "Agent", &task.agent_id);
                    detail_kv(ui, "Slot", &task.slot_id);
                    detail_kv(ui, "Trace", &task.trace_id);
                    detail_kv(ui, "State", &task.state);
                    detail_kv(
                        ui,
                        "Runtime Task",
                        task.runtime_task_id
                            .clone()
                            .unwrap_or_else(|| "None".to_string()),
                    );
                    detail_kv(
                        ui,
                        "Status",
                        task.status.clone().unwrap_or_else(|| "None".to_string()),
                    );
                    detail_kv(
                        ui,
                        "Turns",
                        task.task_turn_count
                            .map(|value| value.to_string())
                            .unwrap_or_else(|| "None".to_string()),
                    );
                    if let Some(output) = &task.output {
                        ui.add_space(8.0);
                        ui.add(cast::CodeOutputPanel::new("Output", output).height(180.0));
                    }
                    if let Some(error) = &task.error {
                        ui.add_space(8.0);
                        ui.add(
                            cast::CodeOutputPanel::new("Error", error)
                                .kind(cast::ToolOutputKind::Error)
                                .height(160.0),
                        );
                    }
                    ui.add_space(12.0);
                    if ui
                        .add(
                            cast::Button::new("Cancel Task")
                                .intent(cast::Intent::Danger)
                                .variant(cast::Variant::Outline),
                        )
                        .clicked()
                    {
                        self.send_command(OperatorCommand::CancelTask {
                            request_id: task.request_id.clone(),
                        });
                    }
                } else {
                    ui.label("No tasks are currently tracked.");
                }
            });
        });
    }

    fn render_channels_tab(&mut self, ui: &mut egui::Ui) {
        let channels = self.filtered_channels();
        self.channel_index = clamp_index(self.channel_index, channels.len());
        let selected = channels.get(self.channel_index).cloned();
        let selected_runtime = selected
            .as_ref()
            .and_then(|channel| self.selected_channel_runtime(&channel.id));

        ui.columns(2, |columns| {
            cast::Panel::new().show(&mut columns[0], |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.heading("Channels");
                    ui.add_space(8.0);
                    ui.add(cast::Badge::new(format!("{} visible", channels.len())));
                });
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.add(
                        cast::SearchInput::new(&mut self.channel_filter)
                            .hint_text("channel id, kind, or agent")
                            .size(cast::Size::Small),
                    );
                    if ui
                        .add(
                            cast::Button::new("Clear")
                                .size(cast::Size::Small)
                                .variant(cast::Variant::Ghost),
                        )
                        .clicked()
                    {
                        self.channel_filter.clear();
                    }
                });
                ui.add_space(8.0);
                ScrollArea::vertical().show(ui, |ui| {
                    let labels = channels
                        .iter()
                        .map(|channel| {
                            format!("{} · {} -> {}", channel.id, channel.kind, channel.agent_id)
                        })
                        .collect::<Vec<_>>();
                    ui.add(
                        cast::NavList::new(&mut self.channel_index, labels).size(cast::Size::Small),
                    );
                });
            });

            cast::Panel::new().show(&mut columns[1], |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.heading("Channel Detail");
                    if let Some(channel) = &selected {
                        ui.add(
                            cast::Badge::new(if channel.enabled {
                                "Enabled"
                            } else {
                                "Disabled"
                            })
                            .intent(if channel.enabled {
                                cast::Intent::Success
                            } else {
                                cast::Intent::Warning
                            })
                            .status_dot(),
                        );
                    }
                });
                ui.add_space(8.0);
                if let Some(channel) = selected {
                    detail_kv(ui, "Channel", &channel.id);
                    detail_kv(ui, "Kind", &channel.kind);
                    detail_kv(ui, "Agent", &channel.agent_id);
                    detail_kv(ui, "Enabled", yes_no(channel.enabled));

                    if let Some(runtime) = selected_runtime {
                        ui.add_space(8.0);
                        ui.horizontal_wrapped(|ui| {
                            ui.label(RichText::new("Runtime").strong());
                            ui.add(
                                cast::Badge::new(runtime.state.clone())
                                    .intent(status_intent(&runtime.state))
                                    .status_dot(),
                            );
                        });
                        detail_kv(ui, "State", &runtime.state);
                        detail_kv(ui, "Start Count", runtime.start_count.to_string());
                        detail_kv(ui, "Restart Count", runtime.restart_count.to_string());
                        detail_kv(ui, "Failure Count", runtime.failure_count.to_string());
                        detail_kv(
                            ui,
                            "Last Error Code",
                            runtime
                                .last_error_code
                                .clone()
                                .unwrap_or_else(|| "None".to_string()),
                        );
                        detail_kv(
                            ui,
                            "Last Error",
                            runtime.last_error.unwrap_or_else(|| "None".to_string()),
                        );
                    }
                } else {
                    ui.label("No channels are configured.");
                }
            });
        });
    }

    fn render_events_tab(&mut self, ui: &mut egui::Ui) {
        let events = self.filtered_events();
        self.event_index = clamp_index(self.event_index, events.len());
        let selected = events.get(self.event_index).cloned();

        ui.columns(2, |columns| {
            cast::Panel::new().show(&mut columns[0], |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.heading("Recent Events");
                    ui.add_space(8.0);
                    ui.add(cast::Badge::new(format!("{} visible", events.len())));
                });
                ui.add_space(8.0);
                ui.horizontal_wrapped(|ui| {
                    ui.add(
                        cast::SearchInput::new(&mut self.event_filter)
                            .hint_text("event name or payload text")
                            .size(cast::Size::Small),
                    );
                    if ui
                        .add(
                            cast::Button::new("Clear")
                                .size(cast::Size::Small)
                                .variant(cast::Variant::Ghost),
                        )
                        .clicked()
                    {
                        self.event_filter.clear();
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    let mut paused = self.events_paused;
                    if ui
                        .add(cast::Checkbox::new(&mut paused, "Pause").size(cast::Size::Small))
                        .changed()
                    {
                        self.set_events_paused(paused);
                    }
                    ui.add(
                        cast::Checkbox::new(&mut self.events_follow_latest, "Follow Latest")
                            .size(cast::Size::Small),
                    );
                    if ui
                        .add(
                            cast::Button::new("Jump Latest")
                                .size(cast::Size::Small)
                                .variant(cast::Variant::Outline),
                        )
                        .clicked()
                    {
                        self.event_index = 0;
                    }
                });
                ui.add_space(8.0);
                ScrollArea::vertical().show(ui, |ui| {
                    let labels = events
                        .iter()
                        .map(|event| {
                            format!(
                                "{} · {}",
                                event.event,
                                truncate_for_list(
                                    &serde_json::to_string(&event.data)
                                        .unwrap_or_else(|_| "{}".to_string()),
                                    40,
                                )
                            )
                        })
                        .collect::<Vec<_>>();
                    ui.add(
                        cast::NavList::new(&mut self.event_index, labels).size(cast::Size::Small),
                    );
                });
            });

            cast::Panel::new().show(&mut columns[1], |ui| {
                ui.horizontal_wrapped(|ui| {
                    ui.heading("Event Detail");
                    ui.add(
                        cast::Badge::new(if self.events_paused {
                            "Paused snapshot"
                        } else {
                            "Live stream"
                        })
                        .intent(if self.events_paused {
                            cast::Intent::Warning
                        } else {
                            cast::Intent::Success
                        })
                        .status_dot(),
                    );
                });
                ui.add_space(8.0);
                detail_kv(
                    ui,
                    "Event Source",
                    if self.events_paused {
                        "paused snapshot".to_string()
                    } else {
                        "live stream".to_string()
                    },
                );
                detail_kv(ui, "Follow Latest", yes_no(self.events_follow_latest));
                detail_kv(ui, "Visible Events", events.len().to_string());
                if let Some(event) = selected {
                    detail_kv(ui, "Event", &event.event);
                    ui.add_space(8.0);
                    ui.add(
                        cast::CodeOutputPanel::new(
                            "Payload",
                            serde_json::to_string_pretty(&event.data)
                                .unwrap_or_else(|_| "{}".to_string()),
                        )
                        .kind(cast::ToolOutputKind::Json)
                        .height(360.0),
                    );
                } else {
                    ui.label("No events have been observed yet.");
                }
            });
        });
    }

    fn render_session_branch_controls(
        &mut self,
        ui: &mut egui::Ui,
        session_id: &str,
        detail: Option<&SessionDetail>,
    ) {
        cast::Panel::new().show(ui, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.label(RichText::new("Branches").strong());
                if let Some(detail) = detail {
                    ui.add(cast::Badge::new(format!(
                        "{} branches",
                        detail.branches.len()
                    )));
                }
            });
            ui.add_space(8.0);

            let Some(detail) = detail else {
                ui.label("Loading branch detail...");
                return;
            };

            let active_branch = detail.branches.iter().find(|branch| branch.active);
            detail_kv(
                ui,
                "Active Branch",
                active_branch
                    .map(branch_descriptor)
                    .unwrap_or_else(|| "main".to_string()),
            );

            ui.add_space(6.0);
            cast::Table::new(["Branch", "Created", "State", "Action"])
                .column_weights([2.4, 1.4, 0.8, 0.9])
                .size(cast::Size::Small)
                .show(ui, detail.branches.len(), |row, index| {
                    let branch = &detail.branches[index];
                    row.text(branch_descriptor(branch));
                    row.text(truncate_for_list(&branch.created_at, 22));
                    row.cell(|ui| {
                        if branch.active {
                            ui.add(
                                cast::Badge::new("Active")
                                    .intent(cast::Intent::Success)
                                    .status_dot(),
                            );
                        } else {
                            ui.add(cast::Badge::new("Available"));
                        }
                    });
                    row.cell(|ui| {
                        if !branch.active
                            && ui
                                .add(
                                    cast::Button::new("Checkout")
                                        .size(cast::Size::Small)
                                        .variant(cast::Variant::Outline),
                                )
                                .clicked()
                        {
                            self.send_command(OperatorCommand::CheckoutSessionBranch {
                                session_id: session_id.to_string(),
                                branch: branch.branch_id.clone(),
                            });
                        }
                    });
                });

            ui.add_space(8.0);
            ui.add(
                cast::TextInput::new(&mut self.branch_name_input)
                    .label("New Branch")
                    .hint_text("branch name"),
            );
            ui.add(
                cast::Checkbox::new(
                    &mut self.activate_new_branch,
                    "Activate immediately after create",
                )
                .size(cast::Size::Small),
            );
            let can_create = !self.branch_name_input.trim().is_empty();
            if ui
                .add(
                    cast::Button::new("Create Branch")
                        .enabled(can_create)
                        .variant(cast::Variant::Outline),
                )
                .clicked()
            {
                let name = self.branch_name_input.trim().to_string();
                self.branch_name_input.clear();
                self.send_command(OperatorCommand::CreateSessionBranch {
                    session_id: session_id.to_string(),
                    name,
                    from_turn_index: None,
                    activate: self.activate_new_branch,
                });
            }
        });
    }
}

impl eframe::App for TurinDesktopApp {
    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        while let Ok(update) = self.controller.update_rx.try_recv() {
            self.apply_update(update);
        }

        self.ensure_session_detail_loaded();

        ctx.request_repaint_after(Duration::from_millis(250));
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.sync_theme(ui.ctx());
        paint_app_canvas(ui);
        if self.dashboard.ui.apps().next().is_some() {
            self.render_operator_shell(ui);
        } else {
            self.render_default_shell(ui);
        }
    }
}

fn paint_app_canvas(ui: &mut egui::Ui) {
    let theme = cast::theme_for_ui(ui);
    ui.visuals_mut().override_text_color = Some(theme.colors.text);
    ui.painter().rect_filled(
        ui.max_rect(),
        egui::CornerRadius::ZERO,
        theme.colors.background,
    );
}

fn default_operator_metric_groups(
    summary: &DefaultOperatorConsoleSummary,
) -> Vec<DefaultOperatorMetricGroup> {
    vec![
        DefaultOperatorMetricGroup {
            title: "Runtime",
            metrics: vec![
                ("Agents", summary.agents),
                ("Harnesses", summary.harnesses),
                ("Channels", summary.channels),
            ],
        },
        DefaultOperatorMetricGroup {
            title: "Work",
            metrics: vec![
                ("Live Sessions", summary.live_sessions),
                ("Stored Sessions", summary.stored_sessions),
                ("Tasks", summary.tasks),
            ],
        },
    ]
}

fn ui_notice_intent(level: Option<UiNoticeLevel>) -> cast::Intent {
    match level {
        Some(UiNoticeLevel::Success) => cast::Intent::Success,
        Some(UiNoticeLevel::Warning) => cast::Intent::Warning,
        Some(UiNoticeLevel::Error) => cast::Intent::Danger,
        Some(UiNoticeLevel::Info) | None => cast::Intent::Info,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turin_daemon_protocol::{UiIntentSource, UiListNode, UiNode, UiPaneIntent, UiScreenIntent};

    #[test]
    fn visible_ui_list_requests_include_active_screen_and_pane_only() {
        let app = test_app();

        let requests = visible_ui_list_requests(&app, 0, Some("notes"));

        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].source, "worklists.home");
        assert_eq!(requests[0].limit, Some(3));
        assert_eq!(requests[1].source, "worklists.notes");
        assert_eq!(requests[1].limit, Some(5));
    }

    #[test]
    fn visible_ui_list_requests_ignore_inactive_surfaces() {
        let app = test_app();

        let requests = visible_ui_list_requests(&app, 0, Some("missing-pane"));

        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].source, "worklists.home");
    }

    #[test]
    fn visible_ui_list_requests_clamp_screen_index() {
        let app = test_app();

        let requests = visible_ui_list_requests(&app, usize::MAX, None);

        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].source, "worklists.other");
        assert_eq!(requests[0].limit, Some(7));
    }

    #[test]
    fn ui_refresh_requests_include_known_and_visible_matching_bindings() {
        let known = BTreeMap::from([(
            list_request("worklists.release", Some(8)).cache_key(),
            list_request("worklists.release", Some(8)),
        )]);
        let selected = vec![
            list_request("worklists.release", Some(25)),
            list_request("worklists.other", Some(10)),
        ];

        let requests = ui_refresh_requests_for_binding("worklists.release", &known, selected);

        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].source, "worklists.release");
        assert_eq!(requests[0].limit, Some(8));
        assert_eq!(requests[1].source, "worklists.release");
        assert_eq!(requests[1].limit, Some(25));
    }

    #[test]
    fn ui_refresh_requests_dedupe_matching_visible_cache_keys() {
        let request = list_request("worklists.release", Some(8));
        let known = BTreeMap::from([(request.cache_key(), request.clone())]);

        let requests = ui_refresh_requests_for_binding("worklists.release", &known, vec![request]);

        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].source, "worklists.release");
        assert_eq!(requests[0].limit, Some(8));
    }

    #[test]
    fn ui_refresh_requests_ignore_other_bindings() {
        let known = BTreeMap::from([(
            list_request("worklists.release", Some(8)).cache_key(),
            list_request("worklists.release", Some(8)),
        )]);
        let selected = vec![list_request("worklists.release", Some(25))];

        let requests = ui_refresh_requests_for_binding("worklists.qa", &known, selected);

        assert!(requests.is_empty());
    }

    #[test]
    fn default_operator_console_copy_explains_no_harness_path() {
        assert_eq!(DEFAULT_OPERATOR_EMPTY_TITLE, "Turin is ready");
        assert!(DEFAULT_OPERATOR_EMPTY_BODY.contains("No custom app surface"));
        assert!(DEFAULT_OPERATOR_EMPTY_BODY.contains("standard console"));
        assert!(DEFAULT_OPERATOR_INTRO.contains("standard console"));
        assert_eq!(DEFAULT_OPERATOR_GUIDANCE_TITLE, "Custom Apps");
        assert!(DEFAULT_OPERATOR_GUIDANCE_BODY.contains("focused screens"));
        assert!(DEFAULT_OPERATOR_GUIDANCE_BODY.contains("lists, forms, reports"));
    }

    #[test]
    fn default_operator_metric_groups_keep_app_local_counts() {
        let summary = DefaultOperatorConsoleSummary {
            connection: "local".to_string(),
            target: ".turin/config.toml".to_string(),
            freshness: "fresh".to_string(),
            agents: 2,
            harnesses: 3,
            channels: 4,
            live_sessions: 5,
            stored_sessions: 6,
            tasks: 7,
            ui_notices: 8,
            ui_requests: 9,
        };

        let groups = default_operator_metric_groups(&summary);

        assert_eq!(
            groups,
            vec![
                DefaultOperatorMetricGroup {
                    title: "Runtime",
                    metrics: vec![("Agents", 2), ("Harnesses", 3), ("Channels", 4)],
                },
                DefaultOperatorMetricGroup {
                    title: "Work",
                    metrics: vec![("Live Sessions", 5), ("Stored Sessions", 6), ("Tasks", 7)],
                },
            ]
        );
    }

    #[test]
    fn harness_action_failure_matching_filters_other_apps() {
        let app = test_app();
        let matching = HarnessActionFailure {
            action: "release.fail_diagnostic".to_string(),
            agent_id: Some("release-agent".to_string()),
            harness_id: Some("release-harness".to_string()),
            message: "Release Operator diagnostic failure".to_string(),
        };
        let other_harness = HarnessActionFailure {
            harness_id: Some("qa-harness".to_string()),
            ..matching.clone()
        };
        let other_agent = HarnessActionFailure {
            agent_id: Some("qa-agent".to_string()),
            ..matching.clone()
        };

        assert!(harness_action_failure_matches_app(&matching, &app));
        assert!(!harness_action_failure_matches_app(&other_harness, &app));
        assert!(!harness_action_failure_matches_app(&other_agent, &app));
    }

    #[test]
    fn harness_action_failure_without_agent_matches_selected_harness() {
        let app = test_app();
        let failure = HarnessActionFailure {
            action: "release.fail_diagnostic".to_string(),
            agent_id: None,
            harness_id: Some("release-harness".to_string()),
            message: "Release Operator diagnostic failure".to_string(),
        };

        assert!(harness_action_failure_matches_app(&failure, &app));
    }

    fn test_app() -> UiAppRecord {
        UiAppRecord {
            id: "release".to_string(),
            source: UiIntentSource {
                harness_id: Some("release-harness".to_string()),
                app_id: Some("release".to_string()),
                agent_id: Some("release-agent".to_string()),
                package_id: None,
            },
            definition: None,
            screens: BTreeMap::from([
                (
                    "home".to_string(),
                    UiScreenIntent {
                        app_id: "release".to_string(),
                        id: "home".to_string(),
                        title: "Home".to_string(),
                        presentation: None,
                        nodes: vec![UiNode::List(list_node("home-list", "worklists.home", 3))],
                    },
                ),
                (
                    "other".to_string(),
                    UiScreenIntent {
                        app_id: "release".to_string(),
                        id: "other".to_string(),
                        title: "Other".to_string(),
                        presentation: None,
                        nodes: vec![UiNode::List(list_node("other-list", "worklists.other", 7))],
                    },
                ),
            ]),
            panes: BTreeMap::from([(
                "notes".to_string(),
                UiPaneIntent {
                    app_id: "release".to_string(),
                    id: "notes".to_string(),
                    title: "Notes".to_string(),
                    presentation: Some("sheet".to_string()),
                    nodes: vec![UiNode::List(list_node("notes-list", "worklists.notes", 5))],
                },
            )]),
            menus: Vec::new(),
            opens_with: None,
            badges: BTreeMap::new(),
        }
    }

    fn list_node(id: &str, source: &str, limit: u32) -> UiListNode {
        UiListNode {
            id: Some(id.to_string()),
            title: id.to_string(),
            source: source.to_string(),
            filter: Default::default(),
            fields: Vec::new(),
            sort: Vec::new(),
            limit: Some(limit),
            intent: Some("items".to_string()),
            render_as: Some("table".to_string()),
        }
    }

    fn list_request(source: &str, limit: Option<u32>) -> UiListRequest {
        UiListRequest {
            source: source.to_string(),
            filter: Default::default(),
            limit,
        }
    }
}
