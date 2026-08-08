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
    SessionDetail, SessionMessageWindow, SessionSummary, TaskStatus,
};
use turin_daemon_protocol::{
    EventEnvelope, HarnessActionRunResult, UiIntent, UiMenuItem, UiNoticeLevel, UiShowIntent,
    WorkItemList,
};
use turin_types::layout::DEFAULT_UI_PROFILES_PATH;
use turin_ui_core::{
    ConnectionDraftHistory, ConnectionOptions, ConnectionPreflightReport,
    ConnectionProfileActivityBook, ConnectionProfileCatalog, ConnectionProfileDraft,
    ConnectionProfileDraftAuthMode, ConnectionProfileDraftDiff, ConnectionProfileDraftValidation,
    ConnectionProfileKind, ConnectionProfileSummary, DashboardState, HarnessActionFailure,
    OperatorCommand, UiAppRecord, UiController, UiListRequest, UiShowTarget, UiUpdate,
    collect_ui_list_requests, connect_dashboard, ensure_local_daemon_for_draft,
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
            .with_min_inner_size(Vec2::new(760.0, 600.0)),
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
    #[cfg(debug_assertions)]
    ctx.global_style_mut(|style| {
        style.debug.show_unaligned = false;
        style.debug.warn_if_rect_changes_id = false;
    });
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
        .with_typography(cast::TypographyTokens::cast())
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
    conversation_message_limits: BTreeMap<String, usize>,
    task_filter: String,
    channel_filter: String,
    event_filter: String,
    events_paused: bool,
    events_follow_latest: bool,
    paused_events: Vec<EventEnvelope>,
    ui_screen_indices: BTreeMap<String, usize>,
    ui_assistant_apps: BTreeSet<String>,
    ui_active_pane: Option<ActiveHarnessPane>,
    ui_open_disclosures: BTreeMap<String, bool>,
    ui_form_values: BTreeMap<String, String>,
    ui_list_filters: BTreeMap<String, String>,
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
    pending_conversation: Option<PendingConversation>,
    pending_resume_session_id: Option<String>,
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct ActiveHarnessPane {
    id: String,
    presentation: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HarnessPanePresentation {
    Sheet,
    Dialog,
}

#[derive(Debug, Clone)]
struct PendingConversation {
    agent_id: String,
    prompt: Option<String>,
    existing_session_ids: BTreeSet<String>,
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
            conversation_message_limits: BTreeMap::new(),
            task_filter: String::new(),
            channel_filter: String::new(),
            event_filter: String::new(),
            events_paused: false,
            events_follow_latest: true,
            paused_events: Vec::new(),
            ui_screen_indices: BTreeMap::new(),
            ui_assistant_apps: BTreeSet::new(),
            ui_active_pane: None,
            ui_open_disclosures: BTreeMap::new(),
            ui_form_values: BTreeMap::new(),
            ui_list_filters: BTreeMap::new(),
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
            pending_conversation: None,
            pending_resume_session_id: None,
            _runtime: runtime,
        }
    }

    fn apply_update(&mut self, update: UiUpdate) {
        let snapshot_updated = matches!(&update, UiUpdate::Snapshot(_));
        let command_failed = matches!(&update, UiUpdate::Error(_));
        let auto_follow_event = matches!(&update, UiUpdate::Event(_))
            && !self.events_paused
            && self.events_follow_latest;
        let harness_action_ran =
            matches!(&update, UiUpdate::Event(event) if event.event == "harness.action_ran");
        if let UiUpdate::SessionEvent(event) = &update {
            if session_event_changes_conversation(event)
                && let Some(session_id) = self.current_detail_session_id()
            {
                self.request_conversation_detail(session_id);
            }
            self.clamp_selection_indices();
            return;
        }
        if let UiUpdate::SessionDetail(detail) = &update
            && self.requested_session_detail.as_deref() == Some(detail.session.session_id.as_str())
        {
            self.requested_session_detail = None;
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
        if command_failed {
            self.pending_conversation = None;
            self.pending_resume_session_id = None;
        }
        if snapshot_updated {
            self.requested_session_detail = None;
            self.complete_pending_conversation();
            self.complete_pending_conversation_resume();
        }
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
        if self.operator_assistant_is_open(&app) {
            return self
                .ui_active_pane
                .as_ref()
                .and_then(|active| app.panes.get(&active.id))
                .map(|pane| collect_ui_list_requests(&pane.nodes))
                .unwrap_or_default();
        }
        let screen_index = self
            .ui_screen_indices
            .get(&app.id)
            .copied()
            .unwrap_or_else(|| harness_ui::default_screen_index(&app));
        visible_ui_list_requests(
            &app,
            screen_index,
            self.ui_active_pane.as_ref().map(|pane| pane.id.as_str()),
        )
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
            self.apply_ui_show_request(&show);
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

    fn apply_ui_show_request(&mut self, show: &UiShowIntent) {
        let Some(app) = self.select_ui_app_by_id(&show.app_id) else {
            return;
        };
        match ui_show_target_for(&app, &show.target) {
            Some(UiShowTarget::Screen { screen_index }) => {
                self.open_harness_screen(&app, screen_index);
            }
            Some(UiShowTarget::Pane { pane_id }) => {
                self.tab = TabKind::UiApps;
                self.ui_active_pane = Some(ActiveHarnessPane {
                    id: pane_id.to_string(),
                    presentation: show.presentation.clone(),
                });
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
        self.ui_assistant_apps.remove(&app.id);
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

    fn operator_assistant_is_open(&self, app: &UiAppRecord) -> bool {
        self.ui_assistant_apps.contains(&app.id)
    }

    fn operator_agent_id<'a>(&'a self, app: Option<&'a UiAppRecord>) -> Option<&'a str> {
        app.and_then(|app| app.source.agent_id.as_deref())
            .or_else(|| {
                self.dashboard
                    .agents()
                    .get(self.agent_index)
                    .map(|agent| agent.id.as_str())
            })
    }

    fn selected_operator_session(&self, app: Option<&UiAppRecord>) -> Option<LiveSession> {
        let session = self.selected_live_session()?;
        if app.is_none() {
            return Some(session);
        }
        self.operator_agent_id(app)
            .is_none_or(|agent_id| session.agent_id == agent_id)
            .then_some(session)
    }

    fn open_operator_assistant(&mut self, app: &UiAppRecord) {
        self.tab = TabKind::UiApps;
        self.ui_assistant_apps.insert(app.id.clone());
        self.ui_active_pane = None;
        self.requested_session_detail = None;

        let target_agent = app.source.agent_id.as_deref();
        if let Some(agent_id) = target_agent
            && let Some(index) = self
                .dashboard
                .agents()
                .iter()
                .position(|agent| agent.id == agent_id)
        {
            self.agent_index = index;
        }

        let selected_matches = self.selected_live_session().is_some_and(|session| {
            target_agent.is_none_or(|agent_id| session.agent_id == agent_id)
        });
        if selected_matches {
            return;
        }

        if let Some((index, session)) = self
            .dashboard
            .live_sessions
            .iter()
            .enumerate()
            .find(|(_, session)| target_agent.is_none_or(|agent_id| session.agent_id == agent_id))
            .map(|(index, session)| (index, session.clone()))
        {
            self.live_session_index = index;
            self.send_command(OperatorCommand::FocusSessionStream {
                session_id: Some(session.session_id),
            });
        } else {
            self.live_session_index = usize::MAX;
            self.send_command(OperatorCommand::FocusSessionStream { session_id: None });
        }
    }

    fn start_default_conversation(&mut self, prompt: Option<String>) {
        let Some(agent_id) = self
            .dashboard
            .agents()
            .get(self.agent_index)
            .map(|agent| agent.id.clone())
        else {
            self.dashboard
                .record_error("No enabled agent is available for a new conversation");
            return;
        };
        let existing_session_ids = self
            .dashboard
            .live_sessions
            .iter()
            .map(|session| session.session_id.clone())
            .collect();
        self.pending_conversation = Some(PendingConversation {
            agent_id: agent_id.clone(),
            prompt,
            existing_session_ids,
        });
        self.pending_resume_session_id = None;
        self.live_session_index = usize::MAX;
        self.send_command(OperatorCommand::OpenSession { agent_id });
    }

    fn complete_pending_conversation(&mut self) {
        let Some(pending) = self.pending_conversation.as_ref() else {
            return;
        };
        let Some(index) =
            pending_conversation_session_index(pending, &self.dashboard.live_sessions)
        else {
            return;
        };
        let session = self.dashboard.live_sessions[index].clone();

        let prompt = self
            .pending_conversation
            .take()
            .and_then(|pending| pending.prompt);
        self.live_session_index = index;
        self.requested_session_detail = None;
        self.send_command(OperatorCommand::FocusSessionStream {
            session_id: Some(session.session_id.clone()),
        });
        if let Some(prompt) = prompt.filter(|prompt| !prompt.trim().is_empty()) {
            if let Some(title) = default_conversation_title_from_prompt(&prompt) {
                self.send_command(OperatorCommand::SetSessionTitle {
                    session_id: session.session_id.clone(),
                    title: Some(title),
                });
            }
            self.send_command(OperatorCommand::SubmitPrompt {
                session_id: session.session_id,
                prompt,
            });
        }
    }

    fn resume_default_conversation(&mut self, session_id: String) {
        if let Some(index) = self
            .dashboard
            .live_sessions
            .iter()
            .position(|session| session.session_id == session_id)
        {
            self.live_session_index = index;
            let live_sessions = self.dashboard.live_sessions.clone();
            self.focus_default_session(&live_sessions);
            return;
        }
        if self.pending_resume_session_id.as_deref() == Some(session_id.as_str()) {
            return;
        }
        self.pending_conversation = None;
        self.pending_resume_session_id = Some(session_id.clone());
        self.send_command(OperatorCommand::ResumeSession { session_id });
    }

    fn complete_pending_conversation_resume(&mut self) {
        let Some(session_id) = self.pending_resume_session_id.as_deref() else {
            return;
        };
        let Some(index) = self
            .dashboard
            .live_sessions
            .iter()
            .position(|session| session.session_id == session_id)
        else {
            return;
        };
        let session_id = self
            .pending_resume_session_id
            .take()
            .expect("pending resume id was checked above");
        self.live_session_index = index;
        self.requested_session_detail = None;
        self.send_command(OperatorCommand::FocusSessionStream {
            session_id: Some(session_id),
        });
    }

    fn default_session_needs_title(&self, session_id: &str) -> bool {
        let already_titled = self
            .dashboard
            .sessions
            .iter()
            .find(|session| session.session_id == session_id)
            .and_then(session_summary_title)
            .is_some();
        !already_titled
            && self
                .dashboard
                .session_detail(session_id)
                .is_some_and(|detail| detail.messages.is_empty())
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
            cast::ThemeMode::Light => "Dark",
            cast::ThemeMode::Dark => "Light",
        }
    }

    fn current_detail_session_id(&self) -> Option<String> {
        let selected_app = self.selected_ui_app();
        if selected_app.is_none()
            || selected_app
                .as_ref()
                .is_some_and(|app| self.operator_assistant_is_open(app))
        {
            return self
                .selected_operator_session(selected_app.as_ref())
                .map(|session| session.session_id);
        }
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
            return;
        }

        self.request_conversation_detail(session_id);
    }

    fn request_conversation_detail(&mut self, session_id: String) {
        if self.requested_session_detail.as_deref() == Some(session_id.as_str()) {
            return;
        }

        let message_limit = self
            .conversation_message_limits
            .get(&session_id)
            .copied()
            .unwrap_or(DEFAULT_CONVERSATION_MESSAGE_LIMIT);
        self.requested_session_detail = Some(session_id.clone());
        self.send_command(OperatorCommand::LoadSessionDetail {
            session_id,
            message_limit: Some(message_limit),
        });
    }

    fn load_earlier_conversation_messages(&mut self, session_id: &str, total: usize) {
        let current = self
            .conversation_message_limits
            .get(session_id)
            .copied()
            .unwrap_or(DEFAULT_CONVERSATION_MESSAGE_LIMIT);
        let next = current
            .saturating_add(CONVERSATION_MESSAGE_PAGE_SIZE)
            .min(total);
        if next <= current {
            return;
        }
        self.conversation_message_limits
            .insert(session_id.to_string(), next);
        self.request_conversation_detail(session_id.to_string());
    }

    fn render_default_shell(&mut self, ui: &mut egui::Ui) {
        let compact = operator_shell_is_compact(ui.available_width());
        let live_sessions = self.dashboard.live_sessions.clone();
        let recent_sessions = recent_default_conversations(
            &live_sessions,
            &self.dashboard.sessions,
            DEFAULT_RECENT_CONVERSATION_LIMIT,
        );
        self.live_session_index = clamp_index(self.live_session_index, live_sessions.len());
        let selected = live_sessions.get(self.live_session_index).cloned();

        let top_frame = {
            let theme = cast::theme_for_ui(ui);
            egui::Frame::new()
                .fill(theme.colors.surface)
                .stroke(egui::Stroke::new(theme.stroke.sm, theme.colors.border))
                .inner_margin(egui::Margin::symmetric(20, 11))
        };
        egui::Panel::top("default_shell_top")
            .exact_size(56.0)
            .frame(top_frame)
            .show(ui, |ui| {
                ui.horizontal_wrapped(|ui| {
                    themed_heading(ui, "Turin", 20.0);
                    self.render_connection_status_inline(ui);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        self.render_theme_controls(ui);
                        if ui
                            .add(
                                cast::Button::new("Runtime Tools")
                                    .size(cast::Size::Small)
                                    .variant(cast::Variant::Ghost),
                            )
                            .clicked()
                        {
                            self.runtime_tools_open = true;
                        }
                        if compact
                            && ui
                                .add(
                                    cast::Button::new("New conversation")
                                        .size(cast::Size::Small)
                                        .intent(cast::Intent::Primary)
                                        .enabled(
                                            self.pending_conversation.is_none()
                                                && self.pending_resume_session_id.is_none(),
                                        ),
                                )
                                .clicked()
                        {
                            self.start_default_conversation(None);
                        }
                    });
                });
            });

        if compact {
            let compact_nav_frame = {
                let theme = cast::theme_for_ui(ui);
                egui::Frame::new()
                    .fill(theme.colors.surface)
                    .stroke(egui::Stroke::new(theme.stroke.sm, theme.colors.border))
                    .inner_margin(egui::Margin::symmetric(16, 10))
            };
            egui::Panel::top("default_session_nav_compact")
                .frame(compact_nav_frame)
                .show(ui, |ui| {
                    self.render_default_session_nav(ui, &live_sessions, &recent_sessions, true);
                });
        } else {
            let theme = cast::theme_for_ui(ui);
            egui::Panel::left("default_session_nav")
                .resizable(false)
                .exact_size(OPERATOR_SIDEBAR_WIDTH)
                .frame(
                    egui::Frame::new()
                        .fill(theme.colors.surface)
                        .stroke(egui::Stroke::new(theme.stroke.sm, theme.colors.border))
                        .inner_margin(egui::Margin::symmetric(16, 16)),
                )
                .show(ui, |ui| {
                    self.render_default_session_nav(ui, &live_sessions, &recent_sessions, false);
                });
        }

        if let Some(session) = selected.as_ref() {
            let composer_frame = {
                let theme = cast::theme_for_ui(ui);
                egui::Frame::new()
                    .fill(theme.colors.surface)
                    .stroke(egui::Stroke::new(theme.stroke.sm, theme.colors.border))
                    .inner_margin(egui::Margin::symmetric(20, 12))
            };
            egui::Panel::bottom("default_conversation_composer")
                .frame(composer_frame)
                .show(ui, |ui| {
                    self.render_default_composer(ui, session);
                });
        }

        egui::CentralPanel::default().show(ui, |ui| {
            ScrollArea::vertical().show(ui, |ui| {
                let outer_margin = if compact {
                    OPERATOR_COMPACT_CONTENT_MARGIN
                } else {
                    OPERATOR_CONTENT_MARGIN
                };
                ui.add_space(outer_margin);
                let (content_width, inset) =
                    default_conversation_geometry(ui.available_width(), outer_margin);
                ui.horizontal_top(|ui| {
                    ui.add_space(inset);
                    ui.vertical(|ui| {
                        ui.set_width(content_width);
                        if let Some(session) = selected.as_ref() {
                            self.render_default_conversation(ui, session);
                        } else {
                            self.render_default_welcome(ui);
                        }
                    });
                });
                ui.add_space(outer_margin);
            });
        });

        self.render_runtime_tools_sheet(ui);
    }

    fn render_default_session_nav(
        &mut self,
        ui: &mut egui::Ui,
        live_sessions: &[LiveSession],
        recent_sessions: &[SessionSummary],
        compact: bool,
    ) {
        let agents = self.dashboard.agents().to_vec();
        self.agent_index = clamp_index(self.agent_index, agents.len());

        if compact {
            ui.horizontal_wrapped(|ui| {
                if !live_sessions.is_empty() || !recent_sessions.is_empty() {
                    let mut labels = vec!["Choose a conversation".to_string()];
                    labels.extend(default_conversation_labels(
                        live_sessions,
                        &self.dashboard.sessions,
                    ));
                    labels.extend(
                        recent_sessions
                            .iter()
                            .enumerate()
                            .map(|(index, session)| stored_conversation_title(session, index)),
                    );
                    let current = live_sessions
                        .get(self.live_session_index)
                        .map(|_| self.live_session_index + 1)
                        .unwrap_or_default();
                    let mut selected = current;
                    ui.add(cast::Select::new(&mut selected, labels).width(280.0));
                    if selected != current && selected > 0 {
                        let target = selected - 1;
                        if target < live_sessions.len() {
                            self.live_session_index = target;
                            self.focus_default_session(live_sessions);
                        } else if let Some(session) =
                            recent_sessions.get(target - live_sessions.len())
                        {
                            self.resume_default_conversation(session.session_id.clone());
                        }
                    }
                }
                if agents.len() > 1 {
                    let agent_labels = agents
                        .iter()
                        .map(|agent| default_agent_label(&agent.id))
                        .collect::<Vec<_>>();
                    ui.add(cast::Select::new(&mut self.agent_index, agent_labels).width(180.0));
                }
            });
            return;
        }

        themed_heading(ui, "Conversations", 18.0);
        ui.add_space(10.0);
        if agents.len() > 1 {
            let agent_labels = agents
                .iter()
                .map(|agent| default_agent_label(&agent.id))
                .collect::<Vec<_>>();
            ui.add(
                cast::Select::new(&mut self.agent_index, agent_labels).width(ui.available_width()),
            );
            ui.add_space(10.0);
        }
        if ui
            .add_sized(
                egui::vec2(ui.available_width(), 32.0),
                cast::Button::new("New conversation")
                    .intent(cast::Intent::Primary)
                    .enabled(
                        self.pending_conversation.is_none()
                            && self.pending_resume_session_id.is_none(),
                    ),
            )
            .clicked()
        {
            self.start_default_conversation(None);
        }
        ui.add_space(12.0);

        if live_sessions.is_empty() && recent_sessions.is_empty() {
            themed_muted(ui, "Your conversations will appear here.");
        }
        if !live_sessions.is_empty() {
            themed_overline(ui, "Active");
            ui.add_space(4.0);
            let labels = default_conversation_labels(live_sessions, &self.dashboard.sessions);
            for (index, (session, label)) in live_sessions.iter().zip(labels).enumerate() {
                let mut row = cast::ListRow::new(label)
                    .selected(index == self.live_session_index)
                    .size(cast::Size::Small);
                if agents.len() > 1 {
                    row = row.subtitle(default_agent_label(&session.agent_id));
                }
                if session.active_tasks > 0 {
                    row = row.trailing("Working");
                }
                if ui.add(row).clicked() && index != self.live_session_index {
                    self.live_session_index = index;
                    self.focus_default_session(live_sessions);
                }
            }
        }
        if !recent_sessions.is_empty() {
            if !live_sessions.is_empty() {
                ui.add_space(14.0);
            }
            themed_overline(ui, "Recent");
            ui.add_space(4.0);
            for (index, session) in recent_sessions.iter().enumerate() {
                let pending =
                    self.pending_resume_session_id.as_deref() == Some(session.session_id.as_str());
                let mut row = cast::ListRow::new(stored_conversation_title(session, index))
                    .enabled(!pending)
                    .size(cast::Size::Small);
                if agents.len() > 1 {
                    row = row.subtitle(default_agent_label(&session.agent_id));
                }
                if pending {
                    row = row.trailing("Opening...");
                }
                if ui.add(row).clicked() {
                    self.resume_default_conversation(session.session_id.clone());
                }
            }
        }
    }

    fn focus_default_session(&mut self, live_sessions: &[LiveSession]) {
        let session_id = live_sessions
            .get(self.live_session_index)
            .map(|session| session.session_id.clone());
        self.requested_session_detail = None;
        self.send_command(OperatorCommand::FocusSessionStream { session_id });
    }

    fn render_default_conversation(&mut self, ui: &mut egui::Ui, session: &LiveSession) {
        let title =
            default_conversation_title(session, self.live_session_index, &self.dashboard.sessions);
        ui.horizontal_wrapped(|ui| {
            themed_heading(ui, title, 28.0);
            if session.active_tasks > 0 {
                ui.add(
                    cast::Badge::new("Working")
                        .intent(cast::Intent::Info)
                        .status_dot(),
                );
            }
        });
        ui.add_space(14.0);

        let Some(detail) = self.dashboard.session_detail(&session.session_id) else {
            render_conversation_loading(ui);
            return;
        };
        let requested_limit = self
            .conversation_message_limits
            .get(&session.session_id)
            .copied()
            .unwrap_or(DEFAULT_CONVERSATION_MESSAGE_LIMIT);
        let (message_start, message_window) = match detail.message_window.clone() {
            Some(window) => (0, Some(window)),
            None => {
                let offset = detail.messages.len().saturating_sub(requested_limit);
                (
                    offset,
                    (offset > 0).then_some(SessionMessageWindow {
                        offset,
                        total: detail.messages.len(),
                    }),
                )
            }
        };
        let mut load_earlier = false;
        if let Some(window) = message_window.as_ref()
            && window.offset > 0
        {
            ui.horizontal(|ui| {
                load_earlier = ui
                    .add(
                        cast::Button::new("Load earlier messages")
                            .size(cast::Size::Small)
                            .variant(cast::Variant::Outline)
                            .enabled(
                                self.requested_session_detail.as_deref()
                                    != Some(session.session_id.as_str()),
                            ),
                    )
                    .clicked();
                themed_muted(
                    ui,
                    format!(
                        "Showing the latest {} of {} messages",
                        window.total.saturating_sub(window.offset),
                        window.total
                    ),
                );
            });
            ui.add_space(14.0);
        }
        let rendered_messages = detail.messages[message_start..]
            .iter()
            .filter_map(|message| {
                let body = session_message_text(&message.content);
                let tools = detail
                    .tool_executions
                    .iter()
                    .filter(|tool| tool.turn_index == message.turn_index)
                    .collect::<Vec<_>>();
                default_conversation_message_is_visible(
                    &message.role,
                    !body.is_empty(),
                    !tools.is_empty(),
                )
                .then_some((message, body, tools))
            })
            .collect::<Vec<_>>();
        if rendered_messages.is_empty() && session.active_tasks == 0 {
            cast::EmptyState::new("Ready when you are")
                .body("Ask a question, delegate a task, or describe an outcome below.")
                .icon("✦")
                .intent(cast::Intent::Primary)
                .show(ui, |_| {});
            return;
        }

        cast::MessageThread::new()
            .width(ui.available_width())
            .show(ui, |thread| {
                let last_message_is_assistant = rendered_messages
                    .last()
                    .is_some_and(|(message, _, _)| message.role.eq_ignore_ascii_case("assistant"));
                let message_count = rendered_messages.len();
                for (index, (message, body, tools)) in rendered_messages.into_iter().enumerate() {
                    let is_assistant = message.role.eq_ignore_ascii_case("assistant");
                    let title = if message.role.eq_ignore_ascii_case("user") {
                        "You".to_string()
                    } else if is_assistant {
                        default_agent_label(&session.agent_id)
                    } else {
                        "Tool".to_string()
                    };
                    let chat = cast::ChatMessage::new(chat_role_from_label(&message.role), body)
                        .title(title)
                        .streaming(
                            session.active_tasks > 0 && is_assistant && index + 1 == message_count,
                        );
                    if tools.is_empty() {
                        thread.message(chat);
                    } else {
                        thread.rich_message(chat, |ui| {
                            for tool in tools {
                                ui.add_space(6.0);
                                let mut call = cast::ToolCall::new(tool.tool_name.clone())
                                    .status(tool_status_from_verdict(&tool.verdict));
                                if let Some(duration_ms) = tool.duration_ms {
                                    call = call.metadata(format!("{duration_ms} ms"));
                                }
                                if tool.is_error
                                    && let Some(output) = tool.output.as_ref()
                                {
                                    call = call.body(truncate_for_list(
                                        &session_message_text(output),
                                        240,
                                    ));
                                }
                                ui.add(call.width(ui.available_width()));
                            }
                        });
                    }
                }
                if session.active_tasks > 0 && !last_message_is_assistant {
                    thread.message(
                        cast::ChatMessage::assistant("")
                            .title(default_agent_label(&session.agent_id))
                            .streaming(true),
                    );
                }
            });
        if load_earlier && let Some(window) = message_window {
            self.load_earlier_conversation_messages(&session.session_id, window.total);
        }
    }

    fn render_default_composer(&mut self, ui: &mut egui::Ui, session: &LiveSession) {
        let (content_width, inset) =
            default_conversation_geometry(ui.available_width(), OPERATOR_COMPACT_CONTENT_MARGIN);
        ui.horizontal_top(|ui| {
            ui.add_space(inset);
            ui.vertical(|ui| {
                ui.set_width(content_width);
                let response = cast::AgentComposer::new(&mut self.prompt_input)
                    .placeholder("Ask Turin to investigate, build, explain, or coordinate...")
                    .send_label("Send")
                    .stop_label("Stop")
                    .rows(3)
                    .enabled(session.running)
                    .loading(session.active_tasks > 0)
                    .width(content_width)
                    .show(ui)
                    .inner;
                if response.submitted && !self.prompt_input.trim().is_empty() {
                    let prompt = mem::take(&mut self.prompt_input);
                    self.requested_session_detail = None;
                    if self.default_session_needs_title(&session.session_id)
                        && let Some(title) = default_conversation_title_from_prompt(&prompt)
                    {
                        self.send_command(OperatorCommand::SetSessionTitle {
                            session_id: session.session_id.clone(),
                            title: Some(title),
                        });
                    }
                    self.send_command(OperatorCommand::SubmitPrompt {
                        session_id: session.session_id.clone(),
                        prompt,
                    });
                }
                if response.stopped {
                    self.send_command(OperatorCommand::CancelSession {
                        session_id: session.session_id.clone(),
                    });
                }
            });
        });
    }

    fn render_default_welcome(&mut self, ui: &mut egui::Ui) {
        if self.pending_resume_session_id.is_some() {
            ui.add_space(48.0);
            ui.vertical_centered(|ui| {
                themed_heading(ui, "Opening conversation", 28.0);
                ui.add_space(6.0);
                themed_muted(ui, "Restoring its context and messages...");
            });
            ui.add_space(24.0);
            render_conversation_loading(ui);
            return;
        }
        let pending = self.pending_conversation.is_some();
        let agents = self.dashboard.agents().to_vec();
        self.agent_index = clamp_index(self.agent_index, agents.len());
        ui.add_space(28.0);
        ui.vertical_centered(|ui| {
            themed_heading(ui, "What should Turin do?", 30.0);
            ui.add_space(6.0);
            themed_muted(
                ui,
                "Ask a question, delegate a task, or describe the outcome you want.",
            );
        });
        ui.add_space(22.0);
        if agents.len() > 1 {
            let labels = agents
                .iter()
                .map(|agent| default_agent_label(&agent.id))
                .collect::<Vec<_>>();
            ui.horizontal(|ui| {
                ui.add_space((ui.available_width() - 260.0).max(0.0) / 2.0);
                ui.add(cast::Select::new(&mut self.agent_index, labels).width(260.0));
            });
            ui.add_space(12.0);
        }
        let response = cast::AgentComposer::new(&mut self.prompt_input)
            .placeholder("Ask Turin to investigate, build, explain, or coordinate...")
            .send_label("Start")
            .rows(4)
            .enabled(!agents.is_empty() && !pending)
            .loading(pending)
            .width(ui.available_width())
            .show(ui)
            .inner;
        if response.submitted && !self.prompt_input.trim().is_empty() {
            let prompt = mem::take(&mut self.prompt_input);
            self.start_default_conversation(Some(prompt));
        }
    }

    fn render_runtime_tools_sheet(&mut self, ui: &mut egui::Ui) {
        if !self.runtime_tools_open {
            return;
        }
        let mut open = true;
        cast::Sheet::new(&mut open, "default_runtime_tools")
            .title("Runtime Tools")
            .width(680.0)
            .show(ui.ctx(), |ui, _sheet| {
                ScrollArea::vertical().show(ui, |ui| {
                    self.render_runtime_tools_content(ui);
                });
            });
        if !open {
            self.runtime_tools_open = false;
        }
    }

    fn render_connection_status_inline(&self, ui: &mut egui::Ui) {
        let ready = self
            .dashboard
            .health
            .as_ref()
            .is_some_and(|health| health.ready);
        if ready && self.dashboard.connection_kind == ConnectionKind::Local {
            return;
        }
        let (connection, intent) = self.operator_connection_summary();
        ui.add(cast::Badge::new(connection).intent(intent).status_dot());
    }

    fn render_operator_shell(&mut self, ui: &mut egui::Ui) {
        self.request_selected_ui_lists(false);
        let apps = self.dashboard.ui.apps().cloned().collect::<Vec<_>>();
        self.ui_app_index = clamp_index(self.ui_app_index, apps.len());
        let Some(app) = apps.get(self.ui_app_index).cloned() else {
            self.render_default_shell(ui);
            return;
        };

        let compact = operator_shell_is_compact(ui.available_width());
        if compact {
            let compact_nav_frame = {
                let theme = cast::theme_for_ui(ui);
                egui::Frame::new()
                    .fill(theme.colors.surface)
                    .stroke(egui::Stroke::new(theme.stroke.sm, theme.colors.border))
                    .inner_margin(egui::Margin::symmetric(16, 12))
            };
            egui::Panel::top("operator_shell_compact_nav")
                .frame(compact_nav_frame)
                .show(ui, |ui| {
                    self.render_compact_operator_nav(ui, &apps, &app);
                });
        } else {
            let theme = cast::theme_for_ui(ui);
            egui::Panel::left("operator_shell_nav")
                .resizable(false)
                .exact_size(OPERATOR_SIDEBAR_WIDTH)
                .frame(
                    egui::Frame::new()
                        .fill(theme.colors.surface)
                        .stroke(egui::Stroke::new(theme.stroke.sm, theme.colors.border))
                        .inner_margin(egui::Margin::symmetric(16, 16)),
                )
                .show(ui, |ui| {
                    self.render_operator_sidebar(ui, &apps, &app);
                });
        }

        let top_frame = {
            let theme = cast::theme_for_ui(ui);
            egui::Frame::new()
                .fill(theme.colors.surface)
                .stroke(egui::Stroke::new(theme.stroke.sm, theme.colors.border))
                .inner_margin(egui::Margin::symmetric(20, 11))
        };
        egui::Panel::top("operator_shell_top")
            .exact_size(56.0)
            .frame(top_frame)
            .show(ui, |ui| self.render_operator_top_bar(ui, &app));

        if self.operator_assistant_is_open(&app)
            && let Some(session) = self.selected_operator_session(Some(&app))
        {
            let composer_frame = {
                let theme = cast::theme_for_ui(ui);
                egui::Frame::new()
                    .fill(theme.colors.surface)
                    .stroke(egui::Stroke::new(theme.stroke.sm, theme.colors.border))
                    .inner_margin(egui::Margin::symmetric(20, 12))
            };
            egui::Panel::bottom("operator_assistant_composer")
                .frame(composer_frame)
                .show(ui, |ui| self.render_default_composer(ui, &session));
        }

        egui::CentralPanel::default().show(ui, |ui| {
            ScrollArea::vertical().show(ui, |ui| {
                let outer_margin = if compact {
                    OPERATOR_COMPACT_CONTENT_MARGIN
                } else {
                    OPERATOR_CONTENT_MARGIN
                };
                ui.add_space(outer_margin);
                let (content_width, inset) =
                    operator_content_geometry(ui.available_width(), outer_margin);
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

        self.render_runtime_tools_sheet(ui);
    }

    fn render_operator_top_bar(&mut self, ui: &mut egui::Ui, app: &UiAppRecord) {
        let current = if self.operator_assistant_is_open(app) {
            "Assistant".to_string()
        } else {
            self.active_ui_screen_id(app)
                .and_then(|screen_id| app.screens.get(&screen_id))
                .map(|screen| screen.title.clone())
                .unwrap_or_else(|| "Workspace".to_string())
        };
        ui.horizontal(|ui| {
            ui.add(cast::Breadcrumb::new([ui_app_title(app), current]).size(cast::Size::Small));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .add(
                        cast::Button::new("Runtime tools")
                            .size(cast::Size::Small)
                            .variant(cast::Variant::Ghost),
                    )
                    .clicked()
                {
                    self.runtime_tools_open = true;
                }
                if self.operator_assistant_is_open(app) {
                    if ui
                        .add(
                            cast::Button::new("New conversation")
                                .size(cast::Size::Small)
                                .intent(cast::Intent::Primary)
                                .enabled(
                                    self.pending_conversation.is_none()
                                        && self.pending_resume_session_id.is_none(),
                                ),
                        )
                        .clicked()
                    {
                        self.open_operator_assistant(app);
                        self.start_default_conversation(None);
                    }
                } else if ui
                    .add(
                        cast::Button::new("Refresh")
                            .size(cast::Size::Small)
                            .variant(cast::Variant::Outline),
                    )
                    .clicked()
                {
                    self.request_selected_ui_lists(true);
                }
                self.render_connection_status_inline(ui);
            });
        });
    }

    fn render_operator_sidebar(
        &mut self,
        ui: &mut egui::Ui,
        apps: &[UiAppRecord],
        app: &UiAppRecord,
    ) {
        let app_title = ui_app_title(app);
        ui.horizontal(|ui| {
            ui.add(cast::Avatar::new(app_title.clone()).size(cast::Size::Medium));
            ui.vertical(|ui| {
                themed_heading(ui, app_title, 18.0);
                if let Some(agent_id) = app.source.agent_id.as_deref() {
                    themed_muted(ui, default_agent_label(agent_id));
                }
            });
        });
        if let Some(about) = app
            .definition
            .as_ref()
            .and_then(|definition| definition.about.as_deref())
        {
            ui.add_space(8.0);
            themed_muted(ui, about);
        }
        ui.add_space(16.0);

        if apps.len() > 1 {
            let previous = self.ui_app_index;
            let labels = apps.iter().map(ui_app_title).collect::<Vec<_>>();
            ui.add(cast::Select::new(&mut self.ui_app_index, labels).width(ui.available_width()));
            if self.ui_app_index != previous
                && let Some(candidate) = apps.get(self.ui_app_index)
            {
                self.open_harness_screen(candidate, harness_ui::default_screen_index(candidate));
            }
            ui.add_space(14.0);
        }

        let assistant_selected = self.operator_assistant_is_open(app);
        if ui
            .add(
                cast::MenuItem::new("Assistant")
                    .selected(assistant_selected)
                    .intent(if assistant_selected {
                        cast::Intent::Primary
                    } else {
                        cast::Intent::Neutral
                    }),
            )
            .clicked()
        {
            self.open_operator_assistant(app);
        }
        ui.add_space(8.0);

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
                    themed_overline(ui, menu.title.clone());
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

    fn render_compact_operator_nav(
        &mut self,
        ui: &mut egui::Ui,
        apps: &[UiAppRecord],
        app: &UiAppRecord,
    ) {
        ui.horizontal_wrapped(|ui| {
            themed_heading(ui, ui_app_title(app), 20.0);
            if apps.len() > 1 {
                let previous = self.ui_app_index;
                let labels = apps.iter().map(ui_app_title).collect::<Vec<_>>();
                ui.add(cast::Select::new(&mut self.ui_app_index, labels).width(220.0));
                if self.ui_app_index != previous
                    && let Some(candidate) = apps.get(self.ui_app_index)
                {
                    self.open_harness_screen(
                        candidate,
                        harness_ui::default_screen_index(candidate),
                    );
                }
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                self.render_theme_controls(ui);
            });
        });

        if !app.screens.is_empty() {
            ui.add_space(8.0);
            let screen_index = self
                .ui_screen_indices
                .get(&app.id)
                .copied()
                .unwrap_or_else(|| harness_ui::default_screen_index(app))
                .min(app.screens.len() - 1);
            let mut route_index = if self.operator_assistant_is_open(app) {
                0
            } else {
                screen_index + 1
            };
            let previous = route_index;
            let mut labels = vec!["Assistant".to_string()];
            labels.extend(app.screens.values().map(|screen| screen.title.clone()));
            ScrollArea::horizontal().show(ui, |ui| {
                ui.add(cast::Tabs::new(&mut route_index, labels).size(cast::Size::Small));
            });
            if route_index != previous {
                if route_index == 0 {
                    self.open_operator_assistant(app);
                } else {
                    self.open_harness_screen(app, route_index - 1);
                }
            }
        }
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
            let descendant_selected = menu_descendant_opens(item, current_screen_id);
            let selected = Some(item.opens.as_str()) == current_screen_id && !descendant_selected;
            ui.horizontal(|ui| {
                ui.add_space(depth as f32 * 14.0);
                if ui
                    .add(
                        cast::MenuItem::new(harness_ui::menu_item_label(app, item))
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
            if !item.items.is_empty() && (selected || descendant_selected) {
                self.render_operator_menu_items(ui, app, &item.items, current_screen_id, depth + 1);
            }
        }
    }

    fn render_operator_stage(&mut self, ui: &mut egui::Ui, app: &UiAppRecord) {
        self.render_operator_feedback(ui, app);

        if self.operator_assistant_is_open(app) {
            self.render_operator_assistant(ui, app);
            self.render_pending_harness_ui_action(ui, app);
            self.render_active_harness_pane(ui, app);
            return;
        }

        let mut screen_index = self
            .ui_screen_indices
            .get(&app.id)
            .copied()
            .unwrap_or_else(|| harness_ui::default_screen_index(app));
        let mut render_state = harness_ui::HarnessRenderState {
            lists: &self.ui_lists,
            requested_lists: &self.requested_ui_lists,
            list_errors: &self.ui_list_errors,
            open_disclosures: &mut self.ui_open_disclosures,
            form_values: &mut self.ui_form_values,
            list_filters: &mut self.ui_list_filters,
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

        self.render_pending_harness_ui_action(ui, app);
        self.render_active_harness_pane(ui, app);
    }

    fn render_operator_assistant(&mut self, ui: &mut egui::Ui, app: &UiAppRecord) {
        let agent_id = self.operator_agent_id(Some(app)).map(str::to_string);
        let live_sessions = self
            .dashboard
            .live_sessions
            .iter()
            .enumerate()
            .filter(|(_, session)| {
                agent_id
                    .as_deref()
                    .is_none_or(|agent_id| session.agent_id == agent_id)
            })
            .map(|(index, session)| (index, session.clone()))
            .collect::<Vec<_>>();
        let live_values = live_sessions
            .iter()
            .map(|(_, session)| session.clone())
            .collect::<Vec<_>>();
        let stored_sessions = self
            .dashboard
            .sessions
            .iter()
            .filter(|session| {
                agent_id
                    .as_deref()
                    .is_none_or(|agent_id| session.agent_id == agent_id)
            })
            .cloned()
            .collect::<Vec<_>>();
        let recent_sessions = recent_default_conversations(
            &live_values,
            &stored_sessions,
            DEFAULT_RECENT_CONVERSATION_LIMIT,
        );

        if !live_sessions.is_empty() || !recent_sessions.is_empty() {
            let active_labels = default_conversation_labels(&live_values, &stored_sessions);
            let mut labels = vec!["Choose a conversation".to_string()];
            labels.extend(active_labels);
            labels.extend(
                recent_sessions
                    .iter()
                    .enumerate()
                    .map(|(index, session)| stored_conversation_title(session, index)),
            );
            let selected_live_position = live_sessions
                .iter()
                .position(|(index, _)| *index == self.live_session_index);
            let mut selected = selected_live_position.map(|index| index + 1).unwrap_or(0);
            let previous = selected;
            cast::FilterBar::new().show(ui, |ui| {
                ui.add(cast::Select::new(&mut selected, labels).width(360.0));
            });
            if selected != previous && selected > 0 {
                let target = selected - 1;
                if let Some((global_index, session)) = live_sessions.get(target) {
                    self.live_session_index = *global_index;
                    self.requested_session_detail = None;
                    self.send_command(OperatorCommand::FocusSessionStream {
                        session_id: Some(session.session_id.clone()),
                    });
                } else if let Some(session) = recent_sessions.get(target - live_sessions.len()) {
                    self.resume_default_conversation(session.session_id.clone());
                }
            }
            ui.add_space(18.0);
        }

        if let Some(session) = self.selected_operator_session(Some(app)) {
            self.render_default_conversation(ui, &session);
        } else {
            self.render_operator_assistant_welcome(ui, app, agent_id.as_deref());
        }
    }

    fn render_operator_assistant_welcome(
        &mut self,
        ui: &mut egui::Ui,
        app: &UiAppRecord,
        agent_id: Option<&str>,
    ) {
        if self.pending_resume_session_id.is_some() {
            ui.add_space(48.0);
            ui.vertical_centered(|ui| {
                themed_heading(ui, "Opening conversation", 28.0);
                ui.add_space(6.0);
                themed_muted(ui, "Restoring its context and messages...");
            });
            ui.add_space(24.0);
            render_conversation_loading(ui);
            return;
        }

        let pending = self.pending_conversation.is_some();
        ui.add_space(48.0);
        ui.vertical_centered(|ui| {
            themed_heading(ui, "What should we build?", 30.0);
            ui.add_space(6.0);
            themed_muted(
                ui,
                agent_id
                    .map(|agent_id| {
                        format!(
                            "Work with {} while this workspace keeps tasks, reviews, and plans durable.",
                            default_agent_label(agent_id)
                        )
                    })
                    .unwrap_or_else(|| {
                        "Use the agent for active work and the surrounding desk for durable context."
                            .to_string()
                    }),
            );
        });
        ui.add_space(22.0);
        let response = cast::AgentComposer::new(&mut self.prompt_input)
            .placeholder("Describe the outcome, investigation, or change you want...")
            .send_label("Start")
            .rows(4)
            .enabled(agent_id.is_some() && !pending)
            .loading(pending)
            .width(ui.available_width())
            .show(ui)
            .inner;
        if response.submitted && !self.prompt_input.trim().is_empty() {
            let prompt = mem::take(&mut self.prompt_input);
            self.open_operator_assistant(app);
            self.start_default_conversation(Some(prompt));
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
        if ui
            .add(cast::MenuItem::new("Settings").size(cast::Size::Small))
            .clicked()
        {
            self.sidebar_settings_open = true;
        }

        if !self.sidebar_settings_open {
            return;
        }

        let (connection, intent) = self.operator_connection_summary();
        let mut open = self.sidebar_settings_open;
        let ready_local = self
            .dashboard
            .health
            .as_ref()
            .is_some_and(|health| health.ready)
            && self.dashboard.connection_kind == ConnectionKind::Local;
        cast::Sheet::new(&mut open, "operator_settings")
            .title("Settings")
            .width(360.0)
            .show(ui.ctx(), |ui, sheet| {
                if !ready_local {
                    ui.add(cast::Badge::new(connection).intent(intent).status_dot());
                    ui.add_space(cast::theme_for_ui(ui).spacing.md);
                }
                themed_overline(ui, "Appearance");
                ui.add_space(6.0);
                self.render_theme_controls(ui);
                ui.add_space(cast::theme_for_ui(ui).spacing.lg);
                ui.add(cast::Separator::new());
                ui.add_space(cast::theme_for_ui(ui).spacing.lg);
                themed_overline(ui, "Workspace");
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
                        cast::Button::new("Open runtime tools")
                            .size(cast::Size::Small)
                            .variant(cast::Variant::Ghost),
                    )
                    .clicked()
                {
                    self.runtime_tools_open = true;
                    sheet.close();
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
                        cast::Button::new("System")
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
        let matching_failure = self
            .latest_harness_action_failure
            .as_ref()
            .filter(|failure| harness_action_failure_matches_app(failure, app));

        if let Some(error) = &self.dashboard.last_error
            && !matching_failure
                .is_some_and(|failure| dashboard_error_is_action_failure(error, failure))
        {
            feedback.push((
                format!("runtime-error:{error}"),
                cast::Toast::new("Something went wrong")
                    .body("Open Runtime Tools for technical details.")
                    .intent(cast::Intent::Danger),
            ));
        }

        if let Some(result) = self
            .latest_harness_action_result
            .as_ref()
            .filter(|result| harness_action_result_matches_app(result, app))
        {
            let label = self.harness_action_label(&app.id, &result.action);
            if !harness_action_has_notice(result, &app.id) {
                feedback.push((
                    format!("action-success:{}", self.harness_feedback_revision),
                    cast::Toast::new(format!("{label} completed")).intent(cast::Intent::Success),
                ));
            }
        } else if let Some(failure) = matching_failure {
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
                    let mut toast = cast::Toast::new(notice.title.clone())
                        .intent(ui_notice_intent(notice.level));
                    if !body.trim().is_empty() {
                        toast = toast.body(body.clone());
                    }
                    (
                        format!("notice:{}:{}:{body}", notice.app_id, notice.title),
                        toast,
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

    fn render_runtime_tools_content(&mut self, ui: &mut egui::Ui) {
        self.render_tab_bar(ui);
        ui.add_space(cast::theme_for_ui(ui).spacing.md);
        self.render_active_tab(ui);
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
                    cast::EmptyState::new("No custom apps")
                        .body("Harness-defined app surfaces will appear here when available.")
                        .icon("UI")
                        .intent(cast::Intent::Neutral)
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
                    cast::EmptyState::new("Select an app")
                        .body("Choose a harness-defined app to inspect its declared surfaces.")
                        .intent(cast::Intent::Neutral)
                        .show(ui, |_| {});
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
                    open_disclosures: &mut self.ui_open_disclosures,
                    form_values: &mut self.ui_form_values,
                    list_filters: &mut self.ui_list_filters,
                    selected_list_items: &mut self.ui_selected_list_items,
                };
                let event =
                    harness_ui::render_harness_app(ui, &app, &mut screen_index, &mut render_state);
                self.ui_screen_indices.insert(app.id.clone(), screen_index);
                if let Some(event) = event {
                    self.handle_harness_ui_event(&app, event);
                }

                self.render_pending_harness_ui_action(ui, &app);
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

    fn render_active_harness_pane(&mut self, ui: &mut egui::Ui, app: &UiAppRecord) {
        let Some(active) = self.ui_active_pane.clone() else {
            return;
        };
        let Some(pane) = app.panes.get(&active.id).cloned() else {
            self.ui_active_pane = None;
            return;
        };

        let mut open = true;
        let overlay_id = format!("harness_ui_pane:{}:{}", app.id, pane.id);
        let dialog_body_height = harness_pane_dialog_height(ui.ctx().content_rect().height());
        let mut pane_event = None;
        match harness_pane_presentation(&active, &pane) {
            HarnessPanePresentation::Sheet => {
                cast::Sheet::new(&mut open, overlay_id)
                    .title(pane.title.clone())
                    .width(520.0)
                    .show(ui.ctx(), |ui, _sheet| {
                        pane_event = self.render_harness_pane_body(ui, app, &pane, None);
                    });
            }
            HarnessPanePresentation::Dialog => {
                cast::Dialog::new(&mut open, overlay_id)
                    .title(pane.title.clone())
                    .width(520.0)
                    .show(ui.ctx(), |ui, _dialog| {
                        pane_event =
                            self.render_harness_pane_body(ui, app, &pane, Some(dialog_body_height));
                    });
            }
        }

        if let Some(event) = pane_event {
            self.handle_harness_ui_event(app, event);
        }
        if !open {
            self.ui_active_pane = None;
        }
    }

    fn render_harness_pane_body(
        &mut self,
        ui: &mut egui::Ui,
        app: &UiAppRecord,
        pane: &turin_daemon_protocol::UiPaneIntent,
        max_height: Option<f32>,
    ) -> Option<HarnessUiEvent> {
        let scroll = ScrollArea::vertical();
        let scroll = if let Some(max_height) = max_height {
            scroll.max_height(max_height)
        } else {
            scroll
        };
        scroll
            .show(ui, |ui| {
                let mut render_state = harness_ui::HarnessRenderState {
                    lists: &self.ui_lists,
                    requested_lists: &self.requested_ui_lists,
                    list_errors: &self.ui_list_errors,
                    open_disclosures: &mut self.ui_open_disclosures,
                    form_values: &mut self.ui_form_values,
                    list_filters: &mut self.ui_list_filters,
                    selected_list_items: &mut self.ui_selected_list_items,
                };
                harness_ui::render_harness_pane(ui, app, pane, &mut render_state)
            })
            .inner
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

    fn render_pending_harness_ui_action(&mut self, ui: &mut egui::Ui, app: &UiAppRecord) {
        let Some(pending) = self
            .pending_harness_ui_action
            .as_ref()
            .filter(|pending| pending.app_id == app.id)
        else {
            return;
        };

        let label = pending.label.clone();
        let title = harness_action_confirmation_title(&label);
        let dialog_id = (
            "harness_ui_action_confirmation",
            app.id.clone(),
            pending.action.clone(),
        );
        let mut open = true;
        let mut confirmed = false;
        let mut cancelled = false;
        cast::Dialog::new(&mut open, dialog_id)
            .title(title)
            .width(420.0)
            .muted_sections()
            .show_with_footer(
                ui.ctx(),
                |ui, _dialog| {
                    themed_muted(
                        ui,
                        format!("{} will run this action immediately.", ui_app_title(app)),
                    );
                },
                |ui, dialog| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add(
                                cast::Button::new(label)
                                    .intent(cast::Intent::Primary)
                                    .size(cast::Size::Small),
                            )
                            .clicked()
                        {
                            confirmed = true;
                            dialog.close();
                        }
                        if ui
                            .add(
                                cast::Button::new("Cancel")
                                    .variant(cast::Variant::Outline)
                                    .size(cast::Size::Small),
                            )
                            .clicked()
                        {
                            cancelled = true;
                            dialog.close();
                        }
                    });
                },
            );

        if confirmed {
            self.confirm_pending_harness_ui_action();
        } else if cancelled || !open {
            self.cancel_pending_harness_ui_action();
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

const OPERATOR_COMPACT_BREAKPOINT: f32 = 900.0;
const OPERATOR_SIDEBAR_WIDTH: f32 = 248.0;
const OPERATOR_CONTENT_MARGIN: f32 = 28.0;
const OPERATOR_COMPACT_CONTENT_MARGIN: f32 = 16.0;
const OPERATOR_MAX_CONTENT_WIDTH: f32 = 1120.0;
const DEFAULT_MAX_CONVERSATION_WIDTH: f32 = 880.0;
const DEFAULT_RECENT_CONVERSATION_LIMIT: usize = 10;
const DEFAULT_CONVERSATION_MESSAGE_LIMIT: usize = 48;
const CONVERSATION_MESSAGE_PAGE_SIZE: usize = 48;

fn operator_shell_is_compact(available_width: f32) -> bool {
    available_width < OPERATOR_COMPACT_BREAKPOINT
}

fn operator_content_geometry(available_width: f32, outer_margin: f32) -> (f32, f32) {
    content_geometry(available_width, outer_margin, OPERATOR_MAX_CONTENT_WIDTH)
}

fn default_conversation_geometry(available_width: f32, outer_margin: f32) -> (f32, f32) {
    content_geometry(
        available_width,
        outer_margin,
        DEFAULT_MAX_CONVERSATION_WIDTH,
    )
}

fn content_geometry(available_width: f32, outer_margin: f32, max_content_width: f32) -> (f32, f32) {
    let usable = (available_width - outer_margin * 2.0).max(0.0);
    let content_width = usable.min(max_content_width);
    let inset = ((available_width - content_width) / 2.0).max(outer_margin);
    (content_width, inset)
}

fn default_conversation_labels(
    live_sessions: &[LiveSession],
    stored_sessions: &[SessionSummary],
) -> Vec<String> {
    live_sessions
        .iter()
        .enumerate()
        .map(|(index, session)| default_conversation_title(session, index, stored_sessions))
        .collect()
}

fn default_agent_label(agent_id: &str) -> String {
    if agent_id.eq_ignore_ascii_case("default") {
        "Turin".to_string()
    } else {
        agent_id.to_string()
    }
}

fn recent_default_conversations(
    live_sessions: &[LiveSession],
    stored_sessions: &[SessionSummary],
    limit: usize,
) -> Vec<SessionSummary> {
    let live_ids = live_sessions
        .iter()
        .map(|session| session.session_id.as_str())
        .collect::<BTreeSet<_>>();
    stored_sessions
        .iter()
        .filter(|session| !live_ids.contains(session.session_id.as_str()))
        .take(limit)
        .cloned()
        .collect()
}

fn pending_conversation_session_index(
    pending: &PendingConversation,
    live_sessions: &[LiveSession],
) -> Option<usize> {
    live_sessions.iter().rposition(|session| {
        session.agent_id == pending.agent_id
            && !pending.existing_session_ids.contains(&session.session_id)
    })
}

fn default_conversation_title(
    session: &LiveSession,
    index: usize,
    stored_sessions: &[SessionSummary],
) -> String {
    stored_sessions
        .iter()
        .find(|stored| stored.session_id == session.session_id)
        .and_then(session_summary_title)
        .unwrap_or_else(|| format!("Conversation {}", index + 1))
}

fn stored_conversation_title(session: &SessionSummary, index: usize) -> String {
    session_summary_title(session).unwrap_or_else(|| format!("Conversation {}", index + 1))
}

fn session_summary_title(session: &SessionSummary) -> Option<String> {
    session
        .metadata
        .as_ref()
        .and_then(|metadata| metadata.get("title"))
        .and_then(|title| title.as_str())
        .map(str::trim)
        .filter(|title| !title.is_empty())
        .map(str::to_string)
}

fn default_conversation_title_from_prompt(prompt: &str) -> Option<String> {
    let title = prompt.split_whitespace().collect::<Vec<_>>().join(" ");
    (!title.is_empty()).then(|| truncate_for_list(&title, 56))
}

fn default_conversation_message_is_visible(role: &str, has_body: bool, has_tools: bool) -> bool {
    !role.eq_ignore_ascii_case("system") && (has_body || has_tools)
}

fn session_event_changes_conversation(event: &EventEnvelope) -> bool {
    matches!(
        event.event.as_str(),
        "task_start" | "tool_result" | "message_end" | "task_complete"
    )
}

fn render_conversation_loading(ui: &mut egui::Ui) {
    let width = ui.available_width();
    for factor in [0.68, 0.92, 0.54] {
        ui.add(cast::Skeleton::new().width(width * factor));
        ui.add_space(8.0);
    }
}

fn menu_descendant_opens(item: &UiMenuItem, current_screen_id: Option<&str>) -> bool {
    let Some(current_screen_id) = current_screen_id else {
        return false;
    };
    item.items.iter().any(|child| {
        child.opens == current_screen_id || menu_descendant_opens(child, Some(current_screen_id))
    })
}

fn harness_action_confirmation_title(label: &str) -> String {
    if label.ends_with(['?', '!', '.']) {
        label.to_string()
    } else {
        format!("{label}?")
    }
}

fn harness_pane_presentation(
    active: &ActiveHarnessPane,
    pane: &turin_daemon_protocol::UiPaneIntent,
) -> HarnessPanePresentation {
    let presentation = active
        .presentation
        .as_deref()
        .or(pane.presentation.as_deref())
        .unwrap_or_default();
    let presentation = presentation.trim();
    if presentation.eq_ignore_ascii_case("modal") || presentation.eq_ignore_ascii_case("dialog") {
        HarnessPanePresentation::Dialog
    } else {
        HarnessPanePresentation::Sheet
    }
}

fn harness_pane_dialog_height(viewport_height: f32) -> f32 {
    (viewport_height * 0.66).clamp(240.0, 640.0)
}

fn harness_action_has_notice(result: &HarnessActionRunResult, app_id: &str) -> bool {
    result.ui_intents.iter().any(
        |message| matches!(&message.intent, UiIntent::Notify(notice) if notice.app_id == app_id),
    )
}

fn dashboard_error_is_action_failure(error: &str, failure: &HarnessActionFailure) -> bool {
    error.starts_with(&format!("Harness action '{}' failed", failure.action))
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
    use turin_daemon_protocol::{
        UiIntentMessage, UiIntentSource, UiListNode, UiNode, UiNoticeIntent, UiPaneIntent,
        UiScreenIntent,
    };

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
    fn nested_menu_prefers_the_matching_child() {
        let item = UiMenuItem {
            label: "Work".to_string(),
            opens: "approvals".to_string(),
            id: None,
            icon: None,
            badge: None,
            items: vec![UiMenuItem {
                label: "Approvals".to_string(),
                opens: "approvals".to_string(),
                id: None,
                icon: None,
                badge: None,
                items: Vec::new(),
            }],
        };

        assert!(menu_descendant_opens(&item, Some("approvals")));
        assert!(!menu_descendant_opens(&item, Some("overview")));
    }

    #[test]
    fn harness_action_confirmation_title_adds_only_missing_punctuation() {
        assert_eq!(
            harness_action_confirmation_title("Approve release"),
            "Approve release?"
        );
        assert_eq!(
            harness_action_confirmation_title("Delete release?"),
            "Delete release?"
        );
    }

    #[test]
    fn action_feedback_prefers_explicit_notices_and_specific_failures() {
        let result = HarnessActionRunResult {
            action: "release.approve".to_string(),
            agent_id: "default".to_string(),
            harness_id: Some("default".to_string()),
            result: serde_json::Value::Null,
            ui_intents: vec![UiIntentMessage::new(UiIntent::Notify(UiNoticeIntent {
                app_id: "release".to_string(),
                title: "Approved".to_string(),
                body: None,
                level: Some(UiNoticeLevel::Success),
            }))],
        };
        assert!(harness_action_has_notice(&result, "release"));
        assert!(!harness_action_has_notice(&result, "qa"));

        let failure = HarnessActionFailure {
            action: "release.reject".to_string(),
            agent_id: Some("default".to_string()),
            harness_id: Some("default".to_string()),
            message: "Release was already approved".to_string(),
        };
        assert!(dashboard_error_is_action_failure(
            "Harness action 'release.reject' failed in harness default: Release was already approved",
            &failure,
        ));
        assert!(!dashboard_error_is_action_failure(
            "Failed to load release data",
            &failure,
        ));
    }

    #[test]
    fn pane_presentation_prefers_dynamic_modal_hint_and_defaults_to_sheet() {
        let mut active = ActiveHarnessPane {
            id: "release-notes".to_string(),
            presentation: None,
        };
        let mut pane = UiPaneIntent {
            app_id: "release".to_string(),
            id: active.id.clone(),
            title: "Release notes".to_string(),
            presentation: None,
            nodes: Vec::new(),
        };

        assert_eq!(
            harness_pane_presentation(&active, &pane),
            HarnessPanePresentation::Sheet
        );

        pane.presentation = Some("modal".to_string());
        assert_eq!(
            harness_pane_presentation(&active, &pane),
            HarnessPanePresentation::Dialog
        );

        active.presentation = Some("sheet".to_string());
        assert_eq!(
            harness_pane_presentation(&active, &pane),
            HarnessPanePresentation::Sheet
        );

        assert_eq!(harness_pane_dialog_height(200.0), 240.0);
        assert_eq!(harness_pane_dialog_height(800.0), 528.0);
        assert_eq!(harness_pane_dialog_height(1200.0), 640.0);
    }

    #[test]
    fn operator_shell_compacts_before_content_becomes_cramped() {
        assert!(!operator_shell_is_compact(1200.0));
        assert!(!operator_shell_is_compact(OPERATOR_COMPACT_BREAKPOINT));
        assert!(operator_shell_is_compact(OPERATOR_COMPACT_BREAKPOINT - 1.0));
    }

    #[test]
    fn operator_content_geometry_centers_and_caps_content() {
        assert_eq!(operator_content_geometry(1400.0, 24.0), (1120.0, 140.0));
        assert_eq!(operator_content_geometry(800.0, 16.0), (768.0, 16.0));
        assert_eq!(operator_content_geometry(300.0, 16.0), (268.0, 16.0));
        assert_eq!(
            default_conversation_geometry(1400.0, OPERATOR_CONTENT_MARGIN),
            (880.0, 260.0)
        );
        assert_eq!(
            default_conversation_geometry(800.0, OPERATOR_COMPACT_CONTENT_MARGIN),
            (768.0, 16.0)
        );
    }

    #[test]
    fn app_theme_uses_cast_typography_and_default_spacing() {
        let seed = app_theme_seed(cast::ThemeMode::Light);
        let cast_typography = cast::TypographyTokens::cast();

        assert_eq!(
            seed.typography.heading.family,
            cast_typography.heading.family
        );
        assert_eq!(seed.controls.min_height, 32.0);
        assert_eq!(seed.spacing.md, 12.0);
    }

    #[test]
    fn default_agent_label_keeps_the_standard_workspace_product_facing() {
        assert_eq!(default_agent_label("default"), "Turin");
        assert_eq!(default_agent_label("release-agent"), "release-agent");
    }

    #[test]
    fn default_conversation_title_prefers_user_title_without_exposing_ids() {
        let live = LiveSession {
            agent_id: "default".to_string(),
            slot_id: "slot-1".to_string(),
            session_id: "session-secret-id".to_string(),
            running: true,
            active_tasks: 0,
            queued_tasks: 0,
            current_request_id: None,
            execution: turin_control_client::LiveExecution {
                execution_id: "execution-1".to_string(),
                context_target: serde_json::Value::Null,
                visibility: "client".to_string(),
                durability: "durable".to_string(),
                write_policy: "read_write".to_string(),
            },
            conflict_policy: "queue".to_string(),
        };
        let stored = SessionSummary {
            internal_id: 1,
            session_id: live.session_id.clone(),
            agent_id: live.agent_id.clone(),
            metadata: Some(serde_json::json!({ "title": "Plan the migration" })),
            created_at: "2026-07-31T00:00:00Z".to_string(),
        };

        assert_eq!(
            default_conversation_title(&live, 0, &[stored]),
            "Plan the migration"
        );
        assert_eq!(default_conversation_title(&live, 1, &[]), "Conversation 2");
    }

    #[test]
    fn recent_conversations_exclude_live_sessions_and_keep_snapshot_order() {
        let live = LiveSession {
            agent_id: "default".to_string(),
            slot_id: "slot-live".to_string(),
            session_id: "live".to_string(),
            running: true,
            active_tasks: 0,
            queued_tasks: 0,
            current_request_id: None,
            execution: turin_control_client::LiveExecution {
                execution_id: "execution-live".to_string(),
                context_target: serde_json::Value::Null,
                visibility: "client".to_string(),
                durability: "durable".to_string(),
                write_policy: "read_write".to_string(),
            },
            conflict_policy: "queue".to_string(),
        };
        let stored = ["live", "most-recent", "older"]
            .into_iter()
            .enumerate()
            .map(|(index, session_id)| SessionSummary {
                internal_id: index as i64,
                session_id: session_id.to_string(),
                agent_id: "default".to_string(),
                metadata: None,
                created_at: "2026-07-31T00:00:00Z".to_string(),
            })
            .collect::<Vec<_>>();

        let recent = recent_default_conversations(&[live], &stored, 1);

        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].session_id, "most-recent");
    }

    #[test]
    fn first_prompt_becomes_a_compact_conversation_title() {
        assert_eq!(
            default_conversation_title_from_prompt("  Plan\n the next   Turin release  "),
            Some("Plan the next Turin release".to_string())
        );
        assert_eq!(default_conversation_title_from_prompt(" \n\t "), None);
        assert!(
            default_conversation_title_from_prompt(&"a".repeat(100))
                .is_some_and(|title| title.ends_with("..."))
        );
    }

    #[test]
    fn default_conversation_hides_system_copy_but_keeps_tool_only_turns() {
        assert!(!default_conversation_message_is_visible(
            "system", true, false
        ));
        assert!(!default_conversation_message_is_visible(
            "assistant",
            false,
            false
        ));
        assert!(default_conversation_message_is_visible(
            "assistant",
            false,
            true
        ));
        assert!(default_conversation_message_is_visible("user", true, false));
    }

    #[test]
    fn conversation_refreshes_only_for_persisted_transcript_events() {
        for event in ["task_start", "tool_result", "message_end", "task_complete"] {
            assert!(session_event_changes_conversation(&EventEnvelope::new(
                event,
                serde_json::Value::Null,
            )));
        }
        for event in ["thinking_delta", "message_delta", "turn_prepare"] {
            assert!(!session_event_changes_conversation(&EventEnvelope::new(
                event,
                serde_json::Value::Null,
            )));
        }
    }

    #[test]
    fn pending_conversation_selects_the_latest_new_session_for_its_agent() {
        let live_session = |agent_id: &str, session_id: &str| LiveSession {
            agent_id: agent_id.to_string(),
            slot_id: format!("slot-{session_id}"),
            session_id: session_id.to_string(),
            running: true,
            active_tasks: 0,
            queued_tasks: 0,
            current_request_id: None,
            execution: turin_control_client::LiveExecution {
                execution_id: format!("execution-{session_id}"),
                context_target: serde_json::Value::Null,
                visibility: "client".to_string(),
                durability: "durable".to_string(),
                write_policy: "read_write".to_string(),
            },
            conflict_policy: "queue".to_string(),
        };
        let pending = PendingConversation {
            agent_id: "default".to_string(),
            prompt: Some("Hello".to_string()),
            existing_session_ids: BTreeSet::from(["existing".to_string()]),
        };
        let sessions = vec![
            live_session("default", "existing"),
            live_session("other", "other-new"),
            live_session("default", "first-new"),
            live_session("default", "latest-new"),
        ];

        assert_eq!(
            pending_conversation_session_index(&pending, &sessions),
            Some(3)
        );
        assert_eq!(
            pending_conversation_session_index(&pending, &sessions[..2]),
            None
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
