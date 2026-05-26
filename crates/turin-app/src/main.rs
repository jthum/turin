use anyhow::{Result, anyhow};
use clap::Parser;
use eframe::egui::{self, Color32, RichText, ScrollArea, TextEdit, Vec2};
use std::collections::{BTreeMap, BTreeSet};
use std::mem;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;
use turin_control_client::{
    AgentRuntime, AgentSummary, ChannelRuntime, ChannelSummary, ConnectionKind, LiveSession,
    SessionBranchDetail, SessionDetail, SessionSummary, TaskStatus,
};
use turin_daemon_protocol::{EventEnvelope, UiNode, WorkItemList};
use turin_types::layout::{
    DEFAULT_BOOTSTRAP_CONFIG_PATH, DEFAULT_BOOTSTRAP_DAEMON_ENDPOINT_PATH, DEFAULT_UI_PROFILES_PATH,
};
use turin_ui_core::{
    ConnectionDraftHistory, ConnectionOptions, ConnectionPreflightOutcome,
    ConnectionPreflightReport, ConnectionProfileActivityBook, ConnectionProfileAuth,
    ConnectionProfileCatalog, ConnectionProfileDraft, ConnectionProfileDraftAuthMode,
    ConnectionProfileDraftDiff, ConnectionProfileDraftValidation, ConnectionProfileKind,
    ConnectionProfileSummary, DashboardFreshness, DashboardNoticeLevel, DashboardState,
    OperatorCommand, UiAppRecord, UiController, UiListRequest, UiUpdate, connect_dashboard,
    ensure_local_daemon_for_draft, preflight_connection_blocking, preflight_draft_blocking,
    spawn_controller,
};

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
            configure_visuals(&cc.egui_ctx);
            Ok(Box::new(TurinDesktopApp::new(
                dashboard,
                controller,
                runtime,
                connection_options,
                profile_catalog,
                active_profile,
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

fn collect_ui_list_requests(nodes: &[UiNode], out: &mut Vec<UiListRequest>) {
    for node in nodes {
        match node {
            UiNode::Section(section) => collect_ui_list_requests(&section.nodes, out),
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

fn configure_visuals(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.override_text_color = Some(Color32::from_rgb(240, 234, 224));
    visuals.widgets.active.bg_fill = Color32::from_rgb(31, 78, 96);
    visuals.widgets.inactive.bg_fill = Color32::from_rgb(23, 33, 40);
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(46, 88, 101);
    visuals.panel_fill = Color32::from_rgb(13, 18, 22);
    visuals.window_fill = Color32::from_rgb(17, 24, 29);
    ctx.set_visuals(visuals);
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
    ui_lists: BTreeMap<String, WorkItemList>,
    requested_ui_lists: BTreeSet<String>,
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
    ) -> Self {
        let profile_draft = connection_options
            .current_profile_draft()
            .unwrap_or_else(|_| ConnectionProfileDraft::default());
        Self {
            dashboard,
            controller,
            connection_options,
            profile_catalog,
            active_profile,
            tab: TabKind::Connections,
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
            ui_lists: BTreeMap::new(),
            requested_ui_lists: BTreeSet::new(),
            _runtime: runtime,
        }
    }

    fn apply_update(&mut self, update: UiUpdate) {
        let auto_follow_event = matches!(&update, UiUpdate::Event(_))
            && !self.events_paused
            && self.events_follow_latest;
        if matches!(&update, UiUpdate::SessionEvent(_)) {
            self.clamp_selection_indices();
            return;
        }
        if let UiUpdate::UiListLoaded { request, items } = &update {
            self.ui_lists
                .insert(request.cache_key(), items.as_ref().clone());
        }
        self.dashboard.apply_update(update);
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
        let mut requests = Vec::new();
        for screen in app.screens.values() {
            collect_ui_list_requests(&screen.nodes, &mut requests);
        }
        for pane in app.panes.values() {
            collect_ui_list_requests(&pane.nodes, &mut requests);
        }
        requests
    }

    fn request_selected_ui_lists(&mut self, force: bool) {
        for request in self.selected_ui_list_requests() {
            let key = request.cache_key();
            if !force
                && (self.ui_lists.contains_key(&key) || self.requested_ui_lists.contains(&key))
            {
                continue;
            }
            self.requested_ui_lists.insert(key);
            self.send_command(OperatorCommand::LoadUiList {
                request: Box::new(request),
            });
        }
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

    fn selected_agent(&self) -> Option<AgentSummary> {
        self.dashboard.agents().get(self.agent_index).cloned()
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

    fn selected_task(&self) -> Option<TaskStatus> {
        self.filtered_tasks().get(self.task_index).cloned()
    }

    fn selected_channel(&self) -> Option<ChannelSummary> {
        self.filtered_channels().get(self.channel_index).cloned()
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

    fn selected_event(&self) -> Option<EventEnvelope> {
        self.filtered_events().get(self.event_index).cloned()
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

    fn render_runtime_overview(&self, ui: &mut egui::Ui) {
        ui.group(|ui| {
            ui.heading("Runtime");
            ui.add_space(8.0);
            if let Some(health) = self.dashboard.health.as_ref() {
                metric_row(
                    ui,
                    "Agents",
                    health.agent_count,
                    "Harnesses",
                    health.harness_count,
                    "Channels",
                    health.channel_count,
                );
                metric_row(
                    ui,
                    "Running",
                    health.running_agent_count,
                    "Active",
                    health.active_task_count,
                    "Queued",
                    health.queued_task_count,
                );
                metric_row(
                    ui,
                    "Awaiting",
                    health.awaiting_result_count,
                    "Issues",
                    health.issue_count,
                    "Failed Channels",
                    health.failed_channel_count,
                );
                ui.add_space(8.0);
                ui.label(
                    RichText::new(format!(
                        "Version {} / protocol {} / {} / {}",
                        health.version,
                        health.protocol_version,
                        health.transport,
                        health.wire_format
                    ))
                    .color(Color32::from_rgb(173, 167, 159)),
                );
            } else {
                ui.label("Loading runtime health...");
            }
        });
    }

    fn render_tab_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            for tab in TabKind::ALL {
                let selected = self.tab == tab;
                if ui.selectable_label(selected, tab.title()).clicked() {
                    self.tab = tab;
                }
            }
        });
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
            columns[0].group(|ui| {
                ui.heading("Connection Profiles");
                ui.add_space(6.0);
                ui.label(format!("Source: {}", profiles_source.display()));
                ui.add_space(8.0);
                if profiles.is_empty() {
                    ui.label(format!(
                        "No profiles loaded. Add {} or pass --profiles-file.",
                        DEFAULT_UI_PROFILES_PATH
                    ));
                } else {
                    ScrollArea::vertical().show(ui, |ui| {
                        for (index, profile) in profiles.iter().enumerate() {
                            let label = format!(
                                "{}{} [{} | {}]",
                                profile.name,
                                if profile.is_default { " (default)" } else { "" },
                                profile_kind_label(profile.kind),
                                profile_auth_label(profile.auth.as_ref())
                            );
                            if ui
                                .selectable_label(index == self.profile_index, label)
                                .clicked()
                            {
                                self.profile_index = index;
                                self.pending_delete_profile = None;
                            }
                        }
                    });
                }

                ui.add_space(10.0);
                ui.separator();
                ui.add_space(8.0);
                ui.label(RichText::new("Manage Profiles").strong());
                ui.add_space(6.0);
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Load Current").clicked() {
                        self.load_current_connection_into_editor();
                    }
                    if ui.button("Load Selected").clicked() {
                        self.load_selected_profile_into_editor();
                    }
                    if ui.button("Load Latest Recent").clicked() {
                        self.load_latest_recent_draft();
                    }
                    if ui.button("New Draft").clicked() {
                        self.reset_profile_editor();
                    }
                });
                ui.add_space(8.0);
                ui.label(RichText::new("Recent Drafts").strong());
                ui.add_space(4.0);
                if recent_drafts.is_empty() {
                    ui.label("No successful draft connections yet.");
                } else {
                    ScrollArea::vertical().max_height(132.0).show(ui, |ui| {
                        for (index, draft) in recent_drafts.iter().enumerate() {
                            if ui
                                .selectable_label(
                                    index == self.recent_draft_index,
                                    draft.summary_label(),
                                )
                                .clicked()
                            {
                                self.recent_draft_index = index;
                            }
                        }
                    });
                    ui.add_space(6.0);
                    if ui.button("Load Selected Recent").clicked() {
                        self.load_selected_recent_draft();
                    }
                }
                ui.add_space(6.0);
                ui.label(RichText::new("Save As Name").strong());
                ui.add(
                    TextEdit::singleline(&mut self.profile_name_input)
                        .hint_text(profile_name_hint.clone()),
                );
                ui.add_space(8.0);
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new("Kind").strong());
                    ui.selectable_value(
                        &mut self.profile_draft.kind,
                        ConnectionProfileKind::LocalConfig,
                        "Local Config",
                    );
                    ui.selectable_value(
                        &mut self.profile_draft.kind,
                        ConnectionProfileKind::LocalEndpoint,
                        "Local Endpoint",
                    );
                    ui.selectable_value(
                        &mut self.profile_draft.kind,
                        ConnectionProfileKind::Remote,
                        "Remote",
                    );
                });
                ui.add_space(6.0);
                ui.label(RichText::new(profile_target_label(self.profile_draft.kind)).strong());
                ui.add(
                    TextEdit::singleline(&mut self.profile_draft.target)
                        .hint_text(profile_target_hint(self.profile_draft.kind)),
                );
                let target_validation = self.profile_draft.validate();
                if let Some(message) = target_validation.target_error.as_ref() {
                    ui.label(
                        RichText::new(message.clone()).color(Color32::from_rgb(255, 138, 128)),
                    );
                } else if let Some(message) = target_validation.target_notice.as_ref() {
                    ui.label(
                        RichText::new(message.clone()).color(Color32::from_rgb(255, 209, 128)),
                    );
                }
                if self.profile_draft.kind == ConnectionProfileKind::Remote {
                    ui.add_space(8.0);
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new("Auth").strong());
                        ui.selectable_value(
                            &mut self.profile_draft.auth_mode,
                            ConnectionProfileDraftAuthMode::TokenEnv,
                            "Token Env",
                        );
                        ui.selectable_value(
                            &mut self.profile_draft.auth_mode,
                            ConnectionProfileDraftAuthMode::InlineToken,
                            "Inline Token",
                        );
                        ui.selectable_value(
                            &mut self.profile_draft.auth_mode,
                            ConnectionProfileDraftAuthMode::None,
                            "None",
                        );
                    });
                    ui.add_space(6.0);
                    ui.label(
                        RichText::new(profile_auth_value_label(self.profile_draft.auth_mode))
                            .strong(),
                    );
                    ui.add(
                        TextEdit::singleline(&mut self.profile_draft.auth_value)
                            .password(
                                self.profile_draft.auth_mode
                                    == ConnectionProfileDraftAuthMode::InlineToken,
                            )
                            .hint_text(profile_auth_value_hint(self.profile_draft.auth_mode)),
                    );
                    let auth_validation = self.profile_draft.validate();
                    if let Some(message) = auth_validation.auth_error.as_ref() {
                        ui.label(
                            RichText::new(message.clone()).color(Color32::from_rgb(255, 138, 128)),
                        );
                    } else if let Some(message) = auth_validation.auth_notice.as_ref() {
                        ui.label(
                            RichText::new(message.clone()).color(Color32::from_rgb(255, 209, 128)),
                        );
                    }
                } else {
                    self.profile_draft.auth_mode = ConnectionProfileDraftAuthMode::None;
                    self.profile_draft.auth_value.clear();
                }
                let draft_validation = self.profile_draft_validation();
                ui.add_space(8.0);
                ui.label(RichText::new("Draft Validation").strong());
                ui.label(RichText::new(draft_validation.summary()).color(
                    if draft_validation.is_valid() {
                        Color32::from_rgb(168, 228, 160)
                    } else {
                        Color32::from_rgb(255, 138, 128)
                    },
                ));
                if self.editor_is_dirty() {
                    ui.label(
                        RichText::new(format!(
                            "Unsaved editor changes vs {}",
                            self.draft_baseline_label
                        ))
                        .color(Color32::from_rgb(255, 209, 128)),
                    );
                }
                ui.checkbox(&mut self.save_profile_as_default, "Set as default");
                ui.add_space(8.0);
                ui.horizontal_wrapped(|ui| {
                    if ui
                        .add_enabled(draft_validation.is_valid(), egui::Button::new("Test Draft"))
                        .clicked()
                    {
                        self.preflight_draft();
                    }
                    if ui
                        .add_enabled(
                            draft_validation.is_valid(),
                            egui::Button::new("Connect Draft"),
                        )
                        .clicked()
                    {
                        self.connect_profile_draft();
                    }
                    if ui
                        .add_enabled(selected.is_some(), egui::Button::new("Test Selected"))
                        .clicked()
                    {
                        self.preflight_selected_profile();
                    }
                    if ui
                        .add_enabled(
                            self.profile_draft.kind == ConnectionProfileKind::LocalConfig,
                            egui::Button::new("Ensure Draft Local"),
                        )
                        .clicked()
                    {
                        self.ensure_local_daemon_for_draft();
                    }
                    if ui
                        .add_enabled(update_selected_ready, egui::Button::new("Update Selected"))
                        .clicked()
                    {
                        self.update_selected_profile();
                    }
                    if ui
                        .add_enabled(
                            draft_validation.is_valid() && typed_name_ready,
                            egui::Button::new("Save As Name"),
                        )
                        .clicked()
                    {
                        self.save_current_profile();
                    }
                    if ui.button("Duplicate Selected").clicked() {
                        self.duplicate_selected_profile();
                    }
                    if ui.button("Rename Selected").clicked() {
                        self.rename_selected_profile();
                    }
                });
                if let Some(action) = self.pending_discard_action.as_ref() {
                    let action_description = action.description();
                    ui.add_space(6.0);
                    ui.horizontal_wrapped(|ui| {
                        ui.label(
                            RichText::new(format!("Pending: {}", action_description))
                                .color(Color32::from_rgb(255, 209, 128)),
                        );
                        if ui.button("Discard Pending Action").clicked() {
                            self.confirm_pending_discard_action();
                        }
                        if ui.button("Cancel Pending Action").clicked() {
                            self.cancel_pending_discard_action();
                        }
                    });
                }
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    let armed = self.is_delete_armed_for_selected();
                    if ui
                        .button(if armed {
                            "Confirm Delete"
                        } else {
                            "Arm Delete"
                        })
                        .clicked()
                    {
                        if armed {
                            self.delete_selected_profile();
                        } else {
                            self.arm_delete_selected_profile();
                        }
                    }
                    if armed && ui.button("Cancel Delete").clicked() {
                        self.cancel_delete_selected_profile();
                    }
                });
                if let Some(profile_name) = self.pending_delete_profile.as_deref() {
                    ui.add_space(6.0);
                    ui.label(
                        RichText::new(format!(
                            "Delete armed for '{}'. Confirm to remove it from the profiles file.",
                            profile_name
                        ))
                        .color(Color32::from_rgb(255, 171, 145)),
                    );
                }
            });

            columns[1].group(|ui| {
                let draft_validation = self.profile_draft_validation();
                ui.heading("Connection Detail");
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
                    if ui.button("Reconnect Current").clicked() {
                        self.reconnect_current();
                    }
                    if ui.button("Reload Profiles").clicked() {
                        self.reload_profiles();
                    }
                });

                if let Some(profile) = selected.as_ref() {
                    ui.add_space(12.0);
                    ui.label(RichText::new("Selected Profile").strong());
                    detail_kv(ui, "Name", profile.name.clone());
                    detail_kv(ui, "Kind", profile_kind_label(profile.kind));
                    detail_kv(ui, "Target", profile.target.clone());
                    detail_kv(ui, "Default", yes_no(profile.is_default));
                    detail_kv(ui, "Auth", profile_auth_label(profile.auth.as_ref()));
                    ui.add_space(10.0);
                    if ui.button("Connect Selected Profile").clicked() {
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
        let selected = self.selected_ui_app();
        let list_requests = self.selected_ui_list_requests();

        ui.columns(2, |columns| {
            columns[0].group(|ui| {
                ui.heading("Harness UI Apps");
                ui.add_space(8.0);
                if apps.is_empty() {
                    ui.label("No harness UI apps are declared by the current runtime.");
                } else {
                    ScrollArea::vertical().show(ui, |ui| {
                        for (index, app) in apps.iter().enumerate() {
                            let title = app
                                .definition
                                .as_ref()
                                .map(|definition| definition.title.as_str())
                                .unwrap_or(app.id.as_str());
                            let label = format!(
                                "{}  screens:{} panes:{} menus:{}",
                                title,
                                app.screens.len(),
                                app.panes.len(),
                                app.menus.len()
                            );
                            if ui
                                .selectable_label(index == self.ui_app_index, label)
                                .clicked()
                            {
                                self.ui_app_index = index;
                            }
                        }
                    });
                }

                ui.add_space(12.0);
                if ui.button("Refresh Selected Lists").clicked() {
                    self.request_selected_ui_lists(true);
                }
                ui.add_space(8.0);
                ui.label(RichText::new("Dynamic UI Signals").strong());
                detail_kv(ui, "Notices", self.dashboard.ui.notices().len().to_string());
                detail_kv(
                    ui,
                    "Open Requests",
                    self.dashboard.ui.opens().len().to_string(),
                );
                detail_kv(
                    ui,
                    "Show Requests",
                    self.dashboard.ui.shows().len().to_string(),
                );
                detail_kv(
                    ui,
                    "Focus Requests",
                    self.dashboard.ui.focuses().len().to_string(),
                );
                detail_kv(
                    ui,
                    "Refresh Requests",
                    self.dashboard.ui.refreshes().len().to_string(),
                );
            });

            columns[1].group(|ui| {
                ui.heading("UI App Detail");
                ui.add_space(8.0);
                let Some(app) = selected else {
                    ui.label("Select a UI app to inspect its declared surfaces.");
                    return;
                };

                detail_kv(ui, "App ID", app.id.clone());
                detail_kv(
                    ui,
                    "Title",
                    app.definition
                        .as_ref()
                        .map(|definition| definition.title.clone())
                        .unwrap_or_else(|| app.id.clone()),
                );
                detail_kv(
                    ui,
                    "Opens With",
                    app.opens_with.clone().unwrap_or_else(|| "None".to_string()),
                );
                detail_kv(ui, "Screens", app.screens.len().to_string());
                detail_kv(ui, "Panes", app.panes.len().to_string());
                detail_kv(ui, "Menus", app.menus.len().to_string());
                detail_kv(ui, "Badges", app.badges.len().to_string());

                ui.add_space(10.0);
                ui.label(RichText::new("Screens").strong());
                for screen in app.screens.values() {
                    ui.label(format!("{}  [{} nodes]", screen.title, screen.nodes.len()));
                }

                ui.add_space(10.0);
                ui.label(RichText::new("Menus").strong());
                for menu in &app.menus {
                    ui.label(format!("{}  [{} items]", menu.title, menu.items.len()));
                }

                if !list_requests.is_empty() {
                    ui.add_space(10.0);
                    ui.label(RichText::new("List Data").strong());
                    for request in &list_requests {
                        let key = request.cache_key();
                        ui.separator();
                        ui.label(RichText::new(&request.source).strong());
                        match self.ui_lists.get(&key) {
                            Some(items) => {
                                ui.label(format!("{} items", items.items.len()));
                                for item in items.items.iter().take(8) {
                                    ui.label(format!(
                                        "{}  [{}]  {}",
                                        item.title, item.status, item.public_id
                                    ));
                                }
                                if items.items.len() > 8 {
                                    ui.label(format!("... and {} more", items.items.len() - 8));
                                }
                            }
                            None if self.requested_ui_lists.contains(&key) => {
                                ui.label("Loading...");
                            }
                            None => {
                                ui.label("Not loaded.");
                            }
                        }
                    }
                }

                if !self.dashboard.ui.notices().is_empty() {
                    ui.add_space(10.0);
                    ui.label(RichText::new("Recent UI Notices").strong());
                    for notice in self.dashboard.ui.notices().iter().rev().take(5) {
                        if notice.app_id == app.id {
                            ui.label(format!("{}: {}", notice.app_id, notice.title));
                        }
                    }
                }
            });
        });
    }

    fn render_agents_tab(&mut self, ui: &mut egui::Ui) {
        let agents = self.dashboard.agents().to_vec();
        let selected_agent = self.selected_agent();
        let selected_runtime = selected_agent
            .as_ref()
            .and_then(|agent| self.selected_agent_runtime(&agent.id));

        ui.columns(2, |columns| {
            columns[0].group(|ui| {
                ui.heading("Agents");
                ui.add_space(8.0);
                ScrollArea::vertical().show(ui, |ui| {
                    for (index, agent) in agents.iter().enumerate() {
                        let label = format!("{}  [{} / {}]", agent.id, agent.provider, agent.model);
                        if ui
                            .selectable_label(index == self.agent_index, label)
                            .clicked()
                        {
                            self.agent_index = index;
                        }
                    }
                });
            });

            columns[1].group(|ui| {
                ui.heading("Agent Detail");
                ui.add_space(8.0);
                if let Some(agent) = selected_agent {
                    detail_kv(ui, "Agent", &agent.id);
                    detail_kv(ui, "Enabled", yes_no(agent.enabled));
                    detail_kv(ui, "Provider", &agent.provider);
                    detail_kv(ui, "Model", &agent.model);
                    detail_kv(ui, "Harness", &agent.harness_ref);

                    if let Some(runtime) = selected_runtime {
                        ui.add_space(8.0);
                        ui.label(RichText::new("Runtime").strong());
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
                    if ui.button("Open Live Session").clicked() {
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
        let selected = self.selected_live_session();
        let selected_detail = self.selected_session_detail().cloned();

        ui.columns(2, |columns| {
            columns[0].group(|ui| {
                ui.heading("Live Sessions");
                ui.add_space(8.0);
                ScrollArea::vertical().show(ui, |ui| {
                    for (index, session) in live_sessions.iter().enumerate() {
                        let label = format!(
                            "{}  [{} | slot {}]",
                            session.session_id, session.agent_id, session.slot_id
                        );
                        if ui
                            .selectable_label(index == self.live_session_index, label)
                            .clicked()
                        {
                            self.live_session_index = index;
                        }
                    }
                });
            });

            columns[1].group(|ui| {
                ui.heading("Live Session Detail");
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
                    ui.add(
                        TextEdit::multiline(&mut self.prompt_input)
                            .desired_rows(8)
                            .hint_text("Write a prompt for the selected live session"),
                    );
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        let can_submit = !self.prompt_input.trim().is_empty();
                        if ui
                            .add_enabled(can_submit, egui::Button::new("Submit Prompt"))
                            .clicked()
                        {
                            let prompt = mem::take(&mut self.prompt_input);
                            self.send_command(OperatorCommand::SubmitPrompt {
                                session_id: session.session_id.clone(),
                                prompt,
                            });
                        }
                        if ui.button("Clear").clicked() {
                            self.prompt_input.clear();
                        }
                    });

                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        if ui.button("Cancel Session").clicked() {
                            self.send_command(OperatorCommand::CancelSession {
                                session_id: session.session_id.clone(),
                            });
                        }
                        if ui.button("Kill Session").clicked() {
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
        let selected = self.selected_session();
        let selected_detail = self.selected_session_detail().cloned();

        ui.columns(2, |columns| {
            columns[0].group(|ui| {
                ui.heading("Stored Sessions");
                ui.add_space(8.0);
                ScrollArea::vertical().show(ui, |ui| {
                    for (index, session) in sessions.iter().enumerate() {
                        let label = format!(
                            "{}  [{}]",
                            session.session_id,
                            truncate_for_list(&session.created_at, 22)
                        );
                        if ui
                            .selectable_label(index == self.session_index, label)
                            .clicked()
                        {
                            self.session_index = index;
                        }
                    }
                });
            });

            columns[1].group(|ui| {
                ui.heading("Stored Session Detail");
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
                    if ui.button("Resume Into Live Session").clicked() {
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
        let selected = self.selected_task();

        ui.columns(2, |columns| {
            columns[0].group(|ui| {
                ui.heading("Tasks");
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Filter").strong());
                    ui.add(
                        TextEdit::singleline(&mut self.task_filter)
                            .hint_text("request id, agent, or state"),
                    );
                    if ui.button("Clear").clicked() {
                        self.task_filter.clear();
                    }
                });
                ui.add_space(8.0);
                ScrollArea::vertical().show(ui, |ui| {
                    for (index, task) in tasks.iter().enumerate() {
                        let label = format!(
                            "{}  [{} / {}]",
                            truncate_for_list(&task.request_id, 18),
                            task.agent_id,
                            task.state
                        );
                        if ui
                            .selectable_label(index == self.task_index, label)
                            .clicked()
                        {
                            self.task_index = index;
                        }
                    }
                });
            });

            columns[1].group(|ui| {
                ui.heading("Task Detail");
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
                        ui.label(RichText::new("Output").strong());
                        ScrollArea::vertical().max_height(180.0).show(ui, |ui| {
                            ui.code(output);
                        });
                    }
                    if let Some(error) = &task.error {
                        ui.add_space(8.0);
                        ui.label(
                            RichText::new("Error")
                                .strong()
                                .color(Color32::from_rgb(255, 171, 145)),
                        );
                        ui.code(error);
                    }
                    ui.add_space(12.0);
                    if ui.button("Cancel Task").clicked() {
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
        let selected = self.selected_channel();
        let selected_runtime = selected
            .as_ref()
            .and_then(|channel| self.selected_channel_runtime(&channel.id));

        ui.columns(2, |columns| {
            columns[0].group(|ui| {
                ui.heading("Channels");
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Filter").strong());
                    ui.add(
                        TextEdit::singleline(&mut self.channel_filter)
                            .hint_text("channel id, kind, or agent"),
                    );
                    if ui.button("Clear").clicked() {
                        self.channel_filter.clear();
                    }
                });
                ui.add_space(8.0);
                ScrollArea::vertical().show(ui, |ui| {
                    for (index, channel) in channels.iter().enumerate() {
                        let label =
                            format!("{}  [{} -> {}]", channel.id, channel.kind, channel.agent_id);
                        if ui
                            .selectable_label(index == self.channel_index, label)
                            .clicked()
                        {
                            self.channel_index = index;
                        }
                    }
                });
            });

            columns[1].group(|ui| {
                ui.heading("Channel Detail");
                ui.add_space(8.0);
                if let Some(channel) = selected {
                    detail_kv(ui, "Channel", &channel.id);
                    detail_kv(ui, "Kind", &channel.kind);
                    detail_kv(ui, "Agent", &channel.agent_id);
                    detail_kv(ui, "Enabled", yes_no(channel.enabled));

                    if let Some(runtime) = selected_runtime {
                        ui.add_space(8.0);
                        ui.label(RichText::new("Runtime").strong());
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
        let selected = self.selected_event();

        ui.columns(2, |columns| {
            columns[0].group(|ui| {
                ui.heading("Recent Events");
                ui.add_space(8.0);
                ui.horizontal_wrapped(|ui| {
                    ui.label(RichText::new("Filter").strong());
                    ui.add(
                        TextEdit::singleline(&mut self.event_filter)
                            .hint_text("event name or payload text"),
                    );
                    if ui.button("Clear").clicked() {
                        self.event_filter.clear();
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    let mut paused = self.events_paused;
                    if ui.checkbox(&mut paused, "Pause").changed() {
                        self.set_events_paused(paused);
                    }
                    ui.checkbox(&mut self.events_follow_latest, "Follow Latest");
                    if ui.button("Jump Latest").clicked() {
                        self.event_index = 0;
                    }
                });
                ui.add_space(8.0);
                ScrollArea::vertical().show(ui, |ui| {
                    for (index, event) in events.iter().enumerate() {
                        let label = format!(
                            "{}  [{}]",
                            event.event,
                            truncate_for_list(
                                &serde_json::to_string(&event.data)
                                    .unwrap_or_else(|_| "{}".to_string()),
                                40,
                            )
                        );
                        if ui
                            .selectable_label(index == self.event_index, label)
                            .clicked()
                        {
                            self.event_index = index;
                        }
                    }
                });
            });

            columns[1].group(|ui| {
                ui.heading("Event Detail");
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
                    ui.label(RichText::new("Payload").strong());
                    ScrollArea::vertical().show(ui, |ui| {
                        ui.code(
                            serde_json::to_string_pretty(&event.data)
                                .unwrap_or_else(|_| "{}".to_string()),
                        );
                    });
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
        ui.label(RichText::new("Branches").strong());
        ui.add_space(8.0);

        let Some(detail) = detail else {
            ui.label("Loading branch detail...");
            return;
        };

        let active_branch = detail.branches.iter().find(|branch| branch.active);
        detail_kv(ui, "Branch Count", detail.branches.len().to_string());
        detail_kv(
            ui,
            "Active Branch",
            active_branch
                .map(branch_descriptor)
                .unwrap_or_else(|| "main".to_string()),
        );

        ui.add_space(6.0);
        ScrollArea::vertical().max_height(140.0).show(ui, |ui| {
            for branch in &detail.branches {
                ui.group(|ui| {
                    ui.horizontal_wrapped(|ui| {
                        let mut label = branch_descriptor(branch);
                        if branch.active {
                            label.push_str("  [active]");
                        }
                        ui.label(
                            RichText::new(label)
                                .strong()
                                .color(Color32::from_rgb(173, 214, 255)),
                        );
                        ui.label(truncate_for_list(&branch.created_at, 22));
                        if !branch.active && ui.button("Checkout").clicked() {
                            self.send_command(OperatorCommand::CheckoutSessionBranch {
                                session_id: session_id.to_string(),
                                branch: branch.branch_id.clone(),
                            });
                        }
                    });
                });
                ui.add_space(4.0);
            }
        });

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label("New Branch");
            ui.text_edit_singleline(&mut self.branch_name_input);
        });
        ui.checkbox(
            &mut self.activate_new_branch,
            "Activate immediately after create",
        );
        let can_create = !self.branch_name_input.trim().is_empty();
        if ui
            .add_enabled(can_create, egui::Button::new("Create Branch"))
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
    }
}

impl eframe::App for TurinDesktopApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        while let Ok(update) = self.controller.update_rx.try_recv() {
            self.apply_update(update);
        }

        self.ensure_session_detail_loaded();

        ctx.request_repaint_after(Duration::from_millis(250));

        let ready = self
            .dashboard
            .health
            .as_ref()
            .is_some_and(|health| health.ready);
        let accent = if ready {
            Color32::from_rgb(111, 214, 161)
        } else {
            Color32::from_rgb(255, 196, 107)
        };
        let connection_kind = match self.dashboard.connection_kind {
            ConnectionKind::Local => "Local",
            ConnectionKind::Remote => "Remote",
        };

        egui::TopBottomPanel::top("top_banner").show(ctx, |ui| {
            ui.add_space(8.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(
                    RichText::new("Turin App")
                        .size(26.0)
                        .color(Color32::from_rgb(142, 214, 255))
                        .strong(),
                );
                ui.add_space(12.0);
                ui.label(
                    RichText::new(if ready { "CONNECTED" } else { "DEGRADED" })
                        .color(accent)
                        .strong(),
                );
                ui.add_space(12.0);
                ui.label(
                    RichText::new(format!("{connection_kind} target"))
                        .color(Color32::from_rgb(201, 195, 187))
                        .strong(),
                );
                ui.label(
                    RichText::new(self.dashboard.connection_target.clone())
                        .color(Color32::from_rgb(201, 195, 187)),
                );
                ui.add_space(12.0);
                ui.label(
                    RichText::new(format!("Source {}", self.active_connection_label()))
                        .color(Color32::from_rgb(151, 214, 255))
                        .strong(),
                );
                ui.add_space(12.0);
                ui.label(
                    RichText::new(format!(
                        "Sync {} ({} / {} / {})",
                        freshness_label(self.dashboard.snapshot_freshness()),
                        self.dashboard.snapshot_age_label(),
                        self.dashboard.last_refresh_status_label(),
                        self.dashboard.last_refresh_latency_label()
                    ))
                    .color(freshness_color(self.dashboard.snapshot_freshness()))
                    .strong(),
                );
                ui.add_space(12.0);
                if ui.button("Refresh").clicked() {
                    self.send_command(OperatorCommand::Refresh);
                }
            });
            ui.add_space(6.0);
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.render_runtime_overview(ui);
            ui.add_space(12.0);

            if let Some(error) = &self.dashboard.last_error {
                ui.group(|ui| {
                    ui.label(
                        RichText::new(error)
                            .color(Color32::from_rgb(255, 171, 145))
                            .strong(),
                    );
                });
                ui.add_space(8.0);
            }

            if let Some(info) = &self.dashboard.last_info {
                ui.group(|ui| {
                    ui.label(
                        RichText::new(info)
                            .color(Color32::from_rgb(151, 214, 255))
                            .strong(),
                    );
                });
                ui.add_space(8.0);
            }

            self.render_tab_bar(ui);
            ui.add_space(10.0);
            self.render_active_tab(ui);

            ui.add_space(12.0);
            ui.collapsing("Diagnostics", |ui| {
                ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
                    ui.code(self.dashboard.status_pretty_json());
                });
            });
        });
    }
}

fn clamp_index(current: usize, len: usize) -> usize {
    if len == 0 { 0 } else { current.min(len - 1) }
}

fn detail_kv(ui: &mut egui::Ui, key: &str, value: impl ToString) {
    ui.horizontal_wrapped(|ui| {
        ui.label(RichText::new(key).strong());
        ui.label(value.to_string());
    });
}

fn metric_row(
    ui: &mut egui::Ui,
    left_label: &str,
    left_value: usize,
    middle_label: &str,
    middle_value: usize,
    right_label: &str,
    right_value: usize,
) {
    ui.horizontal(|ui| {
        metric_chip(ui, left_label, left_value);
        metric_chip(ui, middle_label, middle_value);
        metric_chip(ui, right_label, right_value);
    });
}

fn metric_chip(ui: &mut egui::Ui, label: &str, value: usize) {
    ui.label(
        RichText::new(format!("{}: {}", label, value)).color(Color32::from_rgb(240, 234, 224)),
    );
    ui.add_space(12.0);
}

fn truncate_for_list(value: &str, max_chars: usize) -> String {
    let char_count = value.chars().count();
    if char_count <= max_chars {
        value.to_string()
    } else {
        let prefix: String = value.chars().take(max_chars.saturating_sub(1)).collect();
        format!("{prefix}…")
    }
}

fn yes_no(value: bool) -> &'static str {
    if value { "Yes" } else { "No" }
}

fn connection_kind_label(kind: ConnectionKind) -> &'static str {
    match kind {
        ConnectionKind::Local => "local",
        ConnectionKind::Remote => "remote",
    }
}

fn freshness_label(freshness: DashboardFreshness) -> &'static str {
    match freshness {
        DashboardFreshness::Fresh => "fresh",
        DashboardFreshness::Quiet => "quiet",
        DashboardFreshness::Stale => "stale",
    }
}

fn freshness_color(freshness: DashboardFreshness) -> Color32 {
    match freshness {
        DashboardFreshness::Fresh => Color32::from_rgb(111, 214, 161),
        DashboardFreshness::Quiet => Color32::from_rgb(255, 214, 102),
        DashboardFreshness::Stale => Color32::from_rgb(255, 171, 145),
    }
}

fn preflight_outcome_label(outcome: ConnectionPreflightOutcome) -> &'static str {
    match outcome {
        ConnectionPreflightOutcome::Ready => "ready",
        ConnectionPreflightOutcome::Degraded => "degraded",
        ConnectionPreflightOutcome::Invalid => "invalid",
        ConnectionPreflightOutcome::ConnectFailed => "connect-failed",
    }
}

fn profile_kind_label(kind: ConnectionProfileKind) -> &'static str {
    match kind {
        ConnectionProfileKind::LocalConfig => "local-config",
        ConnectionProfileKind::LocalEndpoint => "local-endpoint",
        ConnectionProfileKind::Remote => "remote",
    }
}

fn profile_target_label(kind: ConnectionProfileKind) -> &'static str {
    match kind {
        ConnectionProfileKind::LocalConfig => "Config Path",
        ConnectionProfileKind::LocalEndpoint => "Endpoint Path",
        ConnectionProfileKind::Remote => "Remote URL",
    }
}

fn profile_target_hint(kind: ConnectionProfileKind) -> &'static str {
    match kind {
        ConnectionProfileKind::LocalConfig => DEFAULT_BOOTSTRAP_CONFIG_PATH,
        ConnectionProfileKind::LocalEndpoint => DEFAULT_BOOTSTRAP_DAEMON_ENDPOINT_PATH,
        ConnectionProfileKind::Remote => "http://127.0.0.1:9324",
    }
}

fn profile_auth_label(auth: Option<&ConnectionProfileAuth>) -> String {
    match auth {
        Some(ConnectionProfileAuth::TokenEnv(name)) => format!("env:{name}"),
        Some(ConnectionProfileAuth::InlineToken) => "inline token".to_string(),
        None => "none".to_string(),
    }
}

fn profile_draft_auth_label(mode: ConnectionProfileDraftAuthMode) -> &'static str {
    match mode {
        ConnectionProfileDraftAuthMode::None => "none",
        ConnectionProfileDraftAuthMode::TokenEnv => "token-env",
        ConnectionProfileDraftAuthMode::InlineToken => "inline-token",
    }
}

fn profile_auth_value_label(mode: ConnectionProfileDraftAuthMode) -> &'static str {
    match mode {
        ConnectionProfileDraftAuthMode::None => "Auth Value",
        ConnectionProfileDraftAuthMode::TokenEnv => "Token Env Var",
        ConnectionProfileDraftAuthMode::InlineToken => "Inline Token",
    }
}

fn profile_auth_value_hint(mode: ConnectionProfileDraftAuthMode) -> &'static str {
    match mode {
        ConnectionProfileDraftAuthMode::None => "remote auth disabled",
        ConnectionProfileDraftAuthMode::TokenEnv => "TURIN_REMOTE_TOKEN",
        ConnectionProfileDraftAuthMode::InlineToken => "paste bearer token",
    }
}

fn notice_level_label(level: DashboardNoticeLevel) -> &'static str {
    match level {
        DashboardNoticeLevel::Error => "ERROR",
        DashboardNoticeLevel::Info => "INFO",
    }
}

fn notice_level_color(level: DashboardNoticeLevel) -> Color32 {
    match level {
        DashboardNoticeLevel::Error => Color32::from_rgb(255, 171, 145),
        DashboardNoticeLevel::Info => Color32::from_rgb(151, 214, 255),
    }
}

fn render_session_detail_panel(ui: &mut egui::Ui, detail: Option<&SessionDetail>) {
    ui.label(RichText::new("Session Detail").strong());
    ui.add_space(8.0);

    let Some(detail) = detail else {
        ui.label("Loading detailed transcript and tool history...");
        return;
    };

    detail_kv(ui, "Messages", detail.messages.len().to_string());
    detail_kv(ui, "Events", detail.events.len().to_string());
    detail_kv(ui, "Tool Calls", detail.tool_executions.len().to_string());

    ui.add_space(8.0);
    ScrollArea::vertical().max_height(280.0).show(ui, |ui| {
        if !detail.messages.is_empty() {
            ui.label(RichText::new("Transcript").strong());
            ui.add_space(4.0);
        }
        for message in detail.messages.iter().rev().take(8).rev() {
            ui.group(|ui| {
                ui.label(
                    RichText::new(format!("{} · turn {}", message.role, message.turn_index))
                        .strong()
                        .color(Color32::from_rgb(142, 214, 255)),
                );
                ui.code(json_preview(&message.content, 360));
            });
            ui.add_space(6.0);
        }

        if !detail.events.is_empty() {
            ui.add_space(8.0);
            ui.label(RichText::new("Recent Events").strong());
            ui.add_space(4.0);
            for event in detail.events.iter().rev().take(4).rev() {
                ui.group(|ui| {
                    ui.label(
                        RichText::new(event.event_type.clone())
                            .strong()
                            .color(Color32::from_rgb(173, 167, 159)),
                    );
                    ui.code(json_preview(&event.payload, 220));
                });
                ui.add_space(6.0);
            }
        }

        if !detail.tool_executions.is_empty() {
            ui.add_space(8.0);
            ui.label(RichText::new("Recent Tool Calls").strong());
            ui.add_space(4.0);
            for tool in detail.tool_executions.iter().rev().take(4).rev() {
                ui.group(|ui| {
                    ui.label(
                        RichText::new(format!("{} · {}", tool.tool_name, tool.verdict))
                            .strong()
                            .color(Color32::from_rgb(255, 196, 107)),
                    );
                    ui.code(json_preview(&tool.args, 260));
                    if let Some(output) = &tool.output {
                        ui.code(json_preview(output, 260));
                    }
                });
                ui.add_space(6.0);
            }
        }
    });
}

fn branch_descriptor(branch: &SessionBranchDetail) -> String {
    match branch.head_turn_index {
        Some(turn_index) => format!("{} · head {}", branch.name, turn_index),
        None => branch.name.clone(),
    }
}

fn json_preview(value: &serde_json::Value, max_chars: usize) -> String {
    let rendered = serde_json::to_string_pretty(value).unwrap_or_else(|_| "null".to_string());
    truncate_for_list(&rendered, max_chars)
}
