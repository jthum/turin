use anyhow::{Result, anyhow};
use clap::Parser;
use eframe::egui::{self, Color32, RichText, ScrollArea, TextEdit, Vec2};
use std::mem;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;
use turin_control_client::{
    AgentRuntime, AgentSummary, ChannelRuntime, ChannelSummary, ConnectionKind, LiveSession,
    SessionDetail, SessionSummary, TaskStatus,
};
use turin_daemon_protocol::EventEnvelope;
use turin_ui_core::{
    ConnectionOptions, ConnectionProfileAuth, ConnectionProfileCatalog, ConnectionProfileKind,
    ConnectionProfileSummary, DashboardFreshness, DashboardNoticeLevel, DashboardState,
    OperatorCommand, UiController, UiUpdate, connect_dashboard, spawn_controller,
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
    Agents,
    LiveSessions,
    Sessions,
    Tasks,
    Channels,
    Events,
}

impl TabKind {
    const ALL: [Self; 7] = [
        Self::Connections,
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
    agent_index: usize,
    live_session_index: usize,
    session_index: usize,
    task_index: usize,
    channel_index: usize,
    event_index: usize,
    prompt_input: String,
    requested_session_detail: Option<String>,
    _runtime: Arc<Runtime>,
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
        Self {
            dashboard,
            controller,
            connection_options,
            profile_catalog,
            active_profile,
            tab: TabKind::Connections,
            profile_index: 0,
            agent_index: 0,
            live_session_index: 0,
            session_index: 0,
            task_index: 0,
            channel_index: 0,
            event_index: 0,
            prompt_input: String::new(),
            requested_session_detail: None,
            _runtime: runtime,
        }
    }

    fn apply_update(&mut self, update: UiUpdate) {
        self.dashboard.apply_update(update);
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
        self.agent_index = clamp_index(self.agent_index, self.dashboard.agents().len());
        self.live_session_index =
            clamp_index(self.live_session_index, self.dashboard.live_sessions.len());
        self.session_index = clamp_index(self.session_index, self.dashboard.sessions.len());
        self.task_index = clamp_index(self.task_index, self.dashboard.tasks.len());
        self.channel_index = clamp_index(self.channel_index, self.dashboard.channels().len());
        self.event_index = clamp_index(self.event_index, self.dashboard.recent_events.len());
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

    fn reload_profiles(&mut self) {
        match self.connection_options.load_profiles() {
            Ok(catalog) => {
                self.profile_catalog = catalog;
                self.clamp_selection_indices();
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

    fn reconnect_current(&mut self) {
        self.switch_connection(self.connection_options.clone());
    }

    fn connect_selected_profile(&mut self) {
        if let Some(options) = self.selected_profile_options() {
            self.switch_connection(options);
        } else {
            self.dashboard
                .record_error("No connection profile is currently selected");
        }
    }

    fn switch_connection(&mut self, connection_options: ConnectionOptions) {
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
                self.dashboard = dashboard;
                self.prompt_input.clear();
                self.requested_session_detail = None;
                self.clamp_selection_indices();
                let target = self.dashboard.connection_target.clone();
                self.dashboard
                    .record_info(format!("Connected UI client to {target}"));
            }
            Err(err) => {
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
        self.dashboard.tasks.get(self.task_index).cloned()
    }

    fn selected_channel(&self) -> Option<ChannelSummary> {
        self.dashboard.channels().get(self.channel_index).cloned()
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
        self.dashboard
            .recent_events
            .iter()
            .rev()
            .nth(self.event_index)
            .cloned()
    }

    fn selected_session_detail(&self) -> Option<&SessionDetail> {
        self.current_detail_session_id()
            .as_deref()
            .and_then(|session_id| self.dashboard.session_detail(session_id))
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
        let selected = self.selected_profile().cloned();
        let profiles_source = self.connection_options.profiles_path();

        ui.columns(2, |columns| {
            columns[0].group(|ui| {
                ui.heading("Connection Profiles");
                ui.add_space(6.0);
                ui.label(format!("Source: {}", profiles_source.display()));
                ui.add_space(8.0);
                if profiles.is_empty() {
                    ui.label("No profiles loaded. Add ui-profiles.toml or pass --profiles-file.");
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
                            }
                        }
                    });
                }
            });

            columns[1].group(|ui| {
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
                    "Active Profile",
                    self.active_profile
                        .clone()
                        .unwrap_or_else(|| "Direct CLI/config".to_string()),
                );
                detail_kv(ui, "Profiles File", profiles_source.display().to_string());
                detail_kv(ui, "Available Profiles", profiles.len().to_string());
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

                if let Some(profile) = selected {
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
                    render_session_detail_panel(ui, selected_detail.as_ref());
                } else {
                    ui.label("No persisted sessions found.");
                }
            });
        });
    }

    fn render_tasks_tab(&mut self, ui: &mut egui::Ui) {
        let tasks = self.dashboard.tasks.clone();
        let selected = self.selected_task();

        ui.columns(2, |columns| {
            columns[0].group(|ui| {
                ui.heading("Tasks");
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
        let channels = self.dashboard.channels().to_vec();
        let selected = self.selected_channel();
        let selected_runtime = selected
            .as_ref()
            .and_then(|channel| self.selected_channel_runtime(&channel.id));

        ui.columns(2, |columns| {
            columns[0].group(|ui| {
                ui.heading("Channels");
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
        let events: Vec<_> = self.dashboard.recent_events.iter().rev().cloned().collect();
        let selected = self.selected_event();

        ui.columns(2, |columns| {
            columns[0].group(|ui| {
                ui.heading("Recent Events");
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
                if let Some(profile) = &self.active_profile {
                    ui.add_space(12.0);
                    ui.label(
                        RichText::new(format!("Profile {profile}"))
                            .color(Color32::from_rgb(151, 214, 255))
                            .strong(),
                    );
                }
                ui.add_space(12.0);
                ui.label(
                    RichText::new(format!(
                        "Sync {} ({})",
                        freshness_label(self.dashboard.snapshot_freshness()),
                        self.dashboard.snapshot_age_label()
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

fn profile_kind_label(kind: ConnectionProfileKind) -> &'static str {
    match kind {
        ConnectionProfileKind::LocalConfig => "local-config",
        ConnectionProfileKind::LocalEndpoint => "local-endpoint",
        ConnectionProfileKind::Remote => "remote",
    }
}

fn profile_auth_label(auth: Option<&ConnectionProfileAuth>) -> String {
    match auth {
        Some(ConnectionProfileAuth::TokenEnv(name)) => format!("env:{name}"),
        Some(ConnectionProfileAuth::InlineToken) => "inline token".to_string(),
        None => "none".to_string(),
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

        if !detail.tool_executions.is_empty() {
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

fn json_preview(value: &serde_json::Value, max_chars: usize) -> String {
    let rendered = serde_json::to_string_pretty(value).unwrap_or_else(|_| "null".to_string());
    truncate_for_list(&rendered, max_chars)
}
