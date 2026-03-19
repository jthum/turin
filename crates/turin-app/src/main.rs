use anyhow::{Result, anyhow};
use clap::Parser;
use eframe::egui::{self, Color32, RichText, ScrollArea, TextEdit, Vec2};
use std::mem;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::runtime::Runtime;
use tokio::sync::mpsc;
use tokio::time;
use turin_control_client::{
    AgentRuntime, AgentSummary, ChannelRuntime, ChannelSummary, ConnectionKind, ConnectionSpec,
    ControlClient, LiveSession, SessionSummary, TaskStatus,
};
use turin_daemon_protocol::{EventEnvelope, RuntimeEventsSubscribeParams};
use turin_ui_core::{DashboardSnapshot, DashboardState};

#[derive(Parser, Debug)]
#[command(name = "turin-app", version, about)]
struct Args {
    #[arg(long, default_value = "turin.toml")]
    config: PathBuf,
    #[arg(long)]
    endpoint: Option<PathBuf>,
    #[arg(long)]
    remote_url: Option<String>,
    #[arg(long)]
    auth_token: Option<String>,
    #[arg(long)]
    auth_token_env: Option<String>,
}

enum UiUpdate {
    Snapshot(Box<DashboardSnapshot>),
    Event(EventEnvelope),
    Error(String),
    Info(String),
}

enum OperatorCommand {
    Refresh,
    OpenSession { agent_id: String },
    ResumeSession { session_id: String },
    SubmitPrompt { session_id: String, prompt: String },
    CancelSession { session_id: String },
    KillSession { session_id: String },
    CancelTask { request_id: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TabKind {
    Agents,
    LiveSessions,
    Sessions,
    Tasks,
    Channels,
    Events,
}

impl TabKind {
    const ALL: [Self; 6] = [
        Self::Agents,
        Self::LiveSessions,
        Self::Sessions,
        Self::Tasks,
        Self::Channels,
        Self::Events,
    ];

    fn title(self) -> &'static str {
        match self {
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
    let spec = connection_spec_from_args(&args)?;
    let runtime = Arc::new(Runtime::new()?);
    let client = runtime.block_on(ControlClient::connect(&spec))?;
    let dashboard = runtime.block_on(DashboardState::load(&client))?;
    let (tx, rx) = mpsc::unbounded_channel::<UiUpdate>();
    let (command_tx, command_rx) = mpsc::unbounded_channel::<OperatorCommand>();

    spawn_event_task(runtime.clone(), client.clone(), tx.clone());
    spawn_refresh_task(runtime.clone(), client.clone(), tx.clone());
    spawn_command_task(runtime.clone(), client, command_rx, tx);

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
                dashboard, rx, command_tx, runtime,
            )))
        }),
    )
    .map_err(|err| anyhow!(err.to_string()))
}

fn connection_spec_from_args(args: &Args) -> Result<ConnectionSpec> {
    if let Some(base_url) = &args.remote_url {
        if let Some(auth_token) = &args.auth_token {
            return Ok(ConnectionSpec::Remote {
                base_url: base_url.clone(),
                auth_token: auth_token.clone(),
            });
        }
        if let Some(auth_token_env) = &args.auth_token_env {
            return Ok(ConnectionSpec::RemoteEnv {
                base_url: base_url.clone(),
                auth_token_env: auth_token_env.clone(),
            });
        }
        return Err(anyhow!(
            "--remote-url requires either --auth-token or --auth-token-env"
        ));
    }

    if let Some(endpoint) = &args.endpoint {
        return Ok(ConnectionSpec::LocalEndpoint {
            endpoint: endpoint.clone(),
        });
    }

    Ok(ConnectionSpec::LocalConfig {
        config_path: args.config.clone(),
    })
}

fn spawn_event_task(
    runtime: Arc<Runtime>,
    client: ControlClient,
    tx: mpsc::UnboundedSender<UiUpdate>,
) {
    runtime.spawn(async move {
        match client
            .subscribe_managed(RuntimeEventsSubscribeParams::default())
            .await
        {
            Ok(mut stream) => loop {
                match stream.next_event().await {
                    Ok(event) => {
                        let _ = tx.send(UiUpdate::Event(event));
                    }
                    Err(err) => {
                        let _ = tx.send(UiUpdate::Error(err.to_string()));
                        break;
                    }
                }
            },
            Err(err) => {
                let _ = tx.send(UiUpdate::Error(err.to_string()));
            }
        }
    });
}

fn spawn_refresh_task(
    runtime: Arc<Runtime>,
    client: ControlClient,
    tx: mpsc::UnboundedSender<UiUpdate>,
) {
    runtime.spawn(async move {
        let mut interval = time::interval(Duration::from_secs(5));
        loop {
            interval.tick().await;
            match DashboardState::snapshot(&client).await {
                Ok(snapshot) => {
                    let _ = tx.send(UiUpdate::Snapshot(Box::new(snapshot)));
                }
                Err(err) => {
                    let _ = tx.send(UiUpdate::Error(err.to_string()));
                }
            }
        }
    });
}

fn spawn_command_task(
    runtime: Arc<Runtime>,
    client: ControlClient,
    mut command_rx: mpsc::UnboundedReceiver<OperatorCommand>,
    tx: mpsc::UnboundedSender<UiUpdate>,
) {
    runtime.spawn(async move {
        while let Some(command) = command_rx.recv().await {
            match handle_command(&client, command).await {
                Ok(message) => {
                    let _ = tx.send(UiUpdate::Info(message));
                    match DashboardState::snapshot(&client).await {
                        Ok(snapshot) => {
                            let _ = tx.send(UiUpdate::Snapshot(Box::new(snapshot)));
                        }
                        Err(err) => {
                            let _ = tx.send(UiUpdate::Error(err.to_string()));
                        }
                    }
                }
                Err(err) => {
                    let _ = tx.send(UiUpdate::Error(err.to_string()));
                }
            }
        }
    });
}

async fn handle_command(client: &ControlClient, command: OperatorCommand) -> Result<String> {
    match command {
        OperatorCommand::Refresh => Ok("Refreshed Turin state".to_string()),
        OperatorCommand::OpenSession { agent_id } => {
            let session = client.open_session(&agent_id, None).await?;
            Ok(format!(
                "Opened live session {} for agent {}",
                session.session_id, session.agent_id
            ))
        }
        OperatorCommand::ResumeSession { session_id } => {
            let session = client.resume_session(&session_id, None).await?;
            Ok(format!(
                "Resumed session {} into live slot {}",
                session.session_id, session.slot_id
            ))
        }
        OperatorCommand::SubmitPrompt { session_id, prompt } => {
            let task = client
                .submit_task(None, Some(session_id.clone()), prompt)
                .await?;
            Ok(format!(
                "Submitted task {} to session {}",
                task.request_id, session_id
            ))
        }
        OperatorCommand::CancelSession { session_id } => {
            let result = client.cancel_session(&session_id).await?;
            Ok(format!(
                "Requested cancel for session {} ({})",
                result.session_id, result.agent_id
            ))
        }
        OperatorCommand::KillSession { session_id } => {
            let result = client.kill_session(&session_id).await?;
            Ok(format!(
                "Killed session {} ({})",
                result.session_id, result.agent_id
            ))
        }
        OperatorCommand::CancelTask { request_id } => {
            let task = client.cancel_task(&request_id).await?;
            Ok(format!(
                "Cancelled task {} -> {}",
                task.request_id, task.state
            ))
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
    rx: mpsc::UnboundedReceiver<UiUpdate>,
    command_tx: mpsc::UnboundedSender<OperatorCommand>,
    tab: TabKind,
    agent_index: usize,
    live_session_index: usize,
    session_index: usize,
    task_index: usize,
    channel_index: usize,
    event_index: usize,
    prompt_input: String,
    _runtime: Arc<Runtime>,
}

impl TurinDesktopApp {
    fn new(
        dashboard: DashboardState,
        rx: mpsc::UnboundedReceiver<UiUpdate>,
        command_tx: mpsc::UnboundedSender<OperatorCommand>,
        runtime: Arc<Runtime>,
    ) -> Self {
        Self {
            dashboard,
            rx,
            command_tx,
            tab: TabKind::Agents,
            agent_index: 0,
            live_session_index: 0,
            session_index: 0,
            task_index: 0,
            channel_index: 0,
            event_index: 0,
            prompt_input: String::new(),
            _runtime: runtime,
        }
    }

    fn apply_update(&mut self, update: UiUpdate) {
        match update {
            UiUpdate::Snapshot(snapshot) => self.dashboard.apply_snapshot(*snapshot),
            UiUpdate::Event(event) => self.dashboard.record_event(event),
            UiUpdate::Error(message) => self.dashboard.record_error(message),
            UiUpdate::Info(message) => self.dashboard.record_info(message),
        }
        self.clamp_selection_indices();
    }

    fn clamp_selection_indices(&mut self) {
        self.agent_index = clamp_index(self.agent_index, self.dashboard.agents().len());
        self.live_session_index =
            clamp_index(self.live_session_index, self.dashboard.live_sessions.len());
        self.session_index = clamp_index(self.session_index, self.dashboard.sessions.len());
        self.task_index = clamp_index(self.task_index, self.dashboard.tasks.len());
        self.channel_index = clamp_index(self.channel_index, self.dashboard.channels().len());
        self.event_index = clamp_index(self.event_index, self.dashboard.recent_events.len());
    }

    fn send_command(&mut self, command: OperatorCommand) {
        if let Err(err) = self.command_tx.send(command) {
            self.dashboard
                .record_error(format!("Failed to dispatch operator command: {err}"));
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
            TabKind::Agents => self.render_agents_tab(ui),
            TabKind::LiveSessions => self.render_live_sessions_tab(ui),
            TabKind::Sessions => self.render_sessions_tab(ui),
            TabKind::Tasks => self.render_tasks_tab(ui),
            TabKind::Channels => self.render_channels_tab(ui),
            TabKind::Events => self.render_events_tab(ui),
        }
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
                } else {
                    ui.label("No live sessions are running right now.");
                }
            });
        });
    }

    fn render_sessions_tab(&mut self, ui: &mut egui::Ui) {
        let sessions = self.dashboard.sessions.clone();
        let selected = self.selected_session();

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
        while let Ok(update) = self.rx.try_recv() {
            self.apply_update(update);
        }

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
