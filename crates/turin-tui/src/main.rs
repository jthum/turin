mod settings;

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event as CEvent, KeyCode, KeyEventKind,
};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::layout::{Constraint, Direction, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Padding, Paragraph, Tabs, Wrap};
use ratatui::{DefaultTerminal, Frame};
use serde::Serialize;
use serde_json::Value;
use settings::{
    ChatInspectorPane, ChatSidebarPane, LoadedTuiSettings, TuiSettings, load_settings,
    save_settings,
};
use std::collections::{BTreeMap, VecDeque};
use std::io::stdout;
use std::path::PathBuf;
use std::time::Duration;
use turin_control_client::{
    AgentRuntime, AgentSummary, ChannelRuntime, ChannelSummary, ConnectionKind, LiveSession,
    SessionSummary, TaskStatus,
};
use turin_daemon_protocol::EventEnvelope;
use turin_ui_core::{
    ConnectionDraftHistory, ConnectionOptions, ConnectionPreflightOutcome,
    ConnectionPreflightReport, ConnectionProfileActivityBook, ConnectionProfileAuth,
    ConnectionProfileCatalog, ConnectionProfileDraft, ConnectionProfileDraftAuthMode,
    ConnectionProfileDraftDiff, ConnectionProfileDraftValidation, ConnectionProfileKind,
    ConnectionProfileSummary, DashboardFreshness, DashboardSnapshot, DashboardState,
    OperatorCommand, UiController, UiUpdate, connect_dashboard, ensure_local_daemon_for_draft,
    preflight_connection_blocking, preflight_draft_blocking, spawn_controller,
};

#[derive(Parser, Debug)]
#[command(name = "turin-tui", version, about)]
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
    #[arg(long)]
    tui_config: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TabKind {
    Chat,
    Connections,
    Agents,
    LiveSessions,
    Sessions,
    Tasks,
    Channels,
    Events,
    Settings,
}

impl TabKind {
    const ALL: [Self; 9] = [
        Self::Chat,
        Self::Connections,
        Self::Agents,
        Self::LiveSessions,
        Self::Sessions,
        Self::Tasks,
        Self::Channels,
        Self::Events,
        Self::Settings,
    ];

    fn title(self) -> &'static str {
        match self {
            Self::Chat => "Chat",
            Self::Connections => "Connections",
            Self::Agents => "Agents",
            Self::LiveSessions => "Live Sessions",
            Self::Sessions => "Sessions",
            Self::Tasks => "Tasks",
            Self::Channels => "Channels",
            Self::Events => "Events",
            Self::Settings => "Settings",
        }
    }

    fn next(self) -> Self {
        let idx = Self::ALL
            .iter()
            .position(|candidate| *candidate == self)
            .expect("tab exists");
        Self::ALL[(idx + 1) % Self::ALL.len()]
    }

    fn prev(self) -> Self {
        let idx = Self::ALL
            .iter()
            .position(|candidate| *candidate == self)
            .expect("tab exists");
        Self::ALL[(idx + Self::ALL.len() - 1) % Self::ALL.len()]
    }

    fn from_digit(digit: char) -> Option<Self> {
        match digit {
            '1' => Some(Self::Chat),
            '2' => Some(Self::Connections),
            '3' => Some(Self::Agents),
            '4' => Some(Self::LiveSessions),
            '5' => Some(Self::Sessions),
            '6' => Some(Self::Tasks),
            '7' => Some(Self::Channels),
            '8' => Some(Self::Events),
            '9' => Some(Self::Settings),
            _ => None,
        }
    }
}

#[derive(Clone)]
enum InputMode {
    SubmitPrompt {
        session_id: String,
    },
    ConfirmDiscard {
        action: PendingDraftAction,
    },
    SaveProfile {
        make_default: bool,
    },
    DuplicateProfile {
        source_name: String,
        make_default: bool,
    },
    RenameProfile {
        source_name: String,
        make_default: bool,
    },
    ConfirmDelete {
        profile_name: String,
    },
    EditDraftTarget,
    EditDraftAuth,
    EditTaskFilter,
    EditChannelFilter,
    EditEventFilter,
    EditTranscriptBudget,
    EditUserLabel,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PendingDraftAction {
    CurrentConnection,
    SelectedProfile(String),
    SelectedRecentDraft,
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
            Self::SelectedRecentDraft => {
                "load the selected recent draft into the editor".to_string()
            }
        }
    }
}

struct TuiApp {
    dashboard: DashboardState,
    connection_options: ConnectionOptions,
    settings: TuiSettings,
    settings_path: PathBuf,
    profile_catalog: Option<ConnectionProfileCatalog>,
    active_profile: Option<String>,
    tab: TabKind,
    profile_index: usize,
    recent_draft_index: usize,
    agent_index: usize,
    live_session_index: usize,
    session_index: usize,
    task_index: usize,
    channel_index: usize,
    event_index: usize,
    settings_index: usize,
    chat_sidebar_index: usize,
    chat_scroll_lines: u16,
    chat_session_id: Option<String>,
    profile_draft: ConnectionProfileDraft,
    draft_baseline: ConnectionProfileDraft,
    draft_baseline_label: String,
    recent_drafts: ConnectionDraftHistory,
    profile_activity: ConnectionProfileActivityBook,
    last_preflight_report: Option<ConnectionPreflightReport>,
    input_mode: Option<InputMode>,
    input: String,
    input_cursor: usize,
    requested_session_detail: Option<String>,
    task_filter: String,
    channel_filter: String,
    event_filter: String,
    events_paused: bool,
    events_follow_latest: bool,
    paused_events: Vec<EventEnvelope>,
    live_transcripts: BTreeMap<String, LiveTranscriptState>,
    pending_chat_prompt: Option<PendingChatPrompt>,
}

#[derive(Debug, Clone, Default)]
struct LiveTranscriptState {
    pending_user_messages: VecDeque<String>,
    assistant_preview: String,
    thinking_preview: String,
    recent_tool_calls: VecDeque<String>,
    recent_events: VecDeque<String>,
}

#[derive(Debug, Clone)]
enum ChatSidebarItem {
    LiveSession { session_id: String, label: String },
    StoredSession { session_id: String, label: String },
    Agent { agent_id: String, label: String },
    Channel { label: String },
    Event { summary: String },
}

impl ChatSidebarItem {
    fn label(&self) -> &str {
        match self {
            Self::LiveSession { label, .. }
            | Self::StoredSession { label, .. }
            | Self::Agent { label, .. }
            | Self::Channel { label, .. }
            | Self::Event { summary: label } => label,
        }
    }

    fn session_id(&self) -> Option<&str> {
        match self {
            Self::LiveSession { session_id, .. } | Self::StoredSession { session_id, .. } => {
                Some(session_id)
            }
            _ => None,
        }
    }
}

enum LoopAction {
    Quit,
    Reconnect {
        options: Box<ConnectionOptions>,
        connected_draft: Option<ConnectionProfileDraft>,
    },
}

#[derive(Debug, Clone)]
enum PendingChatPrompt {
    ResumeSession { session_id: String },
    OpenAgent { agent_id: String },
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let initial_options = connection_options(&args);
    let loaded_settings = load_settings(args.tui_config.as_deref())?;

    enable_raw_mode()?;
    execute!(stdout(), EnterAlternateScreen, EnableBracketedPaste)?;
    let mut terminal = ratatui::init();

    let loop_result = run_shell(&mut terminal, initial_options, loaded_settings).await;

    ratatui::restore();
    disable_raw_mode()?;
    execute!(stdout(), LeaveAlternateScreen, DisableBracketedPaste)?;

    loop_result
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

async fn run_shell(
    terminal: &mut DefaultTerminal,
    initial_connection_options: ConnectionOptions,
    loaded_settings: LoadedTuiSettings,
) -> Result<()> {
    let (mut app, mut controller) = connect_shell_state(
        initial_connection_options,
        loaded_settings,
        ConnectionDraftHistory::default(),
        ConnectionProfileActivityBook::default(),
    )
    .await?;

    loop {
        let action = run_app(terminal, &mut app, &mut controller)?;
        match action {
            LoopAction::Quit => {
                controller.shutdown();
                return Ok(());
            }
            LoopAction::Reconnect {
                options,
                connected_draft,
            } => {
                let mut recent_drafts = app.recent_drafts.clone();
                let mut profile_activity = app.profile_activity.clone();
                if let Some(draft) = connected_draft.as_ref() {
                    recent_drafts.record_success(draft);
                }
                let options = *options;
                let reconnect_profile = options.resolved_profile_name().ok().flatten();
                match connect_shell_state(
                    options,
                    LoadedTuiSettings {
                        settings: app.settings.clone(),
                        path: app.settings_path.clone(),
                    },
                    recent_drafts,
                    profile_activity.clone(),
                )
                .await
                {
                    Ok((next_app, next_controller)) => {
                        controller.shutdown();
                        app = next_app;
                        controller = next_controller;
                        if let Some(profile_name) = app.active_profile.clone() {
                            profile_activity.record_connect_result(
                                profile_name,
                                true,
                                format!("Connected to {}", app.dashboard.connection_target),
                            );
                            app.profile_activity = profile_activity;
                        }
                        let target = app.dashboard.connection_target.clone();
                        app.dashboard
                            .record_info(format!("Connected UI client to {target}"));
                    }
                    Err(err) => {
                        if let Some(profile_name) = reconnect_profile {
                            profile_activity.record_connect_result(
                                profile_name,
                                false,
                                format!("Failed to connect UI client: {err}"),
                            );
                            app.profile_activity = profile_activity;
                        }
                        app.dashboard
                            .record_error(format!("Failed to connect UI client: {err}"));
                    }
                }
            }
        }
    }
}

async fn connect_shell_state(
    connection_options: ConnectionOptions,
    loaded_settings: LoadedTuiSettings,
    recent_drafts: ConnectionDraftHistory,
    profile_activity: ConnectionProfileActivityBook,
) -> Result<(TuiApp, UiController)> {
    let spec = connection_options.to_spec()?;
    let (client, dashboard) = connect_dashboard(&spec).await?;
    let profile_catalog = connection_options.load_profiles()?;
    let active_profile = connection_options.resolved_profile_name()?;
    let app = TuiApp::new(
        dashboard,
        connection_options,
        loaded_settings,
        profile_catalog,
        active_profile,
        recent_drafts,
        profile_activity,
    );
    let controller = spawn_controller(&tokio::runtime::Handle::current(), client);
    Ok((app, controller))
}

fn run_app(
    terminal: &mut DefaultTerminal,
    app: &mut TuiApp,
    controller: &mut UiController,
) -> Result<LoopAction> {
    loop {
        while let Ok(update) = controller.update_rx.try_recv() {
            app.apply_update(update);
        }

        app.ensure_session_detail_loaded(&controller.command_tx)?;

        terminal.draw(|frame| render(frame, app))?;

        if event::poll(Duration::from_millis(120)).context("Failed to poll terminal events")? {
            match event::read().context("Failed to read terminal event")? {
                CEvent::Key(key)
                    if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) =>
                {
                    if let Some(action) = handle_key(app, key.code, &controller.command_tx)? {
                        return Ok(action);
                    }
                }
                CEvent::Paste(text) => handle_paste(app, &text)?,
                _ => {}
            }
        }
    }
}

fn handle_paste(app: &mut TuiApp, text: &str) -> Result<()> {
    if app.input_mode.is_some() {
        app.insert_input_text(text);
        Ok(())
    } else {
        app.dashboard
            .record_info("Paste is only accepted while an input editor is open");
        Ok(())
    }
}

fn handle_key(
    app: &mut TuiApp,
    key: KeyCode,
    command_tx: &tokio::sync::mpsc::UnboundedSender<OperatorCommand>,
) -> Result<Option<LoopAction>> {
    if app.input_mode.is_some() {
        return handle_input_mode(app, key, command_tx);
    }

    match key {
        KeyCode::Char('q') | KeyCode::Esc => return Ok(Some(LoopAction::Quit)),
        KeyCode::Tab => app.tab = app.tab.next(),
        KeyCode::BackTab => app.tab = app.tab.prev(),
        KeyCode::Left => app.tab = app.tab.prev(),
        KeyCode::Right => app.tab = app.tab.next(),
        KeyCode::Char(digit) if TabKind::from_digit(digit).is_some() => {
            app.tab = TabKind::from_digit(digit).expect("checked above");
        }
        KeyCode::Down | KeyCode::Char('j') => app.move_selection(1),
        KeyCode::Up | KeyCode::Char('k') => app.move_selection(-1),
        KeyCode::PageUp if app.tab == TabKind::Chat => app.scroll_chat(10),
        KeyCode::PageDown if app.tab == TabKind::Chat => app.scroll_chat(-10),
        KeyCode::Home if app.tab == TabKind::Chat => app.jump_chat_oldest(),
        KeyCode::End if app.tab == TabKind::Chat => app.jump_chat_latest(),
        KeyCode::Char('r') => send_command(command_tx, OperatorCommand::Refresh)?,
        KeyCode::Char(',') if app.tab == TabKind::Chat => app.cycle_left_chat_pane(),
        KeyCode::Char('.') if app.tab == TabKind::Chat => app.cycle_right_chat_pane(),
        KeyCode::Char('h') if app.tab == TabKind::Chat => app.toggle_show_thinking(),
        KeyCode::Char('v') if app.tab == TabKind::Chat => app.toggle_streaming_preview(),
        KeyCode::Char('f') if app.tab == TabKind::Chat => app.toggle_chat_follow_latest(),
        KeyCode::Char('l') if app.tab == TabKind::Connections => app.reload_profiles(),
        KeyCode::Char('v') if app.tab == TabKind::Connections => {
            app.load_current_connection_into_draft()
        }
        KeyCode::Char('b') if app.tab == TabKind::Connections => {
            app.load_selected_profile_into_draft()
        }
        KeyCode::Char('P') if app.tab == TabKind::Connections => app.preflight_selected_profile(),
        KeyCode::Char('T') if app.tab == TabKind::Connections => app.preflight_draft(),
        KeyCode::Char('E') if app.tab == TabKind::Connections => {
            app.ensure_local_daemon_for_draft()
        }
        KeyCode::Char('m') if app.tab == TabKind::Connections => app.cycle_profile_draft_kind(),
        KeyCode::Char('o') if app.tab == TabKind::Connections => {
            app.cycle_profile_draft_auth_mode()
        }
        KeyCode::Char('t') if app.tab == TabKind::Connections => app.start_edit_draft_target(),
        KeyCode::Char('g') if app.tab == TabKind::Connections => app.start_edit_draft_auth(),
        KeyCode::Char('a') if app.tab == TabKind::Connections => {
            app.start_save_profile_input(false)
        }
        KeyCode::Char('A') if app.tab == TabKind::Connections => app.start_save_profile_input(true),
        KeyCode::Char('S') if app.tab == TabKind::Connections => app.update_selected_profile(),
        KeyCode::Char('y') if app.tab == TabKind::Connections => {
            app.start_duplicate_profile_input(false)
        }
        KeyCode::Char('Y') if app.tab == TabKind::Connections => {
            app.start_duplicate_profile_input(true)
        }
        KeyCode::Char('u') if app.tab == TabKind::Connections => {
            app.start_rename_profile_input(false)
        }
        KeyCode::Char('U') if app.tab == TabKind::Connections => {
            app.start_rename_profile_input(true)
        }
        KeyCode::Char('d') if app.tab == TabKind::Connections => app.start_delete_confirmation(),
        KeyCode::Char('s') if app.tab == TabKind::Connections => {
            if let Some(options) = app.selected_profile_options() {
                return Ok(Some(LoopAction::Reconnect {
                    options: Box::new(options),
                    connected_draft: None,
                }));
            }
        }
        KeyCode::Char('[') if app.tab == TabKind::Connections => {
            app.move_recent_draft_selection(-1)
        }
        KeyCode::Char(']') if app.tab == TabKind::Connections => app.move_recent_draft_selection(1),
        KeyCode::Char('R') if app.tab == TabKind::Connections => app.load_selected_recent_draft(),
        KeyCode::Char('/') if app.tab == TabKind::Tasks => app.start_edit_task_filter(),
        KeyCode::Char('/') if app.tab == TabKind::Channels => app.start_edit_channel_filter(),
        KeyCode::Char('/') if app.tab == TabKind::Events => app.start_edit_event_filter(),
        KeyCode::Char('F') if app.tab == TabKind::Tasks => app.clear_task_filter(),
        KeyCode::Char('F') if app.tab == TabKind::Channels => app.clear_channel_filter(),
        KeyCode::Char('F') if app.tab == TabKind::Events => app.clear_event_filter(),
        KeyCode::Char('z') if app.tab == TabKind::Events => app.toggle_events_paused(),
        KeyCode::Char('f') if app.tab == TabKind::Events => app.toggle_events_follow_latest(),
        KeyCode::Char('G') if app.tab == TabKind::Events => app.jump_latest_event(),
        KeyCode::Char('C') if app.tab == TabKind::Connections => {
            if let Some(options) = app.draft_connection_options() {
                return Ok(Some(LoopAction::Reconnect {
                    options: Box::new(options),
                    connected_draft: Some(app.profile_draft.clone()),
                }));
            }
        }
        KeyCode::Char('w') if app.tab == TabKind::Settings => app.save_settings(),
        KeyCode::Char('n') => {
            if let Some(agent) = app.selected_agent() {
                send_command(
                    command_tx,
                    OperatorCommand::OpenSession {
                        agent_id: agent.id.clone(),
                    },
                )?;
            }
        }
        KeyCode::Char('e') => {
            if let Some(session) = app.selected_persisted_session() {
                send_command(
                    command_tx,
                    OperatorCommand::ResumeSession {
                        session_id: session.session_id.clone(),
                    },
                )?;
            }
        }
        KeyCode::Enter => match app.tab {
            TabKind::Chat => app.handle_chat_enter(command_tx)?,
            TabKind::Connections => {
                if let Some(options) = app.selected_profile_options() {
                    return Ok(Some(LoopAction::Reconnect {
                        options: Box::new(options),
                        connected_draft: None,
                    }));
                }
            }
            TabKind::Agents => {
                if let Some(agent) = app.selected_agent() {
                    send_command(
                        command_tx,
                        OperatorCommand::OpenSession {
                            agent_id: agent.id.clone(),
                        },
                    )?;
                }
            }
            TabKind::Sessions => {
                if let Some(session) = app.selected_persisted_session() {
                    send_command(
                        command_tx,
                        OperatorCommand::ResumeSession {
                            session_id: session.session_id.clone(),
                        },
                    )?;
                }
            }
            TabKind::LiveSessions => {
                app.start_prompt_input();
            }
            TabKind::Settings => app.activate_selected_setting(),
            _ => {}
        },
        KeyCode::Char('p') if app.tab == TabKind::Chat => app.handle_chat_enter(command_tx)?,
        KeyCode::Char('p') => app.start_prompt_input(),
        KeyCode::Char('b') if app.tab == TabKind::Settings => app.start_edit_transcript_budget(),
        KeyCode::Char('c') => match app.tab {
            TabKind::LiveSessions => {
                if let Some(session) = app.selected_live_session() {
                    send_command(
                        command_tx,
                        OperatorCommand::CancelSession {
                            session_id: session.session_id.clone(),
                        },
                    )?;
                }
            }
            TabKind::Tasks => {
                if let Some(task) = app.selected_task() {
                    send_command(
                        command_tx,
                        OperatorCommand::CancelTask {
                            request_id: task.request_id.clone(),
                        },
                    )?;
                }
            }
            _ => {}
        },
        KeyCode::Char('x') => {
            if let Some(session) = app.selected_live_session() {
                send_command(
                    command_tx,
                    OperatorCommand::KillSession {
                        session_id: session.session_id.clone(),
                    },
                )?;
            }
        }
        _ => {}
    }

    app.clamp_selection();
    Ok(None)
}

fn handle_input_mode(
    app: &mut TuiApp,
    key: KeyCode,
    command_tx: &tokio::sync::mpsc::UnboundedSender<OperatorCommand>,
) -> Result<Option<LoopAction>> {
    if let Some(InputMode::ConfirmDiscard { action }) = app.input_mode.as_ref() {
        match key {
            KeyCode::Esc | KeyCode::Char('n') | KeyCode::Char('N') => {
                app.dashboard.record_info("Kept the current editor draft");
                app.clear_input_mode();
                return Ok(None);
            }
            KeyCode::Enter | KeyCode::Char('y') | KeyCode::Char('Y') => {
                let action = action.clone();
                app.apply_draft_action(action);
                app.clear_input_mode();
                return Ok(None);
            }
            _ => return Ok(None),
        }
    }

    if let Some(InputMode::ConfirmDelete { profile_name }) = app.input_mode.as_ref() {
        match key {
            KeyCode::Esc | KeyCode::Char('n') | KeyCode::Char('N') => {
                app.dashboard.record_info(format!(
                    "Cancelled delete for connection profile '{}'",
                    profile_name
                ));
                app.clear_input_mode();
                return Ok(None);
            }
            KeyCode::Enter | KeyCode::Char('y') | KeyCode::Char('Y') => {
                let profile_name = profile_name.clone();
                app.delete_profile(&profile_name);
                app.clear_input_mode();
                return Ok(None);
            }
            _ => return Ok(None),
        }
    }

    match key {
        KeyCode::Esc => app.clear_input_mode(),
        KeyCode::Left => app.move_input_cursor_left(),
        KeyCode::Right => app.move_input_cursor_right(),
        KeyCode::Home => app.move_input_cursor_home(),
        KeyCode::End => app.move_input_cursor_end(),
        KeyCode::Backspace => app.delete_input_left(),
        KeyCode::Delete => app.delete_input_right(),
        KeyCode::Enter => {
            let input = app.input.trim().to_string();
            if input.is_empty() {
                let message = match app.input_mode.as_ref() {
                    Some(InputMode::SubmitPrompt { .. }) => "Prompt cannot be empty",
                    Some(InputMode::SaveProfile { .. })
                    | Some(InputMode::DuplicateProfile { .. })
                    | Some(InputMode::RenameProfile { .. }) => "Profile name cannot be empty",
                    Some(InputMode::EditDraftTarget) => "Profile target cannot be empty",
                    Some(InputMode::EditDraftAuth) => "Profile auth value cannot be empty",
                    Some(InputMode::EditTranscriptBudget) => {
                        "Transcript budget must be a positive integer"
                    }
                    Some(InputMode::EditUserLabel) => "User label cannot be empty",
                    Some(InputMode::EditTaskFilter)
                    | Some(InputMode::EditChannelFilter)
                    | Some(InputMode::EditEventFilter) => "Use Esc to clear the filter",
                    Some(InputMode::ConfirmDiscard { .. }) => "Discard confirmation is required",
                    Some(InputMode::ConfirmDelete { .. }) => "Delete confirmation is required",
                    None => "Input cannot be empty",
                };
                app.dashboard.record_error(message);
                app.clear_input_mode();
                return Ok(None);
            }

            match app.input_mode.clone() {
                Some(InputMode::SubmitPrompt { session_id }) => {
                    app.note_prompt_submitted(&session_id, &input);
                    send_command(
                        command_tx,
                        OperatorCommand::SubmitPrompt {
                            session_id,
                            prompt: input,
                        },
                    )?;
                }
                Some(InputMode::SaveProfile { make_default }) => {
                    app.save_current_profile(&input, make_default);
                }
                Some(InputMode::DuplicateProfile {
                    source_name,
                    make_default,
                }) => {
                    app.duplicate_profile(&source_name, &input, make_default);
                }
                Some(InputMode::RenameProfile {
                    source_name,
                    make_default,
                }) => {
                    app.rename_profile(&source_name, &input, make_default);
                }
                Some(InputMode::EditDraftTarget) => {
                    app.profile_draft.target = input;
                    app.dashboard
                        .record_info("Updated the connection profile draft target");
                }
                Some(InputMode::EditDraftAuth) => {
                    app.profile_draft.auth_value = input;
                    app.dashboard
                        .record_info("Updated the connection profile draft auth value");
                }
                Some(InputMode::EditTranscriptBudget) => match input.parse::<usize>() {
                    Ok(value) if value >= 16 * 1024 => {
                        app.settings.chat.transcript_memory_budget_bytes = value;
                        app.chat_scroll_lines = 0;
                        app.dashboard.record_info(format!(
                            "Updated transcript memory budget to {} bytes",
                            value
                        ));
                    }
                    _ => app
                        .dashboard
                        .record_error("Transcript budget must be an integer >= 16384 bytes"),
                },
                Some(InputMode::EditUserLabel) => {
                    app.settings.chat.user_label = input;
                    app.dashboard
                        .record_info("Updated the UI-only user label for chat rendering");
                }
                Some(InputMode::EditTaskFilter) => {
                    app.task_filter = input;
                    app.dashboard.record_info("Updated the task filter");
                }
                Some(InputMode::EditChannelFilter) => {
                    app.channel_filter = input;
                    app.dashboard.record_info("Updated the channel filter");
                }
                Some(InputMode::EditEventFilter) => {
                    app.event_filter = input;
                    app.dashboard.record_info("Updated the event filter");
                }
                Some(InputMode::ConfirmDiscard { .. }) => {}
                Some(InputMode::ConfirmDelete { .. }) => {}
                None => {}
            }
            app.clear_input_mode();
        }
        KeyCode::Char(ch) => app.insert_input_char(ch),
        _ => {}
    }
    Ok(None)
}

fn send_command(
    command_tx: &tokio::sync::mpsc::UnboundedSender<OperatorCommand>,
    command: OperatorCommand,
) -> Result<()> {
    command_tx
        .send(command)
        .map_err(|_| anyhow!("UI command channel closed"))
}

fn subtle_border_style() -> Style {
    Style::default()
        .fg(Color::Rgb(74, 78, 84))
        .add_modifier(Modifier::DIM)
}

fn panel_block<'a>(title: &'a str) -> Block<'a> {
    Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(subtle_border_style())
        .padding(Padding::horizontal(1))
}

fn render(frame: &mut Frame<'_>, app: &mut TuiApp) {
    let footer_height = if app.input_mode.is_some() { 6 } else { 4 };
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6),
            Constraint::Length(3),
            Constraint::Min(12),
            Constraint::Length(footer_height),
        ])
        .split(frame.area());

    render_banner(frame, app, layout[0]);
    render_tabs(frame, app, layout[1]);

    if app.tab == TabKind::Chat {
        render_chat(frame, app, layout[2]);
    } else {
        let main = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(38), Constraint::Percentage(62)])
            .split(layout[2]);
        render_left_panel(frame, app, main[0]);
        render_right_panel(frame, app, main[1]);
    }
    render_footer(frame, app, layout[3]);
}

fn render_chat(frame: &mut Frame<'_>, app: &mut TuiApp, area: ratatui::layout::Rect) {
    let show_left = app.settings.layout.left_pane != ChatSidebarPane::None;
    let show_right = app.settings.layout.right_pane != ChatInspectorPane::None
        && (app.settings.layout.right_pane != ChatInspectorPane::Thinking
            || app.settings.chat.show_thinking);

    let mut constraints = Vec::new();
    if show_left {
        constraints.push(Constraint::Length(32));
    }
    constraints.push(Constraint::Min(40));
    if show_right {
        constraints.push(Constraint::Length(36));
    }

    let sections = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(constraints)
        .split(area);

    let mut index = 0usize;
    if show_left {
        render_chat_sidebar(frame, app, sections[index]);
        index += 1;
    }
    render_chat_center(frame, app, sections[index]);
    if show_right {
        render_chat_inspector(frame, app, sections[index + 1]);
    }
}

fn render_banner(frame: &mut Frame<'_>, app: &TuiApp, area: ratatui::layout::Rect) {
    let health = app.dashboard.health.as_ref();
    let ready = health.is_some_and(|health| health.ready);
    let status_color = if ready {
        Color::LightGreen
    } else {
        Color::LightYellow
    };
    let banner = Paragraph::new(vec![
        Line::from(vec![
            Span::styled(
                "Turin TUI",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                if ready { "CONNECTED" } else { "DEGRADED" },
                Style::default()
                    .fg(status_color)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Target: ", Style::default().fg(Color::Gray)),
            Span::raw(app.dashboard.connection_target.clone()),
        ]),
        Line::from(vec![
            Span::styled("Profile: ", Style::default().fg(Color::Gray)),
            Span::raw(app.active_connection_label()),
            Span::styled("  Sync: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!(
                    "{} ({} / {} / {})",
                    freshness_label(app.dashboard.snapshot_freshness()),
                    app.dashboard.snapshot_age_label(),
                    app.dashboard.last_refresh_status_label(),
                    app.dashboard.last_refresh_latency_label()
                ),
                Style::default().fg(freshness_color(app.dashboard.snapshot_freshness())),
            ),
        ]),
        Line::from(vec![
            Span::styled("Activity: ", Style::default().fg(Color::Gray)),
            Span::raw(format!(
                "{} events  last event {}  refresh {} ok / {} fail  notice {}",
                app.dashboard.total_event_count,
                app.dashboard.event_age_label(),
                app.dashboard.refresh_success_count,
                app.dashboard.refresh_failure_count,
                app.dashboard.notice_age_label()
            )),
            Span::styled("  Counts: ", Style::default().fg(Color::Gray)),
            Span::raw(format!(
                "{} agents, {} live sessions, {} stored sessions, {} tasks, {} channels",
                app.dashboard.agents().len(),
                app.dashboard.live_sessions.len(),
                app.dashboard.sessions.len(),
                app.dashboard.tasks.len(),
                app.dashboard.channels().len()
            )),
        ]),
    ])
    .block(panel_block("Connection"))
    .wrap(Wrap { trim: true });
    frame.render_widget(banner, area);
}

fn render_tabs(frame: &mut Frame<'_>, app: &TuiApp, area: ratatui::layout::Rect) {
    let titles = TabKind::ALL
        .iter()
        .map(|tab| Line::from(Span::raw(tab.title())))
        .collect::<Vec<_>>();
    let selected = TabKind::ALL
        .iter()
        .position(|candidate| *candidate == app.tab)
        .unwrap_or(0);
    let tabs = Tabs::new(titles)
        .select(selected)
        .highlight_style(
            Style::default()
                .fg(Color::LightCyan)
                .add_modifier(Modifier::BOLD),
        )
        .divider(Span::raw(" "))
        .block(panel_block("Views"));
    frame.render_widget(tabs, area);
}

fn render_left_panel(frame: &mut Frame<'_>, app: &mut TuiApp, area: ratatui::layout::Rect) {
    let title = match app.tab {
        TabKind::Chat => "Chat",
        TabKind::Connections => "Connections",
        TabKind::Agents => "Agents",
        TabKind::LiveSessions => "Live Sessions",
        TabKind::Sessions => "Stored Sessions",
        TabKind::Tasks => "Tasks",
        TabKind::Channels => "Channels",
        TabKind::Events => "Events",
        TabKind::Settings => "Settings",
    };
    let items = app.list_items();
    let mut state = ListState::default();
    if !items.is_empty() {
        state.select(Some(app.selected_index()));
    }
    let list = List::new(items)
        .highlight_style(
            Style::default()
                .bg(Color::Rgb(28, 56, 73))
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(">> ")
        .block(panel_block(title));
    frame.render_stateful_widget(list, area, &mut state);
}

fn render_right_panel(frame: &mut Frame<'_>, app: &TuiApp, area: ratatui::layout::Rect) {
    let detail = Paragraph::new(app.detail_text())
        .block(panel_block("Detail"))
        .wrap(Wrap { trim: false });
    frame.render_widget(detail, area);
}

fn render_footer(frame: &mut Frame<'_>, app: &TuiApp, area: ratatui::layout::Rect) {
    if app.input_mode.is_some() {
        let block = panel_block(app.input_title());
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let sections = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(2), Constraint::Length(1)])
            .split(inner);

        let (input_lines, cursor) = app.visible_input_editor(sections[0]);
        let input_editor = Paragraph::new(input_lines)
            .style(Style::default().fg(Color::White))
            .wrap(Wrap { trim: false });
        frame.render_widget(input_editor, sections[0]);

        let hint = Paragraph::new(Line::from(app.input_hint().to_string()))
            .style(Style::default().fg(Color::Gray))
            .wrap(Wrap { trim: true });
        frame.render_widget(hint, sections[1]);

        if let Some(cursor) = cursor {
            frame.set_cursor_position(cursor);
        }
        return;
    }

    let lines = {
        let mut lines = vec![Line::from(app.help_text())];
        if app.tab == TabKind::Connections {
            let validation = app.profile_draft_validation();
            if !validation.is_valid()
                || validation.target_notice.is_some()
                || validation.auth_notice.is_some()
            {
                lines.push(Line::from(Span::styled(
                    format!("Draft: {}", validation.summary()),
                    Style::default().fg(if validation.is_valid() {
                        Color::Yellow
                    } else {
                        Color::LightRed
                    }),
                )));
            }
            if app.editor_is_dirty() {
                lines.push(Line::from(Span::styled(
                    format!(
                        "Draft differs from {}: {}",
                        app.draft_baseline_label,
                        app.editor_diff().summary()
                    ),
                    Style::default().fg(Color::Yellow),
                )));
            }
        }
        if let Some(error) = &app.dashboard.last_error {
            lines.push(Line::from(Span::styled(
                error.clone(),
                Style::default().fg(Color::LightRed),
            )));
        }
        if let Some(info) = &app.dashboard.last_info {
            lines.push(Line::from(Span::styled(
                info.clone(),
                Style::default().fg(Color::LightGreen),
            )));
        }
        if lines.len() == 1 {
            lines.push(Line::from(
                "Actions update through the shared local/remote control client.",
            ));
        }
        lines
    };

    let footer = Paragraph::new(lines)
        .block(panel_block("Help"))
        .wrap(Wrap { trim: true });
    frame.render_widget(footer, area);
}

fn render_chat_sidebar(frame: &mut Frame<'_>, app: &mut TuiApp, area: ratatui::layout::Rect) {
    let items = app
        .chat_sidebar_items()
        .into_iter()
        .map(|item| ListItem::new(item.label().to_string()))
        .collect::<Vec<_>>();
    let mut state = ListState::default();
    if !items.is_empty() {
        state.select(Some(app.chat_sidebar_index));
    }
    let list = List::new(items)
        .highlight_style(
            Style::default()
                .bg(Color::Rgb(28, 56, 73))
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(">> ")
        .block(panel_block(app.settings.layout.left_pane.title()));
    frame.render_stateful_widget(list, area, &mut state);
}

fn render_chat_center(frame: &mut Frame<'_>, app: &TuiApp, area: ratatui::layout::Rect) {
    let block = panel_block("Chat");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(6)])
        .split(inner);

    let header = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("Session: ", Style::default().fg(Color::Gray)),
            Span::styled(
                app.current_chat_title(),
                Style::default()
                    .fg(Color::LightCyan)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Mode: ", Style::default().fg(Color::Gray)),
            Span::raw(app.current_chat_mode_label()),
            Span::styled("  Stream: ", Style::default().fg(Color::Gray)),
            Span::raw(if app.settings.chat.show_streaming_preview {
                "preview on"
            } else {
                "preview hidden"
            }),
            Span::styled("  Thinking: ", Style::default().fg(Color::Gray)),
            Span::raw(if app.settings.chat.show_thinking {
                "visible"
            } else {
                "hidden"
            }),
        ]),
    ])
    .wrap(Wrap { trim: true });
    frame.render_widget(header, layout[0]);

    let transcript_width = layout[1].width.max(1) as usize;
    let (text, scroll) = app.chat_transcript_text(layout[1].height as usize, transcript_width);
    let transcript = Paragraph::new(text)
        .scroll((scroll, 0))
        .wrap(Wrap { trim: false });
    frame.render_widget(transcript, layout[1]);
}

fn render_chat_inspector(frame: &mut Frame<'_>, app: &TuiApp, area: ratatui::layout::Rect) {
    let title = app.settings.layout.right_pane.title();
    let body = match app.settings.layout.right_pane {
        ChatInspectorPane::Thinking => app.current_thinking_text(),
        ChatInspectorPane::Tools => app.current_tool_text(),
        ChatInspectorPane::Events => app.current_chat_event_text(),
        ChatInspectorPane::SessionMeta => app.current_session_meta_text(),
        ChatInspectorPane::None => String::new(),
    };

    let inspector = Paragraph::new(body)
        .block(panel_block(title))
        .wrap(Wrap { trim: false });
    frame.render_widget(inspector, area);
}

impl TuiApp {
    fn new(
        dashboard: DashboardState,
        connection_options: ConnectionOptions,
        loaded_settings: LoadedTuiSettings,
        profile_catalog: Option<ConnectionProfileCatalog>,
        active_profile: Option<String>,
        recent_drafts: ConnectionDraftHistory,
        profile_activity: ConnectionProfileActivityBook,
    ) -> Self {
        let profile_draft = connection_options
            .current_profile_draft()
            .unwrap_or_else(|_| ConnectionProfileDraft::default());
        let mut app = Self {
            dashboard,
            connection_options,
            settings: loaded_settings.settings,
            settings_path: loaded_settings.path,
            profile_catalog,
            active_profile,
            tab: TabKind::Chat,
            profile_index: 0,
            recent_draft_index: 0,
            agent_index: 0,
            live_session_index: 0,
            session_index: 0,
            task_index: 0,
            channel_index: 0,
            event_index: 0,
            settings_index: 0,
            chat_sidebar_index: 0,
            chat_scroll_lines: 0,
            chat_session_id: None,
            draft_baseline: profile_draft.clone(),
            draft_baseline_label: "current connection".to_string(),
            profile_draft,
            recent_drafts,
            profile_activity,
            last_preflight_report: None,
            input_mode: None,
            input: String::new(),
            input_cursor: 0,
            requested_session_detail: None,
            task_filter: String::new(),
            channel_filter: String::new(),
            event_filter: String::new(),
            events_paused: false,
            events_follow_latest: true,
            paused_events: Vec::new(),
            live_transcripts: BTreeMap::new(),
            pending_chat_prompt: None,
        };
        app.events_follow_latest = app.settings.chat.follow_latest;
        app.initialize_chat_session();
        app.clamp_selection();
        app
    }

    fn apply_update(&mut self, update: UiUpdate) {
        match &update {
            UiUpdate::Event(event) => {
                self.apply_live_event(event);
                if !self.events_paused && self.events_follow_latest {
                    self.event_index = 0;
                }
            }
            UiUpdate::SessionDetail(detail) => {
                self.clear_live_transcript_for(&detail.session.session_id);
                self.clear_pending_messages_for(&detail.session.session_id);
            }
            UiUpdate::Snapshot(snapshot) => {
                if self.chat_session_id.is_none() {
                    self.chat_session_id = snapshot
                        .live_sessions
                        .first()
                        .map(|session| session.session_id.clone())
                        .or_else(|| {
                            snapshot
                                .sessions
                                .first()
                                .map(|session| session.session_id.clone())
                        });
                }
                self.maybe_activate_pending_chat_prompt(snapshot);
            }
            UiUpdate::RefreshTelemetry { .. } | UiUpdate::Error(_) | UiUpdate::Info(_) => {}
        }

        self.dashboard.apply_update(update);
        self.initialize_chat_session();
        self.clamp_selection();
    }

    fn clamp_selection(&mut self) {
        self.profile_index = clamp_index(
            self.profile_index,
            self.profile_catalog
                .as_ref()
                .map(|catalog| catalog.profiles().len())
                .unwrap_or(0),
        );
        self.recent_draft_index =
            clamp_index(self.recent_draft_index, self.recent_drafts.drafts().len());
        self.agent_index = clamp_index(self.agent_index, self.dashboard.agents().len());
        self.live_session_index =
            clamp_index(self.live_session_index, self.dashboard.live_sessions.len());
        self.session_index = clamp_index(self.session_index, self.dashboard.sessions.len());
        self.task_index = clamp_index(self.task_index, self.filtered_tasks().len());
        self.channel_index = clamp_index(self.channel_index, self.filtered_channels().len());
        self.event_index = clamp_index(self.event_index, self.filtered_events().len());
        self.settings_index = clamp_index(self.settings_index, self.settings_items().len());
        self.chat_sidebar_index =
            clamp_index(self.chat_sidebar_index, self.chat_sidebar_items().len());
        self.sync_chat_selection_from_sidebar();
    }

    fn move_selection(&mut self, delta: isize) {
        let len = self.current_len();
        if len == 0 {
            return;
        }
        let index = self.selected_index() as isize + delta;
        let index = index.clamp(0, (len.saturating_sub(1)) as isize) as usize;
        self.set_selected_index(index);
    }

    fn selected_index(&self) -> usize {
        match self.tab {
            TabKind::Chat => self.chat_sidebar_index,
            TabKind::Connections => self.profile_index,
            TabKind::Agents => self.agent_index,
            TabKind::LiveSessions => self.live_session_index,
            TabKind::Sessions => self.session_index,
            TabKind::Tasks => self.task_index,
            TabKind::Channels => self.channel_index,
            TabKind::Events => self.event_index,
            TabKind::Settings => self.settings_index,
        }
    }

    fn set_selected_index(&mut self, value: usize) {
        match self.tab {
            TabKind::Chat => self.chat_sidebar_index = value,
            TabKind::Connections => self.profile_index = value,
            TabKind::Agents => self.agent_index = value,
            TabKind::LiveSessions => self.live_session_index = value,
            TabKind::Sessions => self.session_index = value,
            TabKind::Tasks => self.task_index = value,
            TabKind::Channels => self.channel_index = value,
            TabKind::Events => self.event_index = value,
            TabKind::Settings => self.settings_index = value,
        }
    }

    fn current_len(&self) -> usize {
        match self.tab {
            TabKind::Chat => self.chat_sidebar_items().len(),
            TabKind::Connections => self
                .profile_catalog
                .as_ref()
                .map(|catalog| catalog.profiles().len())
                .unwrap_or(0),
            TabKind::Agents => self.dashboard.agents().len(),
            TabKind::LiveSessions => self.dashboard.live_sessions.len(),
            TabKind::Sessions => self.dashboard.sessions.len(),
            TabKind::Tasks => self.filtered_tasks().len(),
            TabKind::Channels => self.filtered_channels().len(),
            TabKind::Events => self.filtered_events().len(),
            TabKind::Settings => self.settings_items().len(),
        }
    }

    fn move_recent_draft_selection(&mut self, delta: isize) {
        let len = self.recent_drafts.drafts().len();
        if len == 0 {
            self.dashboard
                .record_error("No recent draft connections have been recorded yet");
            return;
        }
        let index = self.recent_draft_index as isize + delta;
        self.recent_draft_index = index.clamp(0, (len.saturating_sub(1)) as isize) as usize;
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

    fn selected_agent(&self) -> Option<&AgentSummary> {
        self.dashboard.agents().get(self.agent_index)
    }

    fn selected_profile(&self) -> Option<&ConnectionProfileSummary> {
        self.profile_catalog
            .as_ref()?
            .profiles()
            .get(self.profile_index)
    }

    fn selected_recent_draft(&self) -> Option<&ConnectionProfileDraft> {
        self.recent_drafts.drafts().get(self.recent_draft_index)
    }

    fn set_profile_draft(
        &mut self,
        draft: ConnectionProfileDraft,
        baseline_label: impl Into<String>,
    ) {
        self.profile_draft = draft.clone();
        self.draft_baseline = draft;
        self.draft_baseline_label = baseline_label.into();
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

    fn selected_profile_options(&self) -> Option<ConnectionOptions> {
        let selected = self.selected_profile()?;
        self.profile_catalog
            .as_ref()?
            .connection_options(&selected.name)
    }

    fn reload_profiles(&mut self) {
        match self.connection_options.load_profiles() {
            Ok(catalog) => {
                self.profile_catalog = catalog;
                self.clamp_selection();
                self.dashboard
                    .record_info("Reloaded UI connection profiles");
            }
            Err(err) => self
                .dashboard
                .record_error(format!("Failed to load UI profiles: {err}")),
        }
    }

    fn load_current_connection_into_draft(&mut self) {
        if self.editor_is_dirty() {
            self.begin_input_mode(
                InputMode::ConfirmDiscard {
                    action: PendingDraftAction::CurrentConnection,
                },
                "",
            );
            self.dashboard.record_info(format!(
                "The profile editor has unsaved changes ({}). Press y or Enter to discard them and load the current connection.",
                self.editor_diff().summary()
            ));
            return;
        }
        self.apply_draft_action(PendingDraftAction::CurrentConnection);
    }

    fn load_selected_profile_into_draft(&mut self) {
        let Some(profile_name) = self.selected_profile().map(|profile| profile.name.clone()) else {
            self.dashboard
                .record_error("No connection profile is currently selected");
            return;
        };
        if self.editor_is_dirty() {
            self.begin_input_mode(
                InputMode::ConfirmDiscard {
                    action: PendingDraftAction::SelectedProfile(profile_name.clone()),
                },
                "",
            );
            self.dashboard.record_info(format!(
                "The profile editor has unsaved changes ({}). Press y or Enter to discard them and load '{}'.",
                self.editor_diff().summary(),
                profile_name
            ));
            return;
        }
        self.apply_draft_action(PendingDraftAction::SelectedProfile(profile_name));
    }

    fn load_selected_recent_draft(&mut self) {
        if self.editor_is_dirty() {
            self.begin_input_mode(
                InputMode::ConfirmDiscard {
                    action: PendingDraftAction::SelectedRecentDraft,
                },
                "",
            );
            self.dashboard.record_info(format!(
                "The profile editor has unsaved changes ({}). Press y or Enter to discard them and load the selected recent draft.",
                self.editor_diff().summary()
            ));
            return;
        }
        self.apply_draft_action(PendingDraftAction::SelectedRecentDraft);
    }

    fn profile_draft_validation(&self) -> ConnectionProfileDraftValidation {
        self.profile_draft.validate()
    }

    fn cycle_profile_draft_kind(&mut self) {
        self.profile_draft.kind = match self.profile_draft.kind {
            ConnectionProfileKind::LocalConfig => ConnectionProfileKind::LocalEndpoint,
            ConnectionProfileKind::LocalEndpoint => ConnectionProfileKind::Remote,
            ConnectionProfileKind::Remote => ConnectionProfileKind::LocalConfig,
        };
        if self.profile_draft.kind != ConnectionProfileKind::Remote {
            self.profile_draft.auth_mode = ConnectionProfileDraftAuthMode::None;
            self.profile_draft.auth_value.clear();
        }
        self.dashboard.record_info(format!(
            "Profile draft kind is now {}",
            profile_kind_label(self.profile_draft.kind)
        ));
    }

    fn cycle_profile_draft_auth_mode(&mut self) {
        if self.profile_draft.kind != ConnectionProfileKind::Remote {
            self.dashboard
                .record_error("Auth mode can only be edited for remote profile drafts");
            return;
        }
        self.profile_draft.auth_mode = match self.profile_draft.auth_mode {
            ConnectionProfileDraftAuthMode::None => ConnectionProfileDraftAuthMode::TokenEnv,
            ConnectionProfileDraftAuthMode::TokenEnv => ConnectionProfileDraftAuthMode::InlineToken,
            ConnectionProfileDraftAuthMode::InlineToken => ConnectionProfileDraftAuthMode::None,
        };
        if self.profile_draft.auth_mode == ConnectionProfileDraftAuthMode::None {
            self.profile_draft.auth_value.clear();
        }
        self.dashboard.record_info(format!(
            "Profile draft auth mode is now {}",
            profile_draft_auth_label(self.profile_draft.auth_mode)
        ));
    }

    fn select_profile_by_name(&mut self, profile_name: &str) {
        if let Some(index) = self.profile_catalog.as_ref().and_then(|catalog| {
            catalog
                .profiles()
                .iter()
                .position(|profile| profile.name == profile_name)
        }) {
            self.profile_index = index;
        }
    }

    fn apply_draft_action(&mut self, action: PendingDraftAction) {
        match action {
            PendingDraftAction::CurrentConnection => {
                match self.connection_options.current_profile_draft() {
                    Ok(draft) => {
                        self.set_profile_draft(draft, "current connection");
                        self.dashboard
                            .record_info("Loaded current connection into the profile draft");
                    }
                    Err(err) => self.dashboard.record_error(format!(
                        "Failed to load current connection into draft: {err}"
                    )),
                }
            }
            PendingDraftAction::SelectedProfile(profile_name) => {
                match self.connection_options.load_profile_draft(&profile_name) {
                    Ok(draft) => {
                        self.set_profile_draft(draft, format!("saved profile '{}'", profile_name));
                        self.dashboard.record_info(format!(
                            "Loaded connection profile '{}' into the draft",
                            profile_name
                        ));
                    }
                    Err(err) => self.dashboard.record_error(format!(
                        "Failed to load connection profile into draft: {err}"
                    )),
                }
            }
            PendingDraftAction::SelectedRecentDraft => {
                let Some(draft) = self.selected_recent_draft().cloned() else {
                    self.dashboard
                        .record_error("No recent draft connection is currently selected");
                    return;
                };
                self.set_profile_draft(draft, "selected recent draft");
                self.dashboard
                    .record_info("Loaded the selected recent draft into the profile draft");
            }
        }
    }

    fn save_current_profile(&mut self, profile_name: &str, make_default: bool) {
        let validation = self.profile_draft_validation();
        if !validation.is_valid() {
            self.dashboard.record_error(format!(
                "Cannot save invalid connection profile draft: {}",
                validation.summary()
            ));
            return;
        }
        match self.connection_options.save_profile_draft(
            profile_name,
            &self.profile_draft,
            make_default,
        ) {
            Ok(catalog) => {
                self.profile_catalog = Some(catalog);
                self.select_profile_by_name(profile_name);
                self.set_profile_draft(
                    self.profile_draft.clone(),
                    format!("saved profile '{}'", profile_name),
                );
                self.clamp_selection();
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

        match self
            .connection_options
            .save_profile_draft(&profile_name, &self.profile_draft, false)
        {
            Ok(catalog) => {
                let active_profile = self.active_profile.as_deref() == Some(profile_name.as_str());
                self.profile_catalog = Some(catalog);
                self.select_profile_by_name(&profile_name);
                self.set_profile_draft(
                    self.profile_draft.clone(),
                    format!("saved profile '{}'", profile_name),
                );
                self.clamp_selection();
                self.dashboard.record_info(if active_profile {
                    format!(
                        "Updated connection profile '{}'. Use reconnect to apply the saved changes.",
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
        self.last_preflight_report = Some(report.clone());
        if report.is_success() {
            self.dashboard
                .record_info(format!("Preflight for '{}' succeeded", selected.name));
        } else {
            self.dashboard.record_error(format!(
                "Preflight for '{}' failed: {}",
                selected.name, report.message
            ));
        }
    }

    fn preflight_draft(&mut self) {
        let report = preflight_draft_blocking(&self.connection_options, &self.profile_draft);
        self.last_preflight_report = Some(report.clone());
        if report.is_success() {
            self.dashboard.record_info("Draft preflight succeeded");
        } else {
            self.dashboard
                .record_error(format!("Draft preflight failed: {}", report.message));
        }
    }

    fn ensure_local_daemon_for_draft(&mut self) {
        match ensure_local_daemon_for_draft(&self.connection_options, &self.profile_draft) {
            Ok(message) => self.dashboard.record_info(message),
            Err(err) => self
                .dashboard
                .record_error(format!("Failed to ensure local daemon: {err}")),
        }
    }

    fn draft_connection_options(&mut self) -> Option<ConnectionOptions> {
        let validation = self.profile_draft_validation();
        if !validation.is_valid() {
            self.dashboard.record_error(format!(
                "Cannot connect invalid connection profile draft: {}",
                validation.summary()
            ));
            return None;
        }

        match self
            .connection_options
            .connection_options_for_draft(&self.profile_draft)
        {
            Ok(options) => Some(options),
            Err(err) => {
                self.dashboard
                    .record_error(format!("Failed to build connection from draft: {err}"));
                None
            }
        }
    }

    fn duplicate_profile(&mut self, source_name: &str, target_name: &str, make_default: bool) {
        match self
            .connection_options
            .duplicate_profile(source_name, target_name, make_default)
        {
            Ok(catalog) => {
                self.profile_catalog = Some(catalog);
                self.select_profile_by_name(target_name);
                self.clamp_selection();
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

    fn rename_profile(&mut self, source_name: &str, target_name: &str, make_default: bool) {
        match self
            .connection_options
            .rename_profile(source_name, target_name, make_default)
        {
            Ok(catalog) => {
                if self.connection_options.profile.as_deref() == Some(source_name) {
                    self.connection_options.profile = Some(target_name.to_string());
                }
                if self.active_profile.as_deref() == Some(source_name) {
                    self.active_profile = Some(target_name.to_string());
                }
                self.profile_catalog = Some(catalog);
                self.select_profile_by_name(target_name);
                self.clamp_selection();
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

    fn delete_profile(&mut self, profile_name: &str) {
        let fallback_connection =
            if self.connection_options.profile.as_deref() == Some(profile_name) {
                self.connection_options.materialized().ok()
            } else {
                None
            };

        match self.connection_options.delete_profile(profile_name) {
            Ok(catalog) => {
                if let Some(options) = fallback_connection {
                    self.connection_options = options;
                }
                if self.active_profile.as_deref() == Some(profile_name) {
                    self.active_profile = None;
                }
                self.profile_catalog = Some(catalog);
                self.clamp_selection();
                self.dashboard
                    .record_info(format!("Deleted connection profile '{}'", profile_name));
            }
            Err(err) => self
                .dashboard
                .record_error(format!("Failed to delete connection profile: {err}")),
        }
    }

    fn selected_live_session(&self) -> Option<&LiveSession> {
        self.dashboard.live_sessions.get(self.live_session_index)
    }

    fn selected_persisted_session(&self) -> Option<&SessionSummary> {
        self.dashboard.sessions.get(self.session_index)
    }

    fn selected_task(&self) -> Option<TaskStatus> {
        self.filtered_tasks().get(self.task_index).cloned()
    }

    fn selected_channel(&self) -> Option<ChannelSummary> {
        self.filtered_channels().get(self.channel_index).cloned()
    }

    fn initialize_chat_session(&mut self) {
        let current_exists = self
            .chat_session_id
            .as_deref()
            .is_some_and(|session_id| self.session_exists(session_id));
        if !current_exists {
            self.chat_session_id = self
                .dashboard
                .live_sessions
                .first()
                .map(|session| session.session_id.clone())
                .or_else(|| {
                    self.dashboard
                        .sessions
                        .first()
                        .map(|session| session.session_id.clone())
                });
        }
        self.sync_chat_selection_to_session();
    }

    fn maybe_activate_pending_chat_prompt(&mut self, snapshot: &DashboardSnapshot) {
        let Some(pending) = self.pending_chat_prompt.clone() else {
            return;
        };

        let resolved_session = match pending {
            PendingChatPrompt::ResumeSession { session_id } => snapshot
                .live_sessions
                .iter()
                .find(|session| session.session_id == session_id)
                .map(|session| session.session_id.clone()),
            PendingChatPrompt::OpenAgent { agent_id } => snapshot
                .live_sessions
                .iter()
                .find(|session| session.agent_id == agent_id)
                .map(|session| session.session_id.clone()),
        };

        if let Some(session_id) = resolved_session {
            self.chat_session_id = Some(session_id.clone());
            self.chat_scroll_lines = 0;
            self.begin_input_mode(InputMode::SubmitPrompt { session_id }, "");
            self.pending_chat_prompt = None;
            self.dashboard
                .record_info("Live chat session is ready. Type the prompt and press Enter.");
        }
    }

    fn session_exists(&self, session_id: &str) -> bool {
        self.dashboard
            .live_sessions
            .iter()
            .any(|session| session.session_id == session_id)
            || self
                .dashboard
                .sessions
                .iter()
                .any(|session| session.session_id == session_id)
    }

    fn sync_chat_selection_to_session(&mut self) {
        let Some(session_id) = self.chat_session_id.as_deref() else {
            return;
        };
        let sidebar = self.chat_sidebar_items();
        if let Some(index) = sidebar
            .iter()
            .position(|item| item.session_id() == Some(session_id))
        {
            self.chat_sidebar_index = index;
        }
    }

    fn sync_chat_selection_from_sidebar(&mut self) {
        if self.tab != TabKind::Chat {
            return;
        }
        if let Some(item) = self.current_chat_sidebar_item()
            && let Some(session_id) = item.session_id()
        {
            self.chat_session_id = Some(session_id.to_string());
            if self.settings.chat.follow_latest {
                self.chat_scroll_lines = 0;
            }
        }
    }

    fn current_chat_sidebar_item(&self) -> Option<ChatSidebarItem> {
        self.chat_sidebar_items()
            .get(self.chat_sidebar_index)
            .cloned()
    }

    fn chat_sidebar_items(&self) -> Vec<ChatSidebarItem> {
        match self.settings.layout.left_pane {
            ChatSidebarPane::Sessions => {
                let mut items = self
                    .dashboard
                    .live_sessions
                    .iter()
                    .map(|session| ChatSidebarItem::LiveSession {
                        session_id: session.session_id.clone(),
                        label: format!(
                            "{}  {}  live",
                            self.session_label(&session.session_id, Some(&session.agent_id)),
                            session.slot_id
                        ),
                    })
                    .collect::<Vec<_>>();
                items.extend(self.dashboard.sessions.iter().map(|session| {
                    ChatSidebarItem::StoredSession {
                        session_id: session.session_id.clone(),
                        label: format!(
                            "{}  {}",
                            self.session_label(&session.session_id, Some(&session.agent_id)),
                            session.created_at
                        ),
                    }
                }));
                items
            }
            ChatSidebarPane::Agents => self
                .dashboard
                .agents()
                .iter()
                .map(|agent| ChatSidebarItem::Agent {
                    agent_id: agent.id.clone(),
                    label: format!("{} [{}]", agent.id, agent.model),
                })
                .collect(),
            ChatSidebarPane::Channels => self
                .dashboard
                .channels()
                .iter()
                .map(|channel| ChatSidebarItem::Channel {
                    label: format!("{} [{}]", channel.id, channel.kind),
                })
                .collect(),
            ChatSidebarPane::Events => self
                .filtered_events()
                .into_iter()
                .take(32)
                .map(|event| ChatSidebarItem::Event {
                    summary: format!("{} {}", event.event, compact_json(&event.data, 72)),
                })
                .collect(),
            ChatSidebarPane::None => Vec::new(),
        }
    }

    fn current_chat_session_id(&self) -> Option<&str> {
        self.chat_session_id.as_deref()
    }

    fn current_chat_title(&self) -> String {
        self.current_chat_session_id()
            .map(|session_id| self.session_label(session_id, None))
            .unwrap_or_else(|| "No session selected".to_string())
    }

    fn current_chat_mode_label(&self) -> &'static str {
        match self.current_chat_session_id() {
            Some(session_id)
                if self
                    .dashboard
                    .live_sessions
                    .iter()
                    .any(|session| session.session_id == session_id) =>
            {
                "live"
            }
            Some(_) => "stored",
            None => "none",
        }
    }

    fn session_label(&self, session_id: &str, fallback_agent: Option<&str>) -> String {
        if let Some(detail) = self.dashboard.session_detail(session_id)
            && let Some(summary) = detail
                .messages
                .iter()
                .find(|message| message.role == "user")
                .and_then(|message| message_content_text(&message.content))
                .map(|text| excerpt(&text, 42))
        {
            return summary;
        }

        let agent_id = self
            .dashboard
            .live_sessions
            .iter()
            .find(|session| session.session_id == session_id)
            .map(|session| session.agent_id.as_str())
            .or_else(|| {
                self.dashboard
                    .sessions
                    .iter()
                    .find(|session| session.session_id == session_id)
                    .map(|session| session.agent_id.as_str())
            })
            .or(fallback_agent)
            .unwrap_or("session");
        format!("{agent_id}:{}", tail(session_id, 8))
    }

    fn apply_live_event(&mut self, event: &EventEnvelope) {
        let Some(session_id) = event
            .data
            .get("session_id")
            .and_then(|value| value.as_str())
        else {
            return;
        };
        let state = self
            .live_transcripts
            .entry(session_id.to_string())
            .or_default();
        match event.event.as_str() {
            "task_start" => {
                state.assistant_preview.clear();
                state.thinking_preview.clear();
                state.recent_tool_calls.clear();
                state.recent_events.clear();
            }
            "message_delta" => {
                if let Some(delta) = event
                    .data
                    .get("content_delta")
                    .and_then(|value| value.as_str())
                {
                    state.assistant_preview.push_str(delta);
                }
            }
            "thinking_delta" => {
                if let Some(delta) = event.data.get("thinking").and_then(|value| value.as_str()) {
                    state.thinking_preview.push_str(delta);
                }
            }
            "tool_call" => {
                let tool_name = event
                    .data
                    .get("tool_name")
                    .and_then(|value| value.as_str())
                    .unwrap_or("tool");
                push_bounded_line(
                    &mut state.recent_tool_calls,
                    format!("{} {}", tool_name, compact_json(&event.data, 72)),
                    8,
                );
            }
            "task_complete" => {
                self.dashboard.session_details.remove(session_id);
                self.requested_session_detail = None;
            }
            _ => {}
        }

        if matches!(
            event.event.as_str(),
            "task_start" | "task_complete" | "message_delta" | "thinking_delta" | "tool_call"
        ) {
            push_bounded_line(
                &mut state.recent_events,
                format!("{} {}", event.event, compact_json(&event.data, 96)),
                12,
            );
        }

        if self.settings.chat.follow_latest && self.current_chat_session_id() == Some(session_id) {
            self.chat_scroll_lines = 0;
        }
    }

    fn clear_live_transcript_for(&mut self, session_id: &str) {
        self.live_transcripts.remove(session_id);
    }

    fn clear_pending_messages_for(&mut self, session_id: &str) {
        if let Some(state) = self.live_transcripts.get_mut(session_id) {
            state.pending_user_messages.clear();
        }
    }

    fn note_prompt_submitted(&mut self, session_id: &str, prompt: &str) {
        let state = self
            .live_transcripts
            .entry(session_id.to_string())
            .or_default();
        push_bounded_line(
            &mut state.pending_user_messages,
            prompt.trim().to_string(),
            8,
        );
        state.assistant_preview.clear();
        state.thinking_preview.clear();
        self.dashboard.session_details.remove(session_id);
        self.requested_session_detail = None;
        self.chat_session_id = Some(session_id.to_string());
        self.chat_scroll_lines = 0;
    }

    fn input_len_chars(&self) -> usize {
        self.input.chars().count()
    }

    fn input_byte_index(&self, char_index: usize) -> usize {
        nth_char_byte_index(&self.input, char_index)
    }

    fn begin_input_mode(&mut self, mode: InputMode, initial: impl Into<String>) {
        self.input_mode = Some(mode);
        self.input = initial.into();
        self.input_cursor = self.input_len_chars();
    }

    fn insert_input_char(&mut self, ch: char) {
        let byte_index = self.input_byte_index(self.input_cursor);
        self.input.insert(byte_index, ch);
        self.input_cursor += 1;
    }

    fn insert_input_text(&mut self, text: &str) {
        let byte_index = self.input_byte_index(self.input_cursor);
        self.input.insert_str(byte_index, text);
        self.input_cursor += text.chars().count();
    }

    fn move_input_cursor_left(&mut self) {
        self.input_cursor = self.input_cursor.saturating_sub(1);
    }

    fn move_input_cursor_right(&mut self) {
        self.input_cursor = (self.input_cursor + 1).min(self.input_len_chars());
    }

    fn move_input_cursor_home(&mut self) {
        self.input_cursor = 0;
    }

    fn move_input_cursor_end(&mut self) {
        self.input_cursor = self.input_len_chars();
    }

    fn delete_input_left(&mut self) {
        if self.input_cursor == 0 {
            return;
        }
        let end = self.input_byte_index(self.input_cursor);
        let start = self.input_byte_index(self.input_cursor - 1);
        self.input.replace_range(start..end, "");
        self.input_cursor -= 1;
    }

    fn delete_input_right(&mut self) {
        if self.input_cursor >= self.input_len_chars() {
            return;
        }
        let start = self.input_byte_index(self.input_cursor);
        let end = self.input_byte_index(self.input_cursor + 1);
        self.input.replace_range(start..end, "");
    }

    fn input_title(&self) -> &'static str {
        match self.input_mode.as_ref() {
            Some(InputMode::SubmitPrompt { .. }) => "Prompt",
            Some(InputMode::ConfirmDiscard { .. }) => "Discard Changes",
            Some(InputMode::SaveProfile { .. }) => "Save Profile",
            Some(InputMode::DuplicateProfile { .. }) => "Duplicate Profile",
            Some(InputMode::RenameProfile { .. }) => "Rename Profile",
            Some(InputMode::ConfirmDelete { .. }) => "Delete Profile",
            Some(InputMode::EditDraftTarget) => "Edit Target",
            Some(InputMode::EditDraftAuth) => "Edit Auth",
            Some(InputMode::EditTaskFilter) => "Task Filter",
            Some(InputMode::EditChannelFilter) => "Channel Filter",
            Some(InputMode::EditEventFilter) => "Event Filter",
            Some(InputMode::EditTranscriptBudget) => "Transcript Budget",
            Some(InputMode::EditUserLabel) => "User Label",
            None => "Help",
        }
    }

    fn input_hint(&self) -> String {
        match self.input_mode.as_ref() {
            Some(InputMode::SubmitPrompt { .. }) => {
                "Enter submits. Esc cancels. Left/Right/Home/End move the cursor. Paste is inserted as one block.".to_string()
            }
            Some(InputMode::ConfirmDiscard { .. }) => {
                "Press y or Enter to discard changes. Press n or Esc to cancel.".to_string()
            }
            Some(InputMode::SaveProfile { make_default }) => {
                if *make_default {
                    "Enter saves the current draft under the typed name and marks it default. Esc cancels.".to_string()
                } else {
                    "Enter saves the current draft under the typed profile name. Esc cancels.".to_string()
                }
            }
            Some(InputMode::DuplicateProfile { source_name, make_default }) => {
                if *make_default {
                    format!(
                        "Enter duplicates '{}' to the typed name and sets it default. Esc cancels.",
                        source_name
                    )
                } else {
                    format!(
                        "Enter duplicates '{}' to the typed name. Esc cancels.",
                        source_name
                    )
                }
            }
            Some(InputMode::RenameProfile { source_name, make_default }) => {
                if *make_default {
                    format!(
                        "Enter renames '{}' and makes the new name default. Esc cancels.",
                        source_name
                    )
                } else {
                    format!("Enter renames '{}'. Esc cancels.", source_name)
                }
            }
            Some(InputMode::ConfirmDelete { .. }) => {
                "Press y or Enter to confirm delete. Press n or Esc to cancel.".to_string()
            }
            Some(InputMode::EditDraftTarget) => {
                "Enter updates the profile draft target. Esc cancels.".to_string()
            }
            Some(InputMode::EditDraftAuth) => {
                "Enter updates the profile draft auth value. Esc cancels.".to_string()
            }
            Some(InputMode::EditTaskFilter) => {
                "Enter updates the task filter. Esc cancels.".to_string()
            }
            Some(InputMode::EditChannelFilter) => {
                "Enter updates the channel filter. Esc cancels.".to_string()
            }
            Some(InputMode::EditEventFilter) => {
                "Enter updates the event filter. Esc cancels.".to_string()
            }
            Some(InputMode::EditTranscriptBudget) => {
                "Enter updates transcript memory budget in bytes (minimum 16384). Esc cancels."
                    .to_string()
            }
            Some(InputMode::EditUserLabel) => {
                "Enter updates the UI-only label used for your chat messages. Esc cancels."
                    .to_string()
            }
            None => String::new(),
        }
    }

    fn input_accepts_text(&self) -> bool {
        !matches!(
            self.input_mode.as_ref(),
            Some(InputMode::ConfirmDiscard { .. } | InputMode::ConfirmDelete { .. }) | None
        )
    }

    fn input_body_text(&self) -> String {
        match self.input_mode.as_ref() {
            Some(InputMode::ConfirmDiscard { action }) => action.description(),
            Some(InputMode::ConfirmDelete { profile_name }) => profile_name.clone(),
            _ => self.input.clone(),
        }
    }

    fn visible_input_editor(&self, area: Rect) -> (Vec<Line<'static>>, Option<Position>) {
        let available_width = area.width.max(1) as usize;
        let available_height = area.height.max(1) as usize;
        let body = self.input_body_text();
        let cursor_index = if self.input_accepts_text() {
            self.input_cursor
        } else {
            0
        };
        let (cursor_line, cursor_col) = cursor_line_col(&body, cursor_index);
        let horizontal_offset = cursor_col.saturating_sub(available_width.saturating_sub(1));
        let all_lines = split_input_lines(&body);
        let start_line = cursor_line.saturating_sub(available_height.saturating_sub(1));
        let end_line = (start_line + available_height).min(all_lines.len());

        let visible_lines = all_lines[start_line..end_line]
            .iter()
            .map(|line| {
                Line::from(slice_chars(
                    line,
                    horizontal_offset,
                    horizontal_offset + available_width,
                ))
            })
            .collect::<Vec<_>>();

        let cursor = if self.input_accepts_text() {
            Some(Position::new(
                area.x + cursor_col.saturating_sub(horizontal_offset) as u16,
                area.y + cursor_line.saturating_sub(start_line) as u16,
            ))
        } else {
            None
        };

        (visible_lines, cursor)
    }

    fn start_prompt_input(&mut self) {
        if let Some(session) = self.selected_live_session() {
            self.begin_input_mode(
                InputMode::SubmitPrompt {
                    session_id: session.session_id.clone(),
                },
                "",
            );
        }
    }

    fn start_chat_prompt_input(&mut self) {
        match self.current_chat_sidebar_item() {
            Some(ChatSidebarItem::Agent { agent_id, label }) => {
                self.pending_chat_prompt = Some(PendingChatPrompt::OpenAgent {
                    agent_id: agent_id.clone(),
                });
                self.dashboard.record_info(format!(
                    "Opening a live session for '{}' and waiting to start prompt input",
                    label
                ));
            }
            Some(ChatSidebarItem::StoredSession { session_id, label }) => {
                self.pending_chat_prompt = Some(PendingChatPrompt::ResumeSession {
                    session_id: session_id.clone(),
                });
                self.dashboard.record_info(format!(
                    "Resuming '{}' and waiting to start prompt input",
                    label
                ));
            }
            _ => {}
        }

        let Some(session_id) = self.current_chat_session_id().map(str::to_string) else {
            self.dashboard
                .record_error("No chat session is currently selected");
            return;
        };
        if self
            .dashboard
            .live_sessions
            .iter()
            .any(|session| session.session_id == session_id)
        {
            self.begin_input_mode(InputMode::SubmitPrompt { session_id }, "");
            self.pending_chat_prompt = None;
        } else if self.pending_chat_prompt.is_none() {
            self.dashboard.record_error(
                "The selected chat session is not live. Resume it first, or open a fresh live session.",
            );
        }
    }

    fn start_save_profile_input(&mut self, make_default: bool) {
        self.begin_input_mode(InputMode::SaveProfile { make_default }, "");
    }

    fn start_edit_draft_target(&mut self) {
        self.begin_input_mode(
            InputMode::EditDraftTarget,
            self.profile_draft.target.clone(),
        );
    }

    fn start_edit_draft_auth(&mut self) {
        if self.profile_draft.kind != ConnectionProfileKind::Remote {
            self.dashboard
                .record_error("Draft auth can only be edited for remote profiles");
            return;
        }
        if self.profile_draft.auth_mode == ConnectionProfileDraftAuthMode::None {
            self.dashboard.record_error(
                "Draft auth value can only be edited after choosing env or inline auth mode",
            );
            return;
        }
        self.begin_input_mode(
            InputMode::EditDraftAuth,
            self.profile_draft.auth_value.clone(),
        );
    }

    fn start_edit_task_filter(&mut self) {
        self.begin_input_mode(InputMode::EditTaskFilter, self.task_filter.clone());
    }

    fn start_edit_channel_filter(&mut self) {
        self.begin_input_mode(InputMode::EditChannelFilter, self.channel_filter.clone());
    }

    fn start_edit_event_filter(&mut self) {
        self.begin_input_mode(InputMode::EditEventFilter, self.event_filter.clone());
    }

    fn clear_task_filter(&mut self) {
        self.task_filter.clear();
        self.clamp_selection();
        self.dashboard.record_info("Cleared the task filter");
    }

    fn clear_channel_filter(&mut self) {
        self.channel_filter.clear();
        self.clamp_selection();
        self.dashboard.record_info("Cleared the channel filter");
    }

    fn clear_event_filter(&mut self) {
        self.event_filter.clear();
        self.clamp_selection();
        self.dashboard.record_info("Cleared the event filter");
    }

    fn toggle_events_paused(&mut self) {
        self.events_paused = !self.events_paused;
        if self.events_paused {
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
        self.clamp_selection();
    }

    fn toggle_events_follow_latest(&mut self) {
        self.events_follow_latest = !self.events_follow_latest;
        if self.events_follow_latest {
            self.event_index = 0;
        }
        self.dashboard.record_info(format!(
            "Event follow-latest is now {}",
            if self.events_follow_latest {
                "on"
            } else {
                "off"
            }
        ));
    }

    fn jump_latest_event(&mut self) {
        self.event_index = 0;
        self.dashboard
            .record_info("Jumped to the latest visible event");
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

    fn start_duplicate_profile_input(&mut self, make_default: bool) {
        let Some(source_name) = self.selected_profile().map(|profile| profile.name.clone()) else {
            self.dashboard
                .record_error("No connection profile is currently selected");
            return;
        };
        self.begin_input_mode(
            InputMode::DuplicateProfile {
                source_name: source_name.clone(),
                make_default,
            },
            format!("{source_name}-copy"),
        );
    }

    fn start_rename_profile_input(&mut self, make_default: bool) {
        let Some(source_name) = self.selected_profile().map(|profile| profile.name.clone()) else {
            self.dashboard
                .record_error("No connection profile is currently selected");
            return;
        };
        self.begin_input_mode(
            InputMode::RenameProfile {
                source_name: source_name.clone(),
                make_default,
            },
            source_name,
        );
    }

    fn start_delete_confirmation(&mut self) {
        let Some(profile_name) = self.selected_profile().map(|profile| profile.name.clone()) else {
            self.dashboard
                .record_error("No connection profile is currently selected");
            return;
        };
        self.begin_input_mode(
            InputMode::ConfirmDelete {
                profile_name: profile_name.clone(),
            },
            "",
        );
        self.dashboard.record_info(format!(
            "Delete confirmation armed for connection profile '{}'",
            profile_name
        ));
    }

    fn clear_input_mode(&mut self) {
        self.input_mode = None;
        self.input.clear();
        self.input_cursor = 0;
    }

    fn handle_chat_enter(
        &mut self,
        command_tx: &tokio::sync::mpsc::UnboundedSender<OperatorCommand>,
    ) -> Result<()> {
        match self.current_chat_sidebar_item() {
            Some(ChatSidebarItem::Agent { agent_id, .. }) => {
                self.pending_chat_prompt = Some(PendingChatPrompt::OpenAgent {
                    agent_id: agent_id.clone(),
                });
                self.dashboard.record_info(format!(
                    "Opening a live session for '{}' and waiting to start prompt input",
                    agent_id
                ));
                send_command(command_tx, OperatorCommand::OpenSession { agent_id })
            }
            Some(ChatSidebarItem::StoredSession { session_id, .. }) => {
                self.pending_chat_prompt = Some(PendingChatPrompt::ResumeSession {
                    session_id: session_id.clone(),
                });
                self.dashboard.record_info(format!(
                    "Resuming '{}' and waiting to start prompt input",
                    tail(&session_id, 8)
                ));
                send_command(command_tx, OperatorCommand::ResumeSession { session_id })
            }
            _ => {
                self.start_chat_prompt_input();
                Ok(())
            }
        }
    }

    fn cycle_left_chat_pane(&mut self) {
        self.settings.layout.left_pane = self.settings.layout.left_pane.next();
        self.chat_sidebar_index = 0;
        self.clamp_selection();
        self.dashboard.record_info(format!(
            "Chat left pane is now {}",
            self.settings.layout.left_pane.title()
        ));
    }

    fn cycle_right_chat_pane(&mut self) {
        self.settings.layout.right_pane = self.settings.layout.right_pane.next();
        self.dashboard.record_info(format!(
            "Chat right pane is now {}",
            self.settings.layout.right_pane.title()
        ));
    }

    fn toggle_show_thinking(&mut self) {
        self.settings.chat.show_thinking = !self.settings.chat.show_thinking;
        self.dashboard.record_info(format!(
            "Thinking pane is now {}",
            if self.settings.chat.show_thinking {
                "visible"
            } else {
                "hidden"
            }
        ));
    }

    fn toggle_streaming_preview(&mut self) {
        self.settings.chat.show_streaming_preview = !self.settings.chat.show_streaming_preview;
        self.dashboard.record_info(format!(
            "Streaming preview is now {}",
            if self.settings.chat.show_streaming_preview {
                "visible"
            } else {
                "hidden"
            }
        ));
    }

    fn toggle_chat_follow_latest(&mut self) {
        self.settings.chat.follow_latest = !self.settings.chat.follow_latest;
        if self.settings.chat.follow_latest {
            self.chat_scroll_lines = 0;
        }
        self.dashboard.record_info(format!(
            "Chat follow-latest is now {}",
            if self.settings.chat.follow_latest {
                "on"
            } else {
                "off"
            }
        ));
    }

    fn scroll_chat(&mut self, delta_from_bottom: i16) {
        let next = self.chat_scroll_lines as i16 + delta_from_bottom;
        self.chat_scroll_lines = next.max(0) as u16;
        if self.chat_scroll_lines > 0 {
            self.settings.chat.follow_latest = false;
        }
    }

    fn jump_chat_latest(&mut self) {
        self.chat_scroll_lines = 0;
        self.settings.chat.follow_latest = true;
        self.dashboard
            .record_info("Chat view jumped back to the latest output");
    }

    fn jump_chat_oldest(&mut self) {
        self.chat_scroll_lines = u16::MAX / 2;
        self.settings.chat.follow_latest = false;
        self.dashboard
            .record_info("Chat view jumped toward the oldest loaded transcript lines");
    }

    fn save_settings(&mut self) {
        match save_settings(&self.settings_path, &self.settings) {
            Ok(()) => self.dashboard.record_info(format!(
                "Saved TUI settings to '{}'",
                self.settings_path.display()
            )),
            Err(err) => self
                .dashboard
                .record_error(format!("Failed to save TUI settings: {err}")),
        }
    }

    fn settings_items(&self) -> Vec<String> {
        vec![
            format!("Left Pane: {}", self.settings.layout.left_pane.title()),
            format!("Right Pane: {}", self.settings.layout.right_pane.title()),
            format!(
                "Streaming Preview: {}",
                if self.settings.chat.show_streaming_preview {
                    "on"
                } else {
                    "off"
                }
            ),
            format!(
                "Thinking Pane: {}",
                if self.settings.chat.show_thinking {
                    "on"
                } else {
                    "off"
                }
            ),
            format!(
                "Follow Latest: {}",
                if self.settings.chat.follow_latest {
                    "on"
                } else {
                    "off"
                }
            ),
            format!("User Label: {}", self.settings.chat.user_label),
            format!(
                "Transcript Budget: {} bytes",
                self.settings.chat.transcript_memory_budget_bytes
            ),
        ]
    }

    fn activate_selected_setting(&mut self) {
        match self.settings_index {
            0 => self.cycle_left_chat_pane(),
            1 => self.cycle_right_chat_pane(),
            2 => self.toggle_streaming_preview(),
            3 => self.toggle_show_thinking(),
            4 => self.toggle_chat_follow_latest(),
            5 => self.start_edit_user_label(),
            6 => self.start_edit_transcript_budget(),
            _ => {}
        }
    }

    fn start_edit_user_label(&mut self) {
        self.begin_input_mode(
            InputMode::EditUserLabel,
            self.settings.chat.user_label.clone(),
        );
    }

    fn start_edit_transcript_budget(&mut self) {
        self.begin_input_mode(
            InputMode::EditTranscriptBudget,
            self.settings
                .chat
                .transcript_memory_budget_bytes
                .to_string(),
        );
    }

    fn chat_transcript_text(
        &self,
        viewport_height: usize,
        viewport_width: usize,
    ) -> (Text<'static>, u16) {
        let Some(session_id) = self.current_chat_session_id() else {
            return (
                Text::from(vec![
                    Line::from("No chat session selected."),
                    Line::from(
                        "Use the left pane to pick a session, or switch the left pane to Agents and press Enter to open one.",
                    ),
                ]),
                0,
            );
        };

        let mut lines = self.build_transcript_lines(session_id);
        let budget = self
            .settings
            .chat
            .transcript_memory_budget_bytes
            .max(16 * 1024);
        trim_lines_to_budget(&mut lines, budget);
        let lines = wrap_lines_for_width(lines, viewport_width.max(1));

        let total_lines = lines.len();
        let visible_lines = viewport_height.max(1);
        let scroll_from_top = total_lines
            .saturating_sub(visible_lines.saturating_add(self.chat_scroll_lines as usize));
        (
            Text::from(lines),
            scroll_from_top.min(u16::MAX as usize) as u16,
        )
    }

    fn build_transcript_lines(&self, session_id: &str) -> Vec<Line<'static>> {
        let mut lines = Vec::new();
        if let Some(detail) = self.dashboard.session_detail(session_id) {
            for message in &detail.messages {
                let content = message_content_text(&message.content)
                    .unwrap_or_else(|| compact_json(&message.content, 120));
                self.push_message_block(
                    &mut lines,
                    message.role.as_str(),
                    Some(format!("turn {}", message.turn_index)),
                    &content,
                );
            }
        }

        if let Some(state) = self.live_transcripts.get(session_id) {
            for prompt in &state.pending_user_messages {
                self.push_message_block(&mut lines, "user", Some("pending".to_string()), prompt);
            }

            if self.settings.chat.show_streaming_preview
                && !state.assistant_preview.trim().is_empty()
            {
                self.push_message_block(
                    &mut lines,
                    "assistant",
                    Some("streaming".to_string()),
                    &state.assistant_preview,
                );
            }
        }

        if lines.is_empty() {
            lines.push(Line::from("No transcript has been loaded yet."));
        }
        lines
    }

    fn push_message_block(
        &self,
        lines: &mut Vec<Line<'static>>,
        role: &str,
        status: Option<String>,
        content: &str,
    ) {
        let (label, color) = self.chat_role_descriptor(role);
        let heading = match status {
            Some(status) => format!("── {label} · {status}"),
            None => format!("── {label}"),
        };
        lines.push(Line::from(Span::styled(
            heading,
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        )));
        let body_prefix = format!("{label} │ ");
        for body_line in content.lines() {
            lines.push(Line::from(format!("{body_prefix}{body_line}")));
        }
        if content.is_empty() {
            lines.push(Line::from(body_prefix));
        }
        lines.push(Line::default());
    }

    fn chat_role_descriptor(&self, role: &str) -> (String, Color) {
        match role {
            "user" => (self.settings.chat.user_label.clone(), Color::LightBlue),
            "assistant" => ("Assistant".to_string(), Color::LightGreen),
            "system" => ("System".to_string(), Color::Yellow),
            _ => (role.to_string(), Color::White),
        }
    }

    fn current_thinking_text(&self) -> String {
        if !self.settings.chat.show_thinking {
            return "Thinking pane is hidden in settings.".to_string();
        }

        let Some(session_id) = self.current_chat_session_id() else {
            return "No chat session selected.".to_string();
        };
        let Some(state) = self.live_transcripts.get(session_id) else {
            return "No streamed thinking for the selected session yet.".to_string();
        };
        if state.thinking_preview.trim().is_empty() {
            "No streamed thinking for the selected session yet.".to_string()
        } else {
            state.thinking_preview.clone()
        }
    }

    fn current_tool_text(&self) -> String {
        let Some(session_id) = self.current_chat_session_id() else {
            return "No chat session selected.".to_string();
        };
        let mut lines = Vec::new();
        if let Some(detail) = self.dashboard.session_detail(session_id) {
            for tool in detail.tool_executions.iter().rev().take(8).rev() {
                lines.push(format!(
                    "{} · {} · {}ms",
                    tool.tool_name,
                    tool.verdict,
                    tool.duration_ms.unwrap_or(0)
                ));
            }
        }
        if let Some(state) = self.live_transcripts.get(session_id) {
            lines.extend(state.recent_tool_calls.iter().cloned());
        }
        if lines.is_empty() {
            "No tool activity for the selected session yet.".to_string()
        } else {
            lines.join("\n")
        }
    }

    fn current_chat_event_text(&self) -> String {
        let Some(session_id) = self.current_chat_session_id() else {
            return "No chat session selected.".to_string();
        };
        if let Some(state) = self.live_transcripts.get(session_id)
            && !state.recent_events.is_empty()
        {
            return state
                .recent_events
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join("\n");
        }
        "No live session events for the selected chat session yet.".to_string()
    }

    fn current_session_meta_text(&self) -> String {
        let Some(session_id) = self.current_chat_session_id() else {
            return "No chat session selected.".to_string();
        };
        let detail = self.dashboard.session_detail(session_id);
        pretty_json(&serde_json::json!({
            "session_id": session_id,
            "title": self.session_label(session_id, None),
            "mode": self.current_chat_mode_label(),
            "loaded_detail": detail.is_some(),
            "messages": detail.map(|value| value.messages.len()).unwrap_or(0),
            "events": detail.map(|value| value.events.len()).unwrap_or(0),
            "tool_calls": detail.map(|value| value.tool_executions.len()).unwrap_or(0),
        }))
    }

    fn list_items(&self) -> Vec<ListItem<'static>> {
        match self.tab {
            TabKind::Chat => self
                .chat_sidebar_items()
                .iter()
                .map(|item| ListItem::new(item.label().to_string()))
                .collect(),
            TabKind::Connections => self
                .profile_catalog
                .as_ref()
                .map(|catalog| {
                    catalog
                        .profiles()
                        .iter()
                        .map(|profile| {
                            let default = if profile.is_default { " default" } else { "" };
                            ListItem::new(format!(
                                "{} [{} | {}{}]",
                                profile.name,
                                profile_kind_label(profile.kind),
                                profile_auth_label(profile.auth.as_ref()),
                                default
                            ))
                        })
                        .collect()
                })
                .unwrap_or_default(),
            TabKind::Agents => self
                .dashboard
                .agents()
                .iter()
                .map(|agent| {
                    let runtime = self.agent_runtime(&agent.id);
                    let summary = format!(
                        "{}  {}  active:{} queued:{}",
                        agent.id,
                        if runtime.is_some_and(|runtime| runtime.running) {
                            "running"
                        } else {
                            "idle"
                        },
                        runtime.map(|runtime| runtime.active_tasks).unwrap_or(0),
                        runtime.map(|runtime| runtime.queued_tasks).unwrap_or(0)
                    );
                    ListItem::new(summary)
                })
                .collect(),
            TabKind::LiveSessions => self
                .dashboard
                .live_sessions
                .iter()
                .map(|session| {
                    ListItem::new(format!(
                        "{} [{}]  active:{} queued:{}",
                        session.agent_id,
                        session.slot_id,
                        session.active_tasks,
                        session.queued_tasks
                    ))
                })
                .collect(),
            TabKind::Sessions => self
                .dashboard
                .sessions
                .iter()
                .map(|session| {
                    ListItem::new(format!(
                        "{}  {}  {}",
                        session.session_id, session.agent_id, session.created_at
                    ))
                })
                .collect(),
            TabKind::Tasks => self
                .filtered_tasks()
                .iter()
                .map(|task| {
                    ListItem::new(format!(
                        "{}  {}  {}",
                        task.request_id, task.state, task.agent_id
                    ))
                })
                .collect(),
            TabKind::Channels => self
                .filtered_channels()
                .iter()
                .map(|channel| {
                    ListItem::new(format!(
                        "{}  {}  {}",
                        channel.id,
                        channel.kind,
                        if channel.enabled {
                            "enabled"
                        } else {
                            "disabled"
                        }
                    ))
                })
                .collect(),
            TabKind::Events => self
                .filtered_events()
                .iter()
                .map(|event| ListItem::new(event.event.clone()))
                .collect(),
            TabKind::Settings => self
                .settings_items()
                .into_iter()
                .map(ListItem::new)
                .collect(),
        }
    }

    fn detail_text(&self) -> String {
        match self.tab {
            TabKind::Chat => pretty_json(&serde_json::json!({
                "chat_session_id": self.current_chat_session_id(),
                "title": self.current_chat_title(),
                "mode": self.current_chat_mode_label(),
                "left_pane": self.settings.layout.left_pane.title(),
                "right_pane": self.settings.layout.right_pane.title(),
                "show_streaming_preview": self.settings.chat.show_streaming_preview,
                "show_thinking": self.settings.chat.show_thinking,
                "follow_latest": self.settings.chat.follow_latest,
                "user_label": self.settings.chat.user_label,
                "transcript_budget_bytes": self.settings.chat.transcript_memory_budget_bytes,
            })),
            TabKind::Connections => {
                let validation = self.profile_draft_validation();
                let editor_diff = self.editor_diff();
                let selected_diff = self.selected_profile_diff();
                pretty_json(&serde_json::json!({
                    "current_connection": {
                        "kind": connection_kind_label(self.dashboard.connection_kind),
                        "target": self.dashboard.connection_target,
                        "active_profile": self.active_connection_label(),
                        "profiles_source": self.connection_options.profiles_path().display().to_string(),
                        "snapshot_freshness": freshness_label(self.dashboard.snapshot_freshness()),
                        "last_snapshot": self.dashboard.snapshot_age_label(),
                        "last_event": self.dashboard.event_age_label(),
                        "last_notice": self.dashboard.notice_age_label(),
                        "total_events": self.dashboard.total_event_count,
                        "refresh_success_count": self.dashboard.refresh_success_count,
                        "refresh_failure_count": self.dashboard.refresh_failure_count,
                        "last_refresh_status": self.dashboard.last_refresh_status_label(),
                        "last_refresh_latency": self.dashboard.last_refresh_latency_label(),
                        "transport": self.dashboard.health.as_ref().map(|health| health.transport.clone()),
                        "wire_format": self.dashboard.health.as_ref().map(|health| health.wire_format.clone()),
                    },
                    "selected_profile": self.selected_profile().map(|profile| serde_json::json!({
                        "name": profile.name,
                        "kind": profile_kind_label(profile.kind),
                        "target": profile.target,
                        "auth": profile_auth_label(profile.auth.as_ref()),
                        "default": profile.is_default,
                    })),
                    "selected_recent_draft": self.selected_recent_draft().map(|draft| draft.summary_label()),
                    "profile_draft": {
                        "kind": profile_kind_label(self.profile_draft.kind),
                        "target": self.profile_draft.target,
                        "auth_mode": profile_draft_auth_label(self.profile_draft.auth_mode),
                        "auth_value": if self.profile_draft.auth_mode == ConnectionProfileDraftAuthMode::InlineToken {
                            "<hidden>"
                        } else {
                            self.profile_draft.auth_value.as_str()
                        },
                    "validation": {
                        "status": if validation.is_valid() { "valid" } else { "invalid" },
                        "connect_ready": validation.is_valid(),
                        "summary": validation.summary(),
                        "target_error": validation.target_error,
                        "auth_error": validation.auth_error,
                        "target_notice": validation.target_notice,
                        "auth_notice": validation.auth_notice,
                    },
                    "dirty": self.editor_is_dirty(),
                    "baseline": self.draft_baseline_label,
                    "changes": editor_diff,
                    "selected_update_ready": selected_diff.as_ref().is_some_and(|diff| !diff.is_empty() && validation.is_valid()),
                },
                "selected_profile_diff": selected_diff,
                    "selected_profile_activity": self
                        .selected_profile()
                        .and_then(|profile| self.profile_activity.entry(&profile.name)),
                "latest_preflight": self.last_preflight_report.as_ref().map(|report| serde_json::json!({
                    "outcome": preflight_outcome_label(report.outcome),
                    "target": report.target,
                    "auth": report.auth,
                    "message": report.message,
                    "latency_ms": report.latency_ms,
                    "ready": report.ready,
                    "transport": report.transport,
                    "wire_format": report.wire_format,
                })),
                "recent_drafts": self
                    .recent_drafts
                    .drafts()
                    .iter()
                    .enumerate()
                    .map(|(index, draft)| serde_json::json!({
                        "selected": index == self.recent_draft_index,
                        "summary": draft.summary_label(),
                    }))
                    .collect::<Vec<_>>(),
                "available_profiles": self.profile_catalog.as_ref().map(|catalog| catalog.profiles().len()).unwrap_or(0),
                "last_error": self.dashboard.last_error.clone(),
                "last_info": self.dashboard.last_info.clone(),
                "recent_notices": self
                        .dashboard
                        .recent_notices
                        .iter()
                        .rev()
                        .take(6)
                        .collect::<Vec<_>>(),
                }))
            }
            TabKind::Agents => self
                .selected_agent()
                .map(|agent| {
                    let runtime = self.agent_runtime(&agent.id);
                    pretty_json(&serde_json::json!({
                        "agent": agent,
                        "runtime": runtime,
                    }))
                })
                .unwrap_or_else(|| "No agents available.".to_string()),
            TabKind::LiveSessions => self
                .selected_live_session()
                .map(|session| {
                    pretty_json(&serde_json::json!({
                        "live_session": session,
                        "detail": session_detail_preview(self.dashboard.session_detail(&session.session_id)),
                    }))
                })
                .unwrap_or_else(|| "No live sessions available.".to_string()),
            TabKind::Sessions => self
                .selected_persisted_session()
                .map(|session| {
                    pretty_json(&serde_json::json!({
                        "session": session,
                        "detail": session_detail_preview(self.dashboard.session_detail(&session.session_id)),
                    }))
                })
                .unwrap_or_else(|| "No stored sessions available.".to_string()),
            TabKind::Tasks => self
                .selected_task()
                .map(|task| {
                    pretty_json(&serde_json::json!({
                        "filter": self.task_filter,
                        "task": task,
                    }))
                })
                .unwrap_or_else(|| "No tasks available.".to_string()),
            TabKind::Channels => self
                .selected_channel()
                .map(|channel| {
                    let runtime = self.channel_runtime(&channel.id);
                    pretty_json(&serde_json::json!({
                        "filter": self.channel_filter,
                        "channel": channel,
                        "runtime": runtime,
                    }))
                })
                .unwrap_or_else(|| "No channels available.".to_string()),
            TabKind::Events => self
                .selected_event()
                .map(|event| {
                    pretty_json(&serde_json::json!({
                        "filter": self.event_filter,
                        "paused": self.events_paused,
                        "follow_latest": self.events_follow_latest,
                        "visible_events": self.filtered_events().len(),
                        "event": event,
                    }))
                })
                .unwrap_or_else(|| "No events yet.".to_string()),
            TabKind::Settings => pretty_json(&serde_json::json!({
                "path": self.settings_path.display().to_string(),
                "layout": {
                    "left_pane": self.settings.layout.left_pane.title(),
                    "right_pane": self.settings.layout.right_pane.title(),
                },
                "chat": {
                    "transcript_memory_budget_bytes": self.settings.chat.transcript_memory_budget_bytes,
                    "show_streaming_preview": self.settings.chat.show_streaming_preview,
                    "show_thinking": self.settings.chat.show_thinking,
                    "follow_latest": self.settings.chat.follow_latest,
                    "user_label": self.settings.chat.user_label,
                },
                "save_hint": "Press w to persist settings",
            })),
        }
    }

    fn help_text(&self) -> String {
        let shared = "1-9 switch views | Tab cycle | arrows/j/k move | r refresh | q quit";
        let scoped = match self.tab {
            TabKind::Chat => {
                "Enter opens/resumes or prompts | p prompt | ,/. cycle panes | h thinking | v preview | f follow-latest | PgUp/PgDn scroll | Home/End jump"
            }
            TabKind::Connections => {
                "Enter/s connect selected | C connect draft | T test draft | P test selected | E ensure draft local | S update selected | R load recent | [/ ] pick recent | v load current | b load selected | m/o cycle draft | t/g edit draft | a/A save as named | y/Y duplicate | u/U rename | d delete(confirm) | l reload"
            }
            TabKind::Agents => "n or Enter opens a live session for the selected agent",
            TabKind::LiveSessions => "p or Enter prompts | c cancel session | x kill session",
            TabKind::Sessions => "e or Enter resumes the selected stored session",
            TabKind::Tasks => "/ edit filter | F clear filter | c cancels the selected task",
            TabKind::Channels => "/ edit filter | F clear filter | channel view is read-only",
            TabKind::Events => {
                "/ edit filter | F clear filter | z pause | f follow latest | G latest"
            }
            TabKind::Settings => {
                "Enter toggles/cycles the selected setting | b edits transcript budget | w saves settings"
            }
        };
        format!("{} | {}", shared, scoped)
    }

    fn agent_runtime(&self, agent_id: &str) -> Option<&AgentRuntime> {
        self.dashboard
            .status
            .as_ref()?
            .agent_runtimes
            .iter()
            .find(|runtime| runtime.agent_id == agent_id)
    }

    fn channel_runtime(&self, channel_id: &str) -> Option<&ChannelRuntime> {
        self.dashboard
            .status
            .as_ref()?
            .channel_runtimes
            .iter()
            .find(|runtime| runtime.id == channel_id)
    }

    fn selected_event(&self) -> Option<EventEnvelope> {
        self.filtered_events().get(self.event_index).cloned()
    }

    fn current_detail_session_id(&self) -> Option<String> {
        match self.tab {
            TabKind::Chat => self.current_chat_session_id().map(str::to_string),
            TabKind::LiveSessions => self
                .selected_live_session()
                .map(|session| session.session_id.clone()),
            TabKind::Sessions => self
                .selected_persisted_session()
                .map(|session| session.session_id.clone()),
            _ => None,
        }
    }

    fn ensure_session_detail_loaded(
        &mut self,
        command_tx: &tokio::sync::mpsc::UnboundedSender<OperatorCommand>,
    ) -> Result<()> {
        let Some(session_id) = self.current_detail_session_id() else {
            self.requested_session_detail = None;
            return Ok(());
        };

        if self.dashboard.session_detail(&session_id).is_some() {
            self.requested_session_detail = Some(session_id);
            return Ok(());
        }

        if self.requested_session_detail.as_deref() == Some(session_id.as_str()) {
            return Ok(());
        }

        self.requested_session_detail = Some(session_id.clone());
        send_command(
            command_tx,
            OperatorCommand::LoadSessionDetail { session_id },
        )
    }
}

fn nth_char_byte_index(value: &str, char_index: usize) -> usize {
    value
        .char_indices()
        .nth(char_index)
        .map(|(index, _)| index)
        .unwrap_or_else(|| value.len())
}

fn split_input_lines(value: &str) -> Vec<String> {
    let mut lines = value
        .split('\n')
        .map(std::string::ToString::to_string)
        .collect::<Vec<_>>();
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn wrap_lines_for_width(lines: Vec<Line<'static>>, width: usize) -> Vec<Line<'static>> {
    let width = width.max(1);
    let mut wrapped = Vec::new();
    for line in lines {
        let style = line
            .spans
            .first()
            .map(|span| span.style)
            .unwrap_or(line.style);
        let text = line
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();
        if text.is_empty() {
            wrapped.push(Line::default());
            continue;
        }
        for chunk in wrap_text_chunk(&text, width) {
            wrapped.push(Line::from(Span::styled(chunk, style)));
        }
    }
    wrapped
}

fn wrap_text_chunk(value: &str, width: usize) -> Vec<String> {
    if value.chars().count() <= width {
        return vec![value.to_string()];
    }

    let mut remaining = value.trim_end_matches('\n').to_string();
    let mut wrapped = Vec::new();
    while !remaining.is_empty() {
        let remaining_len = remaining.chars().count();
        if remaining_len <= width {
            wrapped.push(remaining);
            break;
        }

        let candidate = slice_chars(&remaining, 0, width);
        let next_char_is_whitespace = remaining
            .chars()
            .nth(width)
            .is_some_and(char::is_whitespace);
        let split_at = if next_char_is_whitespace {
            width
        } else {
            candidate
                .char_indices()
                .rev()
                .find(|(_, ch)| ch.is_whitespace())
                .map(|(index, _)| candidate[..index].chars().count())
                .filter(|count| *count > 0)
                .unwrap_or(width)
        };

        wrapped.push(slice_chars(&remaining, 0, split_at).trim_end().to_string());
        remaining = slice_chars(&remaining, split_at, remaining_len)
            .trim_start()
            .to_string();
    }
    wrapped
}

fn cursor_line_col(value: &str, cursor_chars: usize) -> (usize, usize) {
    let mut line = 0usize;
    let mut col = 0usize;
    for (index, ch) in value.chars().enumerate() {
        if index == cursor_chars {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
    }
    (line, col)
}

fn slice_chars(value: &str, start: usize, end: usize) -> String {
    value
        .chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect()
}

fn clamp_index(index: usize, len: usize) -> usize {
    if len == 0 { 0 } else { index.min(len - 1) }
}

fn pretty_json<T: Serialize>(value: &T) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| "<unserializable>".to_string())
}

fn session_detail_preview(
    detail: Option<&turin_control_client::SessionDetail>,
) -> Option<serde_json::Value> {
    detail.map(|detail| {
        serde_json::json!({
            "session": {
                "messages": detail.messages.len(),
                "events": detail.events.len(),
                "tool_calls": detail.tool_executions.len(),
            },
            "recent_messages": detail.messages.iter().rev().take(4).rev().map(|message| serde_json::json!({
                "role": message.role,
                "turn": message.turn_index,
                "content": message.content,
            })).collect::<Vec<_>>(),
            "recent_events": detail.events.iter().rev().take(3).rev().map(|event| serde_json::json!({
                "type": event.event_type,
                "created_at": event.created_at,
                "payload": event.payload,
            })).collect::<Vec<_>>(),
            "recent_tools": detail.tool_executions.iter().rev().take(3).rev().map(|tool| serde_json::json!({
                "tool": tool.tool_name,
                "verdict": tool.verdict,
                "duration_ms": tool.duration_ms,
            })).collect::<Vec<_>>(),
        })
    })
}

fn compact_json(value: &Value, max_chars: usize) -> String {
    excerpt(
        &serde_json::to_string(value).unwrap_or_else(|_| "{}".to_string()),
        max_chars,
    )
}

fn message_content_text(value: &Value) -> Option<String> {
    if let Some(text) = value.as_str() {
        return Some(text.to_string());
    }
    if let Some(content) = value.get("text").and_then(|inner| inner.as_str()) {
        return Some(content.to_string());
    }
    if let Some(items) = value.as_array() {
        let mut parts = Vec::new();
        for item in items {
            if let Some(text) = item.get("text").and_then(|inner| inner.as_str()) {
                parts.push(text.to_string());
            }
        }
        if !parts.is_empty() {
            return Some(parts.join("\n"));
        }
    }
    None
}

fn excerpt(value: &str, max_chars: usize) -> String {
    let trimmed = value.trim();
    let mut out = String::new();
    for ch in trimmed.chars() {
        if out.chars().count() >= max_chars {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

fn tail(value: &str, max_chars: usize) -> String {
    let chars = value.chars().collect::<Vec<_>>();
    if chars.len() <= max_chars {
        return value.to_string();
    }
    chars[chars.len() - max_chars..].iter().collect()
}

fn push_bounded_line(queue: &mut VecDeque<String>, value: String, max_items: usize) {
    if value.trim().is_empty() {
        return;
    }
    queue.push_back(value);
    while queue.len() > max_items {
        queue.pop_front();
    }
}

fn trim_lines_to_budget(lines: &mut Vec<Line<'static>>, budget_bytes: usize) {
    let mut total = 0usize;
    let mut kept = Vec::new();
    for line in lines.iter().rev() {
        let line_text = line
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();
        total = total.saturating_add(line_text.len() + 1);
        if total > budget_bytes && !kept.is_empty() {
            break;
        }
        kept.push(line.clone());
    }
    kept.reverse();
    *lines = kept;
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

fn freshness_color(freshness: DashboardFreshness) -> Color {
    match freshness {
        DashboardFreshness::Fresh => Color::LightGreen,
        DashboardFreshness::Quiet => Color::Yellow,
        DashboardFreshness::Stale => Color::LightRed,
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

#[cfg(test)]
mod tests {
    use super::{
        cursor_line_col, nth_char_byte_index, slice_chars, split_input_lines, wrap_text_chunk,
    };

    #[test]
    fn nth_char_byte_index_handles_unicode_boundaries() {
        let value = "aé中";
        assert_eq!(nth_char_byte_index(value, 0), 0);
        assert_eq!(nth_char_byte_index(value, 1), 1);
        assert_eq!(nth_char_byte_index(value, 2), 3);
        assert_eq!(nth_char_byte_index(value, 3), value.len());
    }

    #[test]
    fn cursor_line_col_tracks_multiline_input() {
        let value = "one\ntwo\nthree";
        assert_eq!(cursor_line_col(value, 0), (0, 0));
        assert_eq!(cursor_line_col(value, 3), (0, 3));
        assert_eq!(cursor_line_col(value, 4), (1, 0));
        assert_eq!(cursor_line_col(value, 7), (1, 3));
        assert_eq!(cursor_line_col(value, 8), (2, 0));
        assert_eq!(cursor_line_col(value, value.chars().count()), (2, 5));
    }

    #[test]
    fn split_input_lines_preserves_empty_input_and_newlines() {
        assert_eq!(split_input_lines(""), vec![String::new()]);
        assert_eq!(
            split_input_lines("a\nb\n"),
            vec!["a".to_string(), "b".to_string(), String::new()]
        );
    }

    #[test]
    fn slice_chars_respects_character_offsets() {
        assert_eq!(slice_chars("abcdef", 1, 4), "bcd");
        assert_eq!(slice_chars("aé中", 1, 3), "é中");
    }

    #[test]
    fn wrap_text_chunk_prefers_word_boundaries() {
        assert_eq!(
            wrap_text_chunk("alpha beta gamma", 10),
            vec!["alpha beta".to_string(), "gamma".to_string()]
        );
    }
}
