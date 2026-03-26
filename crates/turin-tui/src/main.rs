mod settings;

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event as CEvent, KeyCode, KeyEventKind, MouseEvent, MouseEventKind,
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
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use turin_control_client::{
    AgentRuntime, AgentSummary, ChannelRuntime, ChannelSummary, ConnectionKind, LiveSession,
    SessionSearchHit as PersistedSessionSearchHit, SessionSummary, TaskStatus,
};
use turin_daemon_protocol::{EventEnvelope, SessionSearchHitKind, SessionSearchScope};
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
    Search,
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
    const ALL: [Self; 10] = [
        Self::Chat,
        Self::Search,
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
            Self::Search => "Search",
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
            '2' => Some(Self::Search),
            '3' => Some(Self::Connections),
            '4' => Some(Self::Agents),
            '5' => Some(Self::LiveSessions),
            '6' => Some(Self::Sessions),
            '7' => Some(Self::Tasks),
            '8' => Some(Self::Channels),
            '9' => Some(Self::Events),
            '0' => Some(Self::Settings),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SearchScope {
    All,
    Sessions,
    Messages,
    Tools,
    SessionEvents,
    Agents,
    Tasks,
    Channels,
    Events,
}

impl SearchScope {
    fn title(self) -> &'static str {
        match self {
            Self::All => "All",
            Self::Sessions => "Sessions",
            Self::Messages => "Messages",
            Self::Tools => "Tool Calls",
            Self::SessionEvents => "Session Events",
            Self::Agents => "Agents",
            Self::Tasks => "Tasks",
            Self::Channels => "Channels",
            Self::Events => "Events",
        }
    }

    fn next(self) -> Self {
        match self {
            Self::All => Self::Sessions,
            Self::Sessions => Self::Messages,
            Self::Messages => Self::Tools,
            Self::Tools => Self::SessionEvents,
            Self::SessionEvents => Self::Agents,
            Self::Agents => Self::Tasks,
            Self::Tasks => Self::Channels,
            Self::Channels => Self::Events,
            Self::Events => Self::All,
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
    EditSearchQuery,
    EditSessionTitle {
        session_id: String,
    },
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
    search_index: usize,
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
    requested_stream_session: Option<String>,
    requested_session_detail: Option<String>,
    search_query: String,
    search_scope: SearchScope,
    persisted_search_hits: Vec<PersistedSessionSearchHit>,
    search_loading: bool,
    search_offset: usize,
    search_page_size: usize,
    search_has_more: bool,
    task_filter: String,
    channel_filter: String,
    event_filter: String,
    events_paused: bool,
    events_follow_latest: bool,
    paused_events: Vec<EventEnvelope>,
    live_transcripts: BTreeMap<String, LiveTranscriptState>,
    pending_chat_prompt: Option<PendingChatPrompt>,
    detail_retry_until: BTreeMap<String, Instant>,
    detail_last_requested_at: BTreeMap<String, Instant>,
    inline_thinking_expanded: bool,
    focused_chat_turn: Option<(String, u32)>,
    pending_chat_turn_jump: Option<(String, u32)>,
    focused_search_context: Option<(String, SearchChatContext)>,
}

#[derive(Debug, Clone, Default)]
struct LiveTranscriptState {
    pending_user_messages: VecDeque<String>,
    assistant_preview: String,
    thinking_preview: String,
    completed_turn_thinking: BTreeMap<u32, String>,
    recent_tool_calls: VecDeque<String>,
    recent_events: VecDeque<String>,
    awaiting_reply: bool,
    awaiting_reply_for: Option<String>,
}

#[derive(Debug, Clone)]
struct TranscriptLine {
    line: Line<'static>,
    turn_index: Option<u32>,
}

#[derive(Debug, Clone)]
struct TranscriptBlockMeta {
    status: Option<String>,
    focused_turn: bool,
    turn_index: Option<u32>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct TokenUsageTotals {
    input_tokens: u64,
    output_tokens: u64,
}

impl TokenUsageTotals {
    fn total_tokens(self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    fn record(&mut self, input_tokens: u64, output_tokens: u64) {
        self.input_tokens = self.input_tokens.saturating_add(input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(output_tokens);
    }

    fn has_data(self) -> bool {
        self.input_tokens > 0 || self.output_tokens > 0
    }
}

#[derive(Debug, Clone, Default)]
struct SessionTokenUsageSummary {
    total: TokenUsageTotals,
    turns: BTreeMap<u32, TokenUsageTotals>,
}

#[derive(Debug, Clone)]
struct SearchHit {
    kind: &'static str,
    label: String,
    summary: String,
    detail: String,
    rank: i32,
    action: SearchAction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SearchChatContextKind {
    Session,
    Message,
    ToolExecution,
    SessionEvent,
}

#[derive(Debug, Clone)]
struct SearchChatContext {
    kind: SearchChatContextKind,
    label: String,
    summary: String,
    tool_name: Option<String>,
    event_type: Option<String>,
}

#[derive(Debug, Clone)]
enum SearchAction {
    OpenChatSession {
        session_id: String,
        focus_turn: Option<u32>,
        context: Option<SearchChatContext>,
    },
    FocusAgent {
        agent_id: String,
    },
    FocusTask {
        request_id: String,
    },
    FocusChannel {
        channel_id: String,
    },
    FocusEvent {
        event_name: String,
        created_at: Option<String>,
    },
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
    execute!(
        stdout(),
        EnterAlternateScreen,
        EnableBracketedPaste,
        EnableMouseCapture
    )?;
    let mut terminal = ratatui::init();

    let loop_result = run_shell(&mut terminal, initial_options, loaded_settings).await;

    ratatui::restore();
    disable_raw_mode()?;
    execute!(
        stdout(),
        LeaveAlternateScreen,
        DisableBracketedPaste,
        DisableMouseCapture
    )?;

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

        app.ensure_chat_session_stream_loaded(&controller.command_tx)?;
        app.ensure_session_detail_loaded(&controller.command_tx)?;

        if event::poll(Duration::from_millis(16)).context("Failed to poll terminal events")? {
            loop {
                match event::read().context("Failed to read terminal event")? {
                    CEvent::Key(key)
                        if matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat) =>
                    {
                        if let Some(action) = handle_key(app, key.code, &controller.command_tx)? {
                            return Ok(action);
                        }
                    }
                    CEvent::Paste(text) => handle_paste(app, &text)?,
                    CEvent::Mouse(mouse) => handle_mouse(app, mouse),
                    _ => {}
                }

                while let Ok(update) = controller.update_rx.try_recv() {
                    app.apply_update(update);
                }

                if !event::poll(Duration::ZERO).context("Failed to drain terminal events")? {
                    break;
                }
            }
        }

        app.ensure_chat_session_stream_loaded(&controller.command_tx)?;
        app.ensure_session_detail_loaded(&controller.command_tx)?;

        terminal.draw(|frame| render(frame, app))?;
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

fn handle_mouse(app: &mut TuiApp, mouse: MouseEvent) {
    if app.input_mode.is_some() {
        return;
    }

    if app.tab == TabKind::Chat {
        match mouse.kind {
            MouseEventKind::ScrollUp => app.scroll_chat(3),
            MouseEventKind::ScrollDown => app.scroll_chat(-3),
            _ => {}
        }
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
        KeyCode::Char('t') if app.tab == TabKind::Chat => app.toggle_inline_thinking_expansion(),
        KeyCode::Char('v') if app.tab == TabKind::Chat => app.toggle_streaming_preview(),
        KeyCode::Char('f') if app.tab == TabKind::Chat => app.toggle_chat_follow_latest(),
        KeyCode::Char('l') if app.tab == TabKind::Connections => app.reload_profiles(),
        KeyCode::Char('m') if app.tab == TabKind::Search => app.cycle_search_scope(command_tx)?,
        KeyCode::Char('F') if app.tab == TabKind::Search => app.clear_search_query(command_tx)?,
        KeyCode::Char('[') if app.tab == TabKind::Search => app.prev_search_page(command_tx)?,
        KeyCode::Char(']') if app.tab == TabKind::Search => app.next_search_page(command_tx)?,
        KeyCode::Char('v') if app.tab == TabKind::Connections => {
            app.load_current_connection_into_draft()
        }
        KeyCode::Char('b') if app.tab == TabKind::Connections => {
            app.load_selected_profile_into_draft()
        }
        KeyCode::Char('P') if app.tab == TabKind::Connections => app.preflight_selected_profile(),
        KeyCode::Char('T') if app.tab == TabKind::Connections => app.preflight_draft(),
        KeyCode::Char('T')
            if matches!(
                app.tab,
                TabKind::Chat | TabKind::Search | TabKind::LiveSessions | TabKind::Sessions
            ) =>
        {
            app.start_edit_session_title()
        }
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
        KeyCode::Char('/') => {
            app.tab = TabKind::Search;
            app.start_edit_search_query();
        }
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
            TabKind::Search => app.activate_search_result(command_tx)?,
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
            if input.is_empty()
                && !matches!(
                    app.input_mode.as_ref(),
                    Some(InputMode::EditSessionTitle { .. })
                )
            {
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
                    Some(InputMode::EditSearchQuery) => "Search query cannot be empty",
                    Some(InputMode::EditUserLabel) => "User label cannot be empty",
                    Some(InputMode::EditSessionTitle { .. }) => {
                        "Blank title will clear the current session title"
                    }
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
                        app.persist_settings_quietly();
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
                    app.persist_settings_quietly();
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
                Some(InputMode::EditSearchQuery) => {
                    app.search_query = input;
                    app.search_index = 0;
                    app.search_offset = 0;
                    app.tab = TabKind::Search;
                    app.refresh_persisted_search(command_tx)?;
                    app.dashboard.record_info("Updated the search query");
                }
                Some(InputMode::EditSessionTitle { session_id }) => {
                    let title = input.trim().to_string();
                    send_command(
                        command_tx,
                        OperatorCommand::SetSessionTitle {
                            session_id,
                            title: (!title.is_empty()).then_some(title),
                        },
                    )?;
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
        .fg(Color::Rgb(60, 64, 68))
        .add_modifier(Modifier::DIM)
}

fn panel_block<'a>(title: &'a str) -> Block<'a> {
    Block::default()
        .title(Line::from(Span::styled(
            title,
            Style::default()
                .fg(Color::Rgb(124, 130, 142))
                .add_modifier(Modifier::DIM),
        )))
        .borders(Borders::ALL)
        .border_style(subtle_border_style())
        .padding(Padding::horizontal(1))
}

fn pane_block<'a>(title: &'a str) -> Block<'a> {
    panel_block(title)
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
    .block(pane_block("Connection"))
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
    if app.tab == TabKind::Search {
        render_search_panel(frame, app, area);
        return;
    }

    let title = match app.tab {
        TabKind::Chat => "Chat",
        TabKind::Search => "Search",
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
        .block(pane_block(title));
    frame.render_stateful_widget(list, area, &mut state);
}

fn render_search_panel(frame: &mut Frame<'_>, app: &mut TuiApp, area: ratatui::layout::Rect) {
    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(5), Constraint::Min(6)])
        .split(area);

    let (session_count, message_count, tool_count, session_event_count, other_count) =
        app.search_kind_counts();
    let visible_results = app.search_hits().len();
    let range_start = if visible_results == 0 {
        0
    } else {
        app.search_offset + 1
    };
    let range_end = app.search_offset + visible_results;
    let summary = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("Query: ", Style::default().fg(Color::Gray)),
            Span::raw(if app.search_query.trim().is_empty() {
                "<empty>".to_string()
            } else {
                app.search_query.clone()
            }),
        ]),
        Line::from(vec![
            Span::styled("Scope: ", Style::default().fg(Color::Gray)),
            Span::raw(app.search_scope.title()),
            Span::styled("  Results: ", Style::default().fg(Color::Gray)),
            Span::raw(visible_results.to_string()),
            if app.search_loading {
                Span::styled("  loading…", Style::default().fg(Color::Yellow))
            } else {
                Span::raw("")
            },
        ]),
        Line::from(vec![
            Span::styled("Page: ", Style::default().fg(Color::Gray)),
            Span::raw(format!(
                "{}  [{}-{}]",
                (app.search_offset / app.search_page_size) + 1,
                range_start,
                range_end
            )),
            Span::styled("  More: ", Style::default().fg(Color::Gray)),
            Span::raw(if app.search_has_more { "yes" } else { "no" }),
        ]),
        Line::from(vec![
            Span::styled("Kinds: ", Style::default().fg(Color::Gray)),
            Span::raw(format!(
                "sessions {}  messages {}  tools {}  session events {}  other {}",
                session_count, message_count, tool_count, session_event_count, other_count
            )),
        ]),
    ])
    .block(pane_block("Search"));
    frame.render_widget(summary, sections[0]);

    let items = app.search_list_items();
    let mut state = ListState::default();
    if !items.is_empty() {
        state.select(Some(app.search_index.min(items.len().saturating_sub(1))));
    }
    let list = List::new(items)
        .highlight_style(
            Style::default()
                .bg(Color::Rgb(28, 56, 73))
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol(">> ")
        .block(pane_block("Results"));
    frame.render_stateful_widget(list, sections[1], &mut state);
}

fn render_right_panel(frame: &mut Frame<'_>, app: &TuiApp, area: ratatui::layout::Rect) {
    let detail = Paragraph::new(app.detail_text())
        .block(pane_block("Detail"))
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
        .block(pane_block(app.settings.layout.left_pane.title()));
    frame.render_stateful_widget(list, area, &mut state);
}

fn render_chat_center(frame: &mut Frame<'_>, app: &mut TuiApp, area: ratatui::layout::Rect) {
    let block = pane_block("Chat");
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut header_lines = vec![
        Line::from(vec![
            Span::styled("Session: ", Style::default().fg(Color::Gray)),
            Span::styled(
                app.current_chat_title(),
                Style::default()
                    .fg(Color::LightCyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                app.current_chat_status_label(),
                app.current_chat_activity_style(),
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
        Line::from(vec![
            Span::styled("Tokens: ", Style::default().fg(Color::Gray)),
            Span::styled(
                app.current_chat_token_usage_label(),
                Style::default().fg(Color::Rgb(168, 176, 186)),
            ),
        ]),
    ];
    if let Some(summary) = app.current_prompt_context_summary() {
        header_lines.push(Line::from(vec![
            Span::styled("Replying to: ", Style::default().fg(Color::Gray)),
            Span::styled(summary, Style::default().fg(Color::LightBlue)),
        ]));
    }
    if let Some(summary) = app.current_search_context_summary() {
        header_lines.push(Line::from(vec![
            Span::styled("Search hit: ", Style::default().fg(Color::Gray)),
            Span::styled(summary, Style::default().fg(Color::Rgb(198, 208, 122))),
        ]));
    }

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(header_lines.len() as u16),
            Constraint::Min(6),
        ])
        .split(inner);

    let header = Paragraph::new(header_lines).wrap(Wrap { trim: true });
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
        .block(pane_block(title))
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
            search_index: 0,
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
            requested_stream_session: None,
            requested_session_detail: None,
            search_query: String::new(),
            search_scope: SearchScope::All,
            persisted_search_hits: Vec::new(),
            search_loading: false,
            search_offset: 0,
            search_page_size: 64,
            search_has_more: false,
            task_filter: String::new(),
            channel_filter: String::new(),
            event_filter: String::new(),
            events_paused: false,
            events_follow_latest: true,
            paused_events: Vec::new(),
            live_transcripts: BTreeMap::new(),
            pending_chat_prompt: None,
            detail_retry_until: BTreeMap::new(),
            detail_last_requested_at: BTreeMap::new(),
            inline_thinking_expanded: false,
            focused_chat_turn: None,
            pending_chat_turn_jump: None,
            focused_search_context: None,
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
            UiUpdate::SessionEvent(event) => {
                self.apply_live_event(event);
            }
            UiUpdate::SessionDetail(detail) => {
                self.sync_pending_user_messages_from_detail(&detail.session.session_id, detail);
                if self.session_detail_satisfies_pending_reply(&detail.session.session_id, detail) {
                    self.capture_completed_thinking_from_detail(&detail.session.session_id, detail);
                    self.finalize_live_transcript_for(&detail.session.session_id);
                    self.detail_retry_until.remove(&detail.session.session_id);
                    self.detail_last_requested_at
                        .remove(&detail.session.session_id);
                } else {
                    if let Some(state) = self.live_transcripts.get_mut(&detail.session.session_id) {
                        state.awaiting_reply = true;
                    }
                    self.requested_session_detail = None;
                }
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
            UiUpdate::SearchResults {
                query,
                scope,
                offset,
                limit,
                has_more,
                hits,
            } => {
                if query.trim() == self.search_query.trim()
                    && Some(*scope) == self.persisted_search_scope()
                    && *offset == self.search_offset
                    && *limit == self.search_page_size
                {
                    self.persisted_search_hits = hits.clone();
                    self.search_loading = false;
                    self.search_has_more = *has_more;
                }
            }
            UiUpdate::RefreshTelemetry { .. } | UiUpdate::Info(_) => {}
            UiUpdate::Error(_) => {
                if self.search_loading {
                    self.search_loading = false;
                }
            }
        }

        if !matches!(&update, UiUpdate::SessionEvent(_)) {
            self.dashboard.apply_update(update);
        }
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
        self.search_index = clamp_index(self.search_index, self.search_hits().len());
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
            TabKind::Search => self.search_index,
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
            TabKind::Search => self.search_index = value,
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
            TabKind::Search => self.search_hits().len(),
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

    fn activate_search_result(
        &mut self,
        command_tx: &tokio::sync::mpsc::UnboundedSender<OperatorCommand>,
    ) -> Result<()> {
        let Some(hit) = self.selected_search_hit() else {
            self.dashboard
                .record_error("No search result is currently selected");
            return Ok(());
        };

        match hit.action {
            SearchAction::OpenChatSession {
                session_id,
                focus_turn,
                context,
            } => {
                self.chat_session_id = Some(session_id);
                self.tab = TabKind::Chat;
                self.chat_scroll_lines = 0;
                self.settings.chat.follow_latest = false;
                self.persist_settings_quietly();
                self.focused_chat_turn = focus_turn.map(|turn| {
                    (
                        self.chat_session_id
                            .as_deref()
                            .unwrap_or_default()
                            .to_string(),
                        turn,
                    )
                });
                self.pending_chat_turn_jump = self.focused_chat_turn.clone();
                self.focused_search_context = context.map(|context| {
                    (
                        self.chat_session_id
                            .as_deref()
                            .unwrap_or_default()
                            .to_string(),
                        context,
                    )
                });
            }
            SearchAction::FocusAgent { agent_id } => {
                self.focused_chat_turn = None;
                self.pending_chat_turn_jump = None;
                self.focused_search_context = None;
                if let Some(index) = self
                    .dashboard
                    .agents()
                    .iter()
                    .position(|agent| agent.id == agent_id)
                {
                    self.agent_index = index;
                }
                self.tab = TabKind::Agents;
            }
            SearchAction::FocusTask { request_id } => {
                self.focused_chat_turn = None;
                self.pending_chat_turn_jump = None;
                self.focused_search_context = None;
                let filtered = self.filtered_tasks();
                if let Some(index) = filtered
                    .iter()
                    .position(|task| task.request_id == request_id)
                {
                    self.task_index = index;
                }
                self.tab = TabKind::Tasks;
            }
            SearchAction::FocusChannel { channel_id } => {
                self.focused_chat_turn = None;
                self.pending_chat_turn_jump = None;
                self.focused_search_context = None;
                let filtered = self.filtered_channels();
                if let Some(index) = filtered.iter().position(|channel| channel.id == channel_id) {
                    self.channel_index = index;
                }
                self.tab = TabKind::Channels;
            }
            SearchAction::FocusEvent {
                event_name,
                created_at,
            } => {
                self.focused_chat_turn = None;
                self.pending_chat_turn_jump = None;
                self.focused_search_context = None;
                let filtered = self.filtered_events();
                if let Some(index) = filtered.iter().position(|event| {
                    event.event == event_name
                        && created_at.as_deref()
                            == event
                                .data
                                .get("created_at")
                                .and_then(|value| value.as_str())
                }) {
                    self.event_index = index;
                }
                self.tab = TabKind::Events;
            }
        }

        self.ensure_chat_session_stream_loaded(command_tx)?;
        self.ensure_session_detail_loaded(command_tx)?;
        Ok(())
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
            if self.chat_session_id.as_deref() != Some(session_id) {
                self.focused_chat_turn = None;
                self.pending_chat_turn_jump = None;
                self.focused_search_context = None;
            }
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
        if let Some(title) = self.session_title(session_id) {
            return title;
        }

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

    fn session_title(&self, session_id: &str) -> Option<String> {
        self.dashboard
            .session_detail(session_id)
            .and_then(|detail| session_metadata_title(detail.session.metadata.as_ref()))
            .or_else(|| {
                self.dashboard
                    .sessions
                    .iter()
                    .find(|session| session.session_id == session_id)
                    .and_then(|session| session_metadata_title(session.metadata.as_ref()))
            })
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
                self.detail_retry_until.remove(session_id);
                self.detail_last_requested_at.remove(session_id);
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
            "message_end" => {
                self.requested_session_detail = None;
                self.detail_retry_until.insert(
                    session_id.to_string(),
                    Instant::now() + Duration::from_secs(5),
                );
            }
            "task_complete" => {
                self.requested_session_detail = None;
                self.detail_retry_until.insert(
                    session_id.to_string(),
                    Instant::now() + Duration::from_secs(5),
                );
            }
            _ => {}
        }

        if matches!(
            event.event.as_str(),
            "task_start"
                | "task_complete"
                | "message_delta"
                | "thinking_delta"
                | "message_end"
                | "tool_call"
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

    fn finalize_live_transcript_for(&mut self, session_id: &str) {
        if let Some(state) = self.live_transcripts.get_mut(session_id) {
            state.assistant_preview.clear();
            state.thinking_preview.clear();
            state.pending_user_messages.clear();
            state.awaiting_reply = false;
            state.awaiting_reply_for = None;
        }
    }

    fn capture_completed_thinking_from_detail(
        &mut self,
        session_id: &str,
        detail: &turin_control_client::SessionDetail,
    ) {
        let Some(state) = self.live_transcripts.get_mut(session_id) else {
            return;
        };
        let thinking = state.thinking_preview.trim();
        if thinking.is_empty() {
            return;
        }

        let Some(prompt) = state.awaiting_reply_for.as_deref() else {
            return;
        };

        let Some(turn_index) = assistant_turn_index_after_prompt(detail, prompt) else {
            return;
        };

        state
            .completed_turn_thinking
            .entry(turn_index)
            .or_insert_with(|| thinking.to_string());
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
        state.awaiting_reply = true;
        state.awaiting_reply_for = Some(prompt.trim().to_string());
        self.requested_session_detail = None;
        self.detail_retry_until.insert(
            session_id.to_string(),
            Instant::now() + Duration::from_secs(10),
        );
        self.detail_last_requested_at.remove(session_id);
        self.chat_session_id = Some(session_id.to_string());
        self.chat_scroll_lines = 0;
        self.pending_chat_turn_jump = None;
        self.focused_search_context = None;
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
            Some(InputMode::EditSearchQuery) => "Search Query",
            Some(InputMode::EditSessionTitle { .. }) => "Session Title",
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
            Some(InputMode::EditSearchQuery) => {
                "Enter updates the global search query. Esc cancels.".to_string()
            }
            Some(InputMode::EditSessionTitle { .. }) => {
                "Enter updates the current session title. Submit a blank value to clear it. Esc cancels.".to_string()
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

    fn search_hits(&self) -> Vec<SearchHit> {
        let query = self.search_query.trim().to_ascii_lowercase();
        if query.is_empty() {
            return Vec::new();
        }

        let mut hits = Vec::new();

        if self.persisted_search_scope().is_some() {
            for hit in &self.persisted_search_hits {
                let label = if let Some(title) = hit.title.as_deref() {
                    title.to_string()
                } else {
                    self.session_label(&hit.session_id, Some(&hit.agent_id))
                };
                let (kind, line_label, detail) = match hit.kind {
                    SessionSearchHitKind::Session => (
                        "session",
                        format!("{label} [{}]", hit.created_at),
                        pretty_json(&serde_json::json!({
                            "type": "session",
                            "session_id": hit.session_id,
                            "agent_id": hit.agent_id,
                            "title": hit.title,
                            "created_at": hit.created_at,
                        })),
                    ),
                    SessionSearchHitKind::Message => (
                        "message",
                        format!(
                            "{} · {} · turn {}",
                            label,
                            hit.role.as_deref().unwrap_or("message"),
                            hit.turn_index.unwrap_or(0)
                        ),
                        pretty_json(&serde_json::json!({
                            "type": "session_message",
                            "session_id": hit.session_id,
                            "agent_id": hit.agent_id,
                            "title": hit.title,
                            "role": hit.role,
                            "turn_index": hit.turn_index,
                            "created_at": hit.created_at,
                            "content": hit.snippet,
                        })),
                    ),
                    SessionSearchHitKind::ToolExecution => (
                        "tool",
                        format!(
                            "{} · {} · turn {}",
                            label,
                            hit.tool_name.as_deref().unwrap_or("tool"),
                            hit.turn_index.unwrap_or(0)
                        ),
                        pretty_json(&serde_json::json!({
                            "type": "tool_execution",
                            "session_id": hit.session_id,
                            "agent_id": hit.agent_id,
                            "title": hit.title,
                            "tool_name": hit.tool_name,
                            "turn_index": hit.turn_index,
                            "created_at": hit.created_at,
                            "content": hit.snippet,
                        })),
                    ),
                    SessionSearchHitKind::Event => (
                        "event",
                        format!(
                            "{} · {}",
                            label,
                            hit.event_type.as_deref().unwrap_or("event")
                        ),
                        pretty_json(&serde_json::json!({
                            "type": "session_event",
                            "session_id": hit.session_id,
                            "agent_id": hit.agent_id,
                            "title": hit.title,
                            "event_type": hit.event_type,
                            "created_at": hit.created_at,
                            "content": hit.snippet,
                        })),
                    ),
                };
                let context = Some(match hit.kind {
                    SessionSearchHitKind::Session => SearchChatContext {
                        kind: SearchChatContextKind::Session,
                        label: line_label.clone(),
                        summary: hit.summary.clone(),
                        tool_name: None,
                        event_type: None,
                    },
                    SessionSearchHitKind::Message => SearchChatContext {
                        kind: SearchChatContextKind::Message,
                        label: line_label.clone(),
                        summary: hit.summary.clone(),
                        tool_name: None,
                        event_type: None,
                    },
                    SessionSearchHitKind::ToolExecution => SearchChatContext {
                        kind: SearchChatContextKind::ToolExecution,
                        label: line_label.clone(),
                        summary: hit.summary.clone(),
                        tool_name: hit.tool_name.clone(),
                        event_type: None,
                    },
                    SessionSearchHitKind::Event => SearchChatContext {
                        kind: SearchChatContextKind::SessionEvent,
                        label: line_label.clone(),
                        summary: hit.summary.clone(),
                        tool_name: None,
                        event_type: hit.event_type.clone(),
                    },
                });
                let rank = i32::try_from(hit.score)
                    .unwrap_or(if hit.score.is_negative() {
                        i32::MIN
                    } else {
                        i32::MAX
                    });
                hits.push(SearchHit {
                    kind,
                    label: line_label,
                    summary: hit.summary.clone(),
                    detail,
                    rank,
                    action: SearchAction::OpenChatSession {
                        session_id: hit.session_id.clone(),
                        focus_turn: hit.turn_index,
                        context,
                    },
                });
            }
        }

        if matches!(self.search_scope, SearchScope::All | SearchScope::Agents) {
            for agent in self.dashboard.agents() {
                let runtime = self.agent_runtime(&agent.id);
                let summary = format!(
                    "{} [{}] {}",
                    agent.id,
                    agent.model,
                    if runtime.is_some_and(|runtime| runtime.running) {
                        "running"
                    } else {
                        "idle"
                    }
                );
                if search_match(
                    &query,
                    &[
                        &summary,
                        &agent.id,
                        &agent.model,
                        &agent.provider,
                        &agent.harness_ref,
                    ],
                ) && let Some(rank) = search_rank(
                    &query,
                    &[
                        &summary,
                        &agent.id,
                        &agent.model,
                        &agent.provider,
                        &agent.harness_ref,
                    ],
                ) {
                    hits.push(SearchHit {
                        kind: "agent",
                        label: summary,
                        summary: agent.provider.clone(),
                        detail: pretty_json(&serde_json::json!({
                            "type": "agent",
                            "agent": agent,
                            "runtime": runtime,
                        })),
                        rank,
                        action: SearchAction::FocusAgent {
                            agent_id: agent.id.clone(),
                        },
                    });
                }
            }
        }

        if matches!(self.search_scope, SearchScope::All | SearchScope::Tasks) {
            for task in &self.dashboard.tasks {
                let summary = format!("{} {} {}", task.request_id, task.state, task.agent_id);
                if search_match(
                    &query,
                    &[
                        &summary,
                        &task.request_id,
                        &task.agent_id,
                        &task.state,
                        task.error.as_deref().unwrap_or(""),
                        task.output.as_deref().unwrap_or(""),
                    ],
                ) && let Some(rank) = search_rank(
                    &query,
                    &[
                        &summary,
                        &task.request_id,
                        &task.agent_id,
                        &task.state,
                        task.error.as_deref().unwrap_or(""),
                        task.output.as_deref().unwrap_or(""),
                    ],
                ) {
                    hits.push(SearchHit {
                        kind: "task",
                        label: summary,
                        summary: task.trace_id.clone(),
                        detail: pretty_json(&serde_json::json!({
                            "type": "task",
                            "task": task,
                        })),
                        rank,
                        action: SearchAction::FocusTask {
                            request_id: task.request_id.clone(),
                        },
                    });
                }
            }
        }

        if matches!(self.search_scope, SearchScope::All | SearchScope::Channels) {
            for channel in self.dashboard.channels() {
                let runtime = self.channel_runtime(&channel.id);
                let summary = format!("{} [{}] {}", channel.id, channel.kind, channel.agent_id);
                if search_match(
                    &query,
                    &[&summary, &channel.id, &channel.kind, &channel.agent_id],
                ) && let Some(rank) = search_rank(
                    &query,
                    &[&summary, &channel.id, &channel.kind, &channel.agent_id],
                ) {
                    hits.push(SearchHit {
                        kind: "channel",
                        label: summary,
                        summary: if channel.enabled {
                            "enabled".to_string()
                        } else {
                            "disabled".to_string()
                        },
                        detail: pretty_json(&serde_json::json!({
                            "type": "channel",
                            "channel": channel,
                            "runtime": runtime,
                        })),
                        rank,
                        action: SearchAction::FocusChannel {
                            channel_id: channel.id.clone(),
                        },
                    });
                }
            }
        }

        if matches!(self.search_scope, SearchScope::All | SearchScope::Events) {
            for event in self.event_source().iter().rev() {
                let payload =
                    serde_json::to_string(&event.data).unwrap_or_else(|_| "{}".to_string());
                if search_match(&query, &[&event.event, &payload])
                    && let Some(rank) = search_rank(&query, &[&event.event, &payload])
                {
                    hits.push(SearchHit {
                        kind: "event",
                        label: event.event.clone(),
                        summary: compact_json(&event.data, 100),
                        detail: pretty_json(&serde_json::json!({
                            "type": "event",
                            "event": event,
                        })),
                        rank,
                        action: SearchAction::FocusEvent {
                            event_name: event.event.clone(),
                            created_at: event
                                .data
                                .get("created_at")
                                .and_then(|value| value.as_str())
                                .map(str::to_string),
                        },
                    });
                }
            }
        }

        hits.sort_by(|left, right| {
            right
                .rank
                .cmp(&left.rank)
                .then_with(|| left.kind.cmp(right.kind))
                .then_with(|| left.label.cmp(&right.label))
        });

        hits
    }

    fn selected_search_hit(&self) -> Option<SearchHit> {
        self.search_hits().get(self.search_index).cloned()
    }

    fn search_kind_counts(&self) -> (usize, usize, usize, usize, usize) {
        let mut session_count = 0;
        let mut message_count = 0;
        let mut tool_count = 0;
        let mut session_event_count = 0;
        let mut other_count = 0;
        for hit in self.search_hits() {
            match (&hit.kind, &hit.action) {
                (&"session", _) => session_count += 1,
                (&"message", _) => message_count += 1,
                (&"tool", _) => tool_count += 1,
                (&"event", SearchAction::OpenChatSession { .. }) => session_event_count += 1,
                _ => other_count += 1,
            }
        }
        (
            session_count,
            message_count,
            tool_count,
            session_event_count,
            other_count,
        )
    }

    fn search_list_items(&self) -> Vec<ListItem<'static>> {
        let hits = self.search_hits();
        if hits.is_empty() {
            return vec![ListItem::new(if self.search_loading {
                "Searching persisted session history…"
            } else if self.search_query.trim().is_empty() {
                "Use / to enter a search query."
            } else {
                "No search results."
            })];
        }

        hits.into_iter()
            .map(|hit| {
                ListItem::new(vec![
                    Line::from(vec![
                        Span::styled(
                            format!("{} ", search_hit_icon(hit.kind)),
                            Style::default().fg(search_hit_color(hit.kind)),
                        ),
                        Span::styled(hit.label, Style::default().add_modifier(Modifier::BOLD)),
                    ]),
                    Line::from(Span::styled(
                        format!("  {}", hit.summary),
                        Style::default().fg(Color::Rgb(150, 156, 168)),
                    )),
                ])
            })
            .collect()
    }

    fn persisted_search_scope(&self) -> Option<SessionSearchScope> {
        match self.search_scope {
            SearchScope::All => Some(SessionSearchScope::All),
            SearchScope::Sessions => Some(SessionSearchScope::Sessions),
            SearchScope::Messages => Some(SessionSearchScope::Messages),
            SearchScope::Tools => Some(SessionSearchScope::ToolExecutions),
            SearchScope::SessionEvents => Some(SessionSearchScope::Events),
            SearchScope::Agents
            | SearchScope::Tasks
            | SearchScope::Channels
            | SearchScope::Events => None,
        }
    }

    fn refresh_persisted_search(
        &mut self,
        command_tx: &tokio::sync::mpsc::UnboundedSender<OperatorCommand>,
    ) -> Result<()> {
        let query = self.search_query.trim().to_string();
        self.persisted_search_hits.clear();
        self.search_loading = false;
        self.search_has_more = false;

        let Some(scope) = self.persisted_search_scope() else {
            return Ok(());
        };
        if query.is_empty() {
            return Ok(());
        }

        self.search_loading = true;
        send_command(
            command_tx,
            OperatorCommand::SearchSessions {
                query,
                scope,
                limit: self.search_page_size,
                offset: self.search_offset,
            },
        )
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
        self.persist_settings_quietly();
        self.dashboard.record_info(format!(
            "Chat left pane is now {}",
            self.settings.layout.left_pane.title()
        ));
    }

    fn cycle_right_chat_pane(&mut self) {
        self.settings.layout.right_pane = self.settings.layout.right_pane.next();
        self.persist_settings_quietly();
        self.dashboard.record_info(format!(
            "Chat right pane is now {}",
            self.settings.layout.right_pane.title()
        ));
    }

    fn toggle_show_thinking(&mut self) {
        self.settings.chat.show_thinking = !self.settings.chat.show_thinking;
        self.persist_settings_quietly();
        self.dashboard.record_info(format!(
            "Thinking pane is now {}",
            if self.settings.chat.show_thinking {
                "visible"
            } else {
                "hidden"
            }
        ));
    }

    fn toggle_inline_thinking_expansion(&mut self) {
        self.inline_thinking_expanded = !self.inline_thinking_expanded;
        self.dashboard.record_info(format!(
            "Inline thinking is now {}",
            if self.inline_thinking_expanded {
                "expanded"
            } else {
                "collapsed"
            }
        ));
    }

    fn start_edit_search_query(&mut self) {
        self.begin_input_mode(InputMode::EditSearchQuery, self.search_query.clone());
    }

    fn start_edit_session_title(&mut self) {
        let Some(session_id) = self.current_detail_session_id() else {
            self.dashboard
                .record_error("No session is currently selected for titling");
            return;
        };
        let initial = self.session_title(&session_id).unwrap_or_default();
        self.begin_input_mode(InputMode::EditSessionTitle { session_id }, initial);
    }

    fn cycle_search_scope(
        &mut self,
        command_tx: &tokio::sync::mpsc::UnboundedSender<OperatorCommand>,
    ) -> Result<()> {
        self.search_scope = self.search_scope.next();
        self.search_index = 0;
        self.search_offset = 0;
        self.refresh_persisted_search(command_tx)?;
        self.dashboard
            .record_info(format!("Search scope is now {}", self.search_scope.title()));
        Ok(())
    }

    fn clear_search_query(
        &mut self,
        command_tx: &tokio::sync::mpsc::UnboundedSender<OperatorCommand>,
    ) -> Result<()> {
        self.search_query.clear();
        self.search_index = 0;
        self.search_offset = 0;
        self.refresh_persisted_search(command_tx)?;
        self.dashboard
            .record_info("Cleared the global search query");
        Ok(())
    }

    fn next_search_page(
        &mut self,
        command_tx: &tokio::sync::mpsc::UnboundedSender<OperatorCommand>,
    ) -> Result<()> {
        if !self.search_has_more {
            self.dashboard
                .record_info("Already on the last persisted search page");
            return Ok(());
        }
        self.search_offset = self.search_offset.saturating_add(self.search_page_size);
        self.search_index = 0;
        self.refresh_persisted_search(command_tx)?;
        Ok(())
    }

    fn prev_search_page(
        &mut self,
        command_tx: &tokio::sync::mpsc::UnboundedSender<OperatorCommand>,
    ) -> Result<()> {
        if self.search_offset == 0 {
            self.dashboard
                .record_info("Already on the first persisted search page");
            return Ok(());
        }
        self.search_offset = self.search_offset.saturating_sub(self.search_page_size);
        self.search_index = 0;
        self.refresh_persisted_search(command_tx)?;
        Ok(())
    }

    fn toggle_streaming_preview(&mut self) {
        self.settings.chat.show_streaming_preview = !self.settings.chat.show_streaming_preview;
        self.persist_settings_quietly();
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
        self.persist_settings_quietly();
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
        self.pending_chat_turn_jump = None;
        if self.chat_scroll_lines > 0 {
            self.settings.chat.follow_latest = false;
        }
    }

    fn jump_chat_latest(&mut self) {
        self.chat_scroll_lines = 0;
        self.pending_chat_turn_jump = None;
        self.settings.chat.follow_latest = true;
        self.dashboard
            .record_info("Chat view jumped back to the latest output");
    }

    fn jump_chat_oldest(&mut self) {
        self.chat_scroll_lines = u16::MAX / 2;
        self.pending_chat_turn_jump = None;
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

    fn persist_settings_quietly(&mut self) {
        if let Err(err) = save_settings(&self.settings_path, &self.settings) {
            self.dashboard
                .record_error(format!("Failed to persist TUI settings: {err}"));
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
        &mut self,
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
        trim_transcript_lines_to_budget(&mut lines, budget, self.focused_chat_turn_for_session(session_id));
        let lines = wrap_transcript_lines_for_width(lines, viewport_width.max(1));

        let total_lines = lines.len();
        let visible_lines = viewport_height.max(1);
        if let Some((pending_session_id, pending_turn)) = self.pending_chat_turn_jump.clone()
            && pending_session_id == session_id
            && let Some(target_index) = lines
                .iter()
                .position(|line| line.turn_index == Some(pending_turn))
        {
            let target_top = target_index.saturating_sub(2);
            let scroll_from_bottom = total_lines
                .saturating_sub(visible_lines.saturating_add(target_top));
            self.chat_scroll_lines = scroll_from_bottom.min(u16::MAX as usize) as u16;
            self.pending_chat_turn_jump = None;
        }
        let scroll_from_top = total_lines
            .saturating_sub(visible_lines.saturating_add(self.chat_scroll_lines as usize));
        (
            Text::from(lines.into_iter().map(|line| line.line).collect::<Vec<_>>()),
            scroll_from_top.min(u16::MAX as usize) as u16,
        )
    }

    fn build_transcript_lines(&self, session_id: &str) -> Vec<TranscriptLine> {
        let mut lines = Vec::new();
        let focused_turn = self.focused_chat_turn_for_session(session_id);
        let completed_thinking = self
            .live_transcripts
            .get(session_id)
            .map(|state| &state.completed_turn_thinking);
        let token_usage = self.session_token_usage(session_id);
        if let Some(detail) = self.dashboard.session_detail(session_id) {
            for (index, message) in detail.messages.iter().enumerate() {
                let content = message_content_text(&message.content)
                    .unwrap_or_else(|| compact_json(&message.content, 120));
                if let Some(thinking) = completed_thinking
                    .and_then(|turns| turns.get(&message.turn_index))
                    .filter(|_| {
                        message_has_renderable_assistant_text(message)
                            && assistant_turn_message_is_first(detail, index)
                    })
                {
                    self.push_thinking_preview_block(
                        &mut lines,
                        thinking,
                        self.inline_thinking_expanded,
                        true,
                        Some(message.turn_index),
                    );
                }

                let token_footer = token_usage
                    .as_ref()
                    .and_then(|usage| usage.turns.get(&message.turn_index).copied())
                    .filter(|usage| {
                        message_has_renderable_assistant_text(message)
                            && assistant_turn_message_is_last(detail, index)
                            && usage.has_data()
                    })
                    .map(token_usage_footer_line);
                self.push_message_block(
                    &mut lines,
                    message.role.as_str(),
                    &content,
                    token_footer,
                    TranscriptBlockMeta {
                        status: Some(format!("turn {}", message.turn_index)),
                        focused_turn: focused_turn == Some(message.turn_index),
                        turn_index: Some(message.turn_index),
                    },
                );
            }
        }

        if let Some(state) = self.live_transcripts.get(session_id) {
            for prompt in &state.pending_user_messages {
                self.push_message_block(
                    &mut lines,
                    "user",
                    prompt,
                    None,
                    TranscriptBlockMeta {
                        status: Some("pending".to_string()),
                        focused_turn: false,
                        turn_index: None,
                    },
                );
            }

            if self.settings.chat.show_thinking
                && !state.thinking_preview.trim().is_empty()
                && (state.awaiting_reply || !state.assistant_preview.trim().is_empty())
            {
                self.push_thinking_preview_block(
                    &mut lines,
                    &state.thinking_preview,
                    self.inline_thinking_expanded,
                    false,
                    None,
                );
            }

            if self.settings.chat.show_streaming_preview
                && !state.assistant_preview.trim().is_empty()
            {
                self.push_message_block(
                    &mut lines,
                    "assistant",
                    &state.assistant_preview,
                    None,
                    TranscriptBlockMeta {
                        status: Some("streaming".to_string()),
                        focused_turn: false,
                        turn_index: None,
                    },
                );
            }

            if state.awaiting_reply
                && state.assistant_preview.trim().is_empty()
                && state.thinking_preview.trim().is_empty()
            {
                lines.push(TranscriptLine {
                    line: Line::from(Span::styled(
                        format!("{} Thinking…", spinner_frame()),
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    )),
                    turn_index: None,
                });
                lines.push(TranscriptLine {
                    line: Line::default(),
                    turn_index: None,
                });
            }
        }

        if lines.is_empty() {
            lines.push(TranscriptLine {
                line: Line::from("No transcript has been loaded yet."),
                turn_index: None,
            });
        }
        lines
    }

    fn push_message_block(
        &self,
        lines: &mut Vec<TranscriptLine>,
        role: &str,
        content: &str,
        footer: Option<Line<'static>>,
        meta: TranscriptBlockMeta,
    ) {
        let TranscriptBlockMeta {
            status,
            focused_turn,
            turn_index,
        } = meta;
        let (label, color, body_style) = self.chat_role_descriptor(role);
        let heading = match (status, focused_turn) {
            (Some(status), true) => format!("── {label} · {status} · match"),
            (Some(status), false) => format!("── {label} · {status}"),
            (None, true) => format!("── {label} · match"),
            (None, false) => format!("── {label}"),
        };
        let heading_style = if focused_turn {
            Style::default()
                .fg(Color::LightYellow)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(color).add_modifier(Modifier::BOLD)
        };
        let body_style = if focused_turn {
            body_style.bg(Color::Rgb(44, 44, 24))
        } else {
            body_style
        };
        lines.push(TranscriptLine {
            line: Line::from(Span::styled(heading, heading_style)),
            turn_index,
        });
        let body_prefix = "│ ";
        for body_line in content.lines() {
            lines.push(TranscriptLine {
                line: Line::from(Span::styled(
                    format!("{body_prefix}{body_line}"),
                    body_style,
                )),
                turn_index,
            });
        }
        if content.is_empty() {
            lines.push(TranscriptLine {
                line: Line::from(Span::styled(body_prefix.to_string(), body_style)),
                turn_index,
            });
        }
        if let Some(footer) = footer {
            lines.push(TranscriptLine {
                line: footer,
                turn_index,
            });
        }
        lines.push(TranscriptLine {
            line: Line::default(),
            turn_index,
        });
    }

    fn push_thinking_preview_block(
        &self,
        lines: &mut Vec<TranscriptLine>,
        thinking: &str,
        expanded: bool,
        persisted: bool,
        turn_index: Option<u32>,
    ) {
        lines.push(TranscriptLine {
            line: Line::from(Span::styled(
                if persisted {
                    if expanded {
                        "·· Thinking"
                    } else {
                        "·· Thinking (collapsed, press t to expand)"
                    }
                } else if expanded {
                    "·· Thinking preview"
                } else {
                    "·· Thinking preview (press t to expand)"
                },
                Style::default()
                    .fg(Color::Rgb(132, 144, 160))
                    .add_modifier(Modifier::ITALIC | Modifier::DIM),
            )),
            turn_index,
        });

        let preview_lines = if expanded {
            thinking
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .collect::<Vec<_>>()
        } else {
            thinking
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .take(2)
                .collect::<Vec<_>>()
        };

        if preview_lines.is_empty() {
            lines.push(TranscriptLine {
                line: Line::from(Span::styled(
                    "› reasoning stream active",
                    Style::default()
                        .fg(Color::Rgb(110, 122, 138))
                        .add_modifier(Modifier::DIM),
                )),
                turn_index,
            });
        } else {
            for line in preview_lines {
                lines.push(TranscriptLine {
                    line: Line::from(Span::styled(
                        format!("› {}", excerpt(line, 140)),
                        Style::default()
                            .fg(Color::Rgb(110, 122, 138))
                            .add_modifier(Modifier::DIM),
                    )),
                    turn_index,
                });
            }
        }

        if !expanded
            && thinking
                .lines()
                .filter(|line| !line.trim().is_empty())
                .count()
                > 2
        {
            lines.push(TranscriptLine {
                line: Line::from(Span::styled(
                    "  … more available in the Thinking pane or via t",
                    Style::default()
                        .fg(Color::Rgb(96, 108, 124))
                        .add_modifier(Modifier::DIM),
                )),
                turn_index,
            });
        } else {
            lines.push(TranscriptLine {
                line: Line::from(Span::styled(
                    "  full stream available in the Thinking pane",
                    Style::default()
                        .fg(Color::Rgb(96, 108, 124))
                        .add_modifier(Modifier::DIM),
                )),
                turn_index,
            });
        }
        lines.push(TranscriptLine {
            line: Line::default(),
            turn_index,
        });
    }

    fn chat_role_descriptor(&self, role: &str) -> (String, Color, Style) {
        match role {
            "user" => (
                self.settings.chat.user_label.clone(),
                Color::LightBlue,
                Style::default()
                    .fg(Color::LightBlue)
                    .bg(Color::Rgb(18, 33, 48)),
            ),
            "assistant" => (
                "Assistant".to_string(),
                Color::LightGreen,
                Style::default().fg(Color::White),
            ),
            "system" => (
                "System".to_string(),
                Color::Yellow,
                Style::default().fg(Color::Yellow),
            ),
            _ => (
                role.to_string(),
                Color::White,
                Style::default().fg(Color::White),
            ),
        }
    }

    fn current_chat_status_label(&self) -> String {
        if self.current_chat_is_busy() {
            "Busy".to_string()
        } else {
            "Idle".to_string()
        }
    }

    fn current_chat_activity_style(&self) -> Style {
        if self.current_chat_is_busy() {
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Gray)
        }
    }

    fn current_chat_is_busy(&self) -> bool {
        let Some(session_id) = self.current_chat_session_id() else {
            return false;
        };
        self.dashboard
            .live_sessions
            .iter()
            .find(|session| session.session_id == session_id)
            .is_some_and(|session| session.active_tasks > 0 || session.queued_tasks > 0)
            || self.live_transcripts.get(session_id).is_some_and(|state| {
                state.awaiting_reply
                    || !state.assistant_preview.is_empty()
                    || !state.thinking_preview.is_empty()
            })
    }

    fn current_prompt_context_summary(&self) -> Option<String> {
        let session_id = self.current_chat_session_id()?;
        if let Some(state) = self.live_transcripts.get(session_id)
            && let Some(prompt) = state.pending_user_messages.back()
        {
            return Some(excerpt_multiline(prompt, 2, 120));
        }
        if let Some(state) = self.live_transcripts.get(session_id)
            && let Some(prompt) = state.awaiting_reply_for.as_deref()
        {
            return Some(excerpt_multiline(prompt, 2, 120));
        }
        let detail = self.dashboard.session_detail(session_id)?;
        detail
            .messages
            .iter()
            .rev()
            .find(|message| message.role == "user")
            .and_then(|message| message_content_text(&message.content))
            .map(|text| excerpt_multiline(&text, 2, 120))
    }

    fn current_search_context(&self) -> Option<&SearchChatContext> {
        let session_id = self.current_chat_session_id()?;
        self.focused_search_context
            .as_ref()
            .and_then(|(focused_session_id, context)| {
                (focused_session_id == session_id).then_some(context)
            })
    }

    fn current_search_context_summary(&self) -> Option<String> {
        let context = self.current_search_context()?;
        Some(format!(
            "{} · {}",
            context.label,
            excerpt(&context.summary, 96)
        ))
    }

    fn current_chat_token_usage_label(&self) -> String {
        let Some(session_id) = self.current_chat_session_id() else {
            return "n/a".to_string();
        };
        self.session_token_usage(session_id)
            .filter(|usage| usage.total.has_data())
            .map(|usage| format_token_usage_totals(usage.total))
            .unwrap_or_else(|| "loading…".to_string())
    }

    fn session_token_usage(&self, session_id: &str) -> Option<SessionTokenUsageSummary> {
        let detail = self.dashboard.session_detail(session_id)?;
        Some(token_usage_from_detail(detail))
    }

    fn sync_pending_user_messages_from_detail(
        &mut self,
        session_id: &str,
        detail: &turin_control_client::SessionDetail,
    ) {
        let persisted_last_user = detail
            .messages
            .iter()
            .rev()
            .find(|message| message.role == "user")
            .and_then(|message| message_content_text(&message.content));

        let Some(state) = self.live_transcripts.get_mut(session_id) else {
            return;
        };
        let Some(persisted_last_user) = persisted_last_user else {
            return;
        };

        if state
            .pending_user_messages
            .back()
            .is_some_and(|pending| pending.trim() == persisted_last_user.trim())
        {
            state.pending_user_messages.pop_back();
        }
    }

    fn session_detail_satisfies_pending_reply(
        &self,
        session_id: &str,
        detail: &turin_control_client::SessionDetail,
    ) -> bool {
        let Some(state) = self.live_transcripts.get(session_id) else {
            return self.session_detail_has_any_assistant_text(detail);
        };
        let Some(prompt) = state.awaiting_reply_for.as_deref() else {
            return self.session_detail_has_any_assistant_text(detail);
        };

        let Some(user_index) = detail.messages.iter().rposition(|message| {
            message.role == "user"
                && message_content_text(&message.content)
                    .is_some_and(|text| text.trim() == prompt.trim())
        }) else {
            return false;
        };

        detail
            .messages
            .iter()
            .skip(user_index + 1)
            .any(message_has_renderable_assistant_text)
    }

    fn session_detail_has_any_assistant_text(
        &self,
        detail: &turin_control_client::SessionDetail,
    ) -> bool {
        detail
            .messages
            .iter()
            .any(message_has_renderable_assistant_text)
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
        if !state.thinking_preview.trim().is_empty() {
            state.thinking_preview.clone()
        } else if let Some((_, thinking)) = state.completed_turn_thinking.iter().next_back() {
            thinking.clone()
        } else {
            "No streamed thinking for the selected session yet.".to_string()
        }
    }

    fn current_tool_text(&self) -> String {
        let Some(session_id) = self.current_chat_session_id() else {
            return "No chat session selected.".to_string();
        };
        let mut lines = Vec::new();
        if let Some(detail) = self.dashboard.session_detail(session_id) {
            if let Some(context) = self.current_search_context()
                && context.kind == SearchChatContextKind::ToolExecution
            {
                let focused_turn = self.focused_chat_turn_for_session(session_id);
                let matched = detail
                    .tool_executions
                    .iter()
                    .filter(|tool| {
                        focused_turn.is_none_or(|turn| tool.turn_index == turn)
                            && context
                                .tool_name
                                .as_deref()
                                .is_none_or(|tool_name| tool.tool_name == tool_name)
                    })
                    .map(|tool| {
                        format!(
                            "match · turn {} · {} · {} · {}ms",
                            tool.turn_index,
                            tool.tool_name,
                            tool.verdict,
                            tool.duration_ms.unwrap_or(0)
                        )
                    })
                    .collect::<Vec<_>>();
                if !matched.is_empty() {
                    lines.extend(matched);
                    lines.push(String::new());
                }
            }
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
        if let Some(detail) = self.dashboard.session_detail(session_id)
            && let Some(context) = self.current_search_context()
            && context.kind == SearchChatContextKind::SessionEvent
        {
            let matched = detail
                .events
                .iter()
                .filter(|event| {
                    context
                        .event_type
                        .as_deref()
                        .is_none_or(|event_type| event.event_type == event_type)
                })
                .map(|event| {
                    format!(
                        "match · {} · {}",
                        event.event_type,
                        compact_json(&event.payload, 96)
                    )
                })
                .collect::<Vec<_>>();
            if !matched.is_empty() {
                return matched.join("\n");
            }
        }
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
            TabKind::Search => self.search_list_items(),
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

    fn focused_chat_turn_for_session(&self, session_id: &str) -> Option<u32> {
        self.focused_chat_turn
            .as_ref()
            .and_then(|(focused_session_id, turn)| {
                (focused_session_id == session_id).then_some(*turn)
            })
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
                "search_context": self.current_search_context().map(|context| serde_json::json!({
                    "kind": format!("{:?}", context.kind),
                    "label": context.label,
                    "summary": context.summary,
                    "tool_name": context.tool_name,
                    "event_type": context.event_type,
                })),
            })),
            TabKind::Search => self
                .selected_search_hit()
                .map(|hit| {
                    format!(
                        "Query: {}\nScope: {}\nKind: {}\nLabel: {}\nSummary: {}\nAction: open session and focus the matched turn when available\n\n{}",
                        self.search_query,
                        self.search_scope.title(),
                        hit.kind,
                        hit.label,
                        hit.summary,
                        hit.detail
                    )
                })
                .unwrap_or_else(|| {
                    let (sessions, messages, tools, session_events, other) =
                        self.search_kind_counts();
                    pretty_json(&serde_json::json!({
                        "query": self.search_query,
                        "scope": self.search_scope.title(),
                        "result_count": self.search_hits().len(),
                        "offset": self.search_offset,
                        "page_size": self.search_page_size,
                        "has_more": self.search_has_more,
                        "search_loading": self.search_loading,
                        "counts": {
                            "sessions": sessions,
                            "messages": messages,
                            "tools": tools,
                            "session_events": session_events,
                            "other": other,
                        },
                        "note": if self.search_query.trim().is_empty() {
                            "Use / to enter a query. Search covers persisted session history plus loaded runtime state for agents, tasks, channels, and live events."
                        } else if self.search_loading {
                            "Persisted session history search is still loading."
                        } else {
                            "No search hits for the current query."
                        }
                    }))
                }),
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
        let shared = "0-9 switch views | Tab cycle | arrows/j/k move | r refresh | q quit";
        let scoped = match self.tab {
            TabKind::Chat => {
                "Enter opens/resumes or prompts | p prompt | ,/. cycle panes | h thinking pane | t inline thinking | v preview | f follow-latest | PgUp/PgDn scroll | Home/End jump"
            }
            TabKind::Search => {
                "/ edits query | m cycles scope | [ / ] page | F clears query | Enter opens selected hit"
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
            TabKind::Search => self.selected_search_hit().and_then(|hit| match hit.action {
                SearchAction::OpenChatSession { session_id, .. } => Some(session_id),
                _ => None,
            }),
            TabKind::LiveSessions => self
                .selected_live_session()
                .map(|session| session.session_id.clone()),
            TabKind::Sessions => self
                .selected_persisted_session()
                .map(|session| session.session_id.clone()),
            _ => None,
        }
    }

    fn ensure_chat_session_stream_loaded(
        &mut self,
        command_tx: &tokio::sync::mpsc::UnboundedSender<OperatorCommand>,
    ) -> Result<()> {
        let session_id = self.current_chat_session_id().map(str::to_string);
        if self.requested_stream_session == session_id {
            return Ok(());
        }

        self.requested_stream_session = session_id.clone();
        send_command(
            command_tx,
            OperatorCommand::FocusSessionStream { session_id },
        )
    }

    fn ensure_session_detail_loaded(
        &mut self,
        command_tx: &tokio::sync::mpsc::UnboundedSender<OperatorCommand>,
    ) -> Result<()> {
        let Some(session_id) = self.current_detail_session_id() else {
            self.requested_session_detail = None;
            return Ok(());
        };

        let should_retry = self.should_retry_session_detail(&session_id);

        if self.dashboard.session_detail(&session_id).is_some() && !should_retry {
            self.requested_session_detail = Some(session_id);
            return Ok(());
        }

        if self.requested_session_detail.as_deref() == Some(session_id.as_str()) && !should_retry {
            return Ok(());
        }

        let now = Instant::now();
        if should_retry
            && self
                .detail_last_requested_at
                .get(&session_id)
                .is_some_and(|last| now.duration_since(*last) < Duration::from_millis(700))
        {
            return Ok(());
        }

        self.requested_session_detail = Some(session_id.clone());
        self.detail_last_requested_at
            .insert(session_id.clone(), now);
        send_command(
            command_tx,
            OperatorCommand::LoadSessionDetail { session_id },
        )
    }

    fn should_retry_session_detail(&self, session_id: &str) -> bool {
        let now = Instant::now();
        self.detail_retry_until
            .get(session_id)
            .is_some_and(|deadline| *deadline > now)
            && self.pending_reply_not_committed(session_id)
    }

    fn pending_reply_not_committed(&self, session_id: &str) -> bool {
        self.live_transcripts
            .get(session_id)
            .is_some_and(|state| state.awaiting_reply)
            && self
                .dashboard
                .session_detail(session_id)
                .map(|detail| !self.session_detail_satisfies_pending_reply(session_id, detail))
                .unwrap_or(true)
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

fn wrap_transcript_lines_for_width(
    lines: Vec<TranscriptLine>,
    width: usize,
) -> Vec<TranscriptLine> {
    let width = width.max(1);
    let mut wrapped = Vec::new();
    for line in lines {
        let style = line
            .line
            .spans
            .first()
            .map(|span| span.style)
            .unwrap_or(line.line.style);
        let text = line
            .line
            .spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>();
        if text.is_empty() {
            wrapped.push(TranscriptLine {
                line: Line::default(),
                turn_index: line.turn_index,
            });
            continue;
        }
        for chunk in wrap_text_chunk(&text, width) {
            wrapped.push(TranscriptLine {
                line: Line::from(Span::styled(chunk, style)),
                turn_index: line.turn_index,
            });
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

fn search_match(query: &str, haystacks: &[&str]) -> bool {
    haystacks
        .iter()
        .any(|value| value.to_ascii_lowercase().contains(query))
}

fn search_rank(query: &str, haystacks: &[&str]) -> Option<i32> {
    let mut best: Option<i32> = None;
    for (index, value) in haystacks.iter().enumerate() {
        let normalized = value.trim().to_ascii_lowercase();
        if normalized.is_empty() {
            continue;
        }
        let field_weight = match index {
            0 => 140,
            1 => 90,
            _ => 50,
        };
        let score = if normalized == query {
            field_weight + 120
        } else if normalized.starts_with(query) {
            field_weight + 80
        } else if normalized.contains(query) {
            field_weight + 40
        } else {
            continue;
        };
        best = Some(best.map_or(score, |current| current.max(score)));
    }
    best
}

fn search_hit_icon(kind: &str) -> &'static str {
    match kind {
        "session" => "◉",
        "message" => "≡",
        "tool" => "⚒",
        "event" => "◌",
        "agent" => "◆",
        "task" => "▣",
        "channel" => "◎",
        _ => "•",
    }
}

fn search_hit_color(kind: &str) -> Color {
    match kind {
        "session" => Color::Cyan,
        "message" => Color::LightBlue,
        "tool" => Color::Yellow,
        "event" => Color::LightMagenta,
        "agent" => Color::LightGreen,
        "task" => Color::LightYellow,
        "channel" => Color::LightCyan,
        _ => Color::Gray,
    }
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

fn session_metadata_title(metadata: Option<&Value>) -> Option<String> {
    metadata?
        .get("title")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|title| !title.is_empty())
        .map(str::to_string)
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

fn excerpt_multiline(value: &str, max_lines: usize, max_chars: usize) -> String {
    let mut joined = value
        .lines()
        .take(max_lines)
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join(" / ");
    if joined.is_empty() {
        joined = value.trim().to_string();
    }
    excerpt(&joined, max_chars)
}

fn tail(value: &str, max_chars: usize) -> String {
    let chars = value.chars().collect::<Vec<_>>();
    if chars.len() <= max_chars {
        return value.to_string();
    }
    chars[chars.len() - max_chars..].iter().collect()
}

fn spinner_frame() -> &'static str {
    const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_millis())
        .unwrap_or(0);
    let index = ((millis / 120) % FRAMES.len() as u128) as usize;
    FRAMES[index]
}

fn message_has_renderable_assistant_text(
    message: &turin_control_client::SessionMessageDetail,
) -> bool {
    message.role == "assistant"
        && message_content_text(&message.content).is_some_and(|text| !text.trim().is_empty())
}

fn token_usage_from_detail(
    detail: &turin_control_client::SessionDetail,
) -> SessionTokenUsageSummary {
    let mut usage = SessionTokenUsageSummary::default();
    let mut current_turn_index = None;

    for event in &detail.events {
        match event.event_type.as_str() {
            "turn_start" => {
                current_turn_index = turn_index_from_event_payload(&event.payload);
            }
            "message_end" => {
                let input_tokens = payload_u64(&event.payload, "input_tokens");
                let output_tokens = payload_u64(&event.payload, "output_tokens");
                if input_tokens == 0 && output_tokens == 0 {
                    continue;
                }
                usage.total.record(input_tokens, output_tokens);
                if let Some(turn_index) = current_turn_index {
                    usage
                        .turns
                        .entry(turn_index)
                        .or_default()
                        .record(input_tokens, output_tokens);
                }
            }
            "turn_end" => {
                current_turn_index = None;
            }
            "session_end" if !usage.total.has_data() => {
                usage.total.input_tokens = payload_u64(&event.payload, "total_input_tokens");
                usage.total.output_tokens = payload_u64(&event.payload, "total_output_tokens");
            }
            _ => {}
        }
    }

    usage
}

fn turn_index_from_event_payload(payload: &Value) -> Option<u32> {
    payload
        .get("turn_index")
        .and_then(|value| value.as_u64())
        .map(|value| value as u32)
}

fn payload_u64(payload: &Value, key: &str) -> u64 {
    payload
        .get(key)
        .and_then(|value| value.as_u64())
        .unwrap_or(0)
}

fn assistant_turn_message_is_first(
    detail: &turin_control_client::SessionDetail,
    message_index: usize,
) -> bool {
    let message = &detail.messages[message_index];
    detail.messages.iter().take(message_index).all(|candidate| {
        candidate.turn_index != message.turn_index
            || !message_has_renderable_assistant_text(candidate)
    })
}

fn assistant_turn_message_is_last(
    detail: &turin_control_client::SessionDetail,
    message_index: usize,
) -> bool {
    let message = &detail.messages[message_index];
    detail
        .messages
        .iter()
        .skip(message_index + 1)
        .all(|candidate| {
            candidate.turn_index != message.turn_index
                || !message_has_renderable_assistant_text(candidate)
        })
}

fn assistant_turn_index_after_prompt(
    detail: &turin_control_client::SessionDetail,
    prompt: &str,
) -> Option<u32> {
    let user_index = detail.messages.iter().rposition(|message| {
        message.role == "user"
            && message_content_text(&message.content)
                .is_some_and(|text| text.trim() == prompt.trim())
    })?;

    detail
        .messages
        .iter()
        .skip(user_index + 1)
        .rev()
        .find(|message| message_has_renderable_assistant_text(message))
        .map(|message| message.turn_index)
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

fn trim_transcript_lines_to_budget(
    lines: &mut Vec<TranscriptLine>,
    budget_bytes: usize,
    focused_turn: Option<u32>,
) {
    if let Some(focused_turn) = focused_turn
        && let Some(focus_index) = lines
            .iter()
            .position(|line| line.turn_index == Some(focused_turn))
    {
        let line_costs = lines
            .iter()
            .map(transcript_line_cost)
            .collect::<Vec<_>>();
        let mut start = focus_index;
        while start > 0 && lines[start - 1].turn_index == Some(focused_turn) {
            start -= 1;
        }
        let mut end = focus_index + 1;
        while end < lines.len() && lines[end].turn_index == Some(focused_turn) {
            end += 1;
        }

        let mut total = line_costs[start..end].iter().sum::<usize>();
        let mut next_before = start;
        let mut next_after = end;
        let mut bias_before = true;

        while total < budget_bytes && (next_before > 0 || next_after < lines.len()) {
            let mut added = false;
            if bias_before
                && next_before > 0
                && total.saturating_add(line_costs[next_before - 1]) <= budget_bytes
            {
                next_before -= 1;
                total = total.saturating_add(line_costs[next_before]);
                added = true;
            }
            if next_after < lines.len()
                && total.saturating_add(line_costs[next_after]) <= budget_bytes
            {
                total = total.saturating_add(line_costs[next_after]);
                next_after += 1;
                added = true;
            }
            if !bias_before
                && next_before > 0
                && total.saturating_add(line_costs[next_before - 1]) <= budget_bytes
            {
                next_before -= 1;
                total = total.saturating_add(line_costs[next_before]);
                added = true;
            }
            if !added {
                break;
            }
            bias_before = !bias_before;
        }

        *lines = lines[next_before..next_after].to_vec();
        return;
    }

    let mut total = 0usize;
    let mut kept = Vec::new();
    for line in lines.iter().rev() {
        let line_text = line
            .line
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

fn transcript_line_cost(line: &TranscriptLine) -> usize {
    line.line
        .spans
        .iter()
        .map(|span| span.content.len())
        .sum::<usize>()
        + 1
}

fn format_token_usage_totals(usage: TokenUsageTotals) -> String {
    format!(
        "{} total · {} in / {} out",
        format_integer_with_commas(usage.total_tokens()),
        format_integer_with_commas(usage.input_tokens),
        format_integer_with_commas(usage.output_tokens)
    )
}

fn token_usage_footer_line(usage: TokenUsageTotals) -> Line<'static> {
    Line::from(Span::styled(
        format!("↳ Tokens {}", format_token_usage_totals(usage)),
        Style::default()
            .fg(Color::Rgb(128, 136, 148))
            .add_modifier(Modifier::DIM),
    ))
}

fn format_integer_with_commas(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, ch) in digits.chars().rev().enumerate() {
        if index > 0 && index % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out.chars().rev().collect()
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
        cursor_line_col, excerpt_multiline, format_integer_with_commas, nth_char_byte_index,
        slice_chars, split_input_lines, token_usage_from_detail, wrap_text_chunk,
    };
    use serde_json::json;
    use turin_control_client::{
        SessionDetail, SessionEventDetail, SessionSummary, SessionToolExecutionDetail,
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

    #[test]
    fn excerpt_multiline_compacts_multiple_lines() {
        assert_eq!(
            excerpt_multiline("alpha\nbeta\ngamma", 2, 80),
            "alpha / beta"
        );
    }

    #[test]
    fn token_usage_from_detail_tracks_totals_per_turn() {
        let detail = SessionDetail {
            session: SessionSummary {
                internal_id: 1,
                session_id: "s_1".to_string(),
                agent_id: "default".to_string(),
                metadata: None,
                created_at: "2026-03-25T00:00:00Z".to_string(),
            },
            events: vec![
                SessionEventDetail {
                    id: 1,
                    event_type: "turn_start".to_string(),
                    payload: json!({"turn_index": 0}),
                    created_at: "2026-03-25T00:00:01Z".to_string(),
                },
                SessionEventDetail {
                    id: 2,
                    event_type: "message_end".to_string(),
                    payload: json!({"input_tokens": 12, "output_tokens": 34}),
                    created_at: "2026-03-25T00:00:02Z".to_string(),
                },
                SessionEventDetail {
                    id: 3,
                    event_type: "turn_end".to_string(),
                    payload: json!({"turn_index": 0}),
                    created_at: "2026-03-25T00:00:03Z".to_string(),
                },
                SessionEventDetail {
                    id: 4,
                    event_type: "turn_start".to_string(),
                    payload: json!({"turn_index": 1}),
                    created_at: "2026-03-25T00:00:04Z".to_string(),
                },
                SessionEventDetail {
                    id: 5,
                    event_type: "message_end".to_string(),
                    payload: json!({"input_tokens": 5, "output_tokens": 8}),
                    created_at: "2026-03-25T00:00:05Z".to_string(),
                },
            ],
            messages: Vec::new(),
            tool_executions: Vec::<SessionToolExecutionDetail>::new(),
        };

        let usage = token_usage_from_detail(&detail);
        assert_eq!(usage.total.input_tokens, 17);
        assert_eq!(usage.total.output_tokens, 42);
        assert_eq!(usage.turns.get(&0).unwrap().total_tokens(), 46);
        assert_eq!(usage.turns.get(&1).unwrap().total_tokens(), 13);
    }

    #[test]
    fn format_integer_with_commas_groups_thousands() {
        assert_eq!(format_integer_with_commas(0), "0");
        assert_eq!(format_integer_with_commas(12), "12");
        assert_eq!(format_integer_with_commas(1_234), "1,234");
        assert_eq!(format_integer_with_commas(12_345_678), "12,345,678");
    }
}
