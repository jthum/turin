use anyhow::{Context, Result, anyhow};
use clap::Parser;
use crossterm::event::{self, Event as CEvent, KeyCode};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Tabs, Wrap};
use ratatui::{DefaultTerminal, Frame};
use serde::Serialize;
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
    ConnectionProfileSummary, DashboardFreshness, DashboardState, OperatorCommand, UiController,
    UiUpdate, connect_dashboard, ensure_local_daemon_for_draft, preflight_connection_blocking,
    preflight_draft_blocking, spawn_controller,
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
            '1' => Some(Self::Connections),
            '2' => Some(Self::Agents),
            '3' => Some(Self::LiveSessions),
            '4' => Some(Self::Sessions),
            '5' => Some(Self::Tasks),
            '6' => Some(Self::Channels),
            '7' => Some(Self::Events),
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
    profile_draft: ConnectionProfileDraft,
    draft_baseline: ConnectionProfileDraft,
    draft_baseline_label: String,
    recent_drafts: ConnectionDraftHistory,
    profile_activity: ConnectionProfileActivityBook,
    last_preflight_report: Option<ConnectionPreflightReport>,
    input_mode: Option<InputMode>,
    input: String,
    requested_session_detail: Option<String>,
    task_filter: String,
    channel_filter: String,
    event_filter: String,
    events_paused: bool,
    events_follow_latest: bool,
    paused_events: Vec<EventEnvelope>,
}

enum LoopAction {
    Quit,
    Reconnect {
        options: Box<ConnectionOptions>,
        connected_draft: Option<ConnectionProfileDraft>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let initial_options = connection_options(&args);

    enable_raw_mode()?;
    execute!(stdout(), EnterAlternateScreen)?;
    let mut terminal = ratatui::init();

    let loop_result = run_shell(&mut terminal, initial_options).await;

    ratatui::restore();
    disable_raw_mode()?;
    execute!(stdout(), LeaveAlternateScreen)?;

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
) -> Result<()> {
    let (mut app, mut controller) = connect_shell_state(
        initial_connection_options,
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
                match connect_shell_state(options, recent_drafts, profile_activity.clone()).await {
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

        if event::poll(Duration::from_millis(120)).context("Failed to poll terminal events")?
            && let CEvent::Key(key) = event::read().context("Failed to read terminal event")?
            && let Some(action) = handle_key(app, key.code, &controller.command_tx)?
        {
            return Ok(action);
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
        KeyCode::Char('r') => send_command(command_tx, OperatorCommand::Refresh)?,
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
            _ => {}
        },
        KeyCode::Char('p') => app.start_prompt_input(),
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
        KeyCode::Backspace => {
            app.input.pop();
        }
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
        KeyCode::Char(ch) => app.input.push(ch),
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

fn render(frame: &mut Frame<'_>, app: &mut TuiApp) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(6),
            Constraint::Length(3),
            Constraint::Min(12),
            Constraint::Length(4),
        ])
        .split(frame.area());

    render_banner(frame, app, layout[0]);
    render_tabs(frame, app, layout[1]);

    let main = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(38), Constraint::Percentage(62)])
        .split(layout[2]);
    render_left_panel(frame, app, main[0]);
    render_right_panel(frame, app, main[1]);
    render_footer(frame, app, layout[3]);
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
    .block(
        Block::default()
            .title("Connection")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)),
    )
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
        .block(
            Block::default()
                .title("Views")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
    frame.render_widget(tabs, area);
}

fn render_left_panel(frame: &mut Frame<'_>, app: &mut TuiApp, area: ratatui::layout::Rect) {
    let title = match app.tab {
        TabKind::Connections => "Connections",
        TabKind::Agents => "Agents",
        TabKind::LiveSessions => "Live Sessions",
        TabKind::Sessions => "Stored Sessions",
        TabKind::Tasks => "Tasks",
        TabKind::Channels => "Channels",
        TabKind::Events => "Events",
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
        .block(
            Block::default()
                .title(title)
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
    frame.render_stateful_widget(list, area, &mut state);
}

fn render_right_panel(frame: &mut Frame<'_>, app: &TuiApp, area: ratatui::layout::Rect) {
    let detail = Paragraph::new(app.detail_text())
        .block(
            Block::default()
                .title("Detail")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        )
        .wrap(Wrap { trim: false });
    frame.render_widget(detail, area);
}

fn render_footer(frame: &mut Frame<'_>, app: &TuiApp, area: ratatui::layout::Rect) {
    let lines = if app.input_mode.is_some() {
        match app.input_mode.as_ref() {
            Some(InputMode::SubmitPrompt { .. }) => vec![
                Line::from(vec![
                    Span::styled("Prompt> ", Style::default().fg(Color::LightCyan)),
                    Span::raw(app.input.clone()),
                ]),
                Line::from("Enter submits prompt to the selected live session. Esc cancels."),
            ],
            Some(InputMode::ConfirmDiscard { action }) => vec![
                Line::from(vec![
                    Span::styled("Discard> ", Style::default().fg(Color::LightYellow)),
                    Span::raw(action.description()),
                ]),
                Line::from("Press y or Enter to discard editor changes. Press n or Esc to cancel."),
            ],
            Some(InputMode::SaveProfile { make_default }) => vec![
                Line::from(vec![
                    Span::styled("Profile> ", Style::default().fg(Color::LightCyan)),
                    Span::raw(app.input.clone()),
                ]),
                Line::from(if *make_default {
                    "Enter saves the current draft under the typed name and marks it default. Esc cancels."
                } else {
                    "Enter saves the current draft under the typed profile name. Esc cancels."
                }),
            ],
            Some(InputMode::DuplicateProfile {
                source_name,
                make_default,
            }) => vec![
                Line::from(vec![
                    Span::styled("Duplicate> ", Style::default().fg(Color::LightCyan)),
                    Span::raw(app.input.clone()),
                ]),
                Line::from(if *make_default {
                    format!(
                        "Enter duplicates '{}' to the typed name and sets it as default. Esc cancels.",
                        source_name
                    )
                } else {
                    format!(
                        "Enter duplicates '{}' to the typed name. Esc cancels.",
                        source_name
                    )
                }),
            ],
            Some(InputMode::RenameProfile {
                source_name,
                make_default,
            }) => vec![
                Line::from(vec![
                    Span::styled("Rename> ", Style::default().fg(Color::LightCyan)),
                    Span::raw(app.input.clone()),
                ]),
                Line::from(if *make_default {
                    format!(
                        "Enter renames '{}' and marks the new name as default. Esc cancels.",
                        source_name
                    )
                } else {
                    format!("Enter renames '{}'. Esc cancels.", source_name)
                }),
            ],
            Some(InputMode::ConfirmDelete { profile_name }) => vec![
                Line::from(vec![
                    Span::styled("Delete> ", Style::default().fg(Color::LightRed)),
                    Span::raw(profile_name.clone()),
                ]),
                Line::from("Press y or Enter to confirm delete. Press n or Esc to cancel."),
            ],
            Some(InputMode::EditDraftTarget) => vec![
                Line::from(vec![
                    Span::styled("Target> ", Style::default().fg(Color::LightCyan)),
                    Span::raw(app.input.clone()),
                ]),
                Line::from("Enter updates the profile draft target. Esc cancels."),
            ],
            Some(InputMode::EditDraftAuth) => vec![
                Line::from(vec![
                    Span::styled("Auth> ", Style::default().fg(Color::LightCyan)),
                    Span::raw(app.input.clone()),
                ]),
                Line::from("Enter updates the profile draft auth value. Esc cancels."),
            ],
            Some(InputMode::EditTaskFilter) => vec![
                Line::from(vec![
                    Span::styled("Task Filter> ", Style::default().fg(Color::LightCyan)),
                    Span::raw(app.input.clone()),
                ]),
                Line::from("Enter updates the task filter. Esc cancels."),
            ],
            Some(InputMode::EditChannelFilter) => vec![
                Line::from(vec![
                    Span::styled("Channel Filter> ", Style::default().fg(Color::LightCyan)),
                    Span::raw(app.input.clone()),
                ]),
                Line::from("Enter updates the channel filter. Esc cancels."),
            ],
            Some(InputMode::EditEventFilter) => vec![
                Line::from(vec![
                    Span::styled("Event Filter> ", Style::default().fg(Color::LightCyan)),
                    Span::raw(app.input.clone()),
                ]),
                Line::from("Enter updates the event filter. Esc cancels."),
            ],
            None => Vec::new(),
        }
    } else {
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

    let footer = Paragraph::new(lines).block(
        Block::default()
            .title("Help")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)),
    );
    frame.render_widget(footer, area);
}

impl TuiApp {
    fn new(
        dashboard: DashboardState,
        connection_options: ConnectionOptions,
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
            profile_catalog,
            active_profile,
            tab: TabKind::Connections,
            profile_index: 0,
            recent_draft_index: 0,
            agent_index: 0,
            live_session_index: 0,
            session_index: 0,
            task_index: 0,
            channel_index: 0,
            event_index: 0,
            draft_baseline: profile_draft.clone(),
            draft_baseline_label: "current connection".to_string(),
            profile_draft,
            recent_drafts,
            profile_activity,
            last_preflight_report: None,
            input_mode: None,
            input: String::new(),
            requested_session_detail: None,
            task_filter: String::new(),
            channel_filter: String::new(),
            event_filter: String::new(),
            events_paused: false,
            events_follow_latest: true,
            paused_events: Vec::new(),
        };
        app.clamp_selection();
        app
    }

    fn apply_update(&mut self, update: UiUpdate) {
        let auto_follow_event = matches!(update, UiUpdate::Event(_))
            && !self.events_paused
            && self.events_follow_latest;
        self.dashboard.apply_update(update);
        if auto_follow_event {
            self.event_index = 0;
        }
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
            TabKind::Connections => self.profile_index,
            TabKind::Agents => self.agent_index,
            TabKind::LiveSessions => self.live_session_index,
            TabKind::Sessions => self.session_index,
            TabKind::Tasks => self.task_index,
            TabKind::Channels => self.channel_index,
            TabKind::Events => self.event_index,
        }
    }

    fn set_selected_index(&mut self, value: usize) {
        match self.tab {
            TabKind::Connections => self.profile_index = value,
            TabKind::Agents => self.agent_index = value,
            TabKind::LiveSessions => self.live_session_index = value,
            TabKind::Sessions => self.session_index = value,
            TabKind::Tasks => self.task_index = value,
            TabKind::Channels => self.channel_index = value,
            TabKind::Events => self.event_index = value,
        }
    }

    fn current_len(&self) -> usize {
        match self.tab {
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
            self.input_mode = Some(InputMode::ConfirmDiscard {
                action: PendingDraftAction::CurrentConnection,
            });
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
            self.input_mode = Some(InputMode::ConfirmDiscard {
                action: PendingDraftAction::SelectedProfile(profile_name.clone()),
            });
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
            self.input_mode = Some(InputMode::ConfirmDiscard {
                action: PendingDraftAction::SelectedRecentDraft,
            });
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

    fn start_prompt_input(&mut self) {
        if let Some(session) = self.selected_live_session() {
            self.input_mode = Some(InputMode::SubmitPrompt {
                session_id: session.session_id.clone(),
            });
            self.input.clear();
        }
    }

    fn start_save_profile_input(&mut self, make_default: bool) {
        self.input_mode = Some(InputMode::SaveProfile { make_default });
        self.input.clear();
    }

    fn start_edit_draft_target(&mut self) {
        self.input_mode = Some(InputMode::EditDraftTarget);
        self.input = self.profile_draft.target.clone();
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
        self.input_mode = Some(InputMode::EditDraftAuth);
        self.input = self.profile_draft.auth_value.clone();
    }

    fn start_edit_task_filter(&mut self) {
        self.input_mode = Some(InputMode::EditTaskFilter);
        self.input = self.task_filter.clone();
    }

    fn start_edit_channel_filter(&mut self) {
        self.input_mode = Some(InputMode::EditChannelFilter);
        self.input = self.channel_filter.clone();
    }

    fn start_edit_event_filter(&mut self) {
        self.input_mode = Some(InputMode::EditEventFilter);
        self.input = self.event_filter.clone();
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
        self.input_mode = Some(InputMode::DuplicateProfile {
            source_name: source_name.clone(),
            make_default,
        });
        self.input = format!("{source_name}-copy");
    }

    fn start_rename_profile_input(&mut self, make_default: bool) {
        let Some(source_name) = self.selected_profile().map(|profile| profile.name.clone()) else {
            self.dashboard
                .record_error("No connection profile is currently selected");
            return;
        };
        self.input_mode = Some(InputMode::RenameProfile {
            source_name: source_name.clone(),
            make_default,
        });
        self.input = source_name;
    }

    fn start_delete_confirmation(&mut self) {
        let Some(profile_name) = self.selected_profile().map(|profile| profile.name.clone()) else {
            self.dashboard
                .record_error("No connection profile is currently selected");
            return;
        };
        self.input_mode = Some(InputMode::ConfirmDelete {
            profile_name: profile_name.clone(),
        });
        self.input.clear();
        self.dashboard.record_info(format!(
            "Delete confirmation armed for connection profile '{}'",
            profile_name
        ));
    }

    fn clear_input_mode(&mut self) {
        self.input_mode = None;
        self.input.clear();
    }

    fn list_items(&self) -> Vec<ListItem<'static>> {
        match self.tab {
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
        }
    }

    fn detail_text(&self) -> String {
        match self.tab {
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
        }
    }

    fn help_text(&self) -> String {
        let shared = "1-7 switch views | Tab cycle | arrows/j/k move | r refresh | q quit";
        let scoped = match self.tab {
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
