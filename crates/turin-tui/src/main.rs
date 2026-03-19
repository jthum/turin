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
use tokio::sync::mpsc;
use turin_control_client::{
    AgentRuntime, AgentSummary, ChannelRuntime, ChannelSummary, LiveSession, SessionSummary,
    TaskStatus,
};
use turin_daemon_protocol::EventEnvelope;
use turin_ui_core::{
    ConnectionOptions, DashboardState, OperatorCommand, UiUpdate, connect_dashboard,
    spawn_controller,
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
            '1' => Some(Self::Agents),
            '2' => Some(Self::LiveSessions),
            '3' => Some(Self::Sessions),
            '4' => Some(Self::Tasks),
            '5' => Some(Self::Channels),
            '6' => Some(Self::Events),
            _ => None,
        }
    }
}

enum InputMode {
    SubmitPrompt { session_id: String },
}

struct TuiApp {
    dashboard: DashboardState,
    tab: TabKind,
    agent_index: usize,
    live_session_index: usize,
    session_index: usize,
    task_index: usize,
    channel_index: usize,
    event_index: usize,
    input_mode: Option<InputMode>,
    input: String,
    requested_session_detail: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let spec = connection_options(&args).to_spec()?;
    let (client, dashboard) = connect_dashboard(&spec).await?;
    let mut app = TuiApp::new(dashboard);

    let controller = spawn_controller(&tokio::runtime::Handle::current(), client);
    let mut update_rx = controller.update_rx;
    let command_tx = controller.command_tx;

    enable_raw_mode()?;
    execute!(stdout(), EnterAlternateScreen)?;
    let mut terminal = ratatui::init();

    let loop_result = run_app(&mut terminal, &mut app, &mut update_rx, command_tx);

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
    }
}

fn run_app(
    terminal: &mut DefaultTerminal,
    app: &mut TuiApp,
    update_rx: &mut mpsc::UnboundedReceiver<UiUpdate>,
    command_tx: mpsc::UnboundedSender<OperatorCommand>,
) -> Result<()> {
    loop {
        while let Ok(update) = update_rx.try_recv() {
            app.apply_update(update);
        }

        app.ensure_session_detail_loaded(&command_tx)?;

        terminal.draw(|frame| render(frame, app))?;

        if event::poll(Duration::from_millis(120)).context("Failed to poll terminal events")?
            && let CEvent::Key(key) = event::read().context("Failed to read terminal event")?
            && handle_key(app, key.code, &command_tx)?
        {
            return Ok(());
        }
    }
}

fn handle_key(
    app: &mut TuiApp,
    key: KeyCode,
    command_tx: &mpsc::UnboundedSender<OperatorCommand>,
) -> Result<bool> {
    if app.input_mode.is_some() {
        return handle_input_mode(app, key, command_tx);
    }

    match key {
        KeyCode::Char('q') | KeyCode::Esc => return Ok(true),
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
    Ok(false)
}

fn handle_input_mode(
    app: &mut TuiApp,
    key: KeyCode,
    command_tx: &mpsc::UnboundedSender<OperatorCommand>,
) -> Result<bool> {
    match key {
        KeyCode::Esc => app.clear_input_mode(),
        KeyCode::Backspace => {
            app.input.pop();
        }
        KeyCode::Enter => {
            let prompt = app.input.trim().to_string();
            if prompt.is_empty() {
                app.dashboard.record_error("Prompt cannot be empty");
                app.clear_input_mode();
                return Ok(false);
            }
            if let Some(InputMode::SubmitPrompt { session_id }) = &app.input_mode {
                send_command(
                    command_tx,
                    OperatorCommand::SubmitPrompt {
                        session_id: session_id.clone(),
                        prompt,
                    },
                )?;
            }
            app.clear_input_mode();
        }
        KeyCode::Char(ch) => app.input.push(ch),
        _ => {}
    }
    Ok(false)
}

fn send_command(
    command_tx: &mpsc::UnboundedSender<OperatorCommand>,
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
            Constraint::Length(5),
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
            Span::styled("Counts: ", Style::default().fg(Color::Gray)),
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
        vec![
            Line::from(vec![
                Span::styled("Prompt> ", Style::default().fg(Color::LightCyan)),
                Span::raw(app.input.clone()),
            ]),
            Line::from("Enter submits prompt to the selected live session. Esc cancels."),
        ]
    } else {
        let mut lines = vec![Line::from(app.help_text())];
        if let Some(info) = &app.dashboard.last_info {
            lines.push(Line::from(Span::styled(
                info.clone(),
                Style::default().fg(Color::LightGreen),
            )));
        } else if let Some(error) = &app.dashboard.last_error {
            lines.push(Line::from(Span::styled(
                error.clone(),
                Style::default().fg(Color::LightRed),
            )));
        } else {
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
    fn new(dashboard: DashboardState) -> Self {
        let mut app = Self {
            dashboard,
            tab: TabKind::Agents,
            agent_index: 0,
            live_session_index: 0,
            session_index: 0,
            task_index: 0,
            channel_index: 0,
            event_index: 0,
            input_mode: None,
            input: String::new(),
            requested_session_detail: None,
        };
        app.clamp_selection();
        app
    }

    fn apply_update(&mut self, update: UiUpdate) {
        self.dashboard.apply_update(update);
        self.clamp_selection();
    }

    fn clamp_selection(&mut self) {
        self.agent_index = clamp_index(self.agent_index, self.dashboard.agents().len());
        self.live_session_index =
            clamp_index(self.live_session_index, self.dashboard.live_sessions.len());
        self.session_index = clamp_index(self.session_index, self.dashboard.sessions.len());
        self.task_index = clamp_index(self.task_index, self.dashboard.tasks.len());
        self.channel_index = clamp_index(self.channel_index, self.dashboard.channels().len());
        self.event_index = clamp_index(self.event_index, self.dashboard.recent_events.len());
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
            TabKind::Agents => self.dashboard.agents().len(),
            TabKind::LiveSessions => self.dashboard.live_sessions.len(),
            TabKind::Sessions => self.dashboard.sessions.len(),
            TabKind::Tasks => self.dashboard.tasks.len(),
            TabKind::Channels => self.dashboard.channels().len(),
            TabKind::Events => self.dashboard.recent_events.len(),
        }
    }

    fn selected_agent(&self) -> Option<&AgentSummary> {
        self.dashboard.agents().get(self.agent_index)
    }

    fn selected_live_session(&self) -> Option<&LiveSession> {
        self.dashboard.live_sessions.get(self.live_session_index)
    }

    fn selected_persisted_session(&self) -> Option<&SessionSummary> {
        self.dashboard.sessions.get(self.session_index)
    }

    fn selected_task(&self) -> Option<&TaskStatus> {
        self.dashboard.tasks.get(self.task_index)
    }

    fn selected_channel(&self) -> Option<&ChannelSummary> {
        self.dashboard.channels().get(self.channel_index)
    }

    fn start_prompt_input(&mut self) {
        if let Some(session) = self.selected_live_session() {
            self.input_mode = Some(InputMode::SubmitPrompt {
                session_id: session.session_id.clone(),
            });
            self.input.clear();
        }
    }

    fn clear_input_mode(&mut self) {
        self.input_mode = None;
        self.input.clear();
    }

    fn list_items(&self) -> Vec<ListItem<'static>> {
        match self.tab {
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
                .dashboard
                .tasks
                .iter()
                .map(|task| {
                    ListItem::new(format!(
                        "{}  {}  {}",
                        task.request_id, task.state, task.agent_id
                    ))
                })
                .collect(),
            TabKind::Channels => self
                .dashboard
                .channels()
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
                .dashboard
                .recent_events
                .iter()
                .rev()
                .map(|event| ListItem::new(event.event.clone()))
                .collect(),
        }
    }

    fn detail_text(&self) -> String {
        match self.tab {
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
                        "detail": self.dashboard.session_detail(&session.session_id),
                    }))
                })
                .unwrap_or_else(|| "No live sessions available.".to_string()),
            TabKind::Sessions => self
                .selected_persisted_session()
                .map(|session| {
                    pretty_json(&serde_json::json!({
                        "session": session,
                        "detail": self.dashboard.session_detail(&session.session_id),
                    }))
                })
                .unwrap_or_else(|| "No stored sessions available.".to_string()),
            TabKind::Tasks => self
                .selected_task()
                .map(pretty_json)
                .unwrap_or_else(|| "No tasks available.".to_string()),
            TabKind::Channels => self
                .selected_channel()
                .map(|channel| {
                    let runtime = self.channel_runtime(&channel.id);
                    pretty_json(&serde_json::json!({
                        "channel": channel,
                        "runtime": runtime,
                    }))
                })
                .unwrap_or_else(|| "No channels available.".to_string()),
            TabKind::Events => self
                .selected_event()
                .map(pretty_json)
                .unwrap_or_else(|| "No events yet.".to_string()),
        }
    }

    fn help_text(&self) -> String {
        let shared = "1-6 switch views | Tab cycle | arrows/j/k move | r refresh | q quit";
        let scoped = match self.tab {
            TabKind::Agents => "n or Enter opens a live session for the selected agent",
            TabKind::LiveSessions => "p or Enter prompts | c cancel session | x kill session",
            TabKind::Sessions => "e or Enter resumes the selected stored session",
            TabKind::Tasks => "c cancels the selected task",
            TabKind::Channels => "channel view is read-only in this pass",
            TabKind::Events => "event stream is live; latest entries are appended automatically",
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

    fn selected_event(&self) -> Option<&EventEnvelope> {
        self.dashboard
            .recent_events
            .iter()
            .rev()
            .nth(self.event_index)
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
        command_tx: &mpsc::UnboundedSender<OperatorCommand>,
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
