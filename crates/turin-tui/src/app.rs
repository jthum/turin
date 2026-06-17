use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result};
use crossterm::event::{Event as CEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Row, Table, Wrap};
use serde_json::Value;
use turin_daemon_protocol::{EventEnvelope, WorkItemList};
use turin_ui_core::{
    ConnectionOptions, DashboardFreshness, DashboardState, OperatorCommand, UiController,
    UiListRequest, UiUpdate,
};

use crate::{harness_ui, theme};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TuiSignal {
    Continue,
    Quit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TabKind {
    Overview,
    Harness,
    Tasks,
    Events,
}

impl TabKind {
    const ALL: [Self; 4] = [Self::Overview, Self::Harness, Self::Tasks, Self::Events];

    fn title(self) -> &'static str {
        match self {
            Self::Overview => "Overview",
            Self::Harness => "Harness Apps",
            Self::Tasks => "Tasks",
            Self::Events => "Events",
        }
    }

    fn next(self) -> Self {
        let index = Self::ALL
            .iter()
            .position(|candidate| *candidate == self)
            .unwrap_or_default();
        Self::ALL[(index + 1) % Self::ALL.len()]
    }

    fn previous(self) -> Self {
        let index = Self::ALL
            .iter()
            .position(|candidate| *candidate == self)
            .unwrap_or_default();
        Self::ALL[(index + Self::ALL.len() - 1) % Self::ALL.len()]
    }
}

#[derive(Debug, Clone)]
pub struct PendingHarnessAction {
    pub app_id: String,
    pub label: String,
    pub action: String,
    pub agent_id: Option<String>,
    pub harness_id: Option<String>,
    pub params: Value,
}

pub struct TuiApp {
    dashboard: DashboardState,
    controller: UiController,
    connection_options: ConnectionOptions,
    tab: TabKind,
    quit: bool,
    show_help: bool,
    ui_app_index: usize,
    ui_screen_indices: BTreeMap<String, usize>,
    ui_action_index: usize,
    task_index: usize,
    event_index: usize,
    ui_list_requests: BTreeMap<String, UiListRequest>,
    ui_lists: BTreeMap<String, WorkItemList>,
    requested_ui_lists: BTreeSet<String>,
    pending_action: Option<PendingHarnessAction>,
}

impl TuiApp {
    pub fn new(
        dashboard: DashboardState,
        controller: UiController,
        connection_options: ConnectionOptions,
    ) -> Self {
        Self {
            dashboard,
            controller,
            connection_options,
            tab: TabKind::Overview,
            quit: false,
            show_help: false,
            ui_app_index: 0,
            ui_screen_indices: BTreeMap::new(),
            ui_action_index: 0,
            task_index: 0,
            event_index: 0,
            ui_list_requests: BTreeMap::new(),
            ui_lists: BTreeMap::new(),
            requested_ui_lists: BTreeSet::new(),
            pending_action: None,
        }
    }

    pub fn shutdown(&self) {
        self.controller.shutdown();
    }

    pub fn should_quit(&self) -> bool {
        self.quit
    }

    pub fn drain_updates(&mut self) {
        while let Ok(update) = self.controller.update_rx.try_recv() {
            self.apply_update(update);
        }
    }

    fn apply_update(&mut self, update: UiUpdate) {
        let harness_action_ran =
            matches!(&update, UiUpdate::Event(event) if event.event == "harness.action_ran");

        if let UiUpdate::UiListLoaded { request, items } = &update {
            let key = request.cache_key();
            self.ui_list_requests
                .insert(key.clone(), request.as_ref().clone());
            self.requested_ui_lists.remove(&key);
            self.ui_lists.insert(key, items.as_ref().clone());
        }

        self.dashboard.apply_update(update);
        let refreshed = self.apply_ui_refresh_intents();
        if harness_action_ran && refreshed == 0 {
            let _ = self.request_current_harness_lists(true);
        }
        self.clamp_selection();
    }

    pub fn ensure_visible_data(&mut self) -> Result<()> {
        if self.tab == TabKind::Harness {
            self.request_current_harness_lists(false)?;
        }
        Ok(())
    }

    pub fn handle_terminal_event(&mut self, event: CEvent) -> Result<TuiSignal> {
        let CEvent::Key(key) = event else {
            return Ok(TuiSignal::Continue);
        };
        if key.kind != KeyEventKind::Press {
            return Ok(TuiSignal::Continue);
        }

        if self.pending_action.is_some() {
            return self.handle_pending_action_key(key);
        }

        match key.code {
            KeyCode::Char('q') => {
                self.quit = true;
                Ok(TuiSignal::Quit)
            }
            KeyCode::Char('?') => {
                self.show_help = !self.show_help;
                Ok(TuiSignal::Continue)
            }
            KeyCode::Tab | KeyCode::Right => {
                self.tab = self.tab.next();
                self.clamp_selection();
                Ok(TuiSignal::Continue)
            }
            KeyCode::BackTab | KeyCode::Left => {
                self.tab = self.tab.previous();
                self.clamp_selection();
                Ok(TuiSignal::Continue)
            }
            KeyCode::Char('r') => {
                if self.tab == TabKind::Harness {
                    self.request_current_harness_lists(true)?;
                } else {
                    self.send_command(OperatorCommand::Refresh)?;
                }
                Ok(TuiSignal::Continue)
            }
            KeyCode::Char('j') | KeyCode::Down => {
                self.move_selection(1);
                Ok(TuiSignal::Continue)
            }
            KeyCode::Char('k') | KeyCode::Up => {
                self.move_selection(-1);
                Ok(TuiSignal::Continue)
            }
            KeyCode::Char(']') if self.tab == TabKind::Harness => {
                self.ui_app_index = offset_index(self.ui_app_index, self.ui_app_count(), 1);
                self.ui_action_index = 0;
                Ok(TuiSignal::Continue)
            }
            KeyCode::Char('[') if self.tab == TabKind::Harness => {
                self.ui_app_index = offset_index(self.ui_app_index, self.ui_app_count(), -1);
                self.ui_action_index = 0;
                Ok(TuiSignal::Continue)
            }
            KeyCode::Char('h') if key.modifiers.is_empty() => {
                self.move_harness_screen(-1);
                Ok(TuiSignal::Continue)
            }
            KeyCode::Char('l') if key.modifiers.is_empty() => {
                self.move_harness_screen(1);
                Ok(TuiSignal::Continue)
            }
            KeyCode::Enter => {
                self.activate_selection()?;
                Ok(TuiSignal::Continue)
            }
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.quit = true;
                Ok(TuiSignal::Quit)
            }
            _ => Ok(TuiSignal::Continue),
        }
    }

    fn handle_pending_action_key(&mut self, key: KeyEvent) -> Result<TuiSignal> {
        match key.code {
            KeyCode::Enter | KeyCode::Char('y') => {
                self.confirm_pending_action()?;
                Ok(TuiSignal::Continue)
            }
            KeyCode::Esc | KeyCode::Char('n') | KeyCode::Char('q') => {
                self.cancel_pending_action();
                Ok(TuiSignal::Continue)
            }
            _ => Ok(TuiSignal::Continue),
        }
    }

    fn move_selection(&mut self, delta: isize) {
        match self.tab {
            TabKind::Overview => {}
            TabKind::Harness => {
                let actions = self.current_harness_actions();
                if actions.is_empty() {
                    self.ui_app_index = offset_index(self.ui_app_index, self.ui_app_count(), delta);
                } else {
                    self.ui_action_index = offset_index(self.ui_action_index, actions.len(), delta);
                }
            }
            TabKind::Tasks => {
                self.task_index = offset_index(self.task_index, self.dashboard.tasks.len(), delta);
            }
            TabKind::Events => {
                self.event_index =
                    offset_index(self.event_index, self.dashboard.recent_events.len(), delta);
            }
        }
    }

    fn move_harness_screen(&mut self, delta: isize) {
        if self.tab != TabKind::Harness {
            return;
        }
        let Some(app) = self.selected_ui_app() else {
            return;
        };
        let screen_count = app.screens.len();
        let current = self.selected_screen_index(&app);
        self.ui_screen_indices
            .insert(app.id.clone(), offset_index(current, screen_count, delta));
        self.ui_action_index = 0;
    }

    fn activate_selection(&mut self) -> Result<()> {
        if self.tab != TabKind::Harness {
            return Ok(());
        }
        let Some(action) = self
            .current_harness_actions()
            .get(self.ui_action_index)
            .cloned()
        else {
            return Ok(());
        };

        if action.confirm {
            self.pending_action = Some(action.into_pending());
        } else {
            self.run_harness_action(action.into_pending())?;
        }
        Ok(())
    }

    fn confirm_pending_action(&mut self) -> Result<()> {
        let Some(action) = self.pending_action.take() else {
            return Ok(());
        };
        self.run_harness_action(action)
    }

    fn cancel_pending_action(&mut self) {
        if let Some(action) = self.pending_action.take() {
            self.dashboard.record_info(format!(
                "Cancelled harness UI action '{}' ({})",
                action.label, action.action
            ));
        }
    }

    fn run_harness_action(&mut self, action: PendingHarnessAction) -> Result<()> {
        self.dashboard.record_info(format!(
            "Running harness UI action '{}' ({})",
            action.label, action.action
        ));
        self.send_command(OperatorCommand::RunHarnessAction {
            agent_id: action.agent_id,
            harness_id: action.harness_id,
            action: action.action,
            params: action.params,
        })
    }

    fn send_command(&self, command: OperatorCommand) -> Result<()> {
        self.controller
            .command_tx
            .send(command)
            .context("failed to send operator command")
    }

    fn selected_ui_app(&self) -> Option<turin_ui_core::UiAppRecord> {
        self.dashboard.ui.apps().nth(self.ui_app_index).cloned()
    }

    fn ui_app_count(&self) -> usize {
        self.dashboard.ui.apps().count()
    }

    fn selected_screen_index(&self, app: &turin_ui_core::UiAppRecord) -> usize {
        self.ui_screen_indices
            .get(&app.id)
            .copied()
            .unwrap_or_else(|| harness_ui::default_screen_index(app))
            .min(app.screens.len().saturating_sub(1))
    }

    fn current_harness_actions(&self) -> Vec<harness_ui::HarnessAction> {
        let Some(app) = self.selected_ui_app() else {
            return Vec::new();
        };
        let screen_index = self.selected_screen_index(&app);
        harness_ui::screen_at(&app, screen_index)
            .map(|screen| harness_ui::collect_actions(&app, &screen.nodes))
            .unwrap_or_default()
    }

    fn current_harness_list_requests(&self) -> Vec<UiListRequest> {
        let Some(app) = self.selected_ui_app() else {
            return Vec::new();
        };
        let screen_index = self.selected_screen_index(&app);
        harness_ui::screen_at(&app, screen_index)
            .map(|screen| harness_ui::collect_list_requests(&screen.nodes))
            .unwrap_or_default()
    }

    fn request_current_harness_lists(&mut self, force: bool) -> Result<()> {
        for request in self.current_harness_list_requests() {
            self.request_ui_list(request, force)?;
        }
        Ok(())
    }

    fn request_ui_list(&mut self, request: UiListRequest, force: bool) -> Result<()> {
        let key = request.cache_key();
        self.ui_list_requests.insert(key.clone(), request.clone());
        if force {
            self.ui_lists.remove(&key);
            self.requested_ui_lists.remove(&key);
        } else if self.ui_lists.contains_key(&key) || self.requested_ui_lists.contains(&key) {
            return Ok(());
        }
        self.requested_ui_lists.insert(key);
        self.send_command(OperatorCommand::LoadUiList {
            request: Box::new(request),
        })
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
        let mut requests = Vec::new();
        let mut keys = BTreeSet::new();

        for (key, request) in &self.ui_list_requests {
            if request.source == binding {
                keys.insert(key.clone());
                requests.push(request.clone());
            }
        }

        for request in self.current_harness_list_requests() {
            let key = request.cache_key();
            if request.source == binding && keys.insert(key) {
                requests.push(request);
            }
        }

        for request in &requests {
            let key = request.cache_key();
            self.ui_lists.remove(&key);
            self.requested_ui_lists.remove(&key);
        }

        let count = requests.len();
        for request in requests {
            let _ = self.request_ui_list(request, true);
        }
        count
    }

    fn clamp_selection(&mut self) {
        self.ui_app_index = self.ui_app_index.min(self.ui_app_count().saturating_sub(1));
        self.task_index = self
            .task_index
            .min(self.dashboard.tasks.len().saturating_sub(1));
        self.event_index = self
            .event_index
            .min(self.dashboard.recent_events.len().saturating_sub(1));
        let action_count = self.current_harness_actions().len();
        self.ui_action_index = self.ui_action_index.min(action_count.saturating_sub(1));
    }

    pub fn render(&mut self, frame: &mut Frame<'_>) {
        let area = frame.area();
        frame.render_widget(Block::default().style(theme::base()), area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(8),
                Constraint::Length(2),
            ])
            .split(area);

        self.render_header(frame, chunks[0]);
        self.render_body(frame, chunks[1]);
        self.render_footer(frame, chunks[2]);

        if self.show_help {
            self.render_help(frame, centered_rect(70, 55, area));
        }
        if self.pending_action.is_some() {
            self.render_pending_action(frame, centered_rect(68, 42, area));
        }
    }

    fn render_header(&self, frame: &mut Frame<'_>, area: Rect) {
        let titles = TabKind::ALL
            .iter()
            .map(|tab| {
                if *tab == self.tab {
                    Span::styled(format!(" {} ", tab.title()), theme::accent())
                } else {
                    Span::styled(format!(" {} ", tab.title()), theme::muted())
                }
            })
            .collect::<Vec<_>>();
        let line = Line::from(titles);
        let connection = format!(
            "{}  {}",
            connection_kind_label(self.dashboard.connection_kind),
            self.dashboard.connection_target
        );
        let content = vec![
            Line::from(vec![
                Span::styled("Turin TUI", theme::title()),
                Span::raw("  "),
                Span::styled(connection, theme::muted()),
            ]),
            line,
        ];
        frame.render_widget(
            Paragraph::new(content).block(
                Block::default()
                    .borders(Borders::BOTTOM)
                    .style(theme::base()),
            ),
            area,
        );
    }

    fn render_body(&mut self, frame: &mut Frame<'_>, area: Rect) {
        match self.tab {
            TabKind::Overview => self.render_overview(frame, area),
            TabKind::Harness => self.render_harness(frame, area),
            TabKind::Tasks => self.render_tasks(frame, area),
            TabKind::Events => self.render_events(frame, area),
        }
    }

    fn render_overview(&self, frame: &mut Frame<'_>, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(34),
                Constraint::Percentage(33),
                Constraint::Percentage(33),
            ])
            .split(area);

        let health = self.dashboard.health.as_ref();
        let status_lines = vec![
            kv_line(
                "Freshness",
                freshness_label(self.dashboard.snapshot_freshness()),
            ),
            kv_line("Snapshot", self.dashboard.snapshot_age_label()),
            kv_line("Last refresh", self.dashboard.last_refresh_latency_label()),
            kv_line("Refresh status", self.dashboard.last_refresh_status_label()),
            kv_line("Events", self.dashboard.total_event_count.to_string()),
            kv_line(
                "Profile",
                self.connection_options
                    .profile
                    .clone()
                    .unwrap_or_else(|| "direct".to_string()),
            ),
        ];
        frame.render_widget(panel("Connection", status_lines), chunks[0]);

        let runtime_lines = vec![
            kv_line("Agents", self.dashboard.agents().len().to_string()),
            kv_line(
                "Live sessions",
                self.dashboard.live_sessions.len().to_string(),
            ),
            kv_line("Tasks", self.dashboard.tasks.len().to_string()),
            kv_line("Channels", self.dashboard.channels().len().to_string()),
            kv_line(
                "Issues",
                health
                    .map(|health| health.issue_count.to_string())
                    .unwrap_or_else(|| "-".to_string()),
            ),
        ];
        frame.render_widget(panel("Runtime", runtime_lines), chunks[1]);

        let mut notice_lines = Vec::new();
        if let Some(error) = &self.dashboard.last_error {
            notice_lines.push(Line::from(Span::styled(error.clone(), theme::danger())));
        }
        if let Some(info) = &self.dashboard.last_info {
            notice_lines.push(Line::from(Span::styled(info.clone(), theme::success())));
        }
        if notice_lines.is_empty() {
            notice_lines.push(Line::from(Span::styled(
                "No recent notices",
                theme::muted(),
            )));
        }
        frame.render_widget(panel("Notices", notice_lines), chunks[2]);
    }

    fn render_harness(&mut self, frame: &mut Frame<'_>, area: Rect) {
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Length(28),
                Constraint::Min(40),
                Constraint::Length(34),
            ])
            .split(area);
        self.render_harness_nav(frame, columns[0]);
        harness_ui::render_harness_screen(
            frame,
            columns[1],
            self.selected_ui_app().as_ref(),
            &self.ui_screen_indices,
            &self.ui_lists,
            &self.requested_ui_lists,
        );
        self.render_harness_inspector(frame, columns[2]);
    }

    fn render_harness_nav(&self, frame: &mut Frame<'_>, area: Rect) {
        let apps = self.dashboard.ui.apps().cloned().collect::<Vec<_>>();
        let mut items = apps
            .iter()
            .enumerate()
            .map(|(index, app)| {
                let title = app
                    .definition
                    .as_ref()
                    .map(|definition| definition.title.as_str())
                    .unwrap_or(app.id.as_str());
                let style = if index == self.ui_app_index {
                    theme::selected()
                } else {
                    theme::base()
                };
                ListItem::new(Line::from(vec![
                    Span::styled(
                        if index == self.ui_app_index {
                            "● "
                        } else {
                            "  "
                        },
                        style,
                    ),
                    Span::styled(title.to_string(), style),
                ]))
            })
            .collect::<Vec<_>>();

        if let Some(app) = self.selected_ui_app() {
            items.push(ListItem::new(Line::from("")));
            items.push(ListItem::new(Line::from(Span::styled(
                "Screens",
                theme::muted(),
            ))));
            let selected_screen = self.selected_screen_index(&app);
            for (index, screen) in app.screens.values().enumerate() {
                let style = if index == selected_screen {
                    theme::selected()
                } else {
                    theme::base()
                };
                items.push(ListItem::new(Line::from(vec![
                    Span::styled(
                        if index == selected_screen {
                            "● "
                        } else {
                            "  "
                        },
                        style,
                    ),
                    Span::styled(screen.title.clone(), style),
                ])));
            }
        }

        let list = if items.is_empty() {
            List::new(vec![ListItem::new(Line::from(Span::styled(
                "No harness UI apps",
                theme::muted(),
            )))])
        } else {
            List::new(items)
        };
        frame.render_widget(list.block(block("Apps")), area);
    }

    fn render_harness_inspector(&self, frame: &mut Frame<'_>, area: Rect) {
        let Some(app) = self.selected_ui_app() else {
            frame.render_widget(empty_panel("Inspector", "Select a harness app"), area);
            return;
        };
        let screen_index = self.selected_screen_index(&app);
        let screen_title = harness_ui::screen_at(&app, screen_index)
            .map(|screen| screen.title.as_str())
            .unwrap_or("No screen");
        let actions = self.current_harness_actions();

        let mut lines = vec![
            kv_line("Screen", screen_title),
            kv_line("Screens", app.screens.len().to_string()),
            kv_line("Actions", actions.len().to_string()),
            Line::from(""),
            Line::from(Span::styled("Actions", theme::title())),
        ];

        if actions.is_empty() {
            lines.push(Line::from(Span::styled(
                "No actions on this screen",
                theme::muted(),
            )));
        } else {
            for (index, action) in actions.iter().enumerate() {
                let marker = if index == self.ui_action_index {
                    "●"
                } else {
                    " "
                };
                let style = if index == self.ui_action_index {
                    theme::selected()
                } else if action.confirm {
                    theme::warning()
                } else {
                    theme::base()
                };
                lines.push(Line::from(vec![
                    Span::styled(format!("{marker} "), style),
                    Span::styled(action.label.clone(), style),
                ]));
            }
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "h/l screen  enter action",
            theme::muted(),
        )));
        frame.render_widget(panel("Inspector", lines), area);
    }

    fn render_tasks(&self, frame: &mut Frame<'_>, area: Rect) {
        let rows = self.dashboard.tasks.iter().map(|task| {
            Row::new(vec![
                task.request_id.clone(),
                task.agent_id.clone(),
                task.state.clone(),
                task.status.clone().unwrap_or_default(),
            ])
        });
        let table = Table::new(
            rows,
            [
                Constraint::Length(18),
                Constraint::Length(18),
                Constraint::Length(14),
                Constraint::Min(20),
            ],
        )
        .header(Row::new(vec!["Request", "Agent", "State", "Status"]).style(theme::accent()))
        .block(block("Tasks"));
        frame.render_widget(table, area);
    }

    fn render_events(&self, frame: &mut Frame<'_>, area: Rect) {
        let items = self
            .dashboard
            .recent_events
            .iter()
            .rev()
            .map(event_line)
            .collect::<Vec<_>>();
        let items = if items.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "No events yet",
                theme::muted(),
            )))]
        } else {
            items
        };
        frame.render_widget(List::new(items).block(block("Events")), area);
    }

    fn render_footer(&self, frame: &mut Frame<'_>, area: Rect) {
        let info = self
            .dashboard
            .last_info
            .as_deref()
            .or(self.dashboard.last_error.as_deref())
            .unwrap_or(
                "Tab/←/→ switch  j/k move  h/l screen  Enter act  r refresh  ? help  q quit",
            );
        frame.render_widget(
            Paragraph::new(Line::from(vec![Span::styled(
                info.to_string(),
                theme::muted(),
            )]))
            .alignment(Alignment::Center)
            .block(Block::default().borders(Borders::TOP).style(theme::base())),
            area,
        );
    }

    fn render_help(&self, frame: &mut Frame<'_>, area: Rect) {
        frame.render_widget(Clear, area);
        let lines = vec![
            Line::from(Span::styled("Keyboard", theme::title())),
            Line::from(""),
            kv_line("Tab / ← →", "switch workspace"),
            kv_line("j / k", "move selection"),
            kv_line("[ / ]", "switch harness app"),
            kv_line("h / l", "switch harness screen"),
            kv_line("Enter", "run selected harness action"),
            kv_line("r", "refresh current view"),
            kv_line("Esc / n", "cancel confirmation"),
            kv_line("y / Enter", "confirm action"),
            kv_line("?", "toggle this help"),
            kv_line("q", "quit"),
        ];
        frame.render_widget(panel("Help", lines), area);
    }

    fn render_pending_action(&self, frame: &mut Frame<'_>, area: Rect) {
        let Some(action) = self.pending_action.as_ref() else {
            return;
        };
        frame.render_widget(Clear, area);
        let mut lines = vec![
            Line::from(Span::styled("Confirm harness action", theme::warning())),
            Line::from(""),
            kv_line("App", action.app_id.as_str()),
            kv_line("Label", action.label.as_str()),
            kv_line("Action", action.action.as_str()),
        ];
        if let Some(harness_id) = action.harness_id.as_ref() {
            lines.push(kv_line("Harness", harness_id.as_str()));
        }
        if let Some(agent_id) = action.agent_id.as_ref() {
            lines.push(kv_line("Agent", agent_id.as_str()));
        }
        if !action.params.is_null() {
            lines.push(kv_line("Params", action.params.to_string()));
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Press y/Enter to run, Esc/n to cancel",
            theme::muted(),
        )));
        frame.render_widget(panel("Confirmation", lines), area);
    }
}

fn block(title: &'static str) -> Block<'static> {
    Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme::PANEL_HOT).bg(theme::BG))
        .style(Style::default().fg(theme::TEXT).bg(theme::BG))
}

fn panel(title: &'static str, lines: Vec<Line<'static>>) -> Paragraph<'static> {
    Paragraph::new(lines)
        .block(block(title))
        .wrap(Wrap { trim: true })
        .style(Style::default().bg(theme::BG))
}

fn empty_panel(title: &'static str, message: &'static str) -> Paragraph<'static> {
    panel(
        title,
        vec![Line::from(Span::styled(message, theme::muted()))],
    )
}

fn kv_line(label: impl Into<String>, value: impl Into<String>) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{:<14}", label.into()), theme::muted()),
        Span::styled(value.into(), theme::base()),
    ])
}

fn event_line(event: &EventEnvelope) -> ListItem<'static> {
    ListItem::new(Line::from(vec![
        Span::styled("● ", theme::accent()),
        Span::styled(event.event.clone(), theme::base()),
    ]))
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical[1])[1]
}

fn offset_index(current: usize, len: usize, delta: isize) -> usize {
    if len == 0 {
        return 0;
    }
    let len = len as isize;
    (current as isize + delta).rem_euclid(len) as usize
}

fn connection_kind_label(kind: turin_control_client::ConnectionKind) -> &'static str {
    match kind {
        turin_control_client::ConnectionKind::Local => "local",
        turin_control_client::ConnectionKind::Remote => "remote",
    }
}

fn freshness_label(freshness: DashboardFreshness) -> &'static str {
    match freshness {
        DashboardFreshness::Fresh => "fresh",
        DashboardFreshness::Quiet => "quiet",
        DashboardFreshness::Stale => "stale",
    }
}
