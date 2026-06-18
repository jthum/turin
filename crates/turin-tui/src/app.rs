use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result};
use crossterm::event::{Event as CEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Row, Table, Wrap};
use serde_json::Value;
use turin_control_client::TaskStatus;
use turin_daemon_protocol::{EventEnvelope, HarnessActionRunResult, UiFormNode, WorkItemList};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HarnessFocus {
    Navigation,
    Actions,
}

impl HarnessFocus {
    fn next(self) -> Self {
        match self {
            Self::Navigation => Self::Actions,
            Self::Actions => Self::Navigation,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Navigation => "navigation",
            Self::Actions => "actions",
        }
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

#[derive(Debug, Clone)]
struct TuiFormSession {
    app_id: String,
    form: UiFormNode,
    agent_id: Option<String>,
    harness_id: Option<String>,
    values: BTreeMap<String, String>,
    field_index: usize,
    error: Option<String>,
}

impl TuiFormSession {
    fn from_action(action: harness_ui::HarnessAction) -> Option<Self> {
        let form = action.form?;
        let values = form
            .fields
            .iter()
            .map(|field| {
                (
                    field.name.clone(),
                    harness_ui::default_form_value(&form, field),
                )
            })
            .collect::<BTreeMap<_, _>>();
        Some(Self {
            app_id: action.app_id,
            form,
            agent_id: action.agent_id,
            harness_id: action.harness_id,
            values,
            field_index: 0,
            error: None,
        })
    }

    fn selected_field(&self) -> Option<&turin_daemon_protocol::UiFormField> {
        self.form.fields.get(self.field_index)
    }
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
    ui_nav_indices: BTreeMap<String, usize>,
    ui_action_index: usize,
    harness_focus: HarnessFocus,
    task_index: usize,
    event_index: usize,
    ui_list_requests: BTreeMap<String, UiListRequest>,
    ui_lists: BTreeMap<String, WorkItemList>,
    requested_ui_lists: BTreeSet<String>,
    pending_action: Option<PendingHarnessAction>,
    active_form: Option<TuiFormSession>,
    latest_harness_action_result: Option<HarnessActionRunResult>,
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
            ui_nav_indices: BTreeMap::new(),
            ui_action_index: 0,
            harness_focus: HarnessFocus::Navigation,
            task_index: 0,
            event_index: 0,
            ui_list_requests: BTreeMap::new(),
            ui_lists: BTreeMap::new(),
            requested_ui_lists: BTreeSet::new(),
            pending_action: None,
            active_form: None,
            latest_harness_action_result: None,
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
        if let UiUpdate::HarnessActionCompleted(result) = &update {
            self.latest_harness_action_result = Some(result.as_ref().clone());
        }

        self.dashboard.apply_update(update);
        self.apply_ui_navigation_intents();
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

        if self.active_form.is_some() {
            return self.handle_form_key(key);
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
            KeyCode::Char('f') if self.tab == TabKind::Harness => {
                self.harness_focus = self.harness_focus.next();
                self.clamp_selection();
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
                self.sync_harness_nav_to_screen();
                Ok(TuiSignal::Continue)
            }
            KeyCode::Char('[') if self.tab == TabKind::Harness => {
                self.ui_app_index = offset_index(self.ui_app_index, self.ui_app_count(), -1);
                self.ui_action_index = 0;
                self.sync_harness_nav_to_screen();
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

    fn handle_form_key(&mut self, key: KeyEvent) -> Result<TuiSignal> {
        match key.code {
            KeyCode::Esc => {
                self.cancel_active_form();
                Ok(TuiSignal::Continue)
            }
            KeyCode::Enter => {
                self.submit_active_form()?;
                Ok(TuiSignal::Continue)
            }
            KeyCode::Tab | KeyCode::Down => {
                self.move_form_field(1);
                Ok(TuiSignal::Continue)
            }
            KeyCode::BackTab | KeyCode::Up => {
                self.move_form_field(-1);
                Ok(TuiSignal::Continue)
            }
            KeyCode::Left => {
                self.cycle_active_option(-1);
                Ok(TuiSignal::Continue)
            }
            KeyCode::Right => {
                self.cycle_active_option(1);
                Ok(TuiSignal::Continue)
            }
            KeyCode::Backspace => {
                self.delete_active_form_char();
                Ok(TuiSignal::Continue)
            }
            KeyCode::Delete => {
                self.clear_active_form_field();
                Ok(TuiSignal::Continue)
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.clear_active_form_field();
                Ok(TuiSignal::Continue)
            }
            KeyCode::Char(' ') => {
                if !self.toggle_active_bool() {
                    self.append_active_form_char(' ');
                }
                Ok(TuiSignal::Continue)
            }
            KeyCode::Char('h') if self.active_form_field_has_options() => {
                self.cycle_active_option(-1);
                Ok(TuiSignal::Continue)
            }
            KeyCode::Char('l') if self.active_form_field_has_options() => {
                self.cycle_active_option(1);
                Ok(TuiSignal::Continue)
            }
            KeyCode::Char('y') => {
                if !self.set_active_bool(true) {
                    self.append_active_form_char('y');
                }
                Ok(TuiSignal::Continue)
            }
            KeyCode::Char('n') => {
                if !self.set_active_bool(false) {
                    self.append_active_form_char('n');
                }
                Ok(TuiSignal::Continue)
            }
            KeyCode::Char(ch)
                if !key.modifiers.contains(KeyModifiers::CONTROL)
                    && !key.modifiers.contains(KeyModifiers::ALT) =>
            {
                self.append_active_form_char(ch);
                Ok(TuiSignal::Continue)
            }
            _ => Ok(TuiSignal::Continue),
        }
    }

    fn move_selection(&mut self, delta: isize) {
        match self.tab {
            TabKind::Overview => {}
            TabKind::Harness => match self.harness_focus {
                HarnessFocus::Navigation => self.move_harness_nav(delta),
                HarnessFocus::Actions => {
                    let actions = self.current_harness_actions();
                    if actions.is_empty() {
                        self.harness_focus = HarnessFocus::Navigation;
                        self.move_harness_nav(delta);
                    } else {
                        self.ui_action_index =
                            offset_index(self.ui_action_index, actions.len(), delta);
                    }
                }
            },
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
        self.sync_harness_nav_to_screen();
        self.ui_action_index = 0;
    }

    fn activate_selection(&mut self) -> Result<()> {
        if self.tab != TabKind::Harness {
            return Ok(());
        }
        if self.harness_focus == HarnessFocus::Navigation {
            self.open_selected_harness_nav()?;
            return Ok(());
        }
        let Some(action) = self
            .current_harness_actions()
            .get(self.ui_action_index)
            .cloned()
        else {
            return Ok(());
        };

        if action.form.is_some() {
            if let Some(form_session) = TuiFormSession::from_action(action) {
                self.dashboard.record_info(format!(
                    "Editing harness UI form '{}' ({})",
                    form_session.form.title, form_session.form.action
                ));
                self.active_form = Some(form_session);
            }
            return Ok(());
        }

        if action.confirm {
            self.pending_action = Some(action.into_pending());
        } else {
            self.run_harness_action(action.into_pending())?;
        }
        Ok(())
    }

    fn move_harness_nav(&mut self, delta: isize) {
        let Some(app) = self.selected_ui_app() else {
            return;
        };
        let items = harness_ui::collect_nav_items(&app);
        if items.is_empty() {
            self.ui_app_index = offset_index(self.ui_app_index, self.ui_app_count(), delta);
            return;
        }
        let index = self.selected_nav_index(&app, &items);
        self.ui_nav_indices
            .insert(app.id.clone(), offset_index(index, items.len(), delta));
    }

    fn open_selected_harness_nav(&mut self) -> Result<()> {
        let Some(app) = self.selected_ui_app() else {
            return Ok(());
        };
        let items = harness_ui::collect_nav_items(&app);
        let Some(item) = items.get(self.selected_nav_index(&app, &items)) else {
            return Ok(());
        };

        let screen_index = match &item.target {
            harness_ui::HarnessNavTarget::Screen { index } => Some(*index),
            harness_ui::HarnessNavTarget::Menu { opens } => {
                harness_ui::screen_index_for_target(&app, opens)
            }
        };
        let Some(screen_index) = screen_index else {
            self.dashboard.record_error(format!(
                "Navigation target '{}' is not a screen in '{}'",
                item.label, app.id
            ));
            return Ok(());
        };

        self.ui_screen_indices.insert(app.id.clone(), screen_index);
        self.ui_action_index = 0;
        self.request_current_harness_lists(false)
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

    fn submit_active_form(&mut self) -> Result<()> {
        let Some(mut form_session) = self.active_form.take() else {
            return Ok(());
        };
        match harness_ui::form_params(&form_session.form, &form_session.values) {
            Ok(params) => self.run_harness_action(PendingHarnessAction {
                app_id: form_session.app_id,
                label: format!("Submit {}", form_session.form.title),
                action: form_session.form.action,
                agent_id: form_session.agent_id,
                harness_id: form_session.harness_id,
                params,
            }),
            Err(message) => {
                form_session.error = Some(message.clone());
                self.dashboard.record_error(message);
                self.active_form = Some(form_session);
                Ok(())
            }
        }
    }

    fn cancel_active_form(&mut self) {
        if let Some(form_session) = self.active_form.take() {
            self.dashboard.record_info(format!(
                "Cancelled harness UI form '{}' ({})",
                form_session.form.title, form_session.form.action
            ));
        }
    }

    fn move_form_field(&mut self, delta: isize) {
        let Some(form_session) = self.active_form.as_mut() else {
            return;
        };
        form_session.field_index = offset_index(
            form_session.field_index,
            form_session.form.fields.len(),
            delta,
        );
        form_session.error = None;
    }

    fn append_active_form_char(&mut self, ch: char) {
        let Some(form_session) = self.active_form.as_mut() else {
            return;
        };
        let Some(field) = form_session.selected_field().cloned() else {
            return;
        };
        if !field.options.is_empty() || harness_ui::is_bool_field(&field) {
            return;
        }
        form_session.values.entry(field.name).or_default().push(ch);
        form_session.error = None;
    }

    fn delete_active_form_char(&mut self) {
        let Some(form_session) = self.active_form.as_mut() else {
            return;
        };
        let Some(field) = form_session.selected_field().cloned() else {
            return;
        };
        if !field.options.is_empty() || harness_ui::is_bool_field(&field) {
            return;
        }
        if let Some(value) = form_session.values.get_mut(&field.name) {
            value.pop();
        }
        form_session.error = None;
    }

    fn clear_active_form_field(&mut self) {
        let Some(form_session) = self.active_form.as_mut() else {
            return;
        };
        let Some(field) = form_session.selected_field().cloned() else {
            return;
        };
        if !field.options.is_empty() {
            if let Some(first) = field.options.first() {
                form_session
                    .values
                    .insert(field.name, harness_ui::form_value_string(first));
            }
            return;
        }
        if harness_ui::is_bool_field(&field) {
            form_session.values.insert(field.name, "false".to_string());
            return;
        }
        form_session.values.insert(field.name, String::new());
        form_session.error = None;
    }

    fn toggle_active_bool(&mut self) -> bool {
        let Some(form_session) = self.active_form.as_mut() else {
            return false;
        };
        let Some(field) = form_session.selected_field().cloned() else {
            return false;
        };
        if !harness_ui::is_bool_field(&field) {
            return false;
        }
        let current = form_session
            .values
            .get(&field.name)
            .is_some_and(|value| matches!(value.as_str(), "true" | "1" | "yes" | "on"));
        form_session
            .values
            .insert(field.name, (!current).to_string());
        form_session.error = None;
        true
    }

    fn set_active_bool(&mut self, value: bool) -> bool {
        let Some(form_session) = self.active_form.as_mut() else {
            return false;
        };
        let Some(field) = form_session.selected_field().cloned() else {
            return false;
        };
        if !harness_ui::is_bool_field(&field) {
            return false;
        }
        form_session.values.insert(field.name, value.to_string());
        form_session.error = None;
        true
    }

    fn active_form_field_has_options(&self) -> bool {
        self.active_form
            .as_ref()
            .and_then(TuiFormSession::selected_field)
            .is_some_and(|field| !field.options.is_empty())
    }

    fn cycle_active_option(&mut self, delta: isize) -> bool {
        let Some(form_session) = self.active_form.as_mut() else {
            return false;
        };
        let Some(field) = form_session.selected_field().cloned() else {
            return false;
        };
        if field.options.is_empty() {
            return false;
        }
        let labels = field
            .options
            .iter()
            .map(harness_ui::form_value_string)
            .collect::<Vec<_>>();
        let current = form_session
            .values
            .get(&field.name)
            .cloned()
            .unwrap_or_else(|| harness_ui::default_form_value(&form_session.form, &field));
        let index = labels
            .iter()
            .position(|label| *label == current)
            .unwrap_or_default();
        let next = offset_index(index, labels.len(), delta);
        if let Some(label) = labels.get(next) {
            form_session
                .values
                .insert(field.name.clone(), label.clone());
            form_session.error = None;
        }
        true
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

    fn selected_nav_index(
        &self,
        app: &turin_ui_core::UiAppRecord,
        items: &[harness_ui::HarnessNavItem],
    ) -> usize {
        self.ui_nav_indices
            .get(&app.id)
            .copied()
            .unwrap_or_else(|| self.screen_nav_index(app, items))
            .min(items.len().saturating_sub(1))
    }

    fn screen_nav_index(
        &self,
        app: &turin_ui_core::UiAppRecord,
        items: &[harness_ui::HarnessNavItem],
    ) -> usize {
        let screen_index = self.selected_screen_index(app);
        items
            .iter()
            .position(|item| {
                matches!(
                    item.target,
                    harness_ui::HarnessNavTarget::Screen { index } if index == screen_index
                )
            })
            .unwrap_or(0)
    }

    fn sync_harness_nav_to_screen(&mut self) {
        let Some(app) = self.selected_ui_app() else {
            return;
        };
        let items = harness_ui::collect_nav_items(&app);
        if items.is_empty() {
            self.ui_nav_indices.remove(&app.id);
            return;
        }
        self.ui_nav_indices
            .insert(app.id.clone(), self.screen_nav_index(&app, &items));
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

    fn apply_ui_navigation_intents(&mut self) {
        for open in self.dashboard.ui.take_opens() {
            self.apply_ui_open_request(&open.app_id, &open.target, "open");
        }
        for show in self.dashboard.ui.take_shows() {
            self.apply_ui_show_request(&show.app_id, &show.target);
        }
        for focus in self.dashboard.ui.take_focuses() {
            self.apply_ui_focus_request(&focus.app_id, &focus.target);
        }
    }

    fn apply_ui_open_request(&mut self, app_id: &str, target: &str, label: &str) {
        let Some(app) = self.select_ui_app_by_id(app_id) else {
            return;
        };
        let Some(screen_index) = harness_ui::screen_index_for_target(&app, target) else {
            self.dashboard.record_error(format!(
                "UI {label} target '{target}' is not a screen in '{app_id}'"
            ));
            return;
        };
        self.open_harness_screen(&app, screen_index, HarnessFocus::Navigation, 0);
        self.dashboard
            .record_info(format!("Opened '{target}' from ui.{label}"));
    }

    fn apply_ui_show_request(&mut self, app_id: &str, target: &str) {
        let Some(app) = self.select_ui_app_by_id(app_id) else {
            return;
        };
        if harness_ui::screen_index_for_target(&app, target).is_some() {
            self.apply_ui_open_request(app_id, target, "show");
            return;
        }
        if app.panes.contains_key(target) {
            self.dashboard.record_info(format!(
                "TUI noted ui.show pane '{target}' in '{app_id}', but panes are not rendered yet"
            ));
        } else {
            self.dashboard.record_error(format!(
                "UI show target '{target}' is not a screen or pane in '{app_id}'"
            ));
        }
    }

    fn apply_ui_focus_request(&mut self, app_id: &str, target: &str) {
        let Some(app) = self.select_ui_app_by_id(app_id) else {
            return;
        };
        let Some(target) = harness_ui::find_focus_target(&app, target) else {
            self.dashboard.record_error(format!(
                "UI focus target '{target}' was not found in '{app_id}'"
            ));
            return;
        };

        match target {
            harness_ui::HarnessFocusTarget::Screen { screen_index }
            | harness_ui::HarnessFocusTarget::Node { screen_index } => {
                self.open_harness_screen(&app, screen_index, HarnessFocus::Navigation, 0);
            }
            harness_ui::HarnessFocusTarget::Action {
                screen_index,
                action_index,
            } => {
                self.open_harness_screen(&app, screen_index, HarnessFocus::Actions, action_index);
            }
        }
    }

    fn select_ui_app_by_id(&mut self, app_id: &str) -> Option<turin_ui_core::UiAppRecord> {
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

    fn open_harness_screen(
        &mut self,
        app: &turin_ui_core::UiAppRecord,
        screen_index: usize,
        focus: HarnessFocus,
        action_index: usize,
    ) {
        self.tab = TabKind::Harness;
        self.ui_screen_indices.insert(app.id.clone(), screen_index);
        self.harness_focus = focus;
        self.ui_action_index = action_index;
        self.sync_harness_nav_to_screen();
        if let Err(err) = self.request_current_harness_lists(false) {
            self.dashboard
                .record_error(format!("Failed to load harness UI lists: {err}"));
        }
        self.clamp_selection();
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
        if let Some(app) = self.selected_ui_app() {
            let items = harness_ui::collect_nav_items(&app);
            if items.is_empty() {
                self.ui_nav_indices.remove(&app.id);
            } else {
                let index = self.selected_nav_index(&app, &items);
                self.ui_nav_indices.insert(app.id.clone(), index);
            }
        }
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
        if self.active_form.is_some() {
            self.render_active_form(frame, centered_rect(72, 58, area));
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
        for notice in self.dashboard.ui.notices().iter().rev().take(4) {
            notice_lines.push(ui_notice_line(notice));
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
        let mut items = Vec::new();

        items.push(ListItem::new(Line::from(Span::styled(
            "Apps",
            theme::muted(),
        ))));
        if apps.is_empty() {
            items.push(ListItem::new(Line::from(Span::styled(
                "  No harness UI apps",
                theme::muted(),
            ))));
        } else {
            for (index, app) in apps.iter().enumerate() {
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
                items.push(ListItem::new(Line::from(vec![
                    Span::styled(
                        if index == self.ui_app_index {
                            "● "
                        } else {
                            "  "
                        },
                        style,
                    ),
                    Span::styled(title.to_string(), style),
                ])));
            }
        }

        if let Some(app) = self.selected_ui_app() {
            items.push(ListItem::new(Line::from("")));
            let nav_items = harness_ui::collect_nav_items(&app);
            let selected_nav = self.selected_nav_index(&app, &nav_items);
            let active_screen = self.selected_screen_index(&app);
            let mut group = String::new();

            for (index, item) in nav_items.iter().enumerate() {
                if item.group != group {
                    group = item.group.clone();
                    items.push(ListItem::new(Line::from(Span::styled(
                        group.clone(),
                        theme::muted(),
                    ))));
                }
                let selected =
                    self.harness_focus == HarnessFocus::Navigation && index == selected_nav;
                let active = match &item.target {
                    harness_ui::HarnessNavTarget::Screen { index } => *index == active_screen,
                    harness_ui::HarnessNavTarget::Menu { opens } => {
                        harness_ui::screen_index_for_target(&app, opens) == Some(active_screen)
                    }
                };
                let style = if selected {
                    theme::selected()
                } else if active {
                    theme::accent()
                } else {
                    theme::base()
                };
                let prefix = if selected {
                    "● "
                } else if active {
                    "◆ "
                } else {
                    "  "
                };
                let indent = "  ".repeat(item.depth);
                let badge = item
                    .badge
                    .as_ref()
                    .map(|badge| format!("  [{badge}]"))
                    .unwrap_or_default();
                items.push(ListItem::new(Line::from(vec![
                    Span::styled(prefix, style),
                    Span::raw(indent),
                    Span::styled(item.label.clone(), style),
                    Span::styled(badge, theme::muted()),
                ])));
            }
        }

        frame.render_widget(List::new(items).block(block("Navigation")), area);
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
            kv_line("Focus", self.harness_focus.label()),
            kv_line("Screens", app.screens.len().to_string()),
            kv_line("Menus", app.menus.len().to_string()),
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
                let selected =
                    self.harness_focus == HarnessFocus::Actions && index == self.ui_action_index;
                let marker = if selected { "●" } else { " " };
                let style = if selected {
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

        if let Some(result) = self.latest_harness_action_result.as_ref() {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled("Latest Result", theme::title())));
            lines.push(kv_line("Action", result.action.clone()));
            lines.push(kv_line("Agent", result.agent_id.clone()));
            if let Some(harness_id) = result.harness_id.as_ref() {
                lines.push(kv_line("Harness", harness_id.clone()));
            }
            if !result.result.is_null() {
                lines.push(kv_line("Result", json_preview(&result.result, 220)));
            }
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "f focus  enter open/run  h/l screen",
            theme::muted(),
        )));
        frame.render_widget(panel("Inspector", lines), area);
    }

    fn render_tasks(&self, frame: &mut Frame<'_>, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(68), Constraint::Percentage(32)])
            .split(area);
        let rows = self
            .dashboard
            .tasks
            .iter()
            .enumerate()
            .map(|(index, task)| {
                let style = if index == self.task_index {
                    theme::selected()
                } else {
                    theme::base()
                };
                Row::new(vec![
                    task.request_id.clone(),
                    task.agent_id.clone(),
                    task.state.clone(),
                    task.status.clone().unwrap_or_default(),
                ])
                .style(style)
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
        frame.render_widget(table, chunks[0]);
        self.render_task_detail(frame, chunks[1]);
    }

    fn render_events(&self, frame: &mut Frame<'_>, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
            .split(area);
        let items = self
            .dashboard
            .recent_events
            .iter()
            .rev()
            .enumerate()
            .map(|(index, event)| event_line(event, index == self.event_index))
            .collect::<Vec<_>>();
        let items = if items.is_empty() {
            vec![ListItem::new(Line::from(Span::styled(
                "No events yet",
                theme::muted(),
            )))]
        } else {
            items
        };
        frame.render_widget(List::new(items).block(block("Events")), chunks[0]);
        self.render_event_detail(frame, chunks[1]);
    }

    fn render_task_detail(&self, frame: &mut Frame<'_>, area: Rect) {
        let Some(task) = self.dashboard.tasks.get(self.task_index) else {
            frame.render_widget(empty_panel("Task Detail", "No task selected"), area);
            return;
        };
        frame.render_widget(panel("Task Detail", task_detail_lines(task)), area);
    }

    fn render_event_detail(&self, frame: &mut Frame<'_>, area: Rect) {
        let Some(event) = self
            .dashboard
            .recent_events
            .iter()
            .rev()
            .nth(self.event_index)
        else {
            frame.render_widget(empty_panel("Event Detail", "No event selected"), area);
            return;
        };
        frame.render_widget(panel("Event Detail", event_detail_lines(event)), area);
    }

    fn render_footer(&self, frame: &mut Frame<'_>, area: Rect) {
        let fallback = if self.active_form.is_some() {
            "Form: Tab/↑/↓ fields  type edit  Space bool  h/l or ←/→ option  Enter submit  Esc cancel"
        } else {
            "Tab/←/→ tabs  f focus  j/k move  Enter open/run  r refresh  ? help  q quit"
        };
        let info = self
            .dashboard
            .last_info
            .as_deref()
            .or(self.dashboard.last_error.as_deref())
            .unwrap_or(fallback);
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
            kv_line("f", "cycle harness focus"),
            kv_line("j / k", "move selection"),
            kv_line("[ / ]", "switch harness app"),
            kv_line("h / l", "switch harness screen"),
            kv_line("Enter", "open selected nav item or run selected action"),
            kv_line("r", "refresh current view"),
            kv_line("Esc / n", "cancel confirmation"),
            kv_line("y / Enter", "confirm action"),
            kv_line("Form Tab / ↑ ↓", "move between form fields"),
            kv_line(
                "Form type",
                "edit text, number, integer, and textarea fields",
            ),
            kv_line("Form Space", "toggle boolean fields"),
            kv_line("Form h/l / ← →", "cycle option fields"),
            kv_line("Form Enter", "submit form"),
            kv_line("Form Esc", "cancel form"),
            kv_line("?", "toggle this help"),
            kv_line("q", "quit"),
        ];
        frame.render_widget(panel("Help", lines), area);
    }

    fn render_active_form(&self, frame: &mut Frame<'_>, area: Rect) {
        let Some(form_session) = self.active_form.as_ref() else {
            return;
        };
        frame.render_widget(Clear, area);
        let mut lines = vec![
            Line::from(Span::styled(
                format!(
                    "{}  ({})",
                    form_session.form.title, form_session.form.action
                ),
                theme::title(),
            )),
            Line::from(""),
        ];
        if form_session.form.fields.is_empty() {
            lines.push(Line::from(Span::styled(
                "This form has no editable fields.",
                theme::muted(),
            )));
        }
        for (index, field) in form_session.form.fields.iter().enumerate() {
            let selected = index == form_session.field_index;
            let style = if selected {
                theme::selected()
            } else {
                theme::base()
            };
            let value = form_session
                .values
                .get(&field.name)
                .cloned()
                .unwrap_or_else(|| harness_ui::default_form_value(&form_session.form, field));
            lines.push(Line::from(vec![
                Span::styled(if selected { "● " } else { "  " }, style),
                Span::styled(format!("{:<18}", field.label), style),
                Span::styled(format!("{:<22}", form_field_meta(field)), theme::muted()),
                Span::styled(form_field_value_preview(field, &value), style),
            ]));
        }
        if let Some(error) = form_session.error.as_ref() {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(error.clone(), theme::danger())));
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Tab/↑/↓ fields  type edit  Space bool  h/l or ←/→ options  Enter submit  Esc cancel",
            theme::muted(),
        )));
        frame.render_widget(panel("Form", lines), area);
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

fn task_detail_lines(task: &TaskStatus) -> Vec<Line<'static>> {
    let mut lines = vec![
        kv_line("Request", task.request_id.clone()),
        kv_line("Agent", task.agent_id.clone()),
        kv_line("Slot", task.slot_id.clone()),
        kv_line("Trace", task.trace_id.clone()),
        kv_line("State", task.state.clone()),
        kv_line("Execution", task.execution.execution_id.clone()),
        kv_line("Visibility", task.execution.visibility.clone()),
        kv_line("Durability", task.execution.durability.clone()),
    ];
    if let Some(status) = &task.status {
        lines.push(kv_line("Status", status.clone()));
    }
    if let Some(turns) = task.task_turn_count {
        lines.push(kv_line("Turns", turns.to_string()));
    }
    if let Some(runtime_task_id) = &task.runtime_task_id {
        lines.push(kv_line("Runtime", runtime_task_id.clone()));
    }
    if let Some(error) = &task.error {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("Error", theme::danger())));
        lines.push(Line::from(Span::styled(
            truncate(error, 220),
            theme::danger(),
        )));
    }
    if let Some(output) = &task.output {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("Output", theme::title())));
        lines.push(Line::from(Span::styled(
            truncate(output, 260),
            theme::base(),
        )));
    }
    if let Some(branch_outcome) = &task.branch_outcome {
        lines.push(Line::from(""));
        lines.push(kv_line("Branch", json_preview(branch_outcome, 160)));
    }
    lines
}

fn event_detail_lines(event: &EventEnvelope) -> Vec<Line<'static>> {
    let mut lines = vec![kv_line("Event", event.event.clone())];
    if !event.data.is_null() {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled("Data", theme::title())));
        lines.push(Line::from(Span::styled(
            json_preview(&event.data, 700),
            theme::base(),
        )));
    }
    lines
}

fn event_line(event: &EventEnvelope, selected: bool) -> ListItem<'static> {
    let style = if selected {
        theme::selected()
    } else {
        theme::base()
    };
    let preview = if event.data.is_null() {
        String::new()
    } else {
        format!("  {}", json_preview(&event.data, 80))
    };
    ListItem::new(Line::from(vec![
        Span::styled("● ", theme::accent()),
        Span::styled(event.event.clone(), style),
        Span::styled(preview, theme::muted()),
    ]))
    .style(style)
}

fn ui_notice_line(notice: &turin_daemon_protocol::UiNoticeIntent) -> Line<'static> {
    let style = match notice.level {
        Some(turin_daemon_protocol::UiNoticeLevel::Success) => theme::success(),
        Some(turin_daemon_protocol::UiNoticeLevel::Warning) => theme::warning(),
        Some(turin_daemon_protocol::UiNoticeLevel::Error) => theme::danger(),
        Some(turin_daemon_protocol::UiNoticeLevel::Info) | None => theme::base(),
    };
    let body = notice
        .body
        .as_ref()
        .map(|body| format!(" - {body}"))
        .unwrap_or_default();
    Line::from(vec![
        Span::styled(format!("{}: ", notice.app_id), theme::muted()),
        Span::styled(format!("{}{}", notice.title, body), style),
    ])
}

fn form_field_meta(field: &turin_daemon_protocol::UiFormField) -> String {
    let mut parts = vec![harness_ui::normalized_form_field_kind(field)];
    if field.required.unwrap_or(false) {
        parts.push("required".to_string());
    }
    if !field.options.is_empty() {
        parts.push(format!("{} options", field.options.len()));
    }
    truncate(&parts.join(" "), 20)
}

fn form_field_value_preview(field: &turin_daemon_protocol::UiFormField, value: &str) -> String {
    if harness_ui::is_bool_field(field) {
        if matches!(value, "true" | "1" | "yes" | "on") {
            return "[x]".to_string();
        }
        return "[ ]".to_string();
    }
    if !field.options.is_empty() {
        return format!("< {} >", truncate(value, 44));
    }
    if value.is_empty() {
        return "<empty>".to_string();
    }
    truncate(value, 52)
}

fn json_preview(value: &Value, max_chars: usize) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(value) => truncate(value, max_chars),
        Value::Bool(value) => value.to_string(),
        Value::Number(value) => value.to_string(),
        Value::Array(_) | Value::Object(_) => truncate(&value.to_string(), max_chars),
    }
}

fn truncate(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    if max_chars <= 3 {
        return ".".repeat(max_chars);
    }
    let mut out = String::new();
    let take_chars = max_chars - 3;
    for (index, ch) in value.chars().enumerate() {
        if index >= take_chars {
            out.push_str("...");
            return out;
        }
        out.push(ch);
    }
    out
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
