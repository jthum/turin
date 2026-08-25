use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result};
use crossterm::event::{Event as CEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, Paragraph, Row, Table, Wrap};
use serde_json::Value;
use turin_client::TaskStatus;
use turin_daemon_protocol::{EventEnvelope, HarnessActionRunResult, UiFormNode, WorkItemList};
use turin_ui_core::{
    ConnectionOptions, DashboardFreshness, DashboardState, DefaultOperatorConsoleSummary,
    HarnessActionFailure, OperatorCommand, UiController, UiListRequest, UiShowTarget, UiUpdate,
    ui_harness_action_failure_matches_app as harness_action_failure_matches_app,
    ui_harness_action_result_matches_app as harness_action_result_matches_app,
    ui_refresh_requests_for_binding, ui_show_target_for, work_item_key,
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
    Items,
    Actions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PaneFocus {
    Items,
    Actions,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelectionEdge {
    Start,
    End,
}

const SELECTION_PAGE_SIZE: isize = 8;

impl HarnessFocus {
    fn next_available(self, has_items: bool, has_actions: bool) -> Self {
        const ORDER: [HarnessFocus; 3] = [
            HarnessFocus::Navigation,
            HarnessFocus::Items,
            HarnessFocus::Actions,
        ];
        let current = ORDER
            .iter()
            .position(|candidate| *candidate == self)
            .unwrap_or_default();
        for offset in 1..=ORDER.len() {
            let candidate = ORDER[(current + offset) % ORDER.len()];
            if candidate.is_available(has_items, has_actions) {
                return candidate;
            }
        }
        Self::Navigation
    }

    fn is_available(self, has_items: bool, has_actions: bool) -> bool {
        match self {
            Self::Navigation => true,
            Self::Items => has_items,
            Self::Actions => has_actions,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Navigation => "navigation",
            Self::Items => "items",
            Self::Actions => "actions",
        }
    }
}

impl PaneFocus {
    fn next_available(self, has_items: bool, has_actions: bool) -> Self {
        let candidate = match self {
            Self::Items => Self::Actions,
            Self::Actions => Self::Items,
        };
        if candidate.is_available(has_items, has_actions) {
            candidate
        } else if self.is_available(has_items, has_actions) {
            self
        } else if has_items {
            Self::Items
        } else {
            Self::Actions
        }
    }

    fn is_available(self, has_items: bool, has_actions: bool) -> bool {
        match self {
            Self::Items => has_items,
            Self::Actions => has_actions,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Items => "items",
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

    fn clear_selected_field(&mut self) -> bool {
        let Some(field) = self.selected_field().cloned() else {
            return false;
        };
        if !field.options.is_empty() {
            if let Some(first) = field.options.first() {
                self.values
                    .insert(field.name, harness_ui::form_value_string(first));
            }
            self.error = None;
            return true;
        }
        if harness_ui::is_bool_field(&field) {
            self.values.insert(field.name, "false".to_string());
            self.error = None;
            return true;
        }
        self.values.insert(field.name, String::new());
        self.error = None;
        true
    }

    fn toggle_selected_bool(&mut self) -> bool {
        let Some(field) = self.selected_field().cloned() else {
            return false;
        };
        if !harness_ui::is_bool_field(&field) {
            return false;
        }
        let current = self
            .values
            .get(&field.name)
            .is_some_and(|value| matches!(value.as_str(), "true" | "1" | "yes" | "on"));
        self.values.insert(field.name, (!current).to_string());
        self.error = None;
        true
    }

    fn set_selected_bool(&mut self, value: bool) -> bool {
        let Some(field) = self.selected_field().cloned() else {
            return false;
        };
        if !harness_ui::is_bool_field(&field) {
            return false;
        }
        self.values.insert(field.name, value.to_string());
        self.error = None;
        true
    }

    fn cycle_selected_option(&mut self, delta: isize) -> bool {
        let Some(field) = self.selected_field().cloned() else {
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
        let current = self
            .values
            .get(&field.name)
            .cloned()
            .unwrap_or_else(|| harness_ui::default_form_value(&self.form, &field));
        let index = labels
            .iter()
            .position(|label| *label == current)
            .unwrap_or_default();
        let next = offset_index(index, labels.len(), delta);
        if let Some(label) = labels.get(next) {
            self.values.insert(field.name, label.clone());
            self.error = None;
        }
        true
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
    active_pane_id: Option<String>,
    ui_pane_focus: PaneFocus,
    ui_pane_item_index: usize,
    ui_pane_action_index: usize,
    selected_ui_pane_item_key: Option<String>,
    ui_action_index: usize,
    ui_item_index: usize,
    selected_ui_item_key: Option<String>,
    harness_focus: HarnessFocus,
    task_index: usize,
    event_index: usize,
    ui_list_requests: BTreeMap<String, UiListRequest>,
    ui_lists: BTreeMap<String, WorkItemList>,
    requested_ui_lists: BTreeSet<String>,
    ui_list_errors: BTreeMap<String, String>,
    pending_action: Option<PendingHarnessAction>,
    active_form: Option<TuiFormSession>,
    latest_harness_action_result: Option<HarnessActionRunResult>,
    latest_harness_action_failure: Option<HarnessActionFailure>,
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
            active_pane_id: None,
            ui_pane_focus: PaneFocus::Items,
            ui_pane_item_index: 0,
            ui_pane_action_index: 0,
            selected_ui_pane_item_key: None,
            ui_action_index: 0,
            ui_item_index: 0,
            selected_ui_item_key: None,
            harness_focus: HarnessFocus::Navigation,
            task_index: 0,
            event_index: 0,
            ui_list_requests: BTreeMap::new(),
            ui_lists: BTreeMap::new(),
            requested_ui_lists: BTreeSet::new(),
            ui_list_errors: BTreeMap::new(),
            pending_action: None,
            active_form: None,
            latest_harness_action_result: None,
            latest_harness_action_failure: None,
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
        }
        if let UiUpdate::HarnessActionFailed(failure) = &update {
            self.latest_harness_action_failure = Some(failure.as_ref().clone());
            self.latest_harness_action_result = None;
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

        if self.active_pane_id.is_some() {
            return self.handle_pane_key(key);
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
                self.cycle_harness_focus();
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
            KeyCode::PageDown => {
                self.move_selection_page(SELECTION_PAGE_SIZE);
                Ok(TuiSignal::Continue)
            }
            KeyCode::PageUp => {
                self.move_selection_page(-SELECTION_PAGE_SIZE);
                Ok(TuiSignal::Continue)
            }
            KeyCode::Home | KeyCode::End | KeyCode::Char('g') | KeyCode::Char('G') => {
                if let Some(edge) = selection_edge_for_key(key.code) {
                    self.move_selection_to_edge(edge);
                }
                Ok(TuiSignal::Continue)
            }
            KeyCode::Char(']') if self.tab == TabKind::Harness => {
                self.ui_app_index = offset_index(self.ui_app_index, self.ui_app_count(), 1);
                self.ui_action_index = 0;
                self.ui_item_index = 0;
                self.close_active_pane();
                self.sync_harness_nav_to_screen();
                Ok(TuiSignal::Continue)
            }
            KeyCode::Char('[') if self.tab == TabKind::Harness => {
                self.ui_app_index = offset_index(self.ui_app_index, self.ui_app_count(), -1);
                self.ui_action_index = 0;
                self.ui_item_index = 0;
                self.close_active_pane();
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

    fn handle_pane_key(&mut self, key: KeyEvent) -> Result<TuiSignal> {
        match key.code {
            KeyCode::Esc | KeyCode::Char('q') => {
                self.close_active_pane();
                Ok(TuiSignal::Continue)
            }
            KeyCode::Char('?') => {
                self.show_help = !self.show_help;
                Ok(TuiSignal::Continue)
            }
            KeyCode::Char('r') => {
                self.request_current_harness_lists(true)?;
                Ok(TuiSignal::Continue)
            }
            KeyCode::Char('f') => {
                self.cycle_pane_focus();
                self.clamp_selection();
                Ok(TuiSignal::Continue)
            }
            KeyCode::Char('j') | KeyCode::Down => {
                self.move_pane_selection(1);
                Ok(TuiSignal::Continue)
            }
            KeyCode::Char('k') | KeyCode::Up => {
                self.move_pane_selection(-1);
                Ok(TuiSignal::Continue)
            }
            KeyCode::PageDown => {
                self.move_pane_selection_page(SELECTION_PAGE_SIZE);
                Ok(TuiSignal::Continue)
            }
            KeyCode::PageUp => {
                self.move_pane_selection_page(-SELECTION_PAGE_SIZE);
                Ok(TuiSignal::Continue)
            }
            KeyCode::Home | KeyCode::End | KeyCode::Char('g') | KeyCode::Char('G') => {
                if let Some(edge) = selection_edge_for_key(key.code) {
                    self.move_pane_selection_to_edge(edge);
                }
                Ok(TuiSignal::Continue)
            }
            KeyCode::Enter => {
                self.activate_pane_selection()?;
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
            KeyCode::Char('j') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.append_active_form_newline();
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
                HarnessFocus::Items => {
                    let items = self.current_harness_work_items();
                    if !items.is_empty() {
                        self.ui_item_index = offset_index(self.ui_item_index, items.len(), delta);
                        self.remember_current_harness_item();
                    }
                }
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

    fn move_selection_page(&mut self, delta: isize) {
        match self.tab {
            TabKind::Overview => {}
            TabKind::Harness => match self.harness_focus {
                HarnessFocus::Navigation => self.move_harness_nav_page(delta),
                HarnessFocus::Items => {
                    let items = self.current_harness_work_items();
                    self.ui_item_index = page_index(self.ui_item_index, items.len(), delta);
                    self.remember_current_harness_item();
                }
                HarnessFocus::Actions => {
                    let actions = self.current_harness_actions();
                    self.ui_action_index = page_index(self.ui_action_index, actions.len(), delta);
                }
            },
            TabKind::Tasks => {
                self.task_index = page_index(self.task_index, self.dashboard.tasks.len(), delta);
            }
            TabKind::Events => {
                self.event_index =
                    page_index(self.event_index, self.dashboard.recent_events.len(), delta);
            }
        }
    }

    fn move_selection_to_edge(&mut self, edge: SelectionEdge) {
        match self.tab {
            TabKind::Overview => {}
            TabKind::Harness => match self.harness_focus {
                HarnessFocus::Navigation => self.move_harness_nav_to_edge(edge),
                HarnessFocus::Items => {
                    let items = self.current_harness_work_items();
                    self.ui_item_index = edge_index(items.len(), edge);
                    self.remember_current_harness_item();
                }
                HarnessFocus::Actions => {
                    let actions = self.current_harness_actions();
                    self.ui_action_index = edge_index(actions.len(), edge);
                }
            },
            TabKind::Tasks => {
                self.task_index = edge_index(self.dashboard.tasks.len(), edge);
            }
            TabKind::Events => {
                self.event_index = edge_index(self.dashboard.recent_events.len(), edge);
            }
        }
    }

    fn cycle_pane_focus(&mut self) {
        let has_items = !self.current_pane_work_items().is_empty();
        let has_actions = !self.current_pane_actions().is_empty();
        self.ui_pane_focus = self.ui_pane_focus.next_available(has_items, has_actions);
    }

    fn move_pane_selection(&mut self, delta: isize) {
        match self.ui_pane_focus {
            PaneFocus::Items => {
                let items = self.current_pane_work_items();
                if !items.is_empty() {
                    self.ui_pane_item_index =
                        offset_index(self.ui_pane_item_index, items.len(), delta);
                    self.remember_current_pane_item();
                }
            }
            PaneFocus::Actions => {
                let actions = self.current_pane_actions();
                if !actions.is_empty() {
                    self.ui_pane_action_index =
                        offset_index(self.ui_pane_action_index, actions.len(), delta);
                }
            }
        }
    }

    fn move_pane_selection_page(&mut self, delta: isize) {
        match self.ui_pane_focus {
            PaneFocus::Items => {
                let items = self.current_pane_work_items();
                self.ui_pane_item_index = page_index(self.ui_pane_item_index, items.len(), delta);
                self.remember_current_pane_item();
            }
            PaneFocus::Actions => {
                let actions = self.current_pane_actions();
                self.ui_pane_action_index =
                    page_index(self.ui_pane_action_index, actions.len(), delta);
            }
        }
    }

    fn move_pane_selection_to_edge(&mut self, edge: SelectionEdge) {
        match self.ui_pane_focus {
            PaneFocus::Items => {
                let items = self.current_pane_work_items();
                self.ui_pane_item_index = edge_index(items.len(), edge);
                self.remember_current_pane_item();
            }
            PaneFocus::Actions => {
                let actions = self.current_pane_actions();
                self.ui_pane_action_index = edge_index(actions.len(), edge);
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
        self.ui_item_index = 0;
        self.selected_ui_item_key = None;
        self.close_active_pane();
    }

    fn move_harness_nav_page(&mut self, delta: isize) {
        let Some(app) = self.selected_ui_app() else {
            return;
        };
        let items = harness_ui::collect_nav_items(&app);
        if items.is_empty() {
            self.ui_nav_indices.remove(&app.id);
            return;
        }
        let index = self.selected_nav_index(&app, &items);
        self.ui_nav_indices
            .insert(app.id.clone(), page_index(index, items.len(), delta));
    }

    fn move_harness_nav_to_edge(&mut self, edge: SelectionEdge) {
        let Some(app) = self.selected_ui_app() else {
            return;
        };
        let items = harness_ui::collect_nav_items(&app);
        if items.is_empty() {
            self.ui_nav_indices.remove(&app.id);
            return;
        }
        self.ui_nav_indices
            .insert(app.id.clone(), edge_index(items.len(), edge));
    }

    fn cycle_harness_focus(&mut self) {
        let has_items = !self.current_harness_work_items().is_empty();
        let has_actions = !self.current_harness_actions().is_empty();
        self.harness_focus = self.harness_focus.next_available(has_items, has_actions);
    }

    fn activate_selection(&mut self) -> Result<()> {
        if self.tab != TabKind::Harness {
            return Ok(());
        }
        if self.harness_focus == HarnessFocus::Navigation {
            self.open_selected_harness_nav()?;
            return Ok(());
        }
        if self.harness_focus == HarnessFocus::Items {
            if let Some(selection) = self.selected_harness_work_item() {
                if let Some(app) = self.selected_ui_app()
                    && let Some(action) = pending_action_from_work_item(&app, &selection)
                {
                    self.dashboard.record_info(format!(
                        "Work item action '{}' requires confirmation before running",
                        action.action
                    ));
                    self.pending_action = Some(action);
                } else {
                    self.dashboard.record_info(format!(
                        "Selected work item '{}' from {}",
                        selection.item.title, selection.list_title
                    ));
                }
            }
            return Ok(());
        }
        let Some(action) = self
            .current_harness_actions()
            .get(self.ui_action_index)
            .cloned()
        else {
            return Ok(());
        };

        self.start_harness_action(action)
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
        self.ui_item_index = 0;
        self.selected_ui_item_key = None;
        self.close_active_pane();
        self.request_current_harness_lists(false)
    }

    fn activate_pane_selection(&mut self) -> Result<()> {
        match self.ui_pane_focus {
            PaneFocus::Items => self.activate_pane_work_item(),
            PaneFocus::Actions => self.activate_pane_action(),
        }
    }

    fn activate_pane_work_item(&mut self) -> Result<()> {
        if let Some(selection) = self.selected_pane_work_item() {
            if let Some(app) = self.selected_ui_app()
                && let Some(action) = pending_action_from_work_item(&app, &selection)
            {
                self.dashboard.record_info(format!(
                    "Work item action '{}' requires confirmation before running",
                    action.action
                ));
                self.pending_action = Some(action);
            } else {
                self.dashboard.record_info(format!(
                    "Selected pane work item '{}' from {}",
                    selection.item.title, selection.list_title
                ));
            }
        } else {
            self.dashboard
                .record_info("No work item is available in this pane");
        }
        Ok(())
    }

    fn activate_pane_action(&mut self) -> Result<()> {
        let Some(action) = self
            .current_pane_actions()
            .get(self.ui_pane_action_index)
            .cloned()
        else {
            self.dashboard
                .record_info("No action is available in this pane");
            return Ok(());
        };
        self.start_harness_action(action)
    }

    fn start_harness_action(&mut self, action: harness_ui::HarnessAction) -> Result<()> {
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

    fn close_active_pane(&mut self) {
        self.active_pane_id = None;
        self.ui_pane_focus = PaneFocus::Items;
        self.ui_pane_item_index = 0;
        self.ui_pane_action_index = 0;
        self.selected_ui_pane_item_key = None;
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

    fn append_active_form_newline(&mut self) -> bool {
        let Some(form_session) = self.active_form.as_mut() else {
            return false;
        };
        let Some(field) = form_session.selected_field().cloned() else {
            return false;
        };
        if !harness_ui::is_multiline_field(&field) {
            return false;
        }
        form_session
            .values
            .entry(field.name)
            .or_default()
            .push('\n');
        form_session.error = None;
        true
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
        form_session.clear_selected_field();
    }

    fn toggle_active_bool(&mut self) -> bool {
        let Some(form_session) = self.active_form.as_mut() else {
            return false;
        };
        form_session.toggle_selected_bool()
    }

    fn set_active_bool(&mut self, value: bool) -> bool {
        let Some(form_session) = self.active_form.as_mut() else {
            return false;
        };
        form_session.set_selected_bool(value)
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
        form_session.cycle_selected_option(delta)
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

    fn current_pane_actions(&self) -> Vec<harness_ui::HarnessAction> {
        let Some(app) = self.selected_ui_app() else {
            return Vec::new();
        };
        pane_actions(&app, self.active_pane_id.as_deref())
    }

    fn current_pane_work_items(&self) -> Vec<harness_ui::HarnessWorkItemSelection> {
        let Some(app) = self.selected_ui_app() else {
            return Vec::new();
        };
        pane_work_items(&app, self.active_pane_id.as_deref(), &self.ui_lists)
    }

    fn current_harness_list_requests(&self) -> Vec<UiListRequest> {
        let Some(app) = self.selected_ui_app() else {
            return Vec::new();
        };
        let screen_index = self.selected_screen_index(&app);
        visible_harness_list_requests(&app, screen_index, self.active_pane_id.as_deref())
    }

    fn current_harness_work_items(&self) -> Vec<harness_ui::HarnessWorkItemSelection> {
        let Some(app) = self.selected_ui_app() else {
            return Vec::new();
        };
        let screen_index = self.selected_screen_index(&app);
        harness_ui::screen_at(&app, screen_index)
            .map(|screen| harness_ui::collect_work_item_selections(&screen.nodes, &self.ui_lists))
            .unwrap_or_default()
    }

    fn selected_harness_work_item(&self) -> Option<harness_ui::HarnessWorkItemSelection> {
        self.current_harness_work_items()
            .get(self.ui_item_index)
            .cloned()
    }

    fn selected_pane_work_item(&self) -> Option<harness_ui::HarnessWorkItemSelection> {
        self.current_pane_work_items()
            .get(self.ui_pane_item_index)
            .cloned()
    }

    fn remember_current_harness_item(&mut self) {
        let items = self.current_harness_work_items();
        self.selected_ui_item_key = items.get(self.ui_item_index).map(work_item_selection_key);
    }

    fn remember_current_pane_item(&mut self) {
        let items = self.current_pane_work_items();
        self.selected_ui_pane_item_key = items
            .get(self.ui_pane_item_index)
            .map(work_item_selection_key);
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
            self.ui_list_errors.remove(&key);
        } else if self.ui_lists.contains_key(&key)
            || self.requested_ui_lists.contains(&key)
            || self.ui_list_errors.contains_key(&key)
        {
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
        let requests = ui_refresh_requests_for_binding(
            binding,
            &self.ui_list_requests,
            self.current_harness_list_requests(),
        );

        for request in &requests {
            let key = request.cache_key();
            self.ui_lists.remove(&key);
            self.requested_ui_lists.remove(&key);
            self.ui_list_errors.remove(&key);
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
        match ui_show_target_for(&app, target) {
            Some(UiShowTarget::Screen { screen_index }) => {
                self.open_harness_screen(&app, screen_index, HarnessFocus::Navigation, 0);
                self.dashboard
                    .record_info(format!("Opened '{target}' from ui.show"));
            }
            Some(UiShowTarget::Pane { pane_id }) => {
                self.tab = TabKind::Harness;
                self.active_pane_id = Some(pane_id.to_string());
                self.ui_pane_focus = PaneFocus::Items;
                self.ui_pane_item_index = 0;
                self.ui_pane_action_index = 0;
                self.selected_ui_pane_item_key = None;
                if let Err(err) = self.request_current_harness_lists(false) {
                    self.dashboard
                        .record_error(format!("Failed to load harness UI pane lists: {err}"));
                }
                self.dashboard
                    .record_info(format!("Opened pane '{target}' from ui.show"));
            }
            None => {
                self.dashboard.record_error(format!(
                    "UI show target '{target}' is not a screen or pane in '{app_id}'"
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
        self.ui_item_index = 0;
        self.selected_ui_item_key = None;
        self.close_active_pane();
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
        let work_items = self.current_harness_work_items();
        let item_count = work_items.len();
        reconcile_work_item_selection(
            &mut self.ui_item_index,
            &mut self.selected_ui_item_key,
            &work_items,
        );
        if !self
            .harness_focus
            .is_available(item_count > 0, action_count > 0)
        {
            self.harness_focus = HarnessFocus::Navigation;
        }
        if let Some(app) = self.selected_ui_app() {
            if let Some(pane_id) = self.active_pane_id.as_deref()
                && !app.panes.contains_key(pane_id)
            {
                self.close_active_pane();
            }
            let items = harness_ui::collect_nav_items(&app);
            if items.is_empty() {
                self.ui_nav_indices.remove(&app.id);
            } else {
                let index = self.selected_nav_index(&app, &items);
                self.ui_nav_indices.insert(app.id.clone(), index);
            }
        }
        let pane_action_count = self.current_pane_actions().len();
        let pane_work_items = self.current_pane_work_items();
        let pane_item_count = pane_work_items.len();
        reconcile_work_item_selection(
            &mut self.ui_pane_item_index,
            &mut self.selected_ui_pane_item_key,
            &pane_work_items,
        );
        self.ui_pane_action_index = self
            .ui_pane_action_index
            .min(pane_action_count.saturating_sub(1));
        if self.active_pane_id.is_some()
            && !self
                .ui_pane_focus
                .is_available(pane_item_count > 0, pane_action_count > 0)
        {
            self.ui_pane_focus = if pane_item_count > 0 {
                PaneFocus::Items
            } else {
                PaneFocus::Actions
            };
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

        if self.active_pane_id.is_some() {
            self.render_active_pane(frame, centered_rect(78, 72, area));
        }
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
        let selected_item = self.selected_harness_work_item();
        let selected_item_id = selected_item
            .as_ref()
            .map(|selection| selection.item.public_id.as_str());
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Length(28),
                Constraint::Min(40),
                Constraint::Length(34),
            ])
            .split(area);
        self.render_harness_nav(frame, columns[0]);
        let render_state = harness_ui::HarnessRenderState {
            screen_indices: &self.ui_screen_indices,
            lists: &self.ui_lists,
            requested_lists: &self.requested_ui_lists,
            list_errors: &self.ui_list_errors,
            selected_work_item_id: selected_item_id,
            selected_action_index: None,
        };
        harness_ui::render_harness_screen(
            frame,
            columns[1],
            self.selected_ui_app().as_ref(),
            &render_state,
        );
        self.render_harness_inspector(frame, columns[2]);
    }

    fn render_active_pane(&self, frame: &mut Frame<'_>, area: Rect) {
        let selected_item = self.selected_pane_work_item();
        let selected_item_id = selected_item
            .as_ref()
            .map(|selection| selection.item.public_id.as_str());
        let selected_action_index =
            (self.ui_pane_focus == PaneFocus::Actions).then_some(self.ui_pane_action_index);
        let render_state = harness_ui::HarnessRenderState {
            screen_indices: &self.ui_screen_indices,
            lists: &self.ui_lists,
            requested_lists: &self.requested_ui_lists,
            list_errors: &self.ui_list_errors,
            selected_work_item_id: selected_item_id,
            selected_action_index,
        };
        harness_ui::render_harness_pane(
            frame,
            area,
            self.selected_ui_app().as_ref(),
            self.active_pane_id.as_deref(),
            &render_state,
        );
    }

    fn render_harness_nav(&self, frame: &mut Frame<'_>, area: Rect) {
        let apps = self.dashboard.ui.apps().cloned().collect::<Vec<_>>();
        let mut items = Vec::new();

        items.push(ListItem::new(Line::from(Span::styled(
            "Apps",
            theme::muted(),
        ))));
        if apps.is_empty() {
            items.extend(no_custom_harness_nav_lines().into_iter().map(ListItem::new));
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
                let badge_style = if item.badge_level.is_some() {
                    ui_notice_level_style(item.badge_level)
                } else {
                    theme::muted()
                };
                items.push(ListItem::new(Line::from(vec![
                    Span::styled(prefix, style),
                    Span::raw(indent),
                    Span::styled(item.label.clone(), style),
                    Span::styled(badge, badge_style),
                ])));
            }
        }

        frame.render_widget(List::new(items).block(block("Navigation")), area);
    }

    fn render_harness_inspector(&self, frame: &mut Frame<'_>, area: Rect) {
        let Some(app) = self.selected_ui_app() else {
            self.render_default_harness_inspector(frame, area);
            return;
        };
        let screen_index = self.selected_screen_index(&app);
        let screen_title = harness_ui::screen_at(&app, screen_index)
            .map(|screen| screen.title.as_str())
            .unwrap_or("No screen");
        let actions = self.current_harness_actions();
        let work_items = self.current_harness_work_items();

        let mut lines = vec![
            kv_line("Screen", screen_title),
            kv_line("Focus", self.harness_focus.label()),
            kv_line("Screens", app.screens.len().to_string()),
            kv_line("Menus", app.menus.len().to_string()),
            kv_line("Items", work_items.len().to_string()),
            kv_line("Actions", actions.len().to_string()),
            Line::from(""),
            Line::from(Span::styled("Selected Item", theme::title())),
        ];

        if let Some(selection) = work_items.get(self.ui_item_index) {
            lines.extend(work_item_selection_lines(
                selection,
                self.harness_focus == HarnessFocus::Items,
                self.ui_item_index,
                work_items.len(),
            ));
        } else {
            lines.push(Line::from(Span::styled(
                "No loaded worklist rows on this screen",
                theme::muted(),
            )));
        }

        lines.extend([
            Line::from(""),
            Line::from(Span::styled("Actions", theme::title())),
        ]);

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

        if let Some(result) = self
            .latest_harness_action_result
            .as_ref()
            .filter(|result| harness_action_result_matches_app(result, &app))
        {
            lines.extend(latest_harness_action_result_lines(result));
        }
        if let Some(failure) = self
            .latest_harness_action_failure
            .as_ref()
            .filter(|failure| harness_action_failure_matches_app(failure, &app))
        {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "Latest Action Failure",
                theme::title(),
            )));
            lines.push(kv_line("Action", failure.action.clone()));
            if let Some(agent_id) = failure.agent_id.as_ref() {
                lines.push(kv_line("Agent", agent_id.clone()));
            }
            if let Some(harness_id) = failure.harness_id.as_ref() {
                lines.push(kv_line("Harness", harness_id.clone()));
            }
            lines.push(kv_line("Error", truncate(&failure.message, 220)));
        }

        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "f focus  j/k move  enter open/select/run  h/l screen",
            theme::muted(),
        )));
        frame.render_widget(panel("Inspector", lines), area);
    }

    fn render_default_harness_inspector(&self, frame: &mut Frame<'_>, area: Rect) {
        frame.render_widget(
            panel(
                "Inspector",
                default_harness_inspector_lines(&self.dashboard),
            ),
            area,
        );
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
            "Form: Tab/↑/↓ fields  type edit  Ctrl+J newline  Space bool  h/l or ←/→ option  Enter submit  Esc cancel".to_string()
        } else if self.active_pane_id.is_some() {
            format!(
                "Pane ({}): f focus  j/k move  g/G jump  Enter item action/run  r refresh  Esc/q close  ? help",
                self.ui_pane_focus.label()
            )
        } else if self.tab == TabKind::Harness {
            "Harness: f focus  j/k move  PgUp/PgDn/Home/End/g/G jump  Enter open/item action/run  r refresh  ? help".to_string()
        } else {
            "Tab/←/→ tabs  j/k move  PgUp/PgDn/Home/End/g/G jump  Enter open/run  r refresh  ? help  q quit".to_string()
        };
        let info = self
            .dashboard
            .last_info
            .as_deref()
            .or(self.dashboard.last_error.as_deref())
            .unwrap_or(&fallback);
        self.render_footer_text(frame, area, info.to_string());
    }

    fn render_footer_text(&self, frame: &mut Frame<'_>, area: Rect, text: String) {
        frame.render_widget(
            Paragraph::new(Line::from(vec![Span::styled(text, theme::muted())]))
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
            kv_line(
                "f",
                "cycle harness focus through navigation and non-empty regions",
            ),
            kv_line("j / k", "move selection"),
            kv_line("PgUp / PgDn", "move selection by a larger stride"),
            kv_line("Home / End / g / G", "jump to first or last selectable row"),
            kv_line("[ / ]", "switch harness app"),
            kv_line("h / l", "switch harness screen"),
            kv_line(
                "Enter",
                "open nav item, queue selected item action, or run action",
            ),
            kv_line("r", "refresh current view"),
            kv_line("Pane f", "switch pane focus between items and actions"),
            kv_line("Pane j / k", "move selected pane item or action"),
            kv_line(
                "Pane Enter",
                "queue pane item action or run selected pane action",
            ),
            kv_line("Pane Esc / q", "close shown pane"),
            kv_line("Esc / n", "cancel confirmation"),
            kv_line("y / Enter", "confirm action"),
            kv_line("Form Tab / ↑ ↓", "move between form fields"),
            kv_line(
                "Form type",
                "edit text, number, integer, and textarea fields",
            ),
            kv_line("Form Ctrl+J", "insert newline in textarea/markdown fields"),
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
        frame.render_widget(active_form_panel(form_session), area);
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

fn active_form_panel(form_session: &TuiFormSession) -> Paragraph<'static> {
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
        "Tab/↑/↓ fields  type edit  Ctrl+J newline  Space bool  h/l or ←/→ options  Enter submit  Esc cancel",
        theme::muted(),
    )));
    panel("Form", lines)
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

fn work_item_selection_lines(
    selection: &harness_ui::HarnessWorkItemSelection,
    focused: bool,
    index: usize,
    total: usize,
) -> Vec<Line<'static>> {
    let marker = if focused { "● " } else { "  " };
    let style = if focused {
        theme::selected()
    } else {
        theme::base()
    };
    let item = &selection.item;
    let position = if total == 0 {
        "0 / 0".to_string()
    } else {
        format!("{} / {}", index.min(total - 1) + 1, total)
    };
    let mut lines = vec![
        Line::from(vec![
            Span::styled(marker, style),
            Span::styled(truncate(&item.title, 34), style),
        ]),
        kv_line("Position", position),
        kv_line("List", selection.list_title.clone()),
        kv_line("Source", selection.list_source.clone()),
        kv_line("Worklist", item.worklist_id.clone()),
        kv_line(
            "State",
            format!(
                "{} / {} / priority {}",
                item.status, item.kind, item.priority
            ),
        ),
        kv_line("Item", item.public_id.clone()),
        kv_line("Created", item.created_at.clone()),
        kv_line("Updated", item.updated_at.clone()),
    ];
    if let Some(parent_id) = item.parent_id.as_ref() {
        lines.push(kv_line("Parent", parent_id.clone()));
    }
    if item.paused {
        lines.push(Line::from(vec![
            Span::styled(format!("{:<14}", "Paused"), theme::muted()),
            Span::styled("yes", theme::warning()),
        ]));
    }
    if let Some(reason) = item.pause_reason.as_ref() {
        lines.push(kv_line("Pause reason", truncate(reason, 60)));
    }
    if let Some(agent_id) = item.claim_agent_id.as_ref() {
        lines.push(kv_line("Claimed by", agent_id.clone()));
    }
    if let Some(claimed_at) = item.claimed_at.as_ref() {
        lines.push(kv_line("Claimed at", claimed_at.clone()));
    }
    if let Some(completed_at) = item.completed_at.as_ref() {
        lines.push(kv_line("Completed", completed_at.clone()));
    }
    if let Some(action) = item.action.as_ref() {
        lines.push(kv_line("Action", action.name.clone()));
        lines.push(Line::from(Span::styled(
            "Enter queues this work-item action for confirmation",
            theme::muted(),
        )));
    }
    if let Some(reason) = item.failure_reason.as_ref() {
        lines.push(Line::from(Span::styled(
            format!("Failure: {}", truncate(reason, 38)),
            theme::danger(),
        )));
    }
    if let Some(metadata) = item.metadata.as_ref() {
        lines.push(kv_line("Metadata", json_preview(metadata, 90)));
    }
    lines
}

fn latest_harness_action_result_lines(result: &HarnessActionRunResult) -> Vec<Line<'static>> {
    let mut lines = vec![
        Line::from(""),
        Line::from(Span::styled("Latest Result", theme::title())),
        kv_line("Action", result.action.clone()),
        kv_line("Agent", result.agent_id.clone()),
    ];
    if let Some(harness_id) = result.harness_id.as_ref() {
        lines.push(kv_line("Harness", harness_id.clone()));
    }
    if result.result.is_null() {
        lines.push(Line::from(Span::styled(
            "Action completed without a result payload.",
            theme::muted(),
        )));
    } else {
        lines.push(kv_line("Result", json_preview(&result.result, 220)));
    }
    lines
}

fn no_custom_harness_nav_lines() -> Vec<Line<'static>> {
    vec![
        Line::from(vec![
            Span::styled("● ", theme::selected()),
            Span::styled("Default Console", theme::selected()),
            Span::styled("  [runtime]", theme::muted()),
        ]),
        Line::from(Span::styled(
            "  No custom harness UI is declared.",
            theme::muted(),
        )),
        Line::from(Span::styled(
            "  Overview, Tasks, and Events remain available.",
            theme::muted(),
        )),
    ]
}

fn pending_action_from_work_item(
    app: &turin_ui_core::UiAppRecord,
    selection: &harness_ui::HarnessWorkItemSelection,
) -> Option<PendingHarnessAction> {
    let action = selection.item.action.as_ref()?;
    Some(PendingHarnessAction {
        app_id: app.id.clone(),
        label: format!("Work item: {}", selection.item.title),
        action: action.name.clone(),
        agent_id: app.source.agent_id.clone(),
        harness_id: app.source.harness_id.clone(),
        params: action.params.clone().unwrap_or(Value::Null),
    })
}

fn reconcile_work_item_selection(
    index: &mut usize,
    selected_key: &mut Option<String>,
    items: &[harness_ui::HarnessWorkItemSelection],
) {
    if items.is_empty() {
        *index = 0;
        *selected_key = None;
        return;
    }

    if let Some(key) = selected_key.as_deref()
        && let Some(position) = items
            .iter()
            .position(|selection| work_item_selection_key(selection) == key)
    {
        *index = position;
    } else {
        *index = (*index).min(items.len() - 1);
    }

    *selected_key = items.get(*index).map(work_item_selection_key);
}

fn work_item_selection_key(selection: &harness_ui::HarnessWorkItemSelection) -> String {
    format!(
        "{}:{}",
        selection.list_source,
        work_item_key(&selection.item)
    )
}

fn pane_actions(
    app: &turin_ui_core::UiAppRecord,
    pane_id: Option<&str>,
) -> Vec<harness_ui::HarnessAction> {
    pane_id
        .and_then(|pane_id| app.panes.get(pane_id))
        .map(|pane| harness_ui::collect_actions(app, &pane.nodes))
        .unwrap_or_default()
}

fn pane_work_items(
    app: &turin_ui_core::UiAppRecord,
    pane_id: Option<&str>,
    lists: &BTreeMap<String, WorkItemList>,
) -> Vec<harness_ui::HarnessWorkItemSelection> {
    pane_id
        .and_then(|pane_id| app.panes.get(pane_id))
        .map(|pane| harness_ui::collect_work_item_selections(&pane.nodes, lists))
        .unwrap_or_default()
}

fn visible_harness_list_requests(
    app: &turin_ui_core::UiAppRecord,
    screen_index: usize,
    active_pane_id: Option<&str>,
) -> Vec<UiListRequest> {
    let mut requests = harness_ui::screen_at(app, screen_index)
        .map(|screen| harness_ui::collect_list_requests(&screen.nodes))
        .unwrap_or_default();
    if let Some(pane_id) = active_pane_id
        && let Some(pane) = app.panes.get(pane_id)
    {
        requests.extend(harness_ui::collect_list_requests(&pane.nodes));
    }
    requests
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
    let style = ui_notice_level_style(notice.level);
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

fn ui_notice_level_style(level: Option<turin_daemon_protocol::UiNoticeLevel>) -> Style {
    match level {
        Some(turin_daemon_protocol::UiNoticeLevel::Success) => theme::success(),
        Some(turin_daemon_protocol::UiNoticeLevel::Warning) => theme::warning(),
        Some(turin_daemon_protocol::UiNoticeLevel::Error) => theme::danger(),
        Some(turin_daemon_protocol::UiNoticeLevel::Info) | None => theme::base(),
    }
}

fn default_harness_inspector_lines(dashboard: &DashboardState) -> Vec<Line<'static>> {
    let summary = DefaultOperatorConsoleSummary::from_dashboard(dashboard);
    vec![
        Line::from(Span::styled("Default Console", theme::title())),
        Line::from(""),
        kv_line("Connection", summary.connection),
        kv_line("Freshness", summary.freshness),
        kv_line("Target", truncate(&summary.target, 42)),
        Line::from(""),
        Line::from(Span::styled("Runtime", theme::title())),
        kv_line("Agents", summary.agents.to_string()),
        kv_line("Harnesses", summary.harnesses.to_string()),
        Line::from(""),
        Line::from(Span::styled("Work", theme::title())),
        kv_line("Live Sessions", summary.live_sessions.to_string()),
        kv_line("Stored", summary.stored_sessions.to_string()),
        kv_line("Tasks", summary.tasks.to_string()),
        Line::from(""),
        Line::from(Span::styled("UI Signals", theme::title())),
        kv_line("Apps", dashboard.ui.apps().count().to_string()),
        kv_line("Notices", summary.ui_notices.to_string()),
        kv_line("Requests", summary.ui_requests.to_string()),
        Line::from(""),
        Line::from(Span::styled(
            "Use Overview, Tasks, and Events until a harness declares ui.app(...).",
            theme::muted(),
        )),
    ]
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
    if harness_ui::is_password_field(field) {
        return "••••••••".to_string();
    }
    if harness_ui::is_multiline_field(field) {
        let line_count = value.split('\n').count();
        let preview = value.replace('\n', " ↵ ");
        return format!("{line_count} lines · {}", truncate(&preview, 42));
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

fn page_index(current: usize, len: usize, delta: isize) -> usize {
    if len == 0 {
        return 0;
    }
    let last = len.saturating_sub(1) as isize;
    (current as isize + delta).clamp(0, last) as usize
}

fn edge_index(len: usize, edge: SelectionEdge) -> usize {
    match edge {
        SelectionEdge::Start => 0,
        SelectionEdge::End => len.saturating_sub(1),
    }
}

fn selection_edge_for_key(code: KeyCode) -> Option<SelectionEdge> {
    match code {
        KeyCode::Home | KeyCode::Char('g') => Some(SelectionEdge::Start),
        KeyCode::End | KeyCode::Char('G') => Some(SelectionEdge::End),
        _ => None,
    }
}

fn connection_kind_label(kind: turin_client::ConnectionKind) -> &'static str {
    match kind {
        turin_client::ConnectionKind::Local => "local",
        turin_client::ConnectionKind::Remote => "remote",
    }
}

fn freshness_label(freshness: DashboardFreshness) -> &'static str {
    match freshness {
        DashboardFreshness::Fresh => "fresh",
        DashboardFreshness::Quiet => "quiet",
        DashboardFreshness::Stale => "stale",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::buffer::Buffer;
    use serde_json::json;
    use turin_client::ConnectionKind;
    use turin_daemon_protocol::{
        ScheduleActionParams, UiActionNode, UiFormField, UiIntentSource, UiListNode, UiNode,
        UiPaneIntent, UiScreenIntent, WorkItemDetail,
    };
    use turin_ui_core::UiAppRecord;

    #[test]
    fn harness_focus_cycle_skips_empty_regions() {
        assert_eq!(
            HarnessFocus::Navigation.next_available(false, false),
            HarnessFocus::Navigation
        );
        assert_eq!(
            HarnessFocus::Navigation.next_available(false, true),
            HarnessFocus::Actions
        );
        assert_eq!(
            HarnessFocus::Navigation.next_available(true, false),
            HarnessFocus::Items
        );
        assert_eq!(
            HarnessFocus::Items.next_available(true, false),
            HarnessFocus::Navigation
        );
        assert_eq!(
            HarnessFocus::Items.next_available(true, true),
            HarnessFocus::Actions
        );
        assert_eq!(
            HarnessFocus::Actions.next_available(true, true),
            HarnessFocus::Navigation
        );
    }

    #[test]
    fn harness_focus_availability_keeps_navigation_as_fallback() {
        assert!(HarnessFocus::Navigation.is_available(false, false));
        assert!(!HarnessFocus::Items.is_available(false, true));
        assert!(HarnessFocus::Items.is_available(true, false));
        assert!(!HarnessFocus::Actions.is_available(true, false));
        assert!(HarnessFocus::Actions.is_available(false, true));
    }

    #[test]
    fn page_and_edge_navigation_helpers_are_bounded() {
        assert_eq!(offset_index(0, 5, -1), 4);
        assert_eq!(page_index(0, 5, SELECTION_PAGE_SIZE), 4);
        assert_eq!(page_index(4, 5, -SELECTION_PAGE_SIZE), 0);
        assert_eq!(page_index(2, 5, SELECTION_PAGE_SIZE), 4);
        assert_eq!(page_index(2, 5, -SELECTION_PAGE_SIZE), 0);
        assert_eq!(page_index(0, 0, SELECTION_PAGE_SIZE), 0);
        assert_eq!(edge_index(5, SelectionEdge::Start), 0);
        assert_eq!(edge_index(5, SelectionEdge::End), 4);
        assert_eq!(edge_index(0, SelectionEdge::End), 0);
    }

    #[test]
    fn edge_navigation_keys_include_terminal_friendly_aliases() {
        assert_eq!(
            selection_edge_for_key(KeyCode::Home),
            Some(SelectionEdge::Start)
        );
        assert_eq!(
            selection_edge_for_key(KeyCode::Char('g')),
            Some(SelectionEdge::Start)
        );
        assert_eq!(
            selection_edge_for_key(KeyCode::End),
            Some(SelectionEdge::End)
        );
        assert_eq!(
            selection_edge_for_key(KeyCode::Char('G')),
            Some(SelectionEdge::End)
        );
        assert_eq!(selection_edge_for_key(KeyCode::Char('j')), None);
    }

    #[test]
    fn no_custom_harness_nav_lines_explain_default_console() {
        let text = line_text(&no_custom_harness_nav_lines());

        assert!(text.contains("Default Console"));
        assert!(text.contains("[runtime]"));
        assert!(text.contains("No custom harness UI is declared."));
        assert!(text.contains("Overview, Tasks, and Events remain available."));
    }

    #[test]
    fn terminal_golden_default_harness_inspector_stays_stable() {
        let text = rendered_default_inspector_text(&empty_dashboard());

        assert_eq!(
            terminal_compact_content_lines(&text),
            vec![
                "Default Console",
                "Connection local",
                "Freshness stale",
                "Target local-test",
                "Runtime",
                "Agents 0",
                "Harnesses 0",
                "Work",
                "Live Sessions 0",
                "Stored 0",
                "Tasks 0",
                "UI Signals",
                "Apps 0",
                "Notices 0",
                "Requests 0",
                "Use Overview, Tasks, and Events until a harness declares ui.app(...).",
            ]
        );
    }

    #[test]
    fn work_item_action_becomes_pending_harness_action() {
        let app = test_app();
        let selection = harness_ui::HarnessWorkItemSelection {
            list_title: "Approvals".to_string(),
            list_source: "worklists.release".to_string(),
            item: test_work_item(Some(ScheduleActionParams {
                name: "release.approve".to_string(),
                params: Some(json!({ "item": "REL-1" })),
            })),
        };

        let pending = pending_action_from_work_item(&app, &selection).expect("pending action");

        assert_eq!(pending.app_id, "release");
        assert_eq!(pending.label, "Work item: Approve release");
        assert_eq!(pending.action, "release.approve");
        assert_eq!(pending.harness_id.as_deref(), Some("release-harness"));
        assert_eq!(pending.agent_id.as_deref(), Some("release-agent"));
        assert_eq!(pending.params, json!({ "item": "REL-1" }));
    }

    #[test]
    fn work_item_without_action_is_not_runnable() {
        let app = test_app();
        let selection = harness_ui::HarnessWorkItemSelection {
            list_title: "Approvals".to_string(),
            list_source: "worklists.release".to_string(),
            item: test_work_item(None),
        };

        assert!(pending_action_from_work_item(&app, &selection).is_none());
    }

    #[test]
    fn reconcile_work_item_selection_preserves_identity_after_reorder() {
        let mut release_one = test_work_item(None);
        release_one.public_id = "REL-1".to_string();
        release_one.title = "First release gate".to_string();
        let mut release_two = test_work_item(None);
        release_two.id = 2;
        release_two.public_id = "REL-2".to_string();
        release_two.title = "Second release gate".to_string();
        let selected = work_item_selection("worklists.release", release_two.clone());
        let reordered = vec![
            work_item_selection("worklists.release", release_two),
            work_item_selection("worklists.release", release_one),
        ];
        let mut index = 1;
        let mut selected_key = Some(work_item_selection_key(&selected));

        reconcile_work_item_selection(&mut index, &mut selected_key, &reordered);

        assert_eq!(index, 0);
        assert_eq!(selected_key.as_deref(), Some("worklists.release:REL-2"));
    }

    #[test]
    fn reconcile_work_item_selection_falls_back_when_identity_disappears() {
        let item = work_item_selection("worklists.release", test_work_item(None));
        let mut index = 9;
        let mut selected_key = Some("worklists.release:missing".to_string());

        reconcile_work_item_selection(&mut index, &mut selected_key, &[item]);

        assert_eq!(index, 0);
        assert_eq!(selected_key.as_deref(), Some("worklists.release:REL-1"));
    }

    #[test]
    fn reconcile_work_item_selection_clears_empty_selection() {
        let mut index = 3;
        let mut selected_key = Some("worklists.release:REL-1".to_string());

        reconcile_work_item_selection(&mut index, &mut selected_key, &[]);

        assert_eq!(index, 0);
        assert_eq!(selected_key, None);
    }

    #[test]
    fn harness_action_result_matching_filters_other_apps() {
        let app = test_app();
        let matching = HarnessActionRunResult {
            action: "release.seed".to_string(),
            agent_id: "release-agent".to_string(),
            harness_id: Some("release-harness".to_string()),
            result: json!({ "status": "ok" }),
            ui_intents: Vec::new(),
        };
        let other_harness = HarnessActionRunResult {
            harness_id: Some("qa-harness".to_string()),
            ..matching.clone()
        };
        let other_agent = HarnessActionRunResult {
            agent_id: "qa-agent".to_string(),
            ..matching.clone()
        };

        assert!(harness_action_result_matches_app(&matching, &app));
        assert!(!harness_action_result_matches_app(&other_harness, &app));
        assert!(!harness_action_result_matches_app(&other_agent, &app));
    }

    #[test]
    fn harness_action_result_without_harness_matches_selected_agent() {
        let app = test_app();
        let result = HarnessActionRunResult {
            action: "release.seed".to_string(),
            agent_id: "release-agent".to_string(),
            harness_id: None,
            result: json!({ "status": "ok" }),
            ui_intents: Vec::new(),
        };

        assert!(harness_action_result_matches_app(&result, &app));
    }

    #[test]
    fn latest_harness_action_result_lines_explain_null_payload() {
        let result = HarnessActionRunResult {
            action: "release.seed".to_string(),
            agent_id: "release-agent".to_string(),
            harness_id: Some("release-harness".to_string()),
            result: Value::Null,
            ui_intents: Vec::new(),
        };

        let text = line_text(&latest_harness_action_result_lines(&result));

        assert!(text.contains("Latest Result"));
        assert!(text.contains("release.seed"));
        assert!(text.contains("release-agent"));
        assert!(text.contains("release-harness"));
        assert!(text.contains("Action completed without a result payload."));
    }

    #[test]
    fn latest_harness_action_result_lines_preview_non_null_payload() {
        let result = HarnessActionRunResult {
            action: "release.seed".to_string(),
            agent_id: "release-agent".to_string(),
            harness_id: None,
            result: json!({ "status": "ok" }),
            ui_intents: Vec::new(),
        };

        let text = line_text(&latest_harness_action_result_lines(&result));

        assert!(text.contains("Result"));
        assert!(text.contains("status"));
        assert!(text.contains("ok"));
        assert!(!text.contains("without a result payload"));
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

    #[test]
    fn work_item_selection_lines_include_position_and_action_hint() {
        let mut item = test_work_item(Some(ScheduleActionParams {
            name: "release.approve".to_string(),
            params: Some(json!({ "item": "REL-1" })),
        }));
        item.parent_id = Some("REL-0".to_string());
        item.paused = true;
        item.pause_reason = Some("Waiting for release captain signoff".to_string());
        item.claim_agent_id = Some("release-bot".to_string());
        item.claimed_at = Some("2026-06-18T01:00:00Z".to_string());
        item.completed_at = Some("2026-06-18T02:00:00Z".to_string());
        item.failure_reason = Some("Previous gate check failed".to_string());
        item.metadata = Some(json!({ "release": "2026.06" }));
        let selection = harness_ui::HarnessWorkItemSelection {
            list_title: "Approvals".to_string(),
            list_source: "worklists.release".to_string(),
            item,
        };

        let text = line_text(&work_item_selection_lines(&selection, true, 1, 3));

        assert!(text.contains("Position"));
        assert!(text.contains("2 / 3"));
        assert!(text.contains("Worklist"));
        assert!(text.contains("release"));
        assert!(text.contains("Created"));
        assert!(text.contains("2026-06-18T00:00:00Z"));
        assert!(text.contains("Updated"));
        assert!(text.contains("Parent"));
        assert!(text.contains("REL-0"));
        assert!(text.contains("Paused"));
        assert!(text.contains("yes"));
        assert!(text.contains("Pause reason"));
        assert!(text.contains("Waiting for release captain signoff"));
        assert!(text.contains("Claimed by"));
        assert!(text.contains("release-bot"));
        assert!(text.contains("Claimed at"));
        assert!(text.contains("2026-06-18T01:00:00Z"));
        assert!(text.contains("Completed"));
        assert!(text.contains("2026-06-18T02:00:00Z"));
        assert!(text.contains("Action"));
        assert!(text.contains("release.approve"));
        assert!(text.contains("Enter queues this work-item action for confirmation"));
        assert!(text.contains("Failure: Previous gate check failed"));
        assert!(text.contains("Metadata"));
        assert!(text.contains("2026.06"));
    }

    #[test]
    fn form_field_value_preview_summarizes_multiline_text() {
        let field = UiFormField {
            name: "notes".to_string(),
            label: "Notes".to_string(),
            kind: Some("textarea".to_string()),
            default: None,
            required: None,
            options: Vec::new(),
        };

        let preview = form_field_value_preview(&field, "first line\nsecond line");

        assert_eq!(preview, "2 lines · first line ↵ second line");
        assert!(harness_ui::is_multiline_field(&field));
    }

    #[test]
    fn form_field_value_preview_masks_password_text() {
        let field = UiFormField {
            name: "token".to_string(),
            label: "Token".to_string(),
            kind: Some("secret".to_string()),
            default: None,
            required: None,
            options: Vec::new(),
        };

        let preview = form_field_value_preview(&field, "super-secret-token");

        assert_eq!(preview, "••••••••");
        assert!(harness_ui::is_password_field(&field));
        assert!(!preview.contains("super-secret-token"));
    }

    #[test]
    fn terminal_golden_active_form_modal_stays_stable() {
        let form = UiFormNode {
            id: Some("seed-demo-form".to_string()),
            title: "Seed Demo Work".to_string(),
            action: "release.seed_demo_work".to_string(),
            fields: vec![
                UiFormField {
                    name: "title".to_string(),
                    label: "Title".to_string(),
                    kind: Some("text".to_string()),
                    default: Some(json!("Release 2026.06")),
                    required: Some(true),
                    options: Vec::new(),
                },
                UiFormField {
                    name: "confirmed".to_string(),
                    label: "Confirmed".to_string(),
                    kind: Some("boolean".to_string()),
                    default: Some(json!(true)),
                    required: None,
                    options: Vec::new(),
                },
                UiFormField {
                    name: "lane".to_string(),
                    label: "Lane".to_string(),
                    kind: Some("select".to_string()),
                    default: Some(json!("qa")),
                    required: None,
                    options: vec![json!("dev"), json!("qa")],
                },
                UiFormField {
                    name: "notes".to_string(),
                    label: "Notes".to_string(),
                    kind: Some("markdown".to_string()),
                    default: None,
                    required: None,
                    options: Vec::new(),
                },
            ],
            params: Value::Null,
        };
        let session = TuiFormSession {
            app_id: "release".to_string(),
            form,
            agent_id: Some("release-agent".to_string()),
            harness_id: Some("release-harness".to_string()),
            values: BTreeMap::from([
                ("title".to_string(), "Release 2026.06".to_string()),
                ("confirmed".to_string(), "true".to_string()),
                ("lane".to_string(), "qa".to_string()),
                ("notes".to_string(), "first line\nsecond line".to_string()),
            ]),
            field_index: 3,
            error: Some("Form field 'Count' must be a valid integer".to_string()),
        };

        assert_eq!(
            terminal_compact_content_lines(&rendered_form_text(&session)),
            vec![
                "Seed Demo Work (release.seed_demo_work)",
                "Title text required Release 2026.06",
                "Confirmed boolean [x]",
                "Lane select 2 options < qa >",
                "● Notes markdown 2 lines · first line ↵ second line",
                "Form field 'Count' must be a valid integer",
                "Tab/↑/↓ fields type edit Ctrl+J newline Space bool h/l or ←/→ options Enter submit Esc",
                "cancel",
            ]
        );
    }

    #[test]
    fn form_session_field_mutations_clear_stale_errors() {
        let form = UiFormNode {
            id: Some("seed".to_string()),
            title: "Seed".to_string(),
            action: "release.seed".to_string(),
            fields: vec![
                UiFormField {
                    name: "confirmed".to_string(),
                    label: "Confirmed".to_string(),
                    kind: Some("boolean".to_string()),
                    default: Some(json!(true)),
                    required: None,
                    options: Vec::new(),
                },
                UiFormField {
                    name: "lane".to_string(),
                    label: "Lane".to_string(),
                    kind: Some("select".to_string()),
                    default: None,
                    required: None,
                    options: vec![json!("dev"), json!("qa")],
                },
                UiFormField {
                    name: "title".to_string(),
                    label: "Title".to_string(),
                    kind: Some("text".to_string()),
                    default: Some(json!("Release")),
                    required: None,
                    options: Vec::new(),
                },
            ],
            params: Value::Null,
        };
        let mut session = TuiFormSession {
            app_id: "release".to_string(),
            form,
            agent_id: None,
            harness_id: None,
            values: BTreeMap::from([
                ("confirmed".to_string(), "true".to_string()),
                ("lane".to_string(), "dev".to_string()),
                ("title".to_string(), "Release".to_string()),
            ]),
            field_index: 1,
            error: Some("stale validation".to_string()),
        };

        assert!(session.cycle_selected_option(1));
        assert_eq!(session.values["lane"], "qa");
        assert_eq!(session.error, None);

        session.error = Some("stale validation".to_string());
        assert!(session.clear_selected_field());
        assert_eq!(session.values["lane"], "dev");
        assert_eq!(session.error, None);

        session.field_index = 0;
        session.error = Some("stale validation".to_string());
        assert!(session.toggle_selected_bool());
        assert_eq!(session.values["confirmed"], "false");
        assert_eq!(session.error, None);

        session.error = Some("stale validation".to_string());
        assert!(session.set_selected_bool(true));
        assert_eq!(session.values["confirmed"], "true");
        assert_eq!(session.error, None);

        session.error = Some("stale validation".to_string());
        assert!(session.clear_selected_field());
        assert_eq!(session.values["confirmed"], "false");
        assert_eq!(session.error, None);

        session.field_index = 2;
        session.error = Some("stale validation".to_string());
        assert!(session.clear_selected_field());
        assert_eq!(session.values["title"], "");
        assert_eq!(session.error, None);
    }

    #[test]
    fn visible_list_requests_include_active_pane_nodes() {
        let mut app = test_app();
        app.screens.insert(
            "home".to_string(),
            UiScreenIntent {
                app_id: app.id.clone(),
                id: "home".to_string(),
                title: "Home".to_string(),
                presentation: None,
                nodes: vec![UiNode::List(UiListNode {
                    id: Some("screen-list".to_string()),
                    title: "Screen List".to_string(),
                    source: "worklists.screen".to_string(),
                    filter: Default::default(),
                    fields: Vec::new(),
                    sort: Vec::new(),
                    limit: Some(3),
                    intent: Some("screen".to_string()),
                    render_as: Some("table".to_string()),
                })],
            },
        );
        app.panes.insert(
            "notes".to_string(),
            UiPaneIntent {
                app_id: app.id.clone(),
                id: "notes".to_string(),
                title: "Notes".to_string(),
                presentation: Some("sheet".to_string()),
                nodes: vec![UiNode::List(UiListNode {
                    id: Some("pane-list".to_string()),
                    title: "Pane List".to_string(),
                    source: "worklists.pane".to_string(),
                    filter: Default::default(),
                    fields: Vec::new(),
                    sort: Vec::new(),
                    limit: Some(5),
                    intent: Some("pane".to_string()),
                    render_as: Some("table".to_string()),
                })],
            },
        );

        let screen_only = visible_harness_list_requests(&app, 0, None);
        let with_pane = visible_harness_list_requests(&app, 0, Some("notes"));

        assert_eq!(screen_only.len(), 1);
        assert_eq!(screen_only[0].source, "worklists.screen");
        assert_eq!(screen_only[0].limit, Some(3));
        assert_eq!(with_pane.len(), 2);
        assert_eq!(with_pane[0].source, "worklists.screen");
        assert_eq!(with_pane[0].limit, Some(3));
        assert_eq!(with_pane[1].source, "worklists.pane");
        assert_eq!(with_pane[1].limit, Some(5));
    }

    #[test]
    fn pane_actions_collect_runnable_nodes() {
        let mut app = test_app();
        app.panes.insert(
            "actions".to_string(),
            UiPaneIntent {
                app_id: app.id.clone(),
                id: "actions".to_string(),
                title: "Actions".to_string(),
                presentation: Some("sheet".to_string()),
                nodes: vec![UiNode::Action(UiActionNode {
                    id: Some("approve-now".to_string()),
                    label: "Approve now".to_string(),
                    action: "release.approve".to_string(),
                    params: json!({ "force": true }),
                    confirm: true,
                })],
            },
        );

        let actions = pane_actions(&app, Some("actions"));

        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].label, "Approve now");
        assert_eq!(actions[0].action, "release.approve");
        assert!(actions[0].confirm);
        assert_eq!(actions[0].params, json!({ "force": true }));
        assert!(pane_actions(&app, Some("missing")).is_empty());
        assert!(pane_actions(&app, None).is_empty());
    }

    #[test]
    fn pane_work_items_collect_loaded_rows() {
        let mut app = test_app();
        let list = UiListNode {
            id: Some("pane-list".to_string()),
            title: "Pane List".to_string(),
            source: "worklists.pane".to_string(),
            filter: Default::default(),
            fields: Vec::new(),
            sort: Vec::new(),
            limit: Some(5),
            intent: Some("pane".to_string()),
            render_as: Some("table".to_string()),
        };
        app.panes.insert(
            "notes".to_string(),
            UiPaneIntent {
                app_id: app.id.clone(),
                id: "notes".to_string(),
                title: "Notes".to_string(),
                presentation: Some("sheet".to_string()),
                nodes: vec![UiNode::List(list.clone())],
            },
        );
        let request = UiListRequest {
            source: list.source,
            filter: list.filter,
            limit: list.limit,
        };
        let lists = BTreeMap::from([(
            request.cache_key(),
            WorkItemList {
                worklist_id: "pane".to_string(),
                items: vec![test_work_item(Some(ScheduleActionParams {
                    name: "release.approve".to_string(),
                    params: Some(json!({ "item": "REL-1" })),
                }))],
            },
        )]);

        let items = pane_work_items(&app, Some("notes"), &lists);

        assert_eq!(items.len(), 1);
        assert_eq!(items[0].list_title, "Pane List");
        assert_eq!(items[0].list_source, "worklists.pane");
        assert_eq!(items[0].item.public_id, "REL-1");
        assert!(pane_work_items(&app, Some("missing"), &lists).is_empty());
        assert!(pane_work_items(&app, None, &lists).is_empty());
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
            screens: BTreeMap::new(),
            panes: BTreeMap::new(),
            menus: Vec::new(),
            opens_with: None,
            badges: BTreeMap::new(),
        }
    }

    fn test_work_item(action: Option<ScheduleActionParams>) -> WorkItemDetail {
        WorkItemDetail {
            id: 1,
            public_id: "REL-1".to_string(),
            worklist_id: "release".to_string(),
            parent_id: None,
            title: "Approve release".to_string(),
            kind: "approval".to_string(),
            prompt: Some("Check release gates".to_string()),
            content: None,
            tools: None,
            conflict_policy: None,
            action,
            status: "pending".to_string(),
            paused: false,
            pause_reason: None,
            pause_until_unix_ms: None,
            priority: 10,
            after: None,
            metadata: None,
            claim_agent_id: None,
            claim_session_id: None,
            claim_execution_id: None,
            claim_heartbeat_unix_ms: None,
            claimed_at: None,
            completed_at: None,
            failure_reason: None,
            created_at: "2026-06-18T00:00:00Z".to_string(),
            updated_at: "2026-06-18T00:00:00Z".to_string(),
        }
    }

    fn work_item_selection(
        list_source: &str,
        item: WorkItemDetail,
    ) -> harness_ui::HarnessWorkItemSelection {
        harness_ui::HarnessWorkItemSelection {
            list_title: "Approvals".to_string(),
            list_source: list_source.to_string(),
            item,
        }
    }

    fn line_text(lines: &[Line<'static>]) -> String {
        lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn rendered_form_text(session: &TuiFormSession) -> String {
        let backend = TestBackend::new(96, 18);
        let mut terminal = Terminal::new(backend).expect("test terminal");
        terminal
            .draw(|frame| {
                frame.render_widget(active_form_panel(session), frame.area());
            })
            .expect("draw form panel");
        buffer_text(terminal.backend().buffer())
    }

    fn rendered_default_inspector_text(dashboard: &DashboardState) -> String {
        let backend = TestBackend::new(96, 26);
        let mut terminal = Terminal::new(backend).expect("test terminal");
        terminal
            .draw(|frame| {
                frame.render_widget(
                    panel("Inspector", default_harness_inspector_lines(dashboard)),
                    frame.area(),
                );
            })
            .expect("draw default inspector panel");
        buffer_text(terminal.backend().buffer())
    }

    fn terminal_compact_content_lines(text: &str) -> Vec<String> {
        text.lines()
            .filter_map(|line| {
                if line.contains('─') {
                    return None;
                }
                let line = line.trim_end();
                let line = line.strip_prefix('│').unwrap_or(line);
                let line = line.strip_suffix('│').unwrap_or(line);
                let line = line.trim_end();
                if line.trim().is_empty() {
                    None
                } else {
                    Some(line.split_whitespace().collect::<Vec<_>>().join(" "))
                }
            })
            .collect()
    }

    fn buffer_text(buffer: &Buffer) -> String {
        let area = *buffer.area();
        let mut out = String::new();
        for y in area.top()..area.bottom() {
            for x in area.left()..area.right() {
                if let Some(cell) = buffer.cell((x, y)) {
                    out.push_str(cell.symbol());
                }
            }
            out.push('\n');
        }
        out
    }

    fn empty_dashboard() -> DashboardState {
        DashboardState {
            connection_kind: ConnectionKind::Local,
            connection_target: "local-test".to_string(),
            health: None,
            status: None,
            live_sessions: Vec::new(),
            sessions: Vec::new(),
            tasks: Vec::new(),
            session_details: Default::default(),
            ui: Default::default(),
            recent_events: Vec::new(),
            recent_notices: Vec::new(),
            last_snapshot_unix_ms: 0,
            last_event_unix_ms: None,
            last_notice_unix_ms: None,
            total_event_count: 0,
            refresh_success_count: 0,
            refresh_failure_count: 0,
            last_refresh_duration_ms: None,
            last_refresh_ok: None,
            last_error: None,
            last_info: None,
        }
    }
}
