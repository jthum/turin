use std::collections::BTreeMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use turin_daemon_protocol::{
    EventEnvelope, UiAppIntent, UiBadgeIntent, UiFocusIntent, UiIntent, UiIntentMessage,
    UiMenuIntent, UiNoticeIntent, UiOpenIntent, UiOpensWithIntent, UiPaneIntent, UiRefreshIntent,
    UiScreenIntent, UiShowIntent,
};

pub const DEFAULT_MAX_UI_NOTICES: usize = 32;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiIntentState {
    #[serde(default)]
    apps: BTreeMap<String, UiAppState>,
    #[serde(default = "default_max_notices")]
    max_notices: usize,
    #[serde(default)]
    recent_notices: Vec<UiNoticeIntent>,
    #[serde(default)]
    pending_refreshes: Vec<UiRefreshIntent>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiAppState {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub definition: Option<UiAppIntent>,
    #[serde(default)]
    pub screens: BTreeMap<String, UiScreenIntent>,
    #[serde(default)]
    pub panes: BTreeMap<String, UiPaneIntent>,
    #[serde(default)]
    pub menus: Vec<UiMenuIntent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub opens_with: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_target: Option<String>,
    #[serde(default)]
    pub visible_targets: BTreeMap<String, UiShowIntent>,
    #[serde(default)]
    pub badges: BTreeMap<String, UiBadgeIntent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub focused: Option<UiFocusIntent>,
}

impl Default for UiIntentState {
    fn default() -> Self {
        Self::new()
    }
}

impl UiIntentState {
    pub fn new() -> Self {
        Self {
            apps: BTreeMap::new(),
            max_notices: DEFAULT_MAX_UI_NOTICES,
            recent_notices: Vec::new(),
            pending_refreshes: Vec::new(),
        }
    }

    pub fn with_max_notices(max_notices: usize) -> Self {
        Self {
            max_notices,
            ..Self::new()
        }
    }

    pub fn from_messages(messages: impl IntoIterator<Item = UiIntentMessage>) -> Self {
        let mut state = Self::new();
        state.apply_messages(messages);
        state
    }

    pub fn apps(&self) -> impl Iterator<Item = &UiAppState> {
        self.apps.values()
    }

    pub fn app(&self, app_id: &str) -> Option<&UiAppState> {
        self.apps.get(app_id)
    }

    pub fn recent_notices(&self) -> &[UiNoticeIntent] {
        &self.recent_notices
    }

    pub fn pending_refreshes(&self) -> &[UiRefreshIntent] {
        &self.pending_refreshes
    }

    pub fn take_pending_refreshes(&mut self) -> Vec<UiRefreshIntent> {
        std::mem::take(&mut self.pending_refreshes)
    }

    pub fn apply_event(&mut self, event: &EventEnvelope) -> Result<bool> {
        let Some(message) = UiIntentMessage::from_event(event)? else {
            return Ok(false);
        };
        self.apply_message(message);
        Ok(true)
    }

    pub fn apply_messages(&mut self, messages: impl IntoIterator<Item = UiIntentMessage>) {
        for message in messages {
            self.apply_message(message);
        }
    }

    pub fn apply_message(&mut self, message: UiIntentMessage) {
        match message.intent {
            UiIntent::App(intent) => {
                let app = self.ensure_app(intent.id.clone());
                app.definition = Some(intent);
            }
            UiIntent::Screen(intent) => {
                self.ensure_app(intent.app_id.clone())
                    .screens
                    .insert(intent.id.clone(), intent);
            }
            UiIntent::Menu(intent) => {
                self.ensure_app(intent.app_id.clone()).upsert_menu(intent);
            }
            UiIntent::OpensWith(intent) => self.apply_opens_with(intent),
            UiIntent::Pane(intent) => {
                self.ensure_app(intent.app_id.clone())
                    .panes
                    .insert(intent.id.clone(), intent);
            }
            UiIntent::Open(intent) => self.apply_open(intent),
            UiIntent::Show(intent) => {
                self.ensure_app(intent.app_id.clone())
                    .visible_targets
                    .insert(intent.target.clone(), intent);
            }
            UiIntent::Notify(intent) => self.push_notice(intent),
            UiIntent::Badge(intent) => {
                self.ensure_app(intent.app_id.clone())
                    .badges
                    .insert(intent.target.clone(), intent);
            }
            UiIntent::Focus(intent) => {
                let app_id = intent.app_id.clone();
                self.ensure_app(app_id).focused = Some(intent);
            }
            UiIntent::Refresh(intent) => self.pending_refreshes.push(intent),
        }
    }

    fn apply_opens_with(&mut self, intent: UiOpensWithIntent) {
        let app = self.ensure_app(intent.app_id);
        if app.active_target.is_none() {
            app.active_target = Some(intent.screen_id.clone());
        }
        app.opens_with = Some(intent.screen_id);
    }

    fn apply_open(&mut self, intent: UiOpenIntent) {
        self.ensure_app(intent.app_id).active_target = Some(intent.target);
    }

    fn push_notice(&mut self, intent: UiNoticeIntent) {
        self.ensure_app(intent.app_id.clone());
        self.recent_notices.push(intent);
        if self.recent_notices.len() > self.max_notices {
            let drop_count = self.recent_notices.len() - self.max_notices;
            self.recent_notices.drain(0..drop_count);
        }
    }

    fn ensure_app(&mut self, app_id: String) -> &mut UiAppState {
        self.apps
            .entry(app_id.clone())
            .or_insert_with(|| UiAppState::new(app_id))
    }
}

impl UiAppState {
    fn new(id: String) -> Self {
        Self {
            id,
            definition: None,
            screens: BTreeMap::new(),
            panes: BTreeMap::new(),
            menus: Vec::new(),
            opens_with: None,
            active_target: None,
            visible_targets: BTreeMap::new(),
            badges: BTreeMap::new(),
            focused: None,
        }
    }

    fn upsert_menu(&mut self, intent: UiMenuIntent) {
        if let Some(existing) = self
            .menus
            .iter_mut()
            .find(|menu| menu.title == intent.title)
        {
            *existing = intent;
            return;
        }
        self.menus.push(intent);
    }
}

const fn default_max_notices() -> usize {
    DEFAULT_MAX_UI_NOTICES
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use turin_daemon_protocol::{
        EventEnvelope, UI_INTENT_EVENT, UiMenuItem, UiNode, UiNoticeLevel,
    };

    #[test]
    fn static_intents_index_apps_screens_menus_and_default_target() {
        let state = UiIntentState::from_messages([
            UiIntentMessage::new(UiIntent::App(UiAppIntent {
                id: "release".to_string(),
                title: "Release Operator".to_string(),
                about: None,
                icon: None,
            })),
            UiIntentMessage::new(UiIntent::Screen(UiScreenIntent {
                app_id: "release".to_string(),
                id: "home".to_string(),
                title: "Release Desk".to_string(),
                presentation: None,
                nodes: vec![UiNode::Text(turin_daemon_protocol::UiTextNode {
                    id: None,
                    text: "Ready".to_string(),
                })],
            })),
            UiIntentMessage::new(UiIntent::OpensWith(UiOpensWithIntent {
                app_id: "release".to_string(),
                screen_id: "home".to_string(),
            })),
            UiIntentMessage::new(UiIntent::Menu(UiMenuIntent {
                app_id: "release".to_string(),
                title: "Main".to_string(),
                items: vec![UiMenuItem {
                    label: "Dashboard".to_string(),
                    opens: "home".to_string(),
                    id: None,
                    icon: None,
                    badge: None,
                    items: Vec::new(),
                }],
            })),
        ]);

        let app = state.app("release").expect("release app");
        assert_eq!(
            app.definition.as_ref().map(|app| app.title.as_str()),
            Some("Release Operator")
        );
        assert!(app.screens.contains_key("home"));
        assert_eq!(app.opens_with.as_deref(), Some("home"));
        assert_eq!(app.active_target.as_deref(), Some("home"));
        assert_eq!(app.menus.len(), 1);
    }

    #[test]
    fn dynamic_intents_update_navigation_badges_focus_and_bounded_notices() {
        let mut state = UiIntentState::with_max_notices(2);

        state.apply_message(UiIntentMessage::new(UiIntent::Open(UiOpenIntent {
            app_id: "release".to_string(),
            target: "approvals".to_string(),
            presentation: None,
        })));
        state.apply_message(UiIntentMessage::new(UiIntent::Badge(UiBadgeIntent {
            app_id: "release".to_string(),
            target: "approvals".to_string(),
            count: Some(3),
            label: None,
            level: Some(UiNoticeLevel::Warning),
            data: Default::default(),
        })));
        state.apply_message(UiIntentMessage::new(UiIntent::Focus(UiFocusIntent {
            app_id: "release".to_string(),
            target: "open-work".to_string(),
        })));
        for title in ["first", "second", "third"] {
            state.apply_message(UiIntentMessage::new(UiIntent::Notify(UiNoticeIntent {
                app_id: "release".to_string(),
                title: title.to_string(),
                body: None,
                level: None,
            })));
        }
        state.apply_message(UiIntentMessage::new(UiIntent::Refresh(UiRefreshIntent {
            app_id: "release".to_string(),
            binding: "worklists.release".to_string(),
        })));

        let app = state.app("release").expect("release app");
        assert_eq!(app.active_target.as_deref(), Some("approvals"));
        assert_eq!(app.badges["approvals"].count, Some(3));
        assert_eq!(
            app.focused.as_ref().map(|focus| focus.target.as_str()),
            Some("open-work")
        );
        assert_eq!(state.recent_notices().len(), 2);
        assert_eq!(state.recent_notices()[0].title, "second");
        assert_eq!(state.pending_refreshes().len(), 1);
        assert_eq!(state.take_pending_refreshes().len(), 1);
        assert!(state.pending_refreshes().is_empty());
    }

    #[test]
    fn apply_event_ignores_unrelated_events_and_consumes_ui_intents() {
        let mut state = UiIntentState::new();
        let unrelated = EventEnvelope::new("runtime.snapshot", json!({}));
        assert!(!state.apply_event(&unrelated).expect("apply unrelated"));

        let event = EventEnvelope::new(
            UI_INTENT_EVENT,
            json!({
                "type": "notify",
                "app_id": "release",
                "title": "Release blocked"
            }),
        );
        assert!(state.apply_event(&event).expect("apply ui event"));
        assert_eq!(state.recent_notices().len(), 1);
        assert!(state.app("release").is_some());
    }
}
