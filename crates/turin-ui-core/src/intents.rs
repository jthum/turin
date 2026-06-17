use std::collections::BTreeMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use turin_daemon_protocol::{
    EventEnvelope, UiAppIntent, UiBadgeIntent, UiFocusIntent, UiIntent, UiIntentMessage,
    UiIntentSource, UiMenuIntent, UiNoticeIntent, UiOpenIntent, UiOpensWithIntent, UiPaneIntent,
    UiRefreshIntent, UiScreenIntent, UiShowIntent,
};

pub const DEFAULT_MAX_UI_NOTICES: usize = 32;

/// Client-side registry of UI facts and requests decoded from harness UI intents.
///
/// This type deliberately does not own navigation state such as the active app,
/// screen, pane, modal, selection, or scroll position. Each client decides how
/// to honor, defer, degrade, or ignore the recorded requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiRegistry {
    #[serde(default)]
    apps: BTreeMap<String, UiAppRecord>,
    #[serde(default = "default_max_notices")]
    max_notices: usize,
    #[serde(default)]
    notices: Vec<UiNoticeIntent>,
    #[serde(default)]
    opens: Vec<UiOpenIntent>,
    #[serde(default)]
    shows: Vec<UiShowIntent>,
    #[serde(default)]
    focuses: Vec<UiFocusIntent>,
    #[serde(default)]
    refreshes: Vec<UiRefreshIntent>,
}

/// Declared surfaces for one harness-defined app.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiAppRecord {
    pub id: String,
    #[serde(default, skip_serializing_if = "UiIntentSource::is_empty")]
    pub source: UiIntentSource,
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
    #[serde(default)]
    pub badges: BTreeMap<String, UiBadgeIntent>,
}

impl Default for UiRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl UiRegistry {
    pub fn new() -> Self {
        Self {
            apps: BTreeMap::new(),
            max_notices: DEFAULT_MAX_UI_NOTICES,
            notices: Vec::new(),
            opens: Vec::new(),
            shows: Vec::new(),
            focuses: Vec::new(),
            refreshes: Vec::new(),
        }
    }

    pub fn with_max_notices(max_notices: usize) -> Self {
        Self {
            max_notices,
            ..Self::new()
        }
    }

    pub fn from_messages(messages: impl IntoIterator<Item = UiIntentMessage>) -> Self {
        let mut registry = Self::new();
        registry.apply_messages(messages);
        registry
    }

    pub fn apps(&self) -> impl Iterator<Item = &UiAppRecord> {
        self.apps.values()
    }

    pub fn app(&self, app_id: &str) -> Option<&UiAppRecord> {
        self.apps.get(app_id)
    }

    pub fn notices(&self) -> &[UiNoticeIntent] {
        &self.notices
    }

    pub fn opens(&self) -> &[UiOpenIntent] {
        &self.opens
    }

    pub fn shows(&self) -> &[UiShowIntent] {
        &self.shows
    }

    pub fn focuses(&self) -> &[UiFocusIntent] {
        &self.focuses
    }

    pub fn refreshes(&self) -> &[UiRefreshIntent] {
        &self.refreshes
    }

    pub fn take_opens(&mut self) -> Vec<UiOpenIntent> {
        std::mem::take(&mut self.opens)
    }

    pub fn take_shows(&mut self) -> Vec<UiShowIntent> {
        std::mem::take(&mut self.shows)
    }

    pub fn take_focuses(&mut self) -> Vec<UiFocusIntent> {
        std::mem::take(&mut self.focuses)
    }

    pub fn take_refreshes(&mut self) -> Vec<UiRefreshIntent> {
        std::mem::take(&mut self.refreshes)
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
        let source = message.source;
        match message.intent {
            UiIntent::App(intent) => {
                let app = self.ensure_app_with_source(intent.id.clone(), &source);
                app.definition = Some(intent);
            }
            UiIntent::Screen(intent) => {
                self.ensure_app_with_source(intent.app_id.clone(), &source)
                    .screens
                    .insert(intent.id.clone(), intent);
            }
            UiIntent::Menu(intent) => {
                self.ensure_app_with_source(intent.app_id.clone(), &source)
                    .upsert_menu(intent);
            }
            UiIntent::OpensWith(intent) => self.apply_opens_with(intent, &source),
            UiIntent::Pane(intent) => {
                self.ensure_app_with_source(intent.app_id.clone(), &source)
                    .panes
                    .insert(intent.id.clone(), intent);
            }
            UiIntent::Open(intent) => self.apply_open(intent, &source),
            UiIntent::Show(intent) => {
                self.ensure_app_with_source(intent.app_id.clone(), &source);
                self.shows.push(intent);
            }
            UiIntent::Notify(intent) => self.push_notice(intent, &source),
            UiIntent::Badge(intent) => {
                self.ensure_app_with_source(intent.app_id.clone(), &source)
                    .badges
                    .insert(intent.target.clone(), intent);
            }
            UiIntent::Focus(intent) => {
                let app_id = intent.app_id.clone();
                self.ensure_app_with_source(app_id, &source);
                self.focuses.push(intent);
            }
            UiIntent::Refresh(intent) => {
                self.ensure_app_with_source(intent.app_id.clone(), &source);
                self.refreshes.push(intent);
            }
        }
    }

    fn apply_opens_with(&mut self, intent: UiOpensWithIntent, source: &UiIntentSource) {
        let app = self.ensure_app_with_source(intent.app_id, source);
        app.opens_with = Some(intent.screen_id);
    }

    fn apply_open(&mut self, intent: UiOpenIntent, source: &UiIntentSource) {
        self.ensure_app_with_source(intent.app_id.clone(), source);
        self.opens.push(intent);
    }

    fn push_notice(&mut self, intent: UiNoticeIntent, source: &UiIntentSource) {
        self.ensure_app_with_source(intent.app_id.clone(), source);
        self.notices.push(intent);
        if self.notices.len() > self.max_notices {
            let drop_count = self.notices.len() - self.max_notices;
            self.notices.drain(0..drop_count);
        }
    }

    fn ensure_app(&mut self, app_id: String) -> &mut UiAppRecord {
        self.apps
            .entry(app_id.clone())
            .or_insert_with(|| UiAppRecord::new(app_id))
    }

    fn ensure_app_with_source(
        &mut self,
        app_id: String,
        source: &UiIntentSource,
    ) -> &mut UiAppRecord {
        let app = self.ensure_app(app_id);
        app.merge_source(source);
        app
    }
}

impl UiAppRecord {
    fn new(id: String) -> Self {
        Self {
            id,
            source: UiIntentSource::default(),
            definition: None,
            screens: BTreeMap::new(),
            panes: BTreeMap::new(),
            menus: Vec::new(),
            opens_with: None,
            badges: BTreeMap::new(),
        }
    }

    fn merge_source(&mut self, source: &UiIntentSource) {
        if self.source.harness_id.is_none() {
            self.source.harness_id.clone_from(&source.harness_id);
        }
        if self.source.app_id.is_none() {
            self.source.app_id.clone_from(&source.app_id);
        }
        if self.source.agent_id.is_none() {
            self.source.agent_id.clone_from(&source.agent_id);
        }
        if self.source.package_id.is_none() {
            self.source.package_id.clone_from(&source.package_id);
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
        let registry = UiRegistry::from_messages([
            UiIntentMessage::new(UiIntent::App(UiAppIntent {
                id: "release".to_string(),
                title: "Release Operator".to_string(),
                about: None,
                icon: None,
            }))
            .from_harness("release-harness"),
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

        let app = registry.app("release").expect("release app");
        assert_eq!(
            app.definition.as_ref().map(|app| app.title.as_str()),
            Some("Release Operator")
        );
        assert!(app.screens.contains_key("home"));
        assert_eq!(app.opens_with.as_deref(), Some("home"));
        assert_eq!(app.menus.len(), 1);
        assert_eq!(app.source.harness_id.as_deref(), Some("release-harness"));
    }

    #[test]
    fn dynamic_intents_record_requests_badges_focuses_and_bounded_notices() {
        let mut registry = UiRegistry::with_max_notices(2);

        registry.apply_message(UiIntentMessage::new(UiIntent::Open(UiOpenIntent {
            app_id: "release".to_string(),
            target: "approvals".to_string(),
            presentation: None,
        })));
        registry.apply_message(UiIntentMessage::new(UiIntent::Badge(UiBadgeIntent {
            app_id: "release".to_string(),
            target: "approvals".to_string(),
            count: Some(3),
            label: None,
            level: Some(UiNoticeLevel::Warning),
            data: Default::default(),
        })));
        registry.apply_message(UiIntentMessage::new(UiIntent::Focus(UiFocusIntent {
            app_id: "release".to_string(),
            target: "open-work".to_string(),
        })));
        registry.apply_message(UiIntentMessage::new(UiIntent::Show(UiShowIntent {
            app_id: "release".to_string(),
            target: "release-panel".to_string(),
            area: Some("side".to_string()),
            presentation: None,
        })));
        for title in ["first", "second", "third"] {
            registry.apply_message(UiIntentMessage::new(UiIntent::Notify(UiNoticeIntent {
                app_id: "release".to_string(),
                title: title.to_string(),
                body: None,
                level: None,
            })));
        }
        registry.apply_message(UiIntentMessage::new(UiIntent::Refresh(UiRefreshIntent {
            app_id: "release".to_string(),
            binding: "worklists.release".to_string(),
        })));

        let app = registry.app("release").expect("release app");
        assert_eq!(registry.opens()[0].target, "approvals");
        assert_eq!(app.badges["approvals"].count, Some(3));
        assert_eq!(registry.focuses()[0].target, "open-work");
        assert_eq!(registry.shows()[0].target, "release-panel");
        assert_eq!(registry.notices().len(), 2);
        assert_eq!(registry.notices()[0].title, "second");
        assert_eq!(registry.refreshes().len(), 1);
        assert_eq!(registry.take_opens().len(), 1);
        assert!(registry.opens().is_empty());
        assert_eq!(registry.take_shows().len(), 1);
        assert!(registry.shows().is_empty());
        assert_eq!(registry.take_focuses().len(), 1);
        assert!(registry.focuses().is_empty());
        assert_eq!(registry.take_refreshes().len(), 1);
        assert!(registry.refreshes().is_empty());
    }

    #[test]
    fn apply_event_ignores_unrelated_events_and_consumes_ui_intents() {
        let mut registry = UiRegistry::new();
        let unrelated = EventEnvelope::new("runtime.snapshot", json!({}));
        assert!(!registry.apply_event(&unrelated).expect("apply unrelated"));

        let event = EventEnvelope::new(
            UI_INTENT_EVENT,
            json!({
                "type": "notify",
                "app_id": "release",
                "title": "Release blocked"
            }),
        );
        assert!(registry.apply_event(&event).expect("apply ui event"));
        assert_eq!(registry.notices().len(), 1);
        assert!(registry.app("release").is_some());
    }
}
