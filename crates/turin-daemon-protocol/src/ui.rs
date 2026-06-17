use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::EventEnvelope;

pub const UI_INTENT_EVENT: &str = "ui.intent";
pub const UI_INTENT_VERSION: u16 = 1;

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct UiIntentMessage {
    #[serde(default = "default_ui_intent_version")]
    pub version: u16,
    #[serde(default, skip_serializing_if = "UiIntentSource::is_empty")]
    pub source: UiIntentSource,
    #[serde(
        default,
        rename = "recipient",
        skip_serializing_if = "UiIntentTarget::is_empty"
    )]
    pub target: UiIntentTarget,
    #[serde(flatten)]
    pub intent: UiIntent,
}

impl UiIntentMessage {
    pub fn new(intent: UiIntent) -> Self {
        Self {
            version: UI_INTENT_VERSION,
            source: UiIntentSource::default(),
            target: UiIntentTarget::default(),
            intent,
        }
    }

    pub fn from_harness(mut self, harness_id: impl Into<String>) -> Self {
        self.source.harness_id = Some(harness_id.into());
        self
    }

    pub fn for_app(mut self, app_id: impl Into<String>) -> Self {
        self.source.app_id = Some(app_id.into());
        self
    }

    pub fn for_session(mut self, ui_session_id: impl Into<String>) -> Self {
        self.target.ui_session_id = Some(ui_session_id.into());
        self
    }

    pub fn into_event(self) -> Result<EventEnvelope, serde_json::Error> {
        Ok(EventEnvelope::new(
            UI_INTENT_EVENT,
            serde_json::to_value(self)?,
        ))
    }

    pub fn from_event(event: &EventEnvelope) -> Result<Option<Self>, serde_json::Error> {
        if event.event != UI_INTENT_EVENT {
            return Ok(None);
        }
        serde_json::from_value(event.data.clone()).map(Some)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct UiIntentSource {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub harness_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub app_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub package_id: Option<String>,
}

impl UiIntentSource {
    pub fn is_empty(&self) -> bool {
        self.harness_id.is_none()
            && self.app_id.is_none()
            && self.agent_id.is_none()
            && self.package_id.is_none()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub struct UiIntentTarget {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ui_session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_id: Option<String>,
}

impl UiIntentTarget {
    pub fn is_empty(&self) -> bool {
        self.ui_session_id.is_none() && self.client_id.is_none()
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum UiIntent {
    App(UiAppIntent),
    Screen(UiScreenIntent),
    Menu(UiMenuIntent),
    OpensWith(UiOpensWithIntent),
    Pane(UiPaneIntent),
    Open(UiOpenIntent),
    Show(UiShowIntent),
    Notify(UiNoticeIntent),
    Badge(UiBadgeIntent),
    Focus(UiFocusIntent),
    Refresh(UiRefreshIntent),
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct UiAppIntent {
    pub id: String,
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub about: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub icon: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct UiScreenIntent {
    pub app_id: String,
    pub id: String,
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presentation: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<UiNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct UiPaneIntent {
    pub app_id: String,
    pub id: String,
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presentation: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<UiNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct UiMenuIntent {
    pub app_id: String,
    pub title: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub items: Vec<UiMenuItem>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct UiMenuItem {
    pub label: String,
    pub opens: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub icon: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub badge: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub items: Vec<UiMenuItem>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct UiOpensWithIntent {
    pub app_id: String,
    pub screen_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct UiOpenIntent {
    pub app_id: String,
    pub target: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presentation: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct UiShowIntent {
    pub app_id: String,
    pub target: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub area: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub presentation: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct UiNoticeIntent {
    pub app_id: String,
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub level: Option<UiNoticeLevel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UiNoticeLevel {
    Info,
    Success,
    Warning,
    Error,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct UiBadgeIntent {
    pub app_id: String,
    pub target: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub count: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub level: Option<UiNoticeLevel>,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub data: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct UiFocusIntent {
    pub app_id: String,
    pub target: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct UiRefreshIntent {
    pub app_id: String,
    pub binding: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UiNode {
    Section(UiSectionNode),
    Text(UiTextNode),
    Action(UiActionNode),
    List(UiListNode),
    Activity(UiActivityNode),
    Detail(UiDetailNode),
    Form(UiFormNode),
    Report(UiReportNode),
    Chart(UiChartNode),
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct UiSectionNode {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub title: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<UiNode>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct UiTextNode {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct UiActionNode {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub label: String,
    pub action: String,
    #[serde(default, skip_serializing_if = "is_null")]
    pub params: Value,
    #[serde(default, skip_serializing_if = "is_false")]
    pub confirm: bool,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct UiListNode {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub title: String,
    pub source: String,
    #[serde(default, rename = "where", skip_serializing_if = "Map::is_empty")]
    pub filter: Map<String, Value>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fields: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sort: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intent: Option<String>,
    #[serde(default, rename = "as", skip_serializing_if = "Option::is_none")]
    pub render_as: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct UiActivityNode {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub title: String,
    pub source: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct UiDetailNode {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub title: String,
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub item_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct UiFormNode {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub title: String,
    pub action: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fields: Vec<UiFormField>,
    #[serde(default, skip_serializing_if = "is_null")]
    pub params: Value,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct UiFormField {
    pub name: String,
    pub label: String,
    #[serde(default, alias = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub required: Option<bool>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub options: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct UiReportNode {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub title: String,
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct UiChartNode {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub title: String,
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intent: Option<String>,
    #[serde(default, rename = "as", skip_serializing_if = "Option::is_none")]
    pub render_as: Option<String>,
}

const fn default_ui_intent_version() -> u16 {
    UI_INTENT_VERSION
}

fn is_null(value: &Value) -> bool {
    value.is_null()
}

fn is_false(value: &bool) -> bool {
    !*value
}
