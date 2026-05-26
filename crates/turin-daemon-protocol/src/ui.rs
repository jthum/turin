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
    pub agent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub package_id: Option<String>,
}

impl UiIntentSource {
    pub fn is_empty(&self) -> bool {
        self.harness_id.is_none() && self.agent_id.is_none() && self.package_id.is_none()
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
    Home(UiHomeIntent),
    Show(UiShowIntent),
    Notify(UiNoticeIntent),
    Focus(UiFocusIntent),
    Refresh(UiRefreshIntent),
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct UiAppIntent {
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtitle: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub icon: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct UiHomeIntent {
    pub title: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subtitle: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<UiNode>,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct UiShowIntent {
    pub area: String,
    pub node: UiNode,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct UiNoticeIntent {
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

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct UiFocusIntent {
    pub target: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct UiRefreshIntent {
    pub binding: String,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UiNode {
    Section(UiSectionNode),
    Text(UiTextNode),
    Action(UiActionNode),
    Worklist(UiWorklistNode),
    Activity(UiActivityNode),
    Detail(UiDetailNode),
    ApprovalQueue(UiApprovalQueueNode),
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
pub struct UiWorklistNode {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub title: String,
    pub source: String,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub filters: Map<String, Value>,
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
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct UiApprovalQueueNode {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub title: String,
    pub source: String,
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub filters: Map<String, Value>,
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
