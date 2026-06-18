use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::UiIntentMessage;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HarnessActionRunParams {
    pub action: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub harness_id: Option<String>,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub params: Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HarnessActionRunResult {
    pub action: String,
    pub agent_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub harness_id: Option<String>,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub result: Value,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ui_intents: Vec<UiIntentMessage>,
}
