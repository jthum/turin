use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::UiIntentMessage;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct HarnessSourceEntry {
    pub path: String,
    pub hash: String,
    pub bytes: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct HarnessSourceFile {
    pub path: String,
    pub source: String,
    pub hash: String,
    pub bytes: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct HarnessSourceListResult {
    pub harness_id: String,
    pub files: Vec<HarnessSourceEntry>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct HarnessSourceGetParams {
    pub id: String,
    pub path: String,
}

/// An in-memory candidate source. `None` removes the path from the candidate
/// harness without changing the filesystem.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct HarnessSourceOverlay {
    pub path: String,
    pub source: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct HarnessSourceValidateParams {
    pub id: String,
    #[serde(default)]
    pub changes: Vec<HarnessSourceOverlay>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct HarnessSourceValidationResult {
    pub harness_id: String,
    pub directory: String,
    pub script_count: usize,
    pub valid: bool,
}

/// A source save guarded by the exact hash read by the editor. `source = None`
/// deletes the path; `expected_hash = None` requires that it does not exist.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct HarnessSourceSaveChange {
    pub path: String,
    pub source: Option<String>,
    pub expected_hash: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct HarnessSourceSaveParams {
    pub id: String,
    pub changes: Vec<HarnessSourceSaveChange>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct HarnessSourceSaveResult {
    pub harness_id: String,
    #[serde(default)]
    pub saved: Vec<HarnessSourceEntry>,
    #[serde(default)]
    pub deleted: Vec<String>,
}

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
