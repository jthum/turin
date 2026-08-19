use serde::{Deserialize, Serialize};
use turin_types::{TaskInputContent, ToolsConfig};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SubmitTaskParams {
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub slot_id: Option<String>,
    pub prompt: String,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_context: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<TaskInputContent>>,
    #[serde(default)]
    pub tools: Option<ToolsConfig>,
    #[serde(default)]
    pub conflict_policy: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SidestepContextTargetParams {
    BranchHead { branch_head_id: i64 },
    TurnId { turn_id: i64 },
    SelectedPath { turn_ids: Vec<i64> },
    ExternalReference { reference: String },
    SummarySource { source_turn_id: i64 },
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SidestepModeParams {
    #[default]
    Ephemeral,
    ForkSibling,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SidestepTaskParams {
    pub session_id: String,
    #[serde(default)]
    pub slot_id: Option<String>,
    pub prompt: String,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<TaskInputContent>>,
    #[serde(default)]
    pub tools: Option<ToolsConfig>,
    #[serde(default)]
    pub mode: SidestepModeParams,
    #[serde(default)]
    pub context_target: Option<SidestepContextTargetParams>,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TaskIdParams {
    pub request_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WaitTaskParams {
    pub request_id: String,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PromoteTaskParams {
    pub request_id: String,
    #[serde(default)]
    pub branch_name: Option<String>,
    #[serde(default)]
    pub source_turn_id: Option<i64>,
}
