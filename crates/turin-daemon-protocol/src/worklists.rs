use serde::{Deserialize, Serialize};
use serde_json::Value;
use turin_types::{TaskInputContent, ToolsConfig};

use crate::{ContextPersistenceParams, ScheduleActionParams};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorklistListParams {
    #[serde(default)]
    pub persistence: Option<ContextPersistenceParams>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub scope: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorklistTargetParams {
    pub id: String,
    #[serde(default)]
    pub persistence: Option<ContextPersistenceParams>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorkItemTargetParams {
    pub id: String,
    #[serde(default)]
    pub persistence: Option<ContextPersistenceParams>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorklistItemsParams {
    pub id: String,
    #[serde(default)]
    pub persistence: Option<ContextPersistenceParams>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub parent_id: Option<String>,
    #[serde(default)]
    pub r#where: Option<serde_json::Map<String, Value>>,
    #[serde(default)]
    pub claimed_only: bool,
    #[serde(default)]
    pub paused_only: bool,
    #[serde(default)]
    pub due_only: bool,
    #[serde(default)]
    pub limit: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorklistList {
    pub worklists: Vec<WorklistDetail>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorkItemList {
    pub worklist_id: String,
    pub items: Vec<WorkItemDetail>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorklistDetail {
    pub id: i64,
    pub public_id: String,
    pub name: String,
    pub scope_ref: String,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub persistence: Option<ContextPersistenceParams>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct WorkItemDetail {
    pub id: i64,
    pub public_id: String,
    pub worklist_id: String,
    #[serde(default)]
    pub parent_id: Option<String>,
    pub title: String,
    pub kind: String,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub content: Option<Vec<TaskInputContent>>,
    #[serde(default)]
    pub tools: Option<ToolsConfig>,
    #[serde(default)]
    pub conflict_policy: Option<String>,
    #[serde(default)]
    pub action: Option<ScheduleActionParams>,
    pub status: String,
    #[serde(default)]
    pub paused: bool,
    #[serde(default)]
    pub pause_reason: Option<String>,
    #[serde(default)]
    pub pause_until_unix_ms: Option<i64>,
    pub priority: i64,
    #[serde(default)]
    pub after: Option<Vec<String>>,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub claim_agent_id: Option<String>,
    #[serde(default)]
    pub claim_session_id: Option<String>,
    #[serde(default)]
    pub claim_execution_id: Option<String>,
    #[serde(default)]
    pub claim_heartbeat_unix_ms: Option<i64>,
    #[serde(default)]
    pub claimed_at: Option<String>,
    #[serde(default)]
    pub completed_at: Option<String>,
    #[serde(default)]
    pub failure_reason: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}
