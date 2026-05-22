use serde::{Deserialize, Serialize};
use serde_json::Value;
use turin_types::{TaskInputContent, ToolsConfig};

use crate::ContextPersistenceParams;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScheduleActionParams {
    pub name: String,
    #[serde(default)]
    pub params: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScheduleCreateParams {
    pub agent_id: String,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<TaskInputContent>>,
    #[serde(default)]
    pub tools: Option<ToolsConfig>,
    #[serde(default)]
    pub conflict_policy: Option<String>,
    #[serde(default)]
    pub action: Option<ScheduleActionParams>,
    pub next_run_unix_ms: i64,
    #[serde(default)]
    pub interval_seconds: Option<u64>,
    #[serde(default)]
    pub recurring_pattern: Option<String>,
    #[serde(default)]
    pub overlap_policy: Option<String>,
    #[serde(default)]
    pub work_key: Option<String>,
    #[serde(default)]
    pub max_concurrency: Option<u32>,
    #[serde(default)]
    pub persistence: Option<ContextPersistenceParams>,
    #[serde(default = "crate::default_enabled")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScheduleUpdateParams {
    pub id: String,
    #[serde(default)]
    pub agent_id: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Vec<TaskInputContent>>,
    #[serde(default)]
    pub tools: Option<ToolsConfig>,
    #[serde(default)]
    pub conflict_policy: Option<String>,
    #[serde(default)]
    pub action: Option<ScheduleActionParams>,
    #[serde(default)]
    pub next_run_unix_ms: Option<i64>,
    #[serde(default)]
    pub interval_seconds: Option<u64>,
    #[serde(default)]
    pub recurring_pattern: Option<String>,
    #[serde(default)]
    pub overlap_policy: Option<String>,
    #[serde(default)]
    pub work_key: Option<String>,
    #[serde(default)]
    pub max_concurrency: Option<u32>,
    #[serde(default)]
    pub persistence: Option<ContextPersistenceParams>,
    #[serde(default)]
    pub enabled: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScheduleRunsParams {
    pub id: String,
    #[serde(default)]
    pub active_only: bool,
    #[serde(default)]
    pub limit: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScheduleJobList {
    pub jobs: Vec<ScheduleJobDetail>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScheduleJobRunList {
    pub public_id: String,
    pub runs: Vec<ScheduleJobRunDetail>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScheduleJobDetail {
    pub id: i64,
    pub public_id: String,
    pub agent_id: String,
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
    #[serde(default)]
    pub persistence: Option<ContextPersistenceParams>,
    pub next_run_unix_ms: i64,
    #[serde(default)]
    pub interval_seconds: Option<u64>,
    #[serde(default)]
    pub recurring_pattern: Option<String>,
    pub overlap_policy: String,
    #[serde(default)]
    pub work_key: Option<String>,
    #[serde(default)]
    pub max_concurrency: Option<u32>,
    pub enabled: bool,
    pub slot_id: String,
    #[serde(default)]
    pub running_task_id: Option<String>,
    #[serde(default)]
    pub active_run_count: u32,
    pub pending_rerun: bool,
    #[serde(default)]
    pub last_run_unix_ms: Option<i64>,
    #[serde(default)]
    pub last_status: Option<String>,
    #[serde(default)]
    pub last_error_code: Option<String>,
    #[serde(default)]
    pub failure_count: u64,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScheduleJobRunDetail {
    pub id: i64,
    pub task_id: String,
    pub started_unix_ms: i64,
    #[serde(default)]
    pub finished_unix_ms: Option<i64>,
    #[serde(default)]
    pub duration_ms: Option<u64>,
    #[serde(default)]
    pub last_status: Option<String>,
    pub active: bool,
    pub created_at: String,
    pub updated_at: String,
}
