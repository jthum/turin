use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use turin_types::ToolsConfig;

use crate::kernel::config::{
    AgentConfig, ContextPersistenceConfig, InferenceOverrideConfig, ThinkingConfig,
};

#[derive(Debug, Clone, Serialize)]
pub struct RegistryIssue {
    pub path: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SharedHarnessSummary {
    pub id: String,
    pub directory: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentSummary {
    pub id: String,
    pub directory: String,
    pub enabled: bool,
    pub provider: String,
    pub model: String,
    pub idle_timeout_seconds: Option<u64>,
    pub harness_kind: String,
    pub harness_ref: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct RegistrySnapshot {
    pub agents_dir: String,
    pub harnesses_dir: String,
    pub agents: Vec<AgentSummary>,
    pub shared_harnesses: Vec<SharedHarnessSummary>,
    pub issues: Vec<RegistryIssue>,
}

#[derive(Debug, Clone)]
pub struct DiscoveredAgent {
    pub id: String,
    pub directory: PathBuf,
    pub enabled: bool,
    pub agent_config: AgentConfig,
    pub harness_id: String,
    pub harness_kind: HarnessKind,
    pub harness_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarnessKind {
    Local,
    Shared,
}

#[derive(Debug, Clone)]
pub struct SharedHarness {
    pub id: String,
    pub directory: PathBuf,
}

#[derive(Debug, Clone)]
pub struct RegistryLoad {
    pub agents_dir: PathBuf,
    pub harnesses_dir: PathBuf,
    pub agents: Vec<DiscoveredAgent>,
    pub shared_harnesses: Vec<SharedHarness>,
    pub issues: Vec<RegistryIssue>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct AgentFileConfig {
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub system_prompt: Option<String>,
    pub model: String,
    pub provider: String,
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
    #[serde(default)]
    pub harness: Option<String>,
    #[serde(default)]
    pub idle_timeout_seconds: Option<u64>,
    #[serde(default)]
    pub linked_runtime_lanes: Option<usize>,
    #[serde(default)]
    pub tools: ToolsConfig,
    #[serde(default)]
    pub inference: InferenceOverrideConfig,
    #[serde(default)]
    pub persistence: ContextPersistenceConfig,
}

fn default_enabled() -> bool {
    true
}
