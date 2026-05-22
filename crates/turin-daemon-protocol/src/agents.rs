use serde::{Deserialize, Serialize};
use turin_types::{ThinkingConfig, ToolsConfig};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CreateAgentParams {
    pub id: String,
    pub provider: String,
    pub model: String,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
    #[serde(default)]
    pub harness: Option<String>,
    #[serde(default)]
    pub idle_timeout_seconds: Option<u64>,
    #[serde(default)]
    pub tools: ToolsConfig,
    #[serde(default = "crate::default_enabled")]
    pub enabled: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UpdateAgentParams {
    pub id: String,
    #[serde(default)]
    pub provider: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
    #[serde(default)]
    pub idle_timeout_seconds: Option<u64>,
    #[serde(default)]
    pub tools: Option<ToolsConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BindHarnessParams {
    pub id: String,
    pub harness_id: String,
}
