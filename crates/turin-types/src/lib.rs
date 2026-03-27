use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum AgentMode {
    #[default]
    Auto,
    Stateful,
    Stateless,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ThinkingConfig {
    pub enabled: bool,
    pub budget_tokens: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Default)]
pub struct ToolSelectionConfig {
    #[serde(default)]
    pub tools: Option<Vec<String>>,
    #[serde(default)]
    pub tools_exclude: Vec<String>,
}

impl ToolSelectionConfig {
    pub fn is_empty(&self) -> bool {
        self.tools.is_none() && self.tools_exclude.is_empty()
    }
}
