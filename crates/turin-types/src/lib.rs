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

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Default)]
pub struct ToolSettingsConfig {
    #[serde(default)]
    pub web_fetch: WebFetchToolSettings,
    #[serde(default)]
    pub web_search: WebSearchToolSettings,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Default)]
pub struct WebFetchToolSettings {
    #[serde(default)]
    pub user_agent: Option<String>,
    #[serde(default)]
    pub accept: Option<String>,
    #[serde(default)]
    pub accept_language: Option<String>,
    #[serde(default)]
    pub accept_encoding: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Default)]
pub struct WebSearchToolSettings {
    #[serde(default)]
    pub providers: Option<Vec<String>>,
    #[serde(default)]
    pub user_agent: Option<String>,
    #[serde(default)]
    pub brave: BraveSearchToolSettings,
    #[serde(default)]
    pub tavily: TavilySearchToolSettings,
    #[serde(default)]
    pub searxng: SearxngSearchToolSettings,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Default)]
pub struct BraveSearchToolSettings {
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Default)]
pub struct TavilySearchToolSettings {
    #[serde(default)]
    pub api_key_env: Option<String>,
    #[serde(default)]
    pub base_url: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Default)]
pub struct SearxngSearchToolSettings {
    #[serde(default)]
    pub base_url: Option<String>,
}
