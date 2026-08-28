pub mod content;
pub mod layout;

pub use content::TaskInputContent;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum TaskState {
    Queued,
    Running,
    Cancelling,
    Completed,
}

impl TaskState {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Running => "running",
            Self::Cancelling => "cancelling",
            Self::Completed => "completed",
        }
    }

    pub const fn is_active(self) -> bool {
        !matches!(self, Self::Completed)
    }
}

impl std::fmt::Display for TaskState {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl PartialEq<&str> for TaskState {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ThinkingConfig {
    pub enabled: bool,
    pub budget_tokens: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Default)]
pub struct ToolSelectionConfig {
    #[serde(default)]
    pub allow: Option<Vec<String>>,
    #[serde(default)]
    pub exclude: Vec<String>,
}

impl ToolSelectionConfig {
    pub fn is_empty(&self) -> bool {
        self.allow.is_none() && self.exclude.is_empty()
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Default)]
pub struct ToolsConfig {
    #[serde(flatten)]
    pub selection: ToolSelectionConfig,
    #[serde(default)]
    pub web_fetch: WebFetchToolSettings,
    #[serde(default)]
    pub web_search: WebSearchToolSettings,
}

impl ToolsConfig {
    pub fn is_empty(&self) -> bool {
        self == &Self::default()
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Default)]
pub struct WebFetchToolSettings {
    #[serde(default)]
    pub max_response_bytes: Option<usize>,
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

#[cfg(test)]
mod tests {
    use super::TaskState;

    #[test]
    fn task_state_preserves_wire_values() {
        for (state, wire) in [
            (TaskState::Queued, "queued"),
            (TaskState::Running, "running"),
            (TaskState::Cancelling, "cancelling"),
            (TaskState::Completed, "completed"),
        ] {
            assert_eq!(
                serde_json::to_string(&state).unwrap(),
                format!("\"{wire}\"")
            );
            assert_eq!(
                serde_json::from_str::<TaskState>(&format!("\"{wire}\"")).unwrap(),
                state
            );
        }
    }
}
