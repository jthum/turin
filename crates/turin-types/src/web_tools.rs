use http::HeaderValue;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use url::Url;

use crate::{ToolsConfig, WebFetchToolSettings, WebSearchToolSettings};

pub const DEFAULT_BROWSER_USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36";
pub const DEFAULT_BROWSER_ACCEPT: &str =
    "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8";
pub const DEFAULT_BROWSER_ACCEPT_LANGUAGE: &str = "en-US,en;q=0.5";
pub const DEFAULT_BROWSER_ACCEPT_ENCODING: &str = "gzip, deflate, br";
pub const DEFAULT_BRAVE_SEARCH_URL: &str = "https://api.search.brave.com/res/v1/web/search";
pub const DEFAULT_TAVILY_SEARCH_URL: &str = "https://api.tavily.com/search";
pub const DEFAULT_DUCKDUCKGO_SEARCH_URL: &str = "https://lite.duckduckgo.com/lite/";

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WebToolKind {
    Fetch,
    Search,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct WebToolRequest {
    pub tool: WebToolKind,
    pub params: Value,
    pub tools: ToolsConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct WebToolOutput {
    pub content: String,
    pub metadata: Value,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WebToolErrorKind {
    InvalidParams,
    Execution,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WebToolResponse {
    Success {
        output: WebToolOutput,
    },
    Error {
        kind: WebToolErrorKind,
        message: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebSearchProvider {
    Brave,
    Tavily,
    Searxng,
    DuckDuckGoHtml,
}

impl WebSearchProvider {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "brave" => Some(Self::Brave),
            "tavily" => Some(Self::Tavily),
            "searxng" => Some(Self::Searxng),
            "duckduckgo_html" | "duckduckgo" | "duckduckgo_lite" => Some(Self::DuckDuckGoHtml),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Brave => "brave",
            Self::Tavily => "tavily",
            Self::Searxng => "searxng",
            Self::DuckDuckGoHtml => "duckduckgo_html",
        }
    }
}

pub fn validate_tools_config(settings: &ToolsConfig) -> Result<(), String> {
    validate_optional_header_value(
        settings.web_fetch.user_agent.as_deref(),
        "tools.web_fetch.user_agent",
    )?;
    validate_optional_header_value(
        settings.web_fetch.accept.as_deref(),
        "tools.web_fetch.accept",
    )?;
    validate_optional_header_value(
        settings.web_fetch.accept_language.as_deref(),
        "tools.web_fetch.accept_language",
    )?;
    validate_optional_header_value(
        settings.web_fetch.accept_encoding.as_deref(),
        "tools.web_fetch.accept_encoding",
    )?;
    validate_optional_header_value(
        settings.web_search.user_agent.as_deref(),
        "tools.web_search.user_agent",
    )?;

    let providers = configured_search_providers(&settings.web_search)?;
    for provider in providers {
        match provider {
            WebSearchProvider::Brave => validate_api_key_env(
                settings.web_search.brave.api_key_env.as_deref(),
                "tools.web_search.brave.api_key_env",
            )?,
            WebSearchProvider::Tavily => validate_api_key_env(
                settings.web_search.tavily.api_key_env.as_deref(),
                "tools.web_search.tavily.api_key_env",
            )?,
            WebSearchProvider::Searxng => validate_http_url_setting(
                settings.web_search.searxng.base_url.as_deref(),
                "tools.web_search.searxng.base_url",
            )?,
            WebSearchProvider::DuckDuckGoHtml => {}
        }
    }

    validate_optional_http_url(
        settings.web_search.brave.base_url.as_deref(),
        "tools.web_search.brave.base_url",
    )?;
    validate_optional_http_url(
        settings.web_search.tavily.base_url.as_deref(),
        "tools.web_search.tavily.base_url",
    )?;
    Ok(())
}

pub fn configured_search_providers(
    settings: &WebSearchToolSettings,
) -> Result<Vec<WebSearchProvider>, String> {
    let mut providers = Vec::new();
    match settings.providers.as_ref() {
        Some(configured) => {
            for provider in configured {
                let parsed = WebSearchProvider::parse(provider)
                    .ok_or_else(|| format!("unknown web_search provider '{}'", provider))?;
                if !providers.contains(&parsed) {
                    providers.push(parsed);
                }
            }
        }
        None => providers.push(WebSearchProvider::DuckDuckGoHtml),
    }

    if providers.is_empty() {
        return Err("tools.web_search.providers must not be empty".to_string());
    }

    Ok(providers)
}

pub fn validate_web_url(value: &str) -> Result<Url, String> {
    let url = Url::parse(value).map_err(|e| format!("Invalid URL: {e}"))?;
    match url.scheme() {
        "http" | "https" => Ok(url),
        other => Err(format!(
            "Unsupported URL scheme '{other}'; expected http or https"
        )),
    }
}

pub fn fetch_user_agent(settings: &WebFetchToolSettings) -> &str {
    settings
        .user_agent
        .as_deref()
        .unwrap_or(DEFAULT_BROWSER_USER_AGENT)
}

pub fn search_user_agent(settings: &WebSearchToolSettings) -> &str {
    settings
        .user_agent
        .as_deref()
        .unwrap_or(DEFAULT_BROWSER_USER_AGENT)
}

fn validate_optional_header_value(value: Option<&str>, key: &str) -> Result<(), String> {
    if let Some(value) = value {
        if value.trim().is_empty() {
            return Err(format!("{key} must not be empty"));
        }
        HeaderValue::from_str(value)
            .map_err(|e| format!("{key} must be a valid HTTP header value: {e}"))?;
    }
    Ok(())
}

fn validate_api_key_env(value: Option<&str>, key: &str) -> Result<(), String> {
    let Some(value) = value else {
        return Err(format!("{key} is required"));
    };
    if value.trim().is_empty() {
        return Err(format!("{key} must not be empty"));
    }
    Ok(())
}

fn validate_http_url_setting(value: Option<&str>, key: &str) -> Result<(), String> {
    let Some(value) = value else {
        return Err(format!("{key} is required"));
    };
    validate_web_url(value)
        .map(|_| ())
        .map_err(|error| format!("{key}: {error}"))
}

fn validate_optional_http_url(value: Option<&str>, key: &str) -> Result<(), String> {
    if let Some(value) = value {
        validate_web_url(value)
            .map(|_| ())
            .map_err(|error| format!("{key}: {error}"))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_tools_config_rejects_unknown_search_provider() {
        let mut settings = ToolsConfig::default();
        settings.web_search.providers = Some(vec!["unknown".to_string()]);
        let err = validate_tools_config(&settings).unwrap_err();
        assert!(err.contains("unknown web_search provider"));
    }

    #[test]
    fn validate_tools_config_requires_tavily_api_key_env_when_selected() {
        let mut settings = ToolsConfig::default();
        settings.web_search.providers = Some(vec!["tavily".to_string()]);
        let err = validate_tools_config(&settings).unwrap_err();
        assert!(err.contains("tools.web_search.tavily.api_key_env"));
    }
}
