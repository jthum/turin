use anyhow::{Result, bail};
use reqwest::Client;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderValue};
use serde::Deserialize;
use serde_json::Value;
use turin_types::WebSearchToolSettings;

use crate::tools::ToolError;

use super::html::{collapse_whitespace, parse_duckduckgo_lite_results, truncate_chars};
use super::{WebSearchHit, apply_api_headers, apply_html_search_headers, validate_web_url};

const DEFAULT_BRAVE_SEARCH_URL: &str = "https://api.search.brave.com/res/v1/web/search";
const DEFAULT_TAVILY_SEARCH_URL: &str = "https://api.tavily.com/search";
const DEFAULT_DUCKDUCKGO_SEARCH_URL: &str = "https://lite.duckduckgo.com/lite/";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WebSearchProvider {
    Brave,
    Tavily,
    Searxng,
    DuckDuckGoHtml,
}

impl WebSearchProvider {
    fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "brave" => Some(Self::Brave),
            "tavily" => Some(Self::Tavily),
            "searxng" => Some(Self::Searxng),
            "duckduckgo_html" | "duckduckgo" | "duckduckgo_lite" => Some(Self::DuckDuckGoHtml),
            _ => None,
        }
    }

    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Brave => "brave",
            Self::Tavily => "tavily",
            Self::Searxng => "searxng",
            Self::DuckDuckGoHtml => "duckduckgo_html",
        }
    }
}

#[derive(Debug, Deserialize)]
pub(super) struct BraveSearchResponse {
    #[serde(default)]
    web: Option<BraveWebResultSet>,
}

#[derive(Debug, Deserialize)]
struct BraveWebResultSet {
    #[serde(default)]
    results: Vec<BraveWebResult>,
}

#[derive(Debug, Deserialize)]
struct BraveWebResult {
    title: String,
    url: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    extra_snippets: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub(super) struct TavilySearchResponse {
    #[serde(default)]
    results: Vec<TavilySearchResult>,
}

#[derive(Debug, Deserialize)]
struct TavilySearchResult {
    #[serde(default)]
    title: String,
    url: String,
    #[serde(default)]
    content: String,
}

#[derive(Debug, Deserialize)]
pub(super) struct SearxngSearchResponse {
    #[serde(default)]
    results: Vec<SearxngSearchResult>,
}

#[derive(Debug, Deserialize)]
struct SearxngSearchResult {
    #[serde(default)]
    title: String,
    url: String,
    #[serde(default)]
    content: String,
}

pub(super) fn configured_search_providers(
    settings: &WebSearchToolSettings,
) -> Result<Vec<WebSearchProvider>> {
    let mut providers = Vec::new();
    match settings.providers.as_ref() {
        Some(configured) => {
            for provider in configured {
                let parsed = WebSearchProvider::parse(provider)
                    .ok_or_else(|| anyhow::anyhow!("unknown web_search provider '{}'", provider))?;
                if !providers.contains(&parsed) {
                    providers.push(parsed);
                }
            }
        }
        None => providers.push(WebSearchProvider::DuckDuckGoHtml),
    }
    if providers.is_empty() {
        bail!("tools.web_search.providers must not be empty");
    }
    Ok(providers)
}

pub(super) fn normalize_hits_for_output(hits: &[WebSearchHit]) -> Vec<Value> {
    hits.iter()
        .map(|hit| {
            serde_json::json!({
                "title": hit.title,
                "url": hit.url,
                "snippet": hit.snippet,
            })
        })
        .collect()
}

fn search_hit(title: &str, url: String, snippet: &str) -> Option<WebSearchHit> {
    if title.trim().is_empty() || url.trim().is_empty() {
        return None;
    }
    Some(WebSearchHit {
        title: collapse_whitespace(title),
        url,
        snippet: truncate_chars(&collapse_whitespace(snippet), 400),
    })
}

pub(super) fn brave_hits_from_response(
    response: BraveSearchResponse,
    limit: usize,
) -> Vec<WebSearchHit> {
    response
        .web
        .map(|web| {
            web.results
                .into_iter()
                .take(limit)
                .filter_map(|result| {
                    let snippet = if !result.description.trim().is_empty() {
                        result.description
                    } else {
                        result.extra_snippets.into_iter().next().unwrap_or_default()
                    };
                    search_hit(&result.title, result.url, &snippet)
                })
                .collect()
        })
        .unwrap_or_default()
}

pub(super) fn tavily_hits_from_response(
    response: TavilySearchResponse,
    limit: usize,
) -> Vec<WebSearchHit> {
    response
        .results
        .into_iter()
        .take(limit)
        .filter_map(|result| search_hit(&result.title, result.url, &result.content))
        .collect()
}

pub(super) fn searxng_hits_from_response(
    response: SearxngSearchResponse,
    limit: usize,
) -> Vec<WebSearchHit> {
    response
        .results
        .into_iter()
        .take(limit)
        .filter_map(|result| search_hit(&result.title, result.url, &result.content))
        .collect()
}

async fn read_json_response<T: for<'de> Deserialize<'de>>(
    response: reqwest::Response,
    label: &str,
) -> Result<T, ToolError> {
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|e| ToolError::ExecutionError(format!("{label} body read failed: {e}")))?;
    if !status.is_success() {
        return Err(ToolError::ExecutionError(format!(
            "{label} request failed with {}: {}",
            status.as_u16(),
            truncate_chars(&collapse_whitespace(&body), 400)
        )));
    }
    serde_json::from_str(&body)
        .map_err(|e| ToolError::ExecutionError(format!("{label} response decode failed: {e}")))
}

fn load_required_api_key(
    config: Option<&crate::kernel::config::TurinConfig>,
    env_name: &str,
    provider: &str,
) -> Result<String, ToolError> {
    let value = config
        .and_then(|config| config.environment_value(env_name))
        .or_else(|| std::env::var(env_name).ok());
    value.ok_or_else(|| {
        ToolError::ExecutionError(format!(
            "{provider} search requires environment variable '{}'",
            env_name
        ))
    })
}

pub(super) async fn search_duckduckgo_html(
    client: &Client,
    settings: &WebSearchToolSettings,
    query: &str,
    limit: usize,
) -> Result<Vec<WebSearchHit>, ToolError> {
    let request = client
        .get(DEFAULT_DUCKDUCKGO_SEARCH_URL)
        .query(&[("q", query)]);
    let request = apply_html_search_headers(request, settings)?;
    let response = request.send().await.map_err(|e| {
        ToolError::ExecutionError(format!("duckduckgo_html search request failed: {e}"))
    })?;
    let html = response.text().await.map_err(|e| {
        ToolError::ExecutionError(format!("duckduckgo_html search body read failed: {e}"))
    })?;
    Ok(parse_duckduckgo_lite_results(&html, limit))
}

pub(super) async fn search_brave(
    client: &Client,
    settings: &WebSearchToolSettings,
    config: Option<&crate::kernel::config::TurinConfig>,
    query: &str,
    limit: usize,
) -> Result<Vec<WebSearchHit>, ToolError> {
    let env_name =
        settings.brave.api_key_env.as_deref().ok_or_else(|| {
            ToolError::ExecutionError("Brave search is not configured".to_string())
        })?;
    let api_key = load_required_api_key(config, env_name, "Brave")?;
    let request = client
        .get(
            settings
                .brave
                .base_url
                .as_deref()
                .unwrap_or(DEFAULT_BRAVE_SEARCH_URL),
        )
        .query(&[("q", query), ("count", &limit.to_string())])
        .header("X-Subscription-Token", api_key);
    let request = apply_api_headers(request, settings)?;
    let response = request
        .send()
        .await
        .map_err(|e| ToolError::ExecutionError(format!("brave search request failed: {e}")))?;
    let parsed: BraveSearchResponse = read_json_response(response, "brave search").await?;
    Ok(brave_hits_from_response(parsed, limit))
}

pub(super) async fn search_tavily(
    client: &Client,
    settings: &WebSearchToolSettings,
    config: Option<&crate::kernel::config::TurinConfig>,
    query: &str,
    limit: usize,
) -> Result<Vec<WebSearchHit>, ToolError> {
    let env_name =
        settings.tavily.api_key_env.as_deref().ok_or_else(|| {
            ToolError::ExecutionError("Tavily search is not configured".to_string())
        })?;
    let api_key = load_required_api_key(config, env_name, "Tavily")?;
    let request = client
        .post(
            settings
                .tavily
                .base_url
                .as_deref()
                .unwrap_or(DEFAULT_TAVILY_SEARCH_URL),
        )
        .header(AUTHORIZATION, format!("Bearer {api_key}"))
        .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
        .json(&serde_json::json!({
            "query": query,
            "search_depth": "basic",
            "max_results": limit,
            "include_answer": false,
            "include_raw_content": false,
            "include_images": false,
        }));
    let request = apply_api_headers(request, settings)?;
    let response = request
        .send()
        .await
        .map_err(|e| ToolError::ExecutionError(format!("tavily search request failed: {e}")))?;
    let parsed: TavilySearchResponse = read_json_response(response, "tavily search").await?;
    Ok(tavily_hits_from_response(parsed, limit))
}

pub(super) async fn search_searxng(
    client: &Client,
    settings: &WebSearchToolSettings,
    query: &str,
    limit: usize,
) -> Result<Vec<WebSearchHit>, ToolError> {
    let base_url =
        settings.searxng.base_url.as_deref().ok_or_else(|| {
            ToolError::ExecutionError("SearXNG search is not configured".to_string())
        })?;
    let mut search_url = validate_web_url(base_url)?;
    if search_url.path().is_empty() || search_url.path() == "/" {
        search_url.set_path("/search");
    }
    let request = client
        .get(search_url)
        .query(&[("q", query), ("format", "json")]);
    let request = apply_html_search_headers(request, settings)?;
    let response = request
        .send()
        .await
        .map_err(|e| ToolError::ExecutionError(format!("searxng search request failed: {e}")))?;
    let parsed: SearxngSearchResponse = read_json_response(response, "searxng search").await?;
    Ok(searxng_hits_from_response(parsed, limit))
}
