use std::env;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use reqwest::header::{
    ACCEPT, ACCEPT_ENCODING, ACCEPT_LANGUAGE, AUTHORIZATION, CONTENT_TYPE, HeaderValue, USER_AGENT,
};
use reqwest::redirect::Policy;
use reqwest::{Client, RequestBuilder};
use scraper::{Html, Selector};
use serde::Deserialize;
use serde_json::Value;
use turin_types::{ToolsConfig, WebFetchToolSettings, WebSearchToolSettings};
use url::Url;

use crate::tools::{Tool, ToolContext, ToolEffect, ToolError, ToolOutput, parse_args};

pub struct WebFetchTool;
pub struct WebSearchTool;

const DEFAULT_BROWSER_USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36";
const DEFAULT_BROWSER_ACCEPT: &str =
    "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8";
const DEFAULT_BROWSER_ACCEPT_LANGUAGE: &str = "en-US,en;q=0.5";
const DEFAULT_BROWSER_ACCEPT_ENCODING: &str = "gzip, deflate, br";
const DEFAULT_BRAVE_SEARCH_URL: &str = "https://api.search.brave.com/res/v1/web/search";
const DEFAULT_TAVILY_SEARCH_URL: &str = "https://api.tavily.com/search";
const DEFAULT_DUCKDUCKGO_SEARCH_URL: &str = "https://lite.duckduckgo.com/lite/";

#[derive(Debug, Clone, PartialEq, Eq)]
struct WebSearchHit {
    title: String,
    url: String,
    snippet: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WebSearchProvider {
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

    fn as_str(self) -> &'static str {
        match self {
            Self::Brave => "brave",
            Self::Tavily => "tavily",
            Self::Searxng => "searxng",
            Self::DuckDuckGoHtml => "duckduckgo_html",
        }
    }
}

#[derive(Deserialize)]
struct WebFetchArgs {
    url: String,
    #[serde(default = "default_fetch_timeout_seconds")]
    timeout_seconds: u64,
    #[serde(default = "default_fetch_max_chars")]
    max_chars: usize,
}

#[derive(Deserialize)]
struct WebSearchArgs {
    query: String,
    #[serde(default = "default_search_limit")]
    limit: usize,
    #[serde(default = "default_fetch_timeout_seconds")]
    timeout_seconds: u64,
}

#[derive(Debug, Deserialize)]
struct BraveSearchResponse {
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
struct TavilySearchResponse {
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
struct SearxngSearchResponse {
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

fn default_fetch_timeout_seconds() -> u64 {
    20
}

fn default_fetch_max_chars() -> usize {
    12_000
}

fn default_search_limit() -> usize {
    5
}

pub fn validate_tools_config(settings: &ToolsConfig) -> Result<()> {
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

fn validate_optional_header_value(value: Option<&str>, key: &str) -> Result<()> {
    if let Some(value) = value {
        if value.trim().is_empty() {
            bail!("{key} must not be empty");
        }
        let _ = HeaderValue::from_str(value)
            .with_context(|| format!("{key} must be a valid HTTP header value"))?;
    }
    Ok(())
}

fn validate_api_key_env(value: Option<&str>, key: &str) -> Result<()> {
    let Some(value) = value else {
        bail!("{key} is required");
    };
    if value.trim().is_empty() {
        bail!("{key} must not be empty");
    }
    Ok(())
}

fn validate_http_url_setting(value: Option<&str>, key: &str) -> Result<()> {
    let Some(value) = value else {
        bail!("{key} is required");
    };
    validate_web_url(value)
        .map(|_| ())
        .map_err(|error| anyhow::anyhow!("{key}: {error}"))
}

fn validate_optional_http_url(value: Option<&str>, key: &str) -> Result<()> {
    if let Some(value) = value {
        validate_web_url(value)
            .map(|_| ())
            .map_err(|error| anyhow::anyhow!("{key}: {error}"))?;
    }
    Ok(())
}

fn configured_search_providers(settings: &WebSearchToolSettings) -> Result<Vec<WebSearchProvider>> {
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

fn build_http_client(timeout_seconds: u64) -> Result<Client, ToolError> {
    Client::builder()
        .redirect(Policy::limited(10))
        .timeout(Duration::from_secs(timeout_seconds))
        .build()
        .map_err(|e| ToolError::ExecutionError(format!("Failed to build HTTP client: {e}")))
}

fn validate_web_url(value: &str) -> Result<Url, ToolError> {
    let url =
        Url::parse(value).map_err(|e| ToolError::InvalidParams(format!("Invalid URL: {e}")))?;
    match url.scheme() {
        "http" | "https" => Ok(url),
        other => Err(ToolError::InvalidParams(format!(
            "Unsupported URL scheme '{other}'; expected http or https"
        ))),
    }
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    let mut iter = text.chars();
    let out = iter.by_ref().take(max_chars).collect::<String>();
    if iter.next().is_some() {
        format!("{out}...")
    } else {
        out
    }
}

fn extract_html_title(document: &Html) -> Option<String> {
    let selector = Selector::parse("title").ok()?;
    let title = document
        .select(&selector)
        .next()?
        .text()
        .collect::<Vec<_>>()
        .join(" ");
    let title = collapse_whitespace(&title);
    if title.is_empty() { None } else { Some(title) }
}

fn extract_html_text(html: &str) -> String {
    let document = Html::parse_document(html);
    let selector = Selector::parse("body").expect("valid selector");
    let body_text = document
        .select(&selector)
        .next()
        .map(|body| body.text().collect::<Vec<_>>().join(" "))
        .unwrap_or_else(|| document.root_element().text().collect::<Vec<_>>().join(" "));
    collapse_whitespace(&body_text)
}

fn decode_duckduckgo_result_url(raw: &str) -> String {
    let normalized = if raw.starts_with("//") {
        format!("https:{raw}")
    } else if raw.starts_with('/') {
        format!("https://duckduckgo.com{raw}")
    } else {
        raw.to_string()
    };

    let Ok(url) = Url::parse(&normalized) else {
        return normalized;
    };

    let is_duckduckgo = url
        .host_str()
        .is_some_and(|host| host.eq_ignore_ascii_case("duckduckgo.com"));
    if !is_duckduckgo {
        return normalized;
    }

    for (key, value) in url.query_pairs() {
        if key == "uddg" {
            return value.into_owned();
        }
    }

    normalized
}

fn parse_duckduckgo_lite_results(html: &str, limit: usize) -> Vec<WebSearchHit> {
    let document = Html::parse_document(html);
    let title_selector = Selector::parse("a.result-link").expect("valid selector");
    let snippet_selector = Selector::parse("td.result-snippet").expect("valid selector");

    let titles = document
        .select(&title_selector)
        .map(|node| {
            let title = collapse_whitespace(&node.text().collect::<Vec<_>>().join(" "));
            let url = node.value().attr("href").unwrap_or_default().to_string();
            (title, decode_duckduckgo_result_url(&url))
        })
        .filter(|(title, url)| !title.is_empty() && !url.is_empty())
        .collect::<Vec<_>>();

    let snippets = document
        .select(&snippet_selector)
        .map(|node| collapse_whitespace(&node.text().collect::<Vec<_>>().join(" ")))
        .collect::<Vec<_>>();

    titles
        .into_iter()
        .enumerate()
        .take(limit)
        .map(|(index, (title, url))| WebSearchHit {
            title,
            url,
            snippet: snippets.get(index).cloned().unwrap_or_default(),
        })
        .collect()
}

fn normalize_hits_for_output(hits: &[WebSearchHit]) -> Vec<Value> {
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

fn header_value(value: &str, label: &str) -> Result<HeaderValue, ToolError> {
    HeaderValue::from_str(value)
        .map_err(|e| ToolError::ExecutionError(format!("Invalid header value for {label}: {e}")))
}

fn fetch_user_agent(settings: &WebFetchToolSettings) -> &str {
    settings
        .user_agent
        .as_deref()
        .unwrap_or(DEFAULT_BROWSER_USER_AGENT)
}

fn search_user_agent(settings: &WebSearchToolSettings) -> &str {
    settings
        .user_agent
        .as_deref()
        .unwrap_or(DEFAULT_BROWSER_USER_AGENT)
}

fn apply_browser_headers(
    request: RequestBuilder,
    user_agent: &str,
    accept: Option<&str>,
    accept_language: Option<&str>,
    accept_encoding: Option<&str>,
) -> Result<RequestBuilder, ToolError> {
    let request = request.header(USER_AGENT, header_value(user_agent, "User-Agent")?);
    let request = request.header(
        ACCEPT,
        header_value(accept.unwrap_or(DEFAULT_BROWSER_ACCEPT), "Accept")?,
    );
    let request = request.header(
        ACCEPT_LANGUAGE,
        header_value(
            accept_language.unwrap_or(DEFAULT_BROWSER_ACCEPT_LANGUAGE),
            "Accept-Language",
        )?,
    );
    let request = request.header(
        ACCEPT_ENCODING,
        header_value(
            accept_encoding.unwrap_or(DEFAULT_BROWSER_ACCEPT_ENCODING),
            "Accept-Encoding",
        )?,
    );
    Ok(request)
}

fn apply_fetch_headers(
    request: RequestBuilder,
    settings: &WebFetchToolSettings,
) -> Result<RequestBuilder, ToolError> {
    apply_browser_headers(
        request,
        fetch_user_agent(settings),
        settings.accept.as_deref(),
        settings.accept_language.as_deref(),
        settings.accept_encoding.as_deref(),
    )
}

fn apply_html_search_headers(
    request: RequestBuilder,
    settings: &WebSearchToolSettings,
) -> Result<RequestBuilder, ToolError> {
    apply_browser_headers(request, search_user_agent(settings), None, None, None)
}

fn apply_api_headers(
    request: RequestBuilder,
    settings: &WebSearchToolSettings,
) -> Result<RequestBuilder, ToolError> {
    let request = request.header(
        USER_AGENT,
        header_value(search_user_agent(settings), "User-Agent")?,
    );
    let request = request.header(ACCEPT, HeaderValue::from_static("application/json"));
    Ok(request)
}

fn brave_hits_from_response(response: BraveSearchResponse, limit: usize) -> Vec<WebSearchHit> {
    response
        .web
        .map(|web| {
            web.results
                .into_iter()
                .take(limit)
                .filter_map(|result| {
                    if result.title.trim().is_empty() || result.url.trim().is_empty() {
                        return None;
                    }
                    let snippet = if !result.description.trim().is_empty() {
                        result.description
                    } else {
                        result.extra_snippets.into_iter().next().unwrap_or_default()
                    };
                    Some(WebSearchHit {
                        title: collapse_whitespace(&result.title),
                        url: result.url,
                        snippet: truncate_chars(&collapse_whitespace(&snippet), 400),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn tavily_hits_from_response(response: TavilySearchResponse, limit: usize) -> Vec<WebSearchHit> {
    response
        .results
        .into_iter()
        .take(limit)
        .filter_map(|result| {
            if result.title.trim().is_empty() || result.url.trim().is_empty() {
                return None;
            }
            Some(WebSearchHit {
                title: collapse_whitespace(&result.title),
                url: result.url,
                snippet: truncate_chars(&collapse_whitespace(&result.content), 400),
            })
        })
        .collect()
}

fn searxng_hits_from_response(response: SearxngSearchResponse, limit: usize) -> Vec<WebSearchHit> {
    response
        .results
        .into_iter()
        .take(limit)
        .filter_map(|result| {
            if result.title.trim().is_empty() || result.url.trim().is_empty() {
                return None;
            }
            Some(WebSearchHit {
                title: collapse_whitespace(&result.title),
                url: result.url,
                snippet: truncate_chars(&collapse_whitespace(&result.content), 400),
            })
        })
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

fn load_required_api_key(env_name: &str, provider: &str) -> Result<String, ToolError> {
    env::var(env_name).map_err(|_| {
        ToolError::ExecutionError(format!(
            "{provider} search requires environment variable '{}'",
            env_name
        ))
    })
}

async fn search_duckduckgo_html(
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

async fn search_brave(
    client: &Client,
    settings: &WebSearchToolSettings,
    query: &str,
    limit: usize,
) -> Result<Vec<WebSearchHit>, ToolError> {
    let env_name =
        settings.brave.api_key_env.as_deref().ok_or_else(|| {
            ToolError::ExecutionError("Brave search is not configured".to_string())
        })?;
    let api_key = load_required_api_key(env_name, "Brave")?;
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

async fn search_tavily(
    client: &Client,
    settings: &WebSearchToolSettings,
    query: &str,
    limit: usize,
) -> Result<Vec<WebSearchHit>, ToolError> {
    let env_name =
        settings.tavily.api_key_env.as_deref().ok_or_else(|| {
            ToolError::ExecutionError("Tavily search is not configured".to_string())
        })?;
    let api_key = load_required_api_key(env_name, "Tavily")?;
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

async fn search_searxng(
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

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch an HTTP or HTTPS URL and return readable page text. Use this instead of shell_exec for normal web retrieval."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "HTTP or HTTPS URL to fetch"
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Request timeout in seconds",
                    "default": 20
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum number of characters to return",
                    "default": 12000
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, params: Value, ctx: &ToolContext) -> Result<ToolEffect, ToolError> {
        let args: WebFetchArgs = parse_args(params)?;
        let url = validate_web_url(&args.url)?;
        let client = build_http_client(args.timeout_seconds)?;
        let request = apply_fetch_headers(client.get(url.clone()), &ctx.tools.web_fetch)?;
        let response = request
            .send()
            .await
            .map_err(|e| ToolError::ExecutionError(format!("web_fetch request failed: {e}")))?;
        let status = response.status();
        let final_url = response.url().to_string();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/octet-stream")
            .to_string();
        let body = response
            .bytes()
            .await
            .map_err(|e| ToolError::ExecutionError(format!("web_fetch body read failed: {e}")))?;
        let raw_body = String::from_utf8_lossy(&body).into_owned();
        let extracted = if content_type.contains("html") {
            extract_html_text(&raw_body)
        } else {
            collapse_whitespace(&raw_body)
        };
        let document = Html::parse_document(&raw_body);
        let title = if content_type.contains("html") {
            extract_html_title(&document)
        } else {
            None
        };
        let body_text = truncate_chars(&extracted, args.max_chars.max(200));
        let mut content = format!("Fetched {} {}", status.as_u16(), final_url);
        content.push_str(&format!("\nContent-Type: {}", content_type));
        if let Some(title) = &title {
            content.push_str(&format!("\nTitle: {}", title));
        }
        content.push_str("\n\n");
        content.push_str(&body_text);

        Ok(ToolEffect::Output(ToolOutput {
            content,
            metadata: serde_json::json!({
                "status": status.as_u16(),
                "final_url": final_url,
                "content_type": content_type,
                "title": title,
                "bytes": body.len(),
                "user_agent": fetch_user_agent(&ctx.tools.web_fetch),
            }),
        }))
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the public web and return top result titles, URLs, and snippets. Use this instead of shell_exec for normal discovery."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5
                },
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Request timeout in seconds",
                    "default": 20
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, params: Value, ctx: &ToolContext) -> Result<ToolEffect, ToolError> {
        let args: WebSearchArgs = parse_args(params)?;
        let query = args.query.trim();
        if query.is_empty() {
            return Err(ToolError::InvalidParams(
                "web_search query must not be empty".to_string(),
            ));
        }

        let limit = args.limit.clamp(1, 10);
        let client = build_http_client(args.timeout_seconds)?;
        let providers = configured_search_providers(&ctx.tools.web_search)
            .map_err(|e| ToolError::ExecutionError(e.to_string()))?;
        let mut errors = Vec::new();

        for provider in providers.iter().copied() {
            let result = match provider {
                WebSearchProvider::Brave => {
                    search_brave(&client, &ctx.tools.web_search, query, limit).await
                }
                WebSearchProvider::Tavily => {
                    search_tavily(&client, &ctx.tools.web_search, query, limit).await
                }
                WebSearchProvider::Searxng => {
                    search_searxng(&client, &ctx.tools.web_search, query, limit).await
                }
                WebSearchProvider::DuckDuckGoHtml => {
                    search_duckduckgo_html(&client, &ctx.tools.web_search, query, limit).await
                }
            };

            match result {
                Ok(hits) => {
                    let mut content = if hits.is_empty() {
                        format!(
                            "No web search results found for '{}' via {}.",
                            query,
                            provider.as_str()
                        )
                    } else {
                        format!("Top web results for '{}' via {}:", query, provider.as_str())
                    };
                    for (index, hit) in hits.iter().enumerate() {
                        content.push_str(&format!(
                            "\n{}. {}\n   URL: {}\n   Snippet: {}",
                            index + 1,
                            hit.title,
                            hit.url,
                            hit.snippet
                        ));
                    }

                    return Ok(ToolEffect::Output(ToolOutput {
                        content,
                        metadata: serde_json::json!({
                            "query": query,
                            "results": normalize_hits_for_output(&hits),
                            "provider": provider.as_str(),
                            "attempted_providers": providers.iter().map(|value| value.as_str()).collect::<Vec<_>>(),
                        }),
                    }));
                }
                Err(error) => {
                    errors.push(format!("{}: {}", provider.as_str(), error));
                }
            }
        }

        Err(ToolError::ExecutionError(format!(
            "web_search failed across providers: {}",
            errors.join(" | ")
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duckduckgo_redirects_decode_to_final_urls() {
        let raw = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdocs%3Fx%3D1&rut=abc";
        assert_eq!(
            decode_duckduckgo_result_url(raw),
            "https://example.com/docs?x=1"
        );
    }

    #[test]
    fn lite_results_parse_titles_urls_and_snippets() {
        let html = r#"
        <html><body><table>
          <tr><td><a class='result-link' href='//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa'>Alpha Result</a></td></tr>
          <tr><td class='result-snippet'>First snippet.</td></tr>
          <tr><td><a class='result-link' href='//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fb'>Beta Result</a></td></tr>
          <tr><td class='result-snippet'>Second snippet.</td></tr>
        </table></body></html>
        "#;
        let hits = parse_duckduckgo_lite_results(html, 5);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].title, "Alpha Result");
        assert_eq!(hits[0].url, "https://example.com/a");
        assert_eq!(hits[0].snippet, "First snippet.");
        assert_eq!(hits[1].title, "Beta Result");
    }

    #[test]
    fn html_text_extraction_collapses_noise() {
        let html = r#"
        <html>
          <head><title>Example</title></head>
          <body>
            <h1>Hello</h1>
            <p>World</p>
          </body>
        </html>
        "#;
        assert_eq!(
            extract_html_title(&Html::parse_document(html)).as_deref(),
            Some("Example")
        );
        assert_eq!(extract_html_text(html), "Hello World");
    }

    #[test]
    fn brave_results_parse_titles_urls_and_snippets() {
        let response: BraveSearchResponse = serde_json::from_value(serde_json::json!({
            "web": {
                "results": [
                    {
                        "title": "Alpha",
                        "url": "https://example.com/a",
                        "description": "Alpha snippet"
                    }
                ]
            }
        }))
        .unwrap();
        let hits = brave_hits_from_response(response, 5);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].title, "Alpha");
        assert_eq!(hits[0].url, "https://example.com/a");
        assert_eq!(hits[0].snippet, "Alpha snippet");
    }

    #[test]
    fn tavily_results_parse_titles_urls_and_snippets() {
        let response: TavilySearchResponse = serde_json::from_value(serde_json::json!({
            "results": [
                {
                    "title": "Beta",
                    "url": "https://example.com/b",
                    "content": "Beta snippet"
                }
            ]
        }))
        .unwrap();
        let hits = tavily_hits_from_response(response, 5);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].title, "Beta");
        assert_eq!(hits[0].url, "https://example.com/b");
        assert_eq!(hits[0].snippet, "Beta snippet");
    }

    #[test]
    fn searxng_results_parse_titles_urls_and_snippets() {
        let response: SearxngSearchResponse = serde_json::from_value(serde_json::json!({
            "results": [
                {
                    "title": "Gamma",
                    "url": "https://example.com/c",
                    "content": "Gamma snippet"
                }
            ]
        }))
        .unwrap();
        let hits = searxng_hits_from_response(response, 5);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].title, "Gamma");
        assert_eq!(hits[0].url, "https://example.com/c");
        assert_eq!(hits[0].snippet, "Gamma snippet");
    }

    #[test]
    fn validate_tools_config_rejects_unknown_search_provider() {
        let mut settings = ToolsConfig::default();
        settings.web_search.providers = Some(vec!["unknown".to_string()]);
        let err = validate_tools_config(&settings).unwrap_err();
        assert!(err.to_string().contains("unknown web_search provider"));
    }

    #[test]
    fn validate_tools_config_requires_tavily_api_key_env_when_selected() {
        let mut settings = ToolsConfig::default();
        settings.web_search.providers = Some(vec!["tavily".to_string()]);
        let err = validate_tools_config(&settings).unwrap_err();
        assert!(
            err.to_string()
                .contains("tools.web_search.tavily.api_key_env")
        );
    }
}
