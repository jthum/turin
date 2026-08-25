use std::time::Duration;

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use futures::StreamExt;
use reqwest::header::{ACCEPT, ACCEPT_ENCODING, ACCEPT_LANGUAGE, HeaderValue, USER_AGENT};
use reqwest::redirect::Policy;
use reqwest::{Client, RequestBuilder};
use serde::Deserialize;
use serde_json::Value;
use turin_types::{ToolsConfig, WebFetchToolSettings, WebSearchToolSettings};
use url::Url;

use crate::tools::{Tool, ToolContext, ToolEffect, ToolError, ToolOutput, parse_args};

mod html;
mod search;

use html::{collapse_whitespace, extract_html_text, extract_html_title, truncate_chars};
use search::{
    WebSearchProvider, configured_search_providers, normalize_hits_for_output, search_brave,
    search_duckduckgo_html, search_searxng, search_tavily,
};

pub struct WebFetchTool;
pub struct WebSearchTool;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct WebSearchHit {
    pub(super) title: String,
    pub(super) url: String,
    pub(super) snippet: String,
}

const DEFAULT_BROWSER_USER_AGENT: &str = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36";
const DEFAULT_BROWSER_ACCEPT: &str =
    "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8";
const DEFAULT_BROWSER_ACCEPT_LANGUAGE: &str = "en-US,en;q=0.5";
const DEFAULT_BROWSER_ACCEPT_ENCODING: &str = "gzip, deflate, br";
const DEFAULT_FETCH_MAX_RESPONSE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Deserialize)]
struct WebFetchArgs {
    url: String,
    #[serde(default = "default_fetch_timeout_seconds")]
    timeout_seconds: u64,
    #[serde(default = "default_fetch_max_chars")]
    max_chars: usize,
    #[serde(default)]
    max_bytes: Option<usize>,
}

#[derive(Deserialize)]
struct WebSearchArgs {
    query: String,
    #[serde(default = "default_search_limit")]
    limit: usize,
    #[serde(default = "default_fetch_timeout_seconds")]
    timeout_seconds: u64,
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
    if settings.web_fetch.max_response_bytes == Some(0) {
        bail!("tools.web_fetch.max_response_bytes must be greater than zero");
    }
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

fn clamp_timeout_seconds(timeout_seconds: u64) -> u64 {
    timeout_seconds.min(300)
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

fn configured_fetch_max_response_bytes(settings: &WebFetchToolSettings) -> usize {
    settings
        .max_response_bytes
        .unwrap_or(DEFAULT_FETCH_MAX_RESPONSE_BYTES)
}

fn effective_fetch_max_response_bytes(
    args: &WebFetchArgs,
    settings: &WebFetchToolSettings,
) -> usize {
    let configured = configured_fetch_max_response_bytes(settings);
    args.max_bytes.unwrap_or(configured).clamp(1, configured)
}

fn append_bounded_body_chunk(body: &mut Vec<u8>, chunk: &[u8], max_bytes: usize) -> bool {
    let remaining = max_bytes.saturating_sub(body.len());
    if chunk.len() > remaining {
        body.extend_from_slice(&chunk[..remaining]);
        return true;
    }
    body.extend_from_slice(chunk);
    false
}

async fn read_bounded_response_body(
    response: reqwest::Response,
    max_bytes: usize,
) -> Result<(Vec<u8>, bool), ToolError> {
    let initial_capacity = response
        .content_length()
        .and_then(|length| usize::try_from(length).ok())
        .unwrap_or(0)
        .min(max_bytes);
    let mut body = Vec::with_capacity(initial_capacity);
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|error| {
            ToolError::ExecutionError(format!("web_fetch body read failed: {error}"))
        })?;
        if append_bounded_body_chunk(&mut body, &chunk, max_bytes) {
            return Ok((body, true));
        }
    }
    Ok((body, false))
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
                },
                "max_bytes": {
                    "type": "integer",
                    "description": "Optional response read limit; cannot exceed the configured web_fetch ceiling",
                    "minimum": 1
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, params: Value, ctx: &ToolContext) -> Result<ToolEffect, ToolError> {
        let args: WebFetchArgs = parse_args(params)?;
        let url = validate_web_url(&args.url)?;
        let client = build_http_client(clamp_timeout_seconds(args.timeout_seconds))?;
        let request = apply_fetch_headers(client.get(url.clone()), &ctx.tools.web_fetch)?;
        let response = request
            .send()
            .await
            .map_err(|e| ToolError::ExecutionError(format!("web_fetch request failed: {e}")))?;
        let status = response.status();
        let final_url = response.url().to_string();
        let content_length = response.content_length();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("application/octet-stream")
            .to_string();
        let max_response_bytes = effective_fetch_max_response_bytes(&args, &ctx.tools.web_fetch);
        let (body, response_truncated) =
            read_bounded_response_body(response, max_response_bytes).await?;
        let raw_body = String::from_utf8_lossy(&body).into_owned();
        let extracted = if content_type.contains("html") {
            extract_html_text(&raw_body)
        } else {
            collapse_whitespace(&raw_body)
        };
        let title = if content_type.contains("html") {
            let document = scraper::Html::parse_document(&raw_body);
            extract_html_title(&document)
        } else {
            None
        };
        let body_text = truncate_chars(&extracted, args.max_chars.max(200));
        let mut content = format!("Fetched {} {}", status.as_u16(), final_url);
        content.push_str(&format!("\nContent-Type: {}", content_type));
        if response_truncated {
            content.push_str(&format!(
                "\nResponse-Truncated: true (read limit: {max_response_bytes} bytes)"
            ));
        }
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
                "content_length": content_length,
                "max_response_bytes": max_response_bytes,
                "response_truncated": response_truncated,
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
        let client = build_http_client(clamp_timeout_seconds(args.timeout_seconds))?;
        let providers = configured_search_providers(&ctx.tools.web_search)
            .map_err(|e| ToolError::ExecutionError(e.to_string()))?;
        let mut errors = Vec::new();

        for provider in providers.iter().copied() {
            let result = match provider {
                WebSearchProvider::Brave => {
                    search_brave(
                        &client,
                        &ctx.tools.web_search,
                        ctx.config.as_deref(),
                        query,
                        limit,
                    )
                    .await
                }
                WebSearchProvider::Tavily => {
                    search_tavily(
                        &client,
                        &ctx.tools.web_search,
                        ctx.config.as_deref(),
                        query,
                        limit,
                    )
                    .await
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
#[path = "tests/web_tools.rs"]
mod tests;
