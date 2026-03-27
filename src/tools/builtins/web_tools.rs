use std::time::Duration;

use async_trait::async_trait;
use reqwest::redirect::Policy;
use scraper::{Html, Selector};
use serde::Deserialize;
use serde_json::Value;
use url::Url;

use crate::tools::{Tool, ToolContext, ToolEffect, ToolError, ToolOutput, parse_args};

pub struct WebFetchTool;
pub struct WebSearchTool;

const DEFAULT_USER_AGENT: &str = "Turin/0.26.0 (+https://github.com/jthum/turin)";

#[derive(Debug, Clone, PartialEq, Eq)]
struct WebSearchHit {
    title: String,
    url: String,
    snippet: String,
}

#[derive(Deserialize)]
struct WebFetchArgs {
    url: String,
    #[serde(default = "default_fetch_timeout_secs")]
    timeout_secs: u64,
    #[serde(default = "default_fetch_max_chars")]
    max_chars: usize,
}

#[derive(Deserialize)]
struct WebSearchArgs {
    query: String,
    #[serde(default = "default_search_limit")]
    limit: usize,
    #[serde(default = "default_fetch_timeout_secs")]
    timeout_secs: u64,
}

fn default_fetch_timeout_secs() -> u64 {
    20
}

fn default_fetch_max_chars() -> usize {
    12_000
}

fn default_search_limit() -> usize {
    5
}

fn build_http_client(timeout_secs: u64) -> Result<reqwest::Client, ToolError> {
    reqwest::Client::builder()
        .user_agent(DEFAULT_USER_AGENT)
        .redirect(Policy::limited(10))
        .timeout(Duration::from_secs(timeout_secs))
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
                "timeout_secs": {
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

    async fn execute(&self, params: Value, _ctx: &ToolContext) -> Result<ToolEffect, ToolError> {
        let args: WebFetchArgs = parse_args(params)?;
        let url = validate_web_url(&args.url)?;
        let client = build_http_client(args.timeout_secs)?;
        let response = client
            .get(url.clone())
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
                "timeout_secs": {
                    "type": "integer",
                    "description": "Request timeout in seconds",
                    "default": 20
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, params: Value, _ctx: &ToolContext) -> Result<ToolEffect, ToolError> {
        let args: WebSearchArgs = parse_args(params)?;
        let query = args.query.trim();
        if query.is_empty() {
            return Err(ToolError::InvalidParams(
                "web_search query must not be empty".to_string(),
            ));
        }

        let limit = args.limit.clamp(1, 10);
        let client = build_http_client(args.timeout_secs)?;
        let response = client
            .get("https://lite.duckduckgo.com/lite/")
            .query(&[("q", query)])
            .send()
            .await
            .map_err(|e| ToolError::ExecutionError(format!("web_search request failed: {e}")))?;
        let html = response
            .text()
            .await
            .map_err(|e| ToolError::ExecutionError(format!("web_search body read failed: {e}")))?;
        let hits = parse_duckduckgo_lite_results(&html, limit);

        let mut content = if hits.is_empty() {
            format!("No web search results found for '{}'.", query)
        } else {
            format!("Top web results for '{}':", query)
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

        Ok(ToolEffect::Output(ToolOutput {
            content,
            metadata: serde_json::json!({
                "query": query,
                "results": hits.iter().map(|hit| serde_json::json!({
                    "title": hit.title,
                    "url": hit.url,
                    "snippet": hit.snippet,
                })).collect::<Vec<_>>(),
                "provider": "duckduckgo_lite",
            }),
        }))
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
}
