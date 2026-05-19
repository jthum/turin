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
