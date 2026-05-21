use scraper::{Html, Selector};
use url::Url;

use super::WebSearchHit;

pub(super) fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub(super) fn truncate_chars(text: &str, max_chars: usize) -> String {
    let mut iter = text.chars();
    let out = iter.by_ref().take(max_chars).collect::<String>();
    if iter.next().is_some() {
        format!("{out}...")
    } else {
        out
    }
}

pub(super) fn extract_html_title(document: &Html) -> Option<String> {
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

pub(super) fn extract_html_text(html: &str) -> String {
    let document = Html::parse_document(html);
    let selector = Selector::parse("body").expect("valid selector");
    let body_text = document
        .select(&selector)
        .next()
        .map(|body| body.text().collect::<Vec<_>>().join(" "))
        .unwrap_or_else(|| document.root_element().text().collect::<Vec<_>>().join(" "));
    collapse_whitespace(&body_text)
}

pub(super) fn decode_duckduckgo_result_url(raw: &str) -> String {
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

pub(super) fn parse_duckduckgo_lite_results(html: &str, limit: usize) -> Vec<WebSearchHit> {
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
