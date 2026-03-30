use anyhow::Result;
use regex::Regex;
use reqwest::Client;
use scraper::{Html, Selector};
use serde::Serialize;
use std::sync::OnceLock;

#[derive(Debug, Serialize, Clone)]
pub struct SearchResult {
    pub title: String,
    pub link: String,
    pub snippet: String,
}

// Lazily compiled CSS selectors — compiled once, reused forever
fn title_selector() -> &'static Selector {
    static SEL: OnceLock<Selector> = OnceLock::new();
    SEL.get_or_init(|| Selector::parse("a.result-link").expect("hardcoded CSS selector"))
}
fn snippet_selector() -> &'static Selector {
    static SEL: OnceLock<Selector> = OnceLock::new();
    SEL.get_or_init(|| Selector::parse("td.result-snippet").expect("hardcoded CSS selector"))
}
fn fallback_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r#"href="([^"]*?duckduckgo\.com/l/[^"]*?)""#).expect("hardcoded regex")
    })
}

/// Search DuckDuckGo (Lite) for a query and return snippets.
///
/// DuckDuckGo Lite is significantly more robust against anti-bot
/// measures and has a much more stable HTML structure than Google.
pub async fn search_duckduckgo(query: &str) -> Result<String> {
    let client = Client::builder()
        .user_agent("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        .build()?;

    let url = format!(
        "https://lite.duckduckgo.com/lite/?q={}",
        urlencoding::encode(query)
    );

    let res = client.get(&url).send().await?;
    let body = res.text().await?;

    let document = Html::parse_document(&body);
    let mut results = Vec::new();

    for (title_el, snippet_el) in document
        .select(title_selector())
        .zip(document.select(snippet_selector()))
    {
        let title = title_el.text().collect::<String>().trim().to_string();
        let link = title_el
            .value()
            .attr("href")
            .unwrap_or_default()
            .to_string();
        let snippet = snippet_el.text().collect::<String>().trim().to_string();

        if !title.is_empty() && !link.is_empty() {
            results.push(SearchResult {
                title,
                link,
                snippet,
            });
        }
    }

    // Heuristic Fallback: If Lite fails, try regex extraction
    if results.is_empty() {
        tracing::debug!("DDG Lite scraper yielded 0 results. Falling back to regex...");
        for cap in fallback_regex().captures_iter(&body).take(5) {
            results.push(SearchResult {
                title: "Extracted via DDG Link Heuristic".into(),
                link: cap[1].to_string(),
                snippet: "No snippet available".into(),
            });
        }
    }

    Ok(serde_json::to_string_pretty(&results)?)
}
