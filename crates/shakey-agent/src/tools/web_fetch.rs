use super::browser;
use anyhow::Result;
use reqwest::Client;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot, OnceCell, Semaphore};

/// Message type for the web fetcher actor.
pub enum FetchMessage {
    FetchUrl {
        url: String,
        responder: oneshot::Sender<Result<String>>,
    },
}

static FETCHER: OnceCell<mpsc::Sender<FetchMessage>> = OnceCell::const_new();

/// Fetch the content of a URL with tiered performance & robustness.
///
/// 1. Tries Fast Static Layer (reqwest)
/// 2. If it detects dynamic content (JS-only), it triggers Dynamic Tier (Lightpanda)
pub async fn fetch_url(url: &str) -> Result<String> {
    let tx = FETCHER.get_or_init(|| async {
        let (tx, mut rx) = mpsc::channel::<FetchMessage>(1024);

        let max_concurrent = 20;
        let semaphore = Arc::new(Semaphore::new(max_concurrent));

        tokio::spawn(async move {
            let client = Client::builder()
                .user_agent("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 ShakeyBot/0.1")
                .timeout(std::time::Duration::from_secs(30))
                .pool_max_idle_per_host(5)
                .tcp_keepalive(Some(std::time::Duration::from_secs(60)))
                .build()
                .unwrap_or_default();

            let client = Arc::new(client);

            while let Some(msg) = rx.recv().await {
                match msg {
                    FetchMessage::FetchUrl { url, responder } => {
                        let client = Arc::clone(&client);
                        let sem = Arc::clone(&semaphore);

                        tokio::spawn(async move {
                            let _permit = sem.acquire().await.ok();

                            // TIER 1: FAST STATIC (reqwest)
                            let res = client.get(&url).send().await;
                            let outcome = match res {
                                Ok(r) => {
                                    if r.status().is_success() {
                                        let text = r.text().await.unwrap_or_default();

                                        // Robust Heuristic: Check if page is empty or says "JavaScript required"
                                        if is_javascript_required(&text) {
                                            tracing::info!("TIER 1 DETECTED JS REQUIREMENT. PROMOTING TO TIER 2 (DYNAMIC)...");

                                            // TIER 2: DEEP DYNAMIC (Lightpanda)
                                            match browser::fetch_dynamic(&url).await {
                                                Ok(dynamic) => Ok(dynamic),
                                                Err(e) => Err(anyhow::anyhow!("Tier 2 (Dynamic) failed for {}: {}", url, e)),
                                            }
                                        } else {
                                            Ok(text)
                                        }
                                    } else {
                                        Err(anyhow::anyhow!("HTTP {} for {}", r.status(), url))
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!("Tier 1 network error: {}. Triggering Dynamic Fallback...", e);
                                    // Robust Fallback: Try TIER 2 even on Tier 1 network error
                                    browser::fetch_dynamic(&url).await
                                }
                            };
                            let _ = responder.send(outcome);
                        });
                    }
                }
            }
        });
        tx
    }).await;

    let (resp_tx, resp_rx) = oneshot::channel();
    tx.send(FetchMessage::FetchUrl {
        url: url.to_string(),
        responder: resp_tx,
    })
    .await
    .map_err(|_| anyhow::anyhow!("Web Fetcher actor channel broken"))?;

    resp_rx
        .await
        .map_err(|_| anyhow::anyhow!("Web Fetcher responder dropped"))?
}

/// Heuristic detector for JS-required pages.
fn is_javascript_required(html: &str) -> bool {
    let html_lower = html.to_lowercase();

    // Core anti-bot/JS-only markers
    html_lower.contains("enable javascript") ||
    html_lower.contains("javascript is required") ||
    html_lower.contains("js disabled") ||
    (html_lower.contains("<noscript>") && html_lower.len() < 2000) || // Short page with noscript is sus
    html_lower.contains("cloudflare") && html_lower.contains("checking your browser")
}
