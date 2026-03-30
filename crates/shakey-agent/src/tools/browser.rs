use anyhow::Result;
use chromiumoxide::browser::{Browser, BrowserConfig};
use futures::StreamExt;
use std::sync::Arc;
use tokio::sync::{Mutex, OnceCell};

/// Global Lightpanda browser instance (if available).
static BROWSER: OnceCell<Arc<Mutex<Browser>>> = OnceCell::const_new();

/// Connect to the Lightpanda (or Chrome) process via CDP.
///
/// Ultimate Robustness Logic:
/// 1. Attempt to connect to an ALREADY RUNNING Lightpanda at localhost:9222
/// 2. If no process is running, launch a new local browser as a fallback.
pub async fn get_browser() -> Result<Arc<Mutex<Browser>>> {
    let instance = BROWSER.get_or_try_init(|| async {
        // Attempt CDP Connection first (Port 9222)
        match Browser::connect("ws://127.0.0.1:9222").await {
            Ok((browser, mut handler)) => {
                tracing::info!("ULTIMATE ROBUSTNESS: Connected to existing Lightpanda at 127.0.0.1:9222");
                tokio::spawn(async move {
                    while let Some(h) = handler.next().await {
                        if let Err(e) = h {
                            tracing::error!("CDP Handler Error: {}", e);
                            break;
                        }
                    }
                });
                Ok(Arc::new(Mutex::new(browser)))
            }
            Err(_) => {
                tracing::warn!("No running Lightpanda found at 9222. Launching local fallback...");
                let config = match BrowserConfig::builder()
                    .no_sandbox()
                    .window_size(1920, 1080)
                    .build()
                {
                    Ok(c) => c,
                    Err(e) => {
                        tracing::error!("Failed to build browser config: {}. Retrying with minimal config.", e);
                        // Minimal config — no_sandbox only.
                        // Convert String error to anyhow::Error
                        BrowserConfig::builder()
                            .no_sandbox()
                            .build()
                            .map_err(|e2| anyhow::anyhow!("Minimal browser config also failed: {}", e2))?
                    }
                };

                match Browser::launch(config).await {
                    Ok((browser, mut handler)) => {
                        tokio::spawn(async move {
                            while let Some(h) = handler.next().await {
                                if let Err(e) = h {
                                    tracing::error!("CDP Handler Error: {}", e);
                                    break;
                                }
                            }
                        });
                        Ok(Arc::new(Mutex::new(browser)))
                    }
                    Err(e) => {
                        tracing::error!("Failed to launch fallback browser: {}. Browser tools will be unavailable.", e);
                        // Return an error instead of panicking.
                        Err(anyhow::anyhow!("FATAL: No browser available. Install chromium or start Lightpanda on port 9222. Error: {}", e))
                    }
                }
            }
        }
    }).await?;

    Ok(Arc::clone(instance))
}

/// Dynamic Web Fetcher: Uses Lightpanda to render JavaScript.
pub async fn fetch_dynamic(url: &str) -> Result<String> {
    let browser = get_browser().await?;
    let browser = browser.lock().await;

    // Create a new tab (incognito-like)
    let page = browser.new_page(url).await?;

    // Wait until JS reports the page is loaded (networkidle0 equivalent)
    let _ = page.wait_for_navigation().await;

    // Additional wait for dynamic scripts (SPAs)
    tokio::time::sleep(std::time::Duration::from_millis(1000)).await;

    // Get the fully rendered HTML content
    let html = page.content().await?;

    // Close the page to free up Lightpanda resources
    page.close().await?;

    Ok(html)
}
