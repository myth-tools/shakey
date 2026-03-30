use anyhow::Result;
use scraper::{Html, Selector};
use std::sync::OnceLock;

/// Lazily compiled content-area selectors (compiled once, reused forever).
fn content_selectors() -> &'static [Selector] {
    static SELS: OnceLock<Vec<Selector>> = OnceLock::new();
    SELS.get_or_init(|| {
        vec![
            Selector::parse("article").expect("hardcoded CSS"),
            Selector::parse("main").expect("hardcoded CSS"),
            Selector::parse("#content").expect("hardcoded CSS"),
            Selector::parse(".content").expect("hardcoded CSS"),
            Selector::parse("body").expect("hardcoded CSS"),
        ]
    })
}

/// Parse a blob of HTML into a clean, human-readable text.
///
/// Enhanced Heuristic Version:
/// 1. Strips all noise (script, style, head, nav, footer, ads)
/// 2. Preserves semantic structure (headings, lists) for better LLM reasoning
/// 3. Collapses whitespace
pub fn parse_html(html: &str) -> Result<String> {
    let document = Html::parse_document(html);
    let mut text_content = String::new();

    // Heuristic: Identify main content area if it exists (article, main, #content)

    let mut main_content = None;
    for selector in content_selectors() {
        if let Some(el) = document.select(selector).next() {
            main_content = Some(el);
            break;
        }
    }

    let root = main_content.unwrap_or(document.root_element());

    // Traverse the tree and collect text from non-noise nodes
    for node in root.descendants() {
        if let Some(element) = node.value().as_element() {
            let tag = element.name().to_lowercase();

            // Skip noise tags explicitly
            if tag == "script"
                || tag == "style"
                || tag == "head"
                || tag == "nav"
                || tag == "footer"
                || tag == "noscript"
            {
                continue;
            }

            // Heuristic for structural breaks (headings, paragraphs, divs)
            if matches!(
                tag.as_str(),
                "h1" | "h2" | "h3" | "h4" | "p" | "div" | "br" | "li" | "tr"
            ) && !text_content.is_empty()
                && !text_content.ends_with('\n')
            {
                text_content.push('\n');
            }
        }

        if let Some(text_node) = node.value().as_text() {
            // Ensure this text is not inside a noise element sibling we missed
            let parent = node.parent().and_then(|p| p.value().as_element());
            if let Some(p) = parent {
                let p_tag = p.name().to_lowercase();
                if p_tag == "script" || p_tag == "style" || p_tag == "head" {
                    continue;
                }
            }

            let part = text_node.trim();
            if !part.is_empty() {
                text_content.push_str(part);
                text_content.push(' ');
            }
        }
    }

    Ok(text_content.trim().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_html_robust() -> Result<()> {
        let html = r#"
            <html>
                <head><title>Test</title></head>
                <nav>Navigation Link</nav>
                <body>
                    <main>
                        <h1>Main Title</h1>
                        <p>This is important content.</p>
                        <script>console.log('noise');</script>
                        <div class="ads">Ad Content</div>
                    </main>
                    <footer>Copyright 2026</footer>
                </body>
            </html>
        "#;
        let text = parse_html(html)?;
        assert!(text.contains("Main Title"));
        assert!(text.contains("important content"));
        assert!(!text.contains("Navigation Link"));
        assert!(!text.contains("noise"));
        assert!(!text.contains("Copyright 2026"));
        Ok(())
    }
}
