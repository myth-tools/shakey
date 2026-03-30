//! Shared utilities for the Shakey Distill pipeline.

/// Robustly extract JSON from mixed-text/markdown LLM responses.
/// Handles markdown code fences (```json ... ```) and raw JSON.
pub fn extract_json(response: &str) -> String {
    let trimmed = response.trim();
    // ── Sovereign Fix: Strip markdown code fences first ──
    let content = if let Some(fence_start) = trimmed.find("```") {
        let after_fence = &trimmed[fence_start + 3..];
        // Skip optional language specifier (e.g., "json", "JSON")
        let content = if after_fence.starts_with("json") || after_fence.starts_with("JSON") {
            &after_fence[4..]
        } else {
            after_fence
        };
        if let Some(fence_end) = content.find("```") {
            content[..fence_end].trim()
        } else {
            content.trim()
        }
    } else {
        trimmed
    };
    if let Some(start) = content.find('{') {
        if let Some(end) = content.rfind('}') {
            return content[start..=end].to_string();
        }
    }
    content.to_string()
}

/// Extract a numeric score from a model output string.
pub fn parse_score(score_str: &str) -> f32 {
    score_str
        .chars()
        .filter(|c| c.is_ascii_digit() || *c == '.')
        .collect::<String>()
        .parse::<f32>()
        .unwrap_or(0.0)
}
