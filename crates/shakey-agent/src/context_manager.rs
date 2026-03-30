use anyhow::Result;
use crate::nim_client::{ChatMessage, NimClient, ChatRequest};

/// Context Manager: The "Zenith" of memory management.
/// Handles sliding-window truncation and summarization to stay within model limits.
pub struct ContextManager {
    max_tokens: u32,
    nim: Option<NimClient>,
}

impl ContextManager {
    pub fn new(max_tokens: u32, nim: Option<NimClient>) -> Self {
        Self { max_tokens, nim }
    }

    /// Prune history to stay within token limits.
    /// Simple heuristic: if over limit, drop oldest non-system messages.
    pub fn prune(&self, messages: &mut Vec<ChatMessage>) -> Result<()> {
        let estimated_tokens: u32 = messages.iter().map(|m| {
            match &m.content {
                crate::nim_client::MessageContent::Text(t) => t.len() as u32 / 4,
                _ => 100,
            }
        }).sum();

        if estimated_tokens > self.max_tokens {
            tracing::warn!(target: "shakey::context", "Context window pressure: {}/{} tokens. Pruning...", estimated_tokens, self.max_tokens);

            // Keep the system prompt (index 0 usually)
            let system = if !messages.is_empty() && messages[0].role == "system" {
                Some(messages.remove(0))
            } else { None };

            // Remove 20% of oldest history
            let to_remove = (messages.len() as f64 * 0.2) as usize;
            for _ in 0..to_remove.max(1) {
                if !messages.is_empty() {
                    messages.remove(0);
                }
            }

            if let Some(s) = system {
                messages.insert(0, s);
            }
        }
        Ok(())
    }

    /// Summarize old history into a single distilled memory.
    pub async fn summarize(&self, history: &[ChatMessage]) -> Result<String> {
        if let Some(nim) = &self.nim {
            let mut prompt = String::from("Summarize the following conversation history into a concise 'Sovereign Memory' block:\n\n");
            for m in history {
                prompt.push_str(&format!("{}: {:?}\n", m.role, m.content));
            }

            let res = nim.query_for_role(
                shakey_distill::teacher::TeacherRole::Summarizer,
                "system",
                &prompt,
                512,
                0.2
            ).await?;
            Ok(res)
        } else {
            Ok("Summarization skipped: No NIM client available.".into())
        }
    }
}
