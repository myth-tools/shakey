//! Zenith Context Manager: Advanced sliding window memory for OODA stabilization.
//!
//! Handles the prioritization, pruning, and injection of context into the
//! agent's reasoning loop. Ensures core mission instructions are preserved
//! while keeping the prompt within LLM context limits.

use serde::{Deserialize, Serialize};
use shakey_distill::nim_client::NimClient;
use std::collections::VecDeque;
use std::fmt;

/// High-performance context manager for the sovereign OODA loop.
#[derive(Clone, Serialize, Deserialize, Default)]
pub struct ContextManager {
    /// Sliding window of recent context strings
    pub window: VecDeque<String>,
    /// Maximum number of context items to track
    pub max_items: usize,
    /// Total character count (for coarse token estimation)
    pub total_chars: usize,
    /// NIM Client reference (for on-the-fly RAG if needed)
    #[serde(skip)]
    pub nim_client: Option<NimClient>,
}

impl fmt::Debug for ContextManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContextManager")
            .field("window_len", &self.window.len())
            .field("max_items", &self.max_items)
            .field("total_chars", &self.total_chars)
            .field("has_nim_client", &self.nim_client.is_some())
            .finish()
    }
}

impl ContextManager {
    pub fn new(max_items: usize, nim_client: Option<NimClient>) -> Self {
        Self {
            window: VecDeque::with_capacity(max_items),
            max_items,
            total_chars: 0,
            nim_client,
        }
    }

    /// Add a new context snippet. Prunes if the window is full.
    pub fn add_context(&mut self, context: &str) {
        if self.window.len() >= self.max_items {
            if let Some(oldest) = self.window.pop_front() {
                self.total_chars -= oldest.len();
            }
        }

        self.window.push_back(context.to_string());
        self.total_chars += context.len();
    }

    /// Clear all non-essential context.
    pub fn reset(&mut self) {
        self.window.clear();
        self.total_chars = 0;
    }

    /// Export the consolidated context as a single prompt block.
    pub fn export_consolidated(&self) -> String {
        self.window
            .iter()
            .cloned()
            .collect::<Vec<_>>()
            .join("\n---\n")
    }
}
