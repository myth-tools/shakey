//! Memory Consolidation — synthesizing episodic experiences into long-term knowledge.
//!
//! Dual-cognitive approach: Uses LLM summarization for semantic synthesis.

use crate::CycleRecord;
use anyhow::Result;
use shakey_distill::nim_client::NimClient;

pub struct MemoryConsolidator {
    nim_client: NimClient,
}

impl MemoryConsolidator {
    pub fn new(nim_client: NimClient) -> Self {
        Self { nim_client }
    }

    /// Consolidate a series of cycle records into a high-level "Lesson Learned".
    pub async fn consolidate_episodes(&self, episodes: &[CycleRecord]) -> Result<String> {
        if episodes.is_empty() {
            return Ok("No episodes to consolidate.".to_string());
        }

        let mut context =
            String::from("Recently, the agent completed the following evolution cycles:\n\n");
        for ep in episodes {
            context.push_str(&format!(
                "- Cycle {}: Strategy={}, Success={}, Notes={}\n",
                ep.cycle_id, ep.strategy, ep.success, ep.notes
            ));
        }

        let prompt = format!(
            "{}\nBased on these cycles, synthesize a single, high-level factual 'Lesson Learned' or 'Sovereign Knowledge' fact. \
            Format it as a concise, structured statement that will help the agent's future decision making. \
            Avoid placeholders. Be world-class and industry-grade in your reasoning.",
            context
        );

        // Query teacher model for high-fidelity synthesis
        let lesson = self
            .nim_client
            .query_for_role(
                shakey_distill::teacher::TeacherRole::Summarizer,
                "cognition",
                &prompt,
            )
            .await
            .unwrap_or_else(|_| {
                // Safe fallback if the dynamic role query fails
                "Memory Consolidation: Learning remains stable.".to_string()
            });

        Ok(lesson.trim().to_string())
    }
}
