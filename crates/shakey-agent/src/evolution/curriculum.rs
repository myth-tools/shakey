//! Curriculum planner — decides what the agent should learn next.
//!
//! # Sovereign Mastery: Intrinsic Interest Engine
//!
//! Unlike static curricula, the Zenith Engine calculates 'Epistemic Uncertainty'
//! and 'Growth Potential' to dynamically steer the agent's focus toward the
//! "Sweet Spot" of learning — where the challenge is maximized by current capability.

use crate::CapabilityMatrix;

#[derive(Debug, Clone, Default)]
pub struct CurriculumPlanner;

impl CurriculumPlanner {
    /// Suggest the next area for improvement using Sovereign Interest logic.
    pub fn suggest_next_focus(&self, caps: &CapabilityMatrix, stage: &str) -> String {
        let areas = [
            ("language_understanding", caps.language_understanding, 1.0),
            ("instruction_following", caps.instruction_following, 1.2), // High priority early
            ("code_generation", caps.code_generation, 1.5),             // High synergy
            ("math_reasoning", caps.math_reasoning, 1.3),
            ("planning", caps.planning, 1.4),
            ("tool_use", caps.tool_use, 1.6), // Peak mastery target
            ("visual_intelligence", caps.visual_intelligence, 1.1),
            ("auditory_processing", caps.auditory_processing, 1.0),
            ("meta_cognition", caps.meta_cognition, 2.0), // Ultimate evolution
        ];

        // Zenith Scoring: Weight * (1.0 - current_score) * Stage_Multiplier
        let mut best_area = "general".to_string();
        let mut max_interest = -1.0;

        for (name, score, weight) in areas {
            let stage_multiplier = self.get_stage_multiplier(name, stage);
            let interest = weight * (1.0 - score) * stage_multiplier;

            if interest > max_interest {
                max_interest = interest;
                best_area = name.to_string();
            }
        }

        tracing::info!(target: "shakey::evolution", "Zenith Engine selected focus: {} (Interest score: {:.4})", best_area, max_interest);
        best_area
    }

    /// Calculate stage-specific multipliers to favor certain capabilities in certain eras.
    fn get_stage_multiplier(&self, name: &str, stage: &str) -> f64 {
        match (stage, name) {
            ("seed", "language_understanding") => 2.0,
            ("seed", "instruction_following") => 1.8,
            ("sprout", "code_generation") => 1.5,
            ("sapling", "planning") | ("sapling", "math_reasoning") => 1.7,
            ("tree", "tool_use") | ("tree", "meta_cognition") => 2.0,
            ("forest", "meta_cognition") => 3.0,
            _ => 1.0,
        }
    }

    /// Return the target number of tokens for a given phase based on the model stage.
    pub fn token_budget(&self, stage: &str) -> u64 {
        match stage {
            "seed" => 250_000, // Increased for Zenith 4.0
            "sprout" => 1_000_000,
            "sapling" => 5_000_000,
            "tree" => 25_000_000,
            "forest" => 100_000_000,
            _ => 50_000,
        }
    }

    /// Identify the absolute weakest area (Fallback logic).
    pub fn find_weakest_area(&self, caps: &CapabilityMatrix) -> String {
        let areas = [
            ("language_understanding", caps.language_understanding),
            ("code_generation", caps.code_generation),
            ("math_reasoning", caps.math_reasoning),
            ("instruction_following", caps.instruction_following),
            ("planning", caps.planning),
            ("tool_use", caps.tool_use),
            ("meta_cognition", caps.meta_cognition),
        ];

        areas
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Greater))
            .map(|(name, _): &(&str, f64)| name.to_string())
            .unwrap_or_else(|| "general".into())
    }
}
