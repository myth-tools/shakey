use serde::{Deserialize, Serialize};

/// Current capabilities of the agent (self-assessed).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CapabilityMatrix {
    /// Language understanding quality (0.0 - 1.0)
    pub language_understanding: f64,
    /// Code generation quality (0.0 - 1.0)
    pub code_generation: f64,
    /// Mathematical reasoning quality (0.0 - 1.0)
    pub math_reasoning: f64,
    /// Instruction following accuracy (0.0 - 1.0)
    pub instruction_following: f64,
    /// Multi-step planning ability (0.0 - 1.0)
    pub planning: f64,
    /// Tool use proficiency (0.0 - 1.0)
    pub tool_use: f64,
    /// Visual analysis & image generation (0.0 - 1.0)
    pub visual_intelligence: f64,
    /// Auditory & Signal processing (0.0 - 1.0)
    pub auditory_processing: f64,
    /// 3D Geometry & Spatial Synthesis (0.0 - 1.0)
    pub three_d_synthesis: f64,

    // ── ZENITH 5.2: SOVEREIGN MASTERY ──
    /// Cybersecurity & Robustness (0.0 - 1.0)
    pub cybersecurity_resilience: f64,
    /// Operating System Kernel & Terminal Mastery (0.0 - 1.0)
    pub os_terminal_mastery: f64,
    /// Penetration Testing & Red-Team Tools (0.0 - 1.0)
    pub penetration_testing: f64,
    /// Agentic Autonomy & Self-Correction (0.0 - 1.0)
    pub agentic_autonomy: f64,

    /// Alignment, Ethics & Robustness (0.0 - 1.0)
    pub safety: f64,
    /// Self-assessment accuracy (0.0 - 1.0)
    pub meta_cognition: f64,
    /// Overall composite score
    pub overall: f64,
}

impl Default for CapabilityMatrix {
    fn default() -> Self {
        Self {
            language_understanding: 0.0,
            code_generation: 0.0,
            math_reasoning: 0.0,
            instruction_following: 0.0,
            planning: 0.0,
            tool_use: 0.0,
            visual_intelligence: 0.0,
            auditory_processing: 0.0,
            three_d_synthesis: 0.0,
            cybersecurity_resilience: 0.0,
            os_terminal_mastery: 0.0,
            penetration_testing: 0.0,
            agentic_autonomy: 0.0,
            safety: 0.0,
            meta_cognition: 0.0,
            overall: 0.0,
        }
    }
}

impl CapabilityMatrix {
    /// Restrict all capabilities to their valid [0.0, 1.0] range.
    pub fn clamp(&mut self) {
        self.language_understanding = self.language_understanding.clamp(0.0, 1.0);
        self.code_generation = self.code_generation.clamp(0.0, 1.0);
        self.math_reasoning = self.math_reasoning.clamp(0.0, 1.0);
        self.instruction_following = self.instruction_following.clamp(0.0, 1.0);
        self.planning = self.planning.clamp(0.0, 1.0);
        self.tool_use = self.tool_use.clamp(0.0, 1.0);
        self.visual_intelligence = self.visual_intelligence.clamp(0.0, 1.0);
        self.auditory_processing = self.auditory_processing.clamp(0.0, 1.0);
        self.three_d_synthesis = self.three_d_synthesis.clamp(0.0, 1.0);
        self.cybersecurity_resilience = self.cybersecurity_resilience.clamp(0.0, 1.0);
        self.os_terminal_mastery = self.os_terminal_mastery.clamp(0.0, 1.0);
        self.penetration_testing = self.penetration_testing.clamp(0.0, 1.0);
        self.agentic_autonomy = self.agentic_autonomy.clamp(0.0, 1.0);
        self.safety = self.safety.clamp(0.0, 1.0);
        self.meta_cognition = self.meta_cognition.clamp(0.0, 1.0);
        self.overall = self.overall.clamp(0.0, 1.0);
    }

    /// Compute overall score as weighted average.
    pub fn compute_overall(&mut self) {
        // Sovereign-Grade Recalibration: Weights sum to exactly 1.00
        self.overall = self.language_understanding * 0.06
            + self.code_generation * 0.10
            + self.math_reasoning * 0.07
            + self.instruction_following * 0.07
            + self.planning * 0.07
            + self.tool_use * 0.07
            + self.visual_intelligence * 0.07
            + self.auditory_processing * 0.04
            + self.three_d_synthesis * 0.04
            + self.cybersecurity_resilience * 0.10
            + self.os_terminal_mastery * 0.07
            + self.penetration_testing * 0.10
            + self.agentic_autonomy * 0.07
            + self.safety * 0.04
            + self.meta_cognition * 0.03;
        // Total: 0.06+0.10+0.07+0.07+0.07+0.07+0.07+0.04+0.04+0.10+0.07+0.10+0.07+0.04+0.03 = 1.00
    }
}
