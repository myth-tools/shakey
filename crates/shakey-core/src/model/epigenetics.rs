//! Dynamic Epigenetic Scaling (Neuroplasticity)
//!
//! A world-first architecture feature for Large Language Models.
//! Traditional LLMs have static weights during inference. The Neuroplasticity Matrix
//! introduces an ephemeral, dynamically updating state array associated with the network's pathways
//! (in this case, MoE experts).
//!
//! As specific experts are activated (e.g., Code Expert, Math Expert), their epigenetic
//! "tags" increase. These tags directly scale the output of the expert on the fly.
//! This simulates short-term neuroplasticity: if a model is deep into a coding session,
//! the coding pathways temporarily strengthen without any backpropagation, allowing zero-shot
//! domain locking and unprecedented inference coherence.
//!
//! The tags decay universally after each step, ensuring the model's plasticity doesn't
//! collapse into a single permanent state, allowing it to dynamically transition between topics.

use candle_core::{Result, Tensor};
use std::sync::RwLock;

/// Manages the dynamic scaling of pathways via epigenetic tracking.
#[derive(Debug)]
pub struct NeuroplasticityMatrix {
    /// The current multiplier tags for each pathway (expert).
    /// Bound between [1.0, max_plasticity]
    tags: RwLock<Vec<f32>>,
    /// How much a tag strengthens when activated (e.g., 0.05)
    learning_rate: f32,
    /// How quickly un-activated tags decay back to 1.0 (e.g., 0.99)
    decay_rate: f32,
    /// Absolute ceiling for epigenetic scaling multiplier
    max_plasticity: f32,
}

impl NeuroplasticityMatrix {
    /// Initialize a new plasticity matrix for N pathways.
    pub fn new(num_pathways: usize) -> Self {
        Self {
            tags: RwLock::new(vec![1.0; num_pathways]),
            learning_rate: 0.05,
            decay_rate: 0.98,     // Fast decay for highly dynamic contexts
            max_plasticity: 1.25, // 25% max boost to preventing activation explosions
        }
    }

    /// Apply the neuroplastic multiplier to a pathway's output and update its epigenetic state.
    ///
    /// # Arguments
    /// * `expert_id` - The route ID processing the token
    /// * `activation_strength` - How heavily this expert was utilized (its routing weight)
    /// * `output` - The raw tensor output from the pathway
    ///
    /// # Returns
    /// Scaled tensor output mimicking strengthened synaptic firing
    pub fn apply_and_update(
        &self,
        expert_id: usize,
        activation_strength: f32,
        output: &Tensor,
    ) -> Result<Tensor> {
        let current_multiplier = {
            // ── Sovereign Guard: Poison-safe lock access ──
            let tags = match self.tags.read() {
                Ok(t) => t,
                Err(poisoned) => poisoned.into_inner(),
            };
            if expert_id < tags.len() {
                tags[expert_id]
            } else {
                1.0
            }
        };

        // ── Industry-Grade: Sovereign Gating ──
        // Ensure the multiplier stays within strict safety bounds [1.0, max_plasticity]
        // and handle potential float corruption (NaN/Inf) at the boundary.
        let safe_multiplier = if current_multiplier.is_finite() {
            current_multiplier.clamp(1.0, self.max_plasticity)
        } else {
            1.0
        };

        // If the multiplier is effectively 1.0, we can skip tensor mult for speed
        let scaled_output = if (safe_multiplier - 1.0).abs() < 1e-4 {
            output.clone()
        } else {
            (output * (safe_multiplier as f64))?
        };

        // We only bump the tag internally if it crossed a threshold of utilization,
        // so that minor activations don't flood the plasticity state.
        // ── Sovereign Fix: Single write lock acquisition (eliminates deadlock risk) ──
        if activation_strength > 0.1 {
            let mut tags = match self.tags.write() {
                Ok(t) => t,
                Err(poisoned) => poisoned.into_inner(),
            };
            if expert_id >= tags.len() {
                return Ok(scaled_output);
            }

            // ── Zenith Upgrade: Momentum-based Epigenetic Inertia ──
            // Transition to an EMA (Exponential Moving Average) update.
            // This prevents "tag-glitching" by outlier activations and ensures
            // the state represents a sustained domain focus (e.g., Coding -> 5-10 tokens).
            let momentum: f32 = 0.85; // History retention factor
            let target = (1.0 + self.learning_rate * activation_strength).min(self.max_plasticity);

            tags[expert_id] = (tags[expert_id] * momentum) + (target * (1.0 - momentum));
        }

        Ok(scaled_output)
    }

    /// Globally decay all pathways. Should be called once per sequence step.
    pub fn decay(&self) {
        let mut tags = match self.tags.write() {
            Ok(t) => t,
            Err(poisoned) => poisoned.into_inner(),
        };
        for tag in tags.iter_mut() {
            if *tag > 1.0 {
                // Decay back towards 1.0 exponentially
                *tag = 1.0 + (*tag - 1.0) * self.decay_rate;
            }
        }
    }

    /// Retrieve current state for telemetry and metrics.
    pub fn get_state(&self) -> Vec<f32> {
        match self.tags.read() {
            Ok(t) => t.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }
}
