//! Evolution engine — manages model growth and version control.
//!
//! Handles:
//! - A/B comparison between model checkpoints
//! - Commit/rollback decisions
//! - Progressive architecture scaling (seed → sprout → sapling → tree → forest)

pub mod benchmark;
pub mod curriculum;

use crate::CapabilityMatrix;
use serde::{Deserialize, Serialize};

/// Model version record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersion {
    pub version_id: String,
    pub stage: String,
    pub step: u64,
    pub capabilities: CapabilityMatrix,
    pub checkpoint_path: String,
    pub created_at: String,
    pub is_active: bool,
}

/// Evolution controller — decides when to grow or rollback.
pub struct EvolutionController {
    /// All tracked model versions
    pub versions: Vec<ModelVersion>,
    /// Maximum rollback depth
    pub max_rollback: usize,
    /// Minimum improvement to commit a new version
    pub min_improvement: f64,
}

impl EvolutionController {
    pub fn new(max_rollback: usize, min_improvement: f64) -> Self {
        Self {
            versions: Vec::new(),
            max_rollback,
            min_improvement,
        }
    }

    /// Compare two capability matrices and decide whether to commit.
    ///
    /// Returns (should_commit, improvement_delta).
    pub fn should_commit(&self, old: &CapabilityMatrix, new: &CapabilityMatrix) -> (bool, f64) {
        let delta = new.overall - old.overall;
        let commit = delta >= self.min_improvement;

        if commit {
            tracing::info!(
                "Improvement detected: {:.4} → {:.4} (Δ={:.4}) — committing",
                old.overall,
                new.overall,
                delta
            );
        } else {
            tracing::info!(
                "Insufficient improvement: {:.4} → {:.4} (Δ={:.4}, min={:.4}) — rolling back",
                old.overall,
                new.overall,
                delta,
                self.min_improvement
            );
        }

        (commit, delta)
    }

    /// Register a new model version.
    pub fn register_version(&mut self, version: ModelVersion) {
        // Deactivate all previous versions
        for v in &mut self.versions {
            v.is_active = false;
        }
        self.versions.push(version);

        // Trim to max_rollback depth
        while self.versions.len() > self.max_rollback {
            self.versions.remove(0);
        }
    }

    /// Get the currently active version.
    pub fn active_version(&self) -> Option<&ModelVersion> {
        self.versions.iter().rfind(|v| v.is_active)
    }

    /// Check if the model should be promoted to the next architecture stage.
    pub fn should_promote(
        &self,
        current_stage: &str,
        capabilities: &CapabilityMatrix,
    ) -> Option<String> {
        let threshold = match current_stage {
            "seed" => 0.3,    // Promote to sprout when >30% capable
            "sprout" => 0.5,  // Promote to sapling when >50%
            "sapling" => 0.7, // Promote to tree when >70%
            "tree" => 0.85,   // Promote to forest when >85%
            _ => return None,
        };

        if capabilities.overall >= threshold {
            let next_stage = match current_stage {
                "seed" => "sprout",
                "sprout" => "sapling",
                "sapling" => "tree",
                "tree" => "forest",
                _ => return None,
            };
            tracing::info!(
                "Model ready for promotion: {} → {} (score={:.2} >= {:.2})",
                current_stage,
                next_stage,
                capabilities.overall,
                threshold
            );
            Some(next_stage.into())
        } else {
            None
        }
    }
}
