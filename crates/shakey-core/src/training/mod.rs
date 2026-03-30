//! Training infrastructure.
//!
//! - `checkpoint` — Atomic checkpoint/resume system (zero-loss on Colab/Kaggle)
//! - `trainer` — Training loop with gradient accumulation
//! - `distillation` — Knowledge distillation loss functions

pub mod capabilities;
pub mod checkpoint;
pub mod dataloader;
pub mod distillation;
pub mod muon;
pub mod optimizer;
pub mod replay_buffer;
pub mod scheduler;
pub mod trainer;

use anyhow::Result;
use async_trait::async_trait;

/// Trait for the "Sovereign Critic" which provides semantic feedback for Autopoietic Loss modulation.
#[async_trait]
pub trait SovereignCritic: Send + Sync {
    /// Evaluate the logical and semantic quality of a batch's reasoning.
    async fn evaluate_score(&self, prompt: &str, completion: &str) -> Result<f32>;
}
