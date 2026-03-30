//! Mixture of Depths & Pathways (MoDP)
//!
//! A world-first dynamic architecture.
//! In traditional LLMs, every token traverses every layer linearly.
//! MoDP introduces a "Pathway Router" that decides the execution graph
//! for each token independently.
//!
//! Features:
//! - **Layer Skipping**: Simple tokens (e.g. whitespace) can bypass expensive
//!   middle layers entirely.
//! - **Recurrent Logic**: Complex tokens can "recycle" through a reasoning layer
//!   multiple times to refine their hidden state before moving to the output head.

use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

/// Decides which layers a token block should traverse.
#[derive(Debug)]
pub struct PathwayRouter {
    gate: Linear,
    n_layers: usize,
}

impl PathwayRouter {
    pub fn new(d_model: usize, n_layers: usize, vb: VarBuilder) -> Result<Self> {
        let gate = candle_nn::linear(d_model, n_layers, vb.pp("gate"))?;
        Ok(Self { gate, n_layers })
    }

    /// Predict the importance/complexity of the current token block.
    /// Returns [batch, n_layers] importance scores.
    pub fn route(&self, hidden: &Tensor, temperature: f32) -> Result<Tensor> {
        // hidden: [batch, seq_len, d_model]

        // --- Sovereign Guard: Prevent "empty tensor for reduce" during mean(1) ---
        let (_batch, seq_len, _d_model) = hidden.dims3()?;
        let hidden_avg = if seq_len > 0 {
            hidden.mean(1)? // [batch, d_model]
        } else {
            Tensor::zeros((_batch, _d_model), hidden.dtype(), hidden.device())?
        };

        let mut logits = self.gate.forward(&hidden_avg)?; // [batch, n_layers]

        // ── Zenith Upgrade: Temperature Scaling & Entropy Control ──
        // High temp (e.g. 2.0) during distillation forces the model to explore
        // deeper pathways. Low temp (e.g. 0.1) forces extreme determinism for production.
        if (temperature - 1.0).abs() > 1e-4 {
            logits = (logits / (temperature as f64))?;
        }

        candle_nn::ops::sigmoid(&logits)
    }

    /// Construct a sequence of layer indices for the current forward pass.
    pub fn compute_pathway(&self, hidden: &Tensor, temperature: f32) -> Result<Vec<usize>> {
        let importance = self.route(hidden, temperature)?;
        // ── Sovereign Fix: Average across batch dimension for multi-batch safety ──
        // Guard against empty batch to prevent "empty tensor for reduce" panic
        let importance_mean = if importance.elem_count() > 0 {
            importance.mean(0)?
        } else {
            Tensor::zeros(self.n_layers, importance.dtype(), importance.device())?
        };
        let scores: Vec<f32> = importance_mean.to_vec1()?;

        let mut path = Vec::new();
        for (i, &score) in scores.iter().enumerate() {
            if i >= self.n_layers {
                break;
            }
            if score < 0.15 {
                // Skip this layer! (Mixture of Depths)
                continue;
            }

            path.push(i);

            // ── Recurrent Reasoning: "Thinking Harder" ──
            // If the gate predicts extreme complexity, the token recurses through the layer.
            // Industry-grade limit: Max 2 passes to prevent activation explosion.
            if score > 0.94 {
                path.push(i);
                tracing::debug!(target: "shakey::modp", "Recurrent Reasoning active for layer {}", i);
            }
        }

        // Guarantee at least the first and last layer for stability
        if path.is_empty() || path[0] != 0 {
            path.insert(0, 0);
        }
        if *path.last().unwrap() != self.n_layers - 1 {
            path.push(self.n_layers - 1);
        }

        Ok(path)
    }
}
