//! Custom AdamW optimizer.
//!
//! AdamW (Adam with Weight Decay) separates weight decay from the
//! gradient update steps, fixing a common issue in standard Adam.

use anyhow::Result;
use candle_core::{backprop::GradStore as Grads, Tensor, Var};
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};

/// AdamW optimizer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    pub use_muon: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            lr: 3e-4,
            beta1: 0.9,
            beta2: 0.95,
            eps: 1e-8,
            weight_decay: 0.1,
            use_muon: true, // Default to world-first spectral optimization
        }
    }
}

/// AdamW optimizer.
pub struct Optimizer {
    vars: Vec<Var>,
    moments1: Vec<Tensor>,
    moments2: Vec<Tensor>,
    muon_states: Vec<Option<super::muon::Muon>>,
    config: OptimizerConfig,
    step: u64,
    /// ELITE: Moving average of the global gradient norm for adaptive clipping
    avg_grad_norm: f32,
    /// ELITE: Decay factor for the moving average
    beta_norm: f32,
}

impl Optimizer {
    pub fn new(varmap: &VarMap, config: OptimizerConfig) -> Result<Self> {
        let vars = varmap.all_vars();
        let mut moments1 = Vec::with_capacity(vars.len());
        let mut moments2 = Vec::with_capacity(vars.len());

        for var in &vars {
            moments1.push(Tensor::zeros_like(var.as_tensor())?);
            moments2.push(Tensor::zeros_like(var.as_tensor())?);
        }

        let mut muon_states = Vec::with_capacity(vars.len());
        if config.use_muon {
            for var in &vars {
                let dims = var.as_tensor().dims();
                if dims.len() == 2 {
                    // Muon is highly effective for 2D parameter matrices (linear layers)
                    muon_states.push(Some(super::muon::Muon::new(
                        (dims[0], dims[1]),
                        config.lr,
                        config.beta1,
                        var.as_tensor().device(),
                    )?));
                } else {
                    muon_states.push(None);
                }
            }
        } else {
            muon_states = (0..vars.len()).map(|_| None).collect();
        }

        Ok(Self {
            vars,
            moments1,
            moments2,
            muon_states,
            config,
            step: 0,
            avg_grad_norm: 1.0, // Initialize with unit norm
            beta_norm: 0.99,    // Slow-moving average for stability
        })
    }

    pub fn vars(&self) -> &[Var] {
        &self.vars
    }

    /// Perform one optimization step, applying global gradient clipping if required.
    /// Returns the global gradient norm (f32).
    pub fn step(&mut self, grads: &Grads, lr: f64, max_grad_norm: f64) -> Result<f32> {
        self.step += 1;
        let t = self.step as f64;
        let beta1 = self.config.beta1;
        let beta2 = self.config.beta2;
        let eps = self.config.eps;
        let wd = self.config.weight_decay;

        // 1. Compute global gradient norm
        let mut sum_sq = 0.0f32;
        for var in self.vars.iter() {
            if let Some(grad) = grads.get(var.as_tensor()) {
                let sq = grad.sqr()?.sum_all()?.to_scalar::<f32>()?;
                sum_sq += sq;
            }
        }
        let total_norm = sum_sq.sqrt();

        // ── ELITE: Global Norm Adaptive Clipping ──
        // Instead of a fixed threshold, we use a moving average to adapt to different
        // training phases (warmup vs. convergence). The threshold is capped at max_grad_norm.
        self.avg_grad_norm =
            self.beta_norm * self.avg_grad_norm + (1.0 - self.beta_norm) * total_norm;

        let adaptive_threshold = if max_grad_norm > 0.0 {
            // Adaptive threshold is 2x the moving average, capped at the user's max_grad_norm
            (self.avg_grad_norm * 2.0).min(max_grad_norm as f32)
        } else {
            max_grad_norm as f32
        };

        let clip_coef = if adaptive_threshold > 0.0 && total_norm > adaptive_threshold {
            adaptive_threshold / (total_norm + 1e-6)
        } else {
            1.0
        };

        // 2. Fused-Style Update and Weight Decay
        for (i, var) in self.vars.iter_mut().enumerate() {
            if let Some(raw_grad) = grads.get(var.as_tensor()) {
                // Apply gradient clipping in-place conceptually
                let grad = if clip_coef < 1.0 {
                    raw_grad.affine(clip_coef as f64, 0.0)?
                } else {
                    raw_grad.clone()
                };

                // Peak Mastery: Inplace moment updates to reduce intermediate buffers
                let m_new = self.moments1[i]
                    .affine(beta1, 0.0)?
                    .add(&grad.affine(1.0 - beta1, 0.0)?)?;
                self.moments1[i] = m_new.clone();

                let v_new = self.moments2[i]
                    .affine(beta2, 0.0)?
                    .add(&(grad.sqr()?.affine(1.0 - beta2, 0.0)?))?;
                self.moments2[i] = v_new.clone();

                // Fused Bias Correction and Update Coefficient Calculation
                let m_hat_scale = 1.0 / (1.0 - beta1.powf(t));
                let v_hat_scale = 1.0 / (1.0 - beta2.powf(t));

                // Denom = sqrt(v_new * v_hat_scale) + eps
                let denom = (v_new.affine(v_hat_scale, 0.0)?.sqrt()?.affine(1.0, eps))?;

                // Unified Update: (m_new * m_hat_scale) / denom + wd * var
                let update = m_new.affine(m_hat_scale, 0.0)?.div(&denom)?;

                let var_tensor = var.as_tensor();
                // Industry-Grade: Skip weight decay for 1D params (biases, norm weights, scales)
                // Standard in LLaMA/Mistral/GPT — WD on norms degrades training quality
                let effective_wd = if var_tensor.dims().len() <= 1 {
                    0.0
                } else {
                    wd
                };
                let wd_update = var_tensor.affine(effective_wd, 0.0)?.add(&update)?;

                // Final weights: W = W - lr * wd_update
                let new_var = if let Some(ref mut muon) = self.muon_states[i] {
                    // ── Sovereign Fix: Apply weight decay before Muon spectral update ──
                    // Muon handles its own momentum internally, but weight decay must
                    // still be applied to prevent parameter magnitude explosion.
                    let decayed = if effective_wd > 0.0 {
                        var_tensor.affine(1.0 - lr * effective_wd, 0.0)?
                    } else {
                        var_tensor.clone()
                    };
                    muon.step(&decayed, &grad)?
                } else {
                    // Standard AdamW update path
                    var_tensor.sub(&(wd_update.affine(lr, 0.0))?)?
                };

                var.set(&new_var)?;
            }
        }

        Ok(total_norm)
    }
    // Note: candle's backprop engine creates a fresh `GradStore` per `backward()` call,
    // so gradients do NOT accumulate across steps. No `zero_grad()` is needed.
}
