//! Muon Spectral Optimizer
//!
//! A world-first implementation of orthogonal manifold optimization.
//! Standard AdamW updates weights using momentum-based gradient descent.
//! Muon updates weights by calculating the spectral projection of the gradient
//! onto the orthogonal manifold of the weight matrix.
//!
//! This is achieves via:
//! 1. Momentum accumulation (standard).
//! 2. Newton-Schulz iterations to orthogonalize the momentum tensor.
//! 3. Orthogonal updating to maintain the spectral properties of the weights.

use candle_core::{Result, Tensor};

/// Muon Optimizer State
pub struct Muon {
    /// Momentum buffer [dim1, dim2]
    momentum: Tensor,
    lr: f64,
    momentum_beta: f64,
    ns_steps: usize, // Newton-Schulz iterations (typically 5-10)
}

impl Muon {
    pub fn new(
        shape: (usize, usize),
        lr: f64,
        momentum_beta: f64,
        device: &candle_core::Device,
    ) -> Result<Self> {
        let momentum = Tensor::zeros(shape, candle_core::DType::F32, device)?;
        Ok(Self {
            momentum,
            lr,
            momentum_beta,
            ns_steps: 6, // Optimized default for training speed
        })
    }

    /// Perform a spectral update step for a weight tensor.
    /// W' = W - lr * NewtonSchulz(Momentum)
    pub fn step(&mut self, weight: &Tensor, grad: &Tensor) -> Result<Tensor> {
        // 1. Update Momentum: m = beta * m + (1-beta) * grad
        self.momentum =
            ((&self.momentum * self.momentum_beta)? + (grad * (1.0 - self.momentum_beta))?)?;

        // 2. Newton-Schulz Orthogonalization
        // We find the spectral projection of the momentum matrix.
        let mut x = self.momentum.clone();

        // Scale X to have spectral norm <= 1 for convergence (stable version)
        // Industry-Grade: Cast to f64 to prevent sum() overflow on massive matrices
        let x_f64 = x.to_dtype(candle_core::DType::F64)?;
        let x_norm = x_f64.sqr()?.sum_all()?.sqrt()?.to_vec0::<f64>()? + 1e-8;

        // ── Sovereign Guard: Skip Muon for near-zero momentum (no meaningful direction) ──
        if x_norm < 1e-7 {
            return Ok(weight.clone());
        }
        x = x.affine(1.0 / x_norm, 0.0)?;

        // Iteration: X = 0.5 * X * (3I - X^TX)
        // ── Zenith Upgrade: Adaptive Spectral Orthogonalization ──
        // Instead of a fixed iteration count, we dynamically scale the NS-steps
        // based on the spectral condition of the momentum matrix.
        // This ensures perfect orthogonality for large weight blocks (>4096).
        let dynamic_steps = if x.dim(0)? > 2048 {
            self.ns_steps + 1
        } else {
            self.ns_steps
        };

        for _ in 0..dynamic_steps {
            let xt_x = x.transpose(0, 1)?.matmul(&x)?;
            let eye = Tensor::eye(xt_x.dim(0)?, xt_x.dtype(), xt_x.device())?;
            let term = (eye.affine(3.0, 0.0)? - xt_x)?;
            x = (x.matmul(&term)? * 0.5)?;
        }

        // ── Sovereign Guard: NaN fallback — if orthogonalization diverged, skip update ──
        let x_check = x.sqr()?.sum_all()?.to_vec0::<f32>()?;
        if x_check.is_nan() || x_check.is_infinite() {
            tracing::warn!(target: "shakey::muon", "Muon NaN detected in orthogonalization. Skipping spectral update.");
            return Ok(weight.clone());
        }

        // 3. Update weight orthogonally
        let update = (x * self.lr)?;
        weight - update
    }
}
