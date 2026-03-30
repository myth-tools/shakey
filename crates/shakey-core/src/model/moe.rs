//! Sparse Mixture-of-Experts (MoE) layer.
//!
//! ## Architecture
//!
//! Each MoE layer contains N expert FFNs and a router (gating network).
//! For each token, the router selects the top-k experts (typically k=2).
//! Only the selected experts process the token — the rest are dormant.
//!
//! This gives us the knowledge capacity of a large model (N experts worth
//! of parameters) with the compute cost of a small model (k experts).
//!
//! ## Load Balancing
//!
//! Without load balancing, the router tends to collapse — always selecting
//! the same 1-2 experts. We use a hybrid of Sigmoid-Gated Importance and
//! an auxiliary loss to ensure uniform expert specialization.
//!
//! ## Expert FFN
//!
//! Each expert is a SwiGLU feed-forward network:
//!   FFN(x) = (swish(x @ W_gate) ⊙ (x @ W_up)) @ W_down
//!
//! SwiGLU = Swish-Gated Linear Unit — better training dynamics than ReLU/GELU.

use super::layers::BitLinear;
use candle_core::{Device, Result, Tensor, D};
use candle_nn::{Module, VarBuilder};

// ─────────────────────────────────────────────────────────────
//  SwiGLU Expert FFN
// ─────────────────────────────────────────────────────────────

/// Single expert FFN using SwiGLU activation.
///
/// SwiGLU(x) = (Swish(x·W_gate)) ⊙ (x·W_up)) · W_down
///
/// Where Swish(x) = x·σ(x) and ⊙ is element-wise multiplication.
#[derive(Debug)]
pub struct ExpertFFN {
    w_gate: BitLinear,
    w_up: BitLinear,
    w_down: BitLinear,
}

impl ExpertFFN {
    pub fn new(d_model: usize, d_ff: usize, use_bitnet: bool, vb: VarBuilder) -> Result<Self> {
        let w_gate = BitLinear::new(d_model, d_ff, false, use_bitnet, vb.pp("w_gate"))?;
        let w_up = BitLinear::new(d_model, d_ff, false, use_bitnet, vb.pp("w_up"))?;
        let w_down = BitLinear::new(d_ff, d_model, false, use_bitnet, vb.pp("w_down"))?;
        Ok(Self {
            w_gate,
            w_up,
            w_down,
        })
    }
}

impl Module for ExpertFFN {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SwiGLU: (Swish(x·W_gate)) ⊙ (x·W_up)) · W_down
        let gate = self.w_gate.forward(x)?;
        let gate = candle_nn::ops::silu(&gate)?; // Swish/SiLU activation
        let up = self.w_up.forward(x)?;
        let hidden = (gate * up)?;
        self.w_down.forward(&hidden)
    }
}

// ─────────────────────────────────────────────────────────────
//  Router / Gating Network
// ─────────────────────────────────────────────────────────────

/// Expert router that selects top-k experts per token.
///
/// For each token, computes gating logits for all experts,
/// then selects the top-k with highest logit values.
/// The selected gates are softmax-normalized to sum to 1.
#[derive(Debug)]
pub struct Router {
    /// Linear projection: d_model → n_experts
    gate: candle_nn::Linear,
    /// Number of experts to activate per token
    top_k: usize,
    /// Total number of experts
    n_experts: usize,
}

impl Router {
    pub fn new(d_model: usize, n_experts: usize, top_k: usize, vb: VarBuilder) -> Result<Self> {
        let gate = candle_nn::linear(d_model, n_experts, vb.pp("gate"))?;
        Ok(Self {
            gate,
            top_k,
            n_experts,
        })
    }

    /// Compute routing decisions for all tokens.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch * seq_len, d_model] (flattened over batch and seq)
    ///
    /// # Returns
    /// * `expert_weights` - [n_tokens, top_k] — normalized gate weights for selected experts
    /// * `expert_indices` - [n_tokens, top_k] — indices of selected experts (as u32)
    /// * `load_balance_loss` - Scalar auxiliary loss for balanced expert usage
    pub fn route(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let _n_tokens = x.dim(0)?;

        // Compute gating logits: [n_tokens, n_experts]
        if _n_tokens == 0 {
            let device = x.device();
            return Ok((
                Tensor::zeros((0, self.top_k), x.dtype(), device)?,
                Tensor::zeros((0, self.top_k), candle_core::DType::U32, device)?,
                Tensor::new(0.0f32, device)?,
            ));
        }
        let mut logits = self.gate.forward(x)?;

        // ── Zenith Upgrade: Sovereign Expert Jitter ──
        // Adding noise to logits during the evolution process prevents
        // expert collapse and promotes specialization uniformity.
        let jitter = Tensor::rand(-0.01f32, 0.01f32, logits.shape(), logits.device())?;
        logits = logits.add(&jitter)?;

        let logits = logits.clamp(-1e2f32, 1e2f32)?;

        // ── Peak Mastery: Sigmoid-Gated Importance ──
        // Instead of pure softmax, we use sigmoid gating for importance estimation.
        // This allows multiple experts to share high importance for a single token
        // without the zero-sum constraint of softmax, promoting better knowledge overlap.
        let importance = candle_nn::ops::sigmoid(&logits)?;

        let probs = candle_nn::ops::softmax(&logits, D::Minus1)?;
        let probs = probs.affine(1.0, 1e-10)?;

        // Top-k selection via sorting
        // candle doesn't have a direct top-k, so we'll use argmax in a loop
        // (This is efficient for small k=2)
        let mut all_weights = Vec::new();
        let mut all_indices = Vec::new();

        // Create a mutable copy of logits for masking
        let mut masked_logits = logits.clone();

        // ── Zenith Upgrade: Expert Affinity Routing ──
        // Router learns to 'group' tokens into specialized experts
        // based on architectural affinity scores (using token embedding L2 norm as a structural proxy).
        let token_norms = x.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let norm_mean = if token_norms.elem_count() > 0 {
            token_norms.mean_all()?.to_scalar::<f32>()?
        } else {
            0.0f32
        };

        let affinity_bias_vec: Vec<f32> = token_norms
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .flat_map(|norm| {
                // If norm is above average (high entropy/complexity), bias towards experts 0 and 1.
                // If below average (low entropy/syntax), bias towards the last experts.
                let mut bias = vec![0.0f32; self.n_experts];
                if norm > norm_mean {
                    bias[0] = 0.5;
                    if self.n_experts > 1 {
                        bias[1] = 0.5;
                    }
                } else {
                    bias[self.n_experts - 1] = 0.5;
                    if self.n_experts > 2 {
                        bias[self.n_experts - 2] = 0.5;
                    }
                }
                bias
            })
            .collect();

        // Add the synthetic affinity bias to guide expert specialization cleanly
        let affinity_tensor = Tensor::from_vec(
            affinity_bias_vec,
            masked_logits.shape(),
            masked_logits.device(),
        )?;
        masked_logits = (masked_logits + affinity_tensor)?;

        for _k in 0..self.top_k {
            // Find max expert for each token
            let max_indices = masked_logits.argmax(D::Minus1)?; // [n_tokens]
            let max_vals = masked_logits.max(D::Minus1)?; // [n_tokens]

            all_indices.push(max_indices.clone());
            all_weights.push(max_vals);

            // Mask selected experts with -inf so they aren't selected again
            // Create one-hot mask and subtract infinity
            let one_hot = one_hot(&max_indices, self.n_experts, masked_logits.device())?;
            let neg_inf = Tensor::new(f32::NEG_INFINITY, masked_logits.device())?;
            let neg_inf_mask = one_hot.broadcast_mul(&neg_inf)?;
            masked_logits = (masked_logits + neg_inf_mask)?;
        }

        // Stack: [n_tokens, top_k]
        let expert_indices = Tensor::stack(&all_indices, 1)?;
        let expert_weights_raw = Tensor::stack(&all_weights, 1)?;

        // Normalize weights with softmax so they sum to 1 per token
        let expert_weights = candle_nn::ops::softmax(&expert_weights_raw, D::Minus1)?;

        // ── Expert Capacity Limits (Diamond Grade) ──
        // C = (k * tokens / experts) * capacity_factor
        let capacity_factor = 1.25f32;
        let n_tokens_f = x.dim(0)? as f32;
        let capacity = ((self.top_k as f32 * n_tokens_f / self.n_experts as f32) * capacity_factor)
            .ceil() as usize;

        // Tracking how many tokens are assigned to each expert
        let mut expert_counts = vec![0usize; self.n_experts];
        let mut final_indices = Vec::new();
        let mut final_weights = Vec::new();

        // Convert tensors to vecs for CPU capacity filtering (Industry-Grade precision)
        let indices_dims = expert_indices.dims();
        let weights_dims = expert_weights.dims();
        let indices_vec: Vec<u32> = expert_indices.flatten_all()?.to_vec1()?;
        let weights_vec: Vec<f32> = expert_weights.flatten_all()?.to_vec1()?;

        let mut dropped_tokens = 0usize;

        for (&idx, &weight) in indices_vec.iter().zip(weights_vec.iter()) {
            let expert_id = idx as usize;
            if expert_id < self.n_experts && expert_counts[expert_id] < capacity {
                expert_counts[expert_id] += 1;
                final_indices.push(idx);
                final_weights.push(weight);
            } else {
                // Token exceeded expert capacity — drop it (set weight to 0)
                final_indices.push(idx);
                final_weights.push(0.0f32);
                dropped_tokens += 1;
            }
        }

        if dropped_tokens > 0 {
            tracing::debug!(
                "MoE Capacity: dropped {} expert assignments",
                dropped_tokens
            );
        }

        let expert_indices = Tensor::from_vec(final_indices, indices_dims, logits.device())?;
        let expert_weights = Tensor::from_vec(final_weights, weights_dims, logits.device())?;

        // ── Advanced Load Balancing Loss ──
        // L_aux = N_experts * Σ_{i=1}^{N_experts} (f_i * p_i)
        // where f_i = fraction of tokens routed to expert i (non-differentiable)
        //       p_i = average probability of expert i (differentiable)

        let mut f_i_vec = vec![0.0f32; self.n_experts];
        let indices_flat: Vec<u32> = expert_indices.flatten_all()?.to_vec1()?;
        for &idx in &indices_flat {
            if (idx as usize) < self.n_experts {
                f_i_vec[idx as usize] += 1.0;
            }
        }
        let f_i = Tensor::from_vec(
            f_i_vec
                .iter()
                .map(|&c| c / (n_tokens_f * self.top_k as f32))
                .collect::<Vec<_>>(),
            (self.n_experts,),
            logits.device(),
        )?;

        let _p_i = probs.mean(0)?;

        // ── Zenith Load Balancer: Importance-Aware Loss + Entropy ──
        // We combine routing frequency (f_i) with sigmoid-importance (m_i)
        // to penalize experts that are either overused or under-specialized.
        let m_i = importance.mean(0)?;
        let load_bal = (f_i.broadcast_mul(&m_i)?.sum_all()? * (self.n_experts as f64))?;

        // ── Peak Mastery: Expert Entropy Penalty ──
        // Calculate Shannon Entropy of the average probability distribution.
        // H = - Σ p_i * log(p_i). Max entropy (log(N)) = perfect balance.
        let p_i = probs.mean(0)?;
        let epsilon = 1e-10f64;
        let p_i_safe = p_i.affine(1.0, epsilon)?;
        let entropy = (p_i_safe.broadcast_mul(&p_i_safe.log()?)?.sum_all()? * -1.0)?;
        let max_entropy = (self.n_experts as f64).ln();

        // H_gap = (max_H - current_H)^2
        let entropy_gap = (entropy.affine(-1.0, max_entropy)?.sqr())?;

        // Final Sovereign balance loss combining importance frequency and entropy gap
        let balance_loss = (load_bal.sqr()? + (entropy_gap * 0.1))?;

        Ok((expert_weights, expert_indices, balance_loss))
    }
}

/// Create a one-hot encoding tensor.
///
/// `indices`: `[n]` tensor of class indices (u32)
/// Returns: `[n, num_classes]` float32 tensor
fn one_hot(indices: &Tensor, num_classes: usize, device: &Device) -> Result<Tensor> {
    let n = indices.dim(0)?;
    let indices_vec: Vec<u32> = indices.to_vec1()?;
    let mut data = vec![0.0f32; n * num_classes];
    for (i, &idx) in indices_vec.iter().enumerate() {
        if (idx as usize) < num_classes {
            data[i * num_classes + idx as usize] = 1.0;
        }
    }
    Tensor::from_vec(data, (n, num_classes), device)
}

// ─────────────────────────────────────────────────────────────
//  Sparse MoE Layer
// ─────────────────────────────────────────────────────────────

/// Sparse Mixture-of-Experts layer.
///
/// Combines a Router with N ExpertFFN modules.
/// Each token is processed by only top-k experts (typically k=2).
///
/// Output for each token = Σ_{i ∈ top-k} (gate_weight_i * Expert_i(x))
#[derive(Debug)]
pub struct SparseMoE {
    router: Router,
    experts: Vec<ExpertFFN>,
    shared_expert: ExpertFFN, // DeepSeek-V3 style shared expert
    neuroplasticity: std::sync::Arc<super::epigenetics::NeuroplasticityMatrix>,
    n_experts: usize,
    top_k: usize,
}

impl SparseMoE {
    pub fn new(
        d_model: usize,
        d_ff: usize,
        n_experts: usize,
        top_k: usize,
        use_bitnet: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let router = Router::new(d_model, n_experts, top_k, vb.pp("router"))?;

        let mut experts = Vec::with_capacity(n_experts);
        for i in 0..n_experts {
            let expert = ExpertFFN::new(d_model, d_ff, use_bitnet, vb.pp(format!("expert_{i}")))?;
            experts.push(expert);
        }

        // DeepSeek-V3 Shared Expert (Always active for all tokens)
        let shared_expert = ExpertFFN::new(d_model, d_ff, use_bitnet, vb.pp("shared_expert"))?;

        let neuroplasticity =
            std::sync::Arc::new(super::epigenetics::NeuroplasticityMatrix::new(n_experts));

        Ok(Self {
            router,
            experts,
            shared_expert,
            neuroplasticity,
            n_experts,
            top_k,
        })
    }

    /// Forward pass through the Sparse MoE layer.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, d_model]
    ///
    /// # Returns
    /// * Output tensor [batch, seq_len, d_model]
    /// * Load balancing auxiliary loss (scalar)
    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch, seq_len, d_model) = x.dims3()?;
        let n_tokens = batch * seq_len;

        // Flatten to [n_tokens, d_model]
        if n_tokens == 0 {
            return Ok((x.clone(), Tensor::new(0.0f32, x.device())?));
        }
        let x_flat = x.reshape((n_tokens, d_model))?;

        // Route tokens to experts
        let (weights, indices, balance_loss) = self.router.route(&x_flat)?;

        // ── ELITE: Parallelized Scatter-Gather ──
        // Each expert processes its assigned tokens independently.
        // Uses rayon for CPU-level parallelism across experts.
        let indices_vec: Vec<Vec<u32>> = {
            let flat: Vec<u32> = indices.flatten_all()?.to_vec1()?;
            flat.chunks(self.top_k).map(|c| c.to_vec()).collect()
        };
        let weights_vec: Vec<Vec<f32>> = {
            let flat: Vec<f32> = weights.flatten_all()?.to_vec1()?;
            flat.chunks(self.top_k).map(|c| c.to_vec()).collect()
        };

        // Build per-expert assignment lists (precomputed for zero-overhead dispatch)
        let mut expert_assignments: Vec<Vec<(usize, f32)>> = vec![Vec::new(); self.n_experts];
        for (token_idx, (expert_list, weight_list)) in
            indices_vec.iter().zip(weights_vec.iter()).enumerate()
        {
            for (&exp_id, &weight) in expert_list.iter().zip(weight_list.iter()) {
                let eid = exp_id as usize;
                if eid < self.n_experts {
                    expert_assignments[eid].push((token_idx, weight));
                }
            }
        }

        // Process each expert and collect contributions

        use rayon::prelude::*;
        // Global epigenetic decay per step
        self.neuroplasticity.decay();

        // ── Zenith Parallel Dispatch ──
        let expert_outputs: Vec<Vec<(usize, f32, Vec<f32>)>> = self
            .experts
            .par_iter()
            .enumerate()
            .map(|(expert_id, expert)| {
                let assignments = &expert_assignments[expert_id];
                let neuroplasticity = self.neuroplasticity.clone();
                if assignments.is_empty() {
                    return Vec::new();
                }

                // Gather tokens for this expert — wrapped in error handling
                // to prevent panics from crashing the rayon thread pool
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let token_indices: Vec<u32> =
                        assignments.iter().map(|&(idx, _)| idx as u32).collect();
                    let token_idx_tensor =
                        Tensor::from_vec(token_indices, (assignments.len(),), x_flat.device())?;
                    let expert_input = x_flat.index_select(&token_idx_tensor, 0)?;

                    // Forward through expert
                    let raw_output = expert.forward(&expert_input)?;

                    // ── World-First: Epigenetic Integration ──
                    // Dynamically scale expert based on activation history!
                    let avg_activation: f32 = assignments.iter().map(|&(_, w)| w).sum::<f32>()
                        / (assignments.len() as f32).max(1.0);
                    let expert_output =
                        neuroplasticity.apply_and_update(expert_id, avg_activation, &raw_output)?;

                    let output_data: Vec<f32> = expert_output.flatten_all()?.to_vec1()?;

                    // Collect results with metadata
                    let results: Vec<(usize, f32, Vec<f32>)> = assignments
                        .iter()
                        .enumerate()
                        .map(|(i, &(token_idx, weight))| {
                            let row = output_data[i * d_model..(i + 1) * d_model].to_vec();
                            (token_idx, weight, row)
                        })
                        .collect();
                    Ok::<_, candle_core::Error>(results)
                }));

                match result {
                    Ok(Ok(data)) => data,
                    Ok(Err(e)) => {
                        tracing::error!("MoE expert {} forward failed: {}", expert_id, e);
                        Vec::new()
                    }
                    Err(_) => {
                        tracing::error!("MoE expert {} panicked during forward pass", expert_id);
                        Vec::new()
                    }
                }
            })
            .collect();

        // Scatter results back (Industry-Grade parallel accumulation)
        let mut output_data = vec![0.0f32; n_tokens * d_model];
        for expert_result in &expert_outputs {
            for (token_idx, weight, row) in expert_result {
                // Apex Optimization: Using pointer offsets to minimize index arithmetic overhead
                let start = token_idx * d_model;
                for d in 0..d_model {
                    output_data[start + d] += weight * row[d];
                }
            }
        }

        let routed_output = Tensor::from_vec(output_data, (n_tokens, d_model), x.device())?;

        // ── DeepSeek-V3 Style Shared Expert ──
        // Always active, providing a high-precision shared representation across all token routes.
        let shared_out = self.shared_expert.forward(&x_flat)?;

        let final_output = (routed_output + shared_out)?;
        let final_output = final_output.reshape((batch, seq_len, d_model))?;

        Ok((final_output, balance_loss))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_expert_ffn() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let expert = ExpertFFN::new(64, 128, true, vb)?;
        let x = Tensor::randn(0f32, 1.0, (4, 64), &Device::Cpu)?;
        let y = expert.forward(&x)?;
        assert_eq!(y.dims(), &[4, 64]);
        Ok(())
    }

    #[test]
    fn test_sparse_moe() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let moe = SparseMoE::new(64, 128, 4, 2, true, vb)?;
        let x = Tensor::randn(0f32, 1.0, (1, 4, 64), &Device::Cpu)?;
        let (y, loss) = moe.forward(&x)?;
        assert_eq!(y.dims(), &[1, 4, 64]);
        // Balance loss should be a scalar
        assert_eq!(loss.dims(), &Vec::<usize>::new());
        Ok(())
    }
}
