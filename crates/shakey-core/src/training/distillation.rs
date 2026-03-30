//! Knowledge distillation and Preference Optimization loss functions.
//!
//! Features:
//! 1. **KL-Divergence**: Standard teacher-student knowledge transfer.
//! 2. **Cross-Entropy**: Next-token prediction on teacher labels.
//! 3. **DPO (Direct Preference Optimization)**: Alignment via chosen/rejected pairs.

use candle_core::{Result, Tensor, D};
use candle_nn::ops;

/// Configuration for distillation loss computation.
#[derive(Debug, Clone)]
pub struct DistillationConfig {
    /// Temperature for softening distributions (higher = softer)
    pub temperature: f64,
    /// Weight for KL-divergence loss (α)
    pub alpha_kl: f64,
    /// Weight for cross-entropy loss (1 - α)
    pub alpha_ce: f64,
    /// Weight for MoE load balance auxiliary loss
    pub aux_loss_weight: f64,
    /// Weight for MoE expert entropy penalty (prevents expert collapse)
    pub entropy_weight: f64,
    /// ZENITH: Weight boost for reasoning/thinking tokens
    pub reasoning_weight: f64,
    /// ELITE: Adaptive temperature scaling factor (0.0 to disable)
    pub adaptive_temp_scale: f64,
    /// Label smoothing factor (0.0 = none, 0.1 = standard for modern LLMs)
    pub label_smoothing: f64,
    /// Z-loss weight for logit stability (prevents softmax saturation, standard in PaLM/Gemini)
    pub z_loss_weight: f64,
}

/// Configuration for DPO loss computation.
#[derive(Debug, Clone)]
pub struct DpoConfig {
    /// Beta parameter (KL penalty strength) - typically 0.1 to 0.5
    pub beta: f64,
    /// Reference model log-probability weight (usually 1.0)
    pub label_smoothing: f64,
}

impl Default for DpoConfig {
    fn default() -> Self {
        Self {
            beta: 0.1,
            label_smoothing: 0.0,
        }
    }
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 2.0,
            alpha_kl: 0.7,
            alpha_ce: 0.3,
            aux_loss_weight: 0.01,
            entropy_weight: 0.005,
            reasoning_weight: 1.5, // 50% more weight on thinking tokens
            adaptive_temp_scale: 0.1,
            label_smoothing: 0.1,
            z_loss_weight: 1e-4,
        }
    }
}

/// Compute the combined distillation loss.
///
/// # Arguments
/// * `student_logits` - [batch, seq_len, vocab_size]
/// * `teacher_logits` - [batch, seq_len, vocab_size]
/// * `target_ids` - [batch, seq_len]
/// * `aux_loss` - Scalar MoE load balance loss
/// * `reasoning_mask` - Optional mask [batch, seq_len] with 1.0 for reasoning tokens
/// * `config` - Distillation hyperparameters
pub fn distillation_loss(
    student_logits: &Tensor,
    teacher_logits: &Tensor,
    target_ids: &Tensor,
    aux_loss: &Tensor,
    reasoning_mask: Option<&Tensor>,
    config: &DistillationConfig,
) -> Result<(Tensor, LossComponents)> {
    let mut temperature = Tensor::new(config.temperature as f32, student_logits.device())?;

    // ── ELITE: Adaptive Temperature Scaling ──
    // T_adaptive = T_base + scale * entropy(teacher_probs)
    // Higher entropy (uncertainty) -> Softer labels
    if config.adaptive_temp_scale > 0.0 {
        let teacher_probs_raw = ops::softmax(teacher_logits, D::Minus1)?;
        // ── Sovereign Guard: Clamp log input to prevent log(0) = -inf ──
        let clamped_probs = teacher_probs_raw.clone().clamp(1e-10, 1.0)?;
        let entropy = (teacher_probs_raw * clamped_probs.log()?.neg()?)?.sum(D::Minus1)?;
        temperature = (entropy * config.adaptive_temp_scale)?.broadcast_add(&temperature)?;
        // Expand for broadcasting: [batch, seq]
        temperature = temperature.unsqueeze(D::Minus1)?;
    }

    // ── Sovereign Guard: Floor temperature to prevent division-by-zero ──
    // Very low-entropy teacher distributions can collapse temperature to near-zero.
    let temp_floor = Tensor::new(0.1f32, student_logits.device())?;
    temperature = temperature.broadcast_maximum(&temp_floor)?;

    // Apply temperature to logits
    let student_logits_scaled = student_logits.broadcast_div(&temperature)?;
    let teacher_logits_scaled = teacher_logits.broadcast_div(&temperature)?;

    // ── KL-Divergence Loss with Numerical Stability Guard ──
    let student_log_soft = ops::log_softmax(&student_logits_scaled, D::Minus1)?;
    let teacher_log_soft = ops::log_softmax(&teacher_logits_scaled, D::Minus1)?;
    let teacher_soft = ops::softmax(&teacher_logits_scaled, D::Minus1)?;

    // KL divergence: Σ P_t * (log P_t - log P_s)
    let kl_per_token = (&teacher_soft * (teacher_log_soft - student_log_soft)?)?.sum(D::Minus1)?;

    // ── ZENITH: Reasoning Weight Boost ──
    // Multiply KL loss by reasoning_weight if the token is marked as a 'thinking' token.
    let weighted_kl = if let Some(mask) = reasoning_mask {
        let boost = (mask * (config.reasoning_weight - 1.0))?;
        let weight = (boost + 1.0)?;
        kl_per_token.broadcast_mul(&weight)?
    } else {
        kl_per_token
    };

    let kl_loss = weighted_kl.mean_all()?;

    // Scale KL by T^2 (standard distillation practice)
    // Note: If temperature is now a tensor [batch, seq], we use the mean T^2 for scaling
    let t_squared = temperature.sqr()?.mean_all()?;
    let kl_loss = (kl_loss * t_squared)?;

    // ── Cross-Entropy Loss (with Label Smoothing) ──
    let ce_loss = cross_entropy_loss(student_logits, target_ids, config.label_smoothing, None)?;

    // ── Entropy Regularization (prevents overconfident student predictions) ──
    let student_probs_unscaled = ops::softmax(student_logits, D::Minus1)?;
    let student_log_probs_unscaled = ops::log_softmax(student_logits, D::Minus1)?;
    let neg_entropy = (&student_probs_unscaled * &student_log_probs_unscaled)?
        .sum(D::Minus1)?
        .mean_all()?;

    // ── Z-Loss for Logit Stability (PaLM/Gemini standard) ──
    let z = z_loss(student_logits)?;

    // ── Combined Loss ──
    let total_loss = ((&kl_loss * config.alpha_kl)?
        + (&ce_loss * config.alpha_ce)?
        + (aux_loss * config.aux_loss_weight)?
        + (&neg_entropy * config.entropy_weight)?
        + (&z * config.z_loss_weight)?)?;

    let kl_val = kl_loss.to_vec0::<f32>()? as f64;
    let ce_val = ce_loss.to_vec0::<f32>()? as f64;
    let aux_val = aux_loss.to_vec0::<f32>()? as f64;
    let total_val = total_loss.to_vec0::<f32>()? as f64;

    let components = LossComponents {
        total: total_val,
        kl_divergence: kl_val,
        cross_entropy: ce_val,
        aux_balance: aux_val,
    };

    Ok((total_loss, components))
}

/// Z-Loss: Penalizes large pre-softmax logits to prevent softmax saturation.
/// Standard in PaLM, Gemini, and other large-scale LLMs.
/// L_z = mean(log(Σ exp(z)))²
fn z_loss(logits: &Tensor) -> Result<Tensor> {
    // Numerically stable log-sum-exp: max(z) + log(Σ exp(z - max(z)))
    let max_logits = logits.max_keepdim(D::Minus1)?;
    let shifted = logits.broadcast_sub(&max_logits)?;
    let sum_exp = shifted.exp()?.sum_keepdim(D::Minus1)?;
    // ── Sovereign Guard: Prevent log(0) for degenerate logit tensors ──
    let safe_sum_exp = sum_exp.clamp(1e-10, f32::MAX as f64)?;
    let log_sum_exp = (safe_sum_exp.log()? + max_logits)?;
    log_sum_exp.sqr()?.mean_all()
}

/// Cross-entropy loss with optional label smoothing and pad-token masking.
///
/// Label smoothing blends the hard target with a uniform distribution,
/// preventing overconfident predictions and improving generalization.
/// Standard in all modern LLMs (typically ε = 0.1).
///
/// # Arguments
/// * `logits` - [batch, seq_len, vocab_size]
/// * `targets` - [batch, seq_len] u32 token IDs
/// * `label_smoothing` - Smoothing factor (0.0 = none, 0.1 = standard)
/// * `ignore_index` - Optional token ID to exclude from loss (typically pad_id)
pub fn cross_entropy_loss(
    logits: &Tensor,
    targets: &Tensor,
    label_smoothing: f64,
    ignore_index: Option<u32>,
) -> Result<Tensor> {
    let (batch, seq_len, vocab_size) = logits.dims3()?;
    let n_tokens = batch * seq_len;

    // Flatten: [batch * seq_len, vocab_size]
    let logits_flat = logits.reshape((n_tokens, vocab_size))?;
    let targets_flat = targets.flatten_all()?;

    // Log-softmax: [total_tokens, vocab_size]
    let log_probs = ops::log_softmax(&logits_flat, D::Minus1)?;

    // Gather log_prob at each target index: [total_tokens, 1]
    let targets_reshaped = targets_flat.reshape((n_tokens, 1))?;
    let target_log_probs = log_probs
        .gather(&targets_reshaped, D::Minus1)?
        .squeeze(D::Minus1)?;

    // Build loss mask (1.0 for valid tokens, 0.0 for ignored/pad tokens)
    let (masked_nll, n_valid_f) = if let Some(ign_idx) = ignore_index {
        let targets_vec: Vec<u32> = targets_flat.to_vec1()?;
        let mask_vec: Vec<f32> = targets_vec
            .iter()
            .map(|&t| if t == ign_idx { 0.0 } else { 1.0 })
            .collect();
        let mask = Tensor::from_vec(mask_vec, target_log_probs.shape(), logits.device())?;
        let n_valid = mask.sum_all()?;
        // Guard: if all tokens are padding, return zero loss
        let n_valid_scalar = n_valid.to_vec0::<f32>()?;
        if n_valid_scalar < 1.0 {
            return Tensor::new(0.0f32, logits.device());
        }
        let masked = target_log_probs.broadcast_mul(&mask)?;
        (masked.sum_all()?, n_valid)
    } else {
        (
            target_log_probs.sum_all()?,
            Tensor::new(n_tokens as f32, logits.device())?,
        )
    };

    let nll_loss = (masked_nll.broadcast_div(&n_valid_f)? * -1.0)?;

    if label_smoothing > 0.0 {
        // Label smoothing: L = (1 - ε) * NLL + ε * (-mean(all log_probs))
        let smooth_loss = (log_probs.mean_all()? * -1.0)?;
        (&nll_loss * (1.0 - label_smoothing))? + (&smooth_loss * label_smoothing)?
    } else {
        Ok(nll_loss)
    }
}

/// Individual loss components for logging and monitoring.
#[derive(Debug, Clone)]
pub struct LossComponents {
    pub total: f64,
    pub kl_divergence: f64,
    pub cross_entropy: f64,
    pub aux_balance: f64,
}

/// Compute the DPO (Direct Preference Optimization) loss.
///
/// # Arguments
/// * `policy_chosen_logps` - Log-probabilities of chosen responses under policy model
/// * `policy_rejected_logps` - Log-probabilities of rejected responses under policy model
/// * `reference_chosen_logps` - Log-probabilities of chosen responses under reference model
/// * `reference_rejected_logps` - Log-probabilities of rejected responses under reference model
/// * `config` - DPO hyperparameters
pub fn dpo_loss(
    policy_chosen_logps: &Tensor,
    policy_rejected_logps: &Tensor,
    reference_chosen_logps: &Tensor,
    reference_rejected_logps: &Tensor,
    config: &DpoConfig,
) -> Result<Tensor> {
    // chosen_logratios = policy_chosen_logps - reference_chosen_logps
    let chosen_logratios = (policy_chosen_logps - reference_chosen_logps)?;
    let rejected_logratios = (policy_rejected_logps - reference_rejected_logps)?;

    let logits = (chosen_logratios - rejected_logratios)?;
    let logits = (logits * config.beta)?;

    // Loss = -log_sigmoid(beta * log_ratios)
    // Stable log_sigmoid(x) = -softplus(-x) = -[max(0, -x) + log(1 + exp(-|x|))]
    // This avoids exp() overflow for large |x| values.
    let x = logits;
    let neg_abs_x = x.abs()?.neg()?;
    let zero = Tensor::new(0.0f32, x.device())?;
    let softplus_neg_x = (x.neg()?.maximum(&zero)? + (neg_abs_x.exp()? + 1.0)?.log()?)?;

    // Mean over batch
    softplus_neg_x.mean_all()
}

/// Compute log-probabilities for a sequence of tokens given logits.
pub fn get_batch_logps(logits: &Tensor, labels: &Tensor, average_log_prob: bool) -> Result<Tensor> {
    // logits: [batch, seq_len, vocab]
    // labels: [batch, seq_len]
    let (batch, seq, vocab) = logits.dims3()?;

    let log_probs = ops::log_softmax(logits, D::Minus1)?;

    // Gather log_probs at label indices
    // Reshape labels to [batch * seq, 1]
    let labels_flat = labels.reshape((batch * seq, 1))?;
    let log_probs_flat = log_probs.reshape((batch * seq, vocab))?;

    let selected_logps = log_probs_flat.gather(&labels_flat, D::Minus1)?;
    let selected_logps = selected_logps.reshape((batch, seq))?;

    // Sum over sequence
    let logps = selected_logps.sum(D::Minus1)?;

    if average_log_prob {
        logps / seq as f64
    } else {
        Ok(logps)
    }
}

impl std::fmt::Display for LossComponents {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "total={:.4} (kl={:.4} ce={:.4} aux={:.6})",
            self.total, self.kl_divergence, self.cross_entropy, self.aux_balance
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_cross_entropy_loss() -> Result<()> {
        // Create simple logits and targets
        let logits = Tensor::new(&[[[2.0f32, 1.0, 0.1], [0.1, 2.0, 1.0]]], &Device::Cpu)?;
        let targets = Tensor::new(&[[0u32, 1]], &Device::Cpu)?;

        let loss = cross_entropy_loss(&logits, &targets, 0.0, None)?;
        let loss_val: f32 = loss.to_vec0()?;

        // Loss should be positive and finite
        assert!(loss_val > 0.0);
        assert!(loss_val.is_finite());
        Ok(())
    }

    #[test]
    fn test_distillation_loss() -> Result<()> {
        let student = Tensor::randn(0f32, 1.0, (1, 4, 100), &Device::Cpu)?;
        let teacher = Tensor::randn(0f32, 1.0, (1, 4, 100), &Device::Cpu)?;
        let targets = Tensor::new(&[[1u32, 5, 10, 20]], &Device::Cpu)?;
        let aux = Tensor::new(0.01f32, &Device::Cpu)?;

        let config = DistillationConfig::default();
        let (_loss, components) =
            distillation_loss(&student, &teacher, &targets, &aux, None, &config)?;

        assert!(components.total.is_finite());
        assert!(components.kl_divergence.is_finite());
        assert!(components.cross_entropy.is_finite());
        Ok(())
    }
}
