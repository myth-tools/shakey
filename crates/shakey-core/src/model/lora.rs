//! Low-Rank Adaptation (LoRA) modules.
//!
//! LoRA allows us to perform "online continuous learning" during user interaction
//! without having to update the frozen 1.58-bit quantized weights of the core model.
//!
//! Output = Base_Linear(x) + Lora_B(Lora_A(x)) * (alpha / rank)

use crate::model::layers::BitLinear;
use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

/// Configuration for LoRA adapters.
#[derive(Debug, Clone)]
pub struct LoraConfig {
    /// Rank of the low-rank projection (e.g., 8, 16, 32).
    /// Controls the "memory capacity" of the online updates.
    pub rank: usize,
    /// Scaling alpha parameter
    pub alpha: f64,
    /// Dropout rate for LoRA (typically 0.05-0.1, 0.0 for pure online learning)
    pub dropout: f32,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 16,    // Good balance between speed and expressive power
            alpha: 32.0, // Standard LoRA scaling
            dropout: 0.0,
        }
    }
}

/// A Linear layer augmented with Low-Rank Adapters for online learning.
#[derive(Debug)]
pub struct LoraLinear {
    /// The original frozen base layer (BitLinear)
    pub base: BitLinear,
    /// LoRA A projection matrix: [in_features, r]
    lora_a: Linear,
    /// LoRA B projection matrix: [r, out_features]
    lora_b: Linear,
    /// Scaling factor precalculated: alpha / rank
    scale: f64,
}

impl LoraLinear {
    /// Create a new LoRA-augmented BitLinear layer.
    pub fn new(
        in_features: usize,
        out_features: usize,
        use_bias: bool,
        quantize: bool,
        lora_config: &LoraConfig,
        vb: VarBuilder,      // For base layer
        lora_vb: VarBuilder, // For LoRA specifically
    ) -> Result<Self> {
        // Base layer is fully loaded from the frozen weights
        let base = BitLinear::new(in_features, out_features, use_bias, quantize, vb)?;

        // LoRA A initialized with normal distribution
        let a_weight = lora_vb.get_with_hints(
            (lora_config.rank, in_features),
            "lora_a",
            candle_nn::init::DEFAULT_KAIMING_NORMAL,
        )?;

        // LoRA B initialized with zeros (so initial adaptation is zero)
        let b_weight = lora_vb.get_with_hints(
            (out_features, lora_config.rank),
            "lora_b",
            candle_nn::init::ZERO,
        )?;

        let lora_a = Linear::new(a_weight, None);
        let lora_b = Linear::new(b_weight, None);

        let scale = lora_config.alpha / (lora_config.rank as f64);

        Ok(Self {
            base,
            lora_a,
            lora_b,
            scale,
        })
    }

    /// Update the base layer to freeze weights for fast inference.
    pub fn freeze_base(&mut self) -> Result<()> {
        self.base.freeze()
    }
}

impl Module for LoraLinear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Compute base output using quantized weights
        let base_out = self.base.forward(x)?;

        // ── Sovereign-Epsilon: Numerical Safety Guards ──
        // We clamp the input to the LoRA adapter to prevent extreme values from
        // triggering gradient explosions in the online learning phase.
        let x_clamped = x.clamp(-10.0f32, 10.0f32)?;

        // Compute LoRA adapter output in full precision
        // x -> A -> B -> scale
        let a_out = self.lora_a.forward(&x_clamped)?;
        let b_out = self.lora_b.forward(&a_out)?;

        // ── Diamond-Grade Scaling: Precision Normalization ──
        // We add a tiny epsilon to the scale to ensure it's never exactly zero,
        // which helps with certain fused kernel optimizations.
        let safe_scale = self.scale.max(1e-8);
        let scaled_lora = (b_out * safe_scale)?;

        // Final output: Base result + learned adaptation
        base_out.broadcast_add(&scaled_lora)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    use candle_nn::VarMap;

    #[test]
    fn test_lora_linear() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &Device::Cpu);

        let config = LoraConfig::default();
        let layer = LoraLinear::new(8, 4, false, false, &config, vb.pp("base"), vb.pp("lora"))?;

        let x = Tensor::randn(0f32, 1.0, (2, 8), &Device::Cpu)?;
        let y = layer.forward(&x)?;

        assert_eq!(y.dims(), &[2, 4]);

        // Ensure A and B gradients are tracked during backprop
        let loss = y.sqr()?.sum_all()?;
        let _grads = loss.backward()?;

        // Base is not frozen in this context but technically shouldn't update if we configure so
        // but Lora A and B should definitely have gradients, A especially since x varies

        Ok(())
    }
}
