//! Model configuration.
//!
//! Defines all hyperparameters for every model stage (Seed → Forest).
//! Configurations are loaded from `configs/model_stages.yaml`.

use serde::{Deserialize, Serialize};

/// Complete model configuration. Fully serializable for checkpoint/resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Human-readable stage name (seed, sprout, sapling, tree, forest)
    pub name: String,
    /// Approximate total parameter count (for display only)
    #[serde(default = "default_total_params")]
    pub total_params_approx: String,
    /// Vocabulary size (number of unique tokens)
    pub vocab_size: usize,
    /// Hidden dimension of the model
    pub d_model: usize,
    /// Number of transformer layers
    pub n_layers: usize,
    /// Number of attention heads (query heads)
    pub n_heads: usize,
    /// Number of key-value heads (for GQA; n_kv_heads <= n_heads)
    pub n_kv_heads: usize,
    /// Feed-forward hidden dimension (per expert)
    pub d_ff: usize,
    /// Total number of experts in each MoE layer
    pub n_experts: usize,
    /// Number of experts activated per token (top-k routing)
    pub n_active_experts: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// RoPE base frequency
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    /// RMSNorm epsilon
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    /// Weight quantization ("1.58" for ternary, "fp32" for full precision)
    #[serde(default = "default_weight_bits")]
    pub weight_bits: String,
    /// Activation function ("swiglu", "gelu", "relu")
    #[serde(default = "default_activation")]
    pub activation: String,
    /// Number of Medusa heads for speculative decoding
    #[serde(default = "default_n_medusa_heads")]
    pub n_medusa_heads: usize,
    /// RoPE scaling factor (e.g., 2.0 to double the context window)
    #[serde(default = "default_rope_scaling")]
    pub rope_scaling: f64,
    /// Whether to tie input embedding weights with output head (standard in LLaMA/Mistral/GPT)
    #[serde(default = "default_tie_embeddings")]
    pub tie_word_embeddings: bool,
    /// Sliding window size for local attention (None = full attention, used by Mistral/Gemma)
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// Whether to normalize queries and keys before attention (QK-Norm, standard in Gemma 2 / Cohere)
    #[serde(default = "default_qk_norm")]
    pub qk_norm: bool,
}

fn default_qk_norm() -> bool {
    true // Very important for stable training at scale
}

fn default_rope_scaling() -> f64 {
    2.0 // Double context by default for Elite Expansion
}

fn default_n_medusa_heads() -> usize {
    3 // Triple-prediction by default
}

fn default_total_params() -> String {
    "unknown".into()
}
fn default_rope_theta() -> f64 {
    500000.0 // LLaMA-3 standard for massive context support
}
fn default_rms_norm_eps() -> f64 {
    1e-6
}
fn default_weight_bits() -> String {
    "1.58".into()
}
fn default_activation() -> String {
    "swiglu".into()
}
fn default_tie_embeddings() -> bool {
    true // Standard in modern LLMs for parameter efficiency
}

impl ModelConfig {
    /// Head dimension = d_model / n_heads
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }

    /// KV dimension = head_dim * n_kv_heads
    pub fn kv_dim(&self) -> usize {
        self.head_dim() * self.n_kv_heads
    }

    /// Whether to use BitNet 1.58-bit quantization
    pub fn use_bitnet(&self) -> bool {
        self.weight_bits == "1.58"
    }

    /// Returns the Seed (Stage 0) default config
    pub fn seed() -> Self {
        Self {
            name: "seed".into(),
            total_params_approx: "15M".into(),
            vocab_size: 32000,
            d_model: 256,
            n_layers: 6,
            n_heads: 4,
            n_kv_heads: 2,
            d_ff: 512,
            n_experts: 4,
            n_active_experts: 2,
            max_seq_len: 1024,
            rope_theta: 500000.0,
            rms_norm_eps: 1e-6,
            weight_bits: "1.58".into(),
            activation: "swiglu".into(),
            n_medusa_heads: 3,
            rope_scaling: 2.0,
            tie_word_embeddings: true,
            sliding_window: None,
            qk_norm: true,
        }
    }

    /// Returns the Sprout (Stage 1) default config
    pub fn sprout() -> Self {
        Self {
            name: "sprout".into(),
            total_params_approx: "60M".into(),
            vocab_size: 32000,
            d_model: 512,
            n_layers: 12,
            n_heads: 8,
            n_kv_heads: 4,
            d_ff: 1024,
            n_experts: 4,
            n_active_experts: 2,
            max_seq_len: 2048,
            rope_theta: 500000.0,
            rms_norm_eps: 1e-6,
            weight_bits: "1.58".into(),
            activation: "swiglu".into(),
            n_medusa_heads: 3,
            rope_scaling: 2.0,
            tie_word_embeddings: true,
            sliding_window: None,
            qk_norm: true,
        }
    }

    /// Load model configs from YAML file
    pub fn load_stages(path: &str) -> anyhow::Result<Vec<Self>> {
        let content = std::fs::read_to_string(path)?;
        let doc: StagesFile = serde_yaml::from_str(&content)?;
        Ok(doc.stages)
    }

    /// Validate config consistency
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.d_model.is_multiple_of(self.n_heads),
            "d_model ({}) must be divisible by n_heads ({})",
            self.d_model,
            self.n_heads
        );
        anyhow::ensure!(
            self.n_heads.is_multiple_of(self.n_kv_heads),
            "n_heads ({}) must be divisible by n_kv_heads ({})",
            self.n_heads,
            self.n_kv_heads
        );
        anyhow::ensure!(
            self.n_active_experts <= self.n_experts,
            "n_active_experts ({}) must be <= n_experts ({})",
            self.n_active_experts,
            self.n_experts
        );
        anyhow::ensure!(self.vocab_size > 0, "vocab_size must be > 0");
        anyhow::ensure!(self.d_ff > 0, "d_ff must be > 0");
        anyhow::ensure!(self.rope_scaling >= 1.0, "rope_scaling must be >= 1.0");
        if let Some(w) = self.sliding_window {
            anyhow::ensure!(w > 0, "sliding_window must be > 0");
            anyhow::ensure!(
                w <= self.max_seq_len,
                "sliding_window ({}) must be <= max_seq_len ({})",
                w,
                self.max_seq_len
            );
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct StagesFile {
    stages: Vec<ModelConfig>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seed_config_valid() {
        let cfg = ModelConfig::seed();
        cfg.validate().unwrap();
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.kv_dim(), 128);
        assert!(cfg.use_bitnet());
    }

    #[test]
    fn test_sprout_config_valid() {
        let cfg = ModelConfig::sprout();
        cfg.validate().unwrap();
        assert_eq!(cfg.head_dim(), 64);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let cfg = ModelConfig::seed();
        let yaml = serde_yaml::to_string(&cfg).unwrap();
        let cfg2: ModelConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(cfg.d_model, cfg2.d_model);
        assert_eq!(cfg.n_layers, cfg2.n_layers);
    }
}
