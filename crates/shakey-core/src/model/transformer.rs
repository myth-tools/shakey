//! Full Transformer model — assembles all layers into a complete LLM.
//!
//! Architecture per block:
//!   x → RMSNorm → GQA Attention → + residual
//!     → RMSNorm → Sparse MoE FFN → + residual
//!
//! The final output head projects from d_model to vocab_size for
//! next-token prediction. Weight tying with the embedding layer
//! is supported to reduce parameters.

use super::attention::{create_causal_mask, GroupedQueryAttention, QuantizedKvCache};
use super::config::ModelConfig;
use super::layers::{BitLinear, RmsNorm, TokenEmbedding};
use super::moe::SparseMoE;
use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};

// ─────────────────────────────────────────────────────────────
//  Output Types
// ─────────────────────────────────────────────────────────────

/// Output of a single transformer layer.
#[derive(Debug)]
pub struct LayerOutput {
    pub hidden: Tensor,
    pub kv_cache: QuantizedKvCache,
    pub aux_loss: Tensor,
}

/// Output of the full transformer model.
#[derive(Debug)]
pub struct ForwardOutput {
    pub logits: Tensor,
    /// Speculative logits from Medusa heads: [batch, n_heads, seq_len, vocab_size]
    pub medusa_logits: Option<Tensor>,
    pub kv_caches: Vec<QuantizedKvCache>,
    pub aux_loss: Tensor,
}

// ─────────────────────────────────────────────────────────────
//  Transformer Block
// ─────────────────────────────────────────────────────────────

/// Single transformer block with pre-norm architecture.
///
/// Forward:
///   h = x + Attention(RMSNorm(x))
///   out = h + MoE_FFN(RMSNorm(h))
#[derive(Debug)]
pub struct TransformerBlock {
    attn_norm: RmsNorm,
    attention: GroupedQueryAttention,
    /// RMSNorm immediately after attention to stabilize high-depth activation spikes
    post_attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    moe_ffn: SparseMoE,
    deep_norm_alpha: f32,
}

impl TransformerBlock {
    pub fn new(cfg: &ModelConfig, layer_idx: usize, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp(format!("layer_{layer_idx}"));

        let attn_norm = RmsNorm::new(cfg.d_model, cfg.rms_norm_eps, vb.pp("attn_norm"))?;
        let attention = GroupedQueryAttention::new(
            cfg.d_model,
            cfg.n_heads,
            cfg.n_kv_heads,
            cfg.max_seq_len,
            cfg.rope_theta,
            cfg.rope_scaling,
            cfg.use_bitnet(),
            cfg.qk_norm,
            cfg.rms_norm_eps,
            vb.pp("attention"),
        )?;
        let post_attn_norm = RmsNorm::new(cfg.d_model, cfg.rms_norm_eps, vb.pp("post_attn_norm"))?;
        let ffn_norm = RmsNorm::new(cfg.d_model, cfg.rms_norm_eps, vb.pp("ffn_norm"))?;
        let moe_ffn = SparseMoE::new(
            cfg.d_model,
            cfg.d_ff,
            cfg.n_experts,
            cfg.n_active_experts,
            cfg.use_bitnet(),
            vb.pp("moe"),
        )?;

        // ── DeepNorm Residual Scale Factor ──
        // Enhances training stability for extremely deep networks (>1000 layers)
        // by scaling the residual branch gracefully.
        let deep_norm_alpha = (2.0 * cfg.n_layers as f64).powf(0.25) as f32;

        Ok(Self {
            attn_norm,
            attention,
            post_attn_norm,
            ffn_norm,
            moe_ffn,
            deep_norm_alpha,
        })
    }

    /// Forward pass through one transformer block.
    ///
    /// # Returns
    /// * Output tensor [batch, seq_len, d_model]
    /// * Updated KV-cache (k, v)
    /// * Load balance loss from MoE
    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        start_pos: usize,
        kv_cache: Option<&QuantizedKvCache>,
    ) -> Result<LayerOutput> {
        // DeepNorm Scaled Pre-norm attention with residual
        let normed = self.attn_norm.forward(x)?;
        let (mut attn_out, kv_cache_new) =
            self.attention.forward(&normed, mask, start_pos, kv_cache)?;

        // Stabilize attention output with post-attention norm
        attn_out = self.post_attn_norm.forward(&attn_out)?;

        let x_scaled = x.affine(self.deep_norm_alpha as f64, 0.0)?;
        let h = (x_scaled + attn_out)?;

        // Pre-norm MoE FFN with residual
        let normed = self.ffn_norm.forward(&h)?;
        let (ffn_out, balance_loss) = self.moe_ffn.forward(&normed)?;

        let h_scaled = h.affine(self.deep_norm_alpha as f64, 0.0)?;
        let output = (h_scaled + ffn_out)?;

        Ok(LayerOutput {
            hidden: output,
            kv_cache: kv_cache_new,
            aux_loss: balance_loss,
        })
    }
}

// ─────────────────────────────────────────────────────────────
//  Full Transformer Model
// ─────────────────────────────────────────────────────────────

/// Complete Transformer LLM with BitNet 1.58-bit + Sparse MoE.
///
/// Components:
/// - Token embedding
/// - N transformer blocks (each with GQA + MoE)
/// - Final RMSNorm
/// - Output head (d_model → vocab_size)
#[derive(Debug)]
pub struct TransformerModel {
    embedding: TokenEmbedding,
    blocks: Vec<TransformerBlock>,
    final_norm: RmsNorm,
    /// None when using tied embeddings (weight shared with embedding layer)
    output_head: Option<BitLinear>,
    medusa_heads: Vec<super::layers::MedusaHead>,
    pathway_router: Option<super::pathway::PathwayRouter>,
    config: ModelConfig,
}

impl TransformerModel {
    /// Create a new Transformer model from config.
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        cfg.validate()
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let embedding = TokenEmbedding::new(cfg.vocab_size, cfg.d_model, vb.pp("embedding"))?;

        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let block = TransformerBlock::new(cfg, i, vb.pp("blocks"))?;
            blocks.push(block);
        }

        let final_norm = RmsNorm::new(cfg.d_model, cfg.rms_norm_eps, vb.pp("final_norm"))?;

        // Output head: d_model → vocab_size
        // When tie_word_embeddings is true, we reuse the embedding matrix (LLaMA/Mistral standard)
        let output_head = if cfg.tie_word_embeddings {
            None
        } else {
            Some(BitLinear::new(
                cfg.d_model,
                cfg.vocab_size,
                false,
                false, // Don't quantize output head — needs full precision for logits
                vb.pp("output_head"),
            )?)
        };

        // Initialize Medusa heads for speculative decoding
        let mut medusa_heads = Vec::with_capacity(cfg.n_medusa_heads);
        for i in 0..cfg.n_medusa_heads {
            let head = super::layers::MedusaHead::new(
                cfg.d_model,
                cfg.vocab_size,
                i + 1, // offset: head 0 predicts t+2, head 1 predicts t+3, etc.
                vb.pp(format!("medusa_head_{i}")),
            )?;
            medusa_heads.push(head);
        }

        // Mixture of Depths / Pathways Router (Genesis architecture)
        let pathway_router = if cfg.n_layers > 4 {
            Some(super::pathway::PathwayRouter::new(
                cfg.d_model,
                cfg.n_layers,
                vb.pp("pathway"),
            )?)
        } else {
            None
        };

        Ok(Self {
            embedding,
            blocks,
            final_norm,
            output_head,
            medusa_heads,
            pathway_router,
            config: cfg.clone(),
        })
    }

    /// Forward pass: token IDs → logits.
    ///
    /// # Arguments
    /// * `token_ids` - [batch, seq_len] u32 tensor of token indices
    /// * `start_pos` - Position offset for KV-cache
    /// * `kv_caches` - Optional KV-caches from previous generation step
    ///
    /// # Returns
    /// * `logits` - [batch, seq_len, vocab_size] — raw logits for next token prediction
    /// * `kv_caches` - Updated KV-caches for each layer
    /// * `aux_loss` - Total load balance auxiliary loss from all MoE layers
    pub fn forward(
        &self,
        token_ids: &Tensor,
        start_pos: usize,
        kv_caches: Option<&[QuantizedKvCache]>,
        temperature: Option<f32>,
    ) -> Result<ForwardOutput> {
        let seq_len = token_ids.dim(1)?;
        let max_ctx = self.config.max_seq_len;
        let temp = temperature.unwrap_or(1.0);

        // ── Peak Mastery: Hard Sequence Limit ──
        if seq_len + start_pos > max_ctx {
            return Err(candle_core::Error::Msg(format!(
                "Context window exceeded: {} current + {} start > {} max. Truncate input before inference.",
                seq_len, start_pos, max_ctx
            )));
        }

        // Token embedding
        let mut hidden = self.embedding.forward(token_ids)?;

        // Create causal mask with optional Sliding Window Attention
        let mask = if seq_len > 1 {
            Some(create_causal_mask(
                seq_len,
                self.config.sliding_window,
                token_ids.device(),
            )?)
        } else {
            None
        };

        // ── World-First: Dynamic Mixture of Depths & Pathways ──
        // Compute the optimal execution pathway for the current input block.
        let execution_path = if let Some(ref router) = self.pathway_router {
            router.compute_pathway(&hidden, temp)?
        } else {
            (0..self.blocks.len()).collect()
        };

        tracing::debug!(target: "shakey::model", "🚀 Dynamic Pathway: {:?}", execution_path);

        // Track KV-caches and auxiliary losses
        // (Pre-allocate to the base layer count for stability)
        let mut new_kv_caches = vec![None; self.blocks.len()];
        let mut total_aux_loss = Tensor::new(0.0f32, token_ids.device())?;

        // Forward through the dynamic layer sequence
        for &layer_idx in &execution_path {
            let block = &self.blocks[layer_idx];
            let kv_cache = kv_caches.as_ref().and_then(|c| c.get(layer_idx));

            let layer_out = block.forward(&hidden, mask.as_ref(), start_pos, kv_cache)?;

            hidden = layer_out.hidden;
            // We store the KV-cache from the *last* visit of the block in this pass
            new_kv_caches[layer_idx] = Some(layer_out.kv_cache.clone());
            total_aux_loss = (total_aux_loss + layer_out.aux_loss)?;
        }

        // Collect populated caches (skipping any layer that was entirely bypassed)
        let mut final_kv_caches = Vec::with_capacity(self.blocks.len());
        for (i, cache_opt) in new_kv_caches.into_iter().enumerate() {
            if let Some(c) = cache_opt {
                final_kv_caches.push(c);
            } else {
                // If a layer was skipped entirely, it must retain its old cache (if any)
                // or initialize an empty one for consistency.
                let fallback = if let Some(old) = kv_caches.as_ref().and_then(|c| c.get(i)) {
                    old.clone()
                } else {
                    // INDUSTRY-GRADE: Sovereign Cache Initialization
                    // For layers bypassed in the current execution pathway, we maintain a
                    // zero-length head-aligned tensor. RoPE and Masking logic will
                    // correctly handle this as "missing temporal history" for this specific layer.
                    let d_head = self.config.d_model / self.config.n_heads;
                    let empty = Tensor::zeros(
                        (1, self.config.n_kv_heads, 0, d_head),
                        candle_core::DType::F32,
                        token_ids.device(),
                    )?;
                    QuantizedKvCache::new(&empty, &empty)?
                };
                final_kv_caches.push(fallback);
            }
        }

        // ── Peak Mastery: Streaming Context Management ──
        // If the KV-cache exceeds a safety threshold (max_seq_len),
        // we perform sliding-window pruning to maintain O(1) memory overhead.
        for cache in &mut final_kv_caches {
            let current_len = cache.k.dim(2)?;
            if current_len > max_ctx {
                let start = current_len - max_ctx;
                cache.k = cache.k.narrow(2, start, max_ctx)?;
                cache.v = cache.v.narrow(2, start, max_ctx)?;
                cache.k_scale = cache.k_scale.narrow(2, start, max_ctx)?;
                cache.v_scale = cache.v_scale.narrow(2, start, max_ctx)?;
            }
        }

        // Final normalization
        hidden = self.final_norm.forward(&hidden)?;

        // Project to vocabulary (tied embeddings or dedicated head)
        let logits = if let Some(ref head) = self.output_head {
            head.forward(&hidden)?
        } else {
            // Tied embedding: output = hidden @ embedding_weight^T
            hidden.broadcast_matmul(&self.embedding.weight().t()?)?
        };

        // ── Zenith Upgrade: Final Logit Soft-Capping ──
        // Prevents logit divergence in ultra-long context or high-temperature sampling.
        let final_logit_cap = 30.0f64;
        let logits = (logits.affine(1.0 / final_logit_cap, 0.0)?.tanh()? * final_logit_cap)?;

        // ── Peak Mastery: Adaptive Medusa Decoding ──
        // Dynamically scale the number of lookahead heads based on the entropy
        // of the current token prediction. High confidence = more speculative heads.
        let mut medusa_logits = Vec::with_capacity(self.medusa_heads.len());

        // Compute base probabilities to estimate confidence (on the last token)
        // [batch, 1, vocab] -> [batch, vocab]
        let heads_to_use = if seq_len > 0 && !self.medusa_heads.is_empty() {
            let last_logits = logits.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
            let probs = candle_nn::ops::softmax(&last_logits, candle_core::D::Minus1)?;

            // Use mean confidence across batch for Medusa gating
            // Guard against empty tensors to prevent "empty tensor for reduce" panic
            let max_prob = if probs.elem_count() > 0 {
                probs
                    .max(candle_core::D::Minus1)?
                    .mean_all()?
                    .to_vec0::<f32>()?
            } else {
                0.0f32
            };

            // Heuristic: If confidence > 80%, use all heads. If > 50%, use half. Otherwise, use 1.
            if max_prob > 0.8 {
                self.medusa_heads.len()
            } else if max_prob > 0.5 {
                self.medusa_heads.len() / 2
            } else {
                1
            }
            .max(1)
        } else {
            0
        };

        for i in 0..heads_to_use {
            medusa_logits.push(self.medusa_heads[i].forward(&hidden)?);
        }

        // Stack medusa logits: [batch, n_used_medusa, seq_len, vocab_size]
        let medusa_logits = if medusa_logits.is_empty() {
            None
        } else {
            Some(Tensor::stack(&medusa_logits, 1)?)
        };

        Ok(ForwardOutput {
            logits,
            medusa_logits,
            kv_caches: final_kv_caches,
            aux_loss: total_aux_loss,
        })
    }

    /// Get the model configuration.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Count total trainable parameters.
    pub fn count_parameters(varmap: &candle_nn::VarMap) -> usize {
        varmap.all_vars().iter().map(|v| v.elem_count()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_transformer_block() -> Result<()> {
        let cfg = ModelConfig::seed();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let block = TransformerBlock::new(&cfg, 0, vb)?;

        let x = Tensor::randn(0f32, 1.0, (1, 4, 256), &Device::Cpu)?;
        let mask = create_causal_mask(4, None, &Device::Cpu)?;
        let layer_out = block.forward(&x, Some(&mask), 0, None)?;
        assert_eq!(layer_out.hidden.dims(), &[1, 4, 256]);
        Ok(())
    }

    #[test]
    fn test_full_model_seed() -> Result<()> {
        let cfg = ModelConfig::seed();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let model = TransformerModel::new(&cfg, vb)?;

        let token_ids = Tensor::new(&[[1u32, 5, 10, 20]], &Device::Cpu)?;
        let out = model.forward(&token_ids, 0, None, None)?;

        assert_eq!(out.logits.dims(), &[1, 4, 32000]); // [batch, seq, vocab]
        assert_eq!(out.kv_caches.len(), 6); // 6 layers
        assert_eq!(out.aux_loss.dims(), &Vec::<usize>::new()); // scalar

        let total_params = TransformerModel::count_parameters(&varmap);
        tracing::info!("Seed model: {} parameters", total_params);
        Ok(())
    }
}
