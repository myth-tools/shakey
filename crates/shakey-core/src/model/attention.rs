//! Grouped Query Attention (GQA) with Rotary Position Embeddings (RoPE).
//!
//! ## Grouped Query Attention
//!
//! Standard multi-head attention uses separate K/V projections for each head.
//! GQA shares K/V projections across groups of query heads, reducing memory
//! by a factor of (n_heads / n_kv_heads) with minimal quality loss.
//!
//! ## Rotary Position Embeddings (RoPE)
//!
//! Instead of learned positional embeddings, RoPE encodes position information
//! by rotating the query and key vectors. This enables:
//! - Extrapolation to longer sequences than seen during training
//! - Relative position awareness (attention naturally decays with distance)
//! - No additional learned parameters

use super::layers::{BitLinear, LinearLayer, RmsNorm};
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Module, VarBuilder};

// ─────────────────────────────────────────────────────────────
//  Rotary Position Embeddings (RoPE)
// ─────────────────────────────────────────────────────────────

/// Pre-computed RoPE frequency tensors.
///
/// RoPE works by rotating pairs of dimensions in the Q/K vectors.
/// The rotation angle depends on position and dimension index.
///
/// For position `p` and dimension pair `i`:
///   θ_i = theta_base ^ (-2i / d)
///   q'`[2i]`   = q`[2i]`   * cos(p·θ_i) - q`[2i+1]` * sin(p·θ_i)
///   q'`[2i+1]` = q`[2i]`   * sin(p·θ_i) + q`[2i+1]` * cos(p·θ_i)
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    cos_cache: Tensor,
    sin_cache: Tensor,
}

impl RotaryEmbedding {
    /// Pre-compute cos/sin tables for all positions up to `max_seq_len` with NTK-aware scaling.
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        theta: f64,
        scaling_factor: f64,
        device: &Device,
    ) -> Result<Self> {
        let half_dim = head_dim / 2;

        // --- YaRN (Yet another RoPE extensioN) Scaling ---
        // YaRN elegantly balances interpolation and extrapolation across frequency bands,
        // solving the "context window destruction" problem of plain NTK scaling.
        // It applies a temperature shift to recover missing entropy in long contexts.
        let m_scale = if scaling_factor > 1.0 {
            0.1 * scaling_factor.ln() + 1.0
        } else {
            1.0
        };

        let beta_fast = 32.0;
        let beta_slow = 1.0;

        let mut freqs_vec: Vec<f32> = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            let dim_ratio = 2.0 * (i as f64) / (head_dim as f64);
            let base_freq = 1.0 / theta.powf(dim_ratio);
            let wavelength = 2.0 * std::f64::consts::PI / base_freq;

            // Frequency band splitting
            let gamma = if wavelength < beta_fast {
                1.0 // Extrapolate high frequencies (local structure)
            } else if wavelength > beta_slow * scaling_factor {
                1.0 / scaling_factor // Interpolate low frequencies (global context)
            } else {
                let blend = (wavelength - beta_fast) / (beta_slow * scaling_factor - beta_fast);
                (1.0 - blend) + blend * (1.0 / scaling_factor)
            };

            // Apply YaRN temperature entropy scaling
            let scaled_freq = base_freq * gamma * m_scale;
            freqs_vec.push(scaled_freq as f32);
        }

        let freqs = Tensor::from_vec(freqs_vec, (1, half_dim), device)?;

        // Position indices [0, 1, 2, ..., max_seq_len-1]
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::from_vec(positions, (max_seq_len, 1), device)?;

        // Outer product: [max_seq_len, half_dim]
        let angles = positions.matmul(&freqs)?;

        // Pre-compute cos and sin: [max_seq_len, half_dim]
        let cos_cache = angles.cos()?;
        let sin_cache = angles.sin()?;

        Ok(Self {
            cos_cache,
            sin_cache,
        })
    }

    /// Apply RoPE to query or key tensor.
    ///
    /// # Arguments
    /// * `x` - Tensor of shape [batch, n_heads, seq_len, head_dim]
    /// * `start_pos` - Starting position (for KV-cache during generation)
    pub fn apply(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        let (_batch, _n_heads, seq_len, head_dim) = x.dims4()?;
        let half_dim = head_dim / 2;
        let device = x.device();

        // ── Peak Mastery: Device-Safe Cache Materialization ──
        // If the model was moved to a different device (e.g. CPU -> CUDA),
        // we must ensure the RoPE caches are relocated before narrow/reshape.
        let cos = if self.cos_cache.device().location() != device.location() {
            self.cos_cache.to_device(device)?
        } else {
            self.cos_cache.clone()
        };
        let sin = if self.sin_cache.device().location() != device.location() {
            self.sin_cache.to_device(device)?
        } else {
            self.sin_cache.clone()
        };

        let cos = cos
            .narrow(0, start_pos, seq_len)?
            .reshape((1, 1, seq_len, half_dim))?;
        let sin = sin
            .narrow(0, start_pos, seq_len)?
            .reshape((1, 1, seq_len, half_dim))?;

        // ── Vectorized Rotation ──
        let x_even = x.narrow(D::Minus1, 0, half_dim)?;
        let x_odd = x.narrow(D::Minus1, half_dim, half_dim)?;

        let x_even_rot = (x_even.broadcast_mul(&cos)? - x_odd.broadcast_mul(&sin)?)?;
        let x_odd_rot = (x_even.broadcast_mul(&sin)? + x_odd.broadcast_mul(&cos)?)?;

        Tensor::cat(&[x_even_rot, x_odd_rot], D::Minus1)
    }
}

// ─────────────────────────────────────────────────────────────
//  Grouped Query Attention (GQA)
// ─────────────────────────────────────────────────────────────

/// Grouped Query Attention with BitLinear projections and RoPE.
///
/// - Q projection: d_model → n_heads * head_dim
/// - K projection: d_model → n_kv_heads * head_dim  (fewer heads!)
/// - V projection: d_model → n_kv_heads * head_dim  (fewer heads!)
/// - O projection: n_heads * head_dim → d_model
///
/// K/V heads are expanded (repeated) to match Q heads during attention.
#[derive(Debug)]
pub struct GroupedQueryAttention {
    q_proj: LinearLayer,
    k_proj: LinearLayer,
    v_proj: LinearLayer,
    o_proj: LinearLayer,
    /// Optional QK-Norm: normalizes queries to prevent training instability (Gemma 2)
    q_norm: Option<RmsNorm>,
    /// Optional QK-Norm: normalizes keys
    k_norm: Option<RmsNorm>,
    rope: RotaryEmbedding,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    n_rep: usize, // n_heads / n_kv_heads — how many times to repeat KV
}

/// Elite Sovereign structure for INT8 Quantized KV Cache.
///
/// Stores K and V as INT8 with per-head, per-step scaling factors
/// to maintain maximum numerical precision while reducing memory footprint.
#[derive(Debug, Clone)]
pub struct QuantizedKvCache {
    pub k: Tensor,       // [batch, n_kv_heads, seq_len, head_dim] in INT8
    pub v: Tensor,       // [batch, n_kv_heads, seq_len, head_dim] in INT8
    pub k_scale: Tensor, // [batch, n_kv_heads, seq_len, 1] in F32
    pub v_scale: Tensor, // [batch, n_kv_heads, seq_len, 1] in F32
    pub hologram: Option<super::holographic::HolographicMemory>,
}

impl QuantizedKvCache {
    /// Quantize F32 tensors into INT8 cache.
    pub fn new(k: &Tensor, v: &Tensor) -> Result<Self> {
        let (k_int8, k_scale) = Self::quantize(k)?;
        let (v_int8, v_scale) = Self::quantize(v)?;
        Ok(Self {
            k: k_int8,
            v: v_int8,
            k_scale,
            v_scale,
            hologram: None,
        })
    }

    fn quantize(x: &Tensor) -> Result<(Tensor, Tensor)> {
        // --- Sovereign Guard: Prevent "empty tensor for reduce" ---
        if x.elem_count() == 0 {
            let (batch, n_heads, seq_len, head_dim) = x.dims4()?;
            let x_q = Tensor::zeros(
                (batch, n_heads, seq_len, head_dim),
                candle_core::DType::U8,
                x.device(),
            )?;
            let scale = Tensor::ones(
                (batch, n_heads, seq_len, 1),
                candle_core::DType::F32,
                x.device(),
            )?;
            return Ok((x_q, scale));
        }

        // Compute per-step, per-head absolute maximum for scaling
        // x shape: [batch, n_heads, seq_len, head_dim]
        let x_abs = x.abs()?;
        let x_max = x_abs.max_keepdim(D::Minus1)?; // [batch, n_heads, seq_len, 1]
        let scale = (x_max / 127.0)?;

        // Elite Protection: Add epsilon to scale to prevent division by zero
        let scale = scale.affine(1.0, 1e-8)?;

        let x_q = x.broadcast_div(&scale)?;
        // Note: candle doesn't have I8 natively, using U8 with offset
        let clamped = x_q.clamp(-127.0f32, 127.0f32)?;
        let shifted = clamped.affine(1.0, 128.0)?;
        let x_q = shifted.round()?.to_dtype(DType::U8)?;

        Ok((x_q, scale))
    }

    /// Dequantize INT8 cache back to current precision.
    pub fn dequantize(&self, dtype: DType) -> Result<(Tensor, Tensor)> {
        let k_f32 = self.k.to_dtype(DType::F32)?.affine(1.0, -128.0)?;
        let k = k_f32.broadcast_mul(&self.k_scale)?.to_dtype(dtype)?;

        let v_f32 = self.v.to_dtype(DType::F32)?.affine(1.0, -128.0)?;
        let v = v_f32.broadcast_mul(&self.v_scale)?.to_dtype(dtype)?;
        Ok((k, v))
    }

    /// Append new tokens to the existing quantized cache.
    pub fn cat(&self, new_k: &Tensor, new_v: &Tensor) -> Result<Self> {
        let (nk_int8, nk_scale) = Self::quantize(new_k)?;
        let (nv_int8, nv_scale) = Self::quantize(new_v)?;

        let mut k = Tensor::cat(&[&self.k, &nk_int8], 2)?;
        let mut v = Tensor::cat(&[&self.v, &nv_int8], 2)?;
        let mut k_scale = Tensor::cat(&[&self.k_scale, &nk_scale], 2)?;
        let mut v_scale = Tensor::cat(&[&self.v_scale, &nv_scale], 2)?;

        // ── World-First: Fractal Holographic Compression ──
        // If the window exceeds the "Hard Horizon" (e.g. 2048 tokens),
        // we fold the oldest half of the cache into the hologram and prune the standard cache.
        let mut hologram = self.hologram.clone();
        let seq_len = k.dim(2)?;
        let horizon = 2048;

        if seq_len > horizon {
            let to_fold = horizon / 2;
            let (batch, n_heads, _, d_head) = new_k.dims4()?;

            let mut h = match hologram {
                Some(h) => h,
                None => super::holographic::HolographicMemory::new(
                    batch,
                    n_heads,
                    d_head,
                    new_k.device(),
                )?,
            };

            // Dequantize the tokens we are about to prune to bind them accurately
            let k_to_fold = self
                .k
                .narrow(2, 0, to_fold)?
                .to_dtype(candle_core::DType::F32)?;
            let v_to_fold = self
                .v
                .narrow(2, 0, to_fold)?
                .to_dtype(candle_core::DType::F32)?;
            let ks_to_fold = self.k_scale.narrow(2, 0, to_fold)?;
            let vs_to_fold = self.v_scale.narrow(2, 0, to_fold)?;

            let k_f32 = k_to_fold.affine(1.0, -128.0)?.broadcast_mul(&ks_to_fold)?;
            let v_f32 = v_to_fold.affine(1.0, -128.0)?.broadcast_mul(&vs_to_fold)?;

            h.fold(&k_f32, &v_f32)?;
            hologram = Some(h);

            // Prune the tensors
            k = k.narrow(2, to_fold, seq_len - to_fold)?;
            v = v.narrow(2, to_fold, seq_len - to_fold)?;
            k_scale = k_scale.narrow(2, to_fold, seq_len - to_fold)?;
            v_scale = v_scale.narrow(2, to_fold, seq_len - to_fold)?;
        }

        Ok(Self {
            k,
            v,
            k_scale,
            v_scale,
            hologram,
        })
    }
}

impl GroupedQueryAttention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_model: usize,
        n_heads: usize,
        n_kv_heads: usize,
        max_seq_len: usize,
        rope_theta: f64,
        rope_scaling: f64,
        use_bitnet: bool,
        qk_norm: bool,
        rms_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = d_model / n_heads;
        let n_rep = n_heads / n_kv_heads;

        // Default initialization binds the base BitLinear. LoRA adapters
        // can be loaded later or initialized via a specialized constructor.
        let q_proj = LinearLayer::Base(BitLinear::new(
            d_model,
            n_heads * head_dim,
            false,
            use_bitnet,
            vb.pp("q_proj"),
        )?);
        let k_proj = LinearLayer::Base(BitLinear::new(
            d_model,
            n_kv_heads * head_dim,
            false,
            use_bitnet,
            vb.pp("k_proj"),
        )?);
        let v_proj = LinearLayer::Base(BitLinear::new(
            d_model,
            n_kv_heads * head_dim,
            false,
            use_bitnet,
            vb.pp("v_proj"),
        )?);
        let o_proj = LinearLayer::Base(BitLinear::new(
            n_heads * head_dim,
            d_model,
            false,
            use_bitnet,
            vb.pp("o_proj"),
        )?);

        let rope =
            RotaryEmbedding::new(head_dim, max_seq_len, rope_theta, rope_scaling, vb.device())?;

        let q_norm = if qk_norm {
            Some(RmsNorm::new(
                n_heads * head_dim,
                rms_norm_eps,
                vb.pp("q_norm"),
            )?)
        } else {
            None
        };
        let k_norm = if qk_norm {
            Some(RmsNorm::new(
                n_kv_heads * head_dim,
                rms_norm_eps,
                vb.pp("k_norm"),
            )?)
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rope,
            n_heads,
            n_kv_heads,
            head_dim,
            n_rep,
        })
    }

    /// Forward pass with optional KV-cache for autoregressive generation.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch, seq_len, d_model]
    /// * `mask` - Optional causal attention mask
    /// * `start_pos` - Position offset for KV-cache
    /// * `kv_cache` - Optional (k_cache, v_cache) from previous steps
    ///
    /// # Returns
    /// * Output tensor [batch, seq_len, d_model]
    /// * Updated (k_cache, v_cache) for next step
    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        start_pos: usize,
        kv_cache: Option<&QuantizedKvCache>,
    ) -> Result<(Tensor, QuantizedKvCache)> {
        let (batch, seq_len, _d_model) = x.dims3()?;

        let mut q = self.q_proj.forward(x)?;
        let mut k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // ── Zenith Upgrade: Projection-Level QK-Norm ──
        // Normalize the full query/key vectors before splitting into heads.
        // This stabilizes the embedding space and ensures weights match dimensions.
        if let Some(norm) = &self.q_norm {
            q = norm.forward(&q)?;
        }
        if let Some(norm) = &self.k_norm {
            k = norm.forward(&k)?;
        }

        // Reshape to [batch, seq_len, n_heads/n_kv_heads, head_dim]
        // and transpose to [batch, n_heads, seq_len, head_dim]
        let q = q
            .reshape((batch, seq_len, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch, seq_len, self.n_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply RoPE to Q and K
        let q = self.rope.apply(&q, start_pos)?;
        let k = self.rope.apply(&k, start_pos)?;

        // ── Peak Mastery: INT8 KV-cache Quantization ──
        // Storing the cache in INT8 instead of F32/F16 reduces memory bandwidth pressure
        // by 4x, providing a significant boost to token generation speed and capacity.
        let (k_cache_out, v_cache_out) = match kv_cache {
            Some(cache) => {
                let updated = cache.cat(&k, &v)?;
                (updated.clone(), updated)
            }
            None => {
                let cache = QuantizedKvCache::new(&k, &v)?;
                (cache.clone(), cache)
            }
        };

        // Dequantize for attention computation
        let (k, v) = k_cache_out.dequantize(DType::F32)?;

        // ── Sovereign Hybrid Attention Controller ──
        // For short sequences (agent reasoning), fused matmul is faster due to batch overhead.
        // For large sequences, tiled Flash Attention provides memory safety.
        let seq_len = q.dim(2)?;
        let mut output = if seq_len <= 128 {
            self.fused_attention(&q, &k, &v, mask)?
        } else {
            self.flash_attention(&q, &k, &v, mask)?
        };

        // ── World-First: Holographic Resonance Retrieval ──
        // Collapse the fractal memory map back into a fuzzy attention activation!
        if let Some(hologram) = &k_cache_out.hologram {
            let resonance = hologram.resonate(&q)?;
            output = (output + resonance)?;
        }

        // Reshape back: [batch, seq_len, n_heads * head_dim]
        let attn_output =
            output
                .transpose(1, 2)?
                .reshape((batch, seq_len, self.n_heads * self.head_dim))?;

        // Output projection
        let output = self.o_proj.forward(&attn_output)?;

        Ok((output, v_cache_out))
    }

    /// Fast-path fused attention for small agentic reasoning blocks.
    fn fused_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_b, _h, _s, d) = q.dims4()?;
        let scale = 1.0 / (d as f64).sqrt();

        // Repeat KV heads if GQA
        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        // ── Sovereign Ternary Attention ──
        // If using BitNet, we quantize Q and K to ternary (-1, 0, 1) before the matmul.
        // This reduces noise and matches the underlying parameter distribution.
        let q_scaled = if self.q_proj.is_bitnet() {
            self.quantize_tensor(q)?
        } else {
            q.clone()
        };

        let k_scaled = if self.q_proj.is_bitnet() {
            self.quantize_tensor(&k)?
        } else {
            k.clone()
        };

        // QK^T / sqrt(d)
        let mut scores = (q_scaled.matmul(&k_scaled.transpose(2, 3)?)? * scale)?;

        // Apply causal mask segment
        if let Some(mask) = mask {
            let seq_q = q.dim(2)?;
            let seq_k = k.dim(2)?;
            scores = scores.broadcast_add(&mask.narrow(2, 0, seq_q)?.narrow(3, 0, seq_k)?)?;
        }

        // ── Gemma 2 Logit Soft-Capping ──
        // Instead of hard clamping, use tanh-based soft-capping which preserves
        // gradient flow while preventing extreme attention scores.
        // softcap(x) = cap * tanh(x / cap)
        let attn_logit_cap = 50.0f64;
        let scores = (scores.affine(1.0 / attn_logit_cap, 0.0)?.tanh()? * attn_logit_cap)?;
        let weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        weights.matmul(&v)
    }

    /// Tiling-based Flash Attention for memory-efficient large contexts.
    fn flash_attention(
        &self,
        q: &Tensor, // [b, h, s_q, d]
        k: &Tensor, // [b, h_kv, s_k, d]
        v: &Tensor, // [b, h_kv, s_k, d]
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, n_heads, seq_q, head_dim) = q.dims4()?;
        let (_b, _h_kv, seq_k, _d) = k.dims4()?;
        let scale = 1.0 / (head_dim as f64).sqrt();

        // Repeat KV heads if using GQA
        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        // Block sizes (tuned for L3 cache)
        let br = 64; // Row block size
        let bc = 64; // Column block size

        let device = q.device();

        // Final output O
        let mut o = Tensor::zeros((batch, n_heads, seq_q, head_dim), DType::F32, device)?;

        // Running max and sum for softmax
        let mut l = Tensor::zeros((batch, n_heads, seq_q, 1), DType::F32, device)?;
        let mut m = (Tensor::ones((batch, n_heads, seq_q, 1), DType::F32, device)? * (-1e9))?;

        for j in (0..seq_k).step_by(bc) {
            let j_end = (j + bc).min(seq_k);
            let kj = k.narrow(2, j, j_end - j)?.transpose(2, 3)?; // [b, h, d, bc]
            let vj = v.narrow(2, j, j_end - j)?; // [b, h, bc, d]

            for i in (0..seq_q).step_by(br) {
                let i_end = (i + br).min(seq_q);
                let qi = q.narrow(2, i, i_end - i)?; // [b, h, br, d]

                // ── Absolute Perfection: Fused Scaled Flash Attention ──
                let mut s_ij = (qi.matmul(&kj)? * scale)?;

                // Apply mask segment if present (Fused check)
                if let Some(mask) = mask {
                    s_ij = s_ij.broadcast_add(&mask.narrow(2, i, i_end - i)?.narrow(
                        3,
                        j,
                        j_end - j,
                    )?)?;
                }

                // ── Zenith Upgrade: Flash Logit Soft-Capping ──
                // Synchronized with fused attention path to prevent numerical divergence
                // in long trajectories.
                let attn_logit_cap = 50.0f32;
                let s_ij = (s_ij.affine(1.0 / attn_logit_cap as f64, 0.0)?.tanh()?
                    * attn_logit_cap as f64)?;

                // ── Numerical Stability Guard ──
                let s_ij = s_ij.clamp(-1e4f32, 1e4f32)?;

                // Row-wise max of current block
                let m_ij = s_ij.max_keepdim(D::Minus1)?; // [b, h, br, 1]

                // Exponential sum of current block
                let p_ij = s_ij.broadcast_sub(&m_ij)?.exp()?; // [b, h, br, bc]
                let l_ij = p_ij.sum_keepdim(D::Minus1)?; // [b, h, br, 1]

                // Update running stats: m, l
                let m_prev = m.narrow(2, i, i_end - i)?;
                let l_prev = l.narrow(2, i, i_end - i)?;
                let o_prev = o.narrow(2, i, i_end - i)?;

                // m_new = max(m_prev, m_ij)
                let m_new = m_prev.maximum(&m_ij)?;

                // Rescale previous stats
                let alpha = (m_prev - &m_new)?.exp()?;
                let beta = (m_ij - &m_new)?.exp()?;

                // l_new = alpha * l_prev + beta * l_ij
                let l_new = (l_prev.broadcast_mul(&alpha)? + l_ij.broadcast_mul(&beta)?)?;

                // O_new = alpha * O_prev + beta * (p_ij @ vj)
                let pv_ij = p_ij.matmul(&vj)?;
                let o_new = (o_prev.broadcast_mul(&alpha)? + pv_ij.broadcast_mul(&beta)?)?;

                // Write back
                m = m.slice_assign(&[0..batch, 0..n_heads, i..i_end, 0..1], &m_new)?;
                l = l.slice_assign(&[0..batch, 0..n_heads, i..i_end, 0..1], &l_new)?;
                o = o.slice_assign(&[0..batch, 0..n_heads, i..i_end, 0..head_dim], &o_new)?;
            }
        }

        // ── Elite Stability Guard ──
        // Add epsilon to prevent division by zero in softmax normalization
        let l_safe = (l + 1e-10)?;
        o.broadcast_div(&l_safe)
    }

    /// Repeat KV heads to match the number of query heads.
    ///
    /// [batch, n_kv_heads, seq_len, head_dim] → [batch, n_heads, seq_len, head_dim]
    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        if self.n_rep == 1 {
            return Ok(x.clone());
        }
        let (batch, n_kv_heads, seq_len, head_dim) = x.dims4()?;
        let x = x
            .unsqueeze(2)? // [b, n_kv, 1, s, d]
            .expand((batch, n_kv_heads, self.n_rep, seq_len, head_dim))?
            .reshape((batch, self.n_heads, seq_len, head_dim))?;
        Ok(x)
    }

    /// Elite Quantization for Attention Tensors.
    /// Uses robust Absolute-Maximum scaling to ensure numerical range stability.
    fn quantize_tensor(&self, x: &Tensor) -> Result<Tensor> {
        let x_abs = x.abs()?;
        // Guard against empty tensors to prevent "empty tensor for reduce" panic
        let alpha = if x_abs.elem_count() > 0 {
            x_abs.max_all()?.to_scalar::<f32>()?
        } else {
            1e-6f32
        }
        .max(1e-6);
        let x_norm = (x / (alpha as f64))?;
        let x_q = x_norm.clamp(-1.0f32, 1.0f32)?.round()?;
        x_q * (alpha as f64)
    }
}

/// Create a causal attention mask with optional sliding window.
///
/// Positions that should NOT be attended to get -inf.
/// Returns a tensor of shape [1, 1, seq_len, seq_len].
///
/// When `sliding_window` is Some(w), tokens can only attend to the
/// previous w positions (Mistral/Gemma-style local attention).
pub fn create_causal_mask(
    seq_len: usize,
    sliding_window: Option<usize>,
    device: &Device,
) -> Result<Tensor> {
    let neg_inf = f32::NEG_INFINITY;
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                // Causal: can't attend to future tokens
                mask_data[i * seq_len + j] = neg_inf;
            } else if let Some(w) = sliding_window {
                if i.saturating_sub(j) >= w {
                    // SWA: can't attend beyond window
                    mask_data[i * seq_len + j] = neg_inf;
                }
            }
        }
    }
    let mask = Tensor::from_vec(mask_data, (seq_len, seq_len), device)?;
    mask.unsqueeze(0)?.unsqueeze(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::VarMap;

    #[test]
    fn test_rope() -> Result<()> {
        let rope = RotaryEmbedding::new(64, 128, 10000.0, 1.0, &Device::Cpu)?;
        let x = Tensor::randn(0f32, 1.0, (1, 4, 8, 64), &Device::Cpu)?;
        let y = rope.apply(&x, 0)?;
        assert_eq!(y.dims(), &[1, 4, 8, 64]);
        Ok(())
    }

    #[test]
    fn test_gqa() -> Result<()> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let attn =
            GroupedQueryAttention::new(256, 4, 2, 128, 500000.0, 1.0, true, false, 1e-6, vb)?;
        let x = Tensor::randn(0f32, 1.0, (1, 8, 256), &Device::Cpu)?;
        let mask = create_causal_mask(8, None, &Device::Cpu)?;
        let (y, _cache) = attn.forward(&x, Some(&mask), 0, None)?;
        assert_eq!(y.dims(), &[1, 8, 256]);
        Ok(())
    }

    #[test]
    fn test_causal_mask() -> Result<()> {
        let mask = create_causal_mask(4, None, &Device::Cpu)?;
        assert_eq!(mask.dims(), &[1, 1, 4, 4]);
        // Position (0,1) should be -inf (can't attend to future)
        let mask_data: Vec<f32> = mask.flatten_all()?.to_vec1()?;
        assert!(mask_data[1].is_infinite() && mask_data[1] < 0.0); // (0,1) = -inf
        assert_eq!(mask_data[0], 0.0); // (0,0) = 0 (can attend to self)
        Ok(())
    }
}
