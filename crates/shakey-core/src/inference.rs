//! Inference engine — autoregressive text generation with KV-cache and sampling.
//!
//! Supports:
//! - Temperature scaling
//! - Top-k sampling
//! - Top-p (nucleus) sampling
//! - Repetition penalty
//! - KV-cache for O(n) generation instead of O(n²)

use super::metrics::SovereignMetrics;
use super::model::transformer::TransformerModel;
use super::tokenizer::Tokenizer;
use candle_core::{Device, Result, Tensor, D};
use std::time::Instant;

/// Sampling parameters for text generation.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Temperature (higher = more random, lower = more deterministic)
    pub temperature: f64,
    /// Top-k: only sample from the k highest-probability tokens
    pub top_k: usize,
    /// Top-p: only sample from tokens whose cumulative probability ≤ p
    pub top_p: f64,
    /// Repetition penalty (1.0 = no penalty)
    pub repetition_penalty: f64,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Stop generation at EOS token
    pub stop_at_eos: bool,
    /// Enable "Industry-Grade" Internal Reasoning (Hidden Thought Chain)
    pub reasoning: bool,
    /// Medusa confidence threshold (0.0 - 1.0). Higher = more accurate, lower = faster.
    pub medusa_confidence_threshold: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            repetition_penalty: 1.1,
            max_tokens: 1024,
            stop_at_eos: true,
            reasoning: true,
            medusa_confidence_threshold: 0.4,
        }
    }
}

/// Inference engine for autoregressive text generation.
pub struct InferenceEngine<'a> {
    model: &'a TransformerModel,
    tokenizer: &'a Tokenizer,
    device: Device,
}

impl<'a> InferenceEngine<'a> {
    pub fn new(model: &'a TransformerModel, tokenizer: &'a Tokenizer, device: Device) -> Self {
        Self {
            model,
            tokenizer,
            device,
        }
    }

    /// Generate text from a prompt.
    ///
    /// Uses KV-cache for efficient autoregressive generation:
    /// 1. Prefill: process entire prompt in parallel, build KV-cache
    /// 2. Decode: generate tokens one-by-one, appending to KV-cache
    pub fn generate(&self, prompt: &str, params: &SamplingParams) -> Result<String> {
        let mut output = String::new();
        self.generate_stream(prompt, params, |token| {
            output.push_str(&token);
            Ok(())
        })?;
        Ok(output)
    }

    /// Generate text from a prompt with streaming support.
    ///
    /// Calls the provided callback for each decoded token as it is generated.
    pub fn generate_stream<F>(
        &self,
        prompt: &str,
        params: &SamplingParams,
        mut callback: F,
    ) -> Result<()>
    where
        F: FnMut(String) -> Result<()>,
    {
        // Tokenize prompt
        let prompt_ids = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

        let eos_id = self.tokenizer.special_tokens().eos_id;

        // Convert to tensor: [1, seq_len]
        let input_ids = Tensor::new(prompt_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let start_time = Instant::now();

        // ── Prefill Phase ──
        // Process entire prompt at once, get initial INT8 KV-cache
        let forward_out =
            self.model
                .forward(&input_ids, 0, None, Some(params.temperature as f32))?;
        let logits = forward_out.logits;
        let kv_caches = forward_out.kv_caches;

        // Sample first token from last position's logits
        let last_logits = logits.narrow(1, logits.dim(1)? - 1, 1)?.squeeze(1)?;
        let mut generated_ids = prompt_ids.clone();
        let mut next_token = self.sample_token(&last_logits, &generated_ids, params)?;
        generated_ids.push(next_token);

        // Decode and stream the first token
        let first_token_text = self
            .tokenizer
            .decode(&[next_token])
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        callback(first_token_text)?;

        let mut pos = prompt_ids.len();

        // ── Decode Phase with Medusa Speculative Verification ──
        // Generate multiple tokens per step if Medusa heads are confident.
        //
        // ELITE OPTIMIZATION: KV-cache is passed by reference/ownership
        // to avoid expensive memory copies.
        let mut current_kv_caches = kv_caches;
        let mut current_medusa = forward_out.medusa_logits;
        let mut in_thought = false;
        let mut thought_buffer = Vec::new();

        while generated_ids.len() < params.max_tokens {
            if params.stop_at_eos && next_token == eos_id {
                break;
            }

            // 1. Get Speculative Candidates from Medusa heads (Dynamic Window)
            let mut candidates = Vec::new();
            let mut used_medusa = false;

            if let Some(ref m_logits) = current_medusa {
                let last_m = m_logits.narrow(2, m_logits.dim(2)? - 1, 1)?;

                // ELITE: Peak Confidence Analysis
                // We dynamically adjust how many candidates to draft based on the
                // entropy of the first Medusa head.
                for h in 0..m_logits.dim(1)? {
                    let h_logits = last_m.narrow(1, h, 1)?.squeeze(0)?.squeeze(0)?.squeeze(0)?;
                    let probs = candle_nn::ops::softmax(&h_logits, D::Minus1)?;

                    // Guard against empty tensors to prevent "empty tensor for reduce" panic
                    let max_prob = if probs.elem_count() > 0 {
                        probs.max(D::Minus1)?.to_vec0::<f32>()?
                    } else {
                        0.0f32
                    };

                    // Progressive confidence gate: as we go deeper into the
                    // speculation, we require higher confidence.
                    let gate = params.medusa_confidence_threshold + (h as f32 * 0.1);
                    if max_prob > gate && h_logits.elem_count() > 0 {
                        let cand = h_logits.argmax(0)?.to_vec0::<u32>()?;
                        candidates.push(cand);
                        used_medusa = true;
                    } else {
                        break; // Stop drafting when confidence drops
                    }
                }
            }

            // ── Lookahead Decoding / N-Gram Jacobi Fallback ──
            // If Medusa heads aren't confident (or exhausted), search the generation
            // history for an exact structural match and speculatively draft upcoming tokens.
            // This is "free" speculation without extra neural network passes.
            if !used_medusa && generated_ids.len() >= 4 {
                let n_gram_len = 3;
                let current_gram = &generated_ids[generated_ids.len() - n_gram_len..];

                // Search backwards for the same n-gram
                for i in (0..generated_ids.len() - n_gram_len).rev() {
                    if &generated_ids[i..i + n_gram_len] == current_gram {
                        // Match found! Draft the next 3 tokens from history
                        let draft_len = 3.min(generated_ids.len() - (i + n_gram_len));
                        if draft_len > 0 {
                            candidates.extend_from_slice(
                                &generated_ids[i + n_gram_len..i + n_gram_len + draft_len],
                            );
                        }
                        break;
                    }
                }
            }

            // 2. Form a "Speculative Block" for verification
            let mut spec_block = vec![next_token];
            spec_block.extend(&candidates);

            let block_len = spec_block.len();
            let block_tensor = Tensor::new(spec_block.as_slice(), &self.device)?.unsqueeze(0)?;

            // 3. One forward pass to verify all candidates at once
            // ELITE: Pass ownership of INT8 KV-cache — ultra-fast transfer
            let spec_out = self.model.forward(
                &block_tensor,
                pos,
                Some(&current_kv_caches),
                Some(params.temperature as f32),
            )?;
            let spec_logits = spec_out.logits; // [1, block_len, vocab]

            // 4. Verification Logic: Accept candidates if they match base model's argmax
            let mut accepted_count = 1; // Always accept the first token (standard prediction)
            for i in 0..(block_len - 1) {
                let base_pred_logits = spec_logits.narrow(1, i, 1)?.squeeze(1)?.squeeze(0)?;

                // Guard against empty tensors to prevent "empty tensor for reduce" panic
                if base_pred_logits.elem_count() == 0 {
                    break;
                }
                let base_pred = base_pred_logits.argmax(0)?.to_vec0::<u32>()?;

                if base_pred == spec_block[i + 1] {
                    accepted_count += 1;
                } else {
                    next_token = base_pred;
                    break;
                }
            }

            // 5. Update state with accepted tokens
            for (i, &token) in spec_block.iter().take(accepted_count).enumerate() {
                if i > 0 {
                    generated_ids.push(token);
                    let text = self
                        .tokenizer
                        .decode(&[token])
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

                    // --- Peak Mastery: Reasoning Controller ---
                    if params.reasoning {
                        if text.contains("<thought>") {
                            in_thought = true;
                        }
                        if in_thought {
                            thought_buffer.push(text.clone());
                            if text.contains("</thought>") {
                                in_thought = false;
                            }
                            // We don't send thought tokens to the primary callback to keep output clean
                        } else {
                            callback(text)?;
                        }
                    } else {
                        callback(text)?;
                    }
                }

                if params.stop_at_eos && token == eos_id {
                    return Ok(());
                }
            }

            // 6. Final next_token for the next cycle
            if accepted_count == block_len {
                let last_logits = spec_logits
                    .narrow(1, block_len - 1, 1)?
                    .squeeze(1)?
                    .squeeze(0)?;
                if last_logits.elem_count() > 0 {
                    next_token = last_logits.argmax(0)?.to_vec0::<u32>()?;
                }
            }

            // ELITE: Take ownership — no clone
            current_kv_caches = spec_out.kv_caches;
            current_medusa = spec_out.medusa_logits;
            pos += accepted_count;

            if accepted_count == 0 {
                break;
            }
        }

        let duration = start_time.elapsed();
        SovereignMetrics::global().record_tokens(
            (generated_ids.len() - prompt_ids.len()) as u64,
            duration.as_millis() as u64,
        );

        Ok(())
    }

    /// Sample a single token from logits using the configured strategy.
    fn sample_token(
        &self,
        logits: &Tensor,
        past_tokens: &[u32],
        params: &SamplingParams,
    ) -> Result<u32> {
        // logits shape: [1, vocab_size] or [vocab_size]
        let logits = if logits.dims().len() > 1 {
            logits.squeeze(0)?
        } else {
            logits.clone()
        };

        // Apply repetition penalty
        let logits = if params.repetition_penalty != 1.0 {
            self.apply_repetition_penalty(&logits, past_tokens, params.repetition_penalty)?
        } else {
            logits
        };

        // Temperature scaling
        let logits = if params.temperature != 1.0 && params.temperature > 0.0 {
            (logits / params.temperature)?
        } else {
            logits
        };

        // Greedy decoding if temperature ≈ 0
        if params.temperature < 1e-6 {
            let token = logits.argmax(D::Minus1)?;
            return token.to_vec0::<u32>();
        }

        // Convert to probabilities
        let probs = candle_nn::ops::softmax(&logits, D::Minus1)?;
        let probs_vec: Vec<f32> = probs.to_vec1()?;

        // Top-k filtering
        let mut indexed_probs: Vec<(usize, f32)> =
            probs_vec.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Keep only top-k
        let top_k = params.top_k.min(indexed_probs.len());
        indexed_probs.truncate(top_k);

        // Top-p filtering
        if params.top_p < 1.0 {
            let mut cumulative = 0.0;
            let mut cutoff_idx = indexed_probs.len();
            for (i, (_, p)) in indexed_probs.iter().enumerate() {
                cumulative += p;
                if cumulative >= params.top_p as f32 {
                    cutoff_idx = i + 1;
                    break;
                }
            }
            indexed_probs.truncate(cutoff_idx);
        }

        // Renormalize
        let total: f32 = indexed_probs.iter().map(|(_, p)| p).sum();
        for (_, p) in &mut indexed_probs {
            *p /= total;
        }

        // Sample from the filtered distribution
        let mut rng_val: f32 = rand::random();
        for (idx, prob) in &indexed_probs {
            rng_val -= prob;
            if rng_val <= 0.0 {
                return Ok(*idx as u32);
            }
        }

        // Fallback: return the most probable token
        if indexed_probs.is_empty() {
            return Ok(0); // Return PAD/UNK as last resort
        }
        Ok(indexed_probs[0].0 as u32)
    }

    /// Apply repetition penalty to logits.
    fn apply_repetition_penalty(
        &self,
        logits: &Tensor,
        past_tokens: &[u32],
        penalty: f64,
    ) -> Result<Tensor> {
        if past_tokens.is_empty() {
            return Ok(logits.clone());
        }

        let mut unique_tokens = past_tokens.to_vec();
        unique_tokens.sort_unstable();
        unique_tokens.dedup();

        let indices = Tensor::new(unique_tokens.as_slice(), logits.device())?;
        let selected = logits.index_select(&indices, 0)?;
        let selected_vec = selected.to_vec1::<f32>()?;

        let mut deltas = Vec::with_capacity(selected_vec.len());
        for &val in &selected_vec {
            let new_val = if val > 0.0 {
                val / (penalty as f32)
            } else {
                val * (penalty as f32)
            };
            deltas.push(new_val - val);
        }

        let delta_tensor = Tensor::from_vec(deltas, selected.shape(), logits.device())?;
        logits.index_add(&indices, &delta_tensor, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampling_params_default() {
        let params = SamplingParams::default();
        assert!(params.temperature > 0.0);
        assert!(params.top_k > 0);
        assert!(params.top_p > 0.0 && params.top_p <= 1.0);
    }
}
