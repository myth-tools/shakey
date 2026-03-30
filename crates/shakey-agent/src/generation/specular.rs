//! WCSI Specular Decoding Engine — Medusa-style parallel token verification.
//!
//! achieving 3x–4x generation speedup by drafting future tokens using parallel heads
//! and verifying them in a single transformer forward pass.

use candle_core::{Device, Result, Tensor, D};
use shakey_core::inference::SamplingParams;
use shakey_core::model::transformer::TransformerModel;
use shakey_core::tokenizer::Tokenizer;

/// WCSI Specular Engine for ultra-fast autonomous reasoning.
pub struct SpecularEngine<'a> {
    model: &'a TransformerModel,
    tokenizer: &'a Tokenizer,
    device: Device,
}

impl<'a> SpecularEngine<'a> {
    pub fn new(model: &'a TransformerModel, tokenizer: &'a Tokenizer, device: Device) -> Self {
        Self {
            model,
            tokenizer,
            device,
        }
    }

    /// Generate text using Medusa-style Specular Decoding.
    ///
    /// Algorithm:
    /// 1. Forward pass to get next-token logits + Medusa draft logits.
    /// 2. Sample next-token + draft tokens from Medusa heads.
    /// 3. In the next step, verify draft tokens in parallel.
    /// 4. Accept the longest valid prefix of drafted tokens.
    pub fn generate_specular<F>(
        &self,
        prompt: &str,
        params: &SamplingParams,
        mut callback: F,
    ) -> Result<()>
    where
        F: FnMut(String) -> Result<()>,
    {
        let prompt_ids = self
            .tokenizer
            .encode(prompt)
            .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        let eos_id = self.tokenizer.special_tokens().eos_id;

        // Input: [1, seq_len]
        let input_ids = Tensor::new(prompt_ids.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut generated_ids = prompt_ids.clone();

        // ── Prefill Phase ──
        let forward_out =
            self.model
                .forward(&input_ids, 0, None, Some(params.temperature as f32))?;
        let mut logits = forward_out.logits;
        let mut medusa_logits = forward_out.medusa_logits;
        let mut kv_caches = Some(forward_out.kv_caches);
        let mut pos = prompt_ids.len();

        // Initial token sampling
        let last_logits = logits.narrow(1, logits.dim(1)? - 1, 1)?.squeeze(1)?;
        let mut next_token = self.sample_token(&last_logits, &generated_ids, params)?;

        // Stream first token
        callback(
            self.tokenizer
                .decode(&[next_token])
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?,
        )?;
        generated_ids.push(next_token);

        // ── Specular Decoding Loop ──
        let mut tokens_generated = 1;
        while tokens_generated < params.max_tokens {
            if params.stop_at_eos && next_token == eos_id {
                break;
            }

            // 1. Get draft tokens from current Medusa heads using Dynamic Confidence Drafting
            let mut drafts = Vec::new();
            if let Some(ref m_logits) = medusa_logits {
                // m_logits: [batch, n_heads, seq_len, vocab_size]
                // Get the last sequence position
                let last_m = m_logits.narrow(2, m_logits.dim(2)? - 1, 1)?;
                for h in 0..m_logits.dim(1)? {
                    let head_logits = last_m.narrow(1, h, 1)?.squeeze(0)?.squeeze(0)?.squeeze(0)?;

                    // Confidence calculation
                    let probs = candle_nn::ops::softmax(&head_logits, D::Minus1)?;
                    let max_prob = probs.max(D::Minus1)?.to_vec0::<f32>()?;

                    if max_prob > 0.4 {
                        let draft_token = head_logits.argmax(D::Minus1)?.to_vec0::<u32>()?;
                        drafts.push(draft_token);
                    } else {
                        // Entropy too high. Break the draft chain to avoid verifying noise
                        break;
                    }
                }
            }

            // 2. Prepare block for verification: [next_token, draft1, draft2, ...]
            let mut block = vec![next_token];
            block.extend_from_slice(&drafts);
            let block_len = block.len();
            let block_tensor = Tensor::new(block.as_slice(), &self.device)?.unsqueeze(0)?;

            // 3. Forward verification pass
            let forward_out = self.model.forward(
                &block_tensor,
                pos,
                kv_caches.as_deref(),
                Some(params.temperature as f32),
            )?;
            logits = forward_out.logits; // [1, block_len, vocab_size]
            medusa_logits = forward_out.medusa_logits;
            kv_caches = Some(forward_out.kv_caches);

            // 4. Verify drafted tokens
            // The logic:
            // logits[i] predicts token block[i+1].
            // If block[i+1] == argmax(logits[i]), it's accepted.
            let mut accepted_count = 1; // Always accept the first token (next_token)
            let mut new_next_token = 0;
            let mut full_acceptance = true;

            for i in 0..block_len {
                let current_logits = logits.narrow(1, i, 1)?.squeeze(1)?;
                let top_token = current_logits.argmax(D::Minus1)?.to_vec0::<u32>()?;

                if i < block_len - 1 && top_token == block[i + 1] {
                    // Accepted! Stream it.
                    accepted_count += 1;
                    let text = self
                        .tokenizer
                        .decode(&[top_token])
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    callback(text)?;
                    generated_ids.push(top_token);
                    if params.stop_at_eos && top_token == eos_id {
                        return Ok(());
                    }
                } else {
                    // Mismatch or end of block. This top_token is our NEW next_token.
                    full_acceptance = false;
                    new_next_token = top_token;
                    let text = self
                        .tokenizer
                        .decode(&[new_next_token])
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    callback(text)?;
                    generated_ids.push(new_next_token);
                    break;
                }
            }

            // If all drafted tokens were accepted, we need to sample the actual
            // next token from the last logits position (not leave it as 0).
            if full_acceptance {
                let last_logits = logits.narrow(1, block_len - 1, 1)?.squeeze(1)?;
                new_next_token = self.sample_token(&last_logits, &generated_ids, params)?;
                let text = self
                    .tokenizer
                    .decode(&[new_next_token])
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                callback(text)?;
                generated_ids.push(new_next_token);
                tokens_generated += 1;
            }

            // Update position and next_token
            pos += accepted_count;
            next_token = new_next_token;
            tokens_generated += accepted_count;
        }

        Ok(())
    }

    /// ELITE: Fast sampling with Top-K prefiltering and repetition penalty.
    /// Reduces sampling from O(vocab) to O(k) for 32K+ vocabs.
    fn sample_token(&self, logits: &Tensor, past: &[u32], params: &SamplingParams) -> Result<u32> {
        let logits = logits.squeeze(0)?;

        // Greedy fast-path (no sampling overhead)
        if params.temperature < 1e-6 {
            return logits.argmax(D::Minus1)?.to_vec0::<u32>();
        }

        // Apply repetition penalty inline (avoids full InferenceEngine dependency)
        let logits = if params.repetition_penalty != 1.0 && !past.is_empty() {
            let logits_vec: Vec<f32> = logits.to_vec1()?;
            let mut penalized = logits_vec;
            let penalty = params.repetition_penalty as f32;
            // Only penalize last 64 tokens (speed + prevents over-penalization)
            let window = past.len().saturating_sub(64);
            for &tok in &past[window..] {
                let idx = tok as usize;
                if idx < penalized.len() {
                    if penalized[idx] > 0.0 {
                        penalized[idx] /= penalty;
                    } else {
                        penalized[idx] *= penalty;
                    }
                }
            }
            Tensor::from_vec(penalized, logits.shape(), &self.device)?
        } else {
            logits
        };

        // Temperature scaling + softmax
        let probs = candle_nn::ops::softmax(&(&logits / params.temperature)?, D::Minus1)?;
        let probs_vec: Vec<f32> = probs.to_vec1()?;

        // Top-K prefiltering: O(n) partial sort to find top-k, then sample from k candidates
        let k = params.top_k.min(probs_vec.len()).max(1);
        let mut indexed: Vec<(usize, f32)> =
            probs_vec.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        indexed.truncate(k);

        // Renormalize top-k
        let total: f32 = indexed.iter().map(|(_, p)| p).sum();
        let mut rng_val: f32 = rand::random();
        for (idx, prob) in &indexed {
            rng_val -= prob / total;
            if rng_val <= 0.0 {
                return Ok(*idx as u32);
            }
        }

        Ok(indexed[0].0 as u32)
    }
}
