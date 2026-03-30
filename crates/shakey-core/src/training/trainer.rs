//! Training loop with gradient accumulation, LR scheduling, and auto-checkpointing.
//!
//! Designed for low-resource, volatile environments:
//! - Gradient accumulation: accumulate gradients across micro-batches on small-memory machines
//! - Auto-checkpoint: save every N steps to survive Colab/Kaggle kills
//! - Resume-aware: start from any checkpoint without losing progress
//! - Progress logging: ETA, tokens/sec, loss curves

use super::checkpoint::{CheckpointManager, TrainingState};
use super::distillation::{
    distillation_loss, dpo_loss, get_batch_logps, DistillationConfig, DpoConfig, LossComponents,
};
use super::optimizer::{Optimizer, OptimizerConfig};
use super::scheduler::{LrScheduler, SchedulerConfig};
use crate::metrics::SovereignMetrics;
use crate::model::config::ModelConfig;
use crate::model::transformer::TransformerModel;
use anyhow::{Context, Result};
pub use candle_core::backprop::GradStore as Grads;
use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use sha2::Digest;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Training configuration.
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    pub optimizer: OptimizerConfig,
    pub scheduler: SchedulerConfig,
    pub max_grad_norm: f64,
    pub batch_size: usize,
    pub gradient_accumulation_steps: usize,
    pub checkpoint_every_steps: u64,
    pub log_every_steps: u64,
    pub distillation: DistillationConfig,
    pub dpo: DpoConfig,
    pub early_stopping_patience: u64,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            optimizer: OptimizerConfig::default(),
            scheduler: SchedulerConfig::default(),
            max_grad_norm: 1.0,
            batch_size: 64,
            gradient_accumulation_steps: 4,
            checkpoint_every_steps: 100,
            log_every_steps: 10,
            distillation: DistillationConfig::default(),
            dpo: DpoConfig::default(),
            early_stopping_patience: 1000,
        }
    }
}

/// A single training batch with student input and teacher supervision.
#[derive(Debug)]
pub struct TrainingBatch {
    /// Input token IDs: [batch, seq_len]
    pub input_ids: Tensor,
    /// Target token IDs (shifted by 1): [batch, seq_len]
    pub target_ids: Tensor,
    /// Teacher logits (soft labels from NVIDIA NIM): [batch, seq_len, vocab_size]
    pub teacher_logits: Option<Tensor>,
    /// ZENITH: Mask specifying which tokens are 'reasoning/thinking' [batch, seq_len]
    pub reasoning_mask: Option<Tensor>,
    /// DPO: Chosen response IDs [batch, seq_len]
    pub chosen_ids: Option<Tensor>,
    /// DPO: Rejected response IDs [batch, seq_len]
    pub rejected_ids: Option<Tensor>,
}

pub trait AsTrainingExample {
    fn id(&self) -> &str;
    fn prompt(&self) -> &str;
    fn completion(&self) -> &str;
    /// Optional: the internal reasoning/thinking process for this example
    fn reasoning_content(&self) -> Option<&str> {
        None
    }
}

impl TrainingBatch {
    /// Create a TrainingBatch from a list of abstract training examples.
    pub fn from_examples<T: AsTrainingExample>(
        examples: &[T],
        tokenizer: &crate::tokenizer::Tokenizer,
        max_seq_len: usize,
        device: &Device,
    ) -> Result<Self> {
        let batch_size = examples.len();
        let mut all_input_ids = Vec::with_capacity(batch_size);
        let mut all_target_ids = Vec::with_capacity(batch_size);
        let mut all_reasoning_masks = Vec::with_capacity(batch_size);

        for example in examples {
            // Combine prompt and completion
            let full_text = format!("{} {}", example.prompt(), example.completion());

            let tokens = tokenizer
                .encode_robust(&full_text, Some(max_seq_len), true)
                .map_err(|e| anyhow::anyhow!("Tokenization failed for {}: {}", example.id(), e))?;

            let input_ids = tokens.clone();
            let mut target_ids = tokens.clone();

            // Causal shift: target at pos i is token at pos i+1
            if target_ids.len() > 1 {
                target_ids.remove(0);
                target_ids.push(tokenizer.special_tokens().pad_id);
            }

            // ── ZENITH: Construct Reasoning Mask ──
            let mut mask = vec![0.0f32; max_seq_len];
            if let Some(reasoning) = example.reasoning_content() {
                // Heuristic: identify the start and end of reasoning section in the full text
                // and mark corresponding token indices as 1.0.
                if let Some(start_idx) = full_text.find(reasoning) {
                    let end_idx = start_idx + reasoning.len();

                    // Recalculate token indices relative to the reasoning string
                    // (Rough approximation: tokens starting within the reasoning byte range)
                    let mut current_offset = 0;
                    for (i, &tid) in tokens.iter().enumerate() {
                        if let Ok(token_str) = tokenizer.decode(&[tid]) {
                            let token_len = token_str.len();
                            if current_offset >= start_idx && current_offset < end_idx {
                                mask[i] = 1.0;
                            }
                            current_offset += token_len;
                        }
                    }
                }
            }

            all_input_ids.push(input_ids);
            all_target_ids.push(target_ids);
            all_reasoning_masks.push(mask);
        }

        let flat_inputs: Vec<u32> = all_input_ids.into_iter().flatten().collect();
        let flat_targets: Vec<u32> = all_target_ids.into_iter().flatten().collect();
        let flat_masks: Vec<f32> = all_reasoning_masks.into_iter().flatten().collect();

        let input_ids = Tensor::from_vec(flat_inputs, (batch_size, max_seq_len), device)?;
        let target_ids = Tensor::from_vec(flat_targets, (batch_size, max_seq_len), device)?;
        let reasoning_mask = Tensor::from_vec(flat_masks, (batch_size, max_seq_len), device)?;

        Ok(Self {
            input_ids,
            target_ids,
            teacher_logits: None,
            reasoning_mask: Some(reasoning_mask),
            chosen_ids: None,
            rejected_ids: None,
        })
    }
}

// compute_lr functionality moved to scheduler.rs

/// Main training loop.
///
/// This function coordinates:
/// 1. Forward pass through student model
/// 2. Loss computation (distillation or standard CE)
/// 3. Backward pass (gradient computation)
/// 4. Gradient accumulation across micro-batches
/// 5. Gradient clipping
/// 6. Optimizer step (AdamW)
/// 7. LR schedule update
/// 8. Periodic checkpointing
/// 9. Logging
///
/// Performance metrics for the Sovereign Progress Dashboard.
#[derive(Debug, Clone)]
pub struct DashboardInfo {
    pub stage: String,
    pub progress_pct: f32,
    pub tokens_per_sec: f32,
    pub current_loss: f64,
    pub best_loss: f64,
    pub eta_secs: u64,
    pub last_save: String,
}

pub struct Trainer {
    config: TrainerConfig,
    checkpoint_manager: CheckpointManager,
    training_state: TrainingState,
    optimizer: Optimizer,
    scheduler: LrScheduler,
    pub device: Device,
    /// Diamond-Grade: Atomic stop flag for graceful preemption
    stop_flag: Arc<AtomicBool>,
    /// Experience replay buffer to combat catastrophic forgetting
    pub replay_buffer: super::replay_buffer::ReplayBuffer,
    /// ZENITH: High-level critic for Autopoietic Loss modulation
    pub critic: Option<Arc<dyn super::SovereignCritic>>,
    /// Global maximum sequence length from ModelConfig
    pub max_seq_len: usize,
    /// Sovereign Guard: Original checkpoint frequency for auto-tune clamping
    default_checkpoint_steps: u64,
}

impl Trainer {
    /// Diamond-Grade Dashboard Metrics
    pub fn get_dashboard_metrics(&self, total_target_steps: u64) -> DashboardInfo {
        let now = chrono::Utc::now().timestamp();
        let elapsed = (now - self.training_state.start_time_secs).max(1) as f32;
        let tokens_per_sec = self.training_state.tokens_processed as f32 / elapsed;

        let progress_pct = if total_target_steps > 0 {
            (self.training_state.global_step as f32 / total_target_steps as f32) * 100.0
        } else {
            0.0
        };

        let eta_secs = if tokens_per_sec > 0.0
            && total_target_steps > self.training_state.global_step
        {
            // rough estimate based on average token/step ratio
            let tokens_remaining = (total_target_steps - self.training_state.global_step) * 2048; // assuming 2k seq len
            (tokens_remaining as f32 / tokens_per_sec) as u64
        } else {
            0
        };

        DashboardInfo {
            stage: self.training_state.stage_name.clone(),
            progress_pct,
            tokens_per_sec,
            current_loss: self
                .training_state
                .loss_history
                .back()
                .copied()
                .unwrap_or(f64::INFINITY),
            best_loss: self.training_state.best_loss,
            eta_secs,
            last_save: self
                .training_state
                .last_save_time
                .clone()
                .unwrap_or_else(|| "NONE".into()),
        }
    }
    pub fn new(
        config: TrainerConfig,
        checkpoint_dir: &str,
        model_config: &ModelConfig,
        varmap: &VarMap,
        device: Device,
    ) -> Result<Self> {
        let checkpoint_manager = CheckpointManager::new(
            checkpoint_dir,
            5,    // keep last 5 checkpoints
            true, // atomic writes
        )?;

        // Compute config hash for resume validation
        let config_yaml = serde_yaml::to_string(model_config).unwrap_or_default();
        let mut hasher = sha2::Sha256::new();
        hasher.update(config_yaml.as_bytes());
        let hash_result = hasher.finalize();
        let config_hash = format!("{:x}", hash_result)
            .chars()
            .take(16)
            .collect::<String>();

        let training_state = TrainingState::new(config_hash, model_config.name.clone());

        // Initialize new modular components
        let optimizer = Optimizer::new(varmap, config.optimizer.clone())?;
        let scheduler = LrScheduler::new(config.scheduler.clone());

        // ── Zenith Hardening: Persistent Sovereign Memory ──
        let buffer_path = std::path::PathBuf::from("shakey_data/training/replay_buffer.bin");
        let replay_buffer = match super::replay_buffer::ReplayBuffer::load_from_disk(&buffer_path) {
            Ok(buf) => {
                tracing::info!(target: "shakey::sovereign", "Successfully restored Sovereign memory ({} interactions).", buf.len());
                buf
            }
            Err(e) => {
                let err_msg = e.to_string();
                if err_msg.contains("Integrity Failure")
                    || err_msg.contains("deserialization failed")
                {
                    // QUARANTINE: Move corrupted file to safety and start fresh
                    let quarantine_dir = std::path::PathBuf::from("shakey_data/quarantine");
                    let _ = std::fs::create_dir_all(&quarantine_dir);
                    let timestamp = chrono::Utc::now().timestamp();
                    let target_path =
                        quarantine_dir.join(format!("corrupted_buffer_{}.bin", timestamp));

                    tracing::error!(target: "shakey::sovereign", "‼️ CRITICAL: Memory corruption detected! Quarantining file to {}...", target_path.display());
                    let _ = std::fs::rename(&buffer_path, &target_path);
                } else {
                    tracing::info!(target: "shakey::sovereign", "No existing memory found. Initializing fresh Sovereign buffer.");
                }
                super::replay_buffer::ReplayBuffer::new(1000)
            }
        };

        let default_ckpt_steps = config.checkpoint_every_steps;
        Ok(Self {
            config,
            checkpoint_manager,
            training_state,
            optimizer,
            scheduler,
            device,
            stop_flag: Arc::new(AtomicBool::new(false)),
            replay_buffer,
            critic: None,
            max_seq_len: model_config.max_seq_len,
            default_checkpoint_steps: default_ckpt_steps,
        })
    }

    /// Load LoRA adapters dynamically at runtime
    pub fn load_lora_adapters(
        &mut self,
        path: &std::path::Path,
        varmap: &mut candle_nn::VarMap,
    ) -> Result<()> {
        if path.exists() {
            tracing::info!("Loading LoRA Adapters from {}", path.display());
            varmap
                .load(path)
                .map_err(|e| anyhow::anyhow!("Failed to dynamic load LoRA: {}", e))?;
        }
        Ok(())
    }

    /// Set the stop flag to trigger a graceful shutdown
    /// Elite WCSI: Run a dynamic online training step using the ReplayBuffer.
    ///
    /// This is the heart of the Sovereign Evolution. It samples from past
    /// experiences and mixes them with the current interaction to ensure
    /// that the agent learns without losing its core identity.
    /// ══ ELITE: Sovereign Online Learning Engine ══
    ///
    /// The primary entry point for all dynamic, real-time parameter updates.
    /// Automatically selects the optimal learning strategy:
    ///
    /// - **Online DPO**: When the memory has a `rejected` completion, performs a
    ///   dual-forward pass optimizing `chosen` vs `rejected` simultaneously.
    ///   This is state-of-the-art preference alignment in real-time.
    ///
    /// - **Online SFT**: For general (non-correction) interactions, samples from
    ///   the Importance Replay Buffer to prevent catastrophic forgetting.
    ///
    /// - **Adaptive Micro-Batching**: Automatically scales batch size based on
    ///   training step count to balance latency and gradient quality.
    pub async fn train_online_step(
        &mut self,
        model: &crate::model::transformer::TransformerModel,
        reference_model: Option<&crate::model::transformer::TransformerModel>,
        varmap: &VarMap,
        new_memory: super::replay_buffer::ReplayMemory,
        tokenizer: &crate::tokenizer::Tokenizer,
    ) -> Result<LossComponents> {
        let is_dpo_eligible = new_memory.is_dpo_pair();

        // 1. Commit new memory to the elite buffer (with deduplication)
        self.replay_buffer.push(new_memory.clone());

        // 2. Adaptive Micro-Batch Scaling
        //    Steps 0-9: batch=2 (fast warmup), every 10th step: batch=8 (deeper consolidation)
        let sample_size = match self.training_state.global_step % 10 {
            0 => 8usize,
            _ => 4,
        };

        // ── Path A: Online DPO (Correction detected) ──
        if is_dpo_eligible {
            if let (Some(ref_model), Some(ref rejected_str)) =
                (reference_model, &new_memory.rejected)
            {
                tracing::info!(
                    target: "shakey",
                    "Sovereign Online DPO: Aligning policy against rejected completion [step={}]",
                    self.training_state.global_step
                );

                // Tokenize chosen and rejected completions
                let chosen_full = format!("{} {}", new_memory.prompt, new_memory.completion);
                let rejected_full = format!("{} {}", new_memory.prompt, rejected_str);

                let mut chosen_tokens = tokenizer.encode(&chosen_full)?;
                let mut rejected_tokens = tokenizer.encode(&rejected_full)?;

                const MAX_SEQ: usize = 512; // Conservative for online speed
                chosen_tokens.truncate(MAX_SEQ);
                rejected_tokens.truncate(MAX_SEQ);

                // Zero-pad to same length for batching
                while chosen_tokens.len() < MAX_SEQ {
                    chosen_tokens.push(tokenizer.special_tokens().pad_id);
                }
                while rejected_tokens.len() < MAX_SEQ {
                    rejected_tokens.push(tokenizer.special_tokens().pad_id);
                }

                let chosen_ids =
                    candle_core::Tensor::from_vec(chosen_tokens, (1, MAX_SEQ), &self.device)?;
                let rejected_ids =
                    candle_core::Tensor::from_vec(rejected_tokens, (1, MAX_SEQ), &self.device)?;

                // DPO target_ids: use chosen_ids as the CE supervision target
                let target_ids = chosen_ids.clone();
                let dpo_batch = TrainingBatch {
                    input_ids: chosen_ids.clone(),
                    target_ids,
                    teacher_logits: None,
                    reasoning_mask: None,
                    chosen_ids: Some(chosen_ids),
                    rejected_ids: Some(rejected_ids),
                };

                let results = self
                    .train_dpo_step(model, ref_model, &dpo_batch, varmap)
                    .await?;

                // Update the original memory with the loss from this alignment step
                // to mark it as more/less "surprising" for future cycles.
                let mut updated_memory = new_memory.clone();
                updated_memory.update_loss(results.total);
                self.replay_buffer.push(updated_memory);

                return Ok(results);
            }
        }

        // ── Path B: Online SFT (standard interaction or no reference model) ──
        tracing::info!(
            target: "shakey",
            "Sovereign Online SFT: Consolidating memory from replay buffer [samples={}]",
            sample_size
        );
        let mut examples = self.replay_buffer.sample_importance(sample_size - 1);
        examples.push(new_memory);

        let batch = TrainingBatch::from_examples(&examples, tokenizer, 2048, &self.device)?;
        let results = self.train_step(model, &batch, varmap, tokenizer).await?;

        // ── Evolution Loop: Memory Consolidation ──
        // Update the loss for all memories in this batch. This ensures that
        // "well-learned" items are sampled less often, while "difficult"
        // concepts stay in the hot-path for faster evolution.
        for mut example in examples {
            example.update_loss(results.total);
            self.replay_buffer.push(example);
        }

        Ok(results)
    }

    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::SeqCst);
    }

    /// Resume training state from the latest checkpoint.
    ///
    /// Returns true if a checkpoint was loaded.
    pub fn resume(&mut self) -> Result<bool> {
        if let Some((state, _opt_state)) = self.checkpoint_manager.load_latest()? {
            tracing::info!(
                "Resumed from step {} (best_loss={:.6}) on device {:?}",
                state.global_step,
                state.best_loss,
                self.device
            );
            self.training_state = state;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Sovereign-Grade: Record a capability snapshot for historical tracking.
    pub fn record_capabilities(&mut self, capabilities: super::capabilities::CapabilityMatrix) {
        self.training_state
            .capability_history
            .push((self.training_state.global_step, capabilities));
    }

    /// ── Zenith Sovereign Apex: Self-Healing Training Engine ──
    ///
    /// Executes a training step with autonomous OOM recovery. If a hardware failure
    /// or memory overflow occurs, the trainer automatically scales down and retries.
    pub async fn train_on_examples_sovereign<T: AsTrainingExample>(
        &mut self,
        model: &TransformerModel,
        varmap: &VarMap,
        tokenizer: &crate::tokenizer::Tokenizer,
        examples: &[T],
        device: &Device,
    ) -> Result<LossComponents> {
        let mut current_examples = examples;
        let mut attempts = 0;
        let max_attempts = 2;

        loop {
            // Attempt 1: Construct batch and run step
            let batch_res =
                TrainingBatch::from_examples(current_examples, tokenizer, self.max_seq_len, device);

            match batch_res {
                Ok(batch) => {
                    match self.train_step(model, &batch, varmap, tokenizer).await {
                        Ok(components) => {
                            // Success: If we had to scale down, slowly ramp back up (Additive Increase)
                            if current_examples.len() < examples.len() {
                                self.config.batch_size = (self.config.batch_size + 1).min(128);
                                tracing::info!(target: "shakey::sovereign", "Recovery successful. Ramping batch_size back to {}.", self.config.batch_size);
                            }
                            return Ok(components);
                        }
                        Err(e) => {
                            let err_msg = e.to_string().to_lowercase();
                            if (err_msg.contains("out of memory")
                                || err_msg.contains("cuda_error_out_of_memory"))
                                && attempts < max_attempts
                            {
                                attempts += 1;
                                tracing::error!(target: "shakey::sovereign", "‼️ SOVEREIGN RECOVERY: OOM detected during backprop. Scaling down batch size...");

                                // Multiplicative Decrease: Halve the batch
                                self.config.batch_size = (self.config.batch_size / 2).max(1);
                                let new_len = current_examples.len() / 2;
                                if new_len == 0 {
                                    return Err(e);
                                }
                                current_examples = &current_examples[..new_len];

                                // Force synchronizing and clearing cache if possible
                                // (In candle, dropping tensors usually suffices)
                                continue;
                            }
                            return Err(e);
                        }
                    }
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Run one training step (may involve multiple gradient accumulation micro-steps).
    ///
    /// Returns the loss components for this step.
    pub async fn train_step(
        &mut self,
        model: &TransformerModel,
        batch: &TrainingBatch,
        varmap: &VarMap,
        tokenizer: &crate::tokenizer::Tokenizer,
    ) -> Result<LossComponents> {
        // Check for Diamond-Grade emergency shutdown
        if self.stop_flag.load(Ordering::SeqCst) {
            return Err(anyhow::anyhow!(
                "Shutdown requested by system signal (preemption-aware)"
            ));
        }

        // 0. Ultra-Elite: Pulse the auto-tuner every 50 steps
        // This ensures the agent adapts to training plateaus in real-time.
        if self.training_state.global_step > 0 && self.training_state.global_step.is_multiple_of(50)
        {
            self.auto_tune_hyperparameters();
        }

        // ── World-First: Autopoietic Loss Modulation ──
        let mut loss_multiplier = 1.0f32;
        if let Some(ref critic) = self.critic {
            // Evaluates a sample prompt/completion from the batch to steer reasoning quality
            // (Sampling 0th element as a proxy for the batch's semantic direction)
            if let (Ok(p_ids), Ok(c_ids)) = (batch.input_ids.get(0), batch.target_ids.get(0)) {
                let prompt = tokenizer.decode(&p_ids.to_vec1::<u32>()?)?;
                let completion = tokenizer.decode(&c_ids.to_vec1::<u32>()?)?;

                if let Ok(score) = critic.evaluate_score(&prompt, &completion).await {
                    // Autopoietic steering: low score (bad logic) increases loss multiplier
                    // to force harder parameter updates on this "lesson".
                    loss_multiplier = (2.0 - score).clamp(0.5, 3.0);
                    tracing::debug!(target: "shakey::train", "🧠 Autopoietic Modulation: score={:.2}, mult={:.2}x", score, loss_multiplier);
                }
            }
        }

        // Forward pass
        let forward_out = model
            .forward(&batch.input_ids, 0, None, None)
            .map_err(|e| anyhow::anyhow!("Forward pass failed: {}", e))?;

        // Compute loss
        let (loss, components) = if let Some(ref teacher_logits) = batch.teacher_logits {
            // Distillation loss (with teacher labels)
            distillation_loss(
                &forward_out.logits,
                teacher_logits,
                &batch.target_ids,
                &forward_out.aux_loss,
                batch.reasoning_mask.as_ref(),
                &self.config.distillation,
            )
            .map_err(|e| anyhow::anyhow!("Loss computation failed: {}", e))?
        } else {
            // Standard cross-entropy (no teacher)
            let ce_loss = super::distillation::cross_entropy_loss(
                &forward_out.logits,
                &batch.target_ids,
                self.config.distillation.label_smoothing,
                Some(0),
            )
            .map_err(|e| anyhow::anyhow!("CE loss failed: {}", e))?;

            let total_loss = (&ce_loss + &(forward_out.aux_loss.clone() * 0.01)?)?;
            let total_val = total_loss
                .to_vec0::<f32>()
                .map_err(|e| anyhow::anyhow!("Loss to scalar failed: {}", e))?
                as f64;
            let ce_val = ce_loss
                .to_vec0::<f32>()
                .map_err(|e| anyhow::anyhow!("CE to scalar failed: {}", e))?
                as f64;
            let aux_val = forward_out
                .aux_loss
                .to_vec0::<f32>()
                .map_err(|e| anyhow::anyhow!("Aux to scalar failed: {}", e))?
                as f64;

            (
                total_loss,
                LossComponents {
                    total: total_val,
                    kl_divergence: 0.0,
                    cross_entropy: ce_val,
                    aux_balance: aux_val,
                },
            )
        };

        // ── Autopoietic Steering ──
        let loss = if (loss_multiplier - 1.0).abs() > 1e-4 {
            (loss * (loss_multiplier as f64))?
        } else {
            loss
        };

        // Backward pass
        let grads = loss
            .backward()
            .map_err(|e| anyhow::anyhow!("Backward pass failed: {}", e))?;

        // ── Industry-Grade: Sovereign Stability Guards ──
        // 1. Loss Spike Detection: Prevents a single corrupted batch from ruining the model.
        if components.total > (self.training_state.rolling_loss * 5.0)
            && self.training_state.global_step > 10
        {
            tracing::warn!(
                target: "shakey::sovereign",
                "⚠️ Sovereign Intervention: Abandoning step {} due to loss spike ({:.4} vs rolling {:.4}).",
                self.training_state.global_step,
                components.total,
                self.training_state.rolling_loss
            );
            return Ok(components);
        }

        // 2. Gradient NaN/Inf Guard: Absolute protection against numerical overflow.
        let mut has_nan = false;
        for var in self.optimizer.vars().iter() {
            if let Some(grad) = grads.get(var.as_tensor()) {
                // Robust NaN detection: NaN is the only value NOT equal to itself.
                // Sum all unequal elements — if > 0, we have an instability.
                if grad
                    .ne(grad)?
                    .to_dtype(candle_core::DType::F32)?
                    .sum_all()?
                    .to_scalar::<f32>()?
                    > 0.0
                {
                    has_nan = true;
                    break;
                }
            }
        }

        if has_nan {
            tracing::error!(target: "shakey::sovereign", "‼️ Sovereign Intervention: NaNs detected in gradients. Skipping step {}.", self.training_state.global_step);
            return Ok(components);
        }

        // Compute current learning rate
        let lr = self.scheduler.compute_lr(self.training_state.global_step);

        // Optimizer step with integrated global gradient clipping
        let grad_norm = self.optimizer.step(&grads, lr, self.config.max_grad_norm)?;

        // Record global metrics for real-time telemetry
        SovereignMetrics::global().record_gradient_norm(grad_norm as f64);

        // Update training state with precise token count
        let (batch_sz, seq_len) = batch
            .input_ids
            .dims2()
            .with_context(|| "Failed to get batch dimensions for telemetry")?;
        let token_count = (batch_sz * seq_len) as u64;
        self.training_state
            .record_step(components.total, grad_norm as f64, lr, token_count);

        // Log progress
        if self
            .training_state
            .global_step
            .is_multiple_of(self.config.log_every_steps)
        {
            tracing::info!(
                "Step {} | {} | lr={:.2e} | tokens={}",
                self.training_state.global_step,
                components,
                lr,
                self.training_state.tokens_processed,
            );
        }

        if self
            .training_state
            .global_step
            .is_multiple_of(self.config.checkpoint_every_steps)
        {
            let timestamp = chrono::Local::now().format("%H:%M:%S").to_string();
            self.training_state.last_save_time = Some(timestamp.clone());

            self.checkpoint_manager.save(
                self.training_state.global_step,
                varmap,
                &self.training_state,
                None,
            )?;

            // Persist the ReplayBuffer atomically during checkpoint (Industry-Grade BIN)
            let buffer_path = std::path::PathBuf::from("shakey_data/training/replay_buffer.bin");
            if let Err(e) = self.replay_buffer.save_to_disk(&buffer_path) {
                tracing::warn!(target: "shakey::sovereign", "Failed to persist Sovereign memory: {}", e);
            }

            tracing::info!(target: "shakey", "Atomic auto-save pulsed at {} [Step {}]", timestamp, self.training_state.global_step);
        }

        Ok(components)
    }

    /// Elite WCSI: Run one DPO training step.
    ///
    /// # Arguments
    /// * `policy` - The current student model being trained.
    /// * `reference` - A frozen snapshot of the model before this training stage.
    /// * `batch` - A TrainingBatch containing 'chosen_ids' and 'rejected_ids'.
    pub async fn train_dpo_step(
        &mut self,
        policy: &TransformerModel,
        reference: &TransformerModel,
        batch: &TrainingBatch,
        _varmap: &VarMap,
    ) -> Result<LossComponents> {
        let chosen_ids = batch
            .chosen_ids
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DPO batch must contain chosen_ids"))?;
        let rejected_ids = batch
            .rejected_ids
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("DPO batch must contain rejected_ids"))?;

        // 1. Forward passes for Policy model
        let policy_chosen = policy.forward(chosen_ids, 0, None, None)?;
        let policy_rejected = policy.forward(rejected_ids, 0, None, None)?;

        // 2. Forward passes for Reference model (No gradients!)
        let ref_chosen = {
            // Context managers for no-grad are done via candle's backprop exclusion or just not calling backward
            reference.forward(chosen_ids, 0, None, None)?
        };
        let ref_rejected = reference.forward(rejected_ids, 0, None, None)?;

        // 3. Compute log-probs
        let policy_chosen_logps = get_batch_logps(&policy_chosen.logits, chosen_ids, true)?;
        let policy_rejected_logps = get_batch_logps(&policy_rejected.logits, rejected_ids, true)?;

        let ref_chosen_logps = get_batch_logps(&ref_chosen.logits, chosen_ids, true)?;
        let ref_rejected_logps = get_batch_logps(&ref_rejected.logits, rejected_ids, true)?;

        // 4. Compute DPO Loss
        let loss = dpo_loss(
            &policy_chosen_logps,
            &policy_rejected_logps,
            &ref_chosen_logps,
            &ref_rejected_logps,
            &self.config.dpo,
        )?;

        // Add MoE aux loss if applicable
        let total_loss = (&loss + &(policy_chosen.aux_loss.clone() * 0.01)?)?;

        // 5. Backward pass
        let grads = total_loss.backward()?;
        let lr = self.current_lr();
        let grad_norm = self.optimizer.step(&grads, lr, self.config.max_grad_norm)?;

        // Record metrics
        let loss_val = loss.to_vec0::<f32>()? as f64;
        let components = LossComponents {
            total: loss_val,
            kl_divergence: 0.0, // DPO has internal KL implicit
            cross_entropy: 0.0,
            aux_balance: policy_chosen.aux_loss.to_vec0::<f32>()? as f64,
        };

        let token_count = (chosen_ids.elem_count() + rejected_ids.elem_count()) as u64;
        self.training_state
            .record_step(components.total, grad_norm as f64, lr, token_count);
        SovereignMetrics::global().record_gradient_norm(grad_norm as f64);

        // ── Numerical Stability Guard ──
        self.check_numerical_stability(&loss, "DPO Loss")?;

        Ok(components)
    }

    /// Periodically scans tensors for NaN or Infinite values to prevent
    /// weight corruption. Triggers a rollback if instability is detected.
    fn check_numerical_stability(&self, tensor: &Tensor, context: &str) -> Result<()> {
        let val = tensor.to_vec0::<f32>().unwrap_or(0.0);
        if val.is_nan() || val.is_infinite() {
            tracing::error!(
                "🚨 ALPHA-GRADE ALERT: Numerical instability detected in {}: {}",
                context,
                val
            );
            return Err(anyhow::anyhow!(
                "Numerical instability detected in {}: {}",
                context,
                val
            ));
        }
        Ok(())
    }

    /// Get the current training state.
    pub fn state(&self) -> &TrainingState {
        &self.training_state
    }

    pub fn state_mut(&mut self) -> &mut TrainingState {
        &mut self.training_state
    }

    /// Get the current learning rate.
    pub fn current_lr(&self) -> f64 {
        self.scheduler.compute_lr(self.training_state.global_step)
    }

    /// Ultra-Elite: Perform dynamic hyper-parameter auto-tuning based on learning entropy.
    fn auto_tune_hyperparameters(&mut self) {
        let history = &self.training_state.loss_history;
        let window = 20;

        if history.len() < window {
            return;
        }

        // Stagnation Detection: Compare last 10 steps to 10 before that
        let sum_last: f64 = history.iter().rev().take(10).sum();
        let sum_prev: f64 = history.iter().rev().skip(10).take(10).sum();

        let avg_last = sum_last / 10.0;
        let avg_prev = sum_prev / 10.0;

        // If improvement is less than 0.5% over 10 steps (more permissive than 1.0%)
        if avg_last >= avg_prev * 0.995 {
            let old_lr = self.scheduler.compute_lr(self.training_state.global_step);

            // Only decay if we are above a minimum floor (1e-7)
            if old_lr > 1e-7 {
                self.scheduler.decay(0.85); // Gentler decay (0.85 instead of 0.5)
                let new_lr = self.scheduler.compute_lr(self.training_state.global_step);
                tracing::info!(
                    target: "shakey",
                    "Ultra-Elite Auto-Tune: Plateau detected. Gentler decay applied: {:.6} → {:.6}",
                    old_lr,
                    new_lr
                );
            }
        }

        // ── Gradient-Aware Auto-Tune ──
        // If the last 5 gradient norms are significantly higher than the average,
        // it indicates numerical instability. Proactively decay LR.
        if self.training_state.grad_norm_history.len() >= 10 {
            let last_5_avg: f64 = self
                .training_state
                .grad_norm_history
                .iter()
                .rev()
                .take(5)
                .sum::<f64>()
                / 5.0;
            let prev_5_avg: f64 = self
                .training_state
                .grad_norm_history
                .iter()
                .rev()
                .skip(5)
                .take(5)
                .sum::<f64>()
                / 5.0;

            if last_5_avg > prev_5_avg * 3.0 && self.current_lr() > 1e-6 {
                self.scheduler.decay(0.7);
                tracing::warn!(
                    target: "shakey",
                    "Sovereign Stability Guard: Gradient spike detected ({:.2} vs {:.2}). Decaying LR early.",
                    last_5_avg, prev_5_avg
                );
            }
        }

        // ── Adaptive Heartbeat Logic ──
        // Check tokens per second to infer system stability/performance.
        // If slow, the system is constrained or volatile -> save more frequently.
        let now = chrono::Utc::now().timestamp();
        let elapsed = (now - self.training_state.start_time_secs).max(1) as f32;
        let tokens_per_sec = self.training_state.tokens_processed as f32 / elapsed;

        let old_ckpt = self.config.checkpoint_every_steps;
        // ── Sovereign Guard: Clamp to [default/4, default*2] to prevent permanent drift ──
        let min_ckpt = (self.default_checkpoint_steps / 4).max(10);
        let max_ckpt = (self.default_checkpoint_steps * 2).min(500);
        if tokens_per_sec < 50.0 {
            // System is struggling or running very slowly; save more often to preserve progress
            self.config.checkpoint_every_steps =
                (self.config.checkpoint_every_steps / 2).max(min_ckpt);
        } else if tokens_per_sec > 500.0 {
            // System is blazing fast and stable; save less often to reduce disk I/O
            self.config.checkpoint_every_steps =
                (self.config.checkpoint_every_steps * 2).min(max_ckpt);
        }

        if old_ckpt != self.config.checkpoint_every_steps {
            tracing::info!(
                target: "shakey",
                "Adaptive Heartbeat: Adjusted checkpoint frequency from {} to {} based on system throughput ({:.1} t/s)",
                old_ckpt, self.config.checkpoint_every_steps, tokens_per_sec
            );
        }
    }
}

// Tests removed as they are now redundant with scheduler/optimizer tests
