//! Learning rate scheduler.
//!
//! Provides a smooth, configurable LR schedule with:
//! 1. Linear warmup (0 → learning_rate)
//! 2. Cosine decay (learning_rate → min_learning_rate)
//! 3. Reset support for restart-based training strategies.

use std::f64::consts::PI;

/// LR schedule configuration.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_lr: f64,
    pub min_lr: f64,
    pub warmup_steps: u64,
    pub total_steps: u64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_lr: 3e-4,
            min_lr: 1e-6,
            warmup_steps: 1000,
            total_steps: 100_000,
        }
    }
}

/// Learning rate scheduler.
pub struct LrScheduler {
    pub config: SchedulerConfig,
    /// Ultra-Elite: Dynamic scale factor for autonomous auto-tuning (default 1.0)
    pub scale_factor: f64,
}

impl LrScheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            scale_factor: 1.0,
        }
    }

    /// Appy a decay factor to the current schedule (e.g. 0.5 to halve LR).
    /// ── Sovereign Guard: Floor scale_factor at 1e-4 to prevent silent training stall ──
    pub fn decay(&mut self, factor: f64) {
        self.scale_factor = (self.scale_factor * factor).max(1e-4);
    }

    /// Compute the current learning rate based on step.
    pub fn compute_lr(&self, step: u64) -> f64 {
        let warmup = self.config.warmup_steps as f64;
        let total = self.config.total_steps as f64;
        let step = step as f64;

        let base_lr = if step < warmup {
            // Linear warmup: 0 → max_lr
            // (step + 1) / warmup * max_lr
            self.config.max_lr * (step + 1.0) / (warmup + 1.0)
        } else if total <= warmup {
            // Edge case: total_steps == warmup_steps → no decay phase, stay at max_lr
            self.config.max_lr
        } else {
            // Cosine decay: max_lr → min_lr
            let progress = (step - warmup) / (total - warmup);
            // progress = 0..1

            // Cosine annealing formula:
            // η_t = η_min + 0.5 * (η_max - η_min) * (1 + cos(π * t / T))
            let cosine = 0.5 * (1.0 + (PI * progress).cos());
            let lr = self.config.min_lr + (self.config.max_lr - self.config.min_lr) * cosine;

            // Clamp to min_lr just in case
            lr.max(self.config.min_lr)
        };

        base_lr * self.scale_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warmup_cosine_schedule() {
        let config = SchedulerConfig {
            max_lr: 3e-4,
            min_lr: 1e-6,
            warmup_steps: 100,
            total_steps: 1000,
        };
        let sched = LrScheduler::new(config);

        // Step 0: should be near 0
        let lr0 = sched.compute_lr(0);
        assert!(lr0 < 1e-5);

        // Step 100 (end of warmup): should be exactly/near max_lr
        let lr100 = sched.compute_lr(100);
        assert!((lr100 - 3e-4).abs() < 1e-6);

        // Step 550 (midway through cosine): should be roughly in the middle
        let lr550 = sched.compute_lr(550);
        assert!(lr550 > 1e-4 && lr550 < 2e-4);

        // Step 1000 (end of total): should be min_lr
        let lr1000 = sched.compute_lr(1000);
        assert!((lr1000 - 1e-6).abs() < 1e-7);
    }
}
