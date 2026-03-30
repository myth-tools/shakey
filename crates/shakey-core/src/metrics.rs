use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

const LATENCY_WINDOW_SIZE: usize = 1024;
const EMA_ALPHA: f64 = 0.1; // Smoothing factor for EMA

/// Registry of real-time agent 'vital signs' with Z-Grade telemetry.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetricsSnapshot {
    pub total_tokens_generated: u64,
    pub total_inference_time_ms: u64,
    pub hallucinations_filtered: u64,
    pub memory_pressure_events: u64,
    pub tool_executions: u64,
    pub tool_failures: u64,
    pub avg_latency_per_token_ms: f64,
    pub p50_latency_ms: f64,
    pub p90_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub tokens_per_second: f64,

    // Z-GRADE TELEMETRY
    pub ema_tokens_per_second: f64,
    pub success_rate: f64,
    pub system_heat: f64,   // 0.0 - 1.0 (failure density)
    pub gradient_norm: f64, // Latest global gradient norm
    pub uptime_secs: u64,
    pub api_pressure_events: u64,
    pub avg_api_latency_us: u64,
}

pub struct SovereignMetrics {
    // Hot-path atomic counters (Zero-Contention)
    tokens: AtomicU64,
    time_ms: AtomicU64,
    time_us: AtomicU64,
    hallucinations: AtomicU64,
    memory_events: AtomicU64,
    api_pressure_count: AtomicU64,
    tool_execs: AtomicU64,
    tool_fails: AtomicU64,

    // EMA States (Atomic Bits for Lock-free Floating Point updates)
    ema_tps_bits: AtomicU64,
    grad_norm_bits: AtomicU64,
    last_token_count: AtomicU64,

    // Latency distribution (Zero-Allocation, Lock-free Circular Buffer)
    latency_buffer: [AtomicU32; LATENCY_WINDOW_SIZE],
    latency_index: AtomicUsize,
    latency_count: AtomicUsize,

    // Performance Optimization: Cached Percentiles
    cached_p50: AtomicU32,
    cached_p90: AtomicU32,
    cached_p99: AtomicU32,
    last_sorted_count: AtomicUsize,

    start_time: Instant,
}

static GLOBAL_METRICS: Lazy<Arc<SovereignMetrics>> = Lazy::new(|| {
    // Initialize atomic buffer using from_fn for idiomatic, safe setup
    let buffer: [AtomicU32; LATENCY_WINDOW_SIZE] =
        std::array::from_fn(|_| AtomicU32::new(0.0f32.to_bits()));

    Arc::new(SovereignMetrics {
        tokens: AtomicU64::new(0),
        time_ms: AtomicU64::new(0),
        time_us: AtomicU64::new(0),
        hallucinations: AtomicU64::new(0),
        memory_events: AtomicU64::new(0),
        api_pressure_count: AtomicU64::new(0),
        tool_execs: AtomicU64::new(0),
        tool_fails: AtomicU64::new(0),
        ema_tps_bits: AtomicU64::new(0.0f64.to_bits()),
        grad_norm_bits: AtomicU64::new(0.0f64.to_bits()),
        last_token_count: AtomicU64::new(0),
        latency_buffer: buffer,
        latency_index: AtomicUsize::new(0),
        latency_count: AtomicUsize::new(0),
        cached_p50: AtomicU32::new(0),
        cached_p90: AtomicU32::new(0),
        cached_p99: AtomicU32::new(0),
        last_sorted_count: AtomicUsize::new(0),
        start_time: Instant::now(),
    })
});

impl SovereignMetrics {
    pub fn global() -> Arc<Self> {
        GLOBAL_METRICS.clone()
    }

    /// Record token generation event.
    pub fn record_tokens(&self, count: u64, duration_ms: u64) {
        self.tokens.fetch_add(count, Ordering::Relaxed);
        self.time_ms.fetch_add(duration_ms, Ordering::Relaxed);
        self.time_us
            .fetch_add(duration_ms * 1000, Ordering::Relaxed);

        // Track per-token latency for percentiles (Lock-free sample)
        if count > 0 {
            let latency = duration_ms as f32 / count as f32;
            let index = self.latency_index.fetch_add(1, Ordering::Relaxed) % LATENCY_WINDOW_SIZE;
            self.latency_count.fetch_add(1, Ordering::Relaxed);

            // Store f32 bits in the atomic buffer
            self.latency_buffer[index].store(latency.to_bits(), Ordering::Relaxed);
        }

        // Update EMA TPS if a window has passed (industry standard heart-beat)
        self.update_ema_tps();
    }

    fn update_ema_tps(&self) {
        let now = Instant::now();
        let current_tokens = self.tokens.load(Ordering::Relaxed);
        let elapsed = now.duration_since(self.start_time).as_secs_f64();

        // ELITE: Lock-free periodic update check (Industry Standard)
        if elapsed >= 1.0 {
            let last_tokens = self
                .last_token_count
                .swap(current_tokens, Ordering::Relaxed);
            let instant_tps = (current_tokens - last_tokens) as f64 / 1.0; // Assume 1s window approx

            let old_ema_bits = self.ema_tps_bits.load(Ordering::Relaxed);
            let old_ema = f64::from_bits(old_ema_bits);
            let new_ema = (instant_tps * EMA_ALPHA) + (old_ema * (1.0 - EMA_ALPHA));

            self.ema_tps_bits
                .store(new_ema.to_bits(), Ordering::Relaxed);
        }
    }

    /// Record a hallucination detection event.
    pub fn record_hallucination(&self) {
        self.hallucinations.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a resource-based throttling event.
    pub fn record_memory_pressure(&self) {
        self.memory_events.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a tool failure event.
    pub fn record_tool_failure(&self) {
        self.tool_fails.fetch_add(1, Ordering::Relaxed);
    }

    /// Zenith Apex II: Record API pressure event (throttling wait).
    pub fn record_api_pressure(&self) {
        self.api_pressure_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a tool execution event.
    pub fn record_tool_execution(&self, success: bool) {
        self.tool_execs.fetch_add(1, Ordering::Relaxed);
        if !success {
            self.tool_fails.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record the global gradient norm from the latest optimizer step.
    pub fn record_gradient_norm(&self, norm: f64) {
        self.grad_norm_bits.store(norm.to_bits(), Ordering::Relaxed);
    }

    /// Get a point-in-time snapshot of all metrics.
    pub fn snapshot(&self) -> MetricsSnapshot {
        let tokens = self.tokens.load(Ordering::Relaxed);
        let time_ms = self.time_ms.load(Ordering::Relaxed);
        let execs = self.tool_execs.load(Ordering::Relaxed);
        let fails = self.tool_fails.load(Ordering::Relaxed);
        let uptime = self.start_time.elapsed().as_secs();

        // Z-LOSS ESTIMATION: If no gradient norm recorded, use 0.0
        let grad_norm = f64::from_bits(self.grad_norm_bits.load(Ordering::Relaxed));

        let mut snap = MetricsSnapshot {
            total_tokens_generated: tokens,
            total_inference_time_ms: time_ms,
            hallucinations_filtered: self.hallucinations.load(Ordering::Relaxed),
            memory_pressure_events: self.memory_events.load(Ordering::Relaxed),
            tool_executions: execs,
            tool_failures: fails,
            avg_latency_per_token_ms: if tokens > 0 {
                time_ms as f64 / tokens as f64
            } else {
                0.0
            },
            tokens_per_second: if uptime > 0 {
                tokens as f64 / uptime as f64
            } else {
                0.0
            },
            ema_tokens_per_second: f64::from_bits(self.ema_tps_bits.load(Ordering::Relaxed)),
            success_rate: if execs > 0 {
                (execs - fails) as f64 / execs as f64
            } else {
                1.0
            },
            system_heat: if execs > 0 {
                fails as f64 / execs.max(1) as f64
            } else {
                0.0
            },
            gradient_norm: grad_norm,
            uptime_secs: uptime,
            p50_latency_ms: f32::from_bits(self.cached_p50.load(Ordering::Relaxed)) as f64,
            p90_latency_ms: f32::from_bits(self.cached_p90.load(Ordering::Relaxed)) as f64,
            p99_latency_ms: f32::from_bits(self.cached_p99.load(Ordering::Relaxed)) as f64,
            api_pressure_events: self.api_pressure_count.load(Ordering::Relaxed),
            avg_api_latency_us: if tokens > 0 {
                self.time_us.load(Ordering::Relaxed) / tokens
            } else {
                0
            },
        };

        // ── Industry-Grade: Async Percentile Refresh ──
        let count = self
            .latency_count
            .load(Ordering::Relaxed)
            .min(LATENCY_WINDOW_SIZE);
        let last_sorted = self.last_sorted_count.load(Ordering::Relaxed);

        if count > 0
            && (count - last_sorted >= 100
                || (count == LATENCY_WINDOW_SIZE && last_sorted < LATENCY_WINDOW_SIZE))
        {
            let mut sorted: Vec<f32> = Vec::with_capacity(count);
            for i in 0..count {
                let bits = self.latency_buffer[i].load(Ordering::Relaxed);
                sorted.push(f32::from_bits(bits));
            }
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let p50 = sorted[sorted.len() * 50 / 100];
            let p90 = sorted[sorted.len() * 90 / 100];
            let p99 = sorted[sorted.len() * 99 / 100];

            self.cached_p50.store(p50.to_bits(), Ordering::Relaxed);
            self.cached_p90.store(p90.to_bits(), Ordering::Relaxed);
            self.cached_p99.store(p99.to_bits(), Ordering::Relaxed);
            self.last_sorted_count.store(count, Ordering::Relaxed);

            snap.p50_latency_ms = p50 as f64;
            snap.p90_latency_ms = p90 as f64;
            snap.p99_latency_ms = p99 as f64;
        }

        snap
    }

    /// ── Sovereign Dashboard Flush ──
    /// Periodically flushes metrics to a JSON file for industry-grade monitoring.
    pub fn flush_to_file(&self, path: impl AsRef<std::path::Path>) -> anyhow::Result<()> {
        let snapshot = self.snapshot();
        let json_str = serde_json::to_string_pretty(&snapshot)?;

        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let tmp_path = path.with_extension("tmp");
        std::fs::write(&tmp_path, json_str)?;
        std::fs::rename(&tmp_path, path)?;

        Ok(())
    }
}
