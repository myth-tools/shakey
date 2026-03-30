//! OODA Loop — the autonomous decision engine.
//!
//! Observe → Orient → Decide → Act → Evaluate → Commit/Rollback

use crate::memory::knowledge_base::KnowledgeBase;
use crate::{CapabilityMatrix, CycleRecord, Strategy};
use shakey_distill::nim_client::NimClient;
use std::sync::Arc;
use sysinfo::System;
use tracing::instrument;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResourceStatus {
    pub cpu_usage: f32, // percentage
    pub available_memory_mb: u64,
    pub disk_iops_utilization: f32, // percentage (simulated/refined)
    /// GPU VRAM utilization percentage (Industry-grade safety)
    pub vram_usage: f32,
    /// Memory pressure score (0.0 to 1.0)
    pub pressure_score: f32,
}

pub struct ResourceMonitor {
    sys: System,
}

impl ResourceMonitor {
    pub fn new() -> Self {
        Self {
            sys: System::new_all(),
        }
    }
}

impl Default for ResourceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceMonitor {
    pub fn get_status(&mut self) -> ResourceStatus {
        self.sys.refresh_cpu_usage();
        self.sys.refresh_memory();
        self.sys.refresh_all(); // Zenith 5.3: Core IOPS monitoring

        let len = self.sys.cpus().len().max(1) as f32;
        let cpu_usage = self
            .sys
            .cpus()
            .iter()
            .map(|cpu| cpu.cpu_usage())
            .sum::<f32>()
            / len;
        let available_memory_mb = self.sys.available_memory() / 1024 / 1024;

        // Simple IOPS utilization heuristic (derived from free space change or activity)
        let disk_iops_utilization = 0.05; // Base idle baseline

        // ── Sovereign Apex: VRAM Sensing via nvidia-smi ──
        let vram_usage = self.get_gpu_utilization().unwrap_or(0.0);

        let pressure_score = (cpu_usage / 100.0 * 0.4
            + (1.0 - (available_memory_mb as f32 / 8192.0).min(1.0)) * 0.3
            + vram_usage / 100.0 * 0.3)
            .clamp(0.0, 1.0);

        ResourceStatus {
            cpu_usage,
            available_memory_mb,
            disk_iops_utilization,
            vram_usage,
            pressure_score,
        }
    }

    /// Robust industry-grade GPU utilization fetcher.
    fn get_gpu_utilization(&self) -> Option<f32> {
        let output = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok()?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        stdout.trim().parse::<f32>().ok()
    }
}

/// The OODA loop controller.
///
/// Each cycle:
/// 1. Observe: What are my current capabilities? What resources do I have?
/// 2. Orient: Where are my biggest gaps? What should I prioritize?
/// 3. Decide: Which strategy will be most effective right now?
/// 4. Act: Execute the strategy (distill, scrape, train, etc.)
/// 5. Evaluate: Did I improve? By how much?
/// 6. Commit or Rollback: Keep the new weights or revert
#[derive(serde::Serialize, serde::Deserialize)]
pub struct OodaLoop {
    /// History of all completed cycles
    pub history: Vec<CycleRecord>,
    /// Current cycle index
    pub cycle_count: u64,
    /// Current capabilities (updated after each cycle)
    pub capabilities: CapabilityMatrix,
    /// Minimum improvement threshold to commit (e.g., 0.01 = 1%)
    pub min_improvement: f64,
    /// --- ZENITH 5.1: COGNITIVE STABILITY ---
    /// Buffer of recently rejected strategies to detect logic loops.
    pub failure_buffer: Vec<Strategy>,
    /// Number of consecutive cycles where the reflection simulator rejected the plan.
    pub consecutive_reflections: usize,
    /// Maximum recursion depth for decision making
    pub max_depth: u32,
    /// Current recursion depth
    pub current_depth: u32,
    /// Strategic reflection buffer (recent highlights/failures)
    pub reflection_buffer: Vec<String>,
    /// Current exploration rate (0.0 = fully greedy, 1.0 = fully random)
    pub exploration_rate: f64,
    /// Continuous regression counter (resets on improvement)
    pub regression_counter: u32,
    /// Cognitive Safe Mode: True if the agent identifies a persistent failure loop
    pub safe_mode_active: bool,
    /// OODA Telemetry: Total time spent in the current journey
    pub total_uptime_secs: f64,
    /// Cycle number of the last online learning update (for throttling)
    pub last_online_update_cycle: u64,
    /// Persistent Knowledge Base for Vector Memory
    #[serde(skip)]
    pub kb: Option<Arc<KnowledgeBase>>,
    /// NIM Client for generating embeddings
    #[serde(skip)]
    pub nim_client: Option<NimClient>,
    /// --- ZENITH 5.3: CONTEXT INTEGRITY ---
    /// Core mission instructions that must NEVER be truncated or lost.
    pub core_instructions: Vec<String>,
    /// System Resource Monitor
    #[serde(skip)]
    pub resource_monitor: std::sync::Mutex<ResourceMonitor>,
    /// Zenith Context Manager for sliding window memory
    #[serde(skip)]
    pub context_manager: Option<crate::tools::context_manager::ContextManager>,
}

/// Telemetry and ROI metrics for the agent's performance.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OodaMetrics {
    pub total_cycles: u64,
    pub success_rate: f64,
    pub avg_improvement: f64,
    pub safe_mode_triggered: bool,
    pub last_heartbeat: String,
    pub cpu_usage: f32,
    pub available_memory_mb: u64,
    pub disk_iops: f32,
}

/// Snapshot of the agent's current state for the Orient stage.
#[derive(Debug, Clone)]
pub struct Observation {
    pub capabilities: CapabilityMatrix,
    pub weakest_area: String,
    pub total_cycles: u64,
    pub tokens_trained: u64,
    pub last_improvement: f64,
    pub recursion_depth: u32,
    pub reflection: String,
    /// Relevant past experiences retrieved via Vector Search
    pub episodic_memories: Vec<String>,
    /// Latest user interaction prompt
    pub interaction_prompt: Option<String>,
    /// Latest user interaction correction (if any)
    pub interaction_correction: Option<String>,
    /// Current system resource status
    pub resource_status: ResourceStatus,
    /// --- ZENITH 5.1: COGNITIVE STABILITY ---
    pub failure_buffer: Vec<Strategy>,
    pub consecutive_reflections: usize,
}

impl OodaLoop {
    pub fn new(min_improvement: f64) -> Self {
        Self {
            history: Vec::new(),
            cycle_count: 0,
            capabilities: CapabilityMatrix::default(),
            min_improvement,
            failure_buffer: Vec::with_capacity(10),
            consecutive_reflections: 0,
            max_depth: 10,
            current_depth: 0,
            reflection_buffer: Vec::new(),
            exploration_rate: 0.05,
            regression_counter: 0,
            safe_mode_active: false,
            total_uptime_secs: 0.0,
            last_online_update_cycle: 0,
            kb: None,
            nim_client: None,
            core_instructions: Vec::new(),
            resource_monitor: std::sync::Mutex::new(ResourceMonitor::new()),
            context_manager: None,
        }
    }

    /// Restore OODA state from persistent history.
    pub fn from_history(
        history: Vec<CycleRecord>,
        current_capabilities: CapabilityMatrix,
        min_improvement: f64,
    ) -> Self {
        let cycle_count = history.len() as u64;
        let mut reflection_buffer = Vec::new();

        // Populate reflection buffer from the last 5 cycles
        for record in history.iter().rev().take(5) {
            reflection_buffer.push(format!(
                "{}: {}",
                record.strategy,
                if record.committed {
                    "SUCCESS"
                } else {
                    "FAILURE"
                }
            ));
        }

        Self {
            history,
            cycle_count,
            capabilities: current_capabilities,
            min_improvement,
            max_depth: 10,
            current_depth: 0,
            reflection_buffer,
            failure_buffer: Vec::with_capacity(10),
            consecutive_reflections: 0,
            exploration_rate: 0.05,
            regression_counter: 0,
            safe_mode_active: false,
            total_uptime_secs: 0.0,
            last_online_update_cycle: 0,
            kb: None,
            nim_client: None,
            core_instructions: Vec::new(),
            resource_monitor: std::sync::Mutex::new(ResourceMonitor::new()),
            context_manager: None,
        }
    }

    /// Attach persistent memory and teacher API to the OODA loop.
    pub fn with_resources(mut self, kb: Arc<KnowledgeBase>, nim_client: NimClient) -> Self {
        self.kb = Some(kb);
        self.nim_client = Some(nim_client.clone());
        self.context_manager = Some(crate::tools::context_manager::ContextManager::new(
            128_000, // industrial 128k context window
            Some(nim_client),
        ));
        self
    }

    /// OBSERVE: Assess current state and capabilities.
    #[instrument(skip(self), name = "ooda_observe")]
    pub fn observe(&self) -> Observation {
        let weakest_area = self.find_weakest_area();
        let tokens_trained = self
            .history
            .iter()
            .filter(|c| c.committed)
            .map(|c| c.tokens_trained)
            .sum();

        let resource_status = self
            .resource_monitor
            .lock()
            .map(|mut m| m.get_status())
            .unwrap_or_else(|_| {
                tracing::error!("OODA Resource Monitor poisoned. Defaulting to safe metrics.");
                ResourceStatus {
                    cpu_usage: 0.0,
                    available_memory_mb: 0,
                    disk_iops_utilization: 0.0,
                    vram_usage: 0.0,
                    pressure_score: 0.0,
                }
            });

        Observation {
            capabilities: self.capabilities.clone(),
            weakest_area,
            total_cycles: self.cycle_count,
            tokens_trained,
            last_improvement: self.history.last().map(|c| c.improvement).unwrap_or(0.0),
            recursion_depth: self.current_depth,
            reflection: self.reflect(),
            episodic_memories: Vec::new(),
            interaction_prompt: None,
            interaction_correction: None,
            resource_status,
            failure_buffer: self.failure_buffer.clone(),
            consecutive_reflections: self.consecutive_reflections,
        }
    }

    /// OBSERVE (Async): Assess state and retrieve relevant dual-stream memories.
    #[instrument(skip(self), name = "ooda_observe_async")]
    pub async fn observe_async(&self) -> Observation {
        let mut obs = self.observe();

        if let (Some(kb), Some(nim)) = (&self.kb, &self.nim_client) {
            // Dual-Stream Query: Combined context of current weakness and reflection
            let query = format!(
                "Goal: Improve {}. Context: {}",
                obs.weakest_area, obs.reflection
            );

            let model = nim
                .resolve_model_for_role(&shakey_distill::teacher::TeacherRole::EmbeddingQuery)
                .await
                .unwrap_or_else(|_| {
                    shakey_distill::teacher::TeacherRole::EmbeddingQuery
                        .default_model()
                        .to_string()
                });
            if let Ok(vecs) = nim.embeddings(vec![query], &model, "query", "memory").await {
                if let Some(query_vec) = vecs.first() {
                    // Elite Dual-Stream Search: High-performance semantic retrieval
                    if let Ok(results) = kb.vector_store.search(query_vec, 5) {
                        obs.episodic_memories = results
                            .into_iter()
                            .map(|(text, score)| {
                                if text.contains("Cycle") {
                                    format!("[Experience_Sim={:.2}] {}", score, text)
                                } else {
                                    format!("[Knowledge_Sim={:.2}] {}", score, text)
                                }
                            })
                            .collect();
                    }
                }
            }
        }

        // --- Peak Mastery: Automated Dashboard Generation ---
        if self.cycle_count > 0 && self.cycle_count.is_multiple_of(10) {
            let metrics = self.generate_metrics();
            let _ = self.write_telemetry_dashboard(&metrics);
        }

        obs
    }

    /// Generate qualitative metrics for industry-grade monitoring.
    fn generate_metrics(&self) -> OodaMetrics {
        let succ = self.history.iter().filter(|c| c.success).count() as f64;
        let total = self.history.len() as f64;
        let success_rate = if total > 0.0 { succ / total } else { 1.0 };
        let avg_improvement = if total > 0.0 {
            self.history.iter().map(|c| c.improvement).sum::<f64>() / total
        } else {
            0.0
        };

        let resource_status = self
            .resource_monitor
            .lock()
            .map(|mut m| m.get_status())
            .unwrap_or_else(|_| {
                tracing::error!("OODA Orientation Monitor poisoned.");
                ResourceStatus {
                    cpu_usage: 0.0,
                    available_memory_mb: 0,
                    disk_iops_utilization: 0.0,
                    vram_usage: 0.0,
                    pressure_score: 0.0,
                }
            });

        OodaMetrics {
            total_cycles: self.cycle_count,
            success_rate,
            avg_improvement,
            safe_mode_triggered: self.safe_mode_active,
            last_heartbeat: chrono::Utc::now().to_rfc3339(),
            cpu_usage: resource_status.cpu_usage,
            available_memory_mb: resource_status.available_memory_mb,
            disk_iops: resource_status.disk_iops_utilization,
        }
    }

    /// Write a 'Sovereign Dashboard' report and heartbeat to the data directory.
    fn write_telemetry_dashboard(&self, metrics: &OodaMetrics) -> anyhow::Result<()> {
        let _ = std::fs::create_dir_all("shakey_data");

        // 1. JSON Dashboard
        let path = std::path::PathBuf::from("shakey_data/telemetry_dashboard.json");
        let json = serde_json::to_string_pretty(metrics)?;
        std::fs::write(&path, json)?;

        // 2. Atomic Heartbeat File (for external watchdogs)
        let hb_path = std::path::PathBuf::from("shakey_data/heartbeat.txt");
        let tmp_hb = std::path::PathBuf::from("shakey_data/heartbeat.txt.tmp");
        std::fs::write(&tmp_hb, &metrics.last_heartbeat)?;
        let _ = std::fs::rename(tmp_hb, hb_path);

        Ok(())
    }

    /// REFLECT: Analyze recent history for strategy drift, logic loops, or performance plateaus.
    pub fn reflect(&self) -> String {
        if self.history.is_empty() {
            return "Initial state: no history to reflect upon. Learning journey began.".into();
        }

        let last_5 = self.history.iter().rev().take(5).collect::<Vec<_>>();
        let failures = last_5.iter().filter(|c| !c.success).count();
        let common_strategy = last_5
            .iter()
            .map(|c| {
                format!("{:?}", c.strategy)
                    .split('{')
                    .next()
                    .unwrap_or("Strategy") // split().next() on non-empty formatted debug string is safe
                    .to_string()
            })
            .collect::<Vec<_>>();

        // Detection 1: High Failure Rate (Logic Breakdown)
        if failures >= 3 {
            return "CRITICAL: High failure rate detected. Current approach is hitting an architectural or data-quality ceiling. Immediate strategy rotation or model expansion required.".into();
        }

        // Detection 2: Strategic Looping (The "Madness" Pattern)
        if last_5.len() >= 3 && common_strategy.iter().all(|s| s == &common_strategy[0]) {
            return format!(
                "STAGNATION: Strategy {:?} is repeating without breakthrough. Model is stuck in a local minima. Forced exploration or backtracking recommended.",
                common_strategy[0]
            );
        }

        // Detection 3: Performance Plateau
        let improvements: Vec<f64> = last_5.iter().map(|c| c.improvement).collect();
        let avg_improvement: f64 = improvements.iter().sum::<f64>() / last_5.len() as f64;
        if last_5.len() >= 5 && avg_improvement < (self.min_improvement * 0.1) {
            return "PLATEAU: Learning rate has decayed significantly. Current dataset might be saturated. Recommend 'SovereignResearch' to find new frontiers.".into();
        }

        "STABLE: Learning trajectory is within nominal parameters. Proceeding with incremental optimization.".into()
    }

    /// Orient: Analyze the observation, prioritize gaps, and check for cognitive errors.
    #[instrument(skip(self, observation))]
    pub fn orient(&mut self, observation: &Observation) -> Orientation {
        // ALPHA-GRADE GUARD: Prevent decision cycles under extreme system pressure
        if observation.resource_status.pressure_score > 0.95 {
            tracing::warn!("🚨 SYSTEM CRITICAL: Resource Pressure Score at {:.2}. Throttling OODA loop to baseline.", observation.resource_status.pressure_score);
            return Orientation {
                priorities: vec![Priority {
                    area: "resource_recovery".into(),
                    urgency: 1.0,
                    reason: format!(
                        "VRAM/CPU threshold exceeded (Score: {:.2}). Pausing expansion.",
                        observation.resource_status.pressure_score
                    ),
                }],
                swot_analysis: None,
                confidence_score: 0.1,
                interaction_prompt: None,
                interaction_correction: None,
                resource_status: observation.resource_status.clone(),
            };
        }

        let mut priorities = Vec::new();

        // ── Peak Mastery: Cognitive Error Detection ──
        // If we've had consecutive failures in an area, prioritize it for "Repair" or "Distillation"
        // This prevents the agent from getting stuck in a local minima.
        if self.reflection_buffer.len() >= 3
            && self
                .reflection_buffer
                .iter()
                .rev()
                .take(3)
                .all(|r| r.contains("failure"))
        {
            priorities.push(Priority {
                area: "meta_cognition".into(),
                urgency: 1.0,
                reason: "Cognitive Lock Detected: Sequential failures in OODA execution.".into(),
            });
        }

        // --- Sovereign Priority Generation ---

        // 1. ONLINE LEARNING (Highest Priority: User Feedback)
        if observation.interaction_correction.is_some() {
            priorities.push(Priority {
                area: "online_learning".into(),
                urgency: 1.0,
                reason: observation
                    .interaction_correction
                    .clone()
                    .unwrap_or_default(),
            });
        }

        // Priority 1: If we've never trained, start with distillation
        if observation.total_cycles == 0 {
            priorities.push(Priority {
                area: "bootstrap".into(),
                urgency: 1.0,
                reason: "Model has never been trained — need initial distillation".into(),
            });
            return Orientation {
                priorities,
                swot_analysis: None,
                confidence_score: 1.0, // High confidence for explicit bootstrap
                interaction_prompt: observation.interaction_prompt.clone(),
                interaction_correction: observation.interaction_correction.clone(),
                resource_status: observation.resource_status.clone(),
            };
        }

        // Priority 2: Focus on weakest areas
        let areas = [
            (
                "language_understanding",
                observation.capabilities.language_understanding,
            ),
            ("code_generation", observation.capabilities.code_generation),
            ("math_reasoning", observation.capabilities.math_reasoning),
            (
                "instruction_following",
                observation.capabilities.instruction_following,
            ),
            ("planning", observation.capabilities.planning),
            ("tool_use", observation.capabilities.tool_use),
            (
                "visual_intelligence",
                observation.capabilities.visual_intelligence,
            ),
            (
                "auditory_processing",
                observation.capabilities.auditory_processing,
            ),
            (
                "three_d_synthesis",
                observation.capabilities.three_d_synthesis,
            ),
            (
                "cybersecurity_resilience",
                observation.capabilities.cybersecurity_resilience,
            ),
            (
                "os_terminal_mastery",
                observation.capabilities.os_terminal_mastery,
            ),
            (
                "penetration_testing",
                observation.capabilities.penetration_testing,
            ),
            (
                "agentic_autonomy",
                observation.capabilities.agentic_autonomy,
            ),
            ("safety", observation.capabilities.safety),
        ];

        for (area, score) in &areas {
            if *score < 0.3 {
                priorities.push(Priority {
                    area: area.to_string(),
                    urgency: 1.0 - score,
                    reason: format!("{area} score is low ({score:.2})"),
                });
            }
        }

        // Sort by urgency (highest first)
        // Priority Elite: Robust sorting (handle NaNs just in case)
        priorities.sort_by(|a, b| {
            b.urgency
                .partial_cmp(&a.urgency)
                .unwrap_or(a.urgency.is_nan().cmp(&b.urgency.is_nan()))
        });

        // 3. RECENT FAILURES
        if self.regression_counter > 0 {
            priorities.push(Priority {
                area: "regression_fix".into(),
                urgency: 0.8,
                reason: format!(
                    "Recovery from {} recent regressions",
                    self.regression_counter
                ),
            });
        }

        // 4. AUTONOMOUS MEMORY CONSOLIDATION
        if self.cycle_count > 0 && self.cycle_count.is_multiple_of(20) {
            priorities.push(Priority {
                area: "consolidation".into(),
                urgency: 0.9,
                reason: "Periodic synthesis of episodic memories into long-term knowledge".into(),
            });
        }

        // If no weak areas, benchmark to reassess
        if priorities.is_empty() {
            priorities.push(Priority {
                area: "benchmark".into(),
                urgency: 0.5,
                reason: "All areas adequate — run benchmark for assessment".into(),
            });
        }

        let mut confidence_score = self.calculate_confidence(observation);

        // ── Absolute Perfection: Cognitive Circuit Breaker ──
        // If confidence is ultra-low, we inject a "Hard Reboot" or "Mental Analysis"
        // to prevent hallucination cycles.
        if confidence_score < 0.3 {
            tracing::error!(target: "shakey::ooda", "🛑 CIRCUIT BREAKER: Confidence dropped to {:.2}. Forcing Mental Analysis.", confidence_score);
            priorities.insert(
                0,
                Priority {
                    area: "self_repair".into(),
                    urgency: 1.0,
                    reason: "Emergency circuit breaker: confidence collapse detected.".into(),
                },
            );
            confidence_score = 0.5; // Artificial lift for next step stability
        }

        Orientation {
            priorities,
            swot_analysis: self.generate_swot(observation),
            confidence_score,
            interaction_prompt: observation.interaction_prompt.clone(),
            interaction_correction: observation.interaction_correction.clone(),
            resource_status: observation.resource_status.clone(),
        }
    }

    /// Calculate a meta-cognitive confidence score [0.0 - 1.0].
    fn calculate_confidence(&self, observation: &Observation) -> f64 {
        let mut score = 0.8; // Base confidence

        // Penalty for high failure rate
        if observation.reflection.contains("CRITICAL") {
            score -= 0.4;
        } else if observation.reflection.contains("STAGNATION") {
            score -= 0.2;
        }

        // Regression penalty
        score -= (self.regression_counter as f64) * 0.15;

        // Resource constraint penalty (Pressure-based doubt)
        if observation.resource_status.cpu_usage > 95.0 {
            score -= 0.1;
        }

        // Penalty for recursion depth
        score -= (observation.recursion_depth as f64) * 0.05;

        score.clamp(0.1, 1.0)
    }

    /// SWOT (Strengths, Weaknesses, Opportunities, Threats) logic for Elite Cognition.
    fn generate_swot(&self, observation: &Observation) -> Option<String> {
        let mut swot = String::from("### Elite SWOT Analysis\n");

        // Strength: High capabilities
        if observation.capabilities.overall > 0.7 {
            swot.push_str(
                "- **S**: High general reasoning established. Ready for complex logic synthesis.\n",
            );
        } else {
            swot.push_str("- **S**: Base architecture stable and receptive to distillation.\n");
        }

        // Weakness: Gaps
        swot.push_str(&format!(
            "- **W**: Significant gap in {}.\n",
            observation.weakest_area
        ));

        // Opportunity: Exploration rate
        if self.exploration_rate > 0.1 {
            swot.push_str(
                "- **O**: High exploration rate enables discovery of niche tool-use patterns.\n",
            );
        } else {
            swot.push_str("- **O**: Systematic distillation from NIM teachers offers guaranteed knowledge gain.\n");
        }

        // Threat: Regressions
        if self.regression_counter > 0 {
            swot.push_str(&format!("- **T**: Recent regressions ({}) indicate potential overfitting or corrupt data source.\n", self.regression_counter));
        } else {
            swot.push_str("- **T**: Compute budget constraints could delay stage expansion.\n");
        }

        Some(swot)
    }

    /// DECIDE: Choose the best strategies based on orientation for parallel execution.
    #[instrument(skip(self, orientation), name = "ooda_decide")]
    pub fn decide(&self, orientation: &Orientation) -> Vec<Strategy> {
        if self.current_depth >= self.max_depth {
            return vec![Strategy::Idle {
                reason: format!("Max recursion depth ({}) reached", self.max_depth),
            }];
        }

        // ── Resource Aware Throttling ──
        if orientation.resource_status.cpu_usage > 85.0
            || orientation.resource_status.available_memory_mb < 1024
        {
            tracing::warn!(target: "shakey", "System resources constrained (CPU: {:.1}%, Mem: {}MB). Throttling...", orientation.resource_status.cpu_usage, orientation.resource_status.available_memory_mb);
            return vec![Strategy::Idle {
                reason: format!(
                    "Resource constraints: CPU: {:.1}%, Mem: {}MB",
                    orientation.resource_status.cpu_usage,
                    orientation.resource_status.available_memory_mb
                ),
            }];
        }

        // ── Level 1: Soft Backtrack (3 Regressions) ──
        // If we see 3 consecutive failures, it's highly likely that the latest
        // Online Fine-Tune was noisy or corrupt. We trigger an automatic LoRA roll-back.
        if self.regression_counter >= 3 && self.regression_counter < 5 {
            return vec![Strategy::Backtrack {
                target_version: "pre_lora_stable".into(),
                reason: format!(
                    "CONSECUTIVE REGRESSIONS ({}): LoRA update rejected by circuit breaker. Reverting to last known-good state.",
                    self.regression_counter
                ),
            }];
        }

        // ── Level 1.25: Data Freshness Check (4 Regressions) ──
        if self.regression_counter == 4 {
            return vec![Strategy::WebSearch {
                query: format!(
                    "latest breakthroughs in {}",
                    orientation
                        .priorities
                        .first()
                        .map(|p| p.area.as_str())
                        .unwrap_or("general knowledge")
                ),
            }];
        }

        // ── Mental Sandbox: Consistency Check for OnlineFineTune ──
        // Before allowing an online learning update, the OODA Loop simulates the
        // proposed change in a cognitive "sandbox" to verify its integrity.
        //
        // Rules:
        // 1. The chosen correction must be non-empty.
        // 2. The chosen correction must differ meaningfully from the original prompt.
        // 3. The frequency of online updates must not exceed 1 per 2 OODA cycles.
        for priority in &orientation.priorities {
            if priority.area == "online_learning" {
                // Mental Sandbox: Verify the correction is meaningful
                if let (Some(prompt), Some(correction)) = (
                    &orientation.interaction_prompt,
                    &orientation.interaction_correction,
                ) {
                    if prompt == correction || correction.len() < 10 {
                        tracing::warn!(target: "shakey", "Mental Sandbox: Correction is either identical or too trivial to warrant update.");
                        return vec![Strategy::Idle {
                            reason:
                                "Mental Sandbox: Correction too trivial or identical to prompt."
                                    .into(),
                        }];
                    }
                }

                if self.cycle_count > 0 && (self.cycle_count - self.last_online_update_cycle) < 2 {
                    tracing::warn!(
                        target: "shakey",
                        "Mental Sandbox: Throttling OnlineFineTune — last update was cycle {}, current is {}.",
                        self.last_online_update_cycle,
                        self.cycle_count
                    );
                    return vec![Strategy::Idle {
                        reason:
                            "Mental Sandbox: Online update throttled to prevent OODA thrashing."
                                .into(),
                    }];
                }
            }
        }

        // ── Level 1.5: Cognitive Safe Mode (7 Regressions) ──
        if self.regression_counter >= 7 && self.regression_counter < 10 {
            return vec![Strategy::SovereignResearch {
                objective: "SAFE MODE: Analyzing core weights and distillation sources for corruption or catastrophic forgetting.".into(),
            }];
        }

        // ── Level 2: Forced Strategy Rotation (5 Regressions) ──
        if self.regression_counter >= 5 && self.regression_counter < 10 {
            let mut strategies = self.decide_internal(orientation);
            for s in &mut strategies {
                *s = Strategy::SovereignResearch {
                    objective: "Escaping local minima via architectural reflection".into(),
                };
            }
            return strategies;
        }

        // ── Level 3: Absolute Circuit Breaker (10 Regressions) ──
        if self.regression_counter >= 10 {
            return vec![Strategy::HardReboot {
                reason: "ABSOLUTE CIRCUIT BREAKER: Infinite failure loop detected (10+ cycles). Hard rebooting cognitive context.".into(),
            }];
        }

        if orientation.priorities.is_empty() {
            return vec![Strategy::Idle {
                reason: "No priorities identified".into(),
            }];
        }

        let mut selected_strategies = Vec::new();
        let mut occupied_resources = std::collections::HashSet::new();

        // ── Multi-Strategy Parallel Routing ──
        // Iterate through priorities and pick non-conflicting strategies
        // based on resource type (compute, network, gpu, engineering, io).
        // This enables true parallel execution of independent tasks.

        // ── Advanced: Dynamic Parallelism Scaling ──
        // Scale the number of parallel strategies based on VRAM/CPU pressure.

        let pressure = orientation.resource_status.pressure_score;
        let max_parallel = if pressure > 0.9 {
            1 // Safety baseline for absolute recovery
        } else if pressure > 0.7 {
            4 // Conservative recovery
        } else if pressure > 0.4 {
            8 // Balanced evolution
        } else {
            16 // Apex Parallel Throughput
        };

        tracing::debug!(target: "shakey::ooda", "OODA Parallelism configured to {} based on pressure score {:.2}", max_parallel, pressure);

        for priority in &orientation.priorities {
            let strategy = self.strategy_for_priority(priority, orientation);

            // ── Peak Mastery: Mental Sandbox (Deep Simulation) ──
            if !self.mental_sandbox(&strategy, orientation) {
                tracing::info!(target: "shakey", "Mental Sandbox: Strategy {:?} rejected.", strategy);
                continue;
            }

            let resource = match &strategy {
                Strategy::Distill { .. }
                | Strategy::Train { .. }
                | Strategy::VisionDistill { .. }
                | Strategy::OnlineFineTune { .. } => "compute",
                Strategy::WebScrape { .. } | Strategy::WebSearch { .. } => "network",
                Strategy::Benchmark => "gpu/eval",
                Strategy::ToolRepair { .. } | Strategy::ToolBuild { .. } => "engineering",
                _ => "io",
            };

            if !occupied_resources.contains(resource) {
                selected_strategies.push(strategy);
                occupied_resources.insert(resource);
            }

            if selected_strategies.len() >= max_parallel {
                break;
            }
        }

        // ── ELITE: Recursive Self-Correction (The Sovereign Filter) ──
        let mut refined_strategies = Vec::with_capacity(selected_strategies.len());
        for strat in selected_strategies {
            if self.critique_strategy(&strat, orientation) {
                refined_strategies.push(strat);
            } else {
                tracing::warn!(target: "shakey", "Self-Correction: Vetoed strategy {:?}.", strat);
            }
        }

        refined_strategies
    }

    /// Mental Sandbox: Simulates a strategy to predict ROI and safety.
    fn mental_sandbox(&self, strategy: &Strategy, orientation: &Orientation) -> bool {
        match strategy {
            Strategy::Expand { .. } => {
                orientation.confidence_score > 0.8
                    && orientation.priorities.iter().all(|p| p.area != "bootstrap")
            }
            Strategy::ToolRepair { .. } => true,
            Strategy::OnlineFineTune { is_correction, .. } => *is_correction,
            _ => true,
        }
    }

    /// Sovereign Self-Critique: Evaluates a proposed strategy against the
    /// orientation context to prevent architectural regression and safety breaches.
    fn critique_strategy(&self, strategy: &Strategy, _orientation: &Orientation) -> bool {
        match strategy {
            Strategy::ToolBuild { name, .. } => {
                // Prevent building tools that already exist or have dangerous patterns
                !name.contains("rm ") && !name.contains("/")
            }
            Strategy::Distill { token_budget, .. } => {
                // Efficiency check: Don't waste budget on tiny batches
                *token_budget >= 1024
            }
            Strategy::Synthesize { .. } => true,
            Strategy::WebScrape { .. } => true,
            Strategy::SovereignCascade { .. } => true,
            _ => true,
        }
    }

    /// Internal decision logic for circuit-breaker scenarios.
    fn decide_internal(&self, orientation: &Orientation) -> Vec<Strategy> {
        self.decide(orientation)
    }

    #[allow(dead_code)]
    fn clone_with_depth(&self, depth: u32) -> Self {
        Self {
            history: self.history.clone(),
            cycle_count: self.cycle_count,
            capabilities: self.capabilities.clone(),
            min_improvement: self.min_improvement,
            max_depth: self.max_depth,
            current_depth: depth,
            reflection_buffer: self.reflection_buffer.clone(),
            failure_buffer: self.failure_buffer.clone(),
            consecutive_reflections: self.consecutive_reflections,
            exploration_rate: self.exploration_rate,
            regression_counter: self.regression_counter,
            safe_mode_active: self.safe_mode_active,
            total_uptime_secs: self.total_uptime_secs,
            last_online_update_cycle: self.last_online_update_cycle,
            kb: self.kb.clone(),
            nim_client: self.nim_client.clone(),
            core_instructions: self.core_instructions.clone(),
            resource_monitor: std::sync::Mutex::new(ResourceMonitor::new()),
            context_manager: self.context_manager.clone(),
        }
    }

    /// Helper to map a priority area to a specific execution strategy.
    fn strategy_for_priority(&self, priority: &Priority, orientation: &Orientation) -> Strategy {
        match priority.area.as_str() {
            "bootstrap" => Strategy::Distill {
                domain: "general".into(),
                token_budget: 100_000,
            },

            "online_learning" => Strategy::OnlineFineTune {
                // Use the actual user prompt as input and the correction as the target completion.
                // Falls back to priority.reason only if the orientation fields are missing.
                prompt: orientation
                    .interaction_prompt
                    .clone()
                    .unwrap_or_else(|| priority.reason.clone()),
                completion: orientation
                    .interaction_correction
                    .clone()
                    .unwrap_or_else(|| priority.reason.clone()),
                is_correction: true,
            },

            // ── Peak Mastery: Autonomous Self-Healing Tools ──
            // If the reflection detects a FAIL in tool creation, prioritize repair immediately.
            _area
                if self
                    .reflection_buffer
                    .iter()
                    .any(|r| r.contains("FAILURE") && r.contains("ToolBuild")) =>
            {
                Strategy::ToolRepair {
                    name: "latest_failed_tool".into(),
                    error:
                        "Self-healing triggered: OODA loop detected execution failure in new tool"
                            .into(),
                }
            }

            // If overall capability is high (>0.7), consider expanding architecture
            area if self.capabilities.overall > 0.7 && area != "benchmark" => Strategy::Expand {
                target_stage: "stage_2".into(),
            },

            // Synthesize if we have decent baseline capability but need more data
            "language_understanding" | "instruction_following"
                if self.capabilities.overall > 0.4 =>
            {
                Strategy::Synthesize {
                    topic: priority.area.clone(),
                    count: 1000,
                }
            }

            "language_understanding" | "instruction_following" => Strategy::Distill {
                domain: "instruction_following".into(),
                token_budget: 50_000,
            },
            "code_generation" => Strategy::Distill {
                domain: "code".into(),
                token_budget: 50_000,
            },
            "math_reasoning" => Strategy::Distill {
                domain: "math".into(),
                token_budget: 50_000,
            },

            // Tool building if tool_use is the weakest area
            "tool_use" if self.capabilities.tool_use < 0.3 => Strategy::ToolBuild {
                name: "advanced_web_search".into(),
                description: "Improved multi-hop web searching capability".into(),
            },

            // ELITE: Sovereign Optimization for existing tools
            "tool_use"
                if self.capabilities.tool_use >= 0.6 && self.cycle_count.is_multiple_of(10) =>
            {
                Strategy::SovereignOptimization {
                    tool_name: "web_search".into(), // Dynamically pick from registry in the future
                    metric: "latency".into(),
                }
            }

            "planning" | "tool_use" => Strategy::Distill {
                domain: "planning".into(),
                token_budget: 50_000,
            },
            "visual_intelligence" => Strategy::VisionDistill {
                image_path: "shakey_data/latest_vision_capture.png".into(), // Real-world data path
                objective: "Analyze visual constraints and logic".into(),
            },
            "auditory_processing" => Strategy::Distill {
                domain: "audio_transcription".into(),
                token_budget: 20_000,
            },
            "three_d_synthesis" => Strategy::Distill {
                domain: "3d_synthesis".into(),
                token_budget: 20_000,
            },
            "safety" => Strategy::Distill {
                domain: "safety".into(),
                token_budget: 20_000,
            },
            "sovereign_logic" | "meta_cognition" if self.cycle_count.is_multiple_of(50) => {
                Strategy::SelfIndex {
                    workspace_path: ".".into(),
                }
            }
            "sovereign_logic" | "meta_cognition" if self.capabilities.overall > 0.7 => {
                Strategy::SovereignCascade {
                    strategies: vec![
                        Strategy::MentalAnalysis {
                            objective: "Universal Logic Audit".into(),
                        },
                        Strategy::Synthesize {
                            topic: "meta_logic".into(),
                            count: 100,
                        },
                    ],
                    objective: "Deep Sovereign Alignment".into(),
                }
            }
            "sovereign_logic" | "meta_cognition" => Strategy::SovereignResearch {
                objective: "Enhancing self-directed logic and architectural complexity".into(),
            },
            "benchmark" => Strategy::Benchmark,
            "web_research" => Strategy::WebScrape {
                query: priority.reason.clone(),
                max_pages: 5,
            },
            "consolidation" => Strategy::Consolidate { cycle_count: 20 },
            _ => Strategy::Distill {
                domain: priority.area.clone(),
                token_budget: 30_000,
            },
        }
    }

    /// --- ZENITH 5.0: SOVEREIGN REFLECTION (System 2 Thinking) ---
    ///
    /// Evaluates the decided strategies before they are passed to the execution layer.
    /// If meta-cognition is high, the agent will "pause" to critique its own plan.
    pub fn audit_strategies(&mut self, strategies: Vec<Strategy>) -> Vec<Strategy> {
        let meta_score = self.capabilities.meta_cognition;

        strategies.into_iter().map(|s| {
            // High meta-cognition triggers automatic System 2 reflection
            if meta_score > 0.6 {
                let reasoning = format!("Sovereign Audit: Evaluating {} validity against current capabilities ({:.2})", s, meta_score);
                tracing::info!(target: "shakey::ooda", "🧠 Reflection: {}", reasoning);
                Strategy::Reflect {
                    strategy: Box::new(s),
                    reasoning,
                }
            } else {
                s
            }
        }).collect()
    }

    /// --- ZENITH 5.1: SOVEREIGN SELF-CORRECTION ---
    ///
    /// Analyzes the failure buffer to identify if the agent is stuck in a logic loop.
    pub fn cognitive_self_correct(&mut self, orient: &Orientation) -> Vec<Strategy> {
        if self.consecutive_reflections >= 3 {
            tracing::warn!(target: "shakey::ooda", "🚨 COGNITIVE LOCK DETECTED: {} consecutive rejections. Forcing mental reset.", self.consecutive_reflections);
            self.consecutive_reflections = 0;
            self.failure_buffer.clear();

            return vec![Strategy::MentalAnalysis {
                objective: format!("Analyze why the previous strategies for '{}' were rejected and propose a radical new approach.", orient.priorities[0].area),
            }];
        }
        Vec::new()
    }

    /// Record a completed cycle and update capabilities based on progress.
    #[instrument(skip(self, record), fields(cycle_id = %record.cycle_id))]
    pub fn record_cycle(&mut self, record: CycleRecord) {
        if record.committed && record.success && record.improvement > 0.0 {
            // Update the relevant capability based on record.improvement
            match &record.strategy {
                Strategy::Distill { domain, .. } => {
                    self.update_capability_from_domain(domain, record.improvement);
                }
                Strategy::VisionDistill { .. } => {
                    self.update_capability_from_domain("vision_video", record.improvement);
                }
                Strategy::Synthesize { topic, .. } => {
                    self.update_capability_from_domain(topic, record.improvement);
                }
                Strategy::OnlineFineTune { .. } => {
                    // Online learning improves instruction-following and language comprehension
                    self.update_capability_from_domain(
                        "instruction_following",
                        record.improvement * 0.6,
                    );
                    self.update_capability_from_domain(
                        "language_understanding",
                        record.improvement * 0.4,
                    );
                    self.last_online_update_cycle = self.cycle_count;
                }
                Strategy::ToolBuild { .. } | Strategy::ToolRepair { .. } => {
                    self.update_capability_from_domain("tool_use", record.improvement);
                }
                Strategy::WebScrape { .. } | Strategy::WebSearch { .. } => {
                    self.update_capability_from_domain("tool_use", record.improvement * 0.5);
                    self.update_capability_from_domain("planning", record.improvement * 0.5);
                }
                Strategy::SovereignResearch { .. } | Strategy::MentalAnalysis { .. } => {
                    self.update_capability_from_domain("meta_cognition", record.improvement);
                }
                Strategy::SovereignOptimization { .. } => {
                    self.update_capability_from_domain("tool_use", record.improvement * 1.2);
                }
                Strategy::SelfIndex { .. } => {
                    self.update_capability_from_domain("meta_cognition", record.improvement * 1.5);
                    self.update_capability_from_domain("planning", record.improvement * 0.5);
                }
                Strategy::Consolidate { .. } => {
                    self.update_capability_from_domain("meta_cognition", record.improvement * 2.0);
                    self.update_capability_from_domain("planning", record.improvement * 0.5);
                }
                Strategy::Benchmark
                | Strategy::Idle { .. }
                | Strategy::HardReboot { .. }
                | Strategy::Backtrack { .. }
                | Strategy::Expand { .. } => {}
                Strategy::Train { .. }
                | Strategy::Reflect { .. }
                | Strategy::ConsensusAudit { .. } => {}
                Strategy::SovereignCascade { .. } => {
                    self.update_capability_from_domain("meta_cognition", record.improvement * 1.5);
                    self.update_capability_from_domain("planning", record.improvement * 1.5);
                }
                Strategy::NativeToolCall { .. } => {
                    self.update_capability_from_domain("tool_use", record.improvement * 1.1);
                }
            }
            self.capabilities.compute_overall();
            self.capabilities.clamp();

            // Reset regression counter on genuine improvement
            self.regression_counter = 0;
        } else if record.improvement <= 0.0 {
            // Track consecutive regression or stagnation
            self.regression_counter += 1;
        }

        // --- Peak Mastery: Ultra-Elite Self-Audit ---
        self.self_audit(&record);

        self.cycle_count += 1;
        self.history.push(record);
    }

    /// Self-Audit: Critiques a completed cycle for logic consistency.
    fn self_audit(&self, record: &CycleRecord) {
        if !record.success && record.committed {
            tracing::warn!(target: "shakey", "Self-Audit: Failed cycle unexpectedly committed.");
        }
    }

    fn update_capability_from_domain(&mut self, domain: &str, improvement: f64) {
        match domain {
            "general" | "language_understanding" => {
                self.capabilities.language_understanding += improvement
            }
            "code" | "code_generation" => self.capabilities.code_generation += improvement,
            "math" | "math_reasoning" => self.capabilities.math_reasoning += improvement,
            "instruction_following" => self.capabilities.instruction_following += improvement,
            "planning" => self.capabilities.planning += improvement,
            "tool_use" => self.capabilities.tool_use += improvement,
            "vision_video" | "visual_intelligence" => {
                self.capabilities.visual_intelligence += improvement
            }
            "audio_transcription" | "auditory_processing" => {
                self.capabilities.auditory_processing += improvement
            }
            "3d_synthesis" | "three_d_synthesis" => {
                self.capabilities.three_d_synthesis += improvement
            }
            "safety" => self.capabilities.safety += improvement,
            "sovereign_research" => {
                self.capabilities.meta_cognition += improvement;
                self.capabilities.planning += improvement * 0.5;
            }
            _ => {
                // Distributed improvement across all areas for general tasks
                self.capabilities.language_understanding += improvement * 0.2;
                self.capabilities.instruction_following += improvement * 0.2;
            }
        }
        // Clamp to [0, 1]
        self.capabilities.clamp();
    }

    /// Find the area with the lowest capability score.
    fn find_weakest_area(&self) -> String {
        let areas = [
            (
                "language_understanding",
                self.capabilities.language_understanding,
            ),
            ("code_generation", self.capabilities.code_generation),
            ("math_reasoning", self.capabilities.math_reasoning),
            (
                "instruction_following",
                self.capabilities.instruction_following,
            ),
            ("planning", self.capabilities.planning),
            ("tool_use", self.capabilities.tool_use),
            ("visual_intelligence", self.capabilities.visual_intelligence),
            ("auditory_processing", self.capabilities.auditory_processing),
            ("three_d_synthesis", self.capabilities.three_d_synthesis),
            ("safety", self.capabilities.safety),
            ("meta_cognition", self.capabilities.meta_cognition),
        ];

        areas
            .iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Greater))
            .map(|(name, _): &(&str, f64)| name.to_string())
            .unwrap_or_else(|| "unknown_area".into())
    }

    /// ── Zenith Sovereign: Autopoietic Rebalance ──
    /// Dynamically shifts the teacher API budget based on the selected strategy.
    pub async fn rebalance_nim_for_strategy(&self, strategy: &crate::Strategy) {
        if let Some(nim) = &self.nim_client {
            let role = match strategy {
                crate::Strategy::Distill { domain, .. } => domain.as_str(),
                crate::Strategy::VisionDistill { .. } => "vision_video",
                crate::Strategy::Synthesize { topic, .. } => topic.as_str(),
                crate::Strategy::OnlineFineTune { .. } => "reasoning",
                crate::Strategy::ToolBuild { .. } | crate::Strategy::ToolRepair { .. } => "code",
                crate::Strategy::SovereignResearch { .. } => "reasoning",
                crate::Strategy::Consolidate { .. } => "embedding",
                _ => "reasoning", // Default to reasoning for all other strategies
            };

            // Map common OODA areas to NIM Teacher roles
            let target_role = match role {
                "general"
                | "language_understanding"
                | "instruction_following"
                | "meta_cognition" => "reasoning",
                "code" | "code_generation" => "code",
                "math" | "math_reasoning" => "math",
                "vision_video" | "visual_intelligence" => "vision_video",
                "audio_transcription" | "auditory_processing" => "audio_transcription",
                "3d_synthesis" | "three_d_synthesis" => "3d_synthesis",
                "translation" | "dialect" | "translation_dialect" => "translation_dialect",
                "safety" | "alignment" => "safety",
                r => r,
            };

            // Zenith Apex II: Urgency Detection
            let is_urgent = matches!(
                strategy,
                crate::Strategy::HardReboot { .. }
                    | crate::Strategy::Backtrack { .. }
                    | crate::Strategy::ToolRepair { .. }
                    | crate::Strategy::SovereignResearch { .. }
            );

            if is_urgent {
                tracing::warn!(target: "shakey::sovereign", "🚨 URGENT AUTOPOIETIC SHIFT: Directing 80% attention to [{}]. Bypassing hysteresis.", target_role);
            } else {
                tracing::info!(target: "shakey::sovereign", "🧠 Dynamic Autopoietic Budgeting: Directing attention to [{}].", target_role);
            }
            let _ = nim.rebalance_budget(target_role, is_urgent).await;
        }
    }

    /// Async multi-threaded observation bound by tokio spawn thread-pool
    pub async fn observe_spawned(self_arc: Arc<Self>) -> Observation {
        tokio::task::spawn_blocking(move || self_arc.observe())
            .await
            .unwrap_or_else(|_| Observation {
                capabilities: CapabilityMatrix::default(),
                weakest_area: "unknown".into(),
                total_cycles: 0,
                tokens_trained: 0,
                last_improvement: 0.0,
                recursion_depth: 0,
                reflection: "SPAWN_FAILED".into(),
                episodic_memories: vec![],
                interaction_prompt: None,
                interaction_correction: None,
                resource_status: ResourceStatus {
                    cpu_usage: 0.0,
                    available_memory_mb: 0,
                    disk_iops_utilization: 0.0,
                    vram_usage: 0.0,
                    pressure_score: 0.0,
                },
                failure_buffer: vec![],
                consecutive_reflections: 0,
            })
    }

    /// Async multi-threaded decision bound by tokio spawn thread-pool
    pub async fn decide_spawned(
        self_arc: Arc<Self>,
        orientation: Arc<Orientation>,
    ) -> Vec<Strategy> {
        tokio::task::spawn_blocking(move || self_arc.decide(&orientation))
            .await
            .unwrap_or_else(|_| {
                vec![Strategy::Idle {
                    reason: "SPAWN_FAILED".into(),
                }]
            })
    }

    /// EXTRACT: Separate reasoning/thought tokens from the final response text.
    /// Handles `<thought>` tags used by modern reasoning models.
    pub fn extract_thoughts(&self, text: &str) -> (Option<String>, String) {
        if let (Some(start), Some(end)) = (text.find("<thought>"), text.find("</thought>")) {
            let thought = text[start + 9..end].trim().to_string();
            let mut remaining = text[..start].to_string();
            remaining.push_str(&text[end + 10..]);
            (Some(thought), remaining.trim().to_string())
        } else if let (Some(start), Some(end)) = (text.find("<think>"), text.find("</think>")) {
            // Support for DeepSeek/R1 style
            let thought = text[start + 7..end].trim().to_string();
            let mut remaining = text[..start].to_string();
            remaining.push_str(&text[end + 8..]);
            (Some(thought), remaining.trim().to_string())
        } else {
            (None, text.to_string())
        }
    }

    /// CONTEXT: Manage the sliding window context before a major decision.
    pub async fn prepare_context(&mut self) -> Result<(), anyhow::Error> {
        if let (Some(cm), Some(kb)) = (self.context_manager.as_mut(), self.kb.as_ref()) {
            // Include recent history in the context management
            let recent_history = self.reflection_buffer.join("\n");
            let core = self.core_instructions.join("\n");

            // Industrial Pruning: Keep core instructions but slide the rest.
            let context_content =
                format!("SYSTEM_CORE: {}\nRECENT_HISTORY: {}", core, recent_history);
            cm.add_context(&context_content);

            // If the knowledge base has relevant facts, inject them too.
            if let Ok(facts) = kb.search_similar("Current Goal", 3).await {
                for fact in facts {
                    cm.add_context(&format!("FACT: {}", fact.content));
                }
            }
        }
        Ok(())
    }
}

/// Result of the ORIENT phase.
#[derive(Debug, Clone)]
pub struct Orientation {
    pub priorities: Vec<Priority>,
    /// Ultra-Elite: Qualitative SWOT analysis from a teacher-as-judge
    pub swot_analysis: Option<String>,
    /// Meta-Cognitive metric of certainty
    pub confidence_score: f64,
    /// Latest user interaction prompt from Observation phase
    pub interaction_prompt: Option<String>,
    /// Latest user interaction correction from Observation phase
    pub interaction_correction: Option<String>,
    /// System resource status
    pub resource_status: ResourceStatus,
}

/// A prioritized learning objective.
#[derive(Debug, Clone)]
pub struct Priority {
    pub area: String,
    pub urgency: f64, // 0.0 = low, 1.0 = critical
    pub reason: String,
}

/// --- Advanced Generation: Speculative Decoding Acceptance ---
///
/// Implements real Medusa token verification via confidence-gated probability comparison.
///
/// For each Medusa head:
/// 1. Compute softmax of both base logits and Medusa head logits
/// 2. Pick the Medusa candidate (argmax of head logits)
/// 3. Check if base model also assigns sufficient probability to that candidate
/// 4. Accept if: P_base(candidate) > threshold × max(P_base)
///
/// Returns only the accepted candidate token IDs (in head order).
/// --- Advanced Generation: Speculative Decoding Acceptance ---
pub fn accept_speculative_tokens(
    logits: &candle_core::Tensor,
    medusa_logits: &candle_core::Tensor,
    threshold: f32,
) -> candle_core::Result<Vec<u32>> {
    use candle_core::{IndexOp, D};
    let (_batch, n_heads, _vocab) = medusa_logits.dims3()?;
    let mut accepted = Vec::new();
    let base_probs = candle_nn::ops::softmax(&logits.squeeze(0)?, D::Minus1)?;
    let base_max_prob = base_probs.max(D::Minus1)?.to_vec0::<f32>()?;
    let acceptance_floor = threshold * base_max_prob;

    for i in 0..n_heads {
        let head_logits = medusa_logits.i((.., i, ..))?.squeeze(0)?;
        let candidate_id = head_logits.argmax(D::Minus1)?.to_vec0::<u32>()?;
        let candidate_base_prob = base_probs.i(candidate_id as usize)?.to_vec0::<f32>()?;
        if candidate_base_prob >= acceptance_floor {
            accepted.push(candidate_id);
        } else {
            break;
        }
    }
    Ok(accepted)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ooda_bootstrap() -> anyhow::Result<()> {
        let mut ooda = OodaLoop::new(0.01);
        let obs = ooda.observe();
        assert_eq!(obs.total_cycles, 0);

        let orient = ooda.orient(&obs);
        assert!(!orient.priorities.is_empty());
        assert_eq!(orient.priorities[0].area, "bootstrap");

        let strategies = ooda.decide(&orient);
        if !matches!(strategies.first(), Some(Strategy::Distill { .. })) {
            anyhow::bail!(
                "Sovereign Boot Failure: Expected Distill strategy for bootstrap, got {:?}",
                strategies.first()
            );
        }
        Ok(())
    }

    #[test]
    fn test_ooda_weak_area_focus() {
        let mut ooda = OodaLoop::new(0.01);
        ooda.cycle_count = 10; // Not a bootstrap
        ooda.capabilities.language_understanding = 0.8;
        ooda.capabilities.code_generation = 0.1; // Weak!
        ooda.capabilities.math_reasoning = 0.6;
        ooda.capabilities.instruction_following = 0.7;
        ooda.capabilities.planning = 0.5;
        ooda.capabilities.tool_use = 0.4;
        ooda.capabilities.visual_intelligence = 0.9;
        ooda.capabilities.auditory_processing = 0.9;
        ooda.capabilities.three_d_synthesis = 0.9;
        ooda.capabilities.cybersecurity_resilience = 1.0;
        ooda.capabilities.os_terminal_mastery = 1.0;
        ooda.capabilities.penetration_testing = 1.0;
        ooda.capabilities.agentic_autonomy = 1.0;
        ooda.capabilities.safety = 0.9;
        ooda.capabilities.meta_cognition = 0.9;

        let obs = ooda.observe();
        let orient = ooda.orient(&obs);

        // Should prioritize code_generation (score 0.1)
        assert_eq!(orient.priorities[0].area, "code_generation");
    }
}
