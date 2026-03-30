//! Benchmark runner — evaluates the agent across various capabilities.
//!
//! Measures performance using standardized benchmark suites:
//! - MMLU (General Reasoning)
//! - HumanEval (Code Generation)
//! - GSM8K (Math Reasoning)
//! - TruthfulQA (Instruction Following)

use crate::CapabilityMatrix;
use anyhow::Result;
use serde_json::Value;
use std::collections::HashMap;

/// Benchmark suite definition.
pub struct BenchmarkSuite {
    pub name: String,
    pub tests: Vec<BenchmarkTest>,
}

/// A single benchmark test case.
pub struct BenchmarkTest {
    pub prompt: String,
    pub reference: String,
    pub domain: String,
}

/// Benchmark runner for the evolution engine.
pub struct BenchmarkRunner {
    pub suites: HashMap<String, BenchmarkSuite>,
    /// Previous benchmark runs for trend analysis
    pub history: Vec<CapabilityMatrix>,
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchmarkRunner {
    pub fn new() -> Self {
        Self {
            suites: HashMap::new(),
            history: Vec::new(),
        }
    }

    /// Load benchmark suites from JSON files.
    pub fn load_suites(&mut self, path: &str) -> Result<()> {
        let content = std::fs::read_to_string(path)?;
        let json: Value = serde_json::from_str(&content)?;

        if let Some(suites) = json.as_array() {
            for suite_json in suites {
                let name = suite_json["name"].as_str().unwrap_or("unknown").to_string();
                let mut tests = Vec::new();

                if let Some(tests_json) = suite_json["tests"].as_array() {
                    for test in tests_json {
                        tests.push(BenchmarkTest {
                            prompt: test["prompt"].as_str().unwrap_or("").to_string(),
                            reference: test["reference"].as_str().unwrap_or("").to_string(),
                            domain: test["domain"].as_str().unwrap_or("general").to_string(),
                        });
                    }
                }

                self.suites
                    .insert(name.clone(), BenchmarkSuite { name, tests });
            }
        }

        Ok(())
    }

    /// Run all suites and return a capability matrix.
    pub async fn run_all(
        &self,
        model: &shakey_core::model::transformer::TransformerModel,
        tokenizer: &shakey_core::tokenizer::Tokenizer,
        device: &candle_core::Device,
    ) -> Result<CapabilityMatrix> {
        let mut caps = CapabilityMatrix::default();
        let mut counts = HashMap::new();
        let mut scores = HashMap::new();

        let engine = shakey_core::inference::InferenceEngine::new(model, tokenizer, device.clone());
        let params = shakey_core::inference::SamplingParams {
            temperature: 0.2, // Low temperature for deterministic benchmarking
            max_tokens: 128,
            ..Default::default()
        };

        for (name, suite) in &self.suites {
            tracing::info!(target: "shakey", "Running benchmark suite: {}", name);
            for test in &suite.tests {
                // Real Model Inference
                let response = match engine.generate(&test.prompt, &params) {
                    Ok(resp) => resp,
                    Err(e) => {
                        tracing::error!("Benchmark inference failed: {}", e);
                        String::new()
                    }
                };

                // ── ELITE Multi-Tiered Scoring Engine ──
                // Tier 1: Exact match (1.0)
                // Tier 2: Numeric proximity (ratio-based)
                // Tier 3: Substring containment (0.8)
                // Tier 4: Token overlap / Jaccard similarity
                // Tier 5: Character-level Levenshtein similarity
                let response_trim = response.trim().to_lowercase();
                let reference_trim = test.reference.trim().to_lowercase();

                let score = if response_trim == reference_trim {
                    // Tier 1: Perfect match
                    1.0
                } else if let (Ok(r_val), Ok(ref_val)) =
                    (response_trim.parse::<f64>(), reference_trim.parse::<f64>())
                {
                    // Tier 2: Numeric — ratio-based proximity
                    if (r_val - ref_val).abs() < 1e-6 {
                        1.0
                    } else if ref_val.abs() > 1e-12 {
                        let ratio = 1.0 - ((r_val - ref_val).abs() / ref_val.abs()).min(1.0);
                        ratio.max(0.0)
                    } else {
                        0.0
                    }
                } else if response_trim.contains(&reference_trim) {
                    // Tier 3: Response contains reference as substring
                    0.8
                } else {
                    // Tier 4: Token overlap (Jaccard similarity)
                    let resp_tokens: std::collections::HashSet<&str> =
                        response_trim.split_whitespace().collect();
                    let ref_tokens: std::collections::HashSet<&str> =
                        reference_trim.split_whitespace().collect();
                    let intersection = resp_tokens.intersection(&ref_tokens).count();
                    let union = resp_tokens.union(&ref_tokens).count();

                    let jaccard = if union > 0 {
                        intersection as f64 / union as f64
                    } else {
                        0.0
                    };

                    // Tier 5: Character-level Dice coefficient (Sørensen–Dice)
                    // Uses character bigram overlap for robust fuzzy matching.
                    let dice_sim = {
                        let bigrams_a: std::collections::HashSet<(u8, u8)> = response_trim
                            .as_bytes()
                            .windows(2)
                            .map(|w| (w[0], w[1]))
                            .collect();
                        let bigrams_b: std::collections::HashSet<(u8, u8)> = reference_trim
                            .as_bytes()
                            .windows(2)
                            .map(|w| (w[0], w[1]))
                            .collect();

                        if bigrams_a.is_empty() && bigrams_b.is_empty() {
                            1.0 // Both single-char or empty
                        } else {
                            let overlap = bigrams_a.intersection(&bigrams_b).count();
                            (2.0 * overlap as f64) / (bigrams_a.len() + bigrams_b.len()) as f64
                        }
                    };

                    // Combined: weighted average of Jaccard + Dice coefficient
                    (0.6 * jaccard + 0.4 * dice_sim).min(1.0)
                };

                let entry = scores.entry(test.domain.clone()).or_insert(0.0);
                *entry += score;

                let count = counts.entry(test.domain.clone()).or_insert(0);
                *count += 1;
            }
        }

        // Aggregate scores for each dimension
        if let Some(&c) = counts.get("language_understanding") {
            caps.language_understanding =
                scores.get("language_understanding").unwrap_or(&0.0) / c as f64;
        }
        if let Some(&c) = counts.get("code_generation") {
            caps.code_generation = scores.get("code_generation").unwrap_or(&0.0) / c as f64;
        }
        if let Some(&c) = counts.get("math_reasoning") {
            caps.math_reasoning = scores.get("math_reasoning").unwrap_or(&0.0) / c as f64;
        }
        if let Some(&c) = counts.get("instruction_following") {
            caps.instruction_following =
                scores.get("instruction_following").unwrap_or(&0.0) / c as f64;
        }
        if let Some(&c) = counts.get("planning") {
            caps.planning = scores.get("planning").unwrap_or(&0.0) / c as f64;
        }
        if let Some(&c) = counts.get("tool_use") {
            caps.tool_use = scores.get("tool_use").unwrap_or(&0.0) / c as f64;
        }
        if let Some(&c) = counts.get("visual_intelligence") {
            caps.visual_intelligence = scores.get("visual_intelligence").unwrap_or(&0.0) / c as f64;
        }
        if let Some(&c) = counts.get("auditory_processing") {
            caps.auditory_processing = scores.get("auditory_processing").unwrap_or(&0.0) / c as f64;
        }
        if let Some(&c) = counts.get("three_d_synthesis") {
            caps.three_d_synthesis = scores.get("three_d_synthesis").unwrap_or(&0.0) / c as f64;
        }
        if let Some(&c) = counts.get("safety") {
            caps.safety = scores.get("safety").unwrap_or(&0.0) / c as f64;
        }

        // Compute overall
        caps.compute_overall();

        // ── Ultra-Elite: Automated Root Cause Analysis (RCA) ──
        if let Some(prev) = self.history.last() {
            if caps.overall < prev.overall - 0.02 {
                tracing::warn!(
                    "REGRESSION DETECTED: {:.2} -> {:.2}. Triggering RCA.",
                    prev.overall,
                    caps.overall
                );
                self.analyze_regression(prev, &caps);
            }
        }

        Ok(caps)
    }

    /// ELITE: Full Root Cause Analysis across all 10 capability dimensions.
    /// Identifies the top-3 regressing areas with actionable recommendations.
    fn analyze_regression(&self, prev: &CapabilityMatrix, current: &CapabilityMatrix) {
        let mut deltas: Vec<(&str, f64)> = vec![
            (
                "language_understanding",
                prev.language_understanding - current.language_understanding,
            ),
            (
                "code_generation",
                prev.code_generation - current.code_generation,
            ),
            (
                "math_reasoning",
                prev.math_reasoning - current.math_reasoning,
            ),
            (
                "instruction_following",
                prev.instruction_following - current.instruction_following,
            ),
            ("planning", prev.planning - current.planning),
            ("tool_use", prev.tool_use - current.tool_use),
            (
                "visual_intelligence",
                prev.visual_intelligence - current.visual_intelligence,
            ),
            (
                "auditory_processing",
                prev.auditory_processing - current.auditory_processing,
            ),
            (
                "three_d_synthesis",
                prev.three_d_synthesis - current.three_d_synthesis,
            ),
            ("safety", prev.safety - current.safety),
        ];

        // Sort by regression severity (largest positive delta = worst regression)
        deltas.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let regressions: Vec<_> = deltas.iter().filter(|(_, d)| *d > 0.01).take(3).collect();

        if regressions.is_empty() {
            tracing::info!(target: "shakey", "RCA: No single-dimension regression found. Overall decline is distributed across all areas.");
            return;
        }

        for (i, (area, delta)) in regressions.iter().enumerate() {
            let severity = if *delta > 0.10 {
                "CRITICAL"
            } else if *delta > 0.05 {
                "HIGH"
            } else {
                "MODERATE"
            };
            let recommendation = match *area {
                "code_generation" => "Increase code teacher sampling. Check for noisy synthetic code data in the 'Code' role pool.",
                "math_reasoning" => "Schedule focused math distillation with high-priority 'Reasoning' or 'SovereignDistill' teachers.",
                "language_understanding" => "Possible catastrophic forgetting from overtrained LoRA. Consider reducing online_learning rate.",
                "instruction_following" => "DPO correction data may be conflicting. Audit replay buffer for contradictory pairs.",
                "planning" => "Multi-step planning degradation suggests insufficient chain-of-thought training.",
                "tool_use" => "Tool execution patterns may have shifted. Re-calibrate with fresh tool-use examples.",
                _ => "Schedule targeted distillation from specialist teachers.",
            };
            tracing::warn!(
                target: "shakey",
                "RCA #{}: [{}] {} regressed by {:.1}% — {}",
                i + 1, severity, area, delta * 100.0, recommendation
            );
        }
    }

    /// Evaluate a response using a Teacher model as a judge.
    pub async fn teacher_judge(
        &self,
        nim: &shakey_distill::nim_client::NimClient,
        prompt: &str,
        response: &str,
        reference: &str,
    ) -> Result<f64> {
        let judge_prompt = format!(
            "Task: Score the LLM response below based on the reference.\n\
             Criteria: Logic, accuracy, and adherence to instructions.\n\
             Score format: Just a float between 0.0 and 1.0.\n\n\
             Prompt: {}\n\nReference: {}\n\nResponse: {}",
            prompt, reference, response
        );

        let model = nim
            .resolve_model_for_role(&shakey_distill::teacher::TeacherRole::Critique)
            .await
            .unwrap_or_else(|_| {
                shakey_distill::teacher::TeacherRole::Critique
                    .default_model()
                    .to_string()
            });
        let res = nim.query(&model, "judge", &judge_prompt, 10, 0.1).await?;
        let score: f64 = res.trim().parse().unwrap_or(0.0);
        Ok(score.clamp(0.0, 1.0))
    }
}
