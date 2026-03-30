use crate::adversary::AdversarialCritic;
use crate::nim_client::{ChatMessage, ChatRequest, NimClient};
use crate::teacher::TeacherRole;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::Arc;

/// Modality types for universal synthesis.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Text,
    Code,
    Logic,
    #[serde(rename = "3d")]
    ThreeD,
    Audio,
    Visual,
    #[serde(rename = "red_team")]
    RedTeam,
    #[serde(rename = "os_mastery")]
    OperatingSystem,
}

impl fmt::Display for Modality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text => write!(f, "textual"),
            Self::Code => write!(f, "source_code"),
            Self::Logic => write!(f, "abstract_logic"),
            Self::ThreeD => write!(f, "spatial_3d"),
            Self::Audio => write!(f, "auditory_tonal"),
            Self::Visual => write!(f, "visual_compositional"),
            Self::RedTeam => write!(f, "offensive_defensive_security"),
            Self::OperatingSystem => write!(f, "kernel_terminal_internals"),
        }
    }
}

/// A dynamic schema for a newly discovered intellectual domain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlchemistSchema {
    pub domain_name: String,
    pub modality: Modality,
    pub description: String,
    pub instruction_format: String,
    pub sample_seed_prompts: Vec<String>,
    pub complexity_tier: u32, // 1-10
}

/// Universal Alchemist — The "Dreaming" engine of Project Shakey.
pub struct UniversalAlchemist {
    pub nim: NimClient,
    pub critic: Arc<AdversarialCritic>,
}

impl UniversalAlchemist {
    pub fn new(nim: NimClient) -> Self {
        let critic = Arc::new(AdversarialCritic::new(nim.clone()));
        Self { nim, critic }
    }

    /// ALPHA-GRADE HEALER: Autonomously corrects malformed JSON using LLM.
    async fn heal_json(&self, malformed: &str, error: &str, domain: &str) -> Result<String> {
        tracing::warn!(target: "shakey::alchemist", "🩹 Alchemist: Healing malformed JSON for domain '{}'...", domain);

        let prompt = format!(
            "System: You are the Sovereign Healer of Project Shakey.\n\
             Context: A Teacher model generated malformed JSON for the domain: {}.\n\
             Error: {}\n\
             Malformed Input: {}\n\n\
             Task: Fix the JSON and ensure it strictly follows the schema. Output JSON ONLY.",
            domain, error, malformed
        );

        self.nim
            .query_for_role(TeacherRole::Healing, "healer", &prompt)
            .await
    }

    /// Discover a new, high-value intellectual domain for synthetic generation.
    pub async fn discover_schema(&self, existing_capabilities: String) -> Result<AlchemistSchema> {
        tracing::info!(target: "shakey::alchemist", "🧠 Alchemist: Dreaming of new intellectual domains...");

        let prompt = format!(
            "System: You are the Sovereign Alchemist of Project Shakey.\n\
             Context: Current agent capabilities: {}.\n\
             Task: Discover a futuristic intellectual domain (Text, Code, Logic, 3D, Audio, Visual, RedTeam, or OperatingSystem).\n\
             Output a JSON schema matching AlchemistSchema struct.",
            existing_capabilities
        );

        let mut response = self
            .nim
            .query_for_role(TeacherRole::Reasoning, "alchemist", &prompt)
            .await?;

        // Self-Healing Retry Loop
        let mut attempts = 0;
        loop {
            let clean_json = crate::utils::extract_json(&response);
            match serde_json::from_str::<AlchemistSchema>(&clean_json) {
                Ok(s) => {
                    tracing::info!(target: "shakey::alchemist", "✨ Discovered New Domain: {} (Tier {})", s.domain_name, s.complexity_tier);
                    return Ok(s);
                }
                Err(_e) if attempts < 2 => {
                    attempts += 1;
                    response = self
                        .heal_json(&response, &_e.to_string(), "schema_discovery")
                        .await?;
                }
                Err(e) => {
                    return Err(anyhow::anyhow!(
                        "Alchemist failed to dream a valid schema after {} attempts: {}",
                        attempts,
                        e
                    ))
                }
            }
        }
    }

    /// Refine a batch of examples using parallel recursive critique with Multi-Teacher Fallback.
    pub async fn refine_batch(
        &self,
        examples: Vec<(String, String)>,
        critique_objective: &str,
    ) -> Result<Vec<(String, String)>> {
        tracing::info!(target: "shakey::alchemist", "🚀 Refining batch of {} examples in parallel...", examples.len());

        let primary_config = self
            .nim
            .resolve_config_for_role(&crate::teacher::TeacherRole::SovereignDistill)
            .await
            .unwrap_or_else(|_| {
                let role = crate::teacher::TeacherRole::SovereignDistill;
                crate::teacher::TeacherConfig {
                    model: role.default_model().to_string(),
                    role,
                    priority: 1,
                    max_tokens: 8192,
                    temperature: 0.3,
                    top_logprobs: 5,
                    enabled: true,
                }
            });

        let requests: Vec<(ChatRequest, String)> = examples.iter().map(|(prompt, completion)| {
            let refine_prompt = format!(
                "### Sovereign Refinement Objective: {}\n\n\
                 Original Prompt: {}\nOriginal Completion: {}\n\n\
                 Task: Critique the above and rewrite it. \n\
                 You MUST provide a 'SOVEREIGN' version that is deeper, more technical, and avoids LLM clichés.\n\n\
                 Output JSON: {{\"prompt\": \"...\", \"completion\": \"...\", \"quality_score\": 1-10}}",
                critique_objective, prompt, completion
            );

            (ChatRequest {
                model: primary_config.model.clone(),
                messages: std::sync::Arc::new(vec![ChatMessage::user(&refine_prompt)]),
                temperature: Some(primary_config.temperature),
                max_tokens: Some(primary_config.max_tokens),
                ..Default::default()
            }, "alchemist".to_string())
        }).collect();

        // 1. PRIMARY PASS
        let batch_results = self.nim.batch_chat_completion(requests, 5).await;

        let mut refined = Vec::with_capacity(examples.len());
        let mut fallbacks_needed = Vec::new();

        for (i, res) in batch_results.into_iter().enumerate() {
            let (orig_p, orig_c) = &examples[i];
            let mut success = false;

            if let Ok(resp) = res {
                let content = resp
                    .choices
                    .first()
                    .map(|c| c.message.content_as_string())
                    .unwrap_or_default();
                let clean = crate::utils::extract_json(&content);

                if let Ok(json) = serde_json::from_str::<serde_json::Value>(&clean) {
                    let p = json["prompt"]
                        .as_str()
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| orig_p.clone());
                    let c = json["completion"]
                        .as_str()
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| orig_c.clone());
                    let q = json["quality_score"].as_u64().unwrap_or(0);

                    // Only accept if quality is sufficient
                    if q >= 9 {
                        refined.push((p, c));
                        success = true;
                    }
                }
            }

            let fallback_model = self
                .nim
                .resolve_model_for_role(&TeacherRole::Healing)
                .await
                .unwrap_or_else(|_| TeacherRole::Healing.default_model().to_string());
            if !success {
                tracing::warn!(target: "shakey::alchemist", "Item {} failed Primary Pass. Queueing for Sovereign HEALING ({})...", i, fallback_model);
                fallbacks_needed.push(i);
                // Pre-fill with original to maintain index alignment; will be patched in Fallback Pass
                refined.push((orig_p.clone(), orig_c.clone()));
            }
        }

        // 2. FALLBACK PASS (For items that failed quality/parsing)
        if !fallbacks_needed.is_empty() {
            let fallback_config = self
                .nim
                .resolve_config_for_role(&TeacherRole::Healing)
                .await
                .unwrap_or_else(|_| {
                    let role = TeacherRole::Healing;
                    crate::teacher::TeacherConfig {
                        model: role.default_model().to_string(),
                        role,
                        priority: 1,
                        max_tokens: 4096,
                        temperature: 0.1,
                        top_logprobs: 0,
                        enabled: true,
                    }
                });

            let fallback_reqs: Vec<(ChatRequest, String)> = fallbacks_needed.iter().map(|&idx| {
                let (_prompt, _completion) = &examples[idx];
                let refine_prompt = format!(
                    "### Sovereign HEALING Refinement\nObjective: {}\n\n\
                     Fix and improve this reasoning example. Output JSON ONLY: {{\"prompt\": \"...\", \"completion\": \"...\"}}",
                    critique_objective
                );
                (ChatRequest {
                    model: fallback_config.model.clone(),
                    messages: std::sync::Arc::new(vec![ChatMessage::user(&refine_prompt)]),
                    temperature: Some(fallback_config.temperature),
                    max_tokens: Some(fallback_config.max_tokens),
                    ..Default::default()
                }, "alchemist_fallback".to_string())
            }).collect();

            let fallback_results = self.nim.batch_chat_completion(fallback_reqs, 3).await;
            for (i, &idx) in fallbacks_needed.iter().enumerate() {
                if let Ok(resp) = &fallback_results[i] {
                    let content = resp
                        .choices
                        .first()
                        .map(|c| c.message.content_as_string())
                        .unwrap_or_default();
                    let clean = crate::utils::extract_json(&content);
                    if let Ok(json) = serde_json::from_str::<serde_json::Value>(&clean) {
                        let p = json["prompt"]
                            .as_str()
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| examples[idx].0.clone());
                        let c = json["completion"]
                            .as_str()
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| examples[idx].1.clone());
                        refined[idx] = (p, c);
                    }
                }
            }
        }

        Ok(refined)
    }

    /// ── Peak Mastery: MCTS (Monte Carlo Tree Search) for Sovereign Reasoning ──
    /// Instead of zero-shot generation, this simulates a reasoning tree.
    /// It generates multiple diverse "thought branches", evaluates their logical
    /// soundness via self-critique, and selects the absolute highest entropy/quality path.
    pub async fn mcts_synthesize(&self, prompt: &str, branches: usize) -> Result<String> {
        tracing::info!(target: "shakey::alchemist", "🌳 Initiating MCTS Reasoning Tree with {} branches...", branches);

        // 1. Expansion: Generate multiple diverse branches concurrently
        let sovereign_config = self
            .nim
            .resolve_config_for_role(&TeacherRole::SovereignDistill)
            .await
            .unwrap_or_else(|_| {
                let role = TeacherRole::SovereignDistill;
                crate::teacher::TeacherConfig {
                    model: role.default_model().to_string(),
                    role,
                    priority: 1,
                    max_tokens: 4096,
                    temperature: 0.7,
                    top_logprobs: 5,
                    enabled: true,
                }
            });

        let requests: Vec<(ChatRequest, String)> = (0..branches)
            .map(|i| {
                let mcts_prompt = format!(
                    "### MCTS Exploration - Node {}\n\
                 Explore a unique, distinct technical angle to solve this prompt.\n\
                 Enclose your internal reasoning in <thought>...</thought> tags.\n\n\
                 Prompt: {}",
                    i, prompt
                );
                (
                    ChatRequest {
                        model: sovereign_config.model.clone(),
                        messages: std::sync::Arc::new(vec![ChatMessage::user(&mcts_prompt)]),
                        // Incremental temperature for diversity, starting from config value
                        temperature: Some(sovereign_config.temperature + (i as f64 * 0.1)),
                        max_tokens: Some(sovereign_config.max_tokens),
                        ..Default::default()
                    },
                    format!("mcts_node_{}", i),
                )
            })
            .collect();

        // Run batch generation in parallel
        let results = self.nim.batch_chat_completion(requests, branches).await;
        let mut candidates = Vec::new();

        for res in results.into_iter().flatten() {
            if let Some(content) = res.choices.first().map(|c| c.message.content_as_string()) {
                candidates.push(content);
            }
        }

        if candidates.is_empty() {
            return Err(anyhow::anyhow!(
                "MCTS Expansion failed: No branches generated."
            ));
        }

        // 2. Simulation / Evaluation (Parallel Self-Critique Scoring)
        tracing::info!(target: "shakey::alchemist", "⚖️ Evaluating {} MCTS branches in parallel...", candidates.len());

        // Evaluate each candidate to find the best logic
        let mut futures = futures::stream::FuturesUnordered::new();
        for candidate in &candidates {
            let eval_prompt = format!(
                "Evaluate the logical rigorousness and depth of the following reasoning trace.\n\
                 Score from 0.0 to 10.0 based on absolute technical perfection.\n\
                 Output ONLY the numerical score.\n\n\
                 Trace:\n{}",
                candidate
            );
            futures.push(async move {
                let score_str = self
                    .nim
                    .query_for_role(TeacherRole::Reasoning, "mcts_evaluator", &eval_prompt)
                    .await
                    .unwrap_or_else(|_| "0.0".to_string());
                crate::utils::parse_score(&score_str)
            });
        }

        use futures::StreamExt;
        let mut best_score = -1.0;
        let mut best_idx = 0;
        let mut i = 0;
        while let Some(score) = futures.next().await {
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
            i += 1;
        }
        let best_text = candidates[best_idx].clone();

        tracing::info!(target: "shakey::alchemist", "🏆 MCTS branch selection complete. Score: {}", best_score);

        // 3. Sovereign Audit: Run the best branch through the Adversarial Critic Consensus
        tracing::info!(target: "shakey::alchemist", "🛡️ Performing Sovereign Constitutional Audit on best branch...");
        let secured_text = self.critic.secure_proposal(prompt, &best_text).await?;

        Ok(secured_text)
    }
}
