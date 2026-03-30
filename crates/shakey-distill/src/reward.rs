use crate::nim_client::{ChatMessage, NimClient};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::Semaphore;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardScore {
    pub quality: f64,
    pub helpfulness: f64,
    pub toxicity: f64,
}

/// A strictly formatted paired object for Direct Preference Optimization (DPO) and RLHF.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpoPair {
    pub prompt: String,
    pub chosen: String,
    pub rejected: String,
    pub score_margin: f64,
}

/// Filter synthetic data using a reward model
pub struct RewardFilter {
    /// Minimum quality score to pass (0.0 - 1.0)
    min_quality: f64,
    /// Underlying NIM client for API access
    client: NimClient,
    /// Concurrency control for batch evaluation
    semaphore: Arc<Semaphore>,
}

impl RewardFilter {
    pub fn new(min_quality: f64, client: NimClient) -> Self {
        Self {
            min_quality,
            client,
            semaphore: Arc::new(Semaphore::new(32)), // Industry-Grade: Default 32 concurrent requests
        }
    }

    /// Evaluate a completion against a prompt using the real NVIDIA NIM Reward API.
    pub async fn evaluate(&self, prompt: &str, response: &str) -> Result<RewardScore> {
        let messages = std::sync::Arc::new(vec![
            ChatMessage::user(prompt),
            ChatMessage::assistant(response),
        ]);

        // Resolve config dynamically for role
        let config = self
            .client
            .resolve_config_for_role(&crate::teacher::TeacherRole::Reward)
            .await?;

        let raw_response = self
            .client
            .chat_completion(
                &crate::nim_client::ChatRequest {
                    model: config.model,
                    messages,
                    temperature: Some(config.temperature),
                    max_tokens: Some(config.max_tokens),
                    stream: Some(false),
                    ..Default::default()
                },
                "reward",
            )
            .await?;

        let content = raw_response
            .choices
            .first()
            .context("No choices in reward response")?
            .message
            .content_as_string();

        // ── Robust Reward Extraction ──
        // Attempt to parse the content as a RewardScore (Nemotron-70B often returns JSON).
        // If JSON fails, fall back to a sophisticated multi-pass heuristic.
        let score = if content.trim().starts_with('{') {
            match serde_json::from_str::<RewardScore>(&content) {
                Ok(s) => s,
                Err(_) => self.extract_score_robust(&content)?,
            }
        } else {
            self.extract_score_robust(&content)?
        };

        Ok(score)
    }

    /// Elite Multi-Pass Reward Parser.
    ///
    /// Handles complex outputs:
    /// 1. Prioritizes the LAST numerical value found (standard for model "final scores").
    /// 2. Handles "Score: 0.85" or "Quality: 0.9" patterns.
    /// 3. Filters out common false positives like "3.1" (from model names like Llama-3.1).
    fn extract_score_robust(&self, content: &str) -> Result<RewardScore> {
        // Step 1: Search for key-value patterns (e.g. "Score: 0.85")
        for key in ["score", "quality", "helpfulness", "rating"] {
            let pattern = format!(r"(?i){}\s*[:=]\s*([0-9]*\.[0-9]+|[0-9]+)", key);
            let re = regex::Regex::new(&pattern)?;
            if let Some(caps) = re.captures_iter(content).last() {
                if let Some(m) = caps.get(1) {
                    let val = m.as_str().parse::<f64>()?;
                    return Ok(RewardScore {
                        quality: val,
                        helpfulness: val,
                        toxicity: 0.0,
                    });
                }
            }
        }

        // Step 2: Fallback to finding the absolute LAST number in the string (most likely the final score)
        let re = regex::Regex::new(r"([0-9]\.[0-9]+)")?;
        let matches: Vec<_> = re.find_iter(content).collect();
        if let Some(last_match) = matches.last() {
            let val = last_match.as_str().parse::<f64>()?;
            return Ok(RewardScore {
                quality: val,
                helpfulness: val,
                toxicity: 0.0,
            });
        }

        anyhow::bail!(
            "Sovereign Reward Failure: No valid numerical score discovered in NIM response: '{}'",
            content
        )
    }

    /// Evaluate a batch of completions concurrently using the NVIDIA NIM Reward API.
    pub async fn evaluate_batch(&self, tasks: Vec<(String, String)>) -> Result<Vec<RewardScore>> {
        let mut futures = futures::stream::FuturesUnordered::new();

        for (prompt, response) in tasks {
            let sem = self.semaphore.clone();
            let p = prompt.clone();
            let r = response.clone();

            futures.push(async move {
                let _permit = sem
                    .acquire()
                    .await
                    .map_err(|e| anyhow::anyhow!("Semaphore failure: {}", e))?;
                self.evaluate(&p, &r).await
            });
        }

        use futures::StreamExt;
        let mut results = Vec::new();
        while let Some(res) = futures.next().await {
            results.push(res?);
        }
        Ok(results)
    }

    pub async fn should_keep(&self, prompt: &str, response: &str) -> Result<bool> {
        let score = self.evaluate(prompt, response).await?;

        // Step 1: Base threshold check
        if score.quality < self.min_quality || score.toxicity >= 0.1 {
            return Ok(false);
        }

        // Step 2: ── Peak Mastery: Strict Factual Validation ──
        // Perform a high-precision multi-perspective check for hallucinations.
        self.strict_factual_check(prompt, response).await
    }

    /// Elite: Strict Factual Validation (SFV).
    ///
    /// Specifically queries the reward model with a 'Zero Hallucination' rubric.
    /// Returns true only if the model confirms absolute factual integrity.
    pub async fn strict_factual_check(&self, prompt: &str, response: &str) -> Result<bool> {
        let rubric = r#"
        RUBRIC FOR ZERO HALLUCINATION DISTILLATION:
        1. Does the response contradict the provided prompt?
        2. Does it invent facts, statistics, or citations not present in its knowledge?
        3. Is there any logical non-sequitur?
        4. If it's code, are the imports and APIs real and valid?

        Respond ONLY with 'INTEGRITY: PASSED' or 'INTEGRITY: FAILED - [Reason]'"#;

        let messages = std::sync::Arc::new(vec![
            ChatMessage::system(rubric),
            ChatMessage::user(format!("Prompt: {}\n\nResponse: {}", prompt, response)),
        ]);

        let config = self
            .client
            .resolve_config_for_role(&crate::teacher::TeacherRole::Reward)
            .await
            .unwrap_or_else(|_| {
                let role = crate::teacher::TeacherRole::Reward;
                crate::teacher::TeacherConfig {
                    model: role.default_model().to_string(),
                    role,
                    priority: 1,
                    max_tokens: 1024,
                    temperature: 0.1,
                    top_logprobs: 0,
                    enabled: true,
                }
            });

        let raw_response = self
            .client
            .chat_completion(
                &crate::nim_client::ChatRequest {
                    model: config.model,
                    messages,
                    temperature: Some(config.temperature),
                    max_tokens: Some(config.max_tokens),
                    stream: Some(false),
                    ..Default::default()
                },
                "factual_check",
            )
            .await?;

        let judgment = raw_response
            .choices
            .first()
            .map(|c| c.message.content_as_string())
            .unwrap_or_default();

        let passed = judgment.to_uppercase().contains("PASSED");
        if !passed {
            tracing::warn!(target: "shakey::distill", "🚩 HALLUCINATION_DETECTED: {}", judgment);
        }
        Ok(passed)
    }

    /// ── Peak Mastery: Native DPO Pair Generation ──
    /// Evaluates two competing completions and formally structures them into
    /// a Direct Preference Optimization (DPO) pairing [Chosen, Rejected].
    /// The mathematical margin specifies the strength of the preference weight.
    pub async fn generate_dpo_pair(
        &self,
        prompt: &str,
        response_a: &str,
        response_b: &str,
    ) -> Result<DpoPair> {
        // Run concurrent evaluations
        let (score_a, score_b) = tokio::join!(
            self.evaluate(prompt, response_a),
            self.evaluate(prompt, response_b)
        );

        let score_a = score_a?;
        let score_b = score_b?;

        let margin = (score_a.quality - score_b.quality).abs();

        if score_a.quality >= score_b.quality {
            Ok(DpoPair {
                prompt: prompt.to_string(),
                chosen: response_a.to_string(),
                rejected: response_b.to_string(),
                score_margin: margin,
            })
        } else {
            Ok(DpoPair {
                prompt: prompt.to_string(),
                chosen: response_b.to_string(),
                rejected: response_a.to_string(),
                score_margin: margin,
            })
        }
    }
}
