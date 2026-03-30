use crate::nim_client::NimClient;
use anyhow::Result;
use async_trait::async_trait;
use shakey_core::training::SovereignCritic;

/// Adversarial Critic: The "Zenith" of autonomous reasoners.
/// It uses a separate teacher model to aggressively critique its own proposals.
pub struct AdversarialCritic {
    nim: NimClient,
}

impl AdversarialCritic {
    pub fn new(nim: NimClient) -> Self {
        Self { nim }
    }

    /// Critique a proposed solution (Code or Shell Command) for security and logic flaws.
    pub async fn critique(&self, context: &str, proposal: &str, model: &str) -> Result<String> {
        tracing::info!(target: "shakey::zenith", "⚔️ Adversarial Pass ({}): Critiquing proposed solution...", model);

        let prompt = format!(
            "System: You are the Sovereign Adversary of Project Shakey.\n\
             Context: {}\n\
             Proposal: {}\n\n\
             Task: Find every flaw, security vulnerability, or unconstrained logic in the proposal above. \n\
             If it is perfect, say 'PERFECT'. Otherwise, provide a 'fixed_v'.\n\n\
             Output JSON ONLY: {{\"status\": \"FLAWED/PERFECT\", \"reason\": \"...\", \"fixed_v\": \"...\"}}",
            context, proposal
        );

        self.nim.query(model, "adversary", &prompt, 8192, 0.1).await
    }

    /// Systematically fix a proposal until the CONSENSUS of adversaries is satisfied.
    pub async fn secure_proposal(&self, context: &str, initial_proposal: &str) -> Result<String> {
        let mut current_v = initial_proposal.to_string();
        let mut attempts = 0;

        let models = self
            .nim
            .models_for_role(&crate::teacher::TeacherRole::Consensus)
            .await?;
        if models.is_empty() {
            return Err(anyhow::anyhow!(
                "AdversarialCritic: No Consensus models found in configuration."
            ));
        }

        while attempts < 2 {
            let mut all_perfect = true;
            let mut latest_fix = None;
            let mut futures = futures::stream::FuturesUnordered::new();
            for model in &models {
                let context_ref = context;
                let current_v_ref = current_v.clone(); // Pass owned snapshot to parallel critique
                let model_name = model.to_string();

                futures.push(async move {
                    (
                        model_name.clone(),
                        self.critique(context_ref, &current_v_ref, &model_name)
                            .await,
                    )
                });
            }

            use futures::StreamExt;
            while let Some((model_name, res)) = futures.next().await {
                let res = res?;
                let clean = crate::utils::extract_json(&res);

                let audit: serde_json::Value = serde_json::from_str(&clean)?;
                if audit["status"].as_str() != Some("PERFECT") {
                    all_perfect = false;
                    latest_fix = audit["fixed_v"].as_str().map(|s| s.to_string());
                    tracing::warn!(target: "shakey::zenith", "⚔️ Adversary ({}) spotted a flaw! Fix required.", model_name);
                    break;
                }
            }

            if all_perfect {
                tracing::info!(target: "shakey::zenith", "🛡️ Sovereign Consensus ACHIEVED. Proposal is secure.");
                return Ok(current_v);
            }

            if let Some(fixed) = latest_fix {
                current_v = fixed;
            } else {
                break;
            }
            attempts += 1;
        }

        Ok(current_v)
    }
}

#[async_trait]
impl SovereignCritic for AdversarialCritic {
    async fn evaluate_score(&self, prompt: &str, completion: &str) -> Result<f32> {
        // We use the most powerful model for semantic evaluation
        let model = self
            .nim
            .resolve_model_for_role(&crate::teacher::TeacherRole::SecurityAudit)
            .await
            .unwrap_or_else(|_| {
                crate::teacher::TeacherRole::SecurityAudit
                    .default_model()
                    .to_string()
            });
        let res = self.critique(prompt, completion, &model).await?;

        let clean = crate::utils::extract_json(&res);

        let audit: serde_json::Value = serde_json::from_str(&clean)?;
        if audit["status"].as_str() == Some("PERFECT") {
            Ok(1.0)
        } else {
            // If flawed, we return a lower score to increase the loss penalty.
            // In a future update, this could be more granular based on the "reason" field.
            Ok(0.5)
        }
    }
}
