//! High-level batch distillation orchestrator.
//!
//! Manages large-scale synthetic data generation by partitioning
//! a target token budget into parallel NIM requests.

use crate::nim_client::{ChatMessage, ChatRequest, ChatResponse, NimClient};
use anyhow::Result;

/// Strategy for partitioning a large distillation task.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub model: String,
    pub role: String,
    pub total_target_tokens: u64,
    pub tokens_per_request: u32,
    pub concurrency: usize,
    pub temperature: f64,
    pub top_logprobs: u32,
}

pub struct BatchManager {
    client: NimClient,
}

impl BatchManager {
    pub fn new(client: NimClient) -> Self {
        Self { client }
    }

    /// Run a massive distillation batch using parallel NIM workers.
    pub async fn run_distillation_batch(
        &self,
        config: BatchConfig,
        prompts: Vec<String>,
    ) -> Result<Vec<ChatResponse>> {
        let mut requests = Vec::with_capacity(prompts.len());

        for prompt in prompts {
            let req = ChatRequest {
                model: config.model.clone(),
                messages: std::sync::Arc::new(vec![ChatMessage::user(prompt)]),
                temperature: Some(config.temperature),
                max_tokens: Some(config.tokens_per_request),
                logprobs: Some(true),
                top_logprobs: Some(config.top_logprobs),
                stream: Some(false),
                ..Default::default()
            };
            requests.push((req, config.role.clone()));
        }

        tracing::info!(
            "Starting parallel distillation batch: {} requests, concurrency={}",
            requests.len(),
            config.concurrency
        );

        let results = self
            .client
            .batch_chat_completion(requests, config.concurrency)
            .await;

        let mut successful = Vec::new();
        let mut failures = 0;

        for res in results {
            match res {
                Ok(resp) => successful.push(resp),
                Err(e) => {
                    tracing::error!("Batch request failed: {}", e);
                    failures += 1;
                }
            }
        }

        if failures > 0 {
            tracing::warn!("Distillation batch completed with {} failures.", failures);
        }

        Ok(successful)
    }
}
