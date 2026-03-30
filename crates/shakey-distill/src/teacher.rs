//! Teacher model abstraction and selection.
//!
//! Loads teacher configurations from `configs/teachers.yaml` and provides
//! a unified interface for querying the best teacher for each task role
//! (reasoning, code, math, reward, embedding).
//!
//! ## Multi-Teacher Strategy
//!
//! When multiple teachers are available for a role:
//! - **Priority**: Uses the highest-priority (lowest number) available teacher
//! - **Weighted Average**: Queries multiple teachers, averages their outputs
//! - **Best-of-N**: Queries N teachers, uses reward model to pick the best
//! - **Fallback**: If primary teacher fails, automatically tries the next one

use super::nim_client::{ChatMessage, ChatRequest, NimClient};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Role classification for teacher models.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TeacherRole {
    Reasoning,
    Code,
    Math,
    Reward,
    Embedding,
    VisionVideo,
    AudioTranscription,
    MultimodalGeneral,
    TranslationDialect,
    Safety,
    #[serde(alias = "3d_synthesis")]
    ThreeDSynthesis,
    Critique,
    Healing,
    Consensus,
    SovereignDistill,
    SecurityAudit,
    EmbeddingQuery,
    Summarizer,
}

impl std::fmt::Display for TeacherRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reasoning => write!(f, "reasoning"),
            Self::Code => write!(f, "code"),
            Self::Math => write!(f, "math"),
            Self::Reward => write!(f, "reward"),
            Self::Embedding => write!(f, "embedding"),
            Self::VisionVideo => write!(f, "vision_video"),
            Self::AudioTranscription => write!(f, "audio_transcription"),
            Self::MultimodalGeneral => write!(f, "multimodal_general"),
            Self::TranslationDialect => write!(f, "translation_dialect"),
            Self::Safety => write!(f, "safety"),
            Self::ThreeDSynthesis => write!(f, "3d_synthesis"),
            Self::Critique => write!(f, "critique"),
            Self::Healing => write!(f, "healing"),
            Self::Consensus => write!(f, "consensus"),
            Self::SovereignDistill => write!(f, "sovereign_distill"),
            TeacherRole::SecurityAudit => write!(f, "security_audit"),
            TeacherRole::EmbeddingQuery => write!(f, "embedding_query"),
            TeacherRole::Summarizer => write!(f, "summarizer"),
        }
    }
}

impl TeacherRole {
    /// Returns the "Perfect" default model ID for a given role.
    /// This ensures zero hardcoded strings in business logic.
    pub fn default_model(&self) -> &'static str {
        match self {
            TeacherRole::Reasoning => "mistralai/mistral-large-3-675b-instruct-2512",
            TeacherRole::Code => "qwen/qwen3-coder-480b-a35b-instruct",
            TeacherRole::Math => "qwen/qwen3-next-80b-a3b-thinking",
            TeacherRole::Reward => "nvidia/nemotron-4-340b-reward",
            TeacherRole::Embedding => "nvidia/nv-embed-v1",
            TeacherRole::VisionVideo => "meta/llama-3.2-90b-vision-instruct",
            TeacherRole::AudioTranscription => "nvidia/cosmos-reason2-8b",
            TeacherRole::MultimodalGeneral => "meta/llama-3.2-90b-vision-instruct",
            TeacherRole::TranslationDialect => "nvidia/riva-translate-4b-instruct-v1.1",
            TeacherRole::Safety => "nvidia/nemotron-content-safety-reasoning-4b",
            TeacherRole::ThreeDSynthesis => "nvidia/streampetr",
            TeacherRole::Critique => "meta/llama-3.3-70b-instruct",
            TeacherRole::Healing => "meta/llama-3.1-405b-instruct",
            TeacherRole::Consensus => "nvidia/nemotron-4-340b-instruct",
            TeacherRole::SovereignDistill => "deepseek-ai/deepseek-v3.1-terminus",
            TeacherRole::SecurityAudit => "meta/llama-3.1-405b-instruct",
            TeacherRole::EmbeddingQuery => "nvidia/llama-nemotron-embed-1b-v2",
            TeacherRole::Summarizer => "meta/llama-3.3-70b-instruct",
        }
    }
}

/// Configuration for a single teacher model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeacherConfig {
    pub model: String,
    pub role: TeacherRole,
    pub priority: u32,
    pub max_tokens: u32,
    pub temperature: f64,
    #[serde(default = "default_top_logprobs")]
    pub top_logprobs: u32,
    #[serde(default = "default_true")]
    pub enabled: bool,
}

fn default_top_logprobs() -> u32 {
    5
}

fn default_true() -> bool {
    true
}

/// File structure for teachers.yaml
#[derive(Debug, Deserialize)]
pub struct TeachersFile {
    pub rate_budget: Option<std::collections::HashMap<String, u32>>,
    pub teachers: Vec<TeacherConfig>,
}

/// Teacher manager — selects and queries the best teacher for each task.
#[derive(Clone)]
pub struct TeacherManager {
    teachers: Vec<TeacherConfig>,
    client: NimClient,
}

impl TeacherManager {
    /// Load teacher configurations from YAML file.
    pub async fn from_config(config_path: &str, client: NimClient) -> Result<Self> {
        let content = std::fs::read_to_string(config_path)
            .with_context(|| format!("Failed to read teachers config: {config_path}"))?;
        let file: TeachersFile = serde_yaml::from_str(&content)?;

        // Inject rate budget into client if present
        if let Some(budget) = file.rate_budget {
            client.set_rate_budget(40, budget).await;
        }

        let enabled_teachers: Vec<TeacherConfig> =
            file.teachers.into_iter().filter(|t| t.enabled).collect();

        tracing::info!("Loaded {} enabled teacher models", enabled_teachers.len());
        for t in &enabled_teachers {
            tracing::debug!("  [{:>10}] {} (priority={})", t.role, t.model, t.priority);
        }

        Ok(Self {
            teachers: enabled_teachers,
            client,
        })
    }

    /// Create with explicit teacher list (for testing).
    pub fn new(teachers: Vec<TeacherConfig>, client: NimClient) -> Self {
        Self { teachers, client }
    }

    /// Get the best (highest priority) teacher for a given role.
    pub fn best_teacher(&self, role: &TeacherRole) -> Option<&TeacherConfig> {
        self.teachers
            .iter()
            .filter(|t| &t.role == role && t.enabled)
            .min_by_key(|t| t.priority)
    }

    /// Get all teachers for a given role, sorted by priority.
    pub fn teachers_for_role(&self, role: &TeacherRole) -> Vec<&TeacherConfig> {
        let mut teachers: Vec<&TeacherConfig> = self
            .teachers
            .iter()
            .filter(|t| &t.role == role && t.enabled)
            .collect();
        teachers.sort_by_key(|t| t.priority);
        teachers
    }

    /// Query the best teacher for a role with automatic fallback.
    ///
    /// Tries the highest-priority teacher first. If it fails, falls back
    /// to the next one in priority order.
    pub async fn query_with_fallback(
        &self,
        role: &TeacherRole,
        messages: std::sync::Arc<Vec<ChatMessage>>,
        max_tokens: Option<u32>,
    ) -> Result<TeacherResponse> {
        let teachers = self.teachers_for_role(role);
        if teachers.is_empty() {
            return Err(anyhow::anyhow!("No enabled teachers for role: {role}"));
        }

        let mut last_error = None;

        for teacher in &teachers {
            let tokens = max_tokens.or(Some(teacher.max_tokens));

            tracing::info!(target: "shakey::teacher", "🎓 Asking Teacher ({}): {}", role, teacher.model);

            let request = ChatRequest {
                model: teacher.model.clone(),
                messages: messages.clone(),
                max_tokens: tokens,
                temperature: Some(teacher.temperature),
                ..Default::default()
            };

            match self
                .client
                .chat_completion(&request, &role.to_string())
                .await
            {
                Ok(resp) => {
                    let choice = resp.choices.first().ok_or_else(|| {
                        anyhow::anyhow!("Empty response from teacher {}", teacher.model)
                    })?;

                    let msg = &choice.message;

                    tracing::debug!("Teacher {} responded", teacher.model);

                    return Ok(TeacherResponse {
                        model: teacher.model.clone(),
                        role: role.clone(),
                        content: msg.content_as_string(),
                        reasoning_content: msg.reasoning_content.clone(),
                        tool_calls: msg.tool_calls.clone(),
                        logprobs: None,
                    });
                }
                Err(e) => {
                    tracing::warn!(
                        target: "shakey::teacher",
                        "⚠️ Teacher {} failed. Fallback triggered: {}",
                        teacher.model,
                        e
                    );
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("All teachers failed for role: {role}")))
    }

    pub async fn query_for_distillation(
        &self,
        role: &TeacherRole,
        messages: std::sync::Arc<Vec<ChatMessage>>,
        top_logprobs: u32,
    ) -> Result<TeacherResponse> {
        let teacher = self
            .best_teacher(role)
            .ok_or_else(|| anyhow::anyhow!("No teacher for role: {role}"))?
            .clone();

        let req = crate::nim_client::ChatRequest {
            model: teacher.model.clone(),
            messages,
            max_tokens: Some(teacher.max_tokens),
            temperature: Some(teacher.temperature),
            logprobs: Some(true),
            top_logprobs: Some(top_logprobs),
            ..Default::default()
        };

        let response = self.client.chat_completion(&req, "distillation").await?;
        let choice = response
            .choices
            .first()
            .ok_or_else(|| anyhow::anyhow!("Teacher returned no choices"))?;
        let msg = &choice.message;

        Ok(TeacherResponse {
            model: teacher.model,
            role: role.clone(),
            content: msg.content_as_string(),
            reasoning_content: msg.reasoning_content.clone(),
            tool_calls: msg.tool_calls.clone(),
            logprobs: choice.logprobs.as_ref().and_then(|lp| lp.content.clone()),
        })
    }

    /// Get the underlying NIM client (for direct API access).
    pub fn client(&self) -> &NimClient {
        &self.client
    }

    /// List all available teacher models.
    pub fn list_teachers(&self) -> &[TeacherConfig] {
        &self.teachers
    }
}

/// Response from a teacher model.
#[derive(Debug)]
pub struct TeacherResponse {
    pub model: String,
    pub role: TeacherRole,
    pub content: String,
    pub reasoning_content: Option<String>,
    pub tool_calls: Option<Vec<super::nim_client::ToolCall>>,
    pub logprobs: Option<Vec<super::nim_client::TokenLogprob>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_teacher_role_display() {
        assert_eq!(TeacherRole::Reasoning.to_string(), "reasoning");
        assert_eq!(TeacherRole::Code.to_string(), "code");
    }

    #[test]
    fn test_teacher_config_deserialize() {
        let yaml = r#"
            model: "test/sovereign-model"
            role: reasoning
            priority: 1
            max_tokens: 4096
            temperature: 0.7
            top_logprobs: 5
            enabled: true
        "#;
        let config: TeacherConfig = serde_yaml::from_str(yaml).unwrap();
        assert_eq!(config.model, "test/sovereign-model");
        assert_eq!(config.role, TeacherRole::Reasoning);
        assert_eq!(config.priority, 1);
        assert_eq!(config.top_logprobs, 5);
    }

    #[test]
    fn test_all_defaults_exist_in_yaml() {
        let config_path = format!("{}/../../configs/teachers.yaml", env!("CARGO_MANIFEST_DIR"));
        let content =
            std::fs::read_to_string(&config_path).expect("Failed to read teachers.yaml from tests");
        let file: TeachersFile =
            serde_yaml::from_str(&content).expect("Failed to parse teachers.yaml");

        let all_roles = vec![
            TeacherRole::Reasoning,
            TeacherRole::Code,
            TeacherRole::Math,
            TeacherRole::Reward,
            TeacherRole::Embedding,
            TeacherRole::VisionVideo,
            TeacherRole::AudioTranscription,
            TeacherRole::MultimodalGeneral,
            TeacherRole::TranslationDialect,
            TeacherRole::Safety,
            TeacherRole::ThreeDSynthesis,
            TeacherRole::Critique,
            TeacherRole::Healing,
            TeacherRole::Consensus,
            TeacherRole::SovereignDistill,
            TeacherRole::SecurityAudit,
            TeacherRole::EmbeddingQuery,
            TeacherRole::Summarizer,
        ];

        for role in all_roles {
            let default_id = role.default_model();
            let found = file
                .teachers
                .iter()
                .find(|t| t.role == role && t.model == default_id);
            assert!(
                found.is_some(),
                "The default model '{}' for role '{}' is NOT serialized in teachers.yaml! Update teachers.yaml to keep it as the Single Source of Truth.",
                default_id, role
            );
        }
    }
}
