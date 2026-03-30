//! NVIDIA NIM API client with rate limiting.
//!
//! Handles all communication with NVIDIA's inference microservices.
//! Uses the OpenAI-compatible chat completions API.
//!
//! ## Rate Limiting
//!
//! Free tier: 40 requests per minute.
//! Implemented as a sliding window rate limiter:
//! - Tracks timestamps of last N requests
//! - Before each request, checks if we'd exceed the limit
//! - If so, sleeps until a slot opens up
//! - Exponential backoff on 429/5xx errors
//!
//! ## Dynamic Modeling
//!
//! Through the intergrated `TeacherManager`, the client can resolve
//! high-level roles (e.g., `Reasoning`, `Code`, `Critique`) to the best
//! available model ID as configured in `teachers.yaml`.

use anyhow::{Context, Result};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use shakey_core::metrics::SovereignMetrics;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicI64, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tracing;

// ─────────────────────────────────────────────────────────────
//  Rate Limiter
// ─────────────────────────────────────────────────────────────

/// Sliding window rate limiter for the NVIDIA NIM API.
///
/// Tracks the timestamps of recent requests and ensures we don't
/// exceed the configured requests-per-minute limit. Thread-safe
/// via Arc<Mutex<_>>.
#[derive(Debug)]
pub struct RateLimiter {
    /// Maximum requests per minute
    max_rpm: u32,
    /// Timestamps of recent requests
    timestamps: VecDeque<Instant>,
    /// Window duration (1 minute)
    window: Duration,
    /// ── Zenith Apex II: Synchronized Concurrency ──
    active_requests: u32,
    max_concurrency: u32,
}

impl RateLimiter {
    pub fn new(max_rpm: u32) -> Self {
        Self {
            max_rpm,
            timestamps: VecDeque::with_capacity(max_rpm as usize + 1),
            window: Duration::from_secs(60),
            active_requests: 0,
            // Industry standard concurrency (proportional to RPM)
            max_concurrency: (max_rpm / 4).clamp(2, 32),
        }
    }

    pub fn peek_wait(&mut self) -> Option<Duration> {
        let now = Instant::now();

        // Remove timestamps older than the window
        while let Some(&oldest) = self.timestamps.front() {
            if now.duration_since(oldest) > self.window {
                self.timestamps.pop_front();
            } else {
                break;
            }
        }

        // Check if we have capacity
        if (self.timestamps.len() as u32) < self.max_rpm {
            return None;
        }

        // Wait until the oldest request falls outside the window
        if let Some(&oldest) = self.timestamps.front() {
            let wait_time = self
                .window
                .checked_sub(now.duration_since(oldest))
                .unwrap_or(Duration::from_millis(10));

            // ── Adaptive Jitter Logic (Apex 5.5) ──
            // Random jitter between 5ms and 25ms to avoid thundering herd without adding excessive lag.
            let jitter_ms = (rand::random::<u64>() % 20) + 5;
            return Some(wait_time + Duration::from_millis(jitter_ms));
        }
        None
    }

    pub fn record(&mut self) {
        self.timestamps.push_back(Instant::now());
    }

    /// Zenith Sovereign: Dynamically adjust the rate limit for autopoietic rebalancing.
    pub fn set_max_rpm(&mut self, max_rpm: u32) {
        self.max_rpm = max_rpm;
        // Rescale concurrency proportionally
        self.max_concurrency = (max_rpm / 4).clamp(2, 32);
    }

    /// Try to acquire a concurrency slot (Internal).
    pub fn try_occupy_slot(&mut self) -> bool {
        if self.active_requests < self.max_concurrency {
            self.active_requests += 1;
            true
        } else {
            false
        }
    }

    /// Release a concurrency slot (Internal).
    pub fn release_slot(&mut self) {
        self.active_requests = self.active_requests.saturating_sub(1);
    }

    /// Get the current request count in the window.
    pub fn current_count(&self) -> usize {
        self.timestamps.len()
    }
}

/// Multi-bucket rate limiter that manages independent budgets for different roles.
#[derive(Debug)]
pub struct MultiRateLimiter {
    /// Map of role names to their individual rate limiters
    limiters: std::collections::HashMap<String, RateLimiter>,
    /// Global fallback limit (e.g., 40 RPM for total API)
    global_limit: RateLimiter,
    /// ── Zenith Apex II: Stability Hysteresis ──
    last_rebalance: Option<Instant>,
    last_target: Option<String>,
}

impl MultiRateLimiter {
    pub fn new(global_max_rpm: u32, role_budgets: std::collections::HashMap<String, u32>) -> Self {
        let mut limiters = std::collections::HashMap::new();
        for (role, rpm) in role_budgets {
            limiters.insert(role, RateLimiter::new(rpm));
        }

        Self {
            limiters,
            global_limit: RateLimiter::new(global_max_rpm),
            last_rebalance: None,
            last_target: None,
        }
    }

    /// Try to acquire a slot. Returns `Some(Duration)` if waiting is required.
    pub fn try_acquire(&mut self, role: &str) -> Option<Duration> {
        let mut max_wait = None;

        if let Some(wait) = self.global_limit.peek_wait() {
            max_wait = Some(wait.max(max_wait.unwrap_or_default()));
        }

        if let Some(limiter) = self.limiters.get_mut(role) {
            if let Some(wait) = limiter.peek_wait() {
                max_wait = Some(wait.max(max_wait.unwrap_or_default()));
            }
        }

        if let Some(wait) = max_wait {
            return Some(wait);
        }

        // If no wait needed, record the request atomically
        self.global_limit.record();
        if let Some(limiter) = self.limiters.get_mut(role) {
            limiter.record();
        }

        None
    }

    /// Zenith Apex II: Synchronized Slot Release
    pub fn release_slot(&mut self, role: &str) {
        if let Some(limiter) = self.limiters.get_mut(role) {
            limiter.release_slot();
        }
    }

    /// Zenith Apex II: Try to occupy a concurrency slot for the specific role.
    pub fn try_occupy_slot(&mut self, role: &str) -> bool {
        if let Some(limiter) = self.limiters.get_mut(role) {
            limiter.try_occupy_slot()
        } else {
            true // If no role found, global limiter is the only guard
        }
    }

    /// ── Zenith Sovereign: Autopoietic Rebalance ──
    /// Dynamically shifts weights toward the target role while ensuring global bounds.
    pub fn rebalance(&mut self, target_role: &str, global_max: u32, is_urgent: bool) {
        let now = Instant::now();

        // --- 🤖 ZENITH APEX II: HYSTERESIS SMOOTHING ---
        // Avoid frequent budget oscillations if OODA goals switch too quickly.
        if !is_urgent {
            if let Some(last) = self.last_rebalance {
                if now.duration_since(last) < Duration::from_secs(15) {
                    if self.last_target.as_deref() == Some(target_role) {
                        return; // Goal hasn't changed, skip rebalance
                    }
                    // Even if goal changed, only allow rebalance every 15s to keep training stable.
                    tracing::debug!(target: "shakey::api", "Autopoietic: Rebalance suppressed by 15s hysteresis window.");
                    return;
                }
            }
        }

        if !self.limiters.contains_key(target_role) {
            return;
        }

        self.last_rebalance = Some(now);
        self.last_target = Some(target_role.to_string());

        // Allocate to the target role (80% if urgent, 50% for standard Industry-grade focus)
        let modifier = if is_urgent { 0.8 } else { 0.5 };
        let prioritized_rpm = ((global_max as f64) * modifier) as u32;
        let prioritized_rpm = prioritized_rpm.max(1);
        let remaining_rpm = global_max.saturating_sub(prioritized_rpm);

        // Distribute remainder to others
        let other_count = self.limiters.len().saturating_sub(1);
        let per_other_rpm = if other_count > 0 {
            (remaining_rpm / other_count as u32).max(1)
        } else {
            remaining_rpm
        };

        for (role, limiter) in self.limiters.iter_mut() {
            if role == target_role {
                limiter.set_max_rpm(prioritized_rpm);
                tracing::debug!(target: "shakey::api", "Autopoietic: Prioritizing {} at {} RPM", role, prioritized_rpm);
            } else {
                limiter.set_max_rpm(per_other_rpm);
            }
        }
    }

    pub fn role_count(&self, role: &str) -> usize {
        self.limiters
            .get(role)
            .map(|l| l.current_count())
            .unwrap_or(0)
    }

    pub fn global_count(&self) -> usize {
        self.global_limit.current_count()
    }
}

/// ── Zenith Apex II: RAII Concurrency Guard ──
/// Ensures that a role-specific concurrency slot is released when the request finishes.
pub struct RolePermit {
    role: String,
    limiter: Arc<std::sync::Mutex<MultiRateLimiter>>,
}

impl Drop for RolePermit {
    fn drop(&mut self) {
        if let Ok(mut lock) = self.limiter.lock() {
            lock.release_slot(&self.role);
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  Circuit Breaker
// ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

/// Thread-safe circuit breaker to protect against service outages.
pub struct CircuitBreaker {
    failure_count: AtomicU32,
    state: Mutex<CircuitState>,
    last_failure_time: AtomicI64, // Unix timestamp
    failure_threshold: u32,
    base_timeout_secs: i64,
    max_timeout_secs: i64,
}

impl CircuitBreaker {
    pub fn new(failure_threshold: u32, base_timeout_secs: i64) -> Self {
        Self {
            failure_count: AtomicU32::new(0),
            state: Mutex::new(CircuitState::Closed),
            last_failure_time: AtomicI64::new(0),
            failure_threshold,
            base_timeout_secs,
            max_timeout_secs: 300, // Hard limit of 5 minutes
        }
    }

    pub async fn check(&self) -> Result<(), anyhow::Error> {
        let mut state_guard = self.state.lock().await;

        if *state_guard == CircuitState::Open {
            let last_ts = self.last_failure_time.load(Ordering::SeqCst);
            let now_ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs() as i64;

            let count = self.failure_count.load(Ordering::SeqCst);
            // Adaptive Backoff: base * 2^(failures - threshold) + jitter
            let exponent = count.saturating_sub(self.failure_threshold);
            let backoff = (self.base_timeout_secs as f64 * 2.0f64.powi(exponent as i32))
                .min(self.max_timeout_secs as f64);
            let jitter = (rand::random::<f64>() * 5.0).floor() as i64; // Sub-5s jitter
            let wait_time = (backoff as i64) + jitter;

            if now_ts - last_ts > wait_time {
                *state_guard = CircuitState::HalfOpen;
                tracing::info!(
                    "Circuit Breaker: Entering Half-Open state. Probing service (Backoff: {}s)...",
                    wait_time
                );
            } else {
                return Err(anyhow::anyhow!(
                    "NIM API Circuit is OPEN. Cooling down for {} more seconds (Backoff: {}s).",
                    wait_time - (now_ts - last_ts),
                    wait_time
                ));
            }
        }
        Ok(())
    }

    pub async fn record_success(&self) {
        let mut state_guard = self.state.lock().await;
        self.failure_count.store(0, Ordering::SeqCst);
        if *state_guard != CircuitState::Closed {
            tracing::info!("Circuit Breaker: Success detected. Closing circuit.");
            *state_guard = CircuitState::Closed;
        }
    }

    pub async fn record_failure(&self) {
        let count = self.failure_count.fetch_add(1, Ordering::SeqCst) + 1;
        if count >= self.failure_threshold {
            let mut state_guard = self.state.lock().await;
            if *state_guard != CircuitState::Open {
                tracing::warn!("Circuit Breaker: Threshold reached. Opening circuit.");
                *state_guard = CircuitState::Open;
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as i64;
                self.last_failure_time.store(now, Ordering::SeqCst);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────
//  API Types (Sovereign-Zenith Hardened)
// ─────────────────────────────────────────────────────────────

#[derive(Debug, thiserror::Error)]
pub enum NimError {
    #[error("Rate Limit Exceeded: Wait {0:?}")]
    RateLimited(Duration),
    #[error("Invalid Request: {0}")]
    InvalidRequest(String),
    #[error("Server Error ({0}): {1}")]
    ServerError(u16, String),
    #[error("NIM Circuit is OPEN: {0}")]
    CircuitOpen(String),
    #[error("Network Error: {0}")]
    Network(String),
    #[error("Serialization Error: {0}")]
    Serialization(String),
    #[error("Fatal: {0}")]
    Fatal(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl Default for MessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl MessageContent {
    pub fn as_string(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlData },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrlData {
    pub url: String,
}

/// Chat message in OpenAI format.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatMessage {
    pub role: String,
    /// Message content (can be Text, Parts, or Option for tool/reasoning only messages)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<MessageContent>,
    /// Native reasoning/thought field (used by DeepSeek-R1 and similar)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// Native tool calling support
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self::new_system(content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new_user(content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new_assistant(content)
    }

    pub fn new_system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: Some(MessageContent::Text(content.into())),
            ..Default::default()
        }
    }
    pub fn new_user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: Some(MessageContent::Text(content.into())),
            ..Default::default()
        }
    }
    pub fn new_assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: Some(MessageContent::Text(content.into())),
            ..Default::default()
        }
    }
    pub fn new_tool(call_id: String, content: String) -> Self {
        Self {
            role: "tool".into(),
            content: Some(MessageContent::Text(content)),
            tool_call_id: Some(call_id),
            ..Default::default()
        }
    }

    pub fn user_with_image(text: impl Into<String>, image_base64: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: Some(MessageContent::Parts(vec![
                ContentPart::Text { text: text.into() },
                ContentPart::ImageUrl {
                    image_url: ImageUrlData {
                        url: format!("data:image/jpeg;base64,{}", image_base64.into()),
                    },
                },
            ])),
            ..Default::default()
        }
    }

    pub fn content_as_string(&self) -> String {
        self.content
            .as_ref()
            .map(|c| c.as_string())
            .unwrap_or_default()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// Chat completion request body.
#[derive(Debug, Clone, Serialize, Default)]
pub struct ChatRequest {
    pub model: String,
    pub messages: std::sync::Arc<Vec<ChatMessage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<std::collections::HashMap<String, f64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub r#type: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

/// Chat completion response.
#[derive(Debug, Deserialize)]
pub struct ChatResponse {
    pub id: String,
    pub choices: Vec<ChatChoice>,
    pub usage: Option<UsageInfo>,
    pub system_fingerprint: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
    pub logprobs: Option<LogprobsResult>,
}

#[derive(Debug, Deserialize)]
pub struct ChatDelta {
    pub role: Option<String>,
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
    pub stop: Option<Vec<String>>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub logit_bias: Option<std::collections::HashMap<String, f64>>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u32>,
    pub presence_penalty: Option<f64>,
    pub seed: Option<u64>,
    pub user: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct LogprobsResult {
    pub content: Option<Vec<TokenLogprob>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenLogprob {
    pub token: String,
    pub logprob: f64,
    pub top_logprobs: Option<Vec<TopLogprob>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: Option<u32>,
    pub total_tokens: u32,
}

#[derive(Debug, Deserialize)]
pub struct ApiErrorDetail {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: Option<String>,
    pub code: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EmbeddingRequest {
    pub input: Vec<String>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub truncate: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: UsageInfo,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
    pub embedding: Vec<f32>,
    pub index: u32,
}
// ─────────────────────────────────────────────────────────────
//  NIM Client
// ─────────────────────────────────────────────────────────────

/// NVIDIA NIM API client with built-in rate limiting and retry logic.
#[derive(Clone)]
pub struct NimClient {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    rate_limiter: Arc<std::sync::Mutex<MultiRateLimiter>>,
    /// Global concurrency limiter to prevent bursting that triggers OOM killings.
    concurrency_limiter: Arc<tokio::sync::Semaphore>,
    max_retries: u32,
    retry_backoff_ms: u64,
    circuit_breaker: Arc<CircuitBreaker>,
    #[allow(dead_code)]
    timeout: Duration,
    /// Persistent global max RPM for dynamic rebalancing.
    global_max_rpm: Arc<AtomicU32>,
    /// --- ZENITH 5.6: ROLE-BASED DISCOVERY ---
    /// Optional teacher manager for resolving high-level roles to model IDs.
    teacher_manager: Arc<Mutex<Option<Arc<crate::teacher::TeacherManager>>>>,
}

impl NimClient {
    /// Create a new NIM client.
    ///
    /// # Arguments
    /// * `api_key` - NVIDIA API key (can also be set via `NVIDIA_API_KEY` env var)
    /// * `base_url` - API base URL (default: `https://integrate.api.nvidia.com/v1`)
    /// * `max_rpm` - Maximum requests per minute (free tier: 40)
    pub fn new(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        max_rpm: u32,
        role_budgets: std::collections::HashMap<String, u32>,
    ) -> Result<Self> {
        let api_key = api_key.into();
        let api_key = if api_key.is_empty() {
            std::env::var("NVIDIA_API_KEY")
                .context("NVIDIA_API_KEY not set and no api_key provided")?
        } else {
            api_key
        };

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .connect_timeout(Duration::from_secs(30))
            .pool_max_idle_per_host(20)
            .build()?;

        Ok(Self {
            client,
            api_key,
            base_url: base_url.into(),
            rate_limiter: Arc::new(std::sync::Mutex::new(MultiRateLimiter::new(
                max_rpm,
                role_budgets,
            ))),
            concurrency_limiter: Arc::new(tokio::sync::Semaphore::new(128)),
            max_retries: 3,
            retry_backoff_ms: 2000,
            circuit_breaker: Arc::new(CircuitBreaker::new(5, 30)),
            timeout: Duration::from_secs(300),
            global_max_rpm: Arc::new(AtomicU32::new(max_rpm)),
            teacher_manager: Arc::new(Mutex::new(None)),
        })
    }

    /// Create from environment variables and defaults.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("NVIDIA_API_KEY")
            .context("NVIDIA_API_KEY environment variable not set")?;
        Self::new(
            api_key,
            "https://integrate.api.nvidia.com/v1",
            40,
            std::collections::HashMap::new(),
        )
    }

    /// Link a TeacherManager to enable role-based model resolution.
    pub async fn set_teacher_manager(&self, manager: Arc<crate::teacher::TeacherManager>) {
        let mut lock = self.teacher_manager.lock().await;
        *lock = Some(manager);
    }

    /// Resolve all enabled model IDs for a given role from the linked TeacherManager.
    pub async fn models_for_role(&self, role: &crate::teacher::TeacherRole) -> Result<Vec<String>> {
        let lock = self.teacher_manager.lock().await;
        let manager = lock
            .as_ref()
            .context("NimClient: TeacherManager not linked.")?;
        let teachers = manager.teachers_for_role(role);
        Ok(teachers.into_iter().map(|t| t.model.clone()).collect())
    }

    /// Resolve a model ID for a given role from the linked TeacherManager.
    pub async fn resolve_model_for_role(
        &self,
        role: &crate::teacher::TeacherRole,
    ) -> Result<String> {
        let config = self.resolve_config_for_role(role).await?;
        Ok(config.model)
    }

    /// Resolve a teacher config for a given role from the linked TeacherManager.
    pub async fn resolve_config_for_role(
        &self,
        role: &crate::teacher::TeacherRole,
    ) -> Result<crate::teacher::TeacherConfig> {
        let lock = self.teacher_manager.lock().await;
        let manager = lock
            .as_ref()
            .context("NimClient: TeacherManager not linked. Call set_teacher_manager() first.")?;
        let teacher = manager
            .best_teacher(role)
            .ok_or_else(|| anyhow::anyhow!("No enabled teacher found for role: {}", role))?;
        Ok(teacher.clone())
    }

    /// Perform a high-level query using a TeacherRole instead of a hardcoded model ID.
    pub async fn query_for_role(
        &self,
        role: crate::teacher::TeacherRole,
        _system_prompt: &str,
        user_prompt: &str,
    ) -> Result<String> {
        let config = self.resolve_config_for_role(&role).await?;
        self.query(
            &config.model,
            &role.to_string(),
            user_prompt,
            config.max_tokens,
            config.temperature,
        )
        .await
    }

    /// Perform a batch completion using a TeacherRole instead of a hardcoded model ID.
    pub async fn batch_query_for_role(
        &self,
        role: crate::teacher::TeacherRole,
        prompts: Vec<(String, String)>, // (system, user)
        concurrency: usize,
    ) -> Result<Vec<String>> {
        let config = self.resolve_config_for_role(&role).await?;
        let requests: Vec<(ChatRequest, String)> = prompts
            .into_iter()
            .map(|(sys, user)| {
                (
                    ChatRequest {
                        model: config.model.clone(),
                        messages: std::sync::Arc::new(vec![
                            ChatMessage::system(&sys),
                            ChatMessage::user(&user),
                        ]),
                        max_tokens: Some(config.max_tokens),
                        temperature: Some(config.temperature),
                        ..Default::default()
                    },
                    role.to_string(),
                )
            })
            .collect();

        let results = self.batch_chat_completion(requests, concurrency).await;
        let mut outputs = Vec::with_capacity(results.len());
        for res in results {
            let resp = res?;
            let content = resp
                .choices
                .first()
                .map(|c| c.message.content_as_string())
                .unwrap_or_default();
            outputs.push(content);
        }
        Ok(outputs)
    }

    /// Send a streaming chat completion request.
    pub async fn stream_chat_completion(
        &self,
        mut request: ChatRequest,
        _role: &str,
    ) -> Result<impl futures::Stream<Item = Result<ChatResponse>>> {
        request.stream = Some(true);
        let url = format!("{}/chat/completions", self.base_url);

        // 1. RATE LIMIT & CONCURRENCY
        // (Simplified for streaming, assuming caller manages high-level logic)
        let _permit =
            self.concurrency_limiter.acquire().await.map_err(|e| {
                anyhow::anyhow!("Concurrency semaphore poisoned (streaming): {}", e)
            })?;

        let response = self
            .client
            .post(&url)
            .header(CONTENT_TYPE, "application/json")
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await?;
            return Err(anyhow::anyhow!(
                "Streaming request failed ({}): {}",
                status,
                body
            ));
        }

        use futures::StreamExt;
        let stream = response.bytes_stream().map(|res| {
            res.map_err(anyhow::Error::from).and_then(|bytes| {
                let s = String::from_utf8_lossy(&bytes);
                let mut results = Vec::new();
                for line in s.lines() {
                    if line.is_empty() {
                        continue;
                    }
                    if line == "data: [DONE]" {
                        continue;
                    }
                    if let Some(json) = line.strip_prefix("data: ") {
                        if let Ok(resp) = serde_json::from_str::<ChatResponse>(json.trim()) {
                            results.push(Ok(resp));
                        }
                    }
                }
                if results.is_empty() {
                    // Could be a heartbeat or partial chunk, we return an empty-ish success or continue
                    Ok(ChatResponse {
                        id: "nop".into(),
                        choices: vec![],
                        usage: None,
                        system_fingerprint: None,
                    })
                } else {
                    results.remove(0)
                }
            })
        });

        Ok(stream)
    }

    /// Send a chat completion request to a specific model.
    pub async fn chat_completion(&self, request: &ChatRequest, role: &str) -> Result<ChatResponse> {
        // ALPHA-GRADE GUARD: Pre-flight circuit check
        self.circuit_breaker.check().await?;

        let url = format!("{}/chat/completions", self.base_url);
        let mut last_error = None;

        for attempt in 0..=self.max_retries {
            // Rate limit (Check-then-Sleep pattern to avoid locking the Mutex during sleep)
            loop {
                let wait_duration = {
                    let mut limiter = self.rate_limiter.lock().unwrap();

                    // 1. Check Concurrency Budget first
                    if !limiter.try_occupy_slot(role) {
                        tracing::debug!(target: "shakey::api", "Concurrency budget full for [{}]. Waiting for slot...", role);
                        Some(Duration::from_millis(100))
                    } else if let Some(duration) = limiter.try_acquire(role) {
                        tracing::debug!(target: "shakey::api", "Rate limit slot occupied. Sleeping {:.2}s", duration.as_secs_f64());
                        limiter.release_slot(role);
                        Some(duration)
                    } else {
                        None
                    }
                };

                if let Some(dur) = wait_duration {
                    tokio::time::sleep(dur).await;
                    continue;
                }

                break;
            }

            // Create RAII permit to ensure slot is released on drop
            let _role_permit = RolePermit {
                role: role.to_string(),
                limiter: self.rate_limiter.clone(),
            };

            tracing::info!(target: "shakey::api", "📡 SENDING_REQUEST to NIM model: {}", request.model);

            // Send request with precise timing
            let start = Instant::now();
            let request_builder = self
                .client
                .post(&url)
                .header(CONTENT_TYPE, "application/json")
                .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
                .json(&request.clone());

            // ── Robust Global Guard ──
            // Obtain a permit from the global system semaphore to prevent OOM
            let _global_permit = self
                .concurrency_limiter
                .acquire()
                .await
                .map_err(|e| anyhow::anyhow!("Global concurrency semaphore poisoned: {}", e))?;

            let result = request_builder.send().await;

            let latency = start.elapsed();
            if latency > Duration::from_secs(30) {
                tracing::warn!(target: "shakey::api", "NIM High Latency Detected: {:.2}s for model {}", latency.as_secs_f64(), request.model);
            }

            match result {
                Ok(response) => {
                    let status = response.status();

                    if status.is_success() {
                        let body = response.text().await?;
                        let chat_response: ChatResponse = serde_json::from_str(&body)
                            .with_context(|| {
                                format!(
                                    "Failed to parse response: {}",
                                    &body[..body.len().min(500)]
                                )
                            })?;

                        // Circuit Success: Reset failure count
                        self.circuit_breaker.record_success().await;
                        return Ok(chat_response);
                    }

                    // Rate limited (429) or server error (5xx) — retry
                    if status.as_u16() == 429 || status.is_server_error() {
                        // Check for Retry-After header BEFORE consuming body
                        let retry_after = response
                            .headers()
                            .get("retry-after")
                            .and_then(|h| h.to_str().ok())
                            .and_then(|s| s.parse::<u64>().ok())
                            .map(Duration::from_secs);

                        let body = response
                            .text()
                            .await
                            .unwrap_or_else(|_| "Unknown API error body".into());

                        let backoff = retry_after.unwrap_or_else(|| {
                            // ── Industry Grade: Full Jitter Exponential Backoff ──
                            // Prevents 'Thundering Herd' by spreading retries across
                            // a wider time-window during service recovery.
                            let base = self.retry_backoff_ms * 2u64.pow(attempt);
                            let jitter = rand::random::<u64>() % (base / 2).max(100);
                            Duration::from_millis(base + jitter)
                        });

                        SovereignMetrics::global().record_memory_pressure();

                        tracing::warn!(
                            target: "shakey::api",
                            "Model '{}' heavily loaded (HTTP {}). Attempt {}/{}. Retrying in {:.2}s... Error: {}",
                            request.model,
                            status,
                            attempt + 1,
                            self.max_retries + 1,
                            backoff.as_secs_f64(),
                            &body[..body.len().min(300)]
                        );
                        last_error = Some(anyhow::anyhow!(
                            "HTTP {} from {}: {}",
                            status,
                            request.model,
                            body
                        ));

                        // ALPHA-GRADE: Record failure for 5xx/429 errors
                        self.circuit_breaker.record_failure().await;

                        drop(_global_permit);
                        drop(_role_permit);

                        tokio::time::sleep(backoff).await;
                        continue;
                    }

                    // Client error (4xx) — don't retry
                    // Sovereign Client Error: No retry
                    let body = response
                        .text()
                        .await
                        .unwrap_or_else(|_| "Client error without body".into());
                    return Err(anyhow::anyhow!(
                        "NVIDIA API Error [{}]: Model '{}' rejected request with body: {}",
                        status,
                        request.model,
                        body
                    ));
                }
                Err(e) => {
                    let backoff = Duration::from_millis(
                        self.retry_backoff_ms * 2u64.pow(attempt) + (rand::random::<u64>() % 1000),
                    );

                    if e.is_timeout() {
                        tracing::error!(target: "shakey::api", "NIM API Timeout for model {}: {e}. Retrying in {:.2}s...", request.model, backoff.as_secs_f64());
                    } else {
                        tracing::error!(target: "shakey::api", "NIM Connection Error: {e}. Retrying in {:.2}s...", backoff.as_secs_f64());
                    }

                    last_error = Some(e.into());

                    drop(_global_permit);
                    drop(_role_permit);

                    tokio::time::sleep(backoff).await;
                    continue;
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            anyhow::anyhow!(
                "NIM Client: Fatal error after {} attempts",
                self.max_retries + 1
            )
        }))
    }

    /// Simple helper: query a model with a single user message.
    pub async fn query(
        &self,
        model: &str,
        role: &str,
        prompt: &str,
        max_tokens: u32,
        temperature: f64,
    ) -> Result<String> {
        let request = ChatRequest {
            model: model.to_string(),
            messages: std::sync::Arc::new(vec![ChatMessage::user(prompt)]),
            temperature: Some(temperature),
            max_tokens: Some(max_tokens),
            stream: Some(false),
            ..Default::default()
        };

        let response = self.chat_completion(&request, role).await?;
        let content = response
            .choices
            .first()
            .map(|c| c.message.content_as_string())
            .unwrap_or_default();

        Ok(content)
    }

    /// Query with logprobs enabled (for distillation).
    ///
    /// Returns the completion text and per-token log probabilities.
    pub async fn query_with_logprobs(
        &self,
        model: &str,
        role: &str,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f64,
        top_logprobs: u32,
    ) -> Result<(String, Vec<TokenLogprob>)> {
        let request = ChatRequest {
            model: model.to_string(),
            messages: std::sync::Arc::new(messages),
            temperature: Some(temperature),
            max_tokens: Some(max_tokens),
            logprobs: Some(true),
            top_logprobs: Some(top_logprobs),
            stream: Some(false),
            ..Default::default()
        };

        let response = self.chat_completion(&request, role).await?;

        let choice = response.choices.first().context("No choices in response")?;

        let content = choice.message.content_as_string();
        let logprobs = choice
            .logprobs
            .as_ref()
            .and_then(|lp| lp.content.clone())
            .context("Distillation failed: logprobs expected but not found in teacher response")?;

        Ok((content, logprobs))
    }

    /// Get the current global rate limiter status.
    pub async fn rate_status(&self) -> usize {
        let limiter = self.rate_limiter.lock().unwrap();
        limiter.global_count()
    }

    /// Set a new rate budget for the client.
    pub async fn set_rate_budget(
        &self,
        max_rpm: u32,
        role_budgets: std::collections::HashMap<String, u32>,
    ) {
        let mut limiter = self.rate_limiter.lock().unwrap();
        *limiter = MultiRateLimiter::new(max_rpm, role_budgets);
        self.global_max_rpm.store(max_rpm, Ordering::SeqCst);
    }

    /// ── Zenith Sovereign: Dynamic Budget Shifting ──
    /// Directs the agent's API "attention" to a specific learning domain.
    pub async fn rebalance_budget(&self, role: &str, is_urgent: bool) {
        let mut limiter = self.rate_limiter.lock().unwrap();
        let global_max = self.global_max_rpm.load(Ordering::SeqCst);
        limiter.rebalance(role, global_max, is_urgent);
    }

    /// Send a batch of chat completion requests in parallel, respecting rate limits.
    ///
    /// # Arguments
    /// * `requests` - List of (ChatRequest, role) tuples
    /// * `concurrency` - Number of concurrent requests to allow
    pub async fn batch_chat_completion(
        &self,
        requests: Vec<(ChatRequest, String)>,
        concurrency: usize,
    ) -> Vec<Result<ChatResponse>> {
        use futures::StreamExt;

        let stream = futures::stream::iter(requests)
            .map(|(req, role)| {
                let client = self.clone();
                async move { client.chat_completion(&req, &role).await }
            })
            .buffer_unordered(concurrency);

        stream.collect().await
    }

    pub async fn embeddings(
        &self,
        texts: Vec<String>,
        model: &str,
        input_type: &str,
        role: &str,
    ) -> Result<Vec<Vec<f32>>> {
        let request = EmbeddingRequest {
            input: texts,
            model: model.to_string(),
            input_type: Some(input_type.to_string()),
            encoding_format: Some("float".into()),
            truncate: Some("NONE".into()),
        };

        let url = format!("{}/embeddings", self.base_url);
        let mut last_error = None;

        for attempt in 0..=self.max_retries {
            // 1. Rate Limiting
            loop {
                let wait_time = {
                    let mut limiter = self.rate_limiter.lock().unwrap();
                    limiter.try_acquire(role)
                };

                match wait_time {
                    Some(duration) => {
                        tracing::debug!(target: "shakey::api", "Rate limit slot occupied (embeddings). Sleeping {:.2}s", duration.as_secs_f64());
                        tokio::time::sleep(duration).await;
                    }
                    None => break,
                }
            }

            // 2. HTTP Request
            let request_builder = self
                .client
                .post(&url)
                .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
                .header(CONTENT_TYPE, "application/json")
                .json(&request.clone());

            // ── Robust Concurrency Guard ──
            let _permit = self.concurrency_limiter.acquire().await.map_err(|e| {
                anyhow::anyhow!("Concurrency semaphore poisoned (embeddings): {}", e)
            })?;

            let response_result = request_builder.send().await;

            match response_result {
                Ok(response) => {
                    let status = response.status();
                    if status.is_success() {
                        let body = response
                            .text()
                            .await
                            .context("Failed to read NIM response body")?;
                        let embedding_response: EmbeddingResponse = serde_json::from_str(&body)
                            .map_err(|e| {
                                tracing::error!(
                                    "Failed to parse NIM embedding JSON: {}. Full body: {}",
                                    e,
                                    body
                                );
                                anyhow::anyhow!("Failed to parse NIM embedding response: {}", e)
                            })?;

                        let mut data = embedding_response.data;
                        data.sort_by_key(|d| d.index);
                        return Ok(data.into_iter().map(|d| d.embedding).collect());
                    }

                    if status.as_u16() == 429 || status.is_server_error() {
                        let base = self.retry_backoff_ms * 2u64.pow(attempt);
                        let jitter = rand::random::<u64>() % (base / 2).max(100);
                        let backoff = Duration::from_millis(base + jitter);

                        SovereignMetrics::global().record_memory_pressure();

                        tracing::warn!(target: "shakey::api", "Embedding API {} ({}). Retrying in {:.2}s...", status, model, backoff.as_secs_f64());
                        last_error = Some(anyhow::anyhow!(
                            "Embedding API Error [{}]: {}",
                            status,
                            response.text().await.unwrap_or_default()
                        ));
                        tokio::time::sleep(backoff).await;
                        continue;
                    }

                    let body = response.text().await.unwrap_or_default();
                    return Err(anyhow::anyhow!(
                        "NIM Embedding API Fatal Error [{}]: {}",
                        status,
                        body
                    ));
                }
                Err(e) => {
                    let backoff = Duration::from_millis(
                        self.retry_backoff_ms * 2u64.pow(attempt) + (rand::random::<u64>() % 1000),
                    );
                    tracing::error!(target: "shakey::api", "NIM Connection Error: {e}. Retrying in {:.2}s...", backoff.as_secs_f64());
                    last_error = Some(e.into());
                    tokio::time::sleep(backoff).await;
                    continue;
                }
            }
        }

        Err(last_error
            .unwrap_or_else(|| anyhow::anyhow!("NIM Embedding Client: Fatal error after retries")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter_creation() {
        let limiter = RateLimiter::new(40);
        assert_eq!(limiter.max_rpm, 40);
        assert_eq!(limiter.current_count(), 0);
    }

    #[test]
    fn test_chat_message_constructors() {
        let sys = ChatMessage::system("You are helpful");
        assert_eq!(sys.role, "system");

        let user = ChatMessage::user("Hello");
        assert_eq!(user.role, "user");

        let asst = ChatMessage::assistant("Hi there");
        assert_eq!(asst.role, "assistant");
    }

    #[test]
    fn test_request_serialization() {
        let req = ChatRequest {
            model: "test-model".into(),
            messages: std::sync::Arc::new(vec![ChatMessage::user("Hello")]),
            ..Default::default()
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("test-model"));
        // top_p should be absent (skip_serializing_if = None)
        assert!(!json.contains("top_p"));
    }
}
