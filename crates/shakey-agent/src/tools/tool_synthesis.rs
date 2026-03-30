use crate::tools::native_sandbox::NativeSandbox;
use crate::tools::registry::{ToolImpl, ToolMetadata, ToolRegistry};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Requirement for a new tool synthesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolRequirement {
    pub name: String,
    pub description: String,
    pub problematic_context: String,
    pub expected_input_schema: String,
}

/// Structured response from LLM for synthesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SynthesisResponse {
    code: String,
    test_cases: Vec<TestCase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TestCase {
    name: String,
    input: String,
    expected_contains: String,
}

pub struct ToolSynthesizer {
    registry: Arc<RwLock<ToolRegistry>>,
    _sandbox_path: std::path::PathBuf,
}

impl ToolSynthesizer {
    pub fn new(
        registry: Arc<RwLock<ToolRegistry>>,
        sandbox_path: impl Into<std::path::PathBuf>,
    ) -> Self {
        Self {
            registry,
            _sandbox_path: sandbox_path.into(),
        }
    }

    /// Robustly extract JSON from mixed-text/markdown LLM responses.
    fn extract_json(response: &str) -> String {
        let trimmed = response.trim();
        if let Some(start) = trimmed.find('{') {
            if let Some(end) = trimmed.rfind('}') {
                return trimmed[start..=end].to_string();
            }
        }
        trimmed.to_string()
    }

    /// Perform a high-assurance security audit of synthesized code using a Teacher model.
    async fn security_audit(
        &self,
        code: &str,
        prompt_llm: &impl Fn(String) -> Result<String>,
    ) -> Result<()> {
        tracing::info!(target: "shakey::evolution", "🛡️ Performing Sovereign Security Audit...");

        let prompt = format!(
            "System: You are the Sovereign Security Auditor of Project Shakey.\n\
             Task: Audit the following Rust code for sandbox escapes, malicious intent, or command injection. \n\
             Code: \n{}\n\n\
             Output JSON ONLY: {{\"safe\": true/false, \"reason\": \"...\"}}",
            code
        );

        let res = prompt_llm(prompt)?;
        let clean = Self::extract_json(&res);
        let audit: serde_json::Value = serde_json::from_str(&clean)?;

        if audit["safe"].as_bool().unwrap_or(false) {
            Ok(())
        } else {
            let reason = audit["reason"]
                .as_str()
                .unwrap_or("Unknown security violation");
            Err(anyhow::anyhow!("Security Audit FAILED: {}", reason))
        }
    }

    /// Synthesize a new tool autonomously with verification and high-assurance auditing.
    pub async fn synthesize_tool(
        &self,
        req: ToolRequirement,
        prompt_llm: impl Fn(String) -> Result<String>,
    ) -> Result<()> {
        tracing::info!(target: "shakey::evolution", "🔧 Sovereign Synthesis Phase 2: Building tool '{}'", req.name);

        let mut error_context = String::new();
        let mut attempts = 0;

        while attempts < 2 {
            let prompt = format!(
                "### Sovereign Tool Synthesis Request (High-Assurance)\n\
                 Objective: {}\n\
                 Context: {}\n\
                 {}\n\
                 Task: Output a JSON object with 'code' and 3 'test_cases' (HappyPath, EdgeCase, MalformedInput).\n\
                 Required JSON: {{ \"code\": \"...\", \"test_cases\": [{{ \"name\": \"...\", \"input\": \"...\", \"expected_contains\": \"...\" }}] }}",
                req.description,
                req.problematic_context,
                if error_context.is_empty() { "".to_string() } else { format!("Previous Failure Context: {}\nRefine the logic.", error_context) }
            );

            let raw_res = prompt_llm(prompt).context("LLM synthesis generation failed")?;
            let clean_json = Self::extract_json(&raw_res);

            let synth: SynthesisResponse = match serde_json::from_str(&clean_json) {
                Ok(s) => s,
                Err(e) => {
                    error_context = format!("JSON Parsing Failed: {}", e);
                    attempts += 1;
                    continue;
                }
            };

            // ── STEP 1: Security Audit ──
            if let Err(e) = self.security_audit(&synth.code, &prompt_llm).await {
                error_context = e.to_string();
                attempts += 1;
                continue;
            }

            // ── STEP 2: VERIFICATION (Multi-pass) ──
            tracing::info!(target: "shakey::evolution", "🧪 Running {} multi-pass test cases for '{}'...", synth.test_cases.len(), req.name);

            let sandbox = NativeSandbox::new()?;
            let mut all_tests_passed = true;

            for test in &synth.test_cases {
                match sandbox.run_secure_script(&synth.code, 10000).await {
                    Ok(result) => {
                        if !result.contains(&test.expected_contains) {
                            error_context = format!(
                                "Test '{}' FAILED: Expected '{}' but got '{}'",
                                test.name, test.expected_contains, result
                            );
                            all_tests_passed = false;
                            break;
                        }
                    }
                    Err(e) => {
                        error_context = format!("Test '{}' CRASHED: {}", test.name, e);
                        all_tests_passed = false;
                        break;
                    }
                }
            }

            if all_tests_passed {
                tracing::info!(target: "shakey::evolution", "✅ All tests PASSED. Registering tool '{}'.", req.name);
                let metadata = ToolMetadata {
                    name: req.name.clone(),
                    description: req.description.clone(),
                    input_schema: req.expected_input_schema.clone(),
                    permissions: vec!["process".into(), "memory".into()],
                    avg_fuel_consumed: Some(800_000),
                };

                let mut registry = self.registry.write().await;
                registry.register(metadata, ToolImpl::Native);
                return Ok(());
            }

            attempts += 1;
        }

        Err(anyhow::anyhow!(
            "High-assurance synthesis failed for tool '{}'. Last context: {}",
            req.name,
            error_context
        ))
    }
}
