//! Shakey Agent — Autonomous self-evolving agent with OODA loop.
//!
//! Provides the core agent logic, evolution control, and tool execution.

pub mod evolution;
pub mod generation;
pub mod mcp_server;
pub mod memory;
pub mod ooda;
pub mod tools;

use serde::{Deserialize, Serialize};
use std::fmt;

// Re-exporting core capability structure for workspace visibility
pub use shakey_core::training::capabilities::CapabilityMatrix;

/// Strategy identification for OODA loop
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Strategy {
    Distill {
        domain: String,
        token_budget: usize,
    },
    VisionDistill {
        image_path: String,
        objective: String,
    },
    WebScrape {
        query: String,
        max_pages: usize,
    },
    Synthesize {
        topic: String,
        count: usize,
    },
    Benchmark,
    ToolBuild {
        name: String,
        description: String,
    },
    Expand {
        target_stage: String,
    },
    SovereignResearch {
        objective: String,
    },
    Backtrack {
        target_version: String,
        reason: String,
    },
    HardReboot {
        reason: String,
    },
    Train {
        data_path: String,
        epochs: usize,
    },
    WebSearch {
        query: String,
    },
    ToolRepair {
        name: String,
        error: String,
    },
    OnlineFineTune {
        prompt: String,
        completion: String,
        is_correction: bool,
    },
    SelfIndex {
        workspace_path: String,
    },
    Consolidate {
        cycle_count: usize,
    },
    SovereignOptimization {
        tool_name: String,
        metric: String,
    },
    MentalAnalysis {
        objective: String,
    },
    SovereignCascade {
        strategies: Vec<Strategy>,
        objective: String,
    },
    Reflect {
        strategy: Box<Strategy>,
        reasoning: String,
    },
    ConsensusAudit {
        topic: String,
        responses: Vec<String>,
    },
    Idle {
        reason: String,
    },
    /// ZENITH 5.5: Native Tool Call (Industry-Parity)
    NativeToolCall {
        id: String,
        name: String,
        arguments: String,
    },
}

impl fmt::Display for Strategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Distill { domain, .. } => write!(f, "Distill({domain})"),
            Self::VisionDistill { objective, .. } => write!(f, "VisionDistill({objective})"),
            Self::WebScrape { query, .. } => write!(f, "WebScrape({query})"),
            Self::Synthesize { topic, .. } => write!(f, "Synthesize({topic})"),
            Self::Benchmark => write!(f, "Benchmark"),
            Self::ToolBuild { name, .. } => write!(f, "ToolBuild({name})"),
            Self::Expand { target_stage } => write!(f, "Expand({target_stage})"),
            Self::SovereignResearch { objective } => write!(f, "SovereignResearch({objective})"),
            Self::Backtrack { target_version, .. } => write!(f, "Backtrack({target_version})"),
            Self::HardReboot { reason } => write!(f, "HardReboot({reason})"),
            Self::Train { data_path, .. } => write!(f, "Train({data_path})"),
            Self::WebSearch { query } => write!(f, "WebSearch({query})"),
            Self::ToolRepair { name, .. } => write!(f, "ToolRepair({name})"),
            Self::OnlineFineTune { is_correction, .. } => {
                write!(f, "OnlineFineTune(is_correction={is_correction})")
            }
            Self::SelfIndex { workspace_path } => write!(f, "SelfIndex({workspace_path})"),
            Self::Consolidate { cycle_count } => write!(f, "Consolidate({cycle_count} cycles)"),
            Self::SovereignOptimization { tool_name, metric } => {
                write!(f, "SovereignOptimization({tool_name}, {metric})")
            }
            Self::MentalAnalysis { objective } => write!(f, "MentalAnalysis({objective})"),
            Self::SovereignCascade {
                strategies,
                objective,
            } => {
                write!(
                    f,
                    "SovereignCascade(n={}, objective=\"{}\")",
                    strategies.len(),
                    objective
                )
            }
            Self::Reflect { reasoning, .. } => write!(f, "Reflect({reasoning})"),
            Self::ConsensusAudit { topic, .. } => write!(f, "ConsensusAudit({topic})"),
            Self::Idle { reason } => write!(f, "Idle({reason})"),
            Self::NativeToolCall { name, .. } => write!(f, "NativeToolCall({name})"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CycleRecord {
    pub cycle_id: String,
    pub strategy: Strategy,
    pub started_at: String,
    pub completed_at: String,
    pub duration_secs: f64,
    pub success: bool,
    pub improvement: f64,
    pub loss: f64,
    pub tokens_trained: u64,
    pub committed: bool,
    pub notes: String,
}
