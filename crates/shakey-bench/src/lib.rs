//! # Shakey Bench
//!
//! Benchmark harness for the Shakey autonomous agent.
//!
//! Provides shared utilities for micro-benchmarks (criterion)
//! and macro-benchmarks (full OODA cycle evaluation).
//!
//! Re-exports core types needed by benchmark binaries.

pub use shakey_core::inference::{InferenceEngine, SamplingParams};
pub use shakey_core::model::config::ModelConfig;
pub use shakey_core::model::transformer::TransformerModel;
pub use shakey_core::tokenizer::Tokenizer;
