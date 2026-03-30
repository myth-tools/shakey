//! # Shakey Core
//!
//! Core LLM architecture for the autonomous self-evolving agent.
//!
//! ## Architecture: BitNet 1.58-bit Sparse MoE Transformer
//!
//! - **BitLinear**: Ternary weight quantization {-1, 0, +1} — inference uses
//!   only additions and subtractions, no float multiplications.
//! - **Sparse MoE**: Multiple expert FFNs with top-k routing — only a fraction
//!   of parameters are active per token.
//! - **GQA**: Grouped Query Attention — fewer KV heads than Q heads for memory
//!   efficiency.
//! - **RoPE**: Rotary Position Embeddings — supports arbitrary context lengths.
//!
//! ## Modules
//!
//! - `model` — Model architecture (layers, attention, MoE, transformer blocks)
//! - `tokenizer` — BPE tokenizer integration
//! - `inference` — Autoregressive generation engine with KV-cache
//! - `training` — Training loop, distillation loss, checkpoint/resume

pub mod env;
pub mod inference;
pub mod memory;
pub mod metrics;
pub mod model;
pub mod tokenizer;
pub mod training;
