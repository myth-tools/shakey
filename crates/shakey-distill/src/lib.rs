//! # Shakey Distill
//!
//! NVIDIA NIM teacher distillation pipeline.
//!
//! Connects to NVIDIA's cloud API to query powerful teacher models,
//! then uses their outputs to train the local student model via
//! knowledge distillation.
//!
//! ## Key Features
//! - **40 RPM rate limiting** with token bucket + exponential backoff
//! - **Multi-teacher ensemble** — query multiple models, aggregate
//! - **Reward-guided filtering** — use NVIDIA reward models to score quality
//! - **Synthetic data generation** — teacher creates diverse training examples

pub mod adversary;
pub mod alchemist;
pub mod batch;
pub mod data_gen;
pub mod filter;
pub mod nim_client;
pub mod reward;
pub mod teacher;
pub mod utils;
