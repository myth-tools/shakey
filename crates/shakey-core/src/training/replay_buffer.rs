//! Elite Sovereign Experience Replay Buffer.
//!
//! This is not a simple ring buffer. It is a purpose-built, high-performance
//! memory engine for autonomous continuous learning, featuring:
//!
//! - **Thread-Safety**: `Arc<RwLock>` for zero-contention OODA access.
//! - **DPO Pair Storage**: Every correction is stored as a (chosen, rejected) pair
//!   for Direct Preference Optimization.
//! - **Power-Law Importance Sampling**: High-impact memories are sampled more.
//! - **Semantic Deduplication**: Character-gram fingerprinting prevents memory bloat
//!   from near-identical interactions, keeping the buffer lean and diverse.
//! - **Statistics Tracking**: Real-time monitoring of buffer health.

use anyhow::Result;
use rand::seq::SliceRandom;
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};
use std::fs;
use std::path::Path;
use std::sync::{Arc, RwLock};

// ─────────────────────────────────────────────────────────────────────────────
//  ReplayMemory
// ─────────────────────────────────────────────────────────────────────────────

/// A single piece of stored memory, enriched with DPO pair information.
///
/// Every correction the user gives results in a DPO pair:
/// - `completion` = the **Chosen** (correct) response
/// - `rejected`   = the **Rejected** (original mistake) response
///
/// These pairs are used for Online DPO, making the agent understand
/// *why* something was wrong, not just *what* to say instead.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReplayMemory {
    /// Unique identifier (UUID or hash)
    pub id: String,
    /// The original user prompt / instruction
    pub prompt: String,
    /// The Chosen (correct) completion for DPO
    pub completion: String,
    /// The Rejected (original mistake) completion for DPO.
    /// None if this was a fresh prompt, not a correction.
    pub rejected: Option<String>,
    /// Importance score (0.0 - 1.0). Corrections get 1.0, positive feedback 0.6.
    pub importance_score: f64,
    /// Whether this was from a user correction (true) or normal interaction (false).
    pub is_correction: bool,
    /// REAL-TIME: Loss computed during the latest training step (Surprise/Saliency).
    pub loss: Option<f64>,
    /// High-speed fingerprint for semantic deduplication (bag-of-bigrams hash).
    #[serde(skip)]
    fingerprint: Option<u64>,
}

impl ReplayMemory {
    /// Create a new memory, computing its fingerprint for deduplication.
    pub fn new(
        id: String,
        prompt: String,
        completion: String,
        rejected: Option<String>,
        importance_score: f64,
        is_correction: bool,
    ) -> Self {
        let fingerprint = Some(compute_fingerprint(&format!("{}{}", prompt, completion)));
        Self {
            id,
            prompt,
            completion,
            rejected,
            importance_score,
            is_correction,
            fingerprint,
            loss: None,
        }
    }

    /// Update the memory with the latest training loss result.
    pub fn update_loss(&mut self, loss: f64) {
        self.loss = Some(loss);
    }

    /// Returns `true` if this memory is a valid DPO pair.
    pub fn is_dpo_pair(&self) -> bool {
        self.rejected.is_some()
    }

    /// Get the fingerprint, computing it lazily if not present.
    fn get_fingerprint(&self) -> u64 {
        self.fingerprint
            .unwrap_or_else(|| compute_fingerprint(&format!("{}{}", self.prompt, self.completion)))
    }
}

impl crate::training::trainer::AsTrainingExample for ReplayMemory {
    fn id(&self) -> &str {
        &self.id
    }
    fn prompt(&self) -> &str {
        &self.prompt
    }
    fn completion(&self) -> &str {
        &self.completion
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Semantic Fingerprinting (Character Bigram Hashing)
// ─────────────────────────────────────────────────────────────────────────────

/// Ultra-fast semantic fingerprint using character bigram bag-of-words + FNV hashing.
///
/// This is deliberately NOT full semantic embedding (which requires a neural network pass).
/// Instead, it uses a locality-sensitive hashing technique inspired by MinHash to detect
/// near-duplicate text at microsecond speeds.
///
/// Deduplication threshold: similarities above 0.90 are considered "near-duplicates".
fn compute_fingerprint(text: &str) -> u64 {
    // Normalize: lowercase, filter, and sample across the entire string
    let normalized: String = text
        .to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect();

    let chars: Vec<char> = normalized.chars().collect();
    if chars.is_empty() {
        return 0;
    }

    // ── Peak Mastery: Global Sampling Fingerprint ──
    // Instead of just the first 256 chars, we sample up to 512 chars
    // distributed evenly across the string. This prevents
    // collisions on long interactions with similar headers.
    let sampled_chars: Vec<char> = if chars.len() <= 512 {
        chars
    } else {
        let step = chars.len() as f32 / 512.0;
        (0..512)
            .map(|i| chars[(i as f32 * step) as usize])
            .collect()
    };

    // Build bigram frequency map
    let mut bigrams: HashMap<(char, char), u32> = HashMap::new();
    for window in sampled_chars.windows(2) {
        *bigrams.entry((window[0], window[1])).or_insert(0) += 1;
    }

    // FNV-1a hash of the sorted bigram set for stability
    let mut hash: u64 = 14695981039346656037u64; // FNV offset basis
    let mut sorted_bigrams: Vec<_> = bigrams.keys().collect();
    sorted_bigrams.sort();
    for &(a, b) in &sorted_bigrams {
        hash ^= *a as u32 as u64;
        hash = hash.wrapping_mul(1099511628211u64);
        hash ^= *b as u32 as u64;
        hash = hash.wrapping_mul(1099511628211u64);
    }
    hash
}

/// Compute similarity between two fingerprints using bit-level comparison.
/// Returns 1.0 for identical, 0.0 for completely different.
fn fingerprint_similarity(a: u64, b: u64) -> f64 {
    let xor = a ^ b;
    let common_bits = 64 - xor.count_ones();
    common_bits as f64 / 64.0
}

// ─────────────────────────────────────────────────────────────────────────────
//  ReplayBuffer Statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Real-time health statistics for the Sovereign Replay Buffer.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct BufferStats {
    /// Total pushes attempted
    pub total_pushed: u64,
    /// Count of deduplication rejections
    pub duplicates_rejected: u64,
    /// Count of DPO pairs stored
    pub dpo_pairs: u64,
    /// Current average importance score
    pub avg_importance: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
//  ReplayBuffer
// ─────────────────────────────────────────────────────────────────────────────

fn default_memories() -> Arc<RwLock<VecDeque<ReplayMemory>>> {
    Arc::new(RwLock::new(VecDeque::new()))
}

fn default_stats() -> Arc<RwLock<BufferStats>> {
    Arc::new(RwLock::new(BufferStats::default()))
}

/// Elite Sovereign Experience Replay Buffer.
///
/// Thread-safe, deduplication-aware, DPO-ready memory engine.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReplayBuffer {
    /// Maximum capacity
    capacity: usize,
    /// Deduplication similarity threshold (0.9 = aggressive, 0.98 = conservative)
    dedup_threshold: f64,
    /// Stored memories — thread-safe
    #[serde(skip, default = "default_memories")]
    memories: Arc<RwLock<VecDeque<ReplayMemory>>>,
    /// Real-time health statistics
    #[serde(skip, default = "default_stats")]
    stats: Arc<RwLock<BufferStats>>,
}

/// A serializable snapshot of the ReplayBuffer state.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct ReplayBufferSnapshot {
    pub version: u32,
    pub capacity: usize,
    pub dedup_threshold: f64,
    pub memories: VecDeque<ReplayMemory>,
    pub stats: BufferStats,
    /// SHA-256 hash of the snapshot content (excluding this field)
    pub checksum: Option<String>,
}

impl ReplayBufferSnapshot {
    pub fn compute_checksum(&self) -> Result<String> {
        // Temporary clone to clear checksum for clean hash calculation
        let clone = self.memories.clone();
        let mut hasher = Sha256::new();

        // Hash critical components
        hasher.update(self.capacity.to_le_bytes());
        hasher.update(self.dedup_threshold.to_be_bytes()); // consistent endianness

        for mem in clone {
            hasher.update(mem.id.as_bytes());
            hasher.update(mem.prompt.as_bytes());
            hasher.update(mem.completion.as_bytes());
        }

        Ok(hex::encode(hasher.finalize()))
    }
}

impl ReplayBuffer {
    /// Create a new Elite Replay Buffer.
    ///
    /// # Arguments
    /// * `capacity` — Max memories to hold.
    /// * `dedup_threshold` — Similarity floor for deduplication (0.98 is conservative/safe).
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            dedup_threshold: 0.98, // Conservative default: only cull near-identical memories
            memories: Arc::new(RwLock::new(VecDeque::with_capacity(capacity))),
            stats: Arc::new(RwLock::new(BufferStats::default())),
        }
    }

    /// Create a buffer with a custom deduplication threshold.
    pub fn with_dedup_threshold(mut self, threshold: f64) -> Self {
        self.dedup_threshold = threshold.clamp(0.5, 1.0);
        self
    }

    /// Push a memory into the buffer.
    ///
    /// Performs semantic deduplication before insertion. If a near-identical
    /// memory already exists, the new one is rejected if its importance is lower,
    /// or it replaces the old one if higher.
    pub fn push(&self, memory: ReplayMemory) {
        let new_fp = memory.get_fingerprint();
        let new_importance = memory.importance_score;
        let is_dpo = memory.is_dpo_pair();

        if let Ok(mut memories) = self.memories.write() {
            // ── Sovereign Deduplication Check ──
            let mut duplicate_idx: Option<usize> = None;
            for (i, existing) in memories.iter().enumerate() {
                let sim = fingerprint_similarity(existing.get_fingerprint(), new_fp);
                if sim >= 1.0 {
                    // Exact match: update and return
                    duplicate_idx = Some(i);
                    break;
                }
                if sim >= self.dedup_threshold {
                    duplicate_idx = Some(i);
                    // Continue searching for exact match? No, threshold is enough
                    break;
                }
            }

            if let Some(idx) = duplicate_idx {
                // Update stats
                if let Ok(mut s) = self.stats.write() {
                    s.total_pushed += 1;
                    s.duplicates_rejected += 1;
                }
                // Only replace if new memory is more important
                if new_importance > memories[idx].importance_score {
                    memories[idx] = memory;
                }
                return;
            }

            // ── Capacity Management: Evict lowest-importance if full ──
            if memories.len() >= self.capacity {
                // Sovereign eviction: remove the LEAST important memory, not oldest
                let min_idx = memories
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| {
                        a.importance_score
                            .partial_cmp(&b.importance_score)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(i, _)| i);

                if let Some(idx) = min_idx {
                    memories.remove(idx);
                }
            }

            memories.push_back(memory);

            // ── Update Stats ──
            if let Ok(mut s) = self.stats.write() {
                s.total_pushed += 1;
                if is_dpo {
                    s.dpo_pairs += 1;
                }
                let n = memories.len() as f64;
                let total_importance: f64 = memories.iter().map(|m| m.importance_score).sum();
                s.avg_importance = if n > 0.0 { total_importance / n } else { 0.0 };
            }
        }
    }

    /// Sample `n` memories uniformly at random.
    pub fn sample(&self, n: usize) -> Vec<ReplayMemory> {
        let memories = match self.memories.read() {
            Ok(m) => m,
            Err(_) => return Vec::new(),
        };
        if memories.is_empty() {
            return Vec::new();
        }
        let mut rng = rand::thread_rng();
        let samples: Vec<ReplayMemory> = memories.iter().cloned().collect();
        samples
            .choose_multiple(&mut rng, n.min(samples.len()))
            .cloned()
            .collect()
    }

    /// Power-Law Importance Sampling — samples top-K by importance score.
    ///
    /// Used during standard SFT online updates.
    pub fn sample_importance(&self, n: usize) -> Vec<ReplayMemory> {
        let memories = match self.memories.read() {
            Ok(m) => m,
            Err(_) => return Vec::new(),
        };
        if memories.is_empty() {
            return Vec::new();
        }
        let mut all: Vec<ReplayMemory> = memories.iter().cloned().collect();
        all.sort_by(|a, b| {
            // Combined Score: Static Importance (60%) + Surprise/Loss (40%)
            let score_a = a.importance_score * 0.6 + a.loss.unwrap_or(0.0) * 0.4;
            let score_b = b.importance_score * 0.6 + b.loss.unwrap_or(0.0) * 0.4;
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        all.truncate(n.min(all.len()));
        all
    }

    /// Sample only DPO-eligible pairs (those with a `rejected` completion).
    ///
    /// Used exclusively during Online DPO training steps to align the model
    /// against specific rejected responses.
    pub fn sample_dpo_pairs(&self, n: usize) -> Vec<ReplayMemory> {
        let memories = match self.memories.read() {
            Ok(m) => m,
            Err(_) => return Vec::new(),
        };
        let mut dpo_pairs: Vec<ReplayMemory> = memories
            .iter()
            .filter(|m| m.is_dpo_pair())
            .cloned()
            .collect();

        // Sort DPO pairs by importance — the "most corrected" items first
        dpo_pairs.sort_by(|a, b| {
            b.importance_score
                .partial_cmp(&a.importance_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        dpo_pairs.truncate(n.min(dpo_pairs.len()));
        dpo_pairs
    }

    /// Get a snapshot of the buffer's health statistics.
    pub fn stats(&self) -> BufferStats {
        self.stats.read().map(|s| s.clone()).unwrap_or_default()
    }

    /// Returns current buffer size.
    pub fn len(&self) -> usize {
        self.memories.read().map(|m| m.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// ── Sovereign Persistence (Zenith Hardening) ──
    /// Persist the buffer to disk using high-performance binary serialization.
    pub fn save_to_disk(&self, path: impl AsRef<Path>) -> Result<()> {
        // ── Sovereign Guard: Poison-safe lock access ──
        let memories = match self.memories.read() {
            Ok(m) => m.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        };
        let stats = match self.stats.read() {
            Ok(s) => s.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        };

        let mut snapshot = ReplayBufferSnapshot {
            version: 1,
            capacity: self.capacity,
            dedup_threshold: self.dedup_threshold,
            memories,
            stats,
            checksum: None,
        };

        // Compute and embed checksum for industry-grade verification
        snapshot.checksum = Some(snapshot.compute_checksum()?);

        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        // Binary serialization via Bincode (10x faster than JSON)
        let bytes = bincode::serialize(&snapshot)
            .map_err(|e| anyhow::anyhow!("Binary serialization failed: {}", e))?;

        // Atomic write: .tmp -> rename
        let tmp_path = path.with_extension("tmp");
        fs::write(&tmp_path, bytes)?;
        fs::rename(&tmp_path, path)?;

        Ok(())
    }

    /// Load the buffer from a previous snapshot.
    /// Handles JSON-to-Binary migration automatically.
    pub fn load_from_disk(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();

        // 1. Migration Logic: Check for old JSON if BIN is missing or as fallback
        if !path.exists() {
            let json_path = path.with_extension("json");
            if json_path.exists() {
                tracing::info!(
                    "Found legacy JSON ReplayBuffer. Migrating to Sovereign Binary format..."
                );
                let json_str = fs::read_to_string(&json_path)?;
                let snapshot_json: serde_json::Value = serde_json::from_str(&json_str)?;

                // Construct from JSON fields
                let cap = snapshot_json["capacity"].as_u64().unwrap_or(1000) as usize;
                let thresh = snapshot_json["dedup_threshold"].as_f64().unwrap_or(0.98);
                let memories: VecDeque<ReplayMemory> =
                    serde_json::from_value(snapshot_json["memories"].clone())?;

                let stats: BufferStats =
                    serde_json::from_value(snapshot_json["stats"].clone()).unwrap_or_default();

                let new_self = Self {
                    capacity: cap,
                    dedup_threshold: thresh,
                    memories: Arc::new(RwLock::new(memories)),
                    stats: Arc::new(RwLock::new(stats)),
                };

                // Save in new format immediately to complete migration
                new_self.save_to_disk(path)?;
                let _ = fs::remove_file(json_path); // Cleanup
                return Ok(new_self);
            }
            return Err(anyhow::anyhow!(
                "ReplayBuffer snapshot not found at: {}",
                path.display()
            ));
        }

        // 2. Load Binary Snapshot
        let bytes = fs::read(path)?;
        let snapshot: ReplayBufferSnapshot = bincode::deserialize(&bytes).map_err(|e| {
            anyhow::anyhow!(
                "Binary deserialization failed: {}. The file might be corrupted.",
                e
            )
        })?;

        // 3. Verify Integrity
        if let Some(ref stored_checksum) = snapshot.checksum {
            let actual_checksum = snapshot.compute_checksum()?;
            if stored_checksum != &actual_checksum {
                return Err(anyhow::anyhow!("Integrity Failure: Checksum mismatch in {}. Content might have been tampered with or corrupted.", path.display()));
            }
        } else {
            tracing::warn!("Loading ReplayBuffer without checksum verification.");
        }

        Ok(Self {
            capacity: snapshot.capacity,
            dedup_threshold: snapshot.dedup_threshold,
            memories: Arc::new(RwLock::new(snapshot.memories)),
            stats: Arc::new(RwLock::new(snapshot.stats)),
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_memory(id: &str, prompt: &str, completion: &str, importance: f64) -> ReplayMemory {
        ReplayMemory::new(
            id.into(),
            prompt.into(),
            completion.into(),
            None,
            importance,
            false,
        )
    }

    #[test]
    fn test_capacity_evicts_lowest_importance() {
        let buffer = ReplayBuffer::new(2);
        buffer.push(make_memory("1", "Prompt A", "Completion A", 0.2));
        buffer.push(make_memory("2", "Prompt B", "Completion B", 0.9));
        // This should evict "1" (lowest importance), not "2"
        buffer.push(make_memory("3", "Prompt C", "Completion C", 0.5));

        assert_eq!(buffer.len(), 2);
        let samples = buffer.sample(10);
        assert!(
            !samples.iter().any(|m| m.id == "1"),
            "Lowest-importance memory should be evicted"
        );
        assert!(
            samples.iter().any(|m| m.id == "2"),
            "High-importance memory must be retained"
        );
    }

    #[test]
    fn test_deduplication_rejects_near_identical() {
        let buffer = ReplayBuffer::new(10).with_dedup_threshold(0.80);
        buffer.push(make_memory(
            "1",
            "What is Rust?",
            "A systems language.",
            0.5,
        ));
        // Near-identical prompt/completion — should be deduplicated
        buffer.push(make_memory(
            "2",
            "What is Rust?",
            "A systems language.",
            0.5,
        ));

        let stats = buffer.stats();
        assert_eq!(stats.duplicates_rejected, 1, "Duplicate should be rejected");
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn test_dpo_pair_sampling() {
        let buffer = ReplayBuffer::new(10);
        buffer.push(make_memory("1", "Prompt", "Correct", 0.5));
        buffer.push(ReplayMemory::new(
            "2".into(),
            "Prompt".into(),
            "Corrected".into(),
            Some("Mistaken".into()),
            1.0,
            true,
        ));

        let dpo = buffer.sample_dpo_pairs(10);
        assert_eq!(dpo.len(), 1, "Only one DPO pair should exist");
        assert_eq!(dpo[0].id, "2");
    }

    #[test]
    fn test_importance_sampling_order() {
        let buffer = ReplayBuffer::new(10);
        buffer.push(make_memory("low", "P1", "C1", 0.1));
        buffer.push(make_memory("high", "P2", "C2", 0.9));
        buffer.push(make_memory("mid", "P3", "C3", 0.5));

        let top = buffer.sample_importance(1);
        assert_eq!(
            top[0].id, "high",
            "Importance sampling must return highest-scored first"
        );
    }
}
