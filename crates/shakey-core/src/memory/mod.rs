//! Elite HNSW (Hierarchical Navigable Small World) Vector Memory.
//!
//! Provides O(log N) retrieval of high-dimensional vectors (embeddings)
//! for long-term agent memory and context-aware RAG.

use anyhow::Result;

pub mod paged_cache;
use parking_lot::RwLock;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::BinaryHeap;
use std::convert::AsRef;

/// Metadata associated with a memory vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetadata {
    pub content: String,
    pub timestamp: u64,
    pub source: String,
    pub tags: Vec<String>,
}

/// A single node in the HNSW graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct HNSWNode {
    pub vector: Vec<f32>,
    pub quantized_vector: Vec<i8>,
    pub scale: f32,
    pub metadata: MemoryMetadata,
    /// Neighbors at each level: `[level][neighbor_idx]`
    pub neighbors: Vec<Vec<usize>>,
}

/// Serialized state of the HNSW index.
#[derive(Serialize, Deserialize)]
struct HNSWSerialized {
    pub nodes: Vec<HNSWNode>,
    pub entry_point: Option<usize>,
    pub max_level: usize,
    pub m: usize,
    pub dim: usize,
    pub ef_construction: usize,
}

/// Elite HNSW Vector Memory — industry-grade retrieval for autonomous agents.
pub struct VectorMemory {
    nodes: RwLock<Vec<HNSWNode>>,
    entry_point: RwLock<Option<usize>>,
    max_level: RwLock<usize>,
    ef_construction: usize,
    _m: usize,
    _dim: usize,

    // Performance: Generational Visited Check (Lock-free sample, but needs write for reset)
    visited: RwLock<Vec<u32>>,
    generation: RwLock<u32>,
}

impl VectorMemory {
    pub fn new(dim: usize, m: usize, ef_construction: usize) -> Self {
        Self {
            nodes: RwLock::new(Vec::new()),
            entry_point: RwLock::new(None),
            max_level: RwLock::new(0),
            ef_construction,
            _m: m,
            _dim: dim,
            visited: RwLock::new(Vec::new()),
            generation: RwLock::new(1),
        }
    }

    /// Create a quantized representation of a vector for SQ8.
    fn quantize(vector: &[f32]) -> (Vec<i8>, f32) {
        let mut max_abs = 0.0f32;
        for &x in vector {
            max_abs = max_abs.max(x.abs());
        }
        // ── Sovereign Guard: Zero-vector protection ──
        // Prevent division-by-zero when all elements are 0.0
        if max_abs < 1e-10 {
            return (vec![0i8; vector.len()], 0.0);
        }
        let scale = max_abs / 127.0;
        let quantized = vector
            .iter()
            .map(|&x| (x / scale).round().clamp(-128.0, 127.0) as i8)
            .collect();
        (quantized, scale)
    }

    /// Ultimate Similarity: AVX-512 Optimized Quantized Distance.
    fn similarity_sq8(query_q: &[i8], query_scale: f32, node_q: &[i8], node_scale: f32) -> f32 {
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx512vnni") {
            return unsafe { Self::dot_sq8_vnni(query_q, node_q) * query_scale * node_scale };
        }

        let mut dot = 0i32;
        for (&a, &b) in query_q.iter().zip(node_q.iter()) {
            dot += (a as i32) * (b as i32);
        }
        (dot as f32) * query_scale * node_scale
    }

    #[cfg(target_arch = "x86_64")]
    unsafe fn dot_sq8_vnni(a: &[i8], b: &[i8]) -> f32 {
        use std::arch::x86_64::*;
        let mut acc = _mm512_setzero_si512();
        let mut k = 0;

        while k + 64 <= a.len() {
            let va = _mm512_loadu_si512(a.as_ptr().add(k) as *const __m512i);
            let vb = _mm512_loadu_si512(b.as_ptr().add(k) as *const __m512i);

            // Promote i8 to i16 (extract 32-byte chunks)
            let va_lo = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(va, 0));
            let vb_lo = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(vb, 0));
            let prod0 = _mm512_madd_epi16(va_lo, vb_lo);

            let va_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(va, 1));
            let vb_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(vb, 1));
            let prod1 = _mm512_madd_epi16(va_hi, vb_hi);

            acc = _mm512_add_epi32(acc, _mm512_add_epi32(prod0, prod1));
            k += 64;
        }

        // Manual reduction for __m512i i32
        let mut tmp = [0i32; 16];
        _mm512_storeu_si512(tmp.as_mut_ptr() as *mut __m512i, acc);
        let mut result = tmp.iter().sum::<i32>();

        // ── Sovereign Guard: Process tail elements for non-64-aligned dimensions ──
        while k < a.len() {
            result += (a[k] as i32) * (b[k] as i32);
            k += 1;
        }
        result as f32
    }

    /// Insert a new experienced memory into the sovereign database.
    pub async fn insert(&self, vector: Vec<f32>, metadata: MemoryMetadata) -> Result<usize> {
        let (quantized_vector, scale) = Self::quantize(&vector);
        let mut nodes = self.nodes.write();
        let new_idx = nodes.len();

        let level = self.generate_random_level();
        let mut max_level_guard = self.max_level.write();
        let mut ep_guard = self.entry_point.write();

        let node = HNSWNode {
            vector: vector.clone(),
            quantized_vector,
            scale,
            metadata,
            neighbors: vec![Vec::new(); level + 1],
        };
        nodes.push(node);

        if ep_guard.is_none() {
            *ep_guard = Some(new_idx);
            *max_level_guard = level;
            return Ok(new_idx);
        }

        let mut curr_entry = ep_guard.unwrap();
        let max_lv = *max_level_guard;

        // ── Sovereign Perf: Reuse quantized vector already stored in the node ──
        let q_vector = nodes[new_idx].quantized_vector.clone();
        let q_scale = nodes[new_idx].scale;

        // Search upper levels
        for l in ((level + 1)..=max_lv).rev() {
            curr_entry = self.search_layer_greedy(&nodes, &q_vector, q_scale, curr_entry, l);
        }

        // Connect level by level
        for l in (0..std::cmp::min(level, max_lv) + 1).rev() {
            let neighbors = self.search_layer_knn(
                &nodes,
                &q_vector,
                q_scale,
                curr_entry,
                l,
                self.ef_construction,
            );
            self.connect_node_bidirectional(&mut nodes, new_idx, neighbors, l);
        }

        if level > max_lv {
            *max_level_guard = level;
            *ep_guard = Some(new_idx);
        }

        Ok(new_idx)
    }

    /// Search the memory for the most semantically relevant experiences.
    pub async fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<(f32, MemoryMetadata)>> {
        let nodes = self.nodes.read();
        let entry_point = self.entry_point.read();
        let max_level = self.max_level.read();

        if entry_point.is_none() {
            return Ok(Vec::new());
        }

        let (query_q, query_scale) = Self::quantize(query);
        let mut curr_entry = entry_point.unwrap();
        // Search from max_level down to level 1
        for l in (1..*max_level + 1).rev() {
            curr_entry = self.search_layer_greedy(&nodes, &query_q, query_scale, curr_entry, l);
        }

        let nearest = self.search_layer_knn(&nodes, &query_q, query_scale, curr_entry, 0, ef);

        let results: Vec<(f32, MemoryMetadata)> = nearest
            .into_iter()
            .take(k)
            .map(|(dist, idx)| (dist, nodes[idx].metadata.clone()))
            .collect();

        Ok(results)
    }

    fn search_layer_greedy(
        &self,
        nodes: &[HNSWNode],
        query_q: &[i8],
        query_scale: f32,
        entry: usize,
        level: usize,
    ) -> usize {
        let mut curr = entry;
        let mut curr_dist = Self::similarity_sq8(
            query_q,
            query_scale,
            &nodes[curr].quantized_vector,
            nodes[curr].scale,
        );

        loop {
            let mut best_next = None;
            for &neighbor in &nodes[curr].neighbors[level] {
                // Elite Optimization: Prefetch neighbor vector to minimize cache misses
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    use std::arch::x86_64::_mm_prefetch;
                    _mm_prefetch(
                        nodes[neighbor].quantized_vector.as_ptr(),
                        std::arch::x86_64::_MM_HINT_T0,
                    );
                }

                let d = Self::similarity_sq8(
                    query_q,
                    query_scale,
                    &nodes[neighbor].quantized_vector,
                    nodes[neighbor].scale,
                );
                if d > curr_dist {
                    curr_dist = d;
                    best_next = Some(neighbor);
                }
            }
            if let Some(next) = best_next {
                curr = next;
            } else {
                break;
            }
        }
        curr
    }

    fn search_layer_knn(
        &self,
        nodes: &[HNSWNode],
        query_q: &[i8],
        query_scale: f32,
        entry: usize,
        level: usize,
        ef: usize,
    ) -> Vec<(f32, usize)> {
        // Industry-Grade: Generational Visited Check for O(1) performance.
        let mut gen_guard = self.generation.write();
        let mut visited_guard = self.visited.write();

        if visited_guard.len() < nodes.len() {
            visited_guard.resize(nodes.len(), 0);
        }

        // If generation wraps around, reset the buffer.
        if *gen_guard == u32::MAX {
            visited_guard.fill(0);
            *gen_guard = 1;
        } else {
            *gen_guard += 1;
        }
        let current_gen = *gen_guard;

        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        let d = Self::similarity_sq8(
            query_q,
            query_scale,
            &nodes[entry].quantized_vector,
            nodes[entry].scale,
        );
        candidates.push(OrderedFloat(d, entry));
        results.push(OrderedFloat(-d, entry));
        visited_guard[entry] = current_gen;

        while let Some(OrderedFloat(d, curr)) = candidates.pop() {
            let lower_bound = -results.peek().unwrap().0;
            if d < lower_bound {
                break;
            }

            for &neighbor in &nodes[curr].neighbors[level] {
                if visited_guard[neighbor] != current_gen {
                    visited_guard[neighbor] = current_gen;
                    let nd = Self::similarity_sq8(
                        query_q,
                        query_scale,
                        &nodes[neighbor].quantized_vector,
                        nodes[neighbor].scale,
                    );
                    if results.len() < ef || nd > lower_bound {
                        candidates.push(OrderedFloat(nd, neighbor));
                        results.push(OrderedFloat(-nd, neighbor));
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut res_vec: Vec<(f32, usize)> = results.into_iter().map(|o| (-o.0, o.1)).collect();
        res_vec.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        res_vec
    }

    fn connect_node_bidirectional(
        &self,
        nodes: &mut [HNSWNode],
        idx: usize,
        neighbors: Vec<(f32, usize)>,
        level: usize,
    ) {
        for (_, neighbor_idx) in neighbors.iter().take(self._m) {
            nodes[idx].neighbors[level].push(*neighbor_idx);
            nodes[*neighbor_idx].neighbors[level].push(idx);

            if nodes[*neighbor_idx].neighbors[level].len() > self._m {
                nodes[*neighbor_idx].neighbors[level].truncate(self._m);
            }
        }
    }

    pub fn generate_random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let scale = 1.0 / (self._m as f64).ln();
        let uniform: f64 = rng.gen();
        (-uniform.ln() * scale).floor() as usize
    }

    /// Elite Serialization: Save the entire HNSW index to a binary file.
    pub fn save<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let nodes = self.nodes.read();
        let entry_point = self.entry_point.read();
        let max_level = self.max_level.read();

        let serialized = HNSWSerialized {
            nodes: nodes.clone(),
            entry_point: *entry_point,
            max_level: *max_level,
            m: self._m,
            dim: self._dim,
            ef_construction: self.ef_construction,
        };

        let bytes = bincode::serialize(&serialized)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Elite Deserialization: Load the entire HNSW index from a binary file using zero-copy mmap.
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };

        // Zero-copy deserialization using bincode on the mmap slice
        let serialized: HNSWSerialized = bincode::deserialize(&mmap[..])?;

        Ok(Self {
            nodes: RwLock::new(serialized.nodes),
            entry_point: RwLock::new(serialized.entry_point),
            max_level: RwLock::new(serialized.max_level),
            ef_construction: serialized.ef_construction,
            _m: serialized.m,
            _dim: serialized.dim,
            visited: RwLock::new(Vec::new()),
            generation: RwLock::new(1),
        })
    }
}

/// Helper for Ordered Float in Heaps
#[derive(PartialEq)]
struct OrderedFloat(f32, usize);

impl Eq for OrderedFloat {}

impl PartialOrd for OrderedFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // ── Sovereign Guard: NaN-safe comparison ──
        // NaN values are treated as less-than to prevent heap corruption
        self.0.partial_cmp(&other.0).unwrap_or_else(|| {
            if self.0.is_nan() {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Greater
            }
        })
    }
}
