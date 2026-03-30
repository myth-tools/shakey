//! Paged Cache Memory Management (vLLM style PagedAttention Foundations)
//!
//! This module implements the foundational structures for Continuous Batching
//! by managing GPU/CPU memory via a Block Allocator. Instead of allocating
//! contiguous memory for an entire theoretical sequence length, memory is
//! allocated dynamically in fixed-size blocks (e.g. 16 or 32 tokens).
//!
//! ## Architecture
//! - **BlockAllocator**: Manages a pool of free physical memory blocks.
//! - **PageTable**: Maps logical sequence tokens to physical memory blocks.
//! - **PagedKVCache**: The underlying tensor layout representing the blocks.

use candle_core::{DType, Device, Result, Tensor};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, VecDeque};

const BLOCK_SIZE: usize = 16; // Standard industry block size for PagedAttention

/// Represents physical memory blocks allocated for a specific sequence.
#[derive(Debug, Clone)]
pub struct PageTable {
    /// Logical block index to Physical block index mapping
    blocks: Vec<usize>,
    /// Number of tokens currently valid in the final block
    #[allow(dead_code)]
    last_block_len: usize,
}

impl Default for PageTable {
    fn default() -> Self {
        Self::new()
    }
}

impl PageTable {
    pub fn new() -> Self {
        Self {
            blocks: Vec::new(),
            last_block_len: 0,
        }
    }

    /// Add a physical block to this sequence.
    pub fn append_block(&mut self, physical_block_id: usize) {
        self.blocks.push(physical_block_id);
    }
}

/// Central Memory Manager that allocates and frees KV-cache blocks.
#[derive(Debug)]
pub struct BlockAllocator {
    num_blocks: usize,
    free_blocks: VecDeque<usize>,
    /// O(1) Membership Check: True if block is currently in the free_blocks pool.
    is_free: Vec<bool>,
    /// Prefix Cache: Maps token sequence hashes to block list.
    prefix_cache: HashMap<String, Vec<usize>>,
}

impl BlockAllocator {
    /// Initialize a new allocator with a maximum capacity of blocks
    pub fn new(num_blocks: usize) -> Self {
        let mut free_blocks = VecDeque::with_capacity(num_blocks);
        for i in 0..num_blocks {
            free_blocks.push_back(i);
        }
        Self {
            num_blocks,
            free_blocks,
            is_free: vec![true; num_blocks],
            prefix_cache: HashMap::new(),
        }
    }

    /// ── Peak Mastery: Prefix Hashing ──
    /// Compute a stable hash for a sequence of tokens to identify it in the cache.
    pub fn compute_prefix_hash(tokens: &[u32]) -> String {
        let mut hasher = Sha256::new();
        for &t in tokens {
            hasher.update(t.to_le_bytes());
        }
        hex::encode(hasher.finalize())
    }

    /// Try to retrieve a physical block list for a given token prefix.
    pub fn get_prefix_blocks(&self, tokens: &[u32]) -> Option<Vec<usize>> {
        let hash = Self::compute_prefix_hash(tokens);
        self.prefix_cache.get(&hash).cloned()
    }

    /// Store a sequence of blocks in the prefix cache.
    pub fn cache_prefix(&mut self, tokens: &[u32], blocks: Vec<usize>) {
        if tokens.is_empty() || blocks.is_empty() {
            return;
        }
        let hash = Self::compute_prefix_hash(tokens);
        self.prefix_cache.insert(hash, blocks);
    }

    /// Allocate a new physical block if available.
    pub fn allocate(&mut self) -> Option<usize> {
        if let Some(block_id) = self.free_blocks.pop_front() {
            self.is_free[block_id] = false;
            Some(block_id)
        } else {
            None
        }
    }

    /// Free a physical block back to the pool.
    pub fn free(&mut self, block_id: usize) {
        // Industry-Grade: O(1) membership check using bit-mask.
        if block_id < self.num_blocks && !self.is_free[block_id] {
            self.free_blocks.push_back(block_id);
            self.is_free[block_id] = true;
        }
    }

    /// Check remaining free blocks.
    pub fn available_blocks(&self) -> usize {
        self.free_blocks.len()
    }
}

/// Physical memory layout for Paged KV Cache.
///
/// In standard attention, KV is `[batch, n_kv_heads, max_seq, head_dim]`.
/// In PagedAttention, KV is `[num_blocks, n_kv_heads, BLOCK_SIZE, head_dim]`.
#[derive(Debug)]
pub struct PagedKVCache {
    pub k_cache: Tensor,
    pub v_cache: Tensor,
    pub block_size: usize,
}

impl PagedKVCache {
    /// Allocate the physical tensors for the Paged Cache on device.
    pub fn new(
        num_blocks: usize,
        n_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        // Elite Note: Using F16 for memory bandwidth optimization
        let k_cache = Tensor::zeros(
            (num_blocks, n_kv_heads, BLOCK_SIZE, head_dim),
            dtype,
            device,
        )?;
        let v_cache = Tensor::zeros(
            (num_blocks, n_kv_heads, BLOCK_SIZE, head_dim),
            dtype,
            device,
        )?;

        Ok(Self {
            k_cache,
            v_cache,
            block_size: BLOCK_SIZE,
        })
    }
}
