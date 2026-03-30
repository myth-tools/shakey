use sha2::{Digest, Sha256};
use std::hash::Hash;

/// A high-performance Bloom Filter for ultra-fast data deduplication.
///
/// Uses SHA-256 and DefaultHasher to provide multiple independent hash functions.
pub struct BloomFilter {
    bit_vec: Vec<u8>,
    size_bits: usize,
    num_hashes: usize,
}

impl BloomFilter {
    /// Create a new Bloom Filter with specified size (in bits) and number of hash functions.
    pub fn new(size_bits: usize, num_hashes: usize) -> Self {
        let size_bytes = size_bits.div_ceil(8);
        Self {
            bit_vec: vec![0; size_bytes],
            size_bits,
            num_hashes,
        }
    }

    /// Add an item to the filter.
    pub fn insert<T: AsRef<[u8]> + Hash>(&mut self, item: T) {
        for i in 0..self.num_hashes {
            let idx = self.hash(&item, i) % self.size_bits;
            let byte_idx = idx / 8;
            let bit_idx = idx % 8;
            self.bit_vec[byte_idx] |= 1 << bit_idx;
        }
    }

    /// Check if an item is likely in the filter.
    pub fn contains<T: AsRef<[u8]> + Hash>(&self, item: T) -> bool {
        for i in 0..self.num_hashes {
            let idx = self.hash(&item, i) % self.size_bits;
            let byte_idx = idx / 8;
            let bit_idx = idx % 8;
            if (self.bit_vec[byte_idx] & (1 << bit_idx)) == 0 {
                return false;
            }
        }
        true
    }

    /// Internal hashing: Combines SHA-256 with an index to produce independent hashes.
    fn hash<T: AsRef<[u8]> + Hash>(&self, item: &T, index: usize) -> usize {
        let mut hasher = Sha256::new();
        hasher.update(item.as_ref());
        hasher.update((index as u64).to_le_bytes());
        let result = hasher.finalize();

        // Use the first 8 bytes for the index
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&result[0..8]);
        u64::from_le_bytes(bytes) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_filter_basic() {
        let mut filter = BloomFilter::new(1024, 3);
        let item = "https://example.com/unique-data-point";

        assert!(!filter.contains(item));
        filter.insert(item);
        assert!(filter.contains(item));
        assert!(!filter.contains("something-else"));
    }
}
