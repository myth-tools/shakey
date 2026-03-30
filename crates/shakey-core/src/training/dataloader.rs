use crate::tokenizer::Tokenizer;
use anyhow::Result;
use candle_core::{Device, Tensor};
use serde::Deserialize;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

/// A fast Bloom Filter for online semantic deduplication.
pub struct BloomFilter {
    bits: Vec<u64>,
    num_hashes: usize,
    capacity: usize,
}

impl BloomFilter {
    pub fn new(capacity: usize, num_hashes: usize) -> Self {
        let size = capacity.div_ceil(64);
        Self {
            bits: vec![0; size],
            num_hashes,
            capacity,
        }
    }

    pub fn insert(&mut self, item: &str) {
        for i in 0..self.num_hashes {
            let h = self.hash(item, i);
            let idx = h % self.capacity;
            self.bits[idx / 64] |= 1 << (idx % 64);
        }
    }

    pub fn contains(&self, item: &str) -> bool {
        for i in 0..self.num_hashes {
            let h = self.hash(item, i);
            let idx = h % self.capacity;
            if (self.bits[idx / 64] & (1 << (idx % 64))) == 0 {
                return false;
            }
        }
        true
    }

    fn hash(&self, item: &str, seed: usize) -> usize {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        seed.hash(&mut hasher);
        hasher.finish() as usize
    }
}

#[derive(Deserialize)]
struct TrainingExample {
    prompt: String,
    completion: String,
}

/// A streaming batch iterator for real training data (JSONL format).
pub struct StreamingDataLoader {
    batch_size: usize,
    max_seq_len: usize,
    device: Device,
    files: Vec<PathBuf>,
    tokenizer: Tokenizer,
    pub deduplicator: std::sync::Arc<std::sync::Mutex<BloomFilter>>,
}

impl StreamingDataLoader {
    pub fn new(
        data_dir: impl AsRef<Path>,
        batch_size: usize,
        max_seq_len: usize,
        device: Device,
        tokenizer: Tokenizer,
    ) -> Result<Self> {
        let data_dir = data_dir.as_ref().to_path_buf();
        let mut files = Vec::new();

        if data_dir.exists() {
            for entry in std::fs::read_dir(&data_dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "jsonl") {
                    files.push(path);
                }
            }
        }

        let deduplicator =
            std::sync::Arc::new(std::sync::Mutex::new(BloomFilter::new(1_000_000, 7)));

        Ok(Self {
            batch_size,
            max_seq_len,
            device,
            files,
            tokenizer,
            deduplicator,
        })
    }

    /// Iterator over (input_ids, target_ids).
    pub fn iter(&self) -> DataLoaderIter<'_> {
        DataLoaderIter {
            loader: self,
            current_file_idx: 0,
            current_lines: None,
        }
    }

    pub fn file_count(&self) -> usize {
        self.files.len()
    }
}

pub struct DataLoaderIter<'a> {
    loader: &'a StreamingDataLoader,
    current_file_idx: usize,
    current_lines: Option<std::vec::IntoIter<String>>,
}

impl<'a> Iterator for DataLoaderIter<'a> {
    type Item = Result<(Tensor, Tensor)>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.loader.files.is_empty() {
            return None;
        }

        loop {
            if let Some(ref mut lines) = self.current_lines {
                let mut batch_inputs = Vec::with_capacity(self.loader.batch_size);
                let mut batch_targets = Vec::with_capacity(self.loader.batch_size);

                for _ in 0..self.loader.batch_size {
                    if let Some(line) = lines.next() {
                        // Parse JSONL to TrainingExample
                        let example: TrainingExample = match serde_json::from_str(&line) {
                            Ok(ex) => ex,
                            Err(_) => continue, // Skip malformed lines
                        };

                        // ── Peak Mastery: Online Semantic Deduplication ──
                        // Check if we've already trained on this specific prompt/completion pair
                        let digest = format!("{}{}", example.prompt, example.completion);
                        {
                            let mut filter = self.loader.deduplicator.lock().unwrap();
                            if filter.contains(&digest) {
                                continue;
                            }
                            filter.insert(&digest);
                        }

                        // Combine prompt and completion for training
                        let full_text = format!("{} {}", example.prompt, example.completion);
                        let mut tokens = match self.loader.tokenizer.encode(&full_text) {
                            Ok(t) => t,
                            Err(_) => continue,
                        };

                        // Truncate or pad to max_seq_len
                        if tokens.len() > self.loader.max_seq_len {
                            tokens.truncate(self.loader.max_seq_len);
                        }

                        // Targets are the same tokens, but shifted by 1 relative to inputs
                        // In causal LM: at pos i, we predict token at pos i+1
                        let mut input_ids = tokens.clone();
                        let mut target_ids = tokens.clone();

                        // Shift targets for causal prediction
                        if target_ids.len() > 1 {
                            target_ids.remove(0);
                            target_ids.push(self.loader.tokenizer.special_tokens().pad_id);
                        }

                        // Pad up to max_seq_len
                        while input_ids.len() < self.loader.max_seq_len {
                            input_ids.push(self.loader.tokenizer.special_tokens().pad_id);
                            target_ids.push(self.loader.tokenizer.special_tokens().pad_id);
                        }

                        batch_inputs.push(input_ids);
                        batch_targets.push(target_ids);
                    } else {
                        break;
                    }
                }

                if !batch_inputs.is_empty() {
                    // Convert to Tensors [batch, seq_len]
                    let batch_len = batch_inputs.len();
                    let flat_inputs: Vec<u32> = batch_inputs.into_iter().flatten().collect();
                    let flat_targets: Vec<u32> = batch_targets.into_iter().flatten().collect();

                    let inputs = match Tensor::from_vec(
                        flat_inputs,
                        (batch_len, self.loader.max_seq_len),
                        &self.loader.device,
                    ) {
                        Ok(t) => t,
                        Err(e) => {
                            return Some(Err(anyhow::anyhow!("Tensor creation failed: {}", e)))
                        }
                    };
                    let targets = match Tensor::from_vec(
                        flat_targets,
                        (batch_len, self.loader.max_seq_len),
                        &self.loader.device,
                    ) {
                        Ok(t) => t,
                        Err(e) => {
                            return Some(Err(anyhow::anyhow!("Tensor creation failed: {}", e)))
                        }
                    };

                    return Some(Ok((inputs, targets)));
                } else {
                    self.current_lines = None;
                }
            }

            if self.current_file_idx >= self.loader.files.len() {
                return None;
            }

            // Load next file
            let file_path = &self.loader.files[self.current_file_idx];
            self.current_file_idx += 1;

            match std::fs::read_to_string(file_path) {
                Ok(content) => {
                    let lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();
                    self.current_lines = Some(lines.into_iter());
                }
                Err(e) => {
                    return Some(Err(anyhow::anyhow!(
                        "Failed to read {}: {}",
                        file_path.display(),
                        e
                    )))
                }
            }
        }
    }
}
