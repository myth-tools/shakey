//! Tokenizer module.
//!
//! Wraps the HuggingFace `tokenizers` library for BPE tokenization.
//! Falls back to a simple byte-level tokenizer if no trained tokenizer
//! is available (for bootstrapping).

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Special token IDs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokens {
    pub bos_id: u32, // Beginning of sequence
    pub eos_id: u32, // End of sequence
    pub pad_id: u32, // Padding
    pub unk_id: u32, // Unknown token
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos_id: 1,
            eos_id: 2,
            pad_id: 0,
            unk_id: 3,
        }
    }
}

/// Tokenizer abstraction that supports both trained BPE and byte-level fallback.
pub enum Tokenizer {
    /// HuggingFace tokenizers BPE (loaded from file)
    Bpe {
        inner: Box<tokenizers::Tokenizer>,
        special_tokens: SpecialTokens,
    },
    /// Simple byte-level tokenizer (256 tokens, no merges)
    /// Used for bootstrapping before a proper tokenizer is trained
    ByteLevel {
        vocab_size: usize,
        special_tokens: SpecialTokens,
    },
}

impl Tokenizer {
    /// Load a trained tokenizer from a JSON file and dynamically detect special tokens.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|e| {
            anyhow::anyhow!("Failed to load tokenizer from {}: {}", path.display(), e)
        })?;

        // ── Robust Tokenizer Calibration ──
        // Attempt to dynamically resolve special token IDs from the model's vocabulary.
        // Fallback to defaults only if standard names are missing.
        let bos_id = inner
            .token_to_id("<s>")
            .or_else(|| inner.token_to_id("<|begin_of_text|>"))
            .unwrap_or(1);
        let eos_id = inner
            .token_to_id("</s>")
            .or_else(|| inner.token_to_id("<|end_of_text|>"))
            .unwrap_or(2);
        let pad_id = inner
            .token_to_id("<pad>")
            .or_else(|| inner.token_to_id("<|pad|>"))
            .unwrap_or(0);
        let unk_id = inner
            .token_to_id("<unk>")
            .or_else(|| inner.token_to_id("<|unk|>"))
            .unwrap_or(3);

        Ok(Self::Bpe {
            inner: Box::new(inner),
            special_tokens: SpecialTokens {
                bos_id,
                eos_id,
                pad_id,
                unk_id,
            },
        })
    }

    /// Create a simple byte-level tokenizer for bootstrapping.
    pub fn byte_level() -> Self {
        Self::ByteLevel {
            vocab_size: 260,
            special_tokens: SpecialTokens {
                bos_id: 256,
                eos_id: 257,
                pad_id: 258,
                unk_id: 259,
            },
        }
    }

    /// Encode text to token IDs with robust padding and truncation.
    pub fn encode_robust(
        &self,
        text: &str,
        max_len: Option<usize>,
        add_special: bool,
    ) -> Result<Vec<u32>> {
        match self {
            Self::Bpe {
                inner,
                special_tokens,
            } => {
                let encoding = inner
                    .encode(text, add_special)
                    .map_err(|e| anyhow::anyhow!("Tokenization error: {}", e))?;
                let mut ids = encoding.get_ids().to_vec();

                if let Some(limit) = max_len {
                    if ids.len() > limit {
                        ids.truncate(limit);
                    } else {
                        while ids.len() < limit {
                            ids.push(special_tokens.pad_id);
                        }
                    }
                }
                Ok(ids)
            }
            Self::ByteLevel { special_tokens, .. } => {
                let mut ids = if add_special {
                    vec![special_tokens.bos_id]
                } else {
                    vec![]
                };
                ids.extend(text.bytes().map(|b| b as u32));
                if add_special {
                    ids.push(special_tokens.eos_id);
                }

                if let Some(limit) = max_len {
                    if ids.len() > limit {
                        ids.truncate(limit);
                    } else {
                        while ids.len() < limit {
                            ids.push(special_tokens.pad_id);
                        }
                    }
                }
                Ok(ids)
            }
        }
    }

    /// Encode text to token IDs (Compatible with previous version).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.encode_robust(text, None, true)
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        match self {
            Self::Bpe {
                inner,
                special_tokens,
            } => {
                // Filter out special tokens for clean text output
                let filtered: Vec<u32> = ids
                    .iter()
                    .copied()
                    .filter(|&id| {
                        id != special_tokens.bos_id
                            && id != special_tokens.eos_id
                            && id != special_tokens.pad_id
                    })
                    .collect();
                inner
                    .decode(&filtered, true)
                    .map_err(|e| anyhow::anyhow!("Decode error: {}", e))
            }
            Self::ByteLevel { special_tokens, .. } => {
                let bytes: Vec<u8> = ids
                    .iter()
                    .filter(|&&id| {
                        id != special_tokens.bos_id
                            && id != special_tokens.eos_id
                            && id != special_tokens.pad_id
                            && id < 256
                    })
                    .map(|&id| id as u8)
                    .collect();
                Ok(String::from_utf8_lossy(&bytes).into_owned())
            }
        }
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        match self {
            Self::Bpe { inner, .. } => inner.get_vocab_size(true),
            Self::ByteLevel { vocab_size, .. } => *vocab_size,
        }
    }

    /// Get special token IDs.
    pub fn special_tokens(&self) -> &SpecialTokens {
        match self {
            Self::Bpe { special_tokens, .. } => special_tokens,
            Self::ByteLevel { special_tokens, .. } => special_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_level_tokenizer() {
        let tok = Tokenizer::byte_level();
        let ids = tok.encode("Hello").unwrap();
        // Should be: [BOS, H, e, l, l, o, EOS]
        assert_eq!(ids.len(), 7);
        assert_eq!(ids[0], 256); // BOS
        assert_eq!(ids[1], b'H' as u32);
        assert_eq!(ids[6], 257); // EOS

        let decoded = tok.decode(&ids).unwrap();
        assert_eq!(decoded, "Hello");
    }

    #[test]
    fn test_vocab_size() {
        let tok = Tokenizer::byte_level();
        assert_eq!(tok.vocab_size(), 260);
    }
}
