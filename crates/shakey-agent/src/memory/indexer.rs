//! Elite Project Indexing Engine for deep codebase awareness.
//!
//! Features:
//! - Incremental Auditing: Uses SHA256 hashes to skip unchanged files.
//! - Semantic-Aware Chunking: Respects Rust's top-level symbol boundaries.
//! - Sliding Window: Overlapping context to ensure semantic continuity.
//! - Concurrent Indexing: Parallelized embedding generation via NIM.

use anyhow::Result;
use ignore::WalkBuilder;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing;

use super::knowledge_base::KnowledgeBase;
use shakey_distill::nim_client::NimClient;

/// Advanced codebase indexer with incremental sync and semantic awareness.
pub struct ProjectIndexer {
    /// NIM client for embedding generation
    nim: NimClient,
    /// Persistence layer for version tracking and RAG
    kb: Arc<KnowledgeBase>,
    /// Embedding model (resolved dynamically from configuration)
    model: String,
    /// Concurrency throttle to prevent API bursts
    semaphore: Arc<Semaphore>,
}

/// Constant for maximum chunk size in characters to take advantage of the 8k token limit.
/// 20,000 characters is approximately 6,000-7,000 tokens for code.
const MAX_CHUNK_CHARS: usize = 20000;

impl ProjectIndexer {
    pub fn new(nim: NimClient, kb: Arc<KnowledgeBase>) -> Self {
        // Resolve model at runtime or use a safe default for initialization
        // Note: index_project performs a compatibility audit which uses this model.
        let model = shakey_distill::teacher::TeacherRole::EmbeddingQuery
            .default_model()
            .to_string();
        Self {
            nim,
            kb,
            model,
            semaphore: Arc::new(Semaphore::new(8)), // Limit to 8 concurrent API calls
        }
    }

    /// Perform a full project audit and index all relevant files incrementally.
    pub async fn index_project(&self, root: impl AsRef<Path>) -> Result<()> {
        let root = root.as_ref();
        tracing::info!(target: "shakey::audit", "Starting Elite Codebase Audit in: {}", root.display());

        // 0. Compatibility Audit: Detect and handle architectural model-mismatch
        self.compatibility_audit()?;

        let mut files_to_process = Vec::new();

        // 1. Recursive scan with automatic .gitignore filtering
        let walker = WalkBuilder::new(root)
            .hidden(false) // Still index hidden files if they match relevant extensions
            .git_ignore(true)
            .git_global(true)
            .git_exclude(true)
            .build();

        for entry in walker
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|ft| ft.is_file()).unwrap_or(false))
        {
            let path = entry.path();
            if self.is_relevant_file(path) {
                let current_hash = self.calculate_hash(path)?;
                let path_str = path.to_string_lossy().to_string();

                // Check if file has changed
                if let Ok(Some(stored_hash)) = self.kb.get_file_hash(&path_str) {
                    if stored_hash == current_hash {
                        continue; // Skip unchanged file
                    }
                }
                files_to_process.push((path.to_path_buf(), current_hash));
            }
        }

        if files_to_process.is_empty() {
            tracing::info!(target: "shakey::audit", "Codebase is already fully synchronized and grounded.");
            return Ok(());
        }

        tracing::info!(target: "shakey::audit", "Processing {} changed/new files...", files_to_process.len());

        // 2. Parallel Processing with JoinSet
        let mut set = JoinSet::new();
        let self_arc = Arc::new(self.clone_lite()); // Need a way to share self

        for (path, hash) in files_to_process {
            let s = Arc::clone(&self_arc);
            let permit = Arc::clone(&self.semaphore).acquire_owned().await?;
            set.spawn(async move {
                let _permit = permit; // Keep permit alive until task finishes
                s.process_file(&path, &hash).await
            });
        }

        let mut total_chunks = 0;
        let mut processed_files = 0;

        while let Some(res) = set.join_next().await {
            match res {
                Ok(Ok(n)) => {
                    total_chunks += n;
                    processed_files += 1;
                }
                Ok(Err(e)) => tracing::warn!("Indexing error: {}", e),
                Err(e) => tracing::error!("Task join error: {}", e),
            }
        }

        tracing::info!(
            target: "shakey::audit",
            "Elite Audit Complete: Synchronized {} files into {} semantic memory segments.",
            processed_files,
            total_chunks
        );

        Ok(())
    }

    /// Process a single file: delete old segments, chunk, embed, and update hash.
    async fn process_file(&self, path: &Path, hash: &str) -> Result<usize> {
        let path_str = path.to_string_lossy();
        let content = fs::read_to_string(path)?;

        // 1. Clean up old segments for this file
        self.kb.vector_store.delete_file_segments(&path_str)?;

        // 2. Semantic Chunking
        let chunks = self.chunk_file(path, &content);
        if chunks.is_empty() {
            return Ok(0);
        }

        // 3. Global Deduplication Check
        let mut unique_chunks = Vec::new();
        let mut unique_texts = Vec::new();
        let mut deduplicated_count = 0;

        for mut chunk in chunks {
            // Context Inversion: Prepend Breadcrumbs to the raw text
            let breadcrumb = format!(
                "![SHAKEY_CONTEXT_INVERSION]\n![PATH: {}]\n![LOC: {}]\n\n",
                path_str, chunk.metadata
            );
            chunk.text = format!("{}{}", breadcrumb, chunk.text);

            let mut hasher = Sha256::new();
            hasher.update(chunk.text.as_bytes());
            let hash_id = hasher.finalize();
            let mut id = [0u8; 16];
            id.copy_from_slice(&hash_id[0..16]);

            // Track IDs for the file segment table even if we skip embedding it
            chunk.metadata = hex::encode(id); // Temporarily store ID in metadata

            match self.kb.vector_store.contains_chunk(&id) {
                Ok(true) => {
                    deduplicated_count += 1;
                    // Already exists globally! Just map this file path to the existing ID
                    let _ =
                        self.kb
                            .vector_store
                            .store(&id, &chunk.text, &[], Some(&path_str), true);
                }
                _ => {
                    unique_texts.push(chunk.text.clone());
                    unique_chunks.push(chunk);
                }
            }
        }

        if unique_chunks.is_empty() {
            self.kb.update_file_hash(&path_str, hash)?;
            return Ok(deduplicated_count); // All chunks existed globally
        }

        // 4. Batch Embedding with Industrial Exponential Backoff
        let mut vectors = vec![];
        let mut retries = 0;
        let max_retries = 5;
        let mut delay_ms = 1000;

        loop {
            let model = self
                .nim
                .resolve_model_for_role(&shakey_distill::teacher::TeacherRole::EmbeddingQuery)
                .await
                .unwrap_or_else(|_| self.model.clone());
            match self
                .nim
                .embeddings(unique_texts.clone(), &model, "passage", "memory")
                .await
            {
                Ok(v) => {
                    vectors = v;
                    break;
                }
                Err(_e) if retries < max_retries => {
                    retries += 1;
                    tracing::warn!(
                        "OODA: Embedding API Limit Hit. Backing off for {}ms (Attempt {}/{})",
                        delay_ms,
                        retries,
                        max_retries
                    );
                    tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                    delay_ms = (delay_ms * 2) + rand::random::<u64>() % 500; // Jitter
                }
                Err(e) => {
                    tracing::error!(
                        "OODA: Fatal embedding error after {} retries: {}",
                        max_retries,
                        e
                    );
                    return Err(e);
                }
            }
        }

        // 5. Persistence
        let chunk_count = unique_chunks.len() + deduplicated_count;
        for (chunk, vector) in unique_chunks.into_iter().zip(vectors) {
            let mut id = [0u8; 16];
            let decoded = hex::decode(&chunk.metadata).unwrap_or_default();
            for (i, &b) in decoded.iter().take(16).enumerate() {
                id[i] = b;
            }

            // Elite Persistence: Optimized batch storage
            let _ = self
                .kb
                .vector_store
                .store(&id, &chunk.text, &vector, Some(&path_str), true);
        }

        // 5. Update Version Hash
        self.kb.update_file_hash(&path_str, hash)?;

        Ok(chunk_count)
    }

    fn calculate_hash(&self, path: &Path) -> Result<String> {
        let content = fs::read(path)?;
        let mut hasher = Sha256::new();
        hasher.update(content);
        let hash_bytes = hasher.finalize();
        let hex_string: String = hash_bytes.iter().map(|b| format!("{:02x}", b)).collect();
        Ok(hex_string)
    }

    /// Filter for relevant codebase files.
    fn is_relevant_file(&self, path: &Path) -> bool {
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");

        // Only main source and configuration files.
        // Directory exclusions (target, node_modules, etc.) are handled by .gitignore via WalkBuilder.
        matches!(
            ext,
            "rs" | "yaml" | "toml" | "md" | "json" | "sh" | "js" | "py" | "c" | "cpp" | "h"
        )
    }

    /// Advanced chunker with semantic awareness and sliding window.
    fn chunk_file(&self, path: &Path, content: &str) -> Vec<CodeChunk> {
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        let rel_path = path.to_string_lossy().to_string();

        match ext {
            "rs" => self.chunk_rust(&rel_path, content),
            "yaml" | "toml" | "json" => self.chunk_config(&rel_path, content),
            _ => self.chunk_generic(&rel_path, content),
        }
    }

    /// Industrial-Grade Rust Chunker: Splits by top-level symbol (struct, fn, mod, impl).
    fn chunk_rust(&self, path: &str, content: &str) -> Vec<CodeChunk> {
        let mut chunks = Vec::new();
        let mut current_chunk = Vec::new();
        let mut line_num = 1;
        let mut chunk_start_line = 1;

        for line in content.lines() {
            let trimmed = line.trim();
            // Recognition: Treat /// doc comments as boundaries if tied to symbols
            let is_boundary = (trimmed.starts_with("///")
                || trimmed.starts_with("fn ")
                || trimmed.starts_with("pub fn ")
                || trimmed.starts_with("struct ")
                || trimmed.starts_with("pub struct ")
                || trimmed.starts_with("impl ")
                || trimmed.starts_with("mod ")
                || trimmed.starts_with("enum ")
                || trimmed.starts_with("pub enum "))
                && !current_chunk.is_empty()
                && (current_chunk.len() > 10);

            if is_boundary {
                let text = current_chunk.join("\n");
                self.push_recursive_chunks(&mut chunks, text, path, chunk_start_line, line_num - 1);

                // Sliding window: keep last 3 lines for context overlap
                let overlap: Vec<&str> =
                    current_chunk.iter().rev().take(3).rev().cloned().collect();
                current_chunk = overlap;
                chunk_start_line = line_num.saturating_sub(3).max(1);
            }

            current_chunk.push(line);
            line_num += 1;
        }

        if !current_chunk.is_empty() {
            let text = current_chunk.join("\n");
            self.push_recursive_chunks(&mut chunks, text, path, chunk_start_line, line_num - 1);
        }
        chunks
    }

    /// Recursively split a block of text into chunks that fit within [MAX_CHUNK_CHARS].
    fn push_recursive_chunks(
        &self,
        chunks: &mut Vec<CodeChunk>,
        text: String,
        path: &str,
        start_line: usize,
        end_line: usize,
    ) {
        if text.len() <= MAX_CHUNK_CHARS {
            chunks.push(CodeChunk {
                text,
                metadata: format!("{}:L{}-{}", path, start_line, end_line),
            });
            return;
        }

        // Block too large! Perform binary recursive split by line
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() <= 1 {
            // Single massive line (e.g. minified JS or giant string). Hard truncate.
            chunks.push(CodeChunk {
                text: text.chars().take(MAX_CHUNK_CHARS).collect(),
                metadata: format!("{}:L{}-{}(truncated)", path, start_line, end_line),
            });
            return;
        }

        let mid = lines.len() / 2;
        let left = lines[..mid].join("\n");
        let right = lines[mid..].join("\n");

        self.push_recursive_chunks(chunks, left, path, start_line, start_line + mid - 1);
        self.push_recursive_chunks(chunks, right, path, start_line + mid, end_line);
    }

    fn chunk_config(&self, path: &str, content: &str) -> Vec<CodeChunk> {
        // Configs are usually key-value blocks. We split by groups of ~30 lines.
        self.chunk_generic(path, content)
    }

    fn chunk_generic(&self, path: &str, content: &str) -> Vec<CodeChunk> {
        let mut chunks = Vec::new();
        let lines: Vec<&str> = content.lines().collect();

        // Use a smaller block size initially to reduce recursion
        for (i, block) in lines.chunks(30).enumerate() {
            let text = block.join("\n");
            let start_line = i * 30 + 1;
            let end_line = start_line + block.len() - 1;
            self.push_recursive_chunks(&mut chunks, text, path, start_line, end_line);
        }
        chunks
    }

    /// Minimal clone for task spawning.
    fn clone_lite(&self) -> Self {
        Self {
            nim: self.nim.clone(),
            kb: Arc::clone(&self.kb),
            model: self.model.clone(),
            semaphore: Arc::clone(&self.semaphore),
        }
    }

    /// Industrial Robustness: Detects model upgrades and performs surgical memory refreshes.
    fn compatibility_audit(&self) -> Result<()> {
        let stored_model = self.kb.get_fact("active_embedding_model")?;

        if let Some(old_model) = stored_model {
            if old_model != self.model {
                tracing::warn!(target: "shakey::audit",
                    "Elite Migration: Model mismatch detected (Old: {}, New: {}). Initializing Indestructible Re-Index...",
                    old_model, self.model
                );

                // Surgical refresh: Wipes embeddings and hashes but preserves facts/identity
                self.kb.clear_embeddings()?;
                self.kb.clear_file_hashes()?;

                // Update identity with new model signature
                self.kb.store_fact("active_embedding_model", &self.model)?;
            }
        } else {
            // First run: establish model identity
            tracing::info!(target: "shakey::audit", "Elite Setup: Initializing Sovereign Memory for model {}...", self.model);
            self.kb.store_fact("active_embedding_model", &self.model)?;
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
struct CodeChunk {
    text: String,
    metadata: String,
}
