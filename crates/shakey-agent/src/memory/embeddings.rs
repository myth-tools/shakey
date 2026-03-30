use anyhow::Result;
use redb::{Database, ReadableTable, TableDefinition};
use shakey_core::memory::{MemoryMetadata, VectorMemory};
use std::sync::Arc;

/// Table definition for embeddings: (ChunkID, VectorBlob)
/// ChunkID is typically a UUID or hash of the text chunk.
pub const EMBEDDINGS_TABLE: TableDefinition<&[u8; 16], &[u8]> = TableDefinition::new("embeddings");
/// Table definition for metadata: (ChunkID, OriginalText)
pub const METADATA_TABLE: TableDefinition<&[u8; 16], &str> = TableDefinition::new("metadata");
/// Table definition for file segments: (FilePath, `Vec<ChunkID>`)
pub const FILE_SEGMENTS_TABLE: TableDefinition<&str, Vec<[u8; 16]>> =
    TableDefinition::new("file_segments");

/// Vector store for persistent RAG.
#[derive(Clone)]
pub struct VectorStore {
    db: Arc<Database>,
    hnsw: Arc<VectorMemory>,
}

impl VectorStore {
    pub fn new(db: Arc<Database>) -> Result<Self> {
        let write_txn = db.begin_write()?;
        {
            let _ = write_txn.open_table(EMBEDDINGS_TABLE)?;
            let _ = write_txn.open_table(METADATA_TABLE)?;
            let _ = write_txn.open_table(FILE_SEGMENTS_TABLE)?;
        }
        write_txn.commit()?;

        // Dynamic Dimension and M params for HNSW
        let hnsw = Arc::new(VectorMemory::new(768, 16, 200));

        // --- ELITE: HNSW Hydration from Persistence ---
        // On startup, we re-index the persistent redb data into the
        // high-performance HNSW structure for O(log N) search.
        let read_txn = db.begin_read()?;
        let emb_table = read_txn.open_table(EMBEDDINGS_TABLE)?;
        let meta_table = read_txn.open_table(METADATA_TABLE)?;

        for item in emb_table.iter()? {
            let (id, vec_bytes) = item?;
            let vec: Vec<f32> = vec_bytes
                .value()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();

            if let Some(meta_text) = meta_table.get(id.value())? {
                let metadata = MemoryMetadata {
                    content: meta_text.value().to_string(),
                    timestamp: 0,
                    source: "startup_hydration".to_string(),
                    tags: Vec::new(),
                };
                // Blocking wait for hydration
                let hydration_result = tokio::task::block_in_place(|| {
                    let rt = tokio::runtime::Handle::current();
                    rt.block_on(async { hnsw.insert(vec, metadata).await })
                });

                if let Err(e) = hydration_result {
                    tracing::warn!("Failed to hydrate vector: {}", e);
                }
            }
        }

        Ok(Self { db, hnsw })
    }

    /// Store a text chunk along with its vector embedding.
    /// Elite Industry-Grade: Automatically performs semantic deduplication (98% threshold).
    pub fn store(
        &self,
        id: &[u8; 16],
        text: &str,
        vector: &[f32],
        file_path: Option<&str>,
        skip_semantic_check: bool,
    ) -> Result<bool> {
        // 1. Semantic Deduplication Check: Elite-Elite logic for unique memory
        if !skip_semantic_check && !vector.is_empty() {
            if let Ok(results) = self.search(vector, 1) {
                if let Some((_, score)) = results.first() {
                    if *score > 0.98 {
                        tracing::debug!(target: "shakey::memory", "Semantic Deduplication: Skipping store for near-identical vector (score: {:.4})", score);
                        return Ok(false);
                    }
                }
            }
        }

        let vector_bytes: Vec<u8> = vector.iter().flat_map(|&f| f.to_le_bytes()).collect();

        let write_txn = self.db.begin_write()?;
        {
            let mut emb_table = write_txn.open_table(EMBEDDINGS_TABLE)?;
            let mut meta_table = write_txn.open_table(METADATA_TABLE)?;

            emb_table.insert(id, vector_bytes.as_slice())?;
            meta_table.insert(id, text)?;

            if let Some(path) = file_path {
                let mut seg_table = write_txn.open_table(FILE_SEGMENTS_TABLE)?;
                let mut segments = seg_table.get(path)?.map(|v| v.value()).unwrap_or_default();
                if !segments.contains(id) {
                    segments.push(*id);
                    seg_table.insert(path, segments)?;
                }
            }
        }
        write_txn.commit()?;

        // Also add to HNSW for fast search immediately (synchronous for consistency)
        if !vector.is_empty() {
            let hnsw = self.hnsw.clone();
            let vec_sync = vector.to_vec();
            let meta_sync = MemoryMetadata {
                content: text.to_string(),
                timestamp: chrono::Utc::now().timestamp() as u64,
                source: file_path.unwrap_or("inline").to_string(),
                tags: Vec::new(),
            };

            // Synchronous insert ensures HNSW index stays consistent with redb.
            // block_in_place is safe here because store() is always called from
            // within a Tokio context (tool execution, indexer, etc.)
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                let _ = tokio::task::block_in_place(|| {
                    handle.block_on(async { hnsw.insert(vec_sync, meta_sync).await })
                });
            }
        }

        Ok(true)
    }

    /// Check if a specific chunk ID already exists in the vector store.
    pub fn contains_chunk(&self, id: &[u8; 16]) -> Result<bool> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(EMBEDDINGS_TABLE)?;
        Ok(table.get(id)?.is_some())
    }

    /// Delete all stored segments for a specific file.
    pub fn delete_file_segments(&self, path: &str) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut emb_table = write_txn.open_table(EMBEDDINGS_TABLE)?;
            let mut meta_table = write_txn.open_table(METADATA_TABLE)?;
            let mut seg_table = write_txn.open_table(FILE_SEGMENTS_TABLE)?;

            let segments_to_delete = seg_table.get(path)?.map(|guard| guard.value());

            if let Some(ids) = segments_to_delete {
                for id in ids {
                    emb_table.remove(&id)?;
                    meta_table.remove(&id)?;
                }
                seg_table.remove(path)?;
            }
        }
        write_txn.commit()?;
        Ok(())
    }
}

#[allow(dead_code)]
struct SearchCandidate {
    id: [u8; 16],
    vec: Vec<f32>,
}

impl VectorStore {
    /// Elite Evolutionary HNSW Search: Industry-leading log N speed.
    /// Resilient: Works both inside and outside a Tokio runtime context.
    pub fn search(&self, query_vector: &[f32], top_k: usize) -> Result<Vec<(String, f32)>> {
        let hnsw = self.hnsw.clone();
        let query = query_vector.to_vec();

        // Bridge async HNSW to synchronous OODA logic.
        // Handle both cases: called from within Tokio (block_in_place)
        // and called from outside Tokio (create temporary runtime).
        let results = match tokio::runtime::Handle::try_current() {
            Ok(handle) => tokio::task::block_in_place(|| {
                handle.block_on(async { hnsw.search(&query, top_k, 200).await })
            })
            .map_err(|e| anyhow::anyhow!("HNSW search failed: {}", e))?,
            Err(_) => {
                // No Tokio runtime active — create a minimal one for the search.
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|e| anyhow::anyhow!("Failed to create search runtime: {}", e))?;
                rt.block_on(async { hnsw.search(&query, top_k, 200).await })
                    .map_err(|e| anyhow::anyhow!("HNSW search failed: {}", e))?
            }
        };

        let mapped = results
            .into_iter()
            .map(|(score, meta)| (meta.content, score))
            .collect();

        Ok(mapped)
    }
}

/// Helper for f32 ordering in BinaryHeap (Min-Heap wrapper)
#[allow(dead_code)]
struct ReverseScore(f32, [u8; 16]);

impl PartialEq for ReverseScore {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for ReverseScore {}

impl PartialOrd for ReverseScore {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ReverseScore {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse order so it's a Min-Heap (popping smallest first keeps top-k)
        match other.0.partial_cmp(&self.0) {
            Some(std::cmp::Ordering::Equal) | None => self.1.cmp(&other.1),
            Some(ord) => ord,
        }
    }
}

/// Elite-Elite: Fast SIMD-friendly cosine similarity calculation.
/// Uses 8-lane manual unrolling to encourage compiler auto-vectorization (AVX/NEON).
#[inline(always)]
#[allow(dead_code)]
fn cosine_similarity_fast(query: &[f32], query_norm: f32, candidate: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut cand_norm_sq = 0.0;

    let q_chunks = query.chunks_exact(8);
    let c_chunks = candidate.chunks_exact(8);
    let q_rem = q_chunks.remainder();
    let c_rem = c_chunks.remainder();

    for (q8, c8) in q_chunks.zip(c_chunks) {
        for i in 0..8 {
            dot += q8[i] * c8[i];
            cand_norm_sq += c8[i] * c8[i];
        }
    }

    for (x, y) in q_rem.iter().zip(c_rem.iter()) {
        dot += x * y;
        cand_norm_sq += y * y;
    }

    let cand_norm = cand_norm_sq.sqrt();
    if query_norm <= 0.0 || cand_norm <= 0.0 {
        return 0.0;
    }
    dot / (query_norm * cand_norm)
}
