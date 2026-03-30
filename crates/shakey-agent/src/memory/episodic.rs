//! Episodic memory — records of past cycles and their outcomes.
//!
//! Based on `redb` for persistent storage.
//! Allows the agent to learn from its own history.

use crate::CycleRecord;
use anyhow::Result;
use redb::{Database, ReadableTable, TableDefinition};
use std::sync::Arc;

/// Table definition for episodic memory (JSON serialized)
const EPISODIC_TABLE: TableDefinition<&str, &str> = TableDefinition::new("episodic_memory");
/// Table definition for metadata (e.g. schema version)
const METADATA_TABLE: TableDefinition<&str, u64> = TableDefinition::new("episodic_metadata");

const SCHEMA_VERSION: u64 = 1;

pub struct EpisodicMemory {
    db: Arc<Database>,
}

impl EpisodicMemory {
    pub fn new(db: Arc<Database>) -> Result<Self> {
        let write_txn = db.begin_write()?;
        {
            let _ = write_txn.open_table(EPISODIC_TABLE)?;
            let mut meta_table = write_txn.open_table(METADATA_TABLE)?;

            // Schema Version Check
            let current_version = meta_table
                .get("schema_version")?
                .map(|v| v.value())
                .unwrap_or(0);
            if current_version < SCHEMA_VERSION {
                tracing::info!(target: "shakey::memory", "Upgrading EpisodicMemory schema from v{} to v{}", current_version, SCHEMA_VERSION);
                meta_table.insert("schema_version", SCHEMA_VERSION)?;
            }
        }
        write_txn.commit()?;

        Ok(Self { db })
    }

    /// Record a cycle in the episodic memory.
    /// Elite Industry-Grade: Uses timestamp-prefixed keys for O(TopK) reverse chronological retrieval.
    #[tracing::instrument(skip(self, record), fields(cycle_id = %record.cycle_id))]
    pub fn record_cycle(&self, record: &CycleRecord) -> Result<()> {
        let json = serde_json::to_string(record)?;

        // Generate a monotonic key: {timestamp}_{uuid}
        let now = chrono::Utc::now().to_rfc3339();
        let key = format!("{}_{}", now, record.cycle_id);

        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(EPISODIC_TABLE)?;
            table.insert(key.as_str(), json.as_str())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Record a cycle and automatically generate a vector embedding for semantic search.
    /// Resilient: Vector store failures are logged as warnings and do not fail the primary record.
    #[tracing::instrument(skip(self, record, vector_store, nim_client), fields(cycle_id = %record.cycle_id))]
    pub async fn record_cycle_vectorized(
        &self,
        record: &CycleRecord,
        vector_store: &super::embeddings::VectorStore,
        nim_client: &shakey_distill::nim_client::NimClient,
    ) -> Result<()> {
        // 1. Primary Durable Record
        self.record_cycle(record)?;

        // 2. Vector Augmentation (Non-Fatal)
        let summary = format!(
            "Cycle {}: Strategy={}. Success={}. Improvement={:.4}. Notes: {}",
            record.cycle_id, record.strategy, record.success, record.improvement, record.notes
        );

        let model = nim_client
            .resolve_model_for_role(&shakey_distill::teacher::TeacherRole::EmbeddingQuery)
            .await
            .unwrap_or_else(|_| {
                shakey_distill::teacher::TeacherRole::EmbeddingQuery
                    .default_model()
                    .to_string()
            });
        match nim_client
            .embeddings(vec![summary.clone()], &model, "passage", "memory")
            .await
        {
            Ok(vecs) => {
                if let Some(vec) = vecs.first() {
                    let mut id = [0u8; 16];
                    let uuid = uuid::Uuid::parse_str(&record.cycle_id)
                        .unwrap_or_else(|_| uuid::Uuid::new_v4());
                    id.copy_from_slice(uuid.as_bytes());

                    if let Err(e) = vector_store.store(&id, &summary, vec, Some("episodic"), false)
                    {
                        tracing::warn!("Episodic Augmentation Failed (Vector Store Error): {}", e);
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Episodic Augmentation Failed (Embedding API Error): {}", e);
            }
        }

        Ok(())
    }

    /// Retrieve all cycle records (O(N) - use only for archiving or testing).
    #[tracing::instrument(skip(self))]
    pub fn list_cycles(&self) -> Result<Vec<CycleRecord>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(EPISODIC_TABLE)?;
        let mut records = Vec::new();
        for item in table.iter()? {
            let (_id, json) = item?;
            let record: CycleRecord = serde_json::from_str(json.value())?;
            records.push(record);
        }
        Ok(records)
    }

    /// Retrieve the latest N cycles using high-performance reverse range scan (O(limit)).
    #[tracing::instrument(skip(self))]
    pub fn list_latest_cycles(&self, limit: usize) -> Result<Vec<CycleRecord>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(EPISODIC_TABLE)?;
        let mut records = Vec::new();

        // Reverse scan from the end of the table
        for item in table.range::<&str>(..)?.rev().take(limit) {
            let (_key, json) = item?;
            let record: CycleRecord = serde_json::from_str(json.value())?;
            records.push(record);
        }

        Ok(records)
    }

    /// Prune old memories beyond a certain retention limit.
    /// Resilient: Ensures the agent never fills up its local disk with infinite OODA logs.
    #[tracing::instrument(skip(self))]
    pub fn prune_memories(&self, keep_last: usize) -> Result<usize> {
        let write_txn = self.db.begin_write()?;
        let mut deleted_count = 0;
        {
            let table = write_txn.open_table(EPISODIC_TABLE)?;
            let total_count = table.iter()?.count() as u64;

            if total_count > keep_last as u64 {
                let to_delete = total_count - keep_last as u64;
                let mut keys_to_delete = Vec::new();

                // Get the oldest keys (start of the range)
                for item in table.range::<&str>(..)?.take(to_delete as usize) {
                    let (key, _val) = item?;
                    keys_to_delete.push(key.value().to_string());
                }

                // Now drop the table guard so we can get a mutable table
                drop(table);
                let mut mut_table = write_txn.open_table(EPISODIC_TABLE)?;
                for key in keys_to_delete {
                    if mut_table.remove(key.as_str())?.is_some() {
                        deleted_count += 1;
                    }
                }
            }
        }
        write_txn.commit()?;
        Ok(deleted_count)
    }
}
