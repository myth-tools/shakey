//! Persistent knowledge base using the `redb` embedded database.
//!
//! Stores facts, embeddings (metadata), and agentic memory.
//! This ensures the agent's knowledge survives restarts and is
//! easily queryable for RAG or self-reflection.

use anyhow::{Context, Result};
use redb::{Database, ReadableTable, TableDefinition};
use std::fs;
use std::path::Path;
use std::sync::Arc;

use super::embeddings::VectorStore;
use super::knowledge_graph::KnowledgeGraph;

/// Table definition for facts: (Topic, Fact)
const FACTS_TABLE: TableDefinition<&str, &str> = TableDefinition::new("facts");

/// Table definition for capabilities: (Name, Score)
const CAPABILITIES_TABLE: TableDefinition<&str, f64> = TableDefinition::new("capabilities");
/// Table definition for file hashes: (FilePath, SHA256)
const FILE_HASHES_TABLE: TableDefinition<&str, &str> = TableDefinition::new("file_hashes");

/// Persistent knowledge store for the agent.
#[derive(Clone)]
pub struct KnowledgeBase {
    db: Arc<Database>,
    pub vector_store: VectorStore,
    pub knowledge_graph: KnowledgeGraph,
    /// NIM Client for on-the-fly embeddings during search
    pub nim_client: Option<shakey_distill::nim_client::NimClient>,
    /// Ultra-Elite: Distributed state broadcasting hook
    pub webhook_url: Option<String>,
}

impl KnowledgeBase {
    /// Open or create a knowledge base at the specified path.
    pub fn new(
        path: impl AsRef<Path>,
        webhook_url: Option<String>,
        nim_client: Option<shakey_distill::nim_client::NimClient>,
    ) -> Result<Self> {
        let db = Database::create(path)?;
        let db = Arc::new(db);

        // Initialize tables
        let write_txn = db.begin_write()?;
        {
            let _ = write_txn.open_table(FACTS_TABLE)?;
            let _ = write_txn.open_table(CAPABILITIES_TABLE)?;
            let _ = write_txn.open_table(FILE_HASHES_TABLE)?;
        }
        write_txn.commit()?;

        let vector_store = VectorStore::new(Arc::clone(&db))?;
        let knowledge_graph = KnowledgeGraph::new(Arc::clone(&db))?;

        Ok(Self {
            db,
            vector_store,
            knowledge_graph,
            nim_client,
            webhook_url,
        })
    }

    pub fn get_db(&self) -> Arc<Database> {
        Arc::clone(&self.db)
    }

    /// Store a fact in the knowledge base.
    pub fn store_fact(&self, topic: &str, fact: &str) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(FACTS_TABLE)?;
            table.insert(topic, fact)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Retrieve a fact by topic.
    pub fn get_fact(&self, topic: &str) -> Result<Option<String>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(FACTS_TABLE)?;
        let fact = table.get(topic)?.map(|v| v.value().to_string());
        Ok(fact)
    }

    /// Retrieve the stored hash of a file for incremental indexing.
    pub fn get_file_hash(&self, path: &str) -> Result<Option<String>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(FILE_HASHES_TABLE)?;
        let hash = table.get(path)?.map(|v| v.value().to_string());
        Ok(hash)
    }

    /// Update the stored hash of a file after indexing.
    pub fn update_file_hash(&self, path: &str, hash: &str) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(FILE_HASHES_TABLE)?;
            table.insert(path, hash)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Surgical Memory Audit: Clear all embeddings and metadata.
    /// This is necessary during model transitions to prevent "Vector Contamination."
    pub fn clear_embeddings(&self) -> Result<()> {
        use super::embeddings::{EMBEDDINGS_TABLE, FILE_SEGMENTS_TABLE, METADATA_TABLE};
        tracing::warn!(target: "shakey::memory", "System: Wiping vector search index for model upgrade...");

        let write_txn = self.db.begin_write()?;
        {
            let mut emb_table = write_txn.open_table(EMBEDDINGS_TABLE)?;
            let mut meta_table = write_txn.open_table(METADATA_TABLE)?;
            let mut seg_table = write_txn.open_table(FILE_SEGMENTS_TABLE)?;

            // Redb's iter + remove is safe for clearing entire tables
            let ids: Vec<[u8; 16]> = emb_table
                .iter()?
                .filter_map(|item| item.ok().map(|(id, _)| *id.value()))
                .collect();
            for id in ids {
                emb_table.remove(&id)?;
                meta_table.remove(&id)?;
            }

            // Clear entire file segment table
            let paths: Vec<String> = seg_table
                .iter()?
                .filter_map(|item| item.ok().map(|(p, _)| p.value().to_string()))
                .collect();
            for path in paths {
                seg_table.remove(path.as_str())?;
            }
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Surgical Memory Audit: Clear all file-system tracking hashes.
    /// This forces the indexer to perform a "Clean Slate" full project audit.
    pub fn clear_file_hashes(&self) -> Result<()> {
        tracing::warn!(target: "shakey::memory", "System: Wiping file-system indexing cache...");

        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(FILE_HASHES_TABLE)?;
            let paths: Vec<String> = table
                .iter()?
                .filter_map(|item| item.ok().map(|(p, _)| p.value().to_string()))
                .collect();
            for path in paths {
                table.remove(path.as_str())?;
            }
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Update a capability score.
    pub fn update_capability(&self, name: &str, score: f64) -> Result<()> {
        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(CAPABILITIES_TABLE)?;
            table.insert(name, score)?;
        }
        write_txn.commit()?;

        // Ultra-Elite: Async Distributed State Broadcast
        if let Some(url) = &self.webhook_url {
            let url = url.clone();
            let name = name.to_string();
            tokio::spawn(async move {
                let client = reqwest::Client::new();
                let payload = serde_json::json!({
                    "event": "capability_update",
                    "capability": name,
                    "score": score,
                    "timestamp": chrono::Utc::now().to_rfc3339()
                });
                if let Err(e) = client.post(&url).json(&payload).send().await {
                    tracing::warn!("Failed to broadcast state update to {}: {}", url, e);
                }
            });
        }

        Ok(())
    }

    /// Get a capability score.
    pub fn get_capability(&self, name: &str) -> Result<Option<f64>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(CAPABILITIES_TABLE)?;
        let score = table.get(name)?.map(|v| v.value());
        Ok(score)
    }

    /// List all facts for debugging or RAG.
    pub fn list_facts(&self) -> Result<Vec<(String, String)>> {
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(FACTS_TABLE)?;
        let mut facts = Vec::new();
        for item in table.iter()? {
            let (topic, fact) = item?;
            facts.push((topic.value().to_string(), fact.value().to_string()));
        }
        Ok(facts)
    }

    /// Load creator identity and sovereign mission from configs/creator.yaml.
    pub fn load_creator_config(&self, path: impl AsRef<Path>) -> Result<()> {
        if !path.as_ref().exists() {
            tracing::warn!(
                "Creator config not found at {}, skipping identity load.",
                path.as_ref().display()
            );
            return Ok(());
        }

        let content = fs::read_to_string(path)?;
        let yaml: serde_json::Value =
            serde_yaml::from_str(&content).context("Failed to parse creator.yaml")?;

        // Store the raw YAML as a system identity fact for RAG
        self.store_fact("system_identity", &content)?;

        // Extract and store specific facts for quick lookup
        if let Some(creator) = yaml.get("creator") {
            if let Some(name) = creator.get("name").and_then(|v| v.as_str()) {
                self.store_fact("creator_name", name)?;
            }
            if let Some(role) = creator.get("role").and_then(|v| v.as_str()) {
                self.store_fact("creator_role", role)?;
            }
        }

        // Store governance rules as facts
        if let Some(gov) = yaml.get("governance") {
            if let Some(map) = gov.as_object() {
                for (k, v) in map {
                    if let Some(rule) = v.as_str() {
                        self.store_fact(&format!("governance_{}", k), rule)?;
                    }
                }
            }
        }

        tracing::info!("Sovereign Identity Loaded: Knowledge base grounded in Creator's vision.");
        Ok(())
    }

    /// ZENITH 5.5: Search for similar facts across the vector store.
    /// Automatically handles embedding generation via NIM.
    pub async fn search_similar(&self, query: &str, top_k: usize) -> Result<Vec<FactSearchResult>> {
        let nim = self
            .nim_client
            .as_ref()
            .context("Search failed: KnowledgeBase has no NimClient for embeddings")?;

        let model = nim
            .resolve_model_for_role(&shakey_distill::teacher::TeacherRole::EmbeddingQuery)
            .await
            .unwrap_or_else(|_| {
                shakey_distill::teacher::TeacherRole::EmbeddingQuery
                    .default_model()
                    .to_string()
            });

        let embedding = nim
            .embeddings(vec![query.to_string()], &model, "query", "memory")
            .await?
            .first()
            .cloned()
            .context("NIM embedding generation failed")?;

        let results = self.vector_store.search(&embedding, top_k)?;

        Ok(results
            .into_iter()
            .map(|(content, score)| FactSearchResult { content, score })
            .collect())
    }
}

pub struct FactSearchResult {
    pub content: String,
    pub score: f32,
}
