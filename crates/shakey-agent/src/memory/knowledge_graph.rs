use anyhow::Result;
use redb::{Database, TableDefinition};
use std::sync::Arc;

/// Table definition for triples: (Subject|Predicate, Object)
/// We'll use a compound key for efficient querying.
pub const TRIPLES_TABLE: TableDefinition<&str, &str> = TableDefinition::new("triples");

/// Knowledge graph for persistent entity-relationship storage.
#[derive(Clone)]
pub struct KnowledgeGraph {
    db: Arc<Database>,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct TripleValue {
    pub object: String,
    pub confidence: f64,
    pub last_updated: String,
}

impl KnowledgeGraph {
    pub fn new(db: Arc<Database>) -> Result<Self> {
        let write_txn = db.begin_write()?;
        {
            let _ = write_txn.open_table(TRIPLES_TABLE)?;
        }
        write_txn.commit()?;
        Ok(Self { db })
    }

    /// Store a triple relationship with confidence.
    pub fn store_triple(
        &self,
        subject: &str,
        predicate: &str,
        object: &str,
        confidence: f64,
    ) -> Result<()> {
        let key = format!("{}|{}", subject, predicate);
        let value = TripleValue {
            object: object.to_string(),
            confidence,
            last_updated: chrono::Utc::now().to_rfc3339(),
        };
        let value_json = serde_json::to_string(&value)?;

        let write_txn = self.db.begin_write()?;
        {
            let mut table = write_txn.open_table(TRIPLES_TABLE)?;
            table.insert(key.as_str(), value_json.as_str())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// Retrieve a relation by subject and predicate.
    pub fn get_relation(&self, subject: &str, predicate: &str) -> Result<Option<TripleValue>> {
        let key = format!("{}|{}", subject, predicate);
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(TRIPLES_TABLE)?;
        if let Some(guard) = table.get(key.as_str())? {
            let val: TripleValue = serde_json::from_str(guard.value())?;
            return Ok(Some(val));
        }
        Ok(None)
    }

    /// Find all relations for a given subject using an efficient range scan.
    pub fn find_relations(&self, subject: &str) -> Result<Vec<(String, TripleValue)>> {
        let prefix = format!("{}|", subject);
        let read_txn = self.db.begin_read()?;
        let table = read_txn.open_table(TRIPLES_TABLE)?;
        let mut results = Vec::new();

        for item in table.range(prefix.as_str()..)? {
            let (key_guard, value_guard) = item?;
            let key = key_guard.value();

            if !key.starts_with(&prefix) {
                break;
            }

            let parts: Vec<&str> = key.split('|').collect();
            if parts.len() == 2 {
                let val: TripleValue = serde_json::from_str(value_guard.value())?;
                results.push((parts[1].to_string(), val));
            }
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_kg_range_scan() -> Result<()> {
        let tmp = TempDir::new()?;
        let db = Database::create(tmp.path().join("kg.redb"))?;
        let kg = KnowledgeGraph::new(Arc::new(db))?;

        // Insert triples for multiple subjects
        kg.store_triple("rust", "is", "safe", 1.0)?;
        kg.store_triple("rust", "has", "borrow_checker", 0.9)?;
        kg.store_triple("python", "is", "slow", 0.8)?;
        kg.store_triple("rust", "is", "fast", 0.95)?; // Update entry

        let rust_rels = kg.find_relations("rust")?;
        assert_eq!(rust_rels.len(), 2);

        let py_rels = kg.find_relations("python")?;
        assert_eq!(py_rels.len(), 1);
        assert_eq!(py_rels[0].0, "is");
        assert_eq!(py_rels[0].1.object, "slow");
        assert_eq!(py_rels[0].1.confidence, 0.8);

        Ok(())
    }
}
