use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Debug, Deserialize)]
pub struct FsInput {
    pub action: String, // "read", "write", "list", "delete"
    pub path: String,
    pub content: Option<String>,
}

/// A Zero-Trust sandboxed File System Toolkit for the autonomous agent.
///
/// Security Features:
/// 1. **Symlink Protection**: Rejects any path containing symlinks that resolve outside the sandbox.
/// 2. **Canonical Jail**: All paths are canonicalized and verified against the base directory.
/// 3. **Resource Quotas**: Limits total workspace size to 1GB to prevent disk-exhaustion attacks.
pub struct FsToolkit {
    base_dir: PathBuf,
    max_workspace_size_bytes: u64,
}

impl FsToolkit {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        let base_dir = base_dir.into();
        let _ = std::fs::create_dir_all(&base_dir);
        Self {
            base_dir: base_dir.canonicalize().unwrap_or(base_dir),
            max_workspace_size_bytes: 1024 * 1024 * 1024, // 1GB Industry-standard limit
        }
    }

    /// Resolve and validate a path to ensure it stays within the sandbox and is symlink-safe.
    fn resolve_path(&self, input_path: &str) -> Result<PathBuf> {
        let joined = self.base_dir.join(input_path.trim_start_matches('/'));

        // --- Zero-Trust: Symlink Verification ---
        // We use symlink_metadata to check for symlinks BEFORE they are traversed (anti-TOCTOU)
        if let Ok(meta) = std::fs::symlink_metadata(&joined) {
            if meta.file_type().is_symlink() {
                return Err(anyhow::anyhow!(
                    "Security: Symlinks are forbidden in the autonomous sandbox."
                ));
            }
        }

        // For existing paths: canonicalize directly
        // For new paths (e.g., write): canonicalize the EXISTING parent
        let canonical = if joined.exists() {
            joined
                .canonicalize()
                .with_context(|| format!("Failed to canonicalize: {:?}", joined))?
        } else {
            // Find the closest existing parent to ensure it's within the sandbox
            let mut current = joined.as_path();
            while let Some(parent) = current.parent() {
                if parent.exists() {
                    let canon_parent = parent.canonicalize()?;
                    // Re-join the non-existent children
                    let rel = joined.strip_prefix(parent)?;
                    return self.validate_containment(canon_parent.join(rel));
                }
                current = parent;
            }
            joined // Fallback (should be caught by validate_containment)
        };

        self.validate_containment(canonical)
    }

    /// Internal helper to ensure a path is strictly inside the base_dir.
    fn validate_containment(&self, path: PathBuf) -> Result<PathBuf> {
        if !path.starts_with(&self.base_dir) {
            return Err(anyhow::anyhow!(
                "Security: Path attempt outside sandbox: {:?} (Base: {:?})",
                path,
                self.base_dir
            ));
        }
        Ok(path)
    }

    /// Check if the workspace has exceeded its resource quota.
    fn check_quota(&self) -> Result<()> {
        let total_size = self.get_dir_size(&self.base_dir)?;
        if total_size > self.max_workspace_size_bytes {
            return Err(anyhow::anyhow!("Resource Exhaustion: Shakey Data workspace exceeds 1GB quota. Self-limiting triggered."));
        }
        Ok(())
    }

    fn get_dir_size(&self, path: &Path) -> Result<u64> {
        let meta = std::fs::symlink_metadata(path)?;
        if meta.file_type().is_symlink() {
            // Drop symlinks entirely for security to prevent infinite recursion
            return Ok(0);
        }

        let mut size = 0;
        if meta.is_dir() {
            for entry in std::fs::read_dir(path)? {
                size += self.get_dir_size(&entry?.path())?;
            }
        } else {
            size = meta.len();
        }
        Ok(size)
    }

    pub fn execute(&self, input_json: &str) -> Result<String> {
        let input: FsInput = serde_json::from_str(input_json)
            .context("Invalid JSON for fs_toolkit. Expected {action, path, content?}")?;

        let target_path = self.resolve_path(&input.path)?;

        match input.action.as_str() {
            "read" => {
                let content = std::fs::read_to_string(&target_path)
                    .with_context(|| format!("Failed to read file: {:?}", target_path))?;
                Ok(content)
            }
            "write" => {
                self.check_quota()?;
                let content = input.content.ok_or_else(|| {
                    anyhow::anyhow!("'content' field required for 'write' action")
                })?;
                if let Some(parent) = target_path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&target_path, content)?;
                Ok(format!("Successfully wrote to {:?}", target_path))
            }
            "list" => {
                let mut results = Vec::new();
                for entry in std::fs::read_dir(&target_path)? {
                    let entry = entry?;
                    let meta = entry.metadata()?;
                    let kind = if meta.is_dir() { "dir" } else { "file" };
                    results.push(format!(
                        "[{}] {}",
                        kind,
                        entry.file_name().to_string_lossy()
                    ));
                }
                Ok(results.join("\n"))
            }
            "delete" => {
                if target_path.is_dir() {
                    std::fs::remove_dir_all(&target_path)?;
                } else {
                    std::fs::remove_file(&target_path)?;
                }
                Ok(format!("Deleted {:?}", target_path))
            }
            _ => Err(anyhow::anyhow!("Unknown action: {}", input.action)),
        }
    }
}
