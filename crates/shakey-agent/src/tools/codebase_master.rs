use anyhow::Result;
use ignore::WalkBuilder;
use std::path::Path;

/// Codebase Master: Project-wide structural analysis and file discovery.
#[derive(Default)]
pub struct CodebaseMaster;

impl CodebaseMaster {
    pub fn new() -> Self {
        Self
    }

    /// Generate a high-fidelity directory tree, respecting .gitignore.
    pub fn get_structural_tree(&self, root: &Path) -> Result<String> {
        tracing::info!(target: "shakey::coding", "📂 CodebaseMaster: Analyzing project structure at '{:?}'", root);

        let mut tree = String::new();
        let walker = WalkBuilder::new(root)
            .hidden(false)
            .git_ignore(true)
            .build();

        for result in walker {
            match result {
                Ok(entry) => {
                    let path = entry.path();
                    let depth = entry.depth();
                    let indent = "  ".repeat(depth);
                    let name = path.file_name().unwrap_or_default().to_string_lossy();

                    if depth == 0 {
                        tree.push_str(&format!("Project Root: {:?}\n", path));
                    } else {
                        tree.push_str(&format!(
                            "{}{} {}\n",
                            indent,
                            if path.is_dir() { "📁" } else { "📄" },
                            name
                        ));
                    }
                }
                Err(err) => {
                    tracing::error!(target: "shakey::coding", "Error walking path: {}", err)
                }
            }
        }

        Ok(tree)
    }
}
