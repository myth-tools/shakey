//! Environment auto-detection and path management.
//!
//! Provides canonical paths for Kaggle, Colab, and Local environments
//! to ensure zero-loss training and seamless persistence.

use std::env;
use std::path::{Path, PathBuf};

/// Supported execution environments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Environment {
    Kaggle,
    Colab,
    Local,
}

impl Environment {
    /// Detect the current execution environment.
    pub fn detect() -> Self {
        if env::var("KAGGLE_KERNEL_RUN_TYPE").is_ok() || Path::new("/kaggle/working").exists() {
            Environment::Kaggle
        } else if env::var("COLAB_GPU").is_ok() || Path::new("/content").exists() {
            Environment::Colab
        } else {
            Environment::Local
        }
    }

    /// Get the canonical base path for all Project Shakey data.
    ///
    /// `SHAKEY_DATA_DIR` is checked first in **all** environments so that the
    /// Kaggle pipeline script (which sets it via `os.environ`) can override the
    /// default without needing a code change.
    pub fn base_data_dir(&self) -> PathBuf {
        // Bug #9 fix: honour SHAKEY_DATA_DIR regardless of environment.
        // Previously this was only checked in the Local branch, meaning the
        // Python script's `os.environ["SHAKEY_DATA_DIR"]` was silently ignored
        // when --kaggle was passed.
        if let Ok(path) = env::var("SHAKEY_DATA_DIR") {
            return PathBuf::from(path);
        }
        match self {
            Environment::Kaggle => PathBuf::from("/kaggle/working/shakey_data"),
            Environment::Colab => PathBuf::from("/content/shakey_data"),
            Environment::Local => PathBuf::from("shakey_data"),
        }
    }

    /// Get the canonical path for checkpoints.
    pub fn checkpoint_dir(&self) -> PathBuf {
        self.base_data_dir().join("checkpoints")
    }

    /// Get the canonical path for the knowledge base.
    pub fn knowledge_dir(&self) -> PathBuf {
        self.base_data_dir().join("knowledge")
    }

    /// Get the canonical path for training logs.
    pub fn logs_dir(&self) -> PathBuf {
        self.base_data_dir().join("logs")
    }

    /// Is this a "volatile" environment (e.g. cloud VM that resets)?
    pub fn is_volatile(&self) -> bool {
        match self {
            Environment::Kaggle | Environment::Colab => true,
            Environment::Local => false,
        }
    }
}

/// Helper to get a path relevant to the current environment.
pub fn get_env_path(sub_path: &str) -> PathBuf {
    Environment::detect().base_data_dir().join(sub_path)
}
