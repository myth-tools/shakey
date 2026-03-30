use crate::tools::sandbox::ToolSandbox;
use anyhow::Result;

/// Hardened native execution sandbox. Now fully WASM-isolated.
/// Compiles raw agent native code directly to WebAssembly to enforce tight fuel bounds.
pub struct NativeSandbox {
    sandbox: ToolSandbox,
}

impl NativeSandbox {
    pub fn new() -> Result<Self> {
        let temp_dir = std::env::temp_dir().join("shakey_native");

        // ── Sovereign Industry Hardening: Unix Permission Lock (0700) ──
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            let mut builder = std::fs::DirBuilder::new();
            builder.recursive(true).mode(0o700);
            builder.create(&temp_dir)?;
        }
        #[cfg(not(unix))]
        std::fs::create_dir_all(&temp_dir)?;

        let sandbox = ToolSandbox::new(temp_dir)?;
        Ok(Self { sandbox })
    }

    /// Execute Rust source code securely by compiling it to WASM first.
    pub async fn run_secure_script(&self, script: &str, _timeout_ms: u64) -> Result<String> {
        let (bytecode, _fuel1) = self
            .sandbox
            .compile_and_benchmark("native_exec", script)
            .await?;
        let (result, fuel2) = self.sandbox.run_wasm_with_stats(&bytecode, "main").await?;
        Ok(format!(
            "Execution Result: {}, Fuel consumed: {}",
            result, fuel2
        ))
    }
}
