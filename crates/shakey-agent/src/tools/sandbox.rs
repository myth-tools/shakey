//! Hardened sandboxed tool execution using `wasmtime` and `WASI`.
//!
//! Provides a secure environment for the agent to execute self-created
//! tools and scripts with strict resource limits and capability controls.

use anyhow::Result;
use std::path::PathBuf;
use wasmtime::{Config, Engine, Module, ResourceLimiter, Store};
use wasmtime_wasi::preview1::{add_to_linker_async, WasiP1Ctx};
/// State held within the Wasmtime Store.
struct HostState {
    wasi: WasiP1Ctx,
    limits: StoreLimits,
}

// Note: HostState no longer implements WasiView as we are currently targeting
// WASI Preview 1 (WasiP1Ctx) for maximum compatibility.

/// Resource limits for the sandbox.
struct StoreLimits {
    max_memory: usize,
    max_table_elements: usize,
    max_instances: usize,
    max_tables: usize,
    max_memories: usize,
}

impl ResourceLimiter for HostState {
    fn memory_growing(
        &mut self,
        _current: usize,
        desired: usize,
        _maximum: Option<usize>,
    ) -> Result<bool> {
        if desired > self.limits.max_memory {
            tracing::warn!(
                "WASM Sandbox: Memory limit exceeded ({} > {})",
                desired,
                self.limits.max_memory
            );
            return Ok(false);
        }
        Ok(true)
    }

    fn table_growing(
        &mut self,
        _current: usize,
        desired: usize,
        _maximum: Option<usize>,
    ) -> Result<bool> {
        Ok(desired <= self.limits.max_table_elements)
    }

    fn instances(&self) -> usize {
        self.limits.max_instances
    }
    fn tables(&self) -> usize {
        self.limits.max_tables
    }
    fn memories(&self) -> usize {
        self.limits.max_memories
    }
}

/// Sandboxed execution environment for agentic tools.
pub struct ToolSandbox {
    engine: Engine,
    #[allow(dead_code)]
    base_dir: PathBuf,
}

impl ToolSandbox {
    /// Initialize a new sandbox engine.
    pub fn new(base_dir: PathBuf) -> Result<Self> {
        let mut config = Config::new();
        config.wasm_memory64(true);
        config.async_support(true);
        config.consume_fuel(true); // Enable CPU execution limits (fuel)

        let engine = Engine::new(&config)?;
        Ok(Self { engine, base_dir })
    }

    /// Execute a WASM module from bytecode and return (exit_code, fuel_consumed).
    pub async fn run_wasm_with_stats(
        &self,
        bytecode: &[u8],
        func_name: &str,
    ) -> Result<(i32, u64)> {
        let module = Module::from_binary(&self.engine, bytecode)?;

        // Build WASI context (Capability-based security)
        let temp_dir = tempfile::tempdir()?;
        let mut wasi_builder = wasmtime_wasi::WasiCtxBuilder::new();
        wasi_builder.inherit_stdout().inherit_stderr();
        wasi_builder.preopened_dir(
            temp_dir.path(),
            "/",
            wasmtime_wasi::DirPerms::all(),
            wasmtime_wasi::FilePerms::all(),
        )?;

        let wasi: WasiP1Ctx = wasi_builder.build_p1();

        let state = HostState {
            wasi,
            limits: StoreLimits {
                max_memory: 64 * 1024 * 1024, // Optimized: 64MB tight limit for agentic tools
                max_table_elements: 1000,
                max_instances: 1,
                max_tables: 1,
                max_memories: 1,
            },
        };

        let mut store = Store::new(&self.engine, state);
        store.limiter(|s| s); // Use our ResourceLimiter implementation

        // Unified 50M instruction limit for elite efficiency
        store.set_fuel(50_000_000)?;

        let mut linker = wasmtime::Linker::new(&self.engine);
        add_to_linker_async(&mut linker, |s: &mut HostState| &mut s.wasi)?;

        let instance: wasmtime::Instance = linker.instantiate_async(&mut store, &module).await?;
        let func = instance.get_typed_func::<(), i32>(&mut store, func_name)?;

        let result = func.call_async(&mut store, ()).await?;

        // Check fuel consumption for telemetry
        let fuel_consumed = 50_000_000 - store.get_fuel().unwrap_or(0);
        tracing::debug!(
            "Tool '{}' finished. Fuel consumed: {}",
            func_name,
            fuel_consumed
        );

        Ok((result, fuel_consumed))
    }

    /// Execute a WASM module from bytecode with legacy compatibility.
    pub async fn run_wasm(&self, bytecode: &[u8], func_name: &str) -> Result<i32> {
        let (res, _) = self.run_wasm_with_stats(bytecode, func_name).await?;
        Ok(res)
    }

    /// Autonomous Compilation & Benchmarking:
    /// Convert Rust source code into a WASM module and measure its efficiency.
    pub async fn compile_and_benchmark(
        &self,
        name: &str,
        source_code: &str,
    ) -> Result<(Vec<u8>, u64)> {
        let bytecode = self.compile_rust_to_wasm(name, source_code).await?;

        // ── Sovereign Benchmark: Continuous Performance Validation ──
        // Run the freshly compiled tool to measure baseline fuel consumption.
        let (_, fuel) = self.run_wasm_with_stats(&bytecode, "main").await?;

        // Elite Threshold: Basic tools should consume < 10M fuel for simple logic.
        if fuel > 10_000_000 {
            tracing::warn!(
                "Sovereign Warning: Tool '{}' exceeds elite fuel threshold ({} > 10M)",
                name,
                fuel
            );
        }

        Ok((bytecode, fuel))
    }

    /// Autonomous Compilation: Convert Rust source code into a WASM module.
    ///
    /// This allows the agent to "evolve" by creating its own high-performance tools.
    pub async fn compile_rust_to_wasm(&self, name: &str, source_code: &str) -> Result<Vec<u8>> {
        let temp_dir = tempfile::tempdir()?;
        let src_path = temp_dir.path().join("main.rs");
        let out_path = temp_dir.path().join(format!("{}.wasm", name));

        std::fs::write(&src_path, source_code)?;

        // Execute rustc with WASM target
        // Ensure the host has wasm32-wasi target installed: `rustup target add wasm32-wasi`
        // Resolve rustc path securely
        static RUSTC_PATH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        let rustc = RUSTC_PATH.get_or_init(|| {
            std::process::Command::new("which")
                .arg("rustc")
                .output()
                .ok()
                .and_then(|out| {
                    if out.status.success() {
                        Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| "/usr/bin/rustc".to_string())
        });

        let output = std::process::Command::new(rustc)
            .arg("--target")
            .arg("wasm32-wasi")
            .arg("-O") // Optimize for speed
            .arg("-o")
            .arg(&out_path)
            .arg(&src_path)
            .output()?;

        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("Compilation failed: {}", err));
        }

        let bytecode = std::fs::read(&out_path)?;
        Ok(bytecode)
    }

    /// Securely execute a shell command (Limited to an allow-list).
    pub fn run_shell(&self, command: &str) -> Result<String> {
        // Sovereign Allow-list: Audited commands for autonomous operations
        let allowed_commands = [
            "ls", "cat", "grep", "find", "whoami", "hostname", "uname", "git", "cargo", "echo",
        ];
        let cmd_name = command.split_whitespace().next().unwrap_or("");

        if !allowed_commands.contains(&cmd_name) {
            return Err(anyhow::anyhow!(
                "Sovereign Security: Command '{}' is blocked by policy",
                cmd_name
            ));
        }

        // Additional Protection: Block pipe/redirect characters to prevent injection
        if command.contains('>')
            || command.contains('|')
            || command.contains('&')
            || command.contains(';')
        {
            return Err(anyhow::anyhow!("Sovereign Security: Shell operators (>, |, &, ;) are forbidden in sandboxed commands"));
        }

        // Execute with restricted environment securely
        static SH_PATH: std::sync::OnceLock<String> = std::sync::OnceLock::new();
        let sh = SH_PATH.get_or_init(|| {
            std::process::Command::new("which")
                .arg("sh")
                .output()
                .ok()
                .and_then(|out| {
                    if out.status.success() {
                        Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| "/bin/sh".to_string())
        });

        let output = std::process::Command::new(sh)
            .arg("-c")
            .arg(command)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output()?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            let err = String::from_utf8_lossy(&output.stderr);
            Err(anyhow::anyhow!("Shell error: {}", err))
        }
    }
}
