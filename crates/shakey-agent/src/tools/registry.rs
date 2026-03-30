use crate::memory::knowledge_base::KnowledgeBase;
use anyhow::Result;
use shakey_core::metrics::SovereignMetrics;
use shakey_distill::nim_client::NimClient;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub type AsyncToolFn =
    Arc<dyn Fn(String) -> Pin<Box<dyn Future<Output = Result<String>> + Send>> + Send + Sync>;

pub enum ToolImpl {
    Builtin(AsyncToolFn),
    Wasm(Vec<u8>),
    Native,
    Distributed { url: String, schema_version: String },
}

pub struct ToolMetadata {
    pub name: String,
    pub description: String,
    pub input_schema: String,
    pub permissions: Vec<String>,
    pub avg_fuel_consumed: Option<u64>,
}

pub struct ToolEntry {
    pub metadata: ToolMetadata,
    pub implementation: ToolImpl,
}

use std::sync::atomic::{AtomicU64, Ordering};

pub struct ToolRegistry {
    tools: HashMap<String, ToolEntry>,
    pub kb: Option<Arc<KnowledgeBase>>,
    pub nim_client: Option<NimClient>,
    pub fuel_remaining: Arc<AtomicU64>,
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
            kb: None,
            nim_client: None,
            fuel_remaining: Arc::new(AtomicU64::new(10_000_000)), // 10M default budget
        }
    }

    pub fn set_fuel(&self, amount: u64) {
        self.fuel_remaining.store(amount, Ordering::SeqCst);
    }

    pub fn with_resources(mut self, kb: Arc<KnowledgeBase>, nim_client: NimClient) -> Self {
        self.kb = Some(kb);
        self.nim_client = Some(nim_client);
        self.register_builtins();
        self
    }

    pub fn register(&mut self, metadata: ToolMetadata, implementation: ToolImpl) {
        self.tools.insert(
            metadata.name.clone(),
            ToolEntry {
                metadata,
                implementation,
            },
        );
    }

    fn register_builtins(&mut self) {
        self.register_web_tools();
        self.register_execution_tools();
        self.register_sovereign_tools();
        self.register_kali_tools();
        self.register_zenith_core_tools();
    }

    fn register_web_tools(&mut self) {
        self.register(
            ToolMetadata {
                name: "web_fetch".into(),
                description: "Fetch content of a URL (Static/Dynamic Tiered)".into(),
                input_schema: "{\"type\": \"string\", \"format\": \"url\"}".into(),
                permissions: vec!["network".into()],
                avg_fuel_consumed: Some(500_000),
            },
            ToolImpl::Builtin(Arc::new(|url: String| {
                Box::pin(async move { crate::tools::web_fetch::fetch_url(&url).await })
            })),
        );

        self.register(
            ToolMetadata {
                name: "web_search".into(),
                description: "Search the web using DuckDuckGo (Lite)".into(),
                input_schema: "{\"type\": \"string\"}".into(),
                permissions: vec!["network".into()],
                avg_fuel_consumed: Some(1_000_000),
            },
            ToolImpl::Builtin(Arc::new(|query: String| {
                Box::pin(async move { crate::tools::web_search::search_duckduckgo(&query).await })
            })),
        );
    }

    fn register_kali_tools(&mut self) {
        let kali = crate::tools::kali_pro_toolkit::KaliProToolkit::new();
        for tool_name in kali.list_tools() {
            let metadata = ToolMetadata {
                name: format!("kali_{}", tool_name),
                description: format!("Industrial-grade Kali Linux tool: {}", tool_name),
                input_schema:
                    "{\"type\": \"object\", \"additionalProperties\": {\"type\": \"string\"}}"
                        .into(),
                permissions: vec!["terminal".into(), "network".into()],
                avg_fuel_consumed: Some(1_000_000), // Heavy industrial tools
            };

            self.register(metadata, ToolImpl::Native);
        }
    }

    fn register_zenith_core_tools(&mut self) {
        self.register(
            ToolMetadata {
                name: "context_prune".into(),
                description: "Manually prune and summarize agent context memory".into(),
                input_schema: "{\"type\": \"object\"}".into(),
                permissions: vec!["memory_write".into()],
                avg_fuel_consumed: Some(50_000),
            },
            ToolImpl::Builtin(Arc::new(|_| {
                Box::pin(async move { Ok("Context pruning scheduled for next cycle".into()) })
            })),
        );
    }

    fn register_execution_tools(&mut self) {
        if let Some(nim) = &self.nim_client {
            let terminal = Arc::new(crate::tools::shell_exec::SovereignTerminal::new(
                nim.clone(),
            ));
            self.register(
                ToolMetadata {
                    name: "shell_exec".into(),
                    description: "Execute a SOVEREIGN shell command (LLM-Gated for Security)"
                        .into(),
                    input_schema: "{\"type\": \"string\"}".into(),
                    permissions: vec!["process".into()],
                    avg_fuel_consumed: Some(500_000),
                },
                ToolImpl::Builtin(Arc::new(move |cmd: String| {
                    let term = Arc::clone(&terminal);
                    Box::pin(async move { term.execute(&cmd).await })
                })),
            );
        }

        self.register(
            ToolMetadata {
                name: "network_master".into(),
                description: "Sovereign Networking: Advanced Nmap and diagnostics".into(),
                input_schema: "{\"type\": \"object\", \"properties\": {\"target\": {\"type\": \"string\"}, \"aggressive\": {\"type\": \"bool\"}}}".into(),
                permissions: vec!["network".into()],
                avg_fuel_consumed: Some(1_000_000),
            },
            ToolImpl::Builtin(Arc::new(move |input: String| {
                Box::pin(async move {
                    let master = crate::tools::network_master::NetworkMaster::new();
                    let req: serde_json::Value = serde_json::from_str(&input)?;
                    let target = req["target"].as_str().unwrap_or("127.0.0.1");
                    let aggressive = req["aggressive"].as_bool().unwrap_or(false);
                    master.scan(target, None, aggressive)
                })
            })),
        );

        self.register(
            ToolMetadata {
                name: "codebase_master".into(),
                description: "Sovereign Coder: High-fidelity project structural analysis".into(),
                input_schema: "{\"type\": \"string\", \"description\": \"Root path to analyze\"}"
                    .into(),
                permissions: vec!["fs".into()],
                avg_fuel_consumed: Some(500_000),
            },
            ToolImpl::Builtin(Arc::new(move |root_path: String| {
                Box::pin(async move {
                    let master = crate::tools::codebase_master::CodebaseMaster::new();
                    master.get_structural_tree(std::path::Path::new(&root_path))
                })
            })),
        );

        let nim_for_exploit = self.nim_client.clone();
        self.register(
            ToolMetadata {
                name: "exploit_architect".into(),
                description: "Sovereign Pentester: Autonomous vulnerability research and PoC construction".into(),
                input_schema: "{\"type\": \"object\", \"properties\": {\"service\": {\"type\": \"string\"}, \"hint\": {\"type\": \"string\"}}}".into(),
                permissions: vec!["network".into(), "reasoning".into()],
                avg_fuel_consumed: Some(2_000_000),
            },
            ToolImpl::Builtin(Arc::new(move |input: String| {
                let nim_res = nim_for_exploit.clone();
                Box::pin(async move {
                    if let Some(nim) = nim_res {
                        let master = crate::tools::exploit_architect::ExploitArchitect::new(Arc::new(nim));
                        let req: serde_json::Value = serde_json::from_str(&input)?;
                        let service = req["service"].as_str().unwrap_or("Unknown");
                        let hint = req["hint"].as_str().unwrap_or("General audit");
                        master.architect_exploit(service, hint).await
                    } else {
                        Err(anyhow::anyhow!("NimClient not available for ExploitArchitect"))
                    }
                })
            })),
        );

        self.register(
            ToolMetadata {
                name: "native_exec".into(),
                description: "Execute a Rust script in a secure WASM native sandbox".into(),
                input_schema: "{\"type\": \"string\"}".into(),
                permissions: vec!["process".into()],
                avg_fuel_consumed: Some(2_000_000),
            },
            ToolImpl::Native,
        );
    }

    fn register_sovereign_tools(&mut self) {
        // fs_toolkit
        self.register(
            ToolMetadata {
                name: "fs_toolkit".into(),
                description: "Sovereign File System: read/write/list in shakey_data/".into(),
                input_schema: "{\"type\": \"object\", \"properties\": {\"action\": {\"enum\": [\"read\", \"write\", \"list\", \"delete\"]}, \"path\": {\"type\": \"string\"}, \"content\": {\"type\": \"string\"}}}".into(),
                permissions: vec!["fs".into()],
                avg_fuel_consumed: Some(10_000),
            },
            ToolImpl::Builtin(Arc::new(|input: String| {
                Box::pin(async move {
                    let toolkit = crate::tools::fs_tools::FsToolkit::new("shakey_data");
                    toolkit.execute(&input)
                })
            })),
        );

        // Extract resources once to avoid multiple immutable borrows of self during registration
        let kb_res = self.kb.as_ref().map(Arc::clone);
        let nim_res = self.nim_client.clone();

        if let (Some(kb), Some(nim)) = (kb_res, nim_res) {
            let kb_for_recall = Arc::clone(&kb);
            let nim_for_recall = nim.clone();

            self.register(
                ToolMetadata {
                    name: "memory_recall".into(),
                    description: "Retrieve relevant episodic memories via vector search".into(),
                    input_schema: "{\"type\": \"string\"}".into(),
                    permissions: vec!["memory".into()],
                    avg_fuel_consumed: Some(200_000),
                },
                ToolImpl::Builtin(Arc::new(move |query: String| {
                    let kb = Arc::clone(&kb_for_recall);
                    let nim = nim_for_recall.clone();
                    Box::pin(async move {
                        let model = nim
                            .resolve_model_for_role(
                                &shakey_distill::teacher::TeacherRole::EmbeddingQuery,
                            )
                            .await
                            .unwrap_or_else(|_| {
                                shakey_distill::teacher::TeacherRole::EmbeddingQuery
                                    .default_model()
                                    .to_string()
                            });
                        let vecs = nim
                            .embeddings(vec![query], &model, "query", "query")
                            .await?;
                        if let Some(q_vec) = vecs.first() {
                            let results = kb.vector_store.search(q_vec, 5)?;
                            Ok(serde_json::to_string(&results)?)
                        } else {
                            Ok("No memories found.".into())
                        }
                    })
                })),
            );

            let kb_for_query = Arc::clone(&kb);
            self.register(
                ToolMetadata {
                    name: "knowledge_query".into(),
                    description: "Query persistent facts by topic".into(),
                    input_schema: "{\"type\": \"string\"}".into(),
                    permissions: vec!["memory".into()],
                    avg_fuel_consumed: Some(10_000),
                },
                ToolImpl::Builtin(Arc::new(move |topic: String| {
                    let kb = Arc::clone(&kb_for_query);
                    Box::pin(async move {
                        let fact = kb.get_fact(&topic)?;
                        Ok(fact.unwrap_or_else(|| "No fact found for this topic.".into()))
                    })
                })),
            );

            let nim_for_reflect = nim.clone();
            self.register(
                ToolMetadata {
                    name: "agent_reflect".into(),
                    description: "Proactively request a meta-cognitive critique of a complex plan or thought from a Teacher model.".into(),
                    input_schema: "{\"type\": \"string\"}".into(),
                    permissions: vec!["reasoning".into()],
                    avg_fuel_consumed: Some(1_000_000),
                },
                ToolImpl::Builtin(Arc::new(move |context: String| {
                    let nim = nim_for_reflect.clone();
                    Box::pin(async move {
                        let prompt = format!("### Meta-Cognitive Reflection Context\n{}\n\nTask: Analyze the logic above for flaws, loops, or stagnation. Provide a high-level strategic recommendation.", context);
                        let response = nim.query_for_role(shakey_distill::teacher::TeacherRole::Reasoning, "reasoning", &prompt).await?;
                        Ok(format!("### Reflection Analysis\n{}", response))
                    })
                })),
            );
        }
    }

    pub async fn execute(&self, name: &str, input: &str) -> Result<String> {
        let tool = self
            .tools
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("Tool not found: {}", name))?;

        // 1. FUEL AUDIT
        let cost = tool.metadata.avg_fuel_consumed.unwrap_or(10_000);
        let remaining = self.fuel_remaining.load(Ordering::Relaxed);
        if remaining < cost {
            tracing::error!(target: "shakey::fuel", "Sovereign Fuel Exhausted: Task requires {}, only {} remaining.", cost, remaining);
            return Err(anyhow::anyhow!("Sovereign Fuel Exhausted (Budget reached)"));
        }
        self.fuel_remaining.fetch_sub(cost, Ordering::Relaxed);

        let timeout_duration = std::time::Duration::from_secs(30);

        let res = match tokio::time::timeout(
            timeout_duration,
            match &tool.implementation {
                ToolImpl::Builtin(f) => f(input.to_string()),
                ToolImpl::Wasm(bytecode) => {
                    let bytes = bytecode.clone();
                    Box::pin(async move {
                        let sandbox = crate::tools::sandbox::ToolSandbox::new(
                            std::path::PathBuf::from("shakey_data/sandbox"),
                        )?;
                        let res = sandbox.run_wasm(&bytes, "main").await?;
                        Ok(format!("WASM Result: {}", res))
                    })
                }
                ToolImpl::Native => {
                    let script = input.to_string();
                    Box::pin(async move {
                        let sandbox = crate::tools::native_sandbox::NativeSandbox::new()?;
                        sandbox.run_secure_script(&script, 30000).await
                    })
                }
                ToolImpl::Distributed { .. } => Box::pin(async move {
                    Err(anyhow::anyhow!(
                        "Distributed tool (MCP) disabled in current scope."
                    ))
                }),
            },
        )
        .await
        {
            Ok(result) => result,
            Err(_) => Err(anyhow::anyhow!("Tool '{}' timed out after 30s", name)),
        };

        // 2. TELEMETRY: Record tool execution and potential failure heat
        let metrics = SovereignMetrics::global();
        metrics.record_tool_execution(res.is_ok());

        res
    }

    pub fn list(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }

    /// Expose all tool entries for protocol-compliant discovery (e.g. MCP).
    pub fn get_all_tools(&self) -> &HashMap<String, ToolEntry> {
        &self.tools
    }

    /// --- Peak Mastery: Ultra-Elite Tool Verification ---
    ///
    /// Runs a sample input through a tool implementation to verify
    /// its functional integrity before registration.
    pub async fn verify_tool(
        &self,
        name: &str,
        imp: &ToolImpl,
        test_input: &str,
    ) -> Result<String> {
        tracing::info!(target: "shakey", "System: Verifying tool '{}' integrity...", name);

        // Use a tightened timeout for verification
        let timeout_duration = std::time::Duration::from_secs(5);

        let result = tokio::time::timeout(
            timeout_duration,
            match imp {
                ToolImpl::Builtin(f) => f(test_input.to_string()),
                ToolImpl::Wasm(bytecode) => {
                    let bytes = bytecode.clone();
                    Box::pin(async move {
                        let sandbox = crate::tools::sandbox::ToolSandbox::new(
                            std::path::PathBuf::from("shakey_data/sandbox_test"),
                        )?;
                        let res = sandbox.run_wasm(&bytes, "main").await?;
                        Ok(res.to_string())
                    })
                }
                ToolImpl::Native => {
                    let script = test_input.to_string();
                    Box::pin(async move {
                        let sandbox = crate::tools::native_sandbox::NativeSandbox::new()?;
                        sandbox.run_secure_script(&script, 5000).await
                    })
                }
                ToolImpl::Distributed { .. } => Box::pin(async move {
                    Err(anyhow::anyhow!(
                        "Verification not supported for distributed tools."
                    ))
                }),
            },
        )
        .await;

        match result {
            Ok(Ok(res)) => {
                tracing::info!(target: "shakey", "System: Tool '{}' verification PASSED.", name);
                Ok(res)
            }
            Ok(Err(e)) => Err(anyhow::anyhow!("Tool verification FAILED: {}", e)),
            Err(_) => Err(anyhow::anyhow!("Tool verification TIMED OUT")),
        }
    }
}
