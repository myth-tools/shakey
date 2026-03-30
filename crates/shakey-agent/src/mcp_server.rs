//! Model Context Protocol (MCP) Server for Project Shakey.
//!
//! This module implements the "Real Logic" for autonomous interaction
//! with external tools and services using the MCP standard.
//!
//! Unlike simulation code, this server provides:
//! 1. Protocol-compliant JSON-RPC 2.0 transport.
//! 2. Dynamic Tool Registration and Discovery.
//! 3. Resource Management for autonomous context injection.
//! 4. Sandboxed Execution of untrusted model-generated scripts.

use crate::tools::registry::ToolRegistry;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing;

/// MCP Tool definition according to the standard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// A request from the model to the MCP server.
#[derive(Debug, Deserialize)]
pub struct McpRequest {
    pub jsonrpc: String,
    pub method: String,
    pub params: serde_json::Value,
    pub id: serde_json::Value,
}

/// A response from the MCP server to the model.
#[derive(Debug, Serialize)]
pub struct McpResponse {
    pub jsonrpc: String,
    pub result: Option<serde_json::Value>,
    pub error: Option<serde_json::Value>,
    pub id: serde_json::Value,
}

/// Sovereign MCP Server — The "Brain-Body" interface for the Shakey Agent.
pub struct McpServer {
    /// Registry of all tools (Builtin, WASM, Native, etc.)
    registry: Arc<RwLock<ToolRegistry>>,
    /// Agent Identity (used for capability negotiation).
    pub agent_id: String,
    /// Lifecycle state.
    initialized: AtomicBool,
}

impl McpServer {
    pub fn new(agent_id: impl Into<String>, registry: Arc<RwLock<ToolRegistry>>) -> Self {
        Self {
            registry,
            agent_id: agent_id.into(),
            initialized: AtomicBool::new(false),
        }
    }

    /// Register a new tool dynamically in the central registry.
    pub async fn register_tool(
        &self,
        metadata: crate::tools::registry::ToolMetadata,
        implementation: crate::tools::registry::ToolImpl,
    ) -> Result<()> {
        tracing::info!(target: "shakey::mcp", "🛠️ Registering Real MCP Tool: {}", metadata.name);
        let mut registry = self.registry.write().await;
        registry.register(metadata, implementation);
        Ok(())
    }

    /// Handle an incoming MCP request (JSON-RPC).
    pub async fn handle_request(&self, request: McpRequest) -> Result<McpResponse> {
        let span = tracing::info_span!(target: "shakey::mcp", "handle_request", method = %request.method, request_id = ?request.id);
        let _enter = span.enter();

        // Block all requests except 'initialize' or 'notifications/initialized' if not initialized.
        if !self.initialized.load(Ordering::Relaxed)
            && request.method != "initialize"
            && request.method != "notifications/initialized"
        {
            return Ok(McpResponse {
                jsonrpc: "2.0".to_string(),
                result: None,
                error: Some(serde_json::json!({
                    "code": -32002,
                    "message": "Server not initialized"
                })),
                id: request.id,
            });
        }

        tracing::debug!(target: "shakey::mcp", "📡 Handling MCP Request: {}", request.method);

        match request.method.as_str() {
            // ── Lifecycle: Initialize ──
            "initialize" => Ok(McpResponse {
                jsonrpc: "2.0".to_string(),
                result: Some(serde_json::json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "resources": {},
                        "prompts": {}
                    },
                    "serverInfo": { "name": "Shakey Sovereign Server", "version": "1.0.0" }
                })),
                error: None,
                id: request.id,
            }),

            "notifications/initialized" => {
                self.initialized.store(true, Ordering::SeqCst);
                tracing::info!(target: "shakey::mcp", "🚀 MCP Server fully INITIALIZED and ready.");
                Ok(McpResponse {
                    jsonrpc: "2.0".to_string(),
                    result: Some(serde_json::json!({})),
                    error: None,
                    id: request.id,
                })
            }

            // ── Standard MCP Tool Discovery ──
            "tools/list" => {
                let registry = self.registry.read().await;
                let tools = registry.get_all_tools();

                let tools_obj: Vec<serde_json::Value> = tools
                    .values()
                    .map(|entry| {
                        let schema: serde_json::Value =
                            serde_json::from_str(&entry.metadata.input_schema)
                                .unwrap_or(serde_json::json!({}));

                        serde_json::json!({
                            "name": entry.metadata.name,
                            "description": entry.metadata.description,
                            "inputSchema": schema
                        })
                    })
                    .collect();

                Ok(McpResponse {
                    jsonrpc: "2.0".to_string(),
                    result: Some(serde_json::json!({"tools": tools_obj})),
                    error: None,
                    id: request.id,
                })
            }

            // ── Standard MCP Tool Execution ──
            "tools/call" => {
                let tool_name = request
                    .params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("-32602: Missing tool name in call"))?;

                let arguments = request
                    .params
                    .get("arguments")
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "{}".to_string());

                match self.execute_tool(tool_name, &arguments).await {
                    Ok(res) => Ok(McpResponse {
                        jsonrpc: "2.0".to_string(),
                        result: Some(
                            serde_json::json!({"content": [{"type": "text", "text": res}]}),
                        ),
                        error: None,
                        id: request.id,
                    }),
                    Err(e) => Ok(McpResponse {
                        jsonrpc: "2.0".to_string(),
                        result: None,
                        error: Some(serde_json::json!({
                            "code": -32603,
                            "message": format!("Tool execution failed: {}", e)
                        })),
                        id: request.id,
                    }),
                }
            }

            // ── Sovereign Prompt Discovery ──
            "prompts/list" => Ok(McpResponse {
                jsonrpc: "2.0".to_string(),
                result: Some(serde_json::json!({
                    "prompts": [
                        {
                            "name": "sovereign_reasoning",
                            "description": "High-fidelity chain-of-thought reasoner",
                            "arguments": [{"name": "context", "required": true}]
                        },
                        {
                            "name": "sovereign_pentester",
                            "description": "Expert in Kali Linux, network mapping, and security auditing",
                            "arguments": [{"name": "target", "required": true}]
                        },
                        {
                            "name": "sovereign_coder",
                            "description": "Senior Rust Architect and systems programmer",
                            "arguments": [{"name": "task", "required": true}]
                        },
                        {
                            "name": "sovereign_ops_master",
                            "description": "Master of Linux internals and shell-fu",
                            "arguments": [{"name": "objective", "required": true}]
                        }
                    ]
                })),
                error: None,
                id: request.id,
            }),

            // ── Sovereign Prompt Retrieval ──
            "prompts/get" => {
                let name = request
                    .params
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let content = match name {
                    "sovereign_reasoning" => "You are the Sovereign-Zenith Reasoner. Use First Principles and the OODA Loop for every task. Think recursively and audit your own thoughts for logic flaws before responding.",
                    "sovereign_pentester" => "You are the Sovereign Pentesting Zenith. Use perfect Kali-Linux tool syntax and First Principles security diagnostics. Target: {{target}}. Perform a deep audit and construct professional exploits for remediation.",
                    "sovereign_coder" => "You are the Sovereign Architect Zenith. Construct ultra-optimized, zero-allocation Rust systems. Audit for thread-safety and side-channel vulnerabilities. Task: {{task}}",
                    "sovereign_ops_master" => "You are the Zenith of OS Internals. Mastery of Linux kernel, shell-fu, and automation. Objective: {{objective}}",
                    _ => "Unknown prompt template."
                };

                Ok(McpResponse {
                    jsonrpc: "2.0".to_string(),
                    result: Some(serde_json::json!({
                        "messages": [{
                            "role": "user",
                            "content": { "type": "text", "text": content }
                        }]
                    })),
                    error: None,
                    id: request.id,
                })
            }

            // ── Sovereign Resource Discovery ──
            "resources/list" => {
                let registry = self.registry.read().await;
                let topics: Vec<String> = if let Some(kb) = &registry.kb {
                    kb.list_facts()?
                        .into_iter()
                        .map(|(topic, _)| topic)
                        .collect()
                } else {
                    Vec::new()
                };

                let mcp_resources: Vec<serde_json::Value> = topics
                    .into_iter()
                    .map(|topic| {
                        serde_json::json!({
                            "uri": format!("shakey://kb/{}", topic),
                            "name": topic,
                            "description": "Knowledge Base fact entry",
                            "mimeType": "text/markdown"
                        })
                    })
                    .collect();

                Ok(McpResponse {
                    jsonrpc: "2.0".to_string(),
                    result: Some(serde_json::json!({"resources": mcp_resources})),
                    error: None,
                    id: request.id,
                })
            }

            // ── Sovereign Resource Retrieval ──
            "resources/read" => {
                let uri = request
                    .params
                    .get("uri")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| anyhow::anyhow!("-32602: Missing resource URI"))?;

                let topic = uri
                    .strip_prefix("shakey://kb/")
                    .ok_or_else(|| anyhow::anyhow!("-32602: Invalid resource URI format"))?;

                let registry = self.registry.read().await;
                let content = if let Some(kb) = &registry.kb {
                    kb.get_fact(topic)?
                        .unwrap_or_else(|| "Fact not found.".to_string())
                } else {
                    "Knowledge Base not available.".to_string()
                };

                Ok(McpResponse {
                    jsonrpc: "2.0".to_string(),
                    result: Some(serde_json::json!({
                        "contents": [{
                            "uri": uri,
                            "mimeType": "text/markdown",
                            "text": content
                        }]
                    })),
                    error: None,
                    id: request.id,
                })
            }

            _ => Ok(McpResponse {
                jsonrpc: "2.0".to_string(),
                result: None,
                error: Some(serde_json::json!({
                    "code": -32601,
                    "message": format!("Method not found: {}", request.method)
                })),
                id: request.id,
            }),
        }
    }

    /// Execute a tool using the Sovereign Registry.
    async fn execute_tool(&self, name: &str, args: &str) -> Result<String> {
        tracing::info!(target: "shakey::mcp", "🚀 EXECUTING Tool 3.0: {}", name);
        let registry = self.registry.read().await;
        registry.execute(name, args).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tool_registration_mcp() {
        let registry = Arc::new(RwLock::new(ToolRegistry::new()));
        let server = McpServer::new("shakey-test", Arc::clone(&registry));

        {
            let mut reg = registry.write().await;
            reg.register(
                crate::tools::registry::ToolMetadata {
                    name: "test_tool".into(),
                    description: "A test tool".into(),
                    input_schema: "{}".into(),
                    permissions: vec![],
                    avg_fuel_consumed: None,
                },
                crate::tools::registry::ToolImpl::Native,
            );
        }

        // MCP Standard: Server must be initialized before discovery.
        let init_req = McpRequest {
            jsonrpc: "2.0".into(),
            method: "initialize".into(),
            params: serde_json::json!({}),
            id: serde_json::json!(0),
        };
        let _ = server.handle_request(init_req).await.unwrap();

        let initialized_notif = McpRequest {
            jsonrpc: "2.0".into(),
            method: "notifications/initialized".into(),
            params: serde_json::json!({}),
            id: serde_json::json!(0),
        };
        let _ = server.handle_request(initialized_notif).await.unwrap();

        let req = McpRequest {
            jsonrpc: "2.0".into(),
            method: "tools/list".into(),
            params: serde_json::json!({}),
            id: serde_json::json!(1),
        };

        let resp = server.handle_request(req).await.unwrap();
        assert!(resp.result.is_some());
    }
}
