use anyhow::Result;
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

use crate::tools::{Tool, ToolContext, ToolError, ToolOutput, parse_args};
use mcp_sdk::client::McpClient;
use mcp_sdk::transport::StdioTransport;
use mcp_sdk::types::ToolDefinition;

/// The builtin tool that allows agents to request an MCP server connection.
pub struct BridgeMcp;

#[derive(Debug, Deserialize)]
struct BridgeMcpArgs {
    command: String,
    args: Vec<String>,
}

#[async_trait]
impl Tool for BridgeMcp {
    fn name(&self) -> &str {
        "bridge_mcp"
    }

    fn description(&self) -> &str {
        "Connect to a Model Context Protocol (MCP) server to dynamically load its tools. Provide the command and arguments to spawn the server (e.g., `['npx', '-y', '@modelcontextprotocol/server-filesystem', '/path']`)."
    }

    fn parameters_schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The executable command (e.g., 'npx', 'python', '/bin/my-server')"
                },
                "args": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "Arguments for the command"
                }
            },
            "required": ["command", "args"]
        })
    }

    async fn execute(
        &self,
        params: Value,
        _ctx: &ToolContext,
    ) -> Result<crate::tools::ToolEffect, ToolError> {
        let args: BridgeMcpArgs = parse_args(params)?;
        let command = args.command.trim();
        if command.is_empty() {
            return Err(ToolError::InvalidParams(
                "'command' must not be empty".to_string(),
            ));
        }

        Ok(crate::tools::ToolEffect::SpawnMcp {
            command: command.to_string(),
            args: args.args,
        })
    }
}

/// A proxy tool that forwards calls to a remote MCP server.
pub struct McpToolProxy {
    client: Arc<McpClient<StdioTransport>>,
    def: ToolDefinition,
}

impl McpToolProxy {
    pub fn new(client: Arc<McpClient<StdioTransport>>, def: ToolDefinition) -> Self {
        Self { client, def }
    }
}

#[async_trait]
impl Tool for McpToolProxy {
    fn name(&self) -> &str {
        &self.def.name
    }

    fn description(&self) -> &str {
        self.def.description.as_deref().unwrap_or("MCP Tool")
    }

    fn parameters_schema(&self) -> Value {
        self.def.input_schema.clone()
    }

    async fn execute(
        &self,
        params: Value,
        _ctx: &ToolContext,
    ) -> Result<crate::tools::ToolEffect, ToolError> {
        let result = self
            .client
            .call_tool(&self.def.name, params)
            .await
            .map_err(|e| ToolError::ExecutionError(format!("MCP Call Failed: {}", e)))?;

        // Convert MCP content to ToolOutput text
        let text_output = result.as_text();

        if result.is_error {
            return Err(ToolError::ExecutionError(text_output));
        }

        Ok(crate::tools::ToolEffect::Output(ToolOutput::new(
            text_output.trim().to_string(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolEffect;
    use std::collections::BTreeSet;

    fn tool_context() -> ToolContext {
        ToolContext {
            workspace_root: std::path::PathBuf::from("."),
            session_id: "test".to_string(),
            agent_id: "default".to_string(),
            store_manager: None,
            embedding_provider: None,
            config: None,
            allowed_native_tools: Arc::new(BTreeSet::from(["bridge_mcp".to_string()])),
            tools: Arc::new(turin_types::ToolsConfig::default()),
        }
    }

    #[tokio::test]
    async fn bridge_mcp_rejects_non_string_args() {
        let err = BridgeMcp
            .execute(
                json!({
                    "command": "node",
                    "args": ["server.js", 42],
                }),
                &tool_context(),
            )
            .await
            .expect_err("non-string args should fail");

        assert!(matches!(err, ToolError::InvalidParams(_)));
    }

    #[tokio::test]
    async fn bridge_mcp_trims_and_preserves_valid_command() {
        let effect = BridgeMcp
            .execute(
                json!({
                    "command": "  node  ",
                    "args": ["server.js"],
                }),
                &tool_context(),
            )
            .await
            .expect("valid bridge_mcp params should parse");

        let ToolEffect::SpawnMcp { command, args } = effect else {
            panic!("expected SpawnMcp effect");
        };
        assert_eq!(command, "node");
        assert_eq!(args, vec!["server.js".to_string()]);
    }
}
