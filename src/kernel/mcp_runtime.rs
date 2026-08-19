use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use mcp_sdk::client::McpClient;
use mcp_sdk::transport::StdioTransport;
use mcp_sdk::types::ToolDefinition;
use tracing::{info, instrument, warn};

use crate::kernel::execution_host::ExecutionHost;
use crate::tools::mcp::McpToolProxy;
use crate::tools::registry::ToolRegistry;

pub(crate) struct McpClientEntry {
    pub command: String,
    pub args: Vec<String>,
    pub client: Arc<McpClient<mcp_sdk::transport::StdioTransport>>,
}

pub(crate) struct McpAttachReport {
    pub listed_tools: usize,
    pub registered_tools: usize,
    pub skipped_existing_tools: usize,
}

impl ExecutionHost {
    /// Connect to an MCP server, initialize it, and register its tools.
    #[instrument(skip(self, args), fields(command = %command, args = ?args))]
    pub(crate) async fn spawn_mcp_server(
        &mut self,
        command: &str,
        args: &[String],
    ) -> Result<McpAttachReport> {
        let args_str: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

        // Check for existing client.
        if let Some(client) = self
            .mcp_clients
            .iter()
            .find(|e| e.command == command && e.args == args)
            .map(|entry| Arc::clone(&entry.client))
        {
            info!(command = %command, "Reusing existing MCP client");

            let list_result = client
                .list_tools()
                .await
                .with_context(|| "Failed to list MCP tools on reused client")?;
            return register_reused_mcp_tools(&mut self.tool_registry, client, list_result.tools);
        }

        info!("Connecting to MCP server");

        let transport = StdioTransport::new(command, &args_str)
            .with_context(|| format!("Failed to spawn MCP process: {}", command))?;

        let client = McpClient::new(transport);
        client
            .initialize()
            .await
            .with_context(|| "Failed to initialize MCP client")?;

        let list_result = client
            .list_tools()
            .await
            .with_context(|| "Failed to list MCP tools")?;
        validate_new_mcp_tools(&self.tool_registry, &list_result.tools)?;

        let client_arc = Arc::new(client);
        let report = register_new_mcp_tools(
            &mut self.tool_registry,
            client_arc.clone(),
            list_result.tools,
        )?;
        self.mcp_clients.push(McpClientEntry {
            command: command.to_string(),
            args: args.to_vec(),
            client: client_arc,
        });

        info!(count = report.registered_tools, "MCP tools registered");
        Ok(report)
    }

    /// Best-effort shutdown for all active MCP subprocess clients owned by this kernel.
    pub async fn shutdown_mcp_clients(&mut self) {
        if self.mcp_clients.is_empty() {
            return;
        }

        let entries = std::mem::take(&mut self.mcp_clients);
        let shutdown = async move {
            for entry in entries {
                if let Err(err) = entry.client.shutdown().await {
                    warn!(
                        command = %entry.command,
                        args = ?entry.args,
                        error = %err,
                        "Failed to shutdown MCP client cleanly"
                    );
                }
            }
        };
        if tokio::time::timeout(Duration::from_secs(2), shutdown)
            .await
            .is_err()
        {
            warn!("MCP client shutdown exceeded the grace period");
        }
    }
}

fn validate_new_mcp_tools(registry: &ToolRegistry, tools: &[ToolDefinition]) -> Result<()> {
    let mut seen = std::collections::BTreeSet::new();
    for tool in tools {
        let name = tool.name.trim();
        if name.is_empty() {
            bail!("MCP server returned a tool with an empty name");
        }
        if !seen.insert(name.to_string()) {
            bail!("MCP server returned duplicate tool '{}'", name);
        }
        if registry.contains(name) {
            bail!("MCP tool '{}' conflicts with an existing tool", name);
        }
    }
    Ok(())
}

fn register_new_mcp_tools(
    registry: &mut ToolRegistry,
    client: Arc<McpClient<StdioTransport>>,
    tools: Vec<ToolDefinition>,
) -> Result<McpAttachReport> {
    let listed_tools = tools.len();
    for tool_def in tools {
        let proxy = McpToolProxy::new(client.clone(), tool_def);
        registry
            .register(Box::new(proxy))
            .with_context(|| "Failed to register MCP tool")?;
    }
    Ok(McpAttachReport {
        listed_tools,
        registered_tools: listed_tools,
        skipped_existing_tools: 0,
    })
}

fn register_reused_mcp_tools(
    registry: &mut ToolRegistry,
    client: Arc<McpClient<StdioTransport>>,
    tools: Vec<ToolDefinition>,
) -> Result<McpAttachReport> {
    let listed_tools = tools.len();
    let mut registered_tools = 0;
    let mut skipped_existing_tools = 0;
    let mut seen = std::collections::BTreeSet::new();

    for tool_def in tools {
        let name = tool_def.name.trim();
        if name.is_empty() {
            bail!("MCP server returned a tool with an empty name");
        }
        if !seen.insert(name.to_string()) {
            bail!("MCP server returned duplicate tool '{}'", name);
        }
        if registry.contains(name) {
            skipped_existing_tools += 1;
            continue;
        }
        let proxy = McpToolProxy::new(client.clone(), tool_def);
        registry
            .register(Box::new(proxy))
            .with_context(|| "Failed to register MCP tool")?;
        registered_tools += 1;
    }

    Ok(McpAttachReport {
        listed_tools,
        registered_tools,
        skipped_existing_tools,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::builtins::ReadFileTool;
    use serde_json::json;

    fn tool(name: &str) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: None,
            input_schema: json!({ "type": "object" }),
        }
    }

    #[test]
    fn validates_duplicate_mcp_tool_names() {
        let registry = ToolRegistry::new();
        let err = validate_new_mcp_tools(&registry, &[tool("alpha"), tool("alpha")])
            .expect_err("duplicate MCP tool names should fail");

        assert!(err.to_string().contains("duplicate tool 'alpha'"));
    }

    #[test]
    fn validates_mcp_tool_name_conflicts_with_existing_registry() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(ReadFileTool)).unwrap();

        let err = validate_new_mcp_tools(&registry, &[tool("read_file")])
            .expect_err("MCP tool name should not conflict with existing tools");

        assert!(err.to_string().contains("conflicts with an existing tool"));
    }
}
