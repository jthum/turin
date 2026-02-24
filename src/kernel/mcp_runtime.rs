use std::sync::Arc;

use anyhow::{Context, Result};
use mcp_sdk::client::McpClient;
use mcp_sdk::transport::StdioTransport;
use tracing::{info, instrument};

use crate::kernel::Kernel;
use crate::tools::mcp::McpToolProxy;

pub(crate) struct McpClientEntry {
    pub command: String,
    pub args: Vec<String>,
    pub client: Arc<McpClient<mcp_sdk::transport::StdioTransport>>,
}

impl Kernel {
    /// Connect to an MCP server, initialize it, and register its tools.
    #[instrument(skip(self, args), fields(command = %command, args = ?args))]
    pub(crate) async fn spawn_mcp_server(
        &mut self,
        command: &str,
        args: &[String],
    ) -> Result<usize> {
        let args_str: Vec<&str> = args.iter().map(|s| s.as_str()).collect();

        // Check for existing client.
        if let Some(entry) = self
            .mcp_clients
            .iter()
            .find(|e| e.command == command && e.args == args)
        {
            info!(command = %command, "Reusing existing MCP client");

            let list_result = entry
                .client
                .list_tools()
                .await
                .with_context(|| "Failed to list MCP tools on reused client")?;
            let count = list_result.tools.len();

            // Refresh tool registry entries in case the registry changed since initial attach.
            for tool_def in list_result.tools {
                let proxy = McpToolProxy::new(entry.client.clone(), tool_def);
                let _ = self.tool_registry.register(Box::new(proxy));
            }
            return Ok(count);
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
        let count = list_result.tools.len();

        let client_arc = Arc::new(client);
        self.mcp_clients.push(McpClientEntry {
            command: command.to_string(),
            args: args.to_vec(),
            client: client_arc.clone(),
        });

        for tool_def in list_result.tools {
            let proxy = McpToolProxy::new(client_arc.clone(), tool_def);
            self.tool_registry
                .register(Box::new(proxy))
                .with_context(|| "Failed to register MCP tool")?;
        }

        info!(count = count, "MCP tools registered");
        Ok(count)
    }
}
