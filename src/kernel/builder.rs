use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
// Mutex removed

use crate::inference::embeddings::EmbeddingProvider;
use crate::kernel::{Kernel, TurinConfig};
use crate::persistence::manager::StoreManager;
use crate::tools::builtins::create_default_registry;
use crate::tools::registry::ToolRegistry;

/// Builder for constructing a `Kernel` instance.
pub struct RuntimeBuilder {
    config: TurinConfig,
    json: bool,
    tool_registry: ToolRegistry,

    embedding_provider: Option<Arc<dyn EmbeddingProvider>>,
}

impl RuntimeBuilder {
    /// Create a new builder with the given configuration.
    pub fn new(config: TurinConfig) -> Self {
        Self {
            config,
            json: false,
            tool_registry: create_default_registry(),

            embedding_provider: None,
        }
    }

    /// Enable JSON output mode (NDJSON).
    pub fn json_mode(mut self, json: bool) -> Self {
        self.json = json;
        self
    }

    /// Register a custom tool registry (overwriting defaults).
    pub fn with_tool_registry(mut self, registry: ToolRegistry) -> Self {
        self.tool_registry = registry;
        self
    }

    /// Build the Kernel.
    pub fn build(self) -> Result<Kernel> {
        let store_manager = Arc::new(StoreManager::new(&self.config.kernel.workspace_root));
        Ok(Kernel {
            config: Arc::new(self.config),
            json: self.json,
            tool_registry: self.tool_registry,
            store_manager,
            harness: Arc::new(std::sync::Mutex::new(None)),
            check_watcher: None,
            clients: HashMap::new(),
            embedding_provider: self.embedding_provider,
            active_queue: std::sync::Arc::new(tokio::sync::Mutex::new(None)),
            mcp_clients: Vec::new(),
        })
    }
}
