use std::collections::BTreeMap;
use std::sync::Arc;

use super::{Tool, ToolContext};

/// Central registry of available tools.
///
/// The ToolRegistry owns all tool instances and provides:
/// - Tool lookup by name
/// - JSON schema generation for LLM tool definitions
/// - Tool execution dispatch
#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: BTreeMap::new(),
        }
    }

    /// Register a tool. Returns error if a tool with the same name is already registered.
    pub fn register(&mut self, tool: Box<dyn Tool>) -> anyhow::Result<()> {
        let name = tool.name().to_string();
        if self.tools.contains_key(&name) {
            anyhow::bail!("Tool '{}' already registered", name);
        }
        self.tools.insert(name, Arc::from(tool));
        Ok(())
    }

    /// Get a tool by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Generate JSON tool definitions for the LLM API.
    ///
    /// Returns a Vec of tool definition objects matching the standard format:
    /// ```json
    /// { "name": "...", "description": "...", "input_schema": { ... } }
    /// ```
    pub fn tool_definitions(&self) -> Vec<serde_json::Value> {
        self.tools
            .values()
            .map(|tool| {
                serde_json::json!({
                    "name": tool.name(),
                    "description": tool.description(),
                    "input_schema": tool.parameters_schema(),
                })
            })
            .collect()
    }

    pub fn tool_definitions_filtered(
        &self,
        allowed_tools: &std::collections::BTreeSet<String>,
    ) -> Vec<serde_json::Value> {
        self.tools
            .values()
            .filter(|tool| allowed_tools.contains(tool.name()))
            .map(|tool| {
                serde_json::json!({
                    "name": tool.name(),
                    "description": tool.description(),
                    "input_schema": tool.parameters_schema(),
                })
            })
            .collect()
    }

    /// Execute a tool by name with the given arguments.
    pub async fn execute(
        &self,
        name: &str,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<super::ToolEffect, super::ToolError> {
        if !ctx.is_native_tool_allowed(name) {
            return Err(super::ToolError::PermissionDenied(format!(
                "Tool '{}' is not enabled for this turn",
                name
            )));
        }
        let tool = self
            .get(name)
            .ok_or_else(|| super::ToolError::ExecutionError(format!("Unknown tool: {}", name)))?;
        tool.execute(args, ctx).await
    }

    /// Get the number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Returns true if no tools are registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::super::builtins;
    use super::*;

    #[test]
    fn test_registry_register_and_get() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(builtins::ReadFileTool)).unwrap();
        assert!(registry.get("read_file").is_some());
        assert!(registry.get("nonexistent").is_none());
        assert_eq!(registry.len(), 1);
    }

    #[test]
    fn test_tool_definitions() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(builtins::ReadFileTool)).unwrap();
        registry
            .register(Box::new(builtins::WriteFileTool))
            .unwrap();

        let defs = registry.tool_definitions();
        assert_eq!(defs.len(), 2);

        // Check that each definition has the required fields
        for def in &defs {
            assert!(def.get("name").is_some());
            assert!(def.get("description").is_some());
            assert!(def.get("input_schema").is_some());
        }
    }

    #[tokio::test]
    async fn filtered_definitions_and_execution_respect_allowed_set() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(builtins::ReadFileTool)).unwrap();
        registry
            .register(Box::new(builtins::WriteFileTool))
            .unwrap();

        let allowed = std::collections::BTreeSet::from(["read_file".to_string()]);
        let defs = registry.tool_definitions_filtered(&allowed);
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0]["name"], "read_file");

        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolContext {
            workspace_root: dir.path().to_path_buf(),
            session_id: "test".into(),
            agent_id: "default".into(),
            store_manager: None,
            embedding_provider: None,
            config: None,
            allowed_native_tools: std::sync::Arc::new(allowed),
            tools: std::sync::Arc::new(turin_types::ToolsConfig::default()),
        };

        let err = registry
            .execute(
                "write_file",
                serde_json::json!({ "path": "x.txt", "content": "hi" }),
                &ctx,
            )
            .await
            .expect_err("write_file should be denied when excluded");
        assert!(matches!(err, super::super::ToolError::PermissionDenied(_)));
    }

    #[test]
    #[should_panic(expected = "already registered")]
    fn test_duplicate_registration_panics() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(builtins::ReadFileTool)).unwrap();
        registry.register(Box::new(builtins::ReadFileTool)).unwrap(); // should panic
    }

    #[test]
    fn test_registry_clone_is_independent() {
        let mut registry = ToolRegistry::new();
        registry.register(Box::new(builtins::ReadFileTool)).unwrap();

        let mut cloned = registry.clone();
        cloned.register(Box::new(builtins::WriteFileTool)).unwrap();

        assert_eq!(registry.len(), 1);
        assert_eq!(cloned.len(), 2);
        assert!(registry.get("write_file").is_none());
        assert!(cloned.get("write_file").is_some());
    }
}
