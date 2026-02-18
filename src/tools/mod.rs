pub mod provider;
pub mod registry;
pub mod builtins;
pub mod mcp;

use async_trait::async_trait;
use std::path::{Path, PathBuf, Component};
use serde_json::{Value, json};

/// Output from a tool execution.
#[derive(Debug, Clone)]
pub struct ToolOutput {
    /// Primary content returned to the LLM
    pub content: String,
    /// Structured metadata for logging and harness inspection
    pub metadata: Value,
}

impl ToolOutput {
    pub fn new(content: String) -> Self {
        Self {
            content,
            metadata: json!({}),
        }
    }
}

/// Error from a tool execution.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("Tool execution failed: {0}")]
    ExecutionError(String),
    #[error("Invalid parameters: {0}")]
    InvalidParams(String),
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
}

/// Context available to tools during execution.
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// Root directory for workspace-relative paths
    pub workspace_root: std::path::PathBuf,
    /// Current session ID
    pub session_id: String,
}

/// The effect of a tool execution.
#[derive(Debug, Clone)]
pub enum ToolEffect {
    /// Standard output — return to LLM
    Output(ToolOutput),
    /// Enqueue new tasks (usually from `submit_task`)
    EnqueueTask {
        title: String,
        subtasks: Vec<String>,
        clear_existing: bool,
    },
    /// Register new tools from an MCP server (usually from `bridge_mcp`)
    SpawnMcp {
        command: String,
        args: Vec<String>,
    },
}

/// The Tool trait — every tool in Turin implements this.
///
/// Tools are the only way the agent interacts with the outside world.
/// They are pure I/O: no direct access to the harness engine or LLM.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name used in LLM tool definitions
    fn name(&self) -> &str;

    /// Human-readable description for the LLM
    fn description(&self) -> &str;

    /// JSON Schema for parameters
    fn parameters_schema(&self) -> Value;

    /// Execute the tool with validated parameters
    async fn execute(
        &self,
        params: Value,
        ctx: &ToolContext,
    ) -> Result<ToolEffect, ToolError>;
}

/// Helper to deserialize tool arguments from a JSON Value.
#[must_use = "parse result should be checked for InvalidParams errors"]
pub fn parse_args<T: serde::de::DeserializeOwned>(args: Value) -> Result<T, ToolError> {
    serde_json::from_value(args).map_err(|e| ToolError::InvalidParams(e.to_string()))
}

/// Centralized path validation to prevent traversal attacks.
#[must_use = "path validation result must be checked before file operations"]
pub fn is_safe_path(root: &Path, path: &Path) -> Result<PathBuf, ToolError> {
    // 1. Resolve to absolute-ish path within root
    let resolved = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };

    // 2. Reject any path with '..' components to prevent traversal
    if resolved.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(ToolError::PermissionDenied(format!(
            "Path traversal (..) not allowed: {}",
            path.display()
        )));
    }

    // 3. Eager canonicalization of root
    let canonical_root = root.canonicalize().map_err(|e| {
        ToolError::ExecutionError(format!("Failed to canonicalize workspace root: {}", e))
    })?;

    // 4. Resolve the target path's existing ancestor to check against root
    let mut current = resolved.clone();
    while !current.exists() {
        if let Some(parent) = current.parent() {
            current = parent.to_path_buf();
        } else {
            break;
        }
    }

    let canonical_current = current.canonicalize().map_err(|e| {
        ToolError::ExecutionError(format!("Failed to canonicalize path: {}", e))
    })?;

    if !canonical_current.starts_with(&canonical_root) {
        return Err(ToolError::PermissionDenied(format!(
            "Path '{}' is outside workspace root",
            path.display()
        )));
    }

    Ok(resolved)
}

