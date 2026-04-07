//! Built-in tool implementations for Turin.
//!
//! These are the core tools available out of the box:
//! - `apply_patch` — Apply structured multi-file patches
//! - `read_file` — Read file contents
//! - `write_file` — Create or overwrite a file
//! - `edit_file` — Search-and-replace within a file
//! - `shell_exec` — Execute a shell command
//! - `web_fetch` — Fetch a URL and extract text content
//! - `web_search` — Search the web and return top results
//! - `remember` / `recall` — Store and search durable agent memory

pub mod apply_patch;
pub mod edit_file;
mod memory_tools;
pub mod read_file;
pub mod shell_exec;
mod submit_plan;
mod web_tools;
pub mod write_file;

use crate::tools::mcp::BridgeMcp;
pub use apply_patch::ApplyPatchTool;
pub use edit_file::EditFileTool;
pub use memory_tools::{RecallTool, RememberTool};
pub use read_file::ReadFileTool;
pub use shell_exec::ShellExecTool;
pub use submit_plan::SubmitPlanTool;
pub use web_tools::{WebFetchTool, WebSearchTool, validate_tools_config};
pub use write_file::WriteFileTool;

use super::registry::ToolRegistry;

pub const BUILTIN_TOOL_NAMES: &[&str] = &[
    "apply_patch",
    "read_file",
    "write_file",
    "edit_file",
    "shell_exec",
    "web_fetch",
    "web_search",
    "remember",
    "recall",
    "submit_plan",
    "bridge_mcp",
];

pub const DEFAULT_EXPOSED_TOOL_NAMES: &[&str] = &[
    "read_file",
    "write_file",
    "edit_file",
    "shell_exec",
    "web_fetch",
    "web_search",
    "remember",
    "recall",
    "submit_plan",
];

pub fn expand_builtin_group(name: &str) -> Option<&'static [&'static str]> {
    match name {
        "all" => Some(BUILTIN_TOOL_NAMES),
        "fs" => Some(&["apply_patch", "read_file", "write_file", "edit_file"]),
        "shell" => Some(&["shell_exec"]),
        "web" => Some(&["web_fetch", "web_search"]),
        "memory" => Some(&["remember", "recall"]),
        "planning" => Some(&["submit_plan"]),
        "integration" => Some(&["bridge_mcp"]),
        _ => None,
    }
}

/// Create a ToolRegistry with all built-in tools registered.
pub fn create_default_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry
        .register(Box::new(ApplyPatchTool))
        .expect("Failed to register ApplyPatchTool");
    registry
        .register(Box::new(ReadFileTool))
        .expect("Failed to register ReadFileTool");
    registry
        .register(Box::new(WriteFileTool))
        .expect("Failed to register WriteFileTool");
    registry
        .register(Box::new(EditFileTool))
        .expect("Failed to register EditFileTool");
    registry
        .register(Box::new(ShellExecTool))
        .expect("Failed to register ShellExecTool");
    registry
        .register(Box::new(WebFetchTool))
        .expect("Failed to register WebFetchTool");
    registry
        .register(Box::new(WebSearchTool))
        .expect("Failed to register WebSearchTool");
    registry
        .register(Box::new(RememberTool))
        .expect("Failed to register RememberTool");
    registry
        .register(Box::new(RecallTool))
        .expect("Failed to register RecallTool");
    registry
        .register(Box::new(SubmitPlanTool))
        .expect("Failed to register SubmitPlanTool");
    registry
        .register(Box::new(BridgeMcp))
        .expect("Failed to register BridgeMcp");
    registry
}
