//! Built-in tool implementations for Turin.
//!
//! These are the core tools available out of the box:
//! - `read_file` — Read file contents
//! - `write_file` — Create or overwrite a file
//! - `edit_file` — Search-and-replace within a file
//! - `shell_exec` — Execute a shell command

pub mod edit_file;
pub mod read_file;
pub mod shell_exec;
mod submit_plan;
pub mod write_file;

use crate::tools::mcp::BridgeMcp;
pub use edit_file::EditFileTool;
pub use read_file::ReadFileTool;
pub use shell_exec::ShellExecTool;
pub use submit_plan::SubmitPlanTool;
pub use write_file::WriteFileTool;

use super::registry::ToolRegistry;

/// Create a ToolRegistry with all built-in tools registered.
pub fn create_default_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
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
        .register(Box::new(SubmitPlanTool))
        .expect("Failed to register SubmitPlanTool");
    registry
        .register(Box::new(BridgeMcp))
        .expect("Failed to register BridgeMcp");
    registry
}
