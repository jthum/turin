use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;

use crate::tools::{Tool, ToolContext, ToolError, ToolOutput, parse_args};

pub struct EditFileTool;

#[derive(Deserialize)]
struct EditFileArgs {
    /// Path to edit (relative to workspace root, or absolute)
    path: String,
    /// Exact string to search for in the file
    old_text: String,
    /// Replacement string
    new_text: String,
}

#[async_trait]
impl Tool for EditFileTool {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Edit a file by replacing an exact string match. The old_text must appear exactly once in the file. Use read_file first to see the current contents."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit"
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact text to find and replace (must match exactly once)"
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text"
                }
            },
            "required": ["path", "old_text", "new_text"]
        })
    }

    #[tracing::instrument(skip(self, params, ctx), fields(path = %params["path"].as_str().unwrap_or("unknown")))]
    async fn execute(
        &self,
        params: Value,
        ctx: &ToolContext,
    ) -> Result<crate::tools::ToolEffect, ToolError> {
        let args: EditFileArgs = parse_args(params)?;
        tracing::info!(path = %args.path, "Editing file");

        let path =
            crate::tools::is_safe_path(&ctx.workspace_root, std::path::Path::new(&args.path))?;
        const MAX_FILE_BYTES: u64 = 16 * 1024 * 1024;
        if let Ok(meta) = tokio::fs::metadata(&path).await
            && meta.len() > MAX_FILE_BYTES
        {
            return Err(ToolError::ExecutionError(format!(
                "File {} is {} bytes; native edit_file cap is {MAX_FILE_BYTES} bytes",
                path.display(),
                meta.len()
            )));
        }

        let content = tokio::fs::read_to_string(&path).await.map_err(|e| {
            ToolError::ExecutionError(format!("Failed to read {}: {}", path.display(), e))
        })?;

        // Count occurrences
        let count = content.matches(&args.old_text).count();
        if count == 0 {
            return Err(ToolError::ExecutionError(format!(
                "old_text not found in {}. Use read_file to verify the current contents.",
                path.display()
            )));
        }
        if count > 1 {
            return Err(ToolError::ExecutionError(format!(
                "old_text found {} times in {} — it must appear exactly once. Use a more specific match.",
                count,
                path.display()
            )));
        }

        let new_content = content.replacen(&args.old_text, &args.new_text, 1);

        tokio::fs::write(&path, &new_content).await.map_err(|e| {
            ToolError::ExecutionError(format!("Failed to write {}: {}", path.display(), e))
        })?;

        Ok(crate::tools::ToolEffect::Output(ToolOutput {
            content: format!(
                "Successfully edited {}. Replaced {} bytes with {} bytes.",
                path.display(),
                args.old_text.len(),
                args.new_text.len()
            ),
            metadata: serde_json::json!({
                "path": path.display().to_string(),
                "old_len": args.old_text.len(),
                "new_len": args.new_text.len(),
            }),
        }))
    }
}

#[cfg(test)]
#[path = "tests/edit_file.rs"]
mod tests;
