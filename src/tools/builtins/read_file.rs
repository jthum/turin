use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;

use crate::tools::{parse_args, Tool, ToolContext, ToolError, ToolOutput};

pub struct ReadFileTool;

#[derive(Deserialize)]
struct ReadFileArgs {
    /// Path to read (relative to workspace root, or absolute)
    path: String,
}

#[async_trait]
impl Tool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read the contents of a file. Returns the full file content as text."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (relative to workspace root, or absolute)"
                }
            },
            "required": ["path"]
        })
    }

    #[tracing::instrument(skip(self, params, ctx), fields(path = %params["path"].as_str().unwrap_or("unknown")))]
    async fn execute(&self, params: Value, ctx: &ToolContext) -> Result<crate::tools::ToolEffect, ToolError> {
        let args: ReadFileArgs = parse_args(params)?;
        tracing::info!(path = %args.path, "Reading file");
        
        // Security: validate path is within workspace using centralized logic
        let path = crate::tools::is_safe_path(&ctx.workspace_root, std::path::Path::new(&args.path))?;

        let content = tokio::fs::read_to_string(&path)
            .await
            .map_err(|e| ToolError::ExecutionError(format!("Failed to read {}: {}", path.display(), e)))?;

        Ok(crate::tools::ToolEffect::Output(ToolOutput {
            content,
            metadata: serde_json::json!({
                "path": path.display().to_string(),
                "bytes": tokio::fs::metadata(&path).await.map(|m| m.len()).unwrap_or(0),
            }),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolEffect;
    use std::path::PathBuf;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_read_file() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello world").unwrap();

        let tool = ReadFileTool;
        let ctx = ToolContext {
            workspace_root: dir.path().to_path_buf(),
            session_id: "test".to_string(),
        };

        let result = tool
            .execute(serde_json::json!({ "path": "test.txt" }), &ctx)
            .await
            .unwrap();
        
        if let ToolEffect::Output(output) = result {
            assert_eq!(output.content, "hello world");
        } else {
            panic!("Expected ToolEffect::Output");
        }
    }

    #[tokio::test]
    async fn test_read_file_not_found() {
        let dir = TempDir::new().unwrap();
        let tool = ReadFileTool;
        let ctx = ToolContext {
            workspace_root: dir.path().to_path_buf(),
            session_id: "test".to_string(),
        };

        let result = tool
            .execute(serde_json::json!({ "path": "nonexistent.txt" }), &ctx)
            .await;
        assert!(result.is_err());
    }
}
