use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use std::process::Stdio;

use crate::tools::{parse_args, Tool, ToolContext, ToolError, ToolOutput, is_safe_path};

pub struct ShellExecTool;

#[derive(Deserialize)]
struct ShellExecArgs {
    /// The shell command to execute
    command: String,
    /// Optional working directory (relative to workspace root)
    #[serde(default)]
    cwd: Option<String>,
    /// Timeout in seconds (default: 30)
    #[serde(default = "default_timeout")]
    timeout_secs: u64,
}

fn default_timeout() -> u64 {
    30
}

#[async_trait]
impl Tool for ShellExecTool {
    fn name(&self) -> &str {
        "shell_exec"
    }

    fn description(&self) -> &str {
        "Execute a shell command and return its stdout and stderr. The command runs in the workspace root directory by default. Use with caution."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute (runs via /bin/sh -c)"
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory (relative to workspace root). Defaults to workspace root."
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30)",
                    "default": 30
                }
            },
            "required": ["command"]
        })
    }

    #[tracing::instrument(skip(self, params, ctx), fields(command = %params["command"].as_str().unwrap_or("unknown")))]
    async fn execute(&self, params: Value, ctx: &ToolContext) -> Result<crate::tools::ToolEffect, ToolError> {
        let args: ShellExecArgs = parse_args(params)?;
        tracing::info!(command = %args.command, "Executing shell command");

        // Resolve working directory
        let cwd = if let Some(ref dir) = args.cwd {
            is_safe_path(&ctx.workspace_root, std::path::Path::new(dir))?
        } else {
            ctx.workspace_root.clone()
        };

        // Execute command with timeout
        let child = tokio::process::Command::new("/bin/sh")
            .arg("-c")
            .arg(&args.command)
            .current_dir(&cwd)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| ToolError::ExecutionError(format!("Failed to spawn command: {}", e)))?;

        let timeout = std::time::Duration::from_secs(args.timeout_secs);
        let output = tokio::select! {
            result = child.wait_with_output() => {
                result.map_err(|e| ToolError::ExecutionError(format!("Command failed: {}", e)))?
            }
            _ = tokio::time::sleep(timeout) => {
                // Timeout — the child process is killed when `child` is dropped
                // because tokio::process::Child kills the process on drop.
                return Err(ToolError::ExecutionError(format!(
                    "Command timed out after {} seconds (process killed)",
                    args.timeout_secs
                )));
            }
        };

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let exit_code = output.status.code().unwrap_or(-1);

        // Build combined output for the LLM
        let mut content = String::new();
        if !stdout.is_empty() {
            content.push_str(&stdout);
        }
        if !stderr.is_empty() {
            if !content.is_empty() {
                content.push('\n');
            }
            content.push_str("[stderr]\n");
            content.push_str(&stderr);
        }
        if content.is_empty() {
            content = format!("Command exited with code {}", exit_code);
        }

        // Truncate very long output
        const MAX_OUTPUT: usize = 100_000;
        if content.len() > MAX_OUTPUT {
            content.truncate(MAX_OUTPUT);
            content.push_str("\n... [output truncated]");
        }

        Ok(crate::tools::ToolEffect::Output(ToolOutput {
            content,
            metadata: serde_json::json!({
                "command": args.command,
                "exit_code": exit_code,
                "stdout_bytes": stdout.len(),
                "stderr_bytes": stderr.len(),
                "cwd": cwd.display().to_string(),
            }),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolEffect;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_shell_exec_echo() {
        let dir = TempDir::new().unwrap();
        let tool = ShellExecTool;
        let ctx = ToolContext {
            workspace_root: dir.path().to_path_buf(),
            session_id: "test".to_string(),
        };

        let result = tool
            .execute(serde_json::json!({ "command": "echo hello" }), &ctx)
            .await
            .unwrap();

        if let ToolEffect::Output(output) = result {
            assert_eq!(output.content.trim(), "hello");
            assert_eq!(output.metadata["exit_code"], 0);
        } else {
            panic!("Expected ToolEffect::Output");
        }
    }

    #[tokio::test]
    async fn test_shell_exec_exit_code() {
        let dir = TempDir::new().unwrap();
        let tool = ShellExecTool;
        let ctx = ToolContext {
            workspace_root: dir.path().to_path_buf(),
            session_id: "test".to_string(),
        };

        let result = tool
            .execute(serde_json::json!({ "command": "exit 42" }), &ctx)
            .await
            .unwrap();

        if let ToolEffect::Output(output) = result {
            assert_eq!(output.metadata["exit_code"], 42);
        } else {
            panic!("Expected ToolEffect::Output");
        }
    }

    #[tokio::test]
    async fn test_shell_exec_stderr() {
        let dir = TempDir::new().unwrap();
        let tool = ShellExecTool;
        let ctx = ToolContext {
            workspace_root: dir.path().to_path_buf(),
            session_id: "test".to_string(),
        };

        let result = tool
            .execute(
                serde_json::json!({ "command": "echo err >&2" }),
                &ctx,
            )
            .await
            .unwrap();

        if let ToolEffect::Output(output) = result {
            assert!(output.content.contains("[stderr]"));
            assert!(output.content.contains("err"));
        } else {
            panic!("Expected ToolEffect::Output");
        }
    }

    #[tokio::test]
    async fn test_shell_exec_timeout() {
        let dir = TempDir::new().unwrap();
        let tool = ShellExecTool;
        let ctx = ToolContext {
            workspace_root: dir.path().to_path_buf(),
            session_id: "test".to_string(),
        };

        let result = tool
            .execute(
                serde_json::json!({ "command": "sleep 60", "timeout_secs": 1 }),
                &ctx,
            )
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("timed out"));
    }
}
