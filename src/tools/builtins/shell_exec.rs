use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use std::process::Stdio;

use crate::tools::{Tool, ToolContext, ToolError, ToolOutput, is_safe_path, parse_args};

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
    timeout_seconds: u64,
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
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 30)",
                    "default": 30
                }
            },
            "required": ["command"]
        })
    }

    #[tracing::instrument(
        skip(self, params, ctx),
        fields(command_bytes = params["command"].as_str().map_or(0, str::len))
    )]
    async fn execute(
        &self,
        params: Value,
        ctx: &ToolContext,
    ) -> Result<crate::tools::ToolEffect, ToolError> {
        let args: ShellExecArgs = parse_args(params)?;
        tracing::info!(
            command_bytes = args.command.len(),
            "Executing shell command"
        );

        // Resolve working directory
        let cwd = if let Some(ref dir) = args.cwd {
            is_safe_path(&ctx.workspace_root, std::path::Path::new(dir))?
        } else {
            ctx.workspace_root.clone()
        };

        // Execute command with timeout
        let child = tokio::process::Command::new("/bin/sh")
            .kill_on_drop(true)
            .arg("-c")
            .arg(&args.command)
            .current_dir(&cwd)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| ToolError::ExecutionError(format!("Failed to spawn command: {}", e)))?;

        let mut child = child; // Allow mutable borrow for wait

        let mut stdout = child.stdout.take().expect("Failed to open stdout");
        let mut stderr = child.stderr.take().expect("Failed to open stderr");

        let mut stdout_buf = Vec::new();
        let mut stderr_buf = Vec::new();

        // 100KB limit
        const MAX_OUTPUT_BYTES: usize = 100 * 1024;

        let timeout = std::time::Duration::from_secs(args.timeout_seconds);
        let _start = std::time::Instant::now();

        // Use a loop to read until both streams are closed or timeout
        // This is a simplified version; for robust streaming we'd use line framing or proper async reading.
        // Given we just need to grab output up to a limit, we can use `read_buf` or similar.

        // But simpler: just use `tokio::time::timeout` wrapping the wait + background readers?
        // No, we need to stop reading if limit exceeded to save memory.

        use tokio::io::AsyncReadExt;

        let read_stream = async {
            let mut stdout_truncated = false;
            let mut stderr_truncated = false;

            // We use small buffers for reading chunks
            let mut buf_stdout = [0u8; 1024];
            let mut buf_stderr = [0u8; 1024];

            let mut stdout_done = false;
            let mut stderr_done = false;

            loop {
                tokio::select! {
                    res = stdout.read(&mut buf_stdout), if !stdout_done && !stdout_truncated => {
                        match res {
                            Ok(0) => stdout_done = true,
                            Ok(n) => {
                                if stdout_buf.len() + n > MAX_OUTPUT_BYTES {
                                    stdout_buf.extend_from_slice(&buf_stdout[..MAX_OUTPUT_BYTES - stdout_buf.len()]);
                                    stdout_truncated = true;
                                } else {
                                    stdout_buf.extend_from_slice(&buf_stdout[..n]);
                                }
                            }
                            Err(_) => stdout_done = true,
                        }
                    }
                    res = stderr.read(&mut buf_stderr), if !stderr_done && !stderr_truncated => {
                        match res {
                            Ok(0) => stderr_done = true,
                            Ok(n) => {
                                if stderr_buf.len() + n > MAX_OUTPUT_BYTES {
                                    stderr_buf.extend_from_slice(&buf_stderr[..MAX_OUTPUT_BYTES - stderr_buf.len()]);
                                    stderr_truncated = true;
                                } else {
                                    stderr_buf.extend_from_slice(&buf_stderr[..n]);
                                }
                            }
                            Err(_) => stderr_done = true,
                        }
                    }
                    else => break, // Both streams disabled (done or truncated)
                }
            }

            // Wait for clean exit
            child.wait().await
        };

        let status_res = tokio::time::timeout(timeout, read_stream).await;

        let exit_code = match status_res {
            Ok(Ok(status)) => status.code().unwrap_or(-1),
            Ok(Err(e)) => {
                return Err(ToolError::ExecutionError(format!(
                    "Command execution failed: {}",
                    e
                )));
            }
            Err(_) => {
                // Timeout
                return Err(ToolError::ExecutionError(format!(
                    "Command timed out after {} seconds (process killed)",
                    args.timeout_seconds
                )));
            }
        };

        let mut content = String::from_utf8_lossy(&stdout_buf).into_owned();
        if stdout_buf.len() >= MAX_OUTPUT_BYTES {
            content.push_str("\n... [stdout truncated]");
        }

        let stderr_str = String::from_utf8_lossy(&stderr_buf);
        if !stderr_str.is_empty() {
            if !content.is_empty() {
                content.push('\n');
            }
            content.push_str("[stderr]\n");
            content.push_str(&stderr_str);
            if stderr_buf.len() >= MAX_OUTPUT_BYTES {
                content.push_str("\n... [stderr truncated]");
            }
        }

        if content.is_empty() {
            content = format!("Command exited with code {}", exit_code);
        }

        Ok(crate::tools::ToolEffect::Output(ToolOutput {
            content,
            metadata: serde_json::json!({
                "command": args.command,
                "exit_code": exit_code,
                "stdout_bytes": stdout_buf.len(),
                "stderr_bytes": stderr_buf.len(),
                "cwd": cwd.display().to_string(),
            }),
        }))
    }
}

#[cfg(test)]
#[path = "tests/shell_exec.rs"]
mod tests;
