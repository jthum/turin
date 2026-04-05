use std::path::PathBuf;
use std::process::Stdio;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde_json::Value;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use turin_types::ToolsConfig;
use turin_types::web_tools::{
    WebToolErrorKind, WebToolKind, WebToolRequest, WebToolResponse,
    validate_tools_config as validate_shared_tools_config,
};

use crate::tools::{Tool, ToolContext, ToolEffect, ToolError, ToolOutput};

pub struct WebFetchTool;
pub struct WebSearchTool;

#[derive(Debug, Clone)]
struct ExternalWebRunnerCommand {
    program: PathBuf,
    args_prefix: Vec<String>,
    display: String,
}

pub fn validate_tools_config(settings: &ToolsConfig) -> Result<()> {
    validate_shared_tools_config(settings).map_err(|message| anyhow!(message))
}

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "web_fetch"
    }

    fn description(&self) -> &str {
        "Fetch an HTTP or HTTPS URL and return readable page text. Use this instead of shell_exec for normal web retrieval."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "HTTP or HTTPS URL to fetch"
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Request timeout in seconds",
                    "default": 20
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum number of characters to return",
                    "default": 12000
                }
            },
            "required": ["url"]
        })
    }

    async fn execute(&self, params: Value, ctx: &ToolContext) -> Result<ToolEffect, ToolError> {
        execute_via_runner(WebToolKind::Fetch, params, ctx).await
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    fn name(&self) -> &str {
        "web_search"
    }

    fn description(&self) -> &str {
        "Search the public web and return top result titles, URLs, and snippets. Use this instead of shell_exec for normal discovery."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5
                },
                "timeout_secs": {
                    "type": "integer",
                    "description": "Request timeout in seconds",
                    "default": 20
                }
            },
            "required": ["query"]
        })
    }

    async fn execute(&self, params: Value, ctx: &ToolContext) -> Result<ToolEffect, ToolError> {
        execute_via_runner(WebToolKind::Search, params, ctx).await
    }
}

async fn execute_via_runner(
    tool: WebToolKind,
    params: Value,
    ctx: &ToolContext,
) -> Result<ToolEffect, ToolError> {
    let runner = resolve_web_runner_command()?;
    let request = WebToolRequest {
        tool,
        params,
        tools: (*ctx.tools).clone(),
    };
    let request_json = serde_json::to_vec(&request).map_err(|error| {
        ToolError::ExecutionError(format!("Failed to encode web tool request: {error}"))
    })?;

    let mut command = Command::new(&runner.program);
    for arg in &runner.args_prefix {
        command.arg(arg);
    }
    command.arg("run-json");
    command.stdin(Stdio::piped());
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());

    let mut child = command.spawn().map_err(|error| {
        ToolError::ExecutionError(format!("Failed to launch '{}': {error}", runner.display))
    })?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(&request_json).await.map_err(|error| {
            ToolError::ExecutionError(format!(
                "Failed to write web tool request to '{}': {error}",
                runner.display
            ))
        })?;
    }

    let output = child.wait_with_output().await.map_err(|error| {
        ToolError::ExecutionError(format!("Failed waiting for '{}': {error}", runner.display))
    })?;

    let response = serde_json::from_slice::<WebToolResponse>(&output.stdout).ok();
    if let Some(response) = response {
        return map_response(response);
    }

    if !output.status.success() {
        return Err(ToolError::ExecutionError(
            runner_output_detail(&output).unwrap_or_else(|| {
                format!("'{}' exited with status {}", runner.display, output.status)
            }),
        ));
    }

    Err(ToolError::ExecutionError(format!(
        "Failed to decode web tool response from '{}'",
        runner.display
    )))
}

fn map_response(response: WebToolResponse) -> Result<ToolEffect, ToolError> {
    match response {
        WebToolResponse::Success { output } => Ok(ToolEffect::Output(ToolOutput {
            content: output.content,
            metadata: output.metadata,
        })),
        WebToolResponse::Error { kind, message } => Err(match kind {
            WebToolErrorKind::InvalidParams => ToolError::InvalidParams(message),
            WebToolErrorKind::Execution => ToolError::ExecutionError(message),
        }),
    }
}

fn resolve_web_runner_command() -> Result<ExternalWebRunnerCommand, ToolError> {
    const BINARY_NAME: &str = "turin-web";
    const ENV_VAR: &str = "TURIN_WEB_BIN";

    if let Some(path) = std::env::var_os(ENV_VAR).filter(|value| !value.is_empty()) {
        let path = PathBuf::from(path);
        return Ok(ExternalWebRunnerCommand {
            display: path.display().to_string(),
            program: path,
            args_prefix: Vec::new(),
        });
    }

    if let Some(command) = resolve_workspace_runner_command(BINARY_NAME) {
        return Ok(command);
    }

    let sibling_name = if cfg!(windows) {
        format!("{BINARY_NAME}.exe")
    } else {
        BINARY_NAME.to_string()
    };

    if let Ok(current_exe) = std::env::current_exe() {
        let mut candidates = Vec::new();
        if let Some(parent) = current_exe.parent() {
            candidates.push(parent.join(&sibling_name));
            if parent.file_name().is_some_and(|name| name == "deps")
                && let Some(grandparent) = parent.parent()
            {
                candidates.push(grandparent.join(&sibling_name));
            }
        }

        for sibling in candidates {
            if sibling.exists() {
                return Ok(ExternalWebRunnerCommand {
                    display: sibling.display().to_string(),
                    program: sibling,
                    args_prefix: Vec::new(),
                });
            }
        }
    }

    let path = PathBuf::from(BINARY_NAME);
    Ok(ExternalWebRunnerCommand {
        display: path.display().to_string(),
        program: path,
        args_prefix: Vec::new(),
    })
}

fn runner_output_detail(output: &std::process::Output) -> Option<String> {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if !stderr.is_empty() {
        return Some(stderr);
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if !stdout.is_empty() {
        return Some(stdout);
    }
    None
}

fn resolve_workspace_runner_command(binary_name: &str) -> Option<ExternalWebRunnerCommand> {
    if !workspace_has_runner(binary_name) {
        return None;
    }

    let cargo = std::env::var_os("CARGO").filter(|value| !value.is_empty())?;
    Some(ExternalWebRunnerCommand {
        program: PathBuf::from(cargo),
        args_prefix: vec![
            "run".to_string(),
            "-q".to_string(),
            "-p".to_string(),
            binary_name.to_string(),
            "--".to_string(),
        ],
        display: format!("cargo run -q -p {binary_name} --"),
    })
}

fn workspace_has_runner(binary_name: &str) -> bool {
    let Some(metadata) = workspace_metadata() else {
        return false;
    };
    let Some(packages) = metadata.get("packages").and_then(Value::as_array) else {
        return false;
    };
    for package in packages {
        let Some(targets) = package.get("targets").and_then(Value::as_array) else {
            continue;
        };
        for target in targets {
            let Some(name) = target.get("name").and_then(Value::as_str) else {
                continue;
            };
            if name != binary_name {
                continue;
            }
            let Some(kinds_array) = target.get("kind").and_then(Value::as_array) else {
                continue;
            };
            if kinds_array
                .iter()
                .any(|entry| entry.as_str().is_some_and(|kind| kind == "bin"))
            {
                return true;
            }
        }
    }
    false
}

fn workspace_metadata() -> Option<Value> {
    let cargo = std::env::var_os("CARGO").filter(|value| !value.is_empty())?;
    let manifest_path = find_workspace_manifest_path()?;
    let output = std::process::Command::new(cargo)
        .arg("metadata")
        .arg("--no-deps")
        .arg("--format-version")
        .arg("1")
        .arg("--manifest-path")
        .arg(&manifest_path)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    serde_json::from_slice::<Value>(&output.stdout).ok()
}

fn find_workspace_manifest_path() -> Option<PathBuf> {
    let mut cursor = std::env::current_dir().ok()?;
    loop {
        let manifest = cursor.join("Cargo.toml");
        if manifest.is_file() {
            return Some(manifest);
        }
        if !cursor.pop() {
            return None;
        }
    }
}
