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
        agent_id: "test-agent".to_string(),
        store_manager: None,
        embedding_provider: None,
        config: None,
        allowed_native_tools: std::sync::Arc::new(crate::tools::policy::full_native_tool_set()),
        tools: std::sync::Arc::new(turin_types::ToolsConfig::default()),
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
        agent_id: "test-agent".to_string(),
        store_manager: None,
        embedding_provider: None,
        config: None,
        allowed_native_tools: std::sync::Arc::new(crate::tools::policy::full_native_tool_set()),
        tools: std::sync::Arc::new(turin_types::ToolsConfig::default()),
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
        agent_id: "test-agent".to_string(),
        store_manager: None,
        embedding_provider: None,
        config: None,
        allowed_native_tools: std::sync::Arc::new(crate::tools::policy::full_native_tool_set()),
        tools: std::sync::Arc::new(turin_types::ToolsConfig::default()),
    };

    let result = tool
        .execute(serde_json::json!({ "command": "echo err >&2" }), &ctx)
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
        agent_id: "test-agent".to_string(),
        store_manager: None,
        embedding_provider: None,
        config: None,
        allowed_native_tools: std::sync::Arc::new(crate::tools::policy::full_native_tool_set()),
        tools: std::sync::Arc::new(turin_types::ToolsConfig::default()),
    };

    let result = tool
        .execute(
            serde_json::json!({ "command": "sleep 60", "timeout_seconds": 1 }),
            &ctx,
        )
        .await;

    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("timed out"));
}
#[tokio::test]
async fn test_shell_exec_output_truncation() {
    let dir = TempDir::new().unwrap();
    let tool = ShellExecTool;
    let ctx = ToolContext {
        workspace_root: dir.path().to_path_buf(),
        session_id: "test".to_string(),
        agent_id: "test-agent".to_string(),
        store_manager: None,
        embedding_provider: None,
        config: None,
        allowed_native_tools: std::sync::Arc::new(crate::tools::policy::full_native_tool_set()),
        tools: std::sync::Arc::new(turin_types::ToolsConfig::default()),
    };

    // Command that produces > 100KB output.
    // `yes` produces infinite output, piped to head to get enough but not too much.
    // 105000 bytes is > 100KB.
    let result = tool
        .execute(
            serde_json::json!({
                "command": "yes '0123456789' | head -c 105000",
                "timeout_seconds": 10
            }),
            &ctx,
        )
        .await
        .unwrap();

    if let ToolEffect::Output(output) = result {
        assert!(output.content.contains("[stdout truncated]"));
        assert!(!output.content.contains("[stderr]")); // Should be no stderr
        // Content length should be roughly 100KB + message
        assert!(output.content.len() < 105000);
        assert!(output.content.len() >= 100000);
    } else {
        panic!("Expected ToolEffect::Output");
    }
}
