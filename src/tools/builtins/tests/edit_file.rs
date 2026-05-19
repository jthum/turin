use super::*;
use crate::tools::ToolEffect;
use tempfile::TempDir;

#[tokio::test]
async fn test_edit_file() {
    let dir = TempDir::new().unwrap();
    let file_path = dir.path().join("test.txt");
    std::fs::write(&file_path, "hello world").unwrap();

    let tool = EditFileTool;
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
            serde_json::json!({
                "path": "test.txt",
                "old_text": "world",
                "new_text": "rust"
            }),
            &ctx,
        )
        .await
        .unwrap();

    if let ToolEffect::Output(output) = result {
        assert!(output.content.contains("Successfully edited"));
    } else {
        panic!("Expected ToolEffect::Output");
    }
    let content = std::fs::read_to_string(&file_path).unwrap();
    assert_eq!(content, "hello rust");
}

#[tokio::test]
async fn test_edit_file_not_found_text() {
    let dir = TempDir::new().unwrap();
    let file_path = dir.path().join("test.txt");
    std::fs::write(&file_path, "hello world").unwrap();

    let tool = EditFileTool;
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
            serde_json::json!({
                "path": "test.txt",
                "old_text": "nonexistent",
                "new_text": "rust"
            }),
            &ctx,
        )
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_edit_file_multiple_matches() {
    let dir = TempDir::new().unwrap();
    let file_path = dir.path().join("test.txt");
    std::fs::write(&file_path, "aaa aaa aaa").unwrap();

    let tool = EditFileTool;
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
            serde_json::json!({
                "path": "test.txt",
                "old_text": "aaa",
                "new_text": "bbb"
            }),
            &ctx,
        )
        .await;

    assert!(result.is_err());
    // Original file should be unchanged
    let content = std::fs::read_to_string(&file_path).unwrap();
    assert_eq!(content, "aaa aaa aaa");
}
