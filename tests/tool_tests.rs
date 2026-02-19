//! Tool security and boundary tests.
//!
//! Tests for path traversal prevention, workspace escaping,
//! and tool error handling across all built-in tools.

use tempfile::TempDir;
use turin::tools::{Tool, ToolContext, ToolError};

// ─── Helpers ────────────────────────────────────────────────────

fn make_ctx(dir: &std::path::Path) -> ToolContext {
    ToolContext {
        workspace_root: dir.to_path_buf(),
        session_id: "test-session".to_string(),
    }
}

// ─── Path Traversal Tests ───────────────────────────────────────

mod read_file_security {
    use super::*;
    use turin::tools::builtins::read_file::ReadFileTool;

    #[tokio::test]
    async fn rejects_parent_traversal() {
        let dir = TempDir::new().unwrap();
        let tool = ReadFileTool;
        let ctx = make_ctx(dir.path());

        let result: Result<_, ToolError> = tool
            .execute(serde_json::json!({ "path": "../../../etc/passwd" }), &ctx)
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::PermissionDenied(_)));
    }

    #[tokio::test]
    async fn rejects_absolute_outside_workspace() {
        let dir = TempDir::new().unwrap();
        let tool = ReadFileTool;
        let ctx = make_ctx(dir.path());

        let result: Result<_, ToolError> = tool
            .execute(serde_json::json!({ "path": "/etc/passwd" }), &ctx)
            .await;

        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, ToolError::PermissionDenied(_)));
    }

    #[tokio::test]
    async fn allows_nested_relative_path() {
        let dir = TempDir::new().unwrap();
        let nested = dir.path().join("sub/dir");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(nested.join("file.txt"), "nested content").unwrap();

        let tool = ReadFileTool;
        let ctx = make_ctx(dir.path());

        let result: Result<_, ToolError> = tool
            .execute(serde_json::json!({ "path": "sub/dir/file.txt" }), &ctx)
            .await;

        assert!(result.is_ok());
        if let Ok(turin::tools::ToolEffect::Output(output)) = result {
            assert_eq!(output.content, "nested content");
        }
    }

    #[tokio::test]
    async fn handles_missing_params() {
        let dir = TempDir::new().unwrap();
        let tool = ReadFileTool;
        let ctx = make_ctx(dir.path());

        let result: Result<_, ToolError> = tool.execute(serde_json::json!({}), &ctx).await;
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::InvalidParams(_)));
    }
}

mod write_file_security {
    use super::*;
    use turin::tools::builtins::write_file::WriteFileTool;

    #[tokio::test]
    async fn rejects_parent_traversal() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFileTool;
        let ctx = make_ctx(dir.path());

        let result: Result<_, ToolError> = tool
            .execute(
                serde_json::json!({ "path": "../../evil.txt", "content": "pwned" }),
                &ctx,
            )
            .await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
    }

    #[tokio::test]
    async fn rejects_absolute_outside_workspace() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFileTool;
        let ctx = make_ctx(dir.path());

        let result: Result<_, ToolError> = tool
            .execute(
                serde_json::json!({ "path": "/tmp/evil.txt", "content": "pwned" }),
                &ctx,
            )
            .await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
    }

    #[tokio::test]
    async fn writes_within_workspace() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFileTool;
        let ctx = make_ctx(dir.path());

        let result: Result<_, ToolError> = tool
            .execute(
                serde_json::json!({ "path": "output.txt", "content": "safe" }),
                &ctx,
            )
            .await;

        assert!(result.is_ok());
        let written = std::fs::read_to_string(dir.path().join("output.txt")).unwrap();
        assert_eq!(written, "safe");
    }
}

mod edit_file_security {
    use super::*;
    use turin::tools::builtins::edit_file::EditFileTool;

    #[tokio::test]
    async fn rejects_parent_traversal() {
        let dir = TempDir::new().unwrap();
        let tool = EditFileTool;
        let ctx = make_ctx(dir.path());

        let result: Result<_, ToolError> = tool
            .execute(
                serde_json::json!({
                    "path": "../../etc/hosts",
                    "old_text": "127.0.0.1",
                    "new_text": "evil"
                }),
                &ctx,
            )
            .await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
    }

    #[tokio::test]
    async fn rejects_nonexistent_file() {
        let dir = TempDir::new().unwrap();
        let tool = EditFileTool;
        let ctx = make_ctx(dir.path());

        let result: Result<_, ToolError> = tool
            .execute(
                serde_json::json!({
                    "path": "nonexistent.txt",
                    "old_text": "foo",
                    "new_text": "bar"
                }),
                &ctx,
            )
            .await;

        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ToolError::ExecutionError(_)));
    }
}

mod shell_exec_security {
    use super::*;
    use turin::tools::builtins::shell_exec::ShellExecTool;

    #[tokio::test]
    async fn rejects_cwd_traversal() {
        let dir = TempDir::new().unwrap();
        let tool = ShellExecTool;
        let ctx = make_ctx(dir.path());

        let result: Result<_, ToolError> = tool
            .execute(
                serde_json::json!({ "command": "ls", "cwd": "../../" }),
                &ctx,
            )
            .await;

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            ToolError::PermissionDenied(_)
        ));
    }

    #[tokio::test]
    async fn captures_multi_line_output() {
        let dir = TempDir::new().unwrap();
        let tool = ShellExecTool;
        let ctx = make_ctx(dir.path());

        let result: Result<_, ToolError> = tool
            .execute(
                serde_json::json!({ "command": "echo -e 'line1\nline2\nline3'" }),
                &ctx,
            )
            .await;
        let result = result.unwrap();

        if let turin::tools::ToolEffect::Output(output) = result {
            assert!(output.content.contains("line1"));
            assert!(output.content.contains("line3"));
        } else {
            panic!("Expected ToolEffect::Output");
        }
    }

    #[tokio::test]
    async fn timeout_kills_process() {
        let dir = TempDir::new().unwrap();
        let tool = ShellExecTool;
        let ctx = make_ctx(dir.path());

        let start = std::time::Instant::now();
        let result: Result<_, ToolError> = tool
            .execute(
                serde_json::json!({ "command": "sleep 60", "timeout_secs": 1 }),
                &ctx,
            )
            .await;

        let elapsed = start.elapsed();
        assert!(result.is_err());
        assert!(
            elapsed.as_secs() < 5,
            "Timeout took too long: {:?}",
            elapsed
        );
        assert!(result.unwrap_err().to_string().contains("timed out"));
    }
}
