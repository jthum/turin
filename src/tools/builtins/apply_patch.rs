use async_trait::async_trait;
use serde::Deserialize;
use serde_json::Value;
use std::path::Path;

use crate::tools::{
    Tool, ToolContext, ToolEffect, ToolError, ToolOutput, is_safe_path, parse_args,
};

pub struct ApplyPatchTool;

#[derive(Deserialize)]
struct ApplyPatchArgs {
    patch: String,
}

#[derive(Debug)]
enum PatchOperation {
    Add {
        path: String,
        lines: Vec<String>,
    },
    Delete {
        path: String,
    },
    Update {
        path: String,
        move_to: Option<String>,
        hunks: Vec<PatchHunk>,
    },
}

#[derive(Debug)]
struct PatchHunk {
    lines: Vec<HunkLine>,
}

#[derive(Debug)]
enum HunkLine {
    Context(String),
    Delete(String),
    Add(String),
}

#[async_trait]
impl Tool for ApplyPatchTool {
    fn name(&self) -> &str {
        "apply_patch"
    }

    fn description(&self) -> &str {
        "Apply a structured multi-file patch. Supports add, delete, update, and move operations with context-aware hunks."
    }

    fn parameters_schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": "Patch text using Turin apply_patch format beginning with '*** Begin Patch' and ending with '*** End Patch'"
                }
            },
            "required": ["patch"]
        })
    }

    async fn execute(&self, params: Value, ctx: &ToolContext) -> Result<ToolEffect, ToolError> {
        let args: ApplyPatchArgs = parse_args(params)?;
        let operations = parse_patch(&args.patch)?;
        let mut summaries = Vec::new();

        for operation in operations {
            summaries.push(apply_operation(ctx, operation).await?);
        }

        Ok(ToolEffect::Output(ToolOutput {
            content: summaries.join("\n"),
            metadata: serde_json::json!({
                "operations": summaries.len(),
            }),
        }))
    }
}

async fn apply_operation(
    ctx: &ToolContext,
    operation: PatchOperation,
) -> Result<String, ToolError> {
    match operation {
        PatchOperation::Add { path, lines } => {
            let target = is_safe_path(&ctx.workspace_root, Path::new(&path))?;
            if tokio::fs::try_exists(&target).await.map_err(|e| {
                ToolError::ExecutionError(format!("Failed to inspect {}: {}", target.display(), e))
            })? {
                return Err(ToolError::ExecutionError(format!(
                    "Cannot add '{}': file already exists",
                    target.display()
                )));
            }
            if let Some(parent) = target.parent() {
                tokio::fs::create_dir_all(parent).await.map_err(|e| {
                    ToolError::ExecutionError(format!(
                        "Failed to create parent directories for {}: {}",
                        target.display(),
                        e
                    ))
                })?;
            }
            let content = render_lines(&lines, !lines.is_empty());
            tokio::fs::write(&target, content).await.map_err(|e| {
                ToolError::ExecutionError(format!("Failed to write {}: {}", target.display(), e))
            })?;
            Ok(format!("Added {}", target.display()))
        }
        PatchOperation::Delete { path } => {
            let target = is_safe_path(&ctx.workspace_root, Path::new(&path))?;
            tokio::fs::remove_file(&target).await.map_err(|e| {
                ToolError::ExecutionError(format!("Failed to delete {}: {}", target.display(), e))
            })?;
            Ok(format!("Deleted {}", target.display()))
        }
        PatchOperation::Update {
            path,
            move_to,
            hunks,
        } => {
            let source = is_safe_path(&ctx.workspace_root, Path::new(&path))?;
            let original = tokio::fs::read_to_string(&source).await.map_err(|e| {
                ToolError::ExecutionError(format!("Failed to read {}: {}", source.display(), e))
            })?;
            let (mut lines, trailing_newline) = split_lines(&original);
            let mut cursor = 0usize;

            for hunk in hunks {
                let old_chunk = hunk
                    .lines
                    .iter()
                    .filter_map(|line| match line {
                        HunkLine::Context(text) | HunkLine::Delete(text) => Some(text.clone()),
                        HunkLine::Add(_) => None,
                    })
                    .collect::<Vec<_>>();
                let new_chunk = hunk
                    .lines
                    .iter()
                    .filter_map(|line| match line {
                        HunkLine::Context(text) | HunkLine::Add(text) => Some(text.clone()),
                        HunkLine::Delete(_) => None,
                    })
                    .collect::<Vec<_>>();

                if old_chunk.is_empty() {
                    let insert_at = cursor.min(lines.len());
                    lines.splice(insert_at..insert_at, new_chunk.clone());
                    cursor = insert_at + new_chunk.len();
                    continue;
                }

                let start = find_subsequence(&lines, &old_chunk, cursor)
                    .or_else(|| find_subsequence(&lines, &old_chunk, 0))
                    .ok_or_else(|| {
                        ToolError::ExecutionError(format!(
                            "Patch hunk did not match {}",
                            source.display()
                        ))
                    })?;
                let end = start + old_chunk.len();
                lines.splice(start..end, new_chunk.clone());
                cursor = start + new_chunk.len();
            }

            let target = if let Some(move_to) = move_to {
                let target = is_safe_path(&ctx.workspace_root, Path::new(&move_to))?;
                if target != source
                    && tokio::fs::try_exists(&target).await.map_err(|e| {
                        ToolError::ExecutionError(format!(
                            "Failed to inspect {}: {}",
                            target.display(),
                            e
                        ))
                    })?
                {
                    return Err(ToolError::ExecutionError(format!(
                        "Cannot move patch target to '{}': file already exists",
                        target.display()
                    )));
                }
                target
            } else {
                source.clone()
            };

            if let Some(parent) = target.parent() {
                tokio::fs::create_dir_all(parent).await.map_err(|e| {
                    ToolError::ExecutionError(format!(
                        "Failed to create parent directories for {}: {}",
                        target.display(),
                        e
                    ))
                })?;
            }

            let content = render_lines(&lines, trailing_newline);
            tokio::fs::write(&target, content).await.map_err(|e| {
                ToolError::ExecutionError(format!("Failed to write {}: {}", target.display(), e))
            })?;

            if target != source {
                tokio::fs::remove_file(&source).await.map_err(|e| {
                    ToolError::ExecutionError(format!(
                        "Failed to remove original {} after move: {}",
                        source.display(),
                        e
                    ))
                })?;
                Ok(format!(
                    "Updated {} and moved to {}",
                    source.display(),
                    target.display()
                ))
            } else {
                Ok(format!("Updated {}", source.display()))
            }
        }
    }
}

fn parse_patch(patch: &str) -> Result<Vec<PatchOperation>, ToolError> {
    let lines = patch.lines().collect::<Vec<_>>();
    if lines.first().copied() != Some("*** Begin Patch") {
        return Err(ToolError::InvalidParams(
            "patch must begin with '*** Begin Patch'".to_string(),
        ));
    }
    if lines.last().copied() != Some("*** End Patch") {
        return Err(ToolError::InvalidParams(
            "patch must end with '*** End Patch'".to_string(),
        ));
    }

    let mut operations = Vec::new();
    let mut index = 1usize;
    while index + 1 < lines.len() {
        let line = lines[index];
        if let Some(path) = line.strip_prefix("*** Add File: ") {
            index += 1;
            let mut add_lines = Vec::new();
            while index + 1 < lines.len() && !is_patch_header(lines[index]) {
                let raw = lines[index];
                let Some(text) = raw.strip_prefix('+') else {
                    return Err(ToolError::InvalidParams(format!(
                        "add file line must start with '+': {}",
                        raw
                    )));
                };
                add_lines.push(text.to_string());
                index += 1;
            }
            operations.push(PatchOperation::Add {
                path: path.to_string(),
                lines: add_lines,
            });
            continue;
        }

        if let Some(path) = line.strip_prefix("*** Delete File: ") {
            operations.push(PatchOperation::Delete {
                path: path.to_string(),
            });
            index += 1;
            continue;
        }

        if let Some(path) = line.strip_prefix("*** Update File: ") {
            index += 1;
            let mut move_to = None;
            if index + 1 < lines.len() {
                move_to = lines[index]
                    .strip_prefix("*** Move to: ")
                    .map(ToOwned::to_owned);
                if move_to.is_some() {
                    index += 1;
                }
            }

            let mut hunks = Vec::new();
            while index + 1 < lines.len() && !is_patch_header(lines[index]) {
                let header = lines[index];
                if !header.starts_with("@@") {
                    return Err(ToolError::InvalidParams(format!(
                        "update hunk must start with '@@': {}",
                        header
                    )));
                }
                index += 1;

                let mut hunk_lines = Vec::new();
                while index + 1 < lines.len()
                    && !lines[index].starts_with("@@")
                    && !is_patch_header(lines[index])
                {
                    let raw = lines[index];
                    if raw == "*** End of File" {
                        index += 1;
                        continue;
                    }
                    let mut chars = raw.chars();
                    let prefix = chars
                        .next()
                        .ok_or_else(|| ToolError::InvalidParams("empty hunk line".to_string()))?;
                    let text = chars.as_str().to_string();
                    let line = match prefix {
                        ' ' => HunkLine::Context(text),
                        '-' => HunkLine::Delete(text),
                        '+' => HunkLine::Add(text),
                        _ => {
                            return Err(ToolError::InvalidParams(format!(
                                "invalid hunk line prefix '{}'",
                                prefix
                            )));
                        }
                    };
                    hunk_lines.push(line);
                    index += 1;
                }

                if hunk_lines.is_empty() {
                    return Err(ToolError::InvalidParams(
                        "update hunk must contain at least one change or context line".to_string(),
                    ));
                }
                hunks.push(PatchHunk { lines: hunk_lines });
            }

            if hunks.is_empty() {
                return Err(ToolError::InvalidParams(format!(
                    "update file '{}' must contain at least one hunk",
                    path
                )));
            }

            operations.push(PatchOperation::Update {
                path: path.to_string(),
                move_to,
                hunks,
            });
            continue;
        }

        return Err(ToolError::InvalidParams(format!(
            "unexpected patch header '{}'",
            line
        )));
    }

    Ok(operations)
}

fn is_patch_header(line: &str) -> bool {
    line.starts_with("*** Add File: ")
        || line.starts_with("*** Delete File: ")
        || line.starts_with("*** Update File: ")
        || line == "*** End Patch"
}

fn split_lines(content: &str) -> (Vec<String>, bool) {
    let trailing_newline = content.ends_with('\n');
    let mut lines = content
        .split('\n')
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if trailing_newline {
        lines.pop();
    }
    (lines, trailing_newline)
}

fn render_lines(lines: &[String], trailing_newline: bool) -> String {
    let mut out = lines.join("\n");
    if trailing_newline && (!lines.is_empty() || out.is_empty()) {
        out.push('\n');
    }
    out
}

fn find_subsequence(haystack: &[String], needle: &[String], start: usize) -> Option<usize> {
    if needle.is_empty() {
        return Some(start.min(haystack.len()));
    }
    if haystack.len() < needle.len() {
        return None;
    }
    (start..=haystack.len().saturating_sub(needle.len()))
        .find(|offset| haystack[*offset..(*offset + needle.len())] == *needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn applies_multi_hunk_update() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("note.txt");
        std::fs::write(&file_path, "alpha\nbeta\ngamma\ndelta\n").unwrap();
        let ctx = ToolContext {
            workspace_root: dir.path().to_path_buf(),
            session_id: "s".into(),
            agent_id: "a".into(),
            store_manager: None,
            embedding_provider: None,
            config: None,
            allowed_native_tools: std::sync::Arc::new(std::collections::BTreeSet::from([
                "apply_patch".to_string(),
            ])),
            tools: std::sync::Arc::new(turin_types::ToolsConfig::default()),
        };

        let tool = ApplyPatchTool;
        tool.execute(
            serde_json::json!({
                "patch": "*** Begin Patch\n*** Update File: note.txt\n@@\n alpha\n-beta\n+beta2\n gamma\n@@\n gamma\n-delta\n+delta2\n*** End Patch"
            }),
            &ctx,
        )
        .await
        .unwrap();

        let updated = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(updated, "alpha\nbeta2\ngamma\ndelta2\n");
    }

    #[tokio::test]
    async fn adds_new_file() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolContext {
            workspace_root: dir.path().to_path_buf(),
            session_id: "s".into(),
            agent_id: "a".into(),
            store_manager: None,
            embedding_provider: None,
            config: None,
            allowed_native_tools: std::sync::Arc::new(std::collections::BTreeSet::from([
                "apply_patch".to_string(),
            ])),
            tools: std::sync::Arc::new(turin_types::ToolsConfig::default()),
        };

        let tool = ApplyPatchTool;
        tool.execute(
            serde_json::json!({
                "patch": "*** Begin Patch\n*** Add File: hello.txt\n+hello\n+world\n*** End Patch"
            }),
            &ctx,
        )
        .await
        .unwrap();

        let created = std::fs::read_to_string(dir.path().join("hello.txt")).unwrap();
        assert_eq!(created, "hello\nworld\n");
    }
}
