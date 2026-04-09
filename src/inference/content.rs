use anyhow::{Context, Result, anyhow};
use reqwest::header::CONTENT_TYPE;
use std::path::Path;

use crate::inference::provider::{InferenceContent, InferenceMessage, InferenceRole};
use turin_types::TaskInputContent;

pub fn infer_prompt_from_messages(messages: &[InferenceMessage]) -> Option<String> {
    let last = messages.last()?;
    if last.role != InferenceRole::User {
        return None;
    }
    infer_prompt_from_content(&last.content)
}

pub fn infer_prompt_from_content(content: &[InferenceContent]) -> Option<String> {
    let text = content
        .iter()
        .filter_map(|part| match part {
            InferenceContent::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");

    if text.is_empty() { None } else { Some(text) }
}

pub fn replace_user_text_content(
    content: &[InferenceContent],
    prompt: Option<&str>,
) -> Vec<InferenceContent> {
    let mut next = Vec::new();
    if let Some(prompt) = prompt {
        next.push(InferenceContent::Text {
            text: prompt.to_string(),
        });
    }
    next.extend(
        content
            .iter()
            .filter(|part| !matches!(part, InferenceContent::Text { .. }))
            .cloned(),
    );
    next
}

pub fn summarize_content_for_display(content: &[InferenceContent]) -> String {
    let mut summary = String::new();

    for part in content {
        match part {
            InferenceContent::Text { text } => summary.push_str(text),
            InferenceContent::Image {
                name,
                url,
                local_path,
                ..
            } => {
                let label = name
                    .as_deref()
                    .or(url.as_deref())
                    .or(local_path.as_deref())
                    .unwrap_or("image");
                summary.push_str(&format!("[Image: {label}] "));
            }
            InferenceContent::File {
                name,
                url,
                local_path,
                ..
            } => {
                let label = name
                    .as_deref()
                    .or(url.as_deref())
                    .or(local_path.as_deref())
                    .unwrap_or("file");
                summary.push_str(&format!("[File: {label}] "));
            }
            InferenceContent::ToolUse { name, .. } => {
                summary.push_str(&format!("[Tool Call: {name}] "));
            }
            InferenceContent::ToolResult { .. } => summary.push_str("[Tool Result] "),
            InferenceContent::Thinking { .. } => summary.push_str("[Thinking] "),
        }
    }

    summary
}

pub fn encode_content_json(content: &[InferenceContent]) -> serde_json::Value {
    serde_json::to_value(content).unwrap_or_else(|_| serde_json::Value::Array(Vec::new()))
}

pub fn decode_content_json(value: serde_json::Value) -> Result<Vec<InferenceContent>> {
    serde_json::from_value(value).context("Invalid persisted message content payload")
}

pub async fn materialize_task_input_content(
    content: &[TaskInputContent],
    media_root: &Path,
) -> Result<Vec<InferenceContent>> {
    let mut materialized = Vec::with_capacity(content.len());
    if !content.is_empty() {
        tokio::fs::create_dir_all(media_root)
            .await
            .with_context(|| format!("Failed to create '{}'", media_root.display()))?;
    }

    for part in content {
        match part {
            TaskInputContent::Text { text } => {
                materialized.push(InferenceContent::Text { text: text.clone() });
            }
            TaskInputContent::Image {
                name,
                content_type,
                url,
                local_path,
                detail,
            } => {
                let fetched =
                    materialize_attachment(name, content_type, url, local_path, media_root).await?;
                materialized.push(InferenceContent::Image {
                    name: name.clone(),
                    content_type: fetched.content_type.or_else(|| content_type.clone()),
                    url: url.clone(),
                    local_path: Some(fetched.local_path),
                    detail: detail.clone(),
                });
            }
            TaskInputContent::File {
                name,
                content_type,
                url,
                local_path,
            } => {
                let fetched =
                    materialize_attachment(name, content_type, url, local_path, media_root).await?;
                materialized.push(InferenceContent::File {
                    name: name.clone(),
                    content_type: fetched.content_type.or_else(|| content_type.clone()),
                    url: url.clone(),
                    local_path: Some(fetched.local_path),
                });
            }
        }
    }

    Ok(materialized)
}

struct MaterializedAttachment {
    local_path: String,
    content_type: Option<String>,
}

async fn materialize_attachment(
    name: &Option<String>,
    content_type: &Option<String>,
    url: &Option<String>,
    local_path: &Option<String>,
    media_root: &Path,
) -> Result<MaterializedAttachment> {
    let extension = attachment_extension(name.as_deref(), url.as_deref(), local_path.as_deref());
    let target_name = if let Some(extension) = extension {
        format!("{}.{}", uuid::Uuid::now_v7().simple(), extension)
    } else {
        uuid::Uuid::now_v7().simple().to_string()
    };
    let target_path = media_root.join(target_name);

    if let Some(source_path) = local_path {
        tokio::fs::copy(source_path, &target_path)
            .await
            .with_context(|| {
                format!(
                    "Failed to copy attachment '{}' to '{}'",
                    source_path,
                    target_path.display()
                )
            })?;
        return Ok(MaterializedAttachment {
            local_path: target_path.display().to_string(),
            content_type: content_type.clone(),
        });
    }

    if let Some(url) = url {
        let response = reqwest::get(url)
            .await
            .with_context(|| format!("Failed to download attachment '{}'", url))?
            .error_for_status()
            .with_context(|| format!("Attachment request failed for '{}'", url))?;
        let fetched_content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(|value| value.split(';').next().unwrap_or(value).trim().to_string());
        let bytes = response
            .bytes()
            .await
            .with_context(|| format!("Failed to read attachment body '{}'", url))?;
        tokio::fs::write(&target_path, &bytes)
            .await
            .with_context(|| format!("Failed to write attachment '{}'", target_path.display()))?;
        return Ok(MaterializedAttachment {
            local_path: target_path.display().to_string(),
            content_type: fetched_content_type.or_else(|| content_type.clone()),
        });
    }

    Err(anyhow!(
        "attachment content must include at least one of local_path or url"
    ))
}

fn attachment_extension(
    name: Option<&str>,
    url: Option<&str>,
    local_path: Option<&str>,
) -> Option<String> {
    name.and_then(extract_extension)
        .or_else(|| local_path.and_then(extract_extension))
        .or_else(|| url.and_then(extract_extension))
}

fn extract_extension(value: &str) -> Option<String> {
    let candidate = value.split('?').next().unwrap_or(value);
    Path::new(candidate)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::trim)
        .filter(|ext| !ext.is_empty())
        .map(str::to_ascii_lowercase)
}

pub fn task_content_from_parts(content: &[TaskInputContent]) -> Vec<InferenceContent> {
    content
        .iter()
        .map(|part| match part {
            TaskInputContent::Text { text } => InferenceContent::Text { text: text.clone() },
            TaskInputContent::Image {
                name,
                content_type,
                url,
                local_path,
                detail,
            } => InferenceContent::Image {
                name: name.clone(),
                content_type: content_type.clone(),
                url: url.clone(),
                local_path: local_path.clone(),
                detail: detail.clone(),
            },
            TaskInputContent::File {
                name,
                content_type,
                url,
                local_path,
            } => InferenceContent::File {
                name: name.clone(),
                content_type: content_type.clone(),
                url: url.clone(),
                local_path: local_path.clone(),
            },
        })
        .collect()
}

pub fn task_output_content_from_inference(content: &[InferenceContent]) -> Vec<TaskInputContent> {
    content
        .iter()
        .filter_map(|part| match part {
            InferenceContent::Text { text } => Some(TaskInputContent::Text { text: text.clone() }),
            InferenceContent::Image {
                name,
                content_type,
                url,
                local_path,
                detail,
            } => Some(TaskInputContent::Image {
                name: name.clone(),
                content_type: content_type.clone(),
                url: url.clone(),
                local_path: local_path.clone(),
                detail: detail.clone(),
            }),
            InferenceContent::File {
                name,
                content_type,
                url,
                local_path,
            } => Some(TaskInputContent::File {
                name: name.clone(),
                content_type: content_type.clone(),
                url: url.clone(),
                local_path: local_path.clone(),
            }),
            _ => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn infer_prompt_only_uses_text_parts() {
        let message = InferenceMessage {
            role: InferenceRole::User,
            content: vec![
                InferenceContent::Image {
                    name: Some("diagram.png".to_string()),
                    content_type: Some("image/png".to_string()),
                    url: Some("https://example.test/diagram.png".to_string()),
                    local_path: None,
                    detail: None,
                },
                InferenceContent::Text {
                    text: "inspect this".to_string(),
                },
            ],
            tool_call_id: None,
        };

        assert_eq!(
            infer_prompt_from_messages(&[message]),
            Some("inspect this".to_string())
        );
    }

    #[test]
    fn replace_user_text_content_preserves_non_text_parts() {
        let content = vec![
            InferenceContent::Text {
                text: "old".to_string(),
            },
            InferenceContent::File {
                name: Some("spec.pdf".to_string()),
                content_type: Some("application/pdf".to_string()),
                url: None,
                local_path: Some("/tmp/spec.pdf".to_string()),
            },
        ];

        let replaced = replace_user_text_content(&content, Some("new"));
        assert!(matches!(
            &replaced[0],
            InferenceContent::Text { text } if text == "new"
        ));
        assert!(matches!(
            &replaced[1],
            InferenceContent::File { name: Some(name), .. } if name == "spec.pdf"
        ));
    }

    #[test]
    fn encode_and_decode_content_round_trip_image_and_file_parts() {
        let content = vec![
            InferenceContent::Image {
                name: Some("diagram.png".to_string()),
                content_type: Some("image/png".to_string()),
                url: Some("https://example.test/diagram.png".to_string()),
                local_path: Some("/tmp/diagram.png".to_string()),
                detail: Some("high".to_string()),
            },
            InferenceContent::File {
                name: Some("spec.pdf".to_string()),
                content_type: Some("application/pdf".to_string()),
                url: None,
                local_path: Some("/tmp/spec.pdf".to_string()),
            },
        ];

        let decoded = decode_content_json(encode_content_json(&content)).expect("content decode");
        assert!(matches!(
            &decoded[0],
            InferenceContent::Image {
                name: Some(name),
                detail: Some(detail),
                ..
            } if name == "diagram.png" && detail == "high"
        ));
        assert!(matches!(
            &decoded[1],
            InferenceContent::File { name: Some(name), .. } if name == "spec.pdf"
        ));
    }

    #[test]
    fn task_output_content_from_inference_filters_non_display_parts() {
        let content = vec![
            InferenceContent::Thinking {
                content: "internal".to_string(),
                signature: None,
            },
            InferenceContent::Text {
                text: "done".to_string(),
            },
            InferenceContent::Image {
                name: Some("diagram.png".to_string()),
                content_type: Some("image/png".to_string()),
                url: None,
                local_path: Some("/tmp/diagram.png".to_string()),
                detail: Some("high".to_string()),
            },
            InferenceContent::ToolUse {
                id: "call-1".to_string(),
                name: "shell_exec".to_string(),
                input: serde_json::json!({}),
            },
        ];

        let output = task_output_content_from_inference(&content);
        assert_eq!(output.len(), 2);
        assert!(matches!(
            &output[0],
            TaskInputContent::Text { text } if text == "done"
        ));
        assert!(matches!(
            &output[1],
            TaskInputContent::Image { name: Some(name), .. } if name == "diagram.png"
        ));
    }

    #[tokio::test]
    async fn materialize_task_input_content_copies_local_attachments() {
        let temp = tempdir().expect("tempdir");
        let source = temp.path().join("diagram.png");
        std::fs::write(&source, [1_u8, 2, 3, 4]).expect("fixture image");

        let content = materialize_task_input_content(
            &[TaskInputContent::Image {
                name: Some("diagram.png".to_string()),
                content_type: Some("image/png".to_string()),
                url: None,
                local_path: Some(source.display().to_string()),
                detail: None,
            }],
            &temp.path().join("media"),
        )
        .await
        .expect("materialize content");

        match &content[0] {
            InferenceContent::Image {
                local_path: Some(local_path),
                ..
            } => {
                assert!(Path::new(local_path).exists());
                assert!(local_path.contains("/media/"));
            }
            other => panic!("expected image content, got {other:?}"),
        }
    }
}
