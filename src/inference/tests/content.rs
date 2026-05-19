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
