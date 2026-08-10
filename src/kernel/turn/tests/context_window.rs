use super::*;
use crate::inference::provider::InferenceRole;
use crate::kernel::session::ContextCompactionCheckpoint;

fn text_message(role: InferenceRole, text: &str) -> InferenceMessage {
    InferenceMessage {
        role,
        content: vec![InferenceContent::Text {
            text: text.to_string(),
        }],
        tool_call_id: None,
    }
}

#[test]
fn estimate_history_tokens_is_non_zero_for_text() {
    let tokens = estimate_history_input_tokens(
        "System prompt",
        &[text_message(InferenceRole::User, "Hello from Turin")],
    );

    assert!(tokens > 0);
}

#[test]
fn request_breakdown_separates_support_messages_and_reusable_prefix() {
    let messages = vec![
        text_message(InferenceRole::User, "Earlier request"),
        text_message(InferenceRole::Assistant, "Earlier response"),
        text_message(InferenceRole::User, "Current request"),
    ];
    let tools = vec![serde_json::json!({
        "name": "search",
        "description": "Search indexed documents"
    })];

    let estimate = estimate_request_token_breakdown("System prompt", &messages, &tools);

    assert_eq!(
        estimate.total_tokens,
        estimate.system_prompt_tokens + estimate.message_tokens + estimate.tool_definition_tokens
    );
    assert!(estimate.reusable_prefix_tokens < estimate.total_tokens);
    assert!(estimate.reusable_prefix_tokens > estimate.system_prompt_tokens);
    assert!(estimate.estimated_payload_bytes > "System prompt".len());
}

#[test]
fn compaction_truncates_old_tool_results_before_dropping_messages() {
    let messages = vec![
        text_message(InferenceRole::User, "First prompt"),
        InferenceMessage {
            role: InferenceRole::Tool,
            content: vec![InferenceContent::ToolResult {
                tool_use_id: "tool_1".to_string(),
                content: "x".repeat(4_000),
                is_error: false,
            }],
            tool_call_id: None,
        },
        text_message(InferenceRole::User, "Follow-up one"),
        text_message(InferenceRole::Assistant, "Follow-up answer one"),
        text_message(InferenceRole::User, "Follow-up two"),
        text_message(InferenceRole::Assistant, "Follow-up answer two"),
        text_message(InferenceRole::Assistant, "Recent answer"),
        text_message(InferenceRole::User, "Current prompt"),
    ];

    let (compacted, report) = compact_messages_for_input_budget("system", &messages, &[], 512, 384);

    assert!(report.truncated_tool_results > 0);
    assert!(report.applied());
    assert!(report.fits_budget());
    assert_eq!(compacted.len(), messages.len());
    match &compacted[1].content[0] {
        InferenceContent::ToolResult { content, .. } => {
            assert_eq!(content, TRUNCATED_TOOL_RESULT_MARKER);
        }
        other => panic!("expected tool result, got {other:?}"),
    }
}

#[test]
fn compaction_slides_message_window_when_needed() {
    let messages = vec![
        text_message(InferenceRole::User, &"a".repeat(1_000)),
        text_message(InferenceRole::Assistant, &"b".repeat(1_000)),
        text_message(InferenceRole::User, "current prompt"),
    ];

    let (compacted, report) = compact_messages_for_input_budget("system", &messages, &[], 512, 192);

    assert!(report.dropped_messages > 0);
    assert!(report.applied());
    let last = compacted.last().expect("expected last compacted message");
    assert_eq!(last.role, InferenceRole::User);
    match &last.content[0] {
        InferenceContent::Text { text } => assert_eq!(text, "current prompt"),
        other => panic!("expected trailing text message, got {other:?}"),
    }
}

#[test]
fn effective_request_context_injects_checkpoint_and_drops_covered_history() {
    let messages = vec![
        text_message(InferenceRole::User, "old 1"),
        text_message(InferenceRole::Assistant, "old 2"),
        text_message(InferenceRole::User, "recent"),
    ];
    let checkpoint = ContextCompactionCheckpoint {
        summary: "important durable context".to_string(),
        covered_message_count: 2,
        generated_at_turn_index: 4,
        provider_name: "mock".to_string(),
        model: "mock-model".to_string(),
    };

    let effective =
        effective_request_context_from_window("base system", &messages, 0, Some(&checkpoint));

    assert!(
        effective
            .system_prompt
            .contains("important durable context")
    );
    assert_eq!(effective.messages.len(), 1);
    match &effective.messages[0].content[0] {
        InferenceContent::Text { text } => assert_eq!(text, "recent"),
        other => panic!("expected recent text message, got {other:?}"),
    }
}

#[test]
fn effective_request_context_applies_checkpoint_to_hot_window_offset() {
    let messages = vec![
        text_message(InferenceRole::User, "covered hot message"),
        text_message(InferenceRole::Assistant, "recent answer"),
        text_message(InferenceRole::User, "current prompt"),
    ];
    let checkpoint = ContextCompactionCheckpoint {
        summary: "durable context before the hot window".to_string(),
        covered_message_count: 6,
        generated_at_turn_index: 4,
        provider_name: "mock".to_string(),
        model: "mock-model".to_string(),
    };

    let effective =
        effective_request_context_from_window("base system", &messages, 5, Some(&checkpoint));

    assert!(
        effective
            .system_prompt
            .contains("durable context before the hot window")
    );
    assert_eq!(effective.messages.len(), 2);
    match &effective.messages[0].content[0] {
        InferenceContent::Text { text } => assert_eq!(text, "recent answer"),
        other => panic!("expected recent text message, got {other:?}"),
    }
}
