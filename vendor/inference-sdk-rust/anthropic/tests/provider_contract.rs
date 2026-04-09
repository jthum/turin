use anthropic_sdk::normalization::AnthropicStreamAdapter;
use anthropic_sdk::types::message::{
    ContentBlock, ContentBlockDelta, ErrorDetails, MessageDelta, MessageDeltaUsage,
    MessageResponse, Role, StreamEvent, Usage,
};
use inference_sdk_core::{InferenceEvent, SdkError, StopReason, validate_event_sequence};
use serde_json::json;

#[test]
fn test_anthropic_provider_contract_tool_stream_order_and_message_end() {
    let mut adapter = AnthropicStreamAdapter::new();
    let mut out: Vec<Result<InferenceEvent, SdkError>> = Vec::new();

    out.extend(adapter.process_event(StreamEvent::MessageStart {
        message: MessageResponse {
            id: "msg_1".to_string(),
            response_type: "message".to_string(),
            role: Role::Assistant,
            content: vec![],
            model: "claude-3-5-sonnet".to_string(),
            stop_reason: None,
            stop_sequence: None,
            usage: Usage {
                input_tokens: 13,
                output_tokens: 0,
            },
        },
    }));

    out.extend(adapter.process_event(StreamEvent::ContentBlockStart {
        index: 0,
        content_block: ContentBlock::ToolUse {
            id: "call_1".to_string(),
            name: "weather".to_string(),
            input: json!({}),
        },
    }));

    out.extend(adapter.process_event(StreamEvent::ContentBlockDelta {
        index: 0,
        delta: ContentBlockDelta::InputJsonDelta {
            partial_json: "{\"city\":\"S".to_string(),
        },
    }));
    out.extend(adapter.process_event(StreamEvent::ContentBlockDelta {
        index: 0,
        delta: ContentBlockDelta::InputJsonDelta {
            partial_json: "F\"}".to_string(),
        },
    }));

    out.extend(adapter.process_event(StreamEvent::MessageDelta {
        delta: MessageDelta {
            stop_reason: Some("tool_use".to_string()),
            stop_sequence: None,
        },
        usage: MessageDeltaUsage { output_tokens: 21 },
    }));

    let events: Vec<InferenceEvent> = out.into_iter().collect::<Result<_, _>>().unwrap();
    validate_event_sequence(&events).expect("event sequence must satisfy core contract");

    assert!(matches!(events[0], InferenceEvent::MessageStart { .. }));
    assert!(matches!(events[1], InferenceEvent::ToolCallStart { .. }));
    assert!(matches!(events[2], InferenceEvent::ToolCallDelta { .. }));
    assert!(matches!(events[3], InferenceEvent::ToolCallDelta { .. }));
    assert!(matches!(
        events[4],
        InferenceEvent::MessageEnd {
            input_tokens: 13,
            output_tokens: 21,
            stop_reason: Some(StopReason::ToolUse)
        }
    ));
}

#[test]
fn test_anthropic_provider_contract_maps_provider_error_to_err() {
    let mut adapter = AnthropicStreamAdapter::new();
    let events = adapter.process_event(StreamEvent::Error {
        error: ErrorDetails {
            error_type: "invalid_request_error".to_string(),
            message: "boom".to_string(),
        },
    });

    assert_eq!(events.len(), 1);
    assert!(matches!(
        events[0],
        Err(SdkError::ProviderError(ref msg)) if msg == "boom"
    ));
}
