use inference_sdk_core::{
    InferenceEvent, StopReason, StreamInvariantViolation, validate_event_sequence,
};

#[test]
fn test_validate_event_sequence_accepts_valid_order() {
    let events = vec![
        InferenceEvent::MessageStart {
            role: "assistant".to_string(),
            model: "test-model".to_string(),
            provider_id: "test".to_string(),
        },
        InferenceEvent::MessageDelta {
            content: "hello".to_string(),
        },
        InferenceEvent::ToolCallStart {
            id: "call_1".to_string(),
            name: "weather".to_string(),
        },
        InferenceEvent::ToolCallDelta {
            delta: "{\"city\":\"SF\"}".to_string(),
        },
        InferenceEvent::MessageEnd {
            input_tokens: 1,
            output_tokens: 2,
            stop_reason: Some(StopReason::ToolUse),
        },
    ];

    assert!(validate_event_sequence(&events).is_ok());
}

#[test]
fn test_validate_event_sequence_rejects_tool_delta_before_start() {
    let events = vec![
        InferenceEvent::MessageStart {
            role: "assistant".to_string(),
            model: "test-model".to_string(),
            provider_id: "test".to_string(),
        },
        InferenceEvent::ToolCallDelta {
            delta: "{\"bad\":true}".to_string(),
        },
        InferenceEvent::MessageEnd {
            input_tokens: 1,
            output_tokens: 2,
            stop_reason: Some(StopReason::ToolUse),
        },
    ];

    assert!(matches!(
        validate_event_sequence(&events),
        Err(StreamInvariantViolation::ToolCallDeltaBeforeStart)
    ));
}

#[test]
fn test_validate_event_sequence_rejects_missing_message_end() {
    let events = vec![
        InferenceEvent::MessageStart {
            role: "assistant".to_string(),
            model: "test-model".to_string(),
            provider_id: "test".to_string(),
        },
        InferenceEvent::MessageDelta {
            content: "hello".to_string(),
        },
    ];

    assert!(matches!(
        validate_event_sequence(&events),
        Err(StreamInvariantViolation::MissingMessageEnd)
    ));
}

#[test]
fn test_validate_event_sequence_rejects_message_end_before_start() {
    let events = vec![InferenceEvent::MessageEnd {
        input_tokens: 1,
        output_tokens: 2,
        stop_reason: Some(StopReason::EndTurn),
    }];

    assert!(matches!(
        validate_event_sequence(&events),
        Err(StreamInvariantViolation::MessageEndBeforeStart)
    ));
}

#[test]
fn test_validate_event_sequence_rejects_duplicate_message_start() {
    let events = vec![
        InferenceEvent::MessageStart {
            role: "assistant".to_string(),
            model: "test-model".to_string(),
            provider_id: "test".to_string(),
        },
        InferenceEvent::MessageStart {
            role: "assistant".to_string(),
            model: "test-model".to_string(),
            provider_id: "test".to_string(),
        },
    ];

    assert!(matches!(
        validate_event_sequence(&events),
        Err(StreamInvariantViolation::DuplicateMessageStart)
    ));
}
