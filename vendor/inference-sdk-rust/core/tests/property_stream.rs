use futures_util::stream;
use inference_sdk_core::{
    InferenceContent, InferenceEvent, InferenceResult, SdkError, StopReason,
    StreamInvariantViolation, validate_event_sequence,
};
use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_validator_rejects_tool_delta_before_tool_start(delta in ".*") {
        let events = vec![
            InferenceEvent::MessageStart {
                role: "assistant".to_string(),
                model: "test-model".to_string(),
                provider_id: "test".to_string(),
            },
            InferenceEvent::ToolCallDelta { delta },
            InferenceEvent::MessageEnd {
                input_tokens: 1,
                output_tokens: 1,
                stop_reason: Some(StopReason::ToolUse),
            },
        ];

        prop_assert!(matches!(
            validate_event_sequence(&events),
            Err(StreamInvariantViolation::ToolCallDeltaBeforeStart)
        ));
    }
}

proptest! {
    #[test]
    fn prop_from_stream_reassembles_tool_json(
        city in "[a-z]{1,12}",
        temp in 0u8..=99,
        chunk_size in 1usize..8,
    ) {
        let expected = serde_json::json!({ "city": city, "temp": temp });
        let json = expected.to_string();
        let deltas: Vec<String> = json
            .as_bytes()
            .chunks(chunk_size)
            .map(|bytes| String::from_utf8(bytes.to_vec()).expect("chunk should be valid utf8"))
            .collect();

        let mut events = vec![
            Ok(InferenceEvent::MessageStart {
                role: "assistant".to_string(),
                model: "test-model".to_string(),
                provider_id: "test".to_string(),
            }),
            Ok(InferenceEvent::ToolCallStart {
                id: "call_1".to_string(),
                name: "weather".to_string(),
            }),
        ];

        for delta in deltas {
            events.push(Ok(InferenceEvent::ToolCallDelta { delta }));
        }

        events.push(Ok(InferenceEvent::MessageEnd {
            input_tokens: 2,
            output_tokens: 3,
            stop_reason: Some(StopReason::ToolUse),
        }));

        let stream = Box::pin(stream::iter(events));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime");
        let result = runtime.block_on(InferenceResult::from_stream(stream));

        let result = result.expect("stream should assemble");
        let tool_input_matches = match result.content.last() {
            Some(InferenceContent::ToolUse { input, .. }) => input == &expected,
            _ => false,
        };
        prop_assert!(tool_input_matches);
    }
}

#[tokio::test]
async fn test_from_stream_errors_when_stream_starts_with_error() {
    let stream = Box::pin(stream::iter(vec![Err(SdkError::ProviderError(
        "boom".to_string(),
    ))]));
    let result = InferenceResult::from_stream(stream).await;
    assert!(matches!(
        result,
        Err(SdkError::ProviderError(ref msg)) if msg == "boom"
    ));
}
