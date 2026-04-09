use inference_sdk_core::{InferenceEvent, SdkError, StopReason, validate_event_sequence};
use openai_sdk::normalization::OpenAiStreamAdapter;
use openai_sdk::types::chat::{
    ChatCompletionChunk, ChatRole, ChunkChoice, ChunkDelta, ChunkFunctionCall, ChunkToolCall, Usage,
};

fn make_chunk(
    delta: ChunkDelta,
    finish_reason: Option<&str>,
    usage: Option<Usage>,
    choices: Option<Vec<ChunkChoice>>,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: "chk_contract".to_string(),
        object: "chat.completion.chunk".to_string(),
        created: 1_700_000_000,
        model: "gpt-4o".to_string(),
        choices: choices.unwrap_or_else(|| {
            vec![ChunkChoice {
                index: 0,
                delta,
                finish_reason: finish_reason.map(str::to_string),
                logprobs: None,
            }]
        }),
        system_fingerprint: None,
        usage,
    }
}

#[test]
fn test_openai_provider_contract_tool_stream_order_and_message_end() {
    let mut adapter = OpenAiStreamAdapter::new();
    let mut out: Vec<Result<InferenceEvent, SdkError>> = Vec::new();

    let start_tool_chunk = make_chunk(
        ChunkDelta {
            role: Some(ChatRole::Assistant),
            content: None,
            tool_calls: Some(vec![ChunkToolCall {
                index: 0,
                id: Some("call_1".to_string()),
                call_type: Some("function".to_string()),
                function: Some(ChunkFunctionCall {
                    name: Some("weather".to_string()),
                    arguments: Some("{\"city\":\"S".to_string()),
                }),
            }]),
        },
        None,
        None,
        None,
    );
    out.extend(adapter.process_chunk(start_tool_chunk));

    let tool_delta_chunk = make_chunk(
        ChunkDelta {
            role: None,
            content: None,
            tool_calls: Some(vec![ChunkToolCall {
                index: 0,
                id: None,
                call_type: None,
                function: Some(ChunkFunctionCall {
                    name: None,
                    arguments: Some("F\"}".to_string()),
                }),
            }]),
        },
        None,
        None,
        None,
    );
    out.extend(adapter.process_chunk(tool_delta_chunk));

    let finish_chunk = make_chunk(
        ChunkDelta {
            role: None,
            content: None,
            tool_calls: None,
        },
        Some("tool_calls"),
        None,
        None,
    );
    out.extend(adapter.process_chunk(finish_chunk));

    // OpenAI usage is emitted as a final chunk with no choices.
    let usage_chunk = make_chunk(
        ChunkDelta {
            role: None,
            content: None,
            tool_calls: None,
        },
        None,
        Some(Usage {
            prompt_tokens: 11,
            completion_tokens: 22,
            total_tokens: 33,
        }),
        Some(vec![]),
    );
    out.extend(adapter.process_chunk(usage_chunk));

    let events: Vec<InferenceEvent> = out.into_iter().collect::<Result<_, _>>().unwrap();
    validate_event_sequence(&events).expect("event sequence must satisfy core contract");

    assert!(matches!(events[0], InferenceEvent::MessageStart { .. }));
    assert!(matches!(events[1], InferenceEvent::ToolCallStart { .. }));
    assert!(matches!(events[2], InferenceEvent::ToolCallDelta { .. }));
    assert!(matches!(events[3], InferenceEvent::ToolCallDelta { .. }));
    assert!(matches!(
        events[4],
        InferenceEvent::MessageEnd {
            input_tokens: 11,
            output_tokens: 22,
            stop_reason: Some(StopReason::ToolUse)
        }
    ));
}

#[tokio::test]
async fn test_openai_provider_contract_propagates_stream_errors() {
    let stream = Box::pin(futures_util::stream::iter(vec![Err(
        SdkError::ProviderError("boom".to_string()),
    )]));

    let result = inference_sdk_core::InferenceResult::from_stream(stream).await;
    assert!(matches!(
        result,
        Err(SdkError::ProviderError(ref msg)) if msg == "boom"
    ));
}
