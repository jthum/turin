use anthropic_sdk::types::message::StreamEvent;

#[test]
fn deserializes_thinking_content_block_without_signature() {
    let json = r#"{
        "type": "content_block_start",
        "index": 0,
        "content_block": {
            "type": "thinking",
            "thinking": "chain of thought omitted"
        }
    }"#;

    let event: StreamEvent =
        serde_json::from_str(json).expect("MiniMax-compatible thinking block should deserialize");
    match event {
        StreamEvent::ContentBlockStart { .. } => {}
        other => panic!("unexpected event: {other:?}"),
    }
}

#[test]
fn deserializes_message_start_with_thinking_content_without_signature() {
    let json = r#"{
        "type": "message_start",
        "message": {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "chain of thought omitted"
                }
            ],
            "model": "minimax-anthropic",
            "stop_reason": null,
            "stop_sequence": null,
            "usage": { "input_tokens": 1, "output_tokens": 1 }
        }
    }"#;

    let event: StreamEvent =
        serde_json::from_str(json).expect("MiniMax-compatible message_start should deserialize");
    match event {
        StreamEvent::MessageStart { .. } => {}
        other => panic!("unexpected event: {other:?}"),
    }
}
