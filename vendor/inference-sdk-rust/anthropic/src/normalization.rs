use crate::types;
use inference_sdk_core::{
    InferenceContent, InferenceEvent, InferenceRequest, InferenceRole, RequestOptions, SdkError,
    StopReason,
};

pub fn to_anthropic_request(
    req: InferenceRequest,
) -> Result<types::message::MessageRequest, SdkError> {
    if req.response_format.is_some() {
        return Err(SdkError::ConfigError(
            "structured response_format is not supported by the anthropic provider driver"
                .to_string(),
        ));
    }

    let mut messages = Vec::new();

    for msg in req.messages {
        match msg.role {
            InferenceRole::User => {
                let mut content_blocks = Vec::new();
                for content in msg.content {
                    if let InferenceContent::Text { text } = content {
                        content_blocks.push(types::message::ContentBlock::Text { text });
                    }
                }

                if !content_blocks.is_empty() {
                    messages.push(types::message::Message {
                        role: types::message::Role::User,
                        content: types::message::Content::Blocks(content_blocks),
                    });
                }
            }
            InferenceRole::Assistant => {
                let mut content_blocks = Vec::new();
                for content in msg.content {
                    match content {
                        InferenceContent::Text { text } => {
                            content_blocks.push(types::message::ContentBlock::Text { text });
                        }
                        InferenceContent::ToolUse { id, name, input } => {
                            content_blocks.push(types::message::ContentBlock::ToolUse {
                                id,
                                name,
                                input,
                            });
                        }
                        InferenceContent::Thinking { content, signature } => {
                            content_blocks.push(types::message::ContentBlock::Thinking {
                                thinking: content,
                                signature,
                            });
                        }
                        _ => {}
                    }
                }

                if !content_blocks.is_empty() {
                    messages.push(types::message::Message {
                        role: types::message::Role::Assistant,
                        content: types::message::Content::Blocks(content_blocks),
                    });
                }
            }
            InferenceRole::Tool => {
                let mut content_blocks = Vec::new();
                for content in msg.content {
                    if let InferenceContent::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } = content
                    {
                        content_blocks.push(types::message::ContentBlock::ToolResult {
                            tool_use_id,
                            content: Some(types::message::ToolResultContent::Text(content)),
                            is_error: is_error.then_some(true),
                        });
                    }
                }

                if !content_blocks.is_empty() {
                    // Anthropic expects tool results to be sent as a user role message.
                    messages.push(types::message::Message {
                        role: types::message::Role::User,
                        content: types::message::Content::Blocks(content_blocks),
                    });
                }
            }
        }
    }

    let tools: Option<Vec<types::message::Tool>> = req.tools.map(|ts| {
        ts.into_iter()
            .map(|t| types::message::Tool {
                name: t.name,
                description: Some(t.description),
                input_schema: t.input_schema,
            })
            .collect()
    });

    let thinking = req
        .thinking_budget
        .map(|budget| types::message::ThinkingConfig {
            thinking_type: "enabled".to_string(),
            budget_tokens: budget,
        });

    Ok(types::message::MessageRequest::builder()
        .model(req.model)
        .messages(messages)
        .maybe_system(req.system)
        .max_tokens(req.max_tokens.unwrap_or(8192))
        .maybe_temperature(req.temperature)
        .maybe_tools(tools)
        .maybe_thinking(thinking)
        .build())
}

#[derive(Default)]
pub struct AnthropicStreamAdapter {
    input_tokens: u32,
}

impl AnthropicStreamAdapter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn process_event(
        &mut self,
        event: types::message::StreamEvent,
    ) -> Vec<Result<InferenceEvent, SdkError>> {
        match event {
            types::message::StreamEvent::MessageStart { message } => {
                self.input_tokens = message.usage.input_tokens;

                vec![Ok(InferenceEvent::MessageStart {
                    role: "assistant".to_string(),
                    model: message.model,
                    provider_id: "anthropic".to_string(),
                })]
            }
            types::message::StreamEvent::ContentBlockDelta { delta, .. } => match delta {
                types::message::ContentBlockDelta::TextDelta { text } => {
                    vec![Ok(InferenceEvent::MessageDelta { content: text })]
                }
                types::message::ContentBlockDelta::ThinkingDelta { thinking } => {
                    vec![Ok(InferenceEvent::ThinkingDelta { content: thinking })]
                }
                types::message::ContentBlockDelta::SignatureDelta { signature } => {
                    vec![Ok(InferenceEvent::ThinkingSignatureDelta { signature })]
                }
                types::message::ContentBlockDelta::InputJsonDelta { partial_json } => {
                    vec![Ok(InferenceEvent::ToolCallDelta {
                        delta: partial_json,
                    })]
                }
            },
            types::message::StreamEvent::ContentBlockStart {
                content_block: types::message::ContentBlock::ToolUse { id, name, .. },
                ..
            } => vec![Ok(InferenceEvent::ToolCallStart { id, name })],
            types::message::StreamEvent::MessageDelta { delta, usage } => {
                let stop_reason = delta.stop_reason.map(|s| match s.as_str() {
                    "end_turn" => StopReason::EndTurn,
                    "max_tokens" => StopReason::MaxTokens,
                    "tool_use" => StopReason::ToolUse,
                    "stop_sequence" => StopReason::StopSequence,
                    _ => StopReason::Unknown,
                });

                vec![Ok(InferenceEvent::MessageEnd {
                    input_tokens: self.input_tokens,
                    output_tokens: usage.output_tokens,
                    stop_reason,
                })]
            }
            types::message::StreamEvent::Error { error } => {
                vec![Err(SdkError::ProviderError(error.message))]
            }
            _ => vec![],
        }
    }
}

/// Anthropic-specific extensions for `RequestOptions`.
pub trait AnthropicRequestExt {
    /// Add the `anthropic-beta` header to the request options.
    fn beta(self, version: &str) -> Result<RequestOptions, SdkError>;
}

impl AnthropicRequestExt for RequestOptions {
    fn beta(self, version: &str) -> Result<RequestOptions, SdkError> {
        self.with_header("anthropic-beta", version)
            .map_err(|e| SdkError::ConfigError(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::message::{
        MessageDeltaUsage, MessageResponse, StreamEvent, Usage as AnthropicUsage,
    };

    #[test]
    fn test_anthropic_adapter_captures_usage() {
        let mut adapter = AnthropicStreamAdapter::new();

        let start_event = StreamEvent::MessageStart {
            message: MessageResponse {
                id: "msg_123".to_string(),
                response_type: "message".to_string(),
                role: crate::types::message::Role::Assistant,
                content: vec![],
                model: "claude-3-5-sonnet".to_string(),
                stop_reason: None,
                stop_sequence: None,
                usage: AnthropicUsage {
                    input_tokens: 10,
                    output_tokens: 1,
                },
            },
        };

        let events = adapter.process_event(start_event);
        assert_eq!(events.len(), 1);
        if let Ok(InferenceEvent::MessageStart { provider_id, .. }) = &events[0] {
            assert_eq!(provider_id, "anthropic");
        } else {
            panic!("Expected MessageStart");
        }
        assert_eq!(adapter.input_tokens, 10);

        let delta_event = StreamEvent::MessageDelta {
            delta: crate::types::message::MessageDelta {
                stop_reason: Some("end_turn".to_string()),
                stop_sequence: None,
            },
            usage: MessageDeltaUsage { output_tokens: 20 },
        };

        let events = adapter.process_event(delta_event);
        assert_eq!(events.len(), 1);
        if let Ok(InferenceEvent::MessageEnd {
            input_tokens,
            output_tokens,
            stop_reason,
        }) = &events[0]
        {
            assert_eq!(*input_tokens, 10);
            assert_eq!(*output_tokens, 20);
            assert_eq!(*stop_reason, Some(StopReason::EndTurn));
        } else {
            panic!("Expected MessageEnd");
        }
    }

    #[test]
    fn test_anthropic_adapter_emits_tool_argument_deltas() {
        let mut adapter = AnthropicStreamAdapter::new();
        let event = StreamEvent::ContentBlockDelta {
            index: 0,
            delta: types::message::ContentBlockDelta::InputJsonDelta {
                partial_json: "{\"city\":\"S".to_string(),
            },
        };
        let events = adapter.process_event(event);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            Ok(InferenceEvent::ToolCallDelta { ref delta }) if delta == "{\"city\":\"S"
        ));
    }

    #[test]
    fn test_anthropic_adapter_emits_thinking_signature_deltas() {
        let mut adapter = AnthropicStreamAdapter::new();
        let event = StreamEvent::ContentBlockDelta {
            index: 0,
            delta: types::message::ContentBlockDelta::SignatureDelta {
                signature: "sig_abc".to_string(),
            },
        };
        let events = adapter.process_event(event);
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            Ok(InferenceEvent::ThinkingSignatureDelta { ref signature }) if signature == "sig_abc"
        ));
    }
}

#[cfg(test)]
mod request_normalization_tests {
    use super::to_anthropic_request;
    use inference_sdk_core::{InferenceContent, InferenceMessage, InferenceRequest, InferenceRole};

    #[test]
    fn preserves_assistant_thinking_blocks_in_request_history() {
        let req = InferenceRequest::builder()
            .model("test-model")
            .messages(vec![InferenceMessage {
                role: InferenceRole::Assistant,
                content: vec![
                    InferenceContent::Thinking {
                        content: "deliberation".to_string(),
                        signature: Some("sig-123".to_string()),
                    },
                    InferenceContent::ToolUse {
                        id: "toolu_1".to_string(),
                        name: "read_file".to_string(),
                        input: serde_json::json!({ "path": "nonce.txt" }),
                    },
                ],
                tool_call_id: None,
            }])
            .max_tokens(128)
            .build();

        let out = to_anthropic_request(req).expect("request should normalize");
        assert_eq!(out.messages.len(), 1);

        match &out.messages[0].content {
            crate::types::message::Content::Blocks(blocks) => {
                assert!(matches!(
                    &blocks[0],
                    crate::types::message::ContentBlock::Thinking {
                        thinking,
                        signature: Some(signature),
                    } if thinking == "deliberation" && signature == "sig-123"
                ));
                assert!(matches!(
                    &blocks[1],
                    crate::types::message::ContentBlock::ToolUse { id, name, .. }
                    if id == "toolu_1" && name == "read_file"
                ));
            }
            other => panic!("unexpected content form: {other:?}"),
        }
    }
}

#[cfg(test)]
mod tool_result_request_shape_tests {
    use super::to_anthropic_request;
    use inference_sdk_core::{InferenceContent, InferenceMessage, InferenceRequest, InferenceRole};

    #[test]
    fn tool_results_serialize_as_string_content_and_omit_false_is_error() {
        let req = InferenceRequest::builder()
            .model("test-model")
            .messages(vec![InferenceMessage {
                role: InferenceRole::Tool,
                content: vec![InferenceContent::ToolResult {
                    tool_use_id: "toolu_1".to_string(),
                    content: "ok".to_string(),
                    is_error: false,
                }],
                tool_call_id: Some("toolu_1".to_string()),
            }])
            .max_tokens(128)
            .build();

        let out = to_anthropic_request(req).expect("request should normalize");
        let json = serde_json::to_value(out).expect("request should serialize");
        let block = &json["messages"][0]["content"][0];

        assert_eq!(json["messages"][0]["role"], "user");
        assert_eq!(block["type"], "tool_result");
        assert_eq!(block["tool_use_id"], "toolu_1");
        assert_eq!(block["content"], "ok");
        assert!(
            block.get("is_error").is_none(),
            "is_error=false should be omitted"
        );
    }
}
