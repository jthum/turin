use futures_core::Stream;
use futures_util::StreamExt;
use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

pub mod error;
pub mod http;
pub mod stream_contract;

pub use error::{SdkError, StreamInvariantViolation};
pub use http::{RequestOptions, RetryNetworkRule, RetryPolicy, RetryStatusRule, TimeoutPolicy};
pub use stream_contract::{EventOrderValidator, validate_event_sequence};

/// A provider that can fulfill inference requests.
pub trait InferenceProvider: Send + Sync {
    fn supports_response_format(&self, response_format: &InferenceResponseFormat) -> bool {
        let _ = response_format;
        false
    }

    fn complete<'a>(
        &'a self,
        request: InferenceRequest,
        options: Option<RequestOptions>,
    ) -> BoxFuture<'a, Result<InferenceResult, SdkError>> {
        Box::pin(async move {
            let stream = self.stream(request, options).await?;
            InferenceResult::from_stream(stream).await
        })
    }

    fn stream<'a>(
        &'a self,
        request: InferenceRequest,
        options: Option<RequestOptions>,
    ) -> BoxFuture<'a, Result<InferenceStream, SdkError>>;
}

pub type InferenceStream =
    Pin<Box<dyn Stream<Item = Result<InferenceEvent, SdkError>> + Send + 'static>>;

/// A provider that can generate embedding vectors.
pub trait EmbeddingProvider: Send + Sync {
    fn embed<'a>(
        &'a self,
        request: EmbeddingRequest,
        options: Option<RequestOptions>,
    ) -> BoxFuture<'a, Result<EmbeddingResponse, SdkError>>;
}

/// A standardized request for embedding generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
}

#[bon::bon]
impl EmbeddingRequest {
    #[builder]
    pub fn new(
        #[builder(into)] model: String,
        input: Vec<String>,
        dimensions: Option<u32>,
    ) -> Self {
        Self {
            model,
            input,
            dimensions,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub data: Vec<EmbeddingData>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<EmbeddingUsage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub input_tokens: u32,
    pub total_tokens: u32,
}

/// A standardized request for LLM inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRequest {
    /// The model identifier (e.g., "gpt-4o", "claude-3-5-sonnet").
    pub model: String,

    /// The conversation history.
    pub messages: Vec<InferenceMessage>,

    /// Optional system prompt (if not provided in messages).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,

    /// Available tools for the model to use.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,

    /// Sampling temperature (0.0 to 1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Maximum number of tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Optional "thinking" budget for reasoning models (e.g. Claude 3.7, o1).
    /// Providers that support it will use this; others will ignore it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<u32>,

    /// Optional response format hint for structured output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<InferenceResponseFormat>,
}

// Bon builder
#[bon::bon]
impl InferenceRequest {
    #[builder]
    pub fn new(
        #[builder(into)] model: String,
        messages: Vec<InferenceMessage>,
        temperature: Option<f32>,
        max_tokens: Option<u32>,
        #[builder(into)] system: Option<String>,
        tools: Option<Vec<Tool>>,
        thinking_budget: Option<u32>,
        response_format: Option<InferenceResponseFormat>,
    ) -> Self {
        Self {
            model,
            messages,
            temperature,
            max_tokens,
            system,
            tools,
            thinking_budget,
            response_format,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceResponseFormat {
    Text,
    JsonObject,
    JsonSchema {
        json_schema: InferenceJsonSchemaConfig,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferenceJsonSchemaConfig {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// A normalized message in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceMessage {
    pub role: InferenceRole,
    pub content: Vec<InferenceContent>,
    // Optional field to link a tool result to a tool call
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum InferenceRole {
    User,
    Assistant,
    // System is handled via the separate `system` field or normalization logic
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum InferenceContent {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        is_error: bool,
    },
    Thinking {
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
}

/// Normalized definition of a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StopReason {
    EndTurn,
    MaxTokens,
    ToolUse,
    StopSequence,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceResult {
    pub content: Vec<InferenceContent>,
    pub model: String,
    pub stop_reason: Option<StopReason>,
    pub usage: Usage,
}

impl InferenceResult {
    fn parse_tool_input(tool_json: &str) -> Result<serde_json::Value, SdkError> {
        if tool_json.trim().is_empty() {
            return Ok(serde_json::json!({}));
        }
        serde_json::from_str(tool_json).map_err(SdkError::SerializationError)
    }

    fn finalize_pending_tool(
        current_tool_id: &mut Option<String>,
        current_tool_name: &mut Option<String>,
        current_tool_json: &mut String,
        content_parts: &mut Vec<InferenceContent>,
    ) -> Result<(), SdkError> {
        if current_tool_id.is_none() && current_tool_name.is_none() && current_tool_json.is_empty()
        {
            return Ok(());
        }

        let id = current_tool_id.take().ok_or_else(|| {
            SdkError::StreamInvariantViolation(StreamInvariantViolation::ToolCallMissingId)
        })?;
        let name = current_tool_name.take().ok_or_else(|| {
            SdkError::StreamInvariantViolation(StreamInvariantViolation::ToolCallMissingName)
        })?;
        let input = Self::parse_tool_input(current_tool_json)?;
        content_parts.push(InferenceContent::ToolUse { id, name, input });
        current_tool_json.clear();
        Ok(())
    }

    /// Helper to extract only `InferenceContent::Text` blocks combined.
    ///
    /// This intentionally excludes `Thinking`, `ToolUse`, and `ToolResult` content.
    /// Consumers that need full semantic output should inspect `self.content` directly.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|c| match c {
                InferenceContent::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Collects a stream into a single result.
    pub async fn from_stream(mut stream: InferenceStream) -> Result<Self, SdkError> {
        let mut content_parts = Vec::new();
        let mut model = String::new();
        let mut stop_reason = None;
        let mut usage = Usage {
            input_tokens: 0,
            output_tokens: 0,
        };

        // Tool call accumulation state.
        let mut current_tool_id: Option<String> = None;
        let mut current_tool_name: Option<String> = None;
        let mut current_tool_json: String = String::new();
        let mut event_validator = EventOrderValidator::new();

        while let Some(event_res) = stream.next().await {
            match event_res {
                Ok(event) => {
                    event_validator
                        .validate_event(&event)
                        .map_err(SdkError::from)?;

                    match event {
                        InferenceEvent::MessageStart { model: m, .. } => {
                            model = m;
                        }
                        InferenceEvent::MessageDelta { content } => {
                            if let Some(InferenceContent::Text { text }) = content_parts.last_mut()
                            {
                                text.push_str(&content);
                            } else {
                                content_parts.push(InferenceContent::Text { text: content });
                            }
                        }
                        InferenceEvent::ThinkingDelta { content } => {
                            if let Some(InferenceContent::Thinking { content: text, .. }) =
                                content_parts.last_mut()
                            {
                                text.push_str(&content);
                            } else {
                                content_parts.push(InferenceContent::Thinking {
                                    content,
                                    signature: None,
                                });
                            }
                        }
                        InferenceEvent::ThinkingSignatureDelta { signature } => {
                            if let Some(InferenceContent::Thinking {
                                signature: existing,
                                ..
                            }) = content_parts.last_mut()
                            {
                                if let Some(existing) = existing.as_mut() {
                                    existing.push_str(&signature);
                                } else {
                                    *existing = Some(signature);
                                }
                            } else {
                                content_parts.push(InferenceContent::Thinking {
                                    content: String::new(),
                                    signature: Some(signature),
                                });
                            }
                        }
                        InferenceEvent::ToolCallStart { id, name } => {
                            // Providers should not interleave tool-call streams, but if they do,
                            // close the previous pending call before starting the next one.
                            Self::finalize_pending_tool(
                                &mut current_tool_id,
                                &mut current_tool_name,
                                &mut current_tool_json,
                                &mut content_parts,
                            )?;
                            current_tool_id = Some(id);
                            current_tool_name = Some(name);
                            current_tool_json.clear();
                        }
                        InferenceEvent::ToolCallDelta { delta } => {
                            if current_tool_id.is_none() || current_tool_name.is_none() {
                                return Err(
                                    StreamInvariantViolation::ToolCallDeltaBeforeStart.into()
                                );
                            }
                            current_tool_json.push_str(&delta);
                        }
                        InferenceEvent::MessageEnd {
                            input_tokens,
                            output_tokens,
                            stop_reason: sr,
                        } => {
                            Self::finalize_pending_tool(
                                &mut current_tool_id,
                                &mut current_tool_name,
                                &mut current_tool_json,
                                &mut content_parts,
                            )?;
                            usage = Usage {
                                input_tokens,
                                output_tokens,
                            };
                            stop_reason = sr;
                        }
                    }
                }
                Err(e) => return Err(e),
            }
        }

        event_validator.finish().map_err(SdkError::from)?;

        Ok(InferenceResult {
            content: content_parts,
            model,
            stop_reason,
            usage,
        })
    }
}

/// Events emitted during a streaming inference response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum InferenceEvent {
    /// The start of a message response.
    MessageStart {
        role: String,
        model: String, // from API
        /// The provider attempting to fulfill this request.
        provider_id: String,
    },
    /// A text delta for the message content.
    MessageDelta { content: String },
    /// A thought process delta (for reasoning models).
    ThinkingDelta { content: String },
    /// A signature delta for a thinking block (Anthropic-compatible providers).
    ThinkingSignatureDelta { signature: String },
    /// A tool call started.
    ToolCallStart { id: String, name: String },
    /// A delta for a tool call argument (JSON fragment).
    ToolCallDelta { delta: String },
    /// The end of a message response, including usage statistics.
    MessageEnd {
        input_tokens: u32,
        output_tokens: u32,
        stop_reason: Option<StopReason>,
    },
}
