use thiserror::Error;

/// Stream contract violations detected while assembling normalized events.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum StreamInvariantViolation {
    #[error("message_start must be emitted before any message/tool delta events")]
    MessageNotStarted,
    #[error("message_start can only be emitted once per stream")]
    DuplicateMessageStart,
    #[error("tool_call_delta was emitted before tool_call_start")]
    ToolCallDeltaBeforeStart,
    #[error("message_end was emitted before message_start")]
    MessageEndBeforeStart,
    #[error("events were emitted after message_end")]
    EventAfterMessageEnd,
    #[error("stream ended before message_end")]
    MissingMessageEnd,
    #[error("stream ended without a message_start")]
    MissingMessageStart,
    #[error("tool call stream ended without a tool id")]
    ToolCallMissingId,
    #[error("tool call stream ended without a tool name")]
    ToolCallMissingName,
}

/// Base error type shared across all provider SDKs.
#[derive(Error, Debug)]
pub enum SdkError {
    #[error("API error: {0}")]
    ApiError(String),
    #[error("Network error: {0}")]
    NetworkError(#[from] reqwest::Error),
    #[error("Serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    #[error("Invalid configuration: {0}")]
    ConfigError(String),
    #[error("Stream error: {0}")]
    StreamError(String),
    #[error(transparent)]
    StreamInvariantViolation(#[from] StreamInvariantViolation),
    #[error("Provider error: {0}")]
    ProviderError(String),
    #[error("Unknown error: {0}")]
    Unknown(String),
}
