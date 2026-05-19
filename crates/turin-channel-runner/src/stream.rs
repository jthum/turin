use serde_json::Value;
use std::time::Duration;
use tokio::time::Instant;
use turin_channel_core::OutboundMessage;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelStreamMode {
    Off,
    Typing,
    Draft,
    Block,
}

impl ChannelStreamMode {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "off" => Some(Self::Off),
            "typing" => Some(Self::Typing),
            "draft" => Some(Self::Draft),
            "block" => Some(Self::Block),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Typing => "typing",
            Self::Draft => "draft",
            Self::Block => "block",
        }
    }

    pub fn is_allowed_by(self, allowed: &[Self]) -> bool {
        allowed.contains(&self)
    }

    pub fn sends_typing(self) -> bool {
        matches!(self, Self::Typing | Self::Draft | Self::Block)
    }

    pub fn streams_text(self) -> bool {
        matches!(self, Self::Draft | Self::Block)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChannelProgressUpdate {
    Typing,
    StreamingPreview {
        text: String,
        thinking: Option<String>,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct WorkerStreamConfig {
    pub(crate) mode: ChannelStreamMode,
    pub(crate) stream_thinking: bool,
    pub(crate) persist_thinking: bool,
}

pub(crate) fn should_flush_preview(
    stream_mode: ChannelStreamMode,
    text_preview: &str,
    thinking_preview: Option<&str>,
    last_flushed_chars: usize,
    last_flush_at: Instant,
) -> bool {
    let current_chars = preview_char_count(text_preview, thinking_preview);
    if current_chars <= last_flushed_chars {
        return false;
    }

    let new_chars = current_chars.saturating_sub(last_flushed_chars);
    match stream_mode {
        ChannelStreamMode::Draft => {
            new_chars >= 32 || last_flush_at.elapsed() >= Duration::from_millis(800)
        }
        ChannelStreamMode::Block => {
            new_chars >= 160
                || (new_chars >= 64
                    && last_flush_at.elapsed() >= Duration::from_millis(1500)
                    && (text_preview.ends_with('\n') || text_preview.ends_with(". ")))
        }
        _ => false,
    }
}

pub(crate) fn preview_char_count(text_preview: &str, thinking_preview: Option<&str>) -> usize {
    text_preview.chars().count()
        + thinking_preview
            .map(|thinking| thinking.chars().count())
            .unwrap_or_default()
}

pub(crate) fn preview_thinking(include_thinking: bool, thinking_preview: &str) -> Option<String> {
    if !include_thinking || thinking_preview.trim().is_empty() {
        return None;
    }
    Some(thinking_preview.to_string())
}

pub(crate) fn should_subscribe_to_session_events(stream: &WorkerStreamConfig) -> bool {
    stream.mode.streams_text() || stream.stream_thinking || stream.persist_thinking
}

pub(crate) fn attach_final_thinking(
    mut outbound: OutboundMessage,
    thinking: Option<String>,
) -> OutboundMessage {
    let Some(thinking) = thinking.map(|value| value.trim().to_string()) else {
        return outbound;
    };
    if thinking.is_empty() {
        return outbound;
    }
    outbound.metadata.insert(
        "channel_final_thinking".to_string(),
        Value::String(thinking),
    );
    outbound
}
