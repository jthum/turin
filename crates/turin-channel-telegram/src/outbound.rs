use anyhow::{Result, anyhow};
use std::path::{Path, PathBuf};
use turin_channel_core::{
    ChannelAttachment, OutboundMessage, render_plain_text_blocks, split_text_lines_to_char_limit,
};

use crate::inbound::{TelegramAttachmentKind, TelegramAudio};

mod html;

use html::render_html_chunks;

pub(crate) const TELEGRAM_MESSAGE_MAX_LEN: usize = 4_096;

fn stream_preview_text(text: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let mut out = String::new();
    for ch in trimmed.chars() {
        if out.chars().count() >= TELEGRAM_MESSAGE_MAX_LEN.saturating_sub(1) {
            out.push('…');
            break;
        }
        out.push(ch);
    }
    out
}

pub(crate) fn render_stream_preview(text: &str, thinking: Option<&str>) -> String {
    let text = text.trim();
    let thinking = thinking.map(str::trim).unwrap_or_default();

    if text.is_empty() && thinking.is_empty() {
        return String::new();
    }

    let mut preview = String::new();
    if !thinking.is_empty() {
        preview.push_str("Thinking…\n");
        preview.push_str(thinking);
    }
    if !text.is_empty() {
        if !preview.is_empty() {
            preview.push_str("\n\nReply\n");
        }
        preview.push_str(text);
    }

    stream_preview_text(&preview)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TelegramRenderMode {
    PlainText,
    Html,
}

#[derive(Debug, Clone)]
struct TelegramRenderedMessage {
    chunks: Vec<String>,
    parse_mode: Option<&'static str>,
    reply_to_message_id: Option<i64>,
    disable_web_page_preview: bool,
    disable_notification: bool,
}

pub(crate) fn telegram_batches_from_message(
    chat_id: &str,
    message_thread_id: Option<i64>,
    message: &OutboundMessage,
) -> Result<Vec<serde_json::Value>> {
    let rendered = render_telegram_message(message)?;
    Ok(rendered
        .chunks
        .into_iter()
        .map(|text| {
            telegram_payload(
                chat_id,
                message_thread_id,
                text,
                rendered.parse_mode,
                rendered.reply_to_message_id,
                rendered.disable_web_page_preview,
                rendered.disable_notification,
            )
        })
        .collect())
}

fn render_telegram_message(message: &OutboundMessage) -> Result<TelegramRenderedMessage> {
    let render_mode = resolve_render_mode(message);
    let reply_to_message_id = metadata_i64(&message.metadata, "telegram_reply_to_message_id")?;
    let disable_web_page_preview = message
        .metadata
        .get("telegram_disable_web_page_preview")
        .and_then(|value| value.as_bool())
        .unwrap_or(true);
    let disable_notification = message
        .metadata
        .get("telegram_disable_notification")
        .and_then(|value| value.as_bool())
        .unwrap_or(false);
    let final_thinking = message
        .metadata
        .get("channel_final_thinking")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty());

    let mut chunks = match render_mode {
        TelegramRenderMode::PlainText => {
            let mut rendered = render_plain_text_blocks(&message.blocks);
            if let Some(thinking) = final_thinking {
                rendered = prepend_final_thinking_text(&rendered, thinking);
            }
            split_for_telegram_message(rendered)
        }
        TelegramRenderMode::Html => render_html_chunks(message, final_thinking),
    };

    if chunks.is_empty() && message.attachments.is_empty() {
        chunks.push("(no output)".to_string());
    }

    Ok(TelegramRenderedMessage {
        chunks,
        parse_mode: match render_mode {
            TelegramRenderMode::PlainText => None,
            TelegramRenderMode::Html => Some("HTML"),
        },
        reply_to_message_id,
        disable_web_page_preview,
        disable_notification,
    })
}

fn resolve_render_mode(message: &OutboundMessage) -> TelegramRenderMode {
    if message
        .metadata
        .get("telegram_format")
        .and_then(|value| value.as_str())
        .is_some_and(|value| {
            value.eq_ignore_ascii_case("plain") || value.eq_ignore_ascii_case("text")
        })
    {
        return TelegramRenderMode::PlainText;
    }
    TelegramRenderMode::Html
}

fn prepend_final_thinking_text(rendered: &str, thinking: &str) -> String {
    let trimmed = rendered.trim();
    if trimmed.is_empty() {
        format!("Thinking:\n{}\n", thinking)
    } else {
        format!("Thinking:\n{}\n\nReply:\n{}", thinking, trimmed)
    }
}

pub(crate) fn attachment_preview_text(attachments: &[ChannelAttachment]) -> String {
    match attachments.len() {
        0 => "(no output)".to_string(),
        1 => format!("Sent attachment: {}", attachments[0].name),
        count => format!("Sent {count} attachments"),
    }
}

pub(crate) fn attachment_kind_from_content_type(
    content_type: Option<&str>,
) -> TelegramAttachmentKind {
    if content_type.is_some_and(|value| value.starts_with("image/")) {
        TelegramAttachmentKind::Image
    } else {
        TelegramAttachmentKind::File
    }
}

pub(crate) fn infer_audio_name(audio: &TelegramAudio) -> Option<String> {
    match (audio.performer.as_deref(), audio.title.as_deref()) {
        (Some(performer), Some(title))
            if !performer.trim().is_empty() && !title.trim().is_empty() =>
        {
            Some(format!("{performer} - {title}.mp3"))
        }
        (_, Some(title)) if !title.trim().is_empty() => Some(format!("{title}.mp3")),
        _ => None,
    }
}

pub(crate) fn default_media_dir_for_runtime(channel_runtime_id: &str) -> PathBuf {
    std::env::temp_dir()
        .join("turin")
        .join("channels")
        .join("telegram")
        .join(sanitize_runtime_component(channel_runtime_id))
        .join("media")
}

fn sanitize_runtime_component(raw: &str) -> String {
    let mut out = String::new();
    for ch in raw.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
            out.push(ch.to_ascii_lowercase());
        } else {
            out.push('-');
        }
    }
    let trimmed = out.trim_matches('-');
    if trimmed.is_empty() {
        "default".to_string()
    } else {
        trimmed.to_string()
    }
}

pub(crate) fn unique_media_name(name: &str, fallback_path: Option<&str>) -> String {
    let extension = media_extension(name, fallback_path)
        .map(|value| format!(".{value}"))
        .unwrap_or_default();
    format!("{}{}", uuid::Uuid::now_v7().simple(), extension)
}

fn media_extension(name: &str, fallback_path: Option<&str>) -> Option<String> {
    Path::new(name)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(str::trim)
        .filter(|ext| !ext.is_empty())
        .map(str::to_ascii_lowercase)
        .or_else(|| {
            fallback_path
                .and_then(|path| Path::new(path).extension().and_then(|ext| ext.to_str()))
                .map(str::trim)
                .filter(|ext| !ext.is_empty())
                .map(str::to_ascii_lowercase)
        })
}

fn split_for_telegram_message(content: String) -> Vec<String> {
    split_text_lines_to_char_limit(&content, TELEGRAM_MESSAGE_MAX_LEN)
}

pub(crate) fn telegram_payload(
    chat_id: &str,
    message_thread_id: Option<i64>,
    text: String,
    parse_mode: Option<&'static str>,
    reply_to_message_id: Option<i64>,
    disable_web_page_preview: bool,
    disable_notification: bool,
) -> serde_json::Value {
    let mut payload = serde_json::Map::new();
    payload.insert("chat_id".to_string(), serde_json::json!(chat_id));
    payload.insert("text".to_string(), serde_json::json!(text));
    payload.insert(
        "disable_web_page_preview".to_string(),
        serde_json::json!(disable_web_page_preview),
    );
    payload.insert(
        "disable_notification".to_string(),
        serde_json::json!(disable_notification),
    );
    if let Some(message_thread_id) = message_thread_id {
        payload.insert(
            "message_thread_id".to_string(),
            serde_json::json!(message_thread_id),
        );
    }
    if let Some(parse_mode) = parse_mode {
        payload.insert("parse_mode".to_string(), serde_json::json!(parse_mode));
    }
    if let Some(reply_to_message_id) = reply_to_message_id {
        payload.insert(
            "reply_to_message_id".to_string(),
            serde_json::json!(reply_to_message_id),
        );
        payload.insert(
            "allow_sending_without_reply".to_string(),
            serde_json::json!(true),
        );
    }
    serde_json::Value::Object(payload)
}

pub(crate) fn telegram_edit_payload(
    chat_id: &str,
    message_id: i64,
    text: String,
    parse_mode: Option<&str>,
    disable_web_page_preview: bool,
) -> serde_json::Value {
    let mut payload = serde_json::Map::new();
    payload.insert("chat_id".to_string(), serde_json::json!(chat_id));
    payload.insert("message_id".to_string(), serde_json::json!(message_id));
    payload.insert("text".to_string(), serde_json::json!(text));
    payload.insert(
        "disable_web_page_preview".to_string(),
        serde_json::json!(disable_web_page_preview),
    );
    if let Some(parse_mode) = parse_mode {
        payload.insert("parse_mode".to_string(), serde_json::json!(parse_mode));
    }
    serde_json::Value::Object(payload)
}

pub(crate) fn metadata_i64(
    metadata: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<Option<i64>> {
    let Some(value) = metadata.get(key) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    if let Some(number) = value.as_i64() {
        return Ok(Some(number));
    }
    if let Some(number) = value.as_u64() {
        return i64::try_from(number).map(Some).map_err(|_| {
            anyhow!(
                "[telegram_invalid_metadata] Telegram metadata '{}' is too large for i64",
                key
            )
        });
    }
    if let Some(text) = value.as_str() {
        return text.parse::<i64>().map(Some).map_err(|_| {
            anyhow!(
                "[telegram_invalid_metadata] Telegram metadata '{}' must be an integer or integer string",
                key
            )
        });
    }
    anyhow::bail!(
        "[telegram_invalid_metadata] Telegram metadata '{}' must be an integer or integer string",
        key
    );
}
