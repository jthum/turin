use anyhow::{Result, anyhow};
use pulldown_cmark::{CodeBlockKind, Event, Options, Parser, Tag, TagEnd};
use std::path::{Path, PathBuf};
use turin_channel_core::{ChannelAttachment, MessageBlock, OutboundMessage};

use crate::{TelegramAttachmentKind, TelegramAudio};

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
            let mut rendered = render_text_blocks(&message.blocks);
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

fn render_text_blocks(blocks: &[MessageBlock]) -> String {
    let mut chunks = Vec::new();
    for block in blocks {
        match block {
            MessageBlock::Text { text } => {
                if !text.trim().is_empty() {
                    chunks.push(text.clone());
                }
            }
            MessageBlock::CodeBlock { language, code } => {
                let prefix = language.clone().unwrap_or_default();
                chunks.push(format!("```{}\n{}\n```", prefix, code));
            }
        }
    }
    chunks.join("\n\n")
}

fn render_html_chunks(message: &OutboundMessage, final_thinking: Option<&str>) -> Vec<String> {
    let mut segments = Vec::new();
    if let Some(thinking) = final_thinking {
        segments.push("<i>Thinking</i>".to_string());
        segments.extend(split_wrapped_segment(thinking, "<pre>", "</pre>"));
        segments.push("<i>Reply</i>".to_string());
    }
    for block in &message.blocks {
        segments.extend(render_html_segments_for_block(block));
    }

    pack_segments(segments)
}

fn prepend_final_thinking_text(rendered: &str, thinking: &str) -> String {
    let trimmed = rendered.trim();
    if trimmed.is_empty() {
        format!("Thinking:\n{}\n", thinking)
    } else {
        format!("Thinking:\n{}\n\nReply:\n{}", thinking, trimmed)
    }
}

fn render_html_segments_for_block(block: &MessageBlock) -> Vec<String> {
    match block {
        MessageBlock::Text { text } => render_markdown_segments(text),
        MessageBlock::CodeBlock { code, .. } => split_wrapped_segment(code, "<pre>", "</pre>"),
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

#[derive(Debug, Clone, Copy)]
struct MarkdownListState {
    ordered: bool,
    next_index: u64,
}

#[derive(Debug, Default, Clone)]
struct MarkdownTableState {
    rows: Vec<Vec<String>>,
    current_row: Vec<String>,
    current_cell: String,
    header_rows: usize,
}

fn render_markdown_segments(markdown: &str) -> Vec<String> {
    let trimmed = markdown.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    let mut options = Options::empty();
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_TASKLISTS);

    let parser = Parser::new_ext(trimmed, options);
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut blockquote_depth = 0usize;
    let mut list_stack: Vec<MarkdownListState> = Vec::new();
    let mut code_block: Option<String> = None;
    let mut table_state: Option<MarkdownTableState> = None;

    for event in parser {
        match event {
            Event::Start(tag) => match tag {
                Tag::Paragraph => {}
                Tag::Heading { .. } => {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str("<b>");
                }
                Tag::BlockQuote(_) => {
                    blockquote_depth = blockquote_depth.saturating_add(1);
                }
                Tag::List(start) => {
                    list_stack.push(MarkdownListState {
                        ordered: start.is_some(),
                        next_index: start.unwrap_or(1),
                    });
                }
                Tag::Item => {
                    flush_rich_segment(&mut segments, &mut current);
                    current.push_str(&blockquote_prefix(blockquote_depth));
                    if let Some(state) = list_stack.last_mut() {
                        if state.ordered {
                            current.push_str(&format!("{}. ", state.next_index));
                            state.next_index = state.next_index.saturating_add(1);
                        } else {
                            current.push_str("• ");
                        }
                    }
                }
                Tag::Emphasis => {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str("<i>");
                }
                Tag::Strong => {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str("<b>");
                }
                Tag::Strikethrough => {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str("<s>");
                }
                Tag::Link { dest_url, .. } => {
                    if table_state.is_some() {
                        continue;
                    }
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str("<a href=\"");
                    current.push_str(&escape_html(dest_url.as_ref()));
                    current.push_str("\">");
                }
                Tag::Table(_) => {
                    flush_rich_segment(&mut segments, &mut current);
                    table_state = Some(MarkdownTableState::default());
                }
                Tag::TableHead => {}
                Tag::TableRow => {
                    if let Some(table) = table_state.as_mut() {
                        table.current_row.clear();
                    }
                }
                Tag::TableCell => {
                    if let Some(table) = table_state.as_mut() {
                        table.current_cell.clear();
                    }
                }
                Tag::CodeBlock(kind) => {
                    flush_rich_segment(&mut segments, &mut current);
                    let mut rendered = String::new();
                    if let CodeBlockKind::Fenced(language) = kind {
                        let language = language.trim();
                        if !language.is_empty() {
                            rendered.push_str(language);
                            rendered.push('\n');
                        }
                    }
                    code_block = Some(rendered);
                }
                _ => {}
            },
            Event::End(tag) => match tag {
                TagEnd::Paragraph if list_stack.is_empty() => {
                    flush_rich_segment(&mut segments, &mut current);
                }
                TagEnd::Heading(_) => {
                    current.push_str("</b>");
                    flush_rich_segment(&mut segments, &mut current);
                }
                TagEnd::BlockQuote(_) => {
                    flush_rich_segment(&mut segments, &mut current);
                    blockquote_depth = blockquote_depth.saturating_sub(1);
                }
                TagEnd::List(_) => {
                    flush_rich_segment(&mut segments, &mut current);
                    list_stack.pop();
                }
                TagEnd::Item => {
                    flush_rich_segment(&mut segments, &mut current);
                }
                TagEnd::Emphasis => current.push_str("</i>"),
                TagEnd::Strong => current.push_str("</b>"),
                TagEnd::Strikethrough => current.push_str("</s>"),
                TagEnd::Table => {
                    if let Some(table) = table_state.take() {
                        let rendered = render_markdown_table(&table);
                        if !rendered.trim().is_empty() {
                            segments.extend(split_wrapped_segment(&rendered, "<pre>", "</pre>"));
                        }
                    }
                }
                TagEnd::TableHead => {
                    if let Some(table) = table_state.as_mut() {
                        if !table.current_row.is_empty() {
                            table.rows.push(std::mem::take(&mut table.current_row));
                        }
                        table.header_rows = table.rows.len();
                    }
                }
                TagEnd::TableRow => {
                    if let Some(table) = table_state.as_mut()
                        && !table.current_row.is_empty()
                    {
                        table.rows.push(std::mem::take(&mut table.current_row));
                    }
                }
                TagEnd::TableCell => {
                    if let Some(table) = table_state.as_mut() {
                        table
                            .current_row
                            .push(normalize_table_cell(&table.current_cell));
                        table.current_cell.clear();
                    }
                }
                TagEnd::Link if table_state.is_none() => {
                    current.push_str("</a>");
                }
                TagEnd::CodeBlock => {
                    if let Some(rendered) = code_block.take() {
                        segments.extend(split_wrapped_segment(&rendered, "<pre>", "</pre>"));
                    }
                }
                _ => {}
            },
            Event::Text(text) => {
                if let Some(code) = code_block.as_mut() {
                    code.push_str(text.as_ref());
                } else if let Some(table) = table_state.as_mut() {
                    table.current_cell.push_str(text.as_ref());
                } else {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str(&escape_html(text.as_ref()));
                }
            }
            Event::Code(text) => {
                if let Some(table) = table_state.as_mut() {
                    table.current_cell.push_str(text.as_ref());
                } else {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str("<code>");
                    current.push_str(&escape_html(text.as_ref()));
                    current.push_str("</code>");
                }
            }
            Event::SoftBreak | Event::HardBreak => {
                if let Some(code) = code_block.as_mut() {
                    code.push('\n');
                } else if let Some(table) = table_state.as_mut() {
                    if !table.current_cell.ends_with(' ') && !table.current_cell.is_empty() {
                        table.current_cell.push(' ');
                    }
                } else {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push('\n');
                }
            }
            Event::Rule => {
                flush_rich_segment(&mut segments, &mut current);
                segments.push("────────".to_string());
            }
            Event::TaskListMarker(checked) => {
                if let Some(table) = table_state.as_mut() {
                    table
                        .current_cell
                        .push_str(if checked { "[x] " } else { "[ ] " });
                } else {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str(if checked { "[x] " } else { "[ ] " });
                }
            }
            Event::Html(html) | Event::InlineHtml(html) => {
                if let Some(code) = code_block.as_mut() {
                    code.push_str(html.as_ref());
                } else if let Some(table) = table_state.as_mut() {
                    table.current_cell.push_str(html.as_ref());
                } else {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str(&escape_html(html.as_ref()));
                }
            }
            Event::InlineMath(text) | Event::DisplayMath(text) => {
                if let Some(table) = table_state.as_mut() {
                    table.current_cell.push_str(text.as_ref());
                } else {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push_str(&escape_html(text.as_ref()));
                }
            }
            Event::FootnoteReference(reference) => {
                if let Some(table) = table_state.as_mut() {
                    table.current_cell.push('[');
                    table.current_cell.push_str(reference.as_ref());
                    table.current_cell.push(']');
                } else {
                    ensure_prefix(&mut current, blockquote_depth);
                    current.push('[');
                    current.push_str(&escape_html(reference.as_ref()));
                    current.push(']');
                }
            }
        }
    }

    flush_rich_segment(&mut segments, &mut current);
    pack_segments(segments)
}

fn ensure_prefix(current: &mut String, blockquote_depth: usize) {
    if current.is_empty() {
        current.push_str(&blockquote_prefix(blockquote_depth));
    }
}

fn blockquote_prefix(depth: usize) -> String {
    "&gt; ".repeat(depth)
}

fn flush_rich_segment(segments: &mut Vec<String>, current: &mut String) {
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        segments.extend(split_rich_segment(trimmed));
    }
    current.clear();
}

fn normalize_table_cell(cell: &str) -> String {
    cell.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn render_markdown_table(table: &MarkdownTableState) -> String {
    if table.rows.is_empty() {
        return String::new();
    }

    let column_count = table.rows.iter().map(Vec::len).max().unwrap_or(0);
    if column_count == 0 {
        return String::new();
    }

    let mut widths = vec![0usize; column_count];
    for row in &table.rows {
        for (index, cell) in row.iter().enumerate() {
            widths[index] = widths[index].max(cell.chars().count());
        }
    }

    let format_row = |row: &[String]| {
        let mut out = String::from("|");
        for (index, width) in widths.iter().enumerate() {
            let cell = row.get(index).map(String::as_str).unwrap_or("");
            out.push(' ');
            out.push_str(cell);
            let padding = width.saturating_sub(cell.chars().count());
            if padding > 0 {
                out.push_str(&" ".repeat(padding));
            }
            out.push(' ');
            out.push('|');
        }
        out
    };

    let separator = {
        let mut out = String::from("|");
        for width in &widths {
            out.push(' ');
            out.push_str(&"-".repeat((*width).max(3)));
            out.push(' ');
            out.push('|');
        }
        out
    };

    let mut lines = Vec::new();
    for (index, row) in table.rows.iter().enumerate() {
        lines.push(format_row(row));
        if table.header_rows > 0 && index + 1 == table.header_rows {
            lines.push(separator.clone());
        }
    }

    lines.join("\n")
}

fn split_rich_segment(content: &str) -> Vec<String> {
    if content.chars().count() <= TELEGRAM_MESSAGE_MAX_LEN {
        return vec![content.to_string()];
    }

    let mut out = Vec::new();
    let mut current = String::new();
    for line in content.lines() {
        let tentative = if current.is_empty() {
            line.to_string()
        } else {
            format!("{current}\n{line}")
        };
        if tentative.chars().count() > TELEGRAM_MESSAGE_MAX_LEN {
            if !current.is_empty() {
                out.push(current.clone());
                current.clear();
            }
            if line.chars().count() > TELEGRAM_MESSAGE_MAX_LEN {
                out.extend(split_plain_segment(line));
            } else {
                current = line.to_string();
            }
        } else {
            current = tentative;
        }
    }

    if !current.is_empty() {
        out.push(current);
    }

    out
}

fn split_plain_segment(content: &str) -> Vec<String> {
    split_content_to_limit(content, TELEGRAM_MESSAGE_MAX_LEN)
}

fn split_wrapped_segment(content: &str, prefix: &str, suffix: &str) -> Vec<String> {
    let limit = TELEGRAM_MESSAGE_MAX_LEN
        .saturating_sub(prefix.chars().count())
        .saturating_sub(suffix.chars().count())
        .max(1);
    split_content_to_limit(&escape_html(content), limit)
        .into_iter()
        .map(|chunk| format!("{prefix}{chunk}{suffix}"))
        .collect()
}

fn split_content_to_limit(content: &str, limit: usize) -> Vec<String> {
    let mut out = Vec::new();
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return out;
    }

    let mut current = String::new();
    for ch in trimmed.chars() {
        current.push(ch);
        if current.chars().count() >= limit {
            out.push(current.clone());
            current.clear();
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

fn pack_segments(segments: Vec<String>) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();

    for segment in segments {
        let segment = segment.trim().to_string();
        if segment.is_empty() {
            continue;
        }

        let tentative = if current.is_empty() {
            segment.clone()
        } else {
            format!("{current}\n\n{segment}")
        };
        if tentative.chars().count() > TELEGRAM_MESSAGE_MAX_LEN {
            if !current.is_empty() {
                out.push(current.clone());
                current.clear();
            }
            current = segment;
        } else {
            current = tentative;
        }
    }

    if !current.is_empty() {
        out.push(current);
    }

    out
}

fn escape_html(input: &str) -> String {
    input
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn split_for_telegram_message(content: String) -> Vec<String> {
    let mut out = Vec::new();
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return out;
    }

    let mut current = String::new();
    for line in trimmed.lines() {
        if line.chars().count() > TELEGRAM_MESSAGE_MAX_LEN {
            if !current.is_empty() {
                out.push(current.clone());
                current.clear();
            }

            let mut segment = String::new();
            for ch in line.chars() {
                segment.push(ch);
                if segment.chars().count() >= TELEGRAM_MESSAGE_MAX_LEN {
                    out.push(segment.clone());
                    segment.clear();
                }
            }
            if !segment.is_empty() {
                out.push(segment);
            }
            continue;
        }

        let tentative = if current.is_empty() {
            line.to_string()
        } else {
            format!("{current}\n{line}")
        };
        if tentative.chars().count() > TELEGRAM_MESSAGE_MAX_LEN {
            if !current.is_empty() {
                out.push(current.clone());
            }
            current = line.to_string();
        } else {
            current = tentative;
        }
    }

    if !current.is_empty() {
        out.push(current);
    }

    out
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
