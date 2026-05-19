use turin_channel_core::{ChannelConversationKey, MessageBlock, OutboundMessage};

use crate::RocketChatReplyMode;

pub(crate) const ROCKETCHAT_MESSAGE_MAX_LEN: usize = 4_000;

fn render_text_blocks(blocks: &[MessageBlock]) -> String {
    let mut chunks = Vec::new();
    for block in blocks {
        match block {
            MessageBlock::Text { text } => {
                if !text.trim().is_empty() {
                    chunks.push(wrap_markdown_tables(text));
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

#[derive(Debug, Clone, Copy)]
pub(crate) struct RocketChatReplyTarget<'a> {
    pub(crate) thread_id: Option<&'a str>,
    pub(crate) show_in_channel: bool,
}

pub(crate) fn build_rocketchat_send_payload(
    room_id: &str,
    text: &str,
    reply_target: RocketChatReplyTarget<'_>,
    attachments: &[serde_json::Value],
) -> serde_json::Value {
    let mut message = serde_json::Map::new();
    message.insert("rid".to_string(), serde_json::json!(room_id));
    message.insert("msg".to_string(), serde_json::json!(text));
    message.insert("parseUrls".to_string(), serde_json::json!(false));
    if !attachments.is_empty() {
        message.insert(
            "attachments".to_string(),
            serde_json::Value::Array(attachments.to_vec()),
        );
    }
    if let Some(thread_id) = reply_target.thread_id {
        message.insert("tmid".to_string(), serde_json::json!(thread_id));
        if reply_target.show_in_channel {
            message.insert("tshow".to_string(), serde_json::json!(true));
        }
    }

    serde_json::json!({ "message": message })
}

pub(crate) fn render_rocketchat_message(
    message: &OutboundMessage,
    persist_thinking: bool,
) -> String {
    let mut rendered = render_text_blocks(&message.blocks);
    if persist_thinking
        && let Some(thinking) = message
            .metadata
            .get("channel_final_thinking")
            .and_then(|value| value.as_str())
            .map(str::trim)
            .filter(|value| !value.is_empty())
    {
        rendered = prepend_final_thinking_text(&rendered, thinking);
    }

    rendered
}

pub(crate) fn prepend_channel_reply_quote(text: &str, message: &OutboundMessage) -> String {
    let reply_label = message
        .metadata
        .get("rocketchat_reply_to_label")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let reply_link = message
        .metadata
        .get("rocketchat_reply_to_message_link")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let reply_excerpt = message
        .metadata
        .get("rocketchat_reply_to_excerpt")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let mut quote_lines = Vec::new();
    if let Some(reply_label) = reply_label {
        let first_line = if let Some(reply_link) = reply_link {
            format!("> [{}]({})", reply_label, reply_link)
        } else {
            format!("> {}", reply_label)
        };
        quote_lines.push(first_line);
    }
    if let Some(reply_excerpt) = reply_excerpt {
        for line in reply_excerpt.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                quote_lines.push(format!("> {}", trimmed));
            }
        }
    }

    if quote_lines.is_empty() {
        return text.to_string();
    }

    if text.is_empty() {
        quote_lines.join("\n")
    } else {
        format!("{}\n\n{}", quote_lines.join("\n"), text)
    }
}

pub(crate) fn resolve_reply_target<'a>(
    room_id: &'a str,
    conversation: &'a ChannelConversationKey,
    message: &'a OutboundMessage,
    reply_mode: RocketChatReplyMode,
) -> RocketChatReplyTarget<'a> {
    let metadata_thread_id = metadata_str(&message.metadata, "rocketchat_thread_id");
    let reply_to_message_id = metadata_str(&message.metadata, "rocketchat_reply_to_message_id");
    let conversation_thread_id =
        (conversation.thread_id != room_id).then_some(conversation.thread_id.as_str());
    let thread_id = match reply_mode {
        RocketChatReplyMode::Channel => None,
        RocketChatReplyMode::Thread | RocketChatReplyMode::ThreadAndChannel => metadata_thread_id
            .or(reply_to_message_id)
            .or(conversation_thread_id),
    };

    RocketChatReplyTarget {
        thread_id,
        show_in_channel: matches!(reply_mode, RocketChatReplyMode::ThreadAndChannel),
    }
}

fn metadata_str<'a>(
    metadata: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Option<&'a str> {
    metadata.get(key).and_then(|value| value.as_str())
}

fn prepend_final_thinking_text(rendered: &str, thinking: &str) -> String {
    let trimmed = rendered.trim();
    if trimmed.is_empty() {
        format!("Thinking:\n{}", thinking)
    } else {
        format!("Thinking:\n{}\n\nReply:\n{}", thinking, trimmed)
    }
}

pub(crate) fn reply_excerpt(text: &str) -> String {
    let lines = text
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(3)
        .map(|line| {
            let excerpt = line.chars().take(120).collect::<String>();
            if line.chars().count() > excerpt.chars().count() {
                format!("{excerpt}...")
            } else {
                excerpt
            }
        })
        .collect::<Vec<_>>();
    if lines.is_empty() {
        String::new()
    } else {
        lines.join("\n")
    }
}

fn wrap_markdown_tables(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return text.to_string();
    }

    let mut out = Vec::new();
    let mut index = 0usize;
    let mut in_fence = false;
    while index < lines.len() {
        let line = lines[index];
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
            out.push(line.to_string());
            index += 1;
            continue;
        }

        if !in_fence && is_markdown_table_row(line) {
            let start = index;
            let mut end = index;
            let mut has_separator = false;
            while end < lines.len() && is_markdown_table_row(lines[end]) {
                has_separator |= is_markdown_table_separator(lines[end]);
                end += 1;
            }
            if has_separator && end.saturating_sub(start) >= 2 {
                out.push("```".to_string());
                out.extend(lines[start..end].iter().map(|value| (*value).to_string()));
                out.push("```".to_string());
                index = end;
                continue;
            }
        }

        out.push(line.to_string());
        index += 1;
    }

    out.join("\n")
}

fn is_markdown_table_row(line: &str) -> bool {
    let trimmed = line.trim();
    !trimmed.is_empty() && trimmed.contains('|') && !trimmed.starts_with("```")
}

fn is_markdown_table_separator(line: &str) -> bool {
    let trimmed = line.trim();
    !trimmed.is_empty()
        && trimmed.contains('-')
        && trimmed
            .chars()
            .all(|ch| matches!(ch, '|' | ':' | '-' | ' ' | '\t'))
}

pub(crate) fn split_for_rocketchat_content(content: String) -> Vec<String> {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return vec![" ".to_string()];
    }

    let mut out = Vec::new();
    let mut current = String::new();
    for line in trimmed.lines() {
        if line.chars().count() > ROCKETCHAT_MESSAGE_MAX_LEN {
            if !current.is_empty() {
                out.push(current.clone());
                current.clear();
            }
            let mut segment = String::new();
            for ch in line.chars() {
                segment.push(ch);
                if segment.chars().count() >= ROCKETCHAT_MESSAGE_MAX_LEN {
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
        if tentative.chars().count() > ROCKETCHAT_MESSAGE_MAX_LEN {
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

#[cfg(test)]
pub(crate) fn render_text_blocks_for_test(blocks: &[MessageBlock]) -> String {
    render_text_blocks(blocks)
}
