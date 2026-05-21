use std::collections::VecDeque;
use std::path::PathBuf;

use turin_channel_core::{
    OutboundMessage, render_plain_text_blocks, split_text_lines_to_char_limit,
};

pub(crate) const DISCORD_CONTENT_MAX_LEN: usize = 2_000;
const DISCORD_EMBEDS_MAX: usize = 10;
const DISCORD_FILES_MAX: usize = 10;

#[derive(Debug, Clone)]
pub(crate) struct LocalAttachmentRef {
    pub(crate) name: String,
    pub(crate) path: PathBuf,
    pub(crate) content_type: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct DiscordSendMessage {
    pub(crate) content: Option<String>,
    pub(crate) embeds: Vec<serde_json::Value>,
    pub(crate) components: Vec<serde_json::Value>,
    pub(crate) files: Vec<LocalAttachmentRef>,
}

pub(crate) fn render_outbound_messages(message: OutboundMessage) -> Vec<DiscordSendMessage> {
    let mut text_chunks = split_text_lines_to_char_limit(
        &render_plain_text_blocks(&message.blocks),
        DISCORD_CONTENT_MAX_LEN,
    );
    let mut embeds = message.embeds;
    if embeds.is_empty() {
        embeds = extract_embeds_from_metadata(&message.metadata);
    }
    let mut components = extract_components_from_metadata(&message.metadata);
    if components.is_empty() {
        components = message.components;
    }

    let mut local_files = Vec::new();
    let mut remote_attachment_urls = Vec::new();
    for attachment in message.attachments {
        if let Some(local_path) = attachment.local_path {
            local_files.push(LocalAttachmentRef {
                name: attachment.name,
                path: PathBuf::from(local_path),
                content_type: attachment.content_type,
            });
            continue;
        }
        if let Some(url) = attachment.url {
            remote_attachment_urls.push(url);
        }
    }
    if !remote_attachment_urls.is_empty() {
        let urls = remote_attachment_urls.join("\n");
        if !urls.trim().is_empty() {
            text_chunks.extend(split_text_lines_to_char_limit(
                &urls,
                DISCORD_CONTENT_MAX_LEN,
            ));
        }
    }

    let mut embed_queue: VecDeque<serde_json::Value> = embeds.into_iter().collect();
    let mut file_queue: VecDeque<LocalAttachmentRef> = local_files.into_iter().collect();
    let mut text_queue: VecDeque<String> = text_chunks.into_iter().collect();
    let mut output = Vec::new();
    let mut first = true;

    while !text_queue.is_empty() || !embed_queue.is_empty() || !file_queue.is_empty() || first {
        let content = text_queue.pop_front();
        let mut embeds_for_message = Vec::new();
        while embeds_for_message.len() < DISCORD_EMBEDS_MAX {
            let Some(embed) = embed_queue.pop_front() else {
                break;
            };
            embeds_for_message.push(embed);
        }

        let mut files_for_message = Vec::new();
        while files_for_message.len() < DISCORD_FILES_MAX {
            let Some(file) = file_queue.pop_front() else {
                break;
            };
            files_for_message.push(file);
        }

        let components_for_message = if first {
            components.clone()
        } else {
            Vec::new()
        };

        if content.is_none()
            && embeds_for_message.is_empty()
            && files_for_message.is_empty()
            && components_for_message.is_empty()
        {
            break;
        }

        output.push(DiscordSendMessage {
            content,
            embeds: embeds_for_message,
            components: components_for_message,
            files: files_for_message,
        });
        first = false;
    }

    if output.is_empty() {
        output.push(DiscordSendMessage {
            content: Some("(no output)".to_string()),
            embeds: Vec::new(),
            components: Vec::new(),
            files: Vec::new(),
        });
    }

    output
}

fn extract_embeds_from_metadata(
    metadata: &serde_json::Map<String, serde_json::Value>,
) -> Vec<serde_json::Value> {
    metadata
        .get("discord_embeds")
        .or_else(|| metadata.get("embeds"))
        .and_then(|value| value.as_array())
        .map(|entries| {
            entries
                .iter()
                .filter(|entry| entry.is_object())
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

fn extract_components_from_metadata(
    metadata: &serde_json::Map<String, serde_json::Value>,
) -> Vec<serde_json::Value> {
    metadata
        .get("discord_components")
        .or_else(|| metadata.get("components"))
        .and_then(|value| value.as_array())
        .map(|entries| {
            entries
                .iter()
                .filter(|entry| entry.is_object())
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

pub(crate) fn discord_payload_from_message(message: &DiscordSendMessage) -> serde_json::Value {
    let mut payload = serde_json::Map::new();
    if let Some(content) = &message.content {
        payload.insert(
            "content".to_string(),
            serde_json::Value::String(content.clone()),
        );
    }
    if !message.embeds.is_empty() {
        payload.insert(
            "embeds".to_string(),
            serde_json::Value::Array(message.embeds.clone()),
        );
    }
    if !message.components.is_empty() {
        payload.insert(
            "components".to_string(),
            serde_json::Value::Array(message.components.clone()),
        );
    }
    serde_json::Value::Object(payload)
}
