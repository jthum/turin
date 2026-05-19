use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use turin_channel_core::{ChannelAttachment, InboundEvent, MessageBlock, OutboundMessage};
use turin_types::TaskInputContent;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TaskSnapshot {
    pub request_id: String,
    pub agent_id: String,
    pub slot_id: String,
    pub trace_id: String,
    pub state: String,
    pub runtime_task_id: Option<String>,
    pub status: Option<String>,
    pub task_turn_count: Option<u32>,
    pub output: Option<String>,
    #[serde(default)]
    pub assistant_content: Option<Vec<TaskInputContent>>,
    pub error: Option<String>,
}

pub(crate) fn task_to_outbound(task: &TaskSnapshot) -> OutboundMessage {
    if let Some(output) = task.output.as_ref() {
        if let Some(structured) = try_parse_structured_outbound(output) {
            structured
        } else if let Some(content) = task.assistant_content.as_deref() {
            let mapped = outbound_from_task_content(content);
            if !mapped.blocks.is_empty() || !mapped.attachments.is_empty() {
                mapped
            } else {
                OutboundMessage::text(output.clone())
            }
        } else {
            OutboundMessage::text(output.clone())
        }
    } else if let Some(content) = task.assistant_content.as_deref() {
        let mapped = outbound_from_task_content(content);
        if !mapped.blocks.is_empty() || !mapped.attachments.is_empty() {
            mapped
        } else {
            OutboundMessage::text(format!("Task {} finished without output", task.request_id))
        }
    } else if let Some(error) = task.error.as_ref() {
        OutboundMessage::text(format!("Turin error: {}", error))
    } else {
        OutboundMessage::text(format!("Task {} finished without output", task.request_id))
    }
}

fn outbound_from_task_content(content: &[TaskInputContent]) -> OutboundMessage {
    let mut outbound = OutboundMessage::default();
    for part in content {
        match part {
            TaskInputContent::Text { text } => {
                if !text.trim().is_empty() {
                    outbound
                        .blocks
                        .push(MessageBlock::Text { text: text.clone() });
                }
            }
            TaskInputContent::Image {
                name,
                content_type,
                url,
                local_path,
                ..
            } => outbound.attachments.push(ChannelAttachment {
                name: name.clone().unwrap_or_else(|| "image".to_string()),
                content_type: content_type.clone(),
                url: url.clone(),
                local_path: local_path.clone(),
            }),
            TaskInputContent::File {
                name,
                content_type,
                url,
                local_path,
            } => outbound.attachments.push(ChannelAttachment {
                name: name.clone().unwrap_or_else(|| "file".to_string()),
                content_type: content_type.clone(),
                url: url.clone(),
                local_path: local_path.clone(),
            }),
        }
    }
    outbound
}

#[derive(Debug, Clone, Deserialize)]
struct StructuredOutbound {
    #[serde(default)]
    _turin_channel_outbound: bool,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    blocks: Vec<MessageBlock>,
    #[serde(default)]
    attachments: Vec<ChannelAttachment>,
    #[serde(default)]
    embeds: Vec<Value>,
    #[serde(default)]
    components: Vec<Value>,
    #[serde(default)]
    metadata: Map<String, Value>,
}

fn try_parse_structured_outbound(raw: &str) -> Option<OutboundMessage> {
    let trimmed = raw.trim();
    if !trimmed.starts_with('{') {
        return None;
    }

    let parsed: StructuredOutbound = serde_json::from_str(trimmed).ok()?;
    if !parsed._turin_channel_outbound {
        return None;
    }

    let mut blocks = parsed.blocks;
    if blocks.is_empty()
        && let Some(content) = parsed.content
        && !content.trim().is_empty()
    {
        blocks.push(MessageBlock::Text { text: content });
    }

    Some(OutboundMessage {
        blocks,
        attachments: parsed.attachments,
        embeds: parsed.embeds,
        components: parsed.components,
        metadata: parsed.metadata,
    })
}

pub(crate) fn task_prompt_for_submission(event: &InboundEvent) -> String {
    let prompt = event.prompt_text();
    if !prompt.trim().is_empty() {
        return prompt;
    }
    if event.attachments.is_empty() {
        return String::new();
    }

    let image_count = event
        .attachments
        .iter()
        .filter(|attachment| {
            attachment
                .content_type
                .as_deref()
                .is_some_and(|content_type| content_type.starts_with("image/"))
        })
        .count();
    let file_count = event.attachments.len().saturating_sub(image_count);

    match (image_count, file_count) {
        (1, 0) => "[image attachment]".to_string(),
        (count, 0) => format!("[{count} image attachments]"),
        (0, 1) => "[file attachment]".to_string(),
        (0, count) => format!("[{count} file attachments]"),
        (images, files) => format!("[{images} image attachment(s), {files} file attachment(s)]"),
    }
}

pub(crate) fn task_input_content_from_event(event: &InboundEvent) -> Vec<TaskInputContent> {
    if event.attachments.is_empty() {
        return Vec::new();
    }

    let mut content = Vec::with_capacity(event.attachments.len() + 1);
    let prompt = event.prompt_text();
    if !prompt.trim().is_empty() {
        content.push(TaskInputContent::Text { text: prompt });
    }

    for attachment in &event.attachments {
        let is_image = attachment
            .content_type
            .as_deref()
            .is_some_and(|content_type| content_type.starts_with("image/"));
        if is_image {
            content.push(TaskInputContent::Image {
                name: Some(attachment.name.clone()),
                content_type: attachment.content_type.clone(),
                url: attachment.url.clone(),
                local_path: attachment.local_path.clone(),
                detail: None,
            });
        } else {
            content.push(TaskInputContent::File {
                name: Some(attachment.name.clone()),
                content_type: attachment.content_type.clone(),
                url: attachment.url.clone(),
                local_path: attachment.local_path.clone(),
            });
        }
    }

    content
}
