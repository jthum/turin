use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tracing::warn;
use turin_channel_core::{
    ChannelAttachment, ChannelConversationKey, InboundEvent, OutboundMessage,
};
use turin_channel_runner::ChannelStreamMode;

use crate::{
    TelegramApiError, TelegramChannelDriver,
    outbound::{
        attachment_preview_text, metadata_i64, render_stream_preview,
        telegram_batches_from_message, telegram_edit_payload, telegram_payload,
    },
};

#[derive(Debug, Clone, Deserialize)]
struct TelegramSentMessage {
    message_id: i64,
}

#[derive(Debug, Clone)]
pub(crate) struct TelegramProgressState {
    sink: TelegramProgressSink,
}

#[derive(Debug, Clone)]
enum TelegramProgressSink {
    Draft { draft_id: i64 },
    Placeholder { message_id: i64 },
}

impl TelegramChannelDriver {
    async fn send_batches(
        &self,
        conversation: &ChannelConversationKey,
        message: &OutboundMessage,
    ) -> Result<()> {
        let chat_id = conversation_chat_id(self.config.primary_chat_id(), conversation);
        let message_thread_id = resolve_message_thread_id(conversation)?;
        let reply_to_message_id = metadata_i64(&message.metadata, "telegram_reply_to_message_id")?;
        let payloads = telegram_batches_from_message(&chat_id, message_thread_id, message)?;
        let reply_for_attachments = if payloads.is_empty() {
            reply_to_message_id
        } else {
            None
        };
        for payload in payloads {
            let _: TelegramSentMessage = self
                .request_with_retry("sendMessage", &payload)
                .await
                .map_err(TelegramApiError::into_anyhow)?;
        }
        self.send_attachment_messages(
            &chat_id,
            message_thread_id,
            &message.attachments,
            reply_for_attachments,
        )
        .await?;
        Ok(())
    }

    async fn send_attachment_messages(
        &self,
        chat_id: &str,
        message_thread_id: Option<i64>,
        attachments: &[ChannelAttachment],
        mut reply_to_message_id: Option<i64>,
    ) -> Result<()> {
        for attachment in attachments {
            self.send_attachment_message(
                chat_id,
                message_thread_id,
                attachment,
                reply_to_message_id.take(),
            )
            .await?;
        }
        Ok(())
    }

    async fn send_attachment_message(
        &self,
        chat_id: &str,
        message_thread_id: Option<i64>,
        attachment: &ChannelAttachment,
        reply_to_message_id: Option<i64>,
    ) -> Result<()> {
        let method = if attachment
            .content_type
            .as_deref()
            .is_some_and(|content_type| content_type.starts_with("image/"))
        {
            "sendPhoto"
        } else {
            "sendDocument"
        };
        let field_name = if method == "sendPhoto" {
            "photo"
        } else {
            "document"
        };

        if let Some(local_path) = attachment.local_path.as_deref() {
            let attachment_name = attachment.name.clone();
            let content_type = attachment.content_type.clone();
            let local_path = PathBuf::from(local_path);
            let chat_id = chat_id.to_string();
            let _: TelegramSentMessage = self
                .multipart_request_with_retry(method, || {
                    let bytes = std::fs::read(&local_path).with_context(|| {
                        format!(
                            "Failed to read Telegram attachment '{}'",
                            local_path.display()
                        )
                    })?;
                    let mut form = reqwest::multipart::Form::new().text("chat_id", chat_id.clone());
                    if let Some(message_thread_id) = message_thread_id {
                        form = form.text("message_thread_id", message_thread_id.to_string());
                    }
                    if let Some(reply_to_message_id) = reply_to_message_id {
                        form = form.text("reply_to_message_id", reply_to_message_id.to_string());
                    }

                    let mut part =
                        reqwest::multipart::Part::bytes(bytes).file_name(attachment_name.clone());
                    if let Some(content_type) = &content_type {
                        part = part.mime_str(content_type).with_context(|| {
                            format!(
                                "Invalid Telegram attachment content type '{}'",
                                content_type
                            )
                        })?;
                    }
                    Ok(form.part(field_name.to_string(), part))
                })
                .await
                .map_err(TelegramApiError::into_anyhow)?;
            return Ok(());
        }

        let remote = attachment.url.as_deref().ok_or_else(|| {
            anyhow!(
                "[telegram_send_missing_attachment_source] attachment '{}' is missing local_path and url",
                attachment.name
            )
        })?;
        let mut payload = serde_json::Map::new();
        payload.insert("chat_id".to_string(), serde_json::json!(chat_id));
        payload.insert(field_name.to_string(), serde_json::json!(remote));
        if let Some(message_thread_id) = message_thread_id {
            payload.insert(
                "message_thread_id".to_string(),
                serde_json::json!(message_thread_id),
            );
        }
        if let Some(reply_to_message_id) = reply_to_message_id {
            payload.insert(
                "reply_to_message_id".to_string(),
                serde_json::json!(reply_to_message_id),
            );
        }
        let _: TelegramSentMessage = self
            .request_with_retry(method, &serde_json::Value::Object(payload))
            .await
            .map_err(TelegramApiError::into_anyhow)?;
        Ok(())
    }

    pub(crate) async fn send_chat_action(&mut self, event: &InboundEvent) -> Result<()> {
        let key = progress_key(&event.conversation)?;
        let now = Instant::now();
        if self
            .last_chat_action_at
            .get(&key)
            .is_some_and(|previous| now.duration_since(*previous) < Duration::from_secs(4))
        {
            return Ok(());
        }

        let chat_id = conversation_chat_id(self.config.primary_chat_id(), &event.conversation);
        let message_thread_id = resolve_message_thread_id(&event.conversation)?;
        let mut payload = serde_json::Map::new();
        payload.insert("chat_id".to_string(), serde_json::json!(chat_id));
        payload.insert("action".to_string(), serde_json::json!("typing"));
        if let Some(message_thread_id) = message_thread_id {
            payload.insert(
                "message_thread_id".to_string(),
                serde_json::json!(message_thread_id),
            );
        }

        let _: bool = self
            .request_with_retry("sendChatAction", &serde_json::Value::Object(payload))
            .await
            .map_err(TelegramApiError::into_anyhow)?;
        self.last_chat_action_at.insert(key, now);
        Ok(())
    }

    pub(crate) async fn send_stream_preview(
        &mut self,
        event: &InboundEvent,
        text: &str,
        thinking: Option<&str>,
    ) -> Result<()> {
        let preview = render_stream_preview(text, thinking);
        if preview.is_empty() {
            return Ok(());
        }

        let key = progress_key(&event.conversation)?;
        let chat_id = conversation_chat_id(self.config.primary_chat_id(), &event.conversation);
        let message_thread_id = resolve_message_thread_id(&event.conversation)?;
        let reply_to_message_id = event
            .metadata
            .get("telegram_message_id")
            .and_then(|value| value.as_i64())
            .or_else(|| {
                event
                    .metadata
                    .get("telegram_message_id")
                    .and_then(|value| value.as_str())
                    .and_then(|value| value.parse::<i64>().ok())
            });

        let existing_state = self.progress_states.get(&key).cloned();
        let next_state = match existing_state {
            Some(TelegramProgressState {
                sink: TelegramProgressSink::Draft { draft_id },
            }) => {
                self.send_message_draft(&chat_id, message_thread_id, draft_id, &preview)
                    .await?;
                Some(TelegramProgressState {
                    sink: TelegramProgressSink::Draft { draft_id },
                })
            }
            Some(TelegramProgressState {
                sink: TelegramProgressSink::Placeholder { message_id },
            }) => {
                self.edit_stream_placeholder(&chat_id, message_id, &preview)
                    .await?;
                Some(TelegramProgressState {
                    sink: TelegramProgressSink::Placeholder { message_id },
                })
            }
            None => {
                self.start_progress_sink(&chat_id, message_thread_id, reply_to_message_id, &preview)
                    .await?
            }
        };

        if let Some(state) = next_state {
            self.progress_states.insert(key, state);
        } else {
            self.progress_states.remove(&key);
        }
        Ok(())
    }

    async fn start_progress_sink(
        &mut self,
        chat_id: &str,
        message_thread_id: Option<i64>,
        reply_to_message_id: Option<i64>,
        preview: &str,
    ) -> Result<Option<TelegramProgressState>> {
        if self.config.stream_mode == ChannelStreamMode::Draft && chat_id_is_private(chat_id) {
            let draft_id = self.allocate_draft_id();
            match self
                .send_message_draft(chat_id, message_thread_id, draft_id, preview)
                .await
            {
                Ok(()) => {
                    return Ok(Some(TelegramProgressState {
                        sink: TelegramProgressSink::Draft { draft_id },
                    }));
                }
                Err(err) => {
                    warn!(error = %err, "Telegram draft streaming failed; falling back to placeholder edits");
                }
            }
        }

        let payload = telegram_payload(
            chat_id,
            message_thread_id,
            preview.to_string(),
            None,
            reply_to_message_id,
            true,
            false,
        );
        let sent: TelegramSentMessage = self
            .request_with_retry("sendMessage", &payload)
            .await
            .map_err(TelegramApiError::into_anyhow)?;
        Ok(Some(TelegramProgressState {
            sink: TelegramProgressSink::Placeholder {
                message_id: sent.message_id,
            },
        }))
    }

    async fn send_message_draft(
        &self,
        chat_id: &str,
        message_thread_id: Option<i64>,
        draft_id: i64,
        preview: &str,
    ) -> Result<()> {
        let mut payload = serde_json::Map::new();
        payload.insert("chat_id".to_string(), serde_json::json!(chat_id));
        payload.insert("draft_id".to_string(), serde_json::json!(draft_id));
        payload.insert("text".to_string(), serde_json::json!(preview));
        if let Some(message_thread_id) = message_thread_id {
            payload.insert(
                "message_thread_id".to_string(),
                serde_json::json!(message_thread_id),
            );
        }
        let _: bool = self
            .request_with_retry("sendMessageDraft", &serde_json::Value::Object(payload))
            .await
            .map_err(TelegramApiError::into_anyhow)?;
        Ok(())
    }

    async fn edit_stream_placeholder(
        &self,
        chat_id: &str,
        message_id: i64,
        preview: &str,
    ) -> Result<()> {
        let payload = telegram_edit_payload(chat_id, message_id, preview.to_string(), None, true);
        let _: TelegramSentMessage = self
            .request_with_retry("editMessageText", &payload)
            .await
            .map_err(TelegramApiError::into_anyhow)?;
        Ok(())
    }

    pub(crate) async fn send_final_message(
        &mut self,
        conversation: &ChannelConversationKey,
        message: &OutboundMessage,
    ) -> Result<()> {
        let key = progress_key(conversation)?;
        let progress_state = self.progress_states.remove(&key);
        let attachment_placeholder_id = match progress_state.as_ref() {
            Some(TelegramProgressState {
                sink: TelegramProgressSink::Placeholder { message_id },
            }) => Some(*message_id),
            _ => None,
        };
        let chat_id = conversation_chat_id(self.config.primary_chat_id(), conversation);
        let message_thread_id = resolve_message_thread_id(conversation)?;
        let payloads = telegram_batches_from_message(&chat_id, message_thread_id, message)?;
        let reply_to_message_id = metadata_i64(&message.metadata, "telegram_reply_to_message_id")?;

        if let Some(TelegramProgressState {
            sink: TelegramProgressSink::Placeholder { message_id },
        }) = progress_state
            && let Some((first, rest)) = payloads.split_first()
        {
            let payload = telegram_edit_payload(
                &chat_id,
                message_id,
                first["text"].as_str().unwrap_or_default().to_string(),
                first["parse_mode"].as_str(),
                first["disable_web_page_preview"].as_bool().unwrap_or(true),
            );
            match self
                .request_with_retry::<TelegramSentMessage>("editMessageText", &payload)
                .await
            {
                Ok(_) => {
                    for payload in rest {
                        let _: TelegramSentMessage = self
                            .request_with_retry("sendMessage", payload)
                            .await
                            .map_err(TelegramApiError::into_anyhow)?;
                    }
                    self.send_attachment_messages(
                        &chat_id,
                        message_thread_id,
                        &message.attachments,
                        None,
                    )
                    .await?;
                    return Ok(());
                }
                Err(error) if error.is_message_not_modified() => {
                    for payload in rest {
                        let _: TelegramSentMessage = self
                            .request_with_retry("sendMessage", payload)
                            .await
                            .map_err(TelegramApiError::into_anyhow)?;
                    }
                    self.send_attachment_messages(
                        &chat_id,
                        message_thread_id,
                        &message.attachments,
                        None,
                    )
                    .await?;
                    return Ok(());
                }
                Err(error) => {
                    warn!(
                        error_code = %error.code,
                        error = %error.message,
                        "Telegram placeholder finalization failed; sending final message normally"
                    );
                }
            }
        }

        if payloads.is_empty() && !message.attachments.is_empty() {
            if let Some(message_id) = attachment_placeholder_id {
                let summary = attachment_preview_text(&message.attachments);
                let payload = telegram_edit_payload(&chat_id, message_id, summary, None, true);
                let _ = self
                    .request_with_retry::<TelegramSentMessage>("editMessageText", &payload)
                    .await;
            }
            self.send_attachment_messages(
                &chat_id,
                message_thread_id,
                &message.attachments,
                reply_to_message_id,
            )
            .await?;
            return Ok(());
        }

        self.send_batches(conversation, message).await
    }

    fn allocate_draft_id(&mut self) -> i64 {
        let draft_id = self.next_draft_id.max(1);
        self.next_draft_id = self.next_draft_id.saturating_add(1).max(1);
        draft_id
    }
}

fn progress_key(conversation: &ChannelConversationKey) -> Result<String> {
    serde_json::to_string(conversation)
        .with_context(|| "[telegram_progress_key_invalid] Failed to serialize conversation key")
}

fn conversation_chat_id(default_chat_id: &str, conversation: &ChannelConversationKey) -> String {
    conversation
        .room_id
        .as_ref()
        .filter(|value| !value.trim().is_empty())
        .cloned()
        .unwrap_or_else(|| default_chat_id.to_string())
}

fn chat_id_is_private(chat_id: &str) -> bool {
    !chat_id.trim_start().starts_with('-')
}

fn resolve_message_thread_id(conversation: &ChannelConversationKey) -> Result<Option<i64>> {
    let Some(room_id) = conversation.room_id.as_deref() else {
        return Ok(None);
    };
    if conversation.thread_id == room_id {
        return Ok(None);
    }

    conversation
        .thread_id
        .parse::<i64>()
        .map(Some)
        .with_context(|| {
            format!(
                "[telegram_invalid_thread_id] Telegram conversation thread id '{}' is not a valid numeric message thread id",
                conversation.thread_id
            )
        })
}
