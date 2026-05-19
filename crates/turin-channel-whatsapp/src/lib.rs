use std::fs;
use std::io::Cursor;
use std::path::Path;
#[cfg(test)]
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use async_trait::async_trait;
#[cfg(test)]
use serde_json::json;
use serde_json::{Map, Value};
use tokio::sync::{mpsc, watch};
#[cfg(test)]
use turin_channel_core::DEFAULT_MAX_INBOUND_TEXT_CHARS;
use turin_channel_core::{
    ChannelAttachment, ChannelCapabilities, ChannelConversationKey, ChannelKind, ChannelMessageRef,
    ChannelSessionScope, ChannelUser, InboundEvent, OutboundMessage, bound_inbound_text,
};
#[cfg(test)]
use turin_channel_core::{
    ChannelAuthFlowDisplay, ChannelAuthFlowPollRequest, ChannelAuthFlowPollResponse,
};
use turin_channel_runner::ChannelDriver;
use uuid::Uuid;
use whatsapp_rust::Jid;
use whatsapp_rust::bot::BotHandle;
use whatsapp_rust::download::{Downloadable, MediaType};
use whatsapp_rust::proto_helpers::MessageExt;
use whatsapp_rust::types::message::MessageInfo;
use whatsapp_rust::waproto::whatsapp as wa;

mod auth;
mod bot;
mod manifest;
mod render;
mod settings;
#[cfg(test)]
use auth::{AuthStateWriter, WhatsAppAuthPhase, WhatsAppAuthSession, WhatsAppAuthState};
pub use auth::{poll_auth_flow, run_auth_flow_worker, start_auth_flow};
use bot::{DriverEvent, build_bot};
pub use manifest::adapter_manifest;
use render::render_whatsapp_message;
#[cfg(test)]
pub(crate) use settings::validate_pair_code_fields;
pub(crate) use settings::{WhatsAppAccountMode, parse_settings, sanitize_component};
pub use settings::{WhatsAppChannelDriverConfig, validate_settings};

const DEFAULT_WORKSPACE_ID: &str = "whatsapp";
const DEFAULT_AUTH_FLOW_ID: &str = "pair";
const DEFAULT_AUTH_POLL_INTERVAL_SECONDS: u64 = 3;
const DEFAULT_AUTH_TIMEOUT_SECONDS: u64 = 300;
const DEFAULT_RUNTIME_STORE_BASENAME: &str = "whatsapp-session.db";
const DEFAULT_PERSONAL_TRIGGER_PREFIX: &str = "/turin";

pub struct WhatsAppChannelDriver {
    config: WhatsAppChannelDriverConfig,
    shutdown_rx: watch::Receiver<bool>,
    client: Arc<whatsapp_rust::Client>,
    bot_handle: BotHandle,
    event_rx: mpsc::UnboundedReceiver<DriverEvent>,
}

impl WhatsAppChannelDriver {
    pub async fn from_settings(
        _channel_id: &str,
        settings: &Value,
        runtime_dir: &Path,
        shutdown_rx: watch::Receiver<bool>,
        _allow_unconfigured_chats: bool,
    ) -> Result<Self> {
        let config = parse_settings(settings, Some(runtime_dir))?;
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        let (runtime_client, mut bot) = build_bot(
            &config.session_store_path,
            None,
            None,
            None,
            Some(event_tx.clone()),
        )
        .await
        .with_context(|| {
            format!(
                "Failed to initialize WhatsApp runtime session store '{}'",
                config.session_store_path.display()
            )
        })?;

        let bot_handle = bot
            .run()
            .await
            .context("Failed to start WhatsApp runtime bot")?;

        Ok(Self {
            config,
            shutdown_rx,
            client: runtime_client,
            bot_handle,
            event_rx,
        })
    }

    async fn message_to_event(
        &self,
        message: Box<wa::Message>,
        info: MessageInfo,
    ) -> Result<Option<InboundEvent>> {
        if info.source.is_from_me || info.source.chat.to_string() == "status@broadcast" {
            return Ok(None);
        }

        let chat_id = info.source.chat.to_string();
        if !chat_is_allowed(
            &chat_id,
            &self.config.allowed_chats,
            &self.config.banned_chats,
        ) {
            return Ok(None);
        }

        let base_message = message.get_base_message();
        let attachments: Vec<ChannelAttachment> = self
            .collect_inbound_attachments(base_message, &info.id)
            .await?;
        let raw_text = message.text_content().or_else(|| message.get_caption());
        let text = match raw_text {
            Some(text) => match inbound_text(
                text,
                self.config.account_mode,
                self.config.trigger_prefix.as_deref(),
            ) {
                Some(value) => value,
                None => return Ok(None),
            },
            None if attachments.is_empty() => return Ok(None),
            None if matches!(self.config.account_mode, WhatsAppAccountMode::Personal)
                && self.config.trigger_prefix.is_some() =>
            {
                return Ok(None);
            }
            None => String::new(),
        };
        let sender_id = info.source.sender.to_string();
        let thread_id = match self.config.session_scope {
            ChannelSessionScope::User if info.source.is_group => {
                format!("room:{chat_id}:user:{sender_id}")
            }
            ChannelSessionScope::User => format!("user:{sender_id}"),
            ChannelSessionScope::Thread => format!("room:{chat_id}:user:{sender_id}"),
            ChannelSessionScope::Room => format!("room:{chat_id}"),
        };

        let conversation = ChannelConversationKey {
            channel: ChannelKind::new("whatsapp"),
            workspace_id: self.config.workspace_id.clone(),
            room_id: Some(chat_id.clone()),
            thread_id,
            user_id: Some(sender_id.clone()),
        };

        let mut metadata = Map::new();
        metadata.insert("chat_jid".to_string(), Value::String(chat_id.clone()));
        metadata.insert("sender_jid".to_string(), Value::String(sender_id.clone()));
        metadata.insert("is_group".to_string(), Value::Bool(info.source.is_group));
        let text = bound_inbound_text(text, &mut metadata, self.config.max_inbound_text_chars);

        Ok(Some(InboundEvent {
            message: ChannelMessageRef {
                conversation: conversation.clone(),
                message_id: info.id,
            },
            conversation,
            user: ChannelUser {
                id: sender_id,
                display_name: None,
                username: None,
            },
            session_scope: self.config.session_scope,
            text,
            attachments,
            metadata,
        }))
    }

    async fn collect_inbound_attachments(
        &self,
        message: &wa::Message,
        message_id: &str,
    ) -> Result<Vec<ChannelAttachment>> {
        fs::create_dir_all(&self.config.media_dir).with_context(|| {
            format!(
                "Failed to create WhatsApp media directory '{}'",
                self.config.media_dir.display()
            )
        })?;

        let mut attachments = Vec::new();
        if let Some(image) = &message.image_message {
            attachments.push(
                self.download_whatsapp_attachment(
                    &**image,
                    message_id,
                    image.mimetype.clone(),
                    image_name(image, message_id),
                )
                .await?,
            );
        }
        if let Some(document) = &message.document_message {
            attachments.push(
                self.download_whatsapp_attachment(
                    &**document,
                    message_id,
                    document.mimetype.clone(),
                    document_name(document, message_id),
                )
                .await?,
            );
        }
        if let Some(video) = &message.video_message {
            attachments.push(
                self.download_whatsapp_attachment(
                    &**video,
                    message_id,
                    video.mimetype.clone(),
                    format!("video-{message_id}.mp4"),
                )
                .await?,
            );
        }
        if let Some(audio) = &message.audio_message {
            attachments.push(
                self.download_whatsapp_attachment(
                    &**audio,
                    message_id,
                    audio.mimetype.clone(),
                    format!("audio-{message_id}.ogg"),
                )
                .await?,
            );
        }
        Ok(attachments)
    }

    async fn download_whatsapp_attachment<D: Downloadable>(
        &self,
        media: &D,
        message_id: &str,
        content_type: Option<String>,
        suggested_name: String,
    ) -> Result<ChannelAttachment> {
        let mut data = Cursor::new(Vec::new());
        self.client
            .download_to_file(media, &mut data)
            .await
            .context("Failed to download WhatsApp media attachment")?;
        let target_path = self.config.media_dir.join(format!(
            "{}-{}",
            Uuid::new_v4(),
            sanitize_component(&suggested_name)
        ));
        fs::write(&target_path, data.into_inner()).with_context(|| {
            format!(
                "Failed to write WhatsApp media attachment '{}'",
                target_path.display()
            )
        })?;
        let final_name = if Path::new(&suggested_name).extension().is_some() {
            suggested_name
        } else {
            infer_media_name(message_id, content_type.as_deref(), &suggested_name)
        };
        Ok(ChannelAttachment {
            name: final_name,
            content_type,
            url: None,
            local_path: Some(target_path.display().to_string()),
        })
    }
}

impl WhatsAppChannelDriver {
    async fn send_attachment(
        &self,
        chat: Jid,
        attachment: &turin_channel_core::ChannelAttachment,
    ) -> Result<()> {
        let local_path = attachment.local_path.as_deref().ok_or_else(|| {
            anyhow!(
                "[whatsapp_send_missing_attachment_source] attachment '{}' is missing local_path",
                attachment.name
            )
        })?;
        let bytes = fs::read(local_path)
            .with_context(|| format!("Failed to read WhatsApp attachment '{}'", local_path))?;
        let media_type = whatsapp_media_type(attachment.content_type.as_deref());
        let upload = self
            .client
            .upload(bytes, media_type)
            .await
            .context("Failed to upload WhatsApp attachment")?;
        let mime_type = attachment
            .content_type
            .clone()
            .or_else(|| whatsapp_default_mime_type(media_type).map(str::to_string));
        let message = match media_type {
            MediaType::Image => wa::Message {
                image_message: Some(Box::new(wa::message::ImageMessage {
                    mimetype: mime_type,
                    url: Some(upload.url),
                    direct_path: Some(upload.direct_path),
                    media_key: Some(upload.media_key),
                    file_enc_sha256: Some(upload.file_enc_sha256),
                    file_sha256: Some(upload.file_sha256),
                    file_length: Some(upload.file_length),
                    ..Default::default()
                })),
                ..Default::default()
            },
            _ => wa::Message {
                document_message: Some(Box::new(wa::message::DocumentMessage {
                    mimetype: mime_type,
                    title: Some(attachment.name.clone()),
                    file_name: Some(attachment.name.clone()),
                    url: Some(upload.url),
                    direct_path: Some(upload.direct_path),
                    media_key: Some(upload.media_key),
                    file_enc_sha256: Some(upload.file_enc_sha256),
                    file_sha256: Some(upload.file_sha256),
                    file_length: Some(upload.file_length),
                    ..Default::default()
                })),
                ..Default::default()
            },
        };
        self.client
            .send_message(chat, message)
            .await
            .context("Failed to send WhatsApp attachment")?;
        Ok(())
    }
}

#[async_trait]
impl ChannelDriver for WhatsAppChannelDriver {
    fn kind(&self) -> ChannelKind {
        ChannelKind::new("whatsapp")
    }

    fn user_matches_selector(&self, selector: &str, user: &ChannelUser) -> bool {
        let selector = selector.trim().trim_start_matches('@');
        if selector.is_empty() {
            return false;
        }
        user.id.eq_ignore_ascii_case(selector)
            || user
                .id
                .split('@')
                .next()
                .is_some_and(|phone| phone.eq_ignore_ascii_case(selector))
            || user
                .username
                .as_ref()
                .is_some_and(|username| username.eq_ignore_ascii_case(selector))
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities {
            rich_formatting: false,
            threads: false,
            attachments: true,
            ephemeral_messages: false,
        }
    }

    async fn next_event(&mut self) -> Result<Option<InboundEvent>> {
        loop {
            tokio::select! {
                changed = self.shutdown_rx.changed() => {
                    if changed.is_err() || *self.shutdown_rx.borrow() {
                        return Ok(None);
                    }
                }
                maybe_event = self.event_rx.recv() => {
                    match maybe_event {
                        Some(DriverEvent::Message(message, info)) => {
                            if let Some(event) = self.message_to_event(message, *info).await? {
                                return Ok(Some(event));
                            }
                        }
                        Some(DriverEvent::LoggedOut(reason)) => {
                            bail!("WhatsApp linked session was logged out: {reason}");
                        }
                        None => return Ok(None),
                    }
                }
            }
        }
    }

    async fn send(
        &mut self,
        conversation: &ChannelConversationKey,
        message: OutboundMessage,
    ) -> Result<()> {
        let room_id = conversation
            .room_id
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                anyhow!("[whatsapp_send_missing_room] outbound conversation is missing room_id")
            })?;
        let chat: Jid = room_id
            .parse()
            .with_context(|| format!("Invalid WhatsApp chat JID '{room_id}'"))?;
        let rendered = render_whatsapp_message(&message);
        if !rendered.trim().is_empty() {
            self.client
                .send_message(
                    chat.clone(),
                    wa::Message {
                        conversation: Some(rendered),
                        ..Default::default()
                    },
                )
                .await
                .context("Failed to send WhatsApp message")?;
        }
        for attachment in &message.attachments {
            self.send_attachment(chat.clone(), attachment).await?;
        }
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<()> {
        self.client.disconnect().await;
        self.bot_handle.abort();
        Ok(())
    }
}

fn image_name(message: &wa::message::ImageMessage, message_id: &str) -> String {
    let extension = content_type_extension(message.mimetype.as_deref()).unwrap_or("jpg");
    format!("image-{message_id}.{extension}")
}

fn document_name(message: &wa::message::DocumentMessage, message_id: &str) -> String {
    message
        .file_name
        .clone()
        .or_else(|| message.title.clone())
        .unwrap_or_else(|| {
            let extension = content_type_extension(message.mimetype.as_deref()).unwrap_or("bin");
            format!("document-{message_id}.{extension}")
        })
}

fn infer_media_name(message_id: &str, content_type: Option<&str>, fallback_stem: &str) -> String {
    if let Some(extension) = content_type_extension(content_type) {
        format!("{fallback_stem}.{extension}")
    } else {
        format!("{fallback_stem}-{message_id}")
    }
}

fn content_type_extension(content_type: Option<&str>) -> Option<&'static str> {
    match content_type.unwrap_or_default() {
        "image/jpeg" => Some("jpg"),
        "image/png" => Some("png"),
        "image/webp" => Some("webp"),
        "application/pdf" => Some("pdf"),
        "video/mp4" => Some("mp4"),
        "audio/mpeg" => Some("mp3"),
        "audio/ogg" => Some("ogg"),
        _ => None,
    }
}

fn whatsapp_media_type(content_type: Option<&str>) -> MediaType {
    match content_type.unwrap_or_default() {
        value if value.starts_with("image/") => MediaType::Image,
        value if value.starts_with("audio/") => MediaType::Audio,
        value if value.starts_with("video/") => MediaType::Video,
        _ => MediaType::Document,
    }
}

fn whatsapp_default_mime_type(media_type: MediaType) -> Option<&'static str> {
    match media_type {
        MediaType::Image => Some("image/jpeg"),
        MediaType::Video => Some("video/mp4"),
        MediaType::Audio => Some("audio/ogg"),
        MediaType::Document => Some("application/octet-stream"),
        _ => None,
    }
}

fn inbound_text(
    raw: &str,
    account_mode: WhatsAppAccountMode,
    trigger_prefix: Option<&str>,
) -> Option<String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }

    let required_prefix = trigger_prefix.or(match account_mode {
        WhatsAppAccountMode::Personal => Some(DEFAULT_PERSONAL_TRIGGER_PREFIX),
        WhatsAppAccountMode::Dedicated => None,
    });

    let Some(prefix) = required_prefix else {
        return Some(trimmed.to_string());
    };

    let candidate = trimmed.strip_prefix(prefix)?.trim_start();
    if candidate.is_empty() {
        None
    } else {
        Some(candidate.to_string())
    }
}

fn chat_is_allowed(chat_jid: &str, allowed_chats: &[String], banned_chats: &[String]) -> bool {
    if selector_matches_chat_list(chat_jid, banned_chats) {
        return false;
    }
    allowed_chats.is_empty() || selector_matches_chat_list(chat_jid, allowed_chats)
}

fn selector_matches_chat_list(chat_jid: &str, selectors: &[String]) -> bool {
    selectors
        .iter()
        .any(|selector| chat_selector_matches(selector, chat_jid))
}

fn chat_selector_matches(selector: &str, chat_jid: &str) -> bool {
    let selector = selector.trim();
    if selector.is_empty() {
        return false;
    }

    selector.eq_ignore_ascii_case(chat_jid)
        || selector
            .strip_prefix('@')
            .is_some_and(|value| value.eq_ignore_ascii_case(chat_jid))
        || selector.eq_ignore_ascii_case(chat_jid.split('@').next().unwrap_or(chat_jid))
}

#[cfg(test)]
mod tests;
