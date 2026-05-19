use serde::Deserialize;
use turin_channel_core::{
    ChannelConversationKey, ChannelKind, ChannelMessageRef, ChannelSessionScope, ChannelUser,
    InboundEvent, bound_inbound_text,
};

use crate::{
    TelegramChannelDriver, TelegramChannelDriverConfig, TelegramRespondMode,
    outbound::{attachment_kind_from_content_type, infer_audio_name},
};

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct TelegramUpdate {
    pub(crate) update_id: i64,
    #[serde(default)]
    pub(crate) message: Option<TelegramMessage>,
    #[serde(default)]
    pub(crate) channel_post: Option<TelegramMessage>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct TelegramMessage {
    pub(crate) message_id: i64,
    pub(crate) chat: TelegramChat,
    #[serde(default)]
    pub(crate) from: Option<TelegramUser>,
    #[serde(default)]
    pub(crate) sender_chat: Option<TelegramChat>,
    #[serde(default)]
    pub(crate) text: Option<String>,
    #[serde(default)]
    pub(crate) caption: Option<String>,
    #[serde(default)]
    pub(crate) entities: Vec<TelegramMessageEntity>,
    #[serde(default)]
    pub(crate) caption_entities: Vec<TelegramMessageEntity>,
    #[serde(default)]
    pub(crate) photo: Vec<TelegramPhotoSize>,
    #[serde(default)]
    pub(crate) document: Option<TelegramDocument>,
    #[serde(default)]
    pub(crate) video: Option<TelegramVideo>,
    #[serde(default)]
    pub(crate) audio: Option<TelegramAudio>,
    #[serde(default)]
    pub(crate) voice: Option<TelegramVoice>,
    #[serde(default)]
    pub(crate) message_thread_id: Option<i64>,
    #[serde(default)]
    pub(crate) reply_to_message: Option<Box<TelegramMessage>>,
}

impl TelegramMessage {
    fn channel_user(&self) -> Option<ChannelUser> {
        if let Some(user) = &self.from {
            let display_name = match (&user.first_name, &user.last_name) {
                (Some(first), Some(last)) if !last.trim().is_empty() => {
                    Some(format!("{} {}", first, last))
                }
                (Some(first), _) => Some(first.clone()),
                _ => user.username.clone(),
            };
            return Some(ChannelUser {
                id: user.id.to_string(),
                display_name,
                username: user.username.clone(),
            });
        }

        self.sender_chat.as_ref().map(|chat| ChannelUser {
            id: chat.id.to_string(),
            display_name: chat
                .title
                .clone()
                .or_else(|| chat.first_name.clone())
                .or_else(|| chat.username.clone()),
            username: chat.username.clone(),
        })
    }

    fn body_text(&self) -> Option<&String> {
        self.text.as_ref().or(self.caption.as_ref())
    }

    fn body_entities(&self) -> &[TelegramMessageEntity] {
        if self.text.is_some() {
            &self.entities
        } else {
            &self.caption_entities
        }
    }

    pub(crate) fn attachment_refs(&self) -> Vec<TelegramAttachmentRef> {
        let mut attachments = Vec::new();
        if let Some(photo) = self.photo.iter().max_by_key(|photo| {
            (
                u64::from(photo.width) * u64::from(photo.height),
                photo.file_size.unwrap_or_default(),
            )
        }) {
            attachments.push(TelegramAttachmentRef {
                file_id: photo.file_id.clone(),
                name: photo
                    .file_unique_id
                    .as_deref()
                    .map(|id| format!("{id}.jpg"))
                    .unwrap_or_else(|| format!("photo-{}.jpg", self.message_id)),
                content_type: Some("image/jpeg".to_string()),
                kind: TelegramAttachmentKind::Image,
            });
        }
        if let Some(document) = &self.document {
            attachments.push(TelegramAttachmentRef {
                file_id: document.file_id.clone(),
                name: document
                    .file_name
                    .clone()
                    .unwrap_or_else(|| format!("document-{}", self.message_id)),
                content_type: document.mime_type.clone(),
                kind: attachment_kind_from_content_type(document.mime_type.as_deref()),
            });
        }
        if let Some(video) = &self.video {
            attachments.push(TelegramAttachmentRef {
                file_id: video.file_id.clone(),
                name: video
                    .file_name
                    .clone()
                    .unwrap_or_else(|| format!("video-{}.mp4", self.message_id)),
                content_type: video
                    .mime_type
                    .clone()
                    .or_else(|| Some("video/mp4".to_string())),
                kind: TelegramAttachmentKind::File,
            });
        }
        if let Some(audio) = &self.audio {
            attachments.push(TelegramAttachmentRef {
                file_id: audio.file_id.clone(),
                name: audio.file_name.clone().unwrap_or_else(|| {
                    infer_audio_name(audio).unwrap_or_else(|| format!("audio-{}", self.message_id))
                }),
                content_type: audio.mime_type.clone(),
                kind: TelegramAttachmentKind::File,
            });
        }
        if let Some(voice) = &self.voice {
            attachments.push(TelegramAttachmentRef {
                file_id: voice.file_id.clone(),
                name: format!("voice-{}.ogg", self.message_id),
                content_type: voice
                    .mime_type
                    .clone()
                    .or_else(|| Some("audio/ogg".to_string())),
                kind: TelegramAttachmentKind::File,
            });
        }
        attachments
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct TelegramUser {
    pub(crate) id: i64,
    #[serde(default)]
    pub(crate) is_bot: Option<bool>,
    #[serde(default)]
    pub(crate) first_name: Option<String>,
    #[serde(default)]
    pub(crate) last_name: Option<String>,
    #[serde(default)]
    pub(crate) username: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct TelegramPhotoSize {
    file_id: String,
    #[serde(default)]
    file_unique_id: Option<String>,
    width: u32,
    height: u32,
    #[serde(default)]
    file_size: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct TelegramDocument {
    file_id: String,
    #[serde(default)]
    file_name: Option<String>,
    #[serde(default)]
    mime_type: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct TelegramVideo {
    file_id: String,
    #[serde(default)]
    file_name: Option<String>,
    #[serde(default)]
    mime_type: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct TelegramAudio {
    file_id: String,
    #[serde(default)]
    file_name: Option<String>,
    #[serde(default)]
    mime_type: Option<String>,
    #[serde(default)]
    pub(crate) performer: Option<String>,
    #[serde(default)]
    pub(crate) title: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct TelegramVoice {
    file_id: String,
    #[serde(default)]
    mime_type: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct TelegramFile {
    #[serde(default)]
    pub(crate) file_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TelegramAttachmentKind {
    Image,
    File,
}

#[derive(Debug, Clone)]
pub(crate) struct TelegramAttachmentRef {
    pub(crate) file_id: String,
    pub(crate) name: String,
    pub(crate) content_type: Option<String>,
    pub(crate) kind: TelegramAttachmentKind,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct TelegramMessageEntity {
    #[serde(rename = "type")]
    pub(crate) kind: String,
    pub(crate) offset: usize,
    pub(crate) length: usize,
    #[serde(default)]
    pub(crate) user: Option<TelegramUser>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct TelegramChat {
    pub(crate) id: i64,
    #[serde(rename = "type")]
    pub(crate) chat_type: String,
    #[serde(default)]
    pub(crate) title: Option<String>,
    #[serde(default)]
    pub(crate) username: Option<String>,
    #[serde(default)]
    pub(crate) first_name: Option<String>,
}

impl TelegramChat {
    fn is_private(&self) -> bool {
        self.chat_type == "private"
    }
}

impl TelegramChannelDriver {
    pub(crate) fn advance_offset(&mut self, updates: &[TelegramUpdate]) {
        if let Some(next) = updates.iter().map(|update| update.update_id).max() {
            self.next_update_offset = Some(next.saturating_add(1));
        }
    }

    #[cfg(test)]
    pub(crate) fn normalize_update(&self, update: TelegramUpdate) -> Option<InboundEvent> {
        let update_id = update.update_id;
        let message = update.message.or(update.channel_post)?;
        self.normalize_message(update_id, message)
    }

    pub(crate) fn normalize_message(
        &self,
        update_id: i64,
        message: TelegramMessage,
    ) -> Option<InboundEvent> {
        let chat_id = message.chat.id.to_string();
        if !self.config.accept_all_chats && !self.config.allows_chat_id(&chat_id) {
            return None;
        }

        if self.config.ignore_bot_messages
            && message.from.as_ref().and_then(|user| user.is_bot) == Some(true)
        {
            return None;
        }

        if !self.should_accept_message(&message) {
            return None;
        }

        let text = message
            .body_text()
            .map(|value| value.trim().to_string())
            .unwrap_or_default();

        let user = message.channel_user()?;
        let scoped_thread_id = message
            .message_thread_id
            .map(|value| value.to_string())
            .unwrap_or_else(|| chat_id.clone());

        let mut metadata = serde_json::Map::new();
        metadata.insert(
            "telegram_update_id".to_string(),
            serde_json::json!(update_id),
        );
        metadata.insert(
            "telegram_message_id".to_string(),
            serde_json::json!(message.message_id),
        );
        metadata.insert(
            "telegram_chat_id".to_string(),
            serde_json::json!(message.chat.id),
        );
        if let Some(message_thread_id) = message.message_thread_id {
            metadata.insert(
                "telegram_message_thread_id".to_string(),
                serde_json::json!(message_thread_id),
            );
        }
        metadata.insert(
            "telegram_chat_type".to_string(),
            serde_json::json!(message.chat.chat_type),
        );
        let text = bound_inbound_text(text, &mut metadata, self.config.max_inbound_text_chars);

        let session_scope = effective_telegram_session_scope(&self.config, &message.chat);
        let conversation = ChannelConversationKey {
            channel: ChannelKind::new("telegram"),
            workspace_id: self.config.workspace_id.clone(),
            room_id: Some(chat_id.clone()),
            thread_id: match session_scope {
                ChannelSessionScope::User | ChannelSessionScope::Thread => scoped_thread_id,
                ChannelSessionScope::Room => chat_id,
            },
            user_id: match session_scope {
                ChannelSessionScope::User => Some(user.id.clone()),
                ChannelSessionScope::Thread | ChannelSessionScope::Room => None,
            },
        };

        Some(InboundEvent {
            message: ChannelMessageRef {
                conversation: conversation.clone(),
                message_id: message.message_id.to_string(),
            },
            conversation,
            user,
            session_scope,
            text,
            attachments: Vec::new(),
            metadata,
        })
    }

    fn should_accept_message(&self, message: &TelegramMessage) -> bool {
        if message.chat.is_private() {
            return true;
        }

        match self.config.respond_mode {
            TelegramRespondMode::All => true,
            TelegramRespondMode::Mentions => {
                self.message_mentions_bot(message) || self.message_targets_bot_command(message)
            }
            TelegramRespondMode::Replies => self.message_replies_to_bot(message),
            TelegramRespondMode::MentionsOrReplies => {
                self.message_mentions_bot(message)
                    || self.message_targets_bot_command(message)
                    || self.message_replies_to_bot(message)
            }
        }
    }

    fn message_mentions_bot(&self, message: &TelegramMessage) -> bool {
        let Some(identity) = self.bot_identity.as_ref() else {
            return false;
        };
        let Some(username) = identity.username.as_deref() else {
            return false;
        };
        let Some(body) = message.body_text() else {
            return false;
        };
        let mention = format!("@{}", username);

        for entity in message.body_entities() {
            match entity.kind.as_str() {
                "mention" => {
                    let Some(slice) = utf16_slice(body, entity.offset, entity.length) else {
                        continue;
                    };
                    if slice.eq_ignore_ascii_case(&mention) {
                        return true;
                    }
                }
                "text_mention" if entity.user.as_ref().map(|user| user.id) == Some(identity.id) => {
                    return true;
                }
                _ => {}
            }
        }

        false
    }

    fn message_targets_bot_command(&self, message: &TelegramMessage) -> bool {
        let Some(identity) = self.bot_identity.as_ref() else {
            return false;
        };
        let Some(username) = identity.username.as_deref() else {
            return false;
        };
        let Some(body) = message.body_text() else {
            return false;
        };

        for entity in message.body_entities() {
            if entity.kind != "bot_command" {
                continue;
            }
            let Some(slice) = utf16_slice(body, entity.offset, entity.length) else {
                continue;
            };
            let Some((_, target)) = slice.split_once('@') else {
                continue;
            };
            if target.eq_ignore_ascii_case(username) {
                return true;
            }
        }

        false
    }

    fn message_replies_to_bot(&self, message: &TelegramMessage) -> bool {
        let Some(replied) = message.reply_to_message.as_deref() else {
            return false;
        };
        let Some(identity) = self.bot_identity.as_ref() else {
            return false;
        };

        if replied.from.as_ref().map(|user| user.id) == Some(identity.id) {
            return true;
        }

        replied
            .from
            .as_ref()
            .and_then(|user| user.username.as_deref())
            .zip(identity.username.as_deref())
            .is_some_and(|(reply_username, bot_username)| {
                reply_username.eq_ignore_ascii_case(bot_username)
            })
    }
}

pub(crate) fn effective_telegram_session_scope(
    config: &TelegramChannelDriverConfig,
    chat: &TelegramChat,
) -> ChannelSessionScope {
    match chat.chat_type.as_str() {
        "private" => config.session_scope_dm.unwrap_or(config.session_scope),
        "channel" => config.session_scope_channel.unwrap_or(config.session_scope),
        "group" | "supergroup" => config.session_scope_group.unwrap_or(config.session_scope),
        _ => config.session_scope,
    }
}

fn utf16_slice(text: &str, offset: usize, length: usize) -> Option<&str> {
    let end = offset.saturating_add(length);
    let mut utf16_index = 0usize;
    let mut start_byte = None;
    let mut end_byte = None;

    for (byte_index, ch) in text.char_indices() {
        if utf16_index == offset {
            start_byte = Some(byte_index);
        }
        if utf16_index == end {
            end_byte = Some(byte_index);
            break;
        }

        utf16_index = utf16_index.saturating_add(ch.len_utf16());

        if utf16_index == offset {
            start_byte = Some(byte_index + ch.len_utf8());
        }
        if utf16_index == end {
            end_byte = Some(byte_index + ch.len_utf8());
            break;
        }
    }

    if offset == utf16_index && start_byte.is_none() {
        start_byte = Some(text.len());
    }
    if end == utf16_index && end_byte.is_none() {
        end_byte = Some(text.len());
    }

    Some(&text[start_byte?..end_byte?])
}
