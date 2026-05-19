use anyhow::{Result, anyhow};
use turin_channel_core::{
    ChannelAttachment, ChannelConversationKey, ChannelKind, ChannelMessageRef, ChannelSessionScope,
    ChannelUser, InboundEvent, bound_inbound_text,
};

use crate::{
    RocketChatChannelDriver, RocketChatMessage, RocketChatMessageUser, RocketChatReplyMode,
    RocketChatResolvedRoom, RocketChatRespondMode, RocketChatRoomType, absolute_url,
    active_thread_key,
};

impl RocketChatChannelDriver {
    pub(crate) fn message_to_event(
        &self,
        room: &RocketChatResolvedRoom,
        message: RocketChatMessage,
    ) -> Result<Option<InboundEvent>> {
        if message.kind.is_some() {
            return Ok(None);
        }

        let user = message.user.as_ref().ok_or_else(|| {
            anyhow!(
                "[rocketchat_message_missing_user] Rocket.Chat message '{}' is missing user metadata",
                message.id
            )
        })?;

        if self.config.ignore_bot_messages && user.id == self.config.user_id {
            return Ok(None);
        }

        if !self.should_accept_message(room, &message, user) {
            return Ok(None);
        }

        let mut text = message.text.clone().unwrap_or_default();
        let attachments = collect_attachments(&self.config.base_url, &message);
        if text.trim().is_empty() && attachments.is_empty() {
            return Ok(None);
        }
        if text.trim().is_empty() && !attachments.is_empty() {
            text = "[Attachment]".to_string();
        }

        let user = ChannelUser {
            id: user.id.clone(),
            display_name: user.name.clone(),
            username: user.username.clone(),
        };
        let session_scope = self.effective_session_scope(room);
        let conversation = ChannelConversationKey {
            channel: ChannelKind::new("rocketchat"),
            workspace_id: self.config.workspace_id.clone(),
            room_id: Some(room.id.clone()),
            thread_id: self.thread_id_for_message(room, &message, session_scope),
            user_id: if matches!(session_scope, ChannelSessionScope::User) {
                Some(user.id.clone())
            } else {
                None
            },
        };

        let mut metadata = serde_json::Map::new();
        metadata.insert(
            "rocketchat_message_id".to_string(),
            serde_json::json!(message.id),
        );
        metadata.insert(
            "rocketchat_message_ts".to_string(),
            serde_json::json!(message.ts),
        );
        metadata.insert("rocketchat_room_id".to_string(), serde_json::json!(room.id));
        if let Some(message_link) = build_rocketchat_message_link(
            &self.config.base_url,
            room,
            self.bot_username.as_deref(),
            &message.id,
        ) {
            metadata.insert(
                "rocketchat_message_link".to_string(),
                serde_json::json!(message_link),
            );
        }
        if let Some(tmid) = message.thread_root_id {
            metadata.insert("rocketchat_thread_id".to_string(), serde_json::json!(tmid));
        }
        text = bound_inbound_text(text, &mut metadata, self.config.max_inbound_text_chars);

        Ok(Some(InboundEvent {
            message: ChannelMessageRef {
                conversation: conversation.clone(),
                message_id: metadata["rocketchat_message_id"]
                    .as_str()
                    .expect("message id inserted")
                    .to_string(),
            },
            conversation,
            user,
            session_scope,
            text,
            attachments,
            metadata,
        }))
    }

    pub(crate) fn effective_session_scope(
        &self,
        room: &RocketChatResolvedRoom,
    ) -> ChannelSessionScope {
        let configured_scope = match room.room_type {
            RocketChatRoomType::DirectMessage => self
                .config
                .session_scope_dm
                .unwrap_or(self.config.session_scope),
            RocketChatRoomType::Channel => self
                .config
                .session_scope_channel
                .unwrap_or(self.config.session_scope),
            RocketChatRoomType::PrivateGroup => self
                .config
                .session_scope_group
                .unwrap_or(self.config.session_scope),
        };

        if matches!(self.config.reply_mode, RocketChatReplyMode::Channel)
            && matches!(configured_scope, ChannelSessionScope::Thread)
        {
            ChannelSessionScope::Room
        } else {
            configured_scope
        }
    }

    pub(crate) fn should_accept_message(
        &self,
        room: &RocketChatResolvedRoom,
        message: &RocketChatMessage,
        user: &RocketChatMessageUser,
    ) -> bool {
        if matches!(room.room_type, RocketChatRoomType::DirectMessage) {
            return user.id != self.config.user_id || !self.config.ignore_bot_messages;
        }

        match self.config.respond_mode {
            RocketChatRespondMode::All => true,
            RocketChatRespondMode::Mentions => {
                message
                    .mentions
                    .iter()
                    .any(|mention| mention.id.as_deref() == Some(self.config.user_id.as_str()))
                    || self.message_quotes_bot_reply(message)
                    || message.thread_root_id.as_deref().is_some_and(|thread_id| {
                        self.active_thread_keys
                            .contains(&active_thread_key(&room.id, thread_id))
                    })
            }
        }
    }

    fn message_quotes_bot_reply(&self, message: &RocketChatMessage) -> bool {
        message.attachments.iter().any(|attachment| {
            attachment.message_link.as_deref().is_some_and(|link| {
                self.recent_sent_message_ids
                    .iter()
                    .any(|message_id| link.contains(message_id))
            }) || attachment
                .author_name
                .as_deref()
                .is_some_and(|author_name| self.is_bot_identity_label(author_name))
        })
    }

    pub(crate) fn thread_id_for_message(
        &self,
        room: &RocketChatResolvedRoom,
        message: &RocketChatMessage,
        session_scope: ChannelSessionScope,
    ) -> String {
        match session_scope {
            ChannelSessionScope::Room => room.id.clone(),
            ChannelSessionScope::Thread => message
                .thread_root_id
                .clone()
                .unwrap_or_else(|| message.id.clone()),
            ChannelSessionScope::User => message
                .thread_root_id
                .clone()
                .unwrap_or_else(|| room.id.clone()),
        }
    }
}

fn collect_attachments(base_url: &str, message: &RocketChatMessage) -> Vec<ChannelAttachment> {
    let mut attachments = Vec::new();
    if let Some(file) = &message.file {
        attachments.push(ChannelAttachment {
            name: file.name.clone(),
            content_type: file.content_type.clone(),
            url: file.url.as_ref().map(|url| absolute_url(base_url, url)),
            local_path: None,
        });
    }
    for attachment in &message.attachments {
        if let Some(url) = attachment
            .title_link
            .as_ref()
            .or(attachment.image_url.as_ref())
            .or(attachment.audio_url.as_ref())
            .or(attachment.video_url.as_ref())
        {
            attachments.push(ChannelAttachment {
                name: attachment
                    .title
                    .clone()
                    .or_else(|| attachment.text.clone())
                    .unwrap_or_else(|| "attachment".to_string()),
                content_type: None,
                url: Some(absolute_url(base_url, url)),
                local_path: None,
            });
        }
    }
    attachments
}

pub(crate) fn build_rocketchat_message_link(
    base_url: &str,
    room: &RocketChatResolvedRoom,
    bot_username: Option<&str>,
    message_id: &str,
) -> Option<String> {
    let path = match room.room_type {
        RocketChatRoomType::Channel => {
            let name = room.name.as_deref().or(room.friendly_name.as_deref())?;
            Some(format!("channel/{}", name.trim_start_matches('#')))
        }
        RocketChatRoomType::PrivateGroup => {
            let name = room.name.as_deref().or(room.friendly_name.as_deref())?;
            Some(format!("group/{}", name.trim_start_matches('#')))
        }
        RocketChatRoomType::DirectMessage => {
            let bot_username = bot_username.unwrap_or_default();
            room.usernames
                .iter()
                .find(|username| {
                    let username = username.trim();
                    !username.is_empty() && username != bot_username
                })
                .map(|username| format!("direct/{}", username))
        }
    }?;

    Some(format!(
        "{}/{}?msg={}",
        base_url.trim_end_matches('/'),
        path,
        message_id
    ))
}
