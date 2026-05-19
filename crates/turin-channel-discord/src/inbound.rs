use turin_channel_core::{
    ChannelAttachment, ChannelConversationKey, ChannelKind, ChannelMessageRef, ChannelSessionScope,
    ChannelUser, InboundEvent, bound_inbound_text,
};

use crate::{DiscordChannelDriver, SEEN_MESSAGE_IDS_LIMIT, api::DiscordMessage};

impl DiscordChannelDriver {
    pub(crate) fn normalize_message(&mut self, message: DiscordMessage) -> Option<InboundEvent> {
        if !self.track_seen_message(&message.id) {
            return None;
        }
        if self.config.ignore_bot_messages && message.author.bot.unwrap_or(false) {
            return None;
        }
        if message.content.trim().is_empty() && message.attachments.is_empty() {
            return None;
        }

        let room_id = self
            .config
            .room_id
            .clone()
            .or(message.guild_id.clone())
            .or(Some(self.config.channel_id.clone()));

        let attachments = message
            .attachments
            .into_iter()
            .map(|attachment| ChannelAttachment {
                name: attachment.filename,
                content_type: attachment.content_type,
                url: Some(attachment.url),
                local_path: None,
            })
            .collect();

        let conversation = ChannelConversationKey {
            channel: ChannelKind::new("discord"),
            workspace_id: self.config.workspace_id.clone(),
            room_id,
            thread_id: message.channel_id.clone(),
            user_id: match self.config.session_scope {
                ChannelSessionScope::User => Some(message.author.id.clone()),
                ChannelSessionScope::Thread | ChannelSessionScope::Room => None,
            },
        };

        let mut metadata = serde_json::Map::new();
        metadata.insert(
            "discord_message_id".to_string(),
            serde_json::Value::String(message.id.clone()),
        );
        if let Some(guild_id) = message.guild_id {
            metadata.insert(
                "discord_guild_id".to_string(),
                serde_json::Value::String(guild_id),
            );
        }
        metadata.insert(
            "channel_runtime_id".to_string(),
            serde_json::Value::String(self.channel_runtime_id.clone()),
        );
        let text = bound_inbound_text(
            message.content,
            &mut metadata,
            self.config.max_inbound_text_chars,
        );

        Some(InboundEvent {
            message: ChannelMessageRef {
                conversation: conversation.clone(),
                message_id: message.id.clone(),
            },
            conversation,
            user: ChannelUser {
                id: message.author.id,
                display_name: message.author.global_name,
                username: Some(message.author.username),
            },
            session_scope: self.config.session_scope,
            text,
            attachments,
            metadata,
        })
    }

    fn track_seen_message(&mut self, message_id: &str) -> bool {
        if self.seen_message_set.contains(message_id) {
            return false;
        }
        self.seen_message_set.insert(message_id.to_string());
        self.seen_message_ids.push_back(message_id.to_string());
        while self.seen_message_ids.len() > SEEN_MESSAGE_IDS_LIMIT {
            if let Some(old) = self.seen_message_ids.pop_front() {
                self.seen_message_set.remove(&old);
            }
        }
        true
    }
}
