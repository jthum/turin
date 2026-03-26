use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime};

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChannelKind {
    Discord,
    Slack,
    Telegram,
    Matrix,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChannelConversationKey {
    pub channel: ChannelKind,
    pub workspace_id: String,
    pub room_id: Option<String>,
    pub thread_id: String,
    pub user_id: Option<String>,
}

impl ChannelConversationKey {
    pub fn deterministic_slot_id(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(serde_json::to_vec(self).expect("conversation key serializes"));
        let digest = hasher.finalize();
        format!("chan-{}", &hex::encode(digest)[..24])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ChannelSessionScope {
    #[default]
    User,
    Thread,
    Room,
}

impl ChannelSessionScope {
    pub fn is_shared(self) -> bool {
        !matches!(self, Self::User)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelMessageRef {
    pub conversation: ChannelConversationKey,
    pub message_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelUser {
    pub id: String,
    pub display_name: Option<String>,
    pub username: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelAttachment {
    pub name: String,
    pub content_type: Option<String>,
    pub url: Option<String>,
    pub local_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageBlock {
    Text {
        text: String,
    },
    CodeBlock {
        language: Option<String>,
        code: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct OutboundMessage {
    #[serde(default)]
    pub blocks: Vec<MessageBlock>,
    #[serde(default)]
    pub attachments: Vec<ChannelAttachment>,
    #[serde(default)]
    pub embeds: Vec<serde_json::Value>,
    #[serde(default)]
    pub components: Vec<serde_json::Value>,
    #[serde(default)]
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

impl OutboundMessage {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            blocks: vec![MessageBlock::Text { text: text.into() }],
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InboundEvent {
    pub conversation: ChannelConversationKey,
    pub message: ChannelMessageRef,
    pub user: ChannelUser,
    #[serde(default)]
    pub session_scope: ChannelSessionScope,
    pub text: String,
    #[serde(default)]
    pub attachments: Vec<ChannelAttachment>,
    #[serde(default)]
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

impl InboundEvent {
    pub fn prompt_text(&self) -> String {
        if !self.session_scope.is_shared() {
            return self.text.clone();
        }

        format!("[Message from {}]\n{}", self.user.prompt_label(), self.text)
    }
}

impl ChannelUser {
    pub fn prompt_label(&self) -> String {
        match (self.display_name.as_deref(), self.username.as_deref()) {
            (Some(display_name), Some(username))
                if !display_name.trim().is_empty()
                    && !username.trim().is_empty()
                    && !display_name.eq_ignore_ascii_case(username) =>
            {
                format!("{display_name} (@{username})")
            }
            (Some(display_name), _) if !display_name.trim().is_empty() => display_name.to_string(),
            (_, Some(username)) if !username.trim().is_empty() => format!("@{username}"),
            _ => self.id.clone(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ChannelCapabilities {
    pub rich_formatting: bool,
    pub threads: bool,
    pub attachments: bool,
    pub ephemeral_messages: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversationBinding {
    pub agent_id: String,
    pub slot_id: String,
    pub session_id: String,
    pub updated_at_unix_secs: u64,
}

impl ConversationBinding {
    pub fn new(
        agent_id: impl Into<String>,
        session_id: impl Into<String>,
        key: &ChannelConversationKey,
        now: SystemTime,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            slot_id: key.deterministic_slot_id(),
            session_id: session_id.into(),
            updated_at_unix_secs: unix_secs(now),
        }
    }

    pub fn touch(&mut self, now: SystemTime) {
        self.updated_at_unix_secs = unix_secs(now);
    }

    pub fn is_expired(&self, now: SystemTime, ttl: Duration) -> bool {
        unix_secs(now).saturating_sub(self.updated_at_unix_secs) > ttl.as_secs()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoutingDecision {
    Reuse { slot_id: String, session_id: String },
    StartFresh { slot_id: String },
}

pub fn decide_routing(
    key: &ChannelConversationKey,
    binding: Option<&ConversationBinding>,
    now: SystemTime,
    ttl: Option<Duration>,
    reset_requested: bool,
) -> RoutingDecision {
    let slot_id = key.deterministic_slot_id();
    if reset_requested {
        return RoutingDecision::StartFresh { slot_id };
    }

    match binding {
        Some(binding) => {
            if ttl.is_some_and(|ttl| binding.is_expired(now, ttl)) {
                RoutingDecision::StartFresh { slot_id }
            } else {
                RoutingDecision::Reuse {
                    slot_id,
                    session_id: binding.session_id.clone(),
                }
            }
        }
        None => RoutingDecision::StartFresh { slot_id },
    }
}

fn unix_secs(time: SystemTime) -> u64 {
    time.duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key() -> ChannelConversationKey {
        ChannelConversationKey {
            channel: ChannelKind::Discord,
            workspace_id: "guild-1".into(),
            room_id: Some("room-2".into()),
            thread_id: "thread-3".into(),
            user_id: Some("user-4".into()),
        }
    }

    #[test]
    fn slot_id_is_stable() {
        let key = key();
        assert_eq!(key.deterministic_slot_id(), key.deterministic_slot_id());
    }

    #[test]
    fn reset_forces_fresh_session() {
        let key = key();
        let binding = ConversationBinding::new("writer", "sess-1", &key, SystemTime::UNIX_EPOCH);
        let decision = decide_routing(&key, Some(&binding), SystemTime::UNIX_EPOCH, None, true);
        assert!(matches!(decision, RoutingDecision::StartFresh { .. }));
    }

    #[test]
    fn ttl_expiry_forces_fresh_session() {
        let key = key();
        let binding = ConversationBinding::new("writer", "sess-1", &key, SystemTime::UNIX_EPOCH);
        let decision = decide_routing(
            &key,
            Some(&binding),
            SystemTime::UNIX_EPOCH + Duration::from_secs(120),
            Some(Duration::from_secs(60)),
            false,
        );
        assert!(matches!(decision, RoutingDecision::StartFresh { .. }));
    }

    #[test]
    fn structured_outbound_message_keeps_code_blocks() {
        let message = OutboundMessage {
            blocks: vec![
                MessageBlock::Text {
                    text: "Here is code".into(),
                },
                MessageBlock::CodeBlock {
                    language: Some("rust".into()),
                    code: "fn main() {}".into(),
                },
            ],
            ..OutboundMessage::default()
        };
        let value = serde_json::to_value(&message).expect("serialize outbound message");
        assert_eq!(value["blocks"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn shared_scope_prompt_includes_sender_label() {
        let key = key();
        let event = InboundEvent {
            conversation: key.clone(),
            message: ChannelMessageRef {
                conversation: key,
                message_id: "m-1".into(),
            },
            user: ChannelUser {
                id: "user-4".into(),
                display_name: Some("Jay".into()),
                username: Some("jthum".into()),
            },
            session_scope: ChannelSessionScope::Thread,
            text: "hello".into(),
            attachments: vec![],
            metadata: serde_json::Map::new(),
        };

        assert_eq!(event.prompt_text(), "[Message from Jay (@jthum)]\nhello");
    }
}
