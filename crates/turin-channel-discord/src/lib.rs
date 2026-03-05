use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use serde::Deserialize;
use std::collections::VecDeque;
use std::time::Duration;
use tokio::sync::watch;
use tokio::time::sleep;
use turin_channel_core::{
    ChannelAttachment, ChannelCapabilities, ChannelConversationKey, ChannelKind, ChannelMessageRef,
    ChannelUser, InboundEvent, MessageBlock, OutboundMessage,
};
use turin_channel_runner::ChannelDriver;

const DEFAULT_BASE_URL: &str = "https://discord.com/api/v10";

#[derive(Debug, Clone)]
pub struct DiscordChannelDriverConfig {
    pub base_url: String,
    pub workspace_id: String,
    pub room_id: Option<String>,
    pub channel_id: String,
    pub token: String,
    pub poll_interval: Duration,
    pub max_messages_per_poll: u16,
    pub start_from_latest: bool,
    pub ignore_bot_messages: bool,
}

impl DiscordChannelDriverConfig {
    pub fn from_settings(settings: &serde_json::Value) -> Result<Self> {
        let settings = settings
            .as_object()
            .ok_or_else(|| anyhow!("Discord channel settings must be a JSON object"))?;

        let token_env = settings
            .get("token_env")
            .and_then(|v| v.as_str())
            .filter(|v| !v.trim().is_empty())
            .ok_or_else(|| anyhow!("Discord channel setting 'token_env' is required"))?;
        let token = std::env::var(token_env).with_context(|| {
            format!(
                "Discord bot token env var '{}' is not set for channel adapter",
                token_env
            )
        })?;

        let channel_id = settings
            .get("channel_id")
            .and_then(|v| v.as_str())
            .filter(|v| !v.trim().is_empty())
            .ok_or_else(|| anyhow!("Discord channel setting 'channel_id' is required"))?
            .to_string();

        let poll_interval_ms = settings
            .get("poll_interval_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(1_000)
            .max(100);
        let max_messages_per_poll = settings
            .get("max_messages_per_poll")
            .and_then(|v| v.as_u64())
            .unwrap_or(25)
            .clamp(1, 100) as u16;

        Ok(Self {
            base_url: settings
                .get("base_url")
                .and_then(|v| v.as_str())
                .unwrap_or(DEFAULT_BASE_URL)
                .trim_end_matches('/')
                .to_string(),
            workspace_id: settings
                .get("workspace_id")
                .and_then(|v| v.as_str())
                .unwrap_or("discord")
                .to_string(),
            room_id: settings
                .get("room_id")
                .and_then(|v| v.as_str())
                .map(std::string::ToString::to_string),
            channel_id,
            token,
            poll_interval: Duration::from_millis(poll_interval_ms),
            max_messages_per_poll,
            start_from_latest: settings
                .get("start_from_latest")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            ignore_bot_messages: settings
                .get("ignore_bot_messages")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
        })
    }
}

pub struct DiscordChannelDriver {
    channel_runtime_id: String,
    config: DiscordChannelDriverConfig,
    client: reqwest::Client,
    shutdown_rx: watch::Receiver<bool>,
    backlog: VecDeque<InboundEvent>,
    last_seen_message_id: Option<String>,
    initialized: bool,
}

impl DiscordChannelDriver {
    pub async fn from_settings(
        channel_runtime_id: impl Into<String>,
        settings: &serde_json::Value,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Result<Self> {
        let config = DiscordChannelDriverConfig::from_settings(settings)?;
        let client = reqwest::Client::builder()
            .user_agent("turin-channel-discord/0.21.0")
            .build()
            .context("Failed to build Discord adapter HTTP client")?;

        Ok(Self {
            channel_runtime_id: channel_runtime_id.into(),
            config,
            client,
            shutdown_rx,
            backlog: VecDeque::new(),
            last_seen_message_id: None,
            initialized: false,
        })
    }

    async fn poll_once(&mut self) -> Result<()> {
        if !self.initialized && self.config.start_from_latest {
            let latest = self.fetch_latest_message_id().await?;
            self.last_seen_message_id = latest;
            self.initialized = true;
            return Ok(());
        }
        self.initialized = true;

        let mut messages = self
            .fetch_messages(
                self.last_seen_message_id.as_deref(),
                self.config.max_messages_per_poll,
            )
            .await?;
        if messages.is_empty() {
            return Ok(());
        }

        messages.sort_by_key(|message| parse_snowflake(&message.id).unwrap_or_default());
        let mut newest_id = self.last_seen_message_id.clone();
        for message in messages {
            if newest_id
                .as_ref()
                .is_none_or(|current| is_newer_snowflake(&message.id, current))
            {
                newest_id = Some(message.id.clone());
            }
            if let Some(event) = self.normalize_message(message) {
                self.backlog.push_back(event);
            }
        }
        self.last_seen_message_id = newest_id;
        Ok(())
    }

    fn normalize_message(&self, message: DiscordMessage) -> Option<InboundEvent> {
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
            channel: ChannelKind::Discord,
            workspace_id: self.config.workspace_id.clone(),
            room_id,
            thread_id: message.channel_id.clone(),
            user_id: Some(message.author.id.clone()),
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
            text: message.content,
            attachments,
            metadata,
        })
    }

    async fn fetch_latest_message_id(&self) -> Result<Option<String>> {
        let mut messages = self.fetch_messages(None, 1).await?;
        Ok(messages.pop().map(|msg| msg.id))
    }

    async fn fetch_messages(&self, after: Option<&str>, limit: u16) -> Result<Vec<DiscordMessage>> {
        let url = format!(
            "{}/channels/{}/messages",
            self.config.base_url, self.config.channel_id
        );
        let mut params = vec![("limit".to_string(), limit.to_string())];
        if let Some(after) = after {
            params.push(("after".to_string(), after.to_string()));
        }

        let response = self
            .request_with_retry(|| {
                self.client
                    .get(&url)
                    .header("Authorization", format!("Bot {}", self.config.token))
                    .query(&params)
                    .build()
                    .context("Failed to build Discord messages request")
            })
            .await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!(
                "Discord messages request failed with {}: {}",
                status.as_u16(),
                body
            );
        }

        response
            .json::<Vec<DiscordMessage>>()
            .await
            .context("Failed to decode Discord messages response")
    }

    async fn post_message(&self, channel_id: &str, content: String) -> Result<()> {
        let url = format!("{}/channels/{}/messages", self.config.base_url, channel_id);
        let payload = serde_json::json!({ "content": content });

        let response = self
            .request_with_retry(|| {
                self.client
                    .post(&url)
                    .header("Authorization", format!("Bot {}", self.config.token))
                    .json(&payload)
                    .build()
                    .context("Failed to build Discord send request")
            })
            .await?;
        let status = response.status();
        if status == reqwest::StatusCode::OK || status == reqwest::StatusCode::CREATED {
            return Ok(());
        }

        let body = response.text().await.unwrap_or_default();
        anyhow::bail!(
            "Discord send request failed with {}: {}",
            status.as_u16(),
            body
        );
    }

    async fn request_with_retry<F>(&self, request_builder: F) -> Result<reqwest::Response>
    where
        F: Fn() -> Result<reqwest::Request>,
    {
        let mut attempts = 0;
        loop {
            attempts += 1;
            let request = request_builder()?;
            let response = self
                .client
                .execute(request)
                .await
                .context("Discord request failed")?;

            if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS && attempts < 5 {
                let body = response
                    .json::<DiscordRateLimit>()
                    .await
                    .unwrap_or(DiscordRateLimit { retry_after: 1.0 });
                sleep(Duration::from_secs_f64(body.retry_after.max(0.1))).await;
                continue;
            }
            return Ok(response);
        }
    }
}

#[async_trait]
impl ChannelDriver for DiscordChannelDriver {
    fn kind(&self) -> ChannelKind {
        ChannelKind::Discord
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities {
            rich_formatting: true,
            threads: true,
            attachments: true,
            ephemeral_messages: false,
        }
    }

    async fn next_event(&mut self) -> Result<Option<InboundEvent>> {
        loop {
            if let Some(event) = self.backlog.pop_front() {
                return Ok(Some(event));
            }
            if *self.shutdown_rx.borrow() {
                return Ok(None);
            }

            self.poll_once().await?;
            if let Some(event) = self.backlog.pop_front() {
                return Ok(Some(event));
            }

            tokio::select! {
                changed = self.shutdown_rx.changed() => {
                    if changed.is_ok() && *self.shutdown_rx.borrow() {
                        return Ok(None);
                    }
                }
                _ = sleep(self.config.poll_interval) => {}
            }
        }
    }

    async fn send(
        &mut self,
        conversation: &ChannelConversationKey,
        message: OutboundMessage,
    ) -> Result<()> {
        let channel_id = if conversation.thread_id.trim().is_empty() {
            self.config.channel_id.clone()
        } else {
            conversation.thread_id.clone()
        };
        let content = render_outbound_message(&message);
        self.post_message(&channel_id, content).await
    }

    async fn shutdown(&mut self) -> Result<()> {
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
struct DiscordMessage {
    id: String,
    channel_id: String,
    #[serde(default)]
    guild_id: Option<String>,
    #[serde(default)]
    content: String,
    author: DiscordAuthor,
    #[serde(default)]
    attachments: Vec<DiscordAttachment>,
}

#[derive(Debug, Clone, Deserialize)]
struct DiscordAuthor {
    id: String,
    username: String,
    #[serde(default)]
    global_name: Option<String>,
    #[serde(default)]
    bot: Option<bool>,
}

#[derive(Debug, Clone, Deserialize)]
struct DiscordAttachment {
    filename: String,
    #[serde(default)]
    content_type: Option<String>,
    url: String,
}

#[derive(Debug, Clone, Deserialize)]
struct DiscordRateLimit {
    retry_after: f64,
}

fn parse_snowflake(value: &str) -> Option<u64> {
    value.parse::<u64>().ok()
}

fn is_newer_snowflake(candidate: &str, current: &str) -> bool {
    match (parse_snowflake(candidate), parse_snowflake(current)) {
        (Some(candidate), Some(current)) => candidate > current,
        _ => candidate > current,
    }
}

fn render_outbound_message(message: &OutboundMessage) -> String {
    let mut chunks = Vec::new();
    for block in &message.blocks {
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

    if chunks.is_empty() && !message.attachments.is_empty() {
        chunks.push(
            message
                .attachments
                .iter()
                .map(|attachment| attachment.name.clone())
                .collect::<Vec<_>>()
                .join(", "),
        );
    }

    if chunks.is_empty() {
        "(no output)".to_string()
    } else {
        chunks.join("\n\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_outbound_preserves_code_blocks() {
        let output = render_outbound_message(&OutboundMessage {
            blocks: vec![
                MessageBlock::Text {
                    text: "summary".to_string(),
                },
                MessageBlock::CodeBlock {
                    language: Some("rust".to_string()),
                    code: "fn main() {}".to_string(),
                },
            ],
            ..OutboundMessage::default()
        });
        assert!(output.contains("summary"));
        assert!(output.contains("```rust"));
    }

    #[test]
    fn normalize_ignores_bot_messages() {
        let config = DiscordChannelDriverConfig {
            base_url: DEFAULT_BASE_URL.to_string(),
            workspace_id: "discord".to_string(),
            room_id: None,
            channel_id: "123".to_string(),
            token: "token".to_string(),
            poll_interval: Duration::from_millis(250),
            max_messages_per_poll: 10,
            start_from_latest: true,
            ignore_bot_messages: true,
        };
        let (_tx, rx) = watch::channel(false);
        let driver = DiscordChannelDriver {
            channel_runtime_id: "discord-runtime".to_string(),
            config,
            client: reqwest::Client::new(),
            shutdown_rx: rx,
            backlog: VecDeque::new(),
            last_seen_message_id: None,
            initialized: false,
        };
        let message = DiscordMessage {
            id: "1".to_string(),
            channel_id: "chan".to_string(),
            guild_id: Some("guild".to_string()),
            content: "hello".to_string(),
            author: DiscordAuthor {
                id: "bot".to_string(),
                username: "bot".to_string(),
                global_name: None,
                bot: Some(true),
            },
            attachments: Vec::new(),
        };
        assert!(driver.normalize_message(message).is_none());
    }
}
