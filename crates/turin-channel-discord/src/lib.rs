use anyhow::{Context, Result};
use async_trait::async_trait;
use std::collections::{HashSet, VecDeque};
#[cfg(test)]
use std::time::Duration;
use tokio::sync::watch;
use tokio::time::sleep;
use turin_channel_core::{
    ChannelCapabilities, ChannelConversationKey, ChannelKind, ChannelUser, InboundEvent,
    OutboundMessage,
};
#[cfg(test)]
use turin_channel_core::{ChannelSessionScope, DEFAULT_MAX_INBOUND_TEXT_CHARS, MessageBlock};
use turin_channel_runner::ChannelDriver;

mod api;
mod gateway;
mod inbound;
mod manifest;
mod render;
mod settings;
#[cfg(test)]
use api::{DiscordAuthor, DiscordMessage};
use api::{is_newer_snowflake, parse_snowflake};
use gateway::GatewayConnection;
pub use manifest::{adapter_manifest, poll_auth_flow, start_auth_flow};
#[cfg(test)]
use render::DISCORD_CONTENT_MAX_LEN;
use render::render_outbound_messages;
pub use settings::{DiscordChannelDriverConfig, validate_settings};
#[cfg(test)]
pub(crate) use settings::{parse_settings, parse_transport_mode};

const DEFAULT_BASE_URL: &str = "https://discord.com/api/v10";
const DEFAULT_GATEWAY_URL: &str = "wss://gateway.discord.gg/?v=10&encoding=json";
const DEFAULT_GATEWAY_INTENTS: u64 = (1 << 9) | (1 << 12) | (1 << 15);
const SEEN_MESSAGE_IDS_LIMIT: usize = 1_024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscordTransportMode {
    Gateway,
    Polling,
}

pub struct DiscordChannelDriver {
    channel_runtime_id: String,
    config: DiscordChannelDriverConfig,
    client: reqwest::Client,
    shutdown_rx: watch::Receiver<bool>,
    backlog: VecDeque<InboundEvent>,
    last_seen_message_id: Option<String>,
    initialized: bool,
    gateway: Option<GatewayConnection>,
    last_gateway_seq: Option<u64>,
    gateway_session_id: Option<String>,
    resume_gateway_url: Option<String>,
    seen_message_ids: VecDeque<String>,
    seen_message_set: HashSet<String>,
    reconnect_attempts: u32,
}

impl DiscordChannelDriver {
    pub async fn from_settings(
        channel_runtime_id: impl Into<String>,
        settings: &serde_json::Value,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Result<Self> {
        let config = DiscordChannelDriverConfig::from_settings(settings)?;
        let client = reqwest::Client::builder()
            .user_agent("turin-channel-discord/0.22.0")
            .build()
            .context(
                "[discord_http_client_init_failed] Failed to build Discord adapter HTTP client",
            )?;

        Ok(Self {
            channel_runtime_id: channel_runtime_id.into(),
            config,
            client,
            shutdown_rx,
            backlog: VecDeque::new(),
            last_seen_message_id: None,
            initialized: false,
            gateway: None,
            last_gateway_seq: None,
            gateway_session_id: None,
            resume_gateway_url: None,
            seen_message_ids: VecDeque::new(),
            seen_message_set: HashSet::new(),
            reconnect_attempts: 0,
        })
    }

    async fn next_poll_event(&mut self) -> Result<Option<InboundEvent>> {
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
}

#[async_trait]
impl ChannelDriver for DiscordChannelDriver {
    fn kind(&self) -> ChannelKind {
        ChannelKind::new("discord")
    }

    fn user_matches_selector(&self, selector: &str, user: &ChannelUser) -> bool {
        let selector = selector.trim();
        if selector.is_empty() {
            return false;
        }
        let selector = selector.strip_prefix('@').unwrap_or(selector);
        user.id == selector
            || user
                .username
                .as_ref()
                .is_some_and(|username| username.eq_ignore_ascii_case(selector))
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
        match self.config.transport_mode {
            DiscordTransportMode::Gateway => self.next_gateway_event().await,
            DiscordTransportMode::Polling => self.next_poll_event().await,
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
        let outbound_messages = render_outbound_messages(message);
        for outbound in outbound_messages {
            self.post_message(&channel_id, outbound).await?;
        }
        Ok(())
    }

    async fn shutdown(&mut self) -> Result<()> {
        if let Some(mut connection) = self.gateway.take() {
            let _ = connection.stream.close(None).await;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
