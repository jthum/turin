use anyhow::{Context, Result};
use async_trait::async_trait;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tracing::warn;
use turin_channel_core::{
    ChannelCapabilities, ChannelConversationKey, ChannelKind, ChannelUser, InboundEvent,
    OutboundMessage,
};
#[cfg(test)]
use turin_channel_core::{
    ChannelMessageRef, ChannelSessionScope, DEFAULT_MAX_INBOUND_TEXT_CHARS, MessageBlock,
};
use turin_channel_runner::{ChannelDriver, ChannelProgressUpdate, ChannelStreamMode};

mod api;
mod delivery;
mod inbound;
mod manifest;
mod media;
mod outbound;
mod settings;
use api::TelegramApiError;
use delivery::TelegramProgressState;
#[cfg(test)]
use inbound::{
    TelegramChat, TelegramMessage, TelegramMessageEntity, effective_telegram_session_scope,
};
use inbound::{TelegramUpdate, TelegramUser};
pub use manifest::{adapter_manifest, poll_auth_flow, start_auth_flow};
#[cfg(test)]
use outbound::TELEGRAM_MESSAGE_MAX_LEN;
use outbound::default_media_dir_for_runtime;
#[cfg(test)]
use outbound::{render_stream_preview, telegram_batches_from_message};
#[cfg(test)]
pub(crate) use settings::DEFAULT_BASE_URL;
pub use settings::{TelegramChannelDriverConfig, validate_settings};

const MAX_STARTUP_SKIP_BATCHES: usize = 32;
const MAX_API_REQUEST_ATTEMPTS: u32 = 5;
pub struct TelegramChannelDriver {
    channel_runtime_id: String,
    config: TelegramChannelDriverConfig,
    media_dir: PathBuf,
    client: reqwest::Client,
    shutdown_rx: watch::Receiver<bool>,
    backlog: VecDeque<InboundEvent>,
    next_update_offset: Option<i64>,
    initialized: bool,
    consecutive_poll_failures: u32,
    progress_states: HashMap<String, TelegramProgressState>,
    last_chat_action_at: HashMap<String, Instant>,
    next_draft_id: i64,
    bot_identity: Option<TelegramBotIdentity>,
}

impl TelegramChannelDriver {
    pub async fn from_settings(
        channel_runtime_id: impl Into<String>,
        settings: &serde_json::Value,
        shutdown_rx: watch::Receiver<bool>,
        allow_unconfigured_chats: bool,
    ) -> Result<Self> {
        Self::from_settings_with_media_dir(
            channel_runtime_id,
            settings,
            None,
            shutdown_rx,
            allow_unconfigured_chats,
        )
        .await
    }

    pub async fn from_settings_with_media_dir(
        channel_runtime_id: impl Into<String>,
        settings: &serde_json::Value,
        media_dir: Option<PathBuf>,
        shutdown_rx: watch::Receiver<bool>,
        allow_unconfigured_chats: bool,
    ) -> Result<Self> {
        let config =
            TelegramChannelDriverConfig::from_settings(settings, allow_unconfigured_chats)?;
        Self::from_config_with_media_dir(channel_runtime_id, config, media_dir, shutdown_rx)
    }

    pub fn from_config(
        channel_runtime_id: impl Into<String>,
        config: TelegramChannelDriverConfig,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Result<Self> {
        Self::from_config_with_media_dir(channel_runtime_id, config, None, shutdown_rx)
    }

    pub fn from_config_with_media_dir(
        channel_runtime_id: impl Into<String>,
        config: TelegramChannelDriverConfig,
        media_dir: Option<PathBuf>,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Result<Self> {
        let channel_runtime_id = channel_runtime_id.into();
        let timeout = Duration::from_secs(config.poll_timeout_seconds.saturating_add(10).max(10));
        let client = reqwest::Client::builder()
            .user_agent("turin-channel-telegram/0.24.0")
            .timeout(timeout)
            .build()
            .context(
                "[telegram_http_client_init_failed] Failed to build Telegram adapter HTTP client",
            )?;
        let media_dir =
            media_dir.unwrap_or_else(|| default_media_dir_for_runtime(&channel_runtime_id));

        Ok(Self {
            channel_runtime_id,
            config,
            media_dir,
            client,
            shutdown_rx,
            backlog: VecDeque::new(),
            next_update_offset: None,
            initialized: false,
            consecutive_poll_failures: 0,
            progress_states: HashMap::new(),
            last_chat_action_at: HashMap::new(),
            next_draft_id: 1,
            bot_identity: None,
        })
    }

    async fn skip_pending_updates(&mut self) -> std::result::Result<(), TelegramApiError> {
        for _ in 0..MAX_STARTUP_SKIP_BATCHES {
            let updates = self.fetch_updates(self.next_update_offset, 100, 0).await?;
            if updates.is_empty() {
                break;
            }
            self.advance_offset(&updates);
            if updates.len() < 100 {
                break;
            }
        }
        Ok(())
    }

    async fn poll_once(&mut self) -> std::result::Result<bool, TelegramApiError> {
        let updates = self
            .fetch_updates(
                self.next_update_offset,
                self.config.max_updates_per_poll,
                self.config.poll_timeout_seconds,
            )
            .await?;
        if updates.is_empty() {
            return Ok(false);
        }

        self.advance_offset(&updates);
        for update in updates {
            let update_id = update.update_id;
            let Some(message) = update.message.or(update.channel_post) else {
                continue;
            };
            if let Some(mut event) = self.normalize_message(update_id, message.clone()) {
                match self.collect_inbound_attachments(&message).await {
                    Ok(attachments) => {
                        event.attachments = attachments;
                    }
                    Err(error) => {
                        warn!(
                            channel_runtime_id = %self.channel_runtime_id,
                            update_id,
                            message_id = message.message_id,
                            error = %error,
                            "Telegram attachment collection failed; continuing without attachments"
                        );
                    }
                }
                if event.text.trim().is_empty() && event.attachments.is_empty() {
                    continue;
                }
                self.backlog.push_back(event);
            }
        }

        Ok(!self.backlog.is_empty())
    }

    async fn fetch_updates(
        &self,
        offset: Option<i64>,
        limit: u8,
        timeout_seconds: u64,
    ) -> std::result::Result<Vec<TelegramUpdate>, TelegramApiError> {
        let payload = serde_json::json!({
            "offset": offset,
            "limit": limit,
            "timeout": timeout_seconds,
            "allowed_updates": ["message", "channel_post"]
        });
        self.request_with_retry("getUpdates", &payload).await
    }
}

#[async_trait]
impl ChannelDriver for TelegramChannelDriver {
    fn kind(&self) -> ChannelKind {
        ChannelKind::new("telegram")
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
        loop {
            if let Some(event) = self.backlog.pop_front() {
                return Ok(Some(event));
            }
            if *self.shutdown_rx.borrow() {
                return Ok(None);
            }

            if !self.initialized {
                match self.ensure_bot_identity().await {
                    Ok(()) => {
                        self.consecutive_poll_failures = 0;
                    }
                    Err(error) if error.retriable => {
                        if self.handle_transient_poll_error("getMe", error).await {
                            return Ok(None);
                        }
                        continue;
                    }
                    Err(error) => return Err(error.into_anyhow()),
                }
                if self.config.start_from_latest {
                    match self.skip_pending_updates().await {
                        Ok(()) => {
                            self.consecutive_poll_failures = 0;
                        }
                        Err(error) if error.retriable => {
                            if self
                                .handle_transient_poll_error("startup skip", error)
                                .await
                            {
                                return Ok(None);
                            }
                            continue;
                        }
                        Err(error) => return Err(error.into_anyhow()),
                    }
                }
                self.initialized = true;
                continue;
            }

            let mut shutdown_rx = self.shutdown_rx.clone();
            let got_backlog = tokio::select! {
                changed = shutdown_rx.changed() => {
                    if changed.is_ok() && *shutdown_rx.borrow() {
                        return Ok(None);
                    }
                    Ok(false)
                }
                result = self.poll_once() => result,
            };

            let got_backlog = match got_backlog {
                Ok(got_backlog) => {
                    self.consecutive_poll_failures = 0;
                    got_backlog
                }
                Err(error) if error.retriable => {
                    if self.handle_transient_poll_error("poll", error).await {
                        return Ok(None);
                    }
                    continue;
                }
                Err(error) => return Err(error.into_anyhow()),
            };

            if let Some(event) = self.backlog.pop_front() {
                return Ok(Some(event));
            }

            if !got_backlog && self.sleep_or_shutdown(self.config.poll_interval).await {
                return Ok(None);
            }
        }
    }

    async fn send(
        &mut self,
        conversation: &ChannelConversationKey,
        message: OutboundMessage,
    ) -> Result<()> {
        self.send_final_message(conversation, &message).await
    }

    fn enrich_outbound_for_event(
        &self,
        event: &InboundEvent,
        mut outbound: OutboundMessage,
    ) -> OutboundMessage {
        if !outbound
            .metadata
            .contains_key("telegram_reply_to_message_id")
            && let Some(message_id) = event.metadata.get("telegram_message_id")
        {
            outbound.metadata.insert(
                "telegram_reply_to_message_id".to_string(),
                message_id.clone(),
            );
        }
        outbound
    }

    fn stream_mode(&self) -> ChannelStreamMode {
        self.config.stream_mode
    }

    fn stream_thinking(&self) -> bool {
        self.config.stream_mode.streams_text() && self.config.stream_thinking
    }

    fn persist_thinking(&self) -> bool {
        self.config.persist_thinking
    }

    async fn send_progress(
        &mut self,
        event: &InboundEvent,
        update: ChannelProgressUpdate,
    ) -> Result<()> {
        match update {
            ChannelProgressUpdate::Typing => self.send_chat_action(event).await,
            ChannelProgressUpdate::StreamingPreview { text, thinking } => {
                self.send_stream_preview(event, &text, thinking.as_deref())
                    .await
            }
        }
    }

    async fn shutdown(&mut self) -> Result<()> {
        let _ = &self.channel_runtime_id;
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct TelegramBotIdentity {
    id: i64,
    username: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TelegramRespondMode {
    All,
    Mentions,
    Replies,
    MentionsOrReplies,
}

impl TelegramRespondMode {
    fn requires_bot_identity(self) -> bool {
        !matches!(self, Self::All)
    }
}

#[cfg(test)]
mod tests;
