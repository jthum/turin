use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use std::collections::VecDeque;
use std::time::Duration;
use tokio::sync::watch;
use tokio::time::sleep;
use turin_channel_core::{
    ChannelAttachment, ChannelCapabilities, ChannelConversationKey, ChannelKind, ChannelMessageRef,
    ChannelUser, InboundEvent, MessageBlock, OutboundMessage,
};
use turin_channel_runner::ChannelDriver;

const DEFAULT_BASE_URL: &str = "https://api.telegram.org";
const TELEGRAM_MESSAGE_MAX_LEN: usize = 4_096;
const MAX_STARTUP_SKIP_BATCHES: usize = 32;

#[derive(Debug, Clone)]
pub struct TelegramChannelDriverConfig {
    pub base_url: String,
    pub workspace_id: String,
    pub chat_id: String,
    pub token: String,
    pub poll_timeout_secs: u64,
    pub poll_interval: Duration,
    pub max_updates_per_poll: u8,
    pub start_from_latest: bool,
    pub ignore_bot_messages: bool,
}

impl TelegramChannelDriverConfig {
    pub fn from_settings(settings: &serde_json::Value) -> Result<Self> {
        let settings = settings
            .as_object()
            .ok_or_else(|| anyhow!("Telegram channel settings must be a JSON object"))?;

        let token_env = read_required_string(
            settings,
            "token_env",
            "[telegram_config_missing_token_env] Telegram channel setting 'token_env' is required",
            "[telegram_config_invalid_token_env] Telegram channel setting 'token_env' must not be empty",
        )?;
        let token = std::env::var(token_env).map_err(|_| {
            anyhow!(
                "[telegram_auth_missing_token] Telegram bot token env var '{}' is not set for channel adapter",
                token_env
            )
        })?;

        let chat_id = read_chat_id(settings.get("chat_id")).map_err(|err| {
            anyhow!(
                "[telegram_config_missing_chat_id] Telegram channel setting 'chat_id' is required: {}",
                err
            )
        })?;

        let poll_timeout_secs = match settings.get("poll_timeout_secs") {
            None => 30,
            Some(value) => {
                let timeout = value.as_u64().ok_or_else(|| {
                    anyhow!(
                        "[telegram_config_invalid_poll_timeout] Telegram channel setting 'poll_timeout_secs' must be a non-negative integer"
                    )
                })?;
                if timeout > 50 {
                    anyhow::bail!(
                        "[telegram_config_invalid_poll_timeout] Telegram channel setting 'poll_timeout_secs' must be <= 50"
                    );
                }
                timeout
            }
        };

        let poll_interval_ms = match settings.get("poll_interval_ms") {
            None => 250,
            Some(value) => {
                let interval = value.as_u64().ok_or_else(|| {
                    anyhow!(
                        "[telegram_config_invalid_poll_interval] Telegram channel setting 'poll_interval_ms' must be a positive integer"
                    )
                })?;
                if interval < 25 {
                    anyhow::bail!(
                        "[telegram_config_invalid_poll_interval] Telegram channel setting 'poll_interval_ms' must be >= 25"
                    );
                }
                interval
            }
        };

        let max_updates_per_poll = match settings.get("max_updates_per_poll") {
            None => 25,
            Some(value) => {
                let max = value.as_u64().ok_or_else(|| {
                    anyhow!(
                        "[telegram_config_invalid_max_updates] Telegram channel setting 'max_updates_per_poll' must be a positive integer"
                    )
                })?;
                if !(1..=100).contains(&max) {
                    anyhow::bail!(
                        "[telegram_config_invalid_max_updates] Telegram channel setting 'max_updates_per_poll' must be in 1..=100"
                    );
                }
                max as u8
            }
        };

        Ok(Self {
            base_url: settings
                .get("base_url")
                .map(|value| {
                    value.as_str().ok_or_else(|| {
                        anyhow!(
                            "[telegram_config_invalid_base_url] Telegram channel setting 'base_url' must be a string"
                        )
                    })
                })
                .transpose()?
                .unwrap_or(DEFAULT_BASE_URL)
                .trim_end_matches('/')
                .to_string(),
            workspace_id: settings
                .get("workspace_id")
                .map(|value| {
                    let text = value.as_str().ok_or_else(|| {
                        anyhow!(
                            "[telegram_config_invalid_workspace_id] Telegram channel setting 'workspace_id' must be a string"
                        )
                    })?;
                    if text.trim().is_empty() {
                        anyhow::bail!(
                            "[telegram_config_invalid_workspace_id] Telegram channel setting 'workspace_id' must not be empty"
                        );
                    }
                    Ok::<String, anyhow::Error>(text.to_string())
                })
                .transpose()?
                .unwrap_or_else(|| "telegram".to_string()),
            chat_id,
            token,
            poll_timeout_secs,
            poll_interval: Duration::from_millis(poll_interval_ms),
            max_updates_per_poll,
            start_from_latest: settings
                .get("start_from_latest")
                .map(|value| {
                    value.as_bool().ok_or_else(|| {
                        anyhow!(
                            "[telegram_config_invalid_start_from_latest] Telegram channel setting 'start_from_latest' must be a boolean"
                        )
                    })
                })
                .transpose()?
                .unwrap_or(true),
            ignore_bot_messages: settings
                .get("ignore_bot_messages")
                .map(|value| {
                    value.as_bool().ok_or_else(|| {
                        anyhow!(
                            "[telegram_config_invalid_ignore_bot_messages] Telegram channel setting 'ignore_bot_messages' must be a boolean"
                        )
                    })
                })
                .transpose()?
                .unwrap_or(true),
        })
    }
}

pub struct TelegramChannelDriver {
    channel_runtime_id: String,
    config: TelegramChannelDriverConfig,
    client: reqwest::Client,
    shutdown_rx: watch::Receiver<bool>,
    backlog: VecDeque<InboundEvent>,
    next_update_offset: Option<i64>,
    initialized: bool,
}

impl TelegramChannelDriver {
    pub async fn from_settings(
        channel_runtime_id: impl Into<String>,
        settings: &serde_json::Value,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Result<Self> {
        let config = TelegramChannelDriverConfig::from_settings(settings)?;
        Self::from_config(channel_runtime_id, config, shutdown_rx)
    }

    pub fn from_config(
        channel_runtime_id: impl Into<String>,
        config: TelegramChannelDriverConfig,
        shutdown_rx: watch::Receiver<bool>,
    ) -> Result<Self> {
        let timeout = Duration::from_secs(config.poll_timeout_secs.saturating_add(10).max(10));
        let client = reqwest::Client::builder()
            .user_agent("turin-channel-telegram/0.24.0")
            .timeout(timeout)
            .build()
            .context(
                "[telegram_http_client_init_failed] Failed to build Telegram adapter HTTP client",
            )?;

        Ok(Self {
            channel_runtime_id: channel_runtime_id.into(),
            config,
            client,
            shutdown_rx,
            backlog: VecDeque::new(),
            next_update_offset: None,
            initialized: false,
        })
    }

    async fn skip_pending_updates(&mut self) -> Result<()> {
        for _ in 0..MAX_STARTUP_SKIP_BATCHES {
            let updates = self
                .fetch_updates(self.next_update_offset, 100, 0)
                .await
                .context(
                    "[telegram_startup_skip_failed] Failed to skip pending Telegram updates",
                )?;
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

    async fn poll_once(&mut self) -> Result<bool> {
        let updates = self
            .fetch_updates(
                self.next_update_offset,
                self.config.max_updates_per_poll,
                self.config.poll_timeout_secs,
            )
            .await?;
        if updates.is_empty() {
            return Ok(false);
        }

        self.advance_offset(&updates);
        for update in updates {
            if let Some(event) = self.normalize_update(update) {
                self.backlog.push_back(event);
            }
        }

        Ok(!self.backlog.is_empty())
    }

    async fn fetch_updates(
        &self,
        offset: Option<i64>,
        limit: u8,
        timeout_secs: u64,
    ) -> Result<Vec<TelegramUpdate>> {
        let payload = serde_json::json!({
            "offset": offset,
            "limit": limit,
            "timeout": timeout_secs,
            "allowed_updates": ["message", "channel_post"]
        });
        self.api_request("getUpdates", &payload).await
    }

    async fn send_batches(
        &self,
        conversation: &ChannelConversationKey,
        message: &OutboundMessage,
    ) -> Result<()> {
        let chat_id = conversation
            .room_id
            .as_ref()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or(&self.config.chat_id)
            .clone();
        let message_thread_id = resolve_message_thread_id(conversation)?;

        for payload in telegram_batches_from_message(&chat_id, message_thread_id, message) {
            let _: TelegramSentMessage = self.api_request("sendMessage", &payload).await?;
        }
        Ok(())
    }

    fn advance_offset(&mut self, updates: &[TelegramUpdate]) {
        if let Some(next) = updates.iter().map(|update| update.update_id).max() {
            self.next_update_offset = Some(next.saturating_add(1));
        }
    }

    fn normalize_update(&self, update: TelegramUpdate) -> Option<InboundEvent> {
        let message = update.message.or(update.channel_post)?;
        if message.chat.id.to_string() != self.config.chat_id {
            return None;
        }

        if self.config.ignore_bot_messages
            && message.from.as_ref().and_then(|user| user.is_bot) == Some(true)
        {
            return None;
        }

        let text = message
            .text
            .as_ref()
            .or(message.caption.as_ref())
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())?;

        let user = message.channel_user()?;
        let thread_id = message
            .message_thread_id
            .map(|value| value.to_string())
            .unwrap_or_else(|| self.config.chat_id.clone());

        let mut metadata = serde_json::Map::new();
        metadata.insert(
            "telegram_update_id".to_string(),
            serde_json::json!(update.update_id),
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

        let conversation = ChannelConversationKey {
            channel: ChannelKind::Telegram,
            workspace_id: self.config.workspace_id.clone(),
            room_id: Some(self.config.chat_id.clone()),
            thread_id,
            user_id: Some(user.id.clone()),
        };

        Some(InboundEvent {
            message: ChannelMessageRef {
                conversation: conversation.clone(),
                message_id: message.message_id.to_string(),
            },
            conversation,
            user,
            text,
            attachments: Vec::new(),
            metadata,
        })
    }

    async fn api_request<T: DeserializeOwned>(
        &self,
        method: &str,
        payload: &serde_json::Value,
    ) -> Result<T> {
        let url = format!(
            "{}/bot{}/{}",
            self.config.base_url, self.config.token, method
        );
        let response = self
            .client
            .post(&url)
            .json(payload)
            .send()
            .await
            .with_context(|| {
                format!(
                    "[telegram_http_request_failed] Telegram {} request failed",
                    method
                )
            })?;

        let status = response.status();
        let body = response.text().await.with_context(|| {
            format!(
                "[telegram_http_decode_failed] Failed to read Telegram {} response body",
                method
            )
        })?;

        let envelope: TelegramApiEnvelope<T> = serde_json::from_str(&body).with_context(|| {
            format!(
                "[telegram_http_decode_failed] Failed to decode Telegram {} response: {}",
                method, body
            )
        })?;

        if !status.is_success() || !envelope.ok {
            let description = envelope.description.clone().unwrap_or_else(|| body.clone());
            let error_code = envelope.error_code.unwrap_or(status.as_u16() as i64);
            anyhow::bail!(
                "[{}] Telegram {} request failed with {}: {}",
                classify_api_error(method, status.as_u16(), &description),
                method,
                error_code,
                description
            );
        }

        envelope
            .result
            .context(format!("Telegram {} response missing result", method))
    }

    async fn sleep_or_shutdown(&self, duration: Duration) -> bool {
        let mut shutdown_rx = self.shutdown_rx.clone();
        tokio::select! {
            changed = shutdown_rx.changed() => changed.is_ok() && *shutdown_rx.borrow(),
            _ = sleep(duration) => false,
        }
    }
}

#[async_trait]
impl ChannelDriver for TelegramChannelDriver {
    fn kind(&self) -> ChannelKind {
        ChannelKind::Telegram
    }

    fn capabilities(&self) -> ChannelCapabilities {
        ChannelCapabilities {
            rich_formatting: false,
            threads: true,
            attachments: false,
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
                if self.config.start_from_latest {
                    self.skip_pending_updates().await?;
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
                    false
                }
                result = self.poll_once() => result?,
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
        self.send_batches(conversation, &message).await
    }

    async fn shutdown(&mut self) -> Result<()> {
        let _ = &self.channel_runtime_id;
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
struct TelegramApiEnvelope<T> {
    ok: bool,
    result: Option<T>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    error_code: Option<i64>,
}

#[derive(Debug, Clone, Deserialize)]
struct TelegramUpdate {
    update_id: i64,
    #[serde(default)]
    message: Option<TelegramMessage>,
    #[serde(default)]
    channel_post: Option<TelegramMessage>,
}

#[derive(Debug, Clone, Deserialize)]
struct TelegramMessage {
    message_id: i64,
    chat: TelegramChat,
    #[serde(default)]
    from: Option<TelegramUser>,
    #[serde(default)]
    sender_chat: Option<TelegramChat>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    caption: Option<String>,
    #[serde(default)]
    message_thread_id: Option<i64>,
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
}

#[derive(Debug, Clone, Deserialize)]
struct TelegramUser {
    id: i64,
    #[serde(default)]
    is_bot: Option<bool>,
    #[serde(default)]
    first_name: Option<String>,
    #[serde(default)]
    last_name: Option<String>,
    #[serde(default)]
    username: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct TelegramChat {
    id: i64,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    username: Option<String>,
    #[serde(default)]
    first_name: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct TelegramSentMessage {
    _message_id: i64,
}

fn read_required_string<'a>(
    settings: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
    missing_message: &str,
    empty_message: &str,
) -> Result<&'a str> {
    let value = settings
        .get(key)
        .and_then(|value| value.as_str())
        .ok_or_else(|| anyhow!(missing_message.to_string()))?;
    if value.trim().is_empty() {
        anyhow::bail!(empty_message.to_string());
    }
    Ok(value)
}

fn read_chat_id(value: Option<&serde_json::Value>) -> Result<String> {
    let Some(value) = value else {
        anyhow::bail!("missing value");
    };

    if let Some(id) = value.as_i64() {
        if id == 0 {
            anyhow::bail!("chat_id must not be zero");
        }
        return Ok(id.to_string());
    }
    if let Some(id) = value.as_u64() {
        if id == 0 {
            anyhow::bail!("chat_id must not be zero");
        }
        return Ok(id.to_string());
    }

    let text = value
        .as_str()
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| anyhow!("chat_id must be a non-empty integer or integer string"))?;

    let is_valid = text
        .strip_prefix('-')
        .unwrap_or(text)
        .chars()
        .all(|ch| ch.is_ascii_digit());
    if !is_valid || text == "-" || text == "0" || text == "-0" {
        anyhow::bail!("chat_id must be a non-zero integer or integer string");
    }

    Ok(text.to_string())
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

fn telegram_batches_from_message(
    chat_id: &str,
    message_thread_id: Option<i64>,
    message: &OutboundMessage,
) -> Vec<serde_json::Value> {
    let mut text_chunks = split_for_telegram_message(render_text_blocks(&message.blocks));

    let attachment_lines = render_attachment_lines(&message.attachments);
    if !attachment_lines.is_empty() {
        text_chunks.extend(split_for_telegram_message(attachment_lines));
    }

    if text_chunks.is_empty() {
        text_chunks.push("(no output)".to_string());
    }

    text_chunks
        .into_iter()
        .map(|text| {
            serde_json::json!({
                "chat_id": chat_id,
                "text": text,
                "message_thread_id": message_thread_id,
                "disable_web_page_preview": true
            })
        })
        .collect()
}

fn render_text_blocks(blocks: &[MessageBlock]) -> String {
    let mut chunks = Vec::new();
    for block in blocks {
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
    chunks.join("\n\n")
}

fn render_attachment_lines(attachments: &[ChannelAttachment]) -> String {
    let mut lines = Vec::new();
    for attachment in attachments {
        let location = attachment
            .url
            .as_deref()
            .or(attachment.local_path.as_deref())
            .unwrap_or("");
        if location.is_empty() {
            lines.push(format!("Attachment: {}", attachment.name));
        } else {
            lines.push(format!("Attachment: {} ({})", attachment.name, location));
        }
    }
    lines.join("\n")
}

fn split_for_telegram_message(content: String) -> Vec<String> {
    let mut out = Vec::new();
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return out;
    }

    let mut current = String::new();
    for line in trimmed.lines() {
        if line.chars().count() > TELEGRAM_MESSAGE_MAX_LEN {
            if !current.is_empty() {
                out.push(current.clone());
                current.clear();
            }

            let mut segment = String::new();
            for ch in line.chars() {
                segment.push(ch);
                if segment.chars().count() >= TELEGRAM_MESSAGE_MAX_LEN {
                    out.push(segment.clone());
                    segment.clear();
                }
            }
            if !segment.is_empty() {
                out.push(segment);
            }
            continue;
        }

        let tentative = if current.is_empty() {
            line.to_string()
        } else {
            format!("{current}\n{line}")
        };
        if tentative.chars().count() > TELEGRAM_MESSAGE_MAX_LEN {
            if !current.is_empty() {
                out.push(current.clone());
            }
            current = line.to_string();
        } else {
            current = tentative;
        }
    }

    if !current.is_empty() {
        out.push(current);
    }

    out
}

fn classify_api_error(method: &str, status_code: u16, description: &str) -> String {
    let lower = description.to_ascii_lowercase();
    if status_code == 401 || lower.contains("unauthorized") {
        return "telegram_auth_invalid_token".to_string();
    }
    if status_code == 429 || lower.contains("too many requests") {
        return "telegram_rate_limited".to_string();
    }

    match method {
        "getUpdates" => {
            if lower.contains("webhook") {
                "telegram_polling_webhook_active".to_string()
            } else if lower.contains("terminated by other getupdates request")
                || lower.contains("terminated by other long poll")
            {
                "telegram_polling_conflict".to_string()
            } else {
                "telegram_get_updates_failed".to_string()
            }
        }
        "sendMessage" => {
            if lower.contains("chat not found") {
                "telegram_send_chat_not_found".to_string()
            } else {
                "telegram_send_failed".to_string()
            }
        }
        _ => "telegram_api_failed".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> TelegramChannelDriverConfig {
        TelegramChannelDriverConfig {
            base_url: DEFAULT_BASE_URL.to_string(),
            workspace_id: "telegram".to_string(),
            chat_id: "-10012345".to_string(),
            token: "token".to_string(),
            poll_timeout_secs: 30,
            poll_interval: Duration::from_millis(250),
            max_updates_per_poll: 25,
            start_from_latest: false,
            ignore_bot_messages: true,
        }
    }

    fn driver() -> TelegramChannelDriver {
        let (_tx, rx) = watch::channel(false);
        TelegramChannelDriver::from_config("telegram-runtime", config(), rx).unwrap()
    }

    #[test]
    fn normalize_uses_chat_id_as_default_thread() {
        let driver = driver();
        let update = TelegramUpdate {
            update_id: 1,
            message: Some(TelegramMessage {
                message_id: 99,
                chat: TelegramChat {
                    id: -10012345,
                    title: Some("Ops".to_string()),
                    username: None,
                    first_name: None,
                },
                from: Some(TelegramUser {
                    id: 7,
                    is_bot: Some(false),
                    first_name: Some("Ava".to_string()),
                    last_name: Some("Stone".to_string()),
                    username: Some("ava".to_string()),
                }),
                sender_chat: None,
                text: Some("hello".to_string()),
                caption: None,
                message_thread_id: None,
            }),
            channel_post: None,
        };

        let event = driver.normalize_update(update).expect("normalized event");
        assert_eq!(event.conversation.channel, ChannelKind::Telegram);
        assert_eq!(event.conversation.room_id.as_deref(), Some("-10012345"));
        assert_eq!(event.conversation.thread_id, "-10012345");
        assert_eq!(event.user.display_name.as_deref(), Some("Ava Stone"));
    }

    #[test]
    fn normalize_uses_topic_thread_id_when_present() {
        let driver = driver();
        let update = TelegramUpdate {
            update_id: 2,
            message: Some(TelegramMessage {
                message_id: 100,
                chat: TelegramChat {
                    id: -10012345,
                    title: Some("Ops".to_string()),
                    username: None,
                    first_name: None,
                },
                from: Some(TelegramUser {
                    id: 8,
                    is_bot: Some(false),
                    first_name: Some("Mia".to_string()),
                    last_name: None,
                    username: Some("mia".to_string()),
                }),
                sender_chat: None,
                text: Some("topic ping".to_string()),
                caption: None,
                message_thread_id: Some(444),
            }),
            channel_post: None,
        };

        let event = driver.normalize_update(update).expect("normalized event");
        assert_eq!(event.conversation.thread_id, "444");
        assert_eq!(event.metadata["telegram_message_thread_id"], 444);
    }

    #[test]
    fn normalize_ignores_bot_messages() {
        let driver = driver();
        let update = TelegramUpdate {
            update_id: 3,
            message: Some(TelegramMessage {
                message_id: 101,
                chat: TelegramChat {
                    id: -10012345,
                    title: Some("Ops".to_string()),
                    username: None,
                    first_name: None,
                },
                from: Some(TelegramUser {
                    id: 9,
                    is_bot: Some(true),
                    first_name: Some("Bot".to_string()),
                    last_name: None,
                    username: Some("bot".to_string()),
                }),
                sender_chat: None,
                text: Some("ignore me".to_string()),
                caption: None,
                message_thread_id: None,
            }),
            channel_post: None,
        };

        assert!(driver.normalize_update(update).is_none());
    }

    #[test]
    fn outbound_batches_split_long_messages_and_keep_thread() {
        let long_text = "x".repeat(TELEGRAM_MESSAGE_MAX_LEN + 200);
        let payloads = telegram_batches_from_message(
            "-10012345",
            Some(555),
            &OutboundMessage {
                blocks: vec![MessageBlock::Text { text: long_text }],
                ..OutboundMessage::default()
            },
        );

        assert!(payloads.len() >= 2);
        assert!(payloads.iter().all(|payload| {
            payload["text"]
                .as_str()
                .map(|text| text.chars().count() <= TELEGRAM_MESSAGE_MAX_LEN)
                .unwrap_or(false)
        }));
        assert!(
            payloads
                .iter()
                .all(|payload| payload["message_thread_id"] == 555)
        );
    }
}
