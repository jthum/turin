use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use serde::Deserialize;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tracing::warn;
use turin_channel_core::{
    ChannelAttachment, ChannelCapabilities, ChannelConversationKey, ChannelKind, ChannelUser,
    InboundEvent, OutboundMessage,
};
#[cfg(test)]
use turin_channel_core::{
    ChannelMessageRef, ChannelSessionScope, DEFAULT_MAX_INBOUND_TEXT_CHARS, MessageBlock,
};
use turin_channel_runner::{ChannelDriver, ChannelProgressUpdate, ChannelStreamMode};

mod api;
mod inbound;
mod manifest;
mod outbound;
mod settings;
use api::TelegramApiError;
use inbound::{
    TelegramAttachmentKind, TelegramAttachmentRef, TelegramFile, TelegramMessage, TelegramUpdate,
    TelegramUser,
};
#[cfg(test)]
use inbound::{TelegramChat, TelegramMessageEntity, effective_telegram_session_scope};
pub use manifest::{adapter_manifest, poll_auth_flow, start_auth_flow};
#[cfg(test)]
use outbound::TELEGRAM_MESSAGE_MAX_LEN;
use outbound::{
    attachment_kind_from_content_type, attachment_preview_text, default_media_dir_for_runtime,
    infer_audio_name, metadata_i64, render_stream_preview, telegram_batches_from_message,
    telegram_edit_payload, telegram_payload, unique_media_name,
};
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

    async fn collect_inbound_attachments(
        &self,
        message: &TelegramMessage,
    ) -> Result<Vec<ChannelAttachment>> {
        let refs = message.attachment_refs();
        if refs.is_empty() {
            return Ok(Vec::new());
        }

        tokio::fs::create_dir_all(&self.media_dir)
            .await
            .with_context(|| {
                format!(
                    "Failed to create Telegram media directory '{}'",
                    self.media_dir.display()
                )
            })?;

        let mut attachments = Vec::with_capacity(refs.len());
        for attachment in refs {
            attachments.push(self.download_inbound_attachment(&attachment).await?);
        }
        Ok(attachments)
    }

    async fn download_inbound_attachment(
        &self,
        attachment: &TelegramAttachmentRef,
    ) -> Result<ChannelAttachment> {
        let file: TelegramFile = self
            .request_with_retry(
                "getFile",
                &serde_json::json!({ "file_id": attachment.file_id }),
            )
            .await
            .map_err(TelegramApiError::into_anyhow)?;
        let file_path = file.file_path.context(format!(
            "Telegram getFile response missing file_path for '{}'",
            attachment.file_id
        ))?;
        let download_url = self.telegram_file_url(&file_path);
        let response = self
            .client
            .get(&download_url)
            .send()
            .await
            .with_context(|| format!("Telegram file download failed for '{}'", attachment.name))?
            .error_for_status()
            .with_context(|| {
                format!(
                    "Telegram file download returned error status for '{}'",
                    attachment.name
                )
            })?;
        let fetched_content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(|value| value.split(';').next().unwrap_or(value).trim().to_string());
        let bytes = response
            .bytes()
            .await
            .with_context(|| format!("Failed to read Telegram file '{}'", attachment.name))?;
        let target_path = self.media_dir.join(unique_media_name(
            &attachment.name,
            Some(file_path.as_str()),
        ));
        tokio::fs::write(&target_path, bytes)
            .await
            .with_context(|| {
                format!(
                    "Failed to write Telegram media attachment '{}'",
                    target_path.display()
                )
            })?;
        Ok(ChannelAttachment {
            name: attachment.name.clone(),
            content_type: attachment
                .content_type
                .clone()
                .or(fetched_content_type)
                .or_else(|| match attachment.kind {
                    TelegramAttachmentKind::Image => Some("image/jpeg".to_string()),
                    TelegramAttachmentKind::File => None,
                }),
            url: None,
            local_path: Some(target_path.display().to_string()),
        })
    }

    fn telegram_file_url(&self, file_path: &str) -> String {
        format!(
            "{}/file/bot{}/{}",
            self.config.base_url,
            self.config.token,
            file_path.trim_start_matches('/')
        )
    }

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

    async fn send_chat_action(&mut self, event: &InboundEvent) -> Result<()> {
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

    async fn send_stream_preview(
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

    async fn send_final_message(
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

#[derive(Debug, Clone, Deserialize)]
struct TelegramSentMessage {
    message_id: i64,
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

#[derive(Debug, Clone)]
struct TelegramProgressState {
    sink: TelegramProgressSink,
}

#[derive(Debug, Clone)]
enum TelegramProgressSink {
    Draft { draft_id: i64 },
    Placeholder { message_id: i64 },
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

#[cfg(test)]
mod tests;
