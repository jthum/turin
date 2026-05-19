use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tokio::sync::watch;
use tokio::time::sleep;
use tracing::warn;
#[cfg(test)]
use turin_channel_core::MessageBlock;
use turin_channel_core::{
    ChannelAdapterManifest, ChannelAttachment, ChannelAuthFlowPollRequest,
    ChannelAuthFlowPollResponse, ChannelAuthFlowStartRequest, ChannelAuthFlowStartResponse,
    ChannelCapabilities, ChannelConfigField, ChannelConfigFieldOption, ChannelConfigTarget,
    ChannelConfigTargetKind, ChannelConversationKey, ChannelEnumSetting, ChannelIdentitySelectors,
    ChannelInstallManifest, ChannelKind, ChannelMessageRef, ChannelRuntimeCapabilities,
    ChannelRuntimeManifest, ChannelSecretRequirement, ChannelSessionScope, ChannelSetupManifest,
    ChannelUser, ChannelValidationCheck, DEFAULT_MAX_INBOUND_TEXT_CHARS, InboundEvent,
    OutboundMessage, bound_inbound_text,
};
use turin_channel_runner::{ChannelDriver, ChannelProgressUpdate, ChannelStreamMode};

mod outbound;
mod settings;
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

pub fn start_auth_flow(
    _request: &ChannelAuthFlowStartRequest,
) -> Result<ChannelAuthFlowStartResponse> {
    anyhow::bail!("Telegram does not expose manifest auth flows")
}

pub fn poll_auth_flow(
    _request: &ChannelAuthFlowPollRequest,
) -> Result<ChannelAuthFlowPollResponse> {
    anyhow::bail!("Telegram does not expose manifest auth flows")
}

pub fn adapter_manifest() -> ChannelAdapterManifest {
    ChannelAdapterManifest {
        protocol_version: turin_channel_core::CHANNEL_ADAPTER_PROTOCOL_VERSION,
        kind: "telegram".to_string(),
        display_name: "Telegram".to_string(),
        runtime: ChannelRuntimeManifest {
            session_scopes: vec!["user".to_string(), "thread".to_string(), "room".to_string()],
            enum_settings: vec![
                ChannelEnumSetting {
                    key: "respond_mode".to_string(),
                    options: vec![
                        "all".to_string(),
                        "mentions".to_string(),
                        "replies".to_string(),
                        "mentions_or_replies".to_string(),
                    ],
                },
                ChannelEnumSetting {
                    key: "session_scope".to_string(),
                    options: vec!["user".to_string(), "thread".to_string(), "room".to_string()],
                },
                ChannelEnumSetting {
                    key: "session_scope_dm".to_string(),
                    options: vec!["user".to_string(), "thread".to_string(), "room".to_string()],
                },
                ChannelEnumSetting {
                    key: "session_scope_group".to_string(),
                    options: vec!["user".to_string(), "thread".to_string(), "room".to_string()],
                },
                ChannelEnumSetting {
                    key: "session_scope_channel".to_string(),
                    options: vec!["user".to_string(), "thread".to_string(), "room".to_string()],
                },
            ],
            capabilities: ChannelRuntimeCapabilities {
                dm: true,
                groups: true,
                threads: true,
                attachments: true,
                streaming: true,
            },
            identity_selectors: ChannelIdentitySelectors {
                matching_rules: vec!["id".to_string(), "username".to_string()],
                examples: vec!["498502840".to_string(), "jthum".to_string()],
            },
        },
        setup: Some(ChannelSetupManifest {
            required_secrets: vec![ChannelSecretRequirement {
                name: "telegram_bot_token".to_string(),
                env_var: "TELEGRAM_BOT_TOKEN".to_string(),
                display_name: Some("Telegram bot token".to_string()),
                help: Some("Get this from @BotFather on Telegram.".to_string()),
                optional: false,
                hints: vec!["Looks like 123456789:AABBccDDeeFFgg...".to_string()],
                target: Some(ChannelConfigTarget {
                    kind: ChannelConfigTargetKind::ChannelSetting,
                    name: "token_env".to_string(),
                }),
                validate: Some(ChannelValidationCheck {
                    kind: "http_get".to_string(),
                    url_template: Some(
                        "https://api.telegram.org/bot{telegram_bot_token}/getMe".to_string(),
                    ),
                    message: Some(
                        "Verify that the supplied Telegram bot token is valid.".to_string(),
                    ),
                }),
            }],
            instructions: Some("Create a bot with BotFather, copy the token, and choose the channel settings you want Turin to apply.".to_string()),
            setup_url: Some("https://t.me/BotFather".to_string()),
            validation_checks: vec![],
            config_fields: vec![
                ChannelConfigField {
                    key: "workspace_id".to_string(),
                    label: Some("Workspace ID".to_string()),
                    field_type: "text".to_string(),
                    prompt: Some("Workspace identifier used when routing Telegram conversations into Turin".to_string()),
                    help: Some("Defaults to 'telegram' and is usually fine to leave alone.".to_string()),
                    default: Some(serde_json::json!("telegram")),
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "workspace_id".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "pairing_mode".to_string(),
                    label: Some("Pairing Mode".to_string()),
                    field_type: "select".to_string(),
                    prompt: Some("How should new Telegram chats be admitted?".to_string()),
                    help: Some("Auto is the easiest onboarding mode; pending requires explicit approval later.".to_string()),
                    default: Some(serde_json::json!("auto")),
                    options: vec![
                        ChannelConfigFieldOption {
                            value: "auto".to_string(),
                            label: Some("Auto approve new chats".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "pending".to_string(),
                            label: Some("Require manual approval".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "off".to_string(),
                            label: Some("Disable pairing".to_string()),
                        },
                    ],
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "pairing_mode".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "respond_mode".to_string(),
                    label: Some("Respond Mode".to_string()),
                    field_type: "select".to_string(),
                    prompt: Some("When should the bot respond in shared chats?".to_string()),
                    help: Some("Mentions or replies is a safe default for groups.".to_string()),
                    default: Some(serde_json::json!("mentions_or_replies")),
                    options: vec![
                        ChannelConfigFieldOption {
                            value: "all".to_string(),
                            label: Some("Every message".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "mentions".to_string(),
                            label: Some("Mentions only".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "replies".to_string(),
                            label: Some("Replies only".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "mentions_or_replies".to_string(),
                            label: Some("Mentions or replies".to_string()),
                        },
                    ],
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "respond_mode".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "session_scope".to_string(),
                    label: Some("Session Scope".to_string()),
                    field_type: "select".to_string(),
                    prompt: Some("How should Telegram conversation memory be scoped?".to_string()),
                    help: Some("Room shares memory across the room; user keeps memory isolated per sender.".to_string()),
                    default: Some(serde_json::json!("user")),
                    options: vec![
                        ChannelConfigFieldOption {
                            value: "user".to_string(),
                            label: Some("Per user".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "thread".to_string(),
                            label: Some("Per thread".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "room".to_string(),
                            label: Some("Per room".to_string()),
                        },
                    ],
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "session_scope".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "session_scope_dm".to_string(),
                    label: Some("DM Session Scope".to_string()),
                    field_type: "select".to_string(),
                    prompt: Some("Optional session scope override for private Telegram chats".to_string()),
                    help: Some("Leave empty to reuse the main session scope.".to_string()),
                    advanced: true,
                    options: vec![
                        ChannelConfigFieldOption {
                            value: "user".to_string(),
                            label: Some("Per user".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "thread".to_string(),
                            label: Some("Per thread".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "room".to_string(),
                            label: Some("Per room".to_string()),
                        },
                    ],
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "session_scope_dm".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "session_scope_group".to_string(),
                    label: Some("Group Session Scope".to_string()),
                    field_type: "select".to_string(),
                    prompt: Some("Optional session scope override for Telegram groups and supergroups".to_string()),
                    help: Some("Leave empty to reuse the main session scope.".to_string()),
                    advanced: true,
                    options: vec![
                        ChannelConfigFieldOption {
                            value: "user".to_string(),
                            label: Some("Per user".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "thread".to_string(),
                            label: Some("Per thread".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "room".to_string(),
                            label: Some("Per room".to_string()),
                        },
                    ],
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "session_scope_group".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "session_scope_channel".to_string(),
                    label: Some("Channel Session Scope".to_string()),
                    field_type: "select".to_string(),
                    prompt: Some("Optional session scope override for Telegram channels".to_string()),
                    help: Some("Leave empty to reuse the main session scope.".to_string()),
                    advanced: true,
                    options: vec![
                        ChannelConfigFieldOption {
                            value: "user".to_string(),
                            label: Some("Per user".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "thread".to_string(),
                            label: Some("Per thread".to_string()),
                        },
                        ChannelConfigFieldOption {
                            value: "room".to_string(),
                            label: Some("Per room".to_string()),
                        },
                    ],
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "session_scope_channel".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "pairing_users".to_string(),
                    label: Some("Pairing Users".to_string()),
                    field_type: "string_list".to_string(),
                    prompt: Some("Optional usernames or IDs allowed to pair new rooms".to_string()),
                    help: Some("Leave empty to allow any sender to trigger pairing.".to_string()),
                    advanced: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "pairing_users".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "allowed_users".to_string(),
                    label: Some("Allowed Users".to_string()),
                    field_type: "string_list".to_string(),
                    prompt: Some("Optional usernames or IDs allowed to interact after approval".to_string()),
                    help: Some("Leave empty to allow any user in approved rooms.".to_string()),
                    advanced: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "allowed_users".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "banned_users".to_string(),
                    label: Some("Banned Users".to_string()),
                    field_type: "string_list".to_string(),
                    prompt: Some("Optional usernames or IDs that should always be denied".to_string()),
                    advanced: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "banned_users".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
                ChannelConfigField {
                    key: "max_inbound_text_chars".to_string(),
                    label: Some("Max Inbound Text Chars".to_string()),
                    field_type: "number".to_string(),
                    help: Some(
                        "Safety cap for inbound Telegram text retained before Turin truncates it."
                            .to_string(),
                    ),
                    default: Some(serde_json::json!(DEFAULT_MAX_INBOUND_TEXT_CHARS)),
                    advanced: true,
                    target: Some(ChannelConfigTarget {
                        kind: ChannelConfigTargetKind::ChannelSetting,
                        name: "max_inbound_text_chars".to_string(),
                    }),
                    ..ChannelConfigField::default()
                },
            ],
            auth_flows: vec![],
        }),
        install: Some(ChannelInstallManifest {
            binary_name: Some("turin-channel-telegram".to_string()),
        }),
    }
}

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

    fn advance_offset(&mut self, updates: &[TelegramUpdate]) {
        if let Some(next) = updates.iter().map(|update| update.update_id).max() {
            self.next_update_offset = Some(next.saturating_add(1));
        }
    }

    #[cfg(test)]
    fn normalize_update(&self, update: TelegramUpdate) -> Option<InboundEvent> {
        let update_id = update.update_id;
        let message = update.message.or(update.channel_post)?;
        self.normalize_message(update_id, message)
    }

    fn normalize_message(&self, update_id: i64, message: TelegramMessage) -> Option<InboundEvent> {
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

    async fn api_request_once<T: DeserializeOwned>(
        &self,
        method: &str,
        payload: &serde_json::Value,
    ) -> std::result::Result<T, TelegramApiError> {
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
            .map_err(|error| TelegramApiError {
                code: "telegram_http_request_failed".to_string(),
                message: format!("Telegram {} request failed: {}", method, error),
                retriable: true,
                retry_after: None,
            })?;

        self.decode_api_response(method, response).await
    }

    async fn api_multipart_request_once<T: DeserializeOwned>(
        &self,
        method: &str,
        form: reqwest::multipart::Form,
    ) -> std::result::Result<T, TelegramApiError> {
        let url = format!(
            "{}/bot{}/{}",
            self.config.base_url, self.config.token, method
        );
        let response = self
            .client
            .post(&url)
            .multipart(form)
            .send()
            .await
            .map_err(|error| TelegramApiError {
                code: "telegram_http_request_failed".to_string(),
                message: format!("Telegram {} multipart request failed: {}", method, error),
                retriable: true,
                retry_after: None,
            })?;

        self.decode_api_response(method, response).await
    }

    async fn decode_api_response<T: DeserializeOwned>(
        &self,
        method: &str,
        response: reqwest::Response,
    ) -> std::result::Result<T, TelegramApiError> {
        let status = response.status();
        let body = response
            .text()
            .await
            .with_context(|| {
                format!(
                    "[telegram_http_decode_failed] Failed to read Telegram {} response body",
                    method
                )
            })
            .map_err(|error| TelegramApiError {
                code: "telegram_http_decode_failed".to_string(),
                message: error.to_string(),
                retriable: true,
                retry_after: None,
            })?;

        let envelope: TelegramApiEnvelope<T> = serde_json::from_str(&body)
            .with_context(|| {
                format!(
                    "[telegram_http_decode_failed] Failed to decode Telegram {} response: {}",
                    method, body
                )
            })
            .map_err(|error| TelegramApiError {
                code: "telegram_http_decode_failed".to_string(),
                message: error.to_string(),
                retriable: false,
                retry_after: None,
            })?;

        if !status.is_success() || !envelope.ok {
            let description = envelope.description.clone().unwrap_or_else(|| body.clone());
            let error_code = envelope.error_code.unwrap_or(status.as_u16() as i64);
            let code = classify_api_error(method, status.as_u16(), &description);
            let retriable = is_retriable_api_error(&code, status.as_u16())
                && !is_not_modified_description(&description);
            return Err(TelegramApiError {
                retriable,
                retry_after: envelope
                    .parameters
                    .as_ref()
                    .and_then(|parameters| parameters.retry_after)
                    .map(Duration::from_secs),
                code,
                message: format!(
                    "Telegram {} request failed with {}: {}",
                    method, error_code, description
                ),
            });
        }

        envelope
            .result
            .context(format!("Telegram {} response missing result", method))
            .map_err(|error| TelegramApiError {
                code: "telegram_missing_result".to_string(),
                message: error.to_string(),
                retriable: false,
                retry_after: None,
            })
    }

    async fn request_with_retry<T: DeserializeOwned>(
        &self,
        method: &str,
        payload: &serde_json::Value,
    ) -> std::result::Result<T, TelegramApiError> {
        let mut attempts: u32 = 0;
        loop {
            attempts = attempts.saturating_add(1);
            match self.api_request_once(method, payload).await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    if !error.retriable || attempts >= MAX_API_REQUEST_ATTEMPTS {
                        return Err(error);
                    }

                    let delay = error.retry_after.unwrap_or_else(|| retry_backoff(attempts));
                    warn!(
                        channel_runtime_id = %self.channel_runtime_id,
                        method,
                        attempt = attempts,
                        delay_ms = delay.as_millis() as u64,
                        error_code = %error.code,
                        error = %error.message,
                        "Retrying Telegram request after transient failure"
                    );
                    sleep(delay).await;
                }
            }
        }
    }

    async fn multipart_request_with_retry<T, F>(
        &self,
        method: &str,
        form_builder: F,
    ) -> std::result::Result<T, TelegramApiError>
    where
        T: DeserializeOwned,
        F: Fn() -> Result<reqwest::multipart::Form>,
    {
        let mut attempts: u32 = 0;
        loop {
            attempts = attempts.saturating_add(1);
            let form = form_builder().map_err(|error| TelegramApiError {
                code: "telegram_multipart_build_failed".to_string(),
                message: error.to_string(),
                retriable: false,
                retry_after: None,
            })?;

            match self.api_multipart_request_once(method, form).await {
                Ok(result) => return Ok(result),
                Err(error) => {
                    if !error.retriable || attempts >= MAX_API_REQUEST_ATTEMPTS {
                        return Err(error);
                    }

                    let delay = error.retry_after.unwrap_or_else(|| retry_backoff(attempts));
                    warn!(
                        channel_runtime_id = %self.channel_runtime_id,
                        method,
                        attempt = attempts,
                        delay_ms = delay.as_millis() as u64,
                        error_code = %error.code,
                        error = %error.message,
                        "Retrying Telegram multipart request after transient failure"
                    );
                    sleep(delay).await;
                }
            }
        }
    }

    async fn sleep_or_shutdown(&self, duration: Duration) -> bool {
        let mut shutdown_rx = self.shutdown_rx.clone();
        tokio::select! {
            changed = shutdown_rx.changed() => changed.is_ok() && *shutdown_rx.borrow(),
            _ = sleep(duration) => false,
        }
    }

    async fn handle_transient_poll_error(&mut self, phase: &str, error: TelegramApiError) -> bool {
        self.consecutive_poll_failures = self.consecutive_poll_failures.saturating_add(1);
        let delay = error
            .retry_after
            .unwrap_or_else(|| retry_backoff(self.consecutive_poll_failures));
        warn!(
            channel_runtime_id = %self.channel_runtime_id,
            phase,
            error_code = %error.code,
            error = %error.message,
            delay_ms = delay.as_millis() as u64,
            "Telegram polling hit a transient failure; backing off"
        );
        self.sleep_or_shutdown(delay).await
    }

    async fn ensure_bot_identity(&mut self) -> std::result::Result<(), TelegramApiError> {
        if self.bot_identity.is_some() || !self.config.respond_mode.requires_bot_identity() {
            return Ok(());
        }

        let bot: TelegramUser = self
            .request_with_retry("getMe", &serde_json::json!({}))
            .await?;
        self.bot_identity = Some(TelegramBotIdentity {
            id: bot.id,
            username: bot.username.map(|username| username.to_ascii_lowercase()),
        });
        Ok(())
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
struct TelegramApiEnvelope<T> {
    ok: bool,
    result: Option<T>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    error_code: Option<i64>,
    #[serde(default)]
    parameters: Option<TelegramApiParameters>,
}

#[derive(Debug, Clone, Deserialize)]
struct TelegramApiParameters {
    #[serde(default)]
    retry_after: Option<u64>,
}

#[derive(Debug, Clone)]
struct TelegramApiError {
    code: String,
    message: String,
    retriable: bool,
    retry_after: Option<Duration>,
}

impl TelegramApiError {
    fn into_anyhow(self) -> anyhow::Error {
        anyhow!("[{}] {}", self.code, self.message)
    }

    fn is_message_not_modified(&self) -> bool {
        self.code == "telegram_edit_message_failed" && is_not_modified_description(&self.message)
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TelegramUpdate {
    update_id: i64,
    #[serde(default)]
    message: Option<TelegramMessage>,
    #[serde(default)]
    channel_post: Option<TelegramMessage>,
}

#[derive(Debug, Clone, Default, Deserialize)]
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
    entities: Vec<TelegramMessageEntity>,
    #[serde(default)]
    caption_entities: Vec<TelegramMessageEntity>,
    #[serde(default)]
    photo: Vec<TelegramPhotoSize>,
    #[serde(default)]
    document: Option<TelegramDocument>,
    #[serde(default)]
    video: Option<TelegramVideo>,
    #[serde(default)]
    audio: Option<TelegramAudio>,
    #[serde(default)]
    voice: Option<TelegramVoice>,
    #[serde(default)]
    message_thread_id: Option<i64>,
    #[serde(default)]
    reply_to_message: Option<Box<TelegramMessage>>,
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

    fn attachment_refs(&self) -> Vec<TelegramAttachmentRef> {
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

#[derive(Debug, Clone, Default, Deserialize)]
struct TelegramPhotoSize {
    file_id: String,
    #[serde(default)]
    file_unique_id: Option<String>,
    width: u32,
    height: u32,
    #[serde(default)]
    file_size: Option<u64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TelegramDocument {
    file_id: String,
    #[serde(default)]
    file_name: Option<String>,
    #[serde(default)]
    mime_type: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TelegramVideo {
    file_id: String,
    #[serde(default)]
    file_name: Option<String>,
    #[serde(default)]
    mime_type: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TelegramAudio {
    file_id: String,
    #[serde(default)]
    file_name: Option<String>,
    #[serde(default)]
    mime_type: Option<String>,
    #[serde(default)]
    performer: Option<String>,
    #[serde(default)]
    title: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TelegramVoice {
    file_id: String,
    #[serde(default)]
    mime_type: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TelegramFile {
    #[serde(default)]
    file_path: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TelegramAttachmentKind {
    Image,
    File,
}

#[derive(Debug, Clone)]
struct TelegramAttachmentRef {
    file_id: String,
    name: String,
    content_type: Option<String>,
    kind: TelegramAttachmentKind,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TelegramMessageEntity {
    #[serde(rename = "type")]
    kind: String,
    offset: usize,
    length: usize,
    #[serde(default)]
    user: Option<TelegramUser>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TelegramChat {
    id: i64,
    #[serde(rename = "type")]
    chat_type: String,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    username: Option<String>,
    #[serde(default)]
    first_name: Option<String>,
}

impl TelegramChat {
    fn is_private(&self) -> bool {
        self.chat_type == "private"
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

fn effective_telegram_session_scope(
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
        "sendMessageDraft" => "telegram_send_draft_failed".to_string(),
        "editMessageText" => "telegram_edit_message_failed".to_string(),
        "sendChatAction" => "telegram_chat_action_failed".to_string(),
        _ => "telegram_api_failed".to_string(),
    }
}

fn is_retriable_api_error(code: &str, status_code: u16) -> bool {
    status_code == 429
        || status_code >= 500
        || matches!(
            code,
            "telegram_http_request_failed"
                | "telegram_http_decode_failed"
                | "telegram_rate_limited"
                | "telegram_send_failed"
                | "telegram_send_draft_failed"
                | "telegram_edit_message_failed"
                | "telegram_chat_action_failed"
                | "telegram_get_updates_failed"
        )
}

fn is_not_modified_description(description: &str) -> bool {
    description
        .to_ascii_lowercase()
        .contains("message is not modified")
}

fn retry_backoff(attempt: u32) -> Duration {
    let exponent = attempt.min(5);
    let millis = 250u64.saturating_mul(2u64.saturating_pow(exponent));
    Duration::from_millis(millis.min(8_000))
}

#[cfg(test)]
mod tests;
