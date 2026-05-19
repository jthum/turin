use anyhow::{Result, anyhow};
use std::collections::HashSet;
use std::time::Duration;
use turin_channel_core::{ChannelSessionScope, DEFAULT_MAX_INBOUND_TEXT_CHARS};
use turin_channel_runner::ChannelStreamMode;

use crate::TelegramRespondMode;

pub(crate) const DEFAULT_BASE_URL: &str = "https://api.telegram.org";

#[derive(Debug, Clone)]
pub struct TelegramChannelDriverConfig {
    pub base_url: String,
    pub workspace_id: String,
    pub chat_ids: Vec<String>,
    pub accept_all_chats: bool,
    pub token: String,
    pub poll_timeout_seconds: u64,
    pub poll_interval: Duration,
    pub max_updates_per_poll: u8,
    pub max_inbound_text_chars: usize,
    pub start_from_latest: bool,
    pub ignore_bot_messages: bool,
    pub respond_mode: TelegramRespondMode,
    pub session_scope: ChannelSessionScope,
    pub session_scope_dm: Option<ChannelSessionScope>,
    pub session_scope_group: Option<ChannelSessionScope>,
    pub session_scope_channel: Option<ChannelSessionScope>,
    pub stream_mode: ChannelStreamMode,
    pub stream_thinking: bool,
    pub persist_thinking: bool,
}

pub fn validate_settings(
    settings: &serde_json::Value,
    allow_unconfigured_chats: bool,
) -> Result<()> {
    parse_settings(settings, allow_unconfigured_chats).map(|_| ())
}

impl TelegramChannelDriverConfig {
    pub fn from_settings(
        settings: &serde_json::Value,
        allow_unconfigured_chats: bool,
    ) -> Result<Self> {
        let settings = parse_settings(settings, allow_unconfigured_chats)?;
        let token_env = settings.token_env.as_str();
        let token = std::env::var(token_env).map_err(|_| {
            anyhow!(
                "[telegram_auth_missing_token] Telegram bot token env var '{}' is not set for channel adapter",
                token_env
            )
        })?;

        Ok(Self {
            base_url: settings.base_url,
            workspace_id: settings.workspace_id,
            chat_ids: settings.chat_ids,
            accept_all_chats: settings.accept_all_chats,
            token,
            poll_timeout_seconds: settings.poll_timeout_seconds,
            poll_interval: Duration::from_millis(settings.poll_interval_ms),
            max_updates_per_poll: settings.max_updates_per_poll,
            max_inbound_text_chars: settings.max_inbound_text_chars,
            start_from_latest: settings.start_from_latest,
            ignore_bot_messages: settings.ignore_bot_messages,
            respond_mode: settings.respond_mode,
            session_scope: settings.session_scope,
            session_scope_dm: settings.session_scope_dm,
            session_scope_group: settings.session_scope_group,
            session_scope_channel: settings.session_scope_channel,
            stream_mode: settings.stream_mode,
            stream_thinking: settings.stream_thinking,
            persist_thinking: settings.persist_thinking,
        })
    }

    pub(crate) fn primary_chat_id(&self) -> &str {
        self.chat_ids
            .first()
            .map(String::as_str)
            .unwrap_or_default()
    }

    pub(crate) fn allows_chat_id(&self, chat_id: &str) -> bool {
        self.chat_ids.iter().any(|allowed| allowed == chat_id)
    }
}

#[derive(Debug, Clone)]
struct TelegramChannelSettings {
    token_env: String,
    base_url: String,
    workspace_id: String,
    chat_ids: Vec<String>,
    poll_timeout_seconds: u64,
    poll_interval_ms: u64,
    max_updates_per_poll: u8,
    max_inbound_text_chars: usize,
    start_from_latest: bool,
    ignore_bot_messages: bool,
    respond_mode: TelegramRespondMode,
    session_scope: ChannelSessionScope,
    session_scope_dm: Option<ChannelSessionScope>,
    session_scope_group: Option<ChannelSessionScope>,
    session_scope_channel: Option<ChannelSessionScope>,
    stream_mode: ChannelStreamMode,
    stream_thinking: bool,
    persist_thinking: bool,
    accept_all_chats: bool,
}

fn parse_settings(
    settings: &serde_json::Value,
    allow_unconfigured_chats: bool,
) -> Result<TelegramChannelSettings> {
    let settings = settings
        .as_object()
        .ok_or_else(|| anyhow!("Telegram channel settings must be a JSON object"))?;
    reject_deprecated_session_scope_keys(settings)?;

    let token_env = read_required_string(
        settings,
        "token_env",
        "[telegram_config_missing_token_env] Telegram channel setting 'token_env' is required",
        "[telegram_config_invalid_token_env] Telegram channel setting 'token_env' must not be empty",
    )?
    .to_string();

    let chat_ids = match read_chat_ids(settings) {
        Ok(ids) => ids,
        Err(_) if allow_unconfigured_chats => Vec::new(),
        Err(err) => {
            return Err(anyhow!(
                "[telegram_config_missing_chat_id] Telegram channel setting 'chat_id' or 'chat_ids' is required: {}",
                err
            ));
        }
    };

    let poll_timeout_seconds = match settings.get("poll_timeout_seconds") {
        None => 30,
        Some(value) => {
            let timeout = value.as_u64().ok_or_else(|| {
                anyhow!(
                    "[telegram_config_invalid_poll_timeout] Telegram channel setting 'poll_timeout_seconds' must be a non-negative integer"
                )
            })?;
            if timeout > 50 {
                anyhow::bail!(
                    "[telegram_config_invalid_poll_timeout] Telegram channel setting 'poll_timeout_seconds' must be <= 50"
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

    let max_inbound_text_chars = match settings.get("max_inbound_text_chars") {
        None => DEFAULT_MAX_INBOUND_TEXT_CHARS,
        Some(value) => {
            let max = value.as_u64().ok_or_else(|| {
                anyhow!(
                    "[telegram_config_invalid_max_inbound_text_chars] Telegram channel setting 'max_inbound_text_chars' must be a positive integer"
                )
            })?;
            let max = usize::try_from(max).map_err(|_| {
                anyhow!(
                    "[telegram_config_invalid_max_inbound_text_chars] Telegram channel setting 'max_inbound_text_chars' is too large"
                )
            })?;
            if max == 0 {
                anyhow::bail!(
                    "[telegram_config_invalid_max_inbound_text_chars] Telegram channel setting 'max_inbound_text_chars' must be > 0"
                );
            }
            max
        }
    };

    Ok(TelegramChannelSettings {
        token_env,
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
        chat_ids,
        poll_timeout_seconds,
        poll_interval_ms,
        max_updates_per_poll,
        max_inbound_text_chars,
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
        respond_mode: read_respond_mode(settings.get("respond_mode"))?,
        session_scope: read_telegram_session_scope(settings.get("session_scope"))?,
        session_scope_dm: read_optional_telegram_session_scope(
            settings.get("session_scope_dm"),
            "session_scope_dm",
        )?,
        session_scope_group: read_optional_telegram_session_scope(
            settings.get("session_scope_group"),
            "session_scope_group",
        )?,
        session_scope_channel: read_optional_telegram_session_scope(
            settings.get("session_scope_channel"),
            "session_scope_channel",
        )?,
        stream_mode: read_stream_mode(settings.get("stream_mode"))?,
        stream_thinking: settings
            .get("stream_thinking")
            .map(|value| {
                value.as_bool().ok_or_else(|| {
                    anyhow!(
                        "[telegram_config_invalid_stream_thinking] Telegram channel setting 'stream_thinking' must be a boolean"
                    )
                })
            })
            .transpose()?
            .unwrap_or(false),
        persist_thinking: settings
            .get("persist_thinking")
            .map(|value| {
                value.as_bool().ok_or_else(|| {
                    anyhow!(
                        "[telegram_config_invalid_persist_thinking] Telegram channel setting 'persist_thinking' must be a boolean"
                    )
                })
            })
            .transpose()?
            .unwrap_or(false),
        accept_all_chats: allow_unconfigured_chats,
    })
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

fn read_chat_ids(settings: &serde_json::Map<String, serde_json::Value>) -> Result<Vec<String>> {
    if let Some(value) = settings.get("chat_ids") {
        return read_chat_id_list(value);
    }

    Ok(vec![read_chat_id(settings.get("chat_id"))?])
}

fn read_chat_id_list(value: &serde_json::Value) -> Result<Vec<String>> {
    let mut ids = Vec::new();
    match value {
        serde_json::Value::Array(values) => {
            for item in values {
                ids.push(read_chat_id(Some(item))?);
            }
        }
        serde_json::Value::String(text) => {
            for item in text.split(',') {
                ids.push(read_chat_id(Some(&serde_json::Value::String(
                    item.trim().to_string(),
                )))?);
            }
        }
        _ => ids.push(read_chat_id(Some(value))?),
    }

    let mut seen = HashSet::new();
    ids.retain(|id| seen.insert(id.clone()));
    if ids.is_empty() {
        anyhow::bail!("chat_ids must include at least one numeric chat id");
    }
    Ok(ids)
}

fn read_respond_mode(value: Option<&serde_json::Value>) -> Result<TelegramRespondMode> {
    let Some(value) = value else {
        return Ok(TelegramRespondMode::All);
    };
    let mode = value.as_str().ok_or_else(|| {
        anyhow!(
            "[telegram_config_invalid_respond_mode] Telegram channel setting 'respond_mode' must be a string"
        )
    })?;
    match mode.trim().to_ascii_lowercase().as_str() {
        "all" => Ok(TelegramRespondMode::All),
        "mentions" => Ok(TelegramRespondMode::Mentions),
        "replies" => Ok(TelegramRespondMode::Replies),
        "mentions_or_replies" => Ok(TelegramRespondMode::MentionsOrReplies),
        _ => anyhow::bail!(
            "[telegram_config_invalid_respond_mode] Telegram channel setting 'respond_mode' must be one of: all, mentions, replies, mentions_or_replies"
        ),
    }
}

fn read_telegram_session_scope(value: Option<&serde_json::Value>) -> Result<ChannelSessionScope> {
    let Some(value) = value else {
        return Ok(ChannelSessionScope::User);
    };
    let scope = value.as_str().ok_or_else(|| {
        anyhow!(
            "[telegram_config_invalid_session_scope] Telegram channel setting 'session_scope' must be a string"
        )
    })?;
    ChannelSessionScope::parse(scope).ok_or_else(|| {
        anyhow!(
            "[telegram_config_invalid_session_scope] Telegram channel setting 'session_scope' must be one of: user, thread, room"
        )
    })
}

fn read_optional_telegram_session_scope(
    value: Option<&serde_json::Value>,
    key: &str,
) -> Result<Option<ChannelSessionScope>> {
    let Some(value) = value else {
        return Ok(None);
    };
    let scope = value.as_str().ok_or_else(|| {
        anyhow!(
            "[telegram_config_invalid_session_scope] Telegram channel setting '{}' must be a string",
            key
        )
    })?;
    ChannelSessionScope::parse(scope).map(Some).ok_or_else(|| {
        anyhow!(
            "[telegram_config_invalid_session_scope] Telegram channel setting '{}' must be one of: user, thread, room",
            key
        )
    })
}

fn reject_deprecated_session_scope_keys(
    settings: &serde_json::Map<String, serde_json::Value>,
) -> Result<()> {
    for (legacy, replacement) in [
        ("dm_session_scope", "session_scope_dm"),
        ("group_session_scope", "session_scope_group"),
        ("channel_session_scope", "session_scope_channel"),
    ] {
        if settings.contains_key(legacy) {
            anyhow::bail!(
                "[telegram_config_deprecated_session_scope_key] Telegram channel setting '{}' is no longer supported; use '{}' instead",
                legacy,
                replacement
            );
        }
    }
    Ok(())
}

fn read_stream_mode(value: Option<&serde_json::Value>) -> Result<ChannelStreamMode> {
    let Some(value) = value else {
        return Ok(ChannelStreamMode::Off);
    };
    let mode = value.as_str().ok_or_else(|| {
        anyhow!(
            "[telegram_config_invalid_stream_mode] Telegram channel setting 'stream_mode' must be a string"
        )
    })?;
    ChannelStreamMode::parse(mode).ok_or_else(|| {
        anyhow!(
            "[telegram_config_invalid_stream_mode] Telegram channel setting 'stream_mode' must be one of: off, typing, draft, block"
        )
    })
}
