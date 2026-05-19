use anyhow::{Result, anyhow};
use std::time::Duration;
use turin_channel_core::{
    ChannelSessionScope, DEFAULT_MAX_INBOUND_TEXT_CHARS, optional_bool_setting,
    optional_non_empty_setting, optional_session_scope_setting, required_non_empty_setting,
    session_scope_setting, u64_setting_with_min,
};
use turin_channel_runner::ChannelStreamMode;

use crate::{
    DEFAULT_BASE_URL, DEFAULT_MAX_MESSAGES_PER_POLL, DEFAULT_POLL_INTERVAL_MS,
    MAX_MESSAGES_PER_POLL, RocketChatReplyMode, RocketChatRespondMode, RocketChatTransportMode,
};

#[derive(Debug, Clone)]
pub struct RocketChatChannelDriverConfig {
    pub base_url: String,
    pub websocket_url: String,
    pub transport_mode: RocketChatTransportMode,
    pub workspace_id: String,
    pub accept_all_rooms: bool,
    pub room_id: Option<String>,
    pub room_name: Option<String>,
    pub user_id: String,
    pub token: String,
    pub poll_interval: Duration,
    pub max_messages_per_poll: u16,
    pub max_inbound_text_chars: usize,
    pub start_from_latest: bool,
    pub ignore_bot_messages: bool,
    pub respond_mode: RocketChatRespondMode,
    pub session_scope: ChannelSessionScope,
    pub session_scope_dm: Option<ChannelSessionScope>,
    pub session_scope_group: Option<ChannelSessionScope>,
    pub session_scope_channel: Option<ChannelSessionScope>,
    pub reply_mode: RocketChatReplyMode,
    pub stream_mode: ChannelStreamMode,
    pub persist_thinking: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct RocketChatChannelSettings {
    pub(crate) token_env: String,
    pub(crate) base_url: String,
    pub(crate) websocket_url: String,
    pub(crate) transport_mode: RocketChatTransportMode,
    pub(crate) workspace_id: String,
    pub(crate) accept_all_rooms: bool,
    pub(crate) room_id: Option<String>,
    pub(crate) room_name: Option<String>,
    pub(crate) user_id: String,
    pub(crate) poll_interval_ms: u64,
    pub(crate) max_messages_per_poll: u16,
    pub(crate) max_inbound_text_chars: usize,
    pub(crate) start_from_latest: bool,
    pub(crate) ignore_bot_messages: bool,
    pub(crate) respond_mode: RocketChatRespondMode,
    pub(crate) session_scope: ChannelSessionScope,
    pub(crate) session_scope_dm: Option<ChannelSessionScope>,
    pub(crate) session_scope_group: Option<ChannelSessionScope>,
    pub(crate) session_scope_channel: Option<ChannelSessionScope>,
    pub(crate) reply_mode: RocketChatReplyMode,
    pub(crate) stream_mode: ChannelStreamMode,
    pub(crate) persist_thinking: bool,
}

pub fn validate_settings(
    settings: &serde_json::Value,
    allow_unconfigured_rooms: bool,
) -> Result<()> {
    parse_settings(settings, allow_unconfigured_rooms).map(|_| ())
}

impl RocketChatChannelDriverConfig {
    pub fn from_settings(
        settings: &serde_json::Value,
        allow_unconfigured_rooms: bool,
    ) -> Result<Self> {
        let settings = parse_settings(settings, allow_unconfigured_rooms)?;
        let token = std::env::var(&settings.token_env).map_err(|_| {
            anyhow!(
                "[rocketchat_auth_missing_token] Rocket.Chat auth token env var '{}' is not set for channel adapter",
                settings.token_env
            )
        })?;

        Ok(Self {
            base_url: settings.base_url,
            websocket_url: settings.websocket_url,
            transport_mode: settings.transport_mode,
            workspace_id: settings.workspace_id,
            accept_all_rooms: settings.accept_all_rooms,
            room_id: settings.room_id,
            room_name: settings.room_name,
            user_id: settings.user_id,
            token,
            poll_interval: Duration::from_millis(settings.poll_interval_ms),
            max_messages_per_poll: settings.max_messages_per_poll,
            max_inbound_text_chars: settings.max_inbound_text_chars,
            start_from_latest: settings.start_from_latest,
            ignore_bot_messages: settings.ignore_bot_messages,
            respond_mode: settings.respond_mode,
            session_scope: settings.session_scope,
            session_scope_dm: settings.session_scope_dm,
            session_scope_group: settings.session_scope_group,
            session_scope_channel: settings.session_scope_channel,
            reply_mode: settings.reply_mode,
            stream_mode: settings.stream_mode,
            persist_thinking: settings.persist_thinking,
        })
    }
}

pub(crate) fn parse_settings(
    settings: &serde_json::Value,
    allow_unconfigured_rooms: bool,
) -> Result<RocketChatChannelSettings> {
    let settings = settings
        .as_object()
        .ok_or_else(|| anyhow!("Rocket.Chat channel settings must be a JSON object"))?;
    reject_deprecated_session_scope_keys(settings)?;

    let token_env = required_non_empty_setting(
        settings,
        "token_env",
        "[rocketchat_config_missing_token_env] Rocket.Chat channel setting 'token_env' is required",
        "[rocketchat_config_invalid_token_env] Rocket.Chat channel setting 'token_env' must not be empty",
    )?
    .to_string();
    let user_id = required_non_empty_setting(
        settings,
        "user_id",
        "[rocketchat_config_missing_user_id] Rocket.Chat channel setting 'user_id' is required",
        "[rocketchat_config_invalid_user_id] Rocket.Chat channel setting 'user_id' must not be empty",
    )?
    .to_string();
    let room_id = optional_non_empty_setting(
        settings,
        "room_id",
        "[rocketchat_config_invalid_room_id] Rocket.Chat channel setting 'room_id' must not be empty",
    )?
    .map(ToString::to_string);
    let room_name = optional_non_empty_setting(
        settings,
        "room_name",
        "[rocketchat_config_invalid_room_name] Rocket.Chat channel setting 'room_name' must not be empty",
    )?
    .map(ToString::to_string);

    let accept_all_rooms = room_id.is_none() && room_name.is_none() && allow_unconfigured_rooms;

    if room_id.is_none() && room_name.is_none() && !allow_unconfigured_rooms {
        anyhow::bail!(
            "[rocketchat_config_missing_room] Rocket.Chat channel requires 'room_id' or 'room_name' unless pairing is enabled"
        );
    }

    let base_url = optional_non_empty_setting(
        settings,
        "base_url",
        "[rocketchat_config_invalid_base_url] Rocket.Chat channel setting 'base_url' must not be empty",
    )?
    .unwrap_or(DEFAULT_BASE_URL)
    .trim_end_matches('/')
    .to_string();
    let websocket_url = optional_non_empty_setting(
        settings,
        "websocket_url",
        "[rocketchat_config_invalid_websocket_url] Rocket.Chat channel setting 'websocket_url' must not be empty",
    )?
    .map(ToString::to_string)
    .unwrap_or_else(|| default_websocket_url(&base_url));

    let poll_interval_ms = u64_setting_with_min(
        settings.get("poll_interval_ms"),
        DEFAULT_POLL_INTERVAL_MS,
        100,
        "[rocketchat_config_invalid_poll_interval] Rocket.Chat channel setting 'poll_interval_ms' must be a positive integer >= 100",
    )?;

    let max_messages_per_poll = u64_setting_with_min(
        settings.get("max_messages_per_poll"),
        DEFAULT_MAX_MESSAGES_PER_POLL as u64,
        1,
        "[rocketchat_config_invalid_max_messages] Rocket.Chat channel setting 'max_messages_per_poll' must be in 1..=100",
    )?;
    if max_messages_per_poll > MAX_MESSAGES_PER_POLL as u64 {
        anyhow::bail!(
            "[rocketchat_config_invalid_max_messages] Rocket.Chat channel setting 'max_messages_per_poll' must be in 1..=100"
        );
    }

    let max_inbound_text_chars = u64_setting_with_min(
        settings.get("max_inbound_text_chars"),
        DEFAULT_MAX_INBOUND_TEXT_CHARS as u64,
        1,
        "[rocketchat_config_invalid_max_inbound_text_chars] Rocket.Chat channel setting 'max_inbound_text_chars' must be a positive integer",
    )?;
    let max_inbound_text_chars = usize::try_from(max_inbound_text_chars).map_err(|_| {
        anyhow!(
            "[rocketchat_config_invalid_max_inbound_text_chars] Rocket.Chat channel setting 'max_inbound_text_chars' is too large"
        )
    })?;

    Ok(RocketChatChannelSettings {
        token_env,
        base_url,
        websocket_url,
        transport_mode: read_transport_mode(settings.get("transport_mode"))?,
        workspace_id: optional_non_empty_setting(
            settings,
            "workspace_id",
            "[rocketchat_config_invalid_workspace_id] Rocket.Chat channel setting 'workspace_id' must not be empty",
        )?
        .unwrap_or("rocketchat")
        .to_string(),
        accept_all_rooms,
        room_id,
        room_name,
        user_id,
        poll_interval_ms,
        max_messages_per_poll: max_messages_per_poll as u16,
        max_inbound_text_chars,
        start_from_latest: read_bool(settings.get("start_from_latest"), true, "start_from_latest")?,
        ignore_bot_messages: read_bool(
            settings.get("ignore_bot_messages"),
            true,
            "ignore_bot_messages",
        )?,
        respond_mode: read_respond_mode(settings.get("respond_mode"))?,
        session_scope: read_session_scope(settings.get("session_scope"))?,
        session_scope_dm: read_optional_session_scope(
            settings.get("session_scope_dm"),
            "session_scope_dm",
        )?,
        session_scope_group: read_optional_session_scope(
            settings.get("session_scope_group"),
            "session_scope_group",
        )?,
        session_scope_channel: read_optional_session_scope(
            settings.get("session_scope_channel"),
            "session_scope_channel",
        )?,
        reply_mode: read_reply_mode(settings.get("reply_mode"))?,
        stream_mode: read_stream_mode(settings.get("stream_mode"))?,
        persist_thinking: read_bool(
            settings.get("persist_thinking"),
            false,
            "persist_thinking",
        )?,
    })
}

fn read_bool(value: Option<&serde_json::Value>, default: bool, key: &str) -> Result<bool> {
    optional_bool_setting(
        value,
        default,
        format!(
            "[rocketchat_config_invalid_bool] Rocket.Chat channel setting '{}' must be true or false",
            key
        ),
    )
    .map_err(Into::into)
}

fn read_respond_mode(value: Option<&serde_json::Value>) -> Result<RocketChatRespondMode> {
    let raw = match value {
        None => return Ok(RocketChatRespondMode::Mentions),
        Some(value) => value.as_str().ok_or_else(|| {
            anyhow!(
                "[rocketchat_config_invalid_respond_mode] Rocket.Chat channel setting 'respond_mode' must be a string"
            )
        })?,
    };
    match raw {
        "all" => Ok(RocketChatRespondMode::All),
        "mentions" => Ok(RocketChatRespondMode::Mentions),
        _ => anyhow::bail!(
            "[rocketchat_config_invalid_respond_mode] Rocket.Chat channel setting 'respond_mode' must be one of: all, mentions"
        ),
    }
}

fn read_transport_mode(value: Option<&serde_json::Value>) -> Result<RocketChatTransportMode> {
    let raw = match value {
        None => return Ok(RocketChatTransportMode::Realtime),
        Some(value) => value.as_str().ok_or_else(|| {
            anyhow!(
                "[rocketchat_config_invalid_transport_mode] Rocket.Chat channel setting 'transport_mode' must be a string"
            )
        })?,
    };
    match raw {
        "realtime" => Ok(RocketChatTransportMode::Realtime),
        "polling" => Ok(RocketChatTransportMode::Polling),
        _ => anyhow::bail!(
            "[rocketchat_config_invalid_transport_mode] Rocket.Chat channel setting 'transport_mode' must be one of: realtime, polling"
        ),
    }
}

fn read_reply_mode(value: Option<&serde_json::Value>) -> Result<RocketChatReplyMode> {
    let raw = match value {
        None => return Ok(RocketChatReplyMode::Thread),
        Some(value) => value.as_str().ok_or_else(|| {
            anyhow!(
                "[rocketchat_config_invalid_reply_mode] Rocket.Chat channel setting 'reply_mode' must be a string"
            )
        })?,
    };
    match raw {
        "thread" => Ok(RocketChatReplyMode::Thread),
        "channel" => Ok(RocketChatReplyMode::Channel),
        "thread_and_channel" => Ok(RocketChatReplyMode::ThreadAndChannel),
        _ => anyhow::bail!(
            "[rocketchat_config_invalid_reply_mode] Rocket.Chat channel setting 'reply_mode' must be one of: thread, channel, thread_and_channel"
        ),
    }
}

fn read_stream_mode(value: Option<&serde_json::Value>) -> Result<ChannelStreamMode> {
    let raw = match value {
        None => return Ok(ChannelStreamMode::Typing),
        Some(value) => value.as_str().ok_or_else(|| {
            anyhow!(
                "[rocketchat_config_invalid_stream_mode] Rocket.Chat channel setting 'stream_mode' must be a string"
            )
        })?,
    };
    let mode = ChannelStreamMode::parse(raw)
        .filter(|mode| mode.is_allowed_by(&[ChannelStreamMode::Off, ChannelStreamMode::Typing]));
    mode.ok_or_else(|| {
        anyhow!(
            "[rocketchat_config_invalid_stream_mode] Rocket.Chat channel setting 'stream_mode' must be one of: off, typing"
        )
    })
}

fn read_session_scope(value: Option<&serde_json::Value>) -> Result<ChannelSessionScope> {
    session_scope_setting(
        value,
        ChannelSessionScope::Thread,
        &[
            ChannelSessionScope::User,
            ChannelSessionScope::Thread,
            ChannelSessionScope::Room,
        ],
        "[rocketchat_config_invalid_session_scope] Rocket.Chat channel setting 'session_scope' must be a string",
        "[rocketchat_config_invalid_session_scope] Rocket.Chat channel setting 'session_scope' must be one of: user, thread, room",
    )
    .map_err(Into::into)
}

fn read_optional_session_scope(
    value: Option<&serde_json::Value>,
    key: &str,
) -> Result<Option<ChannelSessionScope>> {
    optional_session_scope_setting(
        value,
        &[
            ChannelSessionScope::User,
            ChannelSessionScope::Thread,
            ChannelSessionScope::Room,
        ],
        format!(
            "[rocketchat_config_invalid_session_scope] Rocket.Chat channel setting '{}' must be a string",
            key
        ),
        format!(
            "[rocketchat_config_invalid_session_scope] Rocket.Chat channel setting '{}' must be one of: user, thread, room",
            key
        ),
    )
    .map_err(Into::into)
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
                "[rocketchat_config_deprecated_session_scope_key] Rocket.Chat channel setting '{}' is no longer supported; use '{}' instead",
                legacy,
                replacement
            );
        }
    }
    Ok(())
}

pub(crate) fn default_websocket_url(base_url: &str) -> String {
    if let Some(rest) = base_url.strip_prefix("https://") {
        return format!("wss://{}/websocket", rest.trim_end_matches('/'));
    }
    if let Some(rest) = base_url.strip_prefix("http://") {
        return format!("ws://{}/websocket", rest.trim_end_matches('/'));
    }
    if base_url.starts_with("wss://") || base_url.starts_with("ws://") {
        return format!("{}/websocket", base_url.trim_end_matches('/'));
    }
    format!("ws://{}/websocket", base_url.trim_end_matches('/'))
}
