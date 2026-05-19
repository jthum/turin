use anyhow::{Result, anyhow};
use std::time::Duration;
use turin_channel_core::{
    ChannelSessionScope, DEFAULT_MAX_INBOUND_TEXT_CHARS, optional_bool_setting,
    optional_non_empty_setting, positive_usize_setting, required_non_empty_setting,
    u64_setting_with_min,
};

use crate::{DEFAULT_BASE_URL, DEFAULT_GATEWAY_INTENTS, DEFAULT_GATEWAY_URL, DiscordTransportMode};

#[derive(Debug, Clone)]
pub struct DiscordChannelDriverConfig {
    pub base_url: String,
    pub gateway_url: String,
    pub transport_mode: DiscordTransportMode,
    pub gateway_intents: u64,
    pub workspace_id: String,
    pub room_id: Option<String>,
    pub channel_id: String,
    pub token: String,
    pub poll_interval: Duration,
    pub max_messages_per_poll: u16,
    pub max_inbound_text_chars: usize,
    pub start_from_latest: bool,
    pub ignore_bot_messages: bool,
    pub session_scope: ChannelSessionScope,
}

pub fn validate_settings(settings: &serde_json::Value) -> Result<()> {
    parse_settings(settings).map(|_| ())
}

impl DiscordChannelDriverConfig {
    pub fn from_settings(settings: &serde_json::Value) -> Result<Self> {
        let settings = parse_settings(settings)?;
        let token_env = settings.token_env.as_str();
        let token = std::env::var(token_env).map_err(|_| {
            anyhow!(
                "[discord_auth_missing_token] Discord bot token env var '{}' is not set for channel adapter",
                token_env
            )
        })?;

        Ok(Self {
            base_url: settings.base_url,
            gateway_url: settings.gateway_url,
            transport_mode: settings.transport_mode,
            gateway_intents: settings.gateway_intents,
            workspace_id: settings.workspace_id,
            room_id: settings.room_id,
            channel_id: settings.channel_id,
            token,
            poll_interval: Duration::from_millis(settings.poll_interval_ms),
            max_messages_per_poll: settings.max_messages_per_poll,
            max_inbound_text_chars: settings.max_inbound_text_chars,
            start_from_latest: settings.start_from_latest,
            ignore_bot_messages: settings.ignore_bot_messages,
            session_scope: settings.session_scope,
        })
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DiscordChannelSettings {
    pub(crate) token_env: String,
    pub(crate) base_url: String,
    pub(crate) gateway_url: String,
    pub(crate) transport_mode: DiscordTransportMode,
    pub(crate) gateway_intents: u64,
    pub(crate) workspace_id: String,
    pub(crate) room_id: Option<String>,
    pub(crate) channel_id: String,
    pub(crate) poll_interval_ms: u64,
    pub(crate) max_messages_per_poll: u16,
    pub(crate) max_inbound_text_chars: usize,
    pub(crate) start_from_latest: bool,
    pub(crate) ignore_bot_messages: bool,
    pub(crate) session_scope: ChannelSessionScope,
}

pub(crate) fn parse_settings(settings: &serde_json::Value) -> Result<DiscordChannelSettings> {
    let settings = settings
        .as_object()
        .ok_or_else(|| anyhow!("Discord channel settings must be a JSON object"))?;

    let token_env = required_non_empty_setting(
        settings,
        "token_env",
        "[discord_config_missing_token_env] Discord channel setting 'token_env' is required",
        "[discord_config_invalid_token_env] Discord channel setting 'token_env' must not be empty",
    )?
    .to_string();
    let channel_id = required_non_empty_setting(
        settings,
        "channel_id",
        "[discord_config_missing_channel_id] Discord channel setting 'channel_id' is required",
        "[discord_config_invalid_channel_id] Discord channel setting 'channel_id' must not be empty",
    )?
    .to_string();

    let poll_interval_ms = u64_setting_with_min(
        settings.get("poll_interval_ms"),
        1_000,
        100,
        "[discord_config_invalid_poll_interval] Discord channel setting 'poll_interval_ms' must be >= 100",
    )?;

    let max_messages_per_poll = u64_setting_with_min(
        settings.get("max_messages_per_poll"),
        25,
        1,
        "[discord_config_invalid_max_messages] Discord channel setting 'max_messages_per_poll' must be a positive integer",
    )?;
    if max_messages_per_poll > 100 {
        anyhow::bail!(
            "[discord_config_invalid_max_messages] Discord channel setting 'max_messages_per_poll' must be in 1..=100"
        );
    }

    let max_inbound_text_chars = positive_usize_setting(
        settings.get("max_inbound_text_chars"),
        DEFAULT_MAX_INBOUND_TEXT_CHARS,
        "[discord_config_invalid_max_inbound_text_chars] Discord channel setting 'max_inbound_text_chars' must be a positive integer",
        "[discord_config_invalid_max_inbound_text_chars] Discord channel setting 'max_inbound_text_chars' is too large",
    )?;

    let gateway_intents = match settings.get("gateway_intents") {
        None => DEFAULT_GATEWAY_INTENTS,
        Some(value) => {
            let intents = value.as_u64().ok_or_else(|| {
                anyhow!(
                    "[discord_config_invalid_gateway_intents] Discord channel setting 'gateway_intents' must be a positive integer"
                )
            })?;
            if intents == 0 {
                anyhow::bail!(
                    "[discord_config_invalid_gateway_intents] Discord channel setting 'gateway_intents' must be > 0"
                );
            }
            intents
        }
    };

    Ok(DiscordChannelSettings {
        token_env,
        base_url: read_optional_non_empty_string(settings, "base_url", DEFAULT_BASE_URL)?
            .trim_end_matches('/')
            .to_string(),
        gateway_url: read_optional_non_empty_string(settings, "gateway_url", DEFAULT_GATEWAY_URL)?
            .to_string(),
        transport_mode: parse_transport_mode(
            settings
                .get("transport")
                .map(|value| {
                    value.as_str().ok_or_else(|| {
                        anyhow!(
                            "[discord_config_invalid_transport] Discord channel setting 'transport' must be a string"
                        )
                    })
                })
                .transpose()?,
        )?,
        gateway_intents,
        workspace_id: read_optional_non_empty_string(settings, "workspace_id", "discord")?
            .to_string(),
        room_id: read_optional_string(settings, "room_id")?,
        channel_id,
        poll_interval_ms,
        max_messages_per_poll: max_messages_per_poll as u16,
        max_inbound_text_chars,
        start_from_latest: read_optional_bool(settings, "start_from_latest", true)?,
        ignore_bot_messages: read_optional_bool(settings, "ignore_bot_messages", true)?,
        session_scope: read_discord_session_scope(settings.get("session_scope"))?,
    })
}

fn read_optional_non_empty_string<'a>(
    settings: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
    default: &'a str,
) -> Result<&'a str> {
    optional_non_empty_setting(
        settings,
        key,
        format!("Discord channel setting '{}' must not be empty", key),
    )
    .map(|value| value.unwrap_or(default))
    .map_err(Into::into)
}

fn read_optional_string(
    settings: &serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<Option<String>> {
    match settings.get(key) {
        None => Ok(None),
        Some(value) => {
            let text = value
                .as_str()
                .ok_or_else(|| anyhow!("Discord channel setting '{}' must be a string", key))?;
            if text.trim().is_empty() {
                anyhow::bail!("Discord channel setting '{}' must not be empty", key);
            }
            Ok(Some(text.to_string()))
        }
    }
}

fn read_optional_bool(
    settings: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    default: bool,
) -> Result<bool> {
    optional_bool_setting(
        settings.get(key),
        default,
        format!("Discord channel setting '{}' must be a boolean", key),
    )
    .map_err(Into::into)
}

fn read_discord_session_scope(value: Option<&serde_json::Value>) -> Result<ChannelSessionScope> {
    let Some(value) = value else {
        return Ok(ChannelSessionScope::User);
    };
    let scope = value.as_str().ok_or_else(|| {
        anyhow!(
            "[discord_config_invalid_session_scope] Discord channel setting 'session_scope' must be a string"
        )
    })?;
    ChannelSessionScope::parse(scope)
        .filter(|scope| scope.is_allowed_by(&[ChannelSessionScope::User, ChannelSessionScope::Thread]))
        .ok_or_else(|| {
            anyhow!(
            "[discord_config_invalid_session_scope] Discord channel setting 'session_scope' must be one of: user, thread"
        )
        })
}

pub(crate) fn parse_transport_mode(raw: Option<&str>) -> Result<DiscordTransportMode> {
    match raw.unwrap_or("gateway") {
        "gateway" => Ok(DiscordTransportMode::Gateway),
        "polling" => Ok(DiscordTransportMode::Polling),
        other => anyhow::bail!(
            "[discord_config_invalid_transport] Invalid Discord transport '{}'; expected 'gateway' or 'polling'",
            other
        ),
    }
}
