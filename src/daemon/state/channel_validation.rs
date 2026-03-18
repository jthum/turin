use anyhow::{Result, anyhow};

pub(super) fn validate_channel_settings(kind: &str, settings: &serde_json::Value) -> Result<()> {
    let Some(map) = settings.as_object() else {
        anyhow::bail!("Channel settings must be a JSON object");
    };

    match kind {
        "fs" => validate_fs_settings(map),
        "discord" => validate_discord_settings(map),
        "telegram" => validate_telegram_settings(map),
        _ => Ok(()),
    }
}

fn validate_fs_settings(map: &serde_json::Map<String, serde_json::Value>) -> Result<()> {
    for key in ["inbox_dir", "outbox_dir", "processed_dir", "failed_dir"] {
        if let Some(value) = map.get(key) {
            let Some(path) = value.as_str() else {
                anyhow::bail!("fs channel setting '{}' must be a string", key);
            };
            if path.trim().is_empty() {
                anyhow::bail!("fs channel setting '{}' must not be empty", key);
            }
        }
    }

    if let Some(value) = map.get("poll_interval_ms") {
        let Some(interval) = value.as_u64() else {
            anyhow::bail!("fs channel setting 'poll_interval_ms' must be a positive integer");
        };
        if interval < 10 {
            anyhow::bail!("fs channel setting 'poll_interval_ms' must be >= 10");
        }
    }

    Ok(())
}

fn validate_discord_settings(map: &serde_json::Map<String, serde_json::Value>) -> Result<()> {
    let token_env = map
        .get("token_env")
        .and_then(|value| value.as_str())
        .ok_or_else(|| anyhow!("discord channel setting 'token_env' is required"))?;
    if token_env.trim().is_empty() {
        anyhow::bail!("discord channel setting 'token_env' must not be empty");
    }

    let channel_id = map
        .get("channel_id")
        .and_then(|value| value.as_str())
        .ok_or_else(|| anyhow!("discord channel setting 'channel_id' is required"))?;
    if channel_id.trim().is_empty() {
        anyhow::bail!("discord channel setting 'channel_id' must not be empty");
    }

    if let Some(value) = map.get("transport") {
        let Some(transport) = value.as_str() else {
            anyhow::bail!("discord channel setting 'transport' must be a string");
        };
        if transport != "gateway" && transport != "polling" {
            anyhow::bail!("discord channel setting 'transport' must be 'gateway' or 'polling'");
        }
    }

    if let Some(value) = map.get("poll_interval_ms") {
        let Some(interval) = value.as_u64() else {
            anyhow::bail!("discord channel setting 'poll_interval_ms' must be a positive integer");
        };
        if interval < 100 {
            anyhow::bail!("discord channel setting 'poll_interval_ms' must be >= 100");
        }
    }

    if let Some(value) = map.get("max_messages_per_poll") {
        let Some(max) = value.as_u64() else {
            anyhow::bail!(
                "discord channel setting 'max_messages_per_poll' must be a positive integer"
            );
        };
        if !(1..=100).contains(&max) {
            anyhow::bail!("discord channel setting 'max_messages_per_poll' must be in 1..=100");
        }
    }

    for key in ["workspace_id", "room_id", "base_url", "gateway_url"] {
        if let Some(value) = map.get(key) {
            let Some(text) = value.as_str() else {
                anyhow::bail!("discord channel setting '{}' must be a string", key);
            };
            if text.trim().is_empty() {
                anyhow::bail!("discord channel setting '{}' must not be empty", key);
            }
        }
    }

    for key in ["start_from_latest", "ignore_bot_messages"] {
        if let Some(value) = map.get(key)
            && !value.is_boolean()
        {
            anyhow::bail!("discord channel setting '{}' must be a boolean", key);
        }
    }

    if let Some(value) = map.get("gateway_intents") {
        let Some(intents) = value.as_u64() else {
            anyhow::bail!("discord channel setting 'gateway_intents' must be a positive integer");
        };
        if intents == 0 {
            anyhow::bail!("discord channel setting 'gateway_intents' must be > 0");
        }
    }

    Ok(())
}

fn validate_telegram_settings(map: &serde_json::Map<String, serde_json::Value>) -> Result<()> {
    let token_env = map
        .get("token_env")
        .and_then(|value| value.as_str())
        .ok_or_else(|| anyhow!("telegram channel setting 'token_env' is required"))?;
    if token_env.trim().is_empty() {
        anyhow::bail!("telegram channel setting 'token_env' must not be empty");
    }

    validate_telegram_chat_id(
        map.get("chat_id")
            .ok_or_else(|| anyhow!("telegram channel setting 'chat_id' is required"))?,
    )?;

    for key in ["base_url", "workspace_id"] {
        if let Some(value) = map.get(key) {
            let Some(text) = value.as_str() else {
                anyhow::bail!("telegram channel setting '{}' must be a string", key);
            };
            if text.trim().is_empty() {
                anyhow::bail!("telegram channel setting '{}' must not be empty", key);
            }
        }
    }

    for key in ["start_from_latest", "ignore_bot_messages"] {
        if let Some(value) = map.get(key)
            && !value.is_boolean()
        {
            anyhow::bail!("telegram channel setting '{}' must be a boolean", key);
        }
    }

    if let Some(value) = map.get("poll_timeout_secs") {
        let Some(timeout) = value.as_u64() else {
            anyhow::bail!(
                "telegram channel setting 'poll_timeout_secs' must be a non-negative integer"
            );
        };
        if timeout > 50 {
            anyhow::bail!("telegram channel setting 'poll_timeout_secs' must be <= 50");
        }
    }

    if let Some(value) = map.get("poll_interval_ms") {
        let Some(interval) = value.as_u64() else {
            anyhow::bail!("telegram channel setting 'poll_interval_ms' must be a positive integer");
        };
        if interval < 25 {
            anyhow::bail!("telegram channel setting 'poll_interval_ms' must be >= 25");
        }
    }

    if let Some(value) = map.get("max_updates_per_poll") {
        let Some(max) = value.as_u64() else {
            anyhow::bail!(
                "telegram channel setting 'max_updates_per_poll' must be a positive integer"
            );
        };
        if !(1..=100).contains(&max) {
            anyhow::bail!("telegram channel setting 'max_updates_per_poll' must be in 1..=100");
        }
    }

    Ok(())
}

fn validate_telegram_chat_id(value: &serde_json::Value) -> Result<()> {
    if let Some(id) = value.as_i64() {
        if id == 0 {
            anyhow::bail!("telegram channel setting 'chat_id' must not be zero");
        }
        return Ok(());
    }
    if let Some(id) = value.as_u64() {
        if id == 0 {
            anyhow::bail!("telegram channel setting 'chat_id' must not be zero");
        }
        return Ok(());
    }
    let Some(text) = value.as_str() else {
        anyhow::bail!("telegram channel setting 'chat_id' must be an integer or integer string");
    };
    let trimmed = text.trim();
    if trimmed.is_empty() {
        anyhow::bail!("telegram channel setting 'chat_id' must not be empty");
    }
    let digits = trimmed.strip_prefix('-').unwrap_or(trimmed);
    if digits.is_empty() || !digits.chars().all(|ch| ch.is_ascii_digit()) {
        anyhow::bail!("telegram channel setting 'chat_id' must be an integer or integer string");
    }
    if trimmed == "0" || trimmed == "-0" {
        anyhow::bail!("telegram channel setting 'chat_id' must not be zero");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn discord_requires_token_and_channel_id() {
        let error = validate_channel_settings("discord", &json!({}))
            .expect_err("empty discord settings should fail");
        assert!(error.to_string().contains("token_env"));
    }

    #[test]
    fn discord_transport_must_be_valid() {
        let error = validate_channel_settings(
            "discord",
            &json!({
                "token_env": "DISCORD_TOKEN",
                "channel_id": "123",
                "transport": "invalid"
            }),
        )
        .expect_err("invalid transport should fail");
        assert!(error.to_string().contains("transport"));
    }

    #[test]
    fn fs_poll_interval_must_be_valid() {
        let error = validate_channel_settings("fs", &json!({ "poll_interval_ms": 0 }))
            .expect_err("too-small poll interval should fail");
        assert!(error.to_string().contains("poll_interval_ms"));
    }

    #[test]
    fn discord_gateway_intents_must_be_positive() {
        let error = validate_channel_settings(
            "discord",
            &json!({
                "token_env": "DISCORD_TOKEN",
                "channel_id": "123",
                "gateway_intents": 0
            }),
        )
        .expect_err("zero gateway intents should fail");
        assert!(error.to_string().contains("gateway_intents"));
    }

    #[test]
    fn telegram_requires_token_and_chat_id() {
        let error = validate_channel_settings("telegram", &json!({}))
            .expect_err("empty telegram settings should fail");
        assert!(error.to_string().contains("token_env"));
    }

    #[test]
    fn telegram_chat_id_must_be_numeric() {
        let error = validate_channel_settings(
            "telegram",
            &json!({
                "token_env": "TELEGRAM_BOT_TOKEN",
                "chat_id": "@channel"
            }),
        )
        .expect_err("non-numeric chat id should fail");
        assert!(error.to_string().contains("chat_id"));
    }

    #[test]
    fn telegram_poll_timeout_must_be_bounded() {
        let error = validate_channel_settings(
            "telegram",
            &json!({
                "token_env": "TELEGRAM_BOT_TOKEN",
                "chat_id": -100123,
                "poll_timeout_secs": 60
            }),
        )
        .expect_err("too-large poll timeout should fail");
        assert!(error.to_string().contains("poll_timeout_secs"));
    }
}
