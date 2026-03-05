use anyhow::{Result, anyhow};

pub(super) fn validate_channel_settings(kind: &str, settings: &serde_json::Value) -> Result<()> {
    let Some(map) = settings.as_object() else {
        anyhow::bail!("Channel settings must be a JSON object");
    };

    match kind {
        "fs" => validate_fs_settings(map),
        "discord" => validate_discord_settings(map),
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
}
