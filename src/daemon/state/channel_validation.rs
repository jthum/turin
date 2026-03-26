use anyhow::Result;
use std::path::Path;

pub(super) fn validate_channel_settings(
    kind: &str,
    channel_dir: &Path,
    settings: &serde_json::Value,
    access_policy: &turin_channel_runner::ChannelAccessPolicy,
) -> Result<()> {
    match kind {
        "fs" => turin_channel_fs::validate_settings(channel_dir, settings),
        "discord" => turin_channel_discord::validate_settings(settings),
        "telegram" => turin_channel_telegram::validate_settings(
            settings,
            access_policy.requires_unconfigured_inbound(),
        ),
        _ => {
            if settings.is_object() {
                Ok(())
            } else {
                anyhow::bail!("Channel settings must be a JSON object");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[test]
    fn unknown_channel_kinds_only_require_object_settings() {
        let dir = tempdir().unwrap();
        validate_channel_settings(
            "custom",
            dir.path(),
            &json!({}),
            &turin_channel_runner::ChannelAccessPolicy::default(),
        )
        .expect("unknown channel kinds should accept arbitrary object settings");

        let error = validate_channel_settings(
            "custom",
            dir.path(),
            &json!("bad"),
            &turin_channel_runner::ChannelAccessPolicy::default(),
        )
        .expect_err("unknown channel settings must still be objects");
        assert!(error.to_string().contains("JSON object"));
    }

    #[test]
    fn telegram_dispatch_uses_channel_crate_validation() {
        let dir = tempdir().unwrap();
        let error = validate_channel_settings(
            "telegram",
            dir.path(),
            &json!({
                "token_env": "TELEGRAM_BOT_TOKEN",
                "chat_id": "@ops"
            }),
            &turin_channel_runner::ChannelAccessPolicy::default(),
        )
        .expect_err("invalid telegram chat ids should fail");
        assert!(error.to_string().contains("chat_id"));
    }

    #[test]
    fn discord_dispatch_uses_channel_crate_validation() {
        let dir = tempdir().unwrap();
        let error = validate_channel_settings(
            "discord",
            dir.path(),
            &json!({
                "token_env": "DISCORD_TOKEN",
                "channel_id": "123",
                "transport": "invalid"
            }),
            &turin_channel_runner::ChannelAccessPolicy::default(),
        )
        .expect_err("invalid discord transport should fail");
        assert!(error.to_string().contains("transport"));
    }

    #[test]
    fn fs_dispatch_uses_channel_crate_validation() {
        let dir = tempdir().unwrap();
        let error = validate_channel_settings(
            "fs",
            dir.path(),
            &json!({
                "poll_interval_ms": 0
            }),
            &turin_channel_runner::ChannelAccessPolicy::default(),
        )
        .expect_err("invalid fs poll interval should fail");
        assert!(error.to_string().contains("poll_interval_ms"));
    }

    #[test]
    fn telegram_can_skip_chat_id_when_pairing_requires_unconfigured_inbound() {
        let dir = tempdir().unwrap();
        let policy = turin_channel_runner::ChannelAccessPolicy {
            pairing_mode: turin_channel_runner::PairingMode::Pending,
            ..turin_channel_runner::ChannelAccessPolicy::default()
        };
        validate_channel_settings(
            "telegram",
            dir.path(),
            &json!({
                "token_env": "TELEGRAM_BOT_TOKEN"
            }),
            &policy,
        )
        .expect("pairing-enabled telegram validation should allow missing chat ids");
    }
}
