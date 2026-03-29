use anyhow::Result;
use std::path::Path;

pub(super) fn validate_channel_settings(
    kind: &str,
    channel_dir: &Path,
    settings: &serde_json::Value,
    _access_policy: &turin_channel_runner::ChannelAccessPolicy,
) -> Result<()> {
    match kind {
        "fs" => turin_channel_fs::validate_settings(channel_dir, settings),
        kind if crate::daemon::channel_runners::describe_external_runner(kind).is_ok() => {
            crate::daemon::channel_runners::validate_external_channel_settings(kind, settings)
        }
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
}
