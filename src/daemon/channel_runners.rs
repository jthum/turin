use anyhow::Result;
use turin_channel_core::ChannelAdapterManifest;

pub(crate) use turin_channel_host::{describe_external_runner, resolve_external_runner_command};

pub(crate) fn builtin_channel_manifest(kind: &str) -> Option<ChannelAdapterManifest> {
    match kind {
        "fs" => Some(turin_channel_fs::adapter_manifest()),
        _ => None,
    }
}

pub(crate) fn validate_external_channel_settings(
    kind: &str,
    settings: &serde_json::Value,
) -> Result<()> {
    turin_channel_host::validate_external_runner_settings(
        kind,
        settings,
        &std::collections::BTreeMap::new(),
    )
}
