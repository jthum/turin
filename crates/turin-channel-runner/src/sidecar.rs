use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;
use tokio::sync::watch;
use tracing_subscriber::EnvFilter;
use turin_channel_core::{ChannelAuthFlowPollRequest, ChannelAuthFlowStartRequest};
use turin_daemon_client::DaemonClient;

use crate::{
    ChannelAccessPolicy, ChannelDriver, ChannelRunner, RunnerConfig, task_timeout_ms_from_settings,
    tools_config_from_settings,
};

pub const DEFAULT_TURIN_CONFIG_PATH: &str = ".turin/config.toml";

#[derive(Debug, Clone)]
pub struct ChannelRunArgs {
    pub config_path: PathBuf,
    pub turin_config_path: PathBuf,
    pub expected_kind: String,
}

pub struct PreparedChannelRun {
    pub channel_id: String,
    pub agent_id: String,
    pub settings: Value,
    pub runner: ChannelRunner,
    pub task_timeout_ms: Option<u64>,
    pub shutdown_rx: watch::Receiver<bool>,
    pub allow_unconfigured_inbound: bool,
    pub runtime_dir: PathBuf,
}

impl PreparedChannelRun {
    pub async fn run_driver<D: ChannelDriver + Send>(&self, driver: &mut D) -> Result<()> {
        self.runner
            .run_driver(&self.agent_id, driver, self.task_timeout_ms)
            .await
            .with_context(|| format!("Channel '{}' runner failed", self.channel_id))
    }
}

#[derive(Debug, Deserialize)]
struct ChannelFile {
    #[serde(default = "enabled_by_default")]
    enabled: bool,
    kind: String,
    agent_id: String,
    idle_timeout_seconds: Option<u64>,
    #[serde(flatten)]
    settings: toml::Table,
}

pub async fn prepare_channel_run(args: ChannelRunArgs) -> Result<PreparedChannelRun> {
    let config = load_channel_file(&args.config_path, &args.expected_kind, true)?;
    load_turin_env(&args.turin_config_path)?;

    let settings = serde_json::to_value(config.settings)
        .context("Failed to convert channel settings to runtime values")?;
    let access_policy = ChannelAccessPolicy::from_settings(&settings)?;
    let allow_unconfigured_inbound = access_policy.requires_unconfigured_inbound();
    let tools = tools_config_from_settings(&settings)?;
    let task_timeout_ms = task_timeout_ms_from_settings(&settings)?;
    let channel_dir = args.config_path.parent().unwrap_or(Path::new("."));
    let channel_id = channel_dir
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .context("Channel config must be inside a named channel directory")?
        .to_string();
    let runtime_dir = channel_dir.join("runtime");

    let shutdown_rx = spawn_shutdown_signal();
    let daemon = DaemonClient::from_config(&args.turin_config_path)
        .await
        .with_context(|| {
            format!(
                "Failed to resolve Turin daemon from '{}'",
                args.turin_config_path.display()
            )
        })?;
    daemon.health().await.with_context(|| {
        format!(
            "Turin daemon is not reachable at '{}'; start it before launching the channel",
            daemon.endpoint().display()
        )
    })?;
    tracing::info!(
        channel_id,
        kind = args.expected_kind,
        agent_id = config.agent_id,
        daemon_endpoint = %daemon.endpoint().display(),
        "channel runner connected"
    );
    let runner = ChannelRunner::new(
        daemon.clone(),
        RunnerConfig {
            channel_id: channel_id.clone(),
            state_path: runtime_dir.join("bindings.json"),
            access_state_path: runtime_dir.join("access.json"),
            idle_ttl: config.idle_timeout_seconds.map(Duration::from_secs),
            access_policy,
            tools,
        },
    );

    Ok(PreparedChannelRun {
        channel_id,
        agent_id: config.agent_id,
        settings,
        runner,
        task_timeout_ms,
        shutdown_rx,
        allow_unconfigured_inbound,
        runtime_dir,
    })
}

fn load_channel_file(
    path: &Path,
    expected_kind: &str,
    require_enabled: bool,
) -> Result<ChannelFile> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read channel config '{}'", path.display()))?;
    let config: ChannelFile = toml::from_str(&raw)
        .with_context(|| format!("Failed to parse channel config '{}'", path.display()))?;
    if require_enabled && !config.enabled {
        anyhow::bail!("Channel in '{}' is disabled", path.display());
    }
    if config.kind != expected_kind {
        anyhow::bail!(
            "Channel config '{}' declares kind '{}' but this runner handles '{}'",
            path.display(),
            config.kind,
            expected_kind
        );
    }
    if config.agent_id.trim().is_empty() {
        anyhow::bail!("Channel config '{}' has an empty agent_id", path.display());
    }
    Ok(config)
}

pub(crate) fn validate_channel_file(path: &Path, expected_kind: &str) -> Result<()> {
    load_channel_file(path, expected_kind, false).map(|_| ())
}

fn load_turin_env(turin_config_path: &Path) -> Result<()> {
    let env_path = turin_config_path
        .parent()
        .unwrap_or(Path::new("."))
        .join(".env");
    if env_path.is_file() {
        dotenvy::from_path(&env_path)
            .with_context(|| format!("Failed to load '{}'", env_path.display()))?;
    }
    Ok(())
}

fn enabled_by_default() -> bool {
    true
}

pub fn parse_channel_settings_json(raw: &str) -> Result<Value> {
    let value: Value =
        serde_json::from_str(raw).context("Failed to parse channel settings JSON")?;
    if !value.is_object() {
        anyhow::bail!("Channel settings must be a JSON object");
    }
    Ok(value)
}

pub fn parse_auth_flow_start_request(raw: &str) -> Result<ChannelAuthFlowStartRequest> {
    serde_json::from_str(raw).context("Failed to parse auth flow start JSON")
}

pub fn parse_auth_flow_poll_request(raw: &str) -> Result<ChannelAuthFlowPollRequest> {
    serde_json::from_str(raw).context("Failed to parse auth flow poll JSON")
}

pub fn init_channel_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(true)
        .try_init();
}

fn spawn_shutdown_signal() -> watch::Receiver<bool> {
    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    tokio::spawn(async move {
        wait_for_shutdown_signal().await;
        let _ = shutdown_tx.send(true);
    });
    shutdown_rx
}

async fn wait_for_shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};

        let mut terminate = signal(SignalKind::terminate()).ok();
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = async {
                if let Some(signal) = terminate.as_mut() {
                    signal.recv().await;
                } else {
                    std::future::pending::<()>().await;
                }
            } => {}
        }
    }

    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_file_separates_launch_fields_from_adapter_settings() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            r#"
enabled = true
kind = "telegram"
agent_id = "default"
idle_timeout_seconds = 30
pairing_mode = "pending"
task_timeout_ms = 5000

[tools]
allow = ["read_file"]
"#,
        )
        .unwrap();

        let config = load_channel_file(&path, "telegram", true).unwrap();
        assert_eq!(config.agent_id, "default");
        assert_eq!(config.idle_timeout_seconds, Some(30));
        assert_eq!(
            config
                .settings
                .get("pairing_mode")
                .and_then(toml::Value::as_str),
            Some("pending")
        );
        assert!(!config.settings.contains_key("enabled"));
        assert!(!config.settings.contains_key("kind"));
        assert!(!config.settings.contains_key("agent_id"));
    }

    #[test]
    fn channel_file_rejects_disabled_or_wrong_kind() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(
            &path,
            "enabled = false\nkind = \"telegram\"\nagent_id = \"default\"\n",
        )
        .unwrap();
        assert!(load_channel_file(&path, "telegram", true).is_err());

        std::fs::write(
            &path,
            "enabled = true\nkind = \"discord\"\nagent_id = \"default\"\n",
        )
        .unwrap();
        assert!(load_channel_file(&path, "telegram", true).is_err());
    }
}
