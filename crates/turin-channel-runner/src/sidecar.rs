use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::Value;
use tokio::sync::watch;
use tracing_subscriber::EnvFilter;
use turin_channel_core::{ChannelAuthFlowPollRequest, ChannelAuthFlowStartRequest};
use turin_daemon_client::DaemonClient;

use crate::{
    ChannelAccessPolicy, ChannelDriver, ChannelRunner, RunnerConfig, task_timeout_ms_from_settings,
    tools_config_from_settings,
};

#[derive(Debug, Clone)]
pub struct ChannelSidecarRunArgs {
    pub channel_id: String,
    pub daemon_endpoint: PathBuf,
    pub bindings_path: PathBuf,
    pub access_state_path: PathBuf,
    pub idle_timeout_seconds: Option<u64>,
}

pub struct ChannelSidecarRun {
    pub channel_id: String,
    pub runner: ChannelRunner,
    pub task_timeout_ms: Option<u64>,
    pub shutdown_rx: watch::Receiver<bool>,
    pub allow_unconfigured_inbound: bool,
    pub runtime_dir: PathBuf,
}

impl ChannelSidecarRun {
    pub async fn run_driver<D: ChannelDriver + Send>(
        &self,
        agent_id: &str,
        driver: &mut D,
    ) -> Result<()> {
        self.runner
            .run_driver(agent_id, driver, self.task_timeout_ms)
            .await
            .with_context(|| format!("Channel '{}' runner failed", self.channel_id))
    }
}

pub fn prepare_channel_sidecar_run(
    args: ChannelSidecarRunArgs,
    settings: &Value,
) -> Result<ChannelSidecarRun> {
    let access_policy = ChannelAccessPolicy::from_settings(settings)?;
    let allow_unconfigured_inbound = access_policy.requires_unconfigured_inbound();
    let tools = tools_config_from_settings(settings)?;
    let task_timeout_ms = task_timeout_ms_from_settings(settings)?;
    let runtime_dir = args
        .bindings_path
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    let shutdown_rx = spawn_shutdown_signal();
    let daemon = DaemonClient::new(args.daemon_endpoint);
    let runner = ChannelRunner::new(
        daemon.clone(),
        RunnerConfig {
            channel_id: args.channel_id.clone(),
            state_path: args.bindings_path,
            access_state_path: args.access_state_path,
            idle_ttl: args.idle_timeout_seconds.map(Duration::from_secs),
            access_policy,
            tools,
        },
    );

    Ok(ChannelSidecarRun {
        channel_id: args.channel_id,
        runner,
        task_timeout_ms,
        shutdown_rx,
        allow_unconfigured_inbound,
        runtime_dir,
    })
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
