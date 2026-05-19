use anyhow::Result;
use std::time::Duration;
use tokio::time::MissedTickBehavior;
use turin_channel_core::ChannelAdapterManifest;
use turin_daemon_client::DaemonClient;
use turin_daemon_protocol::{ChannelRunnerHeartbeatParams, ChannelRunnerHelloParams};

const RUNNER_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15);

#[derive(Debug, Clone)]
pub struct RunnerPresence {
    pub manifest: ChannelAdapterManifest,
    pub runner_binary: Option<String>,
    pub runner_version: Option<String>,
    pub pid: Option<u32>,
}

pub async fn announce_runner_presence(
    daemon: &DaemonClient,
    channel_id: &str,
    presence: RunnerPresence,
) -> Result<()> {
    daemon
        .channel_runner_hello(ChannelRunnerHelloParams {
            channel_id: channel_id.to_string(),
            manifest: presence.manifest,
            runner_binary: presence.runner_binary,
            runner_version: presence.runner_version,
            pid: presence.pid,
        })
        .await
}

pub fn spawn_runner_heartbeat(
    daemon: DaemonClient,
    channel_id: String,
    mut shutdown_rx: tokio::sync::watch::Receiver<bool>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(RUNNER_HEARTBEAT_INTERVAL);
        interval.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                changed = shutdown_rx.changed() => {
                    if changed.is_err() || *shutdown_rx.borrow() {
                        break;
                    }
                }
                _ = interval.tick() => {
                    let _ = daemon.channel_runner_heartbeat(ChannelRunnerHeartbeatParams {
                        channel_id: channel_id.clone(),
                    }).await;
                }
            }
        }
    })
}
