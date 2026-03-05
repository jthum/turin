use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Serialize;
use tokio::sync::Mutex;

use crate::daemon::registry::DiscoveredChannel;

#[derive(Debug, Clone, Serialize)]
pub struct ChannelRuntimeSnapshot {
    pub id: String,
    pub kind: String,
    pub agent_id: String,
    pub directory: String,
    pub state: String,
    pub last_error: Option<String>,
}

struct RuntimeHandle {
    signature: String,
    shutdown_tx: tokio::sync::watch::Sender<bool>,
    join: tokio::task::JoinHandle<()>,
}

#[derive(Debug, Clone)]
struct DesiredChannel {
    id: String,
    kind: String,
    agent_id: String,
    directory: PathBuf,
    idle_ttl_secs: Option<u64>,
    settings: serde_json::Value,
}

impl DesiredChannel {
    fn signature(&self) -> String {
        format!(
            "{}|{}|{}|{}|{}",
            self.kind,
            self.agent_id,
            self.directory.display(),
            self.idle_ttl_secs
                .map(|v| v.to_string())
                .unwrap_or_else(|| "none".to_string()),
            serde_json::to_string(&self.settings).unwrap_or_default()
        )
    }
}

struct Inner {
    by_id: HashMap<String, ChannelRuntimeSnapshot>,
    handles: HashMap<String, RuntimeHandle>,
}

pub struct ChannelRuntimeManager {
    socket_path: PathBuf,
    inner: Arc<Mutex<Inner>>,
}

impl ChannelRuntimeManager {
    pub fn new(socket_path: PathBuf) -> Self {
        Self {
            socket_path,
            inner: Arc::new(Mutex::new(Inner {
                by_id: HashMap::new(),
                handles: HashMap::new(),
            })),
        }
    }

    pub async fn sync(
        &self,
        workspace_root: PathBuf,
        channels: Vec<DiscoveredChannel>,
    ) -> Result<()> {
        let desired: Vec<DesiredChannel> = channels
            .into_iter()
            .filter(|channel| channel.enabled)
            .map(|channel| DesiredChannel {
                id: channel.id,
                kind: channel.kind,
                agent_id: channel.agent_id,
                directory: channel.directory,
                idle_ttl_secs: channel.idle_ttl_secs,
                settings: serde_json::to_value(channel.extra).unwrap_or_default(),
            })
            .collect();

        let desired_ids: HashSet<String> =
            desired.iter().map(|channel| channel.id.clone()).collect();

        let mut stops = Vec::new();
        let mut starts = Vec::new();

        {
            let mut inner = self.inner.lock().await;

            let existing_ids: Vec<String> = inner.by_id.keys().cloned().collect();
            for channel_id in existing_ids {
                if !desired_ids.contains(&channel_id) {
                    if let Some(handle) = inner.handles.remove(&channel_id) {
                        stops.push(handle);
                    }
                    inner.by_id.remove(&channel_id);
                }
            }

            for channel in &desired {
                let signature = channel.signature();
                let existing_signature =
                    inner.handles.get(&channel.id).map(|h| h.signature.clone());

                let needs_restart = existing_signature
                    .as_ref()
                    .is_some_and(|existing| existing != &signature);
                let needs_start = existing_signature.is_none();

                if needs_restart && let Some(handle) = inner.handles.remove(&channel.id) {
                    stops.push(handle);
                }

                if !is_supported_kind(&channel.kind) {
                    if let Some(handle) = inner.handles.remove(&channel.id) {
                        stops.push(handle);
                    }
                    inner.by_id.insert(
                        channel.id.clone(),
                        ChannelRuntimeSnapshot {
                            id: channel.id.clone(),
                            kind: channel.kind.clone(),
                            agent_id: channel.agent_id.clone(),
                            directory: channel.directory.display().to_string(),
                            state: "unsupported".to_string(),
                            last_error: Some(format!(
                                "No daemon-owned runner available for channel kind '{}' (supported: fs, discord)",
                                channel.kind,
                            )),
                        },
                    );
                    continue;
                }

                if needs_start || needs_restart {
                    inner.by_id.insert(
                        channel.id.clone(),
                        ChannelRuntimeSnapshot {
                            id: channel.id.clone(),
                            kind: channel.kind.clone(),
                            agent_id: channel.agent_id.clone(),
                            directory: channel.directory.display().to_string(),
                            state: "starting".to_string(),
                            last_error: None,
                        },
                    );
                    starts.push(channel.clone());
                }
            }
        }

        for handle in stops {
            let _ = handle.shutdown_tx.send(true);
            let _ = tokio::time::timeout(Duration::from_secs(2), handle.join).await;
        }

        for channel in starts {
            self.start_channel(workspace_root.clone(), channel).await;
        }

        self.prune_finished().await;
        Ok(())
    }

    pub async fn list(&self) -> Vec<ChannelRuntimeSnapshot> {
        self.prune_finished().await;
        let mut values: Vec<_> = self.inner.lock().await.by_id.values().cloned().collect();
        values.sort_by(|a, b| a.id.cmp(&b.id));
        values
    }

    pub async fn get(&self, channel_id: &str) -> Option<ChannelRuntimeSnapshot> {
        self.prune_finished().await;
        self.inner.lock().await.by_id.get(channel_id).cloned()
    }

    pub async fn shutdown(&self) {
        let mut handles = Vec::new();
        {
            let mut inner = self.inner.lock().await;
            for (_, handle) in inner.handles.drain() {
                handles.push(handle);
            }
            for status in inner.by_id.values_mut() {
                status.state = "stopped".to_string();
            }
        }

        for handle in handles {
            let _ = handle.shutdown_tx.send(true);
            let _ = tokio::time::timeout(Duration::from_secs(2), handle.join).await;
        }
    }

    async fn start_fs_channel(&self, workspace_root: PathBuf, channel: DesiredChannel) {
        let socket_path = self.socket_path.clone();
        let inner = Arc::clone(&self.inner);

        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        let channel_id = channel.id.clone();
        let signature = channel.signature();

        let join = tokio::spawn(async move {
            let run_result = async {
                let daemon = turin_daemon_client::DaemonClient::new(&socket_path);
                let binding_state = workspace_root
                    .join(".turin/channels")
                    .join(format!("{}-bindings.json", channel.id));
                let runner = turin_channel_runner::ChannelRunner::new(
                    daemon,
                    turin_channel_runner::RunnerConfig {
                        state_path: binding_state,
                        idle_ttl: channel.idle_ttl_secs.map(Duration::from_secs),
                    },
                );

                let mut driver = turin_channel_fs::FsChannelDriver::from_settings(
                    &channel.id,
                    &channel.directory,
                    &channel.settings,
                    shutdown_rx,
                )
                .await
                .with_context(|| {
                    format!("Failed to initialize fs channel driver '{}'", channel.id)
                })?;

                {
                    let mut guard = inner.lock().await;
                    if let Some(status) = guard.by_id.get_mut(&channel.id) {
                        status.state = "running".to_string();
                        status.last_error = None;
                    }
                }

                runner
                    .run_driver(&channel.agent_id, &mut driver, Some(120_000))
                    .await
                    .with_context(|| format!("Channel '{}' runner failed", channel.id))
            }
            .await;

            let mut guard = inner.lock().await;
            if let Some(status) = guard.by_id.get_mut(&channel.id) {
                match run_result {
                    Ok(()) => {
                        status.state = "stopped".to_string();
                        status.last_error = None;
                    }
                    Err(err) => {
                        status.state = "failed".to_string();
                        status.last_error = Some(err.to_string());
                    }
                }
            }
        });

        let mut guard = self.inner.lock().await;
        guard.handles.insert(
            channel_id,
            RuntimeHandle {
                signature,
                shutdown_tx,
                join,
            },
        );
    }

    async fn start_discord_channel(&self, workspace_root: PathBuf, channel: DesiredChannel) {
        let socket_path = self.socket_path.clone();
        let inner = Arc::clone(&self.inner);

        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        let channel_id = channel.id.clone();
        let signature = channel.signature();

        let join = tokio::spawn(async move {
            let run_result = async {
                let daemon = turin_daemon_client::DaemonClient::new(&socket_path);
                let binding_state = workspace_root
                    .join(".turin/channels")
                    .join(format!("{}-bindings.json", channel.id));
                let runner = turin_channel_runner::ChannelRunner::new(
                    daemon,
                    turin_channel_runner::RunnerConfig {
                        state_path: binding_state,
                        idle_ttl: channel.idle_ttl_secs.map(Duration::from_secs),
                    },
                );

                let mut driver = turin_channel_discord::DiscordChannelDriver::from_settings(
                    &channel.id,
                    &channel.settings,
                    shutdown_rx,
                )
                .await
                .with_context(|| {
                    format!(
                        "Failed to initialize discord channel driver '{}'",
                        channel.id
                    )
                })?;

                {
                    let mut guard = inner.lock().await;
                    if let Some(status) = guard.by_id.get_mut(&channel.id) {
                        status.state = "running".to_string();
                        status.last_error = None;
                    }
                }

                runner
                    .run_driver(&channel.agent_id, &mut driver, Some(120_000))
                    .await
                    .with_context(|| format!("Channel '{}' runner failed", channel.id))
            }
            .await;

            let mut guard = inner.lock().await;
            if let Some(status) = guard.by_id.get_mut(&channel.id) {
                match run_result {
                    Ok(()) => {
                        status.state = "stopped".to_string();
                        status.last_error = None;
                    }
                    Err(err) => {
                        status.state = "failed".to_string();
                        status.last_error = Some(err.to_string());
                    }
                }
            }
        });

        let mut guard = self.inner.lock().await;
        guard.handles.insert(
            channel_id,
            RuntimeHandle {
                signature,
                shutdown_tx,
                join,
            },
        );
    }

    async fn start_channel(&self, workspace_root: PathBuf, channel: DesiredChannel) {
        match channel.kind.as_str() {
            "fs" => self.start_fs_channel(workspace_root, channel).await,
            "discord" => self.start_discord_channel(workspace_root, channel).await,
            _ => {}
        }
    }

    fn prune_finished_inner(inner: &mut Inner) {
        let finished: Vec<String> = inner
            .handles
            .iter()
            .filter_map(|(channel_id, handle)| {
                handle.join.is_finished().then_some(channel_id.clone())
            })
            .collect();

        for channel_id in finished {
            inner.handles.remove(&channel_id);
            if let Some(status) = inner.by_id.get_mut(&channel_id)
                && status.state == "running"
            {
                status.state = "stopped".to_string();
            }
        }
    }

    async fn prune_finished(&self) {
        let mut inner = self.inner.lock().await;
        Self::prune_finished_inner(&mut inner);
    }
}

fn is_supported_kind(kind: &str) -> bool {
    matches!(kind, "fs" | "discord")
}
