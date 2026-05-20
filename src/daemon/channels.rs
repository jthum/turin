mod runtime_state;

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::sync::{Mutex, broadcast};
use turin_channel_core::ChannelAdapterManifest;
use turin_daemon_protocol::{ChannelRunnerHeartbeatParams, ChannelRunnerHelloParams};

use crate::daemon::channel_runners;
use crate::daemon::protocol::EventEnvelope;
use crate::daemon::registry::DiscoveredChannel;

pub use runtime_state::{ChannelRunnerHandshakeSnapshot, ChannelRuntimeSnapshot};
use runtime_state::{STATE_FAILED, STATE_RUNNING, STATE_STARTING, STATE_UNSUPPORTED};

#[cfg(not(test))]
const CHANNEL_SUPERVISOR_INTERVAL: Duration = Duration::from_secs(5);
#[cfg(test)]
const CHANNEL_SUPERVISOR_INTERVAL: Duration = Duration::from_millis(100);

#[cfg(not(test))]
const EXTERNAL_CHANNEL_HEARTBEAT_TIMEOUT: Duration = Duration::from_secs(45);
#[cfg(test)]
const EXTERNAL_CHANNEL_HEARTBEAT_TIMEOUT: Duration = Duration::from_millis(300);
#[cfg(not(test))]
const CHANNEL_RESTART_BACKOFF_BASE: Duration = Duration::from_secs(2);
#[cfg(test)]
const CHANNEL_RESTART_BACKOFF_BASE: Duration = Duration::from_millis(100);

#[cfg(not(test))]
const CHANNEL_RESTART_BACKOFF_MAX: Duration = Duration::from_secs(30);
#[cfg(test)]
const CHANNEL_RESTART_BACKOFF_MAX: Duration = Duration::from_secs(1);

struct RuntimeHandle {
    signature: String,
    shutdown_tx: tokio::sync::watch::Sender<bool>,
    join: tokio::task::JoinHandle<()>,
}

#[derive(Clone)]
struct ChannelLifecycle {
    channel_id: String,
    kind: String,
    event_tx: broadcast::Sender<EventEnvelope>,
    inner: Arc<Mutex<Inner>>,
}

impl ChannelLifecycle {
    fn new(
        channel_id: String,
        kind: String,
        event_tx: broadcast::Sender<EventEnvelope>,
        inner: Arc<Mutex<Inner>>,
    ) -> Self {
        Self {
            channel_id,
            kind,
            event_tx,
            inner,
        }
    }

    async fn mark_running(&self) {
        let mut guard = self.inner.lock().await;
        if let Some(status) = guard.by_id.get_mut(&self.channel_id) {
            status.mark_running(now_unix_ms());
            emit_runtime_update(&self.event_tx, status);
        }
    }

    async fn finish(&self, run_result: Result<()>) {
        let mut guard = self.inner.lock().await;
        if let Some(status) = guard.by_id.get_mut(&self.channel_id) {
            match run_result {
                Ok(()) => {
                    status.mark_clean_stopped(now_unix_ms());
                    emit_runtime_update(&self.event_tx, status);
                }
                Err(err) => {
                    let error_text = format!("{:#}", err);
                    status.mark_failed(
                        error_text.clone(),
                        classify_runtime_error_code(&self.kind, &error_text),
                        now_unix_ms(),
                    );
                    emit_runtime_update(&self.event_tx, status);
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
struct DesiredChannel {
    id: String,
    kind: String,
    agent_id: String,
    directory: PathBuf,
    idle_timeout_seconds: Option<u64>,
    settings: serde_json::Value,
}

impl DesiredChannel {
    fn signature(&self) -> String {
        format!(
            "{}|{}|{}|{}|{}",
            self.kind,
            self.agent_id,
            self.directory.display(),
            self.idle_timeout_seconds
                .map(|v| v.to_string())
                .unwrap_or_else(|| "none".to_string()),
            serde_json::to_string(&self.settings).unwrap_or_default()
        )
    }
}

impl From<&DiscoveredChannel> for DesiredChannel {
    fn from(channel: &DiscoveredChannel) -> Self {
        Self {
            id: channel.id.clone(),
            kind: channel.kind.clone(),
            agent_id: channel.agent_id.clone(),
            directory: channel.directory.clone(),
            idle_timeout_seconds: channel.idle_timeout_seconds,
            settings: serde_json::to_value(channel.extra.clone()).unwrap_or_default(),
        }
    }
}

struct Inner {
    workspace_root: PathBuf,
    desired_channels: HashMap<String, DiscoveredChannel>,
    by_id: HashMap<String, ChannelRuntimeSnapshot>,
    handles: HashMap<String, RuntimeHandle>,
}

pub struct ChannelRuntimeManager {
    endpoint: PathBuf,
    event_tx: broadcast::Sender<EventEnvelope>,
    inner: Arc<Mutex<Inner>>,
    sync_lock: Arc<Mutex<()>>,
}

impl ChannelRuntimeManager {
    pub fn new(endpoint: PathBuf, event_tx: broadcast::Sender<EventEnvelope>) -> Self {
        Self {
            endpoint,
            event_tx,
            inner: Arc::new(Mutex::new(Inner {
                workspace_root: PathBuf::from("."),
                desired_channels: HashMap::new(),
                by_id: HashMap::new(),
                handles: HashMap::new(),
            })),
            sync_lock: Arc::new(Mutex::new(())),
        }
    }

    pub fn start_supervisor(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(CHANNEL_SUPERVISOR_INTERVAL);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                interval.tick().await;
                if let Err(err) = self.supervise_once().await {
                    tracing::warn!(error = %err, "Channel runtime supervisor pass failed");
                }
            }
        })
    }

    pub async fn sync(
        &self,
        workspace_root: PathBuf,
        channels: Vec<DiscoveredChannel>,
    ) -> Result<()> {
        let _sync_guard = self.sync_lock.lock().await;
        let desired_registry_channels: Vec<DiscoveredChannel> = channels
            .into_iter()
            .filter(|channel| channel.enabled)
            .collect();
        let desired: Vec<DesiredChannel> =
            desired_registry_channels.iter().map(Into::into).collect();

        let desired_ids: HashSet<String> =
            desired.iter().map(|channel| channel.id.clone()).collect();

        let mut stops = Vec::new();
        let mut starts = Vec::new();
        let mut removed = Vec::new();
        let mut updates = Vec::new();

        {
            let mut inner = self.inner.lock().await;
            inner.workspace_root = workspace_root.clone();
            inner.desired_channels = desired_registry_channels
                .iter()
                .map(|channel| (channel.id.clone(), channel.clone()))
                .collect();

            let existing_ids: Vec<String> = inner.by_id.keys().cloned().collect();
            for channel_id in existing_ids {
                if !desired_ids.contains(&channel_id) {
                    if let Some(handle) = inner.handles.remove(&channel_id) {
                        stops.push(handle);
                    }
                    inner.by_id.remove(&channel_id);
                    removed.push(channel_id);
                }
            }

            for channel in &desired {
                let signature = channel.signature();
                let existing_signature =
                    inner.handles.get(&channel.id).map(|h| h.signature.clone());
                let auto_restart = existing_signature.is_none()
                    && inner
                        .by_id
                        .get(&channel.id)
                        .is_some_and(|status| status.start_count > 0);

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
                    let now = now_unix_ms();
                    let mut snapshot = inner.by_id.get(&channel.id).cloned().unwrap_or_else(|| {
                        ChannelRuntimeSnapshot::new_for_channel(channel, STATE_UNSUPPORTED, now)
                    });
                    snapshot.mark_unsupported(channel, now);
                    upsert_snapshot(&mut inner, snapshot, &mut updates);
                    continue;
                }

                if needs_start || needs_restart {
                    let now = now_unix_ms();
                    let mut snapshot = inner.by_id.get(&channel.id).cloned().unwrap_or_else(|| {
                        ChannelRuntimeSnapshot::new_for_channel(channel, STATE_STARTING, now)
                    });
                    snapshot.mark_starting(channel, now, needs_restart || auto_restart);
                    upsert_snapshot(&mut inner, snapshot, &mut updates);
                    starts.push(channel.clone());
                }
            }
        }
        self.emit_runtime_removed(removed);
        self.emit_runtime_updates(updates);

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

    pub async fn record_external_hello(
        &self,
        params: ChannelRunnerHelloParams,
    ) -> Result<ChannelRuntimeSnapshot> {
        let mut inner = self.inner.lock().await;
        let status = inner.by_id.get_mut(&params.channel_id).ok_or_else(|| {
            anyhow::anyhow!(
                "Channel runtime '{}' was not found for runner hello",
                params.channel_id
            )
        })?;

        validate_runner_hello(&status.kind, &params.manifest)?;

        let now = now_unix_ms();
        status.mark_running(now);
        status.handshake = Some(ChannelRunnerHandshakeSnapshot {
            display_name: params.manifest.display_name_or_kind().to_string(),
            protocol_version: params.manifest.protocol_version,
            runner_binary: params.runner_binary,
            runner_version: params.runner_version,
            pid: params.pid,
            last_handshake_unix_ms: now,
        });

        let snapshot = status.clone();
        emit_runtime_update(&self.event_tx, &snapshot);
        Ok(snapshot)
    }

    pub async fn record_external_heartbeat(
        &self,
        params: ChannelRunnerHeartbeatParams,
    ) -> Result<ChannelRuntimeSnapshot> {
        let mut inner = self.inner.lock().await;
        let status = inner.by_id.get_mut(&params.channel_id).ok_or_else(|| {
            anyhow::anyhow!(
                "Channel runtime '{}' was not found for runner heartbeat",
                params.channel_id
            )
        })?;

        let handshake = status.handshake.as_mut().ok_or_else(|| {
            anyhow::anyhow!(
                "Channel runtime '{}' received heartbeat before hello",
                params.channel_id
            )
        })?;

        let now = now_unix_ms();
        handshake.last_handshake_unix_ms = now;
        status.mark_running(now);

        let snapshot = status.clone();
        emit_runtime_update(&self.event_tx, &snapshot);
        Ok(snapshot)
    }

    pub async fn shutdown(&self) {
        let mut handles = Vec::new();
        let mut updates = Vec::new();
        {
            let mut inner = self.inner.lock().await;
            for (_, handle) in inner.handles.drain() {
                handles.push(handle);
            }
            for status in inner.by_id.values_mut() {
                status.mark_stopped(now_unix_ms());
                updates.push(status.clone());
            }
        }
        self.emit_runtime_updates(updates);

        for handle in handles {
            let _ = handle.shutdown_tx.send(true);
            let _ = tokio::time::timeout(Duration::from_secs(2), handle.join).await;
        }
    }

    async fn start_fs_channel(&self, _workspace_root: PathBuf, channel: DesiredChannel) {
        let endpoint = self.endpoint.clone();
        let lifecycle = ChannelLifecycle::new(
            channel.id.clone(),
            channel.kind.clone(),
            self.event_tx.clone(),
            Arc::clone(&self.inner),
        );

        let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
        let channel_id = channel.id.clone();
        let signature = channel.signature();

        let join = tokio::spawn(async move {
            let run_result = async {
                let daemon = turin_daemon_client::DaemonClient::new(&endpoint);
                let binding_state = binding_state_path(&channel.directory);
                let access_state = access_state_path(&channel.directory);
                let access_policy =
                    turin_channel_runner::ChannelAccessPolicy::from_settings(&channel.settings)?;
                let tools = turin_channel_runner::tools_config_from_settings(&channel.settings)?;
                let task_timeout_ms =
                    turin_channel_runner::task_timeout_ms_from_settings(&channel.settings)?;
                let runner = turin_channel_runner::ChannelRunner::new(
                    daemon,
                    turin_channel_runner::RunnerConfig {
                        channel_id: channel.id.clone(),
                        state_path: binding_state,
                        access_state_path: access_state,
                        idle_ttl: channel.idle_timeout_seconds.map(Duration::from_secs),
                        access_policy,
                        tools,
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

                lifecycle.mark_running().await;

                runner
                    .run_driver(&channel.agent_id, &mut driver, task_timeout_ms)
                    .await
                    .with_context(|| format!("Channel '{}' runner failed", channel.id))
            }
            .await;

            lifecycle.finish(run_result).await;
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

    async fn start_external_channel(&self, _workspace_root: PathBuf, channel: DesiredChannel) {
        let endpoint = self.endpoint.clone();
        let lifecycle = ChannelLifecycle::new(
            channel.id.clone(),
            channel.kind.clone(),
            self.event_tx.clone(),
            Arc::clone(&self.inner),
        );

        let (shutdown_tx, mut shutdown_rx) = tokio::sync::watch::channel(false);
        let channel_id = channel.id.clone();
        let signature = channel.signature();

        let join = tokio::spawn(async move {
            let run_result = async {
                let runner_command =
                    channel_runners::resolve_external_runner_command(&channel.kind)?;
                let settings_json = serde_json::to_string(&channel.settings)
                    .context("Failed to encode channel settings JSON")?;
                let binding_state = binding_state_path(&channel.directory);
                let access_state = access_state_path(&channel.directory);

                let mut child = Command::new(&runner_command.program);
                for arg in &runner_command.args_prefix {
                    child.arg(arg);
                }
                child
                    .arg("run")
                    .arg("--channel-id")
                    .arg(&channel.id)
                    .arg("--agent-id")
                    .arg(&channel.agent_id)
                    .arg("--daemon-endpoint")
                    .arg(&endpoint)
                    .arg("--bindings-path")
                    .arg(&binding_state)
                    .arg("--access-state-path")
                    .arg(&access_state)
                    .arg("--settings-json")
                    .arg(&settings_json)
                    .stdin(Stdio::null())
                    .stdout(Stdio::inherit())
                    .stderr(Stdio::piped())
                    .kill_on_drop(true);
                if let Some(idle_timeout_seconds) = channel.idle_timeout_seconds {
                    child
                        .arg("--idle-timeout-seconds")
                        .arg(idle_timeout_seconds.to_string());
                }

                let mut child = child.spawn().with_context(|| {
                    format!(
                        "Failed to spawn external {} runner '{}'",
                        channel.kind,
                        runner_command.display
                    )
                })?;
                let stderr_task = child.stderr.take().map(|mut stderr| {
                    tokio::spawn(async move {
                        let mut buf = Vec::new();
                        stderr.read_to_end(&mut buf).await?;
                        Ok::<Vec<u8>, std::io::Error>(buf)
                    })
                });

                tokio::select! {
                    status = child.wait() => {
                        let status = status.with_context(|| {
                            format!(
                                "Failed waiting for external {} runner for channel '{}'",
                                channel.kind,
                                channel.id
                            )
                        })?;
                        let stderr = collect_child_stderr(stderr_task).await;
                        if status.success() {
                            Ok(())
                        } else {
                            let message = format_external_runner_exit_error(
                                &channel.kind,
                                &channel.id,
                                status,
                                stderr.as_deref(),
                            );
                            anyhow::bail!(message);
                        }
                    }
                    changed = shutdown_rx.changed() => {
                        if changed.is_ok() && *shutdown_rx.borrow() {
                            let _ = child.start_kill();
                            let _ = tokio::time::timeout(Duration::from_secs(1), child.wait()).await;
                        }
                        let _ = collect_child_stderr(stderr_task).await;
                        Ok(())
                    }
                }
            }
            .await;

            lifecycle.finish(run_result).await;
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
        if channel.kind == "fs" {
            self.start_fs_channel(workspace_root, channel).await;
        } else {
            self.start_external_channel(workspace_root, channel).await;
        }
    }

    async fn supervise_once(&self) -> Result<()> {
        let mut updates = Vec::new();
        let mut stale_handles = Vec::new();

        let (workspace_root, desired_channels, needs_resync) = {
            let mut inner = self.inner.lock().await;
            updates.extend(Self::prune_finished_inner(&mut inner));

            let now = now_unix_ms();
            let stale_ids: Vec<String> = inner
                .handles
                .keys()
                .filter_map(|channel_id| {
                    let status = inner.by_id.get(channel_id)?;
                    let handshake = status.handshake.as_ref()?;
                    (status.kind != "fs"
                        && status.state == STATE_RUNNING
                        && now.saturating_sub(handshake.last_handshake_unix_ms)
                            > EXTERNAL_CHANNEL_HEARTBEAT_TIMEOUT.as_millis() as u64)
                        .then(|| channel_id.clone())
                })
                .collect();

            for channel_id in stale_ids {
                if let Some(status) = inner.by_id.get_mut(&channel_id) {
                    status.mark_failed(
                        format!(
                            "Channel runner heartbeat timed out after {} seconds",
                            EXTERNAL_CHANNEL_HEARTBEAT_TIMEOUT.as_secs()
                        ),
                        "channel_runner_heartbeat_stale".to_string(),
                        now,
                    );
                    updates.push(status.clone());
                }
                if let Some(handle) = inner.handles.remove(&channel_id) {
                    stale_handles.push(handle);
                }
            }

            let workspace_root = inner.workspace_root.clone();
            let desired_channels: Vec<DiscoveredChannel> =
                inner.desired_channels.values().cloned().collect();

            let needs_resync = desired_channels.iter().any(|channel| {
                if inner.handles.contains_key(&channel.id) {
                    return false;
                }
                match inner.by_id.get(&channel.id) {
                    Some(status) => restart_backoff_elapsed(status, now),
                    None => true,
                }
            });

            (workspace_root, desired_channels, needs_resync)
        };

        self.emit_runtime_updates(updates);

        for handle in stale_handles {
            let _ = handle.shutdown_tx.send(true);
            tokio::spawn(async move {
                let _ = tokio::time::timeout(Duration::from_secs(2), handle.join).await;
            });
        }

        if needs_resync {
            self.sync(workspace_root, desired_channels).await?;
        }

        Ok(())
    }

    fn prune_finished_inner(inner: &mut Inner) -> Vec<ChannelRuntimeSnapshot> {
        let mut updates = Vec::new();
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
                && status.state == STATE_RUNNING
            {
                status.mark_stopped(now_unix_ms());
                updates.push(status.clone());
            }
        }
        updates
    }

    async fn prune_finished(&self) {
        let mut inner = self.inner.lock().await;
        let updates = Self::prune_finished_inner(&mut inner);
        drop(inner);
        self.emit_runtime_updates(updates);
    }

    fn emit_runtime_updates(&self, updates: Vec<ChannelRuntimeSnapshot>) {
        for snapshot in updates {
            let _ = self.event_tx.send(EventEnvelope::new(
                "channel.runtime.updated",
                serde_json::to_value(snapshot).unwrap_or_else(|_| serde_json::json!({})),
            ));
        }
    }

    fn emit_runtime_removed(&self, removed: Vec<String>) {
        for channel_id in removed {
            let _ = self.event_tx.send(EventEnvelope::new(
                "channel.runtime.removed",
                serde_json::json!({ "id": channel_id }),
            ));
        }
    }
}

fn is_supported_kind(kind: &str) -> bool {
    channel_runners::builtin_channel_manifest(kind).is_some()
        || channel_runners::describe_external_runner(kind).is_ok()
}

fn binding_state_path(channel_dir: &std::path::Path) -> PathBuf {
    channel_dir.join("bindings.json")
}

fn access_state_path(channel_dir: &std::path::Path) -> PathBuf {
    channel_dir.join("access.json")
}

fn emit_runtime_update(
    event_tx: &broadcast::Sender<EventEnvelope>,
    snapshot: &ChannelRuntimeSnapshot,
) {
    let _ = event_tx.send(EventEnvelope::new(
        "channel.runtime.updated",
        serde_json::to_value(snapshot).unwrap_or_else(|_| serde_json::json!({})),
    ));
}

fn upsert_snapshot(
    inner: &mut Inner,
    snapshot: ChannelRuntimeSnapshot,
    updates: &mut Vec<ChannelRuntimeSnapshot>,
) {
    let changed = inner.by_id.get(&snapshot.id) != Some(&snapshot);
    if changed {
        updates.push(snapshot.clone());
    }
    inner.by_id.insert(snapshot.id.clone(), snapshot);
}

fn now_unix_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn classify_runtime_error_code(kind: &str, error: &str) -> String {
    if let Some(code) = extract_bracketed_error_code(error) {
        return code;
    }
    let lower = error.to_ascii_lowercase();
    if lower.contains("token") && lower.contains("not set") {
        return format!("{kind}_auth_missing_token");
    }
    if lower.contains("rate") && lower.contains("limit") {
        return format!("{kind}_rate_limited");
    }
    if lower.contains("connect") || lower.contains("dns") {
        return format!("{kind}_transport_connect_failed");
    }
    if lower.contains("decode") || lower.contains("parse") {
        return format!("{kind}_payload_decode_failed");
    }
    format!("{kind}_runtime_error")
}

async fn collect_child_stderr(
    stderr_task: Option<tokio::task::JoinHandle<Result<Vec<u8>, std::io::Error>>>,
) -> Option<String> {
    let join = stderr_task?;
    let bytes = join.await.ok()?.ok()?;
    let text = String::from_utf8_lossy(&bytes).trim().to_string();
    if text.is_empty() { None } else { Some(text) }
}

fn format_external_runner_exit_error(
    kind: &str,
    channel_id: &str,
    status: std::process::ExitStatus,
    stderr: Option<&str>,
) -> String {
    match stderr {
        Some(stderr) => format!(
            "External {kind} runner for channel '{channel_id}' exited with status {status}: {stderr}"
        ),
        None => {
            format!("External {kind} runner for channel '{channel_id}' exited with status {status}")
        }
    }
}

fn restart_backoff_elapsed(status: &ChannelRuntimeSnapshot, now_unix_ms: u64) -> bool {
    let delay = if status.state == STATE_FAILED {
        restart_backoff_delay(status.failure_count.max(1))
    } else {
        Duration::from_secs(0)
    };
    let Some(last_stopped) = status.last_stopped_unix_ms else {
        return true;
    };
    now_unix_ms.saturating_sub(last_stopped) >= delay.as_millis() as u64
}

fn restart_backoff_delay(failure_count: u64) -> Duration {
    let exponent = failure_count.saturating_sub(1).min(4) as u32;
    let multiplier = 1u32 << exponent;
    std::cmp::min(
        CHANNEL_RESTART_BACKOFF_MAX,
        CHANNEL_RESTART_BACKOFF_BASE.saturating_mul(multiplier),
    )
}

fn validate_runner_hello(expected_kind: &str, manifest: &ChannelAdapterManifest) -> Result<()> {
    manifest
        .validate()
        .map_err(anyhow::Error::msg)
        .context("runner hello contained an invalid adapter manifest")?;
    if manifest.kind != expected_kind {
        anyhow::bail!(
            "runner hello reported kind '{}' but runtime expects '{}'",
            manifest.kind,
            expected_kind
        );
    }
    Ok(())
}

fn extract_bracketed_error_code(error: &str) -> Option<String> {
    let start = error.find('[')?;
    let end = error[start + 1..].find(']')?;
    let code = &error[start + 1..start + 1 + end];
    if code.is_empty() {
        None
    } else {
        Some(code.to_string())
    }
}

#[cfg(test)]
#[path = "tests/channels.rs"]
mod tests;
