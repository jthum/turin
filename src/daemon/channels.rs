use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Serialize;
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::sync::{Mutex, broadcast};
use turin_channel_core::ChannelAdapterManifest;
use turin_daemon_protocol::ChannelRunnerHelloParams;

use crate::daemon::channel_runners;
use crate::daemon::protocol::EventEnvelope;
use crate::daemon::registry::DiscoveredChannel;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChannelRunnerHandshakeSnapshot {
    pub display_name: String,
    pub protocol_version: u32,
    pub runner_binary: Option<String>,
    pub runner_version: Option<String>,
    pub pid: Option<u32>,
    pub last_handshake_unix_ms: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct ChannelRuntimeSnapshot {
    pub id: String,
    pub kind: String,
    pub agent_id: String,
    pub directory: String,
    pub state: String,
    pub last_error: Option<String>,
    pub last_error_code: Option<String>,
    pub start_count: u64,
    pub restart_count: u64,
    pub failure_count: u64,
    pub last_transition_unix_ms: u64,
    pub last_started_unix_ms: Option<u64>,
    pub last_stopped_unix_ms: Option<u64>,
    pub handshake: Option<ChannelRunnerHandshakeSnapshot>,
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
    endpoint: PathBuf,
    event_tx: broadcast::Sender<EventEnvelope>,
    inner: Arc<Mutex<Inner>>,
}

impl ChannelRuntimeManager {
    pub fn new(endpoint: PathBuf, event_tx: broadcast::Sender<EventEnvelope>) -> Self {
        Self {
            endpoint,
            event_tx,
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
        let mut removed = Vec::new();
        let mut updates = Vec::new();

        {
            let mut inner = self.inner.lock().await;

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
                    let mut snapshot =
                        inner
                            .by_id
                            .get(&channel.id)
                            .cloned()
                            .unwrap_or(ChannelRuntimeSnapshot {
                                id: channel.id.clone(),
                                kind: channel.kind.clone(),
                                agent_id: channel.agent_id.clone(),
                                directory: channel.directory.display().to_string(),
                                state: "unsupported".to_string(),
                                last_error: None,
                                last_error_code: None,
                                start_count: 0,
                                restart_count: 0,
                                failure_count: 0,
                                last_transition_unix_ms: now,
                                last_started_unix_ms: None,
                                last_stopped_unix_ms: Some(now),
                                handshake: None,
                            });
                    snapshot.kind = channel.kind.clone();
                    snapshot.agent_id = channel.agent_id.clone();
                    snapshot.directory = channel.directory.display().to_string();
                    snapshot.state = "unsupported".to_string();
                    snapshot.last_error = Some(format!(
                        "No built-in or external runner is available for channel kind '{}'",
                        channel.kind,
                    ));
                    snapshot.last_error_code = Some("channel_kind_unsupported".to_string());
                    snapshot.last_transition_unix_ms = now;
                    snapshot.last_stopped_unix_ms = Some(now);
                    upsert_snapshot(&mut inner, snapshot, &mut updates);
                    continue;
                }

                if needs_start || needs_restart {
                    let now = now_unix_ms();
                    let mut snapshot =
                        inner
                            .by_id
                            .get(&channel.id)
                            .cloned()
                            .unwrap_or(ChannelRuntimeSnapshot {
                                id: channel.id.clone(),
                                kind: channel.kind.clone(),
                                agent_id: channel.agent_id.clone(),
                                directory: channel.directory.display().to_string(),
                                state: "starting".to_string(),
                                last_error: None,
                                last_error_code: None,
                                start_count: 0,
                                restart_count: 0,
                                failure_count: 0,
                                last_transition_unix_ms: now,
                                last_started_unix_ms: None,
                                last_stopped_unix_ms: Some(now),
                                handshake: None,
                            });
                    snapshot.kind = channel.kind.clone();
                    snapshot.agent_id = channel.agent_id.clone();
                    snapshot.directory = channel.directory.display().to_string();
                    snapshot.state = "starting".to_string();
                    snapshot.last_error = None;
                    snapshot.last_error_code = None;
                    snapshot.start_count = snapshot.start_count.saturating_add(1);
                    if needs_restart {
                        snapshot.restart_count = snapshot.restart_count.saturating_add(1);
                    }
                    snapshot.last_transition_unix_ms = now;
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

        status.state = "running".to_string();
        status.last_error = None;
        status.last_error_code = None;
        status.last_transition_unix_ms = now_unix_ms();
        status.last_started_unix_ms = Some(status.last_transition_unix_ms);
        status.handshake = Some(ChannelRunnerHandshakeSnapshot {
            display_name: params.manifest.display_name_or_kind().to_string(),
            protocol_version: params.manifest.protocol_version,
            runner_binary: params.runner_binary,
            runner_version: params.runner_version,
            pid: params.pid,
            last_handshake_unix_ms: now_unix_ms(),
        });

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
                status.state = "stopped".to_string();
                status.last_transition_unix_ms = now_unix_ms();
                status.last_stopped_unix_ms = Some(status.last_transition_unix_ms);
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
        let event_tx = self.event_tx.clone();
        let inner = Arc::clone(&self.inner);

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
                        idle_ttl: channel.idle_ttl_secs.map(Duration::from_secs),
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

                {
                    let mut guard = inner.lock().await;
                    if let Some(status) = guard.by_id.get_mut(&channel.id) {
                        status.state = "running".to_string();
                        status.last_error = None;
                        status.last_error_code = None;
                        status.last_transition_unix_ms = now_unix_ms();
                        status.last_started_unix_ms = Some(status.last_transition_unix_ms);
                        emit_runtime_update(&event_tx, status);
                    }
                }

                runner
                    .run_driver(&channel.agent_id, &mut driver, task_timeout_ms)
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
                        status.last_error_code = None;
                        status.last_transition_unix_ms = now_unix_ms();
                        status.last_stopped_unix_ms = Some(status.last_transition_unix_ms);
                        emit_runtime_update(&event_tx, status);
                    }
                    Err(err) => {
                        status.state = "failed".to_string();
                        let error_text = format!("{:#}", err);
                        status.last_error = Some(error_text.clone());
                        status.last_error_code =
                            Some(classify_runtime_error_code(&channel.kind, &error_text));
                        status.failure_count = status.failure_count.saturating_add(1);
                        status.last_transition_unix_ms = now_unix_ms();
                        status.last_stopped_unix_ms = Some(status.last_transition_unix_ms);
                        emit_runtime_update(&event_tx, status);
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

    async fn start_external_channel(&self, _workspace_root: PathBuf, channel: DesiredChannel) {
        let endpoint = self.endpoint.clone();
        let event_tx = self.event_tx.clone();
        let inner = Arc::clone(&self.inner);

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
                if let Some(idle_ttl_secs) = channel.idle_ttl_secs {
                    child.arg("--idle-ttl-secs").arg(idle_ttl_secs.to_string());
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

            let mut guard = inner.lock().await;
            if let Some(status) = guard.by_id.get_mut(&channel.id) {
                match run_result {
                    Ok(()) => {
                        status.state = "stopped".to_string();
                        status.last_error = None;
                        status.last_error_code = None;
                        status.last_transition_unix_ms = now_unix_ms();
                        status.last_stopped_unix_ms = Some(status.last_transition_unix_ms);
                        emit_runtime_update(&event_tx, status);
                    }
                    Err(err) => {
                        status.state = "failed".to_string();
                        let error_text = format!("{:#}", err);
                        status.last_error = Some(error_text.clone());
                        status.last_error_code =
                            Some(classify_runtime_error_code(&channel.kind, &error_text));
                        status.failure_count = status.failure_count.saturating_add(1);
                        status.last_transition_unix_ms = now_unix_ms();
                        status.last_stopped_unix_ms = Some(status.last_transition_unix_ms);
                        emit_runtime_update(&event_tx, status);
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
        if channel.kind == "fs" {
            self.start_fs_channel(workspace_root, channel).await;
        } else {
            self.start_external_channel(workspace_root, channel).await;
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
                status.last_transition_unix_ms = now_unix_ms();
                status.last_stopped_unix_ms = Some(status.last_transition_unix_ms);
            }
        }
    }

    async fn prune_finished(&self) {
        let mut inner = self.inner.lock().await;
        Self::prune_finished_inner(&mut inner);
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
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[cfg(unix)]
    #[tokio::test]
    async fn external_channel_runner_process_is_supervised() {
        use std::os::unix::fs::PermissionsExt;

        let _env_guard = crate::test_support::env_lock().lock().await;
        let temp = tempdir().unwrap();
        let workspace_root = temp.path().join("workspace");
        fs::create_dir_all(workspace_root.join(".turin/runtime/channels")).unwrap();

        let runner = temp.path().join("fake-telegram-runner.sh");
        fs::write(
            &runner,
            "#!/bin/sh\nif [ \"$1\" = \"describe\" ]; then\n  printf '%s\\n' '{\"protocol_version\":2,\"kind\":\"telegram\"}'\n  exit 0\nfi\nif [ \"$1\" = \"run\" ]; then\n  sleep 30\n  exit 0\nfi\nif [ \"$1\" = \"validate-settings\" ]; then\n  exit 0\nfi\nif [ \"$1\" = \"setup-auth-flow-start\" ]; then\n  exit 1\nfi\nif [ \"$1\" = \"setup-auth-flow-poll\" ]; then\n  exit 1\nfi\nexit 0\n",
        )
        .unwrap();
        let mut perms = fs::metadata(&runner).unwrap().permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&runner, perms).unwrap();

        let event_tx = broadcast::channel(8).0;
        let manager = ChannelRuntimeManager::new(temp.path().join("daemon.sock"), event_tx);
        let previous = std::env::var_os("TURIN_CHANNEL_TELEGRAM_BIN");
        unsafe {
            std::env::set_var("TURIN_CHANNEL_TELEGRAM_BIN", &runner);
        }

        manager
            .sync(
                workspace_root.clone(),
                vec![DiscoveredChannel {
                    id: "telegram-ops".to_string(),
                    directory: workspace_root.join(".turin/runtime/channels/telegram-ops"),
                    enabled: true,
                    kind: "telegram".to_string(),
                    agent_id: "default".to_string(),
                    idle_ttl_secs: Some(60),
                    persistence: Default::default(),
                    extra: toml::Table::new(),
                }],
            )
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_millis(200)).await;
        let runtime = manager.get("telegram-ops").await.expect("runtime exists");
        assert_eq!(runtime.state, "starting");

        manager.shutdown().await;
        if let Some(value) = previous {
            unsafe {
                std::env::set_var("TURIN_CHANNEL_TELEGRAM_BIN", value);
            }
        } else {
            unsafe {
                std::env::remove_var("TURIN_CHANNEL_TELEGRAM_BIN");
            }
        }
    }

    #[tokio::test]
    async fn external_runner_hello_marks_channel_running() {
        let event_tx = broadcast::channel(8).0;
        let manager = ChannelRuntimeManager::new(PathBuf::from("daemon.sock"), event_tx);

        {
            let mut inner = manager.inner.lock().await;
            inner.by_id.insert(
                "telegram-ops".to_string(),
                ChannelRuntimeSnapshot {
                    id: "telegram-ops".to_string(),
                    kind: "telegram".to_string(),
                    agent_id: "default".to_string(),
                    directory: "/tmp/workspace/.turin/channels/telegram-ops".to_string(),
                    state: "starting".to_string(),
                    last_error: None,
                    last_error_code: None,
                    start_count: 1,
                    restart_count: 0,
                    failure_count: 0,
                    last_transition_unix_ms: 1,
                    last_started_unix_ms: None,
                    last_stopped_unix_ms: None,
                    handshake: None,
                },
            );
        }

        let snapshot = manager
            .record_external_hello(ChannelRunnerHelloParams {
                channel_id: "telegram-ops".to_string(),
                manifest: ChannelAdapterManifest {
                    protocol_version: turin_channel_core::CHANNEL_ADAPTER_PROTOCOL_VERSION,
                    kind: "telegram".to_string(),
                    display_name: "Telegram".to_string(),
                    ..ChannelAdapterManifest::default()
                },
                runner_binary: Some("turin-channel-telegram".to_string()),
                runner_version: Some(env!("CARGO_PKG_VERSION").to_string()),
                pid: Some(1234),
            })
            .await
            .expect("hello recorded");

        assert_eq!(snapshot.state, "running");
        assert_eq!(
            snapshot.handshake.as_ref().expect("handshake").display_name,
            "Telegram"
        );
    }
}
