use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;
use turin::daemon::protocol::{RequestEnvelope, ResponseEnvelope};
use turin::kernel::config::TurinConfig;
use turin_daemon_client::{DaemonClient, DaemonHealth, DaemonHealthState};

use super::{DaemonHealthReport, DaemonStartReport};

pub(super) async fn send_request(
    config_path: &Path,
    op: &str,
    params: Value,
) -> Result<ResponseEnvelope> {
    let client = daemon_client_from_config(config_path)?;
    let request: RequestEnvelope = serde_json::from_value(json!({
        "id": format!("req-{}", uuid::Uuid::new_v4()),
        "op": op,
        "params": params,
    }))
    .with_context(|| format!("Failed to build daemon request '{}'", op))?;
    client
        .send(request)
        .await
        .map_err(|err| wrap_daemon_client_error(config_path, err))
}

pub(super) fn resolve_endpoint_path(config_path: &Path) -> Result<PathBuf> {
    let config = TurinConfig::from_file(config_path)?;
    let config_base = config_path.parent().unwrap_or_else(|| Path::new("."));
    Ok(config.resolve_daemon_endpoint(config_base))
}

pub(super) fn daemon_client_from_config(config_path: &Path) -> Result<DaemonClient> {
    let config = TurinConfig::from_file(config_path)?;
    let config_base = config_path.parent().unwrap_or_else(|| Path::new("."));
    Ok(DaemonClient::new(
        config.resolve_daemon_endpoint(config_base),
    ))
}

pub(super) async fn daemon_health_report(config_path: &Path) -> Result<DaemonHealthReport> {
    let client = daemon_client_from_config(config_path)?;
    match client.health().await {
        Ok(health) => Ok(DaemonHealthReport::from_health(health)),
        Err(err) if is_daemon_offline_error(&err) => Ok(DaemonHealthReport::offline(
            client.endpoint().display().to_string(),
            err.to_string(),
        )),
        Err(err) => Err(wrap_daemon_client_error(config_path, err)),
    }
}

pub(super) async fn ensure_background_daemon(
    config_path: &Path,
    timeout: Duration,
    poll_interval: Duration,
    log_level: &str,
    log_file_override: Option<&Path>,
) -> Result<DaemonStartReport> {
    let client = daemon_client_from_config(config_path)?;
    match client.health().await {
        Ok(health) => {
            let log_path = resolve_daemon_log_path(config_path, log_file_override)?;
            return Ok(DaemonStartReport {
                started: false,
                endpoint: health.endpoint.clone(),
                log_path: log_path.display().to_string(),
                health: DaemonHealthReport::from_health(health),
            });
        }
        Err(err) if is_daemon_offline_error(&err) => {}
        Err(err) => return Err(wrap_daemon_client_error(config_path, err)),
    }

    spawn_background_daemon(config_path, log_level, log_file_override)?;
    client
        .wait_until_ready(timeout, poll_interval)
        .await
        .map_err(|err| wrap_daemon_client_error(config_path, err))?;
    let health = client
        .health()
        .await
        .map_err(|err| wrap_daemon_client_error(config_path, err))?;
    let log_path = resolve_daemon_log_path(config_path, log_file_override)?;
    Ok(DaemonStartReport {
        started: true,
        endpoint: health.endpoint.clone(),
        log_path: log_path.display().to_string(),
        health: DaemonHealthReport::from_health(health),
    })
}

pub(super) fn resolve_daemon_log_path(
    config_path: &Path,
    log_file_override: Option<&Path>,
) -> Result<PathBuf> {
    if let Some(path) = log_file_override {
        return absolute_path(path);
    }

    let config = TurinConfig::from_file(config_path)?;
    let config_base = config_path.parent().unwrap_or_else(|| Path::new("."));
    Ok(config
        .resolve_workspace_root(config_base)
        .join(".turin/daemon.log"))
}

pub(super) fn tail_lines(path: &Path, count: usize) -> Result<Vec<String>> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read daemon log '{}'", path.display()))?;
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut lines: Vec<String> = contents.lines().map(str::to_string).collect();
    if lines.len() > count {
        lines = lines.split_off(lines.len() - count);
    }
    Ok(lines)
}

pub(super) fn wrap_daemon_client_error(config_path: &Path, err: anyhow::Error) -> anyhow::Error {
    if is_daemon_offline_error(&err) {
        return err.context(format!(
            "Turin daemon is not running. Start it with `turin daemon ensure --config {}` or `turin daemon start --background --config {}`",
            config_path.display(),
            config_path.display()
        ));
    }
    err
}

fn spawn_background_daemon(
    config_path: &Path,
    log_level: &str,
    log_file_override: Option<&Path>,
) -> Result<()> {
    let current_exe =
        std::env::current_exe().with_context(|| "Failed to resolve current Turin binary path")?;
    let config_path = absolute_path(config_path)?;
    let log_path = resolve_daemon_log_path(config_path.as_path(), log_file_override)?;
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "Failed to create daemon log directory '{}'",
                parent.display()
            )
        })?;
    }
    let log_file = File::options()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("Failed to open daemon log '{}'", log_path.display()))?;
    let stderr = log_file
        .try_clone()
        .with_context(|| format!("Failed to clone daemon log handle '{}'", log_path.display()))?;

    let mut child = Command::new(current_exe);
    child
        .arg("--log-level")
        .arg(log_level)
        .arg("daemon")
        .arg("start")
        .arg("--config")
        .arg(&config_path)
        .stdin(Stdio::null())
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(stderr));

    child.spawn().with_context(|| {
        format!(
            "Failed to spawn background daemon with config '{}'",
            config_path.display()
        )
    })?;
    Ok(())
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path.to_path_buf());
    }
    Ok(std::env::current_dir()
        .with_context(|| "Failed to resolve current working directory")?
        .join(path))
}

fn is_daemon_offline_error(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| cause.is::<std::io::Error>())
        || err.to_string().contains("Failed to connect to")
        || err
            .to_string()
            .contains("Daemon closed connection before response")
        || err
            .to_string()
            .contains("Daemon closed connection before subscription ack")
}

impl DaemonHealthReport {
    fn from_health(health: DaemonHealth) -> Self {
        Self {
            state: match health.state {
                DaemonHealthState::Ready => "ready".to_string(),
                DaemonHealthState::Degraded => "degraded".to_string(),
            },
            ready: health.ready,
            endpoint: health.endpoint,
            error: None,
            version: Some(health.version),
            protocol_version: Some(health.protocol_version),
            transport: Some(health.transport),
            wire_format: Some(health.wire_format),
            issue_count: health.issue_count,
            agent_count: health.agent_count,
            harness_count: health.harness_count,
            channel_count: health.channel_count,
            running_agent_count: health.running_agent_count,
            active_task_count: health.active_task_count,
            queued_task_count: health.queued_task_count,
            awaiting_result_count: health.awaiting_result_count,
            channel_runtime_count: health.channel_runtime_count,
            failed_channel_count: health.failed_channel_count,
        }
    }

    fn offline(endpoint: String, error: String) -> Self {
        Self {
            state: "offline".to_string(),
            ready: false,
            endpoint,
            error: Some(error),
            version: None,
            protocol_version: None,
            transport: None,
            wire_format: None,
            issue_count: 0,
            agent_count: 0,
            harness_count: 0,
            channel_count: 0,
            running_agent_count: 0,
            active_task_count: 0,
            queued_task_count: 0,
            awaiting_result_count: 0,
            channel_runtime_count: 0,
            failed_channel_count: 0,
        }
    }
}
