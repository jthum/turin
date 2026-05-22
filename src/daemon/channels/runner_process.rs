use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::io::AsyncReadExt;
use tokio::process::Command;

use super::{
    ChannelLifecycle, ChannelRuntimeManager, DesiredChannel, RuntimeHandle, access_state_path,
    binding_state_path,
};
use crate::daemon::channel_runners;

impl ChannelRuntimeManager {
    pub(super) async fn start_channel(&self, workspace_root: PathBuf, channel: DesiredChannel) {
        if channel.kind == "fs" {
            self.start_fs_channel(workspace_root, channel).await;
        } else {
            self.start_external_channel(workspace_root, channel).await;
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
                        channel.kind, runner_command.display
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
