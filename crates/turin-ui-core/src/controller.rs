use anyhow::{Result, anyhow};
use std::path::PathBuf;
use std::time::Duration;
use tokio::runtime::Handle;
use tokio::sync::mpsc;
use tokio::time;
use turin_control_client::{ConnectionSpec, ControlClient};
use turin_daemon_protocol::{EventEnvelope, RuntimeEventsSubscribeParams};

use crate::{DashboardSnapshot, DashboardState};

pub const DEFAULT_REFRESH_INTERVAL: Duration = Duration::from_secs(5);

#[derive(Debug, Clone)]
pub struct ConnectionOptions {
    pub config_path: PathBuf,
    pub endpoint: Option<PathBuf>,
    pub remote_url: Option<String>,
    pub auth_token: Option<String>,
    pub auth_token_env: Option<String>,
}

#[derive(Debug)]
pub struct UiController {
    pub update_rx: mpsc::UnboundedReceiver<UiUpdate>,
    pub command_tx: mpsc::UnboundedSender<OperatorCommand>,
}

#[derive(Debug)]
pub enum UiUpdate {
    Snapshot(Box<DashboardSnapshot>),
    Event(EventEnvelope),
    Error(String),
    Info(String),
}

#[derive(Debug, Clone)]
pub enum OperatorCommand {
    Refresh,
    OpenSession { agent_id: String },
    ResumeSession { session_id: String },
    SubmitPrompt { session_id: String, prompt: String },
    CancelSession { session_id: String },
    KillSession { session_id: String },
    CancelTask { request_id: String },
}

impl ConnectionOptions {
    pub fn to_spec(&self) -> Result<ConnectionSpec> {
        if let Some(base_url) = &self.remote_url {
            if let Some(auth_token) = &self.auth_token {
                return Ok(ConnectionSpec::Remote {
                    base_url: base_url.clone(),
                    auth_token: auth_token.clone(),
                });
            }
            if let Some(auth_token_env) = &self.auth_token_env {
                return Ok(ConnectionSpec::RemoteEnv {
                    base_url: base_url.clone(),
                    auth_token_env: auth_token_env.clone(),
                });
            }
            return Err(anyhow!(
                "--remote-url requires either --auth-token or --auth-token-env"
            ));
        }

        if let Some(endpoint) = &self.endpoint {
            return Ok(ConnectionSpec::LocalEndpoint {
                endpoint: endpoint.clone(),
            });
        }

        Ok(ConnectionSpec::LocalConfig {
            config_path: self.config_path.clone(),
        })
    }
}

pub async fn connect_dashboard(spec: &ConnectionSpec) -> Result<(ControlClient, DashboardState)> {
    let client = ControlClient::connect(spec).await?;
    let dashboard = DashboardState::load(&client).await?;
    Ok((client, dashboard))
}

pub fn spawn_controller(handle: &Handle, client: ControlClient) -> UiController {
    spawn_controller_with_interval(handle, client, DEFAULT_REFRESH_INTERVAL)
}

pub fn spawn_controller_with_interval(
    handle: &Handle,
    client: ControlClient,
    refresh_interval: Duration,
) -> UiController {
    let (update_tx, update_rx) = mpsc::unbounded_channel::<UiUpdate>();
    let (command_tx, command_rx) = mpsc::unbounded_channel::<OperatorCommand>();

    spawn_event_task(handle, client.clone(), update_tx.clone());
    spawn_refresh_task(handle, client.clone(), update_tx.clone(), refresh_interval);
    spawn_command_task(handle, client, command_rx, update_tx);

    UiController {
        update_rx,
        command_tx,
    }
}

fn spawn_event_task(handle: &Handle, client: ControlClient, tx: mpsc::UnboundedSender<UiUpdate>) {
    handle.spawn(async move {
        match client
            .subscribe_managed(RuntimeEventsSubscribeParams::default())
            .await
        {
            Ok(mut stream) => loop {
                match stream.next_event().await {
                    Ok(event) => {
                        let _ = tx.send(UiUpdate::Event(event));
                    }
                    Err(err) => {
                        let _ = tx.send(UiUpdate::Error(err.to_string()));
                        break;
                    }
                }
            },
            Err(err) => {
                let _ = tx.send(UiUpdate::Error(err.to_string()));
            }
        }
    });
}

fn spawn_refresh_task(
    handle: &Handle,
    client: ControlClient,
    tx: mpsc::UnboundedSender<UiUpdate>,
    refresh_interval: Duration,
) {
    handle.spawn(async move {
        let mut interval = time::interval(refresh_interval);
        loop {
            interval.tick().await;
            match DashboardState::snapshot(&client).await {
                Ok(snapshot) => {
                    let _ = tx.send(UiUpdate::Snapshot(Box::new(snapshot)));
                }
                Err(err) => {
                    let _ = tx.send(UiUpdate::Error(err.to_string()));
                }
            }
        }
    });
}

fn spawn_command_task(
    handle: &Handle,
    client: ControlClient,
    mut command_rx: mpsc::UnboundedReceiver<OperatorCommand>,
    tx: mpsc::UnboundedSender<UiUpdate>,
) {
    handle.spawn(async move {
        while let Some(command) = command_rx.recv().await {
            match execute_operator_command(&client, command).await {
                Ok(message) => {
                    let _ = tx.send(UiUpdate::Info(message));
                    match DashboardState::snapshot(&client).await {
                        Ok(snapshot) => {
                            let _ = tx.send(UiUpdate::Snapshot(Box::new(snapshot)));
                        }
                        Err(err) => {
                            let _ = tx.send(UiUpdate::Error(err.to_string()));
                        }
                    }
                }
                Err(err) => {
                    let _ = tx.send(UiUpdate::Error(err.to_string()));
                }
            }
        }
    });
}

pub async fn execute_operator_command(
    client: &ControlClient,
    command: OperatorCommand,
) -> Result<String> {
    match command {
        OperatorCommand::Refresh => Ok("Refreshed Turin state".to_string()),
        OperatorCommand::OpenSession { agent_id } => {
            let session = client.open_session(&agent_id, None).await?;
            Ok(format!(
                "Opened live session {} for agent {}",
                session.session_id, session.agent_id
            ))
        }
        OperatorCommand::ResumeSession { session_id } => {
            let session = client.resume_session(&session_id, None).await?;
            Ok(format!(
                "Resumed session {} into live slot {}",
                session.session_id, session.slot_id
            ))
        }
        OperatorCommand::SubmitPrompt { session_id, prompt } => {
            let task = client
                .submit_task(None, Some(session_id.clone()), prompt)
                .await?;
            Ok(format!(
                "Submitted task {} to session {}",
                task.request_id, session_id
            ))
        }
        OperatorCommand::CancelSession { session_id } => {
            let result = client.cancel_session(&session_id).await?;
            Ok(format!(
                "Requested cancel for session {} ({})",
                result.session_id, result.agent_id
            ))
        }
        OperatorCommand::KillSession { session_id } => {
            let result = client.kill_session(&session_id).await?;
            Ok(format!(
                "Killed session {} ({})",
                result.session_id, result.agent_id
            ))
        }
        OperatorCommand::CancelTask { request_id } => {
            let task = client.cancel_task(&request_id).await?;
            Ok(format!(
                "Cancelled task {} -> {}",
                task.request_id, task.state
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connection_options_default_to_local_config() {
        let options = ConnectionOptions {
            config_path: PathBuf::from("turin.toml"),
            endpoint: None,
            remote_url: None,
            auth_token: None,
            auth_token_env: None,
        };

        match options.to_spec().expect("spec") {
            ConnectionSpec::LocalConfig { config_path } => {
                assert_eq!(config_path, PathBuf::from("turin.toml"));
            }
            other => panic!("unexpected spec: {other:?}"),
        }
    }

    #[test]
    fn connection_options_require_remote_auth_material() {
        let options = ConnectionOptions {
            config_path: PathBuf::from("turin.toml"),
            endpoint: None,
            remote_url: Some("http://example.test".to_string()),
            auth_token: None,
            auth_token_env: None,
        };

        let err = options.to_spec().expect_err("missing auth should error");
        assert!(
            err.to_string()
                .contains("--remote-url requires either --auth-token or --auth-token-env")
        );
    }
}
