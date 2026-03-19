use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use tokio::runtime::Handle;
use tokio::sync::{mpsc, watch};
use tokio::time;
use turin_control_client::{ConnectionSpec, ControlClient, SessionDetail};
use turin_daemon_protocol::{EventEnvelope, RuntimeEventsSubscribeParams};

use crate::{DashboardSnapshot, DashboardState};

pub const DEFAULT_REFRESH_INTERVAL: Duration = Duration::from_secs(5);

#[derive(Debug, Clone)]
pub struct ConnectionOptions {
    pub config_path: Option<PathBuf>,
    pub endpoint: Option<PathBuf>,
    pub remote_url: Option<String>,
    pub auth_token: Option<String>,
    pub auth_token_env: Option<String>,
    pub profile: Option<String>,
    pub profiles_file: Option<PathBuf>,
}

#[derive(Debug)]
pub struct UiController {
    pub update_rx: mpsc::UnboundedReceiver<UiUpdate>,
    pub command_tx: mpsc::UnboundedSender<OperatorCommand>,
    shutdown_tx: watch::Sender<bool>,
}

#[derive(Debug)]
pub enum UiUpdate {
    Snapshot(Box<DashboardSnapshot>),
    SessionDetail(Box<SessionDetail>),
    Event(EventEnvelope),
    Error(String),
    Info(String),
}

#[derive(Debug, Clone)]
pub enum OperatorCommand {
    Refresh,
    LoadSessionDetail { session_id: String },
    OpenSession { agent_id: String },
    ResumeSession { session_id: String },
    SubmitPrompt { session_id: String, prompt: String },
    CancelSession { session_id: String },
    KillSession { session_id: String },
    CancelTask { request_id: String },
}

#[derive(Debug, Clone)]
pub struct ConnectionProfileCatalog {
    source_path: PathBuf,
    default_profile: Option<String>,
    profiles: Vec<ConnectionProfileSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectionProfileSummary {
    pub name: String,
    pub kind: ConnectionProfileKind,
    pub target: String,
    pub auth: Option<ConnectionProfileAuth>,
    pub is_default: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionProfileKind {
    LocalConfig,
    LocalEndpoint,
    Remote,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConnectionProfileAuth {
    TokenEnv(String),
    InlineToken,
}

impl ConnectionOptions {
    pub fn to_spec(&self) -> Result<ConnectionSpec> {
        let resolved = self.resolve_profile()?;

        if let Some(base_url) = &resolved.remote_url {
            if let Some(auth_token) = &resolved.auth_token {
                return Ok(ConnectionSpec::Remote {
                    base_url: base_url.clone(),
                    auth_token: auth_token.clone(),
                });
            }
            if let Some(auth_token_env) = &resolved.auth_token_env {
                return Ok(ConnectionSpec::RemoteEnv {
                    base_url: base_url.clone(),
                    auth_token_env: auth_token_env.clone(),
                });
            }
            return Err(anyhow!(
                "--remote-url requires either --auth-token or --auth-token-env"
            ));
        }

        if let Some(endpoint) = &resolved.endpoint {
            return Ok(ConnectionSpec::LocalEndpoint {
                endpoint: endpoint.clone(),
            });
        }

        Ok(ConnectionSpec::LocalConfig {
            config_path: resolved
                .config_path
                .unwrap_or_else(|| PathBuf::from("turin.toml")),
        })
    }

    pub fn resolved_profile_name(&self) -> Result<Option<String>> {
        if self.profile.is_none() && self.profiles_file.is_none() {
            return Ok(None);
        }

        let profiles_path = self.profiles_path();
        let profiles = ConnectionProfiles::load(&profiles_path)?;
        Ok(self
            .profile
            .clone()
            .or_else(|| profiles.default_profile.clone()))
    }

    pub fn load_profiles(&self) -> Result<Option<ConnectionProfileCatalog>> {
        let profiles_path = self.profiles_path();
        if !profiles_path.exists() {
            if self.profile.is_some() || self.profiles_file.is_some() {
                let _ = ConnectionProfiles::load(&profiles_path)?;
            }
            return Ok(None);
        }

        let profiles = ConnectionProfiles::load(&profiles_path)?;
        Ok(Some(ConnectionProfileCatalog::from_raw(
            profiles_path,
            profiles,
        )))
    }

    fn resolve_profile(&self) -> Result<Self> {
        if self.profile.is_none() && self.profiles_file.is_none() {
            return Ok(self.clone());
        }

        let profiles_path = self.profiles_path();
        let profiles = ConnectionProfiles::load(&profiles_path)?;
        let Some(profile_name) = self
            .profile
            .clone()
            .or_else(|| profiles.default_profile.clone())
        else {
            return Ok(self.clone());
        };
        let profile = profiles.profiles.get(&profile_name).with_context(|| {
            format!(
                "Connection profile '{}' was not found in '{}'",
                profile_name,
                profiles_path.display()
            )
        })?;

        Ok(Self {
            config_path: self
                .config_path
                .clone()
                .or_else(|| profile.config_path.clone()),
            endpoint: self.endpoint.clone().or_else(|| profile.endpoint.clone()),
            remote_url: self
                .remote_url
                .clone()
                .or_else(|| profile.remote_url.clone()),
            auth_token: self
                .auth_token
                .clone()
                .or_else(|| profile.auth_token.clone()),
            auth_token_env: self
                .auth_token_env
                .clone()
                .or_else(|| profile.auth_token_env.clone()),
            profile: Some(profile_name),
            profiles_file: Some(profiles_path),
        })
    }

    pub fn profiles_path(&self) -> PathBuf {
        self.profiles_file
            .clone()
            .unwrap_or_else(|| PathBuf::from("ui-profiles.toml"))
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
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    spawn_event_task(
        handle,
        client.clone(),
        update_tx.clone(),
        shutdown_rx.clone(),
    );
    spawn_refresh_task(
        handle,
        client.clone(),
        update_tx.clone(),
        refresh_interval,
        shutdown_rx.clone(),
    );
    spawn_command_task(handle, client, command_rx, update_tx, shutdown_rx);

    UiController {
        update_rx,
        command_tx,
        shutdown_tx,
    }
}

impl UiController {
    pub fn shutdown(&self) {
        let _ = self.shutdown_tx.send(true);
    }
}

impl ConnectionProfileCatalog {
    pub fn source_path(&self) -> &Path {
        &self.source_path
    }

    pub fn default_profile(&self) -> Option<&str> {
        self.default_profile.as_deref()
    }

    pub fn profiles(&self) -> &[ConnectionProfileSummary] {
        &self.profiles
    }

    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    pub fn connection_options(&self, name: &str) -> Option<ConnectionOptions> {
        self.profiles
            .iter()
            .any(|profile| profile.name == name)
            .then(|| ConnectionOptions {
                config_path: None,
                endpoint: None,
                remote_url: None,
                auth_token: None,
                auth_token_env: None,
                profile: Some(name.to_string()),
                profiles_file: Some(self.source_path.clone()),
            })
    }

    fn from_raw(source_path: PathBuf, raw: ConnectionProfiles) -> Self {
        let default_profile = raw.default_profile.clone();
        let profiles = raw
            .profiles
            .into_iter()
            .map(|(name, profile)| ConnectionProfileSummary {
                is_default: default_profile.as_deref() == Some(name.as_str()),
                name,
                kind: profile.kind(),
                target: profile.target(),
                auth: profile.auth_label(),
            })
            .collect();

        Self {
            source_path,
            default_profile,
            profiles,
        }
    }
}

fn spawn_event_task(
    handle: &Handle,
    client: ControlClient,
    tx: mpsc::UnboundedSender<UiUpdate>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    handle.spawn(async move {
        match client
            .subscribe_managed(RuntimeEventsSubscribeParams::default())
            .await
        {
            Ok(mut stream) => loop {
                tokio::select! {
                    changed = shutdown_rx.changed() => {
                        if changed.is_ok() && *shutdown_rx.borrow() {
                            break;
                        }
                    }
                    next = stream.next_event() => match next {
                    Ok(event) => {
                        if tx.send(UiUpdate::Event(event)).is_err() {
                            break;
                        }
                    }
                    Err(err) => {
                        if tx.send(UiUpdate::Error(err.to_string())).is_err() {
                            break;
                        }
                        break;
                    }
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
    mut shutdown_rx: watch::Receiver<bool>,
) {
    handle.spawn(async move {
        let mut interval = time::interval(refresh_interval);
        loop {
            tokio::select! {
                changed = shutdown_rx.changed() => {
                    if changed.is_ok() && *shutdown_rx.borrow() {
                        break;
                    }
                }
                _ = interval.tick() => {}
            }
            match DashboardState::snapshot(&client).await {
                Ok(snapshot) => {
                    if tx.send(UiUpdate::Snapshot(Box::new(snapshot))).is_err() {
                        break;
                    }
                }
                Err(err) => {
                    if tx.send(UiUpdate::Error(err.to_string())).is_err() {
                        break;
                    }
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
    mut shutdown_rx: watch::Receiver<bool>,
) {
    handle.spawn(async move {
        loop {
            let command = tokio::select! {
                changed = shutdown_rx.changed() => {
                    if changed.is_ok() && *shutdown_rx.borrow() {
                        break;
                    }
                    continue;
                }
                maybe_command = command_rx.recv() => {
                    match maybe_command {
                        Some(command) => command,
                        None => break,
                    }
                }
            };

            if let OperatorCommand::LoadSessionDetail { session_id } = &command {
                match client.get_session(session_id.as_str()).await {
                    Ok(detail) => {
                        if tx.send(UiUpdate::SessionDetail(Box::new(detail))).is_err() {
                            break;
                        }
                    }
                    Err(err) => {
                        if tx.send(UiUpdate::Error(err.to_string())).is_err() {
                            break;
                        }
                    }
                }
                continue;
            }

            match execute_operator_command(&client, command).await {
                Ok(message) => {
                    if tx.send(UiUpdate::Info(message)).is_err() {
                        break;
                    }
                    match DashboardState::snapshot(&client).await {
                        Ok(snapshot) => {
                            if tx.send(UiUpdate::Snapshot(Box::new(snapshot))).is_err() {
                                break;
                            }
                        }
                        Err(err) => {
                            if tx.send(UiUpdate::Error(err.to_string())).is_err() {
                                break;
                            }
                        }
                    }
                }
                Err(err) => {
                    if tx.send(UiUpdate::Error(err.to_string())).is_err() {
                        break;
                    }
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
        OperatorCommand::LoadSessionDetail { .. } => Ok("Loaded session detail".to_string()),
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

#[derive(Debug, Clone, Default, Deserialize)]
struct ConnectionProfiles {
    #[allow(dead_code)]
    default_profile: Option<String>,
    #[serde(default)]
    profiles: BTreeMap<String, StoredConnectionProfile>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct StoredConnectionProfile {
    #[serde(default, alias = "config")]
    config_path: Option<PathBuf>,
    #[serde(default)]
    endpoint: Option<PathBuf>,
    #[serde(default)]
    remote_url: Option<String>,
    #[serde(default)]
    auth_token: Option<String>,
    #[serde(default)]
    auth_token_env: Option<String>,
}

impl StoredConnectionProfile {
    fn kind(&self) -> ConnectionProfileKind {
        if self.remote_url.is_some() {
            ConnectionProfileKind::Remote
        } else if self.endpoint.is_some() {
            ConnectionProfileKind::LocalEndpoint
        } else {
            ConnectionProfileKind::LocalConfig
        }
    }

    fn target(&self) -> String {
        self.remote_url
            .clone()
            .or_else(|| {
                self.endpoint
                    .as_ref()
                    .map(|path| path.display().to_string())
            })
            .or_else(|| {
                self.config_path
                    .as_ref()
                    .map(|path| path.display().to_string())
            })
            .unwrap_or_else(|| "turin.toml".to_string())
    }

    fn auth_label(&self) -> Option<ConnectionProfileAuth> {
        self.auth_token_env
            .as_ref()
            .map(|env| ConnectionProfileAuth::TokenEnv(env.clone()))
            .or_else(|| {
                self.auth_token
                    .as_ref()
                    .map(|_| ConnectionProfileAuth::InlineToken)
            })
    }
}

impl ConnectionProfiles {
    fn load(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path).with_context(|| {
            format!(
                "Failed to read connection profiles from '{}'",
                path.display()
            )
        })?;
        toml::from_str(&raw).with_context(|| {
            format!(
                "Failed to parse connection profiles TOML from '{}'",
                path.display()
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn connection_options_default_to_local_config() {
        let options = ConnectionOptions {
            config_path: None,
            endpoint: None,
            remote_url: None,
            auth_token: None,
            auth_token_env: None,
            profile: None,
            profiles_file: None,
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
            config_path: None,
            endpoint: None,
            remote_url: Some("http://example.test".to_string()),
            auth_token: None,
            auth_token_env: None,
            profile: None,
            profiles_file: None,
        };

        let err = options.to_spec().expect_err("missing auth should error");
        assert!(
            err.to_string()
                .contains("--remote-url requires either --auth-token or --auth-token-env")
        );
    }

    #[test]
    fn connection_options_apply_profile_overrides() {
        let temp = tempfile::NamedTempFile::new().expect("temp profile file");
        fs::write(
            temp.path(),
            r#"
[profiles.lab]
remote_url = "http://example.test"
auth_token_env = "TURIN_REMOTE_TOKEN"
"#,
        )
        .expect("write profile file");

        let options = ConnectionOptions {
            config_path: None,
            endpoint: None,
            remote_url: None,
            auth_token: None,
            auth_token_env: None,
            profile: Some("lab".to_string()),
            profiles_file: Some(temp.path().to_path_buf()),
        };

        match options.to_spec().expect("spec") {
            ConnectionSpec::RemoteEnv {
                base_url,
                auth_token_env,
            } => {
                assert_eq!(base_url, "http://example.test");
                assert_eq!(auth_token_env, "TURIN_REMOTE_TOKEN");
            }
            other => panic!("unexpected spec: {other:?}"),
        }
    }
}
