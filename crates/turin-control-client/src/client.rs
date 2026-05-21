use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;
use turin_daemon_client::DaemonClient;
use turin_daemon_protocol::{
    DaemonHandshake, DaemonRequest, EventEnvelope, NoParams, RequestEnvelope, ResponseEnvelope,
    RuntimeEventsSubscribeParams,
};
use turin_remote_client::RemoteClient;

use crate::health::{ControlHealth, build_health};
use crate::models::DaemonStatus;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ConnectionSpec {
    LocalConfig {
        config_path: PathBuf,
    },
    LocalEndpoint {
        endpoint: PathBuf,
    },
    Remote {
        base_url: String,
        auth_token: String,
    },
    RemoteEnv {
        base_url: String,
        auth_token_env: String,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConnectionKind {
    Local,
    Remote,
}

#[derive(Debug, Clone, Copy)]
pub struct ManagedSubscribeOptions {
    pub initial_backoff: Duration,
    pub max_backoff: Duration,
}

impl Default for ManagedSubscribeOptions {
    fn default() -> Self {
        Self {
            initial_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(1),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ControlClient {
    Local(DaemonClient),
    Remote(RemoteClient),
}

pub enum ManagedEventStream {
    Local(turin_daemon_client::ManagedEventStream),
    Remote(turin_remote_client::ManagedRemoteEventStream),
}

impl ConnectionSpec {
    pub async fn from_local_config(config_path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self::LocalConfig {
            config_path: config_path.as_ref().to_path_buf(),
        })
    }
}

impl ControlClient {
    pub async fn connect(spec: &ConnectionSpec) -> Result<Self> {
        match spec {
            ConnectionSpec::LocalConfig { config_path } => Ok(Self::Local(
                DaemonClient::from_config(config_path)
                    .await
                    .with_context(|| {
                        format!(
                            "Failed to resolve daemon endpoint from '{}'",
                            config_path.display()
                        )
                    })?,
            )),
            ConnectionSpec::LocalEndpoint { endpoint } => {
                Ok(Self::Local(DaemonClient::new(endpoint.clone())))
            }
            ConnectionSpec::Remote {
                base_url,
                auth_token,
            } => Ok(Self::Remote(RemoteClient::new(
                base_url.clone(),
                auth_token.clone(),
            ))),
            ConnectionSpec::RemoteEnv {
                base_url,
                auth_token_env,
            } => {
                let auth_token = std::env::var(auth_token_env).with_context(|| {
                    format!("Remote auth token env var '{}' is not set", auth_token_env)
                })?;
                Ok(Self::Remote(RemoteClient::new(
                    base_url.clone(),
                    auth_token,
                )))
            }
        }
    }

    pub fn kind(&self) -> ConnectionKind {
        match self {
            Self::Local(_) => ConnectionKind::Local,
            Self::Remote(_) => ConnectionKind::Remote,
        }
    }

    pub fn target(&self) -> String {
        match self {
            Self::Local(client) => client.endpoint().display().to_string(),
            Self::Remote(client) => client.base_url().to_string(),
        }
    }

    pub async fn send(&self, request: RequestEnvelope) -> Result<ResponseEnvelope> {
        match self {
            Self::Local(client) => client.send(request).await,
            Self::Remote(client) => client.send(request).await,
        }
    }

    pub async fn request(
        &self,
        id: Option<String>,
        request: DaemonRequest,
    ) -> Result<ResponseEnvelope> {
        self.send(RequestEnvelope::new(id, request)).await
    }

    pub async fn request_ok<T: for<'de> Deserialize<'de>>(
        &self,
        id: Option<String>,
        request: DaemonRequest,
    ) -> Result<T> {
        match self {
            Self::Local(client) => client.request_ok(id, request).await,
            Self::Remote(client) => client.request_ok(id, request).await,
        }
    }

    pub async fn handshake(&self) -> Result<DaemonHandshake> {
        match self {
            Self::Local(client) => client.handshake().await,
            Self::Remote(client) => client.handshake().await,
        }
    }

    pub async fn status(&self) -> Result<DaemonStatus> {
        self.request_ok(None, DaemonRequest::DaemonStatus(NoParams::default()))
            .await
    }

    pub async fn health(&self) -> Result<ControlHealth> {
        let (health, _) = self.health_and_status().await?;
        Ok(health)
    }

    pub async fn health_and_status(&self) -> Result<(ControlHealth, DaemonStatus)> {
        let target = self.target();
        let connection_kind = self.kind();
        let handshake = self.handshake().await?;
        let status = self.status().await?;
        let health = build_health(connection_kind, target, handshake, &status);
        Ok((health, status))
    }

    pub async fn subscribe_managed(
        &self,
        filter: RuntimeEventsSubscribeParams,
    ) -> Result<ManagedEventStream> {
        self.subscribe_managed_with_options(filter, ManagedSubscribeOptions::default())
            .await
    }

    pub async fn subscribe_managed_with_options(
        &self,
        filter: RuntimeEventsSubscribeParams,
        options: ManagedSubscribeOptions,
    ) -> Result<ManagedEventStream> {
        match self {
            Self::Local(client) => Ok(ManagedEventStream::Local(
                client
                    .subscribe_managed_with_options(
                        filter,
                        turin_daemon_client::ManagedSubscribeOptions {
                            initial_backoff: options.initial_backoff,
                            max_backoff: options.max_backoff,
                        },
                    )
                    .await?,
            )),
            Self::Remote(client) => Ok(ManagedEventStream::Remote(
                client
                    .subscribe_managed_with_options(
                        filter,
                        turin_remote_client::ManagedSubscribeOptions {
                            initial_backoff: options.initial_backoff,
                            max_backoff: options.max_backoff,
                        },
                    )
                    .await?,
            )),
        }
    }
}

impl ManagedEventStream {
    pub async fn next_event(&mut self) -> Result<EventEnvelope> {
        match self {
            Self::Local(stream) => stream.next_event().await,
            Self::Remote(stream) => stream.next_event().await,
        }
    }
}
