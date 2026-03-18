use anyhow::{Context, Result};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::kernel::config::TurinConfig;

#[derive(Debug, Clone, Default)]
pub struct RemoteServeOptions {
    pub bind: Option<String>,
    pub auth_token: Option<String>,
    pub auth_token_env: Option<String>,
    pub event_keepalive_secs: Option<u64>,
    pub allow_non_loopback: Option<bool>,
}

#[derive(Debug, Clone)]
pub(crate) struct ResolvedRemoteConfig {
    pub(crate) bind: String,
    pub(crate) daemon_endpoint: PathBuf,
    pub(crate) auth_token: String,
    pub(crate) auth_token_env: String,
    pub(crate) event_keepalive: Duration,
    pub(crate) allow_non_loopback: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BindExposure {
    Loopback,
    NonLoopback,
}

impl ResolvedRemoteConfig {
    pub(crate) fn from_config(config_path: &Path, options: RemoteServeOptions) -> Result<Self> {
        let config = TurinConfig::from_file(config_path)?;
        let config_base = config_path.parent().unwrap_or_else(|| Path::new("."));
        let daemon_endpoint = config.resolve_daemon_endpoint(config_base);
        let bind = options.bind.unwrap_or_else(|| config.remote.bind.clone());
        let auth_token_env = options
            .auth_token_env
            .unwrap_or_else(|| config.remote.auth_token_env.clone());
        let auth_token = match options.auth_token {
            Some(token) => token,
            None => std::env::var(&auth_token_env).with_context(|| {
                format!(
                    "Remote auth token env var '{}' is not set for turin-remote",
                    auth_token_env
                )
            })?,
        };
        if auth_token.trim().is_empty() {
            anyhow::bail!("Remote auth token must not be empty");
        }

        let keepalive_secs = options
            .event_keepalive_secs
            .unwrap_or(config.remote.event_keepalive_secs);
        anyhow::ensure!(
            keepalive_secs > 0,
            "Remote event keepalive must be greater than 0"
        );

        Ok(Self {
            bind,
            daemon_endpoint,
            auth_token,
            auth_token_env,
            event_keepalive: Duration::from_secs(keepalive_secs),
            allow_non_loopback: options
                .allow_non_loopback
                .unwrap_or(config.remote.allow_non_loopback),
        })
    }
}

pub(crate) fn bind_exposure(local_addr: SocketAddr) -> BindExposure {
    if local_addr.ip().is_loopback() {
        BindExposure::Loopback
    } else {
        BindExposure::NonLoopback
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    #[test]
    fn bind_exposure_detects_loopback() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 9324);
        assert_eq!(bind_exposure(addr), BindExposure::Loopback);
    }

    #[test]
    fn bind_exposure_detects_non_loopback() {
        let addr = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 9324);
        assert_eq!(bind_exposure(addr), BindExposure::NonLoopback);
    }
}
