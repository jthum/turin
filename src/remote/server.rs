use anyhow::{Context, Result};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::{info, warn};
use turin_daemon_client::DaemonClient;

use super::config::{BindExposure, RemoteServeOptions, ResolvedRemoteConfig, bind_exposure};
use super::routes::{RemoteState, handle_http};

#[derive(Debug)]
pub struct RunningRemoteServer {
    local_addr: SocketAddr,
    shutdown_tx: watch::Sender<bool>,
    join: JoinHandle<Result<()>>,
}

impl RunningRemoteServer {
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    pub async fn stop(self) -> Result<()> {
        let _ = self.shutdown_tx.send(true);
        self.join
            .await
            .context("remote server join failed during shutdown")?
    }

    pub async fn wait(self) -> Result<()> {
        self.join.await.context("remote server join failed")?
    }
}

pub async fn start(config_path: &Path, options: RemoteServeOptions) -> Result<RunningRemoteServer> {
    let resolved = ResolvedRemoteConfig::from_config(config_path, options)?;
    let listener = TcpListener::bind(&resolved.bind)
        .await
        .with_context(|| format!("Failed to bind turin-remote to '{}'", resolved.bind))?;
    let local_addr = listener
        .local_addr()
        .context("Failed to resolve turin-remote local bind address")?;
    assert_bind_policy(local_addr, resolved.allow_non_loopback)?;

    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);
    let state = Arc::new(RemoteState {
        bind: local_addr.to_string(),
        daemon_endpoint: resolved.daemon_endpoint.display().to_string(),
        auth_token: Arc::<str>::from(resolved.auth_token),
        event_keepalive: resolved.event_keepalive,
        client: DaemonClient::new(resolved.daemon_endpoint.clone()),
    });

    info!(
        bind = %local_addr,
        daemon_endpoint = %resolved.daemon_endpoint.display(),
        auth_token_env = %resolved.auth_token_env,
        allow_non_loopback = resolved.allow_non_loopback,
        "turin-remote listening"
    );

    let join = tokio::spawn(async move {
        loop {
            tokio::select! {
                changed = shutdown_rx.changed() => {
                    if changed.is_ok() && *shutdown_rx.borrow() {
                        break;
                    }
                }
                accepted = listener.accept() => {
                    let (stream, peer_addr) = accepted.context("Failed to accept turin-remote TCP connection")?;
                    let state = Arc::clone(&state);
                    tokio::spawn(async move {
                        let io = TokioIo::new(stream);
                        let service = service_fn(move |req| handle_http(req, Arc::clone(&state)));
                        if let Err(err) = http1::Builder::new()
                            .serve_connection(io, service)
                            .with_upgrades()
                            .await
                        {
                            warn!(peer = %peer_addr, error = %err, "turin-remote HTTP connection failed");
                        }
                    });
                }
            }
        }
        Ok(())
    });

    Ok(RunningRemoteServer {
        local_addr,
        shutdown_tx,
        join,
    })
}

pub async fn serve(config_path: &Path, options: RemoteServeOptions) -> Result<()> {
    let server = start(config_path, options).await?;
    tokio::signal::ctrl_c()
        .await
        .context("Failed to wait for turin-remote shutdown signal")?;
    server.stop().await
}

fn assert_bind_policy(local_addr: SocketAddr, allow_non_loopback: bool) -> Result<()> {
    match bind_exposure(local_addr) {
        BindExposure::Loopback => Ok(()),
        BindExposure::NonLoopback if allow_non_loopback => {
            warn!(
                bind = %local_addr,
                "turin-remote is listening on a non-loopback interface; deploy behind TLS or a trusted reverse proxy"
            );
            Ok(())
        }
        BindExposure::NonLoopback => anyhow::bail!(
            "Refusing to bind turin-remote to non-loopback address '{}' without explicit opt-in. Set [remote].allow_non_loopback = true or pass --allow-non-loopback.",
            local_addr
        ),
    }
}
