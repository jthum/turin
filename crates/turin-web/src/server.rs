use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::{info, warn};
use turin_client::{Client, ConnectionSpec};

use crate::routes::{WebState, handle_http};

pub const DEFAULT_WEB_BIND: &str = "127.0.0.1:9330";

#[derive(Debug, Clone)]
pub struct WebServeOptions {
    pub bind: String,
    pub assets_dir: PathBuf,
    pub connection: ConnectionSpec,
    pub allow_non_loopback: bool,
}

#[derive(Debug)]
pub struct RunningWebServer {
    local_addr: SocketAddr,
    shutdown_tx: watch::Sender<bool>,
    join: JoinHandle<Result<()>>,
}

impl RunningWebServer {
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    pub async fn stop(self) -> Result<()> {
        let _ = self.shutdown_tx.send(true);
        self.join
            .await
            .context("turin-web server join failed during shutdown")?
    }

    pub async fn wait(self) -> Result<()> {
        self.join.await.context("turin-web server join failed")?
    }
}

pub async fn start(options: WebServeOptions) -> Result<RunningWebServer> {
    let client = Client::connect(&options.connection)
        .await
        .context("Failed to connect turin-web to the Turin control endpoint")?;
    let listener = TcpListener::bind(&options.bind)
        .await
        .with_context(|| format!("Failed to bind turin-web to '{}'", options.bind))?;
    let local_addr = listener
        .local_addr()
        .context("Failed to resolve turin-web local bind address")?;
    assert_bind_policy(local_addr, options.allow_non_loopback)?;

    let (shutdown_tx, mut shutdown_rx) = watch::channel(false);
    let state = Arc::new(WebState {
        assets_dir: options.assets_dir,
        client: Arc::new(client),
    });

    info!(
        bind = %local_addr,
        connection_kind = ?state.client.kind(),
        connection_target = %state.client.target(),
        assets = %state.assets_dir.display(),
        "turin-web listening"
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
                    let (stream, peer_addr) = accepted.context("Failed to accept turin-web TCP connection")?;
                    let state = Arc::clone(&state);
                    tokio::spawn(async move {
                        let io = TokioIo::new(stream);
                        let service = service_fn(move |request| handle_http(request, Arc::clone(&state)));
                        if let Err(error) = http1::Builder::new().serve_connection(io, service).await {
                            warn!(peer = %peer_addr, %error, "turin-web HTTP connection failed");
                        }
                    });
                }
            }
        }
        Ok(())
    });

    Ok(RunningWebServer {
        local_addr,
        shutdown_tx,
        join,
    })
}

pub async fn serve(options: WebServeOptions) -> Result<()> {
    let server = start(options).await?;
    tokio::signal::ctrl_c()
        .await
        .context("Failed to wait for turin-web shutdown signal")?;
    server.stop().await
}

fn assert_bind_policy(local_addr: SocketAddr, allow_non_loopback: bool) -> Result<()> {
    if local_addr.ip().is_loopback() {
        return Ok(());
    }
    if allow_non_loopback {
        warn!(
            bind = %local_addr,
            "turin-web has no built-in user authentication yet; deploy behind a trusted authenticated boundary"
        );
        return Ok(());
    }

    anyhow::bail!(
        "Refusing to bind turin-web to non-loopback address '{}'. Pass --allow-non-loopback only behind a trusted authenticated boundary.",
        local_addr
    )
}

#[cfg(test)]
mod tests {
    use std::net::{IpAddr, Ipv4Addr};

    use super::*;

    #[test]
    fn bind_policy_is_local_by_default() {
        let local = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 9330);
        let public = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 9330);

        assert!(assert_bind_policy(local, false).is_ok());
        assert!(assert_bind_policy(public, false).is_err());
        assert!(assert_bind_policy(public, true).is_ok());
    }
}
