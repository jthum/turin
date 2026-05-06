use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use turin::remote::{RemoteServeOptions, serve};
use turin::tracing_support::init_tracing;
use turin_types::layout::DEFAULT_BOOTSTRAP_CONFIG_PATH;

#[derive(Parser, Debug)]
#[command(name = "turin-remote", version, about)]
struct Cli {
    /// Path to Turin config file
    #[arg(long, default_value = DEFAULT_BOOTSTRAP_CONFIG_PATH)]
    config: PathBuf,

    /// Override the remote bind address
    #[arg(long)]
    bind: Option<String>,

    /// Override the remote auth token directly
    #[arg(long)]
    auth_token: Option<String>,

    /// Override the remote auth token env var name
    #[arg(long)]
    auth_token_env: Option<String>,

    /// Override the SSE/WebSocket keepalive interval in seconds
    #[arg(long, alias = "event-keepalive-secs")]
    event_keepalive_seconds: Option<u64>,

    /// Allow binding turin-remote to a non-loopback address
    #[arg(long)]
    allow_non_loopback: bool,

    /// Log level (error, warn, info, debug, trace)
    #[arg(long, default_value = "info")]
    log_level: String,

    /// Path to log file
    #[arg(long)]
    log_file: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(&cli.log_level, cli.log_file)?;
    serve(
        &cli.config,
        RemoteServeOptions {
            bind: cli.bind,
            auth_token: cli.auth_token,
            auth_token_env: cli.auth_token_env,
            event_keepalive_seconds: cli.event_keepalive_seconds,
            allow_non_loopback: Some(cli.allow_non_loopback),
        },
    )
    .await
}
