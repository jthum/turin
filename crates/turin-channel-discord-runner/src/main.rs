use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde_json::Value;
use tokio::sync::watch;
use tracing_subscriber::EnvFilter;
use turin_channel_discord::DiscordChannelDriver;
use turin_channel_runner::{ChannelAccessPolicy, ChannelRunner, RunnerConfig};
use turin_daemon_client::DaemonClient;

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Run(RunArgs),
    ValidateSettings(ValidateSettingsArgs),
}

#[derive(Parser)]
struct RunArgs {
    #[arg(long)]
    channel_id: String,
    #[arg(long)]
    agent_id: String,
    #[arg(long)]
    daemon_endpoint: PathBuf,
    #[arg(long)]
    bindings_path: PathBuf,
    #[arg(long)]
    access_state_path: PathBuf,
    #[arg(long)]
    idle_ttl_secs: Option<u64>,
    #[arg(long)]
    settings_json: String,
}

#[derive(Parser)]
struct ValidateSettingsArgs {
    #[arg(long)]
    settings_json: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => run(args).await,
        Command::ValidateSettings(args) => validate_settings(args),
    }
}

async fn run(args: RunArgs) -> Result<()> {
    let settings = parse_settings_json(&args.settings_json)?;
    let access_policy = ChannelAccessPolicy::from_settings(&settings)?;
    let tools = turin_channel_runner::tools_config_from_settings(&settings)?;
    let task_timeout_ms = turin_channel_runner::task_timeout_ms_from_settings(&settings)?;

    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    tokio::spawn(async move {
        wait_for_shutdown_signal().await;
        let _ = shutdown_tx.send(true);
    });

    let daemon = DaemonClient::new(args.daemon_endpoint);
    let runner = ChannelRunner::new(
        daemon,
        RunnerConfig {
            state_path: args.bindings_path,
            access_state_path: args.access_state_path,
            idle_ttl: args.idle_ttl_secs.map(Duration::from_secs),
            access_policy,
            tools,
        },
    );

    let mut driver = DiscordChannelDriver::from_settings(&args.channel_id, &settings, shutdown_rx)
        .await
        .with_context(|| {
            format!(
                "Failed to initialize discord channel driver '{}'",
                args.channel_id
            )
        })?;

    runner
        .run_driver(&args.agent_id, &mut driver, task_timeout_ms)
        .await
        .with_context(|| format!("Channel '{}' runner failed", args.channel_id))
}

fn validate_settings(args: ValidateSettingsArgs) -> Result<()> {
    let settings = parse_settings_json(&args.settings_json)?;
    turin_channel_discord::validate_settings(&settings)
}

fn parse_settings_json(raw: &str) -> Result<Value> {
    let value: Value =
        serde_json::from_str(raw).context("Failed to parse channel settings JSON")?;
    if !value.is_object() {
        anyhow::bail!("Channel settings must be a JSON object");
    }
    Ok(value)
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(true)
        .try_init();
}

async fn wait_for_shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};

        let mut terminate = signal(SignalKind::terminate()).ok();
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = async {
                if let Some(signal) = terminate.as_mut() {
                    signal.recv().await;
                } else {
                    std::future::pending::<()>().await;
                }
            } => {}
        }
    }

    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}
