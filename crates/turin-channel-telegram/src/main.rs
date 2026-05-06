use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde_json::Value;
use tokio::sync::watch;
use tracing_subscriber::EnvFilter;
use turin_channel_core::{ChannelAuthFlowPollRequest, ChannelAuthFlowStartRequest};
use turin_channel_runner::{
    ChannelAccessPolicy, ChannelRunner, RunnerConfig, RunnerPresence, announce_runner_presence,
    spawn_runner_heartbeat,
};
use turin_channel_telegram::TelegramChannelDriver;
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
    Describe,
    ValidateSettings(ValidateSettingsArgs),
    SetupAuthFlowStart(AuthFlowRequestArgs),
    SetupAuthFlowPoll(AuthFlowRequestArgs),
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
    #[arg(long, alias = "idle-ttl-secs")]
    idle_timeout_seconds: Option<u64>,
    #[arg(long)]
    settings_json: String,
}

#[derive(Parser)]
struct ValidateSettingsArgs {
    #[arg(long)]
    settings_json: String,
}

#[derive(Parser)]
struct AuthFlowRequestArgs {
    #[arg(long)]
    request_json: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_tracing();
    let cli = Cli::parse();
    match cli.command {
        Command::Run(args) => run(args).await,
        Command::Describe => describe(),
        Command::ValidateSettings(args) => validate_settings(args),
        Command::SetupAuthFlowStart(args) => setup_auth_flow_start(args),
        Command::SetupAuthFlowPoll(args) => setup_auth_flow_poll(args),
    }
}

async fn run(args: RunArgs) -> Result<()> {
    let settings = parse_settings_json(&args.settings_json)?;
    let access_policy = ChannelAccessPolicy::from_settings(&settings)?;
    let tools = turin_channel_runner::tools_config_from_settings(&settings)?;
    let task_timeout_ms = turin_channel_runner::task_timeout_ms_from_settings(&settings)?;
    let allow_unconfigured_chats = access_policy.requires_unconfigured_inbound();

    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let heartbeat_shutdown_rx = shutdown_rx.clone();
    tokio::spawn(async move {
        wait_for_shutdown_signal().await;
        let _ = shutdown_tx.send(true);
    });

    let daemon = DaemonClient::new(args.daemon_endpoint);
    let runtime_dir = args
        .bindings_path
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let runner = ChannelRunner::new(
        daemon.clone(),
        RunnerConfig {
            channel_id: args.channel_id.clone(),
            state_path: args.bindings_path,
            access_state_path: args.access_state_path,
            idle_ttl: args.idle_timeout_seconds.map(Duration::from_secs),
            access_policy,
            tools,
        },
    );

    let mut driver = TelegramChannelDriver::from_settings_with_media_dir(
        &args.channel_id,
        &settings,
        Some(runtime_dir.join("media")),
        shutdown_rx,
        allow_unconfigured_chats,
    )
    .await
    .with_context(|| {
        format!(
            "Failed to initialize telegram channel driver '{}'",
            args.channel_id
        )
    })?;

    announce_runner_presence(
        &daemon,
        &args.channel_id,
        RunnerPresence {
            manifest: turin_channel_telegram::adapter_manifest(),
            runner_binary: Some(env!("CARGO_BIN_NAME").to_string()),
            runner_version: Some(env!("CARGO_PKG_VERSION").to_string()),
            pid: Some(std::process::id()),
        },
    )
    .await
    .with_context(|| {
        format!(
            "Failed to send runner hello for channel '{}'",
            args.channel_id
        )
    })?;

    let _heartbeat_task = spawn_runner_heartbeat(
        daemon.clone(),
        args.channel_id.clone(),
        heartbeat_shutdown_rx,
    );

    runner
        .run_driver(&args.agent_id, &mut driver, task_timeout_ms)
        .await
        .with_context(|| format!("Channel '{}' runner failed", args.channel_id))
}

fn validate_settings(args: ValidateSettingsArgs) -> Result<()> {
    let settings = parse_settings_json(&args.settings_json)?;
    let access_policy = ChannelAccessPolicy::from_settings(&settings)?;
    turin_channel_telegram::validate_settings(
        &settings,
        access_policy.requires_unconfigured_inbound(),
    )
}

fn describe() -> Result<()> {
    println!(
        "{}",
        serde_json::to_string(&turin_channel_telegram::adapter_manifest())?
    );
    Ok(())
}

fn setup_auth_flow_start(args: AuthFlowRequestArgs) -> Result<()> {
    let request: ChannelAuthFlowStartRequest =
        serde_json::from_str(&args.request_json).context("Failed to parse auth flow start JSON")?;
    println!(
        "{}",
        serde_json::to_string(&turin_channel_telegram::start_auth_flow(&request)?)?
    );
    Ok(())
}

fn setup_auth_flow_poll(args: AuthFlowRequestArgs) -> Result<()> {
    let request: ChannelAuthFlowPollRequest =
        serde_json::from_str(&args.request_json).context("Failed to parse auth flow poll JSON")?;
    println!(
        "{}",
        serde_json::to_string(&turin_channel_telegram::poll_auth_flow(&request)?)?
    );
    Ok(())
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
