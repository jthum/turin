use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use turin_channel_discord::DiscordChannelDriver;
use turin_channel_runner::{
    ChannelSidecarRunArgs, init_channel_tracing, parse_auth_flow_poll_request,
    parse_auth_flow_start_request, parse_channel_settings_json, prepare_channel_sidecar_run,
};

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
    #[arg(long)]
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
    init_channel_tracing();
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
    let settings = parse_channel_settings_json(&args.settings_json)?;
    let sidecar = prepare_channel_sidecar_run(
        ChannelSidecarRunArgs {
            channel_id: args.channel_id.clone(),
            daemon_endpoint: args.daemon_endpoint,
            bindings_path: args.bindings_path,
            access_state_path: args.access_state_path,
            idle_timeout_seconds: args.idle_timeout_seconds,
        },
        &settings,
    )?;

    let mut driver = DiscordChannelDriver::from_settings(
        &args.channel_id,
        &settings,
        sidecar.shutdown_rx.clone(),
    )
    .await
    .with_context(|| {
        format!(
            "Failed to initialize discord channel driver '{}'",
            args.channel_id
        )
    })?;

    sidecar
        .announce_presence(
            turin_channel_discord::adapter_manifest(),
            Some(env!("CARGO_BIN_NAME").to_string()),
            Some(env!("CARGO_PKG_VERSION").to_string()),
        )
        .await?;

    let _heartbeat_task = sidecar.spawn_heartbeat();

    sidecar.run_driver(&args.agent_id, &mut driver).await
}

fn validate_settings(args: ValidateSettingsArgs) -> Result<()> {
    let settings = parse_channel_settings_json(&args.settings_json)?;
    turin_channel_discord::validate_settings(&settings)
}

fn describe() -> Result<()> {
    println!(
        "{}",
        serde_json::to_string(&turin_channel_discord::adapter_manifest())?
    );
    Ok(())
}

fn setup_auth_flow_start(args: AuthFlowRequestArgs) -> Result<()> {
    let request = parse_auth_flow_start_request(&args.request_json)?;
    println!(
        "{}",
        serde_json::to_string(&turin_channel_discord::start_auth_flow(&request)?)?
    );
    Ok(())
}

fn setup_auth_flow_poll(args: AuthFlowRequestArgs) -> Result<()> {
    let request = parse_auth_flow_poll_request(&args.request_json)?;
    println!(
        "{}",
        serde_json::to_string(&turin_channel_discord::poll_auth_flow(&request)?)?
    );
    Ok(())
}
