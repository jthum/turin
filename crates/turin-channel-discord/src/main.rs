use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use turin_channel_discord::DiscordChannelDriver;
use turin_channel_runner::{
    ChannelRunArgs, ChannelStateArgs, DEFAULT_TURIN_CONFIG_PATH, init_channel_tracing,
    parse_auth_flow_poll_request, parse_auth_flow_start_request, parse_channel_settings_json,
    prepare_channel_run,
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
    State(ChannelStateArgs),
    Describe,
    ValidateSettings(ValidateSettingsArgs),
    SetupAuthFlowStart(AuthFlowRequestArgs),
    SetupAuthFlowPoll(AuthFlowRequestArgs),
}

#[derive(Parser)]
struct RunArgs {
    #[arg(long)]
    config: PathBuf,
    #[arg(long, default_value = DEFAULT_TURIN_CONFIG_PATH)]
    turin_config: PathBuf,
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
        Command::State(args) => args.run("discord").await,
        Command::Describe => describe(),
        Command::ValidateSettings(args) => validate_settings(args),
        Command::SetupAuthFlowStart(args) => setup_auth_flow_start(args),
        Command::SetupAuthFlowPoll(args) => setup_auth_flow_poll(args),
    }
}

async fn run(args: RunArgs) -> Result<()> {
    let run = prepare_channel_run(ChannelRunArgs {
        config_path: args.config,
        turin_config_path: args.turin_config,
        expected_kind: "discord".to_string(),
    })
    .await?;

    let mut driver = DiscordChannelDriver::from_settings(
        &run.channel_id,
        &run.settings,
        run.shutdown_rx.clone(),
    )
    .await
    .with_context(|| {
        format!(
            "Failed to initialize discord channel driver '{}'",
            run.channel_id
        )
    })?;

    run.run_driver(&mut driver).await
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
