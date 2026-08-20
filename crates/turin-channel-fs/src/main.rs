use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use turin_channel_fs::FsChannelDriver;
use turin_channel_runner::{
    ChannelSidecarRunArgs, init_channel_tracing, parse_channel_settings_json,
    prepare_channel_sidecar_run,
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
    runtime_dir: PathBuf,
    #[arg(long)]
    settings_json: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    init_channel_tracing();
    match Cli::parse().command {
        Command::Run(args) => run(args).await,
        Command::Describe => {
            println!(
                "{}",
                serde_json::to_string(&turin_channel_fs::adapter_manifest())?
            );
            Ok(())
        }
        Command::ValidateSettings(args) => {
            let settings = parse_channel_settings_json(&args.settings_json)?;
            turin_channel_fs::validate_settings(&args.runtime_dir, &settings)
        }
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
    let mut driver = FsChannelDriver::from_settings(
        &args.channel_id,
        &sidecar.runtime_dir,
        &settings,
        sidecar.shutdown_rx.clone(),
    )
    .await
    .with_context(|| {
        format!(
            "Failed to initialize filesystem relay '{}'",
            args.channel_id
        )
    })?;

    sidecar.run_driver(&args.agent_id, &mut driver).await
}
