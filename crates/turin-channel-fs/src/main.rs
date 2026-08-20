use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use turin_channel_fs::FsChannelDriver;
use turin_channel_runner::{
    ChannelRunArgs, ChannelStateArgs, DEFAULT_TURIN_CONFIG_PATH, init_channel_tracing,
    parse_channel_settings_json, prepare_channel_run,
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
    runtime_dir: PathBuf,
    #[arg(long)]
    settings_json: String,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    init_channel_tracing();
    match Cli::parse().command {
        Command::Run(args) => run(args).await,
        Command::State(args) => args.run("fs").await,
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
    let run = prepare_channel_run(ChannelRunArgs {
        config_path: args.config,
        turin_config_path: args.turin_config,
        expected_kind: "fs".to_string(),
    })
    .await?;
    let mut driver = FsChannelDriver::from_settings(
        &run.channel_id,
        &run.runtime_dir,
        &run.settings,
        run.shutdown_rx.clone(),
    )
    .await
    .with_context(|| {
        format!(
            "Failed to initialize filesystem channel '{}'",
            run.channel_id
        )
    })?;

    run.run_driver(&mut driver).await
}
