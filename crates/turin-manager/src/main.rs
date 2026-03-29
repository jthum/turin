mod files;
mod runner;
mod setup;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(author, version, about = "Install, configure, and troubleshoot Turin")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Init(InitArgs),
    Doctor(DoctorArgs),
    #[command(subcommand)]
    Channels(ChannelsCommand),
}

#[derive(Args, Debug, Clone)]
struct InitArgs {
    #[arg(long, default_value = "turin.toml")]
    config: PathBuf,
    #[arg(long)]
    force: bool,
}

#[derive(Args, Debug, Clone)]
struct DoctorArgs {
    #[arg(long, default_value = "turin.toml")]
    config: PathBuf,
}

#[derive(Subcommand, Debug, Clone)]
enum ChannelsCommand {
    List(ChannelsListArgs),
    Configure(ConfigureChannelArgs),
    Status(ChannelsStatusArgs),
}

#[derive(Args, Debug, Clone)]
struct ChannelsListArgs {
    #[arg(long, default_value = "turin.toml")]
    config: PathBuf,
}

#[derive(Args, Debug, Clone)]
struct ConfigureChannelArgs {
    #[arg(long, default_value = "turin.toml")]
    config: PathBuf,
    kind: String,
    #[arg(long)]
    channel_id: Option<String>,
    #[arg(long)]
    agent_id: Option<String>,
}

#[derive(Args, Debug, Clone)]
struct ChannelsStatusArgs {
    #[arg(long, default_value = "turin.toml")]
    config: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Init(args) => {
            setup::run_init(setup::InitArgs {
                config: args.config,
                force: args.force,
            })
            .await
        }
        Command::Doctor(args) => {
            setup::run_doctor(setup::DoctorArgs {
                config: args.config,
            })
            .await
        }
        Command::Channels(ChannelsCommand::List(args)) => {
            setup::run_channels_list(setup::ChannelsListArgs {
                config: args.config,
            })
            .await
        }
        Command::Channels(ChannelsCommand::Configure(args)) => {
            setup::run_configure_channel(setup::ConfigureChannelArgs {
                config: args.config,
                kind: args.kind,
                channel_id: args.channel_id,
                agent_id: args.agent_id,
            })
            .await
        }
        Command::Channels(ChannelsCommand::Status(args)) => {
            setup::run_channels_status(setup::ChannelsStatusArgs {
                config: args.config,
            })
            .await
        }
    }
}
