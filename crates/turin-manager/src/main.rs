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
    #[command(subcommand)]
    Setup(SetupCommand),
}

#[derive(Args, Debug, Clone)]
struct InitArgs {
    #[arg(long, default_value = "turin.toml")]
    config: PathBuf,
    #[arg(long)]
    force: bool,
}

#[derive(Subcommand, Debug, Clone)]
enum SetupCommand {
    Telegram(TelegramSetupArgs),
}

#[derive(Args, Debug, Clone)]
struct TelegramSetupArgs {
    #[arg(long, default_value = "turin.toml")]
    config: PathBuf,
    #[arg(long)]
    channel_id: Option<String>,
    #[arg(long)]
    agent_id: Option<String>,
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
        Command::Setup(SetupCommand::Telegram(args)) => {
            setup::run_setup_telegram(setup::TelegramSetupArgs {
                config: args.config,
                channel_id: args.channel_id,
                agent_id: args.agent_id,
            })
            .await
        }
    }
}
