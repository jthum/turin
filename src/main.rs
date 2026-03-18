use anyhow::Result;
use clap::Parser;

mod cli;
mod commands;
mod dispatch;

use cli::Cli;
use turin::tracing_support::init_tracing;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(&cli.log_level, cli.log_file.clone())?;
    dispatch::run(cli).await
}
