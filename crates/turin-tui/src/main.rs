mod app;
mod harness_ui;
mod terminal;
mod theme;

use anyhow::{Result, anyhow};
use clap::Parser;
use std::path::PathBuf;
use tokio::runtime::Handle;
use turin_ui_core::{ConnectionOptions, connect_dashboard, spawn_controller};

use app::TuiApp;

#[derive(Parser, Debug)]
#[command(name = "turin-tui", version, about)]
struct Args {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    endpoint: Option<PathBuf>,
    #[arg(long)]
    remote_url: Option<String>,
    #[arg(long)]
    auth_token: Option<String>,
    #[arg(long)]
    auth_token_env: Option<String>,
    #[arg(long)]
    profile: Option<String>,
    #[arg(long)]
    profiles_file: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let connection_options = connection_options(&args);
    let spec = connection_options.to_spec()?;
    let (client, dashboard) = connect_dashboard(&spec).await?;
    let controller = spawn_controller(&Handle::current(), client);
    let app = TuiApp::new(dashboard, controller, connection_options);
    terminal::run(app).await.map_err(|err| anyhow!(err))
}

fn connection_options(args: &Args) -> ConnectionOptions {
    ConnectionOptions {
        config_path: args.config.clone(),
        endpoint: args.endpoint.clone(),
        remote_url: args.remote_url.clone(),
        auth_token: args.auth_token.clone(),
        auth_token_env: args.auth_token_env.clone(),
        profile: args.profile.clone(),
        profiles_file: args.profiles_file.clone(),
        suppress_profile_resolution: false,
    }
}
