use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::Parser;
use turin_client::ConnectionSpec;
use turin_types::layout::DEFAULT_BOOTSTRAP_CONFIG_PATH;
use turin_web::{DEFAULT_WEB_BIND, WebServeOptions, serve};

const DEFAULT_REMOTE_AUTH_TOKEN_ENV: &str = "TURIN_REMOTE_AUTH_TOKEN";

#[derive(Parser, Debug)]
#[command(name = "turin-web", version, about)]
struct Cli {
    /// HTTP bind address
    #[arg(long, default_value = DEFAULT_WEB_BIND)]
    bind: String,

    /// Built SvelteKit assets to serve
    #[arg(long, default_value_os_t = default_assets_dir())]
    assets_dir: PathBuf,

    /// Path to Turin config for a local daemon connection
    #[arg(long, default_value = DEFAULT_BOOTSTRAP_CONFIG_PATH)]
    config: PathBuf,

    /// Connect directly to a local daemon endpoint
    #[arg(long)]
    endpoint: Option<PathBuf>,

    /// Connect through turin-remote at this base URL
    #[arg(long)]
    remote_url: Option<String>,

    /// Bearer token for turin-remote
    #[arg(long)]
    auth_token: Option<String>,

    /// Environment variable containing the turin-remote bearer token
    #[arg(long)]
    auth_token_env: Option<String>,

    /// Allow binding without a local-only authentication boundary
    #[arg(long)]
    allow_non_loopback: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let connection = connection_spec(&cli)?;
    serve(WebServeOptions {
        bind: cli.bind,
        assets_dir: cli.assets_dir,
        connection,
        allow_non_loopback: cli.allow_non_loopback,
    })
    .await
}

fn default_assets_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("frontend/build")
}

fn connection_spec(cli: &Cli) -> Result<ConnectionSpec> {
    if cli.remote_url.is_some() && cli.endpoint.is_some() {
        bail!("--remote-url and --endpoint cannot be used together");
    }

    if let Some(base_url) = &cli.remote_url {
        if let Some(auth_token) = &cli.auth_token {
            return Ok(ConnectionSpec::Remote {
                base_url: base_url.clone(),
                auth_token: auth_token.clone(),
            });
        }
        return Ok(ConnectionSpec::RemoteEnv {
            base_url: base_url.clone(),
            auth_token_env: cli
                .auth_token_env
                .clone()
                .unwrap_or_else(|| DEFAULT_REMOTE_AUTH_TOKEN_ENV.to_string()),
        });
    }

    if cli.auth_token.is_some() || cli.auth_token_env.is_some() {
        bail!("--auth-token and --auth-token-env require --remote-url");
    }

    if let Some(endpoint) = &cli.endpoint {
        return Ok(ConnectionSpec::LocalEndpoint {
            endpoint: endpoint.clone(),
        });
    }

    Ok(ConnectionSpec::LocalConfig {
        config_path: cli.config.clone(),
    })
}
