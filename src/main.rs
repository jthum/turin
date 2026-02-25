use anyhow::{Context, Result};
use clap::Parser;
use std::path::Path;
use std::path::PathBuf;

use turin::kernel::Kernel;
use turin::kernel::config::TurinConfig;

mod commands;

/// Turin: A single-binary, event-driven LLM execution runtime
#[derive(Parser, Debug)]
#[command(name = "turin", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Log level (error, warn, info, debug, trace)
    #[arg(long, default_value = "info", global = true)]
    log_level: String,

    /// Path to log file
    #[arg(long, global = true)]
    log_file: Option<PathBuf>,
}

#[derive(clap::Subcommand, Debug)]
enum Commands {
    /// Run the agent with a prompt
    Run {
        /// The prompt to send to the LLM
        #[arg(long)]
        prompt: String,

        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,

        /// Override the model from config
        #[arg(long)]
        model: Option<String>,

        /// Override the provider from config
        #[arg(long)]
        provider: Option<String>,

        /// Show verbose event-level output
        #[arg(long)]
        verbose: bool,

        /// Output events as NDJSON to stdout
        #[arg(long)]
        json: bool,
    },

    /// Start an interactive REPL session
    Repl {
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,

        /// Override the model from config
        #[arg(long)]
        model: Option<String>,

        /// Override the provider from config
        #[arg(long)]
        provider: Option<String>,

        /// Show verbose event-level output
        #[arg(long)]
        verbose: bool,
    },

    /// Run a specific harness script (for testing)
    Script {
        /// Path to the Lua script to run
        path: PathBuf,

        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,

        /// Override the model from config
        #[arg(long)]
        model: Option<String>,

        /// Override the provider from config
        #[arg(long)]
        provider: Option<String>,
    },

    /// Initialize a new Turin project in the current directory
    Init,

    /// Validate configuration and harness scripts
    Check {
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: std::path::PathBuf,
    },
}

use tracing_subscriber::{EnvFilter, fmt, prelude::*};

fn init_tracing(log_level: &str, log_file: Option<PathBuf>) -> Result<()> {
    let filter = EnvFilter::try_from_default_env()
        .or_else(|_| EnvFilter::try_new(log_level))
        .unwrap_or_else(|_| EnvFilter::new("info"));

    let stdout_layer = fmt::layer().with_writer(std::io::stderr).with_ansi(true);

    let file_layer = log_file.map(|path| {
        let parent = path.parent().unwrap_or_else(|| std::path::Path::new("."));
        let filename = path.file_name().unwrap_or_default();
        let file_appender = tracing_appender::rolling::never(parent, filename);
        fmt::layer()
            .with_writer(file_appender)
            .with_ansi(false)
            .json()
    });

    tracing_subscriber::registry()
        .with(filter)
        .with(stdout_layer)
        .with(file_layer)
        .init();

    Ok(())
}

fn load_config_with_overrides(
    config_path: &Path,
    model: Option<String>,
    provider: Option<String>,
) -> Result<TurinConfig> {
    let mut config =
        TurinConfig::from_file(config_path).with_context(|| "Failed to load config")?;

    if let Some(m) = model {
        config.agent.model = m;
    }
    if let Some(p) = provider {
        config.agent.provider = p;
        config.validate()?;
    }

    Ok(config)
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(&cli.log_level, cli.log_file)?;

    match cli.command {
        Commands::Run {
            prompt,
            config,
            model,
            provider,
            verbose: _,
            json,
        } => {
            let config = load_config_with_overrides(&config, model, provider)?;

            tracing::info!(
                model = %config.agent.model,
                provider = %config.agent.provider,
                workspace = %config.kernel.workspace_root,
                harness_dir = %config.harness.directory,
                db = %config.persistence.database_path,
                "Config loaded"
            );

            // Build kernel, initialize state store, and run
            let mut kernel = Kernel::builder(config).json_mode(json).build()?;
            kernel.init_state().await?;
            kernel.init_clients()?;
            kernel.init_harness().await?;
            kernel.start_watcher()?;
            let mut session = kernel.create_session().await;
            kernel.start_session(&mut session).await?;
            kernel.run(&mut session, Some(prompt)).await?;
            kernel.end_session(&mut session).await?;
            kernel.shutdown_mcp_clients().await;
            if !json {
                commands::common::print_session_summary(&session);
            }
            Ok(())
        }
        Commands::Repl {
            config,
            model,
            provider,
            verbose,
        } => {
            let config = load_config_with_overrides(&config, model, provider)?;
            commands::repl::run_repl(config, verbose).await
        }
        Commands::Script {
            path,
            config,
            model,
            provider,
        } => {
            let config = load_config_with_overrides(&config, model, provider)?;

            // Build kernel
            let mut kernel = Kernel::builder(config).json_mode(false).build()?;
            kernel.init_state().await?;
            kernel.init_clients()?;
            kernel.init_harness().await?;

            // Read script
            let script_content = std::fs::read_to_string(&path)
                .with_context(|| format!("Failed to read script: {}", path.display()))?;

            kernel.run_script(&script_content)?;
            kernel.shutdown_mcp_clients().await;

            Ok(())
        }
        Commands::Init => {
            commands::init::run_init()?;
            Ok(())
        }
        Commands::Check { config } => {
            commands::check::run_check(&config).await?;
            Ok(())
        }
    }
}
