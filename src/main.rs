use anyhow::{Context, Result};
use clap::Parser;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::path::PathBuf;

use turin::kernel::config::TurinConfig;
use turin::kernel::Kernel;

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

use tracing_subscriber::{fmt, prelude::*, EnvFilter};

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
            // Load config
            let mut config =
                TurinConfig::from_file(&config).with_context(|| "Failed to load config")?;

            // Apply CLI overrides
            if let Some(m) = model {
                config.agent.model = m;
            }
            if let Some(p) = provider {
                config.agent.provider = p;
                // Re-validate after override
                config.validate()?;
            }

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
            let mut session = kernel.create_session();
            kernel.start_session(&mut session).await?;
            kernel.run(&mut session, Some(prompt)).await?;
            kernel.end_session(&mut session).await?;
            if !json {
                println!("\n\x1b[36m\x1b[1m── Session Summary ──\x1b[0m");
                println!("  \x1b[1mTotal Tokens:\x1b[0m  {} ({} in, {} out)", session.total_input_tokens + session.total_output_tokens, session.total_input_tokens, session.total_output_tokens);
                println!("  \x1b[1mTurns:\x1b[0m         {}", session.turn_index);
            }
            Ok(())
        }
        Commands::Repl {
            config,
            model,
            provider,
            verbose,
        } => {
            // Load config
            let mut config =
                TurinConfig::from_file(&config).with_context(|| "Failed to load config")?;

            // Apply CLI overrides
            if let Some(m) = model {
                config.agent.model = m;
            }
            if let Some(p) = provider {
                config.agent.provider = p;
                config.validate()?;
            }

            tracing::info!(
                model = %config.agent.model,
                provider = %config.agent.provider,
                "Config loaded (REPL mode)"
            );

            // Build kernel
            let mut kernel = Kernel::builder(config).build()?; // JSON not supported in REPL yet
            kernel.init_state().await?;
            kernel.init_clients()?;
            kernel.init_harness().await?;
            kernel.start_watcher()?;

            // Start REPL loop
            let mut rl = DefaultEditor::new()?;
            tracing::info!("REPL started. Type 'exit' or Ctrl+D to quit.");
            if !verbose {
                println!("Turin REPL v{}", env!("CARGO_PKG_VERSION"));
                println!("Type 'exit' or Ctrl+D to quit. Type '/reload' to reload harness.");
            }

            // Trigger AgentStart
            let mut session = kernel.create_session();
            kernel.start_session(&mut session).await?;

use turin::inference::provider::{InferenceContent, InferenceRole};

            loop {
                let readline = rl.readline("\x1b[36m\x1b[1mturin\x1b[0m\x1b[34m>\x1b[0m ");
                match readline {
                    Ok(line) => {
                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }

                        // Slash command handler
                        if line.starts_with('/') {
                            let parts: Vec<&str> = line.split_whitespace().collect();
                            let cmd = parts[0].to_lowercase();

                            match cmd.as_str() {
                                "/status" => {
                                    println!("\n\x1b[36m\x1b[1m── Session Status ──\x1b[0m");
                                    println!("  \x1b[1mSession ID:\x1b[0m {}", session.id);
                                    println!("  \x1b[1mProvider:\x1b[0m   {}", kernel.config().agent.provider);
                                    println!("  \x1b[1mModel:\x1b[0m      {}", kernel.config().agent.model);
                                    println!("  \x1b[1mTurns:\x1b[0m      {}", session.turn_index);
                                    println!(
                                        "  \x1b[1mTokens:\x1b[0m     {} total ({} in, {} out)",
                                        session.total_input_tokens + session.total_output_tokens,
                                        session.total_input_tokens,
                                        session.total_output_tokens
                                    );
                                    println!();
                                    continue;
                                }
                                "/history" => {
                                    println!("\n\x1b[36m\x1b[1m── Message History ──\x1b[0m");
                                    if session.history.is_empty() {
                                        println!("  (No messages yet)");
                                    }
                                    for (i, msg) in session.history.iter().enumerate() {
                                        let role_color = match msg.role {
                                            InferenceRole::User => "\x1b[32m",      // Green
                                            InferenceRole::Assistant => "\x1b[34m", // Blue
                                            InferenceRole::Tool => "\x1b[33m",      // Yellow
                                        };
                                        let role_name = format!("{:?}", msg.role);

                                        let mut content_summary = String::new();
                                        for content in &msg.content {
                                            match content {
                                                InferenceContent::Text { text } => {
                                                    content_summary.push_str(text);
                                                }
                                                InferenceContent::ToolUse { name, .. } => {
                                                    content_summary.push_str(&format!("[Tool Call: {}] ", name));
                                                }
                                                InferenceContent::ToolResult { .. } => {
                                                    content_summary.push_str("[Tool Result] ");
                                                }
                                                InferenceContent::Thinking { .. } => {
                                                    content_summary.push_str("[Thinking] ");
                                                }
                                            }
                                        }

                                        if content_summary.len() > 80 {
                                            content_summary = format!("{}...", &content_summary[..77]);
                                        }
                                        // Replace newlines with spaces for summary
                                        let cleaned_summary = content_summary.replace('\n', " ");

                                        println!(
                                            "  [{}] {}{:10}\x1b[0m: {}",
                                            i, role_color, role_name, cleaned_summary
                                        );
                                    }
                                    println!();
                                    continue;
                                }
                                "/reload" => {
                                    tracing::info!("Reloading harness...");
                                    match kernel.reload_harness().await {
                                        Ok(_) => tracing::info!("Harness reloaded successfully."),
                                        Err(e) => tracing::error!(error = %e, "Failed to reload harness"),
                                    }
                                    continue;
                                }
                                "/clear" => {
                                    session.history.clear();
                                    session.turn_index = 0;
                                    session.total_input_tokens = 0;
                                    session.total_output_tokens = 0;
                                    println!("\x1b[32m\x1b[1m✓\x1b[0m Session history and stats cleared.");
                                    continue;
                                }
                                "/help" => {
                                    println!("\n\x1b[36m\x1b[1m── Available Commands ──\x1b[0m");
                                    println!("  \x1b[1m/status\x1b[0m   - Show session statistics");
                                    println!("  \x1b[1m/history\x1b[0m  - Show condensed message history");
                                    println!("  \x1b[1m/reload\x1b[0m   - Reload harness scripts");
                                    println!("  \x1b[1m/clear\x1b[0m    - Clear session history and reset stats");
                                    println!("  \x1b[1m/help\x1b[0m     - Show this help message");
                                    println!("  \x1b[1m/quit\x1b[0m     - Exit the REPL");
                                    println!();
                                    continue;
                                }
                                "/quit" | "/exit" => {
                                    break;
                                }
                                _ => {
                                    println!("\x1b[31mUnknown command: {}\x1b[0m. Type /help for assistance.", cmd);
                                    continue;
                                }
                            }
                        }

                        if line.eq_ignore_ascii_case("exit") {
                            break;
                        }

                        let _ = rl.add_history_entry(line);

                        // Push prompt to kernel queue and run until empty
                        kernel.run(&mut session, Some(line.to_string())).await?;
                    }
                    Err(ReadlineError::Interrupted) => {
                        println!("^C");
                        break;
                    }
                    Err(ReadlineError::Eof) => {
                        println!("^D");
                        break;
                    }
                    Err(err) => {
                        println!("Error: {:?}", err);
                        break;
                    }
                }
            }
            kernel.end_session(&mut session).await?;
            println!("\n\x1b[36m\x1b[1m── Session Summary ──\x1b[0m");
            println!("  \x1b[1mTotal Tokens:\x1b[0m  {} ({} in, {} out)", session.total_input_tokens + session.total_output_tokens, session.total_input_tokens, session.total_output_tokens);
            println!("  \x1b[1mTurns:\x1b[0m         {}", session.turn_index);
            Ok(())
        }
        Commands::Script {
            path,
            config,
            model,
            provider,
        } => {
            // Load config
            let mut config =
                TurinConfig::from_file(&config).with_context(|| "Failed to load config")?;

            // Apply CLI overrides
            if let Some(m) = model {
                config.agent.model = m;
            }
            if let Some(p) = provider {
                config.agent.provider = p;
                config.validate()?;
            }

            // Build kernel
            let mut kernel = Kernel::builder(config).json_mode(false).build()?;
            kernel.init_state().await?;
            kernel.init_clients()?;
            kernel.init_harness().await?;

            // Read script
            let script_content = std::fs::read_to_string(&path)
                .with_context(|| format!("Failed to read script: {}", path.display()))?;

            kernel.run_script(&script_content)?;

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
