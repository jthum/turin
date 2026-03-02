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

        /// Run the named configured agent instead of the default root agent
        #[arg(long)]
        agent: Option<String>,

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

        /// Run the REPL against the named configured agent instead of the default root agent
        #[arg(long)]
        agent: Option<String>,

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

    /// Run or control the Turin daemon
    Daemon {
        #[command(subcommand)]
        command: DaemonCommands,
    },
}

#[derive(clap::Subcommand, Debug)]
enum DaemonCommands {
    /// Start the daemon in the foreground
    Start {
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
    },
    /// Ping the daemon
    Ping {
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Show daemon status
    Status {
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Rescan filesystem-backed daemon state
    Rescan {
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Reload filesystem-backed daemon state
    Reload {
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Show isolated runtime loading errors
    Errors {
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Stop the daemon
    Stop {
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Tail daemon runtime events
    Events {
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output NDJSON
        #[arg(long)]
        json: bool,
    },
    /// Manage filesystem-backed daemon agents
    Agent {
        #[command(subcommand)]
        command: DaemonAgentCommands,
    },
    /// Manage daemon tasks
    Task {
        #[command(subcommand)]
        command: DaemonTaskCommands,
    },
    /// Manage daemon harnesses
    Harness {
        #[command(subcommand)]
        command: DaemonHarnessCommands,
    },
    /// Inspect persisted daemon sessions
    Session {
        #[command(subcommand)]
        command: DaemonSessionCommands,
    },
}

#[derive(clap::Subcommand, Debug)]
enum DaemonAgentCommands {
    /// List daemon-managed agents
    List {
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Show one daemon-managed agent
    Get {
        /// Agent ID
        id: String,
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Show live runtime status for one daemon-managed agent
    Status {
        /// Agent ID
        id: String,
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Show isolated daemon issues for one agent directory
    Issues {
        /// Agent ID
        id: String,
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Reload one daemon-managed agent from disk
    Reload {
        /// Agent ID
        id: String,
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Create a daemon-managed agent directory
    Create {
        /// Agent ID / directory name
        id: String,
        /// Provider name
        #[arg(long)]
        provider: String,
        /// Model identifier
        #[arg(long)]
        model: String,
        /// Optional system prompt override
        #[arg(long)]
        system_prompt: Option<String>,
        /// Optional shared harness binding
        #[arg(long)]
        harness: Option<String>,
        /// Create the agent disabled
        #[arg(long)]
        disabled: bool,
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Enable a daemon-managed agent
    Enable {
        /// Agent ID
        id: String,
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Update a daemon-managed agent config
    Update {
        /// Agent ID
        id: String,
        /// Optional provider override
        #[arg(long)]
        provider: Option<String>,
        /// Optional model override
        #[arg(long)]
        model: Option<String>,
        /// Optional system prompt override
        #[arg(long)]
        system_prompt: Option<String>,
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Disable a daemon-managed agent
    Disable {
        /// Agent ID
        id: String,
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Delete a daemon-managed agent directory
    Delete {
        /// Agent ID
        id: String,
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Bind an agent to a shared harness
    BindHarness {
        /// Agent ID
        id: String,
        /// Shared harness ID
        harness_id: String,
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Switch an agent back to a local harness
    UseLocalHarness {
        /// Agent ID
        id: String,
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(clap::Subcommand, Debug)]
enum DaemonTaskCommands {
    /// Submit a task to a daemon-managed agent
    Submit {
        /// Agent ID
        agent_id: String,
        /// Prompt to submit
        prompt: String,
        /// Wait for the task to complete and print the terminal result
        #[arg(long)]
        wait: bool,
        /// Optional wait timeout in milliseconds (only meaningful with --wait)
        #[arg(long)]
        timeout_ms: Option<u64>,
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Show one daemon task by request ID
    Get {
        /// Request ID returned by task submission
        request_id: String,
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Wait for one daemon task to complete
    Wait {
        /// Request ID returned by task submission
        request_id: String,
        /// Optional timeout in milliseconds
        #[arg(long)]
        timeout_ms: Option<u64>,
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Cancel one daemon task by request ID
    Cancel {
        /// Request ID returned by task submission
        request_id: String,
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// List daemon tasks
    List {
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(clap::Subcommand, Debug)]
enum DaemonHarnessCommands {
    /// List daemon-visible harness runtimes
    List {
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Create a shared harness directory
    Create {
        /// Harness ID
        id: String,
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Show one harness by ID
    Get {
        /// Harness ID
        id: String,
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Show isolated daemon issues for one harness directory
    Issues {
        /// Harness ID
        id: String,
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Reload one harness by ID
    Reload {
        /// Harness ID
        id: String,
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Validate one harness by ID without mutating the live runtime
    Validate {
        /// Harness ID
        id: String,
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Delete a shared harness directory
    Delete {
        /// Harness ID
        id: String,
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
}

#[derive(clap::Subcommand, Debug)]
enum DaemonSessionCommands {
    /// List recent persisted sessions
    List {
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Maximum number of sessions to return
        #[arg(long, default_value_t = 50)]
        limit: usize,
        /// Offset into the session list
        #[arg(long, default_value_t = 0)]
        offset: usize,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Show one persisted session by session ID
    Get {
        /// Session ID
        session_id: String,
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Cooperatively cancel one active daemon session
    Cancel {
        /// Session ID
        session_id: String,
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
    },
    /// Force-kill one active daemon session
    Kill {
        /// Session ID
        session_id: String,
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Output JSON
        #[arg(long)]
        json: bool,
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
    agent_id: Option<&str>,
) -> Result<TurinConfig> {
    let mut config =
        TurinConfig::from_file(config_path).with_context(|| "Failed to load config")?;

    let target = if let Some(agent_id) = agent_id {
        if agent_id == config.agent.id {
            &mut config.agent
        } else {
            config
                .agents
                .get_mut(agent_id)
                .ok_or_else(|| anyhow::anyhow!("Unknown agent profile: {}", agent_id))?
        }
    } else {
        &mut config.agent
    };

    if let Some(m) = model {
        target.model = m;
    }
    if let Some(p) = provider {
        target.provider = p;
    }
    config.validate()?;

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
            agent,
            verbose: _,
            json,
        } => {
            let config = load_config_with_overrides(&config, model, provider, agent.as_deref())?;
            let selected_agent_id = agent.unwrap_or_else(|| config.agent.id.clone());
            let selected_agent = if selected_agent_id == config.agent.id {
                &config.agent
            } else {
                config.agents.get(&selected_agent_id).ok_or_else(|| {
                    anyhow::anyhow!("Unknown agent profile: {}", selected_agent_id)
                })?
            };
            let (harness_id, harness_cfg) = config.harness_binding_for_agent(selected_agent)?;

            tracing::info!(
                agent_id = %selected_agent_id,
                model = %selected_agent.model,
                provider = %selected_agent.provider,
                workspace = %config.kernel.workspace_root,
                harness_id = %harness_id,
                harness_dir = %harness_cfg.directory,
                db = %config.persistence.database_path,
                "Config loaded"
            );

            // Build kernel, initialize state store, and run
            let mut kernel = Kernel::builder(config).json_mode(json).build()?;
            kernel.init_state().await?;
            kernel.init_clients()?;
            kernel.init_harness().await?;
            kernel.start_watcher()?;
            let mut session = kernel.create_session_for_agent(&selected_agent_id).await;
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
            agent,
            verbose,
        } => {
            let config = load_config_with_overrides(&config, model, provider, agent.as_deref())?;
            commands::repl::run_repl(config, verbose, agent).await
        }
        Commands::Script {
            path,
            config,
            model,
            provider,
        } => {
            let config = load_config_with_overrides(&config, model, provider, None)?;

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
        Commands::Daemon { command } => match command {
            DaemonCommands::Start { config } => commands::daemon::run_start(&config).await,
            DaemonCommands::Ping { config, json } => {
                commands::daemon::run_ping(&config, json).await
            }
            DaemonCommands::Status { config, json } => {
                commands::daemon::run_status(&config, json).await
            }
            DaemonCommands::Rescan { config, json } => {
                commands::daemon::run_rescan(&config, json).await
            }
            DaemonCommands::Reload { config, json } => {
                commands::daemon::run_reload(&config, json).await
            }
            DaemonCommands::Errors { config, json } => {
                commands::daemon::run_runtime_errors(&config, json).await
            }
            DaemonCommands::Stop { config, json } => {
                commands::daemon::run_stop(&config, json).await
            }
            DaemonCommands::Events { config, json } => {
                commands::daemon::run_events(&config, json).await
            }
            DaemonCommands::Agent { command } => match command {
                DaemonAgentCommands::List { config, json } => {
                    commands::daemon::run_agent_list(&config, json).await
                }
                DaemonAgentCommands::Get { id, config, json } => {
                    commands::daemon::run_agent_get(&config, &id, json).await
                }
                DaemonAgentCommands::Status { id, config, json } => {
                    commands::daemon::run_agent_status(&config, &id, json).await
                }
                DaemonAgentCommands::Issues { id, config, json } => {
                    commands::daemon::run_agent_issues(&config, &id, json).await
                }
                DaemonAgentCommands::Reload { id, config, json } => {
                    commands::daemon::run_agent_reload(&config, &id, json).await
                }
                DaemonAgentCommands::Create {
                    id,
                    provider,
                    model,
                    system_prompt,
                    harness,
                    disabled,
                    config,
                    json,
                } => {
                    commands::daemon::run_agent_create(
                        &config,
                        serde_json::json!({
                            "id": id,
                            "provider": provider,
                            "model": model,
                            "system_prompt": system_prompt,
                            "harness": harness,
                            "enabled": !disabled,
                        }),
                        json,
                    )
                    .await
                }
                DaemonAgentCommands::Enable { id, config, json } => {
                    commands::daemon::run_agent_enable(&config, &id, json).await
                }
                DaemonAgentCommands::Update {
                    id,
                    provider,
                    model,
                    system_prompt,
                    config,
                    json,
                } => {
                    commands::daemon::run_agent_update(
                        &config,
                        serde_json::json!({
                            "id": id,
                            "provider": provider,
                            "model": model,
                            "system_prompt": system_prompt,
                        }),
                        json,
                    )
                    .await
                }
                DaemonAgentCommands::Disable { id, config, json } => {
                    commands::daemon::run_agent_disable(&config, &id, json).await
                }
                DaemonAgentCommands::Delete { id, config, json } => {
                    commands::daemon::run_agent_delete(&config, &id, json).await
                }
                DaemonAgentCommands::BindHarness {
                    id,
                    harness_id,
                    config,
                    json,
                } => {
                    commands::daemon::run_agent_bind_harness(&config, &id, &harness_id, json).await
                }
                DaemonAgentCommands::UseLocalHarness { id, config, json } => {
                    commands::daemon::run_agent_use_local_harness(&config, &id, json).await
                }
            },
            DaemonCommands::Task { command } => match command {
                DaemonTaskCommands::Submit {
                    agent_id,
                    prompt,
                    wait,
                    timeout_ms,
                    config,
                    json,
                } => {
                    commands::daemon::run_task_submit(
                        &config, &agent_id, &prompt, wait, timeout_ms, json,
                    )
                    .await
                }
                DaemonTaskCommands::Get {
                    request_id,
                    config,
                    json,
                } => commands::daemon::run_task_get(&config, &request_id, json).await,
                DaemonTaskCommands::Wait {
                    request_id,
                    timeout_ms,
                    config,
                    json,
                } => commands::daemon::run_task_wait(&config, &request_id, timeout_ms, json).await,
                DaemonTaskCommands::Cancel {
                    request_id,
                    config,
                    json,
                } => commands::daemon::run_task_cancel(&config, &request_id, json).await,
                DaemonTaskCommands::List { config, json } => {
                    commands::daemon::run_task_list(&config, json).await
                }
            },
            DaemonCommands::Harness { command } => match command {
                DaemonHarnessCommands::List { config, json } => {
                    commands::daemon::run_harness_list(&config, json).await
                }
                DaemonHarnessCommands::Create { id, config, json } => {
                    commands::daemon::run_harness_create(&config, &id, json).await
                }
                DaemonHarnessCommands::Get { id, config, json } => {
                    commands::daemon::run_harness_get(&config, &id, json).await
                }
                DaemonHarnessCommands::Issues { id, config, json } => {
                    commands::daemon::run_harness_issues(&config, &id, json).await
                }
                DaemonHarnessCommands::Reload { id, config, json } => {
                    commands::daemon::run_harness_reload(&config, &id, json).await
                }
                DaemonHarnessCommands::Validate { id, config, json } => {
                    commands::daemon::run_harness_validate(&config, &id, json).await
                }
                DaemonHarnessCommands::Delete { id, config, json } => {
                    commands::daemon::run_harness_delete(&config, &id, json).await
                }
            },
            DaemonCommands::Session { command } => match command {
                DaemonSessionCommands::List {
                    config,
                    limit,
                    offset,
                    json,
                } => commands::daemon::run_session_list(&config, limit, offset, json).await,
                DaemonSessionCommands::Get {
                    session_id,
                    config,
                    json,
                } => commands::daemon::run_session_get(&config, &session_id, json).await,
                DaemonSessionCommands::Cancel {
                    session_id,
                    config,
                    json,
                } => commands::daemon::run_session_cancel(&config, &session_id, json).await,
                DaemonSessionCommands::Kill {
                    session_id,
                    config,
                    json,
                } => commands::daemon::run_session_kill(&config, &session_id, json).await,
            },
        },
    }
}
