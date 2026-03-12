use anyhow::{Context, Result};
use clap::{Args, Parser};
use std::path::PathBuf;

mod commands;
use commands::harness::{HarnessNewArgs, HarnessTestArgs};
use commands::init::{InitArgs, QuickstartArgs};
use commands::scaffold::{GovernancePreset, HarnessTemplate, InitProvider};
use turin::kernel::Kernel;

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
    Init {
        /// Starter provider profile for the generated config
        #[arg(long, value_enum)]
        provider: Option<InitProvider>,
        /// Override the default starter model for the chosen provider
        #[arg(long)]
        model: Option<String>,
        /// Initial harness template
        #[arg(long, value_enum)]
        harness_template: Option<HarnessTemplate>,
        /// Governance preset for the generated config
        #[arg(long, value_enum)]
        governance: Option<GovernancePreset>,
        /// Overwrite an existing turin.toml / starter harness files
        #[arg(long)]
        force: bool,
        /// Skip prompts and accept defaults
        #[arg(long)]
        yes: bool,
    },

    /// Initialize a Turin project if needed and run a first prompt immediately
    Quickstart {
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Prompt to run after scaffolding or loading config
        #[arg(long)]
        prompt: Option<String>,
        /// Starter provider profile when scaffolding a new project
        #[arg(long, value_enum)]
        provider: Option<InitProvider>,
        /// Override the default starter model for the chosen provider
        #[arg(long)]
        model: Option<String>,
        /// Initial harness template when scaffolding a new project
        #[arg(long, value_enum)]
        harness_template: Option<HarnessTemplate>,
        /// Governance preset when scaffolding a new project
        #[arg(long, value_enum)]
        governance: Option<GovernancePreset>,
        /// Overwrite an existing turin.toml / starter harness files when scaffolding
        #[arg(long)]
        force: bool,
        /// Skip prompts and accept defaults when scaffolding
        #[arg(long)]
        yes: bool,
    },

    /// Validate configuration and harness scripts
    Check {
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: std::path::PathBuf,
    },

    /// Scaffold and validate harness scripts
    Harness {
        #[command(subcommand)]
        command: HarnessCommands,
    },

    /// Run or control the Turin daemon
    Daemon {
        #[command(subcommand)]
        command: DaemonCommands,
    },
}

#[derive(clap::Subcommand, Debug)]
enum DaemonCommands {
    /// Start the daemon
    Start {
        #[command(flatten)]
        args: DaemonStartArgs,
    },
    /// Ping the daemon
    Ping {
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Show wrapper-friendly daemon health
    Health {
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Show daemon status
    Status {
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Wait for the daemon to become ready
    Wait {
        #[command(flatten)]
        args: DaemonReadyArgs,
    },
    /// Ensure the daemon is running, starting it in the background if needed
    Ensure {
        #[command(flatten)]
        args: DaemonReadyArgs,
    },
    /// Rescan filesystem-backed daemon state
    Rescan {
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Reload filesystem-backed daemon state
    Reload {
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Show isolated runtime loading errors
    Errors {
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Stop the daemon
    Stop {
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Tail daemon runtime events
    Events {
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Inspect daemon log output
    Logs {
        #[command(flatten)]
        args: DaemonLogsArgs,
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
    /// Inspect daemon-managed channels
    Channel {
        #[command(subcommand)]
        command: DaemonChannelCommands,
    },
    /// Inspect persisted daemon sessions
    Session {
        #[command(subcommand)]
        command: DaemonSessionCommands,
    },
}

#[derive(clap::Subcommand, Debug)]
enum HarnessCommands {
    /// Create a starter harness template
    New {
        /// Template to scaffold
        #[arg(value_enum)]
        template: HarnessTemplate,
        /// Target harness directory
        #[arg(long, default_value = ".turin/harnesses")]
        dir: PathBuf,
        /// Overwrite an existing file if the template uses the same path
        #[arg(long)]
        force: bool,
    },
    /// Run the configured harness against the mock provider
    Test {
        /// Path to turin.toml config file
        #[arg(long, default_value = "turin.toml")]
        config: PathBuf,
        /// Override the harness directory just for this test run
        #[arg(long)]
        dir: Option<PathBuf>,
        /// Run the test against a named configured agent
        #[arg(long)]
        agent: Option<String>,
        /// Prompt sent into the mock-backed run
        #[arg(
            long,
            default_value = "Summarize this workspace and mention the active harness files."
        )]
        prompt: String,
        /// Mock provider response returned by the test run
        #[arg(long, default_value = "Harness test OK.")]
        response: String,
    },
}

#[derive(clap::Subcommand, Debug)]
enum DaemonAgentCommands {
    /// List daemon-managed agents
    List {
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Show one daemon-managed agent
    Get {
        /// Agent ID
        id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Show live runtime status for one daemon-managed agent
    Status {
        /// Agent ID
        id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Show isolated daemon issues for one agent directory
    Issues {
        /// Agent ID
        id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Reload one daemon-managed agent from disk
    Reload {
        /// Agent ID
        id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
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
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Enable a daemon-managed agent
    Enable {
        /// Agent ID
        id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
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
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Disable a daemon-managed agent
    Disable {
        /// Agent ID
        id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Delete a daemon-managed agent directory
    Delete {
        /// Agent ID
        id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Bind an agent to a shared harness
    BindHarness {
        /// Agent ID
        id: String,
        /// Shared harness ID
        harness_id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Switch an agent back to a local harness
    UseLocalHarness {
        /// Agent ID
        id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
}

#[derive(clap::Subcommand, Debug)]
enum DaemonTaskCommands {
    /// Submit a task to a daemon-managed agent
    Submit {
        /// Agent ID
        #[arg(required_unless_present = "session_id")]
        agent_id: Option<String>,
        /// Existing live session ID to submit into
        #[arg(long)]
        session_id: Option<String>,
        /// Prompt to submit
        prompt: String,
        /// Wait for the task to complete and print the terminal result
        #[arg(long)]
        wait: bool,
        /// Optional wait timeout in milliseconds (only meaningful with --wait)
        #[arg(long)]
        timeout_ms: Option<u64>,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Show one daemon task by request ID
    Get {
        /// Request ID returned by task submission
        request_id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Wait for one daemon task to complete
    Wait {
        /// Request ID returned by task submission
        request_id: String,
        /// Optional timeout in milliseconds
        #[arg(long)]
        timeout_ms: Option<u64>,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Cancel one daemon task by request ID
    Cancel {
        /// Request ID returned by task submission
        request_id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// List daemon tasks
    List {
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
}

#[derive(clap::Subcommand, Debug)]
enum DaemonHarnessCommands {
    /// List daemon-visible harness runtimes
    List {
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Create a shared harness directory
    Create {
        /// Harness ID
        id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Show one harness by ID
    Get {
        /// Harness ID
        id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Show isolated daemon issues for one harness directory
    Issues {
        /// Harness ID
        id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Reload one harness by ID
    Reload {
        /// Harness ID
        id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Validate one harness by ID without mutating the live runtime
    Validate {
        /// Harness ID
        id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Delete a shared harness directory
    Delete {
        /// Harness ID
        id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
}

#[derive(clap::Subcommand, Debug)]
enum DaemonChannelCommands {
    /// List daemon-managed channels
    List {
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Create a daemon-managed channel directory
    Create {
        /// Channel ID / directory name
        id: String,
        /// Channel kind
        #[arg(long)]
        kind: String,
        /// Bound agent ID
        #[arg(long)]
        agent: String,
        /// Optional idle session TTL in seconds
        #[arg(long)]
        idle_ttl_secs: Option<u64>,
        /// Create the channel disabled
        #[arg(long)]
        disabled: bool,
        /// Channel-specific setting in key=value form; value is parsed as JSON when possible
        #[arg(long = "setting", value_name = "KEY=VALUE")]
        settings: Vec<String>,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Show one daemon-managed channel
    Get {
        /// Channel ID
        id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Show one live daemon channel runtime status
    Status {
        /// Channel ID
        id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Show isolated daemon issues for one channel directory
    Issues {
        /// Channel ID
        id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Enable one daemon-managed channel
    Enable {
        /// Channel ID
        id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Disable one daemon-managed channel
    Disable {
        /// Channel ID
        id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Update one daemon-managed channel
    Update {
        /// Channel ID
        id: String,
        /// Optional replacement channel kind
        #[arg(long)]
        kind: Option<String>,
        /// Optional replacement bound agent ID
        #[arg(long)]
        agent: Option<String>,
        /// Optional replacement idle TTL in seconds
        #[arg(long)]
        idle_ttl_secs: Option<u64>,
        /// Replace channel-specific settings with the provided key=value entries
        #[arg(long = "setting", value_name = "KEY=VALUE")]
        settings: Vec<String>,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Delete one daemon-managed channel directory
    Delete {
        /// Channel ID
        id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
}

#[derive(clap::Subcommand, Debug)]
enum DaemonSessionCommands {
    /// List recent persisted sessions
    List {
        /// Maximum number of sessions to return
        #[arg(long, default_value_t = 50)]
        limit: usize,
        /// Offset into the session list
        #[arg(long, default_value_t = 0)]
        offset: usize,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// List current live daemon-managed sessions
    Live {
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Open or reuse a live session slot for an agent
    Open {
        /// Agent ID
        agent_id: String,
        /// Optional custom slot ID
        #[arg(long)]
        slot_id: Option<String>,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Resume a persisted session into a live daemon session slot
    Resume {
        /// Persisted session ID
        session_id: String,
        /// Optional custom slot ID
        #[arg(long)]
        slot_id: Option<String>,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Show one persisted session by session ID
    Get {
        /// Session ID
        session_id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Cooperatively cancel one active daemon session
    Cancel {
        /// Session ID
        session_id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Force-kill one active daemon session
    Kill {
        /// Session ID
        session_id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
}

#[derive(Args, Debug, Clone)]
struct DaemonConfigArgs {
    /// Path to turin.toml config file
    #[arg(long, default_value = "turin.toml")]
    config: PathBuf,
}

#[derive(Args, Debug, Clone)]
struct DaemonStartArgs {
    #[command(flatten)]
    config: DaemonConfigArgs,
    /// Start the daemon in the background and wait for readiness
    #[arg(long)]
    background: bool,
    /// When starting in the background, how long to wait for readiness
    #[arg(long, default_value_t = 5000)]
    wait_timeout_ms: u64,
    /// Output wrapper-friendly JSON when using --background
    #[arg(long)]
    json: bool,
}

#[derive(Args, Debug, Clone)]
struct DaemonOutputArgs {
    #[command(flatten)]
    config: DaemonConfigArgs,
    /// Output JSON
    #[arg(long)]
    json: bool,
}

#[derive(Args, Debug, Clone)]
struct DaemonReadyArgs {
    #[command(flatten)]
    config: DaemonConfigArgs,
    /// Maximum time to wait for the daemon to become ready
    #[arg(long, default_value_t = 5000)]
    timeout_ms: u64,
    /// Poll interval used while waiting for readiness
    #[arg(long, default_value_t = 100)]
    poll_interval_ms: u64,
    /// Output JSON
    #[arg(long)]
    json: bool,
}

#[derive(Args, Debug, Clone)]
struct DaemonLogsArgs {
    #[command(flatten)]
    config: DaemonConfigArgs,
    /// Output JSON
    #[arg(long)]
    json: bool,
    /// Show only the resolved daemon log path
    #[arg(long)]
    path_only: bool,
    /// Number of trailing log lines to show
    #[arg(long, default_value_t = 40)]
    lines: usize,
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

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(&cli.log_level, cli.log_file.clone())?;

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
            let config = commands::common::load_config_with_overrides(
                &config,
                model,
                provider,
                agent.as_deref(),
            )?;
            commands::common::run_prompt_once(config, prompt, agent, json).await
        }
        Commands::Repl {
            config,
            model,
            provider,
            agent,
            verbose,
        } => {
            let config = commands::common::load_config_with_overrides(
                &config,
                model,
                provider,
                agent.as_deref(),
            )?;
            commands::repl::run_repl(config, verbose, agent).await
        }
        Commands::Script {
            path,
            config,
            model,
            provider,
        } => {
            let config =
                commands::common::load_config_with_overrides(&config, model, provider, None)?;

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
        Commands::Init {
            provider,
            model,
            harness_template,
            governance,
            force,
            yes,
        } => {
            commands::init::run_init(InitArgs {
                provider,
                model,
                harness_template,
                governance,
                force,
                yes,
            })?;
            Ok(())
        }
        Commands::Quickstart {
            config,
            prompt,
            provider,
            model,
            harness_template,
            governance,
            force,
            yes,
        } => {
            commands::init::run_quickstart(QuickstartArgs {
                config,
                prompt,
                provider,
                model,
                harness_template,
                governance,
                force,
                yes,
            })
            .await?;
            Ok(())
        }
        Commands::Check { config } => {
            commands::check::run_check(&config).await?;
            Ok(())
        }
        Commands::Harness { command } => match command {
            HarnessCommands::New {
                template,
                dir,
                force,
            } => {
                commands::harness::run_harness_new(HarnessNewArgs {
                    template,
                    dir,
                    force,
                })?;
                Ok(())
            }
            HarnessCommands::Test {
                config,
                dir,
                agent,
                prompt,
                response,
            } => {
                commands::harness::run_harness_test(HarnessTestArgs {
                    config,
                    dir,
                    agent,
                    prompt,
                    response,
                })
                .await?;
                Ok(())
            }
        },
        Commands::Daemon { command } => match command {
            DaemonCommands::Start { args } => {
                commands::daemon::run_start(
                    &args.config.config,
                    args.background,
                    args.wait_timeout_ms,
                    args.json,
                    &cli.log_level,
                    cli.log_file.as_deref(),
                )
                .await
            }
            DaemonCommands::Ping { args } => {
                commands::daemon::run_ping(&args.config.config, args.json).await
            }
            DaemonCommands::Health { args } => {
                commands::daemon::run_health(&args.config.config, args.json).await
            }
            DaemonCommands::Status { args } => {
                commands::daemon::run_status(&args.config.config, args.json).await
            }
            DaemonCommands::Wait { args } => {
                commands::daemon::run_wait(
                    &args.config.config,
                    args.timeout_ms,
                    args.poll_interval_ms,
                    args.json,
                )
                .await
            }
            DaemonCommands::Ensure { args } => {
                commands::daemon::run_ensure(
                    &args.config.config,
                    args.timeout_ms,
                    args.poll_interval_ms,
                    args.json,
                    &cli.log_level,
                    cli.log_file.as_deref(),
                )
                .await
            }
            DaemonCommands::Rescan { args } => {
                commands::daemon::run_rescan(&args.config.config, args.json).await
            }
            DaemonCommands::Reload { args } => {
                commands::daemon::run_reload(&args.config.config, args.json).await
            }
            DaemonCommands::Errors { args } => {
                commands::daemon::run_runtime_errors(&args.config.config, args.json).await
            }
            DaemonCommands::Stop { args } => {
                commands::daemon::run_stop(&args.config.config, args.json).await
            }
            DaemonCommands::Events { args } => {
                commands::daemon::run_events(&args.config.config, args.json).await
            }
            DaemonCommands::Logs { args } => {
                commands::daemon::run_logs(
                    &args.config.config,
                    args.lines,
                    args.path_only,
                    args.json,
                    cli.log_file.as_deref(),
                )
                .await
            }
            DaemonCommands::Agent { command } => match command {
                DaemonAgentCommands::List { args } => {
                    commands::daemon::run_agent_list(&args.config.config, args.json).await
                }
                DaemonAgentCommands::Get { id, args } => {
                    commands::daemon::run_agent_get(&args.config.config, &id, args.json).await
                }
                DaemonAgentCommands::Status { id, args } => {
                    commands::daemon::run_agent_status(&args.config.config, &id, args.json).await
                }
                DaemonAgentCommands::Issues { id, args } => {
                    commands::daemon::run_agent_issues(&args.config.config, &id, args.json).await
                }
                DaemonAgentCommands::Reload { id, args } => {
                    commands::daemon::run_agent_reload(&args.config.config, &id, args.json).await
                }
                DaemonAgentCommands::Create {
                    id,
                    provider,
                    model,
                    system_prompt,
                    harness,
                    disabled,
                    args,
                } => {
                    commands::daemon::run_agent_create(
                        &args.config.config,
                        serde_json::json!({
                            "id": id,
                            "provider": provider,
                            "model": model,
                            "system_prompt": system_prompt,
                            "harness": harness,
                            "enabled": !disabled,
                        }),
                        args.json,
                    )
                    .await
                }
                DaemonAgentCommands::Enable { id, args } => {
                    commands::daemon::run_agent_enable(&args.config.config, &id, args.json).await
                }
                DaemonAgentCommands::Update {
                    id,
                    provider,
                    model,
                    system_prompt,
                    args,
                } => {
                    commands::daemon::run_agent_update(
                        &args.config.config,
                        serde_json::json!({
                            "id": id,
                            "provider": provider,
                            "model": model,
                            "system_prompt": system_prompt,
                        }),
                        args.json,
                    )
                    .await
                }
                DaemonAgentCommands::Disable { id, args } => {
                    commands::daemon::run_agent_disable(&args.config.config, &id, args.json).await
                }
                DaemonAgentCommands::Delete { id, args } => {
                    commands::daemon::run_agent_delete(&args.config.config, &id, args.json).await
                }
                DaemonAgentCommands::BindHarness {
                    id,
                    harness_id,
                    args,
                } => {
                    commands::daemon::run_agent_bind_harness(
                        &args.config.config,
                        &id,
                        &harness_id,
                        args.json,
                    )
                    .await
                }
                DaemonAgentCommands::UseLocalHarness { id, args } => {
                    commands::daemon::run_agent_use_local_harness(
                        &args.config.config,
                        &id,
                        args.json,
                    )
                    .await
                }
            },
            DaemonCommands::Task { command } => match command {
                DaemonTaskCommands::Submit {
                    agent_id,
                    session_id,
                    prompt,
                    wait,
                    timeout_ms,
                    args,
                } => {
                    commands::daemon::run_task_submit(
                        &args.config.config,
                        agent_id.as_deref(),
                        session_id.as_deref(),
                        &prompt,
                        wait,
                        timeout_ms,
                        args.json,
                    )
                    .await
                }
                DaemonTaskCommands::Get { request_id, args } => {
                    commands::daemon::run_task_get(&args.config.config, &request_id, args.json)
                        .await
                }
                DaemonTaskCommands::Wait {
                    request_id,
                    timeout_ms,
                    args,
                } => {
                    commands::daemon::run_task_wait(
                        &args.config.config,
                        &request_id,
                        timeout_ms,
                        args.json,
                    )
                    .await
                }
                DaemonTaskCommands::Cancel { request_id, args } => {
                    commands::daemon::run_task_cancel(&args.config.config, &request_id, args.json)
                        .await
                }
                DaemonTaskCommands::List { args } => {
                    commands::daemon::run_task_list(&args.config.config, args.json).await
                }
            },
            DaemonCommands::Harness { command } => match command {
                DaemonHarnessCommands::List { args } => {
                    commands::daemon::run_harness_list(&args.config.config, args.json).await
                }
                DaemonHarnessCommands::Create { id, args } => {
                    commands::daemon::run_harness_create(&args.config.config, &id, args.json).await
                }
                DaemonHarnessCommands::Get { id, args } => {
                    commands::daemon::run_harness_get(&args.config.config, &id, args.json).await
                }
                DaemonHarnessCommands::Issues { id, args } => {
                    commands::daemon::run_harness_issues(&args.config.config, &id, args.json).await
                }
                DaemonHarnessCommands::Reload { id, args } => {
                    commands::daemon::run_harness_reload(&args.config.config, &id, args.json).await
                }
                DaemonHarnessCommands::Validate { id, args } => {
                    commands::daemon::run_harness_validate(&args.config.config, &id, args.json)
                        .await
                }
                DaemonHarnessCommands::Delete { id, args } => {
                    commands::daemon::run_harness_delete(&args.config.config, &id, args.json).await
                }
            },
            DaemonCommands::Channel { command } => match command {
                DaemonChannelCommands::List { args } => {
                    commands::daemon::run_channel_list(&args.config.config, args.json).await
                }
                DaemonChannelCommands::Create {
                    id,
                    kind,
                    agent,
                    idle_ttl_secs,
                    disabled,
                    settings,
                    args,
                } => {
                    commands::daemon::run_channel_create(
                        &args.config.config,
                        serde_json::json!({
                            "id": id,
                            "kind": kind,
                            "agent_id": agent,
                            "idle_ttl_secs": idle_ttl_secs,
                            "enabled": !disabled,
                            "settings": parse_cli_settings(&settings)?,
                        }),
                        args.json,
                    )
                    .await
                }
                DaemonChannelCommands::Get { id, args } => {
                    commands::daemon::run_channel_get(&args.config.config, &id, args.json).await
                }
                DaemonChannelCommands::Status { id, args } => {
                    commands::daemon::run_channel_status(&args.config.config, &id, args.json).await
                }
                DaemonChannelCommands::Issues { id, args } => {
                    commands::daemon::run_channel_issues(&args.config.config, &id, args.json).await
                }
                DaemonChannelCommands::Enable { id, args } => {
                    commands::daemon::run_channel_enable(&args.config.config, &id, args.json).await
                }
                DaemonChannelCommands::Disable { id, args } => {
                    commands::daemon::run_channel_disable(&args.config.config, &id, args.json).await
                }
                DaemonChannelCommands::Update {
                    id,
                    kind,
                    agent,
                    idle_ttl_secs,
                    settings,
                    args,
                } => {
                    let settings = if settings.is_empty() {
                        None
                    } else {
                        Some(parse_cli_settings(&settings)?)
                    };
                    commands::daemon::run_channel_update(
                        &args.config.config,
                        serde_json::json!({
                            "id": id,
                            "kind": kind,
                            "agent_id": agent,
                            "idle_ttl_secs": idle_ttl_secs,
                            "settings": settings,
                        }),
                        args.json,
                    )
                    .await
                }
                DaemonChannelCommands::Delete { id, args } => {
                    commands::daemon::run_channel_delete(&args.config.config, &id, args.json).await
                }
            },
            DaemonCommands::Session { command } => match command {
                DaemonSessionCommands::List {
                    limit,
                    offset,
                    args,
                } => {
                    commands::daemon::run_session_list(
                        &args.config.config,
                        limit,
                        offset,
                        args.json,
                    )
                    .await
                }
                DaemonSessionCommands::Live { args } => {
                    commands::daemon::run_session_list_live(&args.config.config, args.json).await
                }
                DaemonSessionCommands::Open {
                    agent_id,
                    slot_id,
                    args,
                } => {
                    commands::daemon::run_session_open(
                        &args.config.config,
                        &agent_id,
                        slot_id.as_deref(),
                        args.json,
                    )
                    .await
                }
                DaemonSessionCommands::Resume {
                    session_id,
                    slot_id,
                    args,
                } => {
                    commands::daemon::run_session_resume(
                        &args.config.config,
                        &session_id,
                        slot_id.as_deref(),
                        args.json,
                    )
                    .await
                }
                DaemonSessionCommands::Get { session_id, args } => {
                    commands::daemon::run_session_get(&args.config.config, &session_id, args.json)
                        .await
                }
                DaemonSessionCommands::Cancel { session_id, args } => {
                    commands::daemon::run_session_cancel(
                        &args.config.config,
                        &session_id,
                        args.json,
                    )
                    .await
                }
                DaemonSessionCommands::Kill { session_id, args } => {
                    commands::daemon::run_session_kill(&args.config.config, &session_id, args.json)
                        .await
                }
            },
        },
    }
}

fn parse_cli_settings(entries: &[String]) -> Result<serde_json::Value> {
    let mut settings = serde_json::Map::new();
    for entry in entries {
        let (key, raw_value) = entry
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("Invalid setting '{}'; expected key=value", entry))?;
        if key.trim().is_empty() {
            anyhow::bail!("Invalid setting '{}'; key cannot be empty", entry);
        }
        let value = serde_json::from_str(raw_value)
            .unwrap_or_else(|_| serde_json::Value::String(raw_value.to_string()));
        settings.insert(key.to_string(), value);
    }
    Ok(serde_json::Value::Object(settings))
}
