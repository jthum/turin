use clap::Args;
use std::path::PathBuf;
use turin_types::layout::{DEFAULT_BOOTSTRAP_CONFIG_PATH, DEFAULT_LAYOUT_HARNESSES_DIR};

use crate::commands::scaffold::{GovernancePreset, HarnessTemplate, InitProvider};

/// Turin: An event-driven LLM execution runtime
#[derive(clap::Parser, Debug)]
#[command(name = "turin", version, about)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Commands,

    /// Log level (error, warn, info, debug, trace)
    #[arg(long, default_value = "info", global = true)]
    pub(crate) log_level: String,

    /// Path to log file
    #[arg(long, global = true)]
    pub(crate) log_file: Option<PathBuf>,
}

#[derive(clap::Subcommand, Debug)]
pub(crate) enum Commands {
    /// Run the agent with a prompt
    Run {
        /// The prompt to send to the LLM
        #[arg(long)]
        prompt: String,

        /// Path to Turin config file
        #[arg(long, default_value = DEFAULT_BOOTSTRAP_CONFIG_PATH)]
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
        /// Path to Turin config file
        #[arg(long, default_value = DEFAULT_BOOTSTRAP_CONFIG_PATH)]
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

        /// Path to Turin config file
        #[arg(long, default_value = DEFAULT_BOOTSTRAP_CONFIG_PATH)]
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
        /// Overwrite an existing Turin config / starter harness files
        #[arg(long)]
        force: bool,
        /// Skip prompts and accept defaults
        #[arg(long)]
        yes: bool,
    },

    /// Initialize a Turin project if needed and run a first prompt immediately
    Quickstart {
        /// Path to Turin config file
        #[arg(long, default_value = DEFAULT_BOOTSTRAP_CONFIG_PATH)]
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
        /// Overwrite an existing Turin config / starter harness files when scaffolding
        #[arg(long)]
        force: bool,
        /// Skip prompts and accept defaults when scaffolding
        #[arg(long)]
        yes: bool,
    },

    /// Validate configuration and harness scripts
    Check {
        /// Path to Turin config file
        #[arg(long, default_value = DEFAULT_BOOTSTRAP_CONFIG_PATH)]
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
pub(crate) enum DaemonCommands {
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
pub(crate) enum HarnessCommands {
    /// Create a starter harness template
    New {
        /// Template to scaffold
        #[arg(value_enum)]
        template: HarnessTemplate,
        /// Target harness directory
        #[arg(long, default_value = DEFAULT_LAYOUT_HARNESSES_DIR)]
        dir: PathBuf,
        /// Overwrite an existing file if the template uses the same path
        #[arg(long)]
        force: bool,
    },
    /// Run the configured harness against the mock provider
    Test {
        /// Path to Turin config file
        #[arg(long, default_value = DEFAULT_BOOTSTRAP_CONFIG_PATH)]
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
pub(crate) enum DaemonAgentCommands {
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
pub(crate) enum DaemonTaskCommands {
    /// Submit a task to a daemon-managed agent
    Submit {
        /// Agent ID
        #[arg(required_unless_present = "session_id")]
        agent_id: Option<String>,
        /// Existing live session ID to submit into
        #[arg(long)]
        session_id: Option<String>,
        /// Optional runtime slot ID when the session is live in multiple slots
        #[arg(long)]
        slot_id: Option<String>,
        /// Prompt to submit
        prompt: String,
        /// Conflict policy for stale branch-head writes
        #[arg(long, value_parser = ["reject", "detached", "fork_sibling"])]
        conflict_policy: Option<String>,
        /// Wait for the task to complete and print the terminal result
        #[arg(long)]
        wait: bool,
        /// Optional wait timeout in milliseconds (only meaningful with --wait)
        #[arg(long)]
        timeout_ms: Option<u64>,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Run a one-shot sidestep from a persisted session
    Sidestep {
        /// Persisted session ID to sidestep from
        session_id: String,
        /// Optional temporary runtime slot ID for the sidestep execution
        #[arg(long)]
        slot_id: Option<String>,
        /// Sidestep execution mode
        #[arg(long, default_value = "ephemeral", value_parser = ["ephemeral", "fork_sibling"])]
        mode: String,
        /// Target a specific branch head instead of the session's active branch
        #[arg(long, conflicts_with = "turn_id")]
        branch_head_id: Option<i64>,
        /// Target a specific turn as the sidestep context root
        #[arg(long, conflicts_with = "branch_head_id")]
        turn_id: Option<i64>,
        /// Optional wait timeout in milliseconds
        #[arg(long)]
        timeout_ms: Option<u64>,
        /// Prompt to submit
        prompt: String,
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
pub(crate) enum DaemonHarnessCommands {
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
pub(crate) enum DaemonChannelCommands {
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
    /// Show pending and approved access state for one channel
    Access {
        /// Channel ID
        id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Approve one discovered room/thread for a channel
    Approve {
        /// Channel ID
        id: String,
        /// Channel workspace identifier
        #[arg(long)]
        workspace_id: String,
        /// Optional room identifier when the channel distinguishes room and thread
        #[arg(long)]
        room_id: Option<String>,
        /// Thread identifier used by the channel conversation key
        #[arg(long)]
        thread_id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Reject one pending room/thread for a channel
    Reject {
        /// Channel ID
        id: String,
        /// Channel workspace identifier
        #[arg(long)]
        workspace_id: String,
        /// Optional room identifier when the channel distinguishes room and thread
        #[arg(long)]
        room_id: Option<String>,
        /// Thread identifier used by the channel conversation key
        #[arg(long)]
        thread_id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Revoke one previously approved room/thread for a channel
    Revoke {
        /// Channel ID
        id: String,
        /// Channel workspace identifier
        #[arg(long)]
        workspace_id: String,
        /// Optional room identifier when the channel distinguishes room and thread
        #[arg(long)]
        room_id: Option<String>,
        /// Thread identifier used by the channel conversation key
        #[arg(long)]
        thread_id: String,
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
pub(crate) enum DaemonSessionCommands {
    /// List recent persisted sessions
    List {
        /// Maximum number of sessions to return
        #[arg(long, default_value_t = 50)]
        limit: usize,
        /// Offset into the session list
        #[arg(long, default_value_t = 0)]
        offset: usize,
        #[command(flatten)]
        store: DaemonStoreFilterArgs,
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
    /// List persisted branches for one session
    BranchList {
        /// Session ID
        session_id: String,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Create one persisted branch for a session
    BranchCreate {
        /// Session ID
        session_id: String,
        /// Branch name
        name: String,
        /// Optional runtime slot ID when activating against a live session with multiple slots
        #[arg(long)]
        slot_id: Option<String>,
        /// Optional turn index to branch from; defaults to current active head
        #[arg(long)]
        from_turn: Option<u32>,
        /// Activate the new branch immediately
        #[arg(long)]
        activate: bool,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Check out one persisted branch for a stopped session
    BranchCheckout {
        /// Session ID
        session_id: String,
        /// Branch name or branch ID
        branch: String,
        /// Optional runtime slot ID when checking out against a live session with multiple slots
        #[arg(long)]
        slot_id: Option<String>,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Cooperatively cancel one active daemon session
    Cancel {
        /// Session ID
        session_id: String,
        /// Optional runtime slot ID when the session is live in multiple slots
        #[arg(long)]
        slot_id: Option<String>,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
    /// Force-kill one active daemon session
    Kill {
        /// Session ID
        session_id: String,
        /// Optional runtime slot ID when the session is live in multiple slots
        #[arg(long)]
        slot_id: Option<String>,
        #[command(flatten)]
        args: DaemonOutputArgs,
    },
}

#[derive(Args, Debug, Clone)]
pub(crate) struct DaemonConfigArgs {
    /// Path to Turin config file
    #[arg(long, default_value = DEFAULT_BOOTSTRAP_CONFIG_PATH)]
    pub(crate) config: PathBuf,
}

#[derive(Args, Debug, Clone)]
pub(crate) struct DaemonStartArgs {
    #[command(flatten)]
    pub(crate) config: DaemonConfigArgs,
    /// Start the daemon in the background and wait for readiness
    #[arg(long)]
    pub(crate) background: bool,
    /// When starting in the background, how long to wait for readiness
    #[arg(long, default_value_t = 5000)]
    pub(crate) wait_timeout_ms: u64,
    /// Output wrapper-friendly JSON when using --background
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Args, Debug, Clone)]
pub(crate) struct DaemonOutputArgs {
    #[command(flatten)]
    pub(crate) config: DaemonConfigArgs,
    /// Output JSON
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Args, Debug, Clone, Default)]
pub(crate) struct DaemonStoreFilterArgs {
    /// Named state/store alias to query for persisted sessions
    #[arg(long)]
    pub(crate) store: Option<String>,
    /// Explicit state DB path to query for persisted sessions
    #[arg(long)]
    pub(crate) path: Option<String>,
}

#[derive(Args, Debug, Clone)]
pub(crate) struct DaemonReadyArgs {
    #[command(flatten)]
    pub(crate) config: DaemonConfigArgs,
    /// Maximum time to wait for the daemon to become ready
    #[arg(long, default_value_t = 5000)]
    pub(crate) timeout_ms: u64,
    /// Poll interval used while waiting for readiness
    #[arg(long, default_value_t = 100)]
    pub(crate) poll_interval_ms: u64,
    /// Output JSON
    #[arg(long)]
    pub(crate) json: bool,
}

#[derive(Args, Debug, Clone)]
pub(crate) struct DaemonLogsArgs {
    #[command(flatten)]
    pub(crate) config: DaemonConfigArgs,
    /// Output JSON
    #[arg(long)]
    pub(crate) json: bool,
    /// Show only the resolved daemon log path
    #[arg(long)]
    pub(crate) path_only: bool,
    /// Number of trailing log lines to show
    #[arg(long, default_value_t = 40)]
    pub(crate) lines: usize,
}
