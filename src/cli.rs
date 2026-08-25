#[path = "cli/daemon.rs"]
mod daemon;

use std::path::PathBuf;
use turin_types::layout::{DEFAULT_BOOTSTRAP_CONFIG_PATH, DEFAULT_LAYOUT_HARNESSES_DIR};

use crate::commands::scaffold::{GovernancePreset, HarnessTemplate, InitProvider};

pub(crate) use daemon::*;

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
        config: PathBuf,
        /// Output a machine-readable diagnostic report
        #[arg(long)]
        json: bool,
    },

    /// Check project validity and local daemon readiness
    Doctor {
        /// Path to Turin config file
        #[arg(long, default_value = DEFAULT_BOOTSTRAP_CONFIG_PATH)]
        config: PathBuf,
        /// Output a machine-readable diagnostic report
        #[arg(long)]
        json: bool,
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

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn parse_reference_diagnostic_commands() {
        let check = Cli::try_parse_from(["turin", "check", "--config", "project.toml", "--json"])
            .expect("check command should parse");
        assert!(matches!(
            check.command,
            Commands::Check { config, json }
                if config.as_path() == std::path::Path::new("project.toml") && json
        ));

        let doctor = Cli::try_parse_from(["turin", "doctor", "--config", "project.toml", "--json"])
            .expect("doctor command should parse");
        assert!(matches!(
            doctor.command,
            Commands::Doctor { config, json }
                if config.as_path() == std::path::Path::new("project.toml") && json
        ));
    }
}
