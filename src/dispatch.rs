#[path = "dispatch/daemon.rs"]
mod daemon;

use anyhow::{Context, Result};

use crate::cli::{Cli, Commands, HarnessCommands};
use crate::commands;
use crate::commands::harness::{HarnessNewArgs, HarnessTestArgs};
use crate::commands::init::{InitArgs, QuickstartArgs};
use daemon::handle_daemon_command;

pub(crate) async fn run(cli: Cli) -> Result<()> {
    let Cli {
        command,
        log_level,
        log_file,
    } = cli;

    match command {
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
        Commands::Script {
            path,
            config,
            model,
            provider,
        } => {
            let config =
                commands::common::load_config_with_overrides(&config, model, provider, None)?;

            let mut kernel = crate::composition::kernel_builder(config)
                .json_mode(false)
                .build()?;
            kernel.init_state().await?;
            kernel.init_clients()?;
            kernel.init_harness().await?;

            let script_content = std::fs::read_to_string(&path)
                .with_context(|| format!("Failed to read script: {}", path.display()))?;

            kernel.run_script(&script_content)?;
            kernel.shutdown().await;

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
        Commands::Check { config, json } => {
            commands::check::run_check(&config, json).await?;
            Ok(())
        }
        Commands::Doctor { config, json } => {
            commands::check::run_doctor(&config, json).await?;
            Ok(())
        }
        Commands::Harness { command } => handle_harness_command(command).await,
        Commands::Daemon { command } => {
            handle_daemon_command(command, &log_level, log_file.as_deref()).await
        }
    }
}

async fn handle_harness_command(command: HarnessCommands) -> Result<()> {
    match command {
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
    }
}
