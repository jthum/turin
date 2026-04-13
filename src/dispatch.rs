use anyhow::{Context, Result};

use crate::cli::{
    Cli, Commands, DaemonAgentCommands, DaemonChannelCommands, DaemonCommands,
    DaemonHarnessCommands, DaemonSessionCommands, DaemonTaskCommands, HarnessCommands,
};
use crate::commands;
use crate::commands::harness::{HarnessNewArgs, HarnessTestArgs};
use crate::commands::init::{InitArgs, QuickstartArgs};
use turin::kernel::Kernel;

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

            let mut kernel = Kernel::builder(config).json_mode(false).build()?;
            kernel.init_state().await?;
            kernel.init_clients()?;
            kernel.init_harness().await?;

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

async fn handle_daemon_command(
    command: DaemonCommands,
    log_level: &str,
    log_file: Option<&std::path::Path>,
) -> Result<()> {
    match command {
        DaemonCommands::Start { args } => {
            commands::daemon::run_start(
                &args.config.config,
                args.background,
                args.wait_timeout_ms,
                args.json,
                log_level,
                log_file,
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
                log_level,
                log_file,
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
                log_file,
            )
            .await
        }
        DaemonCommands::Agent { command } => handle_daemon_agent_command(command).await,
        DaemonCommands::Task { command } => handle_daemon_task_command(command).await,
        DaemonCommands::Harness { command } => handle_daemon_harness_command(command).await,
        DaemonCommands::Channel { command } => handle_daemon_channel_command(command).await,
        DaemonCommands::Session { command } => handle_daemon_session_command(command).await,
    }
}

async fn handle_daemon_agent_command(command: DaemonAgentCommands) -> Result<()> {
    match command {
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
            commands::daemon::run_agent_use_local_harness(&args.config.config, &id, args.json).await
        }
    }
}

async fn handle_daemon_task_command(command: DaemonTaskCommands) -> Result<()> {
    match command {
        DaemonTaskCommands::Submit {
            agent_id,
            session_id,
            slot_id,
            prompt,
            wait,
            timeout_ms,
            args,
        } => {
            commands::daemon::run_task_submit(commands::daemon::TaskSubmitCommand {
                config_path: &args.config.config,
                agent_id: agent_id.as_deref(),
                session_id: session_id.as_deref(),
                slot_id: slot_id.as_deref(),
                prompt: &prompt,
                wait,
                timeout_ms,
                json_output: args.json,
            })
            .await
        }
        DaemonTaskCommands::Get { request_id, args } => {
            commands::daemon::run_task_get(&args.config.config, &request_id, args.json).await
        }
        DaemonTaskCommands::Wait {
            request_id,
            timeout_ms,
            args,
        } => {
            commands::daemon::run_task_wait(&args.config.config, &request_id, timeout_ms, args.json)
                .await
        }
        DaemonTaskCommands::Cancel { request_id, args } => {
            commands::daemon::run_task_cancel(&args.config.config, &request_id, args.json).await
        }
        DaemonTaskCommands::List { args } => {
            commands::daemon::run_task_list(&args.config.config, args.json).await
        }
    }
}

async fn handle_daemon_harness_command(command: DaemonHarnessCommands) -> Result<()> {
    match command {
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
            commands::daemon::run_harness_validate(&args.config.config, &id, args.json).await
        }
        DaemonHarnessCommands::Delete { id, args } => {
            commands::daemon::run_harness_delete(&args.config.config, &id, args.json).await
        }
    }
}

async fn handle_daemon_channel_command(command: DaemonChannelCommands) -> Result<()> {
    match command {
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
        DaemonChannelCommands::Access { id, args } => {
            commands::daemon::run_channel_access(&args.config.config, &id, args.json).await
        }
        DaemonChannelCommands::Approve {
            id,
            workspace_id,
            room_id,
            thread_id,
            args,
        } => {
            commands::daemon::run_channel_approve(
                &args.config.config,
                serde_json::json!({
                    "id": id,
                    "workspace_id": workspace_id,
                    "room_id": room_id,
                    "thread_id": thread_id,
                }),
                args.json,
            )
            .await
        }
        DaemonChannelCommands::Reject {
            id,
            workspace_id,
            room_id,
            thread_id,
            args,
        } => {
            commands::daemon::run_channel_reject(
                &args.config.config,
                serde_json::json!({
                    "id": id,
                    "workspace_id": workspace_id,
                    "room_id": room_id,
                    "thread_id": thread_id,
                }),
                args.json,
            )
            .await
        }
        DaemonChannelCommands::Revoke {
            id,
            workspace_id,
            room_id,
            thread_id,
            args,
        } => {
            commands::daemon::run_channel_revoke(
                &args.config.config,
                serde_json::json!({
                    "id": id,
                    "workspace_id": workspace_id,
                    "room_id": room_id,
                    "thread_id": thread_id,
                }),
                args.json,
            )
            .await
        }
        DaemonChannelCommands::Delete { id, args } => {
            commands::daemon::run_channel_delete(&args.config.config, &id, args.json).await
        }
    }
}

async fn handle_daemon_session_command(command: DaemonSessionCommands) -> Result<()> {
    match command {
        DaemonSessionCommands::List {
            limit,
            offset,
            store,
            args,
        } => {
            commands::daemon::run_session_list(
                &args.config.config,
                limit,
                offset,
                store.store.as_deref(),
                store.path.as_deref(),
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
            commands::daemon::run_session_get(&args.config.config, &session_id, args.json).await
        }
        DaemonSessionCommands::BranchList { session_id, args } => {
            commands::daemon::run_session_branch_list(&args.config.config, &session_id, args.json)
                .await
        }
        DaemonSessionCommands::BranchCreate {
            session_id,
            name,
            slot_id,
            from_turn,
            activate,
            args,
        } => {
            commands::daemon::run_session_branch_create(
                &args.config.config,
                &session_id,
                &name,
                slot_id.as_deref(),
                from_turn,
                activate,
                args.json,
            )
            .await
        }
        DaemonSessionCommands::BranchCheckout {
            session_id,
            branch,
            slot_id,
            args,
        } => {
            commands::daemon::run_session_branch_checkout(
                &args.config.config,
                &session_id,
                &branch,
                slot_id.as_deref(),
                args.json,
            )
            .await
        }
        DaemonSessionCommands::Cancel {
            session_id,
            slot_id,
            args,
        } => {
            commands::daemon::run_session_cancel(
                &args.config.config,
                &session_id,
                slot_id.as_deref(),
                args.json,
            )
            .await
        }
        DaemonSessionCommands::Kill {
            session_id,
            slot_id,
            args,
        } => {
            commands::daemon::run_session_kill(
                &args.config.config,
                &session_id,
                slot_id.as_deref(),
                args.json,
            )
            .await
        }
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

#[cfg(test)]
mod tests {
    use super::parse_cli_settings;
    use serde_json::json;

    #[test]
    fn parse_cli_settings_parses_json_and_strings() {
        let value = parse_cli_settings(&[
            "chat_id=123".to_string(),
            "token_env=TELEGRAM_BOT_TOKEN".to_string(),
            "enabled=true".to_string(),
        ])
        .expect("settings should parse");

        assert_eq!(
            value,
            json!({
                "chat_id": 123,
                "token_env": "TELEGRAM_BOT_TOKEN",
                "enabled": true,
            })
        );
    }

    #[test]
    fn parse_cli_settings_rejects_missing_separator() {
        let err = parse_cli_settings(&["broken".to_string()]).expect_err("should reject");
        assert!(err.to_string().contains("expected key=value"));
    }
}
