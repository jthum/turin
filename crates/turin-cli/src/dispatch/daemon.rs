use anyhow::Result;

use crate::cli::{
    DaemonAgentCommands, DaemonCommands, DaemonHarnessCommands, DaemonSessionCommands,
    DaemonTaskCommands,
};
use crate::commands;

pub(super) async fn handle_daemon_command(
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
            commands::daemon::run_stop(
                &args.config.config,
                args.timeout_ms,
                args.poll_interval_ms,
                args.json,
            )
            .await
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
            conflict_policy,
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
                conflict_policy: conflict_policy.as_deref(),
                wait,
                timeout_ms,
                json_output: args.json,
            })
            .await
        }
        DaemonTaskCommands::Sidestep {
            session_id,
            slot_id,
            mode,
            branch_head_id,
            turn_id,
            timeout_ms,
            prompt,
            args,
        } => {
            commands::daemon::run_task_sidestep(commands::daemon::TaskSidestepCommand {
                config_path: &args.config.config,
                session_id: &session_id,
                slot_id: slot_id.as_deref(),
                mode: &mode,
                branch_head_id,
                turn_id,
                prompt: &prompt,
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
