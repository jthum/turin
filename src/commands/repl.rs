use anyhow::Result;
use rustyline::DefaultEditor;
use rustyline::error::ReadlineError;

use turin::display;
use turin::inference::content::summarize_content_for_display;
use turin::inference::provider::InferenceRole;
use turin::kernel::Kernel;
use turin::kernel::config::TurinConfig;
use turin::kernel::session::SessionState;

use crate::commands::common::print_session_summary;

pub(crate) async fn run_repl(
    config: TurinConfig,
    verbose: bool,
    agent_id: Option<String>,
) -> Result<()> {
    let selected_agent_id = agent_id.unwrap_or_else(|| config.agent.id.clone());
    let selected_agent = if selected_agent_id == config.agent.id {
        &config.agent
    } else {
        config
            .agents
            .get(&selected_agent_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown agent profile: {}", selected_agent_id))?
    };

    tracing::info!(
        agent_id = %selected_agent_id,
        model = %selected_agent.model,
        provider = %selected_agent.provider,
        "Config loaded (REPL mode)"
    );

    let mut kernel = crate::composition::kernel_builder(config).build()?; // JSON not supported in REPL yet
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;
    kernel.start_watcher()?;

    let mut rl = DefaultEditor::new()?;
    tracing::info!("REPL started. Type 'exit' or Ctrl+D to quit.");
    if !verbose {
        println!("Turin REPL v{}", env!("CARGO_PKG_VERSION"));
        println!("Type 'exit' or Ctrl+D to quit. Type '/reload' to reload harness.");
    }

    let mut session = kernel.create_session_for_agent(&selected_agent_id).await;
    kernel.start_session(&mut session).await?;

    let ansi = display::stdout_ansi();
    loop {
        let readline = rl.readline(&display::repl_prompt(ansi));
        match readline {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                if line.starts_with('/') {
                    let should_continue = handle_slash_command(
                        &mut kernel,
                        &mut session,
                        line,
                        ansi,
                        selected_agent_id.as_str(),
                    )
                    .await?;
                    if !should_continue {
                        break;
                    }
                    continue;
                }

                if line.eq_ignore_ascii_case("exit") {
                    break;
                }

                let _ = rl.add_history_entry(line);
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
    kernel.shutdown().await;
    print_session_summary(&session);
    Ok(())
}

async fn handle_slash_command(
    kernel: &mut Kernel,
    session: &mut SessionState,
    line: &str,
    ansi: bool,
    selected_agent_id: &str,
) -> Result<bool> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    let cmd = parts[0].to_lowercase();

    match cmd.as_str() {
        "/status" => {
            println!("\n{}", display::header("Session Status", ansi));
            println!(
                "  {} {}",
                display::bold("Session ID:", ansi),
                session.identity.session_id()
            );
            println!(
                "  {}      {}",
                display::bold("Agent:", ansi),
                selected_agent_id
            );
            println!(
                "  {}   {}",
                display::bold("Provider:", ansi),
                active_agent_config(kernel, selected_agent_id)?.provider
            );
            println!(
                "  {}      {}",
                display::bold("Model:", ansi),
                active_agent_config(kernel, selected_agent_id)?.model
            );
            println!(
                "  {}      {}",
                display::bold("Turns:", ansi),
                session.turn_index
            );
            println!(
                "  {}     {} total ({} in, {} out)",
                display::bold("Tokens:", ansi),
                session.total_input_tokens + session.total_output_tokens,
                session.total_input_tokens,
                session.total_output_tokens
            );
            println!();
        }
        "/history" => {
            println!("\n{}", display::header("Message History", ansi));
            if session.history.is_empty() {
                println!("  (No messages yet)");
            }
            for (i, msg) in session.history.iter().enumerate() {
                let role_name = format!("{:?}", msg.role);
                let role_colored = match msg.role {
                    InferenceRole::User => display::paint(&format!("{:10}", role_name), "32", ansi),
                    InferenceRole::Assistant => {
                        display::paint(&format!("{:10}", role_name), "34", ansi)
                    }
                    InferenceRole::Tool => display::paint(&format!("{:10}", role_name), "33", ansi),
                };

                let mut content_summary = summarize_content_for_display(&msg.content);

                if content_summary.len() > 80 {
                    content_summary = format!("{}...", &content_summary[..77]);
                }
                let cleaned_summary = content_summary.replace('\n', " ");
                println!("  [{}] {}: {}", i, role_colored, cleaned_summary);
            }
            println!();
        }
        "/reload" => {
            tracing::info!("Reloading harness...");
            match kernel.reload_harness().await {
                Ok(_) => tracing::info!("Harness reloaded successfully."),
                Err(e) => tracing::error!(error = %e, "Failed to reload harness"),
            }
        }
        "/clear" => {
            session.history.clear();
            session.turn_index = 0;
            session.total_input_tokens = 0;
            session.total_output_tokens = 0;
            println!(
                "{} Session history and stats cleared.",
                display::paint("✓", "32;1", ansi)
            );
        }
        "/help" => {
            println!("\n{}", display::header("Available Commands", ansi));
            println!(
                "  {}   - Show session statistics",
                display::bold("/status", ansi)
            );
            println!(
                "  {}  - Show condensed message history",
                display::bold("/history", ansi)
            );
            println!(
                "  {}   - Reload harness scripts",
                display::bold("/reload", ansi)
            );
            println!(
                "  {}    - Clear session history and reset stats",
                display::bold("/clear", ansi)
            );
            println!(
                "  {}     - Show this help message",
                display::bold("/help", ansi)
            );
            println!("  {}     - Exit the REPL", display::bold("/quit", ansi));
            println!();
        }
        "/quit" | "/exit" => {
            return Ok(false);
        }
        _ => {
            let msg = format!("Unknown command: {}. Type /help for assistance.", cmd);
            println!("{}", display::paint(&msg, "31", ansi));
        }
    }

    Ok(true)
}

fn active_agent_config<'a>(
    kernel: &'a Kernel,
    selected_agent_id: &str,
) -> Result<&'a turin::kernel::config::AgentConfig> {
    if selected_agent_id == kernel.config().agent.id {
        Ok(&kernel.config().agent)
    } else {
        kernel
            .config()
            .agents
            .get(selected_agent_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown agent profile: {}", selected_agent_id))
    }
}
