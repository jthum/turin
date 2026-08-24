use anyhow::{Context, Result};
use std::path::Path;
use turin::display;
use turin::kernel::config::TurinConfig;
use turin::kernel::session::SessionState;

pub(crate) fn print_session_summary(session: &SessionState) {
    let ansi = display::stdout_ansi();
    println!("\n{}", display::header("Session Summary", ansi));
    println!(
        "  {}  {} ({} in, {} out)",
        display::bold("Total Tokens:", ansi),
        session.total_input_tokens + session.total_output_tokens,
        session.total_input_tokens,
        session.total_output_tokens
    );
    println!(
        "  {}         {}",
        display::bold("Turns:", ansi),
        session.turn_index
    );
}

pub(crate) fn load_config_with_overrides(
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

pub(crate) async fn run_prompt_once(
    config: TurinConfig,
    prompt: String,
    agent_id: Option<String>,
    json: bool,
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
    let (harness_id, harness_cfg) = config.harness_binding_for_agent(selected_agent)?;

    tracing::info!(
        agent_id = %selected_agent_id,
        model = %selected_agent.model,
        provider = %selected_agent.provider,
        workspace = %config.kernel.workspace_root,
        harness_id = %harness_id,
        harness_dir = %harness_cfg.directory,
        db = ?config.persistence.top_level_state_selector().ok(),
        "Config loaded"
    );

    let builder = crate::composition::kernel_builder(config).json_mode(json);
    let builder = if json {
        builder
    } else {
        crate::commands::tool_authorization::with_interactive_authorization(builder)
    };
    let mut kernel = builder.build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;
    kernel.start_watcher()?;
    let mut session = kernel.create_session_for_agent(&selected_agent_id).await;
    kernel.start_session(&mut session).await?;
    kernel.run(&mut session, Some(prompt)).await?;
    kernel.end_session(&mut session).await?;
    kernel.shutdown().await;
    if !json {
        print_session_summary(&session);
    }
    Ok(())
}
