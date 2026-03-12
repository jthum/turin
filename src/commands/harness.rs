use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

use turin::display;
use turin::kernel::config::ProviderConfig;

use crate::commands::common;
use crate::commands::scaffold::{HarnessTemplate, scaffold_harness_template};

#[derive(Clone, Debug)]
pub struct HarnessNewArgs {
    pub template: HarnessTemplate,
    pub dir: PathBuf,
    pub force: bool,
}

#[derive(Clone, Debug)]
pub struct HarnessTestArgs {
    pub config: PathBuf,
    pub dir: Option<PathBuf>,
    pub agent: Option<String>,
    pub prompt: String,
    pub response: String,
}

pub fn run_harness_new(args: HarnessNewArgs) -> Result<()> {
    let ansi = display::stdout_ansi();
    let written = scaffold_harness_template(&args.dir, args.template, args.force)?;

    println!("{}", display::header("Created harness template", ansi));
    println!(
        "{} Template: {}",
        display::ok_mark(ansi),
        display::paint(args.template.name(), "34", ansi)
    );
    println!(
        "{} Directory: {}",
        display::ok_mark(ansi),
        display::paint(&args.dir.display().to_string(), "34", ansi)
    );
    println!("{} Files:", display::ok_mark(ansi));
    for path in written {
        println!("  - {}", path.display());
    }
    println!(
        "\nTry {} to validate the harness with the mock provider.",
        display::paint("turin harness test", "34", ansi)
    );

    Ok(())
}

pub async fn run_harness_test(args: HarnessTestArgs) -> Result<()> {
    let mut config = common::load_config_with_overrides(
        &args.config,
        Some("mock-model".to_string()),
        Some("mock".to_string()),
        args.agent.as_deref(),
    )?;
    let selected_agent_id = args
        .agent
        .clone()
        .unwrap_or_else(|| config.agent.id.clone());

    if let Some(dir) = args.dir.as_ref() {
        override_harness_dir(&mut config, &selected_agent_id, dir)?;
    }

    config.providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some(args.response.clone()),
            headers: Default::default(),
            max_retries: None,
            request_timeout_secs: None,
            total_timeout_secs: None,
        },
    );

    let harness_dir = harness_dir_for_agent(&config, &selected_agent_id)?.to_string();
    common::run_prompt_once(config, args.prompt, Some(selected_agent_id.clone()), false)
        .await
        .with_context(|| {
            format!(
                "harness test failed for agent '{}' using '{}'",
                selected_agent_id, harness_dir
            )
        })
}

fn override_harness_dir(
    config: &mut turin::kernel::config::TurinConfig,
    agent_id: &str,
    dir: &Path,
) -> Result<()> {
    let harness_id = if agent_id == config.agent.id {
        config.harness_id_for_agent(&config.agent).to_string()
    } else {
        let agent = config
            .agents
            .get(agent_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown agent profile: {}", agent_id))?;
        config.harness_id_for_agent(agent).to_string()
    };

    if harness_id == "default" {
        config.harness.directory = dir.display().to_string();
    } else {
        let harness = config
            .harnesses
            .get_mut(&harness_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown harness id: {}", harness_id))?;
        harness.directory = dir.display().to_string();
    }

    Ok(())
}

fn harness_dir_for_agent<'a>(
    config: &'a turin::kernel::config::TurinConfig,
    agent_id: &str,
) -> Result<&'a str> {
    let agent = if agent_id == config.agent.id {
        &config.agent
    } else {
        config
            .agents
            .get(agent_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown agent profile: {}", agent_id))?
    };
    let (_, harness) = config.harness_binding_for_agent(agent)?;
    Ok(&harness.directory)
}
