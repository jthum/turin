use anyhow::{Context, Result};
use std::io::{self, IsTerminal, Write};
use std::path::Path;

use turin::display;

use crate::commands::common;
use crate::commands::scaffold::{
    GovernancePreset, HarnessTemplate, InitOptions, InitProvider, ScaffoldResult, scaffold_project,
};

#[derive(Clone, Debug)]
pub struct InitArgs {
    pub provider: Option<InitProvider>,
    pub model: Option<String>,
    pub harness_template: Option<HarnessTemplate>,
    pub governance: Option<GovernancePreset>,
    pub force: bool,
    pub yes: bool,
}

#[derive(Clone, Debug)]
pub struct QuickstartArgs {
    pub config: std::path::PathBuf,
    pub prompt: Option<String>,
    pub provider: Option<InitProvider>,
    pub model: Option<String>,
    pub harness_template: Option<HarnessTemplate>,
    pub governance: Option<GovernancePreset>,
    pub force: bool,
    pub yes: bool,
}

pub fn run_init(args: InitArgs) -> Result<()> {
    let ansi = display::stdout_ansi();
    let options = resolve_init_options(args, false)?;
    let summary = scaffold_project(Path::new("."), &options)?;

    println!("{}", display::header("Initialized Turin project", ansi));
    print_scaffold_summary(&summary, ansi);

    println!("\nNext steps:");
    match summary.provider.default_api_key_env() {
        Some(env_name) => {
            println!(
                "  1. Set {} before your first real run.",
                display::paint(env_name, "33", ansi)
            );
            println!(
                "  2. Try {}",
                display::paint(
                    "turin quickstart --prompt \"Summarize this workspace.\"",
                    "34",
                    ansi
                )
            );
        }
        None => {
            println!(
                "  1. Try {}",
                display::paint(
                    "turin quickstart --prompt \"Summarize this workspace.\"",
                    "34",
                    ansi
                )
            );
            println!(
                "  2. Replace [providers.mock] in turin.toml when you are ready for a real model."
            );
        }
    }
    println!(
        "  3. Add checked-in context such as {} or {} when it helps the harness.",
        display::paint("TURIN.md", "36", ansi),
        display::paint("CONSTRAINTS.md", "36", ansi)
    );

    Ok(())
}

pub async fn run_quickstart(args: QuickstartArgs) -> Result<()> {
    let ansi = display::stdout_ansi();
    let config_exists = args.config.exists();

    if !config_exists {
        let options = resolve_quickstart_options(&args)?;
        let root = args
            .config
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let summary = scaffold_project(&root, &options)?;
        println!("{}", display::header("Quickstart scaffolded", ansi));
        print_scaffold_summary(&summary, ansi);
        println!();
    }

    let config = common::load_config_with_overrides(&args.config, None, None, None)?;
    let prompt = args.prompt.unwrap_or_else(default_quickstart_prompt);
    let provider = config.agent.provider.clone();
    if provider == "mock" {
        println!(
            "{} Using the mock provider. Edit {} when you are ready for real inference.\n",
            display::info_mark(ansi),
            display::paint(&args.config.display().to_string(), "36", ansi)
        );
    }

    common::run_prompt_once(config, prompt, None, false).await
}

fn resolve_init_options(args: InitArgs, quickstart_defaults: bool) -> Result<InitOptions> {
    if args.yes || !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        return Ok(noninteractive_init_options(args, quickstart_defaults));
    }

    let provider = match args.provider {
        Some(provider) => provider,
        None => prompt_choice(
            "Inference provider",
            &[
                InitProvider::Anthropic,
                InitProvider::Openai,
                InitProvider::Mock,
            ],
            if quickstart_defaults {
                InitProvider::Mock
            } else {
                InitProvider::Anthropic
            },
            |choice| format!("{} ({})", choice.alias(), choice.description()),
        )?,
    };

    let default_model = args
        .model
        .clone()
        .unwrap_or_else(|| provider.default_model().to_string());
    let model = prompt_text("Model", &default_model)?;

    let harness_template = match args.harness_template {
        Some(template) => template,
        None => prompt_choice(
            "Harness template",
            &[
                HarnessTemplate::Starter,
                HarnessTemplate::Safety,
                HarnessTemplate::CodingAssistant,
                HarnessTemplate::Reviewer,
            ],
            if quickstart_defaults {
                HarnessTemplate::CodingAssistant
            } else {
                HarnessTemplate::CodingAssistant
            },
            |choice| format!("{} ({})", choice.name(), choice.description()),
        )?,
    };

    let governance = match args.governance {
        Some(governance) => governance,
        None => prompt_choice(
            "Governance preset",
            &[
                GovernancePreset::Open,
                GovernancePreset::Balanced,
                GovernancePreset::Governed,
            ],
            GovernancePreset::Balanced,
            |choice| format!("{} ({})", choice.profile(), choice.description()),
        )?,
    };

    Ok(InitOptions {
        provider,
        model,
        harness_template,
        governance,
        force: args.force,
    })
}

fn noninteractive_init_options(args: InitArgs, quickstart_defaults: bool) -> InitOptions {
    let provider = args.provider.unwrap_or(if quickstart_defaults {
        InitProvider::Mock
    } else {
        InitProvider::Anthropic
    });
    let model = args
        .model
        .unwrap_or_else(|| provider.default_model().to_string());

    InitOptions {
        provider,
        model,
        harness_template: args
            .harness_template
            .unwrap_or(HarnessTemplate::CodingAssistant),
        governance: args.governance.unwrap_or(GovernancePreset::Balanced),
        force: args.force,
    }
}

fn resolve_quickstart_options(args: &QuickstartArgs) -> Result<InitOptions> {
    resolve_init_options(
        InitArgs {
            provider: args.provider.or(Some(InitProvider::Mock)),
            model: args.model.clone(),
            harness_template: args
                .harness_template
                .or(Some(HarnessTemplate::CodingAssistant)),
            governance: args.governance.or(Some(GovernancePreset::Balanced)),
            force: args.force,
            yes: args.yes,
        },
        true,
    )
}

fn print_scaffold_summary(summary: &ScaffoldResult, ansi: bool) {
    println!(
        "{} Provider: {} ({})",
        display::ok_mark(ansi),
        display::paint(summary.provider.alias(), "34", ansi),
        summary.model
    );
    println!(
        "{} Harness template: {}",
        display::ok_mark(ansi),
        display::paint(summary.harness_template.name(), "34", ansi)
    );
    println!(
        "{} Governance preset: {}",
        display::ok_mark(ansi),
        display::paint(summary.governance.profile(), "34", ansi)
    );

    if !summary.created_paths.is_empty() {
        println!("{} Created:", display::ok_mark(ansi));
        for path in &summary.created_paths {
            println!("  - {}", path.display());
        }
    }

    if !summary.updated_paths.is_empty() {
        println!("{} Updated:", display::info_mark(ansi));
        for path in &summary.updated_paths {
            println!("  - {}", path.display());
        }
    }
}

fn prompt_text(label: &str, default: &str) -> Result<String> {
    print!("{label} [{default}]: ");
    io::stdout().flush().context("failed to flush prompt")?;
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .context("failed to read prompt")?;
    let trimmed = input.trim();
    if trimmed.is_empty() {
        Ok(default.to_string())
    } else {
        Ok(trimmed.to_string())
    }
}

fn prompt_choice<T, F>(label: &str, choices: &[T], default: T, render: F) -> Result<T>
where
    T: Copy + Eq,
    F: Fn(T) -> String,
{
    println!("{label}:");
    for (idx, choice) in choices.iter().enumerate() {
        let suffix = if *choice == default { " [default]" } else { "" };
        println!("  {}. {}{}", idx + 1, render(*choice), suffix);
    }

    loop {
        print!(
            "Select 1-{} [{}]: ",
            choices.len(),
            default_index(choices, default) + 1
        );
        io::stdout().flush().context("failed to flush prompt")?;
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .context("failed to read prompt")?;
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Ok(default);
        }
        if let Ok(index) = trimmed.parse::<usize>()
            && index >= 1
            && index <= choices.len()
        {
            return Ok(choices[index - 1]);
        }
        println!("Please enter a number between 1 and {}.", choices.len());
    }
}

fn default_index<T: Copy + Eq>(choices: &[T], default: T) -> usize {
    choices
        .iter()
        .position(|choice| *choice == default)
        .unwrap_or(0)
}

fn default_quickstart_prompt() -> String {
    "Summarize this workspace and tell me which harness files are active.".to_string()
}
