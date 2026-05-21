use std::collections::BTreeMap;

use anyhow::{Context, Result};
use dialoguer::{Confirm, Input, Password, Select};
use serde::Serialize;
use turin_types::layout::{DEFAULT_LAYOUT_HARNESSES_DIR, default_layout_root_for_workspace};

use crate::files::{
    PlannedWrite, config_dir, confirm_and_write, default_workspace_root_for_missing_config,
    load_existing, merge_env_file,
};

use super::InitArgs;

#[derive(Debug, Clone, Copy)]
enum ProviderChoice {
    Anthropic,
    OpenAi,
    Mock,
}

impl ProviderChoice {
    fn name(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAi => "openai",
            Self::Mock => "mock",
        }
    }

    fn api_key_env(self) -> Option<&'static str> {
        match self {
            Self::Anthropic => Some("ANTHROPIC_API_KEY"),
            Self::OpenAi => Some("OPENAI_API_KEY"),
            Self::Mock => None,
        }
    }

    fn default_model(self) -> &'static str {
        match self {
            Self::Anthropic => "claude-sonnet-4-20250514",
            Self::OpenAi => "gpt-5.4",
            Self::Mock => "mock",
        }
    }
}

#[derive(Debug, Serialize)]
struct GeneratedTurinConfig {
    agent: GeneratedAgentConfig,
    kernel: GeneratedKernelConfig,
    persistence: GeneratedPersistenceConfig,
    harness: GeneratedHarnessConfig,
    providers: BTreeMap<String, GeneratedProviderConfig>,
}

#[derive(Debug, Serialize)]
struct GeneratedAgentConfig {
    id: String,
    system_prompt: String,
    model: String,
    provider: String,
    idle_timeout_seconds: u64,
}

#[derive(Debug, Serialize)]
struct GeneratedKernelConfig {
    workspace_root: String,
    max_turns: u32,
    heartbeat_interval_seconds: u32,
    initial_spawn_depth: u32,
}

#[derive(Debug, Serialize)]
struct GeneratedPersistenceConfig {
    state: GeneratedStoreTargetConfig,
}

#[derive(Debug, Serialize)]
struct GeneratedStoreTargetConfig {
    path: String,
}

#[derive(Debug, Serialize)]
struct GeneratedHarnessConfig {
    directory: String,
}

#[derive(Debug, Serialize)]
struct GeneratedProviderConfig {
    #[serde(rename = "type")]
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    api_key_env: Option<String>,
}

pub(crate) async fn run_init(args: InitArgs) -> Result<()> {
    let config_path = args.config;
    if config_path.exists()
        && !args.force
        && !Confirm::new()
            .with_prompt(format!(
                "'{}' already exists. Overwrite it?",
                config_path.display()
            ))
            .default(false)
            .interact()?
    {
        anyhow::bail!("Aborted without overwriting '{}'", config_path.display());
    }

    let provider = prompt_provider()?;
    let model: String = Input::new()
        .with_prompt("Default model")
        .default(provider.default_model().to_string())
        .interact_text()?;
    let system_prompt: String = Input::new()
        .with_prompt("Default system prompt")
        .default("You are a helpful coding assistant.".to_string())
        .interact_text()?;

    let api_key = if let Some(env_var) = provider.api_key_env() {
        if Confirm::new()
            .with_prompt(format!("Store {env_var} in .env now?"))
            .default(true)
            .interact()?
        {
            let value = Password::new()
                .with_prompt(format!("Enter {env_var}"))
                .allow_empty_password(false)
                .interact()?;
            Some((env_var.to_string(), value))
        } else {
            None
        }
    } else {
        None
    };

    let config_body = generate_turin_config(provider, &model, &system_prompt)?;
    let harness_path =
        default_layout_root_for_workspace(&default_workspace_root_for_missing_config(&config_path))
            .join(DEFAULT_LAYOUT_HARNESSES_DIR)
            .join("main.lua");
    let mut plans = vec![
        PlannedWrite::new(config_path.clone(), config_body),
        PlannedWrite::new(harness_path, starter_harness().to_string()),
    ];

    if let Some((env_key, env_value)) = api_key {
        let env_path = config_dir(&config_path).join(".env");
        let existing = load_existing(&env_path)?;
        let mut updates = BTreeMap::new();
        updates.insert(env_key, env_value);
        let (body, display) = merge_env_file(existing.as_deref(), &updates);
        plans.push(PlannedWrite::new(env_path, body).with_display_contents(display));
    }

    confirm_and_write(&plans)?;

    println!("\nCreated '{}'.", config_path.display());
    println!(
        "Next steps:\n  1. Review the generated config if needed.\n  2. Run `turin daemon start --config {}`.",
        config_path.display()
    );

    Ok(())
}

fn prompt_provider() -> Result<ProviderChoice> {
    let options = ["Anthropic", "OpenAI", "Mock"];
    let index = Select::new()
        .with_prompt("Default provider")
        .items(options)
        .default(0)
        .interact()?;
    Ok(match index {
        0 => ProviderChoice::Anthropic,
        1 => ProviderChoice::OpenAi,
        _ => ProviderChoice::Mock,
    })
}

fn generate_turin_config(
    provider: ProviderChoice,
    model: &str,
    system_prompt: &str,
) -> Result<String> {
    let mut providers = BTreeMap::new();
    providers.insert(
        provider.name().to_string(),
        GeneratedProviderConfig {
            kind: provider.name().to_string(),
            api_key_env: provider.api_key_env().map(ToString::to_string),
        },
    );

    let config = GeneratedTurinConfig {
        agent: GeneratedAgentConfig {
            id: "default".to_string(),
            system_prompt: system_prompt.to_string(),
            model: model.to_string(),
            provider: provider.name().to_string(),
            idle_timeout_seconds: 20,
        },
        kernel: GeneratedKernelConfig {
            workspace_root: ".".to_string(),
            max_turns: 50,
            heartbeat_interval_seconds: 30,
            initial_spawn_depth: 0,
        },
        persistence: GeneratedPersistenceConfig {
            state: GeneratedStoreTargetConfig {
                path: "data/state.db".to_string(),
            },
        },
        harness: GeneratedHarnessConfig {
            directory: "harnesses".to_string(),
        },
        providers,
    };

    toml::to_string_pretty(&config).context("Failed to render Turin config")
}

fn starter_harness() -> &'static str {
    r#"function on_turn_prepare(ctx)
  return ALLOW
end

function on_tool_call(call)
  return ALLOW
end
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_turin_config_mentions_selected_provider() {
        let body = generate_turin_config(
            ProviderChoice::Anthropic,
            "claude-sonnet-4-20250514",
            "You are useful.",
        )
        .expect("rendered");
        assert!(body.contains("[providers.anthropic]"));
        assert!(body.contains("api_key_env = \"ANTHROPIC_API_KEY\""));
    }
}
