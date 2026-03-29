use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use dialoguer::{Confirm, Input, Password, Select};
use serde::Serialize;
use serde_json::Value;
use turin_channel_core::{
    ChannelAdapterManifest, ChannelConfigField, ChannelConfigTarget, ChannelConfigTargetKind,
    ChannelFieldVisibilityRule, ChannelSecretRequirement, ChannelValidationCheck,
};

use crate::files::{
    PlannedWrite, config_dir, confirm_and_write, load_existing, merge_env_file,
    render_channel_file, resolve_channels_dir,
};
use crate::runner::describe_external_runner;

#[derive(Debug, Clone)]
pub(crate) struct InitArgs {
    pub(crate) config: PathBuf,
    pub(crate) force: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct TelegramSetupArgs {
    pub(crate) config: PathBuf,
    pub(crate) channel_id: Option<String>,
    pub(crate) agent_id: Option<String>,
}

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
    mode: String,
}

#[derive(Debug, Serialize)]
struct GeneratedKernelConfig {
    workspace_root: String,
    max_turns: u32,
    heartbeat_interval_secs: u32,
    initial_spawn_depth: u32,
}

#[derive(Debug, Serialize)]
struct GeneratedPersistenceConfig {
    database_path: String,
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
    let harness_path = config_dir(&config_path).join(".turin/harnesses/main.lua");
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

pub(crate) async fn run_setup_telegram(args: TelegramSetupArgs) -> Result<()> {
    let config_path = args.config;
    if !config_path.is_file() {
        anyhow::bail!(
            "'{}' does not exist. Run `turin-manager init` first.",
            config_path.display()
        );
    }

    let manifest = describe_external_runner("telegram")
        .context("Failed to inspect the Telegram sidecar manifest")?;
    let setup = manifest.setup.clone().ok_or_else(|| {
        anyhow!(
            "Telegram sidecar did not expose setup metadata; cannot continue with setup flow"
        )
    })?;

    print_setup_intro(&manifest);

    let channel_id = match args.channel_id {
        Some(channel_id) => channel_id,
        None => Input::new()
            .with_prompt("Channel ID")
            .default("telegram".to_string())
            .interact_text()?,
    };
    let agent_id = match args.agent_id {
        Some(agent_id) => agent_id,
        None => Input::new()
            .with_prompt("Agent ID")
            .default("default".to_string())
            .interact_text()?,
    };

    let advanced_enabled = setup.config_fields.iter().any(|field| field.advanced)
        && Confirm::new()
            .with_prompt("Configure advanced Telegram options?")
            .default(false)
            .interact()?;

    let mut channel_settings = BTreeMap::new();
    let mut env_updates = BTreeMap::new();

    for secret in &setup.required_secrets {
        capture_secret(secret, &mut channel_settings, &mut env_updates).await?;
    }

    for field in &setup.config_fields {
        if field.advanced && !advanced_enabled {
            continue;
        }
        if !field_is_visible(field.visible_if.as_ref(), &channel_settings) {
            continue;
        }

        let target = field.target.as_ref().ok_or_else(|| {
            anyhow!(
                "Config field '{}' is missing a target; cannot apply it generically",
                field.key
            )
        })?;
        let value = prompt_field(field)?;
        apply_target_value(target, value, &mut channel_settings, &mut env_updates)?;
    }

    let channel_path = resolve_channels_dir(&config_path)?
        .join(&channel_id)
        .join("channel.toml");
    let existing_channel = load_existing(&channel_path)?;
    let channel_body = render_channel_file(
        existing_channel.as_deref(),
        true,
        "telegram",
        &agent_id,
        &channel_settings,
    )?;

    let mut plans = vec![PlannedWrite::new(channel_path, channel_body)];
    let mut secrets_written = false;
    if !env_updates.is_empty()
        && Confirm::new()
            .with_prompt("Write validated secrets to .env next to turin.toml?")
            .default(true)
            .interact()?
    {
        let env_path = config_dir(&config_path).join(".env");
        let existing_env = load_existing(&env_path)?;
        let (env_body, env_display) = merge_env_file(existing_env.as_deref(), &env_updates);
        plans.push(PlannedWrite::new(env_path, env_body).with_display_contents(env_display));
        secrets_written = true;
    }

    confirm_and_write(&plans)?;

    println!("\nConfigured Telegram channel '{}'.", channel_id);
    if !secrets_written {
        for (key, _) in &env_updates {
            println!("Remember to export {} before starting Turin.", key);
        }
    }
    println!(
        "Next step: `turin daemon start --config {}`",
        config_path.display()
    );

    Ok(())
}

async fn capture_secret(
    secret: &ChannelSecretRequirement,
    channel_settings: &mut BTreeMap<String, Value>,
    env_updates: &mut BTreeMap<String, String>,
) -> Result<()> {
    if let Some(help) = &secret.help {
        println!("{help}");
    }
    for hint in &secret.hints {
        println!("Hint: {hint}");
    }

    let prompt = secret
        .display_name
        .clone()
        .unwrap_or_else(|| secret.name.clone());
    let secret_value = Password::new()
        .with_prompt(prompt)
        .allow_empty_password(secret.optional)
        .interact()?;

    if secret_value.is_empty() && !secret.optional {
        anyhow::bail!("Secret '{}' must not be empty", secret.name);
    }

    if let Some(validation) = &secret.validate {
        validate_secret(secret, &secret_value, validation).await?;
    }

    env_updates.insert(secret.env_var.clone(), secret_value);
    if let Some(target) = &secret.target {
        apply_target_value(
            target,
            Value::String(secret.env_var.clone()),
            channel_settings,
            env_updates,
        )?;
    }

    Ok(())
}

async fn validate_secret(
    secret: &ChannelSecretRequirement,
    value: &str,
    validation: &ChannelValidationCheck,
) -> Result<()> {
    match validation.kind.as_str() {
        "http_get" => {
            let template = validation.url_template.as_ref().ok_or_else(|| {
                anyhow!(
                    "Validation for '{}' is missing 'url_template'",
                    secret.name
                )
            })?;
            let url = template.replace(&format!("{{{}}}", secret.name), value);
            let response = reqwest::Client::new()
                .get(&url)
                .send()
                .await
                .with_context(|| format!("Validation request for '{}' failed", secret.name))?;
            if !response.status().is_success() {
                anyhow::bail!(
                    "{}",
                    validation.message.clone().unwrap_or_else(|| {
                        format!(
                            "Validation for '{}' failed with status {}",
                            secret.name,
                            response.status()
                        )
                    })
                );
            }
            Ok(())
        }
        other => anyhow::bail!("Unsupported validation kind '{}'", other),
    }
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
            mode: "auto".to_string(),
        },
        kernel: GeneratedKernelConfig {
            workspace_root: ".".to_string(),
            max_turns: 50,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        persistence: GeneratedPersistenceConfig {
            database_path: ".turin/state.db".to_string(),
        },
        harness: GeneratedHarnessConfig {
            directory: ".turin/harnesses".to_string(),
        },
        providers,
    };

    toml::to_string_pretty(&config).context("Failed to render turin.toml")
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

fn print_setup_intro(manifest: &ChannelAdapterManifest) {
    println!("Setting up {}.", manifest.display_name_or_kind());
    if let Some(setup) = &manifest.setup {
        if let Some(instructions) = &setup.instructions {
            println!("{instructions}");
        }
        if let Some(url) = &setup.setup_url {
            println!("Setup URL: {url}");
        }
    }
}

fn field_is_visible(
    visible_if: Option<&ChannelFieldVisibilityRule>,
    values: &BTreeMap<String, Value>,
) -> bool {
    let Some(rule) = visible_if else {
        return true;
    };
    values.get(&rule.key) == Some(&rule.equals)
}

fn prompt_field(field: &ChannelConfigField) -> Result<Value> {
    if let Some(help) = &field.help {
        println!("{help}");
    }
    if let Some(hint) = &field.hint {
        println!("Hint: {hint}");
    }
    let prompt = field
        .prompt
        .clone()
        .or_else(|| field.label.clone())
        .unwrap_or_else(|| field.key.clone());

    match field.field_type.as_str() {
        "text" => {
            let default = value_as_default_string(field.default.as_ref());
            let mut input = Input::<String>::new().with_prompt(prompt);
            if let Some(default) = default {
                input = input.default(default);
            }
            let value = input.interact_text()?;
            if field.required && value.trim().is_empty() {
                anyhow::bail!("Field '{}' must not be empty", field.key);
            }
            Ok(Value::String(value))
        }
        "secret" => {
            let value = Password::new()
                .with_prompt(prompt)
                .allow_empty_password(!field.required)
                .interact()?;
            if field.required && value.trim().is_empty() {
                anyhow::bail!("Field '{}' must not be empty", field.key);
            }
            Ok(Value::String(value))
        }
        "boolean" => {
            let default = field
                .default
                .as_ref()
                .and_then(Value::as_bool)
                .unwrap_or(false);
            let value = Confirm::new()
                .with_prompt(prompt)
                .default(default)
                .interact()?;
            Ok(Value::Bool(value))
        }
        "number" => {
            let default = value_as_default_string(field.default.as_ref());
            let mut input = Input::<String>::new().with_prompt(prompt);
            if let Some(default) = default {
                input = input.default(default);
            }
            let raw = input.interact_text()?;
            if raw.contains('.') {
                let parsed: f64 = raw
                    .parse()
                    .with_context(|| format!("Field '{}' must be numeric", field.key))?;
                Ok(serde_json::json!(parsed))
            } else {
                let parsed: i64 = raw
                    .parse()
                    .with_context(|| format!("Field '{}' must be numeric", field.key))?;
                Ok(serde_json::json!(parsed))
            }
        }
        "select" => {
            if field.options.is_empty() {
                anyhow::bail!("Field '{}' has no select options", field.key);
            }
            let labels: Vec<String> = field
                .options
                .iter()
                .map(|option| option.label.clone().unwrap_or_else(|| option.value.clone()))
                .collect();
            let default_value = field.default.as_ref().and_then(Value::as_str);
            let default_index = default_value
                .and_then(|wanted| field.options.iter().position(|option| option.value == wanted))
                .unwrap_or(0);
            let index = Select::new()
                .with_prompt(prompt)
                .items(&labels)
                .default(default_index)
                .interact()?;
            Ok(Value::String(field.options[index].value.clone()))
        }
        "multi_select" => {
            anyhow::bail!(
                "Field '{}' uses 'multi_select', which is not implemented yet in turin-manager",
                field.key
            )
        }
        "string_list" => {
            let default = field.default.as_ref().and_then(value_as_string_list).map(|items| {
                items.join(", ")
            });
            let mut input = Input::<String>::new()
                .with_prompt(format!("{prompt} (comma-separated)"));
            if let Some(default) = default {
                input = input.default(default);
            }
            let raw = input.interact_text()?;
            let values: Vec<String> = raw
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToString::to_string)
                .collect();
            Ok(serde_json::json!(values))
        }
        other => anyhow::bail!("Unsupported field type '{}'", other),
    }
}

fn apply_target_value(
    target: &ChannelConfigTarget,
    value: Value,
    channel_settings: &mut BTreeMap<String, Value>,
    env_updates: &mut BTreeMap<String, String>,
) -> Result<()> {
    match target.kind {
        ChannelConfigTargetKind::ChannelSetting => {
            channel_settings.insert(target.name.clone(), value);
            Ok(())
        }
        ChannelConfigTargetKind::EnvVar => {
            let value = value
                .as_str()
                .ok_or_else(|| anyhow!("Env-var target '{}' requires a string value", target.name))?
                .to_string();
            env_updates.insert(target.name.clone(), value);
            Ok(())
        }
        ChannelConfigTargetKind::RootConfig
        | ChannelConfigTargetKind::AgentConfig
        | ChannelConfigTargetKind::LocalSecretStore => anyhow::bail!(
            "Target kind '{:?}' is not implemented yet in turin-manager",
            target.kind
        ),
    }
}

fn value_as_default_string(value: Option<&Value>) -> Option<String> {
    match value {
        Some(Value::String(value)) => Some(value.clone()),
        Some(Value::Number(value)) => Some(value.to_string()),
        Some(Value::Bool(value)) => Some(value.to_string()),
        _ => None,
    }
}

fn value_as_string_list(value: &Value) -> Option<Vec<String>> {
    let array = value.as_array()?;
    let mut items = Vec::with_capacity(array.len());
    for item in array {
        items.push(item.as_str()?.to_string());
    }
    Some(items)
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

    #[test]
    fn visible_if_matches_existing_value() {
        let mut values = BTreeMap::new();
        values.insert("pairing_mode".to_string(), Value::String("auto".to_string()));
        let rule = ChannelFieldVisibilityRule {
            key: "pairing_mode".to_string(),
            equals: Value::String("auto".to_string()),
        };
        assert!(field_is_visible(Some(&rule), &values));
    }
}
