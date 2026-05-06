use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use dialoguer::{Confirm, Input, MultiSelect, Password, Select};
use dotenvy::from_path_iter;
use serde::Serialize;
use serde_json::Value;
use turin_channel_core::{
    ChannelAdapterManifest, ChannelAuthFlow, ChannelAuthFlowDisplay, ChannelAuthFlowPollRequest,
    ChannelAuthFlowPollResponse, ChannelAuthFlowStartRequest, ChannelConfigField,
    ChannelConfigTarget, ChannelConfigTargetKind, ChannelFieldVisibilityRule, ChannelKind,
    ChannelSecretRequirement, ChannelValidationCheck,
};
use turin_control_client::{ConnectionSpec, ControlClient};
use turin_types::layout::{DEFAULT_LAYOUT_HARNESSES_DIR, default_layout_root_for_workspace};

use crate::files::{
    ConfiguredChannel, PlannedWrite, config_dir, confirm_and_write,
    default_workspace_root_for_missing_config, load_configured_channels, load_existing,
    merge_env_file, render_channel_file, resolve_channels_dir,
};
use crate::runner::{
    describe_external_runner, discover_external_runner_kinds, poll_external_auth_flow,
    start_external_auth_flow, validate_external_runner_settings,
};

#[derive(Debug, Clone)]
pub(crate) struct InitArgs {
    pub(crate) config: PathBuf,
    pub(crate) force: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct DoctorArgs {
    pub(crate) config: PathBuf,
}

#[derive(Debug, Clone)]
pub(crate) struct ChannelsListArgs {
    pub(crate) config: PathBuf,
}

#[derive(Debug, Clone)]
pub(crate) struct ConfigureChannelArgs {
    pub(crate) config: PathBuf,
    pub(crate) kind: String,
    pub(crate) channel_id: Option<String>,
    pub(crate) agent_id: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct ChannelsStatusArgs {
    pub(crate) config: PathBuf,
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

pub(crate) async fn run_doctor(args: DoctorArgs) -> Result<()> {
    let config_path = args.config;
    let mut issues = 0usize;

    if config_path.is_file() {
        println!("[ok] config: {}", config_path.display());
    } else {
        println!("[fail] config: '{}' does not exist", config_path.display());
        issues += 1;
    }

    let configured_channels = load_configured_channels(&config_path)?;
    if configured_channels.is_empty() {
        println!("[warn] channels: no configured channels found");
    } else {
        println!("[ok] channels: {} configured", configured_channels.len());
    }

    let env_values = load_adjacent_env_values(&config_path)?;
    for channel in &configured_channels {
        match describe_external_runner(&channel.kind) {
            Ok(manifest) => {
                println!(
                    "[ok] sidecar: {} ({})",
                    manifest.display_name_or_kind(),
                    channel.kind
                );
                if let Some(setup) = &manifest.setup {
                    for secret in &setup.required_secrets {
                        let present = std::env::var_os(&secret.env_var)
                            .is_some_and(|value| !value.is_empty())
                            || env_values
                                .get(&secret.env_var)
                                .is_some_and(|value| !value.is_empty());
                        if present {
                            println!(
                                "[ok] secret: {} for channel '{}'",
                                secret.env_var, channel.id
                            );
                        } else if secret.optional {
                            println!(
                                "[warn] secret: optional {} is not configured for channel '{}'",
                                secret.env_var, channel.id
                            );
                        } else {
                            println!(
                                "[fail] secret: required {} is not configured for channel '{}'",
                                secret.env_var, channel.id
                            );
                            issues += 1;
                        }
                    }
                }
            }
            Err(err) => {
                println!(
                    "[fail] sidecar: kind '{}' for channel '{}' is not available: {}",
                    channel.kind, channel.id, err
                );
                issues += 1;
            }
        }
    }

    match ControlClient::connect(&ConnectionSpec::LocalConfig {
        config_path: config_path.clone(),
    })
    .await
    {
        Ok(client) => match client.status().await {
            Ok(status) => {
                println!("[ok] daemon: reachable at {}", status.endpoint);
                let runtimes: BTreeMap<_, _> = status
                    .channel_runtimes
                    .into_iter()
                    .map(|runtime| (runtime.id.clone(), runtime))
                    .collect();
                for channel in &configured_channels {
                    match runtimes.get(&channel.id) {
                        Some(runtime) if runtime.state == "running" => {
                            println!("[ok] runtime: channel '{}' is running", channel.id);
                        }
                        Some(runtime) => {
                            println!(
                                "[warn] runtime: channel '{}' is {}{}",
                                channel.id,
                                runtime.state,
                                runtime
                                    .last_error
                                    .as_deref()
                                    .map(|error| format!(" ({error})"))
                                    .unwrap_or_default()
                            );
                            if runtime.state == "failed" {
                                issues += 1;
                            }
                        }
                        None => {
                            println!(
                                "[warn] runtime: channel '{}' has no active runtime snapshot",
                                channel.id
                            );
                        }
                    }
                }
            }
            Err(err) => {
                println!("[warn] daemon: status unavailable: {err}");
            }
        },
        Err(err) => {
            println!(
                "[warn] daemon: not reachable (start Turin with `turin daemon start --config {}`): {}",
                config_path.display(),
                err
            );
        }
    }

    if issues > 0 {
        anyhow::bail!("doctor found {issues} blocking issue(s)");
    }

    println!("Doctor completed without blocking issues.");
    Ok(())
}

pub(crate) async fn run_channels_list(args: ChannelsListArgs) -> Result<()> {
    let configured_channels = load_configured_channels(&args.config)?;
    let configured_by_kind = configured_channels_by_kind(&configured_channels);

    let mut discovered_manifests = BTreeMap::new();
    for kind in discover_external_runner_kinds() {
        if let Ok(manifest) = describe_external_runner(&kind) {
            discovered_manifests.insert(kind, manifest);
        }
    }

    let mut all_kinds: BTreeSet<String> = discovered_manifests.keys().cloned().collect();
    all_kinds.extend(configured_by_kind.keys().cloned());

    if all_kinds.is_empty() {
        println!("No channels discovered.");
        println!(
            "Install or place a `turin-channel-<kind>` sidecar where Turin can resolve it, then run `turin-manager channels configure <kind>`."
        );
        return Ok(());
    }

    let mut rows = Vec::new();
    rows.push(vec![
        "KIND".to_string(),
        "NAME".to_string(),
        "INSTALLED".to_string(),
        "CONFIGURED".to_string(),
        "CHANNEL IDS".to_string(),
    ]);

    for kind in all_kinds {
        let configured_ids = configured_by_kind.get(&kind).cloned().unwrap_or_default();
        let manifest = discovered_manifests
            .get(&kind)
            .cloned()
            .or_else(|| describe_external_runner(&kind).ok());
        let display_name = manifest
            .as_ref()
            .map(|manifest| manifest.display_name_or_kind().to_string())
            .unwrap_or_else(|| kind.clone());
        rows.push(vec![
            kind,
            display_name,
            yes_no(manifest.is_some()),
            yes_no(!configured_ids.is_empty()),
            if configured_ids.is_empty() {
                "-".to_string()
            } else {
                configured_ids.join(", ")
            },
        ]);
    }

    print_table(&rows);
    Ok(())
}

pub(crate) async fn run_configure_channel(args: ConfigureChannelArgs) -> Result<()> {
    let config_path = args.config;
    if !config_path.is_file() {
        anyhow::bail!(
            "'{}' does not exist. Run `turin-manager init` first.",
            config_path.display()
        );
    }

    let kind = ChannelKind::parse(&args.kind).map_err(anyhow::Error::msg)?;
    let manifest = describe_external_runner(kind.as_str())
        .with_context(|| format!("Failed to inspect the '{}' sidecar manifest", kind.as_str()))?;
    let setup = manifest.setup.clone().ok_or_else(|| {
        anyhow!(
            "{} does not expose setup metadata; cannot build a generic configuration flow",
            manifest.display_name_or_kind()
        )
    })?;

    print_setup_intro(&manifest);

    let channel_id = match args.channel_id {
        Some(channel_id) => channel_id,
        None => Input::new()
            .with_prompt("Channel ID")
            .default(kind.as_str().to_string())
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
            .with_prompt(format!(
                "Configure advanced {} options?",
                manifest.display_name_or_kind()
            ))
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
        if let Some(validation) = &field.validate {
            validate_field(field, &value, validation).await?;
        }
        apply_target_value(target, value, &mut channel_settings, &mut env_updates)?;
    }

    for flow in &setup.auth_flows {
        if flow.advanced && !advanced_enabled {
            continue;
        }
        if !field_is_visible(flow.visible_if.as_ref(), &channel_settings) {
            continue;
        }
        run_auth_flow(
            kind.as_str(),
            manifest.display_name_or_kind(),
            flow,
            &mut channel_settings,
            &mut env_updates,
        )
        .await?;
    }

    let settings_value = current_settings_value(&channel_settings);
    validate_external_runner_settings(kind.as_str(), &settings_value, &env_updates).with_context(
        || {
            format!(
                "Final {} settings validation failed",
                manifest.display_name_or_kind()
            )
        },
    )?;

    let channel_path = resolve_channels_dir(&config_path)?
        .join(&channel_id)
        .join("config.toml");
    let existing_channel = load_existing(&channel_path)?;
    let channel_body = render_channel_file(
        existing_channel.as_deref(),
        true,
        kind.as_str(),
        &agent_id,
        &channel_settings,
    )?;

    let mut plans = vec![PlannedWrite::new(channel_path, channel_body)];
    let mut secrets_written = false;
    if !env_updates.is_empty()
        && Confirm::new()
            .with_prompt("Write generated secrets and env vars to .env next to the Turin config?")
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

    println!(
        "\nConfigured {} channel '{}'.",
        manifest.display_name_or_kind(),
        channel_id
    );
    if !secrets_written {
        for key in env_updates.keys() {
            println!("Remember to export {} before starting Turin.", key);
        }
    }
    println!(
        "Next step: `turin daemon start --config {}`",
        config_path.display()
    );

    Ok(())
}

pub(crate) async fn run_channels_status(args: ChannelsStatusArgs) -> Result<()> {
    let configured_channels = load_configured_channels(&args.config)?;
    if configured_channels.is_empty() {
        println!("No configured channels found.");
        println!("Use `turin-manager channels configure <kind>` to add one.");
        return Ok(());
    }

    let mut runtimes_by_id = BTreeMap::new();
    let daemon_note = match ControlClient::connect(&ConnectionSpec::LocalConfig {
        config_path: args.config.clone(),
    })
    .await
    {
        Ok(client) => match client.status().await {
            Ok(status) => {
                for runtime in status.channel_runtimes {
                    runtimes_by_id.insert(runtime.id.clone(), runtime);
                }
                None
            }
            Err(err) => Some(format!("Daemon status unavailable: {err}")),
        },
        Err(err) => Some(format!("Daemon not reachable: {err}")),
    };

    if let Some(note) = &daemon_note {
        println!("{note}");
        println!(
            "Showing configured channels only. Start Turin with `turin daemon start --config {}` for runtime state.",
            args.config.display()
        );
        println!();
    }

    let mut rows = Vec::new();
    rows.push(vec![
        "CHANNEL".to_string(),
        "KIND".to_string(),
        "ENABLED".to_string(),
        "AGENT".to_string(),
        "STATE".to_string(),
        "ERROR".to_string(),
    ]);

    for channel in configured_channels {
        let runtime = runtimes_by_id.get(&channel.id);
        rows.push(vec![
            channel.id,
            channel.kind,
            yes_no(channel.enabled),
            channel.agent_id.unwrap_or_else(|| "-".to_string()),
            runtime
                .map(|runtime| runtime.state.clone())
                .unwrap_or_else(|| "unknown".to_string()),
            runtime
                .and_then(|runtime| {
                    runtime
                        .last_error_code
                        .clone()
                        .or_else(|| runtime.last_error.clone())
                })
                .unwrap_or_else(|| "-".to_string()),
        ]);
    }

    print_table(&rows);
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
    if secret_value.is_empty() {
        return Ok(());
    }

    if let Some(validation) = &secret.validate {
        validate_named_value(&secret.name, &secret_value, validation).await?;
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

async fn run_auth_flow(
    kind: &str,
    display_name: &str,
    flow: &ChannelAuthFlow,
    channel_settings: &mut BTreeMap<String, Value>,
    env_updates: &mut BTreeMap<String, String>,
) -> Result<()> {
    if let Some(help) = &flow.help {
        println!("{help}");
    }
    if let Some(hint) = &flow.hint {
        println!("Hint: {hint}");
    }

    let prompt = flow
        .prompt
        .clone()
        .or_else(|| flow.label.clone())
        .unwrap_or_else(|| flow.id.clone());
    if !Confirm::new()
        .with_prompt(format!("{display_name}: {prompt}?"))
        .default(true)
        .interact()?
    {
        anyhow::bail!("Skipped required auth flow '{}'", flow.id);
    }

    let start = start_external_auth_flow(
        kind,
        &ChannelAuthFlowStartRequest {
            flow_id: flow.id.clone(),
            current_settings: current_settings_value(channel_settings),
        },
    )?;
    render_auth_flow_display(&start.display)?;

    let session = start.session;
    let mut poll_interval = start.display.poll_interval_seconds.unwrap_or(5);

    loop {
        tokio::time::sleep(Duration::from_secs(poll_interval.max(1))).await;
        match poll_external_auth_flow(
            kind,
            &ChannelAuthFlowPollRequest {
                flow_id: flow.id.clone(),
                session: session.clone(),
                current_settings: current_settings_value(channel_settings),
            },
        )? {
            ChannelAuthFlowPollResponse::Pending { display } => {
                render_auth_flow_display(&display)?;
                poll_interval = display.poll_interval_seconds.unwrap_or(poll_interval);
            }
            ChannelAuthFlowPollResponse::Complete { values, message } => {
                if let Some(message) = message {
                    println!("{message}");
                }
                for resolved in values {
                    apply_target_value(
                        &resolved.target,
                        resolved.value,
                        channel_settings,
                        env_updates,
                    )?;
                }
                break;
            }
            ChannelAuthFlowPollResponse::Failed { message } => {
                anyhow::bail!("Auth flow '{}' failed: {}", flow.id, message);
            }
        }
    }

    Ok(())
}

fn render_auth_flow_display(display: &ChannelAuthFlowDisplay) -> Result<()> {
    if let Some(message) = &display.message {
        println!("{message}");
    }
    if let Some(uri) = &display.verification_uri {
        println!("Verification URL: {uri}");
    }
    if let Some(uri) = &display.verification_uri_complete {
        println!("Direct verification URL: {uri}");
    }
    if let Some(code) = &display.user_code {
        println!("User code: {code}");
    }
    if let Some(code) = &display.pairing_code {
        println!("Pairing code: {code}");
    }
    if let Some(qr_text) = &display.qr_text {
        println!("Scan this QR code:");
        if let Err(err) = qr2term::print_qr(qr_text) {
            println!("Failed to render QR code cleanly: {err}");
            println!("QR payload: {qr_text}");
        }
    }
    if let Some(expires_in_seconds) = display.expires_in_seconds {
        println!("This step expires in {} seconds.", expires_in_seconds);
    }
    Ok(())
}

fn current_settings_value(settings: &BTreeMap<String, Value>) -> Value {
    let object = settings
        .iter()
        .filter(|(_, value)| !value.is_null())
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    Value::Object(object)
}

async fn validate_field(
    field: &ChannelConfigField,
    value: &Value,
    validation: &ChannelValidationCheck,
) -> Result<()> {
    if value.is_null() {
        return Ok(());
    }
    let raw = value_as_validation_string(value)
        .ok_or_else(|| anyhow!("Field '{}' cannot be validated as text", field.key))?;
    validate_named_value(&field.key, &raw, validation).await
}

async fn validate_named_value(
    key: &str,
    value: &str,
    validation: &ChannelValidationCheck,
) -> Result<()> {
    match validation.kind.as_str() {
        "http_get" => {
            let template = validation
                .url_template
                .as_ref()
                .ok_or_else(|| anyhow!("Validation for '{}' is missing 'url_template'", key))?;
            let url = template.replace(&format!("{{{}}}", key), value);
            let response = reqwest::Client::new()
                .get(&url)
                .send()
                .await
                .with_context(|| format!("Validation request for '{}' failed", key))?;
            if !response.status().is_success() {
                anyhow::bail!(
                    "{}",
                    validation.message.clone().unwrap_or_else(|| {
                        format!(
                            "Validation for '{}' failed with status {}",
                            key,
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

fn print_setup_intro(manifest: &ChannelAdapterManifest) {
    println!("Configuring {}.", manifest.display_name_or_kind());
    if let Some(setup) = &manifest.setup {
        if let Some(instructions) = &setup.instructions {
            println!("{instructions}");
        }
        if let Some(url) = &setup.setup_url {
            println!("Setup URL: {url}");
        }
    }
}

fn configured_channels_by_kind(channels: &[ConfiguredChannel]) -> BTreeMap<String, Vec<String>> {
    let mut by_kind = BTreeMap::new();
    for channel in channels {
        by_kind
            .entry(channel.kind.clone())
            .or_insert_with(Vec::new)
            .push(channel.id.clone());
    }
    by_kind
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
    if let Some(example) = &field.example {
        println!("Example: {example}");
    }
    let prompt = field
        .prompt
        .clone()
        .or_else(|| field.label.clone())
        .unwrap_or_else(|| field.key.clone());

    match field.field_type.as_str() {
        "text" => {
            let default = value_as_default_string(field.default.as_ref());
            let mut input = Input::<String>::new()
                .with_prompt(prompt)
                .allow_empty(!field.required);
            if let Some(default) = default {
                input = input.default(default);
            }
            let value = input.interact_text()?;
            if field.required && value.trim().is_empty() {
                anyhow::bail!("Field '{}' must not be empty", field.key);
            }
            if value.trim().is_empty() {
                Ok(Value::Null)
            } else {
                Ok(Value::String(value))
            }
        }
        "secret" => {
            let value = Password::new()
                .with_prompt(prompt)
                .allow_empty_password(!field.required)
                .interact()?;
            if field.required && value.trim().is_empty() {
                anyhow::bail!("Field '{}' must not be empty", field.key);
            }
            if value.trim().is_empty() {
                Ok(Value::Null)
            } else {
                Ok(Value::String(value))
            }
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
            let mut input = Input::<String>::new()
                .with_prompt(prompt)
                .allow_empty(field.default.is_none());
            if let Some(default) = default {
                input = input.default(default);
            }
            let raw = input.interact_text()?;
            if raw.trim().is_empty() {
                return Ok(Value::Null);
            }
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
                .and_then(|wanted| {
                    field
                        .options
                        .iter()
                        .position(|option| option.value == wanted)
                })
                .unwrap_or(0);
            let index = Select::new()
                .with_prompt(prompt)
                .items(&labels)
                .default(default_index)
                .interact()?;
            Ok(Value::String(field.options[index].value.clone()))
        }
        "multi_select" => {
            if field.options.is_empty() {
                anyhow::bail!("Field '{}' has no multi-select options", field.key);
            }
            let labels: Vec<String> = field
                .options
                .iter()
                .map(|option| option.label.clone().unwrap_or_else(|| option.value.clone()))
                .collect();
            let default_values = field
                .default
                .as_ref()
                .and_then(value_as_string_list)
                .unwrap_or_default();
            let defaults: Vec<bool> = field
                .options
                .iter()
                .map(|option| default_values.iter().any(|value| value == &option.value))
                .collect();
            let indices = MultiSelect::new()
                .with_prompt(prompt)
                .items(&labels)
                .defaults(&defaults)
                .interact()?;
            let values: Vec<String> = indices
                .into_iter()
                .map(|index| field.options[index].value.clone())
                .collect();
            Ok(serde_json::json!(values))
        }
        "string_list" => {
            let default = field
                .default
                .as_ref()
                .and_then(value_as_string_list)
                .map(|items| items.join(", "));
            let mut input = Input::<String>::new()
                .with_prompt(format!("{prompt} (comma-separated)"))
                .allow_empty(true);
            if let Some(default) = default {
                input = input.default(default);
            }
            let raw = input.interact_text()?;
            if raw.trim().is_empty() {
                return Ok(Value::Null);
            }
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
            if value.is_null() {
                channel_settings.insert(target.name.clone(), Value::Null);
                return Ok(());
            }
            channel_settings.insert(target.name.clone(), value);
            Ok(())
        }
        ChannelConfigTargetKind::EnvVar => {
            if value.is_null() {
                env_updates.remove(&target.name);
                return Ok(());
            }
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

fn value_as_validation_string(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        Value::Bool(value) => Some(value.to_string()),
        Value::Array(values) => {
            let mut rendered = Vec::with_capacity(values.len());
            for value in values {
                rendered.push(value.as_str()?.to_string());
            }
            Some(rendered.join(","))
        }
        _ => None,
    }
}

fn load_adjacent_env_values(config_path: &std::path::Path) -> Result<BTreeMap<String, String>> {
    let env_path = config_dir(config_path).join(".env");
    if !env_path.is_file() {
        return Ok(BTreeMap::new());
    }

    let mut values = BTreeMap::new();
    for item in from_path_iter(&env_path)
        .with_context(|| format!("Failed to parse '{}'", env_path.display()))?
    {
        let (key, value) =
            item.with_context(|| format!("Failed to parse '{}'", env_path.display()))?;
        values.insert(key, value);
    }
    Ok(values)
}

fn yes_no(value: bool) -> String {
    if value {
        "yes".to_string()
    } else {
        "no".to_string()
    }
}

fn print_table(rows: &[Vec<String>]) {
    if rows.is_empty() {
        return;
    }

    let cols = rows[0].len();
    let mut widths = vec![0usize; cols];
    for row in rows {
        for (idx, cell) in row.iter().enumerate() {
            widths[idx] = widths[idx].max(cell.len());
        }
    }

    for (row_idx, row) in rows.iter().enumerate() {
        let line = row
            .iter()
            .enumerate()
            .map(|(idx, cell)| format!("{:width$}", cell, width = widths[idx]))
            .collect::<Vec<_>>()
            .join("  ");
        println!("{line}");
        if row_idx == 0 {
            let sep = widths
                .iter()
                .map(|width| "-".repeat(*width))
                .collect::<Vec<_>>()
                .join("  ");
            println!("{sep}");
        }
    }
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
        values.insert(
            "pairing_mode".to_string(),
            Value::String("auto".to_string()),
        );
        let rule = ChannelFieldVisibilityRule {
            key: "pairing_mode".to_string(),
            equals: Value::String("auto".to_string()),
        };
        assert!(field_is_visible(Some(&rule), &values));
    }

    #[test]
    fn groups_configured_channels_by_kind() {
        let channels = vec![
            ConfiguredChannel {
                id: "telegram-main".to_string(),
                kind: "telegram".to_string(),
                enabled: true,
                agent_id: Some("default".to_string()),
            },
            ConfiguredChannel {
                id: "telegram-ops".to_string(),
                kind: "telegram".to_string(),
                enabled: true,
                agent_id: Some("default".to_string()),
            },
            ConfiguredChannel {
                id: "discord".to_string(),
                kind: "discord".to_string(),
                enabled: true,
                agent_id: Some("default".to_string()),
            },
        ];

        let grouped = configured_channels_by_kind(&channels);
        assert_eq!(
            grouped.get("telegram"),
            Some(&vec![
                "telegram-main".to_string(),
                "telegram-ops".to_string()
            ])
        );
        assert_eq!(grouped.get("discord"), Some(&vec!["discord".to_string()]));
    }

    #[test]
    fn current_settings_value_omits_null_entries() {
        let mut settings = BTreeMap::new();
        settings.insert("websocket_url".to_string(), Value::Null);
        settings.insert(
            "respond_mode".to_string(),
            Value::String("mentions".to_string()),
        );

        let rendered = current_settings_value(&settings);
        let object = rendered.as_object().expect("object");

        assert!(!object.contains_key("websocket_url"));
        assert_eq!(
            object.get("respond_mode"),
            Some(&Value::String("mentions".to_string()))
        );
    }
}
