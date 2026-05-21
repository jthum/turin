use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use dialoguer::{Confirm, Input, MultiSelect, Password, Select};
use serde_json::Value;
use turin_channel_core::{
    ChannelAdapterManifest, ChannelAuthFlow, ChannelAuthFlowDisplay, ChannelAuthFlowPollRequest,
    ChannelAuthFlowPollResponse, ChannelAuthFlowStartRequest, ChannelConfigField,
    ChannelConfigTarget, ChannelConfigTargetKind, ChannelFieldVisibilityRule, ChannelKind,
    ChannelSecretRequirement, ChannelValidationCheck,
};
use turin_control_client::{ConnectionSpec, ControlClient};

use crate::files::{
    ConfiguredChannel, PlannedWrite, config_dir, confirm_and_write, load_configured_channels,
    load_existing, merge_env_file, render_channel_file, resolve_channels_dir,
};
use crate::runner::{
    describe_external_runner, discover_external_runner_kinds, poll_external_auth_flow,
    start_external_auth_flow, validate_external_runner_settings,
};

use super::{ChannelsListArgs, ChannelsStatusArgs, ConfigureChannelArgs};

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
