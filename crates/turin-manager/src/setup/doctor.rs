use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result};
use dotenvy::from_path_iter;
use turin_client::{Client, ConnectionSpec};

use crate::files::{config_dir, load_configured_channels};
use crate::runner::describe_external_runner;

use super::DoctorArgs;

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
                    "[ok] channel runner: {} ({})",
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
                    "[fail] channel runner: kind '{}' for channel '{}' is not available: {}",
                    channel.kind, channel.id, err
                );
                issues += 1;
            }
        }
    }

    match Client::connect(&ConnectionSpec::LocalConfig {
        config_path: config_path.clone(),
    })
    .await
    {
        Ok(client) => match client.status().await {
            Ok(status) => {
                println!("[ok] daemon: reachable at {}", status.endpoint);
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

fn load_adjacent_env_values(config_path: &Path) -> Result<BTreeMap<String, String>> {
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
