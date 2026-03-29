use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result, anyhow};
use serde_json::Value;
use turin_channel_core::{ChannelAdapterManifest, validate_adapter_manifest};

pub(crate) fn builtin_channel_manifest(kind: &str) -> Option<ChannelAdapterManifest> {
    match kind {
        "fs" => Some(turin_channel_fs::adapter_manifest()),
        _ => None,
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ExternalRunnerCommand {
    pub program: PathBuf,
    pub args_prefix: Vec<String>,
    pub display: String,
}

pub(crate) fn external_runner_binary_name(kind: &str) -> String {
    format!("turin-channel-{}", normalize_binary_component(kind))
}

pub(crate) fn external_runner_env_var(kind: &str) -> String {
    format!("TURIN_CHANNEL_{}_BIN", normalize_env_component(kind))
}

pub(crate) fn resolve_external_runner_command(kind: &str) -> Result<ExternalRunnerCommand> {
    let binary_name = external_runner_binary_name(kind);
    let env_var = external_runner_env_var(kind);

    if let Some(path) = std::env::var_os(&env_var).filter(|value| !value.is_empty()) {
        let path = PathBuf::from(path);
        return Ok(ExternalRunnerCommand {
            display: path.display().to_string(),
            program: path,
            args_prefix: Vec::new(),
        });
    }

    if let Some(command) = resolve_workspace_runner_command(&binary_name) {
        return Ok(command);
    }

    let sibling_name = if cfg!(windows) {
        format!("{binary_name}.exe")
    } else {
        binary_name.clone()
    };

    if let Ok(current_exe) = std::env::current_exe() {
        let mut candidates = Vec::new();
        if let Some(parent) = current_exe.parent() {
            candidates.push(parent.join(&sibling_name));
            if parent.file_name().is_some_and(|name| name == "deps")
                && let Some(grandparent) = parent.parent()
            {
                candidates.push(grandparent.join(&sibling_name));
            }
        }

        for sibling in candidates {
            if sibling.exists() {
                return Ok(ExternalRunnerCommand {
                    display: sibling.display().to_string(),
                    program: sibling,
                    args_prefix: Vec::new(),
                });
            }
        }
    }

    let path = PathBuf::from(binary_name);
    Ok(ExternalRunnerCommand {
        display: path.display().to_string(),
        program: path,
        args_prefix: Vec::new(),
    })
}

pub(crate) fn describe_external_runner(kind: &str) -> Result<ChannelAdapterManifest> {
    let runner = resolve_external_runner_command(kind)?;
    let mut command = Command::new(&runner.program);
    for arg in &runner.args_prefix {
        command.arg(arg);
    }
    command.arg("describe");

    let output = command
        .output()
        .with_context(|| format!("Failed to launch '{}'", runner.display))?;

    if !output.status.success() {
        return Err(anyhow!(runner_output_detail(&output).unwrap_or_else(
            || { format!("runner exited with status {}", output.status) }
        )));
    }

    let manifest: ChannelAdapterManifest = serde_json::from_slice(&output.stdout)
        .context("Failed to decode channel runner manifest")?;
    if manifest.kind != kind {
        anyhow::bail!(
            "Channel runner '{}' reported kind '{}' but '{}' was requested",
            runner.display,
            manifest.kind,
            kind
        );
    }
    validate_adapter_manifest(&manifest)
        .map_err(anyhow::Error::msg)
        .context("Channel runner returned an invalid adapter manifest")?;
    Ok(manifest)
}

pub(crate) fn validate_external_channel_settings(
    kind: &str,
    settings: &serde_json::Value,
) -> Result<()> {
    let runner = resolve_external_runner_command(kind)?;
    let settings_json =
        serde_json::to_string(settings).context("Failed to encode channel settings JSON")?;

    let mut command = Command::new(&runner.program);
    for arg in &runner.args_prefix {
        command.arg(arg);
    }
    command
        .arg("validate-settings")
        .arg("--settings-json")
        .arg(&settings_json);

    let output = command
        .output()
        .with_context(|| format!("Failed to launch '{}'", runner.display))?;

    if output.status.success() {
        return Ok(());
    }

    Err(anyhow!(runner_output_detail(&output).unwrap_or_else(
        || format!("runner exited with status {}", output.status)
    )))
}

fn runner_output_detail(output: &std::process::Output) -> Option<String> {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if !stderr.is_empty() {
        return Some(stderr);
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if !stdout.is_empty() {
        return Some(stdout);
    }
    None
}

fn resolve_workspace_runner_command(binary_name: &str) -> Option<ExternalRunnerCommand> {
    if !workspace_has_runner(binary_name) {
        return None;
    }

    let cargo = std::env::var_os("CARGO").filter(|value| !value.is_empty())?;
    Some(ExternalRunnerCommand {
        program: PathBuf::from(cargo),
        args_prefix: vec![
            "run".to_string(),
            "-q".to_string(),
            "-p".to_string(),
            binary_name.to_string(),
            "--".to_string(),
        ],
        display: format!("cargo run -q -p {binary_name} --"),
    })
}

fn workspace_has_runner(binary_name: &str) -> bool {
    let Some(metadata) = workspace_metadata() else {
        return false;
    };
    let Some(packages) = metadata.get("packages").and_then(Value::as_array) else {
        return false;
    };
    for package in packages {
        let Some(targets) = package.get("targets").and_then(Value::as_array) else {
            continue;
        };
        for target in targets {
            let Some(name) = target.get("name").and_then(Value::as_str) else {
                continue;
            };
            if name != binary_name {
                continue;
            }
            let Some(kinds_array) = target.get("kind").and_then(Value::as_array) else {
                continue;
            };
            if kinds_array
                .iter()
                .any(|entry| entry.as_str().is_some_and(|kind| kind == "bin"))
            {
                return true;
            }
        }
    }
    false
}

fn workspace_metadata() -> Option<Value> {
    let cargo = std::env::var_os("CARGO").filter(|value| !value.is_empty())?;
    let manifest_path = find_workspace_manifest_path()?;
    let output = Command::new(cargo)
        .arg("metadata")
        .arg("--no-deps")
        .arg("--format-version")
        .arg("1")
        .arg("--manifest-path")
        .arg(&manifest_path)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    serde_json::from_slice::<Value>(&output.stdout).ok()
}

fn find_workspace_manifest_path() -> Option<PathBuf> {
    let mut cursor = std::env::current_dir().ok()?;
    loop {
        let manifest = cursor.join("Cargo.toml");
        if manifest.is_file() {
            return Some(manifest);
        }
        if !cursor.pop() {
            return None;
        }
    }
}

fn normalize_binary_component(kind: &str) -> String {
    kind.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect()
}

fn normalize_env_component(kind: &str) -> String {
    kind.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derives_external_runner_names_from_kind() {
        assert_eq!(
            external_runner_binary_name("telegram"),
            "turin-channel-telegram"
        );
        assert_eq!(
            external_runner_binary_name("rocketchat"),
            "turin-channel-rocketchat"
        );
        assert_eq!(
            external_runner_binary_name("email.smtp"),
            "turin-channel-email-smtp"
        );
    }

    #[test]
    fn derives_external_runner_env_overrides_from_kind() {
        assert_eq!(
            external_runner_env_var("telegram"),
            "TURIN_CHANNEL_TELEGRAM_BIN"
        );
        assert_eq!(
            external_runner_env_var("rocketchat"),
            "TURIN_CHANNEL_ROCKETCHAT_BIN"
        );
        assert_eq!(
            external_runner_env_var("email.smtp"),
            "TURIN_CHANNEL_EMAIL_SMTP_BIN"
        );
    }
}
