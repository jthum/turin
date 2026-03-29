use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result, anyhow};
use turin_channel_core::ChannelAdapterManifest;

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

    if let Some(cargo) = std::env::var_os("CARGO").filter(|value| !value.is_empty()) {
        let package = binary_name.clone();
        return Ok(ExternalRunnerCommand {
            program: PathBuf::from(cargo),
            args_prefix: vec![
                "run".to_string(),
                "-q".to_string(),
                "-p".to_string(),
                package.clone(),
                "--".to_string(),
            ],
            display: format!("cargo run -q -p {package} --"),
        });
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
