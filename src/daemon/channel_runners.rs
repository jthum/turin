use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result, anyhow};

const TELEGRAM_RUNNER_BIN: &str = "turin-channel-telegram";
const DISCORD_RUNNER_BIN: &str = "turin-channel-discord";
const TELEGRAM_RUNNER_ENV: &str = "TURIN_CHANNEL_TELEGRAM_BIN";
const DISCORD_RUNNER_ENV: &str = "TURIN_CHANNEL_DISCORD_BIN";

pub(crate) fn uses_external_runner(kind: &str) -> bool {
    matches!(kind, "discord" | "telegram")
}

#[derive(Debug, Clone)]
pub(crate) struct ExternalRunnerCommand {
    pub program: PathBuf,
    pub args_prefix: Vec<String>,
    pub display: String,
}

pub(crate) fn external_runner_binary_name(kind: &str) -> Option<&'static str> {
    match kind {
        "discord" => Some(DISCORD_RUNNER_BIN),
        "telegram" => Some(TELEGRAM_RUNNER_BIN),
        _ => None,
    }
}

pub(crate) fn resolve_external_runner_command(kind: &str) -> Result<ExternalRunnerCommand> {
    let binary_name = external_runner_binary_name(kind)
        .ok_or_else(|| anyhow!("No external runner is defined for channel kind '{kind}'"))?;

    if let Some(path) = std::env::var_os(external_runner_env_var(kind).unwrap_or_default())
        .filter(|value| !value.is_empty())
    {
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
        binary_name.to_string()
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
        let package = binary_name.to_string();
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

pub(crate) fn validate_external_channel_settings(
    kind: &str,
    settings: &serde_json::Value,
    allow_unconfigured_chats: bool,
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
    if kind == "telegram" && allow_unconfigured_chats {
        command.arg("--allow-unconfigured-chats");
    }

    let output = command
        .output()
        .with_context(|| format!("Failed to launch '{}'", runner.display))?;

    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let detail = if !stderr.is_empty() {
        stderr
    } else if !stdout.is_empty() {
        stdout
    } else {
        format!("runner exited with status {}", output.status)
    };

    Err(anyhow!(detail))
}

fn external_runner_env_var(kind: &str) -> Option<&'static str> {
    match kind {
        "discord" => Some(DISCORD_RUNNER_ENV),
        "telegram" => Some(TELEGRAM_RUNNER_ENV),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_external_runner_names() {
        assert_eq!(
            external_runner_binary_name("telegram"),
            Some("turin-channel-telegram")
        );
        assert_eq!(
            external_runner_binary_name("discord"),
            Some("turin-channel-discord")
        );
        assert_eq!(external_runner_binary_name("fs"), None);
    }

    #[test]
    fn reports_external_runner_usage_by_kind() {
        assert!(uses_external_runner("telegram"));
        assert!(uses_external_runner("discord"));
        assert!(!uses_external_runner("fs"));
    }
}
