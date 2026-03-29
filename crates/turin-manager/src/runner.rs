use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, anyhow};
use serde_json::Value;
use turin_channel_core::{ChannelAdapterManifest, ChannelKind};

#[derive(Debug, Clone)]
pub(crate) struct ExternalRunnerCommand {
    pub(crate) program: PathBuf,
    pub(crate) args_prefix: Vec<String>,
    pub(crate) display: String,
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

    let sibling_name = platform_binary_name(&binary_name);
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

pub(crate) fn discover_external_runner_kinds() -> Vec<String> {
    let mut kinds = BTreeSet::new();

    discover_runner_kinds_from_current_exe(&mut kinds);
    discover_runner_kinds_from_path(&mut kinds);
    discover_runner_kinds_from_workspace(&mut kinds);

    kinds.into_iter().collect()
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

fn discover_runner_kinds_from_current_exe(kinds: &mut BTreeSet<String>) {
    let Ok(current_exe) = std::env::current_exe() else {
        return;
    };
    let Some(parent) = current_exe.parent() else {
        return;
    };

    discover_runner_kinds_in_dir(parent, kinds);
    if parent.file_name().is_some_and(|name| name == "deps")
        && let Some(grandparent) = parent.parent()
    {
        discover_runner_kinds_in_dir(grandparent, kinds);
    }
}

fn discover_runner_kinds_from_path(kinds: &mut BTreeSet<String>) {
    let Some(path) = std::env::var_os("PATH") else {
        return;
    };
    for dir in std::env::split_paths(&path) {
        discover_runner_kinds_in_dir(&dir, kinds);
    }
}

fn discover_runner_kinds_from_workspace(kinds: &mut BTreeSet<String>) {
    let Some(cargo) = std::env::var_os("CARGO").filter(|value| !value.is_empty()) else {
        return;
    };
    let Some(manifest_path) = find_workspace_manifest_path() else {
        return;
    };

    let output = match Command::new(cargo)
        .arg("metadata")
        .arg("--no-deps")
        .arg("--format-version")
        .arg("1")
        .arg("--manifest-path")
        .arg(&manifest_path)
        .output()
    {
        Ok(output) if output.status.success() => output,
        _ => return,
    };

    let Ok(metadata) = serde_json::from_slice::<Value>(&output.stdout) else {
        return;
    };
    let Some(packages) = metadata.get("packages").and_then(Value::as_array) else {
        return;
    };

    for package in packages {
        let Some(targets) = package.get("targets").and_then(Value::as_array) else {
            continue;
        };
        for target in targets {
            let Some(kinds_array) = target.get("kind").and_then(Value::as_array) else {
                continue;
            };
            let is_bin = kinds_array
                .iter()
                .any(|entry| entry.as_str().is_some_and(|kind| kind == "bin"));
            if !is_bin {
                continue;
            }
            let Some(name) = target.get("name").and_then(Value::as_str) else {
                continue;
            };
            if let Some(kind) = kind_from_binary_name(name) {
                kinds.insert(kind);
            }
        }
    }
}

fn discover_runner_kinds_in_dir(dir: &Path, kinds: &mut BTreeSet<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };

    for entry in entries.flatten() {
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();
        if let Some(kind) = kind_from_binary_name(&file_name) {
            kinds.insert(kind);
        }
    }
}

fn kind_from_binary_name(binary_name: &str) -> Option<String> {
    let name = binary_name.strip_suffix(".exe").unwrap_or(binary_name);
    let raw_kind = name.strip_prefix("turin-channel-")?;
    let kind = ChannelKind::parse(raw_kind).ok()?;
    Some(kind.to_string())
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

fn platform_binary_name(binary_name: &str) -> String {
    if cfg!(windows) {
        format!("{binary_name}.exe")
    } else {
        binary_name.to_string()
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
            external_runner_binary_name("email.smtp"),
            "turin-channel-email-smtp"
        );
    }

    #[test]
    fn extracts_channel_kind_from_binary_name() {
        assert_eq!(
            kind_from_binary_name("turin-channel-rocketchat"),
            Some("rocketchat".to_string())
        );
        assert_eq!(
            kind_from_binary_name("turin-channel-whatsapp.exe"),
            Some("whatsapp".to_string())
        );
        assert_eq!(kind_from_binary_name("turin"), None);
    }
}
