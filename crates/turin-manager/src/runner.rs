use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, anyhow};
use serde_json::Value;
use turin_channel_core::{
    ChannelAdapterManifest, ChannelAuthFlowPollRequest, ChannelAuthFlowPollResponse,
    ChannelAuthFlowStartRequest, ChannelAuthFlowStartResponse, ChannelKind,
    validate_adapter_manifest,
};

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

    if let Some(command) = resolve_workspace_runner_command(&binary_name) {
        return Ok(command);
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

pub(crate) fn validate_external_runner_settings(
    kind: &str,
    settings: &Value,
    env_overrides: &std::collections::BTreeMap<String, String>,
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
        .arg(&settings_json)
        .envs(env_overrides);

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

pub(crate) fn start_external_auth_flow(
    kind: &str,
    request: &ChannelAuthFlowStartRequest,
) -> Result<ChannelAuthFlowStartResponse> {
    let runner = resolve_external_runner_command(kind)?;
    let request_json =
        serde_json::to_string(request).context("Failed to encode auth flow start request")?;

    let mut command = Command::new(&runner.program);
    for arg in &runner.args_prefix {
        command.arg(arg);
    }
    command
        .arg("setup-auth-flow-start")
        .arg("--request-json")
        .arg(&request_json);

    let output = command
        .output()
        .with_context(|| format!("Failed to launch '{}'", runner.display))?;

    if !output.status.success() {
        return Err(anyhow!(runner_output_detail(&output).unwrap_or_else(
            || format!("runner exited with status {}", output.status)
        )));
    }

    serde_json::from_slice(&output.stdout)
        .context("Failed to decode auth flow start response from channel runner")
}

pub(crate) fn poll_external_auth_flow(
    kind: &str,
    request: &ChannelAuthFlowPollRequest,
) -> Result<ChannelAuthFlowPollResponse> {
    let runner = resolve_external_runner_command(kind)?;
    let request_json =
        serde_json::to_string(request).context("Failed to encode auth flow poll request")?;

    let mut command = Command::new(&runner.program);
    for arg in &runner.args_prefix {
        command.arg(arg);
    }
    command
        .arg("setup-auth-flow-poll")
        .arg("--request-json")
        .arg(&request_json);

    let output = command
        .output()
        .with_context(|| format!("Failed to launch '{}'", runner.display))?;

    if !output.status.success() {
        return Err(anyhow!(runner_output_detail(&output).unwrap_or_else(
            || format!("runner exited with status {}", output.status)
        )));
    }

    serde_json::from_slice(&output.stdout)
        .context("Failed to decode auth flow poll response from channel runner")
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
    let Some(metadata) = workspace_metadata() else {
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
    use turin_channel_core::{ChannelAuthFlowPollResponse, ChannelAuthFlowStartRequest};

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

    #[cfg(unix)]
    #[test]
    fn auth_flow_commands_round_trip_through_runner() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().expect("tempdir");
        let runner = temp.path().join("fake-whatsapp-runner.sh");
        std::fs::write(
            &runner,
            "#!/bin/sh\nif [ \"$1\" = \"setup-auth-flow-start\" ]; then\n  printf '%s\\n' '{\"session\":{\"ticket\":\"abc\"},\"display\":{\"message\":\"Scan the QR code\",\"qr_text\":\"otpauth://pair\"}}'\n  exit 0\nfi\nif [ \"$1\" = \"setup-auth-flow-poll\" ]; then\n  printf '%s\\n' '{\"state\":\"complete\",\"values\":[{\"target\":{\"kind\":\"channel_setting\",\"name\":\"session_id\"},\"value\":\"session-1\"}],\"message\":\"Pairing complete\"}'\n  exit 0\nfi\nexit 1\n",
        )
        .expect("script written");
        let mut perms = std::fs::metadata(&runner).expect("metadata").permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&runner, perms).expect("permissions");

        let previous = std::env::var_os("TURIN_CHANNEL_WHATSAPP_BIN");
        unsafe {
            std::env::set_var("TURIN_CHANNEL_WHATSAPP_BIN", &runner);
        }

        let start = start_external_auth_flow(
            "whatsapp",
            &ChannelAuthFlowStartRequest {
                flow_id: "pair".to_string(),
                current_settings: serde_json::json!({}),
            },
        )
        .expect("start response");
        assert_eq!(start.session["ticket"], "abc");
        assert_eq!(start.display.message.as_deref(), Some("Scan the QR code"));

        let poll = poll_external_auth_flow(
            "whatsapp",
            &turin_channel_core::ChannelAuthFlowPollRequest {
                flow_id: "pair".to_string(),
                session: start.session,
                current_settings: serde_json::json!({}),
            },
        )
        .expect("poll response");
        match poll {
            ChannelAuthFlowPollResponse::Complete { values, message } => {
                assert_eq!(message.as_deref(), Some("Pairing complete"));
                assert_eq!(values.len(), 1);
                assert_eq!(values[0].target.name, "session_id");
                assert_eq!(values[0].value, serde_json::json!("session-1"));
            }
            other => panic!("unexpected poll response: {other:?}"),
        }

        if let Some(value) = previous {
            unsafe {
                std::env::set_var("TURIN_CHANNEL_WHATSAPP_BIN", value);
            }
        } else {
            unsafe {
                std::env::remove_var("TURIN_CHANNEL_WHATSAPP_BIN");
            }
        }
    }
}
