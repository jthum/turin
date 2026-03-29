use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use dialoguer::Confirm;
use similar::TextDiff;
use toml_edit::{DocumentMut, Item, Table, Value, value as toml_value};

#[derive(Debug, Clone)]
pub(crate) struct PlannedWrite {
    pub(crate) path: PathBuf,
    pub(crate) contents: String,
    pub(crate) display_contents: Option<String>,
}

impl PlannedWrite {
    pub(crate) fn new(path: PathBuf, contents: String) -> Self {
        Self {
            path,
            contents,
            display_contents: None,
        }
    }

    pub(crate) fn with_display_contents(mut self, display_contents: String) -> Self {
        self.display_contents = Some(display_contents);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ConfiguredChannel {
    pub(crate) id: String,
    pub(crate) kind: String,
    pub(crate) enabled: bool,
    pub(crate) agent_id: Option<String>,
}

pub(crate) fn config_dir(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

pub(crate) fn resolve_channels_dir(config_path: &Path) -> Result<PathBuf> {
    let base = config_dir(config_path);
    if !config_path.is_file() {
        return Ok(base.join("channels"));
    }

    let raw = std::fs::read_to_string(config_path)
        .with_context(|| format!("Failed to read '{}'", config_path.display()))?;
    let parsed: toml::Value = toml::from_str(&raw)
        .with_context(|| format!("Failed to parse '{}'", config_path.display()))?;
    let channels_dir = parsed
        .get("daemon")
        .and_then(|value| value.get("channels_dir"))
        .and_then(toml::Value::as_str)
        .unwrap_or("channels");
    Ok(base.join(channels_dir))
}

pub(crate) fn load_existing(path: &Path) -> Result<Option<String>> {
    if !path.exists() {
        return Ok(None);
    }
    let body = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read '{}'", path.display()))?;
    Ok(Some(body))
}

pub(crate) fn load_configured_channels(config_path: &Path) -> Result<Vec<ConfiguredChannel>> {
    let channels_dir = resolve_channels_dir(config_path)?;
    if !channels_dir.exists() {
        return Ok(Vec::new());
    }

    let mut channels = Vec::new();
    for entry in std::fs::read_dir(&channels_dir)
        .with_context(|| format!("Failed to read '{}'", channels_dir.display()))?
    {
        let entry = entry?;
        let entry_path = entry.path();
        if !entry_path.is_dir() {
            continue;
        }

        let channel_id = entry.file_name().to_string_lossy().to_string();
        let channel_path = entry_path.join("channel.toml");
        if !channel_path.is_file() {
            continue;
        }

        let raw = std::fs::read_to_string(&channel_path)
            .with_context(|| format!("Failed to read '{}'", channel_path.display()))?;
        let parsed: toml::Value = toml::from_str(&raw)
            .with_context(|| format!("Failed to parse '{}'", channel_path.display()))?;
        let Some(kind) = parsed.get("kind").and_then(toml::Value::as_str) else {
            continue;
        };

        channels.push(ConfiguredChannel {
            id: channel_id,
            kind: kind.to_string(),
            enabled: parsed
                .get("enabled")
                .and_then(toml::Value::as_bool)
                .unwrap_or(true),
            agent_id: parsed
                .get("agent_id")
                .and_then(toml::Value::as_str)
                .map(ToString::to_string),
        });
    }

    channels.sort_by(|left, right| left.id.cmp(&right.id));
    Ok(channels)
}

pub(crate) fn confirm_and_write(plans: &[PlannedWrite]) -> Result<()> {
    if plans.is_empty() {
        return Ok(());
    }

    for plan in plans {
        let existing = load_existing(&plan.path)?;
        println!("\n==> {}", plan.path.display());
        match existing {
            Some(existing) => {
                let redacted_existing;
                let before = if plan.display_contents.is_some() {
                    redacted_existing = redact_env_contents(&existing);
                    redacted_existing.as_str()
                } else {
                    existing.as_str()
                };
                print_diff(
                    &plan.path.display().to_string(),
                    before,
                    plan.display_contents
                        .as_deref()
                        .unwrap_or(plan.contents.as_str()),
                );
            }
            None => print_diff(
                &plan.path.display().to_string(),
                "",
                plan.display_contents
                    .as_deref()
                    .unwrap_or(plan.contents.as_str()),
            ),
        }
    }

    if !Confirm::new()
        .with_prompt("Apply these changes?")
        .default(true)
        .interact()?
    {
        anyhow::bail!("Aborted without writing changes");
    }

    for plan in plans {
        if let Some(parent) = plan.path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create '{}'", parent.display()))?;
        }
        std::fs::write(&plan.path, &plan.contents)
            .with_context(|| format!("Failed to write '{}'", plan.path.display()))?;
    }

    Ok(())
}

fn print_diff(label: &str, old: &str, new: &str) {
    let diff = TextDiff::from_lines(old, new);
    let rendered = diff
        .unified_diff()
        .context_radius(3)
        .header(&format!("a/{label}"), &format!("b/{label}"))
        .to_string();
    if rendered.trim().is_empty() {
        println!("(no changes)");
    } else {
        print!("{rendered}");
    }
}

pub(crate) fn merge_env_file(
    existing: Option<&str>,
    updates: &BTreeMap<String, String>,
) -> (String, String) {
    let mut current: BTreeMap<String, String> = BTreeMap::new();
    let mut preserved_lines = Vec::new();

    if let Some(existing) = existing {
        for line in existing.lines() {
            if let Some((key, value)) = parse_env_assignment(line) {
                current.insert(key.to_string(), value.to_string());
            } else {
                preserved_lines.push(line.to_string());
            }
        }
    }

    for (key, value) in updates {
        current.insert(key.clone(), value.clone());
    }

    let mut body = String::new();
    for line in preserved_lines {
        if !line.trim().is_empty() {
            body.push_str(&line);
            body.push('\n');
        }
    }
    for (key, value) in &current {
        body.push_str(key);
        body.push('=');
        body.push_str(value);
        body.push('\n');
    }

    let mut display = String::new();
    display.push_str(&redact_env_contents(&body));

    (body, display)
}

fn redact_env_contents(body: &str) -> String {
    let mut display = String::new();
    for line in body.lines() {
        if let Some((key, _)) = parse_env_assignment(line) {
            display.push_str(key);
            display.push_str("=***REDACTED***\n");
        } else {
            display.push_str(line);
            display.push('\n');
        }
    }
    display
}

fn parse_env_assignment(line: &str) -> Option<(&str, &str)> {
    if line.trim_start().starts_with('#') {
        return None;
    }
    let (key, value) = line.split_once('=')?;
    let key = key.trim();
    if key.is_empty() {
        return None;
    }
    Some((key, value.trim()))
}

pub(crate) fn render_channel_file(
    existing: Option<&str>,
    enabled: bool,
    kind: &str,
    agent_id: &str,
    settings: &BTreeMap<String, serde_json::Value>,
) -> Result<String> {
    let mut doc = if let Some(existing) = existing {
        existing.parse::<DocumentMut>().with_context(
            || "Failed to parse existing channel.toml while staging updated channel settings",
        )?
    } else {
        DocumentMut::new()
    };

    doc["enabled"] = toml_value(enabled);
    doc["kind"] = toml_value(kind);
    doc["agent_id"] = toml_value(agent_id);
    for (key, json_value) in settings {
        doc[key] = json_to_toml_item(json_value)?;
    }

    Ok(doc.to_string())
}

fn json_to_toml_item(value: &serde_json::Value) -> Result<Item> {
    Ok(match value {
        serde_json::Value::Null => Item::None,
        serde_json::Value::Bool(flag) => toml_value(*flag),
        serde_json::Value::Number(number) => {
            if let Some(integer) = number.as_i64() {
                toml_value(integer)
            } else if let Some(integer) = number.as_u64() {
                toml_value(integer as i64)
            } else if let Some(float) = number.as_f64() {
                toml_value(float)
            } else {
                anyhow::bail!("Unsupported numeric value '{number}' for TOML rendering");
            }
        }
        serde_json::Value::String(text) => toml_value(text.as_str()),
        serde_json::Value::Array(values) => {
            let mut array = toml_edit::Array::default();
            for value in values {
                array.push(json_to_toml_value(value)?);
            }
            Item::Value(Value::Array(array))
        }
        serde_json::Value::Object(map) => {
            let mut table = Table::new();
            for (key, value) in map {
                table[key] = json_to_toml_item(value)?;
            }
            Item::Table(table)
        }
    })
}

fn json_to_toml_value(value: &serde_json::Value) -> Result<Value> {
    match json_to_toml_item(value)? {
        Item::Value(value) => Ok(value),
        _ => Err(anyhow!(
            "Only scalar or array values can be nested in TOML arrays"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_merge_replaces_updated_keys_and_redacts_display() {
        let mut updates = BTreeMap::new();
        updates.insert("TOKEN".to_string(), "secret".to_string());
        let (body, display) = merge_env_file(Some("FOO=bar\nTOKEN=old\n"), &updates);
        assert!(body.contains("TOKEN=secret"));
        assert!(display.contains("TOKEN=***REDACTED***"));
        assert!(!display.contains("secret"));
    }

    #[test]
    fn load_configured_channels_reads_channel_directories() {
        let temp = tempfile::tempdir().expect("tempdir");
        let config_path = temp.path().join("turin.toml");
        std::fs::write(&config_path, "[daemon]\nchannels_dir = \"channels\"\n")
            .expect("config written");
        let channel_dir = temp.path().join("channels/telegram");
        std::fs::create_dir_all(&channel_dir).expect("channel dir");
        std::fs::write(
            channel_dir.join("channel.toml"),
            "enabled = true\nkind = \"telegram\"\nagent_id = \"default\"\n",
        )
        .expect("channel file");

        let channels = load_configured_channels(&config_path).expect("channels loaded");
        assert_eq!(
            channels,
            vec![ConfiguredChannel {
                id: "telegram".to_string(),
                kind: "telegram".to_string(),
                enabled: true,
                agent_id: Some("default".to_string()),
            }]
        );
    }
}
