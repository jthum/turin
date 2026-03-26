use std::path::Path;

use anyhow::{Context, Result};

use crate::daemon::state::SessionSummary;
use crate::kernel::config::TurinConfig;

pub(super) fn normalize_bootstrap_paths(config: &mut TurinConfig, config_base: &Path) {
    let workspace_root = config.resolve_workspace_root(config_base);
    config.kernel.workspace_root = workspace_root.display().to_string();

    if Path::new(&config.harness.directory).is_relative() {
        config.harness.directory = workspace_root
            .join(&config.harness.directory)
            .display()
            .to_string();
    }

    if Path::new(&config.harness.fs_root).is_relative() && config.harness.fs_root != "." {
        config.harness.fs_root = workspace_root
            .join(&config.harness.fs_root)
            .display()
            .to_string();
    }
}

pub(super) fn validate_agent_id(agent_id: &str) -> Result<()> {
    if agent_id.trim().is_empty() {
        anyhow::bail!("Agent ID cannot be empty");
    }
    if agent_id == "default" {
        anyhow::bail!("'default' is reserved for the bootstrap agent");
    }
    if agent_id.contains('/') || agent_id.contains('\\') || agent_id.contains("..") {
        anyhow::bail!("Agent ID '{}' contains invalid path characters", agent_id);
    }
    Ok(())
}

pub(super) fn validate_harness_id(harness_id: &str) -> Result<()> {
    if harness_id.trim().is_empty() {
        anyhow::bail!("Harness ID cannot be empty");
    }
    if harness_id == "default" {
        anyhow::bail!("'default' is reserved for the bootstrap harness");
    }
    if harness_id.starts_with("agent::") {
        anyhow::bail!("Harness IDs cannot start with 'agent::'");
    }
    if harness_id.contains('/') || harness_id.contains('\\') || harness_id.contains("..") {
        anyhow::bail!(
            "Harness ID '{}' contains invalid path characters",
            harness_id
        );
    }
    Ok(())
}

pub(super) fn validate_channel_id(channel_id: &str) -> Result<()> {
    if channel_id.trim().is_empty() {
        anyhow::bail!("Channel ID cannot be empty");
    }
    if channel_id.contains('/') || channel_id.contains('\\') || channel_id.contains("..") {
        anyhow::bail!(
            "Channel ID '{}' contains invalid path characters",
            channel_id
        );
    }
    Ok(())
}

pub(super) fn json_object_to_toml_table(value: serde_json::Value) -> Result<toml::Table> {
    match json_to_toml_value(value)? {
        toml::Value::Table(table) => Ok(table),
        _ => anyhow::bail!("Channel settings must be a JSON object"),
    }
}

pub(super) fn merge_json_object_into_toml_table(
    table: &mut toml::Table,
    value: serde_json::Value,
) -> Result<()> {
    let updates = json_object_to_toml_table(value)?;
    for (key, value) in updates {
        table.insert(key, value);
    }
    Ok(())
}

pub(super) fn scaffold_local_harness(agent_dir: &Path) -> Result<()> {
    let harness_dir = agent_dir.join("harness");
    std::fs::create_dir_all(&harness_dir)
        .with_context(|| format!("Failed to create '{}'", harness_dir.display()))?;
    scaffold_harness_main(&harness_dir)
}

pub(super) fn local_harness_is_scaffold(harness_dir: &Path) -> Result<bool> {
    let mut entries = std::fs::read_dir(harness_dir)
        .with_context(|| format!("Failed to read '{}'", harness_dir.display()))?
        .collect::<std::io::Result<Vec<_>>>()
        .with_context(|| format!("Failed to enumerate '{}'", harness_dir.display()))?;
    entries.sort_by_key(|entry| entry.file_name());

    if entries.len() != 1 {
        return Ok(false);
    }

    let entry = &entries[0];
    if entry.file_name() != "main.lua" {
        return Ok(false);
    }

    let body = std::fs::read_to_string(entry.path())
        .with_context(|| format!("Failed to read '{}'", entry.path().display()))?;
    Ok(body == "-- Turin daemon scaffold\n")
}

pub(super) fn scaffold_shared_harness(harness_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(harness_dir)
        .with_context(|| format!("Failed to create '{}'", harness_dir.display()))?;
    scaffold_harness_main(harness_dir)
}

pub(super) fn scaffold_harness_main(harness_dir: &Path) -> Result<()> {
    let main_lua = harness_dir.join("main.lua");
    if main_lua.exists() {
        return Ok(());
    }

    let tmp_path = harness_dir.join(format!(".main.lua.{}.tmp", uuid::Uuid::now_v7().simple()));
    std::fs::write(&tmp_path, "-- Turin daemon scaffold\n")
        .with_context(|| format!("Failed to write '{}'", tmp_path.display()))?;
    std::fs::rename(&tmp_path, &main_lua).with_context(|| {
        format!(
            "Failed to atomically replace '{}' from '{}'",
            main_lua.display(),
            tmp_path.display()
        )
    })?;
    Ok(())
}

pub(super) fn session_summary_from_row(
    row: &crate::persistence::schema::SessionRow,
) -> SessionSummary {
    SessionSummary {
        internal_id: row.id,
        session_id: format_uuid_bytes_simple(&row.public_id),
        agent_id: row.agent_id.clone(),
        metadata: row
            .metadata
            .as_deref()
            .and_then(|raw| serde_json::from_str(raw).ok())
            .or_else(|| {
                row.metadata
                    .as_ref()
                    .map(|raw| serde_json::Value::String(raw.clone()))
            }),
        created_at: row.created_at.clone(),
    }
}

pub(super) fn format_uuid_bytes_simple(bytes: &[u8]) -> String {
    uuid::Uuid::from_slice(bytes)
        .map(|uuid| uuid.simple().to_string())
        .unwrap_or_else(|_| {
            let mut out = String::with_capacity(bytes.len() * 2);
            for byte in bytes {
                use std::fmt::Write as _;
                let _ = write!(&mut out, "{:02x}", byte);
            }
            out
        })
}

pub(super) fn parse_json_or_string(raw: &str) -> serde_json::Value {
    serde_json::from_str(raw).unwrap_or_else(|_| serde_json::Value::String(raw.to_string()))
}

pub(super) fn session_title_from_metadata(metadata: Option<&str>) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(metadata?).ok()?;
    value
        .get("title")
        .and_then(|value| value.as_str())
        .map(str::trim)
        .filter(|title| !title.is_empty())
        .map(str::to_string)
}

pub(super) fn issue_path_is_under(issue_path: &str, root: &Path) -> bool {
    let issue_path = Path::new(issue_path);
    issue_path == root || issue_path.starts_with(root)
}

fn json_to_toml_value(value: serde_json::Value) -> Result<toml::Value> {
    Ok(match value {
        serde_json::Value::Null => {
            anyhow::bail!("Null values are not supported in channel settings")
        }
        serde_json::Value::Bool(v) => toml::Value::Boolean(v),
        serde_json::Value::Number(v) => {
            if let Some(i) = v.as_i64() {
                toml::Value::Integer(i)
            } else if let Some(f) = v.as_f64() {
                toml::Value::Float(f)
            } else {
                anyhow::bail!("Unsupported numeric value in channel settings")
            }
        }
        serde_json::Value::String(v) => toml::Value::String(v),
        serde_json::Value::Array(values) => toml::Value::Array(
            values
                .into_iter()
                .map(json_to_toml_value)
                .collect::<Result<Vec<_>>>()?,
        ),
        serde_json::Value::Object(map) => toml::Value::Table(
            map.into_iter()
                .map(|(key, value)| Ok((key, json_to_toml_value(value)?)))
                .collect::<Result<toml::Table>>()?,
        ),
    })
}
