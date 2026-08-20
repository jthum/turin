use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

use super::AgentFileConfig;

pub(crate) fn read_agent_file(agent_dir: &Path) -> Result<Option<AgentFileConfig>> {
    read_toml_file(&agent_dir.join("config.toml"))
}

pub(crate) fn write_agent_file(agent_dir: &Path, config: &AgentFileConfig) -> Result<()> {
    write_config_file(agent_dir, config, "agent")
}

fn read_toml_file<T>(path: &Path) -> Result<Option<T>>
where
    T: for<'de> serde::Deserialize<'de>,
{
    if !path.exists() {
        return Ok(None);
    }

    let raw =
        fs::read_to_string(path).with_context(|| format!("Failed to read '{}'", path.display()))?;
    let parsed =
        toml::from_str(&raw).with_context(|| format!("Failed to parse '{}'", path.display()))?;
    Ok(Some(parsed))
}

fn write_config_file<T>(dir: &Path, config: &T, label: &str) -> Result<()>
where
    T: Serialize,
{
    fs::create_dir_all(dir)
        .with_context(|| format!("Failed to create {} directory '{}'", label, dir.display()))?;
    let config_path = dir.join("config.toml");
    let tmp_path = dir.join(format!(
        ".config.toml.{}.tmp",
        uuid::Uuid::now_v7().simple()
    ));
    let body = toml::to_string_pretty(config)
        .with_context(|| format!("Failed to serialize '{}'", config_path.display()))?;
    fs::write(&tmp_path, body)
        .with_context(|| format!("Failed to write '{}'", tmp_path.display()))?;
    fs::rename(&tmp_path, &config_path).with_context(|| {
        format!(
            "Failed to atomically replace '{}' from '{}'",
            config_path.display(),
            tmp_path.display()
        )
    })?;
    Ok(())
}
