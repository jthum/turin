use std::path::Path;

use anyhow::{Context, Result};
use turin_daemon_protocol::{ContextPersistenceParams, StoreTargetParams};

use crate::daemon::state::SessionSummary;
use crate::kernel::config::{ContextPersistenceConfig, StoreTargetConfig, TurinConfig};
use crate::persistence::manager::StoreSelector;

pub(super) fn normalize_bootstrap_paths(config: &mut TurinConfig, config_base: &Path) {
    config.normalize_runtime_paths(config_base);
}

pub(super) fn context_store_selector_from_params(
    config: &TurinConfig,
    persistence: Option<&ContextPersistenceParams>,
) -> Result<StoreSelector> {
    let Some(persistence) = persistence else {
        return config.persistence.top_level_state_selector();
    };
    let context = ContextPersistenceConfig {
        state: persistence
            .state
            .as_ref()
            .map(store_target_config_from_params),
        store: persistence
            .store
            .as_ref()
            .map(store_target_config_from_params),
    };
    if persistence.store.is_some() {
        config
            .persistence
            .resolve_context_store_selector(Some(&context))
    } else if persistence.state.is_some() {
        config
            .persistence
            .resolve_context_state_selector(Some(&context))
    } else {
        config.persistence.top_level_state_selector()
    }
}

fn store_target_config_from_params(value: &StoreTargetParams) -> StoreTargetConfig {
    StoreTargetConfig {
        path: value.path.clone(),
        alias: value.alias.clone(),
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
        origin_id: row.origin_id.clone(),
        metadata: row
            .metadata
            .as_deref()
            .and_then(|raw| serde_json::from_str(raw).ok())
            .or_else(|| {
                row.metadata
                    .as_ref()
                    .map(|raw| serde_json::Value::String(raw.clone()))
            }),
        parent_internal_id: row.parent_session_id,
        root_internal_id: row.root_session_id,
        origin_turn_id: row.origin_turn_id,
        relation_kind: row.relation_kind.clone(),
        thread_key: row.thread_key.clone(),
        visibility: row.visibility.clone(),
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
