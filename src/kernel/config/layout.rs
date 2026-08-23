use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use turin_types::layout::{
    DEFAULT_LAYOUT_AGENTS_DIR, DEFAULT_LAYOUT_DAEMON_SOCKET, DEFAULT_LAYOUT_DATA_DIR,
    DEFAULT_LAYOUT_ENV_FILE, DEFAULT_LAYOUT_HARNESSES_DIR, DEFAULT_LAYOUT_SCOPES_DIR,
    DEFAULT_LAYOUT_STATES_DIR, DEFAULT_LAYOUT_STORES_DIR, config_dir, config_workspace_anchor,
    resolve_relative_to,
};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LayoutConfig {
    #[serde(skip)]
    environment: BTreeMap<String, String>,
    #[serde(default)]
    pub root: Option<String>,
    #[serde(default = "default_layout_data_dir")]
    pub data_dir: String,
    #[serde(default = "default_layout_states_dir")]
    pub states_dir: String,
    #[serde(default = "default_layout_stores_dir")]
    pub stores_dir: String,
    #[serde(default = "default_layout_harnesses_dir")]
    pub harnesses_dir: String,
    #[serde(default = "default_layout_agents_dir")]
    pub agents_dir: String,
    #[serde(default = "default_layout_scopes_dir")]
    pub scopes_dir: String,
    #[serde(default = "default_layout_env_file")]
    pub env_file: String,
    #[serde(default = "default_layout_daemon_socket")]
    pub daemon_socket: String,
}

impl Default for LayoutConfig {
    fn default() -> Self {
        Self {
            environment: BTreeMap::new(),
            root: None,
            data_dir: default_layout_data_dir(),
            states_dir: default_layout_states_dir(),
            stores_dir: default_layout_stores_dir(),
            harnesses_dir: default_layout_harnesses_dir(),
            agents_dir: default_layout_agents_dir(),
            scopes_dir: default_layout_scopes_dir(),
            env_file: default_layout_env_file(),
            daemon_socket: default_layout_daemon_socket(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ResolvedLayout {
    pub config_path: PathBuf,
    pub config_dir: PathBuf,
    pub workspace_anchor: PathBuf,
    pub root: PathBuf,
    pub data_dir: PathBuf,
    pub states_dir: PathBuf,
    pub stores_dir: PathBuf,
    pub default_state_db: PathBuf,
    pub harnesses_dir: PathBuf,
    pub agents_dir: PathBuf,
    pub scopes_dir: PathBuf,
    pub env_file: PathBuf,
    pub daemon_socket: PathBuf,
}

impl LayoutConfig {
    pub(super) fn load_environment(&mut self, env_path: &Path) -> Result<()> {
        self.environment.clear();
        if !env_path.is_file() {
            return Ok(());
        }
        for item in dotenvy::from_path_iter(env_path)
            .with_context(|| format!("Failed to parse '{}'", env_path.display()))?
        {
            let (key, value) =
                item.with_context(|| format!("Failed to parse '{}'", env_path.display()))?;
            self.environment.insert(key, value);
        }
        Ok(())
    }

    pub(super) fn environment_value(&self, key: &str) -> Option<String> {
        self.environment.get(key).cloned()
    }

    pub fn resolve(&self, config_path: &Path) -> ResolvedLayout {
        let config_path = config_path.to_path_buf();
        let config_dir = config_dir(&config_path);
        self.resolve_from_config_dir(config_path, config_dir)
    }

    pub fn resolve_from_config_dir(
        &self,
        config_path: PathBuf,
        config_dir: PathBuf,
    ) -> ResolvedLayout {
        let workspace_anchor = config_workspace_anchor(&config_dir);
        let root = self
            .root
            .as_deref()
            .map(Path::new)
            .map(|path| resolve_relative_to(&workspace_anchor, path))
            .unwrap_or_else(|| config_dir.clone());
        let data_dir = resolve_relative_to(&root, Path::new(&self.data_dir));
        let states_dir = resolve_relative_to(&data_dir, Path::new(&self.states_dir));
        let stores_dir = resolve_relative_to(&data_dir, Path::new(&self.stores_dir));
        let harnesses_dir = resolve_relative_to(&root, Path::new(&self.harnesses_dir));
        let agents_dir = resolve_relative_to(&root, Path::new(&self.agents_dir));
        let scopes_dir = resolve_relative_to(&root, Path::new(&self.scopes_dir));
        let env_file = resolve_relative_to(&root, Path::new(&self.env_file));
        let daemon_socket = resolve_relative_to(&root, Path::new(&self.daemon_socket));

        ResolvedLayout {
            config_path,
            config_dir,
            workspace_anchor,
            root,
            data_dir: data_dir.clone(),
            states_dir: states_dir.clone(),
            stores_dir,
            default_state_db: data_dir.join("state.db"),
            harnesses_dir,
            agents_dir,
            scopes_dir,
            env_file,
            daemon_socket,
        }
    }
}

pub fn default_layout_data_dir() -> String {
    DEFAULT_LAYOUT_DATA_DIR.to_string()
}

pub fn default_layout_states_dir() -> String {
    DEFAULT_LAYOUT_STATES_DIR.to_string()
}

pub fn default_layout_stores_dir() -> String {
    DEFAULT_LAYOUT_STORES_DIR.to_string()
}

pub fn default_layout_harnesses_dir() -> String {
    DEFAULT_LAYOUT_HARNESSES_DIR.to_string()
}

pub fn default_layout_agents_dir() -> String {
    DEFAULT_LAYOUT_AGENTS_DIR.to_string()
}

pub fn default_layout_scopes_dir() -> String {
    DEFAULT_LAYOUT_SCOPES_DIR.to_string()
}

pub fn default_layout_env_file() -> String {
    DEFAULT_LAYOUT_ENV_FILE.to_string()
}

pub fn default_layout_daemon_socket() -> String {
    DEFAULT_LAYOUT_DAEMON_SOCKET.to_string()
}
