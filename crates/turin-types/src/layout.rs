use std::path::{Path, PathBuf};

pub const DEFAULT_BOOTSTRAP_CONFIG_PATH: &str = ".turin/config.toml";
pub const DEFAULT_LAYOUT_ROOT: &str = ".turin";
pub const DEFAULT_LAYOUT_DATA_DIR: &str = "data";
pub const DEFAULT_LAYOUT_STATES_DIR: &str = "states";
pub const DEFAULT_LAYOUT_STORES_DIR: &str = "stores";
pub const DEFAULT_LAYOUT_HARNESSES_DIR: &str = "harnesses";
pub const DEFAULT_LAYOUT_AGENTS_DIR: &str = "runtime/agents";
pub const DEFAULT_LAYOUT_SCOPES_DIR: &str = "scopes";
pub const DEFAULT_LAYOUT_ENV_FILE: &str = ".env";
pub const DEFAULT_LAYOUT_DAEMON_SOCKET: &str = "daemon.sock";
pub const DEFAULT_LAYOUT_DAEMON_LOG_FILE: &str = "daemon.log";
pub const DEFAULT_CODE_INDEX_DB_FILE: &str = "codebase.db";
pub const DEFAULT_UI_PROFILES_PATH: &str = ".turin/ui-profiles.toml";
pub const DEFAULT_BOOTSTRAP_DAEMON_ENDPOINT_PATH: &str = ".turin/daemon.sock";

pub fn config_dir(config_path: &Path) -> PathBuf {
    config_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn config_workspace_anchor(config_dir: &Path) -> PathBuf {
    if config_dir.file_name().and_then(|name| name.to_str()) == Some(DEFAULT_LAYOUT_ROOT) {
        return config_dir.parent().unwrap_or(config_dir).to_path_buf();
    }
    config_dir.to_path_buf()
}

pub fn resolve_relative_to(base: &Path, value: &Path) -> PathBuf {
    if value.is_absolute() {
        value.to_path_buf()
    } else {
        base.join(value)
    }
}

pub fn default_layout_root_for_workspace(workspace_root: &Path) -> PathBuf {
    workspace_root.join(DEFAULT_LAYOUT_ROOT)
}

pub fn default_data_dir_for_workspace(workspace_root: &Path) -> PathBuf {
    default_layout_root_for_workspace(workspace_root).join(DEFAULT_LAYOUT_DATA_DIR)
}

pub fn default_states_dir_for_workspace(workspace_root: &Path) -> PathBuf {
    default_data_dir_for_workspace(workspace_root).join(DEFAULT_LAYOUT_STATES_DIR)
}

pub fn default_stores_dir_for_workspace(workspace_root: &Path) -> PathBuf {
    default_data_dir_for_workspace(workspace_root).join(DEFAULT_LAYOUT_STORES_DIR)
}

pub fn default_state_db_for_workspace(workspace_root: &Path) -> PathBuf {
    default_data_dir_for_workspace(workspace_root).join("state.db")
}

pub fn default_daemon_log_for_workspace(workspace_root: &Path) -> PathBuf {
    default_layout_root_for_workspace(workspace_root).join(DEFAULT_LAYOUT_DAEMON_LOG_FILE)
}

pub fn default_code_index_db_for_workspace(workspace_root: &Path) -> PathBuf {
    default_layout_root_for_workspace(workspace_root).join(DEFAULT_CODE_INDEX_DB_FILE)
}
