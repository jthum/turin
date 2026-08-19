use super::StoreTargetConfig;
use turin_types::layout::{
    DEFAULT_LAYOUT_AGENTS_DIR, DEFAULT_LAYOUT_CHANNELS_DIR, DEFAULT_LAYOUT_DAEMON_SOCKET,
    DEFAULT_LAYOUT_DATA_DIR, DEFAULT_LAYOUT_HARNESSES_DIR,
};

pub(super) fn default_system_prompt() -> String {
    "You are a helpful coding assistant.".to_string()
}

pub(super) fn default_agent_id() -> String {
    "default".to_string()
}

pub(super) fn default_workspace_root() -> String {
    ".".to_string()
}

pub(super) fn default_max_turns() -> u32 {
    50
}

pub(super) fn default_heartbeat_interval() -> u32 {
    30
}

pub(super) fn default_idle_timeout_seconds() -> Option<u64> {
    Some(20)
}

pub(super) fn default_linked_runtime_lanes() -> usize {
    4
}

pub(super) fn default_state_path() -> String {
    format!("{DEFAULT_LAYOUT_DATA_DIR}/state.db")
}

pub(super) fn default_harness_directory() -> String {
    DEFAULT_LAYOUT_HARNESSES_DIR.to_string()
}

pub(super) fn default_harness_fs_root() -> String {
    ".".to_string()
}

pub(super) fn default_harness_memory_limit_mb() -> u32 {
    32
}

pub(super) fn default_embedding_model() -> String {
    "text-embedding-3-small".to_string()
}

pub(super) fn default_embedding_dimensions() -> usize {
    1536
}

pub(super) fn default_daemon_agents_dir() -> String {
    DEFAULT_LAYOUT_AGENTS_DIR.to_string()
}

pub(super) fn default_daemon_harnesses_dir() -> String {
    DEFAULT_LAYOUT_HARNESSES_DIR.to_string()
}

pub(super) fn default_daemon_channels_dir() -> String {
    DEFAULT_LAYOUT_CHANNELS_DIR.to_string()
}

pub(super) fn default_daemon_endpoint() -> String {
    DEFAULT_LAYOUT_DAEMON_SOCKET.to_string()
}

pub(super) fn default_daemon_runtime_db() -> String {
    format!("{DEFAULT_LAYOUT_DATA_DIR}/runtime.db")
}

pub(super) fn default_remote_bind() -> String {
    "127.0.0.1:9324".to_string()
}

pub(super) fn default_remote_auth_token_env() -> String {
    "TURIN_REMOTE_TOKEN".to_string()
}

pub(super) fn default_remote_event_keepalive_seconds() -> u64 {
    15
}

pub(super) fn default_state_target() -> StoreTargetConfig {
    StoreTargetConfig::from_path(default_state_path())
}
