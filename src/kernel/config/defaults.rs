use super::StoreTargetConfig;

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

pub(super) fn default_state_path() -> String {
    ".turin/state.db".to_string()
}

pub(super) fn default_harness_directory() -> String {
    ".turin/harnesses".to_string()
}

pub(super) fn default_harness_fs_root() -> String {
    ".".to_string()
}

pub(super) fn default_embedding_model() -> String {
    "text-embedding-3-small".to_string()
}

pub(super) fn default_embedding_dimensions() -> usize {
    1536
}

pub(super) fn default_daemon_agents_dir() -> String {
    "agents".to_string()
}

pub(super) fn default_daemon_harnesses_dir() -> String {
    "harnesses".to_string()
}

pub(super) fn default_daemon_channels_dir() -> String {
    "channels".to_string()
}

pub(super) fn default_daemon_endpoint() -> String {
    ".turin/daemon.sock".to_string()
}

pub(super) fn default_remote_bind() -> String {
    "127.0.0.1:9324".to_string()
}

pub(super) fn default_remote_auth_token_env() -> String {
    "TURIN_REMOTE_TOKEN".to_string()
}

pub(super) fn default_remote_event_keepalive_secs() -> u64 {
    15
}

pub(super) fn default_state_target() -> StoreTargetConfig {
    StoreTargetConfig::from_path(default_state_path())
}
