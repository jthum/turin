use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use turin_local_ipc::resolve_endpoint as resolve_local_ipc_endpoint;

pub use turin_types::{AgentMode, ThinkingConfig, ToolSelectionConfig, ToolsConfig};

/// Top-level Turin configuration, parsed from `turin.toml`.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct TurinConfig {
    #[serde(default)]
    pub tools: ToolsConfig,
    pub agent: AgentConfig,
    /// Optional map of additional peer agents that can be orchestrated by the `AgentManager`
    #[serde(default)]
    pub agents: std::collections::HashMap<String, AgentConfig>,
    #[serde(default)]
    pub kernel: KernelConfig,
    #[serde(default)]
    pub persistence: PersistenceConfig,
    #[serde(default)]
    pub harness: HarnessConfig,
    #[serde(default)]
    pub harnesses: std::collections::HashMap<String, HarnessConfig>,
    #[serde(default)]
    pub providers: ProvidersConfig,
    #[serde(default)]
    pub embeddings: Option<EmbeddingConfig>,
    #[serde(default)]
    pub governance: GovernanceConfig,
    #[serde(default)]
    pub daemon: DaemonConfig,
    #[serde(default)]
    pub remote: RemoteConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingConfig {
    pub provider: String,
    #[serde(default = "default_embedding_model")]
    pub model: String,
    #[serde(default = "default_embedding_dimensions")]
    pub dimensions: usize,
}

impl EmbeddingConfig {
    pub fn noop() -> Self {
        Self {
            provider: "noop".to_string(),
            model: default_embedding_model(),
            dimensions: default_embedding_dimensions(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct AgentConfig {
    /// The identity string for the agent instance (e.g. "default", "coder", "reviewer")
    #[serde(default = "default_agent_id")]
    pub id: String,
    /// System prompt for the LLM
    #[serde(default = "default_system_prompt")]
    pub system_prompt: String,
    /// Model identifier (e.g., "claude-sonnet-4-20250514", "gpt-4o")
    pub model: String,
    /// Provider name ("anthropic" or "openai")
    pub provider: String,
    /// Extended thinking configuration
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
    /// Agent execution mode ("auto", "stateful", "stateless")
    #[serde(default)]
    pub mode: AgentMode,
    /// Optional per-agent harness binding. Omit to use the default `[harness]`.
    #[serde(default)]
    pub harness: Option<String>,
    /// Optional idle shutdown grace period for peer runtimes.
    #[serde(default)]
    pub idle_grace_secs: Option<u64>,
    #[serde(default)]
    pub tools: ToolsConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct KernelConfig {
    /// Root directory for workspace-relative paths
    #[serde(default = "default_workspace_root")]
    pub workspace_root: String,
    /// Maximum turns before the agent loop exits
    #[serde(default = "default_max_turns")]
    pub max_turns: u32,
    /// Heartbeat interval in seconds
    #[serde(default = "default_heartbeat_interval")]
    pub heartbeat_interval_secs: u32,
    /// Initial spawn depth (for recursive agents)
    #[serde(default)]
    pub initial_spawn_depth: u32,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            workspace_root: default_workspace_root(),
            max_turns: default_max_turns(),
            heartbeat_interval_secs: default_heartbeat_interval(),
            initial_spawn_depth: 0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct PersistenceConfig {
    /// Path to the libSQL database file
    #[serde(default = "default_database_path")]
    pub database_path: String,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            database_path: default_database_path(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct HarnessConfig {
    /// Directory containing harness Lua scripts
    #[serde(default = "default_harness_directory")]
    pub directory: String,
    /// Root directory for harness `fs.*` functions (default: workspace root).
    /// Set to "/" for unrestricted filesystem access.
    #[serde(default = "default_harness_fs_root")]
    pub fs_root: String,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            directory: default_harness_directory(),
            fs_root: default_harness_fs_root(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct DaemonConfig {
    #[serde(default = "default_daemon_agents_dir")]
    pub agents_dir: String,
    #[serde(default = "default_daemon_harnesses_dir")]
    pub harnesses_dir: String,
    #[serde(default = "default_daemon_channels_dir")]
    pub channels_dir: String,
    #[serde(default = "default_daemon_endpoint")]
    pub endpoint: String,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            agents_dir: default_daemon_agents_dir(),
            harnesses_dir: default_daemon_harnesses_dir(),
            channels_dir: default_daemon_channels_dir(),
            endpoint: default_daemon_endpoint(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct RemoteConfig {
    #[serde(default = "default_remote_bind")]
    pub bind: String,
    #[serde(default = "default_remote_auth_token_env")]
    pub auth_token_env: String,
    #[serde(default = "default_remote_event_keepalive_secs")]
    pub event_keepalive_secs: u64,
    #[serde(default)]
    pub allow_non_loopback: bool,
}

impl Default for RemoteConfig {
    fn default() -> Self {
        Self {
            bind: default_remote_bind(),
            auth_token_env: default_remote_auth_token_env(),
            event_keepalive_secs: default_remote_event_keepalive_secs(),
            allow_non_loopback: false,
        }
    }
}

pub type ProvidersConfig = std::collections::HashMap<String, ProviderConfig>;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ProviderConfig {
    /// The type of provider ("anthropic", "openai", "mock")
    #[serde(rename = "type")]
    pub kind: String,
    /// Environment variable name containing the API key
    pub api_key_env: Option<String>,
    /// Optional base URL override (for proxies)
    pub base_url: Option<String>,
    /// Additional request headers sent on every provider request.
    #[serde(default)]
    pub headers: std::collections::HashMap<String, String>,
    /// Optional max retry attempts for provider HTTP calls.
    pub max_retries: Option<u32>,
    /// Optional per-request timeout in seconds.
    pub request_timeout_secs: Option<u64>,
    /// Optional total timeout budget in seconds (across retries).
    pub total_timeout_secs: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, serde::Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum GovernanceProfile {
    #[default]
    Open,
    Balanced,
    Governed,
    Custom,
}

#[derive(Debug, Clone, Deserialize, serde::Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum GovernanceAuditMode {
    #[default]
    Off,
    Observational,
    Immutable,
}

#[derive(Debug, Clone, Deserialize, serde::Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum GovernanceImportMode {
    #[default]
    Legacy,
    Scoped,
    Mixed,
}

#[derive(Debug, Clone, Deserialize, serde::Serialize, Default)]
pub struct GovernanceAuditConfig {
    #[serde(default)]
    pub mode: GovernanceAuditMode,
    #[serde(default)]
    pub include_capability_context: bool,
    #[serde(default)]
    pub persist_before_hooks: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct GovernanceImportConfig {
    #[serde(default)]
    pub mode: GovernanceImportMode,
    #[serde(default)]
    pub default_root: Option<String>,
    #[serde(default)]
    pub allow_unscoped_in_open: bool,
}

impl Default for GovernanceImportConfig {
    fn default() -> Self {
        Self {
            mode: GovernanceImportMode::Legacy,
            default_root: None,
            allow_unscoped_in_open: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize, serde::Serialize, Default)]
pub struct GovernanceRootConfig {
    pub path: String,
    #[serde(default)]
    pub writable_hint: bool,
    #[serde(default)]
    pub default_profile: Option<String>,
    #[serde(default)]
    pub max_capabilities: std::collections::HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize, serde::Serialize, Default)]
pub struct GovernanceAgentCapabilitiesConfig {
    #[serde(default)]
    pub capability_profile: Option<String>,
    #[serde(default)]
    pub max_capabilities: std::collections::HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub allowed_child_agents: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, serde::Serialize, Default)]
pub struct GovernanceGrantsConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub max_ttl_ms: Option<u64>,
    #[serde(default)]
    pub require_audit_reason: bool,
}

#[derive(Debug, Clone, Deserialize, serde::Serialize, Default)]
pub struct GovernanceConfig {
    #[serde(default)]
    pub profile: GovernanceProfile,
    #[serde(default)]
    pub enforcement_enabled: bool,
    #[serde(default)]
    pub audit: GovernanceAuditConfig,
    #[serde(default)]
    pub import: GovernanceImportConfig,
    #[serde(default)]
    pub roots: std::collections::HashMap<String, GovernanceRootConfig>,
    #[serde(default)]
    pub capability_profiles:
        std::collections::HashMap<String, std::collections::HashMap<String, serde_json::Value>>,
    #[serde(default)]
    pub agents: std::collections::HashMap<String, GovernanceAgentCapabilitiesConfig>,
    #[serde(default)]
    pub grants: GovernanceGrantsConfig,
}

// ─── Defaults ────────────────────────────────────────────────────

fn default_system_prompt() -> String {
    "You are a helpful coding assistant.".to_string()
}

fn default_agent_id() -> String {
    "default".to_string()
}

fn default_workspace_root() -> String {
    ".".to_string()
}

fn default_max_turns() -> u32 {
    50
}

fn default_heartbeat_interval() -> u32 {
    30
}

fn default_database_path() -> String {
    ".turin/state.db".to_string()
}

fn default_harness_directory() -> String {
    ".turin/harnesses".to_string()
}

fn default_harness_fs_root() -> String {
    ".".to_string()
}

fn default_embedding_model() -> String {
    "text-embedding-3-small".to_string()
}

fn default_embedding_dimensions() -> usize {
    1536
}

fn default_daemon_agents_dir() -> String {
    "agents".to_string()
}

fn default_daemon_harnesses_dir() -> String {
    "harnesses".to_string()
}

fn default_daemon_channels_dir() -> String {
    "channels".to_string()
}

fn default_daemon_endpoint() -> String {
    ".turin/daemon.sock".to_string()
}

fn default_remote_bind() -> String {
    "127.0.0.1:9324".to_string()
}

fn default_remote_auth_token_env() -> String {
    "TURIN_REMOTE_TOKEN".to_string()
}

fn default_remote_event_keepalive_secs() -> u64 {
    15
}

// ─── Loading ─────────────────────────────────────────────────────

impl TurinConfig {
    /// Load configuration from a TOML file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Could not read config file: {}", path.display()))?;
        Self::from_str(&contents)
    }

    /// Parse configuration from a TOML string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(toml_str: &str) -> Result<Self> {
        let config: TurinConfig =
            toml::from_str(toml_str).with_context(|| "Failed to parse turin.toml")?;
        config.validate()?;
        Ok(config)
    }

    /// Validate semantic invariants that serde can't enforce.
    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            !self.agent.model.trim().is_empty(),
            "agent.model must not be empty"
        );
        if !self.providers.contains_key(&self.agent.provider) {
            anyhow::bail!(
                "Provider '{}' configured in [agent] but not found in [providers]",
                self.agent.provider
            );
        }
        anyhow::ensure!(
            self.kernel.max_turns > 0,
            "kernel.max_turns must be greater than 0"
        );
        anyhow::ensure!(
            self.kernel.heartbeat_interval_secs > 0,
            "kernel.heartbeat_interval_secs must be greater than 0"
        );

        anyhow::ensure!(
            !self.harness.directory.trim().is_empty(),
            "harness.directory must not be empty"
        );
        anyhow::ensure!(
            !self.daemon.agents_dir.trim().is_empty(),
            "daemon.agents_dir must not be empty"
        );
        anyhow::ensure!(
            !self.daemon.harnesses_dir.trim().is_empty(),
            "daemon.harnesses_dir must not be empty"
        );
        anyhow::ensure!(
            !self.daemon.channels_dir.trim().is_empty(),
            "daemon.channels_dir must not be empty"
        );
        anyhow::ensure!(
            !self.daemon.endpoint.trim().is_empty(),
            "daemon.endpoint must not be empty"
        );
        anyhow::ensure!(
            !self.remote.bind.trim().is_empty(),
            "remote.bind must not be empty"
        );
        anyhow::ensure!(
            !self.remote.auth_token_env.trim().is_empty(),
            "remote.auth_token_env must not be empty"
        );
        anyhow::ensure!(
            self.remote.event_keepalive_secs > 0,
            "remote.event_keepalive_secs must be greater than 0"
        );

        for (harness_id, harness_cfg) in &self.harnesses {
            anyhow::ensure!(
                !harness_id.trim().is_empty(),
                "harnesses contains an empty harness id"
            );
            anyhow::ensure!(
                harness_id != "default",
                "harnesses.default is reserved; use [harness] for the default harness"
            );
            anyhow::ensure!(
                !harness_cfg.directory.trim().is_empty(),
                "harnesses.{}.directory must not be empty",
                harness_id
            );
        }

        for (provider_name, provider) in &self.providers {
            if let Some(timeout_secs) = provider.request_timeout_secs {
                anyhow::ensure!(
                    timeout_secs > 0,
                    "providers.{}.request_timeout_secs must be greater than 0",
                    provider_name
                );
            }

            if let Some(timeout_secs) = provider.total_timeout_secs {
                anyhow::ensure!(
                    timeout_secs > 0,
                    "providers.{}.total_timeout_secs must be greater than 0",
                    provider_name
                );
            }

            if let (Some(request_secs), Some(total_secs)) =
                (provider.request_timeout_secs, provider.total_timeout_secs)
            {
                anyhow::ensure!(
                    total_secs >= request_secs,
                    "providers.{}.total_timeout_secs must be >= request_timeout_secs",
                    provider_name
                );
            }

            for header in provider.headers.keys() {
                anyhow::ensure!(
                    !header.trim().is_empty(),
                    "providers.{}.headers contains an empty header name",
                    provider_name
                );
            }
        }

        if let Some(ttl_ms) = self.governance.grants.max_ttl_ms {
            anyhow::ensure!(
                ttl_ms > 0,
                "governance.grants.max_ttl_ms must be greater than 0"
            );
        }

        for (root_name, root) in &self.governance.roots {
            anyhow::ensure!(
                !root_name.trim().is_empty(),
                "governance.roots contains an empty root name"
            );
            anyhow::ensure!(
                !root.path.trim().is_empty(),
                "governance.roots.{}.path must not be empty",
                root_name
            );
        }

        for profile_name in self.governance.capability_profiles.keys() {
            anyhow::ensure!(
                !profile_name.trim().is_empty(),
                "governance.capability_profiles contains an empty profile name"
            );
        }

        for (agent_id, agent_cfg) in &self.governance.agents {
            if let Some(profile_name) = &agent_cfg.capability_profile {
                anyhow::ensure!(
                    self.governance
                        .capability_profiles
                        .contains_key(profile_name),
                    "governance.agents.{}.capability_profile '{}' not found in governance.capability_profiles",
                    agent_id,
                    profile_name
                );
            }
        }

        for (agent_id, agent_cfg) in
            std::iter::once((&self.agent.id, &self.agent)).chain(self.agents.iter())
        {
            if let Some(harness_id) = &agent_cfg.harness {
                anyhow::ensure!(
                    harness_id == "default" || self.harnesses.contains_key(harness_id),
                    "agent '{}': harness '{}' not found in [harnesses.*]",
                    agent_id,
                    harness_id
                );
            }
        }

        for (agent_id, _) in
            std::iter::once((&self.agent.id, &self.agent)).chain(self.agents.iter())
        {
            let _ = crate::tools::policy::resolve_effective_tools_config(self, agent_id, None)
                .with_context(|| {
                    format!("invalid [tools] configuration for agent '{}'", agent_id)
                })?;
        }

        Ok(())
    }

    pub fn harness_id_for_agent<'a>(&self, agent: &'a AgentConfig) -> &'a str {
        agent.harness.as_deref().unwrap_or("default")
    }

    pub fn harness_config_by_id(&self, harness_id: &str) -> Result<&HarnessConfig> {
        if harness_id == "default" {
            Ok(&self.harness)
        } else {
            self.harnesses
                .get(harness_id)
                .ok_or_else(|| anyhow::anyhow!("Unknown harness id: {}", harness_id))
        }
    }

    pub fn harness_binding_for_agent<'a, 'b>(
        &'a self,
        agent: &'b AgentConfig,
    ) -> Result<(&'b str, &'a HarnessConfig)> {
        let harness_id = self.harness_id_for_agent(agent);
        let harness = self.harness_config_by_id(harness_id)?;
        Ok((harness_id, harness))
    }

    /// Resolve the workspace root path relative to a base directory.
    pub fn resolve_workspace_root(&self, base: &Path) -> PathBuf {
        let root = Path::new(&self.kernel.workspace_root);
        if root.is_absolute() {
            root.to_path_buf()
        } else {
            base.join(root)
        }
    }

    pub fn resolve_daemon_agents_dir(&self, base: &Path) -> PathBuf {
        resolve_under_workspace(base, &self.kernel.workspace_root, &self.daemon.agents_dir)
    }

    pub fn resolve_daemon_harnesses_dir(&self, base: &Path) -> PathBuf {
        resolve_under_workspace(
            base,
            &self.kernel.workspace_root,
            &self.daemon.harnesses_dir,
        )
    }

    pub fn resolve_daemon_channels_dir(&self, base: &Path) -> PathBuf {
        resolve_under_workspace(base, &self.kernel.workspace_root, &self.daemon.channels_dir)
    }

    pub fn resolve_daemon_endpoint(&self, base: &Path) -> PathBuf {
        resolve_local_ipc_endpoint(base, &self.kernel.workspace_root, &self.daemon.endpoint)
    }
}

fn resolve_under_workspace(base: &Path, workspace_root: &str, value: &str) -> PathBuf {
    let workspace_root = Path::new(workspace_root);
    let workspace = if workspace_root.is_absolute() {
        workspace_root.to_path_buf()
    } else {
        base.join(workspace_root)
    };

    let path = Path::new(value);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace.join(path)
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            id: "default".to_string(),
            system_prompt: default_system_prompt(),
            model: "test-model".to_string(),
            provider: "mock".to_string(),
            thinking: None,
            mode: AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            tools: ToolsConfig::default(),
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_full_config() {
        let toml = r#"
[agent]
system_prompt = "You are a helpful coding assistant."
model = "claude-sonnet-4-20250514"
provider = "anthropic"

[agent.thinking]
enabled = false

[kernel]
workspace_root = "."
max_turns = 50
heartbeat_interval_secs = 30

[persistence]
database_path = ".turin/state.db"

[harness]
directory = ".turin/harnesses"

[providers.anthropic]
type = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"

[providers.openai]
type = "openai"
api_key_env = "OPENAI_API_KEY"
"#;

        let config = TurinConfig::from_str(toml).unwrap();
        assert_eq!(config.agent.model, "claude-sonnet-4-20250514");
        assert_eq!(config.agent.provider, "anthropic");
        assert_eq!(config.kernel.max_turns, 50);
        assert_eq!(config.persistence.database_path, ".turin/state.db");
        assert_eq!(config.harness.directory, ".turin/harnesses");
        assert_eq!(
            config
                .providers
                .get("anthropic")
                .unwrap()
                .api_key_env
                .as_ref()
                .unwrap(),
            "ANTHROPIC_API_KEY"
        );
    }

    #[test]
    fn test_parse_minimal_config() {
        let toml = r#"
[agent]
model = "gpt-4o"
provider = "openai"

[providers.openai]
type = "openai"
"#;

        let config = TurinConfig::from_str(toml).unwrap();
        assert_eq!(config.agent.model, "gpt-4o");
        assert_eq!(config.agent.provider, "openai");
        // Defaults should be applied
        assert_eq!(config.kernel.workspace_root, ".");
        assert_eq!(config.kernel.max_turns, 50);
        assert_eq!(config.persistence.database_path, ".turin/state.db");
        assert_eq!(config.harness.directory, ".turin/harnesses");
        assert_eq!(config.remote.bind, "127.0.0.1:9324");
        assert_eq!(config.remote.auth_token_env, "TURIN_REMOTE_TOKEN");
        assert!(!config.remote.allow_non_loopback);
    }

    #[test]
    fn test_parse_with_base_url_override() {
        let toml = r#"
[agent]
model = "claude-sonnet-4-20250514"
provider = "anthropic"

[providers.anthropic]
type = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
base_url = "https://my-proxy.example.com/v1"
"#;

        let config = TurinConfig::from_str(toml).unwrap();
        let provider = config.providers.get("anthropic").unwrap();
        assert_eq!(
            provider.base_url.as_ref().unwrap(),
            "https://my-proxy.example.com/v1"
        );
    }

    #[test]
    fn test_resolve_workspace_root_relative() {
        let toml = r#"
[agent]
model = "test"
provider = "anthropic"

[providers.anthropic]
type = "anthropic"

[kernel]
workspace_root = "src"
"#;

        let config = TurinConfig::from_str(toml).unwrap();
        let resolved = config.resolve_workspace_root(Path::new("/home/user/project"));
        assert_eq!(resolved, PathBuf::from("/home/user/project/src"));
    }

    #[test]
    fn test_resolve_workspace_root_absolute() {
        let toml = r#"
[agent]
model = "test"
provider = "anthropic"

[providers.anthropic]
type = "anthropic"

[kernel]
workspace_root = "/absolute/path"
"#;

        let config = TurinConfig::from_str(toml).unwrap();
        let resolved = config.resolve_workspace_root(Path::new("/home/user/project"));
        assert_eq!(resolved, PathBuf::from("/absolute/path"));
    }

    #[test]
    fn test_validate_empty_model() {
        let toml = r#"
[agent]
model = ""
provider = "anthropic"
"#;
        assert!(TurinConfig::from_str(toml).is_err());
    }

    #[test]
    fn test_validate_invalid_provider() {
        let toml = r#"
[agent]
model = "gpt-4o"
provider = "google"
"#;
        let err = TurinConfig::from_str(toml).unwrap_err();
        assert!(err.to_string().contains("google"));
    }

    #[test]
    fn test_validate_zero_max_turns() {
        let toml = r#"
[agent]
model = "gpt-4o"
provider = "openai"

[kernel]
max_turns = 0
"#;
        assert!(TurinConfig::from_str(toml).is_err());
    }

    #[test]
    fn test_parse_provider_transport_tuning() {
        let toml = r#"
[agent]
model = "claude-sonnet-4-20250514"
provider = "anthropic"

[providers.anthropic]
type = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
max_retries = 4
request_timeout_secs = 20
total_timeout_secs = 90

[providers.anthropic.headers]
anthropic-beta = "output-128k-2025-02-19"
x-request-tag = "turin-test"
"#;

        let config = TurinConfig::from_str(toml).unwrap();
        let provider = config.providers.get("anthropic").unwrap();
        assert_eq!(provider.max_retries, Some(4));
        assert_eq!(provider.request_timeout_secs, Some(20));
        assert_eq!(provider.total_timeout_secs, Some(90));
        assert_eq!(
            provider.headers.get("anthropic-beta").map(|s| s.as_str()),
            Some("output-128k-2025-02-19")
        );
        assert_eq!(
            provider.headers.get("x-request-tag").map(|s| s.as_str()),
            Some("turin-test")
        );
    }

    #[test]
    fn test_validate_timeout_budget_order() {
        let toml = r#"
[agent]
model = "gpt-4o"
provider = "openai"

[providers.openai]
type = "openai"
request_timeout_secs = 30
total_timeout_secs = 10
"#;
        let err = TurinConfig::from_str(toml).unwrap_err();
        assert!(err.to_string().contains("total_timeout_secs"));
    }

    #[test]
    fn test_validate_empty_header_name() {
        let toml = r#"
[agent]
model = "gpt-4o"
provider = "openai"

[providers.openai]
type = "openai"

[providers.openai.headers]
" " = "bad"
"#;
        let err = TurinConfig::from_str(toml).unwrap_err();
        assert!(err.to_string().contains("empty header name"));
    }

    #[test]
    fn test_validate_remote_keepalive_must_be_positive() {
        let toml = r#"
[agent]
model = "gpt-4o"
provider = "openai"

[providers.openai]
type = "openai"

[remote]
event_keepalive_secs = 0
"#;
        let err = TurinConfig::from_str(toml).unwrap_err();
        assert!(err.to_string().contains("remote.event_keepalive_secs"));
    }

    #[test]
    fn test_parse_governance_config() {
        let toml = r#"
[agent]
model = "gpt-4o"
provider = "openai"

[providers.openai]
type = "openai"

[governance]
profile = "balanced"
enforcement_enabled = false

[governance.audit]
mode = "observational"
include_capability_context = true

[governance.import]
mode = "mixed"
default_root = "core"

[governance.roots.core]
path = "harness/core"
writable_hint = false
default_profile = "core_full"

[governance.roots.core.max_capabilities]
"runtime.db.query" = true
"runtime.db.exec" = false

[governance.capability_profiles.reviewer_ro]
"runtime.db.query" = true
"runtime.policy.set" = false

[governance.agents.reviewer]
capability_profile = "reviewer_ro"
allowed_child_agents = ["worker"]

[governance.agents.reviewer.max_capabilities]
"fs.write" = false
"runtime.db.query" = true

[governance.grants]
enabled = true
max_ttl_ms = 60000
require_audit_reason = true
"#;

        let config = TurinConfig::from_str(toml).unwrap();
        assert_eq!(config.governance.profile, GovernanceProfile::Balanced);
        assert_eq!(
            config.governance.audit.mode,
            GovernanceAuditMode::Observational
        );
        assert_eq!(config.governance.import.mode, GovernanceImportMode::Mixed);
        assert_eq!(
            config.governance.roots.get("core").map(|r| r.path.as_str()),
            Some("harness/core")
        );
        assert_eq!(
            config
                .governance
                .capability_profiles
                .get("reviewer_ro")
                .and_then(|p| p.get("runtime.policy.set"))
                .and_then(|v| v.as_bool()),
            Some(false)
        );
        assert_eq!(
            config
                .governance
                .agents
                .get("reviewer")
                .and_then(|a| a.allowed_child_agents.first())
                .map(|s| s.as_str()),
            Some("worker")
        );
        assert!(config.governance.grants.enabled);
    }
}
