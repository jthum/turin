use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use turin_local_ipc::resolve_endpoint as resolve_local_ipc_endpoint;
use turin_types::layout::config_workspace_anchor;

use crate::persistence::manager::StoreSelector;
pub use turin_types::{ThinkingConfig, ToolSelectionConfig, ToolsConfig};

mod defaults;
mod inference;
mod layout;
mod persistence;
#[cfg(test)]
mod tests;
mod validation;

use defaults::*;
pub use inference::{
    HotHistoryConfig, HotHistoryProfile, InferenceCompactionConfig, InferenceCompactionMode,
    InferenceConfig, InferenceContextConfig, InferenceContextOverrideConfig,
    InferenceOverrideConfig, ResolvedInferenceCandidate, ResolvedInferenceRoute,
};
pub use layout::{LayoutConfig, ResolvedLayout};
pub use persistence::ResolvedPersistenceConfig;

/// Top-level Turin configuration, parsed from the workspace config file.
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
    pub layout: LayoutConfig,
    #[serde(default)]
    pub persistence: PersistenceConfig,
    #[serde(default)]
    pub harness: HarnessConfig,
    #[serde(default)]
    pub harnesses: std::collections::HashMap<String, HarnessConfig>,
    #[serde(default)]
    pub inference: InferenceConfig,
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
    /// Optional per-agent harness binding. Omit to use the default `[harness]`.
    #[serde(default)]
    pub harness: Option<String>,
    /// How long the runtime stays hot after a logical request completes.
    ///
    /// `Some(0)` hibernates immediately, `Some(n)` waits `n` idle seconds,
    /// and `None` keeps the runtime hot indefinitely.
    #[serde(default = "default_idle_timeout_seconds")]
    pub idle_timeout_seconds: Option<u64>,
    #[serde(default)]
    pub tools: ToolsConfig,
    #[serde(default)]
    pub inference: InferenceOverrideConfig,
    #[serde(default)]
    pub persistence: ContextPersistenceConfig,
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
    pub heartbeat_interval_seconds: u32,
    /// Initial spawn depth (for recursive agents)
    #[serde(default)]
    pub initial_spawn_depth: u32,
}

impl Default for KernelConfig {
    fn default() -> Self {
        Self {
            workspace_root: default_workspace_root(),
            max_turns: default_max_turns(),
            heartbeat_interval_seconds: default_heartbeat_interval(),
            initial_spawn_depth: 0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct PersistenceConfig {
    /// Primary owning state target for session/runtime data.
    #[serde(default = "default_state_target")]
    pub state: StoreTargetConfig,
    /// Optional default auxiliary store for scoped non-session data.
    #[serde(default)]
    pub store: Option<StoreTargetConfig>,
    /// Optional named state stores that can be referenced by alias.
    #[serde(default)]
    pub states: std::collections::HashMap<String, NamedStoreConfig>,
    /// Optional named auxiliary stores that can be referenced by alias.
    #[serde(default)]
    pub stores: std::collections::HashMap<String, NamedStoreConfig>,
    /// Optional Level 1 placement rules for scoped memory/KV when no store is provided.
    #[serde(default)]
    pub placements: Vec<ScopedStorePlacementConfig>,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            state: default_state_target(),
            store: None,
            states: std::collections::HashMap::new(),
            stores: std::collections::HashMap::new(),
            placements: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, serde::Serialize, Default, PartialEq, Eq)]
pub struct StoreTargetConfig {
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub alias: Option<String>,
}

impl StoreTargetConfig {
    pub fn from_path(path: impl Into<String>) -> Self {
        Self {
            path: Some(path.into()),
            alias: None,
        }
    }

    pub fn from_alias(alias: impl Into<String>) -> Self {
        Self {
            path: None,
            alias: Some(alias.into()),
        }
    }

    fn validate(&self, label: &str) -> Result<()> {
        let has_path = self
            .path
            .as_ref()
            .is_some_and(|value| !value.trim().is_empty());
        let has_alias = self
            .alias
            .as_ref()
            .is_some_and(|value| !value.trim().is_empty());

        anyhow::ensure!(
            has_path || has_alias,
            "{} requires either 'path' or 'alias'",
            label
        );
        anyhow::ensure!(
            !(has_path && has_alias),
            "{} cannot set both 'path' and 'alias'",
            label
        );
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, serde::Serialize, Default, PartialEq, Eq)]
pub struct ContextPersistenceConfig {
    #[serde(default)]
    pub state: Option<StoreTargetConfig>,
    #[serde(default)]
    pub store: Option<StoreTargetConfig>,
}

#[derive(Debug, Clone, Deserialize, serde::Serialize, Default)]
pub struct NamedStoreConfig {
    pub path: String,
}

#[derive(Debug, Clone, Deserialize, serde::Serialize, Default)]
pub struct ScopedStorePlacementConfig {
    pub scope_kind: String,
    #[serde(default)]
    pub scope_key: Option<String>,
    #[serde(default)]
    pub namespace: Option<String>,
    pub store: String,
}

impl PersistenceConfig {
    pub fn with_state_path(path: impl Into<String>) -> Self {
        Self {
            state: StoreTargetConfig::from_path(path),
            ..Self::default()
        }
    }

    pub fn top_level_state_selector(&self) -> Result<StoreSelector> {
        self.resolve_state_target(&self.state)
    }

    pub fn top_level_store_selector(&self) -> Result<StoreSelector> {
        match &self.store {
            Some(target) => self.resolve_store_target(target),
            None => self.top_level_state_selector(),
        }
    }

    pub fn resolve_context_state_selector(
        &self,
        context: Option<&ContextPersistenceConfig>,
    ) -> Result<StoreSelector> {
        if let Some(target) = context.and_then(|context| context.state.as_ref()) {
            return self.resolve_state_target(target);
        }
        self.top_level_state_selector()
    }

    pub fn resolve_context_store_selector(
        &self,
        context: Option<&ContextPersistenceConfig>,
    ) -> Result<StoreSelector> {
        if let Some(target) = context.and_then(|context| context.store.as_ref()) {
            return self.resolve_store_target(target);
        }
        if let Some(target) = context.and_then(|context| context.state.as_ref()) {
            return self.resolve_state_target(target);
        }
        self.top_level_store_selector()
    }

    pub fn resolve_state_target(&self, target: &StoreTargetConfig) -> Result<StoreSelector> {
        target.validate("persistence.state")?;
        if let Some(path) = &target.path {
            return Ok(StoreSelector::Path(path.clone()));
        }
        let alias = target
            .alias
            .as_deref()
            .expect("validated state target has alias");
        anyhow::ensure!(
            alias == "state" || self.states.contains_key(alias),
            "persistence.state alias '{}' not found in persistence.states",
            alias
        );
        Ok(StoreSelector::Alias(alias.to_string()))
    }

    pub fn resolve_store_target(&self, target: &StoreTargetConfig) -> Result<StoreSelector> {
        target.validate("persistence.store")?;
        if let Some(path) = &target.path {
            return Ok(StoreSelector::Path(path.clone()));
        }
        let alias = target
            .alias
            .as_deref()
            .expect("validated store target has alias");
        anyhow::ensure!(
            alias == "state" || self.states.contains_key(alias) || self.stores.contains_key(alias),
            "persistence.store alias '{}' not found in persistence.states or persistence.stores",
            alias
        );
        Ok(StoreSelector::Alias(alias.to_string()))
    }

    pub fn resolve_store_alias_for_scope(
        &self,
        scope_kind: &str,
        raw_scope_key: Option<&str>,
        namespace: &str,
    ) -> Option<&str> {
        let exact = self.placements.iter().find(|placement| {
            placement.scope_kind == scope_kind
                && placement.scope_key.as_deref() == raw_scope_key
                && placement.namespace.as_deref().unwrap_or("default") == namespace
        });
        if let Some(exact) = exact {
            return Some(exact.store.as_str());
        }

        self.placements
            .iter()
            .find(|placement| placement.scope_kind == scope_kind && placement.scope_key.is_none())
            .map(|placement| placement.store.as_str())
    }

    pub fn resolve_store_selector_for_scope(
        &self,
        scope_kind: &str,
        raw_scope_key: Option<&str>,
        namespace: &str,
    ) -> Option<StoreSelector> {
        if let Some(alias) =
            self.resolve_store_alias_for_scope(scope_kind, raw_scope_key, namespace)
        {
            return Some(StoreSelector::Alias(alias.to_string()));
        }
        self.top_level_store_selector().ok()
    }
}

impl TurinConfig {
    pub fn effective_inference_config_for_agent(
        &self,
        agent_id: &str,
        session_override: Option<&InferenceOverrideConfig>,
    ) -> Result<InferenceConfig> {
        let agent = if agent_id == self.agent.id {
            &self.agent
        } else {
            self.agents
                .get(agent_id)
                .ok_or_else(|| anyhow::anyhow!("Unknown agent profile: {}", agent_id))?
        };

        let mut effective = self.inference.clone();
        if !agent.inference.is_empty() {
            effective = effective.merged_with(&agent.inference);
        }
        if let Some(override_cfg) = session_override
            && !override_cfg.is_empty()
        {
            effective = effective.merged_with(override_cfg);
        }
        Ok(effective)
    }

    pub fn resolve_inference_route(
        &self,
        agent_id: &str,
        base_provider_name: &str,
        base_model: &str,
        base_thinking_budget: u32,
        requested_context: Option<&str>,
        session_override: Option<&InferenceOverrideConfig>,
    ) -> Result<ResolvedInferenceRoute> {
        let effective = self.effective_inference_config_for_agent(agent_id, session_override)?;
        Ok(effective.resolve_route(
            base_provider_name,
            base_model,
            base_thinking_budget,
            requested_context,
        ))
    }

    pub fn resolve_root_inference_route(
        &self,
        base_provider_name: &str,
        base_model: &str,
        base_thinking_budget: u32,
        requested_context: Option<&str>,
    ) -> ResolvedInferenceRoute {
        self.inference.resolve_route(
            base_provider_name,
            base_model,
            base_thinking_budget,
            requested_context,
        )
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
    /// Maximum Lua VM memory in MiB for a single harness engine.
    #[serde(default = "default_harness_memory_limit_mb")]
    pub memory_limit_mb: u32,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            directory: default_harness_directory(),
            fs_root: default_harness_fs_root(),
            memory_limit_mb: default_harness_memory_limit_mb(),
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
    #[serde(default = "default_daemon_runtime_db")]
    pub runtime_db: String,
    #[serde(default = "default_daemon_endpoint")]
    pub endpoint: String,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            agents_dir: default_daemon_agents_dir(),
            harnesses_dir: default_daemon_harnesses_dir(),
            channels_dir: default_daemon_channels_dir(),
            runtime_db: default_daemon_runtime_db(),
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
    #[serde(default = "default_remote_event_keepalive_seconds")]
    pub event_keepalive_seconds: u64,
    #[serde(default)]
    pub allow_non_loopback: bool,
}

impl Default for RemoteConfig {
    fn default() -> Self {
        Self {
            bind: default_remote_bind(),
            auth_token_env: default_remote_auth_token_env(),
            event_keepalive_seconds: default_remote_event_keepalive_seconds(),
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
    pub request_timeout_seconds: Option<u64>,
    /// Optional total timeout budget in seconds (across retries).
    pub total_timeout_seconds: Option<u64>,
    /// Optional provider-level context window used for token budgeting and compaction.
    pub context_window_tokens: Option<u32>,
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

// ─── Loading ─────────────────────────────────────────────────────

impl TurinConfig {
    /// Load configuration from a TOML file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Could not read config file: {}", path.display()))?;
        let mut config: TurinConfig =
            toml::from_str(&contents).with_context(|| "Failed to parse Turin config")?;
        let config_dir = turin_types::layout::config_dir(path);
        load_env_file(&config.layout.resolve(path).env_file)?;
        config.normalize_runtime_paths(&config_dir);
        config.validate()?;
        Ok(config)
    }

    /// Parse configuration from a TOML string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(toml_str: &str) -> Result<Self> {
        let config: TurinConfig =
            toml::from_str(toml_str).with_context(|| "Failed to parse Turin config")?;
        config.validate()?;
        Ok(config)
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

    pub fn resolved_layout(&self, config_base: &Path) -> ResolvedLayout {
        self.layout
            .resolve_from_config_dir(config_base.join("config.toml"), config_base.to_path_buf())
    }

    pub fn resolved_persistence(&self, config_base: &Path) -> ResolvedPersistenceConfig {
        let layout = self.resolved_layout(config_base);
        let workspace_root = self.resolve_workspace_root(config_base);
        ResolvedPersistenceConfig::from_parts(&workspace_root, &layout, &self.persistence)
    }

    /// Resolve the workspace root path relative to a base directory.
    pub fn resolve_workspace_root(&self, base: &Path) -> PathBuf {
        let root = Path::new(&self.kernel.workspace_root);
        if root.is_absolute() {
            root.to_path_buf()
        } else {
            config_workspace_anchor(base).join(root)
        }
    }

    pub fn resolve_daemon_agents_dir(&self, base: &Path) -> PathBuf {
        let layout = self.resolved_layout(base);
        resolve_runtime_path(
            base,
            &self.kernel.workspace_root,
            &self.daemon.agents_dir,
            default_daemon_agents_dir().as_str(),
            &layout.agents_dir,
        )
    }

    pub fn resolve_daemon_harnesses_dir(&self, base: &Path) -> PathBuf {
        let layout = self.resolved_layout(base);
        resolve_runtime_path(
            base,
            &self.kernel.workspace_root,
            &self.daemon.harnesses_dir,
            default_daemon_harnesses_dir().as_str(),
            &layout.harnesses_dir,
        )
    }

    pub fn resolve_daemon_channels_dir(&self, base: &Path) -> PathBuf {
        let layout = self.resolved_layout(base);
        resolve_runtime_path(
            base,
            &self.kernel.workspace_root,
            &self.daemon.channels_dir,
            default_daemon_channels_dir().as_str(),
            &layout.channels_dir,
        )
    }

    pub fn resolve_daemon_runtime_db(&self, base: &Path) -> PathBuf {
        let layout = self.resolved_layout(base);
        resolve_runtime_path(
            base,
            &self.kernel.workspace_root,
            &self.daemon.runtime_db,
            default_daemon_runtime_db().as_str(),
            &layout.data_dir.join("runtime.db"),
        )
    }

    pub fn resolve_daemon_endpoint(&self, base: &Path) -> PathBuf {
        let layout = self.resolved_layout(base);
        if self.daemon.endpoint == default_daemon_endpoint() {
            return layout.daemon_socket;
        }
        resolve_local_ipc_endpoint(
            &config_workspace_anchor(base),
            &self.kernel.workspace_root,
            &self.daemon.endpoint,
        )
    }

    pub fn normalize_runtime_paths(&mut self, config_base: &Path) {
        let layout = self.resolved_layout(config_base);
        let workspace_root = self.resolve_workspace_root(config_base);
        let resolved_persistence = self.resolved_persistence(config_base);
        self.kernel.workspace_root = workspace_root.display().to_string();
        self.layout.root = Some(layout.root.display().to_string());
        self.layout.data_dir = layout.data_dir.display().to_string();
        self.layout.states_dir = layout.states_dir.display().to_string();
        self.layout.stores_dir = layout.stores_dir.display().to_string();
        self.layout.harnesses_dir = layout.harnesses_dir.display().to_string();
        self.layout.agents_dir = layout.agents_dir.display().to_string();
        self.layout.channels_dir = layout.channels_dir.display().to_string();
        self.layout.scopes_dir = layout.scopes_dir.display().to_string();
        self.layout.env_file = layout.env_file.display().to_string();
        self.layout.daemon_socket = layout.daemon_socket.display().to_string();
        self.persistence.state = resolved_persistence.state;
        self.persistence.store = resolved_persistence.store;
        self.persistence.states = resolved_persistence.states;
        self.persistence.stores = resolved_persistence.stores;

        normalize_harness_config_paths(&workspace_root, &layout.harnesses_dir, &mut self.harness);
        for harness in self.harnesses.values_mut() {
            normalize_harness_config_paths(&workspace_root, &layout.harnesses_dir, harness);
        }

        for root in self.governance.roots.values_mut() {
            if Path::new(&root.path).is_relative() {
                root.path = workspace_root.join(&root.path).display().to_string();
            }
        }

        if self.daemon.agents_dir == default_daemon_agents_dir() {
            self.daemon.agents_dir = layout.agents_dir.display().to_string();
        } else if Path::new(&self.daemon.agents_dir).is_relative() {
            self.daemon.agents_dir = workspace_root
                .join(&self.daemon.agents_dir)
                .display()
                .to_string();
        }
        if self.daemon.harnesses_dir == default_daemon_harnesses_dir() {
            self.daemon.harnesses_dir = layout.harnesses_dir.display().to_string();
        } else if Path::new(&self.daemon.harnesses_dir).is_relative() {
            self.daemon.harnesses_dir = workspace_root
                .join(&self.daemon.harnesses_dir)
                .display()
                .to_string();
        }
        if self.daemon.channels_dir == default_daemon_channels_dir() {
            self.daemon.channels_dir = layout.channels_dir.display().to_string();
        } else if Path::new(&self.daemon.channels_dir).is_relative() {
            self.daemon.channels_dir = workspace_root
                .join(&self.daemon.channels_dir)
                .display()
                .to_string();
        }
        if self.daemon.runtime_db == default_daemon_runtime_db() {
            self.daemon.runtime_db = layout.data_dir.join("runtime.db").display().to_string();
        } else if Path::new(&self.daemon.runtime_db).is_relative() {
            self.daemon.runtime_db = workspace_root
                .join(&self.daemon.runtime_db)
                .display()
                .to_string();
        }
        if self.daemon.endpoint == default_daemon_endpoint() {
            self.daemon.endpoint = layout.daemon_socket.display().to_string();
        } else if Path::new(&self.daemon.endpoint).is_relative() {
            self.daemon.endpoint = resolve_local_ipc_endpoint(
                &config_workspace_anchor(config_base),
                &self.kernel.workspace_root,
                &self.daemon.endpoint,
            )
            .display()
            .to_string();
        }
    }
}

fn load_env_file(env_path: &Path) -> Result<()> {
    if !env_path.is_file() {
        return Ok(());
    }

    for item in dotenvy::from_path_iter(env_path)
        .with_context(|| format!("Failed to parse '{}'", env_path.display()))?
    {
        let (key, value) =
            item.with_context(|| format!("Failed to parse '{}'", env_path.display()))?;
        if std::env::var_os(&key).is_none() {
            unsafe {
                std::env::set_var(&key, value);
            }
        }
    }

    Ok(())
}

fn resolve_under_workspace(base: &Path, workspace_root: &str, value: &str) -> PathBuf {
    let workspace_root = Path::new(workspace_root);
    let workspace = if workspace_root.is_absolute() {
        workspace_root.to_path_buf()
    } else {
        config_workspace_anchor(base).join(workspace_root)
    };

    let path = Path::new(value);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        workspace.join(path)
    }
}

fn resolve_runtime_path(
    base: &Path,
    workspace_root: &str,
    value: &str,
    default_value: &str,
    layout_path: &Path,
) -> PathBuf {
    if value == default_value {
        layout_path.to_path_buf()
    } else {
        resolve_under_workspace(base, workspace_root, value)
    }
}

fn normalize_harness_config_paths(
    workspace_root: &Path,
    default_harnesses_dir: &Path,
    harness: &mut HarnessConfig,
) {
    if harness.directory == default_harness_directory() {
        harness.directory = default_harnesses_dir.display().to_string();
    } else if Path::new(&harness.directory).is_relative() {
        harness.directory = workspace_root
            .join(&harness.directory)
            .display()
            .to_string();
    }

    if Path::new(&harness.fs_root).is_relative() && harness.fs_root != "." {
        harness.fs_root = workspace_root.join(&harness.fs_root).display().to_string();
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
            harness: None,
            idle_timeout_seconds: default_idle_timeout_seconds(),
            tools: ToolsConfig::default(),
            inference: InferenceOverrideConfig::default(),
            persistence: ContextPersistenceConfig::default(),
        }
    }
}
