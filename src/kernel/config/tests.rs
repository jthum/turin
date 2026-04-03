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

[persistence.state]
path = ".turin/state.db"

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
    assert_eq!(
        config.persistence.state,
        StoreTargetConfig::from_path(".turin/state.db")
    );
    assert_eq!(config.harness.directory, ".turin/harnesses");
    assert_eq!(config.harness.memory_limit_mb, 32);
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
    assert_eq!(config.kernel.workspace_root, ".");
    assert_eq!(config.kernel.max_turns, 50);
    assert_eq!(
        config.persistence.state,
        StoreTargetConfig::from_path("data/state.db")
    );
    assert_eq!(config.harness.directory, "harnesses");
    assert_eq!(config.harness.memory_limit_mb, 32);
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
fn test_parse_persistence_stores_and_placements() {
    let toml = r#"
[agent]
model = "gpt-4o"
provider = "openai"

[providers.openai]
type = "openai"

[persistence.state]
path = ".turin/state.db"

[persistence.stores.rust_kb]
path = ".turin/kb/rust.db"

[[persistence.placements]]
scope_kind = "project"
store = "rust_kb"

[[persistence.placements]]
scope_kind = "project"
scope_key = "alpha"
namespace = "notes"
store = "rust_kb"
"#;

    let config = TurinConfig::from_str(toml).unwrap();
    assert_eq!(
        config.persistence.stores.get("rust_kb").unwrap().path,
        ".turin/kb/rust.db"
    );
    assert_eq!(
        config
            .persistence
            .resolve_store_alias_for_scope("project", Some("alpha"), "notes"),
        Some("rust_kb")
    );
    assert_eq!(
        config
            .persistence
            .resolve_store_alias_for_scope("project", Some("beta"), "default"),
        Some("rust_kb")
    );
}

#[test]
fn test_resolved_persistence_separates_layout_defaults_from_effective_targets() {
    let toml = r#"
[agent]
model = "gpt-4o"
provider = "openai"

[providers.openai]
type = "openai"

[persistence.stores.rust_kb]
path = "kb/rust.db"
"#;

    let config = TurinConfig::from_str(toml).unwrap();
    let workspace_root = config.resolve_workspace_root(Path::new("/tmp/workspace/.turin"));
    let resolved = config.resolved_persistence(Path::new("/tmp/workspace/.turin"));

    assert_eq!(
        resolved.state,
        StoreTargetConfig::from_path("/tmp/workspace/.turin/data/state.db")
    );
    assert!(resolved.store.is_none());
    assert_eq!(
        resolved.stores.get("rust_kb").unwrap().path,
        workspace_root.join("kb/rust.db").display().to_string()
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
fn test_validate_zero_harness_memory_limit() {
    let toml = r#"
[agent]
model = "gpt-4o"
provider = "openai"

[providers.openai]
type = "openai"

[harness]
memory_limit_mb = 0
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

#[tokio::test]
async fn from_file_loads_adjacent_dotenv_without_overriding_existing_env() {
    let _env_guard = crate::test_support::env_lock().lock().await;
    let temp = tempfile::tempdir().expect("tempdir");
    let config_path = temp.path().join("turin.toml");
    let env_path = temp.path().join(".env");
    let key = "TURIN_TEST_CONFIG_DOTENV_KEY";

    std::fs::write(
        &config_path,
        format!(
            r#"
[agent]
model = "gpt-4o"
provider = "openai"

[providers.openai]
type = "openai"
api_key_env = "{key}"
"#
        ),
    )
    .expect("write config");
    std::fs::write(&env_path, format!("{key}=from-dotenv\n")).expect("write env");

    unsafe {
        std::env::remove_var(key);
    }
    let _ = TurinConfig::from_file(&config_path).expect("config loads");
    assert_eq!(std::env::var(key).as_deref(), Ok("from-dotenv"));

    unsafe {
        std::env::set_var(key, "from-env");
    }
    let _ = TurinConfig::from_file(&config_path).expect("config loads twice");
    assert_eq!(std::env::var(key).as_deref(), Ok("from-env"));

    unsafe {
        std::env::remove_var(key);
    }
}
