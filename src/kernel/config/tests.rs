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
heartbeat_interval_seconds = 30

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
fn test_parse_inference_contexts_and_defaults() {
    let toml = r#"
[agent]
model = "gpt-4o"
provider = "openai"

[providers.openai]
type = "openai"

[providers.anthropic]
type = "anthropic"

[inference.contexts.default]
provider = "openai"
model = "gpt-4o-mini"
temperature = 0.2

[inference.contexts.reasoning]
provider = "anthropic"
model = "claude-opus-4-6"
fallback = "default"
thinking_budget = 4096
"#;

    let config = TurinConfig::from_str(toml).unwrap();
    assert!(config.inference.default.is_none());
    assert_eq!(config.inference.default_context_name(), "default");
    assert_eq!(
        config
            .inference
            .contexts
            .get("default")
            .unwrap()
            .temperature,
        Some(0.2)
    );
    assert_eq!(
        config
            .inference
            .contexts
            .get("reasoning")
            .unwrap()
            .fallback
            .as_deref(),
        Some("default")
    );
}

#[test]
fn test_parse_inference_compaction_policy() {
    let toml = r#"
[agent]
model = "gpt-4o"
provider = "openai"

[providers.openai]
type = "openai"

[inference.compaction]
mode = "summary_only"
inference = "fast"
trigger_ratio = 0.8

[inference.contexts.default]
provider = "openai"
model = "gpt-4o"

[inference.contexts.fast]
provider = "openai"
model = "gpt-4o-mini"
"#;

    let config = TurinConfig::from_str(toml).unwrap();
    assert_eq!(
        config.inference.compaction.mode,
        crate::kernel::config::InferenceCompactionMode::SummaryOnly
    );
    assert_eq!(
        config.inference.compaction.inference.as_deref(),
        Some("fast")
    );
    assert_eq!(config.inference.compaction.trigger_ratio, 0.8);
}

#[test]
fn test_validate_invalid_inference_compaction_trigger_ratio() {
    let toml = r#"
[agent]
model = "gpt-4o"
provider = "openai"

[providers.openai]
type = "openai"

[inference.compaction]
trigger_ratio = 1.5
"#;

    assert!(TurinConfig::from_str(toml).is_err());
}

#[test]
fn test_parse_inference_hot_history_policy() {
    let toml = r#"
[agent]
model = "gpt-4o"
provider = "openai"

[providers.openai]
type = "openai"

[inference.hot_history]
profile = "performance"
max_messages = 128
"#;

    let config = TurinConfig::from_str(toml).unwrap();
    assert_eq!(
        config.inference.hot_history.profile,
        crate::kernel::config::HotHistoryProfile::Performance
    );
    assert_eq!(
        config.inference.hot_history.effective_max_messages(),
        Some(128)
    );
}

#[test]
fn test_validate_invalid_hot_history_max_messages() {
    let toml = r#"
[agent]
model = "gpt-4o"
provider = "openai"

[providers.openai]
type = "openai"

[inference.hot_history]
max_messages = 0
"#;

    assert!(TurinConfig::from_str(toml).is_err());
}

#[test]
fn test_resolve_inference_route_uses_requested_context_then_fallback_then_base() {
    let toml = r#"
[agent]
model = "gpt-4o"
provider = "openai"

[providers.openai]
type = "openai"

[providers.anthropic]
type = "anthropic"

[inference.contexts.default]
provider = "openai"
model = "gpt-4o-mini"

[inference.contexts.reasoning]
provider = "anthropic"
model = "claude-opus-4-6"
fallback = "default"
temperature = 0.1
max_tokens = 4096
"#;

    let config = TurinConfig::from_str(toml).unwrap();
    let route = config.resolve_root_inference_route("openai", "gpt-4o", 1024, Some("reasoning"));
    assert_eq!(route.requested_context.as_deref(), Some("reasoning"));
    assert_eq!(route.candidates.len(), 3);
    assert_eq!(
        route.candidates[0].context_name.as_deref(),
        Some("reasoning")
    );
    assert_eq!(route.candidates[0].provider_name, "anthropic");
    assert_eq!(route.candidates[0].model, "claude-opus-4-6");
    assert_eq!(route.candidates[0].temperature, Some(0.1));
    assert_eq!(route.candidates[0].max_tokens, Some(4096));
    assert_eq!(route.candidates[0].thinking_budget, Some(1024));
    assert_eq!(route.candidates[1].context_name.as_deref(), Some("default"));
    assert_eq!(route.candidates[2].context_name, None);
    assert_eq!(route.candidates[2].provider_name, "openai");
    assert_eq!(route.candidates[2].model, "gpt-4o");
}

#[test]
fn test_resolve_inference_route_unknown_context_warns_and_falls_back() {
    let toml = r#"
[agent]
model = "gpt-4o"
provider = "openai"

[providers.openai]
type = "openai"

[inference.contexts.default]
provider = "openai"
model = "gpt-4o-mini"
"#;

    let config = TurinConfig::from_str(toml).unwrap();
    let route = config.resolve_root_inference_route("openai", "gpt-4o", 1024, Some("creative"));
    assert_eq!(route.candidates.len(), 2);
    assert_eq!(route.candidates[0].context_name.as_deref(), Some("default"));
    assert_eq!(route.candidates[1].context_name, None);
    assert!(
        route
            .warnings
            .iter()
            .any(|warning| warning.contains("creative")),
        "warnings: {:?}",
        route.warnings
    );
}

#[test]
fn test_effective_inference_config_merges_agent_and_session_overrides() {
    let toml = r#"
[agent]
model = "gpt-4o"
provider = "openai"

[providers.openai]
type = "openai"

[providers.anthropic]
type = "anthropic"

[inference.contexts.default]
provider = "openai"
model = "gpt-4o"
temperature = 0.2

[inference.contexts.fast]
provider = "openai"
model = "gpt-4o-mini"
temperature = 0.1

[agents.reviewer]
model = "claude-sonnet-4"
provider = "anthropic"
harness = "review"

[agents.reviewer.inference]
default = "default"

[agents.reviewer.inference.contexts.fast]
provider = "anthropic"
model = "claude-haiku-4"

[harnesses.review]
directory = "review"
fs_root = "."
"#;

    let config = TurinConfig::from_str(toml).unwrap();

    let session_override = InferenceOverrideConfig {
        default: None,
        contexts: std::iter::once((
            "fast".to_string(),
            InferenceContextOverrideConfig {
                temperature: Some(0.4),
                ..InferenceContextOverrideConfig::default()
            },
        ))
        .collect(),
    };

    let effective = config
        .effective_inference_config_for_agent("reviewer", Some(&session_override))
        .unwrap();
    let fast = effective.contexts.get("fast").unwrap();
    assert_eq!(fast.provider, "anthropic");
    assert_eq!(fast.model, "claude-haiku-4");
    assert_eq!(fast.temperature, Some(0.4));
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
fn test_validate_rejects_missing_explicit_default_inference_context() {
    let toml = r#"
[agent]
model = "gpt-4o"
provider = "openai"

[providers.openai]
type = "openai"

[inference]
default = "fast"
"#;

    let err = TurinConfig::from_str(toml).unwrap_err();
    assert!(
        err.to_string()
            .contains("inference.default 'fast' not found in inference.contexts")
    );
}

#[test]
fn test_validate_rejects_inference_fallback_cycles() {
    let toml = r#"
[agent]
model = "gpt-4o"
provider = "openai"

[providers.openai]
type = "openai"

[inference.contexts.a]
provider = "openai"
model = "gpt-4o-mini"
fallback = "b"

[inference.contexts.b]
provider = "openai"
model = "gpt-4o"
fallback = "a"
"#;

    let err = TurinConfig::from_str(toml).unwrap_err();
    assert!(
        err.to_string()
            .contains("inference context fallback cycle detected")
    );
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
fn test_validate_zero_provider_context_window_tokens() {
    let toml = r#"
[agent]
model = "gpt-4o"
provider = "openai"

[providers.openai]
type = "openai"
context_window_tokens = 0
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
request_timeout_seconds = 20
total_timeout_seconds = 90

[providers.anthropic.headers]
anthropic-beta = "output-128k-2025-02-19"
x-request-tag = "turin-test"
"#;

    let config = TurinConfig::from_str(toml).unwrap();
    let provider = config.providers.get("anthropic").unwrap();
    assert_eq!(provider.max_retries, Some(4));
    assert_eq!(provider.request_timeout_seconds, Some(20));
    assert_eq!(provider.total_timeout_seconds, Some(90));
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
request_timeout_seconds = 30
total_timeout_seconds = 10
"#;
    let err = TurinConfig::from_str(toml).unwrap_err();
    assert!(err.to_string().contains("total_timeout_seconds"));
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
event_keepalive_seconds = 0
"#;
    let err = TurinConfig::from_str(toml).unwrap_err();
    assert!(err.to_string().contains("remote.event_keepalive_seconds"));
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
