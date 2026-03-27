use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;
use turin::code_index_writer::build_index;
use turin::kernel::config::{
    AgentConfig, GovernanceConfig, GovernanceGrantsConfig, GovernanceProfile,
};

mod support;

use support::{
    base_config, bind_named_harness, build_kernel, copy_file, copy_tree, mock_provider, repo_path,
};

fn fixture_path(name: &str) -> PathBuf {
    repo_path(Path::new("tests").join("fixtures").join("dx").join(name))
}

fn copy_fixture(name: &str, dest: impl AsRef<Path>) -> Result<()> {
    copy_file(fixture_path(name), dest)
}

fn copy_fixture_tree(name: &str, dest_dir: impl AsRef<Path>) -> Result<()> {
    copy_tree(fixture_path(name), dest_dir)
}

fn seed_code_review_workspace(root: &Path) -> Result<()> {
    let src_dir = root.join("src");
    fs::create_dir_all(&src_dir)?;
    fs::write(
        src_dir.join("governance.rs"),
        r#"
pub fn capability_decision(capability: &str) -> bool {
    capability == "runtime.code.search.hybrid"
}
"#,
    )?;
    fs::write(
        root.join("README.md"),
        "# DX review fixture\nThis workspace exercises cache.file and code.find.\n",
    )?;
    fs::write(root.join("notes.txt"), "cached text")?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_fixture_session_memory_assistant() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    fs::create_dir(&harness_dir)?;
    copy_fixture(
        "session_memory_assistant.lua",
        harness_dir.join("session_memory_assistant.lua"),
    )?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider("MEMORY_OK"));
    let config = base_config(tmp.path(), &harness_dir, "mock_main", providers);

    let mut kernel = build_kernel(config).await?;
    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Recall memory".to_string()))
        .await?;
    assert!(session.turn_index > 0);
    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_fixture_code_cache_shortcuts() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    fs::create_dir(&harness_dir)?;
    copy_fixture(
        "code_cache_shortcuts.lua",
        harness_dir.join("code_cache_shortcuts.lua"),
    )?;
    seed_code_review_workspace(tmp.path())?;
    build_index(tmp.path(), None).await?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider("DX_OK"));
    let config = base_config(tmp.path(), &harness_dir, "mock_main", providers);

    let mut kernel = build_kernel(config).await?;
    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Use DX shortcuts".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_fixture_code_search_fallback() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    fs::create_dir(&harness_dir)?;
    copy_fixture(
        "code_search_fallback.lua",
        harness_dir.join("code_search_fallback.lua"),
    )?;
    seed_code_review_workspace(tmp.path())?;
    build_index(tmp.path(), None).await?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider("FALLBACK_OK"));
    let config = base_config(tmp.path(), &harness_dir, "mock_main", providers);

    let mut kernel = build_kernel(config).await?;
    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Use lexical fallback".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_fixture_workspace_review_assistant() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    fs::create_dir(&harness_dir)?;
    copy_fixture(
        "workspace_review_assistant.lua",
        harness_dir.join("workspace_review_assistant.lua"),
    )?;
    seed_code_review_workspace(tmp.path())?;
    build_index(tmp.path(), None).await?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider("REVIEW_OK"));
    let config = base_config(tmp.path(), &harness_dir, "mock_main", providers);

    let mut kernel = build_kernel(config).await?;
    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Review the workspace".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_fixture_config_driven_agent() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    let config_dir = tmp.path().join("config");
    fs::create_dir(&harness_dir)?;
    fs::create_dir(&config_dir)?;
    copy_fixture(
        "config_driven_agent.lua",
        harness_dir.join("config_driven_agent.lua"),
    )?;
    fs::write(
        config_dir.join("agent.json"),
        serde_json::json!({ "mode": "draft" }).to_string(),
    )?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider("CONFIG_OK"));
    let config = base_config(tmp.path(), &harness_dir, "mock_main", providers);

    let mut kernel = build_kernel(config).await?;
    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Touch config".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;

    let updated: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(config_dir.join("agent.json"))?)?;
    assert_eq!(updated.get("mode").and_then(|v| v.as_str()), Some("draft"));
    assert_eq!(updated.get("touches").and_then(|v| v.as_i64()), Some(1));
    assert!(updated.get("last_seen").and_then(|v| v.as_str()).is_some());
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_fixture_db_journal() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    fs::create_dir(&harness_dir)?;
    copy_fixture("db_journal.lua", harness_dir.join("db_journal.lua"))?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider("DB_OK"));
    let config = base_config(tmp.path(), &harness_dir, "mock_main", providers);

    let mut kernel = build_kernel(config).await?;
    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Write journal".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;

    let store = kernel.store_manager().get_default().await?;
    let conn = store.get_connection().await?;
    let mut rows = conn
        .query("SELECT note FROM dx_journal ORDER BY id DESC LIMIT 1", ())
        .await?;
    let row = rows.next().await?.expect("expected dx_journal row");
    let note: String = row.get(0)?;
    assert_eq!(note, "seed");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_fixture_governed_capability_gate() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    fs::create_dir(&harness_dir)?;
    copy_fixture(
        "governed_capability_gate.lua",
        harness_dir.join("governed_capability_gate.lua"),
    )?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider("GOVERNED_OK"));
    let mut config = base_config(tmp.path(), &harness_dir, "mock_main", providers);
    config.governance = GovernanceConfig {
        profile: GovernanceProfile::Governed,
        enforcement_enabled: true,
        ..GovernanceConfig::default()
    };

    let mut kernel = build_kernel(config).await?;
    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Check capabilities".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_fixture_peer_review_orchestrator() -> Result<()> {
    let tmp = tempdir()?;
    let orchestrator_harness_dir = tmp.path().join("harnesses");
    let reviewer_harness_dir = tmp.path().join("reviewer_harnesses");
    fs::create_dir(&orchestrator_harness_dir)?;
    fs::create_dir(&reviewer_harness_dir)?;
    copy_fixture(
        "peer_review_orchestrator.lua",
        orchestrator_harness_dir.join("peer_review_orchestrator.lua"),
    )?;
    copy_fixture(
        "peer_review_worker.lua",
        reviewer_harness_dir.join("peer_review_worker.lua"),
    )?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider("MAIN_OK"));
    providers.insert("mock_review".to_string(), mock_provider("REVIEW_OK"));
    let mut config = base_config(
        tmp.path(),
        &orchestrator_harness_dir,
        "mock_main",
        providers,
    );
    bind_named_harness(&mut config, "reviewer", &reviewer_harness_dir);
    config.governance = GovernanceConfig {
        profile: GovernanceProfile::Balanced,
        enforcement_enabled: true,
        grants: GovernanceGrantsConfig {
            enabled: true,
            max_ttl_ms: Some(60_000),
            require_audit_reason: false,
        },
        ..GovernanceConfig::default()
    };
    config.agents.insert(
        "reviewer".to_string(),
        AgentConfig {
            tool_selection: Default::default(),
            id: "reviewer".to_string(),
            model: "mock-model".to_string(),
            provider: "mock_review".to_string(),
            system_prompt: "You are a reviewer.".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness: Some("reviewer".to_string()),
            idle_grace_secs: None,
        },
    );

    let mut kernel = build_kernel(config).await?;
    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Review the patch".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_fixture_import_scoped_capability_delegate() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    fs::create_dir(&harness_dir)?;
    copy_fixture_tree("import_scoped_capability_delegate", &harness_dir)?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider("IMPORT_OK"));
    let mut config = base_config(tmp.path(), &harness_dir, "mock_main", providers);
    config.governance = GovernanceConfig {
        profile: GovernanceProfile::Balanced,
        enforcement_enabled: true,
        import: turin::kernel::config::GovernanceImportConfig {
            mode: turin::kernel::config::GovernanceImportMode::Mixed,
            default_root: Some("core".to_string()),
            allow_unscoped_in_open: false,
        },
        roots: HashMap::from([(
            "core".to_string(),
            turin::kernel::config::GovernanceRootConfig {
                path: harness_dir.to_string_lossy().to_string(),
                writable_hint: false,
                default_profile: Some("core_full".to_string()),
                max_capabilities: HashMap::new(),
            },
        )]),
        ..GovernanceConfig::default()
    };

    let mut kernel = build_kernel(config).await?;
    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("Exercise scoped import delegation".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_fixture_import_scoped_complete_delegate() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    let reviewer_harness_dir = tmp.path().join("reviewer_harnesses");
    fs::create_dir(&harness_dir)?;
    fs::create_dir(&reviewer_harness_dir)?;
    copy_fixture_tree("import_scoped_complete_delegate", &harness_dir)?;
    copy_fixture(
        "peer_review_worker.lua",
        reviewer_harness_dir.join("peer_review_worker.lua"),
    )?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider("IMPORT_COMPLETE_OK"));
    providers.insert("mock_review".to_string(), mock_provider("REVIEW_OK"));
    let mut config = base_config(tmp.path(), &harness_dir, "mock_main", providers);
    bind_named_harness(&mut config, "reviewer", &reviewer_harness_dir);
    config.governance = GovernanceConfig {
        profile: GovernanceProfile::Balanced,
        enforcement_enabled: true,
        import: turin::kernel::config::GovernanceImportConfig {
            mode: turin::kernel::config::GovernanceImportMode::Mixed,
            default_root: Some("core".to_string()),
            allow_unscoped_in_open: false,
        },
        roots: HashMap::from([(
            "core".to_string(),
            turin::kernel::config::GovernanceRootConfig {
                path: harness_dir.to_string_lossy().to_string(),
                writable_hint: false,
                default_profile: Some("core_full".to_string()),
                max_capabilities: HashMap::new(),
            },
        )]),
        agents: HashMap::from([(
            "default".to_string(),
            turin::kernel::config::GovernanceAgentCapabilitiesConfig {
                capability_profile: None,
                max_capabilities: HashMap::new(),
                allowed_child_agents: vec!["reviewer".to_string()],
            },
        )]),
        ..GovernanceConfig::default()
    };
    config.agents.insert(
        "reviewer".to_string(),
        AgentConfig {
            tool_selection: Default::default(),
            id: "reviewer".to_string(),
            model: "mock-model".to_string(),
            provider: "mock_review".to_string(),
            system_prompt: "You are a reviewer.".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness: Some("reviewer".to_string()),
            idle_grace_secs: None,
        },
    );

    let mut kernel = build_kernel(config).await?;
    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("Exercise scoped import delegated complete".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;

    let store = kernel.store_manager().get_default().await?;
    let conn = store.get_connection().await?;
    let mut rows = conn
        .query(
            "SELECT review FROM delegated_complete_probe ORDER BY id DESC LIMIT 1",
            (),
        )
        .await?;
    let row = rows
        .next()
        .await?
        .expect("expected delegated_complete_probe row");
    let review: String = row.get(0)?;
    assert_eq!(review, "REVIEW_OK");

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_fixture_nested_import_widen_denial() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    fs::create_dir(&harness_dir)?;
    copy_fixture_tree("nested_import_widen", &harness_dir)?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider("NESTED_OK"));
    let mut config = base_config(tmp.path(), &harness_dir, "mock_main", providers);
    config.governance = GovernanceConfig {
        profile: GovernanceProfile::Balanced,
        enforcement_enabled: true,
        import: turin::kernel::config::GovernanceImportConfig {
            mode: turin::kernel::config::GovernanceImportMode::Mixed,
            default_root: Some("core".to_string()),
            allow_unscoped_in_open: false,
        },
        roots: HashMap::from([(
            "core".to_string(),
            turin::kernel::config::GovernanceRootConfig {
                path: harness_dir.to_string_lossy().to_string(),
                writable_hint: false,
                default_profile: Some("core_full".to_string()),
                max_capabilities: HashMap::new(),
            },
        )]),
        ..GovernanceConfig::default()
    };

    let mut kernel = build_kernel(config).await?;
    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("Exercise nested import widening denial".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_fixture_peer_agent_denial() -> Result<()> {
    let tmp = tempdir()?;
    let orchestrator_harness_dir = tmp.path().join("harnesses");
    let blocked_harness_dir = tmp.path().join("blocked_harnesses");
    fs::create_dir(&orchestrator_harness_dir)?;
    fs::create_dir(&blocked_harness_dir)?;
    copy_fixture(
        "peer_agent_denial.lua",
        orchestrator_harness_dir.join("peer_agent_denial.lua"),
    )?;
    copy_fixture(
        "peer_review_worker.lua",
        blocked_harness_dir.join("peer_review_worker.lua"),
    )?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider("DENIAL_OK"));
    providers.insert("mock_blocked".to_string(), mock_provider("BLOCKED_OK"));
    let mut config = base_config(
        tmp.path(),
        &orchestrator_harness_dir,
        "mock_main",
        providers,
    );
    bind_named_harness(&mut config, "blocked", &blocked_harness_dir);
    config.governance = GovernanceConfig {
        profile: GovernanceProfile::Balanced,
        enforcement_enabled: true,
        agents: HashMap::from([(
            "default".to_string(),
            turin::kernel::config::GovernanceAgentCapabilitiesConfig {
                capability_profile: None,
                max_capabilities: HashMap::new(),
                allowed_child_agents: vec!["reviewer".to_string()],
            },
        )]),
        ..GovernanceConfig::default()
    };
    config.agents.insert(
        "blocked".to_string(),
        AgentConfig {
            tool_selection: Default::default(),
            id: "blocked".to_string(),
            model: "mock-model".to_string(),
            provider: "mock_blocked".to_string(),
            system_prompt: "You are blocked.".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness: Some("blocked".to_string()),
            idle_grace_secs: None,
        },
    );

    let mut kernel = build_kernel(config).await?;
    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Exercise peer agent denial".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_dx_fixture_peer_complete_delegated_caps() -> Result<()> {
    let tmp = tempdir()?;
    let orchestrator_harness_dir = tmp.path().join("harnesses");
    let reviewer_harness_dir = tmp.path().join("reviewer_harnesses");
    fs::create_dir(&orchestrator_harness_dir)?;
    fs::create_dir(&reviewer_harness_dir)?;
    copy_fixture(
        "peer_complete_delegated_caps.lua",
        orchestrator_harness_dir.join("peer_complete_delegated_caps.lua"),
    )?;
    copy_fixture(
        "peer_complete_delegated_caps_worker.lua",
        reviewer_harness_dir.join("peer_complete_delegated_caps_worker.lua"),
    )?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider("MAIN_OK"));
    providers.insert("mock_review".to_string(), mock_provider("REVIEW_QUERY_OK"));
    let mut config = base_config(
        tmp.path(),
        &orchestrator_harness_dir,
        "mock_main",
        providers,
    );
    bind_named_harness(&mut config, "reviewer", &reviewer_harness_dir);
    config.governance = GovernanceConfig {
        profile: GovernanceProfile::Balanced,
        enforcement_enabled: true,
        agents: HashMap::from([(
            "default".to_string(),
            turin::kernel::config::GovernanceAgentCapabilitiesConfig {
                capability_profile: None,
                max_capabilities: HashMap::new(),
                allowed_child_agents: vec!["reviewer".to_string()],
            },
        )]),
        ..GovernanceConfig::default()
    };
    config.agents.insert(
        "reviewer".to_string(),
        AgentConfig {
            tool_selection: Default::default(),
            id: "reviewer".to_string(),
            model: "mock-model".to_string(),
            provider: "mock_review".to_string(),
            system_prompt: "You are a reviewer.".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness: Some("reviewer".to_string()),
            idle_grace_secs: None,
        },
    );

    let mut kernel = build_kernel(config).await?;
    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("Exercise delegated caps via runtime.agent(...):complete".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;
    Ok(())
}
