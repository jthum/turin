use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;
use turin::kernel::Kernel;
use turin::kernel::config::{
    AgentConfig, EmbeddingConfig, GovernanceConfig, GovernanceGrantsConfig, GovernanceProfile,
    HarnessConfig, KernelConfig, PersistenceConfig, ProviderConfig, TurinConfig,
};

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("dx")
        .join(name)
}

fn copy_fixture(name: &str, dest: impl AsRef<Path>) -> Result<()> {
    fs::copy(fixture_path(name), dest)?;
    Ok(())
}

fn copy_fixture_tree(name: &str, dest_dir: impl AsRef<Path>) -> Result<()> {
    let src_dir = fixture_path(name);
    let dest_dir = dest_dir.as_ref();
    fs::create_dir_all(dest_dir)?;
    for entry in fs::read_dir(src_dir)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let dest_path = dest_dir.join(entry.file_name());
        if file_type.is_dir() {
            copy_fixture_tree(
                &format!("{}/{}", name, entry.file_name().to_string_lossy()),
                &dest_path,
            )?;
        } else {
            fs::copy(entry.path(), dest_path)?;
        }
    }
    Ok(())
}

fn mock_provider(response: &str) -> ProviderConfig {
    ProviderConfig {
        kind: "mock".to_string(),
        api_key_env: None,
        base_url: Some(response.to_string()),
        ..ProviderConfig::default()
    }
}

fn base_config(
    workspace_root: &Path,
    harness_dir: &Path,
    default_provider: &str,
    providers: HashMap<String, ProviderConfig>,
) -> TurinConfig {
    TurinConfig {
        agent: AgentConfig {
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: default_provider.to_string(),
            system_prompt: "DX fixture test".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness_dir: None,
            idle_grace_secs: None,
        },
        agents: HashMap::new(),
        kernel: KernelConfig {
            workspace_root: workspace_root.to_string_lossy().to_string(),
            max_turns: 4,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        persistence: PersistenceConfig {
            database_path: workspace_root.join("test.db").to_string_lossy().to_string(),
        },
        harness: HarnessConfig {
            directory: harness_dir.to_string_lossy().to_string(),
            fs_root: ".".to_string(),
        },
        providers,
        embeddings: Some(EmbeddingConfig::NoOp),
        governance: GovernanceConfig::default(),
    }
}

async fn build_kernel(config: TurinConfig) -> Result<Kernel> {
    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;
    Ok(kernel)
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
            id: "reviewer".to_string(),
            model: "mock-model".to_string(),
            provider: "mock_review".to_string(),
            system_prompt: "You are a reviewer.".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness_dir: Some(reviewer_harness_dir.to_string_lossy().to_string()),
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
            id: "blocked".to_string(),
            model: "mock-model".to_string(),
            provider: "mock_blocked".to_string(),
            system_prompt: "You are blocked.".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness_dir: Some(blocked_harness_dir.to_string_lossy().to_string()),
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
