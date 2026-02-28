use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;
use turin::kernel::config::{
    AgentConfig, GovernanceConfig, GovernanceGrantsConfig, GovernanceProfile,
};

mod support;

use support::{base_config, build_kernel, copy_dir_contents, mock_provider, repo_path};

fn library_block_path(name: &str) -> PathBuf {
    repo_path(Path::new("library").join("blocks").join(name))
}

fn library_workflow_path(name: &str) -> PathBuf {
    repo_path(Path::new("library").join("workflows").join(name))
}

#[tokio::test(flavor = "multi_thread")]
async fn test_openclaw_style_personal_assistant_workflow() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    fs::create_dir(&harness_dir)?;
    copy_dir_contents(
        library_workflow_path("openclaw_style_personal_assistant").join("workspace"),
        tmp.path(),
    )?;
    copy_dir_contents(
        library_workflow_path("openclaw_style_personal_assistant").join("harness"),
        &harness_dir,
    )?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider("OPENCLAW_OK"));
    let config = base_config(tmp.path(), &harness_dir, "mock_main", providers);

    let mut kernel = build_kernel(config).await?;
    let mut session = kernel.create_session().await;
    let prompt = "Review src/main.rs and propose a safe plan".to_string();
    kernel.run(&mut session, Some(prompt.clone())).await?;
    kernel.end_session(&mut session).await?;

    let contract = fs::read_to_string(tmp.path().join(".turin/runtime/openclaw-contract.md"))?;
    let last_prompt =
        fs::read_to_string(tmp.path().join(".turin/runtime/openclaw-last-prompt.txt"))?;
    assert!(contract.contains("You are Turin"));
    assert!(contract.contains("reviewer"));
    assert_eq!(last_prompt, prompt);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_governed_peer_review_example() -> Result<()> {
    let tmp = tempdir()?;
    let main_harness_dir = tmp.path().join("main_harnesses");
    let reviewer_harness_dir = tmp.path().join("reviewer_harnesses");
    fs::create_dir(&main_harness_dir)?;
    fs::create_dir(&reviewer_harness_dir)?;
    copy_dir_contents(
        library_block_path("governed_peer_review").join("harness"),
        &main_harness_dir,
    )?;
    copy_dir_contents(
        library_block_path("governed_peer_review")
            .join("agents")
            .join("reviewer"),
        &reviewer_harness_dir,
    )?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider("MAIN_OK"));
    providers.insert("mock_review".to_string(), mock_provider("REVIEW_OK"));
    let mut config = base_config(tmp.path(), &main_harness_dir, "mock_main", providers);
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
    let prompt = "Review the patch for race conditions".to_string();
    kernel.run(&mut session, Some(prompt.clone())).await?;
    kernel.end_session(&mut session).await?;

    let review_artifact = tmp.path().join(".turin/runtime/peer-review.txt");
    let input_artifact = tmp.path().join(".turin/runtime/peer-review-input.txt");
    assert_eq!(fs::read_to_string(review_artifact)?, "REVIEW_OK");
    assert_eq!(fs::read_to_string(input_artifact)?, prompt);

    let store = kernel.store_manager().get_default().await?;
    let conn = store.get_connection().await?;
    let mut rows = conn
        .query(
            "SELECT s.agent_id, m.role, m.content \
             FROM sessions s \
             JOIN messages m ON m.session_id = s.id \
             WHERE s.agent_id = 'reviewer' \
             ORDER BY m.id",
            (),
        )
        .await?;
    let mut saw_reviewer_output = false;
    while let Some(row) = rows.next().await? {
        let agent_id: String = row.get(0)?;
        let role: String = row.get(1)?;
        let content: String = row.get(2)?;
        if agent_id == "reviewer" && role == "assistant" && content.contains("REVIEW_OK") {
            saw_reviewer_output = true;
            break;
        }
    }
    assert!(
        saw_reviewer_output,
        "expected persisted reviewer assistant output"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_durable_journal_example() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    fs::create_dir(&harness_dir)?;
    copy_dir_contents(
        library_block_path("durable_journal").join("harness"),
        &harness_dir,
    )?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider("JOURNAL_OK"));
    let config = base_config(tmp.path(), &harness_dir, "mock_main", providers);

    let mut kernel = build_kernel(config).await?;
    let mut session = kernel.create_session().await;
    let prompt = "Deployment note: restart API after schema migration".to_string();
    kernel.run(&mut session, Some(prompt.clone())).await?;
    kernel.end_session(&mut session).await?;

    let snapshot = fs::read_to_string(tmp.path().join(".turin/runtime/journal-last.txt"))?;
    assert_eq!(snapshot, prompt);

    let store = kernel.store_manager().get_default().await?;
    let conn = store.get_connection().await?;
    let mut rows = conn
        .query(
            "SELECT prompt FROM example_journal ORDER BY id DESC LIMIT 1",
            (),
        )
        .await?;
    let row = rows.next().await?.expect("expected example_journal row");
    let stored_prompt: String = row.get(0)?;
    assert_eq!(stored_prompt, prompt);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_delegated_peer_capabilities_example() -> Result<()> {
    let tmp = tempdir()?;
    let main_harness_dir = tmp.path().join("main_harnesses");
    let reviewer_harness_dir = tmp.path().join("reviewer_harnesses");
    fs::create_dir(&main_harness_dir)?;
    fs::create_dir(&reviewer_harness_dir)?;
    copy_dir_contents(
        library_block_path("delegated_peer_capabilities").join("harness"),
        &main_harness_dir,
    )?;
    copy_dir_contents(
        library_block_path("delegated_peer_capabilities")
            .join("agents")
            .join("reviewer"),
        &reviewer_harness_dir,
    )?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider("MAIN_OK"));
    providers.insert(
        "mock_review".to_string(),
        mock_provider("DELEGATED_REVIEW_OK"),
    );
    let mut config = base_config(tmp.path(), &main_harness_dir, "mock_main", providers);
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
    let prompt = "Review the request with constrained peer capabilities".to_string();
    kernel.run(&mut session, Some(prompt.clone())).await?;
    kernel.end_session(&mut session).await?;

    let review_artifact = tmp.path().join(".turin/runtime/delegated-review.txt");
    let input_artifact = tmp.path().join(".turin/runtime/delegated-review-input.txt");
    assert_eq!(fs::read_to_string(review_artifact)?, "DELEGATED_REVIEW_OK");
    assert_eq!(fs::read_to_string(input_artifact)?, prompt);

    let store = kernel.store_manager().get_default().await?;
    let conn = store.get_connection().await?;
    let mut rows = conn
        .query(
            "SELECT s.agent_id, m.role, m.content \
             FROM sessions s \
             JOIN messages m ON m.session_id = s.id \
             WHERE s.agent_id = 'reviewer' \
             ORDER BY m.id",
            (),
        )
        .await?;
    let mut saw_reviewer_output = false;
    while let Some(row) = rows.next().await? {
        let agent_id: String = row.get(0)?;
        let role: String = row.get(1)?;
        let content: String = row.get(2)?;
        if agent_id == "reviewer" && role == "assistant" && content.contains("DELEGATED_REVIEW_OK")
        {
            saw_reviewer_output = true;
            break;
        }
    }
    assert!(
        saw_reviewer_output,
        "expected persisted reviewer assistant output"
    );

    Ok(())
}
