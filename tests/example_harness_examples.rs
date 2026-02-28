use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;
use turin::kernel::Kernel;
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

struct WorkflowFixture {
    tmp: tempfile::TempDir,
    kernel: Kernel,
}

async fn build_openclaw_fixture(
    main_response: &str,
    planner_response: &str,
    reviewer_response: &str,
) -> Result<WorkflowFixture> {
    let tmp = tempdir()?;
    let main_harness_dir = tmp.path().join("harnesses");
    let planner_harness_dir = tmp.path().join("planner_harnesses");
    let reviewer_harness_dir = tmp.path().join("reviewer_harnesses");
    fs::create_dir(&main_harness_dir)?;
    fs::create_dir(&planner_harness_dir)?;
    fs::create_dir(&reviewer_harness_dir)?;

    copy_dir_contents(
        library_workflow_path("openclaw_style_personal_assistant").join("workspace"),
        tmp.path(),
    )?;
    copy_dir_contents(
        library_workflow_path("openclaw_style_personal_assistant").join("harness"),
        &main_harness_dir,
    )?;
    copy_dir_contents(
        library_workflow_path("openclaw_style_personal_assistant")
            .join("agents")
            .join("planner"),
        &planner_harness_dir,
    )?;
    copy_dir_contents(
        library_workflow_path("openclaw_style_personal_assistant")
            .join("agents")
            .join("reviewer"),
        &reviewer_harness_dir,
    )?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider(main_response));
    providers.insert("mock_planner".to_string(), mock_provider(planner_response));
    providers.insert(
        "mock_reviewer".to_string(),
        mock_provider(reviewer_response),
    );
    let mut config = base_config(tmp.path(), &main_harness_dir, "mock_main", providers);
    config.agents.insert(
        "planner".to_string(),
        AgentConfig {
            id: "planner".to_string(),
            model: "mock-model".to_string(),
            provider: "mock_planner".to_string(),
            system_prompt: "You are a planner.".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness_dir: Some(planner_harness_dir.to_string_lossy().to_string()),
            idle_grace_secs: None,
        },
    );
    config.agents.insert(
        "reviewer".to_string(),
        AgentConfig {
            id: "reviewer".to_string(),
            model: "mock-model".to_string(),
            provider: "mock_reviewer".to_string(),
            system_prompt: "You are a reviewer.".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness_dir: Some(reviewer_harness_dir.to_string_lossy().to_string()),
            idle_grace_secs: None,
        },
    );

    let kernel = build_kernel(config).await?;
    Ok(WorkflowFixture { tmp, kernel })
}

async fn build_full_coding_harness_fixture(
    main_response: &str,
    planner_response: &str,
    reviewer_response: &str,
) -> Result<WorkflowFixture> {
    let tmp = tempdir()?;
    let main_harness_dir = tmp.path().join("harnesses");
    let planner_harness_dir = tmp.path().join("planner_harnesses");
    let reviewer_harness_dir = tmp.path().join("reviewer_harnesses");
    fs::create_dir(&main_harness_dir)?;
    fs::create_dir(&planner_harness_dir)?;
    fs::create_dir(&reviewer_harness_dir)?;

    copy_dir_contents(
        library_workflow_path("full_coding_harness").join("workspace"),
        tmp.path(),
    )?;
    copy_dir_contents(
        library_workflow_path("full_coding_harness").join("harness"),
        &main_harness_dir,
    )?;
    copy_dir_contents(
        library_workflow_path("full_coding_harness")
            .join("agents")
            .join("planner"),
        &planner_harness_dir,
    )?;
    copy_dir_contents(
        library_workflow_path("full_coding_harness")
            .join("agents")
            .join("reviewer"),
        &reviewer_harness_dir,
    )?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider(main_response));
    providers.insert("mock_planner".to_string(), mock_provider(planner_response));
    providers.insert(
        "mock_reviewer".to_string(),
        mock_provider(reviewer_response),
    );
    let mut config = base_config(tmp.path(), &main_harness_dir, "mock_main", providers);
    config.agents.insert(
        "planner".to_string(),
        AgentConfig {
            id: "planner".to_string(),
            model: "mock-model".to_string(),
            provider: "mock_planner".to_string(),
            system_prompt: "You are a planner.".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness_dir: Some(planner_harness_dir.to_string_lossy().to_string()),
            idle_grace_secs: None,
        },
    );
    config.agents.insert(
        "reviewer".to_string(),
        AgentConfig {
            id: "reviewer".to_string(),
            model: "mock-model".to_string(),
            provider: "mock_reviewer".to_string(),
            system_prompt: "You are a reviewer.".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness_dir: Some(reviewer_harness_dir.to_string_lossy().to_string()),
            idle_grace_secs: None,
        },
    );

    let kernel = build_kernel(config).await?;
    Ok(WorkflowFixture { tmp, kernel })
}

async fn build_bug_triage_fixture(
    main_response: &str,
    triager_response: &str,
    responder_response: &str,
) -> Result<WorkflowFixture> {
    let tmp = tempdir()?;
    let main_harness_dir = tmp.path().join("harnesses");
    let triager_harness_dir = tmp.path().join("triager_harnesses");
    let responder_harness_dir = tmp.path().join("responder_harnesses");
    fs::create_dir(&main_harness_dir)?;
    fs::create_dir(&triager_harness_dir)?;
    fs::create_dir(&responder_harness_dir)?;

    copy_dir_contents(
        library_workflow_path("bug_triage_desk").join("workspace"),
        tmp.path(),
    )?;
    copy_dir_contents(
        library_workflow_path("bug_triage_desk").join("harness"),
        &main_harness_dir,
    )?;
    copy_dir_contents(
        library_workflow_path("bug_triage_desk")
            .join("agents")
            .join("triager"),
        &triager_harness_dir,
    )?;
    copy_dir_contents(
        library_workflow_path("bug_triage_desk")
            .join("agents")
            .join("responder"),
        &responder_harness_dir,
    )?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider(main_response));
    providers.insert("mock_triager".to_string(), mock_provider(triager_response));
    providers.insert(
        "mock_responder".to_string(),
        mock_provider(responder_response),
    );
    let mut config = base_config(tmp.path(), &main_harness_dir, "mock_main", providers);
    config.agents.insert(
        "triager".to_string(),
        AgentConfig {
            id: "triager".to_string(),
            model: "mock-model".to_string(),
            provider: "mock_triager".to_string(),
            system_prompt: "You are a triager.".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness_dir: Some(triager_harness_dir.to_string_lossy().to_string()),
            idle_grace_secs: None,
        },
    );
    config.agents.insert(
        "responder".to_string(),
        AgentConfig {
            id: "responder".to_string(),
            model: "mock-model".to_string(),
            provider: "mock_responder".to_string(),
            system_prompt: "You are a responder.".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness_dir: Some(responder_harness_dir.to_string_lossy().to_string()),
            idle_grace_secs: None,
        },
    );

    let kernel = build_kernel(config).await?;
    Ok(WorkflowFixture { tmp, kernel })
}

#[tokio::test(flavor = "multi_thread")]
async fn test_openclaw_style_personal_assistant_routes_review_prompts() -> Result<()> {
    let mut fixture = build_openclaw_fixture("MAIN_OK", "PLAN_OK", "REVIEW_OK").await?;
    let prompt = "Review src/main.rs for regressions and missing checks".to_string();

    let mut session = fixture.kernel.create_session().await;
    fixture
        .kernel
        .run(&mut session, Some(prompt.clone()))
        .await?;
    fixture.kernel.end_session(&mut session).await?;

    let runtime_dir = fixture.tmp.path().join(".turin/runtime/personal-assistant");
    let contract = fs::read_to_string(runtime_dir.join("contract.md"))?;
    let brief = fs::read_to_string(runtime_dir.join("brief.md"))?;
    let route = fs::read_to_string(runtime_dir.join("route.txt"))?;
    let delegated_output = fs::read_to_string(runtime_dir.join("delegated-output.txt"))?;
    let reviewer_prompt = fs::read_to_string(runtime_dir.join("reviewer-last-request.txt"))?;

    assert!(contract.contains("# SOUL.md"));
    assert!(contract.contains("# PROFILE.md"));
    assert!(contract.contains("# AGENTS.md"));
    assert!(contract.contains("# INBOX.md"));
    assert!(brief.contains("Route: reviewer"));
    assert_eq!(route, "reviewer");
    assert_eq!(delegated_output, "REVIEW_OK");
    assert!(reviewer_prompt.contains("User request"));

    let store = fixture.kernel.store_manager().get_default().await?;
    let conn = store.get_connection().await?;
    let mut rows = conn
        .query(
            "SELECT route, delegated_agent, delegated_output, prompt \
             FROM personal_assistant_activity ORDER BY id DESC LIMIT 1",
            (),
        )
        .await?;
    let row = rows.next().await?.expect("expected activity row");
    let route_value: String = row.get(0)?;
    let delegated_agent: String = row.get(1)?;
    let delegated_output_value: String = row.get(2)?;
    let prompt_value: String = row.get(3)?;
    assert_eq!(route_value, "reviewer");
    assert_eq!(delegated_agent, "reviewer");
    assert_eq!(delegated_output_value, "REVIEW_OK");
    assert_eq!(prompt_value, prompt);
    drop(rows);

    let mut reviewer_rows = conn
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
    while let Some(row) = reviewer_rows.next().await? {
        let agent_id: String = row.get(0)?;
        let role: String = row.get(1)?;
        let content: String = row.get(2)?;
        if agent_id == "reviewer" && role == "assistant" && content.contains("REVIEW_OK") {
            saw_reviewer_output = true;
            break;
        }
    }
    assert!(saw_reviewer_output, "expected reviewer assistant output");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_openclaw_style_personal_assistant_routes_planning_prompts() -> Result<()> {
    let mut fixture = build_openclaw_fixture("MAIN_OK", "PLAN_OK", "REVIEW_OK").await?;
    let prompt = "Plan the next three steps for stabilizing the harness library".to_string();

    let mut session = fixture.kernel.create_session().await;
    fixture
        .kernel
        .run(&mut session, Some(prompt.clone()))
        .await?;
    fixture.kernel.end_session(&mut session).await?;

    let runtime_dir = fixture.tmp.path().join(".turin/runtime/personal-assistant");
    let route = fs::read_to_string(runtime_dir.join("route.txt"))?;
    let delegated_output = fs::read_to_string(runtime_dir.join("delegated-output.txt"))?;
    let planner_prompt = fs::read_to_string(runtime_dir.join("planner-last-request.txt"))?;

    assert_eq!(route, "planner");
    assert_eq!(delegated_output, "PLAN_OK");
    assert!(planner_prompt.contains("User request"));

    let store = fixture.kernel.store_manager().get_default().await?;
    let conn = store.get_connection().await?;
    let mut rows = conn
        .query(
            "SELECT route, delegated_agent, delegated_output \
             FROM personal_assistant_activity ORDER BY id DESC LIMIT 1",
            (),
        )
        .await?;
    let row = rows.next().await?.expect("expected activity row");
    let route_value: String = row.get(0)?;
    let delegated_agent: String = row.get(1)?;
    let delegated_output_value: String = row.get(2)?;
    assert_eq!(route_value, "planner");
    assert_eq!(delegated_agent, "planner");
    assert_eq!(delegated_output_value, "PLAN_OK");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_full_coding_harness_workflow() -> Result<()> {
    let mut fixture = build_full_coding_harness_fixture("MAIN_OK", "PLAN_OK", "REVIEW_OK").await?;
    let prompt = "Implement a practical coding workflow for Turin's harness library".to_string();

    let mut session = fixture.kernel.create_session().await;
    fixture
        .kernel
        .run(&mut session, Some(prompt.clone()))
        .await?;
    fixture.kernel.end_session(&mut session).await?;

    let runtime_dir = fixture.tmp.path().join(".turin/runtime/coding-harness");
    let context = fs::read_to_string(runtime_dir.join("context.md"))?;
    let plan = fs::read_to_string(runtime_dir.join("plan.md"))?;
    let review = fs::read_to_string(runtime_dir.join("review.md"))?;
    let brief = fs::read_to_string(runtime_dir.join("brief.md"))?;
    let planner_prompt = fs::read_to_string(runtime_dir.join("planner-last-request.txt"))?;
    let reviewer_prompt = fs::read_to_string(runtime_dir.join("reviewer-last-request.txt"))?;

    assert!(context.contains("# SPEC.md"));
    assert!(context.contains("# TASKS.md"));
    assert!(context.contains("# CONSTRAINTS.md"));
    assert!(context.contains("# NOTES.md"));
    assert_eq!(plan, "PLAN_OK");
    assert_eq!(review, "REVIEW_OK");
    assert!(brief.contains("## Plan"));
    assert!(brief.contains("## Review"));
    assert!(planner_prompt.contains("Workspace context"));
    assert!(reviewer_prompt.contains("Proposed plan"));

    let store = fixture.kernel.store_manager().get_default().await?;
    let conn = store.get_connection().await?;
    let mut rows = conn
        .query(
            "SELECT prompt, plan, review FROM coding_harness_runs ORDER BY id DESC LIMIT 1",
            (),
        )
        .await?;
    let row = rows.next().await?.expect("expected coding harness run");
    let prompt_value: String = row.get(0)?;
    let plan_value: String = row.get(1)?;
    let review_value: String = row.get(2)?;
    assert_eq!(prompt_value, prompt);
    assert_eq!(plan_value, "PLAN_OK");
    assert_eq!(review_value, "REVIEW_OK");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_bug_triage_desk_workflow() -> Result<()> {
    let mut fixture = build_bug_triage_fixture("MAIN_OK", "TRIAGE_OK", "RESPONSE_OK").await?;
    let prompt = "Bug: saving settings sometimes resets the theme after restart".to_string();

    let mut session = fixture.kernel.create_session().await;
    fixture
        .kernel
        .run(&mut session, Some(prompt.clone()))
        .await?;
    fixture.kernel.end_session(&mut session).await?;

    let runtime_dir = fixture.tmp.path().join(".turin/runtime/bug-triage");
    let context = fs::read_to_string(runtime_dir.join("context.md"))?;
    let triage = fs::read_to_string(runtime_dir.join("triage.md"))?;
    let response = fs::read_to_string(runtime_dir.join("response.md"))?;
    let brief = fs::read_to_string(runtime_dir.join("brief.md"))?;
    let triager_prompt = fs::read_to_string(runtime_dir.join("triager-last-request.txt"))?;
    let responder_prompt = fs::read_to_string(runtime_dir.join("responder-last-request.txt"))?;

    assert!(context.contains("# SEVERITY_POLICY.md"));
    assert!(context.contains("# OWNERSHIP.md"));
    assert!(context.contains("# KNOWN_ISSUES.md"));
    assert!(context.contains("# RUNBOOK.md"));
    assert_eq!(triage, "TRIAGE_OK");
    assert_eq!(response, "RESPONSE_OK");
    assert!(brief.contains("## Triage"));
    assert!(brief.contains("## Response"));
    assert!(triager_prompt.contains("Bug report"));
    assert!(responder_prompt.contains("Triage summary"));

    let store = fixture.kernel.store_manager().get_default().await?;
    let conn = store.get_connection().await?;
    let mut rows = conn
        .query(
            "SELECT prompt, triage, response FROM bug_triage_runs ORDER BY id DESC LIMIT 1",
            (),
        )
        .await?;
    let row = rows.next().await?.expect("expected bug triage run");
    let prompt_value: String = row.get(0)?;
    let triage_value: String = row.get(1)?;
    let response_value: String = row.get(2)?;
    assert_eq!(prompt_value, prompt);
    assert_eq!(triage_value, "TRIAGE_OK");
    assert_eq!(response_value, "RESPONSE_OK");
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
async fn test_code_reviewer_block() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    fs::create_dir(&harness_dir)?;
    copy_dir_contents(
        library_block_path("code_reviewer").join("workspace"),
        tmp.path(),
    )?;
    copy_dir_contents(
        library_block_path("code_reviewer").join("harness"),
        &harness_dir,
    )?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider("REVIEW_REPLY"));
    let config = base_config(tmp.path(), &harness_dir, "mock_main", providers);

    let mut kernel = build_kernel(config).await?;
    let mut session = kernel.create_session().await;
    let prompt = "Review the patch for risky assumptions in the provider layer".to_string();
    kernel.run(&mut session, Some(prompt.clone())).await?;
    kernel.end_session(&mut session).await?;

    let runtime_dir = tmp.path().join(".turin/runtime/code-review");
    let context = fs::read_to_string(runtime_dir.join("context.md"))?;
    let last_request = fs::read_to_string(runtime_dir.join("last-request.txt"))?;
    assert!(context.contains("# REVIEW_STYLE.md"));
    assert!(context.contains("# RISK_AREAS.md"));
    assert_eq!(last_request, prompt);

    let store = kernel.store_manager().get_default().await?;
    let conn = store.get_connection().await?;
    let mut rows = conn
        .query(
            "SELECT prompt FROM code_review_runs ORDER BY id DESC LIMIT 1",
            (),
        )
        .await?;
    let row = rows.next().await?.expect("expected code review row");
    let stored_prompt: String = row.get(0)?;
    assert_eq!(stored_prompt, prompt);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_task_planner_block() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    fs::create_dir(&harness_dir)?;
    copy_dir_contents(
        library_block_path("task_planner").join("workspace"),
        tmp.path(),
    )?;
    copy_dir_contents(
        library_block_path("task_planner").join("harness"),
        &harness_dir,
    )?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider("PLAN_REPLY"));
    let config = base_config(tmp.path(), &harness_dir, "mock_main", providers);

    let mut kernel = build_kernel(config).await?;
    let mut session = kernel.create_session().await;
    let prompt = "Break down the next phase of the harness library work".to_string();
    kernel.run(&mut session, Some(prompt.clone())).await?;
    kernel.end_session(&mut session).await?;

    let runtime_dir = tmp.path().join(".turin/runtime/task-planner");
    let context = fs::read_to_string(runtime_dir.join("context.md"))?;
    let last_request = fs::read_to_string(runtime_dir.join("last-request.txt"))?;
    assert!(context.contains("# PLANNING_STYLE.md"));
    assert!(context.contains("# DELIVERY_CONSTRAINTS.md"));
    assert_eq!(last_request, prompt);

    let store = kernel.store_manager().get_default().await?;
    let conn = store.get_connection().await?;
    let mut rows = conn
        .query(
            "SELECT prompt FROM task_planner_runs ORDER BY id DESC LIMIT 1",
            (),
        )
        .await?;
    let row = rows.next().await?.expect("expected task planner row");
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
