use anyhow::Result;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tempfile::tempdir;
use tokio::time::sleep;
use turin_core::kernel::Kernel;
use turin_core::kernel::config::{AgentConfig, GovernanceConfig, GovernanceGrantsConfig};
use turin_core::persistence::state::SessionReadTarget;

mod support;

use support::{
    base_config, bind_named_harness, build_kernel, copy_dir_contents, mock_provider, repo_path,
};

fn library_block_path(name: &str) -> PathBuf {
    repo_path(Path::new("library").join("blocks").join(name))
}

fn library_workflow_path(name: &str) -> PathBuf {
    repo_path(Path::new("library").join("workflows").join(name))
}

struct HarnessOwnedStore {
    database: turso::Database,
}

impl HarnessOwnedStore {
    async fn get_connection(&self) -> Result<turso::Connection> {
        Ok(self.database.connect()?)
    }
}

async fn open_harness_owned_store(workspace: &Path) -> Result<HarnessOwnedStore> {
    let database = turso::Builder::new_local(
        &workspace
            .join(".turin/runtime/harness.db")
            .to_string_lossy(),
    )
    .build()
    .await?;
    Ok(HarnessOwnedStore { database })
}

async fn wait_for_persisted_agent_output(
    kernel: &Kernel,
    agent_id: &str,
    expected_fragment: &str,
) -> Result<bool> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let store = kernel.store_manager().get_default().await?;
        let conn = store.get_connection().await?;
        let mut rows = conn
            .query(
                "SELECT s.agent_id, tm.role, tm.content \
                 FROM sessions s \
                 JOIN turns t ON t.session_id = s.id \
                 JOIN messages tm ON tm.turn_id = t.id \
                 WHERE s.agent_id = ?1 \
                 ORDER BY tm.id",
                [agent_id],
            )
            .await?;
        while let Some(row) = rows.next().await? {
            let row_agent_id: String = row.get(0)?;
            let role: String = row.get(1)?;
            let content: String = row.get(2)?;
            if row_agent_id == agent_id
                && role == "assistant"
                && content.contains(expected_fragment)
            {
                return Ok(true);
            }
        }
        if Instant::now() >= deadline {
            return Ok(false);
        }
        sleep(Duration::from_millis(50)).await;
    }
}

async fn wait_for_persisted_session_events(
    kernel: &Kernel,
    session_internal_id: i64,
    expected_event_types: &[&str],
) -> Result<bool> {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let store = kernel.store_manager().get_default().await?;
        let events = store
            .get_events(session_internal_id, &SessionReadTarget::ActiveBranch)
            .await?;
        if expected_event_types
            .iter()
            .all(|expected| events.iter().any(|event| event.event_type == *expected))
        {
            return Ok(true);
        }
        if Instant::now() >= deadline {
            return Ok(false);
        }
        sleep(Duration::from_millis(50)).await;
    }
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
    bind_named_harness(&mut config, "planner", &planner_harness_dir);
    bind_named_harness(&mut config, "reviewer", &reviewer_harness_dir);
    config.agents.insert(
        "planner".to_string(),
        AgentConfig {
            tools: Default::default(),
            id: "planner".to_string(),
            model: "mock-model".to_string(),
            provider: "mock_planner".to_string(),
            system_prompt: "You are a planner.".to_string(),
            thinking: None,
            harness: Some("planner".to_string()),
            idle_timeout_seconds: None,
            linked_runtime_lanes: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
    );
    config.agents.insert(
        "reviewer".to_string(),
        AgentConfig {
            tools: Default::default(),
            id: "reviewer".to_string(),
            model: "mock-model".to_string(),
            provider: "mock_reviewer".to_string(),
            system_prompt: "You are a reviewer.".to_string(),
            thinking: None,
            harness: Some("reviewer".to_string()),
            idle_timeout_seconds: None,
            linked_runtime_lanes: None,
            inference: Default::default(),
            persistence: Default::default(),
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
    bind_named_harness(&mut config, "planner", &planner_harness_dir);
    bind_named_harness(&mut config, "reviewer", &reviewer_harness_dir);
    config.agents.insert(
        "planner".to_string(),
        AgentConfig {
            tools: Default::default(),
            id: "planner".to_string(),
            model: "mock-model".to_string(),
            provider: "mock_planner".to_string(),
            system_prompt: "You are a planner.".to_string(),
            thinking: None,
            harness: Some("planner".to_string()),
            idle_timeout_seconds: None,
            linked_runtime_lanes: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
    );
    config.agents.insert(
        "reviewer".to_string(),
        AgentConfig {
            tools: Default::default(),
            id: "reviewer".to_string(),
            model: "mock-model".to_string(),
            provider: "mock_reviewer".to_string(),
            system_prompt: "You are a reviewer.".to_string(),
            thinking: None,
            harness: Some("reviewer".to_string()),
            idle_timeout_seconds: None,
            linked_runtime_lanes: None,
            inference: Default::default(),
            persistence: Default::default(),
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
    bind_named_harness(&mut config, "triager", &triager_harness_dir);
    bind_named_harness(&mut config, "responder", &responder_harness_dir);
    config.agents.insert(
        "triager".to_string(),
        AgentConfig {
            tools: Default::default(),
            id: "triager".to_string(),
            model: "mock-model".to_string(),
            provider: "mock_triager".to_string(),
            system_prompt: "You are a triager.".to_string(),
            thinking: None,
            harness: Some("triager".to_string()),
            idle_timeout_seconds: None,
            linked_runtime_lanes: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
    );
    config.agents.insert(
        "responder".to_string(),
        AgentConfig {
            tools: Default::default(),
            id: "responder".to_string(),
            model: "mock-model".to_string(),
            provider: "mock_responder".to_string(),
            system_prompt: "You are a responder.".to_string(),
            thinking: None,
            harness: Some("responder".to_string()),
            idle_timeout_seconds: None,
            linked_runtime_lanes: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
    );

    let kernel = build_kernel(config).await?;
    Ok(WorkflowFixture { tmp, kernel })
}

async fn build_release_manager_fixture(
    main_response: &str,
    reviewer_response: &str,
    changelog_response: &str,
) -> Result<WorkflowFixture> {
    let tmp = tempdir()?;
    let main_harness_dir = tmp.path().join("harnesses");
    let reviewer_harness_dir = tmp.path().join("readiness_harnesses");
    let changelog_harness_dir = tmp.path().join("changelog_harnesses");
    fs::create_dir(&main_harness_dir)?;
    fs::create_dir(&reviewer_harness_dir)?;
    fs::create_dir(&changelog_harness_dir)?;

    copy_dir_contents(
        library_workflow_path("release_manager").join("workspace"),
        tmp.path(),
    )?;
    copy_dir_contents(
        library_workflow_path("release_manager").join("harness"),
        &main_harness_dir,
    )?;
    copy_dir_contents(
        library_workflow_path("release_manager")
            .join("agents")
            .join("readiness_reviewer"),
        &reviewer_harness_dir,
    )?;
    copy_dir_contents(
        library_workflow_path("release_manager")
            .join("agents")
            .join("changelog_writer"),
        &changelog_harness_dir,
    )?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider(main_response));
    providers.insert(
        "mock_reviewer".to_string(),
        mock_provider(reviewer_response),
    );
    providers.insert(
        "mock_changelog".to_string(),
        mock_provider(changelog_response),
    );
    let mut config = base_config(tmp.path(), &main_harness_dir, "mock_main", providers);
    bind_named_harness(&mut config, "readiness_reviewer", &reviewer_harness_dir);
    bind_named_harness(&mut config, "changelog_writer", &changelog_harness_dir);
    config.agents.insert(
        "readiness_reviewer".to_string(),
        AgentConfig {
            tools: Default::default(),
            id: "readiness_reviewer".to_string(),
            model: "mock-model".to_string(),
            provider: "mock_reviewer".to_string(),
            system_prompt: "You are a readiness reviewer.".to_string(),
            thinking: None,
            harness: Some("readiness_reviewer".to_string()),
            idle_timeout_seconds: None,
            linked_runtime_lanes: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
    );
    config.agents.insert(
        "changelog_writer".to_string(),
        AgentConfig {
            tools: Default::default(),
            id: "changelog_writer".to_string(),
            model: "mock-model".to_string(),
            provider: "mock_changelog".to_string(),
            system_prompt: "You are a changelog writer.".to_string(),
            thinking: None,
            harness: Some("changelog_writer".to_string()),
            idle_timeout_seconds: None,
            linked_runtime_lanes: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
    );

    let kernel = build_kernel(config).await?;
    Ok(WorkflowFixture { tmp, kernel })
}

async fn build_docs_team_fixture(
    main_response: &str,
    reviewer_response: &str,
    draft_response: &str,
) -> Result<WorkflowFixture> {
    let tmp = tempdir()?;
    let main_harness_dir = tmp.path().join("harnesses");
    let reviewer_harness_dir = tmp.path().join("docs_reviewer_harnesses");
    let draft_harness_dir = tmp.path().join("draft_writer_harnesses");
    fs::create_dir(&main_harness_dir)?;
    fs::create_dir(&reviewer_harness_dir)?;
    fs::create_dir(&draft_harness_dir)?;

    copy_dir_contents(
        library_workflow_path("docs_team_assistant").join("workspace"),
        tmp.path(),
    )?;
    copy_dir_contents(
        library_workflow_path("docs_team_assistant").join("harness"),
        &main_harness_dir,
    )?;
    copy_dir_contents(
        library_workflow_path("docs_team_assistant")
            .join("agents")
            .join("docs_reviewer"),
        &reviewer_harness_dir,
    )?;
    copy_dir_contents(
        library_workflow_path("docs_team_assistant")
            .join("agents")
            .join("draft_writer"),
        &draft_harness_dir,
    )?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider(main_response));
    providers.insert(
        "mock_reviewer".to_string(),
        mock_provider(reviewer_response),
    );
    providers.insert("mock_draft".to_string(), mock_provider(draft_response));
    let mut config = base_config(tmp.path(), &main_harness_dir, "mock_main", providers);
    bind_named_harness(&mut config, "docs_reviewer", &reviewer_harness_dir);
    bind_named_harness(&mut config, "draft_writer", &draft_harness_dir);
    config.agents.insert(
        "docs_reviewer".to_string(),
        AgentConfig {
            tools: Default::default(),
            id: "docs_reviewer".to_string(),
            model: "mock-model".to_string(),
            provider: "mock_reviewer".to_string(),
            system_prompt: "You are a docs reviewer.".to_string(),
            thinking: None,
            harness: Some("docs_reviewer".to_string()),
            idle_timeout_seconds: None,
            linked_runtime_lanes: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
    );
    config.agents.insert(
        "draft_writer".to_string(),
        AgentConfig {
            tools: Default::default(),
            id: "draft_writer".to_string(),
            model: "mock-model".to_string(),
            provider: "mock_draft".to_string(),
            system_prompt: "You are a draft writer.".to_string(),
            thinking: None,
            harness: Some("draft_writer".to_string()),
            idle_timeout_seconds: None,
            linked_runtime_lanes: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
    );

    let kernel = build_kernel(config).await?;
    Ok(WorkflowFixture { tmp, kernel })
}

#[tokio::test(flavor = "multi_thread")]
async fn test_openclaw_style_personal_assistant_routes_review_prompts() -> Result<()> {
    let mut fixture = build_openclaw_fixture("MAIN_OK", "PLAN_OK", "REVIEW_OK").await?;
    let prompt = "Review src/main.rs for regressions and missing checks".to_string();

    let mut session = fixture.kernel.create_session().await.unwrap();
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

    let harness_store = open_harness_owned_store(fixture.tmp.path()).await?;
    let conn = harness_store.get_connection().await?;
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

    let store = fixture.kernel.store_manager().get_default().await?;
    let conn = store.get_connection().await?;
    let mut reviewer_rows = conn
        .query(
            "SELECT s.agent_id, tm.role, tm.content \
             FROM sessions s \
             JOIN turns t ON t.session_id = s.id \
             JOIN messages tm ON tm.turn_id = t.id \
             WHERE s.agent_id = 'reviewer' \
             ORDER BY tm.id",
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

    let mut session = fixture.kernel.create_session().await.unwrap();
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

    let store = open_harness_owned_store(fixture.tmp.path()).await?;
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

    let mut session = fixture.kernel.create_session().await.unwrap();
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

    let store = open_harness_owned_store(fixture.tmp.path()).await?;
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

    let mut session = fixture.kernel.create_session().await.unwrap();
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

    let store = open_harness_owned_store(fixture.tmp.path()).await?;
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
async fn test_release_manager_workflow() -> Result<()> {
    let mut fixture = build_release_manager_fixture("MAIN_OK", "READY_OK", "CHANGELOG_OK").await?;
    let prompt = "Prepare the next Turin pre-release checkpoint".to_string();

    let mut session = fixture.kernel.create_session().await.unwrap();
    fixture
        .kernel
        .run(&mut session, Some(prompt.clone()))
        .await?;
    fixture.kernel.end_session(&mut session).await?;

    let runtime_dir = fixture.tmp.path().join(".turin/runtime/release-manager");
    let context = fs::read_to_string(runtime_dir.join("context.md"))?;
    let readiness = fs::read_to_string(runtime_dir.join("readiness.md"))?;
    let changelog = fs::read_to_string(runtime_dir.join("changelog.md"))?;
    let brief = fs::read_to_string(runtime_dir.join("brief.md"))?;
    let reviewer_prompt =
        fs::read_to_string(runtime_dir.join("readiness-reviewer-last-request.txt"))?;
    let changelog_prompt =
        fs::read_to_string(runtime_dir.join("changelog-writer-last-request.txt"))?;

    assert!(context.contains("# RELEASE_GOALS.md"));
    assert!(context.contains("# CHANGELOG_NOTES.md"));
    assert!(context.contains("# OPEN_ISSUES.md"));
    assert!(context.contains("# CHECKLIST.md"));
    assert!(context.contains("# CONSTRAINTS.md"));
    assert_eq!(readiness, "READY_OK");
    assert_eq!(changelog, "CHANGELOG_OK");
    assert!(brief.contains("## Readiness Review"));
    assert!(brief.contains("## Draft Release Notes"));
    assert!(reviewer_prompt.contains("Release request"));
    assert!(changelog_prompt.contains("Readiness review"));

    let store = open_harness_owned_store(fixture.tmp.path()).await?;
    let conn = store.get_connection().await?;
    let mut rows = conn
        .query(
            "SELECT prompt, readiness, changelog FROM release_manager_runs ORDER BY id DESC LIMIT 1",
            (),
        )
        .await?;
    let row = rows.next().await?.expect("expected release manager row");
    let prompt_value: String = row.get(0)?;
    let readiness_value: String = row.get(1)?;
    let changelog_value: String = row.get(2)?;
    assert_eq!(prompt_value, prompt);
    assert_eq!(readiness_value, "READY_OK");
    assert_eq!(changelog_value, "CHANGELOG_OK");
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_docs_team_assistant_workflow() -> Result<()> {
    let mut fixture = build_docs_team_fixture("MAIN_OK", "DOCS_REVIEW_OK", "DOCS_DRAFT_OK").await?;
    let prompt = "Update the docs to reflect the latest harness library additions".to_string();

    let mut session = fixture.kernel.create_session().await.unwrap();
    fixture
        .kernel
        .run(&mut session, Some(prompt.clone()))
        .await?;
    fixture.kernel.end_session(&mut session).await?;

    let runtime_dir = fixture.tmp.path().join(".turin/runtime/docs-team");
    let context = fs::read_to_string(runtime_dir.join("context.md"))?;
    let review = fs::read_to_string(runtime_dir.join("review.md"))?;
    let draft = fs::read_to_string(runtime_dir.join("draft.md"))?;
    let brief = fs::read_to_string(runtime_dir.join("brief.md"))?;
    let reviewer_prompt = fs::read_to_string(runtime_dir.join("docs-reviewer-last-request.txt"))?;
    let draft_prompt = fs::read_to_string(runtime_dir.join("draft-writer-last-request.txt"))?;

    assert!(context.contains("# PUBLIC_SURFACE.md"));
    assert!(context.contains("# DOCS_TARGETS.md"));
    assert!(context.contains("# DRIFT_NOTES.md"));
    assert!(context.contains("# STYLE_NOTES.md"));
    assert_eq!(review, "DOCS_REVIEW_OK");
    assert_eq!(draft, "DOCS_DRAFT_OK");
    assert!(brief.contains("## Review Findings"));
    assert!(brief.contains("## Draft Update"));
    assert!(reviewer_prompt.contains("Docs task"));
    assert!(draft_prompt.contains("Review findings"));

    let store = open_harness_owned_store(fixture.tmp.path()).await?;
    let conn = store.get_connection().await?;
    let mut rows = conn
        .query(
            "SELECT prompt, review, draft FROM docs_team_runs ORDER BY id DESC LIMIT 1",
            (),
        )
        .await?;
    let row = rows.next().await?.expect("expected docs team row");
    let prompt_value: String = row.get(0)?;
    let review_value: String = row.get(1)?;
    let draft_value: String = row.get(2)?;
    assert_eq!(prompt_value, prompt);
    assert_eq!(review_value, "DOCS_REVIEW_OK");
    assert_eq!(draft_value, "DOCS_DRAFT_OK");
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
    bind_named_harness(&mut config, "reviewer", &reviewer_harness_dir);
    config.governance = GovernanceConfig {
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
            tools: Default::default(),
            id: "reviewer".to_string(),
            model: "mock-model".to_string(),
            provider: "mock_review".to_string(),
            system_prompt: "You are a reviewer.".to_string(),
            thinking: None,
            harness: Some("reviewer".to_string()),
            idle_timeout_seconds: None,
            linked_runtime_lanes: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
    );

    let mut kernel = build_kernel(config).await?;
    let mut session = kernel.create_session().await.unwrap();
    let prompt = "Review the patch for race conditions".to_string();
    kernel.run(&mut session, Some(prompt.clone())).await?;
    kernel.end_session(&mut session).await?;

    let review_artifact = tmp.path().join(".turin/runtime/peer-review.txt");
    let input_artifact = tmp.path().join(".turin/runtime/peer-review-input.txt");
    assert_eq!(fs::read_to_string(review_artifact)?, "REVIEW_OK");
    assert_eq!(fs::read_to_string(input_artifact)?, prompt);

    assert!(
        wait_for_persisted_session_events(
            &kernel,
            session
                .internal_id
                .expect("main session should have internal id"),
            &[
                "governance_grant_issue",
                "governance_grant_use",
                "governance_grant_revoke",
            ],
        )
        .await?,
        "expected persisted governance grant audit events"
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
    let mut session = kernel.create_session().await.unwrap();
    let prompt = "Deployment note: restart API after schema migration".to_string();
    kernel.run(&mut session, Some(prompt.clone())).await?;
    kernel.end_session(&mut session).await?;

    let snapshot = fs::read_to_string(tmp.path().join(".turin/runtime/journal-last.txt"))?;
    assert_eq!(snapshot, prompt);

    let store = open_harness_owned_store(tmp.path()).await?;
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
    let mut session = kernel.create_session().await.unwrap();
    let prompt = "Review the patch for risky assumptions in the provider layer".to_string();
    kernel.run(&mut session, Some(prompt.clone())).await?;
    kernel.end_session(&mut session).await?;

    let runtime_dir = tmp.path().join(".turin/runtime/code-review");
    let context = fs::read_to_string(runtime_dir.join("context.md"))?;
    let last_request = fs::read_to_string(runtime_dir.join("last-request.txt"))?;
    assert!(context.contains("# REVIEW_STYLE.md"));
    assert!(context.contains("# RISK_AREAS.md"));
    assert_eq!(last_request, prompt);

    let store = open_harness_owned_store(tmp.path()).await?;
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
    let mut session = kernel.create_session().await.unwrap();
    let prompt = "Break down the next phase of the harness library work".to_string();
    kernel.run(&mut session, Some(prompt.clone())).await?;
    kernel.end_session(&mut session).await?;

    let runtime_dir = tmp.path().join(".turin/runtime/task-planner");
    let context = fs::read_to_string(runtime_dir.join("context.md"))?;
    let last_request = fs::read_to_string(runtime_dir.join("last-request.txt"))?;
    assert!(context.contains("# PLANNING_STYLE.md"));
    assert!(context.contains("# DELIVERY_CONSTRAINTS.md"));
    assert_eq!(last_request, prompt);

    let store = open_harness_owned_store(tmp.path()).await?;
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
async fn test_spec_writer_block() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    fs::create_dir(&harness_dir)?;
    copy_dir_contents(
        library_block_path("spec_writer").join("workspace"),
        tmp.path(),
    )?;
    copy_dir_contents(
        library_block_path("spec_writer").join("harness"),
        &harness_dir,
    )?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider("SPEC_REPLY"));
    let config = base_config(tmp.path(), &harness_dir, "mock_main", providers);

    let mut kernel = build_kernel(config).await?;
    let mut session = kernel.create_session().await.unwrap();
    let prompt = "Turn the rough idea into a concrete implementation spec".to_string();
    kernel.run(&mut session, Some(prompt.clone())).await?;
    kernel.end_session(&mut session).await?;

    let runtime_dir = tmp.path().join(".turin/runtime/spec-writer");
    let contract = fs::read_to_string(runtime_dir.join("contract.md"))?;
    let last_request = fs::read_to_string(runtime_dir.join("last-request.txt"))?;
    assert!(contract.contains("# IDEA.md"));
    assert!(contract.contains("# ACCEPTANCE.md"));
    assert!(contract.contains("# CONTEXT.md"));
    assert_eq!(last_request, prompt);

    let store = open_harness_owned_store(tmp.path()).await?;
    let conn = store.get_connection().await?;
    let mut rows = conn
        .query(
            "SELECT prompt FROM spec_writer_runs ORDER BY id DESC LIMIT 1",
            (),
        )
        .await?;
    let row = rows.next().await?.expect("expected spec writer row");
    let stored_prompt: String = row.get(0)?;
    assert_eq!(stored_prompt, prompt);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_test_gap_finder_block() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    fs::create_dir(&harness_dir)?;
    copy_dir_contents(
        library_block_path("test_gap_finder").join("workspace"),
        tmp.path(),
    )?;
    copy_dir_contents(
        library_block_path("test_gap_finder").join("harness"),
        &harness_dir,
    )?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider("TEST_GAP_REPLY"));
    let config = base_config(tmp.path(), &harness_dir, "mock_main", providers);

    let mut kernel = build_kernel(config).await?;
    let mut session = kernel.create_session().await.unwrap();
    let prompt = "Identify the likely missing tests for the governance refactor".to_string();
    kernel.run(&mut session, Some(prompt.clone())).await?;
    kernel.end_session(&mut session).await?;

    let runtime_dir = tmp.path().join(".turin/runtime/test-gap-finder");
    let contract = fs::read_to_string(runtime_dir.join("contract.md"))?;
    let last_request = fs::read_to_string(runtime_dir.join("last-request.txt"))?;
    assert!(contract.contains("# CHANGE_SUMMARY.md"));
    assert!(contract.contains("# TESTING_POLICY.md"));
    assert!(contract.contains("# RISK_AREAS.md"));
    assert_eq!(last_request, prompt);

    let store = open_harness_owned_store(tmp.path()).await?;
    let conn = store.get_connection().await?;
    let mut rows = conn
        .query(
            "SELECT prompt FROM test_gap_runs ORDER BY id DESC LIMIT 1",
            (),
        )
        .await?;
    let row = rows.next().await?.expect("expected test gap row");
    let stored_prompt: String = row.get(0)?;
    assert_eq!(stored_prompt, prompt);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_repo_librarian_block() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    fs::create_dir(&harness_dir)?;
    copy_dir_contents(
        library_block_path("repo_librarian").join("workspace"),
        tmp.path(),
    )?;
    copy_dir_contents(
        library_block_path("repo_librarian").join("harness"),
        &harness_dir,
    )?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider("REPO_REPLY"));
    let config = base_config(tmp.path(), &harness_dir, "mock_main", providers);

    let mut kernel = build_kernel(config).await?;
    let mut session = kernel.create_session().await.unwrap();
    let prompt = "Route this task according to the repository contracts".to_string();
    kernel.run(&mut session, Some(prompt.clone())).await?;
    kernel.end_session(&mut session).await?;

    let runtime_dir = tmp.path().join(".turin/runtime/repo-librarian");
    let contract = fs::read_to_string(runtime_dir.join("contract.md"))?;
    let last_request = fs::read_to_string(runtime_dir.join("last-request.txt"))?;
    assert!(contract.contains("# SOUL.md"));
    assert!(contract.contains("# AGENTS.md"));
    assert!(contract.contains("# ARCHITECTURE.md"));
    assert!(contract.contains("# CONVENTIONS.md"));
    assert_eq!(last_request, prompt);

    let store = open_harness_owned_store(tmp.path()).await?;
    let conn = store.get_connection().await?;
    let mut rows = conn
        .query(
            "SELECT prompt FROM repo_librarian_runs ORDER BY id DESC LIMIT 1",
            (),
        )
        .await?;
    let row = rows.next().await?.expect("expected repo librarian row");
    let stored_prompt: String = row.get(0)?;
    assert_eq!(stored_prompt, prompt);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_release_readiness_checker_block() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    fs::create_dir(&harness_dir)?;
    copy_dir_contents(
        library_block_path("release_readiness_checker").join("workspace"),
        tmp.path(),
    )?;
    copy_dir_contents(
        library_block_path("release_readiness_checker").join("harness"),
        &harness_dir,
    )?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider("READY_REPLY"));
    let config = base_config(tmp.path(), &harness_dir, "mock_main", providers);

    let mut kernel = build_kernel(config).await?;
    let mut session = kernel.create_session().await.unwrap();
    let prompt = "Assess whether the next release looks ready to ship".to_string();
    kernel.run(&mut session, Some(prompt.clone())).await?;
    kernel.end_session(&mut session).await?;

    let runtime_dir = tmp.path().join(".turin/runtime/release-readiness");
    let contract = fs::read_to_string(runtime_dir.join("contract.md"))?;
    let last_request = fs::read_to_string(runtime_dir.join("last-request.txt"))?;
    assert!(contract.contains("# CHECKLIST.md"));
    assert!(contract.contains("# RISK_REGISTER.md"));
    assert!(contract.contains("# RELEASE_NOTES_CONTEXT.md"));
    assert_eq!(last_request, prompt);

    let store = open_harness_owned_store(tmp.path()).await?;
    let conn = store.get_connection().await?;
    let mut rows = conn
        .query(
            "SELECT prompt FROM release_readiness_runs ORDER BY id DESC LIMIT 1",
            (),
        )
        .await?;
    let row = rows.next().await?.expect("expected release readiness row");
    let stored_prompt: String = row.get(0)?;
    assert_eq!(stored_prompt, prompt);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_docs_maintainer_block() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    fs::create_dir(&harness_dir)?;
    copy_dir_contents(
        library_block_path("docs_maintainer").join("workspace"),
        tmp.path(),
    )?;
    copy_dir_contents(
        library_block_path("docs_maintainer").join("harness"),
        &harness_dir,
    )?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider("DOCS_REPLY"));
    let config = base_config(tmp.path(), &harness_dir, "mock_main", providers);

    let mut kernel = build_kernel(config).await?;
    let mut session = kernel.create_session().await.unwrap();
    let prompt =
        "Identify the docs that need updating after the latest library changes".to_string();
    kernel.run(&mut session, Some(prompt.clone())).await?;
    kernel.end_session(&mut session).await?;

    let runtime_dir = tmp.path().join(".turin/runtime/docs-maintainer");
    let contract = fs::read_to_string(runtime_dir.join("contract.md"))?;
    let last_request = fs::read_to_string(runtime_dir.join("last-request.txt"))?;
    assert!(contract.contains("# PUBLIC_SURFACE.md"));
    assert!(contract.contains("# DOCS_POLICY.md"));
    assert!(contract.contains("# DRIFT_SIGNALS.md"));
    assert_eq!(last_request, prompt);

    let store = open_harness_owned_store(tmp.path()).await?;
    let conn = store.get_connection().await?;
    let mut rows = conn
        .query(
            "SELECT prompt FROM docs_maintainer_runs ORDER BY id DESC LIMIT 1",
            (),
        )
        .await?;
    let row = rows.next().await?.expect("expected docs maintainer row");
    let stored_prompt: String = row.get(0)?;
    assert_eq!(stored_prompt, prompt);
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_changelog_writer_block() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    fs::create_dir(&harness_dir)?;
    copy_dir_contents(
        library_block_path("changelog_writer").join("workspace"),
        tmp.path(),
    )?;
    copy_dir_contents(
        library_block_path("changelog_writer").join("harness"),
        &harness_dir,
    )?;

    let mut providers = HashMap::new();
    providers.insert("mock_main".to_string(), mock_provider("CHANGELOG_REPLY"));
    let config = base_config(tmp.path(), &harness_dir, "mock_main", providers);

    let mut kernel = build_kernel(config).await?;
    let mut session = kernel.create_session().await.unwrap();
    let prompt = "Draft a concise changelog entry for the latest harness library work".to_string();
    kernel.run(&mut session, Some(prompt.clone())).await?;
    kernel.end_session(&mut session).await?;

    let runtime_dir = tmp.path().join(".turin/runtime/changelog-writer");
    let contract = fs::read_to_string(runtime_dir.join("contract.md"))?;
    let last_request = fs::read_to_string(runtime_dir.join("last-request.txt"))?;
    assert!(contract.contains("# RELEASE_SCOPE.md"));
    assert!(contract.contains("# MERGED_CHANGES.md"));
    assert!(contract.contains("# WRITING_STYLE.md"));
    assert_eq!(last_request, prompt);

    let store = open_harness_owned_store(tmp.path()).await?;
    let conn = store.get_connection().await?;
    let mut rows = conn
        .query(
            "SELECT prompt FROM changelog_writer_runs ORDER BY id DESC LIMIT 1",
            (),
        )
        .await?;
    let row = rows.next().await?.expect("expected changelog writer row");
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
    bind_named_harness(&mut config, "reviewer", &reviewer_harness_dir);
    config.governance = GovernanceConfig {
        enforcement_enabled: true,
        agents: HashMap::from([(
            "default".to_string(),
            turin_core::kernel::config::GovernanceAgentCapabilitiesConfig {
                capability_set: None,
                max_capabilities: HashMap::new(),
                allowed_child_agents: vec!["reviewer".to_string()],
            },
        )]),
        ..GovernanceConfig::default()
    };
    config.agents.insert(
        "reviewer".to_string(),
        AgentConfig {
            tools: Default::default(),
            id: "reviewer".to_string(),
            model: "mock-model".to_string(),
            provider: "mock_review".to_string(),
            system_prompt: "You are a reviewer.".to_string(),
            thinking: None,
            harness: Some("reviewer".to_string()),
            idle_timeout_seconds: None,
            linked_runtime_lanes: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
    );

    let mut kernel = build_kernel(config).await?;
    let mut session = kernel.create_session().await.unwrap();
    let prompt = "Review the request with constrained peer capabilities".to_string();
    kernel.run(&mut session, Some(prompt.clone())).await?;
    kernel.end_session(&mut session).await?;

    let review_artifact = tmp.path().join(".turin/runtime/delegated-review.txt");
    let input_artifact = tmp.path().join(".turin/runtime/delegated-review-input.txt");
    assert_eq!(fs::read_to_string(review_artifact)?, "DELEGATED_REVIEW_OK");
    assert_eq!(fs::read_to_string(input_artifact)?, prompt);

    let saw_reviewer_output =
        wait_for_persisted_agent_output(&kernel, "reviewer", "DELEGATED_REVIEW_OK").await?;
    assert!(
        saw_reviewer_output,
        "expected persisted reviewer assistant output"
    );

    Ok(())
}
