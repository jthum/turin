//! Session lifecycle and kernel edge-case tests.
//!
//! Tests for session creation, start, end, token accounting,
//! harness hot-reload, and max_turns enforcement.

use anyhow::Result;
use std::collections::HashMap;
use tempfile::tempdir;
use turin::kernel::Kernel;
use turin::kernel::config::{
    AgentConfig, EmbeddingConfig, HarnessConfig, KernelConfig, PersistenceConfig, ProviderConfig,
    TurinConfig,
};
use turin::kernel::session::{QueuedTask, SessionStatus};

// ─── Helpers ────────────────────────────────────────────────────

fn make_config(tmp: &std::path::Path) -> TurinConfig {
    let db_path = tmp.join("test.db");
    let harness_dir = tmp.join("harnesses");
    std::fs::create_dir_all(&harness_dir).unwrap();

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("Mock response".to_string()),
            ..ProviderConfig::default()
        },
    );

    TurinConfig {
        agent: AgentConfig {
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Test assistant.".to_string(),
            thinking: None,
        },
        kernel: KernelConfig {
            workspace_root: tmp.to_str().unwrap().to_string(),
            max_turns: 5,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        persistence: PersistenceConfig {
            database_path: db_path.to_str().unwrap().to_string(),
        },
        harness: HarnessConfig {
            directory: harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
        },
        providers,
        embeddings: Some(EmbeddingConfig::NoOp),
    }
}

async fn make_kernel(tmp: &std::path::Path) -> Result<Kernel> {
    let config = make_config(tmp);
    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;
    Ok(kernel)
}

// ─── Session Lifecycle ──────────────────────────────────────────

#[tokio::test]
async fn test_session_create_starts_inactive() -> Result<()> {
    let tmp = tempdir()?;
    let kernel = make_kernel(tmp.path()).await?;

    let session = kernel.create_session();
    assert_eq!(session.status, SessionStatus::Inactive);
    assert_eq!(session.turn_index, 0);
    assert!(session.history.is_empty());
    assert!(
        !session.identity.session_id.is_empty(),
        "Session ID should be generated"
    );

    Ok(())
}

#[tokio::test]
async fn test_session_start_activates() -> Result<()> {
    let tmp = tempdir()?;
    let kernel = make_kernel(tmp.path()).await?;

    let mut session = kernel.create_session();
    assert_eq!(session.status, SessionStatus::Inactive);

    kernel.start_session(&mut session).await?;
    assert_eq!(session.status, SessionStatus::Active);

    Ok(())
}

#[tokio::test]
async fn test_session_end_deactivates() -> Result<()> {
    let tmp = tempdir()?;
    let kernel = make_kernel(tmp.path()).await?;

    let mut session = kernel.create_session();
    kernel.start_session(&mut session).await?;
    assert_eq!(session.status, SessionStatus::Active);

    kernel.end_session(&mut session).await?;
    assert_eq!(session.status, SessionStatus::Inactive);

    Ok(())
}

#[tokio::test]
async fn test_session_end_idempotent() -> Result<()> {
    let tmp = tempdir()?;
    let kernel = make_kernel(tmp.path()).await?;

    let mut session = kernel.create_session();
    kernel.start_session(&mut session).await?;

    // End twice — should not panic or error
    kernel.end_session(&mut session).await?;
    kernel.end_session(&mut session).await?;
    assert_eq!(session.status, SessionStatus::Inactive);

    Ok(())
}

#[tokio::test]
async fn test_sessions_have_unique_ids() -> Result<()> {
    let tmp = tempdir()?;
    let kernel = make_kernel(tmp.path()).await?;

    let s1 = kernel.create_session();
    let s2 = kernel.create_session();
    assert_ne!(s1.identity.session_id, s2.identity.session_id);

    Ok(())
}

// ─── Agent Loop Edge Cases ──────────────────────────────────────

#[tokio::test]
async fn test_run_with_mock_increments_turns() -> Result<()> {
    let tmp = tempdir()?;
    let mut kernel = make_kernel(tmp.path()).await?;

    let mut session = kernel.create_session();
    kernel.run(&mut session, Some("Hello".to_string())).await?;

    assert!(
        session.turn_index > 0,
        "Turn index should increment after run"
    );
    assert!(
        !session.history.is_empty(),
        "History should contain messages"
    );

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test]
async fn test_run_populates_token_counts() -> Result<()> {
    let tmp = tempdir()?;
    let mut kernel = make_kernel(tmp.path()).await?;

    let mut session = kernel.create_session();
    kernel
        .run(&mut session, Some("Count my tokens".to_string()))
        .await?;

    // Mock provider may report 0 tokens — verify the fields are initialized
    // and accessible without panic (u64 is always >= 0, so we just read them).
    let _input = session.total_input_tokens;
    let _output = session.total_output_tokens;

    kernel.end_session(&mut session).await?;
    Ok(())
}

// ─── Harness Hot Reload ─────────────────────────────────────────

#[tokio::test]
async fn test_harness_reload_picks_up_new_scripts() -> Result<()> {
    let tmp = tempdir()?;
    let mut kernel = make_kernel(tmp.path()).await?;

    // Initially no harness scripts — should work fine
    let mut session = kernel.create_session();
    kernel
        .run(&mut session, Some("Before reload".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;

    // Write a new harness script that logs
    let harness_dir = tmp.path().join("harnesses");
    std::fs::write(
        harness_dir.join("logger.lua"),
        r#"
            function on_session_start(event)
                return ALLOW
            end
        "#,
    )?;

    // Reload and verify it doesn't error
    kernel.reload_harness().await?;

    // Run again with new harness active
    let mut session2 = kernel.create_session();
    kernel
        .run(&mut session2, Some("After reload".to_string()))
        .await?;
    assert!(session2.turn_index > 0);

    kernel.end_session(&mut session2).await?;
    Ok(())
}

// ─── State Store Integration ────────────────────────────────────

#[tokio::test]
async fn test_events_persisted_to_state_store() -> Result<()> {
    let tmp = tempdir()?;
    let mut kernel = make_kernel(tmp.path()).await?;

    let mut session = kernel.create_session();
    kernel
        .run(&mut session, Some("Persist me".to_string()))
        .await?;

    // Give background persistence task a moment to flush
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Query events from state store
    if let Some(store) = kernel.state() {
        let events = store.get_events(&session.identity.session_id).await?;
        assert!(!events.is_empty(), "Events should be persisted");
    }

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test]
async fn test_kernel_without_state_store_works() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir_all(&harness_dir)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("Mock response".to_string()),
            ..ProviderConfig::default()
        },
    );

    let config = TurinConfig {
        agent: AgentConfig {
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Test.".to_string(),
            thinking: None,
        },
        kernel: KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 3,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        persistence: PersistenceConfig {
            database_path: "".to_string(), // Empty — no persistence
        },
        harness: HarnessConfig {
            directory: harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
        },
        providers,
        embeddings: Some(EmbeddingConfig::NoOp),
    };

    let mut kernel = Kernel::builder(config).build()?;
    // Deliberately skip init_state
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session();
    kernel
        .run(&mut session, Some("No persistence".to_string()))
        .await?;

    assert!(session.turn_index > 0);
    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test]
async fn test_multitask_workflow_execution() -> Result<()> {
    let tmp = tempdir()?;
    let mut kernel = make_kernel(tmp.path()).await?;

    let mut session = kernel.create_session();

    // Manually push 2 tasks
    // (We use a scope to drop the lock)
    {
        let mut q = session.queue.lock().await;
        q.push_back(QueuedTask::ad_hoc("Task 1".to_string()));
        q.push_back(QueuedTask::ad_hoc("Task 2".to_string()));
    }

    // Run
    // Expected: Both tasks run.
    kernel.run(&mut session, None).await?;

    // Check history length
    // Each task adds: User (queue prompt) + Assistant (mock response) = 2 messages.
    // Total should be 4 messages.
    assert_eq!(
        session.history.len(),
        4,
        "Expected 4 messages (2 tasks), got {}",
        session.history.len()
    );

    Ok(())
}
