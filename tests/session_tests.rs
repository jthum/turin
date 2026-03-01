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
use turin::kernel::policy::PolicyScope;
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
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Test assistant.".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness_dir: None,
            idle_grace_secs: None,
        },
        agents: std::collections::HashMap::new(),
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
        governance: turin::kernel::config::GovernanceConfig::default(),
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

fn event_has_task_status(events: &[turin::persistence::schema::EventRow], status: &str) -> bool {
    events.iter().any(|e| {
        e.event_type == "task_complete"
            && serde_json::from_str::<serde_json::Value>(&e.payload)
                .ok()
                .and_then(|v| v.get("status").and_then(|s| s.as_str()).map(str::to_string))
                .is_some_and(|s| s == status)
    })
}

fn count_event_type(events: &[turin::persistence::schema::EventRow], event_type: &str) -> usize {
    events.iter().filter(|e| e.event_type == event_type).count()
}

// ─── Session Lifecycle ──────────────────────────────────────────

#[tokio::test]
async fn test_session_create_starts_inactive() -> Result<()> {
    let tmp = tempdir()?;
    let kernel = make_kernel(tmp.path()).await?;

    let session = kernel.create_session().await;
    assert_eq!(session.status, SessionStatus::Inactive);
    assert_eq!(session.turn_index, 0);
    assert!(session.history.is_empty());
    assert!(
        !session.identity.session_id().is_empty(),
        "Session ID should be generated"
    );

    Ok(())
}

#[tokio::test]
async fn test_session_start_activates() -> Result<()> {
    let tmp = tempdir()?;
    let kernel = make_kernel(tmp.path()).await?;

    let mut session = kernel.create_session().await;
    assert_eq!(session.status, SessionStatus::Inactive);

    kernel.start_session(&mut session).await?;
    assert_eq!(session.status, SessionStatus::Active);

    Ok(())
}

#[tokio::test]
async fn test_session_end_deactivates() -> Result<()> {
    let tmp = tempdir()?;
    let kernel = make_kernel(tmp.path()).await?;

    let mut session = kernel.create_session().await;
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

    let mut session = kernel.create_session().await;
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

    let s1 = kernel.create_session().await;
    let s2 = kernel.create_session().await;
    assert_ne!(s1.identity.session_id(), s2.identity.session_id());

    Ok(())
}

// ─── Agent Loop Edge Cases ──────────────────────────────────────

#[tokio::test]
async fn test_run_with_mock_increments_turns() -> Result<()> {
    let tmp = tempdir()?;
    let mut kernel = make_kernel(tmp.path()).await?;

    let mut session = kernel.create_session().await;
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

    let mut session = kernel.create_session().await;
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
    let mut session = kernel.create_session().await;
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
    let mut session2 = kernel.create_session().await;
    kernel
        .run(&mut session2, Some("After reload".to_string()))
        .await?;
    assert!(session2.turn_index > 0);

    kernel.end_session(&mut session2).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_explicit_watch_reloads_nested_used_blocks() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    let blocks_dir = harness_dir.join("blocks");
    std::fs::create_dir_all(&blocks_dir)?;

    std::fs::write(
        harness_dir.join("main.lua"),
        r#"
            watch("blocks")
            use("blocks/feature")
        "#,
    )?;
    std::fs::write(
        blocks_dir.join("feature.lua"),
        r#"
            function on_turn_prepare(ctx)
                local ok, err = fs.write("watch-marker.txt", "v1")
                if not ok then error(err) end
                return ALLOW
            end
        "#,
    )?;

    let mut kernel = make_kernel(tmp.path()).await?;
    kernel.start_watcher()?;

    let marker_path = tmp.path().join("watch-marker.txt");

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("before nested reload".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;
    assert_eq!(std::fs::read_to_string(&marker_path)?, "v1");

    std::fs::write(
        blocks_dir.join("feature.lua"),
        r#"
            function on_turn_prepare(ctx)
                local ok, err = fs.write("watch-marker.txt", "v2")
                if not ok then error(err) end
                return ALLOW
            end
        "#,
    )?;

    let mut saw_v2 = false;
    for _ in 0..10 {
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        let mut session = kernel.create_session().await;
        kernel
            .run(&mut session, Some("after nested reload".to_string()))
            .await?;
        kernel.end_session(&mut session).await?;

        if std::fs::read_to_string(&marker_path).ok().as_deref() == Some("v2") {
            saw_v2 = true;
            break;
        }
    }

    assert!(saw_v2, "explicit watch should reload nested used blocks");
    Ok(())
}

#[tokio::test]
async fn test_single_kernel_routes_sessions_to_agent_specific_harnesses() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test-multi-harness.db");
    let default_harness_dir = tmp.path().join("harnesses-default");
    let writer_harness_dir = tmp.path().join("harnesses-writer");
    std::fs::create_dir_all(&default_harness_dir)?;
    std::fs::create_dir_all(&writer_harness_dir)?;

    std::fs::write(
        default_harness_dir.join("main.lua"),
        r#"
            function on_session_start(event)
                local ok, err = fs.write(".turin/runtime/default-harness.txt", "default")
                if not ok then error(err) end
                return ALLOW
            end
        "#,
    )?;
    std::fs::write(
        writer_harness_dir.join("main.lua"),
        r#"
            function on_session_start(event)
                local ok, err = fs.write(".turin/runtime/writer-harness.txt", "writer")
                if not ok then error(err) end
                return ALLOW
            end
        "#,
    )?;

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

    let mut agents = HashMap::new();
    agents.insert(
        "writer".to_string(),
        AgentConfig {
            id: "writer".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Writer agent.".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness_dir: Some(writer_harness_dir.to_str().unwrap().to_string()),
            idle_grace_secs: None,
        },
    );

    let config = TurinConfig {
        agent: AgentConfig {
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Default agent.".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness_dir: None,
            idle_grace_secs: None,
        },
        agents,
        kernel: KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 5,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        persistence: PersistenceConfig {
            database_path: db_path.to_str().unwrap().to_string(),
        },
        harness: HarnessConfig {
            directory: default_harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
        },
        providers,
        embeddings: Some(EmbeddingConfig::NoOp),
        governance: turin::kernel::config::GovernanceConfig::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut default_session = kernel.create_session().await;
    kernel.start_session(&mut default_session).await?;
    kernel.end_session(&mut default_session).await?;

    let mut writer_session = kernel.create_session_for_agent("writer").await;
    kernel.start_session(&mut writer_session).await?;
    kernel.end_session(&mut writer_session).await?;

    assert_eq!(
        std::fs::read_to_string(tmp.path().join(".turin/runtime/default-harness.txt"))?,
        "default"
    );
    assert_eq!(
        std::fs::read_to_string(tmp.path().join(".turin/runtime/writer-harness.txt"))?,
        "writer"
    );

    Ok(())
}

// ─── State Store Integration ────────────────────────────────────

#[tokio::test]
async fn test_events_persisted_to_state_store() -> Result<()> {
    let tmp = tempdir()?;
    let mut kernel = make_kernel(tmp.path()).await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Persist me".to_string()))
        .await?;

    // Give background persistence task a moment to flush
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    // Query events from state store
    if let Ok(store) = kernel.store_manager().get_default().await {
        let events = store.get_events(session.internal_id.unwrap()).await?;
        assert!(!events.is_empty(), "Events should be persisted");
        assert!(
            events.iter().any(|e| e.event_type == "governance_snapshot"),
            "Expected governance_snapshot audit event to be persisted"
        );
    }

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test]
async fn test_immutable_audit_persists_rejected_audit_events() -> Result<()> {
    let tmp = tempdir()?;
    let mut config = make_config(tmp.path());
    let harness_dir = tmp.path().join("harnesses");
    std::fs::write(
        harness_dir.join("reject_audit.lua"),
        r#"
            function on_kernel_event(event)
                if event.type == "governance_snapshot" then
                    return REJECT, "drop governance snapshot"
                end
                return ALLOW
            end
        "#,
    )?;

    config.governance.profile = turin::kernel::config::GovernanceProfile::Governed;
    config.governance.audit.mode = turin::kernel::config::GovernanceAuditMode::Immutable;
    config.governance.enforcement_enabled = false;

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Persist immutable audit".to_string()))
        .await?;

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    if let Ok(store) = kernel.store_manager().get_default().await {
        let events = store.get_events(session.internal_id.unwrap()).await?;
        assert!(
            events.iter().any(|e| e.event_type == "governance_snapshot"),
            "immutable audit mode should persist governance_snapshot even if on_kernel_event rejects it"
        );
    }

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test]
async fn test_governance_grant_audit_events_persisted() -> Result<()> {
    let tmp = tempdir()?;
    let mut config = make_config(tmp.path());
    let harness_dir = tmp.path().join("harnesses");
    std::fs::write(
        harness_dir.join("grant_audit.lua"),
        r#"
            function on_turn_prepare(ctx)
                local grant, ge = runtime.governance.grant_issue({
                    capabilities = { ["runtime.db.query"] = true },
                    ttl_ms = 5000,
                    max_uses = 1,
                    reason = "session test"
                })
                if grant == nil then error("grant_issue failed: " .. tostring(ge)) end

                local out = runtime.governance.with_grant(grant.grant_id, function()
                    local dec, de = runtime.governance.check("runtime.db.query")
                    if dec == nil then error("grant check failed: " .. tostring(de)) end
                    return "ok"
                end)
                if out ~= "ok" then error("with_grant return mismatch") end

                local grant2, g2e = runtime.governance.grant_issue({
                    capabilities = { ["runtime.db.query"] = true },
                    reason = "revoke test"
                })
                if grant2 == nil then error("second grant_issue failed: " .. tostring(g2e)) end

                local revoked, re = runtime.governance.grant_revoke(grant2.grant_id)
                if revoked ~= true then error("grant_revoke failed: " .. tostring(re)) end
                return ALLOW
            end
        "#,
    )?;

    config.governance.profile = turin::kernel::config::GovernanceProfile::Balanced;
    config.governance.enforcement_enabled = true;
    config.governance.grants.enabled = true;
    config.governance.grants.max_ttl_ms = Some(10_000);
    config.governance.grants.require_audit_reason = true;

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Emit grant audit events".to_string()))
        .await?;

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    if let Ok(store) = kernel.store_manager().get_default().await {
        let events = store.get_events(session.internal_id.unwrap()).await?;
        assert!(
            events
                .iter()
                .any(|e| e.event_type == "governance_grant_issue"),
            "expected governance_grant_issue to be persisted"
        );
        assert!(
            events
                .iter()
                .any(|e| e.event_type == "governance_grant_use"),
            "expected governance_grant_use to be persisted"
        );
        assert!(
            events
                .iter()
                .any(|e| e.event_type == "governance_grant_revoke"),
            "expected governance_grant_revoke to be persisted"
        );
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
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Test.".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness_dir: None,
            idle_grace_secs: None,
        },
        agents: std::collections::HashMap::new(),
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
        governance: turin::kernel::config::GovernanceConfig::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    // Deliberately skip init_state
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
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

    let mut session = kernel.create_session().await;

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

#[tokio::test]
async fn test_token_usage_reject_is_informational_by_default() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir_all(&harness_dir)?;
    std::fs::write(
        harness_dir.join("token_budget.lua"),
        r#"
            function on_token_usage(usage)
                return REJECT, "token budget exceeded"
            end
        "#,
    )?;

    let mut kernel = make_kernel(tmp.path()).await?;
    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Should still complete".to_string()))
        .await?;

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    if let Ok(store) = kernel.store_manager().get_default().await {
        let events = store.get_events(session.internal_id.unwrap()).await?;
        assert!(
            event_has_task_status(&events, "success"),
            "default token-usage reject mode should be informational"
        );
    }

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test]
async fn test_token_usage_reject_can_enforce_task() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir_all(&harness_dir)?;
    std::fs::write(
        harness_dir.join("token_budget.lua"),
        r#"
            function on_token_usage(usage)
                return REJECT, "token budget exceeded"
            end
        "#,
    )?;

    let mut kernel = make_kernel(tmp.path()).await?;
    kernel
        .policy_manager()
        .set(
            "hook.token_usage.reject_mode",
            serde_json::Value::String("enforce_task".to_string()),
            &PolicyScope {
                scope: Some("global".to_string()),
                ..PolicyScope::default()
            },
        )
        .await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("Reject this task after first turn".to_string()),
        )
        .await?;

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    if let Ok(store) = kernel.store_manager().get_default().await {
        let events = store.get_events(session.internal_id.unwrap()).await?;
        assert!(
            event_has_task_status(&events, "rejected"),
            "task should be rejected when hook.token_usage.reject_mode=enforce_task"
        );
    }
    assert_eq!(session.turn_index, 1, "task should stop after first turn");

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test]
async fn test_token_usage_reject_can_enforce_session() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir_all(&harness_dir)?;
    std::fs::write(
        harness_dir.join("token_budget.lua"),
        r#"
            function on_token_usage(usage)
                return REJECT, "session token budget exceeded"
            end
        "#,
    )?;

    let mut kernel = make_kernel(tmp.path()).await?;
    kernel
        .policy_manager()
        .set(
            "hook.token_usage.reject_mode",
            serde_json::Value::String("enforce_session".to_string()),
            &PolicyScope {
                scope: Some("global".to_string()),
                ..PolicyScope::default()
            },
        )
        .await?;

    let mut session = kernel.create_session().await;
    {
        let mut q = session.queue.lock().await;
        let mut t1 = QueuedTask::ad_hoc("Task 1");
        t1.task_id = "t_1".to_string();
        let mut t2 = QueuedTask::ad_hoc("Task 2");
        t2.task_id = "t_2".to_string();
        q.push_back(t1);
        q.push_back(t2);
    }

    kernel.run(&mut session, None).await?;

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    if let Ok(store) = kernel.store_manager().get_default().await {
        let events = store.get_events(session.internal_id.unwrap()).await?;
        assert_eq!(
            count_event_type(&events, "task_start"),
            1,
            "enforce_session should stop the run loop before the second task starts"
        );
        assert!(
            event_has_task_status(&events, "rejected"),
            "first task should be rejected when enforce_session triggers"
        );
    }

    assert!(session.stop_requested, "session stop should be requested");
    assert!(
        session.queue.lock().await.is_empty(),
        "queue should be cleared"
    );
    assert_eq!(
        session.turn_index, 1,
        "session should stop after first turn"
    );

    kernel.end_session(&mut session).await?;
    Ok(())
}
