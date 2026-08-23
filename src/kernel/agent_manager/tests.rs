use super::*;
use crate::harness::scheduler::HarnessSchedulerAccess;
use crate::kernel::Kernel;
use crate::kernel::config::{
    AgentConfig, EmbeddingConfig, GovernanceConfig, HarnessConfig, InferenceConfig,
    InferenceOverrideConfig, KernelConfig, LayoutConfig, PersistenceConfig, ProviderConfig,
    TurinConfig,
};
use crate::kernel::session::QueuedTask;
use crate::kernel::session_refs::{parse_session_reference, session_references_match};
use crate::persistence::state::StateStore;
use crate::tools::{Tool, ToolContext, ToolEffect, ToolError};
use async_trait::async_trait;
use serde_json::json;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use tempfile::tempdir;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

#[test]
fn task_status_prompt_preview_is_normalized_and_bounded() {
    assert_eq!(
        task_prompt_preview("  inspect\nthis   change  "),
        "inspect this change"
    );
    let preview = task_prompt_preview(&"x".repeat(300));
    assert_eq!(preview.chars().count(), 240);
    assert!(preview.ends_with("..."));
}

#[test]
fn linked_runtime_keys_are_bounded_to_physical_lanes() {
    const LANES: usize = 4;
    let keys = (0..100)
        .map(|index| {
            RuntimeSlotKey::linked_for_excluding(
                "worker",
                "parent@state",
                &format!("thread-{index}"),
                &std::collections::HashSet::new(),
                LANES,
            )
            .expect("an empty exclusion set leaves a lane")
            .slot_id
        })
        .collect::<std::collections::HashSet<_>>();

    assert_eq!(keys.len(), LANES);
    assert!(keys.iter().all(|slot_id| slot_id.starts_with("linked_")));
}

#[test]
fn linked_runtime_keys_probe_around_occupied_ancestor_lanes() {
    const LANES: usize = 4;
    let first = RuntimeSlotKey::linked_for_excluding(
        "worker",
        "parent@state",
        "thread",
        &std::collections::HashSet::new(),
        LANES,
    )
    .expect("an empty exclusion set leaves a lane");
    let second = RuntimeSlotKey::linked_for_excluding(
        "worker",
        "parent@state",
        "thread",
        &std::collections::HashSet::from([first.slot_id.clone()]),
        LANES,
    )
    .expect("three lanes remain");
    assert_ne!(first.slot_id, second.slot_id);

    let all_lanes = (0..LANES).map(|lane| format!("linked_{lane}")).collect();
    assert!(
        RuntimeSlotKey::linked_for_excluding(
            "worker",
            "parent@state",
            "thread",
            &all_lanes,
            LANES,
        )
        .is_none()
    );
}

#[test]
fn linked_runtime_keys_support_large_configured_lane_counts() {
    let excluded = (0..99).map(|lane| format!("linked_{lane}")).collect();
    let selected =
        RuntimeSlotKey::linked_for_excluding("worker", "parent@state", "thread", &excluded, 100)
            .expect("the hundredth lane remains available");
    assert_eq!(selected.slot_id, "linked_99");
}

#[tokio::test]
async fn linked_runtime_routing_excludes_busy_same_agent_ancestor_lanes() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harness");
    std::fs::create_dir_all(&harness_dir)?;
    let mut config = test_config(tmp.path(), &harness_dir);
    config.persistence = PersistenceConfig::with_state_path(
        tmp.path().join("state.db").to_string_lossy().to_string(),
    );
    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    let manager = Arc::clone(kernel.agent_manager());
    let store = kernel.store_manager().get_default().await?;

    let root = store
        .create_session(uuid::Uuid::now_v7(), "default", None)
        .await?;
    let parent_public_id = uuid::Uuid::now_v7();
    let parent = store
        .create_linked_session(
            parent_public_id,
            "worker",
            None,
            &crate::persistence::schema::LinkedSessionCreate {
                parent_session_id: root,
                origin_turn_id: None,
                relation_kind: "delegated".to_string(),
                thread_key: "parent".to_string(),
                visibility: "contextual".to_string(),
            },
        )
        .await?;
    let child_public_id = uuid::Uuid::now_v7();
    let child = store
        .create_linked_session(
            child_public_id,
            "worker",
            None,
            &crate::persistence::schema::LinkedSessionCreate {
                parent_session_id: parent,
                origin_turn_id: None,
                relation_kind: "delegated".to_string(),
                thread_key: "child".to_string(),
                visibility: "contextual".to_string(),
            },
        )
        .await?;

    for (lane, public_id) in [(0, parent_public_id), (1, child_public_id)] {
        let control = Arc::new(RuntimeControl::default());
        control.set_current_session_id(Some(public_id.simple().to_string()));
        let handle = Arc::new(AgentRuntimeHandle {
            queue: Arc::new(Mutex::new(VecDeque::new())),
            notify: Arc::new(Notify::new()),
            control,
            shutdown_token: CancellationToken::new(),
            task: None,
            queued_tasks: Arc::new(AtomicUsize::new(0)),
            active_tasks: Arc::new(AtomicUsize::new(1)),
        });
        manager.runtimes.write().await.insert(
            RuntimeSlotKey {
                agent_id: "worker".to_string(),
                slot_id: format!("linked_{lane}"),
            },
            handle,
        );
    }

    let child = store
        .get_session_row(child)
        .await?
        .expect("child session should exist");
    let occupied = manager
        .occupied_ancestor_linked_slots(&store, &child, "worker", 4)
        .await?;
    assert_eq!(
        occupied,
        std::collections::HashSet::from(["linked_0".to_string(), "linked_1".to_string()])
    );
    Ok(())
}

struct TestTool;

#[async_trait]
impl Tool for TestTool {
    fn name(&self) -> &str {
        "test_tool"
    }

    fn description(&self) -> &str {
        "test tool"
    }

    fn parameters_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn execute(
        &self,
        _params: serde_json::Value,
        _ctx: &ToolContext,
    ) -> anyhow::Result<ToolEffect, ToolError> {
        Ok(ToolEffect::Output(crate::tools::ToolOutput::new(
            "ok".to_string(),
        )))
    }
}

fn test_config(workspace_root: &std::path::Path, harness_dir: &std::path::Path) -> TurinConfig {
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
        environment: Default::default(),
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "test".to_string(),
            thinking: None,
            harness: None,
            idle_timeout_seconds: None,
            linked_runtime_lanes: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: HashMap::new(),
        runtime: Default::default(),
        kernel: KernelConfig {
            workspace_root: workspace_root.to_string_lossy().to_string(),
            max_turns: 4,
            heartbeat_interval_seconds: 30,
            initial_spawn_depth: 0,
        },
        layout: LayoutConfig::default(),
        inference: InferenceConfig::default(),
        persistence: PersistenceConfig::with_state_path(
            workspace_root.join("test.db").to_string_lossy().to_string(),
        ),
        harness: HarnessConfig {
            directory: harness_dir.to_string_lossy().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: HashMap::new(),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    }
}

fn signal_test_config(
    workspace_root: &std::path::Path,
    publisher_harness_dir: &std::path::Path,
    reviewer_harness_dir: &std::path::Path,
) -> TurinConfig {
    let mut config = test_config(workspace_root, publisher_harness_dir);
    config.agents.insert(
        "reviewer".to_string(),
        AgentConfig {
            tools: Default::default(),
            id: "reviewer".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "reviewer".to_string(),
            thinking: None,
            harness: Some("reviewer".to_string()),
            idle_timeout_seconds: Some(30),
            linked_runtime_lanes: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
    );
    config.harnesses.insert(
        "reviewer".to_string(),
        HarnessConfig {
            directory: reviewer_harness_dir.to_string_lossy().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
    );
    config
}

async fn abort_all_runtime_slots(manager: &Arc<AgentManager>) {
    let handles: Vec<_> = manager
        .runtimes
        .write()
        .await
        .drain()
        .map(|(_, handle)| handle)
        .collect();
    for handle in handles {
        if let Some(task) = handle.task.as_ref() {
            task.abort();
        }
    }
    tokio::task::yield_now().await;
}

#[tokio::test]
async fn manager_shutdown_stops_cooperative_runtime() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harness");
    std::fs::create_dir_all(&harness_dir)?;

    let kernel = Kernel::builder(test_config(tmp.path(), &harness_dir)).build()?;
    let manager = Arc::clone(kernel.agent_manager());
    let shutdown_token = CancellationToken::new();
    let shutdown_bg = shutdown_token.clone();
    let handle = Arc::new(AgentRuntimeHandle {
        queue: Arc::new(Mutex::new(VecDeque::new())),
        notify: Arc::new(Notify::new()),
        control: Arc::new(RuntimeControl::default()),
        shutdown_token,
        task: Some(tokio::spawn(async move {
            shutdown_bg.cancelled().await;
        })),
        queued_tasks: Arc::new(AtomicUsize::new(0)),
        active_tasks: Arc::new(AtomicUsize::new(0)),
    });
    manager
        .runtimes
        .write()
        .await
        .insert(RuntimeSlotKey::default_for("default"), Arc::clone(&handle));

    manager
        .shutdown_with_grace(std::time::Duration::from_millis(100))
        .await;

    assert!(!handle.is_running());
    assert!(manager.runtimes.read().await.is_empty());
    Ok(())
}

#[tokio::test]
async fn manager_shutdown_aborts_stalled_runtime_after_grace() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harness");
    std::fs::create_dir_all(&harness_dir)?;

    let kernel = Kernel::builder(test_config(tmp.path(), &harness_dir)).build()?;
    let manager = Arc::clone(kernel.agent_manager());
    let runtime_key = RuntimeSlotKey::default_for("default");
    let request_id = "req_stalled_shutdown".to_string();
    let control = Arc::new(RuntimeControl::default());
    control.activate_task(
        Some(request_id.clone()),
        "t_shutdown".to_string(),
        CancellationToken::new(),
    );
    manager.pending_task_states.write().await.insert(
        request_id.clone(),
        PendingTaskRecord {
            runtime_key: runtime_key.clone(),
            session_target: TaskSessionTarget::default(),
            trace_id: "tr_shutdown".to_string(),
            title: None,
            prompt_preview: "stalled shutdown".to_string(),
            state: PendingTaskState::Running,
            runtime_task_id: Some("t_shutdown".to_string()),
            execution: test_execution_snapshot(),
        },
    );
    let handle = Arc::new(AgentRuntimeHandle {
        queue: Arc::new(Mutex::new(VecDeque::new())),
        notify: Arc::new(Notify::new()),
        control,
        shutdown_token: CancellationToken::new(),
        task: Some(tokio::spawn(std::future::pending())),
        queued_tasks: Arc::new(AtomicUsize::new(0)),
        active_tasks: Arc::new(AtomicUsize::new(1)),
    });
    manager
        .runtimes
        .write()
        .await
        .insert(runtime_key, Arc::clone(&handle));

    manager
        .shutdown_with_grace(std::time::Duration::from_millis(20))
        .await;

    assert!(!handle.is_running());
    let completed = manager
        .completed_result(&request_id)
        .await
        .expect("stalled request should become terminal");
    assert_eq!(completed.status, TaskTerminalStatus::Killed);
    assert_eq!(
        completed.error.as_deref(),
        Some("Runtime killed after shutdown grace period")
    );
    Ok(())
}

fn test_execution_snapshot() -> ExecutionStatusSnapshot {
    ExecutionStatusSnapshot::from_execution(
        &crate::kernel::session::ExecutionContext::new(),
        crate::kernel::session::ExecutionWritePolicy::AdvanceBranchHead,
    )
}

#[test]
fn runtime_control_publishes_coherent_snapshots() {
    let control = RuntimeControl::default();
    let (event_tx, _event_rx) = tokio::sync::broadcast::channel(8);
    let cancel_token = CancellationToken::new();
    let execution = test_execution_snapshot();

    control.set_current_session(
        Some("session-1".to_string()),
        Some(event_tx),
        SessionContextOverrides {
            origin_id: Some("client-1".to_string()),
            inference: Default::default(),
        },
        Some(execution.clone()),
        ExecutionConflictPolicy::Detached,
        Some(LiveSessionHistorySnapshot {
            len: 7,
            has_prior_history: true,
        }),
    );
    control.activate_task(
        Some("request-1".to_string()),
        "task-1".to_string(),
        cancel_token.clone(),
    );

    let snapshot = control.snapshot();
    assert_eq!(snapshot.session_id.as_deref(), Some("session-1"));
    assert_eq!(
        snapshot.session_context.origin_id.as_deref(),
        Some("client-1")
    );
    assert_eq!(snapshot.execution, Some(execution));
    assert_eq!(snapshot.conflict_policy, ExecutionConflictPolicy::Detached);
    assert_eq!(snapshot.request_id.as_deref(), Some("request-1"));
    assert_eq!(snapshot.runtime_task_id.as_deref(), Some("task-1"));
    assert_eq!(
        snapshot.history.as_ref().map(|history| history.len),
        Some(7)
    );
    assert!(control.request_task_cancel());
    assert!(cancel_token.is_cancelled());
}

#[tokio::test]
async fn build_shared_peer_kernel_reuses_configured_tool_registry() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harness");
    std::fs::create_dir_all(&harness_dir)?;

    let mut registry = ToolRegistry::new();
    registry.register(Box::new(TestTool))?;

    let kernel = Kernel::builder(test_config(tmp.path(), &harness_dir))
        .with_tool_registry(registry.clone())
        .build()?;

    let peer_kernel = super::peer_session::fork_peer_kernel(&kernel.agent_manager);

    assert_eq!(peer_kernel.tool_registry.len(), registry.len());
    assert!(peer_kernel.tool_registry.get("test_tool").is_some());
    assert!(Arc::ptr_eq(
        &kernel.persistence_locks,
        &peer_kernel.persistence_locks
    ));

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn peer_runtime_idle_zero_waits_for_first_submitted_task() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harness");
    std::fs::create_dir_all(&harness_dir)?;
    std::fs::write(harness_dir.join("main.lua"), "-- idle zero test\n")?;

    let mut config = test_config(tmp.path(), &harness_dir);
    config.agent.idle_timeout_seconds = Some(0);

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let manager = kernel.agent_manager();
    let request_id = manager
        .submit("default", QueuedTask::ad_hoc("hello".to_string()), None)
        .await?;
    let result = manager.await_result(&request_id, Some(5_000)).await?;
    assert_eq!(result.status, TaskTerminalStatus::Success);
    assert_eq!(result.output.as_deref(), Some("Mock response"));

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let running = manager
            .get_status("default")
            .await
            .is_some_and(|status| status.running);
        if manager.list_live_sessions(None).await.is_empty() && !running {
            break;
        }
        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!("timed out waiting for idle-zero peer runtime shutdown");
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }

    Ok(())
}

#[tokio::test]
async fn cancel_task_removes_queued_work_and_records_terminal_result() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harness");
    std::fs::create_dir_all(&harness_dir)?;

    let kernel = Kernel::builder(test_config(tmp.path(), &harness_dir)).build()?;
    let manager = kernel.agent_manager();
    let request_id = "req_cancelled".to_string();
    let (tx_result, rx_result) = oneshot::channel();

    manager
        .pending_results
        .write()
        .await
        .insert(request_id.clone(), rx_result);
    manager.pending_task_states.write().await.insert(
        request_id.clone(),
        PendingTaskRecord {
            runtime_key: RuntimeSlotKey::default_for("default"),
            session_target: TaskSessionTarget::default(),
            trace_id: "tr_cancelled".to_string(),
            title: None,
            prompt_preview: "cancel me".to_string(),
            state: PendingTaskState::Queued,
            runtime_task_id: None,
            execution: test_execution_snapshot(),
        },
    );

    let mut queue = VecDeque::new();
    queue.push_back(PeerAgentTaskEnvelope {
        task: QueuedTask::ad_hoc("cancel me".to_string()),
        request_id: Some(request_id.clone()),
        result_tx: Some(tx_result),
        delegated_capabilities: None,
        promotion_candidate: None,
        linked_session: None,
        session_target: TaskSessionTarget::default(),
    });

    manager.runtimes.write().await.insert(
        RuntimeSlotKey::default_for("default"),
        Arc::new(AgentRuntimeHandle {
            queue: Arc::new(Mutex::new(queue)),
            notify: Arc::new(Notify::new()),
            control: Arc::new(RuntimeControl::default()),
            shutdown_token: CancellationToken::new(),
            task: None,
            queued_tasks: Arc::new(AtomicUsize::new(1)),
            active_tasks: Arc::new(AtomicUsize::new(0)),
        }),
    );

    let snapshot = manager.cancel_task(&request_id).await?;
    assert_eq!(snapshot.state, "completed");
    assert_eq!(snapshot.status, Some(TaskTerminalStatus::Cancelled));
    assert_eq!(
        snapshot.error.as_deref(),
        Some("Task cancelled before execution")
    );
    assert!(
        manager
            .pending_task_states
            .read()
            .await
            .get(&request_id)
            .is_none()
    );
    assert!(
        manager
            .pending_results
            .read()
            .await
            .get(&request_id)
            .is_none()
    );

    let completed = manager
        .get_task(&request_id)
        .await
        .expect("cancelled task should be visible");
    assert_eq!(completed.status, Some(TaskTerminalStatus::Cancelled));

    let later_request_id = "req_cancelled_after_settled".to_string();
    let runtime_key = RuntimeSlotKey::default_for("default");
    let handle = manager
        .runtimes
        .read()
        .await
        .get(&runtime_key)
        .cloned()
        .expect("test runtime");
    manager.pending_task_states.write().await.insert(
        later_request_id.clone(),
        PendingTaskRecord {
            runtime_key,
            session_target: TaskSessionTarget::default(),
            trace_id: "tr_cancelled_later".to_string(),
            title: None,
            prompt_preview: "cancel me too".to_string(),
            state: PendingTaskState::Queued,
            runtime_task_id: None,
            execution: test_execution_snapshot(),
        },
    );
    handle
        .queue
        .lock()
        .expect("runtime queue mutex poisoned")
        .push_back(PeerAgentTaskEnvelope {
            task: QueuedTask::ad_hoc("cancel me too"),
            request_id: Some(later_request_id.clone()),
            result_tx: None,
            delegated_capabilities: None,
            promotion_candidate: None,
            linked_session: None,
            session_target: TaskSessionTarget::default(),
        });
    handle.queued_tasks.fetch_add(1, Ordering::Relaxed);

    manager
        .cancel_pending_requests(&[request_id, later_request_id.clone()])
        .await?;
    assert_eq!(
        manager
            .get_task(&later_request_id)
            .await
            .and_then(|task| task.status),
        Some(TaskTerminalStatus::Cancelled)
    );

    Ok(())
}

#[tokio::test]
async fn closed_result_channel_terminally_fails_pending_task() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harness");
    std::fs::create_dir_all(&harness_dir)?;

    let kernel = Kernel::builder(test_config(tmp.path(), &harness_dir)).build()?;
    let manager = kernel.agent_manager();
    let request_id = "req_lost_result".to_string();
    let (tx_result, rx_result) = oneshot::channel::<PeerAgentTaskResult>();
    drop(tx_result);

    manager
        .pending_results
        .write()
        .await
        .insert(request_id.clone(), rx_result);
    manager.pending_task_states.write().await.insert(
        request_id.clone(),
        PendingTaskRecord {
            runtime_key: RuntimeSlotKey::default_for("default"),
            session_target: TaskSessionTarget {
                session_id: Some("lost-session".to_string()),
                ..TaskSessionTarget::default()
            },
            trace_id: "tr_lost_result".to_string(),
            title: Some("Lost result".to_string()),
            prompt_preview: "lost result".to_string(),
            state: PendingTaskState::Running,
            runtime_task_id: Some("task_lost_result".to_string()),
            execution: test_execution_snapshot(),
        },
    );

    let error = manager
        .await_result(&request_id, None)
        .await
        .expect_err("closed result sender should fail awaiting caller");
    assert!(error.to_string().contains("result channel closed"));
    assert!(
        !manager
            .pending_task_states
            .read()
            .await
            .contains_key(&request_id)
    );
    assert!(
        !manager
            .pending_results
            .read()
            .await
            .contains_key(&request_id)
    );
    let completed = manager
        .get_task(&request_id)
        .await
        .expect("lost result should remain observable as terminal work");
    assert_eq!(completed.status, Some(TaskTerminalStatus::Error));
    assert_eq!(
        completed.error.as_deref(),
        Some("Peer task result channel closed")
    );
    Ok(())
}

#[tokio::test]
async fn cancel_task_marks_running_work_cancelling() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harness");
    std::fs::create_dir_all(&harness_dir)?;

    let kernel = Kernel::builder(test_config(tmp.path(), &harness_dir)).build()?;
    let manager = kernel.agent_manager();
    let request_id = "req_running".to_string();
    let cancel_token = CancellationToken::new();
    let control = Arc::new(RuntimeControl::default());
    control.activate_task(
        Some(request_id.clone()),
        "t_1".to_string(),
        cancel_token.clone(),
    );

    manager.pending_task_states.write().await.insert(
        request_id.clone(),
        PendingTaskRecord {
            runtime_key: RuntimeSlotKey::default_for("default"),
            session_target: TaskSessionTarget::default(),
            trace_id: "tr_running".to_string(),
            title: None,
            prompt_preview: "running".to_string(),
            state: PendingTaskState::Running,
            runtime_task_id: Some("t_1".to_string()),
            execution: test_execution_snapshot(),
        },
    );

    manager.runtimes.write().await.insert(
        RuntimeSlotKey::default_for("default"),
        Arc::new(AgentRuntimeHandle {
            queue: Arc::new(Mutex::new(VecDeque::new())),
            notify: Arc::new(Notify::new()),
            control,
            shutdown_token: CancellationToken::new(),
            task: None,
            queued_tasks: Arc::new(AtomicUsize::new(0)),
            active_tasks: Arc::new(AtomicUsize::new(1)),
        }),
    );

    let snapshot = manager.cancel_task(&request_id).await?;
    assert_eq!(snapshot.state, "cancelling");
    assert!(snapshot.status.is_none());
    assert!(cancel_token.is_cancelled());

    Ok(())
}

#[tokio::test]
async fn cancellation_during_task_activation_is_applied_when_token_becomes_available()
-> anyhow::Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harness");
    std::fs::create_dir_all(&harness_dir)?;

    let kernel = Kernel::builder(test_config(tmp.path(), &harness_dir)).build()?;
    let manager = kernel.agent_manager();
    let request_id = "req_activating".to_string();
    let runtime_key = RuntimeSlotKey::default_for("default");
    let control = Arc::new(RuntimeControl::default());

    manager.pending_task_states.write().await.insert(
        request_id.clone(),
        PendingTaskRecord {
            runtime_key: runtime_key.clone(),
            session_target: TaskSessionTarget::default(),
            trace_id: "tr_activating".to_string(),
            title: None,
            prompt_preview: "activating".to_string(),
            state: PendingTaskState::Running,
            runtime_task_id: None,
            execution: test_execution_snapshot(),
        },
    );
    manager.runtimes.write().await.insert(
        runtime_key,
        Arc::new(AgentRuntimeHandle {
            queue: Arc::new(Mutex::new(VecDeque::new())),
            notify: Arc::new(Notify::new()),
            control: Arc::clone(&control),
            shutdown_token: CancellationToken::new(),
            task: None,
            queued_tasks: Arc::new(AtomicUsize::new(0)),
            active_tasks: Arc::new(AtomicUsize::new(1)),
        }),
    );

    let snapshot = manager.cancel_task(&request_id).await?;
    assert_eq!(snapshot.state, "cancelling");

    let cancel_token = CancellationToken::new();
    control.activate_task(
        Some(request_id.clone()),
        "t_activating".to_string(),
        cancel_token.clone(),
    );
    let cancellation_requested = manager
        .mark_task_running(&request_id, "t_activating".to_string(), None)
        .await;
    if cancellation_requested {
        control.request_task_cancel();
    }

    assert!(cancellation_requested);
    assert!(cancel_token.is_cancelled());
    assert_eq!(
        manager
            .pending_task_states
            .read()
            .await
            .get(&request_id)
            .map(|pending| pending.state),
        Some(PendingTaskState::Cancelling)
    );
    Ok(())
}

#[tokio::test]
async fn cancel_session_cancels_queued_work_and_requests_reset() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harness");
    std::fs::create_dir_all(&harness_dir)?;

    let kernel = Kernel::builder(test_config(tmp.path(), &harness_dir)).build()?;
    let manager = kernel.agent_manager();
    let session_id = "s_cancel";
    let request_id = "req_session_cancel".to_string();
    let (tx_result, rx_result) = oneshot::channel();
    let control = Arc::new(RuntimeControl::default());
    control.set_current_session_id(Some(session_id.to_string()));

    manager
        .pending_results
        .write()
        .await
        .insert(request_id.clone(), rx_result);
    manager.pending_task_states.write().await.insert(
        request_id.clone(),
        PendingTaskRecord {
            runtime_key: RuntimeSlotKey::default_for("default"),
            session_target: TaskSessionTarget {
                session_id: Some(session_id.to_string()),
                ..TaskSessionTarget::default()
            },
            trace_id: "tr_session_cancel".to_string(),
            title: None,
            prompt_preview: "queued".to_string(),
            state: PendingTaskState::Queued,
            runtime_task_id: None,
            execution: test_execution_snapshot(),
        },
    );

    let mut queue = VecDeque::new();
    queue.push_back(PeerAgentTaskEnvelope {
        task: QueuedTask::ad_hoc("queued".to_string()),
        request_id: Some(request_id.clone()),
        result_tx: Some(tx_result),
        delegated_capabilities: None,
        promotion_candidate: None,
        linked_session: None,
        session_target: TaskSessionTarget {
            session_id: Some(session_id.to_string()),
            ..TaskSessionTarget::default()
        },
    });

    manager.runtimes.write().await.insert(
        RuntimeSlotKey::default_for("default"),
        Arc::new(AgentRuntimeHandle {
            queue: Arc::new(Mutex::new(queue)),
            notify: Arc::new(Notify::new()),
            control: Arc::clone(&control),
            shutdown_token: CancellationToken::new(),
            task: None,
            queued_tasks: Arc::new(AtomicUsize::new(1)),
            active_tasks: Arc::new(AtomicUsize::new(0)),
        }),
    );

    let (agent_id, returned_slot_id, returned_session_id) =
        manager.cancel_session(session_id, None).await?;
    assert_eq!(agent_id, "default");
    assert_eq!(returned_slot_id, RuntimeSlotKey::DEFAULT_SLOT_ID);
    assert_eq!(returned_session_id, session_id);
    assert!(matches!(
        control.take_session_reset_request(),
        Some(SessionResetRequest::Fresh(_))
    ));

    let completed = manager
        .get_task(&request_id)
        .await
        .expect("cancelled queued task should be visible");
    assert_eq!(completed.status, Some(TaskTerminalStatus::Cancelled));
    assert_eq!(
        completed.error.as_deref(),
        Some("Session cancelled before execution")
    );

    Ok(())
}

#[tokio::test]
async fn kill_session_marks_running_and_queued_work_killed() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harness");
    std::fs::create_dir_all(&harness_dir)?;

    let kernel = Kernel::builder(test_config(tmp.path(), &harness_dir)).build()?;
    let manager = kernel.agent_manager();
    let session_id = "s_kill";
    let running_request_id = "req_running_kill".to_string();
    let queued_request_id = "req_queued_kill".to_string();
    let (tx_result, rx_result) = oneshot::channel();
    let control = Arc::new(RuntimeControl::default());
    control.set_current_session_id(Some(session_id.to_string()));
    control.activate_task(
        Some(running_request_id.clone()),
        "t_running".to_string(),
        CancellationToken::new(),
    );

    manager
        .pending_results
        .write()
        .await
        .insert(queued_request_id.clone(), rx_result);
    manager.pending_task_states.write().await.insert(
        running_request_id.clone(),
        PendingTaskRecord {
            runtime_key: RuntimeSlotKey::default_for("default"),
            session_target: TaskSessionTarget {
                session_id: Some(session_id.to_string()),
                ..TaskSessionTarget::default()
            },
            trace_id: "tr_running_kill".to_string(),
            title: None,
            prompt_preview: "running".to_string(),
            state: PendingTaskState::Running,
            runtime_task_id: Some("t_running".to_string()),
            execution: test_execution_snapshot(),
        },
    );
    manager.pending_task_states.write().await.insert(
        queued_request_id.clone(),
        PendingTaskRecord {
            runtime_key: RuntimeSlotKey::default_for("default"),
            session_target: TaskSessionTarget {
                session_id: Some(session_id.to_string()),
                ..TaskSessionTarget::default()
            },
            trace_id: "tr_queued_kill".to_string(),
            title: None,
            prompt_preview: "queued".to_string(),
            state: PendingTaskState::Queued,
            runtime_task_id: None,
            execution: test_execution_snapshot(),
        },
    );

    let mut queue = VecDeque::new();
    queue.push_back(PeerAgentTaskEnvelope {
        task: QueuedTask::ad_hoc("queued".to_string()),
        request_id: Some(queued_request_id.clone()),
        result_tx: Some(tx_result),
        delegated_capabilities: None,
        promotion_candidate: None,
        linked_session: None,
        session_target: TaskSessionTarget {
            session_id: Some(session_id.to_string()),
            ..TaskSessionTarget::default()
        },
    });

    manager.runtimes.write().await.insert(
        RuntimeSlotKey::default_for("default"),
        Arc::new(AgentRuntimeHandle {
            queue: Arc::new(Mutex::new(queue)),
            notify: Arc::new(Notify::new()),
            control,
            shutdown_token: CancellationToken::new(),
            task: Some(tokio::spawn(async {
                tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            })),
            queued_tasks: Arc::new(AtomicUsize::new(1)),
            active_tasks: Arc::new(AtomicUsize::new(1)),
        }),
    );

    let (agent_id, returned_slot_id, returned_session_id) =
        manager.kill_session(session_id, None).await?;
    assert_eq!(agent_id, "default");
    assert_eq!(returned_slot_id, RuntimeSlotKey::DEFAULT_SLOT_ID);
    assert_eq!(returned_session_id, session_id);
    assert!(
        manager
            .runtimes
            .read()
            .await
            .get(&RuntimeSlotKey::default_for("default"))
            .is_none()
    );

    let running = manager
        .get_task(&running_request_id)
        .await
        .expect("killed running task should be visible");
    assert_eq!(running.status, Some(TaskTerminalStatus::Killed));

    let queued = manager
        .get_task(&queued_request_id)
        .await
        .expect("killed queued task should be visible");
    assert_eq!(queued.status, Some(TaskTerminalStatus::Killed));
    assert_eq!(queued.error.as_deref(), Some("Session killed"));

    Ok(())
}

#[tokio::test]
async fn pooled_linked_lane_cancels_only_the_target_session() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harness");
    std::fs::create_dir_all(&harness_dir)?;
    let kernel = Kernel::builder(test_config(tmp.path(), &harness_dir)).build()?;
    let manager = kernel.agent_manager();
    let runtime_key = RuntimeSlotKey {
        agent_id: "worker".to_string(),
        slot_id: "linked_0".to_string(),
    };
    let active_session = "active-linked-session";
    let cancelled_session = "cancelled-linked-session";
    let preserved_session = "preserved-linked-session";
    let cancelled_request = "req_cancel_linked".to_string();
    let preserved_request = "req_preserve_linked".to_string();
    let control = Arc::new(RuntimeControl::default());
    control.set_current_session_id(Some(active_session.to_string()));

    for (request_id, session_id) in [
        (&cancelled_request, cancelled_session),
        (&preserved_request, preserved_session),
    ] {
        manager.pending_task_states.write().await.insert(
            request_id.clone(),
            PendingTaskRecord {
                runtime_key: runtime_key.clone(),
                session_target: TaskSessionTarget {
                    session_id: Some(session_id.to_string()),
                    ..TaskSessionTarget::default()
                },
                trace_id: format!("trace-{request_id}"),
                title: None,
                prompt_preview: session_id.to_string(),
                state: PendingTaskState::Queued,
                runtime_task_id: None,
                execution: test_execution_snapshot(),
            },
        );
    }
    let queue = VecDeque::from([
        PeerAgentTaskEnvelope {
            task: QueuedTask::ad_hoc("cancel".to_string()),
            request_id: Some(cancelled_request.clone()),
            result_tx: None,
            delegated_capabilities: None,
            promotion_candidate: None,
            linked_session: None,
            session_target: TaskSessionTarget {
                session_id: Some(cancelled_session.to_string()),
                ..TaskSessionTarget::default()
            },
        },
        PeerAgentTaskEnvelope {
            task: QueuedTask::ad_hoc("preserve".to_string()),
            request_id: Some(preserved_request.clone()),
            result_tx: None,
            delegated_capabilities: None,
            promotion_candidate: None,
            linked_session: None,
            session_target: TaskSessionTarget {
                session_id: Some(preserved_session.to_string()),
                ..TaskSessionTarget::default()
            },
        },
    ]);
    let handle = Arc::new(AgentRuntimeHandle {
        queue: Arc::new(Mutex::new(queue)),
        notify: Arc::new(Notify::new()),
        control,
        shutdown_token: CancellationToken::new(),
        task: None,
        queued_tasks: Arc::new(AtomicUsize::new(2)),
        active_tasks: Arc::new(AtomicUsize::new(0)),
    });
    manager
        .runtimes
        .write()
        .await
        .insert(runtime_key.clone(), Arc::clone(&handle));

    manager.cancel_session(cancelled_session, None).await?;

    {
        let queue = handle
            .queue
            .lock()
            .expect("agent runtime queue mutex poisoned");
        assert_eq!(queue.len(), 1);
        assert_eq!(
            queue.front().and_then(|task| task.request_id.as_deref()),
            Some(preserved_request.as_str())
        );
    }
    assert_eq!(handle.queued_tasks.load(Ordering::Relaxed), 1);
    assert!(manager.runtimes.read().await.contains_key(&runtime_key));
    assert_eq!(
        manager
            .get_task(&cancelled_request)
            .await
            .and_then(|task| task.status),
        Some(TaskTerminalStatus::Cancelled)
    );
    assert!(manager.get_task(&preserved_request).await.is_some());

    let error = manager
        .kill_session(active_session, None)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("unrelated queued work"));
    assert!(manager.runtimes.read().await.contains_key(&runtime_key));
    assert_eq!(handle.queued_tasks.load(Ordering::Relaxed), 1);
    Ok(())
}

#[tokio::test]
async fn pooled_linked_lane_kills_a_queued_session_without_stopping_the_lane() -> anyhow::Result<()>
{
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harness");
    std::fs::create_dir_all(&harness_dir)?;
    let kernel = Kernel::builder(test_config(tmp.path(), &harness_dir)).build()?;
    let manager = kernel.agent_manager();
    let runtime_key = RuntimeSlotKey {
        agent_id: "worker".to_string(),
        slot_id: "linked_0".to_string(),
    };
    let queued_session = "queued-linked-session";
    let request_id = "req_kill_queued_linked".to_string();
    let control = Arc::new(RuntimeControl::default());
    control.set_current_session_id(Some("active-linked-session".to_string()));
    manager.pending_task_states.write().await.insert(
        request_id.clone(),
        PendingTaskRecord {
            runtime_key: runtime_key.clone(),
            session_target: TaskSessionTarget {
                session_id: Some(queued_session.to_string()),
                ..TaskSessionTarget::default()
            },
            trace_id: "trace-kill-queued".to_string(),
            title: None,
            prompt_preview: "kill queued".to_string(),
            state: PendingTaskState::Queued,
            runtime_task_id: None,
            execution: test_execution_snapshot(),
        },
    );
    let queue = VecDeque::from([PeerAgentTaskEnvelope {
        task: QueuedTask::ad_hoc("kill queued".to_string()),
        request_id: Some(request_id.clone()),
        result_tx: None,
        delegated_capabilities: None,
        promotion_candidate: None,
        linked_session: None,
        session_target: TaskSessionTarget {
            session_id: Some(queued_session.to_string()),
            ..TaskSessionTarget::default()
        },
    }]);
    let handle = Arc::new(AgentRuntimeHandle {
        queue: Arc::new(Mutex::new(queue)),
        notify: Arc::new(Notify::new()),
        control,
        shutdown_token: CancellationToken::new(),
        task: None,
        queued_tasks: Arc::new(AtomicUsize::new(1)),
        active_tasks: Arc::new(AtomicUsize::new(0)),
    });
    manager
        .runtimes
        .write()
        .await
        .insert(runtime_key.clone(), Arc::clone(&handle));

    manager.kill_session(queued_session, None).await?;

    assert!(manager.runtimes.read().await.contains_key(&runtime_key));
    assert!(
        handle
            .queue
            .lock()
            .expect("agent runtime queue mutex poisoned")
            .is_empty()
    );
    assert_eq!(handle.queued_tasks.load(Ordering::Relaxed), 0);
    let completed = manager
        .get_task(&request_id)
        .await
        .expect("killed queued task should be visible");
    assert_eq!(completed.session_id.as_deref(), Some(queued_session));
    assert_eq!(completed.status, Some(TaskTerminalStatus::Killed));
    Ok(())
}

#[tokio::test]
async fn session_family_work_count_includes_an_unmaterialized_linked_task() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harness");
    std::fs::create_dir_all(&harness_dir)?;
    let kernel = Kernel::builder(test_config(tmp.path(), &harness_dir)).build()?;
    let manager = kernel.agent_manager();
    let store_selector = kernel.config().persistence.top_level_state_selector()?;
    manager.pending_task_states.write().await.insert(
        "req-unmaterialized-child".to_string(),
        PendingTaskRecord {
            runtime_key: RuntimeSlotKey {
                agent_id: "worker".to_string(),
                slot_id: "linked_0".to_string(),
            },
            session_target: TaskSessionTarget {
                session_id: None,
                store_selector: Some(store_selector.clone()),
                linked_parent_session_id: Some(42),
                thread_key: Some("child".to_string()),
                reserves_new_child: true,
            },
            trace_id: "trace-unmaterialized-child".to_string(),
            title: None,
            prompt_preview: "queued child".to_string(),
            state: PendingTaskState::Queued,
            runtime_task_id: None,
            execution: test_execution_snapshot(),
        },
    );

    let count = manager
        .session_family_work_count(
            &["parent-session".to_string()],
            &std::collections::HashSet::from([(store_selector, 42)]),
        )
        .await;

    assert_eq!(count, 1);
    Ok(())
}

#[tokio::test]
async fn linked_submission_admission_bounds_outstanding_children_and_fan_out() -> anyhow::Result<()>
{
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harness");
    std::fs::create_dir_all(&harness_dir)?;
    let kernel = Kernel::builder(test_config(tmp.path(), &harness_dir)).build()?;
    let manager = kernel.agent_manager();
    let runtime_key = RuntimeSlotKey {
        agent_id: "worker".to_string(),
        slot_id: "linked_0".to_string(),
    };
    let handle = Arc::new(AgentRuntimeHandle {
        queue: Arc::new(Mutex::new(VecDeque::new())),
        notify: Arc::new(Notify::new()),
        control: Arc::new(RuntimeControl::default()),
        shutdown_token: CancellationToken::new(),
        task: None,
        queued_tasks: Arc::new(AtomicUsize::new(0)),
        active_tasks: Arc::new(AtomicUsize::new(0)),
    });
    manager
        .runtimes
        .write()
        .await
        .insert(runtime_key.clone(), Arc::clone(&handle));
    let store_selector = crate::persistence::manager::StoreSelector::Alias("state".to_string());
    let target = |thread: &str| TaskSessionTarget {
        session_id: None,
        store_selector: Some(store_selector.clone()),
        linked_parent_session_id: Some(42),
        thread_key: Some(thread.to_string()),
        reserves_new_child: true,
    };
    let admission = DelegationAdmission {
        parent_session_id: 42,
        persisted_direct_children: 0,
        max_fan_out: 2,
        max_concurrent_children: 1,
    };
    let first = manager
        .submit_to_handle(
            runtime_key.clone(),
            Arc::clone(&handle),
            QueuedTask::ad_hoc("first"),
            PeerTaskSubmission {
                delegated_capabilities: None,
                promotion_candidate: None,
                linked_session: None,
                session_target: target("first"),
                delegation_admission: Some(admission),
            },
        )
        .await?;
    let error = manager
        .submit_to_handle(
            runtime_key.clone(),
            Arc::clone(&handle),
            QueuedTask::ad_hoc("second"),
            PeerTaskSubmission {
                delegated_capabilities: None,
                promotion_candidate: None,
                linked_session: None,
                session_target: target("second"),
                delegation_admission: Some(admission),
            },
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("max_concurrent_children"));
    manager.cancel_task(&first).await?;

    let error = manager
        .submit_to_handle(
            runtime_key,
            handle,
            QueuedTask::ad_hoc("fan-out"),
            PeerTaskSubmission {
                delegated_capabilities: None,
                promotion_candidate: None,
                linked_session: None,
                session_target: target("third"),
                delegation_admission: Some(DelegationAdmission {
                    persisted_direct_children: 2,
                    max_concurrent_children: 4,
                    ..admission
                }),
            },
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("max_fan_out"));
    assert!(manager.pending_results.read().await.is_empty());
    Ok(())
}

#[tokio::test]
async fn stopped_runtime_rejects_submission_without_pending_state() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harness");
    std::fs::create_dir_all(&harness_dir)?;
    let kernel = Kernel::builder(test_config(tmp.path(), &harness_dir)).build()?;
    let manager = kernel.agent_manager();
    let shutdown_token = CancellationToken::new();
    shutdown_token.cancel();
    let handle = Arc::new(AgentRuntimeHandle {
        queue: Arc::new(Mutex::new(VecDeque::new())),
        notify: Arc::new(Notify::new()),
        control: Arc::new(RuntimeControl::default()),
        shutdown_token,
        task: None,
        queued_tasks: Arc::new(AtomicUsize::new(0)),
        active_tasks: Arc::new(AtomicUsize::new(0)),
    });

    let error = manager
        .submit_to_handle(
            RuntimeSlotKey::default_for("default"),
            Arc::clone(&handle),
            QueuedTask::ad_hoc("must not be stranded"),
            PeerTaskSubmission {
                delegated_capabilities: None,
                promotion_candidate: None,
                linked_session: None,
                session_target: TaskSessionTarget::default(),
                delegation_admission: None,
            },
        )
        .await
        .expect_err("stopped runtime should reject submission");

    assert!(error.to_string().contains("stopped before task submission"));
    assert!(
        handle
            .queue
            .lock()
            .expect("runtime queue mutex poisoned")
            .is_empty()
    );
    assert!(manager.pending_task_states.read().await.is_empty());
    assert!(manager.pending_results.read().await.is_empty());
    Ok(())
}

#[tokio::test]
async fn linked_submission_depth_uses_persisted_session_ancestry() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harness");
    std::fs::create_dir_all(&harness_dir)?;
    let mut config = test_config(tmp.path(), &harness_dir);
    config.persistence = PersistenceConfig::with_state_path(
        tmp.path().join("state.db").to_string_lossy().to_string(),
    );
    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    let store = kernel.store_manager().get_default().await?;
    let root = store
        .create_session(uuid::Uuid::now_v7(), "default", None)
        .await?;
    let child_public_id = uuid::Uuid::now_v7();
    store
        .create_linked_session(
            child_public_id,
            "reviewer",
            None,
            &crate::persistence::schema::LinkedSessionCreate {
                parent_session_id: root,
                origin_turn_id: None,
                relation_kind: "delegated".to_string(),
                thread_key: "review".to_string(),
                visibility: "contextual".to_string(),
            },
        )
        .await?;
    kernel
        .policy_manager()
        .set(
            "spawn.max_depth",
            serde_json::Value::from(1u64),
            &crate::kernel::policy::PolicyScope::default(),
        )
        .await?;

    let error = kernel
        .agent_manager()
        .submit_linked(
            &child_public_id.simple().to_string(),
            None,
            "worker",
            LinkedSessionMode::Thread("too-deep".to_string()),
            QueuedTask::ad_hoc("reject"),
            None,
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("spawn.max_depth=1"));
    Ok(())
}

#[tokio::test]
async fn recursive_cancellation_includes_materialized_and_queued_descendants() -> anyhow::Result<()>
{
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harness");
    std::fs::create_dir_all(&harness_dir)?;
    let mut config = test_config(tmp.path(), &harness_dir);
    config.persistence = PersistenceConfig::with_state_path(
        tmp.path().join("state.db").to_string_lossy().to_string(),
    );
    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    let manager = kernel.agent_manager();
    let store = kernel.store_manager().get_default().await?;
    let root_public_id = uuid::Uuid::now_v7();
    let root = store
        .create_session(root_public_id, "default", None)
        .await?;
    let child_public_id = uuid::Uuid::now_v7();
    let child = store
        .create_linked_session(
            child_public_id,
            "worker",
            None,
            &crate::persistence::schema::LinkedSessionCreate {
                parent_session_id: root,
                origin_turn_id: None,
                relation_kind: "delegated".to_string(),
                thread_key: "child".to_string(),
                visibility: "contextual".to_string(),
            },
        )
        .await?;
    let runtime_key = RuntimeSlotKey {
        agent_id: "worker".to_string(),
        slot_id: "linked_0".to_string(),
    };
    let store_selector = kernel.config().persistence.top_level_state_selector()?;
    let child_session_id = child_public_id.simple().to_string();
    let materialized_request = "req-materialized-child".to_string();
    let queued_request = "req-queued-grandchild".to_string();
    let targets = [
        (
            materialized_request.clone(),
            TaskSessionTarget {
                session_id: Some(child_session_id.clone()),
                store_selector: Some(store_selector.clone()),
                linked_parent_session_id: Some(root),
                thread_key: Some("child".to_string()),
                reserves_new_child: false,
            },
        ),
        (
            queued_request.clone(),
            TaskSessionTarget {
                session_id: None,
                store_selector: Some(store_selector),
                linked_parent_session_id: Some(child),
                thread_key: Some("grandchild".to_string()),
                reserves_new_child: true,
            },
        ),
    ];
    let mut queue = VecDeque::new();
    for (request_id, target) in targets {
        manager.pending_task_states.write().await.insert(
            request_id.clone(),
            PendingTaskRecord {
                runtime_key: runtime_key.clone(),
                session_target: target.clone(),
                trace_id: format!("trace-{request_id}"),
                title: None,
                prompt_preview: request_id.clone(),
                state: PendingTaskState::Queued,
                runtime_task_id: None,
                execution: test_execution_snapshot(),
            },
        );
        queue.push_back(PeerAgentTaskEnvelope {
            task: QueuedTask::ad_hoc(request_id.clone()),
            request_id: Some(request_id),
            result_tx: None,
            delegated_capabilities: None,
            promotion_candidate: None,
            linked_session: None,
            session_target: target,
        });
    }
    let handle = Arc::new(AgentRuntimeHandle {
        queue: Arc::new(Mutex::new(queue)),
        notify: Arc::new(Notify::new()),
        control: Arc::new(RuntimeControl::default()),
        shutdown_token: CancellationToken::new(),
        task: None,
        queued_tasks: Arc::new(AtomicUsize::new(2)),
        active_tasks: Arc::new(AtomicUsize::new(0)),
    });
    manager
        .runtimes
        .write()
        .await
        .insert(runtime_key, Arc::clone(&handle));

    let (_, returned_session_id, affected_tasks) = manager
        .cancel_session_family(&root_public_id.simple().to_string())
        .await?;

    assert_eq!(returned_session_id, root_public_id.simple().to_string());
    assert_eq!(affected_tasks, 2);
    assert_eq!(handle.queued_tasks.load(Ordering::Relaxed), 0);
    assert_eq!(
        manager
            .get_task(&materialized_request)
            .await
            .and_then(|task| task.status),
        Some(TaskTerminalStatus::Cancelled)
    );
    assert_eq!(
        manager
            .get_task(&queued_request)
            .await
            .and_then(|task| task.status),
        Some(TaskTerminalStatus::Cancelled)
    );
    Ok(())
}

#[tokio::test]
async fn resume_session_restarts_dead_requested_slot() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harness");
    std::fs::create_dir_all(&harness_dir)?;

    let kernel = Kernel::builder(test_config(tmp.path(), &harness_dir)).build()?;
    let manager = kernel.agent_manager();
    let slot_id = "chan-stale";
    let opened = manager
        .open_session(
            "default",
            Some(slot_id),
            None,
            None,
            Some("telegram-ops".to_string()),
            InferenceOverrideConfig::default(),
        )
        .await?;
    let session_id = opened.session_id.clone();

    manager.kill_session(&session_id, Some(slot_id)).await?;

    let control = Arc::new(RuntimeControl::default());
    control.set_current_session_id(Some(session_id.clone()));

    manager.runtimes.write().await.insert(
        RuntimeSlotKey {
            agent_id: "default".to_string(),
            slot_id: slot_id.to_string(),
        },
        Arc::new(AgentRuntimeHandle {
            queue: Arc::new(Mutex::new(VecDeque::new())),
            notify: Arc::new(Notify::new()),
            control,
            shutdown_token: CancellationToken::new(),
            task: None,
            queued_tasks: Arc::new(AtomicUsize::new(0)),
            active_tasks: Arc::new(AtomicUsize::new(0)),
        }),
    );

    let resumed = manager
        .resume_session(
            &session_id,
            Some(slot_id),
            Some("telegram-ops".to_string()),
            InferenceOverrideConfig::default(),
        )
        .await?;

    assert_eq!(resumed.slot_id, slot_id);
    assert_eq!(resumed.session_id, session_id);
    assert_eq!(resumed.agent_id, "default");

    Ok(())
}

#[tokio::test]
async fn failed_runtime_bootstrap_is_not_published() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harness");
    std::fs::create_dir_all(&harness_dir)?;

    let mut kernel = Kernel::builder(test_config(tmp.path(), &harness_dir)).build()?;
    kernel.init_state().await?;
    let manager = Arc::clone(kernel.agent_manager());
    let runtime_key = RuntimeSlotKey {
        agent_id: "default".to_string(),
        slot_id: "failed-bootstrap".to_string(),
    };
    let missing_session = format!("{}@state", uuid::Uuid::now_v7().simple());

    let result = manager
        .ensure_runtime_slot_resumed(
            runtime_key.clone(),
            missing_session,
            SessionContextOverrides::default(),
        )
        .await;

    let error = match result {
        Ok(_) => anyhow::bail!("missing session unexpectedly started a runtime"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("failed to start"));
    assert!(!manager.runtimes.read().await.contains_key(&runtime_key));
    assert!(manager.pending_task_states.read().await.is_empty());
    Ok(())
}

#[tokio::test]
async fn resume_session_accepts_bare_id_for_path_backed_state_store() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harness");
    std::fs::create_dir_all(&harness_dir)?;

    let mut kernel = Kernel::builder(test_config(tmp.path(), &harness_dir)).build()?;
    kernel.init_state().await?;
    let manager = kernel.agent_manager();
    let opened = manager
        .open_session(
            "default",
            Some("original"),
            None,
            None,
            None,
            InferenceOverrideConfig::default(),
        )
        .await?;
    let bare_session_id = parse_session_reference(&opened.session_id)?.public_id;

    manager
        .kill_session(&opened.session_id, Some("original"))
        .await?;
    let resumed = manager
        .resume_session(
            &bare_session_id,
            Some("resumed"),
            None,
            InferenceOverrideConfig::default(),
        )
        .await?;

    assert_eq!(resumed.slot_id, "resumed");
    assert!(session_references_match(
        &resumed.session_id,
        &bare_session_id
    ));
    Ok(())
}

#[tokio::test]
async fn explicit_runtime_slots_allow_multiple_live_runtimes_for_one_session() -> anyhow::Result<()>
{
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harness");
    std::fs::create_dir_all(&harness_dir)?;

    let kernel = Kernel::builder(test_config(tmp.path(), &harness_dir)).build()?;
    let manager = kernel.agent_manager();

    let slot_a = manager
        .open_session(
            "default",
            Some("slot-a"),
            None,
            None,
            None,
            InferenceOverrideConfig::default(),
        )
        .await?;
    let slot_b = manager
        .resume_session(
            &slot_a.session_id,
            Some("slot-b"),
            None,
            InferenceOverrideConfig::default(),
        )
        .await?;

    let live = manager.list_live_sessions(None).await;
    let matching: Vec<_> = live
        .into_iter()
        .filter(|snapshot| snapshot.session_id == slot_a.session_id)
        .collect();
    assert_eq!(matching.len(), 2);
    assert!(matching.iter().any(|snapshot| snapshot.slot_id == "slot-a"));
    assert!(matching.iter().any(|snapshot| snapshot.slot_id == "slot-b"));
    assert!(
        matching
            .iter()
            .all(|snapshot| snapshot.conflict_policy == ExecutionConflictPolicy::Reject)
    );
    assert_eq!(slot_b.slot_id, "slot-b");

    let resume_err = manager
        .resume_session(
            &slot_a.session_id,
            None,
            None,
            InferenceOverrideConfig::default(),
        )
        .await
        .expect_err("slot-agnostic resume should reject ambiguity");
    assert!(resume_err.to_string().contains("multiple runtime slots"));

    let submit_err = manager
        .submit_to_session(
            &slot_a.session_id,
            None,
            QueuedTask::ad_hoc("ambiguous".to_string()),
            None,
        )
        .await
        .expect_err("slot-agnostic submit should reject ambiguity");
    assert!(submit_err.to_string().contains("multiple runtime slots"));

    let reload_err = manager
        .reload_session(&slot_a.session_id, None)
        .await
        .expect_err("slot-agnostic reload should reject ambiguity");
    assert!(reload_err.to_string().contains("multiple runtime slots"));

    let reloaded = manager
        .reload_session(&slot_a.session_id, Some("slot-b"))
        .await?;
    assert_eq!(reloaded.slot_id, "slot-b");

    let reload_if_live_err = manager
        .reload_session_if_live(&slot_a.session_id, None)
        .await
        .expect_err("slot-agnostic conditional reload should reject ambiguity");
    assert!(
        reload_if_live_err
            .to_string()
            .contains("multiple runtime slots")
    );

    assert!(
        manager
            .subscribe_session_events(&slot_a.session_id, None)
            .await
            .is_none(),
        "slot-agnostic event subscription should reject ambiguity"
    );

    let targeted = manager
        .submit_to_session(
            &slot_a.session_id,
            Some("slot-b"),
            QueuedTask::ad_hoc("targeted".to_string()),
            None,
        )
        .await?;
    let targeted_task = manager
        .get_task(&targeted)
        .await
        .expect("targeted task should be visible");
    assert_eq!(targeted_task.slot_id, "slot-b");

    let (_, subscribed_slot_id, _receiver) = manager
        .subscribe_session_events(&slot_a.session_id, Some("slot-b"))
        .await
        .expect("slot-targeted event subscription should resolve");
    assert_eq!(subscribed_slot_id, "slot-b");

    let cancel_err = manager
        .cancel_session(&slot_a.session_id, None)
        .await
        .expect_err("slot-agnostic cancel should reject ambiguity");
    assert!(cancel_err.to_string().contains("multiple runtime slots"));

    let (killed_agent_id, killed_slot_id, killed_session_id) = manager
        .kill_session(&slot_a.session_id, Some("slot-b"))
        .await?;
    assert_eq!(killed_agent_id, "default");
    assert_eq!(killed_slot_id, "slot-b");
    assert_eq!(killed_session_id, slot_a.session_id);

    let live_after_targeted_kill = manager.list_live_sessions(None).await;
    assert_eq!(live_after_targeted_kill.len(), 1);
    assert_eq!(live_after_targeted_kill[0].slot_id, "slot-a");

    let (final_killed_agent_id, final_killed_slot_id, final_killed_session_id) =
        manager.kill_session(&slot_a.session_id, None).await?;
    assert_eq!(final_killed_agent_id, "default");
    assert_eq!(final_killed_slot_id, "slot-a");
    assert_eq!(final_killed_session_id, slot_a.session_id);

    abort_all_runtime_slots(manager).await;
    Ok(())
}

#[tokio::test]
async fn live_session_snapshots_expose_effective_conflict_policy() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harness");
    std::fs::create_dir_all(&harness_dir)?;

    let kernel = Kernel::builder(test_config(tmp.path(), &harness_dir)).build()?;
    let manager = kernel.agent_manager();
    let session_id = "s_live_conflict";
    let control = Arc::new(RuntimeControl::default());
    control.set_current_session_id(Some(session_id.to_string()));
    control.set_current_execution_snapshot(ExecutionStatusSnapshot {
        execution_id: "ex_live_conflict".to_string(),
        context_target: crate::kernel::session::ExecutionContextTarget::BranchHead {
            branch_head_id: Some(7),
        },
        visibility: crate::kernel::session::ExecutionVisibility::Visible,
        durability: crate::kernel::session::ExecutionDurability::Durable,
        write_policy: crate::kernel::session::ExecutionWritePolicy::AdvanceBranchHead,
    });
    control.set_current_execution_conflict_policy(ExecutionConflictPolicy::Detached);
    control.set_current_history_snapshot(LiveSessionHistorySnapshot {
        len: 64,
        has_prior_history: true,
    });

    manager.runtimes.write().await.insert(
        RuntimeSlotKey::default_for("default"),
        Arc::new(AgentRuntimeHandle {
            queue: Arc::new(Mutex::new(VecDeque::new())),
            notify: Arc::new(Notify::new()),
            control,
            shutdown_token: CancellationToken::new(),
            task: None,
            queued_tasks: Arc::new(AtomicUsize::new(0)),
            active_tasks: Arc::new(AtomicUsize::new(1)),
        }),
    );

    let live = manager.list_live_sessions(None).await;
    assert_eq!(live.len(), 1);
    assert_eq!(live[0].session_id, session_id);
    assert_eq!(live[0].execution.execution_id, "ex_live_conflict");
    assert_eq!(live[0].conflict_policy, ExecutionConflictPolicy::Detached);
    let history = live[0].history.as_ref().expect("history snapshot");
    assert_eq!(history.len, 64);
    assert!(history.has_prior_history);

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runtime_signals_can_wake_subscribed_agent_and_dispatch_to_worklist() -> anyhow::Result<()>
{
    let tmp = tempdir()?;
    let publisher_harness = tmp.path().join("publisher-harness");
    let reviewer_harness = tmp.path().join("reviewer-harness");
    std::fs::create_dir_all(&publisher_harness)?;
    std::fs::create_dir_all(&reviewer_harness)?;

    std::fs::write(
        publisher_harness.join("main.lua"),
        r#"
            action.define("signals.publish", function(ctx, params)
                return {
                    delivered = runtime.emit(params.topic, params.payload)
                }
            end)
        "#,
    )?;
    std::fs::write(
        reviewer_harness.join("main.lua"),
        r#"
            runtime.on("code.ready", function(ready, _meta)
                worklist("reviews"):add({
                    title = "Review " .. ready.branch,
                    prompt = "Review " .. ready.branch
                })
            end)
        "#,
    )?;

    let mut kernel = Kernel::builder(signal_test_config(
        tmp.path(),
        &publisher_harness,
        &reviewer_harness,
    ))
    .build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    let runtime_store = Arc::new(StateStore::open_memory().await?);
    kernel.host.scheduler = Some(Arc::new(HarnessSchedulerAccess::new(
        Arc::clone(&runtime_store),
        None,
    )));
    kernel
        .agent_manager()
        .bind_scheduler_access(kernel.host.scheduler.clone());
    kernel.init_harness().await?;

    let instance = kernel
        .harness_definition_for_agent("default")
        .create_instance(kernel.harness_init_context())?;
    let result = instance.invoke_action(crate::kernel::harness_contract::HarnessActionRequest {
        agent_id: "default",
        name: "signals.publish",
        params: json!({
            "topic": "code.ready",
            "payload": { "branch": "feature-x" }
        }),
    })?;

    assert_eq!(
        result.as_ref().and_then(|value| value.get("delivered")),
        Some(&json!(1))
    );

    let store = kernel
        .store_manager()
        .open(&crate::persistence::manager::StoreSelector::Alias(
            "state".to_string(),
        ))
        .await?;

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let list = store.open_worklist("reviews", "", None).await?;
        let rows = store.list_work_items(list.id).await?;
        if rows.iter().any(|row| row.title == "Review feature-x") {
            break;
        }

        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!("timed out waiting for runtime signal delivery");
        }

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    abort_all_runtime_slots(kernel.agent_manager()).await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runtime_signals_hydrate_reference_payloads_in_subscribed_agent() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let publisher_harness = tmp.path().join("publisher-harness");
    let reviewer_harness = tmp.path().join("reviewer-harness");
    std::fs::create_dir_all(&publisher_harness)?;
    std::fs::create_dir_all(&reviewer_harness)?;

    std::fs::write(
        publisher_harness.join("main.lua"),
        r#"
            action.define("signals.publish_ref", function(this, params)
                local project = scope("project", "alpha")
                project:set("status", "ready")

                local tickets = worklist("tickets")
                local item = tickets:add({
                    title = "Classify checkout",
                    action = "ticket.classify",
                    params = {
                        project = ref(project),
                    },
                })

                return {
                    item_id = item.id,
                    delivered = runtime.emit("ticket.ready", {
                        project = ref(project),
                        item = ref(item),
                    })
                }
            end)
        "#,
    )?;
    std::fs::write(
        reviewer_harness.join("main.lua"),
        r#"
            runtime.on("ticket.ready", function(ready, meta)
                if ready.project:get("status") ~= "ready" then
                    error("expected project ref to hydrate")
                end
                if ready.item.title ~= "Classify checkout" then
                    error("expected work item ref to hydrate")
                end
                if type(ready.item.done) ~= "function" then
                    error("expected hydrated work item methods")
                end

                ready.item:done({
                    reviewer = meta.target_agent_id,
                    project_status = ready.project:get("status"),
                })

                worklist("reviews"):add({
                    title = "Reviewed " .. ready.item.title,
                    prompt = "Reviewed " .. ready.item.title,
                    metadata = {
                        item_id = ready.item.id,
                        source = meta.source_agent_id,
                    },
                })
            end)
        "#,
    )?;

    let mut kernel = Kernel::builder(signal_test_config(
        tmp.path(),
        &publisher_harness,
        &reviewer_harness,
    ))
    .build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    let runtime_store = Arc::new(StateStore::open_memory().await?);
    kernel.host.scheduler = Some(Arc::new(HarnessSchedulerAccess::new(
        Arc::clone(&runtime_store),
        None,
    )));
    kernel
        .agent_manager()
        .bind_scheduler_access(kernel.host.scheduler.clone());
    kernel.init_harness().await?;

    let instance = kernel
        .harness_definition_for_agent("default")
        .create_instance(kernel.harness_init_context())?;
    let result = instance.invoke_action(crate::kernel::harness_contract::HarnessActionRequest {
        agent_id: "default",
        name: "signals.publish_ref",
        params: json!({}),
    })?;

    assert_eq!(
        result.as_ref().and_then(|value| value.get("delivered")),
        Some(&json!(1))
    );

    let store = kernel
        .store_manager()
        .open(&crate::persistence::manager::StoreSelector::Alias(
            "state".to_string(),
        ))
        .await?;

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        let tickets = store.open_worklist("tickets", "", None).await?;
        let ticket_rows = store.list_work_items(tickets.id).await?;
        let ticket_done = ticket_rows
            .iter()
            .any(|row| row.title == "Classify checkout" && row.status == "done");

        let reviews = store.open_worklist("reviews", "", None).await?;
        let review_rows = store.list_work_items(reviews.id).await?;
        let review_created = review_rows
            .iter()
            .any(|row| row.title == "Reviewed Classify checkout");

        if ticket_done && review_created {
            break;
        }

        if tokio::time::Instant::now() >= deadline {
            anyhow::bail!("timed out waiting for hydrated runtime signal delivery");
        }

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    abort_all_runtime_slots(kernel.agent_manager()).await;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runtime_signal_subscriptions_sync_on_harness_reload() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let publisher_harness = tmp.path().join("publisher-harness");
    let reviewer_harness = tmp.path().join("reviewer-harness");
    std::fs::create_dir_all(&publisher_harness)?;
    std::fs::create_dir_all(&reviewer_harness)?;

    std::fs::write(
        publisher_harness.join("main.lua"),
        r#"
            action.define("signals.publish", function(ctx, params)
                return {
                    delivered = runtime.emit(params.topic, params.payload)
                }
            end)
        "#,
    )?;
    std::fs::write(
        reviewer_harness.join("main.lua"),
        r#"
            runtime.on("code.ready", function(_data, _meta) end)
        "#,
    )?;

    let mut kernel = Kernel::builder(signal_test_config(
        tmp.path(),
        &publisher_harness,
        &reviewer_harness,
    ))
    .build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    let runtime_store = Arc::new(StateStore::open_memory().await?);
    kernel.host.scheduler = Some(Arc::new(HarnessSchedulerAccess::new(
        Arc::clone(&runtime_store),
        None,
    )));
    kernel
        .agent_manager()
        .bind_scheduler_access(kernel.host.scheduler.clone());
    kernel.init_harness().await?;

    assert_eq!(
        runtime_store
            .list_signal_subscriber_agent_ids("code.ready")
            .await?,
        vec!["reviewer".to_string()]
    );
    assert!(
        runtime_store
            .list_signal_subscriber_agent_ids("review.ready")
            .await?
            .is_empty()
    );

    std::fs::write(
        reviewer_harness.join("main.lua"),
        r#"
            runtime.on("review.ready", function(_data, _meta) end)
        "#,
    )?;

    kernel.reload_named_harness("reviewer").await?;

    assert!(
        runtime_store
            .list_signal_subscriber_agent_ids("code.ready")
            .await?
            .is_empty()
    );
    assert_eq!(
        runtime_store
            .list_signal_subscriber_agent_ids("review.ready")
            .await?,
        vec!["reviewer".to_string()]
    );

    abort_all_runtime_slots(kernel.agent_manager()).await;
    Ok(())
}
