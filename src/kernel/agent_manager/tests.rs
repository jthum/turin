use super::*;
use crate::harness::scheduler::HarnessSchedulerAccess;
use crate::kernel::Kernel;
use crate::kernel::config::{
    AgentConfig, EmbeddingConfig, GovernanceConfig, HarnessConfig, InferenceConfig, KernelConfig,
    LayoutConfig, PersistenceConfig, ProviderConfig, TurinConfig,
};
use crate::persistence::state::StateStore;
use crate::tools::{Tool, ToolContext, ToolEffect, ToolError};
use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;
use tempfile::tempdir;

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
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: HashMap::new(),
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

fn test_execution_snapshot() -> ExecutionStatusSnapshot {
    ExecutionStatusSnapshot::from_execution(
        &crate::kernel::session::ExecutionContext::new(),
        crate::kernel::session::ExecutionWritePolicy::AdvanceBranchHead,
    )
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

    let peer_kernel = super::peer_runtime::fork_peer_kernel(&kernel.agent_manager);

    assert_eq!(peer_kernel.tool_registry.len(), registry.len());
    assert!(peer_kernel.tool_registry.get("test_tool").is_some());
    assert!(Arc::ptr_eq(
        &kernel.persistence_locks,
        &peer_kernel.persistence_locks
    ));

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
            trace_id: "tr_cancelled".to_string(),
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
    });

    manager.runtimes.write().await.insert(
        RuntimeSlotKey::default_for("default"),
        Arc::new(AgentRuntimeHandle {
            queue: Arc::new(Mutex::new(queue)),
            notify: Arc::new(Notify::new()),
            control: Arc::new(RuntimeControl::default()),
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
            trace_id: "tr_running".to_string(),
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
            trace_id: "tr_session_cancel".to_string(),
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
    });

    manager.runtimes.write().await.insert(
        RuntimeSlotKey::default_for("default"),
        Arc::new(AgentRuntimeHandle {
            queue: Arc::new(Mutex::new(queue)),
            notify: Arc::new(Notify::new()),
            control: Arc::clone(&control),
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
            trace_id: "tr_running_kill".to_string(),
            state: PendingTaskState::Running,
            runtime_task_id: Some("t_running".to_string()),
            execution: test_execution_snapshot(),
        },
    );
    manager.pending_task_states.write().await.insert(
        queued_request_id.clone(),
        PendingTaskRecord {
            runtime_key: RuntimeSlotKey::default_for("default"),
            trace_id: "tr_queued_kill".to_string(),
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
    });

    manager.runtimes.write().await.insert(
        RuntimeSlotKey::default_for("default"),
        Arc::new(AgentRuntimeHandle {
            queue: Arc::new(Mutex::new(queue)),
            notify: Arc::new(Notify::new()),
            control,
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

    manager.runtimes.write().await.insert(
        RuntimeSlotKey::default_for("default"),
        Arc::new(AgentRuntimeHandle {
            queue: Arc::new(Mutex::new(VecDeque::new())),
            notify: Arc::new(Notify::new()),
            control,
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
        .runtime_for_agent("default")
        .create_instance(kernel.harness_init_context())?;
    let result = instance.invoke_declared_action_for_agent(
        "default",
        "signals.publish",
        json!({
            "topic": "code.ready",
            "payload": { "branch": "feature-x" }
        }),
    )?;

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
        .runtime_for_agent("default")
        .create_instance(kernel.harness_init_context())?;
    let result =
        instance.invoke_declared_action_for_agent("default", "signals.publish_ref", json!({}))?;

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
