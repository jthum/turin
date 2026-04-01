use super::*;
use crate::kernel::Kernel;
use crate::kernel::config::{
    AgentConfig, EmbeddingConfig, GovernanceConfig, HarnessConfig, KernelConfig, PersistenceConfig,
    ProviderConfig, TurinConfig,
};
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
            mode: crate::kernel::config::AgentMode::Auto,
            harness: None,
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
            ..PersistenceConfig::default()
        },
        harness: HarnessConfig {
            directory: harness_dir.to_string_lossy().to_string(),
            fs_root: ".".to_string(),
        },
        harnesses: HashMap::new(),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    }
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

    let (agent_id, returned_session_id) = manager.cancel_session(session_id).await?;
    assert_eq!(agent_id, "default");
    assert_eq!(returned_session_id, session_id);
    assert!(matches!(
        control.take_session_reset_request(),
        Some(SessionResetRequest::Fresh)
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
        },
    );
    manager.pending_task_states.write().await.insert(
        queued_request_id.clone(),
        PendingTaskRecord {
            runtime_key: RuntimeSlotKey::default_for("default"),
            trace_id: "tr_queued_kill".to_string(),
            state: PendingTaskState::Queued,
            runtime_task_id: None,
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

    let (agent_id, returned_session_id) = manager.kill_session(session_id).await?;
    assert_eq!(agent_id, "default");
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
