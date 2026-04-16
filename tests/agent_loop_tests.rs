use anyhow::Result;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream;
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::tempdir;
use turin::inference::provider::{
    InferenceEvent, InferenceProvider, InferenceRequest, InferenceStream, ProviderClient,
    RequestOptions, SdkError,
};
use turin::kernel::Kernel;
use turin::kernel::config::{
    AgentConfig, EmbeddingConfig, HarnessConfig, InferenceConfig, PersistenceConfig,
    ProviderConfig, TurinConfig,
};
use turin::kernel::event::{AuditEvent, KernelEvent, LifecycleEvent, StreamEvent};
use turin::kernel::session::{ExecutionConflictPolicy, QueuedTask};

/// A mock provider that returns a text response followed by a tool call in the next turn.
struct SequenceMockProvider {
    responses: Arc<std::sync::Mutex<Vec<Vec<InferenceEvent>>>>,
}

impl InferenceProvider for SequenceMockProvider {
    fn stream<'a>(
        &'a self,
        _request: InferenceRequest,
        _options: Option<RequestOptions>,
    ) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        let responses = self.responses.clone();
        Box::pin(async move {
            let mut guard = responses.lock().unwrap();
            let events = if !guard.is_empty() {
                guard.remove(0).into_iter().map(Ok).collect()
            } else {
                vec![
                    Ok(InferenceEvent::MessageStart {
                        role: "assistant".to_string(),
                        model: "mock-model".to_string(),
                        provider_id: "mock".to_string(),
                    }),
                    Ok(InferenceEvent::MessageDelta {
                        content: "Finishing.".to_string(),
                    }),
                    Ok(InferenceEvent::MessageEnd {
                        input_tokens: 1,
                        output_tokens: 1,
                        stop_reason: None,
                    }),
                ]
            };
            let stream = stream::iter(events).then(|event| async move {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                event
            });
            Ok(Box::pin(stream) as InferenceStream)
        })
    }
}

struct FailThenRecoverProvider {
    should_fail: Arc<std::sync::Mutex<bool>>,
}

impl InferenceProvider for FailThenRecoverProvider {
    fn stream<'a>(
        &'a self,
        _request: InferenceRequest,
        _options: Option<RequestOptions>,
    ) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        let should_fail = self.should_fail.clone();
        Box::pin(async move {
            let mut fail_flag = should_fail.lock().unwrap();
            if *fail_flag {
                *fail_flag = false;
                let events = vec![Err(SdkError::ProviderError(
                    "simulated failure".to_string(),
                ))];
                return Ok(Box::pin(stream::iter(events)) as InferenceStream);
            }

            let events = vec![
                Ok(InferenceEvent::MessageStart {
                    role: "assistant".to_string(),
                    model: "mock-model".to_string(),
                    provider_id: "mock".to_string(),
                }),
                Ok(InferenceEvent::MessageDelta {
                    content: "Recovered".to_string(),
                }),
                Ok(InferenceEvent::MessageEnd {
                    input_tokens: 1,
                    output_tokens: 1,
                    stop_reason: None,
                }),
            ];
            Ok(Box::pin(stream::iter(events)) as InferenceStream)
        })
    }
}

#[tokio::test]
async fn test_agent_loop_event_sequence() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_events.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: None,
            ..ProviderConfig::default()
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Test".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: std::collections::HashMap::new(),
        kernel: turin::kernel::config::KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 5,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        layout: Default::default(),
        inference: InferenceConfig::default(),
        persistence: PersistenceConfig::with_state_path(db_path.to_str().unwrap().to_string()),
        harness: HarnessConfig {
            directory: harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::new(),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: turin::kernel::config::GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;

    // Setup multi-turn sequence:
    // Turn 1: Tool Call
    // Turn 2: Final response
    let responses = vec![
        vec![
            InferenceEvent::MessageStart {
                role: "assistant".to_string(),
                model: "mock-model".to_string(),
                provider_id: "mock".to_string(),
            },
            InferenceEvent::ToolCallStart {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
            },
            InferenceEvent::ToolCallDelta {
                delta: serde_json::json!({"path": "test.txt"}).to_string(),
            },
            InferenceEvent::MessageEnd {
                input_tokens: 10,
                output_tokens: 5,
                stop_reason: None,
            },
        ],
        vec![
            InferenceEvent::MessageStart {
                role: "assistant".to_string(),
                model: "mock-model".to_string(),
                provider_id: "mock".to_string(),
            },
            InferenceEvent::MessageDelta {
                content: "I read it.".to_string(),
            },
            InferenceEvent::MessageEnd {
                input_tokens: 5,
                output_tokens: 2,
                stop_reason: None,
            },
        ],
    ];

    let mock_provider = Arc::new(SequenceMockProvider {
        responses: Arc::new(std::sync::Mutex::new(responses)),
    });
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new("mock", mock_provider),
    );

    let mut session = kernel.create_session().await;

    // Capture events from the session broadcast
    let mut rx = session.event_tx.subscribe();

    kernel.run(&mut session, Some("Hello".to_string())).await?;

    let mut events = Vec::new();
    // Gather all events emitted so far
    while let Ok(event) = rx.try_recv() {
        events.push(event.1);
    }

    // --- Assertions on sequence and types ---

    // 1. Session Lifecycle
    assert!(matches!(
        events[0],
        KernelEvent::Lifecycle(LifecycleEvent::SessionStart { .. })
    ));

    // 2. First Turn
    assert!(events.iter().any(|e| matches!(
        e,
        KernelEvent::Lifecycle(LifecycleEvent::TurnStart { turn_index: 0, .. })
    )));
    assert!(events.iter().any(|e| matches!(e, KernelEvent::Stream(StreamEvent::ToolCall { name, .. }) if name == "read_file")));
    assert!(events.iter().any(|e| matches!(
        e,
        KernelEvent::Lifecycle(LifecycleEvent::TurnEnd {
            turn_index: 0,
            has_tool_calls: true,
            ..
        })
    )));

    // 3. Tool Audit Events
    assert!(events.iter().any(|e| matches!(e, KernelEvent::Audit(AuditEvent::ToolExecStart { name, .. }) if name == "read_file")));
    assert!(
        events
            .iter()
            .any(|e| matches!(e, KernelEvent::Audit(AuditEvent::ToolResult { .. })))
    );
    assert!(events.iter().any(|e| matches!(
        e,
        KernelEvent::Audit(AuditEvent::ToolExecEnd { success: false, .. })
    ))); // read_file will fail because file doesn't exist

    // 4. Second Turn
    assert!(events.iter().any(|e| matches!(
        e,
        KernelEvent::Lifecycle(LifecycleEvent::TurnStart { turn_index: 1, .. })
    )));
    assert!(
        events
            .iter()
            .any(|e| matches!(e, KernelEvent::Stream(StreamEvent::MessageDelta { .. })))
    );
    assert!(events.iter().any(|e| matches!(
        e,
        KernelEvent::Lifecycle(LifecycleEvent::TurnEnd {
            turn_index: 1,
            has_tool_calls: false,
            ..
        })
    )));

    kernel.end_session(&mut session).await?;

    // Re-check for SessionEnd
    while let Ok(event) = rx.try_recv() {
        events.push(event.1);
    }
    assert!(matches!(
        events.last().unwrap(),
        KernelEvent::Lifecycle(LifecycleEvent::SessionEnd { .. })
    ));

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_harness_observation() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_obs.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    // This harness script records detected events into a global KV store
    let harness_code = r#"
        function on_kernel_event(event)
            if event.type == "message_delta" then
                local k = kv.as(runtime.context("project", "state"))
                local current, _ = k.get("observed_tokens")
                current = current or ""
                k.set("observed_tokens", current .. event.content_delta)
            end
            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("observer.lua"), harness_code)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: None,
            ..ProviderConfig::default()
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Test".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: std::collections::HashMap::new(),
        kernel: turin::kernel::config::KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 1,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        layout: Default::default(),
        inference: InferenceConfig::default(),
        persistence: PersistenceConfig::with_state_path(db_path.to_str().unwrap().to_string()),
        harness: HarnessConfig {
            directory: harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::new(),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: turin::kernel::config::GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;

    let responses = vec![vec![
        InferenceEvent::MessageStart {
            role: "assistant".to_string(),
            model: "mock-model".to_string(),
            provider_id: "mock".to_string(),
        },
        InferenceEvent::MessageDelta {
            content: "Hello".to_string(),
        },
        InferenceEvent::MessageDelta {
            content: " World".to_string(),
        },
        InferenceEvent::MessageEnd {
            input_tokens: 1,
            output_tokens: 1,
            stop_reason: None,
        },
    ]];

    let mock_provider = Arc::new(SequenceMockProvider {
        responses: Arc::new(std::sync::Mutex::new(responses)),
    });
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new("mock", mock_provider),
    );
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel.run(&mut session, Some("Hi".to_string())).await?;

    // Check KV store if it was updated by the harness (project:state context)
    let store = kernel.store_manager().get_default().await?;
    let val: Option<String> = store.kv_get("project", "state", "observed_tokens").await?;
    assert_eq!(val, Some("Hello World".to_string()));

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_nested_agent_spawning() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_nest.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    // This harness script spawns a sub-agent when it sees a "nest" keyword
    let harness_code = r#"
        function on_turn_prepare(ctx)
            if ctx.prompt and ctx.prompt:find("trigger_nesting") then
                local result = agent.spawn("nest_inner_work")
                ctx.prompt = "Sub-agent result: " .. result
                return ALLOW
            end
            if ctx.prompt and ctx.prompt:find("nest_inner_work") then
                local k = kv.as(runtime.context("project", "state"))
                k.set("nested_executed", "true")
            end
            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("nester.lua"), harness_code)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: None,
            ..ProviderConfig::default()
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Outer".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: std::collections::HashMap::new(),
        kernel: turin::kernel::config::KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 1,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        layout: Default::default(),
        inference: InferenceConfig::default(),
        persistence: PersistenceConfig::with_state_path(db_path.to_str().unwrap().to_string()),
        harness: HarnessConfig {
            directory: harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::new(),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: turin::kernel::config::GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;

    let responses = vec![
        vec![
            InferenceEvent::MessageStart {
                role: "assistant".to_string(),
                model: "mock-model".to_string(),
                provider_id: "mock".to_string(),
            },
            InferenceEvent::MessageDelta {
                content: "NEST_SUCCESS".to_string(),
            },
            InferenceEvent::MessageEnd {
                input_tokens: 1,
                output_tokens: 1,
                stop_reason: None,
            },
        ],
        vec![
            InferenceEvent::MessageStart {
                role: "assistant".to_string(),
                model: "mock-model".to_string(),
                provider_id: "mock".to_string(),
            },
            InferenceEvent::MessageDelta {
                content: "Final Response".to_string(),
            },
            InferenceEvent::MessageEnd {
                input_tokens: 1,
                output_tokens: 1,
                stop_reason: None,
            },
        ],
    ];

    let mock_provider = Arc::new(SequenceMockProvider {
        responses: Arc::new(std::sync::Mutex::new(responses)),
    });
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new("mock", mock_provider),
    );
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("trigger_nesting now".to_string()))
        .await?;

    // Verify sub-agent work happened (observed via shared DB)
    let store = kernel.store_manager().get_default().await?;
    let val: Option<String> = store.kv_get("project", "state", "nested_executed").await?;
    assert_eq!(val, Some("true".to_string()));

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_on_inference_error_can_queue_fallback_task() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_inference_error.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let harness_code = r#"
        function on_inference_error(event)
            local k = kv.as(runtime.context("project", "state"))
            k.set("last_inference_error", tostring(event.error))
            return MODIFY, { "retry with fallback task" }
        end
    "#;
    std::fs::write(harness_dir.join("recover.lua"), harness_code)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: None,
            ..ProviderConfig::default()
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Recover on stream errors".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: std::collections::HashMap::new(),
        kernel: turin::kernel::config::KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 3,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        layout: Default::default(),
        inference: InferenceConfig::default(),
        persistence: PersistenceConfig::with_state_path(db_path.to_str().unwrap().to_string()),
        harness: HarnessConfig {
            directory: harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::new(),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: turin::kernel::config::GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_harness().await?;

    let provider = Arc::new(FailThenRecoverProvider {
        should_fail: Arc::new(std::sync::Mutex::new(true)),
    });
    kernel.add_client("mock".to_string(), ProviderClient::new("mock", provider));

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("trigger".to_string()))
        .await?;

    let saw_recovered = session.history.iter().any(|msg| {
        msg.content.iter().any(|content| {
            matches!(
                content,
                turin::inference::provider::InferenceContent::Text { text } if text.contains("Recovered")
            )
        })
    });
    assert!(
        saw_recovered,
        "expected fallback task to complete successfully"
    );

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_stale_branch_conflict_does_not_trigger_inference_recovery() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_branch_conflict.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let harness_code = r#"
        function on_inference_error(event)
            local k = kv.as(runtime.context("project", "state"))
            k.set("last_inference_error", tostring(event.error))
            return MODIFY, { "retry with fallback task" }
        end
    "#;
    std::fs::write(harness_dir.join("recover.lua"), harness_code)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: None,
            ..ProviderConfig::default()
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Conflict classification".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: std::collections::HashMap::new(),
        kernel: turin::kernel::config::KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 3,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        layout: Default::default(),
        inference: InferenceConfig::default(),
        persistence: PersistenceConfig::with_state_path(db_path.to_str().unwrap().to_string()),
        harness: HarnessConfig {
            directory: harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::new(),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: turin::kernel::config::GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_harness().await?;
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new(
            "mock",
            Arc::new(SequenceMockProvider {
                responses: Arc::new(std::sync::Mutex::new(Vec::new())),
            }),
        ),
    );

    let mut session = kernel.create_session().await;
    let store = kernel.store_manager().open(&session.store_selector).await?;
    let internal_id = session.internal_id.expect("session should be persisted");

    let first_turn = store
        .prepare_turn_write_target(
            internal_id,
            turin::persistence::state::TurnWriteTarget::branch_head_with_expectation(
                session.selected_branch_head_id(),
                session.selected_branch_head_turn_id(),
                0,
            ),
        )
        .await?
        .expect("first turn should be created");
    let first_turn_id = match first_turn {
        turin::persistence::state::TurnWriteTarget::ExistingTurn { turn_id, .. } => turn_id,
        _ => unreachable!("prepared turn targets should resolve to an existing turn"),
    };

    let second_turn = store
        .prepare_turn_write_target(
            internal_id,
            turin::persistence::state::TurnWriteTarget::branch_head_with_expectation(
                session.selected_branch_head_id(),
                Some(first_turn_id),
                1,
            ),
        )
        .await?
        .expect("second turn should be created");
    let second_turn_id = match second_turn {
        turin::persistence::state::TurnWriteTarget::ExistingTurn { turn_id, .. } => turn_id,
        _ => unreachable!("prepared turn targets should resolve to an existing turn"),
    };
    let existing_message = serde_json::json!([
        {
            "type": "text",
            "text": "existing"
        }
    ]);

    store
        .insert_message(
            internal_id,
            second_turn,
            "assistant",
            &existing_message,
            None,
        )
        .await?;

    session.set_selected_branch_head_turn_id(Some(first_turn_id));
    session.set_selected_branch_head_turn_index(Some(0));

    kernel
        .run(&mut session, Some("trigger".to_string()))
        .await?;

    let conflict_event = store
        .get_all_events(internal_id)
        .await?
        .into_iter()
        .filter(|event| event.event_type == "task_complete")
        .find_map(|event| serde_json::from_str::<serde_json::Value>(&event.payload).ok())
        .and_then(|payload| {
            payload
                .get("status")
                .and_then(|value| value.as_str().map(str::to_string))
        });
    assert_eq!(conflict_event.as_deref(), Some("conflict"));

    let recovery: Option<String> = store
        .kv_get("project", "state", "last_inference_error")
        .await?;
    assert_eq!(recovery, None);

    session.set_selected_branch_head_turn_id(Some(second_turn_id));
    session.set_selected_branch_head_turn_index(Some(1));
    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_stale_branch_conflict_can_continue_detached() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_branch_conflict_detached.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: None,
            ..ProviderConfig::default()
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Conflict detaches".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: std::collections::HashMap::new(),
        kernel: turin::kernel::config::KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 3,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        layout: Default::default(),
        inference: InferenceConfig::default(),
        persistence: PersistenceConfig::with_state_path(db_path.to_str().unwrap().to_string()),
        harness: HarnessConfig {
            directory: harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::new(),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: turin::kernel::config::GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_harness().await?;
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new(
            "mock",
            Arc::new(SequenceMockProvider {
                responses: Arc::new(std::sync::Mutex::new(Vec::new())),
            }),
        ),
    );

    let mut session = kernel.create_session().await;

    let store = kernel.store_manager().open(&session.store_selector).await?;
    let internal_id = session.internal_id.expect("session should be persisted");

    let first_turn = store
        .prepare_turn_write_target(
            internal_id,
            turin::persistence::state::TurnWriteTarget::branch_head_with_expectation(
                session.selected_branch_head_id(),
                session.selected_branch_head_turn_id(),
                0,
            ),
        )
        .await?
        .expect("first turn should be created");
    let first_turn_id = match first_turn {
        turin::persistence::state::TurnWriteTarget::ExistingTurn { turn_id, .. } => turn_id,
        _ => unreachable!("prepared turn targets should resolve to an existing turn"),
    };

    let second_turn = store
        .prepare_turn_write_target(
            internal_id,
            turin::persistence::state::TurnWriteTarget::branch_head_with_expectation(
                session.selected_branch_head_id(),
                Some(first_turn_id),
                1,
            ),
        )
        .await?
        .expect("second turn should be created");
    let existing_message = serde_json::json!([
        {
            "type": "text",
            "text": "existing"
        }
    ]);
    store
        .insert_message(
            internal_id,
            second_turn,
            "assistant",
            &existing_message,
            None,
        )
        .await?;

    session.set_selected_branch_head_turn_id(Some(first_turn_id));
    session.set_selected_branch_head_turn_index(Some(0));

    {
        let mut queue = session.queue.lock().await;
        queue.push_back(
            QueuedTask::ad_hoc("trigger")
                .with_conflict_policy(Some(ExecutionConflictPolicy::Detached)),
        );
    }

    kernel.run(&mut session, None).await?;

    let statuses = store
        .get_all_events(internal_id)
        .await?
        .into_iter()
        .filter(|event| event.event_type == "task_complete")
        .filter_map(|event| serde_json::from_str::<serde_json::Value>(&event.payload).ok())
        .filter_map(|payload| {
            payload
                .get("status")
                .and_then(|value| value.as_str().map(str::to_string))
        })
        .collect::<Vec<_>>();
    assert!(statuses.contains(&"success".to_string()));
    assert!(!statuses.contains(&"conflict".to_string()));

    let messages = store
        .get_messages(
            internal_id,
            &turin::persistence::state::SessionReadTarget::branch_head(
                session.selected_branch_head_id(),
            ),
        )
        .await?;
    assert_eq!(messages.len(), 1);
    assert!(session.history.iter().any(|msg| {
        msg.content.iter().any(|content| {
            matches!(
                content,
                turin::inference::provider::InferenceContent::Text { text } if text.contains("Finishing.")
            )
        })
    }));

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_stale_branch_conflict_can_fork_sibling_durably() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_branch_conflict_fork_sibling.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: None,
            ..ProviderConfig::default()
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Conflict forks".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: std::collections::HashMap::new(),
        kernel: turin::kernel::config::KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 3,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        layout: Default::default(),
        inference: InferenceConfig::default(),
        persistence: PersistenceConfig::with_state_path(db_path.to_str().unwrap().to_string()),
        harness: HarnessConfig {
            directory: harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::new(),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: turin::kernel::config::GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_harness().await?;
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new(
            "mock",
            Arc::new(SequenceMockProvider {
                responses: Arc::new(std::sync::Mutex::new(Vec::new())),
            }),
        ),
    );

    let mut session = kernel.create_session().await;

    let store = kernel.store_manager().open(&session.store_selector).await?;
    let internal_id = session.internal_id.expect("session should be persisted");

    let first_turn = store
        .prepare_turn_write_target(
            internal_id,
            turin::persistence::state::TurnWriteTarget::branch_head_with_expectation(
                session.selected_branch_head_id(),
                session.selected_branch_head_turn_id(),
                0,
            ),
        )
        .await?
        .expect("first turn should be created");
    let first_turn_id = match first_turn {
        turin::persistence::state::TurnWriteTarget::ExistingTurn { turn_id, .. } => turn_id,
        _ => unreachable!("prepared turn targets should resolve to an existing turn"),
    };

    let second_turn = store
        .prepare_turn_write_target(
            internal_id,
            turin::persistence::state::TurnWriteTarget::branch_head_with_expectation(
                session.selected_branch_head_id(),
                Some(first_turn_id),
                1,
            ),
        )
        .await?
        .expect("second turn should be created");
    let existing_message = serde_json::json!([
        {
            "type": "text",
            "text": "existing"
        }
    ]);
    store
        .insert_message(
            internal_id,
            second_turn,
            "assistant",
            &existing_message,
            None,
        )
        .await?;

    session.set_selected_branch_head_turn_id(Some(first_turn_id));
    session.set_selected_branch_head_turn_index(Some(0));

    {
        let mut queue = session.queue.lock().await;
        queue.push_back(
            QueuedTask::ad_hoc("trigger")
                .with_conflict_policy(Some(ExecutionConflictPolicy::ForkSibling)),
        );
    }

    kernel.run(&mut session, None).await?;

    let statuses = store
        .get_all_events(internal_id)
        .await?
        .into_iter()
        .filter(|event| event.event_type == "task_complete")
        .filter_map(|event| serde_json::from_str::<serde_json::Value>(&event.payload).ok())
        .filter_map(|payload| {
            payload
                .get("status")
                .and_then(|value| value.as_str().map(str::to_string))
        })
        .collect::<Vec<_>>();
    assert!(statuses.contains(&"success".to_string()));
    assert!(!statuses.contains(&"conflict".to_string()));

    let task_complete_payload = store
        .get_all_events(internal_id)
        .await?
        .into_iter()
        .filter(|event| event.event_type == "task_complete")
        .filter_map(|event| serde_json::from_str::<serde_json::Value>(&event.payload).ok())
        .find(|payload| payload["status"] == "success")
        .expect("successful task_complete payload should be persisted");
    assert_eq!(
        task_complete_payload["branch_outcome"]["kind"],
        "fork_sibling"
    );
    assert_eq!(
        task_complete_payload["branch_outcome"]["persisted_active_head_unchanged"],
        true
    );

    let active_messages = store
        .get_messages(
            internal_id,
            &turin::persistence::state::SessionReadTarget::branch_head(
                session.selected_branch_head_id(),
            ),
        )
        .await?;
    assert!(
        active_messages.iter().any(|message| {
            message.role == "assistant"
                && serde_json::from_str::<serde_json::Value>(&message.content)
                    .ok()
                    .and_then(|content| {
                        content
                            .as_array()
                            .and_then(|parts| parts.first())
                            .and_then(|part| part.get("text"))
                            .and_then(|value| value.as_str())
                            .map(|text| text.contains("Finishing."))
                    })
                    .unwrap_or(false)
        }),
        "forked branch should persist the assistant completion"
    );

    let main_messages = store
        .get_messages(
            internal_id,
            &turin::persistence::state::SessionReadTarget::ActiveBranch,
        )
        .await?;
    assert_eq!(main_messages.len(), 1);
    assert_eq!(main_messages[0].role, "assistant");

    let branch_heads = store.list_branch_heads(internal_id).await?;
    assert_eq!(branch_heads.len(), 2);
    assert!(
        branch_heads
            .iter()
            .any(|branch| branch.id == session.selected_branch_head_id().unwrap()),
        "session should be retargeted to the forked sibling branch"
    );

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test]
async fn test_dynamic_mode_switching_stateless() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_stateless.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let harness_code = r#"
        function on_turn_start(event)
            agent.mode.set("stateless")
        end
    "#;
    std::fs::write(harness_dir.join("stateless.lua"), harness_code)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: None,
            ..ProviderConfig::default()
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Stateless classifier".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: std::collections::HashMap::new(),
        kernel: turin::kernel::config::KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 5,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        layout: Default::default(),
        inference: InferenceConfig::default(),
        persistence: PersistenceConfig::with_state_path(db_path.to_str().unwrap().to_string()),
        harness: HarnessConfig {
            directory: harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::new(),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: turin::kernel::config::GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_harness().await?;

    let provider = Arc::new(SequenceMockProvider {
        responses: Arc::new(std::sync::Mutex::new(vec![vec![
            InferenceEvent::MessageStart {
                role: "assistant".to_string(),
                model: "mock-model".to_string(),
                provider_id: "mock".to_string(),
            },
            InferenceEvent::MessageDelta {
                content: "One shot only.".to_string(),
            },
            InferenceEvent::MessageEnd {
                input_tokens: 1,
                output_tokens: 1,
                stop_reason: None,
            },
        ]])),
    });
    kernel.add_client("mock".to_string(), ProviderClient::new("mock", provider));

    let mut session = kernel.create_session().await;
    // The run normally would loop 2 times due to the tool call, but stateless drops it instantly after the first yield
    kernel
        .run(&mut session, Some("process".to_string()))
        .await?;

    // Verify it terminated strictly after exactly 1 turn
    assert_eq!(
        session.turn_index, 1,
        "Agent should only complete exactly 1 turn due to stateless mode"
    );
    assert_eq!(session.mode, turin::kernel::config::AgentMode::Stateless);

    kernel.end_session(&mut session).await?;
    Ok(())
}
