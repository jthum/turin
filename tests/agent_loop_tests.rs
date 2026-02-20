use anyhow::Result;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream;
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::tempdir;
use turin::inference::provider::{
    InferenceEvent, InferenceProvider, InferenceRequest, InferenceStream, ProviderClient,
    ProviderKind, RequestOptions, SdkError,
};
use turin::kernel::Kernel;
use turin::kernel::config::{
    AgentConfig, EmbeddingConfig, HarnessConfig, PersistenceConfig, ProviderConfig, TurinConfig,
};
use turin::kernel::event::{AuditEvent, KernelEvent, LifecycleEvent, StreamEvent};

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
        agent: AgentConfig {
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Test".to_string(),
            thinking: None,
        },
        kernel: turin::kernel::config::KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
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
        ProviderClient::new(ProviderKind::Mock, mock_provider),
    );

    let mut session = kernel.create_session();

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
                local current = db.kv_get("observed_tokens") or ""
                db.kv_set("observed_tokens", current .. event.content_delta)
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
        agent: AgentConfig {
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Test".to_string(),
            thinking: None,
        },
        kernel: turin::kernel::config::KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 1,
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
        ProviderClient::new(ProviderKind::Mock, mock_provider),
    );
    kernel.init_harness().await?;

    let mut session = kernel.create_session();
    kernel.run(&mut session, Some("Hi".to_string())).await?;

    // Check KV store if it was updated by the harness
    if let Some(store) = kernel.state() {
        let val: Option<String> = store.kv_get("observed_tokens").await?;
        assert_eq!(val, Some("Hello World".to_string()));
    }

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
                -- Sub-agent will write to DB
                local result = turin.agent.spawn("nest_inner_work")
                ctx.prompt = "Sub-agent result: " .. result
                return ALLOW
            end
            if ctx.prompt and ctx.prompt:find("nest_inner_work") then
                db.kv_set("nested_executed", "true")
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
        agent: AgentConfig {
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Outer".to_string(),
            thinking: None,
        },
        kernel: turin::kernel::config::KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 1,
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
        ProviderClient::new(ProviderKind::Mock, mock_provider),
    );
    kernel.init_harness().await?;

    let mut session = kernel.create_session();
    kernel
        .run(&mut session, Some("trigger_nesting now".to_string()))
        .await?;

    // Verify sub-agent work happened (observed via shared DB)
    if let Some(store) = kernel.state() {
        let val: Option<String> = store.kv_get("nested_executed").await?;
        assert_eq!(val, Some("true".to_string()));
    }

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
            db.kv_set("last_inference_error", tostring(event.error))
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
        agent: AgentConfig {
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Recover on stream errors".to_string(),
            thinking: None,
        },
        kernel: turin::kernel::config::KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 3,
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
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_harness().await?;

    let provider = Arc::new(FailThenRecoverProvider {
        should_fail: Arc::new(std::sync::Mutex::new(true)),
    });
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new(ProviderKind::Mock, provider),
    );

    let mut session = kernel.create_session();
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
