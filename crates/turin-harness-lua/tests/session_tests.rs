//! Session lifecycle and kernel edge-case tests.
//!
//! Tests for session creation, start, end, token accounting,
//! harness hot-reload, and max_turns enforcement.

#[macro_use]
#[path = "support/config_fixture.rs"]
mod config_fixture;

use anyhow::Result;
use async_trait::async_trait;
use futures::future::BoxFuture;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tempfile::tempdir;
use turin_core::inference::provider::{
    InferenceEvent, InferenceMessage, InferenceProvider, InferenceRequest, InferenceResponseFormat,
    InferenceResult, InferenceStream, ProviderClient, RequestOptions, SdkError, Usage,
};
use turin_core::kernel::Kernel;
use turin_core::kernel::config::{
    AgentConfig, EmbeddingConfig, HarnessConfig, InferenceConfig, KernelConfig, PersistenceConfig,
    ProviderConfig, StoreTargetConfig, TurinConfig,
};
use turin_core::kernel::policy::PolicyScope;
use turin_core::kernel::session::{
    ExecutionContextTarget, ExecutionVisibility, ExecutionWritePolicy, QueuedTask, SessionStatus,
    TaskExecutionOverrides,
};
use turin_core::kernel::session_refs::parse_session_reference;
use turin_core::persistence::state::{SessionReadTarget, TurnWriteTarget};
use turin_types::TaskInputContent;

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

    config_fixture! {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Test assistant.".to_string(),
            thinking: None,
            harness: None,
            idle_timeout_seconds: None,
            linked_runtime_lanes: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: std::collections::HashMap::new(),
        runtime: Default::default(),
        kernel: KernelConfig {
            workspace_root: tmp.to_str().unwrap().to_string(),
            max_turns: 5,
            heartbeat_interval_seconds: 30,
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
        governance: turin_core::kernel::config::GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    }
}

async fn make_kernel(tmp: &std::path::Path) -> Result<Kernel> {
    let config = make_config(tmp);
    let mut kernel = turin_harness_lua::runtime_builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;
    Ok(kernel)
}

#[tokio::test]
async fn test_run_script_delegates_to_lua_source_capability() -> Result<()> {
    let tmp = tempdir()?;
    let kernel = make_kernel(tmp.path()).await?;

    kernel.run_script("local value = 6 * 7; assert(value == 42)")?;
    let error = kernel
        .run_script("error('direct-source-sentinel')")
        .expect_err("direct script errors should propagate");
    assert!(error.to_string().contains("direct-source-sentinel"));
    Ok(())
}

fn event_has_task_status(
    events: &[turin_core::persistence::schema::EventRow],
    status: &str,
) -> bool {
    events.iter().any(|e| {
        e.event_type == "task_complete"
            && serde_json::from_str::<serde_json::Value>(&e.payload)
                .ok()
                .and_then(|v| v.get("status").and_then(|s| s.as_str()).map(str::to_string))
                .is_some_and(|s| s == status)
    })
}

fn count_event_type(
    events: &[turin_core::persistence::schema::EventRow],
    event_type: &str,
) -> usize {
    events.iter().filter(|e| e.event_type == event_type).count()
}

#[derive(Default)]
struct CaptureMessagesProvider {
    seen_messages: Arc<Mutex<Vec<turin_core::inference::provider::InferenceMessage>>>,
}

#[async_trait]
impl InferenceProvider for CaptureMessagesProvider {
    fn stream<'a>(
        &'a self,
        request: InferenceRequest,
        _options: Option<RequestOptions>,
    ) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        {
            let mut seen = self
                .seen_messages
                .lock()
                .expect("capture messages mutex poisoned");
            *seen = request.messages.clone();
        }

        Box::pin(async move {
            let stream = futures::stream::iter(vec![
                Ok(InferenceEvent::MessageStart {
                    role: "assistant".to_string(),
                    model: "capture-model".to_string(),
                    provider_id: "capture".to_string(),
                }),
                Ok(InferenceEvent::MessageDelta {
                    content: "CAPTURED".to_string(),
                }),
                Ok(InferenceEvent::MessageEnd {
                    input_tokens: 10,
                    output_tokens: 5,
                    cache_read_input_tokens: None,
                    cache_creation_input_tokens: None,
                    stop_reason: None,
                }),
            ]);
            Ok(Box::pin(stream) as InferenceStream)
        })
    }
}

#[derive(Default)]
struct ContextCheckpointProvider {
    seen_stream_messages: Arc<Mutex<Vec<turin_core::inference::provider::InferenceMessage>>>,
    seen_stream_system: Arc<Mutex<Option<String>>>,
    complete_calls: Arc<Mutex<usize>>,
}

struct StructuredOutputProvider {
    seen_response_format: Arc<Mutex<Option<InferenceResponseFormat>>>,
}

struct PromptStructuredFallbackProvider {
    seen_system_prompt: Arc<Mutex<Option<String>>>,
    seen_response_format: Arc<Mutex<Option<InferenceResponseFormat>>>,
}

struct SequenceCaptureProvider {
    responses: Arc<Mutex<Vec<Vec<InferenceEvent>>>>,
    seen_messages: Arc<Mutex<Vec<Vec<InferenceMessage>>>>,
}

struct StalledStreamProvider;

#[async_trait]
impl InferenceProvider for StalledStreamProvider {
    fn stream<'a>(
        &'a self,
        _request: InferenceRequest,
        _options: Option<RequestOptions>,
    ) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        Box::pin(async move { Ok(Box::pin(futures::stream::pending()) as InferenceStream) })
    }
}

struct ToolCallProvider;

#[async_trait]
impl InferenceProvider for ToolCallProvider {
    fn stream<'a>(
        &'a self,
        _request: InferenceRequest,
        _options: Option<RequestOptions>,
    ) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        Box::pin(async move {
            let stream = futures::stream::iter(vec![
                Ok(InferenceEvent::MessageStart {
                    role: "assistant".to_string(),
                    model: "tool-call-model".to_string(),
                    provider_id: "tool-call".to_string(),
                }),
                Ok(InferenceEvent::ToolCallStart {
                    id: "call_stall_1".to_string(),
                    name: "shell_exec".to_string(),
                }),
                Ok(InferenceEvent::ToolCallDelta {
                    delta: serde_json::json!({
                        "command": "touch tool-started && sleep 60",
                        "timeout_seconds": 60
                    })
                    .to_string(),
                }),
                Ok(InferenceEvent::MessageEnd {
                    input_tokens: 1,
                    output_tokens: 1,
                    cache_read_input_tokens: None,
                    cache_creation_input_tokens: None,
                    stop_reason: None,
                }),
            ]);
            Ok(Box::pin(stream) as InferenceStream)
        })
    }
}

#[async_trait]
impl InferenceProvider for SequenceCaptureProvider {
    fn stream<'a>(
        &'a self,
        request: InferenceRequest,
        _options: Option<RequestOptions>,
    ) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        {
            let mut seen = self
                .seen_messages
                .lock()
                .expect("sequence capture messages mutex poisoned");
            seen.push(request.messages.clone());
        }

        let responses = Arc::clone(&self.responses);
        Box::pin(async move {
            let events = {
                let mut guard = responses
                    .lock()
                    .expect("sequence capture responses mutex poisoned");
                if guard.is_empty() {
                    vec![
                        InferenceEvent::MessageStart {
                            role: "assistant".to_string(),
                            model: "sequence-model".to_string(),
                            provider_id: "sequence".to_string(),
                        },
                        InferenceEvent::MessageDelta {
                            content: "SEQUENCE FALLBACK".to_string(),
                        },
                        InferenceEvent::MessageEnd {
                            input_tokens: 1,
                            output_tokens: 1,
                            cache_read_input_tokens: None,
                            cache_creation_input_tokens: None,
                            stop_reason: None,
                        },
                    ]
                } else {
                    guard.remove(0)
                }
            };
            let stream = futures::stream::iter(events.into_iter().map(Ok));
            Ok(Box::pin(stream) as InferenceStream)
        })
    }
}

#[async_trait]
impl InferenceProvider for StructuredOutputProvider {
    fn supports_response_format(&self, response_format: &InferenceResponseFormat) -> bool {
        matches!(response_format, InferenceResponseFormat::JsonSchema { .. })
    }

    fn complete<'a>(
        &'a self,
        request: InferenceRequest,
        _options: Option<RequestOptions>,
    ) -> BoxFuture<'a, Result<InferenceResult, SdkError>> {
        {
            let mut seen = self
                .seen_response_format
                .lock()
                .expect("structured provider mutex poisoned");
            *seen = request.response_format.clone();
        }

        Box::pin(async move {
            Ok(InferenceResult {
                content: vec![turin_core::inference::provider::InferenceContent::Text {
                    text: r#"{"approved":true,"summary":"structured ok"}"#.to_string(),
                }],
                model: "structured-model".to_string(),
                stop_reason: None,
                usage: Usage {
                    input_tokens: 12,
                    output_tokens: 6,
                    cache_read_input_tokens: None,
                    cache_creation_input_tokens: None,
                },
            })
        })
    }

    fn stream<'a>(
        &'a self,
        _request: InferenceRequest,
        _options: Option<RequestOptions>,
    ) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        Box::pin(async move {
            let stream = futures::stream::iter(vec![
                Ok(InferenceEvent::MessageStart {
                    role: "assistant".to_string(),
                    model: "structured-model".to_string(),
                    provider_id: "structured".to_string(),
                }),
                Ok(InferenceEvent::MessageDelta {
                    content: "MAIN TURN".to_string(),
                }),
                Ok(InferenceEvent::MessageEnd {
                    input_tokens: 4,
                    output_tokens: 2,
                    cache_read_input_tokens: None,
                    cache_creation_input_tokens: None,
                    stop_reason: None,
                }),
            ]);
            Ok(Box::pin(stream) as InferenceStream)
        })
    }
}

#[async_trait]
impl InferenceProvider for PromptStructuredFallbackProvider {
    fn complete<'a>(
        &'a self,
        request: InferenceRequest,
        _options: Option<RequestOptions>,
    ) -> BoxFuture<'a, Result<InferenceResult, SdkError>> {
        {
            let mut seen_system = self
                .seen_system_prompt
                .lock()
                .expect("structured fallback system mutex poisoned");
            *seen_system = request.system.clone();
        }
        {
            let mut seen_response_format = self
                .seen_response_format
                .lock()
                .expect("structured fallback response format mutex poisoned");
            *seen_response_format = request.response_format.clone();
        }

        Box::pin(async move {
            Ok(InferenceResult {
                content: vec![turin_core::inference::provider::InferenceContent::Text {
                    text: r#"{"decision":"fallback","confidence":0.8}"#.to_string(),
                }],
                model: "fallback-model".to_string(),
                stop_reason: None,
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 4,
                    cache_read_input_tokens: None,
                    cache_creation_input_tokens: None,
                },
            })
        })
    }

    fn stream<'a>(
        &'a self,
        _request: InferenceRequest,
        _options: Option<RequestOptions>,
    ) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        Box::pin(async move {
            let stream = futures::stream::iter(vec![
                Ok(InferenceEvent::MessageStart {
                    role: "assistant".to_string(),
                    model: "fallback-model".to_string(),
                    provider_id: "fallback".to_string(),
                }),
                Ok(InferenceEvent::MessageDelta {
                    content: "MAIN TURN".to_string(),
                }),
                Ok(InferenceEvent::MessageEnd {
                    input_tokens: 4,
                    output_tokens: 2,
                    cache_read_input_tokens: None,
                    cache_creation_input_tokens: None,
                    stop_reason: None,
                }),
            ]);
            Ok(Box::pin(stream) as InferenceStream)
        })
    }
}

#[async_trait]
impl InferenceProvider for ContextCheckpointProvider {
    fn complete<'a>(
        &'a self,
        request: InferenceRequest,
        _options: Option<RequestOptions>,
    ) -> BoxFuture<'a, Result<turin_core::inference::provider::InferenceResult, SdkError>> {
        {
            let mut calls = self
                .complete_calls
                .lock()
                .expect("context checkpoint complete mutex poisoned");
            *calls += 1;
        }

        let _ = request;
        Box::pin(async move {
            Ok(turin_core::inference::provider::InferenceResult {
                content: vec![turin_core::inference::provider::InferenceContent::Text {
                    text: "CHECKPOINT SUMMARY".to_string(),
                }],
                model: "checkpoint-model".to_string(),
                stop_reason: None,
                usage: turin_core::inference::provider::Usage {
                    input_tokens: 32,
                    output_tokens: 8,
                    cache_read_input_tokens: None,
                    cache_creation_input_tokens: None,
                },
            })
        })
    }

    fn stream<'a>(
        &'a self,
        request: InferenceRequest,
        _options: Option<RequestOptions>,
    ) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        {
            let mut seen_messages = self
                .seen_stream_messages
                .lock()
                .expect("context checkpoint stream messages mutex poisoned");
            *seen_messages = request.messages.clone();
        }
        {
            let mut seen_system = self
                .seen_stream_system
                .lock()
                .expect("context checkpoint stream system mutex poisoned");
            *seen_system = request.system.clone();
        }

        Box::pin(async move {
            let stream = futures::stream::iter(vec![
                Ok(InferenceEvent::MessageStart {
                    role: "assistant".to_string(),
                    model: "checkpoint-model".to_string(),
                    provider_id: "checkpoint".to_string(),
                }),
                Ok(InferenceEvent::MessageDelta {
                    content: "CHECKPOINTED".to_string(),
                }),
                Ok(InferenceEvent::MessageEnd {
                    input_tokens: 12,
                    output_tokens: 6,
                    cache_read_input_tokens: None,
                    cache_creation_input_tokens: None,
                    stop_reason: None,
                }),
            ]);
            Ok(Box::pin(stream) as InferenceStream)
        })
    }
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
async fn test_session_end_is_terminal() -> Result<()> {
    let tmp = tempdir()?;
    let kernel = make_kernel(tmp.path()).await?;

    let mut session = kernel.create_session().await;
    kernel.start_session(&mut session).await?;
    assert_eq!(session.status, SessionStatus::Active);

    kernel.end_session(&mut session).await?;
    assert_eq!(session.status, SessionStatus::Ended);

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
    assert_eq!(session.status, SessionStatus::Ended);

    Ok(())
}

#[tokio::test]
async fn test_never_started_session_can_be_ended_cleanly() -> Result<()> {
    let tmp = tempdir()?;
    let kernel = make_kernel(tmp.path()).await?;

    let mut session = kernel.create_session().await;
    kernel.end_session(&mut session).await?;

    assert_eq!(session.status, SessionStatus::Ended);
    assert!(session.cancel_token.is_cancelled());
    assert!(session.durability_tx.is_none());
    assert!(
        session
            .event_task
            .as_ref()
            .expect("event task slot")
            .lock()
            .await
            .is_none()
    );

    Ok(())
}

#[tokio::test]
async fn test_ended_session_cannot_be_restarted() -> Result<()> {
    let tmp = tempdir()?;
    let kernel = make_kernel(tmp.path()).await?;

    let mut session = kernel.create_session().await;
    kernel.start_session(&mut session).await?;
    kernel.end_session(&mut session).await?;

    let error = kernel
        .start_session(&mut session)
        .await
        .expect_err("ended session must remain terminal");
    assert!(
        error
            .to_string()
            .contains("Ended sessions cannot be restarted")
    );

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
async fn test_cancelling_stalled_inference_does_not_append_assistant_output() -> Result<()> {
    let tmp = tempdir()?;
    let config = make_config(tmp.path());
    let mut kernel = turin_harness_lua::runtime_builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_harness().await?;
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new("mock", Arc::new(StalledStreamProvider)),
    );
    let mut session = kernel.create_session().await;
    let cancel_token = session.cancel_token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        cancel_token.cancel();
    });

    kernel
        .run(&mut session, Some("Wait for cancellation".to_string()))
        .await?;
    assert_eq!(session.history.len(), 1);
    assert_eq!(
        session.history.messages()[0].role,
        turin_core::inference::provider::InferenceRole::User
    );
    Ok(())
}

#[tokio::test]
async fn test_cancelling_stalled_tool_does_not_append_tool_result() -> Result<()> {
    let tmp = tempdir()?;
    let mut config = make_config(tmp.path());
    config.tools.selection.allow = Some(vec!["shell_exec".to_string()]);
    config.agent.tools.selection.allow = Some(vec!["shell_exec".to_string()]);
    let mut kernel = turin_harness_lua::runtime_builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_harness().await?;
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new("mock", Arc::new(ToolCallProvider)),
    );
    let mut session = kernel.create_session().await;
    let cancel_token = session.cancel_token.clone();
    let started_path = tmp.path().join("tool-started");
    tokio::spawn(async move {
        for _ in 0..100 {
            if started_path.exists() {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
        cancel_token.cancel();
    });

    kernel
        .run(&mut session, Some("Run the stalled tool".to_string()))
        .await?;
    assert_eq!(session.history.len(), 2);
    assert!(
        session.history.messages().iter().all(|message| {
            message.role != turin_core::inference::provider::InferenceRole::Tool
        })
    );
    Ok(())
}

#[tokio::test]
async fn test_run_stops_when_user_message_persistence_fails() -> Result<()> {
    let tmp = tempdir()?;
    let mut kernel = make_kernel(tmp.path()).await?;
    let mut session = kernel.create_session().await;
    let store = kernel.store_manager().open(&session.store_selector).await?;
    let conn = store.get_connection().await?;
    conn.execute_batch(
        r#"
        CREATE TRIGGER fail_message_insert
        BEFORE INSERT ON messages
        BEGIN
            SELECT RAISE(FAIL, 'injected message persistence failure');
        END;
        "#,
    )
    .await?;

    let error = kernel
        .run(&mut session, Some("Do not lose this message".to_string()))
        .await
        .expect_err("durable task should stop when its user message cannot be stored");
    assert!(
        error
            .to_string()
            .contains("Failed to persist user turn message")
    );
    assert!(session.history.is_empty());

    let messages = store
        .get_messages(
            session.internal_id.expect("persisted session"),
            &SessionReadTarget::ActiveBranch,
        )
        .await?;
    assert!(messages.is_empty());
    Ok(())
}

#[tokio::test]
async fn test_run_stops_when_assistant_message_persistence_fails() -> Result<()> {
    let tmp = tempdir()?;
    let mut kernel = make_kernel(tmp.path()).await?;
    let mut session = kernel.create_session().await;
    let store = kernel.store_manager().open(&session.store_selector).await?;
    let conn = store.get_connection().await?;
    conn.execute_batch(
        r#"
        CREATE TRIGGER fail_assistant_message_insert
        BEFORE INSERT ON messages
        WHEN NEW.role = 'assistant'
        BEGIN
            SELECT RAISE(FAIL, 'injected assistant persistence failure');
        END;
        "#,
    )
    .await?;

    let error = kernel
        .run(&mut session, Some("Persist my response".to_string()))
        .await
        .expect_err("durable task should stop when its assistant message cannot be stored");
    assert!(
        error
            .to_string()
            .contains("Failed to persist assistant turn message")
    );
    assert_eq!(session.history.len(), 1);

    let messages = store
        .get_messages(
            session.internal_id.expect("persisted session"),
            &SessionReadTarget::ActiveBranch,
        )
        .await?;
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0].role, "user");
    Ok(())
}

#[tokio::test]
async fn test_run_reports_background_event_persistence_failure() -> Result<()> {
    let tmp = tempdir()?;
    let mut kernel = make_kernel(tmp.path()).await?;
    let mut session = kernel.create_session().await;
    let store = kernel.store_manager().open(&session.store_selector).await?;
    let conn = store.get_connection().await?;
    conn.execute_batch(
        r#"
        CREATE TRIGGER fail_event_insert
        BEFORE INSERT ON events
        BEGIN
            SELECT RAISE(FAIL, 'injected event persistence failure');
        END;
        "#,
    )
    .await?;

    let original_execution = session.execution.clone();
    let mut task = QueuedTask::ad_hoc("Persist transcript and events");
    task.task_id = "t_durability_failure".to_string();
    task.execution = Some(TaskExecutionOverrides {
        visibility: Some(ExecutionVisibility::Hidden),
        ..TaskExecutionOverrides::default()
    });
    session.queue.lock().await.push_back(task);

    let error = kernel
        .run(&mut session, None)
        .await
        .expect_err("task barrier should report background event write failure");
    assert!(error.to_string().contains("Event durability write failed"));
    assert_eq!(session.execution, original_execution);

    let messages = store
        .get_messages(
            session.internal_id.expect("persisted session"),
            &SessionReadTarget::ActiveBranch,
        )
        .await?;
    assert_eq!(messages.len(), 2);

    conn.execute("DROP TRIGGER fail_event_insert", ()).await?;
    kernel
        .run(
            &mut session,
            Some("The durability lane should recover".to_string()),
        )
        .await?;
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

#[tokio::test]
async fn test_harness_module_locals_are_isolated_per_live_session() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir_all(&harness_dir)?;
    std::fs::write(
        harness_dir.join("session_counter.lua"),
        r#"
            local task_counter = 0

            function on_task_start(event)
                task_counter = task_counter + 1
                return MODIFY, {
                    prompt = event.prompt .. " [session_counter=" .. tostring(task_counter) .. "]"
                }
            end
        "#,
    )?;

    let mut config = make_config(tmp.path());
    config.harness.directory = harness_dir.to_string_lossy().to_string();

    let mut kernel = turin_harness_lua::runtime_builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let seen_messages = Arc::new(Mutex::new(Vec::new()));
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new(
            "mock",
            Arc::new(CaptureMessagesProvider {
                seen_messages: Arc::clone(&seen_messages),
            }),
        ),
    );

    let mut session1 = kernel.create_session().await;
    kernel
        .run(&mut session1, Some("Investigate alpha".to_string()))
        .await?;

    let first_seen = seen_messages
        .lock()
        .expect("capture messages mutex poisoned")
        .clone();
    let first_user_message = first_seen
        .iter()
        .rev()
        .find(|message| message.role == turin_core::inference::provider::InferenceRole::User)
        .expect("captured first session user message");
    assert!(matches!(
        &first_user_message.content[0],
        turin_core::inference::provider::InferenceContent::Text { text }
        if text == "Investigate alpha [session_counter=1]"
    ));

    let mut session2 = kernel.create_session().await;
    kernel
        .run(&mut session2, Some("Investigate beta".to_string()))
        .await?;

    let second_seen = seen_messages
        .lock()
        .expect("capture messages mutex poisoned")
        .clone();
    let second_user_message = second_seen
        .iter()
        .rev()
        .find(|message| message.role == turin_core::inference::provider::InferenceRole::User)
        .expect("captured second session user message");
    assert!(matches!(
        &second_user_message.content[0],
        turin_core::inference::provider::InferenceContent::Text { text }
        if text == "Investigate beta [session_counter=1]"
    ));

    kernel.end_session(&mut session1).await?;
    kernel.end_session(&mut session2).await?;
    Ok(())
}

#[tokio::test]
async fn test_local_branch_selection_does_not_mutate_persisted_active_head() -> Result<()> {
    let tmp = tempdir()?;
    let mut kernel = make_kernel(tmp.path()).await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Investigate main line".to_string()))
        .await?;

    let store = kernel.store_manager().open(&session.store_selector).await?;
    let session_id = session.internal_id.expect("session internal id");
    let main_head = store
        .get_active_branch_head(session_id)
        .await?
        .expect("main head");
    let alt_head = store
        .create_branch_head_from_turn_index(session_id, "alt", Some(0), false)
        .await?;
    store
        .insert_message(
            session_id,
            TurnWriteTarget::branch_head(Some(alt_head.id), 1),
            "assistant",
            &serde_json::json!([{"type": "text", "text": "ALT PATH ONLY"}]),
            None,
        )
        .await?;

    let execution_id = session.execution_id().to_string();
    let switched = kernel
        .select_session_branch_by_name_local(&mut session, "alt")
        .await?;
    assert!(switched, "expected local branch selection to succeed");
    assert_eq!(session.selected_branch_head_id(), Some(alt_head.id));
    assert_eq!(
        session.execution.write_policy,
        ExecutionWritePolicy::AdvanceBranchHead
    );

    let has_alt_message = session.history.iter().any(|message| {
        message.content.iter().any(|content| {
            matches!(
                content,
                turin_core::inference::provider::InferenceContent::Text { text }
                if text == "ALT PATH ONLY"
            )
        })
    });
    assert!(
        has_alt_message,
        "local branch selection should materialize the alternate branch history"
    );

    let persisted_active = store
        .get_active_branch_head(session_id)
        .await?
        .expect("persisted active head");
    assert_eq!(
        persisted_active.id, main_head.id,
        "local branch selection should not mutate the persisted active head"
    );
    assert_eq!(
        session.execution_id(),
        execution_id,
        "local branch selection should not create a new execution context"
    );

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test]
async fn test_local_branch_selection_rejects_checkpoint_from_sibling_path() -> Result<()> {
    let tmp = tempdir()?;
    let mut kernel = make_kernel(tmp.path()).await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Shared root".to_string()))
        .await?;

    let store = kernel.store_manager().open(&session.store_selector).await?;
    let session_id = session.internal_id.expect("session internal id");
    let alt_head = store
        .create_branch_head_from_turn_index(session_id, "alt", Some(0), false)
        .await?;

    kernel
        .run(&mut session, Some("Main path only".to_string()))
        .await?;
    let main_head = store
        .get_active_branch_head(session_id)
        .await?
        .and_then(|branch| branch.head_turn_id)
        .expect("main head turn");
    let checkpoint = turin_core::kernel::session::ContextCompactionCheckpoint {
        summary: "MAIN PATH SUMMARY".to_string(),
        covered_through_turn_id: main_head,
        covered_through_turn_index: 1,
        generated_at_turn_index: session.turn_index,
        provider_name: "mock".to_string(),
        model: "mock-model".to_string(),
    };
    let event = turin_core::kernel::event::KernelEvent::Audit(
        turin_core::kernel::event::AuditEvent::ContextCompaction { checkpoint },
    );
    store
        .insert_event(
            session_id,
            None,
            "context_compaction",
            &serde_json::to_value(event)?,
        )
        .await?;

    store
        .insert_message(
            session_id,
            TurnWriteTarget::branch_head(Some(alt_head.id), 1),
            "assistant",
            &serde_json::json!([{"type": "text", "text": "ALT PATH ONLY"}]),
            None,
        )
        .await?;

    assert!(
        kernel
            .select_session_branch_by_name_local(&mut session, "alt")
            .await?
    );
    assert!(
        session.context_checkpoint.is_none(),
        "a sibling-path checkpoint must not compact the selected branch"
    );

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test]
async fn test_refresh_skips_corrupted_compaction_event_and_restores_latest_valid() -> Result<()> {
    let tmp = tempdir()?;
    let mut kernel = make_kernel(tmp.path()).await?;
    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Stable history".to_string()))
        .await?;

    let store = kernel.store_manager().open(&session.store_selector).await?;
    let session_id = session.internal_id.expect("session internal id");
    let head_turn_id = store
        .get_active_branch_head(session_id)
        .await?
        .and_then(|branch| branch.head_turn_id)
        .expect("active head turn");
    let checkpoint = turin_core::kernel::session::ContextCompactionCheckpoint {
        summary: "VALID SUMMARY".to_string(),
        covered_through_turn_id: head_turn_id,
        covered_through_turn_index: 0,
        generated_at_turn_index: session.turn_index,
        provider_name: "mock".to_string(),
        model: "mock-model".to_string(),
    };
    let valid_event = turin_core::kernel::event::KernelEvent::Audit(
        turin_core::kernel::event::AuditEvent::ContextCompaction {
            checkpoint: checkpoint.clone(),
        },
    );
    store
        .insert_event(
            session_id,
            None,
            "context_compaction",
            &serde_json::to_value(valid_event)?,
        )
        .await?;
    store
        .insert_event(
            session_id,
            None,
            "context_compaction",
            &serde_json::json!({"not": "a kernel event"}),
        )
        .await?;

    let original_history = format!("{:?}", session.history);
    let original_has_prior_history = session.history.has_prior_history();
    let original_target = session.context_target().clone();
    let original_turn_index = session.turn_index;
    kernel
        .refresh_session_from_persistence(&mut session)
        .await?;
    assert_eq!(format!("{:?}", session.history), original_history);
    assert_eq!(
        session.history.has_prior_history(),
        original_has_prior_history
    );
    assert_eq!(session.context_target(), &original_target);
    assert_eq!(session.turn_index, original_turn_index);
    assert_eq!(session.context_checkpoint.as_ref(), Some(&checkpoint));

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test]
async fn test_local_turn_selection_materializes_prefix_without_new_execution() -> Result<()> {
    let tmp = tempdir()?;
    let mut kernel = make_kernel(tmp.path()).await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Turn zero".to_string()))
        .await?;
    kernel
        .run(&mut session, Some("Turn one".to_string()))
        .await?;

    let store = kernel.store_manager().open(&session.store_selector).await?;
    let session_id = session.internal_id.expect("session internal id");
    let active_head = store
        .get_active_branch_head(session_id)
        .await?
        .expect("active head");
    let first_turn_branch = store
        .create_branch_head_from_turn_index(session_id, "checkpoint", Some(0), false)
        .await?;
    let first_turn_id = first_turn_branch
        .created_from_turn_id
        .expect("checkpoint source turn id");

    let execution_id = session.execution_id().to_string();
    let switched = kernel
        .select_session_turn_local(&mut session, first_turn_id)
        .await?;
    assert!(switched, "expected local turn selection to succeed");
    assert_eq!(session.selected_turn_id(), Some(first_turn_id));
    assert_eq!(session.selected_branch_head_id(), None);
    assert_eq!(session.execution_id(), execution_id);
    assert_eq!(session.turn_index, 1);
    assert_eq!(
        session.execution.write_policy,
        ExecutionWritePolicy::Detached
    );

    let seen_turn_zero = session.history.iter().any(|message| {
        message.content.iter().any(|content| {
            matches!(
                content,
                turin_core::inference::provider::InferenceContent::Text { text }
                if text == "Turn zero"
            )
        })
    });
    let seen_turn_one = session.history.iter().any(|message| {
        message.content.iter().any(|content| {
            matches!(
                content,
                turin_core::inference::provider::InferenceContent::Text { text }
                if text == "Turn one"
            )
        })
    });
    assert!(
        seen_turn_zero,
        "turn-target materialization should retain earlier context"
    );
    assert!(
        !seen_turn_one,
        "turn-target materialization should stop before later turns"
    );

    let persisted_active = store
        .get_active_branch_head(session_id)
        .await?
        .expect("persisted active head");
    assert_eq!(persisted_active.id, active_head.id);

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test]
async fn test_local_external_reference_selection_materializes_remote_context_detached() -> Result<()>
{
    let tmp = tempdir()?;
    let mut kernel = make_kernel(tmp.path()).await?;

    let mut source = kernel.create_session().await;
    kernel
        .run(&mut source, Some("Remote context".to_string()))
        .await?;

    let mut session = kernel.create_session().await;
    let execution_id = session.execution_id().to_string();
    let source_reference = source.identity.session_id().to_string();

    let switched = kernel
        .select_session_external_reference_local(&mut session, &source_reference)
        .await?;
    assert!(
        switched,
        "expected local external reference selection to succeed"
    );
    assert_eq!(session.execution_id(), execution_id);
    assert_eq!(session.selected_branch_head_id(), None);
    assert_eq!(
        session.execution.write_policy,
        ExecutionWritePolicy::Detached
    );
    let turin_core::kernel::session::ExecutionContextTarget::ExternalReference { reference } =
        session.context_target()
    else {
        panic!("expected external reference context target");
    };
    let parsed_reference = parse_session_reference(reference)?;
    assert_eq!(parsed_reference.public_id, source_reference);
    assert!(
        session
            .history
            .iter()
            .any(|message| message.content.iter().any(|content| {
                matches!(
                    content,
                    turin_core::inference::provider::InferenceContent::Text { text }
                    if text == "Remote context"
                )
            })),
        "external reference selection should materialize the referenced session history"
    );

    kernel
        .run(&mut session, Some("Detached follow-up".to_string()))
        .await?;

    let store = kernel.store_manager().open(&session.store_selector).await?;
    let persisted_messages = store
        .get_messages(
            session.internal_id.expect("session internal id"),
            &SessionReadTarget::ActiveBranch,
        )
        .await?;
    assert!(
        persisted_messages.is_empty(),
        "detached external-reference execution should not mutate the local transcript"
    );

    let persisted_events = store
        .get_events(
            session.internal_id.expect("session internal id"),
            &SessionReadTarget::ActiveBranch,
        )
        .await?;
    assert!(
        event_has_task_status(&persisted_events, "success"),
        "detached external-reference execution should still persist task completion"
    );
    assert_eq!(
        count_event_type(&persisted_events, "turn_start"),
        0,
        "branch-scoped turn events should not persist for detached executions"
    );

    kernel.end_session(&mut session).await?;
    kernel.end_session(&mut source).await?;
    Ok(())
}

#[tokio::test]
async fn test_task_execution_override_materializes_temp_context_and_restores_session() -> Result<()>
{
    let tmp = tempdir()?;
    let mut kernel = make_kernel(tmp.path()).await?;

    let seen_messages = Arc::new(Mutex::new(Vec::new()));
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new(
            "mock",
            Arc::new(CaptureMessagesProvider {
                seen_messages: Arc::clone(&seen_messages),
            }),
        ),
    );

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Turn zero".to_string()))
        .await?;
    kernel
        .run(&mut session, Some("Turn one".to_string()))
        .await?;

    let original_execution_id = session.execution_id().to_string();
    let original_target = session.context_target().clone();
    let original_branch_head_id = session.selected_branch_head_id();

    let store = kernel.store_manager().open(&session.store_selector).await?;
    let session_id = session.internal_id.expect("session internal id");
    let checkpoint = store
        .create_branch_head_from_turn_index(session_id, "checkpoint", Some(0), false)
        .await?;
    let first_turn_id = checkpoint
        .created_from_turn_id
        .expect("checkpoint source turn id");

    let queued_task =
        QueuedTask::ad_hoc("Revisit old context").with_execution(Some(TaskExecutionOverrides {
            context_target: Some(ExecutionContextTarget::TurnId {
                turn_id: first_turn_id,
            }),
            visibility: None,
            durability: None,
            write_policy: None,
        }));
    {
        let mut queue = session.queue.lock().await;
        queue.push_back(queued_task);
    }

    kernel.run(&mut session, None).await?;

    let seen = seen_messages
        .lock()
        .expect("capture messages mutex poisoned")
        .clone();
    let rendered = format!("{seen:?}");
    assert!(
        rendered.contains("Turn zero"),
        "task override should materialize the selected turn context"
    );
    assert!(
        !rendered.contains("Turn one"),
        "task override should not include later turns outside the selected target"
    );
    assert!(
        rendered.contains("Revisit old context"),
        "task override should still include the queued task prompt"
    );

    assert_eq!(session.execution_id(), original_execution_id);
    assert_eq!(session.context_target(), &original_target);
    assert_eq!(session.selected_branch_head_id(), original_branch_head_id);
    assert_eq!(session.turn_index, 2);
    assert!(
        session
            .history
            .iter()
            .any(|message| message.content.iter().any(|content| {
                matches!(
                    content,
                    turin_core::inference::provider::InferenceContent::Text { text }
                    if text == "Turn one"
                )
            })),
        "restored session should return to the original visible history"
    );
    assert!(
        !session
            .history
            .iter()
            .any(|message| message.content.iter().any(|content| {
                matches!(
                    content,
                    turin_core::inference::provider::InferenceContent::Text { text }
                    if text == "Revisit old context"
                )
            })),
        "detached task execution should not durably mutate the restored session transcript"
    );

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test]
async fn test_resumed_live_sessions_share_persistence_lock() -> Result<()> {
    let tmp = tempdir()?;
    let kernel = make_kernel(tmp.path()).await?;

    let session = kernel.create_session().await;
    let session_id = session.identity.session_id().to_string();
    let resumed = kernel
        .resume_session_for_agent("default", &session_id)
        .await?;

    assert!(std::sync::Arc::ptr_eq(
        &session.persistence_lock,
        &resumed.persistence_lock
    ));

    Ok(())
}

#[tokio::test]
async fn test_resumed_live_session_gets_distinct_execution_id() -> Result<()> {
    let tmp = tempdir()?;
    let kernel = make_kernel(tmp.path()).await?;

    let session = kernel.create_session().await;
    let session_id = session.identity.session_id().to_string();
    let resumed = kernel
        .resume_session_for_agent("default", &session_id)
        .await?;

    assert_eq!(session.identity.session_id(), resumed.identity.session_id());
    assert_ne!(session.execution_id(), resumed.execution_id());

    Ok(())
}

#[tokio::test]
async fn test_resume_advances_past_allocated_turn_without_messages() -> Result<()> {
    let tmp = tempdir()?;
    let kernel = make_kernel(tmp.path()).await?;
    let session = kernel.create_session().await;
    let session_id = session.identity.session_id().to_string();
    let store = kernel.store_manager().open(&session.store_selector).await?;
    store
        .prepare_turn_write_target(
            session.internal_id.expect("persisted session"),
            TurnWriteTarget::active_branch(0),
        )
        .await?
        .expect("allocated turn");

    let resumed = kernel
        .resume_session_for_agent("default", &session_id)
        .await?;
    assert!(resumed.history.is_empty());
    assert_eq!(resumed.turn_index, 1);
    Ok(())
}

#[tokio::test]
async fn test_resume_preserves_user_only_partial_turn_without_replay() -> Result<()> {
    let tmp = tempdir()?;
    let mut kernel = make_kernel(tmp.path()).await?;
    let session = kernel.create_session().await;
    let session_id = session.identity.session_id().to_string();
    let internal_id = session.internal_id.expect("persisted session");
    let store = kernel.store_manager().open(&session.store_selector).await?;
    store
        .insert_message(
            internal_id,
            TurnWriteTarget::active_branch(0),
            "user",
            &serde_json::json!([{"type": "text", "text": "Committed before crash"}]),
            None,
        )
        .await?;

    let mut resumed = kernel
        .resume_session_for_agent("default", &session_id)
        .await?;
    assert_eq!(resumed.turn_index, 1);
    assert_eq!(resumed.history.len(), 1);
    assert!(resumed.history.iter().any(|message| {
        message.content.iter().any(|content| {
            matches!(
                content,
                turin_core::inference::provider::InferenceContent::Text { text }
                if text == "Committed before crash"
            )
        })
    }));

    kernel
        .run(&mut resumed, Some("Continue after crash".to_string()))
        .await?;
    assert_eq!(resumed.turn_index, 2);
    let messages = store
        .get_messages(internal_id, &SessionReadTarget::ActiveBranch)
        .await?;
    assert!(
        messages
            .iter()
            .any(|message| message.turn_index == 0 && message.role == "user")
    );
    assert!(
        messages
            .iter()
            .any(|message| message.turn_index == 1 && message.role == "user")
    );

    kernel.end_session(&mut resumed).await?;
    Ok(())
}

#[tokio::test]
async fn test_on_turn_prepare_exposes_estimated_tokens_and_context_limit() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir_all(&harness_dir)?;
    std::fs::write(
        harness_dir.join("token_probe.lua"),
        r#"
            function on_turn_prepare(ctx)
                if ctx.token_count < 1 then
                    error("expected token_count to be estimated")
                end
                if ctx.estimated_input_tokens ~= ctx.token_count then
                    error("expected estimated_input_tokens alias to match token_count")
                end
                if ctx.token_limit ~= 2048 then
                    error("expected token_limit to come from provider context_window_tokens")
                end
                if ctx.max_input_tokens ~= ctx.token_limit then
                    error("expected max_input_tokens alias to match token_limit")
                end
                return ALLOW
            end
        "#,
    )?;

    let mut config = make_config(tmp.path());
    config.harness.directory = harness_dir.to_string_lossy().to_string();
    if let Some(provider) = config.providers.get_mut("mock") {
        provider.context_window_tokens = Some(2048);
    }

    let mut kernel = turin_harness_lua::runtime_builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Probe token estimates".to_string()))
        .await?;

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_on_turn_prepare_structured_output_uses_native_response_format() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir_all(&harness_dir)?;
    std::fs::write(
        harness_dir.join("structured.lua"),
        r#"
            function on_turn_prepare(ctx)
                local result = ctx:structured({
                    prompt = "Review this change",
                    name = "review_result",
                    schema = {
                        type = "object",
                        properties = {
                            approved = { type = "boolean" },
                            summary = { type = "string" },
                        },
                        required = { "approved", "summary" },
                        additionalProperties = false,
                    },
                })

                if result.approved ~= true then
                    error("expected approved=true from structured result")
                end

                if result.summary ~= "structured ok" then
                    error("expected structured summary to round-trip")
                end

                return ALLOW
            end
        "#,
    )?;

    let mut config = make_config(tmp.path());
    config.harness.directory = harness_dir.to_string_lossy().to_string();
    let mut kernel = turin_harness_lua::runtime_builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let seen_response_format = Arc::new(Mutex::new(None));
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new(
            "mock",
            Arc::new(StructuredOutputProvider {
                seen_response_format: Arc::clone(&seen_response_format),
            }),
        ),
    );

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Use structured output".to_string()))
        .await?;

    let seen = seen_response_format
        .lock()
        .expect("structured provider mutex poisoned")
        .clone()
        .expect("structured response format should have been recorded");

    match seen {
        InferenceResponseFormat::JsonSchema { json_schema } => {
            assert_eq!(json_schema.name, "review_result");
            assert_eq!(
                json_schema.schema["required"],
                serde_json::json!(["approved", "summary"])
            );
        }
        other => panic!("expected json_schema response format, got {other:?}"),
    }

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_on_turn_prepare_structured_output_falls_back_to_prompt_and_validate() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir_all(&harness_dir)?;
    std::fs::write(
        harness_dir.join("structured_fallback.lua"),
        r#"
            function on_turn_prepare(ctx)
                local result = ctx:structured({
                    prompt = "Classify this",
                    name = "classification_result",
                    description = "Classify the request for routing",
                    schema = {
                        type = "object",
                        properties = {
                            decision = { type = "string", enum = { "fallback", "ignore" } },
                            confidence = { type = "number" },
                        },
                        required = { "decision", "confidence" },
                        additionalProperties = false,
                    },
                })

                if result.decision ~= "fallback" then
                    error("expected fallback decision from structured result")
                end

                if result.confidence ~= 0.8 then
                    error("expected structured confidence to round-trip")
                end

                return ALLOW
            end
        "#,
    )?;

    let mut config = make_config(tmp.path());
    config.harness.directory = harness_dir.to_string_lossy().to_string();
    let mut kernel = turin_harness_lua::runtime_builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let seen_system_prompt = Arc::new(Mutex::new(None));
    let seen_response_format = Arc::new(Mutex::new(None));
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new(
            "mock",
            Arc::new(PromptStructuredFallbackProvider {
                seen_system_prompt: Arc::clone(&seen_system_prompt),
                seen_response_format: Arc::clone(&seen_response_format),
            }),
        ),
    );

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Use structured output".to_string()))
        .await?;

    let seen_system = seen_system_prompt
        .lock()
        .expect("structured fallback system mutex poisoned")
        .clone()
        .expect("fallback system prompt should have been recorded");
    assert!(seen_system.contains("Return a single valid JSON value"));
    assert!(seen_system.contains("Schema name: classification_result"));
    assert!(seen_system.contains("Classify the request for routing"));

    let seen_response_format = seen_response_format
        .lock()
        .expect("structured fallback response format mutex poisoned")
        .clone();
    assert!(
        seen_response_format.is_none(),
        "fallback path should not send a native response format"
    );

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test]
async fn test_long_history_is_compacted_before_inference_request() -> Result<()> {
    let tmp = tempdir()?;
    let mut config = make_config(tmp.path());
    config.agent.provider = "capture".to_string();
    config.agent.model = "capture-model".to_string();
    config.agent.system_prompt = "Compact long histories".to_string();
    config.providers.insert(
        "capture".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            base_url: None,
            context_window_tokens: Some(512),
            ..ProviderConfig::default()
        },
    );

    let mut kernel = turin_harness_lua::runtime_builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_harness().await?;

    let seen_messages = Arc::new(Mutex::new(Vec::new()));
    kernel.add_client(
        "capture".to_string(),
        ProviderClient::new(
            "capture",
            Arc::new(CaptureMessagesProvider {
                seen_messages: Arc::clone(&seen_messages),
            }),
        ),
    );

    let mut session = kernel.create_session().await;
    session
        .history
        .push(turin_core::inference::provider::InferenceMessage {
            role: turin_core::inference::provider::InferenceRole::User,
            content: vec![turin_core::inference::provider::InferenceContent::Text {
                text: "Old prompt".to_string(),
            }],
            tool_call_id: None,
        });
    session
        .history
        .push(turin_core::inference::provider::InferenceMessage {
            role: turin_core::inference::provider::InferenceRole::Tool,
            content: vec![
                turin_core::inference::provider::InferenceContent::ToolResult {
                    tool_use_id: "tool_1".to_string(),
                    content: "x".repeat(4_000),
                    is_error: false,
                },
            ],
            tool_call_id: None,
        });
    session
        .history
        .push(turin_core::inference::provider::InferenceMessage {
            role: turin_core::inference::provider::InferenceRole::Assistant,
            content: vec![turin_core::inference::provider::InferenceContent::Text {
                text: "Recent assistant context".to_string(),
            }],
            tool_call_id: None,
        });

    kernel
        .run(&mut session, Some("Current compacted prompt".to_string()))
        .await?;

    let seen = seen_messages
        .lock()
        .expect("capture messages mutex poisoned")
        .clone();
    let rendered = format!("{seen:?}");
    assert!(
        rendered.contains("[tool result omitted to fit context window]")
            || seen.len() < session.history.len(),
        "expected preflight compaction to truncate tool results or drop old messages"
    );
    assert!(
        rendered.contains("Current compacted prompt"),
        "expected latest prompt to remain in the inference request"
    );

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test]
async fn test_resume_keeps_bounded_history_while_inference_reads_a_larger_context() -> Result<()> {
    let tmp = tempdir()?;
    let mut config = make_config(tmp.path());
    config.agent.provider = "capture".to_string();
    config.agent.model = "capture-model".to_string();
    config.inference.hot_history.max_messages = Some(4);
    config.inference.compaction.mode =
        turin_core::kernel::config::InferenceCompactionMode::TrimOnly;
    config.providers.insert(
        "capture".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            base_url: None,
            context_window_tokens: Some(32_768),
            ..ProviderConfig::default()
        },
    );

    let mut kernel = turin_harness_lua::runtime_builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_harness().await?;

    let seen_messages = Arc::new(Mutex::new(Vec::new()));
    kernel.add_client(
        "capture".to_string(),
        ProviderClient::new(
            "capture",
            Arc::new(CaptureMessagesProvider {
                seen_messages: Arc::clone(&seen_messages),
            }),
        ),
    );

    let mut session = kernel.create_session().await;
    for i in 0..8 {
        kernel
            .run(&mut session, Some(format!("Persisted context {i}")))
            .await?;
    }
    let session_id = session.identity.session_id().to_string();
    kernel.end_session(&mut session).await?;

    let mut resumed = kernel
        .resume_session_for_agent("default", &session_id)
        .await?;
    assert!(resumed.history.has_prior_history());
    assert!(
        resumed.history.len() <= 4,
        "resume should materialize only the configured resident history window"
    );
    assert!(
        !format!("{:?}", resumed.history.messages()).contains("Persisted context 0"),
        "the oldest persisted turn should not remain resident"
    );

    kernel
        .run(
            &mut resumed,
            Some("Use the full token-bounded context".to_string()),
        )
        .await?;

    let rendered_request = format!(
        "{:?}",
        seen_messages
            .lock()
            .expect("capture messages mutex poisoned")
    );
    assert!(
        rendered_request.contains("Persisted context 0"),
        "inference should retrieve older persisted turns beyond the resident window: {rendered_request}"
    );
    assert!(rendered_request.contains("Use the full token-bounded context"));

    kernel.end_session(&mut resumed).await?;
    Ok(())
}

#[tokio::test]
async fn test_complete_current_resident_history_bypasses_context_rematerialization() -> Result<()> {
    let tmp = tempdir()?;
    let mut config = make_config(tmp.path());
    config.agent.provider = "capture".to_string();
    config.agent.model = "capture-model".to_string();
    config.inference.compaction.mode =
        turin_core::kernel::config::InferenceCompactionMode::TrimOnly;
    config.providers.insert(
        "capture".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            base_url: None,
            context_window_tokens: Some(32_768),
            ..ProviderConfig::default()
        },
    );

    let mut kernel = turin_harness_lua::runtime_builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_harness().await?;

    let seen_messages = Arc::new(Mutex::new(Vec::new()));
    kernel.add_client(
        "capture".to_string(),
        ProviderClient::new(
            "capture",
            Arc::new(CaptureMessagesProvider {
                seen_messages: Arc::clone(&seen_messages),
            }),
        ),
    );

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Persisted original".to_string()))
        .await?;
    let resident_text = session
        .history
        .iter_mut()
        .flat_map(|message| message.content.iter_mut())
        .find_map(|content| match content {
            turin_core::inference::provider::InferenceContent::Text { text }
                if text == "Persisted original" =>
            {
                Some(text)
            }
            _ => None,
        })
        .expect("expected persisted prompt in resident history");
    *resident_text = "Resident fast-path marker".to_string();

    kernel
        .run(&mut session, Some("Next prompt".to_string()))
        .await?;

    let rendered_request = format!(
        "{:?}",
        seen_messages
            .lock()
            .expect("capture messages mutex poisoned")
    );
    assert!(rendered_request.contains("Resident fast-path marker"));
    assert!(!rendered_request.contains("Persisted original"));

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test]
async fn test_auto_compaction_creates_and_restores_context_checkpoint() -> Result<()> {
    let tmp = tempdir()?;
    let mut config = make_config(tmp.path());
    config.agent.provider = "checkpoint".to_string();
    config.agent.model = "checkpoint-model".to_string();
    config.agent.system_prompt = "Auto-compact long histories".to_string();
    config.inference.compaction.trigger_ratio = 0.1;
    config.providers.insert(
        "checkpoint".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            base_url: None,
            context_window_tokens: Some(8_192),
            ..ProviderConfig::default()
        },
    );

    let mut kernel = turin_harness_lua::runtime_builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_harness().await?;

    let seen_stream_messages = Arc::new(Mutex::new(Vec::new()));
    let seen_stream_system = Arc::new(Mutex::new(None));
    let complete_calls = Arc::new(Mutex::new(0usize));
    kernel.add_client(
        "checkpoint".to_string(),
        ProviderClient::new(
            "checkpoint",
            Arc::new(ContextCheckpointProvider {
                seen_stream_messages: Arc::clone(&seen_stream_messages),
                seen_stream_system: Arc::clone(&seen_stream_system),
                complete_calls: Arc::clone(&complete_calls),
            }),
        ),
    );

    let mut session = kernel.create_session().await;
    for i in 0..12 {
        kernel
            .run(
                &mut session,
                Some(format!("Historic prompt {i}: {}", "x".repeat(240))),
            )
            .await?;
    }

    kernel
        .run(
            &mut session,
            Some("Newest prompt after checkpoint".to_string()),
        )
        .await?;

    let checkpoint = session
        .context_checkpoint
        .clone()
        .expect("expected context checkpoint to be generated");
    assert_eq!(checkpoint.summary, "CHECKPOINT SUMMARY");
    assert!(
        checkpoint.covered_through_turn_id > 0,
        "expected checkpoint to cover earlier history"
    );
    assert!(
        *complete_calls
            .lock()
            .expect("context checkpoint complete mutex poisoned")
            >= 1,
        "expected at least one semantic compaction completion call"
    );

    let stream_system = seen_stream_system
        .lock()
        .expect("context checkpoint system mutex poisoned")
        .clone()
        .expect("expected system prompt to be sent");
    assert!(
        stream_system.contains("CHECKPOINT SUMMARY"),
        "expected final inference request to include the checkpoint summary in the system prompt"
    );
    let seen_messages = seen_stream_messages
        .lock()
        .expect("context checkpoint messages mutex poisoned")
        .clone();
    let rendered = format!("{seen_messages:?}");
    assert!(
        !rendered.contains("Historic prompt 0"),
        "expected covered history to be omitted from the final request messages"
    );
    assert!(
        rendered.contains("Newest prompt after checkpoint"),
        "expected the newest prompt to remain in the final request"
    );

    let session_id = session.identity.session_id().to_string();
    kernel.end_session(&mut session).await?;

    let resumed = kernel
        .resume_session_for_agent("default", &session_id)
        .await?;
    let resumed_checkpoint = resumed
        .context_checkpoint
        .expect("expected context checkpoint to restore from persistence");
    assert_eq!(resumed_checkpoint.summary, "CHECKPOINT SUMMARY");

    Ok(())
}

#[tokio::test]
async fn test_multimodal_task_content_persists_and_restores() -> Result<()> {
    let tmp = tempdir()?;
    let mut config = make_config(tmp.path());
    config.agent.provider = "capture".to_string();
    config.agent.model = "capture-model".to_string();
    config.providers.insert(
        "capture".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            base_url: None,
            ..ProviderConfig::default()
        },
    );

    let mut kernel = turin_harness_lua::runtime_builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_harness().await?;

    let seen_messages = Arc::new(Mutex::new(Vec::new()));
    kernel.add_client(
        "capture".to_string(),
        ProviderClient::new(
            "capture",
            Arc::new(CaptureMessagesProvider {
                seen_messages: Arc::clone(&seen_messages),
            }),
        ),
    );

    let source_image = tmp.path().join("diagram.png");
    let source_file = tmp.path().join("spec.pdf");
    std::fs::write(&source_image, [1_u8, 2, 3, 4])?;
    std::fs::write(&source_file, b"spec body")?;

    let mut session = kernel.create_session().await;
    let session_id = session.identity.session_id().to_string();

    let mut task = QueuedTask::ad_hoc("[attachments]");
    task.content = Some(vec![
        TaskInputContent::Text {
            text: "Inspect the attached files".to_string(),
        },
        TaskInputContent::Image {
            name: Some("diagram.png".to_string()),
            content_type: Some("image/png".to_string()),
            url: None,
            local_path: Some(source_image.display().to_string()),
            detail: Some("high".to_string()),
        },
        TaskInputContent::File {
            name: Some("spec.pdf".to_string()),
            content_type: Some("application/pdf".to_string()),
            url: None,
            local_path: Some(source_file.display().to_string()),
        },
    ]);
    session.queue.lock().await.push_back(task);

    kernel.run(&mut session, None).await?;

    let seen = seen_messages
        .lock()
        .expect("capture messages mutex poisoned")
        .clone();
    let user_message = seen
        .iter()
        .rev()
        .find(|message| message.role == turin_core::inference::provider::InferenceRole::User)
        .expect("captured user message");
    assert!(matches!(
        &user_message.content[0],
        turin_core::inference::provider::InferenceContent::Text { text }
        if text == "Inspect the attached files"
    ));

    let captured_image_path = match &user_message.content[1] {
        turin_core::inference::provider::InferenceContent::Image { local_path, .. } => {
            local_path.clone().expect("captured image local_path")
        }
        other => panic!("expected image content, got {other:?}"),
    };
    assert!(std::path::Path::new(&captured_image_path).exists());
    assert!(
        captured_image_path.starts_with(
            tmp.path()
                .join(".turin/data/media")
                .to_string_lossy()
                .as_ref()
        ),
        "captured image should be stored under the workspace media dir: {captured_image_path}"
    );

    let captured_file_path = match &user_message.content[2] {
        turin_core::inference::provider::InferenceContent::File { local_path, .. } => {
            local_path.clone().expect("captured file local_path")
        }
        other => panic!("expected file content, got {other:?}"),
    };
    assert!(std::path::Path::new(&captured_file_path).exists());
    assert!(
        captured_file_path.starts_with(
            tmp.path()
                .join(".turin/data/media")
                .to_string_lossy()
                .as_ref()
        ),
        "captured file should be stored under the workspace media dir: {captured_file_path}"
    );

    kernel.end_session(&mut session).await?;

    let resumed = kernel
        .resume_session_for_agent("default", &session_id)
        .await?;
    let resumed_user_message = resumed
        .history
        .iter()
        .find(|message| message.role == turin_core::inference::provider::InferenceRole::User)
        .expect("resumed user message");

    assert!(matches!(
        &resumed_user_message.content[1],
        turin_core::inference::provider::InferenceContent::Image {
            name: Some(name),
            local_path: Some(path),
            ..
        } if name == "diagram.png" && path == &captured_image_path
    ));
    assert!(matches!(
        &resumed_user_message.content[2],
        turin_core::inference::provider::InferenceContent::File {
            name: Some(name),
            local_path: Some(path),
            ..
        } if name == "spec.pdf" && path == &captured_file_path
    ));

    Ok(())
}

#[tokio::test]
async fn test_tool_transcript_restores_and_continues_after_cold_resume() -> Result<()> {
    let tmp = tempdir()?;
    let mut kernel = make_kernel(tmp.path()).await?;
    std::fs::write(tmp.path().join("test.txt"), "tool transcript body")?;

    let responses = Arc::new(Mutex::new(vec![
        vec![
            InferenceEvent::MessageStart {
                role: "assistant".to_string(),
                model: "sequence-model".to_string(),
                provider_id: "sequence".to_string(),
            },
            InferenceEvent::ToolCallStart {
                id: "call_read_1".to_string(),
                name: "read_file".to_string(),
            },
            InferenceEvent::ToolCallDelta {
                delta: serde_json::json!({"path": "test.txt"}).to_string(),
            },
            InferenceEvent::MessageEnd {
                input_tokens: 10,
                output_tokens: 4,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
                stop_reason: None,
            },
        ],
        vec![
            InferenceEvent::MessageStart {
                role: "assistant".to_string(),
                model: "sequence-model".to_string(),
                provider_id: "sequence".to_string(),
            },
            InferenceEvent::MessageDelta {
                content: "I read it.".to_string(),
            },
            InferenceEvent::MessageEnd {
                input_tokens: 8,
                output_tokens: 3,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
                stop_reason: None,
            },
        ],
        vec![
            InferenceEvent::MessageStart {
                role: "assistant".to_string(),
                model: "sequence-model".to_string(),
                provider_id: "sequence".to_string(),
            },
            InferenceEvent::MessageDelta {
                content: "Continued after resume.".to_string(),
            },
            InferenceEvent::MessageEnd {
                input_tokens: 12,
                output_tokens: 4,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
                stop_reason: None,
            },
        ],
    ]));
    let seen_messages = Arc::new(Mutex::new(Vec::new()));
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new(
            "mock",
            Arc::new(SequenceCaptureProvider {
                responses: Arc::clone(&responses),
                seen_messages: Arc::clone(&seen_messages),
            }),
        ),
    );

    let mut session = kernel.create_session().await;
    let session_id = session.identity.session_id().to_string();
    kernel
        .run(&mut session, Some("Read test.txt".to_string()))
        .await?;

    assert!(
        session.history.iter().any(|message| {
            message.content.iter().any(|content| {
                matches!(
                    content,
                    turin_core::inference::provider::InferenceContent::ToolUse { name, .. }
                    if name == "read_file"
                )
            })
        }),
        "initial run should record the assistant tool call in memory"
    );
    assert!(
        session.history.iter().any(|message| {
            message.content.iter().any(|content| {
                matches!(
                    content,
                    turin_core::inference::provider::InferenceContent::ToolResult { content, .. }
                    if content.contains("tool transcript body")
                )
            })
        }),
        "initial run should record the tool result in memory"
    );

    kernel.end_session(&mut session).await?;

    let mut resumed = kernel
        .resume_session_for_agent("default", &session_id)
        .await?;
    assert!(resumed.restored_from_persistence);
    assert!(
        resumed.history.iter().any(|message| {
            message.content.iter().any(|content| {
                matches!(
                    content,
                    turin_core::inference::provider::InferenceContent::ToolUse { name, .. }
                    if name == "read_file"
                )
            })
        }),
        "cold resume should restore the assistant tool call"
    );
    assert!(
        resumed.history.iter().any(|message| {
            message.content.iter().any(|content| {
                matches!(
                    content,
                    turin_core::inference::provider::InferenceContent::ToolResult { content, .. }
                    if content.contains("tool transcript body")
                )
            })
        }),
        "cold resume should restore the tool result"
    );

    kernel
        .run(
            &mut resumed,
            Some("Continue from restored history".to_string()),
        )
        .await?;

    let seen = seen_messages
        .lock()
        .expect("sequence capture messages mutex poisoned")
        .clone();
    let final_request = seen
        .last()
        .expect("expected resumed continuation inference request");
    let rendered = format!("{final_request:?}");
    assert!(rendered.contains("Read test.txt"));
    assert!(rendered.contains("read_file"));
    assert!(rendered.contains("tool transcript body"));
    assert!(rendered.contains("I read it."));
    assert!(rendered.contains("Continue from restored history"));

    kernel.end_session(&mut resumed).await?;
    Ok(())
}

#[tokio::test]
async fn test_multimodal_task_content_respects_relative_layout_root() -> Result<()> {
    let tmp = tempdir()?;
    let mut config = make_config(tmp.path());
    config.layout.root = Some("runtime-data".to_string());
    config.agent.provider = "capture".to_string();
    config.agent.model = "capture-model".to_string();
    config.providers.insert(
        "capture".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            base_url: None,
            ..ProviderConfig::default()
        },
    );

    let mut kernel = turin_harness_lua::runtime_builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_harness().await?;

    let seen_messages = Arc::new(Mutex::new(Vec::new()));
    kernel.add_client(
        "capture".to_string(),
        ProviderClient::new(
            "capture",
            Arc::new(CaptureMessagesProvider {
                seen_messages: Arc::clone(&seen_messages),
            }),
        ),
    );

    let source_image = tmp.path().join("diagram.png");
    std::fs::write(&source_image, [1_u8, 2, 3, 4])?;

    let mut session = kernel.create_session().await;
    let mut task = QueuedTask::ad_hoc("[attachments]");
    task.content = Some(vec![TaskInputContent::Image {
        name: Some("diagram.png".to_string()),
        content_type: Some("image/png".to_string()),
        url: None,
        local_path: Some(source_image.display().to_string()),
        detail: None,
    }]);
    session.queue.lock().await.push_back(task);

    kernel.run(&mut session, None).await?;

    let seen = seen_messages
        .lock()
        .expect("capture messages mutex poisoned")
        .clone();
    let user_message = seen
        .iter()
        .rev()
        .find(|message| message.role == turin_core::inference::provider::InferenceRole::User)
        .expect("captured user message");

    let captured_image_path = match &user_message.content[0] {
        turin_core::inference::provider::InferenceContent::Image { local_path, .. } => {
            local_path.clone().expect("captured image local_path")
        }
        other => panic!("expected image content, got {other:?}"),
    };
    assert!(
        captured_image_path.starts_with(
            tmp.path()
                .join("runtime-data/data/media")
                .to_string_lossy()
                .as_ref()
        ),
        "captured image should respect the relative layout root: {captured_image_path}"
    );

    Ok(())
}

#[tokio::test]
async fn test_trim_only_compaction_skips_semantic_checkpoint_generation() -> Result<()> {
    let tmp = tempdir()?;
    let mut config = make_config(tmp.path());
    config.agent.provider = "checkpoint".to_string();
    config.agent.model = "checkpoint-model".to_string();
    config.agent.system_prompt = "Trim-only history compaction".to_string();
    config.inference.compaction.mode =
        turin_core::kernel::config::InferenceCompactionMode::TrimOnly;
    config.providers.insert(
        "checkpoint".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            base_url: None,
            context_window_tokens: Some(512),
            ..ProviderConfig::default()
        },
    );

    let mut kernel = turin_harness_lua::runtime_builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_harness().await?;

    let seen_stream_messages = Arc::new(Mutex::new(Vec::new()));
    let seen_stream_system = Arc::new(Mutex::new(None));
    let complete_calls = Arc::new(Mutex::new(0usize));
    kernel.add_client(
        "checkpoint".to_string(),
        ProviderClient::new(
            "checkpoint",
            Arc::new(ContextCheckpointProvider {
                seen_stream_messages: Arc::clone(&seen_stream_messages),
                seen_stream_system: Arc::clone(&seen_stream_system),
                complete_calls: Arc::clone(&complete_calls),
            }),
        ),
    );

    let mut session = kernel.create_session().await;
    for i in 0..12 {
        session
            .history
            .push(turin_core::inference::provider::InferenceMessage {
                role: turin_core::inference::provider::InferenceRole::User,
                content: vec![turin_core::inference::provider::InferenceContent::Text {
                    text: format!("Trim-only history {i}: {}", "x".repeat(240)),
                }],
                tool_call_id: None,
            });
    }

    kernel
        .run(
            &mut session,
            Some("Newest prompt in trim-only mode".to_string()),
        )
        .await?;

    assert!(
        session.context_checkpoint.is_none(),
        "trim_only should not create semantic checkpoints"
    );
    assert_eq!(
        *complete_calls
            .lock()
            .expect("trim-only complete mutex poisoned"),
        0,
        "trim_only should not call provider completion for summarization"
    );

    let stream_system = seen_stream_system
        .lock()
        .expect("trim-only system mutex poisoned")
        .clone()
        .expect("expected system prompt to be sent");
    assert!(
        !stream_system.contains("CHECKPOINT SUMMARY"),
        "trim_only should not inject semantic summary text"
    );

    let seen_messages = seen_stream_messages
        .lock()
        .expect("trim-only messages mutex poisoned")
        .clone();
    let rendered = format!("{seen_messages:?}");
    assert!(
        !rendered.contains("Trim-only history 0"),
        "expected structural trimming to drop older history"
    );
    assert!(
        rendered.contains("Newest prompt in trim-only mode"),
        "expected newest prompt to remain after trimming"
    );

    Ok(())
}

#[tokio::test]
async fn test_compaction_inference_uses_dedicated_inference_context() -> Result<()> {
    let tmp = tempdir()?;
    let mut config = make_config(tmp.path());
    config.agent.provider = "main".to_string();
    config.agent.model = "main-model".to_string();
    config.agent.system_prompt = "Dedicated compaction route".to_string();
    config.inference.compaction.inference = Some("summary".to_string());
    config.inference.compaction.trigger_ratio = 0.2;
    config.inference.contexts.insert(
        "summary".to_string(),
        turin_core::kernel::config::InferenceContextConfig {
            provider: "summary".to_string(),
            model: "summary-model".to_string(),
            fallback: None,
            temperature: Some(0.1),
            max_tokens: Some(256),
            thinking_budget: None,
        },
    );
    config.providers.insert(
        "main".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            base_url: None,
            context_window_tokens: Some(8_192),
            ..ProviderConfig::default()
        },
    );
    config.providers.insert(
        "summary".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            base_url: None,
            context_window_tokens: Some(8_192),
            ..ProviderConfig::default()
        },
    );

    let mut kernel = turin_harness_lua::runtime_builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_harness().await?;

    let main_seen_stream_messages = Arc::new(Mutex::new(Vec::new()));
    let main_seen_stream_system = Arc::new(Mutex::new(None));
    let main_complete_calls = Arc::new(Mutex::new(0usize));
    let summary_seen_stream_messages = Arc::new(Mutex::new(Vec::new()));
    let summary_seen_stream_system = Arc::new(Mutex::new(None));
    let summary_complete_calls = Arc::new(Mutex::new(0usize));

    kernel.add_client(
        "main".to_string(),
        ProviderClient::new(
            "main",
            Arc::new(ContextCheckpointProvider {
                seen_stream_messages: Arc::clone(&main_seen_stream_messages),
                seen_stream_system: Arc::clone(&main_seen_stream_system),
                complete_calls: Arc::clone(&main_complete_calls),
            }),
        ),
    );
    kernel.add_client(
        "summary".to_string(),
        ProviderClient::new(
            "summary",
            Arc::new(ContextCheckpointProvider {
                seen_stream_messages: Arc::clone(&summary_seen_stream_messages),
                seen_stream_system: Arc::clone(&summary_seen_stream_system),
                complete_calls: Arc::clone(&summary_complete_calls),
            }),
        ),
    );

    let mut session = kernel.create_session().await;
    for i in 0..12 {
        kernel
            .run(
                &mut session,
                Some(format!("Compaction-route history {i}: {}", "x".repeat(240))),
            )
            .await?;
        if session.context_checkpoint.is_some() {
            break;
        }
    }

    let checkpoint = session
        .context_checkpoint
        .clone()
        .expect("expected context checkpoint");
    assert_eq!(checkpoint.provider_name, "summary");
    assert_eq!(checkpoint.model, "summary-model");
    assert_eq!(
        *summary_complete_calls
            .lock()
            .expect("summary complete mutex poisoned"),
        1,
        "expected compaction summary to use the dedicated summary context"
    );
    assert_eq!(
        *main_complete_calls
            .lock()
            .expect("main complete mutex poisoned"),
        0,
        "main inference provider should not be used for semantic compaction when a dedicated context is configured"
    );

    let main_stream_system = main_seen_stream_system
        .lock()
        .expect("main stream system mutex poisoned")
        .clone()
        .expect("expected main stream system prompt");
    assert!(
        main_stream_system.contains("CHECKPOINT SUMMARY"),
        "expected main inference request to consume the semantic checkpoint"
    );

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

#[tokio::test(flavor = "multi_thread")]
async fn test_watcher_rebuilds_when_watch_roots_change() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    let blocks_dir = harness_dir.join("blocks");
    let extras_dir = harness_dir.join("extras");
    std::fs::create_dir_all(&blocks_dir)?;
    std::fs::create_dir_all(&extras_dir)?;

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
                local ok, err = fs.write("dynamic-watch-marker.txt", "blocks-v1")
                if not ok then error(err) end
                return ALLOW
            end
        "#,
    )?;
    std::fs::write(
        extras_dir.join("feature.lua"),
        r#"
            function on_turn_prepare(ctx)
                local ok, err = fs.write("dynamic-watch-marker.txt", "extras-v1")
                if not ok then error(err) end
                return ALLOW
            end
        "#,
    )?;

    let mut kernel = make_kernel(tmp.path()).await?;
    kernel.start_watcher()?;

    let marker_path = tmp.path().join("dynamic-watch-marker.txt");

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("before watch-root change".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;
    assert_eq!(std::fs::read_to_string(&marker_path)?, "blocks-v1");

    std::fs::write(
        harness_dir.join("main.lua"),
        r#"
            watch("blocks")
            watch("extras")
            use("blocks/feature")
            use("extras/feature")
        "#,
    )?;

    let mut saw_extra_v1 = false;
    for _ in 0..10 {
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        let mut session = kernel.create_session().await;
        kernel
            .run(&mut session, Some("after watch-root change".to_string()))
            .await?;
        kernel.end_session(&mut session).await?;

        if std::fs::read_to_string(&marker_path).ok().as_deref() == Some("extras-v1") {
            saw_extra_v1 = true;
            break;
        }
    }

    assert!(
        saw_extra_v1,
        "reloading main.lua should rebuild watcher roots and activate extras block"
    );

    std::fs::write(
        extras_dir.join("feature.lua"),
        r#"
            function on_turn_prepare(ctx)
                local ok, err = fs.write("dynamic-watch-marker.txt", "extras-v2")
                if not ok then error(err) end
                return ALLOW
            end
        "#,
    )?;

    let mut saw_extra_v2 = false;
    for _ in 0..10 {
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        let mut session = kernel.create_session().await;
        kernel
            .run(&mut session, Some("after nested extras reload".to_string()))
            .await?;
        kernel.end_session(&mut session).await?;

        if std::fs::read_to_string(&marker_path).ok().as_deref() == Some("extras-v2") {
            saw_extra_v2 = true;
            break;
        }
    }

    assert!(
        saw_extra_v2,
        "watcher should rebuild after watch-root changes so new nested roots hot-reload"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_peer_agent_harness_reload_uses_shared_runtime_manager() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test-peer-reload.db");
    let default_harness_dir = tmp.path().join("harnesses-default");
    let reviewer_harness_dir = tmp.path().join("harnesses-reviewer");
    std::fs::create_dir_all(&default_harness_dir)?;
    std::fs::create_dir_all(&reviewer_harness_dir)?;

    std::fs::write(
        default_harness_dir.join("main.lua"),
        r#"
            function on_turn_prepare(ctx)
                local out = runtime.agent.ask("reviewer", "review this")
                return ALLOW
            end
        "#,
    )?;
    std::fs::write(
        reviewer_harness_dir.join("main.lua"),
        r#"
            function on_turn_prepare(ctx)
                local ok, err = fs.write(".turin/runtime/peer-watch-marker.txt", "v1")
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
        "reviewer".to_string(),
        AgentConfig {
            tools: Default::default(),
            id: "reviewer".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Reviewer agent.".to_string(),
            thinking: None,
            harness: Some("reviewer".to_string()),
            idle_timeout_seconds: None,
            linked_runtime_lanes: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
    );

    let config = config_fixture! {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Default agent.".to_string(),
            thinking: None,
            harness: None,
            idle_timeout_seconds: None,
            linked_runtime_lanes: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents,
        runtime: Default::default(),
        kernel: KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 5,
            heartbeat_interval_seconds: 30,
            initial_spawn_depth: 0,
        },
        layout: Default::default(),
        inference: InferenceConfig::default(),
        persistence: PersistenceConfig::with_state_path(db_path.to_str().unwrap().to_string()),
        harness: HarnessConfig {
            directory: default_harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::from([(
            "reviewer".to_string(),
            HarnessConfig {
                directory: reviewer_harness_dir.to_string_lossy().to_string(),
                fs_root: ".".to_string(),
                memory_limit_mb: 32,
            },
        )]),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: turin_core::kernel::config::GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = turin_harness_lua::runtime_builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;
    kernel.start_watcher()?;

    let marker_path = tmp.path().join(".turin/runtime/peer-watch-marker.txt");

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("initial review".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;
    assert_eq!(std::fs::read_to_string(&marker_path)?, "v1");

    std::fs::write(
        reviewer_harness_dir.join("main.lua"),
        r#"
            function on_turn_prepare(ctx)
                local ok, err = fs.write(".turin/runtime/peer-watch-marker.txt", "v2")
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
            .run(&mut session, Some("post reload review".to_string()))
            .await?;
        kernel.end_session(&mut session).await?;

        if std::fs::read_to_string(&marker_path).ok().as_deref() == Some("v2") {
            saw_v2 = true;
            break;
        }
    }

    assert!(
        saw_v2,
        "shared harness manager should hot-reload peer agent harnesses"
    );
    Ok(())
}

#[tokio::test]
async fn test_hot_reload_only_reloads_affected_harness_runtime() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test-affected-reload.db");
    let default_harness_dir = tmp.path().join("harnesses-default");
    let writer_harness_dir = tmp.path().join("harnesses-writer");
    std::fs::create_dir_all(&default_harness_dir)?;
    std::fs::create_dir_all(&writer_harness_dir)?;

    std::fs::write(
        default_harness_dir.join("main.lua"),
        r#"
            local current = try(fs.read, ".turin/runtime/default-load-count.txt")
            if current == nil then current = "" end
            local next_count = (tonumber(current or "0") or 0) + 1
            fs.write(".turin/runtime/default-load-count.txt", tostring(next_count))

            function on_session_start(event)
                return ALLOW
            end
        "#,
    )?;
    std::fs::write(
        writer_harness_dir.join("main.lua"),
        r#"
            local current = try(fs.read, ".turin/runtime/writer-load-count.txt")
            if current == nil then current = "" end
            local next_count = (tonumber(current or "0") or 0) + 1
            fs.write(".turin/runtime/writer-load-count.txt", tostring(next_count))

            function on_session_start(event)
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
            tools: Default::default(),
            id: "writer".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Writer agent.".to_string(),
            thinking: None,
            harness: Some("writer".to_string()),
            idle_timeout_seconds: None,
            linked_runtime_lanes: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
    );

    let config = config_fixture! {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Default agent.".to_string(),
            thinking: None,
            harness: None,
            idle_timeout_seconds: None,
            linked_runtime_lanes: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents,
        runtime: Default::default(),
        kernel: KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 5,
            heartbeat_interval_seconds: 30,
            initial_spawn_depth: 0,
        },
        layout: Default::default(),
        inference: InferenceConfig::default(),
        persistence: PersistenceConfig::with_state_path(db_path.to_str().unwrap().to_string()),
        harness: HarnessConfig {
            directory: default_harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::from([(
            "writer".to_string(),
            HarnessConfig {
                directory: writer_harness_dir.to_string_lossy().to_string(),
                fs_root: ".".to_string(),
                memory_limit_mb: 32,
            },
        )]),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: turin_core::kernel::config::GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = turin_harness_lua::runtime_builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;
    kernel.start_watcher()?;

    let default_count_path = tmp.path().join(".turin/runtime/default-load-count.txt");
    let writer_count_path = tmp.path().join(".turin/runtime/writer-load-count.txt");

    assert_eq!(std::fs::read_to_string(&default_count_path)?, "1");
    assert_eq!(std::fs::read_to_string(&writer_count_path)?, "1");

    std::fs::write(
        writer_harness_dir.join("main.lua"),
        r#"
            local current = try(fs.read, ".turin/runtime/writer-load-count.txt")
            if current == nil then current = "" end
            local next_count = (tonumber(current or "0") or 0) + 1
            fs.write(".turin/runtime/writer-load-count.txt", tostring(next_count))

            function on_session_start(event)
                return ALLOW
            end

            -- trigger writer harness reload only
        "#,
    )?;

    let mut saw_writer_reload = false;
    for _ in 0..10 {
        tokio::time::sleep(std::time::Duration::from_millis(250)).await;
        let default_count = std::fs::read_to_string(&default_count_path).ok();
        let writer_count = std::fs::read_to_string(&writer_count_path).ok();
        if default_count.as_deref() == Some("1") && writer_count.as_deref() == Some("2") {
            saw_writer_reload = true;
            break;
        }
    }

    assert!(
        saw_writer_reload,
        "changing a named harness should reload only that harness runtime"
    );
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
            tools: Default::default(),
            id: "writer".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Writer agent.".to_string(),
            thinking: None,
            harness: Some("writer".to_string()),
            idle_timeout_seconds: None,
            linked_runtime_lanes: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
    );

    let config = config_fixture! {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Default agent.".to_string(),
            thinking: None,
            harness: None,
            idle_timeout_seconds: None,
            linked_runtime_lanes: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents,
        runtime: Default::default(),
        kernel: KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 5,
            heartbeat_interval_seconds: 30,
            initial_spawn_depth: 0,
        },
        layout: Default::default(),
        inference: InferenceConfig::default(),
        persistence: PersistenceConfig::with_state_path(db_path.to_str().unwrap().to_string()),
        harness: HarnessConfig {
            directory: default_harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::from([(
            "writer".to_string(),
            HarnessConfig {
                directory: writer_harness_dir.to_string_lossy().to_string(),
                fs_root: ".".to_string(),
                memory_limit_mb: 32,
            },
        )]),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: turin_core::kernel::config::GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = turin_harness_lua::runtime_builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut default_session = kernel.create_session().await;
    kernel.start_session(&mut default_session).await?;
    kernel.end_session(&mut default_session).await?;

    let mut writer_session = kernel.create_session_for_agent("writer").await;
    kernel.start_session(&mut writer_session).await?;
    kernel.end_session(&mut writer_session).await?;

    let writer_scripts = kernel.loaded_scripts_for_agent("writer")?;
    assert_eq!(writer_scripts, vec!["main".to_string()]);

    let snapshots = kernel.harness_snapshots();
    assert_eq!(snapshots.len(), 2);
    let default_snapshot = snapshots
        .iter()
        .find(|snapshot| snapshot.harness_id == "default")
        .expect("expected default harness snapshot");
    assert_eq!(default_snapshot.bound_agents, vec!["default".to_string()]);
    assert_eq!(default_snapshot.loaded_scripts, vec!["main".to_string()]);
    assert_eq!(
        default_snapshot.watched_roots,
        vec![default_harness_dir.to_string_lossy().to_string()]
    );

    let writer_snapshot = snapshots
        .iter()
        .find(|snapshot| snapshot.harness_id == "writer")
        .expect("expected writer harness snapshot");
    assert_eq!(writer_snapshot.bound_agents, vec!["writer".to_string()]);
    assert_eq!(writer_snapshot.loaded_scripts, vec!["main".to_string()]);
    assert_eq!(
        writer_snapshot.watched_roots,
        vec![writer_harness_dir.to_string_lossy().to_string()]
    );

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
        let events = store
            .get_events(
                session.internal_id.unwrap(),
                &SessionReadTarget::ActiveBranch,
            )
            .await?;
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

    config.governance.profile = "governed".to_string();
    config.governance.audit.mode = turin_core::kernel::config::GovernanceAuditMode::Immutable;
    config.governance.enforcement_enabled = false;

    let mut kernel = turin_harness_lua::runtime_builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Persist immutable audit".to_string()))
        .await?;

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    if let Ok(store) = kernel.store_manager().get_default().await {
        let events = store
            .get_events(
                session.internal_id.unwrap(),
                &SessionReadTarget::ActiveBranch,
            )
            .await?;
        assert!(
            events.iter().any(|e| e.event_type == "governance_snapshot"),
            "immutable audit mode should persist governance_snapshot even if on_kernel_event rejects it"
        );
    }

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
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

    config.governance.profile = "balanced".to_string();
    config.governance.enforcement_enabled = true;
    config.governance.grants.enabled = true;
    config.governance.grants.max_ttl_ms = Some(10_000);
    config.governance.grants.require_audit_reason = true;

    let mut kernel = turin_harness_lua::runtime_builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Emit grant audit events".to_string()))
        .await?;

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;

    if let Ok(store) = kernel.store_manager().get_default().await {
        let events = store
            .get_events(
                session.internal_id.unwrap(),
                &SessionReadTarget::ActiveBranch,
            )
            .await?;
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

    let config = config_fixture! {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Test.".to_string(),
            thinking: None,
            harness: None,
            idle_timeout_seconds: None,
            linked_runtime_lanes: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: std::collections::HashMap::new(),
        runtime: Default::default(),
        kernel: KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 3,
            heartbeat_interval_seconds: 30,
            initial_spawn_depth: 0,
        },
        layout: Default::default(),
        inference: InferenceConfig::default(),
        persistence: PersistenceConfig {
            state: StoreTargetConfig::from_path(""), // Empty — no persistence
            ..PersistenceConfig::default()
        },
        harness: HarnessConfig {
            directory: harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::new(),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: turin_core::kernel::config::GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = turin_harness_lua::runtime_builder(config).build()?;
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
        let events = store
            .get_events(
                session.internal_id.unwrap(),
                &SessionReadTarget::ActiveBranch,
            )
            .await?;
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
        let events = store
            .get_events(
                session.internal_id.unwrap(),
                &SessionReadTarget::ActiveBranch,
            )
            .await?;
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
        let events = store
            .get_events(
                session.internal_id.unwrap(),
                &SessionReadTarget::ActiveBranch,
            )
            .await?;
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

#[tokio::test]
async fn test_token_usage_and_task_complete_include_task_budget_metrics() -> Result<()> {
    let tmp = tempdir()?;
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir_all(&harness_dir)?;
    std::fs::write(
        harness_dir.join("task_budget.lua"),
        r#"
            local queued_followup = false

            local function valid_budget(payload)
                return payload.task_started_at_unix_ms ~= nil
                    and type(payload.task_elapsed_ms) == "number"
                    and payload.task_elapsed_ms >= 0
                    and payload.task_input_tokens == 10
                    and payload.task_output_tokens == 5
                    and payload.task_total_tokens == 15
                    and payload.task_turn_count == 1
                    and (payload.total_tokens == nil or payload.total_tokens >= payload.task_total_tokens)
            end

            function on_token_usage(usage)
                if not valid_budget(usage) then
                    return REJECT, "missing or invalid task budget metrics"
                end
                return ALLOW
            end

            function on_task_complete(event)
                if event.status == "success" and not queued_followup and valid_budget(event) then
                    queued_followup = true
                    return MODIFY, { "Follow-up from budget metrics" }
                end
                return ALLOW
            end
        "#,
    )?;

    let mut config = make_config(tmp.path());
    config.agent.provider = "capture".to_string();
    config.agent.model = "capture-model".to_string();
    config.providers.insert(
        "capture".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            base_url: None,
            ..ProviderConfig::default()
        },
    );

    let mut kernel = turin_harness_lua::runtime_builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_harness().await?;
    kernel.add_client(
        "capture".to_string(),
        ProviderClient::new(
            "capture",
            Arc::new(CaptureMessagesProvider {
                seen_messages: Arc::new(Mutex::new(Vec::new())),
            }),
        ),
    );
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
        .run(&mut session, Some("Check budget metrics".to_string()))
        .await?;

    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    if let Ok(store) = kernel.store_manager().get_default().await {
        let events = store
            .get_events(
                session.internal_id.unwrap(),
                &SessionReadTarget::ActiveBranch,
            )
            .await?;
        assert_eq!(
            count_event_type(&events, "task_start"),
            2,
            "on_task_complete should queue a follow-up only when budget metrics are valid"
        );
        assert!(
            events.iter().all(|event| {
                event.event_type != "task_complete"
                    || serde_json::from_str::<serde_json::Value>(&event.payload)
                        .ok()
                        .and_then(|value| value.get("status").cloned())
                        .and_then(|status| status.as_str().map(str::to_string))
                        .is_none_or(|status| status == "success")
            }),
            "token usage enforcement should not reject when task budget metrics are valid"
        );
    }

    kernel.end_session(&mut session).await?;
    Ok(())
}
