use anyhow::Result;
use futures::future::BoxFuture;
use futures::stream;
use std::collections::HashMap;
use std::sync::Arc;
use tempfile::tempdir;
use turin::inference::provider::{
    InferenceContent, InferenceEvent, InferenceProvider, InferenceRequest, InferenceStream,
    ProviderClient, RequestOptions, SdkError,
};
use turin::kernel::Kernel;
use turin::kernel::config::{
    AgentConfig, EmbeddingConfig, HarnessConfig, PersistenceConfig, ProviderConfig, TurinConfig,
};

struct ToolMockProvider {
    tool_name: String,
    tool_args: serde_json::Value,
}

impl InferenceProvider for ToolMockProvider {
    fn stream<'a>(
        &'a self,
        _request: InferenceRequest,
        _options: Option<RequestOptions>,
    ) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        let tool_name = self.tool_name.clone();
        let tool_args = self.tool_args.clone();
        Box::pin(async move {
            let events = vec![
                Ok(InferenceEvent::MessageStart {
                    role: "assistant".to_string(),
                    model: "mock-model".to_string(),
                    provider_id: "mock".to_string(),
                }),
                Ok(InferenceEvent::ToolCallStart {
                    id: "test-call-id".to_string(),
                    name: tool_name,
                }),
                Ok(InferenceEvent::ToolCallDelta {
                    delta: tool_args.to_string(),
                }),
                Ok(InferenceEvent::MessageEnd {
                    input_tokens: 10,
                    output_tokens: 5,
                    stop_reason: None,
                }),
            ];
            Ok(Box::pin(stream::iter(events)) as InferenceStream)
        })
    }
}

struct HeaderCaptureProvider {
    seen: Arc<std::sync::Mutex<(bool, bool, Option<u32>)>>,
}

impl InferenceProvider for HeaderCaptureProvider {
    fn stream<'a>(
        &'a self,
        _request: InferenceRequest,
        options: Option<RequestOptions>,
    ) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        let seen = self.seen.clone();
        Box::pin(async move {
            let (static_header, dynamic_header, max_retries) = if let Some(opts) = options {
                (
                    opts.headers.get("x-static").is_some(),
                    opts.headers.get("x-dynamic").is_some(),
                    opts.max_retries,
                )
            } else {
                (false, false, None)
            };
            *seen.lock().unwrap() = (static_header, dynamic_header, max_retries);

            let events = vec![
                Ok(InferenceEvent::MessageStart {
                    role: "assistant".to_string(),
                    model: "mock-model".to_string(),
                    provider_id: "mock".to_string(),
                }),
                Ok(InferenceEvent::MessageDelta {
                    content: "ok".to_string(),
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
async fn test_harness_rejection() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    // Create a safety harness that blocks 'shell_exec'
    let harness_code = r#"
        function on_tool_call(call)
            if call.name == "shell_exec" then
                return REJECT, "Security policy: shell_exec is forbidden"
            end
            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("safety.lua"), harness_code)?;

    let mut providers = HashMap::new();
    // We add a dummy config, but we will inject our custom provider client
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
            system_prompt: "You are a test assistant.".to_string(),
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

    // Inject custom provider
    let mock_provider = Arc::new(ToolMockProvider {
        tool_name: "shell_exec".to_string(),
        tool_args: serde_json::json!({"command": "rm -rf /"}),
    });
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new("mock", mock_provider),
    );

    kernel.init_harness().await?;

    let mut session = kernel.create_session();

    // Run the agent. The mock provider will trigger 'shell_exec'.
    // The harness should reject it.
    kernel
        .run(&mut session, Some("Run a dangerous command".to_string()))
        .await?;

    // Verify turn index incremented
    assert!(session.turn_index > 0);

    // Verify that the assistant history contains the rejection message
    // Turn 0: User message
    // Turn 1: Assistant ToolUse -> [Harness Rejection] -> [Assistant responds to rejection?]
    // Actually, Kernel::run_task does:
    // 1. Assistant streams ToolCall
    // 2. Kernel evaluates harness
    // 3. Harness REJECTS.
    // 4. Kernel appends [HARNESS REJECTED] as ToolResult to history.
    // 5. Kernel enters next turn of loop because tool_results were added.
    // 6. Next turn: LLM sees rejection.

    // In our test, the MockProvider ALWAYS returns the same ToolCall.
    // To prevent infinite loop, max_turns=5 will eventually stop it.

    // Let's check history for the rejection string
    let mut found_rejection = false;
    for msg in &session.history {
        for content in &msg.content {
            if let InferenceContent::ToolResult { content, .. } = content
                && content.contains("Security policy: shell_exec is forbidden")
            {
                found_rejection = true;
            }
        }
    }

    assert!(
        found_rejection,
        "Harness rejection message not found in session history"
    );

    kernel.end_session(&mut session).await?;

    Ok(())
}

#[tokio::test]
async fn test_harness_request_options_passthrough() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_headers.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let harness_code = r#"
        function on_turn_prepare(ctx)
            local opts = ctx.request_options or {}
            opts.headers = opts.headers or {}
            opts.headers["x-dynamic"] = "from-harness"
            opts.max_retries = 1
            ctx.request_options = opts
            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("headers.lua"), harness_code)?;

    let mut static_headers = HashMap::new();
    static_headers.insert("x-static".to_string(), "from-config".to_string());

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: None,
            headers: static_headers,
            max_retries: Some(2),
            ..ProviderConfig::default()
        },
    );

    let config = TurinConfig {
        agent: AgentConfig {
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Header test".to_string(),
            thinking: None,
        },
        kernel: turin::kernel::config::KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 2,
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

    let seen = Arc::new(std::sync::Mutex::new((false, false, None)));
    let provider = Arc::new(HeaderCaptureProvider { seen: seen.clone() });
    kernel.add_client("mock".to_string(), ProviderClient::new("mock", provider));

    let mut session = kernel.create_session();
    kernel
        .run(&mut session, Some("emit headers".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;

    let captured = *seen.lock().unwrap();
    assert!(captured.0, "expected config header to be passed through");
    assert!(captured.1, "expected harness header to be passed through");
    assert_eq!(captured.2, Some(1), "expected harness override for retries");

    Ok(())
}
