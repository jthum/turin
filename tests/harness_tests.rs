use anyhow::{Context, Result};
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
    AgentConfig, AgentMode, ContextPersistenceConfig, EmbeddingConfig, GovernanceConfig,
    GovernanceGrantsConfig, GovernanceProfile, HarnessConfig, InferenceConfig,
    InferenceContextConfig, KernelConfig, NamedStoreConfig, PersistenceConfig, ProviderConfig,
    ScopedStorePlacementConfig, StoreTargetConfig, TurinConfig,
};
use turin::kernel::policy::PolicyScope;
use turin::persistence::manager::StoreSelector;
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

struct FixedTextProvider {
    text: String,
}

impl InferenceProvider for FixedTextProvider {
    fn stream<'a>(
        &'a self,
        _request: InferenceRequest,
        _options: Option<RequestOptions>,
    ) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        let text = self.text.clone();
        Box::pin(async move {
            let events = vec![
                Ok(InferenceEvent::MessageStart {
                    role: "assistant".to_string(),
                    model: "mock-model".to_string(),
                    provider_id: "mock".to_string(),
                }),
                Ok(InferenceEvent::MessageDelta { content: text }),
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

struct VirtualToolProvider {
    tool_name: String,
    tool_args: serde_json::Value,
    seen_tools: Arc<std::sync::Mutex<Vec<String>>>,
    stage: Arc<std::sync::Mutex<u32>>,
}

impl InferenceProvider for VirtualToolProvider {
    fn stream<'a>(
        &'a self,
        request: InferenceRequest,
        _options: Option<RequestOptions>,
    ) -> BoxFuture<'a, Result<InferenceStream, SdkError>> {
        let seen_tools = self.seen_tools.clone();
        let stage = self.stage.clone();
        let tool_name = self.tool_name.clone();
        let tool_args = self.tool_args.clone();
        Box::pin(async move {
            let tool_names = request
                .tools
                .as_ref()
                .map(|tools| tools.iter().map(|tool| tool.name.clone()).collect())
                .unwrap_or_default();
            *seen_tools.lock().unwrap() = tool_names;

            let current_stage = {
                let mut lock = stage.lock().unwrap();
                let current = *lock;
                *lock += 1;
                current
            };

            let events = if current_stage == 0 {
                vec![
                    Ok(InferenceEvent::MessageStart {
                        role: "assistant".to_string(),
                        model: "mock-model".to_string(),
                        provider_id: "mock".to_string(),
                    }),
                    Ok(InferenceEvent::ToolCallStart {
                        id: "virtual-call-id".to_string(),
                        name: tool_name,
                    }),
                    Ok(InferenceEvent::ToolCallDelta {
                        delta: tool_args.to_string(),
                    }),
                    Ok(InferenceEvent::MessageEnd {
                        input_tokens: 8,
                        output_tokens: 5,
                        stop_reason: None,
                    }),
                ]
            } else {
                vec![
                    Ok(InferenceEvent::MessageStart {
                        role: "assistant".to_string(),
                        model: "mock-model".to_string(),
                        provider_id: "mock".to_string(),
                    }),
                    Ok(InferenceEvent::MessageDelta {
                        content: "done".to_string(),
                    }),
                    Ok(InferenceEvent::MessageEnd {
                        input_tokens: 4,
                        output_tokens: 2,
                        stop_reason: None,
                    }),
                ]
            };

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
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "You are a test assistant.".to_string(),
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
        embeddings: None,
        governance: turin::kernel::config::GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
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

    let mut session = kernel.create_session().await;

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
async fn test_virtual_tool_is_exposed_and_executes_native_call() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;
    std::fs::write(tmp.path().join("note.txt"), "hello from virtual tool")?;

    std::fs::write(
        harness_dir.join("main.lua"),
        r#"
        tool.declare("read_note", {
            description = "Read a note from the workspace",
            params = {
                path = { type = "string", required = true }
            },
            handler = function(args)
                return tool.call("read_file", { path = args.path })
            end
        })
        "#,
    )?;

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
            system_prompt: "You are a test assistant.".to_string(),
            thinking: None,
            mode: AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: std::collections::HashMap::new(),
        kernel: KernelConfig {
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
        embeddings: None,
        governance: GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let seen_tools = Arc::new(std::sync::Mutex::new(Vec::new()));
    let stage = Arc::new(std::sync::Mutex::new(0));

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new(
            "mock",
            Arc::new(VirtualToolProvider {
                tool_name: "read_note".to_string(),
                tool_args: serde_json::json!({ "path": "note.txt" }),
                seen_tools: seen_tools.clone(),
                stage,
            }),
        ),
    );
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Read the note".to_string()))
        .await?;

    assert!(
        seen_tools
            .lock()
            .unwrap()
            .iter()
            .any(|tool| tool == "read_note"),
        "expected virtual tool to be exposed to provider request tools"
    );

    let mut found_virtual_result = false;
    for msg in &session.history {
        for content in &msg.content {
            if let InferenceContent::ToolResult { content, .. } = content
                && content.contains("hello from virtual tool")
            {
                found_virtual_result = true;
            }
        }
    }

    assert!(
        found_virtual_result,
        "expected virtual tool result content to reach the outer tool result history"
    );

    Ok(())
}

#[tokio::test]
async fn test_virtual_tool_sequence_aggregates_multiple_native_calls() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;
    std::fs::write(tmp.path().join("one.txt"), "first file")?;
    std::fs::write(tmp.path().join("two.txt"), "second file")?;

    std::fs::write(
        harness_dir.join("main.lua"),
        r#"
        tool.declare("read_pair", {
            description = "Read two files in order",
            params = {
                first = { type = "string", required = true },
                second = { type = "string", required = true }
            },
            handler = function(args)
                return tool.sequence({
                    tool.call("read_file", { path = args.first }),
                    tool.call("read_file", { path = args.second })
                })
            end
        })
        "#,
    )?;

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
            system_prompt: "You are a test assistant.".to_string(),
            thinking: None,
            mode: AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: std::collections::HashMap::new(),
        kernel: KernelConfig {
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
        embeddings: None,
        governance: GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new(
            "mock",
            Arc::new(VirtualToolProvider {
                tool_name: "read_pair".to_string(),
                tool_args: serde_json::json!({ "first": "one.txt", "second": "two.txt" }),
                seen_tools: Arc::new(std::sync::Mutex::new(Vec::new())),
                stage: Arc::new(std::sync::Mutex::new(0)),
            }),
        ),
    );
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Read both files".to_string()))
        .await?;

    let mut aggregated_result = None;
    for msg in &session.history {
        for content in &msg.content {
            if let InferenceContent::ToolResult { content, .. } = content
                && content.contains("Call 1: read_file [ok]")
            {
                aggregated_result = Some(content.clone());
            }
        }
    }

    let aggregated_result = aggregated_result.expect("expected aggregated virtual tool output");
    assert!(aggregated_result.contains("first file"));
    assert!(aggregated_result.contains("second file"));

    Ok(())
}

#[tokio::test]
async fn test_virtual_tool_sequence_callback_shapes_outer_result() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;
    std::fs::write(tmp.path().join("one.txt"), "first file")?;
    std::fs::write(tmp.path().join("two.txt"), "second file")?;

    std::fs::write(
        harness_dir.join("main.lua"),
        r#"
        tool.declare("summarize_pair", {
            description = "Read two files and summarize them",
            params = {
                first = { type = "string", required = true },
                second = { type = "string", required = true }
            },
            handler = function(args)
                return tool.sequence({
                    tool.call("read_file", { path = args.first }),
                    tool.call("read_file", { path = args.second })
                }, function(results)
                    return {
                        content = "Combined: " .. results[1].content .. " | " .. results[2].content,
                        is_error = results[1].is_error or results[2].is_error
                    }
                end)
            end
        })
        "#,
    )?;

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
            system_prompt: "You are a test assistant.".to_string(),
            thinking: None,
            mode: AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: std::collections::HashMap::new(),
        kernel: KernelConfig {
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
        embeddings: None,
        governance: GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new(
            "mock",
            Arc::new(VirtualToolProvider {
                tool_name: "summarize_pair".to_string(),
                tool_args: serde_json::json!({ "first": "one.txt", "second": "two.txt" }),
                seen_tools: Arc::new(std::sync::Mutex::new(Vec::new())),
                stage: Arc::new(std::sync::Mutex::new(0)),
            }),
        ),
    );
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Summarize both files".to_string()))
        .await?;

    let mut shaped_result = None;
    for msg in &session.history {
        for content in &msg.content {
            if let InferenceContent::ToolResult { content, .. } = content
                && content.contains("Combined: first file | second file")
            {
                shaped_result = Some(content.clone());
            }
        }
    }

    assert_eq!(
        shaped_result.as_deref(),
        Some("Combined: first file | second file")
    );

    Ok(())
}

#[tokio::test]
async fn test_virtual_tool_can_call_another_virtual_tool() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;
    std::fs::write(
        tmp.path().join("note.txt"),
        "hello from nested virtual tool",
    )?;

    std::fs::write(
        harness_dir.join("main.lua"),
        r#"
        tool.declare("read_note", {
            description = "Read a note from disk",
            params = {
                path = { type = "string", required = true }
            },
            handler = function(args)
                return tool.call("read_file", { path = args.path })
            end
        })

        tool.declare("read_note_wrapped", {
            description = "Read a note through another virtual tool",
            params = {
                path = { type = "string", required = true }
            },
            handler = function(args)
                return tool.call("read_note", { path = args.path }, function(result)
                    return "wrapped: " .. result.content
                end)
            end
        })
        "#,
    )?;

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
            system_prompt: "You are a test assistant.".to_string(),
            thinking: None,
            mode: AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: std::collections::HashMap::new(),
        kernel: KernelConfig {
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
        embeddings: None,
        governance: GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new(
            "mock",
            Arc::new(VirtualToolProvider {
                tool_name: "read_note_wrapped".to_string(),
                tool_args: serde_json::json!({ "path": "note.txt" }),
                seen_tools: Arc::new(std::sync::Mutex::new(Vec::new())),
                stage: Arc::new(std::sync::Mutex::new(0)),
            }),
        ),
    );
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Read the wrapped note".to_string()))
        .await?;

    let mut wrapped_result = None;
    for msg in &session.history {
        for content in &msg.content {
            if let InferenceContent::ToolResult { content, .. } = content
                && content.contains("wrapped: hello from nested virtual tool")
            {
                wrapped_result = Some(content.clone());
            }
        }
    }

    assert_eq!(
        wrapped_result.as_deref(),
        Some("wrapped: hello from nested virtual tool")
    );

    Ok(())
}

#[tokio::test]
async fn test_virtual_tool_can_forward_reference_later_declaration() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;
    std::fs::write(
        tmp.path().join("note.txt"),
        "hello from forward referenced virtual tool",
    )?;

    std::fs::write(
        harness_dir.join("main.lua"),
        r#"
        tool.declare("read_note_wrapped", {
            description = "Read a note through a later-declared virtual tool",
            params = {
                path = { type = "string", required = true }
            },
            handler = function(args)
                return tool.call("read_note", { path = args.path }, function(result)
                    return "wrapped later: " .. result.content
                end)
            end
        })

        tool.declare("read_note", {
            description = "Read a note from disk",
            params = {
                path = { type = "string", required = true }
            },
            handler = function(args)
                return tool.call("read_file", { path = args.path })
            end
        })
        "#,
    )?;

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
            system_prompt: "You are a test assistant.".to_string(),
            thinking: None,
            mode: AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: std::collections::HashMap::new(),
        kernel: KernelConfig {
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
        embeddings: None,
        governance: GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new(
            "mock",
            Arc::new(VirtualToolProvider {
                tool_name: "read_note_wrapped".to_string(),
                tool_args: serde_json::json!({ "path": "note.txt" }),
                seen_tools: Arc::new(std::sync::Mutex::new(Vec::new())),
                stage: Arc::new(std::sync::Mutex::new(0)),
            }),
        ),
    );
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("Read the later-declared note".to_string()),
        )
        .await?;

    let mut wrapped_result = None;
    for msg in &session.history {
        for content in &msg.content {
            if let InferenceContent::ToolResult { content, .. } = content
                && content.contains("wrapped later: hello from forward referenced virtual tool")
            {
                wrapped_result = Some(content.clone());
            }
        }
    }

    assert_eq!(
        wrapped_result.as_deref(),
        Some("wrapped later: hello from forward referenced virtual tool")
    );

    Ok(())
}

#[tokio::test]
async fn test_virtual_tool_recursion_is_rejected() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    std::fs::write(
        harness_dir.join("main.lua"),
        r#"
        tool.declare("loop_a", {
            description = "Loop entry A",
            params = {
                path = { type = "string", required = true }
            },
            handler = function(args)
                return tool.call("loop_b", { path = args.path })
            end
        })

        tool.declare("loop_b", {
            description = "Loop entry B",
            params = {
                path = { type = "string", required = true }
            },
            handler = function(args)
                return tool.call("loop_a", { path = args.path })
            end
        })
        "#,
    )?;

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
            system_prompt: "You are a test assistant.".to_string(),
            thinking: None,
            mode: AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: std::collections::HashMap::new(),
        kernel: KernelConfig {
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
        embeddings: None,
        governance: GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new(
            "mock",
            Arc::new(VirtualToolProvider {
                tool_name: "loop_a".to_string(),
                tool_args: serde_json::json!({ "path": "note.txt" }),
                seen_tools: Arc::new(std::sync::Mutex::new(Vec::new())),
                stage: Arc::new(std::sync::Mutex::new(0)),
            }),
        ),
    );
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Trigger recursion".to_string()))
        .await?;

    let mut recursion_error = None;
    for msg in &session.history {
        for content in &msg.content {
            if let InferenceContent::ToolResult { content, .. } = content
                && content.contains("virtual tool recursion detected")
            {
                recursion_error = Some(content.clone());
            }
        }
    }

    let recursion_error = recursion_error.expect("expected recursion error result");
    assert!(recursion_error.contains("loop_a -> loop_b -> loop_a"));

    Ok(())
}

#[tokio::test]
async fn test_virtual_tool_depth_limit_is_enforced() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;
    std::fs::write(tmp.path().join("note.txt"), "depth test")?;

    let mut harness_code = String::new();
    for index in 1..=9 {
        if index < 9 {
            harness_code.push_str(&format!(
                r#"
tool.declare("tool_{index}", {{
  description = "Depth test tool {index}",
  params = {{
    path = {{ type = "string", required = true }}
  }},
  handler = function(args)
    return tool.call("tool_{next}", {{ path = args.path }})
  end
}})
"#,
                next = index + 1
            ));
        } else {
            harness_code.push_str(
                r#"
tool.declare("tool_9", {
  description = "Depth test tool 9",
  params = {
    path = { type = "string", required = true }
  },
  handler = function(args)
    return tool.call("read_file", { path = args.path })
  end
})
"#,
            );
        }
    }
    std::fs::write(harness_dir.join("main.lua"), harness_code)?;

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
            system_prompt: "You are a test assistant.".to_string(),
            thinking: None,
            mode: AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: std::collections::HashMap::new(),
        kernel: KernelConfig {
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
        embeddings: None,
        governance: GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new(
            "mock",
            Arc::new(VirtualToolProvider {
                tool_name: "tool_1".to_string(),
                tool_args: serde_json::json!({ "path": "note.txt" }),
                seen_tools: Arc::new(std::sync::Mutex::new(Vec::new())),
                stage: Arc::new(std::sync::Mutex::new(0)),
            }),
        ),
    );
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Trigger depth overflow".to_string()))
        .await?;

    let mut depth_error = None;
    for msg in &session.history {
        for content in &msg.content {
            if let InferenceContent::ToolResult { content, .. } = content
                && content.contains("virtual tool nesting depth exceeded")
            {
                depth_error = Some(content.clone());
            }
        }
    }

    let depth_error = depth_error.expect("expected depth limit error result");
    assert!(depth_error.contains("max 8"));
    assert!(depth_error.contains("tool_1 -> tool_2 -> tool_3"));

    Ok(())
}

#[tokio::test]
async fn test_virtual_tool_callback_can_return_follow_up_plan() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;
    std::fs::write(tmp.path().join("pointer.txt"), "note.txt")?;
    std::fs::write(
        tmp.path().join("note.txt"),
        "resolved through callback plan",
    )?;

    std::fs::write(
        harness_dir.join("main.lua"),
        r#"
        tool.declare("read_note", {
            description = "Read a note from disk",
            params = {
                path = { type = "string", required = true }
            },
            handler = function(args)
                return tool.call("read_file", { path = args.path })
            end
        })

        tool.declare("resolve_pointer", {
            description = "Resolve a pointer file and read the final note",
            params = {
                pointer = { type = "string", required = true }
            },
            handler = function(args)
                return tool.call("read_file", { path = args.pointer }, function(result)
                    return tool.call("read_note", { path = result.content })
                end)
            end
        })
        "#,
    )?;

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
            system_prompt: "You are a test assistant.".to_string(),
            thinking: None,
            mode: AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: std::collections::HashMap::new(),
        kernel: KernelConfig {
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
        embeddings: None,
        governance: GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new(
            "mock",
            Arc::new(VirtualToolProvider {
                tool_name: "resolve_pointer".to_string(),
                tool_args: serde_json::json!({ "pointer": "pointer.txt" }),
                seen_tools: Arc::new(std::sync::Mutex::new(Vec::new())),
                stage: Arc::new(std::sync::Mutex::new(0)),
            }),
        ),
    );
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Resolve the pointer".to_string()))
        .await?;

    let mut resolved_result = None;
    for msg in &session.history {
        for content in &msg.content {
            if let InferenceContent::ToolResult { content, .. } = content
                && content.contains("resolved through callback plan")
            {
                resolved_result = Some(content.clone());
            }
        }
    }

    assert_eq!(
        resolved_result.as_deref(),
        Some("resolved through callback plan")
    );

    Ok(())
}

#[tokio::test]
async fn test_governed_mode_denies_shell_exec_tool_at_kernel_fallback() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_governed_tool_fallback.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    // No rejection hook here; governance should deny at kernel tool execution layer.
    std::fs::write(
        harness_dir.join("allow.lua"),
        r#"
            function on_tool_call(call)
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
            system_prompt: "Governed tool fallback test".to_string(),
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
        embeddings: None,
        governance: turin::kernel::config::GovernanceConfig {
            profile: turin::kernel::config::GovernanceProfile::Governed,
            enforcement_enabled: true,
            ..turin::kernel::config::GovernanceConfig::default()
        },
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;

    let mock_provider = Arc::new(ToolMockProvider {
        tool_name: "shell_exec".to_string(),
        tool_args: serde_json::json!({"command": "echo governed-check"}),
    });
    kernel.add_client(
        "mock".to_string(),
        ProviderClient::new("mock", mock_provider),
    );

    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("Try shell exec".to_string()))
        .await?;

    let mut found_governance_denial = false;
    for msg in &session.history {
        for content in &msg.content {
            if let InferenceContent::ToolResult { content, .. } = content
                && content.contains("Governance denial")
                && content.contains("shell.exec")
            {
                found_governance_denial = true;
            }
        }
    }

    assert!(
        found_governance_denial,
        "Expected governed-mode shell_exec denial from kernel tool fallback"
    );

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_runtime_agent_submit_applies_delegated_capability_ceiling() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_runtime_agent_peer_delegation.db");
    let orchestrator_harness_dir = tmp.path().join("harnesses_orchestrator");
    let worker_harness_dir = tmp.path().join("harnesses_worker");
    std::fs::create_dir(&orchestrator_harness_dir)?;
    std::fs::create_dir(&worker_harness_dir)?;

    let orchestrator_harness = r#"
        function on_turn_prepare(ctx)
            local self_dec, sde = runtime.governance.check("runtime.policy.set")
            if self_dec == nil then error("orchestrator governance.check failed: " .. tostring(sde)) end
            if not self_dec.allowed then
                error("balanced profile should allow runtime.policy.set before delegation")
            end

            local task_id, se = runtime.agent.submit("worker", { prompt = "delegated worker run" }, {
                capabilities = {
                    ["runtime.db.query"] = true
                }
            })
            if task_id == nil then error("runtime.agent.submit failed: " .. tostring(se)) end

            local res, ae = runtime.agent.await(task_id, { timeout_ms = 5000 })
            if res == nil then error("runtime.agent.await failed: " .. tostring(ae)) end
            if res.agent_id ~= "worker" then error("runtime.agent.await wrong agent") end
            if res.status ~= "success" then
                error("delegated worker task should succeed, got status " .. tostring(res.status))
            end
            if res.output ~= "worker-ok" then
                error("delegated worker output mismatch: " .. tostring(res.output))
            end

            return ALLOW
        end
    "#;
    std::fs::write(
        orchestrator_harness_dir.join("orchestrator.lua"),
        orchestrator_harness,
    )?;

    let worker_harness = r#"
        function on_turn_prepare(ctx)
            local dec, de = runtime.governance.check("runtime.policy.set")
            if dec == nil then error("worker governance.check failed: " .. tostring(de)) end
            if dec.subject_agent_id ~= "worker" then error("worker subject_agent_id mismatch") end
            if dec.allowed then
                error("delegated worker should have runtime.policy.set denied")
            end

            local ok, err = runtime.policy.set("peer.delegation.test", true)
            if ok ~= false or err == nil then
                error("worker runtime.policy.set should be denied by delegated ceiling")
            end
            if string.find(tostring(err), "delegated capabilities", 1, true) == nil then
                error("worker denial should mention delegated capabilities")
            end

            return ALLOW
        end
    "#;
    std::fs::write(worker_harness_dir.join("worker.lua"), worker_harness)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("worker-ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let mut agents = std::collections::HashMap::new();
    agents.insert(
        "worker".to_string(),
        AgentConfig {
            tools: Default::default(),
            id: "worker".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Worker".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Stateless,
            harness: Some("worker".to_string()),
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "orchestrator".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Orchestrator".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents,
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
            directory: orchestrator_harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::from([(
            "worker".to_string(),
            HarnessConfig {
                directory: worker_harness_dir.to_string_lossy().to_string(),
                fs_root: ".".to_string(),
                memory_limit_mb: 32,
            },
        )]),
        providers,
        embeddings: None,
        governance: turin::kernel::config::GovernanceConfig {
            profile: turin::kernel::config::GovernanceProfile::Balanced,
            enforcement_enabled: true,
            ..turin::kernel::config::GovernanceConfig::default()
        },
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise runtime agent delegation".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_agent_allowed_child_agents_enforced_across_aliases() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_agent_allowed_child_agents.db");
    let orchestrator_harness_dir = tmp.path().join("harnesses_orchestrator");
    let worker_harness_dir = tmp.path().join("harnesses_worker");
    std::fs::create_dir(&orchestrator_harness_dir)?;
    std::fs::create_dir(&worker_harness_dir)?;

    let orchestrator_harness = r#"
        function on_turn_prepare(ctx)
            local blocked_id, blocked_err = runtime.agent.submit("worker_blocked", { prompt = "nope" })
            if blocked_id ~= nil or blocked_err == nil then
                error("runtime.agent.submit to blocked worker should fail")
            end
            if string.find(tostring(blocked_err), "allowed_child_agents", 1, true) == nil then
                error("blocked runtime.agent.submit should mention allowed_child_agents")
            end

            local complete_out, complete_err = agent.complete("nope", { agent_id = "worker_blocked", timeout_ms = 500 })
            if complete_out ~= nil or complete_err == nil then
                error("agent.complete to blocked worker should fail")
            end
            if string.find(tostring(complete_err), "allowed_child_agents", 1, true) == nil then
                error("blocked agent.complete should mention allowed_child_agents")
            end

            local allowed_id, allowed_err = runtime.agent.submit("worker_allowed", { prompt = "say hello" })
            if allowed_id == nil then error("runtime.agent.submit to allowed worker failed: " .. tostring(allowed_err)) end
            local res, ae = runtime.agent.await(allowed_id, { timeout_ms = 5000 })
            if res == nil then error("runtime.agent.await allowed worker failed: " .. tostring(ae)) end
            if res.status ~= "success" then error("allowed worker should succeed") end
            if res.output ~= "worker-ok" then error("allowed worker output mismatch") end

            return ALLOW
        end
    "#;
    std::fs::write(
        orchestrator_harness_dir.join("orchestrator.lua"),
        orchestrator_harness,
    )?;

    std::fs::write(worker_harness_dir.join("worker.lua"), "-- worker harness\n")?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("worker-ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let mut agents = std::collections::HashMap::new();
    agents.insert(
        "worker_allowed".to_string(),
        AgentConfig {
            tools: Default::default(),
            id: "worker_allowed".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Worker Allowed".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Stateless,
            harness: Some("worker".to_string()),
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
    );
    agents.insert(
        "worker_blocked".to_string(),
        AgentConfig {
            tools: Default::default(),
            id: "worker_blocked".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Worker Blocked".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Stateless,
            harness: Some("worker".to_string()),
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
    );

    let mut governance_agents = std::collections::HashMap::new();
    governance_agents.insert(
        "orchestrator".to_string(),
        turin::kernel::config::GovernanceAgentCapabilitiesConfig {
            capability_profile: None,
            max_capabilities: std::collections::HashMap::new(),
            allowed_child_agents: vec!["worker_allowed".to_string()],
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "orchestrator".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Orchestrator".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents,
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
            directory: orchestrator_harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::from([(
            "worker".to_string(),
            HarnessConfig {
                directory: worker_harness_dir.to_string_lossy().to_string(),
                fs_root: ".".to_string(),
                memory_limit_mb: 32,
            },
        )]),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: turin::kernel::config::GovernanceConfig {
            profile: turin::kernel::config::GovernanceProfile::Balanced,
            enforcement_enabled: true,
            agents: governance_agents,
            ..turin::kernel::config::GovernanceConfig::default()
        },
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise allowed_child_agents enforcement".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_agent_complete_applies_delegated_capability_ceiling() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_agent_complete_peer_delegation.db");
    let orchestrator_harness_dir = tmp.path().join("harnesses_orchestrator");
    let worker_harness_dir = tmp.path().join("harnesses_worker");
    std::fs::create_dir(&orchestrator_harness_dir)?;
    std::fs::create_dir(&worker_harness_dir)?;

    let orchestrator_harness = r#"
        function on_turn_prepare(ctx)
            local out, err = runtime.agent.complete("worker", "delegated worker run", {
                timeout_ms = 5000,
                capabilities = {
                    ["runtime.db.query"] = true
                }
            })
            if out == nil then error("runtime.agent.complete failed: " .. tostring(err)) end
            if out ~= "worker-ok" then error("runtime.agent.complete output mismatch: " .. tostring(out)) end
            return ALLOW
        end
    "#;
    std::fs::write(
        orchestrator_harness_dir.join("orchestrator.lua"),
        orchestrator_harness,
    )?;

    let worker_harness = r#"
        function on_turn_prepare(ctx)
            local query_dec, qe = runtime.governance.check("runtime.db.query")
            if query_dec == nil then error("worker runtime.db.query check failed: " .. tostring(qe)) end
            if query_dec.subject_agent_id ~= "worker" then error("worker subject_agent_id mismatch") end
            if not query_dec.allowed then
                error("delegated worker should have runtime.db.query allowed")
            end

            local rows, qerr = runtime.db.query("SELECT 42 AS n")
            if rows == nil then
                error("worker runtime.db.query should be allowed by delegated ceiling: " .. tostring(qerr))
            end
            if #rows < 1 or rows[1].n ~= 42 then
                error("worker runtime.db.query returned unexpected rows")
            end

            local exec_dec, ede = runtime.governance.check("runtime.db.exec")
            if exec_dec == nil then error("worker runtime.db.exec check failed: " .. tostring(ede)) end
            if exec_dec.allowed then
                error("delegated worker should have runtime.db.exec denied")
            end

            local changed, err = runtime.db.exec("CREATE TABLE IF NOT EXISTS peer_forbidden (id INTEGER)")
            if changed ~= nil or err == nil then
                error("worker runtime.db.exec should be denied by delegated ceiling")
            end
            if string.find(tostring(err), "delegated capabilities", 1, true) == nil then
                error("worker runtime.db.exec denial should mention delegated capabilities")
            end

            return ALLOW
        end
    "#;
    std::fs::write(worker_harness_dir.join("worker.lua"), worker_harness)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("worker-ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let mut agents = std::collections::HashMap::new();
    agents.insert(
        "worker".to_string(),
        AgentConfig {
            tools: Default::default(),
            id: "worker".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Worker".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Stateless,
            harness: Some("worker".to_string()),
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "orchestrator".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Orchestrator".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents,
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
            directory: orchestrator_harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::from([(
            "worker".to_string(),
            HarnessConfig {
                directory: worker_harness_dir.to_string_lossy().to_string(),
                fs_root: ".".to_string(),
                memory_limit_mb: 32,
            },
        )]),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: turin::kernel::config::GovernanceConfig {
            profile: turin::kernel::config::GovernanceProfile::Balanced,
            enforcement_enabled: true,
            ..turin::kernel::config::GovernanceConfig::default()
        },
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise agent.complete delegation".to_string()),
        )
        .await?;
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
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Header test".to_string(),
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
            max_turns: 2,
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
        embeddings: None,
        governance: turin::kernel::config::GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_harness().await?;

    let seen = Arc::new(std::sync::Mutex::new((false, false, None)));
    let provider = Arc::new(HeaderCaptureProvider { seen: seen.clone() });
    kernel.add_client("mock".to_string(), ProviderClient::new("mock", provider));

    let mut session = kernel.create_session().await;
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

#[tokio::test]
async fn test_harness_can_select_named_inference_context() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_inference_contexts.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let harness_code = r#"
        function on_turn_prepare(ctx)
            ctx.inference = "fast"
            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("routing.lua"), harness_code)?;

    let mut providers = HashMap::new();
    providers.insert(
        "primary".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: None,
            ..ProviderConfig::default()
        },
    );
    providers.insert(
        "secondary".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: None,
            ..ProviderConfig::default()
        },
    );

    let mut inference_contexts = HashMap::new();
    inference_contexts.insert(
        "fast".to_string(),
        InferenceContextConfig {
            provider: "secondary".to_string(),
            model: "secondary-model".to_string(),
            fallback: None,
            temperature: Some(0.2),
            max_tokens: Some(256),
            thinking_budget: None,
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "primary-model".to_string(),
            provider: "primary".to_string(),
            system_prompt: "Inference routing test".to_string(),
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
            max_turns: 2,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        layout: Default::default(),
        inference: InferenceConfig {
            default: None,
            contexts: inference_contexts,
            ..InferenceConfig::default()
        },
        persistence: PersistenceConfig::with_state_path(db_path.to_str().unwrap().to_string()),
        harness: HarnessConfig {
            directory: harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::new(),
        providers,
        embeddings: None,
        governance: turin::kernel::config::GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_harness().await?;
    kernel.add_client(
        "primary".to_string(),
        ProviderClient::new(
            "primary",
            Arc::new(FixedTextProvider {
                text: "PRIMARY".to_string(),
            }),
        ),
    );
    kernel.add_client(
        "secondary".to_string(),
        ProviderClient::new(
            "secondary",
            Arc::new(FixedTextProvider {
                text: "SECONDARY".to_string(),
            }),
        ),
    );

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("route via fast".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;

    let mut saw_secondary = false;
    let mut saw_primary = false;
    for msg in &session.history {
        for content in &msg.content {
            if let InferenceContent::Text { text } = content {
                if text.contains("SECONDARY") {
                    saw_secondary = true;
                }
                if text.contains("PRIMARY") {
                    saw_primary = true;
                }
            }
        }
    }

    assert!(
        saw_secondary,
        "expected secondary inference provider output"
    );
    assert!(
        !saw_primary,
        "did not expect primary inference provider output"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_stdlib_context_api_kv_memory_and_tier2() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_stdlib_ctx.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let harness_code = r#"
        function on_turn_prepare(ctx)
            local ident = agent.session.identity()
            if not ident.agent_id or not ident.session_id then
                error("agent.session.identity missing fields")
            end

            local project = runtime.context("project", "alpha", { namespace = "notes" })
            local ok, err = runtime.kv.set("raw_key", "raw_val", project)
            if not ok then error("runtime.kv.set failed: " .. tostring(err)) end

            local raw_val, raw_err = runtime.kv.get("raw_key", project)
            if raw_err ~= nil then error("runtime.kv.get err: " .. tostring(raw_err)) end
            if raw_val ~= "raw_val" then error("runtime.kv.get mismatch") end

            local k = kv.as(project)
            local ok2, err2 = k.set("scoped_key", "scoped_val")
            if not ok2 then error("kv.as.set failed: " .. tostring(err2)) end
            local scoped_val, scoped_err = k.get("scoped_key")
            if scoped_err ~= nil then error("kv.as.get err: " .. tostring(scoped_err)) end
            if scoped_val ~= "scoped_val" then error("kv.as.get mismatch") end

            local m = memory.as(project)
            local mem, em = m.store("alpha memory", { source = "test" }, {
                storage = "lexical_only",
                tags = { "note" },
            })
            if mem == nil then error("memory.as.store failed: " .. tostring(em)) end
            if mem.id == nil then error("memory.as.store missing id") end
            if mem.storage ~= "lexical_only" then error("memory.as.store wrong storage") end

            local hits, hm = m.search("alpha", {
                limit = 5,
                include_metadata = true,
            })
            if hits == nil then error("memory.as.search failed: " .. tostring(hm)) end
            if #hits < 1 then error("memory.as.search returned no hits") end
            if hits[1].id == nil then error("memory.as.search missing id") end
            if hits[1].retrieval_count < 1 then error("memory.as.search missing retrieval_count") end
            if hits[1].metadata == nil or hits[1].metadata.source ~= "test" then
                error("memory.as.search missing metadata")
            end
            if hits[1].metadata._turin == nil or hits[1].metadata._turin.tags[1] ~= "note" then
                error("memory.as.search missing turin metadata")
            end

            local feedback, fe = m.feedback(mem.id, "up", { step = 0.2 })
            if feedback == nil then error("memory.as.feedback failed: " .. tostring(fe)) end
            if feedback.weight <= 1.0 then error("memory.as.feedback did not increase weight") end

            local strict_hits, strict_err = m.search("alpha", {
                mode = "semantic",
                strict = true,
            })
            if strict_hits ~= nil or strict_err == nil then
                error("strict semantic search should fail without embeddings")
            end

            local fallback_hits, fallback_err = m.search("alpha", { mode = "semantic" })
            if fallback_hits == nil then
                error("semantic fallback failed: " .. tostring(fallback_err))
            end
            if #fallback_hits < 1 then
                error("semantic fallback returned no hits")
            end

            local correction, ce = m.correct(
                mem.id,
                "fresh beta memory",
                { source = "corrected" },
                { storage = "lexical_only" }
            )
            if correction == nil then error("memory.as.correct failed: " .. tostring(ce)) end
            if correction.replacement_id == nil then error("memory.as.correct missing replacement id") end

            local corrected_hits, che = m.search("fresh", {
                limit = 5,
                include_metadata = true,
            })
            if corrected_hits == nil or #corrected_hits < 1 then
                error("corrected memory not searchable: " .. tostring(che))
            end
            if corrected_hits[1].id ~= correction.replacement_id then
                error("corrected memory id mismatch")
            end

            local hidden_old, hoe = m.search("alpha", { limit = 5 })
            if hidden_old == nil then error("hidden-old search failed: " .. tostring(hoe)) end
            if #hidden_old ~= 0 then error("superseded memory should be hidden by default") end

            local old_visible, ove = m.search("alpha", {
                limit = 5,
                include_superseded = true,
            })
            if old_visible == nil or #old_visible < 1 then
                error("superseded memory should be visible when requested: " .. tostring(ove))
            end

            local dry_run, dre = m.purge({ only_superseded = true })
            if dry_run == nil then error("memory.as.purge dry-run failed: " .. tostring(dre)) end
            if dry_run.matched < 1 or dry_run.deleted ~= 0 or dry_run.dry_run ~= true then
                error("memory.as.purge dry-run report mismatch")
            end

            local purge, pe = m.purge({
                only_superseded = true,
                dry_run = false,
            })
            if purge == nil then error("memory.as.purge failed: " .. tostring(pe)) end
            if purge.deleted < 1 then error("memory.as.purge deleted nothing") end

            local after_purge, ape = m.search("alpha", {
                limit = 5,
                include_superseded = true,
            })
            if after_purge == nil then error("post-purge search failed: " .. tostring(ape)) end
            if #after_purge ~= 0 then error("purged memory should be gone") end

            local oks, es = session.kv.set("session_seen", "1")
            if not oks then error("session.kv.set failed: " .. tostring(es)) end
            local sv, se = session.kv.get("session_seen")
            if se ~= nil then error("session.kv.get err: " .. tostring(se)) end
            if sv ~= "1" then error("session.kv.get mismatch") end

            local oku, ue = user.kv.set("x", "y")
            if oku ~= false or ue == nil then
                error("user.kv.set should fail without user_id")
            end

            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("stdlib_ctx.lua"), harness_code)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("ok".to_string()),
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
            system_prompt: "Stdlib API test".to_string(),
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
        embeddings: None,
        governance: turin::kernel::config::GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("exercise stdlib".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;

    let project_scope_key = serde_json::json!({
        "namespace": "notes",
        "key": "alpha",
    })
    .to_string();
    let project_store = kernel.store_manager().get_default().await?;
    assert_eq!(
        project_store
            .kv_get("project", &project_scope_key, "raw_key")
            .await?,
        Some("raw_val".to_string())
    );
    assert_eq!(
        project_store
            .kv_get("project", &project_scope_key, "scoped_key")
            .await?,
        Some("scoped_val".to_string())
    );
    let hits = project_store
        .search_memories(
            "project",
            &project_scope_key,
            None,
            None,
            None,
            Some("fresh"),
            5,
            0.0,
            true,
            false,
        )
        .await?;
    assert!(!hits.is_empty(), "expected context memory rows");
    assert!(hits[0].retrieval_count >= 1);
    assert!(
        hits[0].metadata.is_some(),
        "expected context memory metadata"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_runtime_memory_and_kv_support_explicit_store_targets() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_runtime_store_targets.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let harness_code = r#"
        function on_turn_prepare(ctx)
            local project = runtime.context("project", "rust")

            local stored, se = runtime.memory.store(
                "Rust lifetimes require explicit ownership flow",
                project,
                { topic = "rust" },
                { storage = "lexical_only", store = "rust_kb" }
            )
            if stored == nil then error("runtime.memory.store failed: " .. tostring(se)) end

            local hits, he = runtime.memory.search("lifetimes", project, {
                store = "rust_kb",
                include_metadata = true,
            })
            if hits == nil then error("runtime.memory.search failed: " .. tostring(he)) end
            if #hits < 1 then error("runtime.memory.search returned no hits") end
            if hits[1].metadata == nil or hits[1].metadata.topic ~= "rust" then
                error("runtime.memory.search metadata missing")
            end

            local ok, ke = runtime.kv.set(
                "owner",
                "Ferris",
                project,
                { path = ".turin/kb/project.db" }
            )
            if not ok then error("runtime.kv.set failed: " .. tostring(ke)) end

            local value, ve = runtime.kv.get("owner", project, { path = ".turin/kb/project.db" })
            if ve ~= nil then error("runtime.kv.get failed: " .. tostring(ve)) end
            if value ~= "Ferris" then error("runtime.kv.get mismatch") end

            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("runtime_store_targets.lua"), harness_code)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("ok".to_string()),
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
            system_prompt: "Store routing test".to_string(),
            thinking: None,
            mode: AgentMode::Auto,
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
        embeddings: None,
        governance: GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise explicit store targets".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;

    let default_store = kernel.store_manager().get_default().await?;
    let default_hits = default_store
        .search_memories(
            "project",
            "rust",
            None,
            None,
            None,
            Some("lifetimes"),
            5,
            0.0,
            true,
            false,
        )
        .await?;
    assert!(
        default_hits.is_empty(),
        "explicit store memory should not land in default state db"
    );
    assert_eq!(
        default_store.kv_get("project", "rust", "owner").await?,
        None,
        "explicit path kv should not land in default state db"
    );

    let alias_store = kernel
        .store_manager()
        .open(&turin::persistence::manager::StoreSelector::Alias(
            "rust_kb".to_string(),
        ))
        .await?;
    let alias_hits = alias_store
        .search_memories(
            "project",
            "rust",
            None,
            None,
            None,
            Some("lifetimes"),
            5,
            0.0,
            true,
            false,
        )
        .await?;
    assert_eq!(alias_hits.len(), 1);
    assert!(
        alias_hits[0]
            .metadata
            .as_deref()
            .unwrap_or_default()
            .contains("\"topic\":\"rust\"")
    );

    let explicit_path = tmp.path().join(".turin/kb/project.db");
    assert!(
        explicit_path.exists(),
        "expected explicit path db to be created"
    );
    let path_store = kernel
        .store_manager()
        .open(&turin::persistence::manager::StoreSelector::Path(
            ".turin/kb/project.db".to_string(),
        ))
        .await?;
    assert_eq!(
        path_store.kv_get("project", "rust", "owner").await?,
        Some("Ferris".to_string())
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_runtime_memory_and_kv_respect_scope_store_placements() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_scope_store_placement.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let harness_code = r#"
        function on_turn_prepare(ctx)
            local project = runtime.context("project", "rust")

            local stored, se = runtime.memory.store(
                "Placement-routed Rust note",
                project,
                { topic = "rust", layer = "project" },
                { storage = "lexical_only" }
            )
            if stored == nil then error("runtime.memory.store failed: " .. tostring(se)) end

            local hits, he = runtime.memory.search("Placement-routed", project, {
                include_metadata = true,
            })
            if hits == nil then error("runtime.memory.search failed: " .. tostring(he)) end
            if #hits ~= 1 then error("runtime.memory.search returned wrong hit count") end
            if hits[1].metadata == nil or hits[1].metadata.topic ~= "rust" then
                error("runtime.memory.search metadata missing")
            end

            local ok, ke = runtime.kv.set("owner", "Ferris", project)
            if not ok then error("runtime.kv.set failed: " .. tostring(ke)) end

            local value, ve = runtime.kv.get("owner", project)
            if ve ~= nil then error("runtime.kv.get failed: " .. tostring(ve)) end
            if value ~= "Ferris" then error("runtime.kv.get mismatch") end

            return ALLOW
        end
    "#;
    std::fs::write(
        harness_dir.join("runtime_scope_store_placement.lua"),
        harness_code,
    )?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("ok".to_string()),
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
            system_prompt: "Store placement test".to_string(),
            thinking: None,
            mode: AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: std::collections::HashMap::new(),
        kernel: KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 1,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        layout: Default::default(),
        inference: InferenceConfig::default(),
        persistence: PersistenceConfig {
            state: StoreTargetConfig::from_path(db_path.to_str().unwrap().to_string()),
            stores: HashMap::from([(
                "rust_kb".to_string(),
                NamedStoreConfig {
                    path: ".turin/kb/rust.db".to_string(),
                },
            )]),
            placements: vec![ScopedStorePlacementConfig {
                scope_kind: "project".to_string(),
                scope_key: None,
                namespace: None,
                store: "rust_kb".to_string(),
            }],
            ..PersistenceConfig::default()
        },
        harness: HarnessConfig {
            directory: harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::new(),
        providers,
        embeddings: None,
        governance: GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("exercise scope placement".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;

    let default_store = kernel.store_manager().get_default().await?;
    let default_hits = default_store
        .search_memories(
            "project",
            "rust",
            None,
            None,
            None,
            Some("Placement-routed"),
            5,
            0.0,
            true,
            false,
        )
        .await?;
    assert!(
        default_hits.is_empty(),
        "placement-routed memory should not land in default state db"
    );
    assert_eq!(
        default_store.kv_get("project", "rust", "owner").await?,
        None,
        "placement-routed kv should not land in default state db"
    );

    let kb_store = kernel
        .store_manager()
        .open(&turin::persistence::manager::StoreSelector::Alias(
            "rust_kb".to_string(),
        ))
        .await?;
    let kb_hits = kb_store
        .search_memories(
            "project",
            "rust",
            None,
            None,
            None,
            Some("Placement-routed"),
            5,
            0.0,
            true,
            false,
        )
        .await?;
    assert_eq!(kb_hits.len(), 1);
    assert_eq!(
        kb_store.kv_get("project", "rust", "owner").await?,
        Some("Ferris".to_string())
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_runtime_memory_search_supports_multi_source_queries() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_multi_source_search.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let harness_code = r#"
        function on_turn_prepare(ctx)
            local project = runtime.context("project", "rust")
            local global_ctx = runtime.context("global")

            local stored_project, pe = runtime.memory.store(
                "Borrow checker advice from the current workspace",
                project,
                { layer = "project" },
                { storage = "lexical_only" }
            )
            if stored_project == nil then error("project store failed: " .. tostring(pe)) end

            local stored_global, ge = runtime.memory.store(
                "Borrow checker advice from the shared Rust KB",
                global_ctx,
                { layer = "global" },
                { storage = "lexical_only", store = "rust_kb" }
            )
            if stored_global == nil then error("global store failed: " .. tostring(ge)) end

            local hits, he = runtime.memory.search("borrow checker advice", project, {
                include_metadata = true,
                sources = {
                    { scope_kind = "project", scope_key = "rust" },
                    { store = "rust_kb", scope_kind = "global" },
                }
            })
            if hits == nil then error("multi-source search failed: " .. tostring(he)) end
            if #hits ~= 2 then error("multi-source search returned wrong hit count") end

            local seen_project = false
            local seen_global = false
            for _, hit in ipairs(hits) do
                if hit.metadata ~= nil and hit.metadata.layer == "project" then
                    seen_project = true
                end
                if hit.metadata ~= nil and hit.metadata.layer == "global" then
                    seen_global = true
                end
            end
            if not seen_project or not seen_global then
                error("multi-source search did not include both scopes")
            end

            return ALLOW
        end
    "#;
    std::fs::write(
        harness_dir.join("runtime_multi_source_search.lua"),
        harness_code,
    )?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("ok".to_string()),
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
            system_prompt: "Multi-source search test".to_string(),
            thinking: None,
            mode: AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: std::collections::HashMap::new(),
        kernel: KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 1,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        layout: Default::default(),
        inference: InferenceConfig::default(),
        persistence: PersistenceConfig {
            state: StoreTargetConfig::from_path(db_path.to_str().unwrap().to_string()),
            stores: HashMap::from([(
                "rust_kb".to_string(),
                NamedStoreConfig {
                    path: ".turin/kb/rust.db".to_string(),
                },
            )]),
            ..PersistenceConfig::default()
        },
        harness: HarnessConfig {
            directory: harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::new(),
        providers,
        embeddings: None,
        governance: GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise multi-source search".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_runtime_policy_api_round_trip() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_runtime_policy.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let harness_code = r#"
        function on_turn_prepare(ctx)
            local ok, err = runtime.policy.set("spawn.max_depth", 2)
            if not ok then error("global policy set failed: " .. tostring(err)) end

            local v, ge = runtime.policy.get("spawn.max_depth")
            if ge ~= nil then error("global policy get err: " .. tostring(ge)) end
            if v ~= 2 then error("global policy mismatch: " .. tostring(v)) end

            local aok, ae = runtime.policy.set("queue.max_depth", 9, {
                scope = "agent",
                agent_id = "default",
            })
            if not aok then error("agent policy set failed: " .. tostring(ae)) end

            local av, ave = runtime.policy.get("queue.max_depth", {
                scope = "agent",
                agent_id = "default",
            })
            if ave ~= nil then error("agent policy get err: " .. tostring(ave)) end
            if av ~= 9 then error("agent policy mismatch: " .. tostring(av)) end

            local bad_ok, bad_err = runtime.policy.set("spawn.max_depth", "bad")
            if bad_ok ~= false or bad_err == nil then
                error("invalid policy value should return false + err")
            end

            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("policy.lua"), harness_code)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("ok".to_string()),
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
            system_prompt: "Runtime policy test".to_string(),
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
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("exercise runtime policy".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;

    let global_value = kernel
        .policy_manager()
        .get(
            "spawn.max_depth",
            &turin::kernel::policy::PolicyScope::default(),
        )
        .await?;
    assert_eq!(global_value, Some(serde_json::json!(2)));

    let agent_value = kernel
        .policy_manager()
        .get(
            "queue.max_depth",
            &turin::kernel::policy::PolicyScope {
                agent_id: Some("default".to_string()),
                ..turin::kernel::policy::PolicyScope::default()
            },
        )
        .await?;
    assert_eq!(agent_value, Some(serde_json::json!(9)));

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_runtime_cache_api_round_trip() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_runtime_cache.db");
    let harness_dir = tmp.path().join("harnesses");
    let cache_dir = tmp.path().join("cache");
    std::fs::create_dir(&harness_dir)?;
    std::fs::create_dir(&cache_dir)?;
    std::fs::write(cache_dir.join("sample.txt"), "alpha\nbeta\n")?;

    let harness_code = r#"
        function on_turn_prepare(ctx)
            local fresh, fe = runtime.cache.read("cache/sample.txt")
            if fresh == nil then error("fresh cache read failed: " .. tostring(fe)) end
            if fresh.status ~= "fresh" then error("fresh status mismatch") end
            if fresh.content == nil then error("fresh content missing") end

            local unchanged, ue = runtime.cache.read("cache/sample.txt")
            if unchanged == nil then error("unchanged cache read failed: " .. tostring(ue)) end
            if unchanged.status ~= "unchanged" then error("unchanged status mismatch") end
            if unchanged.content ~= nil then error("unchanged should omit content by default") end
            if unchanged.estimated_tokens_saved < 1 then error("unchanged token estimate missing") end

            local okw, we = fs.write("cache/sample.txt", "alpha\ngamma\n")
            if not okw then error("fs.write failed: " .. tostring(we)) end

            local changed, ce = runtime.cache.read("cache/sample.txt", {
                include_previous = true,
            })
            if changed == nil then error("changed cache read failed: " .. tostring(ce)) end
            if changed.status ~= "changed" then error("changed status mismatch") end
            if changed.previous_hash == nil then error("changed previous_hash missing") end
            if changed.previous_content == nil then error("changed previous_content missing") end
            if changed.diff == nil then error("changed diff missing") end

            local stats, se = runtime.cache.stats({ scope = "both" })
            if stats == nil then error("cache stats failed: " .. tostring(se)) end
            if stats.global == nil or stats.global.cached_versions < 2 then
                error("cache global stats mismatch")
            end
            if stats.session == nil or stats.session.files_seen ~= 1 then
                error("cache session stats mismatch")
            end

            local invalidated, ie = runtime.cache.invalidate("cache/sample.txt")
            if invalidated ~= true then error("cache invalidate failed: " .. tostring(ie)) end

            local fresh_again, fae = runtime.cache.read("cache/sample.txt")
            if fresh_again == nil then error("fresh-again read failed: " .. tostring(fae)) end
            if fresh_again.status ~= "fresh" then error("fresh-again status mismatch") end

            local dry_run, dre = runtime.cache.reset({ scope = "session" })
            if dry_run == nil then error("session cache dry-run reset failed: " .. tostring(dre)) end
            if dry_run.dry_run ~= true or dry_run.removed_reads < 1 then
                error("session cache dry-run reset report mismatch")
            end

            local reset, re = runtime.cache.reset({
                scope = "session",
                dry_run = false,
            })
            if reset == nil then error("session cache reset failed: " .. tostring(re)) end
            if reset.removed_reads < 1 then error("session cache reset removed nothing") end

            local after_reset, are = runtime.cache.stats({ scope = "session" })
            if after_reset == nil then error("cache stats after reset failed: " .. tostring(are)) end
            if after_reset.session == nil or after_reset.session.files_seen ~= 0 then
                error("session cache reset did not clear session reads")
            end

            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("runtime_cache.lua"), harness_code)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("ok".to_string()),
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
            system_prompt: "Runtime cache test".to_string(),
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
        embeddings: None,
        governance: turin::kernel::config::GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    let session_uuid = uuid::Uuid::parse_str(session.identity.session_id())?;
    kernel
        .run(&mut session, Some("exercise runtime cache".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;

    let store = kernel.store_manager().get_default().await?;
    let internal_session_id = store
        .get_session_by_public_id(session_uuid)
        .await?
        .expect("main session row missing");
    let cache_stats = store
        .cache_stats(Some(internal_session_id), true, true)
        .await?;
    assert!(
        cache_stats
            .global
            .expect("global cache stats")
            .cached_versions
            >= 2,
        "global cache versions should persist after session reset"
    );

    Ok(())
}

async fn create_synthetic_code_index(
    root: &std::path::Path,
    semantic: bool,
    hybrid: bool,
) -> Result<()> {
    fn vector_blob(fill: f32) -> Vec<u8> {
        let mut blob = Vec::with_capacity(1536 * std::mem::size_of::<f32>());
        for _ in 0..1536 {
            blob.extend_from_slice(&fill.to_le_bytes());
        }
        blob
    }

    fn sparse_vector_blob() -> Vec<u8> {
        let mut blob = Vec::with_capacity(1536 * std::mem::size_of::<f32>());
        blob.extend_from_slice(&1.0_f32.to_le_bytes());
        for _ in 1..1536 {
            blob.extend_from_slice(&0.0_f32.to_le_bytes());
        }
        blob
    }

    let index_dir = root.join(".turin");
    std::fs::create_dir_all(&index_dir)?;
    let index_path = index_dir.join("codebase.db");
    let db = turso::Builder::new_local(index_path.to_str().unwrap())
        .experimental_index_method(true)
        .build()
        .await?;
    let conn = db.connect()?;
    let root_path = std::fs::canonicalize(root)?;
    let capabilities = serde_json::json!({
        "lexical": true,
        "semantic": semantic,
        "hybrid": hybrid,
        "languages": ["rust", "lua"],
    })
    .to_string();
    conn.execute_batch(
        r#"
CREATE TABLE index_meta (
    schema_revision INTEGER NOT NULL,
    root_path TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    capabilities TEXT NOT NULL,
    codebase_id TEXT,
    embedding_key TEXT,
    embedding_dimensions INTEGER,
    embedding_vector_format TEXT,
    embedded_chunks INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE code_chunks (
    chunk_key TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    language TEXT NOT NULL,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    signature TEXT,
    snippet TEXT NOT NULL,
    search_text TEXT NOT NULL,
    embedding BLOB,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    lexical_score REAL NOT NULL,
    semantic_score REAL
);
CREATE INDEX idx_code_chunks_search_fts ON code_chunks USING fts(search_text);
"#,
    )
    .await?;
    conn.execute(
        "INSERT INTO index_meta (schema_revision, root_path, updated_at, capabilities, codebase_id, embedding_key, embedding_dimensions, embedding_vector_format, embedded_chunks) VALUES (?1, ?2, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), ?3, ?4, ?5, ?6, ?7, ?8)",
        turso::params![
            2026031004_i64,
            root_path.to_string_lossy().to_string(),
            capabilities,
            "repo-main",
            if semantic { Some("test:synthetic".to_string()) } else { None },
            if semantic { Some(1536_i64) } else { None },
            if semantic { Some("float8".to_string()) } else { None },
            if semantic { 2_i64 } else { 0_i64 }
        ],
    )
    .await?;
    conn.execute(
        "INSERT INTO code_chunks (chunk_key, path, language, kind, name, signature, snippet, search_text, embedding, start_line, end_line, lexical_score, semantic_score)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, vector8(?9), ?10, ?11, ?12, ?13)",
        turso::params![
            "chunk_rust",
            "src/kernel/governance.rs",
            "rust",
            "function",
            "capability_decision",
            "fn capability_decision(...)",
            "pub fn capability_decision(capability: &str) -> CapabilityDecision",
            "src/kernel/governance.rs\ncapability_decision\nfn capability_decision(...)\npub fn capability_decision(capability: &str) -> CapabilityDecision",
            vector_blob(0.001_f32),
            101_i64,
            132_i64,
            0.91_f64,
            0.82_f64
        ],
    )
    .await?;
    conn.execute(
        "INSERT INTO code_chunks (chunk_key, path, language, kind, name, signature, snippet, search_text, embedding, start_line, end_line, lexical_score, semantic_score)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, vector8(?9), ?10, ?11, ?12, ?13)",
        turso::params![
            "chunk_lua",
            "harnesses/runtime_cache.lua",
            "lua",
            "function",
            "on_turn_prepare",
            "function on_turn_prepare(ctx)",
            "function on_turn_prepare(ctx) local rows = runtime.cache.stats() end",
            "harnesses/runtime_cache.lua\non_turn_prepare\nfunction on_turn_prepare(ctx)\nfunction on_turn_prepare(ctx) local rows = runtime.cache.stats() end",
            sparse_vector_blob(),
            1_i64,
            18_i64,
            0.67_f64,
            0.48_f64
        ],
    )
    .await?;
    conn.execute_batch(
        r#"
CREATE VIEW v_code_lexical AS
SELECT
    chunk_key,
    path,
    language,
    kind,
    name,
    signature,
    snippet,
    start_line,
    end_line,
    lexical_score AS score,
    lexical_score,
    NULL AS semantic_score,
    search_text
FROM code_chunks;
"#,
    )
    .await?;
    if semantic {
        conn.execute_batch(
            r#"
CREATE VIEW v_code_semantic AS
SELECT
    chunk_key,
    path,
    language,
    kind,
    name,
    signature,
    snippet,
    start_line,
    end_line,
    semantic_score AS score,
    NULL AS lexical_score,
    semantic_score,
    embedding
FROM code_chunks
WHERE semantic_score IS NOT NULL;
"#,
        )
        .await?;
    }
    if hybrid {
        conn.execute_batch(
            r#"
CREATE VIEW v_code_hybrid AS
SELECT
    chunk_key,
    path,
    language,
    kind,
    name,
    signature,
    snippet,
    start_line,
    end_line,
    ((lexical_score * 0.5) + (COALESCE(semantic_score, 0.0) * 0.5)) AS score,
    lexical_score,
    semantic_score,
    search_text,
    embedding
FROM code_chunks;
"#,
        )
        .await?;
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_agent_persistence_store_overrides_default_scoped_data_store() -> Result<()> {
    let tmp = tempdir()?;
    let top_state_db = tmp.path().join("top-state.db");
    let agent_state_db = tmp.path().join("agent-state.db");
    let agent_store_db = tmp.path().join("agent-store.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let harness_code = r#"
        function on_turn_prepare(ctx)
            local ok, err = kv.set("agent_marker", "agent")
            if not ok then error("kv.set failed: " .. tostring(err)) end
            local agent_value, agent_err = kv.get("agent_marker")
            if agent_err ~= nil then error("kv.get failed: " .. tostring(agent_err)) end
            if agent_value ~= "agent" then
                error("kv.get mismatch: " .. tostring(agent_value))
            end

            local sok, serr = session.set("session_marker", "session")
            if not sok then error("session.set failed: " .. tostring(serr)) end
            local session_value, session_err = session.get("session_marker")
            if session_err ~= nil then error("session.get failed: " .. tostring(session_err)) end
            if session_value ~= "session" then
                error("session.get mismatch: " .. tostring(session_value))
            end

            local current_session_id = agent.session.identity().session_id
            local loaded, load_err = agent.session.load(current_session_id)
            if load_err ~= nil then error("agent.session.load failed: " .. tostring(load_err)) end
            if loaded == nil or loaded.session_id ~= current_session_id then
                error("agent.session.load mismatch")
            end

            local sessions, list_err = agent.session.list()
            if list_err ~= nil then error("agent.session.list failed: " .. tostring(list_err)) end
            local found = false
            for _, row in ipairs(sessions) do
                if row.session_id == current_session_id then
                    found = true
                    break
                end
            end
            if not found then
                error("agent.session.list missing current session")
            end

            local branches, branch_err = agent.session.branch_list()
            if branch_err ~= nil then error("agent.session.branch_list failed: " .. tostring(branch_err)) end
            if #branches ~= 1 or branches[1].name ~= "main" or branches[1].active ~= true then
                error("agent.session.branch_list missing main branch")
            end

            local created, create_err = agent.session.branch_create("alt", { from_turn_index = 0 })
            if create_err ~= nil then error("agent.session.branch_create failed: " .. tostring(create_err)) end
            if created == nil or created.name ~= "alt" or created.active ~= false or created.deferred ~= false then
                error("agent.session.branch_create mismatch")
            end

            local created_active, created_active_err = agent.session.branch_create("queued", {
                from_turn_index = 0,
                activate = true,
            })
            if created_active_err ~= nil then error("agent.session.branch_create activate=true failed: " .. tostring(created_active_err)) end
            if created_active == nil or created_active.name ~= "queued" or created_active.active ~= false or created_active.deferred ~= true then
                error("agent.session.branch_create activate=true should defer activation")
            end

            local branches_after, branches_after_err = agent.session.branch_list({ session_id = current_session_id })
            if branches_after_err ~= nil then error("agent.session.branch_list(session_id) failed: " .. tostring(branches_after_err)) end
            local saw_alt = false
            local saw_queued = false
            local alt_source_turn_id = nil
            for _, row in ipairs(branches_after) do
                if row.name == "alt" then
                    saw_alt = true
                    alt_source_turn_id = row.source_turn_id
                elseif row.name == "queued" then
                    saw_queued = true
                end
            end
            if not saw_alt or not saw_queued then
                error("agent.session.branch_list missing created branches")
            end
            if alt_source_turn_id == nil then
                error("agent.session.branch_list should surface source_turn_id")
            end

            local siblings, siblings_err = agent.session.branch_siblings(alt_source_turn_id)
            if siblings_err ~= nil then error("agent.session.branch_siblings failed: " .. tostring(siblings_err)) end
            if #siblings ~= 2 then
                error("agent.session.branch_siblings should return the two alternate branches")
            end
            local saw_alt_sibling = false
            local saw_queued_sibling = false
            for _, row in ipairs(siblings) do
                if row.name == "alt" then
                    saw_alt_sibling = true
                elseif row.name == "queued" then
                    saw_queued_sibling = true
                end
                if row.source_turn_id ~= alt_source_turn_id then
                    error("agent.session.branch_siblings returned mismatched source turn")
                end
            end
            if not saw_alt_sibling or not saw_queued_sibling then
                error("agent.session.branch_siblings missing expected siblings")
            end

            local checked_out, checkout_err = agent.session.branch_checkout("alt")
            if checkout_err ~= nil then error("agent.session.branch_checkout failed: " .. tostring(checkout_err)) end
            if checked_out == nil or checked_out.name ~= "alt" or checked_out.active ~= false or checked_out.deferred ~= true then
                error("agent.session.branch_checkout should defer current-session checkout")
            end

            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("main.lua"), harness_code)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let cfg = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            system_prompt: "agent store override".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            thinking: None,
            mode: AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: ContextPersistenceConfig {
                state: Some(StoreTargetConfig::from_path(
                    agent_state_db.to_string_lossy().to_string(),
                )),
                store: Some(StoreTargetConfig::from_path(
                    agent_store_db.to_string_lossy().to_string(),
                )),
            },
        },
        agents: std::collections::HashMap::new(),
        kernel: KernelConfig {
            workspace_root: tmp.path().to_string_lossy().to_string(),
            max_turns: 1,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        layout: Default::default(),
        inference: InferenceConfig::default(),
        persistence: PersistenceConfig::with_state_path(top_state_db.to_string_lossy().to_string()),
        harness: HarnessConfig {
            directory: harness_dir.to_string_lossy().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::new(),
        providers,
        embeddings: None,
        governance: GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(cfg).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    let session_uuid = uuid::Uuid::parse_str(session.identity.session_id())?;
    kernel
        .run(
            &mut session,
            Some("exercise agent store override".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;

    let top_store = kernel.store_manager().get_default().await?;
    assert!(
        top_store
            .get_session_by_public_id(session_uuid)
            .await?
            .is_none(),
        "top-level state DB should not own the session"
    );

    let agent_state_store = kernel
        .store_manager()
        .open(&StoreSelector::Path(
            agent_state_db.to_string_lossy().to_string(),
        ))
        .await?;
    assert!(
        agent_state_store
            .get_session_by_public_id(session_uuid)
            .await?
            .is_some(),
        "agent state DB should own the session"
    );
    assert_eq!(
        agent_state_store
            .kv_get("session", session.identity.session_id(), "session_marker")
            .await?
            .as_deref(),
        Some("session")
    );
    assert_eq!(
        agent_state_store
            .kv_get("agent", "default", "agent_marker")
            .await?,
        None
    );

    let agent_store = kernel
        .store_manager()
        .open(&StoreSelector::Path(
            agent_store_db.to_string_lossy().to_string(),
        ))
        .await?;
    assert_eq!(
        agent_store
            .kv_get("agent", "default", "agent_marker")
            .await?
            .as_deref(),
        Some("agent")
    );
    assert_eq!(
        agent_store
            .kv_get("session", session.identity.session_id(), "session_marker")
            .await?,
        None
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_runtime_code_search_api_round_trip() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_runtime_code_search.db");
    let harness_dir = tmp.path().join("harnesses");
    let repo_root = tmp.path().join("repo");
    let lexical_only_root = tmp.path().join("repo_lexical_only");
    std::fs::create_dir(&harness_dir)?;
    std::fs::create_dir(&repo_root)?;
    std::fs::create_dir(&lexical_only_root)?;
    create_synthetic_code_index(&repo_root, true, true).await?;
    create_synthetic_code_index(&lexical_only_root, false, false).await?;

    let harness_code = r#"
        function on_turn_prepare(ctx)
            local status, se = runtime.code.search.status("repo")
            if status == nil then error("code search status failed: " .. tostring(se)) end
            if status.root == nil or status.index_path == nil then
                error("status root/index_path missing")
            end
            if status.schema_revision ~= 2026031004 then error("schema revision mismatch") end
            if status.capabilities == nil or status.capabilities.lexical ~= true then
                error("status capabilities mismatch")
            end
            if status.codebase_id ~= "repo-main" then error("status codebase_id mismatch") end
            if status.semantic == nil or status.semantic.vector_format ~= "float8" then
                error("status semantic vector format mismatch")
            end

            local rows, le = runtime.code.search.lexical("repo", "capability", {
                limit = 5,
                languages = { "rust" },
                kinds = { "function" },
                min_score = 0.1,
            })
            if rows == nil then error("lexical search failed: " .. tostring(le)) end
            if #rows ~= 1 then error("lexical row count mismatch") end
            if rows[1].name ~= "capability_decision" then error("lexical row mismatch") end
            if rows[1].rank ~= 1 then error("lexical rank mismatch") end

            local hybrid_rows, he = runtime.code.search.hybrid("repo", "capability", {
                trace = true,
            })
            if hybrid_rows == nil then error("hybrid search failed: " .. tostring(he)) end
            if hybrid_rows[1].lexical_score == nil or hybrid_rows[1].semantic_score == nil then
                error("hybrid scores missing")
            end
            if hybrid_rows[1].trace == nil or hybrid_rows[1].trace.fusion ~= "rrf" then
                error("hybrid trace missing")
            end

            local fallback_rows, fe = runtime.code.search.semantic("repo_lexical_only", "cache", {
                trace = true,
            })
            if fallback_rows == nil then error("semantic fallback failed: " .. tostring(fe)) end
            if fallback_rows[1].name ~= "on_turn_prepare" then
                error("semantic fallback row mismatch")
            end
            if fallback_rows[1].semantic_score ~= nil then
                error("semantic fallback should return lexical rows")
            end
            if fallback_rows[1].trace == nil or fallback_rows[1].trace.effective_mode ~= "lexical" then
                error("semantic fallback trace missing")
            end
            if fallback_rows[1].trace.fallback_reason ~= "capability_fallback" then
                error("semantic fallback reason mismatch")
            end

            local strict_rows, strict_err = runtime.code.search.semantic(
                "repo_lexical_only",
                "cache",
                { strict = true }
            )
            if strict_rows ~= nil then error("strict semantic call should fail") end
            if string.find(tostring(strict_err), "semantic capability not available", 1, true) == nil then
                error("strict semantic error mismatch")
            end

            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("runtime_code.lua"), harness_code)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("ok".to_string()),
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
            system_prompt: "Runtime code search test".to_string(),
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
        governance: GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise runtime code search".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_runtime_code_search_falls_back_without_embedding_provider() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_runtime_code_search_fallback.db");
    let harness_dir = tmp.path().join("harnesses");
    let repo_root = tmp.path().join("repo");
    std::fs::create_dir(&harness_dir)?;
    std::fs::create_dir(&repo_root)?;
    create_synthetic_code_index(&repo_root, true, true).await?;

    let harness_code = r#"
        function on_turn_prepare(ctx)
            local semantic_rows, semantic_err = runtime.code.search.semantic("repo", "capability", {
                trace = true,
            })
            if semantic_rows == nil then error("semantic fallback failed: " .. tostring(semantic_err)) end
            if semantic_rows[1].semantic_score ~= nil then
                error("semantic fallback should return lexical rows when no embedding provider exists")
            end
            if semantic_rows[1].trace == nil or semantic_rows[1].trace.fallback_reason ~= "missing_embedding_provider" then
                error("semantic fallback trace mismatch")
            end

            local hybrid_rows, hybrid_err = runtime.code.search.hybrid("repo", "capability", {
                trace = true,
            })
            if hybrid_rows == nil then error("hybrid fallback failed: " .. tostring(hybrid_err)) end
            if hybrid_rows[1].semantic_score ~= nil then
                error("hybrid fallback should return lexical rows when no embedding provider exists")
            end
            if hybrid_rows[1].trace == nil or hybrid_rows[1].trace.fallback_reason ~= "missing_embedding_provider" then
                error("hybrid fallback trace mismatch")
            end

            local strict_rows, strict_err = runtime.code.search.semantic(
                "repo",
                "capability",
                { strict = true }
            )
            if strict_rows ~= nil then error("strict semantic call should fail without embeddings") end
            if string.find(tostring(strict_err), "embedding provider", 1, true) == nil then
                error("strict semantic provider error mismatch")
            end

            local strict_hybrid_rows, strict_hybrid_err = runtime.code.search.hybrid(
                "repo",
                "capability",
                { strict = true }
            )
            if strict_hybrid_rows ~= nil then error("strict hybrid call should fail without embeddings") end
            if string.find(tostring(strict_hybrid_err), "embedding provider", 1, true) == nil then
                error("strict hybrid provider error mismatch")
            end

            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("runtime_code_fallback.lua"), harness_code)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("ok".to_string()),
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
            system_prompt: "Runtime code search fallback test".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness: Some("runtime_code_fallback".to_string()),
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
        embeddings: None,
        governance: GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise runtime code search fallback".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_runtime_governance_observability_api() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_runtime_governance.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let harness_code = r#"
        function on_turn_prepare(ctx)
            local profile = runtime.governance.profile()
            if profile ~= "balanced" then
                error("runtime.governance.profile mismatch: " .. tostring(profile))
            end

            local snap, se = runtime.governance.snapshot()
            if snap == nil then error("runtime.governance.snapshot failed: " .. tostring(se)) end
            if snap.profile ~= "balanced" then error("snapshot.profile mismatch") end
            if snap.enforcement_enabled ~= false then error("snapshot.enforcement_enabled mismatch") end
            if snap.capabilities_observability_only ~= true then error("snapshot should be observability-only in G1") end
            if snap.audit_mode ~= "observational" then error("snapshot.audit_mode mismatch") end
            if snap.import_mode ~= "mixed" then error("snapshot.import_mode mismatch") end
            if snap.grants_enabled ~= true then error("snapshot.grants_enabled mismatch") end

            local saw_db_query = false
            for k, v in pairs(snap.preset_capabilities or {}) do
                if k == "runtime.db.query" and v == true then
                    saw_db_query = true
                end
            end
            if not saw_db_query then error("expected preset runtime.db.query capability") end

            local reviewer, re = runtime.governance.agent("reviewer")
            if reviewer == nil then error("runtime.governance.agent failed: " .. tostring(re)) end
            if reviewer.subject_agent_id ~= "reviewer" then error("reviewer subject_agent_id mismatch") end

            local dec, de = runtime.governance.check("runtime.db.query")
            if dec == nil then error("runtime.governance.check failed: " .. tostring(de)) end
            if dec.subject_agent_id ~= "default" then error("decision subject_agent_id mismatch") end
            if dec.subject_module_name ~= "governance" then
                error("decision subject_module_name mismatch: " .. tostring(dec.subject_module_name))
            end

            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("governance.lua"), harness_code)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let mut roots = std::collections::HashMap::new();
    roots.insert(
        "core".to_string(),
        turin::kernel::config::GovernanceRootConfig {
            path: "harness/core".to_string(),
            writable_hint: false,
            default_profile: Some("core_full".to_string()),
            max_capabilities: std::collections::HashMap::new(),
        },
    );
    let mut agents = std::collections::HashMap::new();
    agents.insert(
        "reviewer".to_string(),
        turin::kernel::config::GovernanceAgentCapabilitiesConfig {
            capability_profile: Some("reviewer_ro".to_string()),
            max_capabilities: std::collections::HashMap::new(),
            allowed_child_agents: vec!["worker".to_string()],
        },
    );
    let mut capability_profiles = std::collections::HashMap::new();
    capability_profiles.insert(
        "reviewer_ro".to_string(),
        std::collections::HashMap::from([(
            "runtime.db.query".to_string(),
            serde_json::Value::Bool(true),
        )]),
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Runtime governance test".to_string(),
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
        governance: turin::kernel::config::GovernanceConfig {
            profile: turin::kernel::config::GovernanceProfile::Balanced,
            enforcement_enabled: false,
            audit: turin::kernel::config::GovernanceAuditConfig {
                mode: turin::kernel::config::GovernanceAuditMode::Observational,
                include_capability_context: true,
                persist_before_hooks: None,
            },
            import: turin::kernel::config::GovernanceImportConfig {
                mode: turin::kernel::config::GovernanceImportMode::Mixed,
                default_root: Some("core".to_string()),
                allow_unscoped_in_open: false,
            },
            roots,
            capability_profiles,
            agents,
            grants: turin::kernel::config::GovernanceGrantsConfig {
                enabled: true,
                max_ttl_ms: Some(60_000),
                require_audit_reason: true,
            },
        },
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise runtime governance api".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_import_scoped_tracks_imported_module_subject_and_root() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_import_scoped.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let util_code = r#"
        return {
            check_subject = function()
                local dec, err = runtime.governance.check("runtime.db.query")
                if dec == nil then error("runtime.governance.check failed: " .. tostring(err)) end
                return dec
            end,
            nested = {
                check_subject = function()
                    local dec, err = runtime.governance.check("runtime.db.query")
                    if dec == nil then error("nested runtime.governance.check failed: " .. tostring(err)) end
                    return dec
                end,
            }
        }
    "#;
    std::fs::write(harness_dir.join("util.lua"), util_code)?;

    let main_code = r#"
        function on_turn_prepare(ctx)
            local util = import_scoped("util", { root = "core" })
            if util == nil then error("import_scoped returned nil") end
            if util.__meta == nil then error("import_scoped missing __meta") end
            if util.__meta.root ~= "core" then error("import_scoped root metadata mismatch") end

            local dec = util.check_subject()
            if dec.subject_agent_id ~= "default" then error("subject_agent_id mismatch") end
            if dec.subject_module_name ~= "util" then
                error("subject_module_name should be util, got " .. tostring(dec.subject_module_name))
            end

            local nested_dec = util.nested.check_subject()
            if nested_dec.subject_module_name ~= "util" then
                error("nested subject_module_name should be util, got " .. tostring(nested_dec.subject_module_name))
            end

            local ok, _ = pcall(function()
                return import_scoped("util", { root = "wrong_root" })
            end)
            if ok then error("import_scoped wrong root should fail") end

            local self_dec, se = runtime.governance.check("runtime.db.query")
            if self_dec == nil then error("self runtime.governance.check failed: " .. tostring(se)) end
            if self_dec.subject_module_name ~= "main" then
                error("caller module context should be restored after import call")
            end

            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("main.lua"), main_code)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let mut roots = std::collections::HashMap::new();
    roots.insert(
        "core".to_string(),
        turin::kernel::config::GovernanceRootConfig {
            path: harness_dir.to_str().unwrap().to_string(),
            writable_hint: false,
            default_profile: Some("core_full".to_string()),
            max_capabilities: std::collections::HashMap::new(),
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "import_scoped test".to_string(),
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
        governance: turin::kernel::config::GovernanceConfig {
            profile: turin::kernel::config::GovernanceProfile::Balanced,
            enforcement_enabled: false,
            roots,
            ..turin::kernel::config::GovernanceConfig::default()
        },
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("exercise import_scoped".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_governed_scoped_import_mode_blocks_unscoped_import() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_governed_scoped_import_mode.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    std::fs::write(
        harness_dir.join("util.lua"),
        r#"
            return {
                ping = function() return "pong" end
            }
        "#,
    )?;

    std::fs::write(
        harness_dir.join("main.lua"),
        r#"
            function on_turn_prepare(ctx)
                local ok_unscoped, _ = pcall(function()
                    return import("util")
                end)
                if ok_unscoped then
                    error("unscoped import() should be blocked in governed scoped mode")
                end

                local util = import_scoped("util")
                if util == nil then error("import_scoped should succeed in scoped mode") end
                if util.ping() ~= "pong" then error("import_scoped util.ping mismatch") end
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
            base_url: Some("ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let mut roots = std::collections::HashMap::new();
    roots.insert(
        "core".to_string(),
        turin::kernel::config::GovernanceRootConfig {
            path: harness_dir.to_str().unwrap().to_string(),
            writable_hint: false,
            default_profile: Some("core_full".to_string()),
            max_capabilities: std::collections::HashMap::new(),
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "governed scoped import mode test".to_string(),
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
        governance: turin::kernel::config::GovernanceConfig {
            profile: turin::kernel::config::GovernanceProfile::Governed,
            enforcement_enabled: true,
            import: turin::kernel::config::GovernanceImportConfig {
                mode: turin::kernel::config::GovernanceImportMode::Scoped,
                default_root: Some("core".to_string()),
                allow_unscoped_in_open: false,
            },
            roots,
            ..turin::kernel::config::GovernanceConfig::default()
        },
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise governed scoped import mode".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_governed_scoped_import_mode_blocks_unscoped_use() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_governed_scoped_use_mode.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    std::fs::write(
        harness_dir.join("util.lua"),
        r#"
            function on_turn_prepare(ctx)
                local dec, err = runtime.governance.check("runtime.db.query")
                if dec == nil then error("runtime.governance.check failed: " .. tostring(err)) end
                if dec.subject_root_name ~= "core" then
                    error("use_scoped root attribution mismatch")
                end
                return ALLOW
            end
        "#,
    )?;

    std::fs::write(
        harness_dir.join("main.lua"),
        r#"
            local ok_unscoped, _ = pcall(function()
                use("util")
            end)
            if ok_unscoped then
                error("unscoped use() should be blocked in governed scoped mode")
            end

            use_scoped("util")

            function on_turn_prepare(ctx)
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
            base_url: Some("ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let mut roots = std::collections::HashMap::new();
    roots.insert(
        "core".to_string(),
        turin::kernel::config::GovernanceRootConfig {
            path: harness_dir.to_str().unwrap().to_string(),
            writable_hint: false,
            default_profile: Some("core_full".to_string()),
            max_capabilities: std::collections::HashMap::new(),
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "governed scoped use mode test".to_string(),
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
        governance: turin::kernel::config::GovernanceConfig {
            profile: turin::kernel::config::GovernanceProfile::Balanced,
            enforcement_enabled: true,
            import: turin::kernel::config::GovernanceImportConfig {
                mode: turin::kernel::config::GovernanceImportMode::Scoped,
                default_root: Some("core".to_string()),
                allow_unscoped_in_open: false,
            },
            roots,
            ..turin::kernel::config::GovernanceConfig::default()
        },
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("exercise use_scoped".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_use_scoped_root_mismatch_fails_harness_init() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_use_scoped_root_mismatch.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    std::fs::write(
        harness_dir.join("util.lua"),
        r#"
            function on_turn_prepare(ctx)
                return ALLOW
            end
        "#,
    )?;

    std::fs::write(
        harness_dir.join("main.lua"),
        r#"
            use_scoped("util", { root = "wrong_root" })
        "#,
    )?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let mut roots = std::collections::HashMap::new();
    roots.insert(
        "core".to_string(),
        turin::kernel::config::GovernanceRootConfig {
            path: harness_dir.to_str().unwrap().to_string(),
            writable_hint: false,
            default_profile: Some("core_full".to_string()),
            max_capabilities: std::collections::HashMap::new(),
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "use_scoped root mismatch test".to_string(),
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
        governance: turin::kernel::config::GovernanceConfig {
            profile: turin::kernel::config::GovernanceProfile::Balanced,
            enforcement_enabled: false,
            roots,
            ..turin::kernel::config::GovernanceConfig::default()
        },
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    let err = kernel.init_harness().await.unwrap_err();
    assert!(
        err.chain()
            .any(|cause| cause.to_string().contains("use_scoped root mismatch")),
        "unexpected error: {err}"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_root_max_capabilities_applies_to_top_level_hooks() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_root_max_caps.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    std::fs::write(
        harness_dir.join("main.lua"),
        r#"
            function on_turn_prepare(ctx)
                local dec, derr = runtime.governance.check("runtime.policy.set")
                if dec == nil then error("runtime.governance.check failed: " .. tostring(derr)) end
                if dec.subject_root_name ~= "core" then
                    error("subject_root_name should be core, got " .. tostring(dec.subject_root_name))
                end
                if dec.allowed then
                    error("runtime.policy.set should be denied by root max_capabilities")
                end
                if dec.reason == nil or string.find(dec.reason, "root max_capabilities 'core'", 1, true) == nil then
                    error("denial reason should reference root max_capabilities")
                end

                local ok, err = runtime.policy.set("root_cap.test", true)
                if ok then
                    error("runtime.policy.set should fail under root max_capabilities")
                end
                if err == nil or string.find(err, "root max_capabilities 'core'", 1, true) == nil then
                    error("runtime.policy.set denial should mention root max_capabilities")
                end
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
            base_url: Some("ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let mut roots = std::collections::HashMap::new();
    let mut root_caps = std::collections::HashMap::new();
    root_caps.insert(
        "runtime.policy.set".to_string(),
        serde_json::Value::Bool(false),
    );
    roots.insert(
        "core".to_string(),
        turin::kernel::config::GovernanceRootConfig {
            path: harness_dir.to_str().unwrap().to_string(),
            writable_hint: false,
            default_profile: Some("core_locked".to_string()),
            max_capabilities: root_caps,
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "root max capabilities test".to_string(),
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
        governance: turin::kernel::config::GovernanceConfig {
            profile: turin::kernel::config::GovernanceProfile::Balanced,
            enforcement_enabled: true,
            roots,
            ..turin::kernel::config::GovernanceConfig::default()
        },
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise root max_capabilities on top-level hooks".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_agent_max_capabilities_denies_runtime_policy_set() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_agent_max_caps.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    std::fs::write(
        harness_dir.join("main.lua"),
        r#"
            function on_turn_prepare(ctx)
                local dec, derr = runtime.governance.check("runtime.policy.set")
                if dec == nil then error("runtime.governance.check failed: " .. tostring(derr)) end
                if dec.allowed then
                    error("runtime.policy.set should be denied by agent max_capabilities")
                end
                if dec.reason == nil or string.find(dec.reason, "agent max_capabilities 'default'", 1, true) == nil then
                    error("denial reason should reference agent max_capabilities")
                end

                local ok, err = runtime.policy.set("agent_cap.test", true)
                if ok then
                    error("runtime.policy.set should fail under agent max_capabilities")
                end
                if err == nil or string.find(err, "agent max_capabilities 'default'", 1, true) == nil then
                    error("runtime.policy.set denial should mention agent max_capabilities")
                end
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
            base_url: Some("ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let mut governance_agents = std::collections::HashMap::new();
    let mut agent_caps = std::collections::HashMap::new();
    agent_caps.insert(
        "runtime.policy.set".to_string(),
        serde_json::Value::Bool(false),
    );
    governance_agents.insert(
        "default".to_string(),
        turin::kernel::config::GovernanceAgentCapabilitiesConfig {
            capability_profile: None,
            max_capabilities: agent_caps,
            allowed_child_agents: vec![],
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "agent max capabilities test".to_string(),
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
        governance: turin::kernel::config::GovernanceConfig {
            profile: turin::kernel::config::GovernanceProfile::Balanced,
            enforcement_enabled: true,
            agents: governance_agents,
            ..turin::kernel::config::GovernanceConfig::default()
        },
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise agent max_capabilities".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_agent_capability_profile_denies_peer_runtime_policy_set() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_agent_capability_profile_peer.db");
    let orchestrator_harness_dir = tmp.path().join("harnesses_orchestrator");
    let reviewer_harness_dir = tmp.path().join("harnesses_reviewer");
    std::fs::create_dir(&orchestrator_harness_dir)?;
    std::fs::create_dir(&reviewer_harness_dir)?;

    std::fs::write(
        orchestrator_harness_dir.join("orchestrator.lua"),
        r#"
            function on_turn_prepare(ctx)
                local self_dec, se = runtime.governance.check("runtime.policy.set")
                if self_dec == nil then error("orchestrator governance.check failed: " .. tostring(se)) end
                if not self_dec.allowed then
                    error("orchestrator should keep runtime.policy.set in balanced mode")
                end

                local task_id, te = runtime.agent.submit("reviewer", { prompt = "capability profile check" })
                if task_id == nil then error("runtime.agent.submit failed: " .. tostring(te)) end

                local res, ae = runtime.agent.await(task_id, { timeout_ms = 5000 })
                if res == nil then error("runtime.agent.await failed: " .. tostring(ae)) end
                if res.agent_id ~= "reviewer" then error("reviewer agent_id mismatch") end
                if res.status ~= "success" then
                    error("reviewer task should succeed, got status " .. tostring(res.status))
                end
                if res.output ~= "reviewer-ok" then
                    error("reviewer output mismatch: " .. tostring(res.output))
                end

                return ALLOW
            end
        "#,
    )?;

    std::fs::write(
        reviewer_harness_dir.join("reviewer.lua"),
        r#"
            function on_turn_prepare(ctx)
                local dec_policy, pe = runtime.governance.check("runtime.policy.set")
                if dec_policy == nil then error("policy decision failed: " .. tostring(pe)) end
                if dec_policy.subject_agent_id ~= "reviewer" then
                    error("reviewer subject_agent_id mismatch")
                end
                if dec_policy.allowed then
                    error("runtime.policy.set should be denied by agent capability_profile")
                end
                if dec_policy.reason == nil or string.find(dec_policy.reason, "agent capability_profile 'reviewer_ro'", 1, true) == nil then
                    error("policy denial reason should mention reviewer_ro capability_profile")
                end

                local dec_query, qe = runtime.governance.check("runtime.db.query")
                if dec_query == nil then error("query decision failed: " .. tostring(qe)) end
                if not dec_query.allowed then
                    error("runtime.db.query should be allowed by reviewer_ro capability_profile")
                end

                local ok, err = runtime.policy.set("reviewer.cap.profile.test", true)
                if ok ~= false or err == nil then
                    error("runtime.policy.set should fail under agent capability_profile")
                end
                if string.find(tostring(err), "agent capability_profile 'reviewer_ro'", 1, true) == nil then
                    error("runtime.policy.set denial should mention reviewer_ro capability_profile")
                end

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
            base_url: Some("reviewer-ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let mut agents = std::collections::HashMap::new();
    agents.insert(
        "reviewer".to_string(),
        AgentConfig {
            tools: Default::default(),
            id: "reviewer".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Reviewer".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Stateless,
            harness: Some("reviewer".to_string()),
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
    );

    let mut capability_profiles = std::collections::HashMap::new();
    capability_profiles.insert(
        "reviewer_ro".to_string(),
        std::collections::HashMap::from([
            (
                "runtime.db.query".to_string(),
                serde_json::Value::Bool(true),
            ),
            (
                "runtime.policy.set".to_string(),
                serde_json::Value::Bool(false),
            ),
        ]),
    );
    let mut governance_agents = std::collections::HashMap::new();
    governance_agents.insert(
        "reviewer".to_string(),
        turin::kernel::config::GovernanceAgentCapabilitiesConfig {
            capability_profile: Some("reviewer_ro".to_string()),
            max_capabilities: std::collections::HashMap::new(),
            allowed_child_agents: vec![],
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "orchestrator".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Orchestrator".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents,
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
            directory: orchestrator_harness_dir.to_str().unwrap().to_string(),
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
        governance: turin::kernel::config::GovernanceConfig {
            profile: turin::kernel::config::GovernanceProfile::Balanced,
            enforcement_enabled: true,
            capability_profiles,
            agents: governance_agents,
            ..turin::kernel::config::GovernanceConfig::default()
        },
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise agent capability_profile enforcement".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_runtime_governance_temporary_grants_issue_use_revoke() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_governance_grants.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    std::fs::write(
        harness_dir.join("main.lua"),
        r#"
            function on_turn_prepare(ctx)
                local before_dec, bde = runtime.governance.check("runtime.policy.set")
                if before_dec == nil then error("pre-grant governance.check failed: " .. tostring(bde)) end
                if not before_dec.allowed then
                    error("runtime.policy.set should be allowed before temporary grant")
                end

                local grant, ge = runtime.governance.grant_issue({
                    capabilities = {
                        ["runtime.db.query"] = true
                    },
                    ttl_ms = 5000,
                    max_uses = 1,
                    reason = "narrow to db.query for one callback"
                })
                if grant == nil then error("grant_issue failed: " .. tostring(ge)) end
                if grant.grant_id == nil then error("grant_issue missing grant_id") end
                if grant.max_uses ~= 1 then error("grant_issue max_uses mismatch") end
                if grant.uses_remaining ~= 1 then error("grant_issue uses_remaining mismatch") end

                local snap, se = runtime.governance.grant_get(grant.grant_id)
                if snap == nil then error("grant_get failed: " .. tostring(se)) end
                if snap.grant_id ~= grant.grant_id then error("grant_get grant_id mismatch") end

                local cb_ret = runtime.governance.with_grant(grant.grant_id, function()
                    local granted_policy, gpe = runtime.governance.check("runtime.policy.set")
                    if granted_policy == nil then error("granted policy check failed: " .. tostring(gpe)) end
                    if granted_policy.allowed then
                        error("temporary grant should deny runtime.policy.set")
                    end
                    if granted_policy.subject_grant_id ~= grant.grant_id then
                        error("subject_grant_id mismatch inside with_grant")
                    end
                    if granted_policy.reason == nil or string.find(granted_policy.reason, "temporary grant", 1, true) == nil then
                        error("temporary grant denial reason missing")
                    end

                    local granted_query, gqe = runtime.governance.check("runtime.db.query")
                    if granted_query == nil then error("granted query check failed: " .. tostring(gqe)) end
                    if not granted_query.allowed then
                        error("temporary grant should allow runtime.db.query")
                    end

                    local ok, err = runtime.policy.set("grant.test", true)
                    if ok ~= false or err == nil then
                        error("runtime.policy.set should fail inside temporary grant")
                    end
                    if string.find(tostring(err), "temporary grant", 1, true) == nil then
                        error("runtime.policy.set denial should mention temporary grant")
                    end
                    return "grant-callback-ok"
                end)
                if cb_ret ~= "grant-callback-ok" then
                    error("with_grant callback return mismatch: " .. tostring(cb_ret))
                end

                local after_dec, ade = runtime.governance.check("runtime.policy.set")
                if after_dec == nil then error("post-grant governance.check failed: " .. tostring(ade)) end
                if not after_dec.allowed then
                    error("runtime.policy.set should be restored after with_grant")
                end
                if after_dec.subject_grant_id ~= nil then
                    error("subject_grant_id should be cleared after with_grant")
                end

                local ok_again, err_again = pcall(function()
                    return runtime.governance.with_grant(grant.grant_id, function()
                        return "unexpected"
                    end)
                end)
                if ok_again then
                    error("with_grant should fail after one-shot grant is consumed")
                end
                if err_again == nil or string.find(tostring(err_again), "not found", 1, true) == nil then
                    error("consumed grant error should mention not found")
                end

                local snap2, se2 = runtime.governance.grant_get(grant.grant_id)
                if snap2 ~= nil then error("consumed grant should not be returned by grant_get") end
                if se2 == nil or string.find(tostring(se2), "not found", 1, true) == nil then
                    error("consumed grant_get should report not found")
                end

                local grant2, g2e = runtime.governance.grant_issue({
                    capabilities = { ["runtime.db.query"] = true },
                    reason = "revoke test"
                })
                if grant2 == nil then error("second grant_issue failed: " .. tostring(g2e)) end
                local revoked, re = runtime.governance.grant_revoke(grant2.grant_id)
                if revoked ~= true then error("grant_revoke failed: " .. tostring(re)) end

                local ok_revoked, err_revoked = pcall(function()
                    return runtime.governance.with_grant(grant2.grant_id, function()
                        return "unexpected"
                    end)
                end)
                if ok_revoked then
                    error("with_grant should fail for revoked grant")
                end
                if err_revoked == nil or string.find(tostring(err_revoked), "not found", 1, true) == nil then
                    error("revoked grant error should mention not found")
                end

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
            base_url: Some("ok".to_string()),
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
            system_prompt: "Governance grant test".to_string(),
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
        governance: turin::kernel::config::GovernanceConfig {
            profile: turin::kernel::config::GovernanceProfile::Balanced,
            enforcement_enabled: true,
            grants: turin::kernel::config::GovernanceGrantsConfig {
                enabled: true,
                max_ttl_ms: Some(10_000),
                require_audit_reason: true,
            },
            ..turin::kernel::config::GovernanceConfig::default()
        },
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise temporary governance grants".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_temporary_grant_ceiling_propagates_to_peer_submit() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_grant_peer_submit.db");
    let orchestrator_harness_dir = tmp.path().join("harnesses_orchestrator");
    let worker_harness_dir = tmp.path().join("harnesses_worker");
    std::fs::create_dir(&orchestrator_harness_dir)?;
    std::fs::create_dir(&worker_harness_dir)?;

    std::fs::write(
        orchestrator_harness_dir.join("orchestrator.lua"),
        r#"
            function on_turn_prepare(ctx)
                local before_dec, be = runtime.governance.check("runtime.policy.set")
                if before_dec == nil then error("pre-grant check failed: " .. tostring(be)) end
                if not before_dec.allowed then error("pre-grant runtime.policy.set should be allowed") end

                local grant, ge = runtime.governance.grant_issue({
                    capabilities = {
                        ["runtime.db.query"] = true
                    },
                    reason = "propagate to peer submit"
                })
                if grant == nil then error("grant_issue failed: " .. tostring(ge)) end

                local cb_out = runtime.governance.with_grant(grant.grant_id, function()
                    local task_id, se = runtime.agent.submit("worker", { prompt = "grant-constrained peer" })
                    if task_id == nil then error("runtime.agent.submit failed: " .. tostring(se)) end

                    local res, ae = runtime.agent.await(task_id, { timeout_ms = 5000 })
                    if res == nil then error("runtime.agent.await failed: " .. tostring(ae)) end
                    if res.status ~= "success" then
                        error("worker task should succeed, got status " .. tostring(res.status))
                    end
                    if res.output ~= "worker-grant-ok" then
                        error("worker output mismatch: " .. tostring(res.output))
                    end
                    return "ok"
                end)
                if cb_out ~= "ok" then error("with_grant return mismatch") end

                local after_dec, ae = runtime.governance.check("runtime.policy.set")
                if after_dec == nil then error("post-grant check failed: " .. tostring(ae)) end
                if not after_dec.allowed then error("post-grant runtime.policy.set should be restored") end
                return ALLOW
            end
        "#,
    )?;

    std::fs::write(
        worker_harness_dir.join("worker.lua"),
        r#"
            function on_turn_prepare(ctx)
                local dec, de = runtime.governance.check("runtime.policy.set")
                if dec == nil then error("worker governance.check failed: " .. tostring(de)) end
                if dec.allowed then
                    error("worker runtime.policy.set should be denied by propagated grant ceiling")
                end
                if dec.reason == nil or string.find(dec.reason, "delegated capabilities", 1, true) == nil then
                    error("worker denial should mention delegated capabilities")
                end

                local ok, err = runtime.policy.set("grant.peer.test", true)
                if ok ~= false or err == nil then
                    error("worker runtime.policy.set should fail under propagated grant ceiling")
                end
                if string.find(tostring(err), "delegated capabilities", 1, true) == nil then
                    error("runtime.policy.set denial should mention delegated capabilities")
                end
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
            base_url: Some("worker-grant-ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let mut agents = std::collections::HashMap::new();
    agents.insert(
        "worker".to_string(),
        AgentConfig {
            tools: Default::default(),
            id: "worker".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Worker".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Stateless,
            harness: Some("worker".to_string()),
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "orchestrator".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Orchestrator".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents,
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
            directory: orchestrator_harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::from([(
            "worker".to_string(),
            HarnessConfig {
                directory: worker_harness_dir.to_string_lossy().to_string(),
                fs_root: ".".to_string(),
                memory_limit_mb: 32,
            },
        )]),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: turin::kernel::config::GovernanceConfig {
            profile: turin::kernel::config::GovernanceProfile::Balanced,
            enforcement_enabled: true,
            grants: turin::kernel::config::GovernanceGrantsConfig {
                enabled: true,
                max_ttl_ms: Some(10_000),
                require_audit_reason: true,
            },
            ..turin::kernel::config::GovernanceConfig::default()
        },
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise grant ceiling peer propagation".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_import_scoped_capability_delegation_is_downward_only() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_import_scoped_caps.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    std::fs::write(
        harness_dir.join("util.lua"),
        r#"
            return {
                try_policy_set = function()
                    local dec_query, qe = runtime.governance.check("runtime.db.query")
                    if dec_query == nil then error("query decision failed: " .. tostring(qe)) end
                    local dec_policy, pe = runtime.governance.check("runtime.policy.set")
                    if dec_policy == nil then error("policy decision failed: " .. tostring(pe)) end

                    local ok, err = runtime.policy.set("delegation.flag", true)
                    return dec_query, dec_policy, ok, err
                end
            }
        "#,
    )?;

    std::fs::write(
        harness_dir.join("main.lua"),
        r#"
            function on_turn_prepare(ctx)
                local util = import_scoped("util", {
                    root = "core",
                    capabilities = {
                        ["runtime.db.*"] = true
                    }
                })

                local dec_query, dec_policy, ok, err = util.try_policy_set()
                if not dec_query.allowed then
                    error("delegated runtime.db.query should stay allowed")
                end
                if dec_query.subject_root_name ~= "core" then
                    error("delegated subject_root_name should be core")
                end
                if dec_policy.allowed then
                    error("delegated runtime.policy.set should be denied by import capability allowlist")
                end
                if ok then
                    error("runtime.policy.set should be denied inside delegated import")
                end
                if err == nil or string.find(err, "delegated capabilities", 1, true) == nil then
                    error("denial should mention delegated capabilities")
                end

                local self_dec, se = runtime.governance.check("runtime.policy.set")
                if self_dec == nil then error("caller runtime.governance.check failed: " .. tostring(se)) end
                if not self_dec.allowed then
                    error("caller capability context should be restored after imported call")
                end

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
            base_url: Some("ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let mut roots = std::collections::HashMap::new();
    roots.insert(
        "core".to_string(),
        turin::kernel::config::GovernanceRootConfig {
            path: harness_dir.to_str().unwrap().to_string(),
            writable_hint: false,
            default_profile: Some("core_full".to_string()),
            max_capabilities: std::collections::HashMap::new(),
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "import scoped capability delegation test".to_string(),
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
        governance: turin::kernel::config::GovernanceConfig {
            profile: turin::kernel::config::GovernanceProfile::Balanced,
            enforcement_enabled: true,
            import: turin::kernel::config::GovernanceImportConfig {
                mode: turin::kernel::config::GovernanceImportMode::Mixed,
                default_root: Some("core".to_string()),
                allow_unscoped_in_open: false,
            },
            roots,
            ..turin::kernel::config::GovernanceConfig::default()
        },
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise import_scoped capability delegation".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_use_scoped_capability_delegation_is_downward_only() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_use_scoped_caps.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    std::fs::write(
        harness_dir.join("util.lua"),
        r#"
            function on_turn_prepare(ctx)
                local dec_query, qe = runtime.governance.check("runtime.db.query")
                if dec_query == nil then error("query decision failed: " .. tostring(qe)) end
                local dec_policy, pe = runtime.governance.check("runtime.policy.set")
                if dec_policy == nil then error("policy decision failed: " .. tostring(pe)) end

                if not dec_query.allowed then
                    error("delegated runtime.db.query should stay allowed")
                end
                if dec_query.subject_root_name ~= "core" then
                    error("delegated subject_root_name should be core")
                end
                if dec_policy.allowed then
                    error("delegated runtime.policy.set should be denied by use capability allowlist")
                end

                local ok, err = runtime.policy.set("use.delegation.flag", true)
                if ok then
                    error("runtime.policy.set should be denied inside delegated use")
                end
                if err == nil or string.find(err, "delegated capabilities", 1, true) == nil then
                    error("denial should mention delegated capabilities")
                end

                return ALLOW
            end
        "#,
    )?;

    std::fs::write(
        harness_dir.join("main.lua"),
        r#"
            use_scoped("util", {
                root = "core",
                capabilities = {
                    ["runtime.db.*"] = true
                }
            })

            function on_turn_prepare(ctx)
                local self_dec, se = runtime.governance.check("runtime.policy.set")
                if self_dec == nil then error("caller runtime.governance.check failed: " .. tostring(se)) end
                if not self_dec.allowed then
                    error("caller capability context should be restored after used block")
                end
                if self_dec.subject_module_name ~= "main" then
                    error("caller module context should be main after used block")
                end
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
            base_url: Some("ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let mut roots = std::collections::HashMap::new();
    roots.insert(
        "core".to_string(),
        turin::kernel::config::GovernanceRootConfig {
            path: harness_dir.to_str().unwrap().to_string(),
            writable_hint: false,
            default_profile: Some("core_full".to_string()),
            max_capabilities: std::collections::HashMap::new(),
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "use scoped capability delegation test".to_string(),
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
        governance: turin::kernel::config::GovernanceConfig {
            profile: turin::kernel::config::GovernanceProfile::Balanced,
            enforcement_enabled: true,
            import: turin::kernel::config::GovernanceImportConfig {
                mode: turin::kernel::config::GovernanceImportMode::Mixed,
                default_root: Some("core".to_string()),
                allow_unscoped_in_open: false,
            },
            roots,
            ..turin::kernel::config::GovernanceConfig::default()
        },
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise use_scoped capability delegation".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_nested_import_cannot_widen_import_delegation() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_nested_import_delegation.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    std::fs::write(
        harness_dir.join("child.lua"),
        r#"
            return {
                ping = function() return "pong" end
            }
        "#,
    )?;

    std::fs::write(
        harness_dir.join("util.lua"),
        r#"
            return {
                try_nested_widen = function()
                    local ok, err = pcall(function()
                        local child = import_scoped("child", {
                            root = "core",
                            capabilities = {
                                ["harness.import.scoped"] = true,
                                ["runtime.policy.set"] = true
                            }
                        })
                        if child == nil then error("child import unexpectedly returned nil") end
                        return child.ping()
                    end)
                    return ok, err
                end
            }
        "#,
    )?;

    std::fs::write(
        harness_dir.join("main.lua"),
        r#"
            function on_turn_prepare(ctx)
                local util = import_scoped("util", {
                    root = "core",
                    capabilities = {
                        ["harness.import.scoped"] = true,
                        ["runtime.db.query"] = true
                    }
                })
                if util == nil then error("util import failed") end

                local ok, err = util.try_nested_widen()
                if ok then
                    error("nested import should not be allowed to widen delegated capabilities")
                end
                if err == nil or string.find(err, "cannot grant 'runtime.policy.set' beyond importer delegation", 1, true) == nil then
                    error("nested import denial should mention delegation widening")
                end
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
            base_url: Some("ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let mut roots = std::collections::HashMap::new();
    roots.insert(
        "core".to_string(),
        turin::kernel::config::GovernanceRootConfig {
            path: harness_dir.to_str().unwrap().to_string(),
            writable_hint: false,
            default_profile: Some("core_full".to_string()),
            max_capabilities: std::collections::HashMap::new(),
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "nested import delegation widening test".to_string(),
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
        governance: turin::kernel::config::GovernanceConfig {
            profile: turin::kernel::config::GovernanceProfile::Balanced,
            enforcement_enabled: true,
            import: turin::kernel::config::GovernanceImportConfig {
                mode: turin::kernel::config::GovernanceImportMode::Mixed,
                default_root: None,
                allow_unscoped_in_open: false,
            },
            roots,
            ..turin::kernel::config::GovernanceConfig::default()
        },
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise nested import delegation widening guard".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_governance_profile_enforcement_blocks_high_risk_runtime_apis() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_governed_enforcement.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let harness_code = r#"
        function on_turn_prepare(ctx)
            local d, de = runtime.governance.check("runtime.db.exec")
            if d == nil then error("runtime.governance.check failed: " .. tostring(de)) end
            if d.allowed ~= false then error("runtime.db.exec should be denied in governed mode") end
            if d.enforcement_enabled ~= true then error("enforcement_enabled should be true") end

            local q, qe = runtime.db.query("SELECT 1 AS n")
            if q == nil then error("runtime.db.query should be allowed: " .. tostring(qe)) end
            if #q < 1 then error("runtime.db.query should return one row") end

            local changed, ce = runtime.db.exec("CREATE TABLE IF NOT EXISTS g2_guard (id INTEGER)")
            if changed ~= nil or ce == nil then
                error("runtime.db.exec should be denied in governed mode")
            end

            local okp, ep = runtime.policy.set("spawn.max_depth", 1)
            if okp ~= false or ep == nil then
                error("runtime.policy.set should be denied in governed mode")
            end

            local token, as_err = agent.spawn("queued subtask")
            if token ~= nil or as_err == nil then
                error("top-level agent.spawn should be denied in governed mode")
            end

            local wrote, fw_err = fs.write("governed-denied.txt", "x")
            if wrote ~= false or fw_err == nil then
                error("top-level fs.write should be denied in governed mode")
            end

            local status_list, sle = runtime.agent.list()
            if status_list == nil then error("runtime.agent.list should still be allowed: " .. tostring(sle)) end

            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("governed.lua"), harness_code)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("ok".to_string()),
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
            system_prompt: "Governed enforcement test".to_string(),
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
        governance: turin::kernel::config::GovernanceConfig {
            profile: turin::kernel::config::GovernanceProfile::Governed,
            enforcement_enabled: true,
            ..turin::kernel::config::GovernanceConfig::default()
        },
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise governed enforcement".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_runtime_db_api_and_context_glob() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_runtime_db_api.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let harness_code = r#"
        function on_turn_prepare(ctx)
            local aliases, ae = runtime.context.glob("*")
            if ae ~= nil then error("runtime.context.glob err: " .. tostring(ae)) end
            local saw_state = false
            for _, a in ipairs(aliases) do
                if a == "state" then saw_state = true end
            end
            if not saw_state then error("runtime.context.glob missing state alias") end

            local h, he = runtime.db.open({ path = ".turin/runtime/test_dynamic.db" })
            if h == nil then error("runtime.db.open failed: " .. tostring(he)) end
            if h.handle == nil then error("runtime.db.open missing handle field") end

            local changed, ce = runtime.db.exec(
                "CREATE TABLE IF NOT EXISTS items (id INTEGER PRIMARY KEY, name TEXT)",
                nil,
                { handle = h.handle }
            )
            if changed == nil and ce ~= nil then error("runtime.db.exec create failed: " .. tostring(ce)) end

            local ins, ie = runtime.db.exec(
                "INSERT INTO items (name) VALUES (?1)",
                { "alice" },
                { handle = h.handle }
            )
            if ins == nil then error("runtime.db.exec insert failed: " .. tostring(ie)) end

            local rows, qe = runtime.db.query(
                "SELECT name FROM items WHERE name = ?1",
                { "alice" },
                { handle = h.handle }
            )
            if rows == nil then error("runtime.db.query failed: " .. tostring(qe)) end
            if #rows < 1 or rows[1].name ~= "alice" then error("runtime.db.query mismatch") end

            local list, le = runtime.db.list()
            if list == nil then error("runtime.db.list failed: " .. tostring(le)) end
            if #list < 1 then error("runtime.db.list should include open handle") end

            local closed, clo_err = runtime.db.close(h)
            if not closed then error("runtime.db.close failed: " .. tostring(clo_err)) end

            local rows2, qe2 = runtime.db.query(
                "SELECT name FROM items",
                nil,
                { handle = h.handle }
            )
            if rows2 ~= nil or qe2 == nil then
                error("runtime.db.query on closed handle should fail")
            end

            local okp, ep = runtime.policy.set("db.allow_dynamic_open", false)
            if not okp then error("policy set failed: " .. tostring(ep)) end
            local h2, he2 = runtime.db.open({ path = ".turin/runtime/blocked.db" })
            if h2 ~= nil or he2 == nil then
                error("runtime.db.open should be blocked when db.allow_dynamic_open=false")
            end

            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("runtime_db.lua"), harness_code)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("ok".to_string()),
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
            system_prompt: "Runtime DB API test".to_string(),
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
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("exercise runtime db api".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;

    let dyn_db = tmp
        .path()
        .join(".turin")
        .join("runtime")
        .join("test_dynamic.db");
    assert!(
        dyn_db.exists(),
        "expected runtime.db.open path to create database"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_runtime_graph_api_records_sparse_relationships() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_runtime_graph_api.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let harness_code = r#"
        function on_turn_prepare(ctx)
            local before, be = runtime.graph.nodes()
            if before == nil then error("runtime.graph.nodes before failed: " .. tostring(be)) end
            if #before ~= 0 then error("ordinary session should not start with graph nodes") end

            local node, ne = runtime.graph.node_create({
                kind = "experiment",
                label = "compare approaches",
                origin_task_id = "task-from-harness",
                metadata = { purpose = "speculation" }
            })
            if node == nil then error("runtime.graph.node_create failed: " .. tostring(ne)) end
            if node.node_id == nil then error("graph node missing node_id") end
            if node.metadata.purpose ~= "speculation" then error("graph node metadata mismatch") end

            local edge, ee = runtime.graph.edge_create({
                source = { kind = "graph_node", id = node.node_id },
                target = { kind = "external_path", id = "wiki://note/interface-design" },
                relation_kind = "contains",
                source_role = "group",
                target_role = "candidate",
                metadata = { rank = 1 }
            })
            if edge == nil then error("runtime.graph.edge_create failed: " .. tostring(ee)) end
            if edge.relation_kind ~= "contains" then error("graph edge relation mismatch") end
            if edge.target_role ~= "candidate" then error("graph edge role mismatch") end

            local edges, le = runtime.graph.edges({
                source = { kind = "graph_node", id = node.node_id }
            })
            if edges == nil then error("runtime.graph.edges failed: " .. tostring(le)) end
            if #edges ~= 1 then error("runtime.graph.edges source lookup mismatch") end

            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("runtime_graph.lua"), harness_code)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("ok".to_string()),
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
            system_prompt: "Runtime graph API test".to_string(),
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
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("exercise runtime graph api".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;

    let store = kernel.store_manager().open(&session.store_selector).await?;
    let internal_id = session.internal_id.expect("session internal id");
    let nodes = store.list_graph_nodes_for_session(internal_id).await?;
    assert_eq!(nodes.len(), 1);
    assert_eq!(nodes[0].kind, "experiment");
    assert_eq!(
        nodes[0].origin_task_id.as_deref(),
        Some("task-from-harness")
    );
    let edges = store.list_graph_edges_for_session(internal_id).await?;
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].relation_kind, "contains");
    assert_eq!(edges[0].target.kind, "external_path");

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_runtime_agent_peer_submit_await_and_status() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_runtime_agent_peer.db");
    let orchestrator_harness_dir = tmp.path().join("harnesses_orchestrator");
    let worker_harness_dir = tmp.path().join("harnesses_worker");
    std::fs::create_dir(&orchestrator_harness_dir)?;
    std::fs::create_dir(&worker_harness_dir)?;

    let orchestrator_harness = r#"
        function on_turn_prepare(ctx)
            local before, be = runtime.agent.get_status("worker")
            if before == nil then error("runtime.agent.get_status before submit failed: " .. tostring(be)) end
            if before.running ~= false then
                error("worker should not be running before submit")
            end

            local task_id, se = runtime.agent.submit("worker", { prompt = "say hello", title = "hello" })
            if task_id == nil then error("runtime.agent.submit failed: " .. tostring(se)) end

            local res, ae = runtime.agent.await(task_id, { timeout_ms = 5000 })
            if res == nil then error("runtime.agent.await failed: " .. tostring(ae)) end
            if res.agent_id ~= "worker" then error("runtime.agent.await wrong agent") end
            if res.status ~= "success" then error("runtime.agent.await status should be success") end
            if res.output ~= "worker-ok" then
                error("runtime.agent.await output mismatch: " .. tostring(res.output))
            end

            local list, le = runtime.agent.list()
            if list == nil then error("runtime.agent.list failed: " .. tostring(le)) end
            local saw_worker = false
            for _, item in ipairs(list) do
                if item.agent_id == "worker" then saw_worker = true end
            end
            if not saw_worker then error("runtime.agent.list missing worker") end

            local after, afe = runtime.agent.get_status("worker")
            if after == nil then error("runtime.agent.get_status after submit failed: " .. tostring(afe)) end
            if after.awaiting_results ~= 0 then error("awaiting_results should be 0 after await") end

            return ALLOW
        end
    "#;
    std::fs::write(
        orchestrator_harness_dir.join("orchestrator.lua"),
        orchestrator_harness,
    )?;

    // Worker harness intentionally empty to validate per-agent harness routing.
    std::fs::write(worker_harness_dir.join("worker.lua"), "-- worker harness\n")?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("worker-ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let mut agents = std::collections::HashMap::new();
    agents.insert(
        "worker".to_string(),
        AgentConfig {
            tools: Default::default(),
            id: "worker".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Worker".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Stateless,
            harness: Some("worker".to_string()),
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "orchestrator".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Orchestrator".to_string(),
            thinking: None,
            mode: turin::kernel::config::AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents,
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
            directory: orchestrator_harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::from([(
            "worker".to_string(),
            HarnessConfig {
                directory: worker_harness_dir.to_string_lossy().to_string(),
                fs_root: ".".to_string(),
                memory_limit_mb: 32,
            },
        )]),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: turin::kernel::config::GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise runtime agent peer".to_string()),
        )
        .await?;

    kernel.end_session(&mut session).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_runtime_agent_sidestep_runs_on_peer_sibling_branch() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_runtime_agent_sidestep.db");
    let orchestrator_harness_dir = tmp.path().join("harnesses_orchestrator");
    let worker_harness_dir = tmp.path().join("harnesses_worker");
    std::fs::create_dir(&orchestrator_harness_dir)?;
    std::fs::create_dir(&worker_harness_dir)?;

    let orchestrator_harness = r#"
        function on_turn_prepare(ctx)
            local warm_id, warm_err = runtime.agent.submit("worker", { prompt = "warm up" })
            if warm_id == nil then error("runtime.agent.submit warmup failed: " .. tostring(warm_err)) end

            local warm_res, await_err = runtime.agent.await(warm_id, { timeout_ms = 5000 })
            if warm_res == nil then error("runtime.agent.await warmup failed: " .. tostring(await_err)) end
            if warm_res.status ~= "success" then
                error("warmup task should succeed, got " .. tostring(warm_res.status))
            end

            local sidestep_id, sidestep_err = runtime.agent.sidestep("worker", "branch-only", {
                mode = "fork_sibling"
            })
            if sidestep_id == nil then error("runtime.agent.sidestep failed: " .. tostring(sidestep_err)) end

            local sidestep_res, sidestep_await_err = runtime.agent.await(sidestep_id, { timeout_ms = 5000 })
            if sidestep_res == nil then error("runtime.agent.await sidestep failed: " .. tostring(sidestep_await_err)) end
            if sidestep_res.status ~= "success" then
                error("sidestep task should succeed, got " .. tostring(sidestep_res.status))
            end
            if sidestep_res.output ~= "worker-ok" then
                error("sidestep output mismatch: " .. tostring(sidestep_res.output))
            end
            if sidestep_res.branch_outcome == nil then
                error("sidestep task should report branch_outcome")
            end
            if sidestep_res.branch_outcome.kind ~= "sidestep_sibling" then
                error("unexpected branch_outcome kind: " .. tostring(sidestep_res.branch_outcome.kind))
            end
            if sidestep_res.branch_outcome.persisted_active_head_unchanged ~= true then
                error("sidestep should preserve persisted active head")
            end

            return ALLOW
        end
    "#;
    std::fs::write(
        orchestrator_harness_dir.join("orchestrator.lua"),
        orchestrator_harness,
    )?;
    std::fs::write(worker_harness_dir.join("worker.lua"), "-- worker harness\n")?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("worker-ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let mut agents = std::collections::HashMap::new();
    agents.insert(
        "worker".to_string(),
        AgentConfig {
            tools: Default::default(),
            id: "worker".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Worker".to_string(),
            thinking: None,
            mode: AgentMode::Stateless,
            harness: Some("worker".to_string()),
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "orchestrator".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Orchestrator".to_string(),
            thinking: None,
            mode: AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents,
        kernel: KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 1,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        layout: Default::default(),
        inference: InferenceConfig::default(),
        persistence: PersistenceConfig::with_state_path(db_path.to_str().unwrap().to_string()),
        harness: HarnessConfig {
            directory: orchestrator_harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::from([(
            "worker".to_string(),
            HarnessConfig {
                directory: worker_harness_dir.to_string_lossy().to_string(),
                fs_root: ".".to_string(),
                memory_limit_mb: 32,
            },
        )]),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("exercise runtime sidestep".to_string()))
        .await?;
    kernel.end_session(&mut session).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_runtime_agent_can_promote_detached_sidestep_result() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_runtime_agent_promote.db");
    let orchestrator_harness_dir = tmp.path().join("harnesses_orchestrator");
    let worker_harness_dir = tmp.path().join("harnesses_worker");
    std::fs::create_dir(&orchestrator_harness_dir)?;
    std::fs::create_dir(&worker_harness_dir)?;

    let orchestrator_harness = r#"
        function on_turn_prepare(ctx)
            local warm_id, warm_err = runtime.agent.submit("worker", { prompt = "warm up" })
            if warm_id == nil then error("runtime.agent.submit warmup failed: " .. tostring(warm_err)) end

            local warm_res, warm_await_err = runtime.agent.await(warm_id, { timeout_ms = 5000 })
            if warm_res == nil then error("runtime.agent.await warmup failed: " .. tostring(warm_await_err)) end
            if warm_res.status ~= "success" then
                error("warmup task should succeed, got " .. tostring(warm_res.status))
            end

            local sidestep_id, sidestep_err = runtime.agent.sidestep("worker", "explore detached", {
                mode = "ephemeral"
            })
            if sidestep_id == nil then error("runtime.agent.sidestep failed: " .. tostring(sidestep_err)) end

            local sidestep_res, sidestep_await_err = runtime.agent.await(sidestep_id, { timeout_ms = 5000 })
            if sidestep_res == nil then error("runtime.agent.await sidestep failed: " .. tostring(sidestep_await_err)) end
            if sidestep_res.status ~= "success" then
                error("detached sidestep should succeed, got " .. tostring(sidestep_res.status))
            end
            if sidestep_res.promotion_candidate == nil then
                error("detached sidestep should expose promotion_candidate")
            end

            local promoted, promote_err = runtime.agent.promote(sidestep_id, {
                branch_name = "kept-runtime-sidestep"
            })
            if promoted == nil then error("runtime.agent.promote failed: " .. tostring(promote_err)) end
            if promoted.name ~= "kept-runtime-sidestep" then
                error("promoted branch name mismatch: " .. tostring(promoted.name))
            end
            if promoted.source_turn_id ~= sidestep_res.promotion_candidate.source_turn_id then
                error("promoted branch source turn mismatch")
            end
            local promoted_again, promote_again_err = runtime.agent.promote(sidestep_id, {
                branch_name = "should-not-create-new-branch"
            })
            if promoted_again == nil then error("runtime.agent.promote retry failed: " .. tostring(promote_again_err)) end
            if promoted_again.branch_id ~= promoted.branch_id then
                error("runtime.agent.promote should be idempotent")
            end
            return ALLOW
        end
    "#;
    std::fs::write(
        orchestrator_harness_dir.join("orchestrator.lua"),
        orchestrator_harness,
    )?;
    std::fs::write(worker_harness_dir.join("worker.lua"), "-- worker harness\n")?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("worker-ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let mut agents = std::collections::HashMap::new();
    agents.insert(
        "worker".to_string(),
        AgentConfig {
            tools: Default::default(),
            id: "worker".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Worker".to_string(),
            thinking: None,
            mode: AgentMode::Stateless,
            harness: Some("worker".to_string()),
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
    );

    let config = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "orchestrator".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            system_prompt: "Orchestrator".to_string(),
            thinking: None,
            mode: AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents,
        kernel: KernelConfig {
            workspace_root: tmp.path().to_str().unwrap().to_string(),
            max_turns: 1,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        layout: Default::default(),
        inference: InferenceConfig::default(),
        persistence: PersistenceConfig::with_state_path(db_path.to_str().unwrap().to_string()),
        harness: HarnessConfig {
            directory: orchestrator_harness_dir.to_str().unwrap().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::from([(
            "worker".to_string(),
            HarnessConfig {
                directory: worker_harness_dir.to_string_lossy().to_string(),
                fs_root: ".".to_string(),
                memory_limit_mb: 32,
            },
        )]),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise runtime sidestep promotion".to_string()),
        )
        .await?;

    kernel.end_session(&mut session).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_agent_sidestep_creates_hidden_sibling_branch_on_current_session() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_agent_sidestep.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let harness = r#"
        local queued = false

        function on_turn_prepare(ctx)
            if not queued then
                queued = true
                local token, err = agent.sidestep("branch-only", { mode = "fork_sibling" })
                if token == nil then error("agent.sidestep failed: " .. tostring(err)) end
            end
            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("default.lua"), harness)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("worker-ok".to_string()),
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
            system_prompt: "Default".to_string(),
            thinking: None,
            mode: AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: Default::default(),
        kernel: KernelConfig {
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
        harnesses: Default::default(),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(&mut session, Some("seed visible path".to_string()))
        .await?;

    let store = kernel.store_manager().open(&session.store_selector).await?;
    let branches = store
        .list_branch_heads(session.internal_id.expect("session internal id"))
        .await?;
    let sidestep_branch = branches
        .iter()
        .find(|branch| !branch.is_active && branch.name.starts_with("sidestep-"))
        .expect("agent.sidestep should create a hidden sibling branch");
    assert_eq!(sidestep_branch.origin_kind, "sidestep");
    let branch_messages = store
        .get_messages(
            session.internal_id.expect("session internal id"),
            &turin::persistence::state::SessionReadTarget::BranchHead(sidestep_branch.id),
        )
        .await?;
    assert!(
        branch_messages
            .iter()
            .any(|message| message.content.contains("branch-only")),
        "sidestep branch should contain the sidestep prompt"
    );

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_agent_can_promote_detached_local_sidestep_result() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_agent_local_promote.db");
    let harness_dir = tmp.path().join("harnesses");
    std::fs::create_dir(&harness_dir)?;

    let harness = r#"
        local sidestep_id = nil
        local task_count = 0

        function on_turn_prepare(ctx)
            task_count = task_count + 1
            if task_count == 1 then
                sidestep_id, err = agent.sidestep("explore detached", { mode = "ephemeral" })
                if sidestep_id == nil then error("agent.sidestep failed: " .. tostring(err)) end
                local queued, queue_err = agent.session.queue("promote local sidestep")
                if not queued then error("agent.session.queue failed: " .. tostring(queue_err)) end
            elseif task_count == 3 then
                local task, task_err = agent.task(sidestep_id)
                if task == nil then error("agent.task failed: " .. tostring(task_err)) end
                if task.status ~= "success" then
                    error("local sidestep status mismatch: " .. tostring(task.status))
                end
                if task.output ~= "worker-ok" then
                    error("local sidestep output mismatch: " .. tostring(task.output))
                end
                if task.promotion_candidate == nil then
                    error("local sidestep should expose promotion_candidate")
                end
                local branch, promote_err = agent.promote(sidestep_id, {
                    branch_name = "kept-local-sidestep"
                })
                if branch == nil then error("agent.promote failed: " .. tostring(promote_err)) end
                if branch.name ~= "kept-local-sidestep" then
                    error("promoted branch name mismatch: " .. tostring(branch.name))
                end
                if branch.origin_kind ~= "promotion" then
                    error("promoted branch origin kind mismatch: " .. tostring(branch.origin_kind))
                end
                if branch.origin_task_id ~= sidestep_id then
                    error("promoted branch origin task mismatch")
                end
                local task_after_promote, task_after_err = agent.task(sidestep_id)
                if task_after_promote == nil then error("agent.task after promote failed: " .. tostring(task_after_err)) end
                if task_after_promote.promoted_branch == nil then
                    error("local task should expose promoted_branch after promotion")
                end
                if task_after_promote.promoted_branch.branch_id ~= branch.branch_id then
                    error("local promoted branch metadata mismatch")
                end
                local branch_again, promote_again_err = agent.promote(sidestep_id, {
                    branch_name = "should-not-create-new-branch"
                })
                if branch_again == nil then error("agent.promote retry failed: " .. tostring(promote_again_err)) end
                if branch_again.branch_id ~= branch.branch_id then
                    error("agent.promote should be idempotent")
                end
            end
            return ALLOW
        end
    "#;
    std::fs::write(harness_dir.join("default.lua"), harness)?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("worker-ok".to_string()),
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
            system_prompt: "Default".to_string(),
            thinking: None,
            mode: AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents: Default::default(),
        kernel: KernelConfig {
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
        harnesses: Default::default(),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: GovernanceConfig::default(),
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(config).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;
    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise local sidestep promotion".to_string()),
        )
        .await?;
    let store = kernel.store_manager().open(&session.store_selector).await?;
    let branches = store
        .list_branch_heads(session.internal_id.expect("session internal id"))
        .await?;
    let promoted = branches
        .iter()
        .find(|branch| branch.name == "kept-local-sidestep")
        .context("promoted local sidestep branch should exist")?;
    assert_eq!(promoted.origin_kind, "promotion");
    let messages = store
        .get_messages(
            session.internal_id.expect("session internal id"),
            &turin::persistence::state::SessionReadTarget::BranchHead(promoted.id),
        )
        .await?;
    let transcript = messages
        .iter()
        .map(|message| format!("{}:{}", message.role, message.content))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        transcript.contains("explore detached"),
        "promoted branch should contain detached sidestep prompt"
    );
    assert!(
        transcript.contains("worker-ok"),
        "promoted branch should contain detached sidestep output"
    );

    kernel.end_session(&mut session).await?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_runtime_agent_complete_allows_post_complete_side_effects() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp.path().join("test_runtime_agent_complete.db");
    let orchestrator_harness_dir = tmp.path().join("harnesses_orchestrator");
    let worker_harness_dir = tmp.path().join("harnesses_worker");
    std::fs::create_dir(&orchestrator_harness_dir)?;
    std::fs::create_dir(&worker_harness_dir)?;

    let orchestrator_harness = r#"
        function on_turn_prepare(ctx)
            local review = runtime.governance.grant({
                ttl_ms = 5000,
                capabilities = {
                    ["runtime.agent.submit"] = true,
                    ["runtime.agent.await"] = true,
                    ["runtime.agent.status"] = true,
                }
            }, function()
                local out, err = runtime.agent.complete("worker", "say hello", { timeout_ms = 5000, title = "hello" })
                if out == nil then error("runtime.agent.complete failed: " .. tostring(err)) end
                return out
            end)

            if review ~= "worker-ok" then
                error("runtime.agent.complete output mismatch: " .. tostring(review))
            end

            local ok, err = fs.write(".turin/runtime/peer-complete.txt", review)
            if not ok then error("fs.write after runtime.agent.complete failed: " .. tostring(err)) end

            local changed, derr = runtime.db.exec([[
                CREATE TABLE IF NOT EXISTS peer_complete_probe (
                    id INTEGER PRIMARY KEY,
                    review TEXT NOT NULL
                )
            ]])
            if changed == nil then error("runtime.db.exec create after runtime.agent.complete failed: " .. tostring(derr)) end

            changed, derr = runtime.db.exec(
                "INSERT INTO peer_complete_probe(review) VALUES (?)",
                { review }
            )
            if changed == nil then error("runtime.db.exec insert after runtime.agent.complete failed: " .. tostring(derr)) end

            local pok, perr = runtime.policy.set("queue.max_depth", 77)
            if not pok then error("runtime.policy.set after runtime.agent.complete failed: " .. tostring(perr)) end

            local pval, pgerr = runtime.policy.get("queue.max_depth")
            if pgerr ~= nil then error("runtime.policy.get after runtime.agent.complete failed: " .. tostring(pgerr)) end
            if pval ~= 77 then error("runtime.policy.get after runtime.agent.complete mismatch: " .. tostring(pval)) end

            session.set("peer_complete_marker", "done")
            local marker = session.get("peer_complete_marker")
            if marker ~= "done" then error("session.get after runtime.agent.complete mismatch: " .. tostring(marker)) end
            return ALLOW
        end
    "#;
    std::fs::write(
        orchestrator_harness_dir.join("orchestrator.lua"),
        orchestrator_harness,
    )?;
    std::fs::write(worker_harness_dir.join("worker.lua"), "-- worker harness\n")?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("worker-ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let mut agents = std::collections::HashMap::new();
    agents.insert(
        "worker".to_string(),
        AgentConfig {
            tools: Default::default(),
            id: "worker".to_string(),
            system_prompt: "worker".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            thinking: None,
            mode: AgentMode::Auto,
            harness: Some("worker".to_string()),
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
    );

    let cfg = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            system_prompt: "orchestrator".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            thinking: None,
            mode: AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents,
        kernel: KernelConfig {
            workspace_root: tmp.path().to_string_lossy().to_string(),
            max_turns: 4,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        layout: Default::default(),
        inference: InferenceConfig::default(),
        persistence: PersistenceConfig::with_state_path(db_path.to_string_lossy().to_string()),
        harness: HarnessConfig {
            directory: orchestrator_harness_dir.to_string_lossy().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::from([(
            "worker".to_string(),
            HarnessConfig {
                directory: worker_harness_dir.to_string_lossy().to_string(),
                fs_root: ".".to_string(),
                memory_limit_mb: 32,
            },
        )]),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: GovernanceConfig {
            profile: GovernanceProfile::Balanced,
            enforcement_enabled: true,
            grants: GovernanceGrantsConfig {
                enabled: true,
                max_ttl_ms: Some(60_000),
                require_audit_reason: false,
            },
            ..GovernanceConfig::default()
        },
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(cfg).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise runtime agent complete".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;

    let artifact = tmp
        .path()
        .join(".turin")
        .join("runtime")
        .join("peer-complete.txt");
    assert!(
        artifact.exists(),
        "expected post-complete artifact to exist"
    );
    assert_eq!(std::fs::read_to_string(&artifact)?, "worker-ok");

    let store = kernel.store_manager().get_default().await?;
    let conn = store.get_connection().await?;
    let mut rows = conn
        .query(
            "SELECT review FROM peer_complete_probe ORDER BY id DESC LIMIT 1",
            (),
        )
        .await?;
    let row = rows
        .next()
        .await?
        .expect("expected peer_complete_probe row");
    let stored_review: String = row.get(0)?;
    assert_eq!(stored_review, "worker-ok");

    let policy_value = kernel
        .policy_manager()
        .get("queue.max_depth", &PolicyScope::default())
        .await?;
    assert_eq!(policy_value, Some(serde_json::json!(77)));

    let session_store = kernel.store_manager().get_default().await?;
    let marker = session_store
        .kv_get(
            "session",
            session.identity.session_id(),
            "peer_complete_marker",
        )
        .await?;
    assert_eq!(marker.as_deref(), Some("done"));

    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn test_runtime_agent_complete_preserves_nested_grant_context() -> Result<()> {
    let tmp = tempdir()?;
    let db_path = tmp
        .path()
        .join("test_runtime_agent_complete_nested_grant.db");
    let orchestrator_harness_dir = tmp.path().join("harnesses_orchestrator");
    let worker_harness_dir = tmp.path().join("harnesses_worker");
    std::fs::create_dir(&orchestrator_harness_dir)?;
    std::fs::create_dir(&worker_harness_dir)?;

    let orchestrator_harness = r#"
        function on_turn_prepare(ctx)
            local outer_gid = nil
            local inner_gid = nil

            local result = runtime.governance.grant({
                ttl_ms = 5000,
                capabilities = {
                    ["runtime.agent.submit"] = true,
                    ["runtime.agent.await"] = true,
                    ["runtime.agent.status"] = true,
                    ["runtime.governance.grant.issue"] = true,
                    ["runtime.governance.grant.use"] = true,
                    ["runtime.governance.grant.revoke"] = true,
                    ["runtime.db.exec"] = true,
                }
            }, function()
                local submit_dec = access.check("runtime.agent.submit")
                if submit_dec == nil or not submit_dec.allowed then
                    error("outer grant should allow runtime.agent.submit")
                end
                outer_gid = submit_dec.subject_grant_id
                if outer_gid == nil or outer_gid == "" then
                    error("outer grant id missing")
                end

                local review, err = runtime.agent.complete("worker", "say hello", { timeout_ms = 5000, title = "hello" })
                if review == nil then
                    error("runtime.agent.complete failed inside nested grant test: " .. tostring(err))
                end
                if review ~= "worker-ok" then
                    error("runtime.agent.complete output mismatch: " .. tostring(review))
                end

                local nested = runtime.governance.grant({
                    ttl_ms = 5000,
                    capabilities = {
                        ["runtime.db.exec"] = true,
                    }
                }, function()
                    local db_dec = access.check("runtime.db.exec")
                    if db_dec == nil or not db_dec.allowed then
                        error("inner grant should allow runtime.db.exec")
                    end
                    inner_gid = db_dec.subject_grant_id
                    if inner_gid == nil or inner_gid == "" then
                        error("inner grant id missing")
                    end
                    if inner_gid == outer_gid then
                        error("inner grant should have a distinct grant id")
                    end

                    local policy_dec = access.check("runtime.policy.set")
                    if policy_dec == nil or policy_dec.allowed then
                        error("inner grant should keep runtime.policy.set denied")
                    end

                    local changed, derr = runtime.db.exec([[
                        CREATE TABLE IF NOT EXISTS nested_complete_probe (
                            id INTEGER PRIMARY KEY,
                            review TEXT NOT NULL,
                            outer_grant_id TEXT NOT NULL,
                            inner_grant_id TEXT NOT NULL
                        )
                    ]])
                    if changed == nil then
                        error("runtime.db.exec create after nested grant failed: " .. tostring(derr))
                    end

                    changed, derr = runtime.db.exec(
                        "INSERT INTO nested_complete_probe(review, outer_grant_id, inner_grant_id) VALUES (?, ?, ?)",
                        { review, outer_gid, inner_gid }
                    )
                    if changed == nil then
                        error("runtime.db.exec insert after nested grant failed: " .. tostring(derr))
                    end

                    return "nested-ok"
                end)

                if nested ~= "nested-ok" then
                    error("nested grant result mismatch: " .. tostring(nested))
                end

                local restored = access.check("runtime.agent.submit")
                if restored == nil or not restored.allowed then
                    error("outer grant should be restored after nested grant")
                end
                if restored.subject_grant_id ~= outer_gid then
                    error("outer grant id should be restored after nested grant")
                end

                return review
            end)

            if result ~= "worker-ok" then
                error("outer grant result mismatch: " .. tostring(result))
            end

            local base_dec = access.check("runtime.agent.submit")
            if base_dec ~= nil and base_dec.subject_grant_id ~= nil then
                error("grant id should be cleared after outer grant")
            end

            return ALLOW
        end
    "#;
    std::fs::write(
        orchestrator_harness_dir.join("orchestrator.lua"),
        orchestrator_harness,
    )?;
    std::fs::write(worker_harness_dir.join("worker.lua"), "-- worker harness\n")?;

    let mut providers = HashMap::new();
    providers.insert(
        "mock".to_string(),
        ProviderConfig {
            kind: "mock".to_string(),
            api_key_env: None,
            base_url: Some("worker-ok".to_string()),
            ..ProviderConfig::default()
        },
    );

    let mut agents = std::collections::HashMap::new();
    agents.insert(
        "worker".to_string(),
        AgentConfig {
            tools: Default::default(),
            id: "worker".to_string(),
            system_prompt: "worker".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            thinking: None,
            mode: AgentMode::Auto,
            harness: Some("worker".to_string()),
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
    );

    let cfg = TurinConfig {
        tools: Default::default(),
        agent: AgentConfig {
            tools: Default::default(),
            id: "default".to_string(),
            system_prompt: "orchestrator".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            thinking: None,
            mode: AgentMode::Auto,
            harness: None,
            idle_grace_secs: None,
            inference: Default::default(),
            persistence: Default::default(),
        },
        agents,
        kernel: KernelConfig {
            workspace_root: tmp.path().to_string_lossy().to_string(),
            max_turns: 4,
            heartbeat_interval_secs: 30,
            initial_spawn_depth: 0,
        },
        layout: Default::default(),
        inference: InferenceConfig::default(),
        persistence: PersistenceConfig::with_state_path(db_path.to_string_lossy().to_string()),
        harness: HarnessConfig {
            directory: orchestrator_harness_dir.to_string_lossy().to_string(),
            fs_root: ".".to_string(),
            memory_limit_mb: 32,
        },
        harnesses: std::collections::HashMap::from([(
            "worker".to_string(),
            HarnessConfig {
                directory: worker_harness_dir.to_string_lossy().to_string(),
                fs_root: ".".to_string(),
                memory_limit_mb: 32,
            },
        )]),
        providers,
        embeddings: Some(EmbeddingConfig::noop()),
        governance: GovernanceConfig {
            profile: GovernanceProfile::Balanced,
            enforcement_enabled: true,
            grants: GovernanceGrantsConfig {
                enabled: true,
                max_ttl_ms: Some(60_000),
                require_audit_reason: false,
            },
            ..GovernanceConfig::default()
        },
        daemon: Default::default(),
        remote: Default::default(),
    };

    let mut kernel = Kernel::builder(cfg).build()?;
    kernel.init_state().await?;
    kernel.init_clients()?;
    kernel.init_harness().await?;

    let mut session = kernel.create_session().await;
    kernel
        .run(
            &mut session,
            Some("exercise runtime agent complete nested grant".to_string()),
        )
        .await?;
    kernel.end_session(&mut session).await?;

    let store = kernel.store_manager().get_default().await?;
    let conn = store.get_connection().await?;
    let mut rows = conn
        .query(
            "SELECT review, outer_grant_id, inner_grant_id FROM nested_complete_probe ORDER BY id DESC LIMIT 1",
            (),
        )
        .await?;
    let row = rows
        .next()
        .await?
        .expect("expected nested_complete_probe row");
    let stored_review: String = row.get(0)?;
    let outer_grant_id: String = row.get(1)?;
    let inner_grant_id: String = row.get(2)?;
    assert_eq!(stored_review, "worker-ok");
    assert!(
        !outer_grant_id.is_empty(),
        "outer grant id should be persisted"
    );
    assert!(
        !inner_grant_id.is_empty(),
        "inner grant id should be persisted"
    );
    assert_ne!(
        outer_grant_id, inner_grant_id,
        "nested grants should use distinct grant ids"
    );

    Ok(())
}
